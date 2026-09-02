"""
evaluate_cpagrn_stratified.py — Stratified ADE/FDE by encounter risk.

Motivation: the aggregate ADE/FDE metric weighs every vessel equally,
regardless of whether it's in a genuine close encounter. A model trained
with risk-weighted loss (train_cpagrn_riskloss.py) may have improved
specifically on the risky subset while getting (relatively) worse on the
"easy" majority — a change invisible in the aggregate number, which is
dominated by the majority class.

This script re-computes the same per-vessel risk classification used in
training (DCPA to closest APPROACHING neighbor at the last observed
timestep), splits the test set into "risky" (DCPA <= --risk_threshold)
and "non-risky" (DCPA > threshold, including vessels with no approaching
neighbor at all) groups, and reports ADE/FDE separately for each.

Works on ANY checkpoint from model_cpagrn.CPAGRN (plain gru2, risk-loss
variant, top_k=15, etc.) — the stratification is purely a test-time
diagnostic, unrelated to how the checkpoint was trained.

Usage:
    python evaluate_cpagrn_stratified.py --tag CPAGRN_v5_gru2_obs10_pred10_s42 \
        --obs_len 10 --pred_len 10 --split test --gpu_num <GPU>

    python evaluate_cpagrn_stratified.py --tag CPAGRN_v5_gru2_riskloss_obs10_pred10_s42 \
        --obs_len 10 --pred_len 10 --split test --gpu_num <GPU>

Compare the two runs' "risky" group numbers directly — that is the real
test of whether risk-weighted loss achieved its intended effect.
"""

from __future__ import annotations
import os
import argparse
import numpy as np

import torch
from dataset import get_dataloaders, denorm
from model_cpagrn import CPAGRN


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tag',            type=str,   required=True)
    p.add_argument('--split',          type=str,   default='test', choices=['val', 'test'])
    p.add_argument('--data_dir',       type=str,   default='dataset/noaa_dec2021_1min')
    p.add_argument('--obs_len',        type=int,   default=10)
    p.add_argument('--pred_len',       type=int,   default=10)
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--gpu_num',        type=int,   default=0)
    # Same semantics as train_cpagrn_riskloss.py's DCPA threshold.
    # Default matches the empirical p25 found during risk-loss calibration
    # (30/8 DGX run): p10=0.0017, p25=0.0056, median=0.0150.
    p.add_argument('--risk_threshold', type=float, default=0.0056,
                    help='DCPA cutoff (z-score position units) below which a vessel is '
                         'classified "risky". Default = empirical p25 from calibration run.')
    return p.parse_args()


def compute_min_dcpa(obs: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Same math as compute_risk_weights() in train_cpagrn_riskloss.py, returns raw min_dcpa."""
    B, N, T, _ = obs.shape
    pos_last = obs[:, :, -1, :2]
    vel_last = obs[:, :, -1, :2] - obs[:, :, -2, :2] if T >= 2 else torch.zeros_like(pos_last)

    pos_i = pos_last.unsqueeze(2).expand(B, N, N, 2)
    pos_j = pos_last.unsqueeze(1).expand(B, N, N, 2)
    vel_i = vel_last.unsqueeze(2).expand(B, N, N, 2)
    vel_j = vel_last.unsqueeze(1).expand(B, N, N, 2)

    r = pos_j - pos_i
    v = vel_j - vel_i
    v_sq = (v * v).sum(dim=-1) + eps
    tcpa = (-(r * v).sum(dim=-1) / v_sq).clamp(-5.0, 5.0)
    dcpa = (r + tcpa.unsqueeze(-1) * v).norm(dim=-1).clamp(0.0, 10.0)

    approaching = tcpa > 0
    valid = approaching
    if mask is not None:
        mask_j = mask.unsqueeze(1).expand(B, N, N)
        valid = valid & mask_j
    eye = torch.eye(N, device=obs.device, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
    valid = valid & (~eye)

    dcpa_masked = dcpa.masked_fill(~valid, float('inf'))
    min_dcpa, _ = dcpa_masked.min(dim=-1)
    return torch.nan_to_num(min_dcpa, nan=10.0, posinf=10.0)


def l2_degrees(pred_lat, pred_lon, true_lat, true_lon):
    return np.sqrt((pred_lat - true_lat) ** 2 + (pred_lon - true_lon) ** 2)


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = os.path.join('checkpoints', args.tag, 'val_best.pth')
    assert os.path.exists(ckpt_path), f'Not found: {ckpt_path}'
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    stats = ckpt.get('stats', None)
    print(f'Loaded epoch {ckpt["epoch"]}  val_loss={ckpt.get("val_loss","?")}')

    model = CPAGRN(
        feature_size = 4,
        d_model      = saved.get('d_model',    64),
        gru_layers   = saved.get('gru_layers', 1),
        pred_len     = saved.get('pred_len',   args.pred_len),
        top_k        = saved.get('top_k',      10),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    _, val_loader, test_loader, file_stats = get_dataloaders(
        args.data_dir, args.obs_len, args.pred_len, args.batch_size
    )
    if stats is None:
        stats = file_stats
    loader = test_loader if args.split == 'test' else val_loader

    lon_mean, lon_std = stats['LON']['mean'], stats['LON']['std']
    lat_mean, lat_std = stats['LAT']['mean'], stats['LAT']['std']

    T = args.pred_len
    # Per-vessel: (ade, fde, is_risky)
    records = []

    with torch.no_grad():
        for obs, pred_gt, mask, _ in loader:
            obs     = obs.to(device)
            pred_gt = pred_gt.to(device)
            mask    = mask.to(device)

            last_obs    = obs[:, :, -1, :2]
            target_disp = pred_gt - last_obs.unsqueeze(2)
            pred_disp   = model(obs, mask=mask, stats=stats)

            min_dcpa = compute_min_dcpa(obs, mask).cpu().numpy()  # [B, N]

            pred_abs   = (pred_disp   + last_obs.unsqueeze(2)).cpu().numpy()
            target_abs = (target_disp + last_obs.unsqueeze(2)).cpu().numpy()
            mask_np    = mask.cpu().numpy()
            B, N       = mask_np.shape

            pred_lon = denorm(pred_abs[..., 0],   lon_mean, lon_std)
            pred_lat = denorm(pred_abs[..., 1],   lat_mean, lat_std)
            true_lon = denorm(target_abs[..., 0], lon_mean, lon_std)
            true_lat = denorm(target_abs[..., 1], lat_mean, lat_std)

            for b in range(B):
                for n in range(N):
                    if not mask_np[b, n]:
                        continue
                    err = l2_degrees(pred_lat[b,n,:], pred_lon[b,n,:],
                                     true_lat[b,n,:], true_lon[b,n,:])
                    ade = float(np.mean(err))
                    fde = float(err[-1])
                    is_risky = bool(min_dcpa[b, n] <= args.risk_threshold)
                    records.append((ade, fde, is_risky))

    ade_all = np.array([r[0] for r in records])
    fde_all = np.array([r[1] for r in records])
    risky   = np.array([r[2] for r in records])

    print(f'\n{"="*60}')
    print(f'  Stratified evaluation | {args.tag} | {args.split}')
    print(f'  risk_threshold = {args.risk_threshold} (DCPA, z-score units)')
    print('='*60)
    print(f'  Overall     (n={len(ade_all):>5}): ADE={ade_all.mean():.6f}°  FDE={fde_all.mean():.6f}°')
    if risky.sum() > 0:
        print(f'  RISKY       (n={risky.sum():>5}): ADE={ade_all[risky].mean():.6f}°  FDE={fde_all[risky].mean():.6f}°')
    else:
        print(f'  RISKY       (n=0): no vessels below threshold in this split')
    print(f'  Non-risky   (n={(~risky).sum():>5}): ADE={ade_all[~risky].mean():.6f}°  FDE={fde_all[~risky].mean():.6f}°')
    print(f'  Risky fraction of test set: {100*risky.mean():.1f}%')
    print('='*60)


if __name__ == '__main__':
    main()
