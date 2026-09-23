"""
evaluate_cpagrn_envship.py — Evaluation for CPA-GRN checkpoints trained on
the EnvShip NOAA Track A benchmark (see train_cpagrn_envship.py).

No geodetic conversion needed here: EnvShip's hist/fut coordinates are
already in a local East/North metric frame (meters), unlike our own
lon/lat-based San Diego dataset. ADE/FDE below are directly comparable
to the paper's Table 3 (TCN, LSTM+SDF, LSTM+Gated-Neighbor-Attn, ...)
in units of meters — PHASE 1 caveat still applies (see dataset_envship.py):
this is a trajectory-only comparison until Phase 2 (neighbor context) exists.

Usage:
    python evaluate_cpagrn_envship.py \
        --tag CPAGRN_v5_huber_d05_gru2_lr5e4_envship_noaa_obs30_pred30_s42 \
        --data_dir dataset/envship_noaa_track_a/track_a_short-term_Cross-domain_Datasets/noaa_track_v1 \
        --split test --gpu_num <GPU>
"""

from __future__ import annotations
import os
import argparse
import numpy as np

import torch
from dataset_envship import get_dataloaders_envship, T_PRED_ENVSHIP
from model_cpagrn import CPAGRN

CKPT_ROOT = 'checkpoints_envship'  # matches train_cpagrn_envship.py


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tag',        type=str, required=True)
    p.add_argument('--data_dir',   type=str, required=True)
    p.add_argument('--split',      type=str, default='test', choices=['val', 'test'])
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--gpu_num',    type=int, default=0)
    return p.parse_args()


def l2_meters(pred_xy, true_xy):
    """Already in meters — plain Euclidean distance, no conversion."""
    return np.sqrt(((pred_xy - true_xy) ** 2).sum(axis=-1))


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = os.path.join(CKPT_ROOT, args.tag, 'val_best.pth')
    assert os.path.exists(ckpt_path), f'Not found: {ckpt_path}'
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    stats = ckpt.get('stats', None)
    print(f'Loaded epoch {ckpt["epoch"]}  val_loss={ckpt.get("val_loss","?")}')

    model = CPAGRN(
        feature_size = 4,
        d_model      = saved.get('d_model',    64),
        gru_layers   = saved.get('gru_layers', 2),
        pred_len     = T_PRED_ENVSHIP,
        top_k        = saved.get('top_k',      10),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    _, val_loader, test_loader, file_stats = get_dataloaders_envship(
        args.data_dir, args.batch_size
    )
    if stats is None:
        stats = file_stats
    loader = test_loader if args.split == 'test' else val_loader

    T = T_PRED_ENVSHIP
    ade_per_horizon = [[] for _ in range(T)]
    fde_list = []

    with torch.no_grad():
        for obs, pred_gt, mask, _ in loader:
            obs     = obs.to(device)
            pred_gt = pred_gt.to(device)
            mask    = mask.to(device)

            last_obs    = obs[:, :, -1, :2]
            target_disp = pred_gt - last_obs.unsqueeze(2)
            pred_disp   = model(obs, mask=mask, stats=stats)

            pred_abs   = (pred_disp   + last_obs.unsqueeze(2)).cpu().numpy()
            target_abs = (target_disp + last_obs.unsqueeze(2)).cpu().numpy()
            mask_np    = mask.cpu().numpy()
            B, N       = mask_np.shape

            for b in range(B):
                for n in range(N):
                    if not mask_np[b, n]:
                        continue
                    err = l2_meters(pred_abs[b, n, :, :], target_abs[b, n, :, :])
                    for t in range(T):
                        ade_per_horizon[t].append(err[t])
                    fde_list.append(err[-1])

    ade_h = [np.mean(h) for h in ade_per_horizon]
    ade   = np.mean(ade_h)
    fde   = np.mean(fde_list)

    print(f'\n{"="*55}')
    print(f'  CPA-GRN (EnvShip NOAA Track A, Phase 1) | {args.tag} | {args.split}')
    print('='*55)
    for horizon_min, step in [(3, 9), (6, 18), (10, 29)]:  # 20s steps -> minutes
        print(f'  ADE {horizon_min:>2}min (step {step+1:>2}) : {ade_h[step]:.2f} m')
    print('-'*55)
    print(f'  ADE (avg, full 10min) : {ade:.2f} m')
    print(f'  FDE                   : {fde:.2f} m')
    print('='*55)
    print(f'\n  EnvShip paper Table 3 reference (NOAA, in-domain):')
    print(f'  TCN (trajectory-only, best)    : ADE=92.8 m')
    print(f'  LSTM + SDF (env-aware, best)   : ADE=100.4 m')
    print(f'  (Gated-Neighbor-Attn NOAA number not given in main text —')
    print(f'   check full result grid in the HF release for exact figure)')
    print('='*55)


if __name__ == '__main__':
    main()
