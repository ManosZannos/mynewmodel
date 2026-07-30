"""
evaluate_cpagrn_smoothvel.py — Evaluation for CPA-GRN v4 + smoothed velocity.

Mirrors evaluate_cpagrn.py, but imports model_cpagrn_smoothvel.py instead
of model_cpagrn.py (needed for the same reason as the multihead evaluate
script: the class comes from a different module, so evaluate_cpagrn.py
can't be reused even though shapes match).

Also includes the top_k fix: passes saved['top_k'] to the model constructor
instead of silently falling back to the default (10). See experiment_log.md,
30/7/2026 — evaluate_cpagrn.py was found to be missing this, which means
any past evaluation of a non-default top_k checkpoint (e.g. top_k=15) via
the original script may have silently evaluated with top_k=10 instead.

Usage:
    python evaluate_cpagrn_smoothvel.py --tag CPAGRN_smoothvel_obs20_pred20_s42 \
        --obs_len 20 --pred_len 20 --split test --gpu_num 0
"""

from __future__ import annotations
import os
import argparse
import numpy as np

import torch
from dataset import get_dataloaders, denorm
from model_cpagrn_smoothvel import CPAGRN


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tag',            type=str,   default='CPAGRN_smoothvel_obs5_pred5_s42')
    p.add_argument('--split',          type=str,   default='test',
                   choices=['val', 'test'])
    p.add_argument('--data_dir',       type=str,   default='dataset/noaa_dec2021_1min')
    p.add_argument('--obs_len',        type=int,   default=5)
    p.add_argument('--pred_len',       type=int,   default=5)
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--gpu_num',        type=int,   default=0)
    return p.parse_args()


def l2_degrees(pred_lat, pred_lon, true_lat, true_lon):
    return np.sqrt((pred_lat - true_lat) ** 2 + (pred_lon - true_lon) ** 2)


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt_path = os.path.join('checkpoints', args.tag, 'val_best.pth')
    assert os.path.exists(ckpt_path), f'Not found: {ckpt_path}'
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    stats = ckpt.get('stats', None)
    print(f'Loaded epoch {ckpt["epoch"]}  val_loss={ckpt.get("val_loss","?"):.6f}')

    model = CPAGRN(
        feature_size = 4,
        d_model      = saved.get('d_model',    64),
        gru_layers   = saved.get('gru_layers', 1),
        pred_len     = saved.get('pred_len',   args.pred_len),
        top_k        = saved.get('top_k',      10),   # ← fix: honor the trained top_k
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Data
    _, val_loader, test_loader, file_stats = get_dataloaders(
        args.data_dir, args.obs_len, args.pred_len, args.batch_size
    )
    if stats is None:
        stats = file_stats
    loader = test_loader if args.split == 'test' else val_loader

    lon_mean, lon_std = stats['LON']['mean'], stats['LON']['std']
    lat_mean, lat_std = stats['LAT']['mean'], stats['LAT']['std']

    T = args.pred_len
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
                    for t in range(T):
                        ade_per_horizon[t].append(err[t])
                    fde_list.append(err[-1])

    ade_h = [np.mean(h) for h in ade_per_horizon]
    ade   = np.mean(ade_h)
    fde   = np.mean(fde_list)

    print(f'\n{"="*55}')
    print(f'  CPA-GRN (smoothvel) | {args.tag} | {args.split}')
    print('='*55)
    for t, a in enumerate(ade_h, 1):
        print(f'  ADE {t:>2}min : {a:.6f}°  ({a*60:.5f} nm)')
    print('-'*55)
    print(f'  ADE (avg) : {ade:.6f}°  ({ade*60:.5f} nm)')
    print(f'  FDE       : {fde:.6f}°  ({fde*60:.5f} nm)')
    print('='*55)


if __name__ == '__main__':
    main()
