"""
evaluate_cpagrn_uniontopk.py — Evaluation for CPA-GRN, Union Top-K variant.

Same ADE/FDE reporting format as evaluate_cpagrn.py, but loads
CPAGRNUnionTopK and reads top_k_dist/top_k_risk from the checkpoint's
saved args (NOT from CLI — same convention as top_k in evaluate_cpagrn.py).

Usage:
    python evaluate_cpagrn_uniontopk.py --tag CPAGRN_v5_uniontopk_gru2_obs10_pred10_s42 \
        --obs_len 10 --pred_len 10 --split test --gpu_num <GPU>
"""

from __future__ import annotations
import os
import argparse
import numpy as np

import torch
from dataset import get_dataloaders, denorm
from model_cpagrn_uniontopk import CPAGRNUnionTopK


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tag',            type=str,   default='CPAGRN_uniontopk_obs5_pred5_s42')
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

    ckpt_path = os.path.join('checkpoints', args.tag, 'val_best.pth')
    assert os.path.exists(ckpt_path), f'Not found: {ckpt_path}'
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    stats = ckpt.get('stats', None)
    print(f'Loaded epoch {ckpt["epoch"]}  val_loss={ckpt.get("val_loss","?"):.6f}')
    print(f"top_k_dist={saved.get('top_k_dist', 12)}  "
          f"top_k_risk={saved.get('top_k_risk', 3)}  "
          f"gru_layers={saved.get('gru_layers', 1)}")

    model = CPAGRNUnionTopK(
        feature_size = 4,
        d_model      = saved.get('d_model',    64),
        gru_layers   = saved.get('gru_layers', 1),
        pred_len     = saved.get('pred_len',   args.pred_len),
        top_k_dist   = saved.get('top_k_dist', 12),
        top_k_risk   = saved.get('top_k_risk', 3),
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
    print(f'  CPA-GRN (union top-k) | {args.tag} | {args.split}')
    print('='*55)
    for t, a in enumerate(ade_h, 1):
        print(f'  ADE {t:>2}min : {a:.6f}°  ({a*60:.5f} nm)')
    print('-'*55)
    print(f'  ADE (avg) : {ade:.6f}°  ({ade*60:.5f} nm)')
    print(f'  FDE       : {fde:.6f}°  ({fde*60:.5f} nm)')
    print('='*55)


if __name__ == '__main__':
    main()
