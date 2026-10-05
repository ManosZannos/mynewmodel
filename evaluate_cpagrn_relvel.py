"""
evaluate_cpagrn_relvel.py - Evaluation for the relative-velocity control
ablation (trained with train_cpagrn_relvel_huberloss.py).

Identical to evaluate_cpagrn.py (including the geodetically-correct ADE/FDE in
metres via local equirectangular projection) EXCEPT that the model class comes
from model_cpagrn_relvel_huberloss. The checkpoint's state_dict contains the
buffer cpa_features.vel_scale (estimated on the training data), which
load_state_dict restores automatically - nothing to pass on the command line.
Architecture hyperparameters (d_model, gru_layers, top_k) are read from the
checkpoint's saved args, as in the other evaluate scripts.

Usage:
    python evaluate_cpagrn_relvel.py \\
        --tag CPAGRN_v5_relvel_huber_d05_gru2_lr5e4_obs10_pred10_s42 \\
        --obs_len 10 --pred_len 10 --split test --gpu_num <GPU>
"""

from __future__ import annotations
import os
import argparse
import numpy as np

import torch
from dataset import get_dataloaders, denorm
from model_cpagrn_relvel_huberloss import CPAGRN


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tag',            type=str,   default='CPAGRN_obs5_pred5_s42')
    p.add_argument('--split',          type=str,   default='test',
                   choices=['val', 'test'])
    p.add_argument('--data_dir',       type=str,   default='dataset/noaa_dec2021_1min')
    p.add_argument('--obs_len',        type=int,   default=5)
    p.add_argument('--pred_len',       type=int,   default=5)
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--gpu_num',        type=int,   default=0)
    return p.parse_args()


M_PER_DEG_LAT = 111_320.0  # ~constant across latitudes (varies <1% pole-to-equator)


def l2_degrees(pred_lat, pred_lon, true_lat, true_lon):
    """Legacy metric: raw Euclidean distance in mixed lat/lon degree units.
    Kept only for continuity with prior reported numbers — NOT geodetically
    correct (treats 1 deg lon == 1 deg lat), see l2_meters()."""
    return np.sqrt((pred_lat - true_lat) ** 2 + (pred_lon - true_lon) ** 2)


def l2_meters(pred_lat, pred_lon, true_lat, true_lon):
    """Geodetically correct L2 error in meters via local equirectangular
    projection: each point's own true latitude sets the lon->meters scale
    (m_per_deg_lon = 111320 * cos(lat)), so lat and lon contribute their
    actual physical distance before being combined."""
    m_per_deg_lon = M_PER_DEG_LAT * np.cos(np.radians(true_lat))
    dlat_m = (pred_lat - true_lat) * M_PER_DEG_LAT
    dlon_m = (pred_lon - true_lon) * m_per_deg_lon
    return np.sqrt(dlat_m ** 2 + dlon_m ** 2)


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
        top_k        = saved.get('top_k',      10),
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
    ade_per_horizon_m = [[] for _ in range(T)]
    fde_list_m = []

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
                    err_m = l2_meters(pred_lat[b,n,:], pred_lon[b,n,:],
                                       true_lat[b,n,:], true_lon[b,n,:])
                    for t in range(T):
                        ade_per_horizon[t].append(err[t])
                        ade_per_horizon_m[t].append(err_m[t])
                    fde_list.append(err[-1])
                    fde_list_m.append(err_m[-1])

    ade_h = [np.mean(h) for h in ade_per_horizon]
    ade   = np.mean(ade_h)
    fde   = np.mean(fde_list)

    ade_h_m = [np.mean(h) for h in ade_per_horizon_m]
    ade_m   = np.mean(ade_h_m)
    fde_m   = np.mean(fde_list_m)

    print(f'\n{"="*55}')
    print(f'  CPA-GRN RelVel | {args.tag} | {args.split}')
    print('='*55)
    print('  [legacy: raw degree-mixed L2, x111320 naive conversion]')
    for t, a in enumerate(ade_h, 1):
        print(f'  ADE {t:>2}min : {a:.6f}°  (naive {a*M_PER_DEG_LAT:.1f} m)')
    print('-'*55)
    print(f'  ADE (avg) : {ade:.6f}°  (naive {ade*M_PER_DEG_LAT:.1f} m)')
    print(f'  FDE       : {fde:.6f}°  (naive {fde*M_PER_DEG_LAT:.1f} m)')
    print('='*55)
    print('  [corrected: geodetic local equirectangular projection]')
    for t, a in enumerate(ade_h_m, 1):
        print(f'  ADE {t:>2}min : {a:.2f} m')
    print('-'*55)
    print(f'  ADE (avg) : {ade_m:.2f} m')
    print(f'  FDE       : {fde_m:.2f} m')
    print(f'  Correction factor (corrected/naive): {ade_m/(ade*M_PER_DEG_LAT):.4f}')
    print('='*55)
    print(f'\n  SMCHN Table 2 reference (same dataset, naive-degree convention):')
    print(f'  Vanilla LSTM {args.pred_len}min ADE = '
          f'{"0.0019°" if args.pred_len==5 else "0.0031°"}')
    print(f'  SMCHN        {args.pred_len}min ADE = '
          f'{"0.0013°" if args.pred_len==5 else "0.0010°"}')
    print('='*55)


if __name__ == '__main__':
    main()