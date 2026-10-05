"""
dump_errors.py — run ONE checkpoint over a split and save PER-VESSEL errors
(geodetically correct, metres, every horizon step) plus metadata to a .npz.

Why: every publication analysis (scene-clustered bootstrap, seen-vs-unseen
vessels, speed strata, risky vs non-risky, per-day, per-horizon, paired model
comparisons) becomes a cheap CPU post-processing of these dumps
(analyze_errors.py). The GPU pass is done once per checkpoint.

Supported architectures (--arch):
    cpagrn : model_cpagrn.CPAGRN            (headline; also gru2 MSE etc.)
    nocpa  : model_cpagrn_nocpa.CPAGRN      (No-CPA ablation)
    nograph: model_cpagrn_nograph_huberloss.CPAGRN (no neighbor aggregation)
    relvel : model_cpagrn_relvel_huberloss.CPAGRN  (relative-velocity control)
    lstm   : model_lstm.VanillaLSTM         (baseline)
Architecture hyper-parameters are read from the checkpoint's saved args, as in
the evaluate_*.py scripts — do not pass them on the command line.

The error definition is IMPORTED from evaluate_cpagrn.py (l2_meters/l2_degrees)
and the risk definition from evaluate_cpagrn_stratified.py (compute_min_dcpa),
so the numbers are the same as the official evaluation by construction.
SANITY CHECK: the printed mean ADE (deg) must equal the value from the official
evaluate script for the same checkpoint (e.g. headline s42: 0.000836 deg,
84.55 m; n = 83,336 vessel-samples on the test split).

Usage (run 1 dump per checkpoint; same tags as training):
    python dump_errors.py --arch cpagrn --tag CPAGRN_v5_huber_d05_gru2_lr5e4_obs10_pred10_s42 --gpu_num 2
    python dump_errors.py --arch nocpa  --tag CPAGRN_v5_nocpa_huber_d05_gru2_lr5e4_obs10_pred10_s42 --gpu_num 2
    python dump_errors.py --arch lstm   --tag <lstm tag> --gpu_num 2
"""

from __future__ import annotations
import os
import json
import argparse

import numpy as np
import torch

from dataset import collate_fn, denorm
from dataset_meta import AISDatasetMeta, verify_against_parent, train_vessel_ids
from evaluate_cpagrn import l2_meters, l2_degrees
from evaluate_cpagrn_stratified import compute_min_dcpa


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arch',       type=str, required=True, choices=['cpagrn', 'nocpa', 'nograph', 'relvel', 'lstm'])
    p.add_argument('--tag',        type=str, required=True)
    p.add_argument('--split',      type=str, default='test', choices=['val', 'test'])
    p.add_argument('--data_dir',   type=str, default='dataset/noaa_dec2021_1min')
    p.add_argument('--obs_len',    type=int, default=10)
    p.add_argument('--pred_len',   type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--gpu_num',    type=int, default=0)
    p.add_argument('--out_dir',    type=str, default='error_dumps')
    p.add_argument('--skip_verify', action='store_true',
                   help='skip the check that AISDatasetMeta == AISDataset (default: verify)')
    return p.parse_args()


def build_model(arch: str, saved: dict, pred_len: int, device):
    if arch in ('cpagrn', 'nocpa', 'nograph', 'relvel'):
        if arch == 'cpagrn':
            from model_cpagrn import CPAGRN
        elif arch == 'nocpa':
            from model_cpagrn_nocpa import CPAGRN
        elif arch == 'relvel':
            from model_cpagrn_relvel_huberloss import CPAGRN
        else:
            from model_cpagrn_nograph_huberloss import CPAGRN
        model = CPAGRN(
            feature_size = 4,
            d_model      = saved.get('d_model',    64),
            gru_layers   = saved.get('gru_layers', 1),
            pred_len     = saved.get('pred_len',   pred_len),
            top_k        = saved.get('top_k',      10),
        )
    else:
        from model_lstm import VanillaLSTM
        model = VanillaLSTM(
            feature_size = 4,
            hidden_size  = saved.get('hidden_size', 64),
            num_layers   = saved.get('num_layers',  1),
            pred_len     = saved.get('pred_len',    pred_len),
        )
    return model.to(device)


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = os.path.join('checkpoints', args.tag, 'val_best.pth')
    assert os.path.exists(ckpt_path), f'Not found: {ckpt_path}'
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    stats = ckpt.get('stats', None)
    if stats is None:
        with open(os.path.join(args.data_dir, 'global_stats.json')) as f:
            stats = json.load(f)
    print(f'Loaded epoch {ckpt["epoch"]}  arch={args.arch}  tag={args.tag}')

    model = build_model(args.arch, saved, args.pred_len, device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Same stride as dataset.get_dataloaders: eval_stride = obs_len + pred_len
    stride  = args.obs_len + args.pred_len
    csv_dir = os.path.join(args.data_dir, args.split)
    if not args.skip_verify:
        verify_against_parent(csv_dir, args.obs_len, args.pred_len, stride)
    ds = AISDatasetMeta(csv_dir, args.obs_len, args.pred_len, stride=stride)
    seen_ids = np.fromiter(train_vessel_ids(args.data_dir), dtype=np.int64)

    lon_mean, lon_std = stats['LON']['mean'], stats['LON']['std']
    lat_mean, lat_std = stats['LAT']['mean'], stats['LAT']['std']
    sog_mean, sog_std = stats['SOG']['mean'], stats['SOG']['std']

    keys = ['err_m', 'err_deg', 'window_idx', 'day', 'frame_start', 'vessel_id',
            'n_in_window', 'mean_sog_kn', 'min_dcpa']
    rows = {k: [] for k in keys}

    with torch.no_grad():
        for start in range(0, len(ds), args.batch_size):
            idxs = list(range(start, min(start + args.batch_size, len(ds))))
            obs, pred_gt, mask, counts = collate_fn([ds[i] for i in idxs])
            obs, pred_gt, mask = obs.to(device), pred_gt.to(device), mask.to(device)

            last_obs    = obs[:, :, -1, :2]
            target_disp = pred_gt - last_obs.unsqueeze(2)
            if args.arch == 'lstm':
                pred_disp = model(obs, mask=mask)
            else:
                pred_disp = model(obs, mask=mask, stats=stats)

            # identical construction to evaluate_cpagrn.py
            pred_abs   = (pred_disp   + last_obs.unsqueeze(2)).cpu().numpy()
            target_abs = (target_disp + last_obs.unsqueeze(2)).cpu().numpy()
            mask_np    = mask.cpu().numpy()

            pred_lon = denorm(pred_abs[..., 0],   lon_mean, lon_std)
            pred_lat = denorm(pred_abs[..., 1],   lat_mean, lat_std)
            true_lon = denorm(target_abs[..., 0], lon_mean, lon_std)
            true_lat = denorm(target_abs[..., 1], lat_mean, lat_std)

            err_m   = l2_meters(pred_lat, pred_lon, true_lat, true_lon)    # [B, N, T]
            err_deg = l2_degrees(pred_lat, pred_lon, true_lat, true_lon)   # [B, N, T]

            sog_kn = denorm(obs.cpu().numpy()[..., 2], sog_mean, sog_std).mean(axis=-1)  # [B, N]
            dcpa   = compute_min_dcpa(obs, mask).cpu().numpy()                            # [B, N]

            rows['err_m'].append(err_m[mask_np])           # [Nv_batch, T], window-major order
            rows['err_deg'].append(err_deg[mask_np])
            rows['mean_sog_kn'].append(sog_kn[mask_np])
            rows['min_dcpa'].append(dcpa[mask_np])

            for b, w in enumerate(idxs):
                c = int(counts[b])
                assert int(mask_np[b].sum()) == c == len(ds.vessel_ids_list[w]), f'count mismatch, window {w}'
                rows['window_idx'].append(np.full(c, w, dtype=np.int64))
                rows['day'].append(np.full(c, ds.day_list[w], dtype=np.int64))
                rows['frame_start'].append(np.full(c, ds.frame_start_list[w], dtype=np.int64))
                rows['vessel_id'].append(ds.vessel_ids_list[w])
                rows['n_in_window'].append(np.full(c, c, dtype=np.int64))

    out = {k: np.concatenate(v, axis=0) for k, v in rows.items()}
    out['err_m']   = out['err_m'].astype(np.float32)
    out['err_deg'] = out['err_deg'].astype(np.float32)
    out['seen_in_train'] = np.isin(out['vessel_id'], seen_ids)

    n = len(out['vessel_id'])
    ade_deg = out['err_deg'].mean()
    print(f'\nvessel-samples: {n:,}   windows: {len(ds):,}   '
          f'days: {sorted(set(int(d) for d in out["day"]))}')
    print(f'mean ADE = {ade_deg:.6f} deg  |  {out["err_m"].mean():.2f} m   '
          f'FDE = {out["err_deg"][:, -1].mean():.6f} deg  |  {out["err_m"][:, -1].mean():.2f} m')
    print(f'seen-in-train vessel-samples: {100 * out["seen_in_train"].mean():.1f}%   '
          f'median ADE = {np.median(out["err_m"].mean(axis=1)):.2f} m')
    print('>>> Compare mean ADE/FDE with the official evaluate script for this checkpoint.')

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'{args.tag}__{args.split}.npz')
    np.savez_compressed(
        path, **out,
        arch=np.array(args.arch), tag=np.array(args.tag), split=np.array(args.split),
        obs_len=np.array(args.obs_len), pred_len=np.array(args.pred_len),
    )
    print(f'saved -> {path}')


if __name__ == '__main__':
    main()