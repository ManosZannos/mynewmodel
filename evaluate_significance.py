"""
evaluate_significance.py — Paired statistical significance test: headline
CPA-GRN (gru2) vs Vanilla LSTM baseline, on the SAME test scenes.

Motivation: the project's "confirmed" standard so far has been "3/3 seeds
beat the baseline, low CV" — a reasonable but informal bar. For publication,
a paired significance test (Wilcoxon signed-rank, non-parametric — makes no
assumption that per-vessel errors are normally distributed, which they
usually aren't) on matched per-vessel ADE/FDE gives a defensible p-value
instead.

Why PAIRED and why Wilcoxon:
  - Paired: LSTM and gru2 are evaluated on the exact SAME (scene, vessel)
    samples, in the same order (test_loader has shuffle=False and both
    models use the same obs_len/pred_len/data_dir, so sample order is
    identical run-to-run) — this lets us test "for the same vessel in the
    same scene, is gru2's error smaller?" rather than comparing two
    unrelated distributions. Paired tests have much higher statistical
    power than unpaired ones for exactly this reason.
  - Wilcoxon signed-rank (not a paired t-test): per-vessel displacement
    error is a non-negative, right-skewed quantity (it's a Euclidean norm),
    not remotely Gaussian — Wilcoxon only assumes the distribution of
    PAIRED DIFFERENCES is symmetric, a much weaker and more plausible
    assumption here.

Usage:
    python evaluate_significance.py \
        --tag_a LSTM_obs10_pred10_s42 --tag_b CPAGRN_v5_gru2_lr5e4_obs10_pred10_s42 \
        --model_a lstm --model_b gru2 \
        --obs_len 10 --pred_len 10 --split test --gpu_num <GPU>

Reports paired Wilcoxon signed-rank test on final-horizon ADE (per vessel,
averaged over the whole pred_len) and on FDE (last-step error), with the
median paired difference and its sign, so the direction of any effect is
explicit alongside the p-value.
"""

from __future__ import annotations
import os
import argparse
import numpy as np
from scipy.stats import wilcoxon

import torch
from dataset import get_dataloaders, denorm


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tag_a',    type=str, required=True, help='Checkpoint tag for model A (e.g. LSTM baseline)')
    p.add_argument('--tag_b',    type=str, required=True, help='Checkpoint tag for model B (e.g. gru2 headline)')
    p.add_argument('--model_a',  type=str, required=True, choices=['lstm', 'gru2'])
    p.add_argument('--model_b',  type=str, required=True, choices=['lstm', 'gru2'])
    p.add_argument('--label_a',  type=str, default=None, help='Display name for model A (default: tag_a)')
    p.add_argument('--label_b',  type=str, default=None, help='Display name for model B (default: tag_b)')
    p.add_argument('--split',    type=str, default='test', choices=['val', 'test'])
    p.add_argument('--data_dir', type=str, default='dataset/noaa_dec2021_1min')
    p.add_argument('--obs_len',  type=int, default=10)
    p.add_argument('--pred_len', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--gpu_num',  type=int, default=0)
    return p.parse_args()


def load_model(model_type: str, tag: str, args, device):
    ckpt_path = os.path.join('checkpoints', tag, 'val_best.pth')
    assert os.path.exists(ckpt_path), f'Not found: {ckpt_path}'
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    stats = ckpt.get('stats', None)

    if model_type == 'lstm':
        from model_lstm import VanillaLSTM
        model = VanillaLSTM(
            feature_size = 4,
            hidden_size  = saved.get('hidden_size', 64),
            num_layers   = saved.get('num_layers',  1),
            pred_len     = saved.get('pred_len',    args.pred_len),
        ).to(device)
    elif model_type == 'gru2':
        from model_cpagrn import CPAGRN
        model = CPAGRN(
            feature_size = 4,
            d_model      = saved.get('d_model',    64),
            gru_layers   = saved.get('gru_layers', 2),
            pred_len     = saved.get('pred_len',   args.pred_len),
            top_k        = saved.get('top_k',      10),
        ).to(device)
    else:
        raise ValueError(model_type)

    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, stats


def l2_degrees(pred_lat, pred_lon, true_lat, true_lon):
    return np.sqrt((pred_lat - true_lat) ** 2 + (pred_lon - true_lon) ** 2)


def collect_errors(model, model_type, loader, stats, device, T):
    """Returns (ade_per_vessel, fde_per_vessel) as flat numpy arrays, in the
    exact order the loader yields (scene, vessel) pairs — this ordering is
    what makes the two models' arrays pairable, PROVIDED both are run with
    the same loader construction (same obs_len/pred_len/data_dir/batch_size,
    shuffle=False for val/test — already the get_dataloaders default)."""
    lon_mean, lon_std = stats['LON']['mean'], stats['LON']['std']
    lat_mean, lat_std = stats['LAT']['mean'], stats['LAT']['std']

    ade_list, fde_list = [], []

    with torch.no_grad():
        for obs, pred_gt, mask, _ in loader:
            obs, pred_gt, mask = obs.to(device), pred_gt.to(device), mask.to(device)
            last_obs    = obs[:, :, -1, :2]
            target_disp = pred_gt - last_obs.unsqueeze(2)

            if model_type == 'lstm':
                pred_disp = model(obs, mask=mask)
            else:
                pred_disp = model(obs, mask=mask, stats=stats)

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
                    err = l2_degrees(pred_lat[b, n, :], pred_lon[b, n, :],
                                      true_lat[b, n, :], true_lon[b, n, :])
                    ade_list.append(err.mean())
                    fde_list.append(err[-1])

    return np.array(ade_list), np.array(fde_list)


def run_paired_test(name, err_a, err_b, label_a, label_b):
    assert len(err_a) == len(err_b), (
        f'{name}: sample count mismatch ({len(err_a)} vs {len(err_b)}) — '
        f'the two models were NOT evaluated on matched samples. Check that '
        f'both used the same --obs_len/--pred_len/--data_dir/--batch_size.'
    )
    diff = err_a - err_b  # positive => A worse (larger error) than B
    n = len(diff)
    n_nonzero = int(np.sum(diff != 0))

    stat, p = wilcoxon(err_a, err_b, alternative='two-sided', zero_method='wilcox')

    median_diff = np.median(diff)
    direction = (f'{label_b} has LOWER error (better) than {label_a}' if median_diff > 0
                 else f'{label_a} has LOWER error (better) than {label_b}' if median_diff < 0
                 else 'no median difference')

    print(f'\n--- {name} ---')
    print(f'  n = {n} paired samples ({n_nonzero} non-tied)')
    print(f'  {label_a} median: {np.median(err_a):.6f}°   mean: {err_a.mean():.6f}°')
    print(f'  {label_b} median: {np.median(err_b):.6f}°   mean: {err_b.mean():.6f}°')
    print(f'  Median paired difference (A - B): {median_diff:+.6f}°  → {direction}')
    print(f'  Wilcoxon signed-rank statistic: {stat:.1f}')
    print(f'  p-value (two-sided): {p:.4e}')
    if p < 0.001:
        sig = 'p < 0.001 — highly significant'
    elif p < 0.01:
        sig = 'p < 0.01 — significant'
    elif p < 0.05:
        sig = 'p < 0.05 — significant'
    else:
        sig = 'NOT significant at α=0.05'
    print(f'  → {sig}')


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    label_a = args.label_a or args.tag_a
    label_b = args.label_b or args.tag_b

    model_a, stats_a = load_model(args.model_a, args.tag_a, args, device)
    model_b, stats_b = load_model(args.model_b, args.tag_b, args, device)

    # Use model B's stats for both (should be identical z-score stats either
    # way, since both were trained on the same data_dir/train split) — kept
    # separate variables only for clarity of provenance.
    _, val_loader, test_loader, file_stats = get_dataloaders(
        args.data_dir, args.obs_len, args.pred_len, args.batch_size
    )
    stats = stats_b if stats_b is not None else (stats_a if stats_a is not None else file_stats)
    loader = test_loader if args.split == 'test' else val_loader

    print(f'Evaluating {label_a} ({args.model_a}) ...')
    ade_a, fde_a = collect_errors(model_a, args.model_a, loader, stats, device, args.pred_len)
    print(f'Evaluating {label_b} ({args.model_b}) ...')
    ade_b, fde_b = collect_errors(model_b, args.model_b, loader, stats, device, args.pred_len)

    print(f'\n{"="*60}')
    print(f'  Paired significance test | {label_a}  vs  {label_b}')
    print(f'  split={args.split}  obs_len={args.obs_len}  pred_len={args.pred_len}')
    print('='*60)
    run_paired_test('ADE (mean over horizon, per vessel)', ade_a, ade_b, label_a, label_b)
    run_paired_test('FDE (final-step error, per vessel)',  fde_a, fde_b, label_a, label_b)
    print('='*60)


if __name__ == '__main__':
    main()
