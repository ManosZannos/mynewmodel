"""
evaluate.py — Paper-aligned evaluation for SMCHN vs DualSTMA

Normalization (METO-S2S, from max_min.json):
  JSON field order: [timestamp, lat_norm, lon_norm, sog_norm, heading_norm, ...]
  lat_norm = (lat - 20.90883) / 28.32044
  lon_norm = (lon - (-133.29703)) / 72.60811

  Denormalization:
    lat = lat_norm * 28.32044 + 20.90883
    lon = lon_norm * 72.60811 + (-133.29703)

CSV field order (from convert_json_to_csv.py):
  frame_id, vessel_id, LON, LAT, SOG, Heading
  where LON = JSON[2] = lon_norm, LAT = JSON[1] = lat_norm

Usage:
  python evaluate.py \\
    --dataset marinecadastre_2021 \\
    --checkpoint checkpoints/SMCHN_dualstma/marinecadastre_2021/val_best.pth \\
    --split test
"""

import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TrajectoryModel
from utils import TrajectoryDataset


# ── METO-S2S normalization constants (from max_min.json) ──────────────────
LAT_MIN   =  20.90883
LAT_RANGE =  28.32044   # lat_max - lat_min = 49.22927 - 20.90883
LON_MIN   = -133.29703
LON_RANGE =  72.60811   # lon_max - lon_min = -60.68892 - (-133.29703)

# CSV column order: frame_id, vessel_id, LON, LAT, SOG, Heading
# LON col (index 0 in features) = lon_norm  → denorm with LON constants
# LAT col (index 1 in features) = lat_norm  → denorm with LAT constants


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str, default='marinecadastre_2021')
    parser.add_argument('--split',          type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--checkpoint',     type=str, required=True)
    parser.add_argument('--obs_len',        type=int, default=10)
    parser.add_argument('--pred_len',       type=int, default=5)
    parser.add_argument('--batch_size',     type=int, default=1)
    parser.add_argument('--device',         type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--embedding_dims', type=int, default=64)
    parser.add_argument('--num_gcn_layers', type=int, default=1)
    parser.add_argument('--num_heads',      type=int, default=4)
    return parser.parse_args()


def evaluate_model(model, loader, device):
    """
    Evaluate deterministic SMCHN model (out_dims=2).

    Model predicts velocity (lon_vel, lat_vel) directly.
    ADE/FDE computed in real-world degrees using METO-S2S denormalization.
    """
    model.eval()

    all_ade = []
    all_fde = []
    step_ade = [[] for _ in range(5)]
    total   = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = [t.to(device) for t in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
                non_linear_ped, loss_mask, V_obs, V_tr = batch

            T = V_obs.shape[1]
            N = V_obs.shape[2]

            identity_spatial  = torch.ones((T, N, N), device=device) * torch.eye(N, device=device)
            identity_temporal = torch.ones((N, T, T), device=device) * torch.eye(T, device=device)
            identity = [identity_spatial, identity_temporal]

            # Forward pass → [pred_len, N, 2] velocities (deterministic)
            V_pred = model(V_obs, identity)
            V_pred = V_pred.squeeze(0) if V_pred.dim() == 4 else V_pred

            # Last observed absolute position (norm space): [N, 2]
            last_obs = obs_traj.squeeze(0)[:, :2, -1]

            # Cumsum of predicted velocities → absolute positions in norm space
            pred_abs = torch.cumsum(V_pred, dim=0) + last_obs.unsqueeze(0)  # [pred_len, N, 2]

            # Ground truth absolute positions (norm space): [pred_len, N, 2]
            gt_abs = pred_traj_gt.squeeze(0).permute(2, 0, 1)[:, :, :2]

            # Denormalize both to degrees
            pred_lon = pred_abs[:, :, 0] * LON_RANGE + LON_MIN
            pred_lat = pred_abs[:, :, 1] * LAT_RANGE + LAT_MIN
            gt_lon   = gt_abs[:, :, 0]   * LON_RANGE + LON_MIN
            gt_lat   = gt_abs[:, :, 1]   * LAT_RANGE + LAT_MIN

            # Displacement per step: [pred_len, N]
            disp = torch.sqrt((pred_lon - gt_lon)**2 + (pred_lat - gt_lat)**2)

            all_ade.append(disp.mean().item())
            all_fde.append(disp[-1].mean().item())
            for t in range(min(5, disp.shape[0])):
                step_ade[t].append(disp[t].mean().item())
            total += 1

    return {
        'ADE':             float(np.mean(all_ade)),
        'FDE':             float(np.mean(all_fde)),
        'step_ade':        [float(np.mean(s)) for s in step_ade],
        'total_sequences': total,
    }


def main():
    args   = setup_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CSV dataset
    data_path = os.path.join('./dataset', args.dataset, args.split)
    print(f"\nLoading dataset from: {data_path}")
    dataset = TrajectoryDataset(
        data_path, obs_len=args.obs_len, pred_len=args.pred_len, skip=1
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    print(f"Sequences: {len(dataset)}")

    print(f"\nNormalization (METO-S2S max_min.json):")
    print(f"  LAT: min={LAT_MIN}, range={LAT_RANGE}  → [{LAT_MIN:.2f}, {LAT_MIN+LAT_RANGE:.2f}]°")
    print(f"  LON: min={LON_MIN}, range={LON_RANGE} → [{LON_MIN:.2f}, {LON_MIN+LON_RANGE:.2f}]°")

    # Initialize model (deterministic, out_dims=2)
    model = TrajectoryModel(
        number_asymmetric_conv_layer=2,
        embedding_dims=args.embedding_dims,
        number_gcn_layers=args.num_gcn_layers,
        dropout=0.0,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        out_dims=2,
        num_heads=args.num_heads
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    print(f"\nCheckpoint loaded: {args.checkpoint}")

    # Evaluate
    results = evaluate_model(model, loader, device)

    # DualSTMA baseline (Huang et al., 2024, Table 2)
    dualstma_ade = {10: 0.000609, 20: 0.000853, 30: 0.001157, 40: 0.001522, 50: 0.002436}
    dualstma_fde = {10: 0.000807, 20: 0.001164, 30: 0.001771, 40: 0.002378, 50: 0.003946}

    horizon = args.pred_len * 10
    ade_d   = dualstma_ade.get(horizon)
    fde_d   = dualstma_fde.get(horizon)

    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS — DualSTMA COMPARISON")
    print(f"{'='*70}")
    print(f"Dataset:   {args.dataset} ({args.split})")
    print(f"Sequences: {results['total_sequences']}")
    print(f"Model:     deterministic (out_dims=2, Huber position loss)")
    print()
    print(f"  {'Horizon':>8} | {'SMCHN (ours)':>14} | {'DualSTMA':>12} | {'Ratio':>8}")
    print(f"  {'-'*52}")

    for t, step_val in enumerate(results['step_ade']):
        h   = (t + 1) * 10
        ref = dualstma_ade.get(h)
        ratio = f"{step_val/ref:.2f}x" if ref else "N/A"
        ref_s = f"{ref}" if ref else "N/A"
        print(f"  {'ADE '+str(h)+'min':>8} | {step_val:>13.6f}° | {ref_s:>12} | {ratio:>8}")

    print(f"  {'-'*52}")
    ade_ratio = f"{results['ADE']/ade_d:.2f}x" if ade_d else "N/A"
    fde_ratio = f"{results['FDE']/fde_d:.2f}x" if fde_d else "N/A"
    print(f"  {'ADE':>8} | {results['ADE']:>13.6f}° | {ade_d if ade_d else 'N/A':>12} | {ade_ratio:>8}")
    print(f"  {'FDE':>8} | {results['FDE']:>13.6f}° | {fde_d if fde_d else 'N/A':>12} | {fde_ratio:>8}")
    print(f"{'='*70}")

    # Save results
    out_dir  = os.path.dirname(args.checkpoint)
    out_file = os.path.join(out_dir, f'eval_{args.split}.txt')
    with open(out_file, 'w') as f:
        f.write(f"SMCHN vs DualSTMA — {args.split}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Horizon: {horizon}min\n\n")
        f.write(f"ADE: {results['ADE']:.6f}° (DualSTMA: {ade_d})\n")
        f.write(f"FDE: {results['FDE']:.6f}° (DualSTMA: {fde_d})\n")
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()