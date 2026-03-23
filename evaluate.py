"""
Evaluation Script with Best-of-K Sampling (Paper-Aligned)

Paper quote:
"During the inference stage, 20 samples are extracted from the learned 
bivariate Gaussian distribution and the closest sample to the ground-truth 
is used to calculate the performance index of the model."

This script implements proper paper-aligned evaluation:
- K=20 samples from predicted Gaussian distribution
- Selection of closest sample to ground truth (by minADE)
- Denormalization to real-world coordinates (degrees)
- Computation of minADE-20 and FDE (of best sample) in degrees

Expected metrics: 0.001-0.003° (as reported in paper)

Usage:
  python evaluate.py --dataset noaa_dec2021 --checkpoint checkpoints/AddGCN_10_10/noaa_dec2021/val_best.pth --split test --num_samples 20
"""

import argparse
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TrajectoryModel
from utils import TrajectoryDataset
from metrics import evaluate_best_of_k


def setup_args():
    parser = argparse.ArgumentParser(description='Trajectory Prediction Evaluation')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='noaa_dec2021',
                        help='Dataset name (folder under ./dataset/)')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate on')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--obs_len', type=int, default=10,
                        help='Observation sequence length')
    parser.add_argument('--pred_len', type=int, default=10,
                        help='Prediction sequence length')
    
    # Evaluation parameters
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples for best-of-K evaluation (K=20 in paper)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (keep 1 for compatibility)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    
    # Model architecture (must match training)
    parser.add_argument('--num_gcn_layers', type=int, default=1,
                        help='Number of GCN layers')
    parser.add_argument('--embedding_dims', type=int, default=64,
                        help='Embedding dimensions')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    
    return parser.parse_args()


def denormalize_predictions(V_pred, global_stats):
    """
    Denormalize Gaussian parameters from z-score to real-world coordinates.
    
    For Gaussian distribution, when denormalizing x = x_norm * std + mean:
    - μ_denorm = μ_norm * std + mean
    - σ_denorm = σ_norm * std
    
    Since the model outputs log(σ), we need:
    - log(σ_denorm) = log(σ_norm * std) = log(σ_norm) + log(std)
    
    Args:
        V_pred: [pred_len, N, 5] - Gaussian parameters (μx, μy, log(σx), log(σy), ρ)
        global_stats: dict with 'LON' and 'LAT' statistics
    
    Returns:
        V_pred_denorm: [pred_len, N, 5] - with denormalized μx, μy, log(σx), log(σy)
    """
    V_pred_denorm = V_pred.clone()
    
    # Get statistics
    lon_mean = global_stats['LON']['mean']
    lon_std = global_stats['LON']['std']
    lat_mean = global_stats['LAT']['mean']
    lat_std = global_stats['LAT']['std']
    
    # Get device from V_pred to avoid CPU/GPU mismatch
    device = V_pred.device
    
    # Denormalize μx (LON): μ_denorm = μ_norm * std + mean
    V_pred_denorm[:, :, 0] = V_pred[:, :, 0] * lon_std + lon_mean
    
    # Denormalize μy (LAT): μ_denorm = μ_norm * std + mean
    V_pred_denorm[:, :, 1] = V_pred[:, :, 1] * lat_std + lat_mean
    
    # Denormalize log(σx): log(σ_denorm) = log(σ_norm) + log(std)
    V_pred_denorm[:, :, 2] = V_pred[:, :, 2] + torch.log(torch.tensor(lon_std, device=device))
    
    # Denormalize log(σy): log(σ_denorm) = log(σ_norm) + log(std)
    V_pred_denorm[:, :, 3] = V_pred[:, :, 3] + torch.log(torch.tensor(lat_std, device=device))
    
    # ρ (correlation) is scale-invariant, remains unchanged
    # V_pred_denorm[:, :, 4] stays the same
    
    return V_pred_denorm


def denormalize_coordinates(V_target, global_stats):
    """
    Denormalize ground truth coordinates from z-score to real-world (degrees).
    
    Args:
        V_target: [pred_len, N, 2] - (LON, LAT) in normalized space
        global_stats: dict with 'LON' and 'LAT' statistics
    
    Returns:
        V_target_denorm: [pred_len, N, 2] - (LON, LAT) in degrees
    """
    V_target_denorm = V_target.clone()
    
    # Denormalize LON
    lon_mean = global_stats['LON']['mean']
    lon_std = global_stats['LON']['std']
    V_target_denorm[:, :, 0] = V_target[:, :, 0] * lon_std + lon_mean
    
    # Denormalize LAT
    lat_mean = global_stats['LAT']['mean']
    lat_std = global_stats['LAT']['std']
    V_target_denorm[:, :, 1] = V_target[:, :, 1] * lat_std + lat_mean
    
    return V_target_denorm


def evaluate_model(model, loader, device, global_stats, num_samples=20):
    """
    Evaluate model with best-of-K sampling in real-world coordinates (degrees).
    
    Args:
        global_stats: dict with normalization statistics from global_stats.json
    
    Returns:
        dict with average minADE and FDE in degrees (paper-aligned)
    """
    model.eval()
    
    all_min_ade = []
    all_fde = []
    total_sequences = 0
    
    print(f"\nEvaluating with best-of-{num_samples} sampling...")
    print(f"Metrics will be computed in real-world coordinates (degrees)")
    print(f"Total batches: {len(loader)}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            # Get data
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, V_tr = batch
            
            T = V_obs.shape[1]   # obs_len
            N = V_obs.shape[2]   # num vessels in this sequence
            
            # Identity matrices (original repo)
            identity_spatial  = torch.ones((T, N, N), device=device) * torch.eye(N, device=device)
            identity_temporal = torch.ones((N, T, T), device=device) * torch.eye(T, device=device)
            identity = [identity_spatial, identity_temporal]
            
            # Forward pass - get Gaussian parameters (in normalized space)
            # Model predicts VELOCITIES (matching V_tr training target)
            V_pred = model(V_obs, identity)  # [pred_len, N, 5]
            V_pred = V_pred.squeeze(0) if V_pred.dim() == 4 else V_pred  # [pred_len, N, 5]

            # Last observed absolute position (normalized): [N, 2]
            # obs_traj: [N, 4, obs_len] → last timestep → first 2 features (LON, LAT)
            last_obs = obs_traj.squeeze(0)[:, :2, -1]  # [N, 2]

            # Convert predicted velocity Gaussian parameters to absolute position Gaussian:
            # μ_abs[t] = last_obs + cumsum(μ_vel[0:t+1])
            # σ stays the same (velocity σ ≈ position σ for small steps)
            mu_vel = V_pred[:, :, :2]  # [pred_len, N, 2]
            mu_abs = torch.cumsum(mu_vel, dim=0) + last_obs.unsqueeze(0)  # [pred_len, N, 2]

            # Build absolute-position Gaussian parameters [pred_len, N, 5]
            V_pred_abs = V_pred.clone()
            V_pred_abs[:, :, :2] = mu_abs  # replace velocity means with absolute means

            # Target: absolute positions from pred_traj_gt [pred_len, N, 2]
            V_target = pred_traj_gt.squeeze(0).permute(2, 0, 1)[:, :, :2]  # [pred_len, N, 2]

            # DENORMALIZE to real-world coordinates (degrees)
            V_pred_denorm = denormalize_predictions(V_pred_abs, global_stats)
            V_target_denorm = denormalize_coordinates(V_target, global_stats)
            
            # Evaluate with best-of-K sampling in REAL-WORLD coordinates
            results = evaluate_best_of_k(V_pred_denorm, V_target_denorm, num_samples=num_samples)
            
            all_min_ade.append(results['minADE'])
            all_fde.append(results['FDE'])
            total_sequences += 1
    
    # Compute average metrics
    avg_min_ade = np.mean(all_min_ade)
    avg_fde = np.mean(all_fde)
    
    return {
        'minADE': avg_min_ade,
        'FDE': avg_fde,
        'total_sequences': total_sequences,
        'all_min_ade': all_min_ade,
        'all_fde': all_fde
    }


def main():
    args = setup_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    data_path = os.path.join('./dataset', args.dataset, args.split)
    print(f"\nLoading dataset from: {data_path}")
    
    dataset = TrajectoryDataset(
        data_path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=1
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Dataset loaded: {len(dataset)} sequences")
    
    # Load global statistics for denormalization
    global_stats_path = os.path.join('./dataset', args.dataset, 'global_stats.json')
    print(f"\nLoading global statistics from: {global_stats_path}")
    
    if not os.path.exists(global_stats_path):
        raise FileNotFoundError(
            f"Global statistics file not found: {global_stats_path}\n"
            f"Please run preprocessing first (preprocess_ais.py) to generate this file."
        )
    
    with open(global_stats_path, 'r') as f:
        global_stats = json.load(f)
    
    print(f"Global stats loaded:")
    print(f"  LON: mean={global_stats['LON']['mean']:.4f}, std={global_stats['LON']['std']:.4f}")
    print(f"  LAT: mean={global_stats['LAT']['mean']:.4f}, std={global_stats['LAT']['std']:.4f}")
    
    # Initialize model
    print(f"\nInitializing model...")
    model = TrajectoryModel(
        number_asymmetric_conv_layer=2,
        embedding_dims=args.embedding_dims,
        number_gcn_layers=args.num_gcn_layers,
        dropout=0.0,  # No dropout during evaluation
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        out_dims=5,  # Gaussian parameters: μx, μy, log(σx), log(σy), ρ
        num_heads=args.num_heads
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    print("Checkpoint loaded successfully!")
    
    # Evaluate with denormalization (paper-aligned)
    results = evaluate_model(model, loader, device, global_stats, num_samples=args.num_samples)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS (Best-of-{args.num_samples}) - PAPER ALIGNED")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset} ({args.split} split)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Total sequences: {results['total_sequences']}")
    print(f"\nMetrics (in real-world coordinates - degrees):")
    print(f"  minADE-{args.num_samples}: {results['minADE']:.6f}°")
    print(f"  FDE (of best sample):      {results['FDE']:.6f}°")
    print(f"\nNote: Metrics computed after denormalization using global_stats.json")
    print(f"      Expected range: 0.001-0.003° (as per paper)")
    print(f"{'='*80}")
    
    # Optionally save detailed results
    output_dir = os.path.dirname(args.checkpoint)
    results_file = os.path.join(output_dir, f'eval_results_{args.split}_K{args.num_samples}.txt')
    
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results (Best-of-{args.num_samples}) - PAPER ALIGNED\n")
        f.write(f"{'='*80}\n")
        f.write(f"Dataset: {args.dataset} ({args.split} split)\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Total sequences: {results['total_sequences']}\n\n")
        f.write(f"Metrics (in real-world coordinates - degrees):\n")
        f.write(f"minADE-{args.num_samples}: {results['minADE']:.6f}°\n")
        f.write(f"FDE (of best sample):      {results['FDE']:.6f}°\n\n")
        f.write(f"Note: Metrics computed after denormalization using global_stats.json\n")
        f.write(f"      Expected range: 0.001-0.003° (as per paper)\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()