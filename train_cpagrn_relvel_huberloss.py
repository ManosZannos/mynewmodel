"""
train_cpagrn_relvel_huberloss.py - Training script for the relative-velocity
CONTROL ablation, matched to the FINAL headline recipe (gru2, lr=5e-4,
Huber delta=0.05).

Differs from train_cpagrn_huberloss.py in two things only:
  1. the model comes from model_cpagrn_relvel_huberloss (CPA channels TCPA/DCPA
     replaced by the standardized relative-velocity vector, same edge_dim=7);
  2. before training, a single constant `vel_scale` is estimated from the
     TRAINING data and written into the model buffer. The CPU RNG state is
     saved and restored around this estimation, and the model is built BEFORE
     it, so with the same --seed this run has exactly the same initial weights
     and the same batch order as the headline run (only the 2 edge channels
     differ).

Checkpoints must be evaluated with evaluate_cpagrn_relvel.py (the state_dict
contains the extra buffer cpa_features.vel_scale, so loading into the headline
model fails loudly instead of silently mixing architectures).

Usage (run the same seeds as the headline):
    python train_cpagrn_relvel_huberloss.py --obs_len 10 --pred_len 10 --gru_layers 2 \\
        --top_k 10 --huber_delta 0.05 --seed 42 --lr 5e-4 \\
        --tag CPAGRN_v5_relvel_huber_d05_gru2_lr5e4_obs10_pred10_s42 --gpu_num <GPU>
"""

from __future__ import annotations
import os
import sys
import math
import time
import argparse
import logging

import torch
import torch.nn as nn
import numpy as np

from dataset import get_dataloaders
from model_cpagrn_relvel_huberloss import CPAGRN, huber_cpagrn_loss, estimate_rel_vel_scale


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',       type=str,   default='dataset/noaa_dec2021_1min')
    p.add_argument('--obs_len',        type=int,   default=10)
    p.add_argument('--pred_len',       type=int,   default=10)
    p.add_argument('--d_model',        type=int,   default=64)
    p.add_argument('--gru_layers',     type=int,   default=2)
    p.add_argument('--top_k',          type=int,   default=10)
    p.add_argument('--huber_delta',    type=float, default=0.05)
    p.add_argument('--epochs',         type=int,   default=200)
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--lr',             type=float, default=5e-4)   # headline recipe default
    p.add_argument('--weight_decay',   type=float, default=0.0)
    p.add_argument('--optimizer',      type=str,   default='adam', choices=['adam', 'adamw'])
    p.add_argument('--clip_grad',      type=float, default=1.0)
    p.add_argument('--gpu_num',        type=int,   default=0)
    p.add_argument('--tag',            type=str,   default='CPAGRN_relvel_huber_obs10_pred10')
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--log_every',      type=int,   default=10)
    p.add_argument('--scale_batches',  type=int,   default=100,
                    help='train batches used to estimate vel_scale (relvel only)')
    return p.parse_args()


def get_lr(epoch, args):
    warmup = 10
    if epoch < warmup:
        return args.lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, args.epochs - warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_epoch(loader, model, optimizer, device, args, stats, train: bool):
    model.train(train)
    total_loss = 0.0
    n_batches  = 0

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for obs, pred_gt, mask, _ in loader:
            obs     = obs.to(device)
            pred_gt = pred_gt.to(device)
            mask    = mask.to(device)

            last_obs    = obs[:, :, -1, :2]
            target_disp = pred_gt - last_obs.unsqueeze(2)

            pred_disp = model(obs, mask=mask, stats=stats)

            loss = huber_cpagrn_loss(pred_disp, target_disp, mask, delta=args.huber_delta)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    ckpt_dir = os.path.join('checkpoints', args.tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(ckpt_dir, 'train.log')),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger()
    log.info(f'Tag: {args.tag}')
    log.info(f'Model: model_cpagrn_relvel_huberloss.py (relative-velocity control: '
              f'edges [dvx, dvy, dist, sin/cos bearing, dhdg, |dhdg|]) + '
              f'Huber loss (delta={args.huber_delta})')
    log.info(f'Args: {vars(args)}')

    train_loader, val_loader, _, stats = get_dataloaders(
        data_dir   = args.data_dir,
        obs_len    = args.obs_len,
        pred_len   = args.pred_len,
        batch_size = args.batch_size,
    )
    log.info(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

    model = CPAGRN(
        feature_size = 4,
        d_model      = args.d_model,
        gru_layers   = args.gru_layers,
        pred_len     = args.pred_len,
        top_k        = args.top_k,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Parameters: {n_params:,}')

    # -- relative-velocity scale: data-driven constant, fixed before training --
    rng_state = torch.get_rng_state()          # keep batch order identical to the headline run
    vel_scale = estimate_rel_vel_scale(train_loader, n_batches=args.scale_batches)
    torch.set_rng_state(rng_state)
    model.cpa_features.vel_scale.fill_(vel_scale)
    log.info(f'Relative-velocity scale (per-component RMS, z-score units/step): {vel_scale:.6f}')

    # ── Calibration print (epoch-0, untrained-weights) — see module docstring ──
    with torch.no_grad():
        obs0, pred_gt0, mask0, _ = next(iter(train_loader))
        obs0, pred_gt0, mask0 = obs0.to(device), pred_gt0.to(device), mask0.to(device)
        last_obs0    = obs0[:, :, -1, :2]
        target_disp0 = pred_gt0 - last_obs0.unsqueeze(2)
        pred_disp0   = model(obs0, mask=mask0, stats=stats)
        dist0 = (pred_disp0 - target_disp0).norm(dim=-1)
        m0 = mask0.unsqueeze(-1).expand_as(dist0)
        vals = dist0[m0].detach().cpu().numpy()
        p10, p25, p50, p75 = np.percentile(vals, [10, 25, 50, 75])
        log.info(
            f'[UNTRAINED-MODEL calibration] displacement-error magnitude '
            f'(first batch, z-score units): p10={p10:.4f}  p25={p25:.4f}  '
            f'p50={p50:.4f}  p75={p75:.4f}'
        )
        log.info(
            f'  These are UNTRAINED-model errors (much larger than a converged '
            f'model will produce) — use only as a rough scale sanity check, not '
            f'as the calibration target itself. Current --huber_delta={args.huber_delta}.'
        )

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val   = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        lr = get_lr(epoch, args)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        t0 = time.time()
        train_loss = run_epoch(train_loader, model, optimizer, device, args, stats, train=True)
        val_loss   = run_epoch(val_loader,   model, optimizer, device, args, stats, train=False)
        elapsed    = time.time() - t0

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            log.info(
                f'Epoch {epoch+1:>3}/{args.epochs} | lr={lr:.2e} | '
                f'train={train_loss:.6f} | val={val_loss:.6f} | t={elapsed:.1f}s'
            )

        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch':    epoch + 1,
                'model':    model.state_dict(),
                'val_loss': val_loss,
                'args':     vars(args),
                'stats':    stats,
            }, os.path.join(ckpt_dir, 'val_best.pth'))

        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
        }, os.path.join(ckpt_dir, 'latest.pth'))

    log.info(f'Done. Best val: {best_val:.6f} at epoch {best_epoch}')


if __name__ == '__main__':
    main()
