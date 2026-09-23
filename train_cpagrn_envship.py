"""
train_cpagrn_envship.py — Trains the headline recipe (gru2 + Huber loss,
delta=0.05) on the EnvShip NOAA Track A benchmark, for external validation
against their published baselines (TCN, LSTM+SDF, LSTM+Gated-Neighbor-Attn).

PHASE 1 ONLY — see dataset_envship.py docstring. This trains the
trajectory-only pathway (N=1 per sample, no neighbor context yet).
Valid for comparison against EnvShip's LSTM/TCN/Transformer-NAR baselines.
NOT yet valid for comparison against their neighbor-aware baselines —
that needs Phase 2 (dataset_envship.load_neighbor_context, not yet written).

Model architecture (model_cpagrn_huberloss.CPAGRN) is UNCHANGED from the
official San Diego headline — only the data source differs. obs_len/pred_len
are fixed at 30/30 (EnvShip Track A's own definition, 20s sampling) and are
NOT CLI-configurable here, unlike train_cpagrn.py's --obs_len/--pred_len.

Checkpoints are written to checkpoints_envship/, never checkpoints/ —
this is hardcoded (not a --tag-dependent default) specifically so this run
can never collide with or overwrite an official San Diego checkpoint dir.

Usage:
    python train_cpagrn_envship.py \
        --data_dir dataset/envship_noaa_track_a/track_a_short-term_Cross-domain_Datasets/noaa_track_v1 \
        --huber_delta 0.05 --seed 42 --lr 5e-4 \
        --tag CPAGRN_v5_huber_d05_gru2_lr5e4_envship_noaa_obs30_pred30_s42 \
        --gpu_num <GPU>
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

from dataset_envship import get_dataloaders_envship, T_OBS_ENVSHIP, T_PRED_ENVSHIP
from model_cpagrn_huberloss import CPAGRN, huber_cpagrn_loss

CKPT_ROOT = 'checkpoints_envship'  # hardcoded — never 'checkpoints/', see module docstring


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',       type=str,   required=True,
                    help='Path to .../noaa_track_v1 (containing train/val/test/part-000.csv.gz)')
    p.add_argument('--d_model',        type=int,   default=64)
    p.add_argument('--gru_layers',     type=int,   default=2)     # headline recipe
    p.add_argument('--top_k',          type=int,   default=10)
    p.add_argument('--huber_delta',    type=float, default=0.05)  # headline recipe
    p.add_argument('--epochs',         type=int,   default=200)
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--lr',             type=float, default=5e-4)  # headline recipe
    p.add_argument('--weight_decay',   type=float, default=0.0)
    p.add_argument('--optimizer',      type=str,   default='adam', choices=['adam', 'adamw'])
    p.add_argument('--clip_grad',      type=float, default=1.0)
    p.add_argument('--gpu_num',        type=int,   default=0)
    p.add_argument('--tag',            type=str,   required=True,
                    help='Must contain "envship_noaa" by convention — see file-separation runbook.')
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--log_every',      type=int,   default=10)
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
    if 'envship' not in args.tag:
        print(f'WARNING: --tag "{args.tag}" does not contain "envship" — '
              f'this breaks the file-separation convention. Continuing anyway.')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    ckpt_dir = os.path.join(CKPT_ROOT, args.tag)
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
    log.info(f'Checkpoint root: {CKPT_ROOT}/ (hardcoded, separate from official checkpoints/)')
    log.info(f'Data source: EnvShip NOAA Track A, PHASE 1 (trajectory-only, N=1 per sample)')
    log.info(f'Model: model_cpagrn_huberloss.py (gru2 architecture, unchanged) + '
              f'Huber loss (delta={args.huber_delta})')
    log.info(f'Args: {vars(args)}')

    train_loader, val_loader, _, stats = get_dataloaders_envship(
        data_dir   = args.data_dir,
        batch_size = args.batch_size,
    )
    log.info(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')
    log.info(f'obs_len={T_OBS_ENVSHIP} pred_len={T_PRED_ENVSHIP} (fixed by EnvShip Track A)')

    model = CPAGRN(
        feature_size = 4,
        d_model      = args.d_model,
        gru_layers   = args.gru_layers,
        pred_len     = T_PRED_ENVSHIP,
        top_k        = args.top_k,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Parameters: {n_params:,}')

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
