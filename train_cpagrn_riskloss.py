"""
train_cpagrn_riskloss.py — CPA-GRN (gru2 backbone) + risk-weighted loss.

Idea #3 from the "how does the model actually USE the CPA signal" family
of proposals (as opposed to top_k / gru_layers / velocity-smoothing, which
only change HOW MUCH context the model sees).

Motivation:
  - Documented ablation finding: TCPA/DCPA contribute more to FDE than ADE
    — i.e. the CPA signal is most informative near the closest-approach
    moment, for the vessels actually involved in a close encounter.
  - The current MSE loss weighs every vessel equally regardless of whether
    it is in a genuine close-quarters encounter or just cruising alone.
    With ~207 vessels/window but only top_k=10-15 neighbors attended to,
    many training examples are "easy" (no real interaction) and dilute
    the training signal for the hard, encounter-heavy cases.
  - This is architecturally ORTHOGONAL to the already-explored "how much
    context" family (top_k, gru_layers, velocity smoothing) — it changes
    what the optimizer is pushed to get right, not what the model can see.
    Expected to not be redundant with gru2 the way top_k+gru2 was.

Mechanism:
  For each vessel, at the LAST OBSERVED timestep, compute DCPA to every
  other (masked, valid) vessel using the same CPA math as CPAFeatures in
  model_cpagrn.py. Take the minimum DCPA over neighbors with TCPA > 0
  (i.e. still approaching, not already receding) as that vessel's "risk
  level" for this training window. Convert to a per-vessel loss weight:

      weight = 1 + risk_boost * exp(-min_dcpa / risk_scale)

  Vessels with a close, approaching neighbor (small DCPA) get weight up
  to (1 + risk_boost); vessels with no close encounter get weight ~1
  (unchanged from standard MSE). This is purely a LOSS reweighting — the
  model architecture, forward pass, and therefore evaluate_cpagrn.py are
  completely unchanged. Only training differs.

  IMPORTANT: risk_scale is in the same units as DCPA computed by
  CPAFeatures, i.e. z-score-normalized position units — NOT nautical
  miles or degrees. There is no universally "correct" default. This
  script prints the empirical DCPA distribution (median, p10, p25) from
  the first training batch so you can sanity-check that risk_scale sits
  in a sensible part of that distribution (roughly: pick something near
  the p10-p25 mark, so only genuinely close encounters get boosted).

Usage:
    python train_cpagrn_riskloss.py --obs_len 10 --pred_len 10 --gru_layers 2 \
        --top_k 10 --seed 42 --risk_boost 2.0 --risk_scale <see printed stats> \
        --tag CPAGRN_v5_gru2_riskloss_obs10_pred10_s42 --gpu_num <GPU>

Evaluation: use the EXISTING evaluate_cpagrn.py unchanged — same model class,
same checkpoint format, no risk-loss-specific logic needed at test time.
    python evaluate_cpagrn.py --tag CPAGRN_v5_gru2_riskloss_obs10_pred10_s42 \
        --obs_len 10 --pred_len 10 --split test --gpu_num <GPU>
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
from model_cpagrn import CPAGRN


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',       type=str,   default='dataset/noaa_dec2021_1min')
    p.add_argument('--obs_len',        type=int,   default=10)
    p.add_argument('--pred_len',       type=int,   default=10)
    p.add_argument('--d_model',        type=int,   default=64)
    p.add_argument('--gru_layers',     type=int,   default=2)
    p.add_argument('--top_k',          type=int,   default=10)
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--epochs',         type=int,   default=200)
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--clip_grad',      type=float, default=1.0)
    p.add_argument('--gpu_num',        type=int,   default=0)
    p.add_argument('--tag',            type=str,   default='CPAGRN_v5_gru2_riskloss_obs10_pred10')
    p.add_argument('--log_every',      type=int,   default=10)
    # Risk-weighting hyperparameters (Idea #3)
    p.add_argument('--risk_boost',     type=float, default=2.0,
                    help='Max additional weight for the closest-encounter vessels (weight range: [1, 1+risk_boost]).')
    p.add_argument('--risk_scale',     type=float, default=1.0,
                    help='DCPA decay scale, in z-score position units. Check the printed DCPA '
                         'distribution stats at startup and set this near the p10-p25 mark.')
    return p.parse_args()


def get_lr(epoch, args):
    warmup = 10
    if epoch < warmup:
        return args.lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, args.epochs - warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def compute_risk_weights(obs: torch.Tensor, mask: torch.Tensor,
                          risk_boost: float, risk_scale: float,
                          eps: float = 1e-6) -> torch.Tensor:
    """
    Per-vessel risk weight based on DCPA to the closest APPROACHING
    (TCPA > 0) neighbor at the last observed timestep. Same CPA math as
    CPAFeatures in model_cpagrn.py, computed here standalone (no need to
    touch the model itself).

    obs:  [B, N, T, 4]  (LON, LAT, SOG, Heading — z-score)
    mask: [B, N] bool
    Returns: weight [B, N], in range [1, 1+risk_boost]
    """
    B, N, T, _ = obs.shape
    pos_last = obs[:, :, -1, :2]
    vel_last = obs[:, :, -1, :2] - obs[:, :, -2, :2] if T >= 2 else torch.zeros_like(pos_last)

    pos_i = pos_last.unsqueeze(2).expand(B, N, N, 2)
    pos_j = pos_last.unsqueeze(1).expand(B, N, N, 2)
    vel_i = vel_last.unsqueeze(2).expand(B, N, N, 2)
    vel_j = vel_last.unsqueeze(1).expand(B, N, N, 2)

    r = pos_j - pos_i
    v = vel_j - vel_i
    v_sq = (v * v).sum(dim=-1) + eps
    tcpa = (-(r * v).sum(dim=-1) / v_sq).clamp(-5.0, 5.0)
    dcpa = (r + tcpa.unsqueeze(-1) * v).norm(dim=-1).clamp(0.0, 10.0)

    approaching = tcpa > 0
    valid = approaching
    if mask is not None:
        mask_j = mask.unsqueeze(1).expand(B, N, N)
        valid = valid & mask_j
    eye = torch.eye(N, device=obs.device, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
    valid = valid & (~eye)

    dcpa_masked = dcpa.masked_fill(~valid, float('inf'))
    min_dcpa, _ = dcpa_masked.min(dim=-1)          # [B, N]
    min_dcpa = torch.nan_to_num(min_dcpa, nan=10.0, posinf=10.0)

    weight = 1.0 + risk_boost * torch.exp(-min_dcpa / risk_scale)
    return weight, min_dcpa


def cpagrn_loss_riskweighted(pred_disp: torch.Tensor, target_disp: torch.Tensor,
                              mask: torch.Tensor, risk_weight: torch.Tensor) -> torch.Tensor:
    sq_err = (pred_disp - target_disp) ** 2
    sq_err = sq_err.sum(dim=-1)                     # [B, N, pred_len]
    w = risk_weight.unsqueeze(-1).expand_as(sq_err)  # [B, N, pred_len]
    m = mask.unsqueeze(-1).expand_as(sq_err)
    weighted_err = sq_err * w
    # Weighted mean, normalized by sum of weights (keeps loss scale comparable
    # to plain MSE — weight~1 for most vessels, up to 1+risk_boost for risky ones)
    return weighted_err[m].sum() / w[m].sum()


def run_epoch(loader, model, optimizer, device, args, stats, train: bool,
              print_dcpa_stats: bool = False):
    model.train(train)
    total_loss = 0.0
    n_batches  = 0

    for obs, pred_gt, mask, _ in loader:
        obs     = obs.to(device)
        pred_gt = pred_gt.to(device)
        mask    = mask.to(device)

        last_obs    = obs[:, :, -1, :2]
        target_disp = pred_gt - last_obs.unsqueeze(2)

        risk_weight, min_dcpa = compute_risk_weights(
            obs, mask, args.risk_boost, args.risk_scale
        )

        if print_dcpa_stats:
            valid_dcpa = min_dcpa[mask].detach().cpu().numpy()
            valid_dcpa = valid_dcpa[valid_dcpa < 10.0]  # drop "no approaching neighbor" sentinel
            if len(valid_dcpa) > 0:
                logging.getLogger().info(
                    f'DCPA distribution (first batch, approaching neighbors only, n={len(valid_dcpa)}): '
                    f'p10={np.percentile(valid_dcpa,10):.4f}  p25={np.percentile(valid_dcpa,25):.4f}  '
                    f'median={np.median(valid_dcpa):.4f}  p75={np.percentile(valid_dcpa,75):.4f}'
                )
                logging.getLogger().info(
                    f'--risk_scale is currently {args.risk_scale} — check it sits near the p10-p25 mark above.'
                )
            print_dcpa_stats = False  # only once

        with torch.set_grad_enabled(train):
            pred_disp = model(obs, mask=mask, stats=stats)
            loss = cpagrn_loss_riskweighted(pred_disp, target_disp, mask, risk_weight)

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
    log.info(f'Model: model_cpagrn.py (CPAGRN, gru_layers={args.gru_layers}) + risk-weighted loss (Πρόταση 3)')
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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val   = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        lr = get_lr(epoch, args)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        t0 = time.time()
        train_loss = run_epoch(train_loader, model, optimizer, device, args, stats,
                                train=True, print_dcpa_stats=(epoch == 0))
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
