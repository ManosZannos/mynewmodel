"""
overfit_check.py — Overfit-a-tiny-batch sanity check, reusable across model variants.

Purpose (Karpathy's "Recipe for Training Neural Networks" — the cheapest,
highest-value debugging tool): before trusting ANY new architecture, verify
that it can drive training loss to ~0 on a tiny, fixed subset of data. If it
can't, something in the wiring is broken (mask direction, target alignment,
a frozen/disconnected sub-module) — it is NOT a "the task is hard" problem.

This is cheaper and catches a DIFFERENT class of bugs than the usual
3-epoch/full-dataset sanity check (which mainly catches OOM/crashes): it
catches silent logic errors that still "run" without error.

Usage:
    python overfit_check.py --model decodercond --obs_len 10 --pred_len 10 \
        --gru_layers 2 --top_k 10 --n_samples 8 --epochs 300 --gpu_num 0

Add new variants by extending MODEL_REGISTRY below.
"""

from __future__ import annotations
import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from dataset import AISDataset, collate_fn


def build_gru2(args):
    from model_cpagrn import CPAGRN, cpagrn_loss
    model = CPAGRN(
        feature_size=4, d_model=args.d_model, gru_layers=args.gru_layers,
        pred_len=args.pred_len, top_k=args.top_k,
    )
    return model, cpagrn_loss


def build_decodercond(args):
    from model_cpagrn_decodercond import CPAGRNDecoderCond, cpagrn_loss
    model = CPAGRNDecoderCond(
        feature_size=4, d_model=args.d_model, gru_layers=args.gru_layers,
        pred_len=args.pred_len, top_k=args.top_k, cond_dim=args.cond_dim,
    )
    return model, cpagrn_loss


def build_uniontopk(args):
    from model_cpagrn_uniontopk import CPAGRNUnionTopK, cpagrn_loss
    model = CPAGRNUnionTopK(
        feature_size=4, d_model=args.d_model, gru_layers=args.gru_layers,
        pred_len=args.pred_len, top_k_dist=args.top_k_dist, top_k_risk=args.top_k_risk,
    )
    return model, cpagrn_loss


def build_auxrisk(args):
    from model_cpagrn_auxrisk import CPAGRNAuxRisk, cpagrn_loss
    model = CPAGRNAuxRisk(
        feature_size=4, d_model=args.d_model, gru_layers=args.gru_layers,
        pred_len=args.pred_len, top_k=args.top_k,
    )
    # This model's forward returns (pred_disp, aux_pred) instead of just
    # pred_disp. The generic harness below only has target_disp (not the raw
    # future positions needed for compute_true_future_dcpa), so this overfit
    # check only exercises the MAIN displacement loss — it still catches
    # wiring bugs in the shared encoder/decoder path. The aux_head will show
    # a ZERO-gradient warning below; that is EXPECTED here, not a bug — the
    # aux branch itself is only exercised by the full training script
    # (train_cpagrn_auxrisk.py), which has access to pred_gt.
    def wrapped_loss(pred_out, target_disp, mask):
        pred_disp, _aux_pred = pred_out
        return cpagrn_loss(pred_disp, target_disp, mask)
    return model, wrapped_loss


MODEL_REGISTRY = {
    'gru2':        build_gru2,
    'decodercond': build_decodercond,
    'uniontopk':   build_uniontopk,
    'auxrisk':     build_auxrisk,
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',       type=str, required=True, choices=list(MODEL_REGISTRY))
    p.add_argument('--data_dir',    type=str, default='dataset/noaa_dec2021_1min')
    p.add_argument('--obs_len',     type=int, default=10)
    p.add_argument('--pred_len',    type=int, default=10)
    p.add_argument('--d_model',     type=int, default=64)
    p.add_argument('--gru_layers',  type=int, default=2)
    p.add_argument('--top_k',       type=int, default=10)
    p.add_argument('--top_k_dist',  type=int, default=12)
    p.add_argument('--top_k_risk',  type=int, default=3)
    p.add_argument('--cond_dim',    type=int, default=16)
    p.add_argument('--n_samples',   type=int, default=8,
                   help='How many training samples to overfit on (tiny, fixed subset)')
    p.add_argument('--epochs',      type=int, default=300)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--gpu_num',     type=int, default=0)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--log_every',   type=int, default=25)
    p.add_argument('--target_loss', type=float, default=1e-4,
                   help='Train loss below this after --epochs is considered a PASS')
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} (physical GPU {args.gpu_num})')
    print(f'Overfit check for model="{args.model}", n_samples={args.n_samples}, '
          f'epochs={args.epochs}')

    # Build a tiny, FIXED subset (same samples every epoch — no shuffling, no val split)
    full_ds = AISDataset(f'{args.data_dir}/train', args.obs_len, args.pred_len)
    n = min(args.n_samples, len(full_ds))
    subset = [full_ds[i] for i in range(n)]
    obs_b, pred_b, mask_b, meta_b = collate_fn(subset)
    obs_b, pred_b, mask_b = obs_b.to(device), pred_b.to(device), mask_b.to(device)
    print(f'Overfitting on {n} fixed samples (no shuffling, no val).')

    model, loss_fn = MODEL_REGISTRY[args.model](args)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    last_obs    = obs_b[:, :, -1, :2]
    target_disp = pred_b - last_obs.unsqueeze(2)

    model.train()
    losses = []
    for epoch in range(args.epochs):
        pred_disp = model(obs_b, mask=mask_b)
        loss = loss_fn(pred_disp, target_disp, mask_b)

        optimizer.zero_grad()
        loss.backward()

        # Check for a common silent-bug symptom: a sub-module with no gradient
        # at all (disconnected from the loss) — worth flagging even if loss falls.
        if epoch == 0:
            no_grad_params = [
                name for name, p in model.named_parameters()
                if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)
            ]
            if no_grad_params:
                print(f'  WARNING: {len(no_grad_params)} parameter(s) got ZERO gradient '
                      f'on the first backward pass — possible disconnected sub-module:')
                for name in no_grad_params[:10]:
                    print(f'    - {name}')

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:>4}/{args.epochs} | loss={loss.item():.6f}')

    final_loss = losses[-1]
    initial_loss = losses[0]
    print(f'\n{"="*55}')
    print(f'  Initial loss : {initial_loss:.6f}')
    print(f'  Final loss   : {final_loss:.6f}')
    print(f'  Reduction    : {(1 - final_loss/initial_loss)*100:.1f}%')
    if final_loss < args.target_loss:
        print(f'  RESULT: PASS — model can overfit a tiny batch (loss < {args.target_loss}).')
        print(f'  Safe to proceed to the full sanity check / full training.')
    else:
        print(f'  RESULT: FAIL — loss did not reach {args.target_loss}.')
        print(f'  Do NOT proceed to full training yet — check for a wiring bug')
        print(f'  (mask direction, target alignment, disconnected sub-module, LR too low).')
    print('='*55)


if __name__ == '__main__':
    main()