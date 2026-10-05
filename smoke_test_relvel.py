"""
smoke_test_relvel.py - run ONCE on the DGX (needs torch) before launching the
relative-velocity training runs. Takes ~1 minute; no GPU needed.

Checks:
  1. headline and relvel models built with the same seed have IDENTICAL initial
     weights and the same parameter count (the control differs only in 2 edge
     channels).
  2. RelVelFeatures: channels 2..6 equal the headline CPAFeatures channels 2..6
     exactly, and channels 0..1 equal (vel_j - vel_i) / vel_scale.
  3. forward + Huber loss + backward run, outputs are finite, gradients reach the
     attention MLP of both aggregation blocks.
  4. loading a headline state_dict into the relvel model FAILS (no silent mixing).
  5. (unless --skip_data) estimate_rel_vel_scale on real training batches.

Usage:
    python smoke_test_relvel.py
    python smoke_test_relvel.py --skip_data
"""

import argparse
import torch

from model_cpagrn import CPAGRN as Headline, CPAFeatures
from model_cpagrn_relvel_huberloss import (
    CPAGRN as RelVel, RelVelFeatures, huber_cpagrn_loss, estimate_rel_vel_scale,
)

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', default='dataset/noaa_dec2021_1min')
ap.add_argument('--skip_data', action='store_true')
args = ap.parse_args()

KW = dict(feature_size=4, d_model=64, gru_layers=2, pred_len=10, top_k=10)

# 1. identical initialisation
torch.manual_seed(42); h = Headline(**KW)
torch.manual_seed(42); r = RelVel(**KW)
n_h = sum(p.numel() for p in h.parameters())
n_r = sum(p.numel() for p in r.parameters())
print(f'[1] parameters: headline={n_h:,}  relvel={n_r:,}')
assert n_h == n_r, 'parameter counts differ'
sd_h, sd_r = h.state_dict(), r.state_dict()
extra = set(sd_r) - set(sd_h)
missing = set(sd_h) - set(sd_r)
assert extra == {'cpa_features.vel_scale'} and not missing, f'unexpected keys: extra={extra} missing={missing}'
for k in sd_h:
    assert torch.equal(sd_h[k], sd_r[k]), f'initial weights differ at {k}'
print('    identical initial weights for the same seed: OK')

# 2. edge features
torch.manual_seed(0)
B, N = 2, 6
pos = torch.randn(B, N, 2); vel = torch.randn(B, N, 2) * 0.03; hdg = torch.randn(B, N)
e_h = CPAFeatures()(pos, vel, hdg)
rvf = RelVelFeatures(vel_scale=2.0)
e_r = rvf(pos, vel, hdg)
assert e_r.shape == e_h.shape == (B, N, N, 7)
assert torch.allclose(e_r[..., 2:], e_h[..., 2:]), 'channels 2..6 differ from headline'
dv = (vel.unsqueeze(1) - vel.unsqueeze(2)) / 2.0          # [B, i, j, 2] = (vel_j - vel_i)/scale
assert torch.allclose(e_r[..., :2], dv), 'relative velocity channels wrong'
print('[2] edge features: channels 2..6 identical to headline; channels 0..1 = (vel_j - vel_i)/scale: OK')

# 3. forward / loss / backward
obs = torch.randn(B, N, 10, 4)
mask = torch.ones(B, N, dtype=torch.bool); mask[0, -1] = False
tgt = torch.randn(B, N, 10, 2)
for name, m in (('headline', h), ('relvel', r)):
    m.zero_grad()
    out = m(obs, mask=mask)
    assert out.shape == (B, N, 10, 2) and torch.isfinite(out).all()
    loss = huber_cpagrn_loss(out, tgt, mask, delta=0.05)
    loss.backward()
    g1 = m.neighbor_agg.attn_mlp[0].weight.grad
    g2 = m.final_spatial.attn_mlp[0].weight.grad
    assert g1 is not None and g1.abs().sum() > 0 and g2 is not None and g2.abs().sum() > 0
    print(f'[3] {name}: forward/backward OK, loss={loss.item():.4f}, grads reach both attention MLPs')

# 4. no silent checkpoint mixing
try:
    RelVel(**KW).load_state_dict(sd_h)
    raise SystemExit('[4] FAIL: headline state_dict loaded into relvel model')
except RuntimeError:
    print('[4] loading a headline state_dict into the relvel model fails as intended: OK')

# 5. scale on real data
if not args.skip_data:
    from dataset import get_dataloaders
    train_loader, _, _, _ = get_dataloaders(args.data_dir, 10, 10, 32)
    s = estimate_rel_vel_scale(train_loader, n_batches=20)
    print(f'[5] estimated vel_scale on 20 real training batches: {s:.6f} (z-score units/step)')
    print('    (raw dv would be ~1/%.0f of a unit-scale feature; after dividing it is O(1))' % (1.0 / s))

print('\nALL SMOKE TESTS PASSED')
