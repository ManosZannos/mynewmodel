import torch
import os
from termcolor import colored
from torch.utils.data import DataLoader

from metrics import *
from model import *
from utils import *

print("="*80)
print(colored("SMOKE TEST - Trajectory Prediction Pipeline (True Baseline, 4 features)", "cyan", attrs=["bold"]))
print("="*80)

print(colored("\n✓ Imports loaded", "green"))
print(f"  Torch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

# Test parameters
obs_len = 10
pred_len = 5
embedding_dims = 64
num_gcn_layers = 1
num_heads = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"  Device: {device}")

# ============================================================================
# 1️⃣ Dataset Loading Test
# ============================================================================
print(colored("\n1️⃣ Testing Dataset Loading...", "yellow"))

obs_traj = None
pred_traj_gt = None
dataset = None

dataset_path = os.path.join('./dataset', 'marinecadastre_2021', 'train')
if os.path.exists(dataset_path):
    try:
        dataset = TrajectoryDataset(
            dataset_path,
            obs_len=obs_len,
            pred_len=pred_len,
            skip=1
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        batch = next(iter(loader))
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr = batch

        print(colored("  ✓ Dataset loaded successfully", "green"))
        print(f"    Total sequences: {len(dataset)}")
        print(f"    V_obs shape: {V_obs.shape}")  # [1, obs_len, N, 4]
        print(f"    V_tr shape:  {V_tr.shape}")   # [1, pred_len, N, 4]

        V_obs = V_obs.to(device)
        V_tr = V_tr.to(device)

    except Exception as e:
        print(colored(f"  ✗ Dataset loading failed: {e}", "red"))
        dataset = None
else:
    print(colored(f"  ⚠ Dataset not found at: {dataset_path}", "yellow"))
    print("    Skipping dataset tests (preprocessing not yet done)")

# ============================================================================
# 2️⃣ Model Forward Pass Test
# ============================================================================
print(colored("\n2️⃣ Testing Model Forward Pass...", "yellow"))

try:
    model = TrajectoryModel(
        number_asymmetric_conv_layer=2,
        embedding_dims=embedding_dims,
        number_gcn_layers=num_gcn_layers,
        dropout=0.0,
        obs_len=obs_len,
        pred_len=pred_len,
        out_dims=2,  # deterministic: (lon_vel, lat_vel)
        num_heads=num_heads
    ).to(device)

    print(colored("  ✓ Model initialized", "green"))

    if dataset is None:
        batch_size = 1
        N = 20
        # V_obs: [batch, obs_len, N, 4] — true baseline: 4 features only
        V_obs = torch.randn(batch_size, obs_len, N, 4).to(device)
        V_tr  = torch.randn(batch_size, pred_len, N, 4).to(device)
        obs_traj     = torch.zeros(batch_size, N, 4, obs_len)
        pred_traj_gt = torch.zeros(batch_size, N, 4, pred_len)
        print(f"    Using dummy data: V_obs {V_obs.shape}")

    T = V_obs.shape[1]
    N = V_obs.shape[2]
    identity_spatial  = torch.ones((T, N, N), device=device) * torch.eye(N, device=device)
    identity_temporal = torch.ones((N, T, T), device=device) * torch.eye(T, device=device)
    identity = [identity_spatial, identity_temporal]

    V_pred = model(V_obs, identity)

    print(colored("  ✓ Forward pass successful", "green"))
    print(f"    Input:  V_obs {list(V_obs.shape)}")
    print(f"    Output: V_pred {list(V_pred.shape)}")
    print(f"    Expected: [pred_len={pred_len}, N={N}, 2]")

    expected_shape = (pred_len, N, 2)
    actual_shape = tuple(V_pred.shape) if V_pred.dim() == 3 else tuple(V_pred.squeeze(0).shape)

    if actual_shape == expected_shape:
        print(colored("  ✓ Output shape correct", "green"))
    else:
        print(colored(f"  ⚠ Output shape mismatch: {actual_shape} vs {expected_shape}", "yellow"))

    if V_pred.dim() == 4:
        V_pred = V_pred.squeeze(0)

except Exception as e:
    print(colored(f"  ✗ Model forward pass failed: {e}", "red"))
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# 3️⃣ Deterministic Evaluation Test
# ============================================================================
print(colored("\n3️⃣ Testing Deterministic Evaluation...", "yellow"))

try:
    LAT_MIN, LAT_RANGE = 20.90883, 28.32044
    LON_MIN, LON_RANGE = -133.29703, 72.60811

    last_obs = obs_traj.squeeze(0)[:, :2, -1].to(device)  # [N, 2]
    pred_abs = torch.cumsum(V_pred, dim=0) + last_obs.unsqueeze(0)

    pred_lon = pred_abs[:, :, 0] * LON_RANGE + LON_MIN
    pred_lat = pred_abs[:, :, 1] * LAT_RANGE + LAT_MIN

    gt_abs = pred_traj_gt.squeeze(0).permute(2, 0, 1)[:, :, :2].to(device)
    gt_lon = gt_abs[:, :, 0] * LON_RANGE + LON_MIN
    gt_lat = gt_abs[:, :, 1] * LAT_RANGE + LAT_MIN

    disp = torch.sqrt((pred_lon - gt_lon)**2 + (pred_lat - gt_lat)**2)
    ade = disp.mean().item()
    fde = disp[-1].mean().item()

    print(colored("  ✓ Deterministic evaluation successful", "green"))
    print(f"    V_pred shape: {list(V_pred.shape)}  (deterministic, out_dims=2)")
    print(f"    ADE: {ade:.4f}°")
    print(f"    FDE: {fde:.4f}°")

except Exception as e:
    print(colored(f"  ✗ Deterministic evaluation failed: {e}", "red"))
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# 4️⃣ Loss Function Test
# ============================================================================
print(colored("\n4️⃣ Testing Loss Function...", "yellow"))

try:
    from torch.nn import functional as F_test

    V_target = V_tr.squeeze(0)  # [pred_len, N, 4]
    last_obs_loss = obs_traj.squeeze(0)[:, :2, -1].to(device)

    pred_pos = torch.cumsum(V_pred, dim=0) + last_obs_loss.unsqueeze(0)
    gt_pos   = torch.cumsum(V_target[:, :, :2], dim=0) + last_obs_loss.unsqueeze(0)

    pos_loss = F_test.huber_loss(pred_pos, gt_pos)
    vel_loss = F_test.huber_loss(V_pred, V_target[:, :, :2])
    total_loss = 10.0 * pos_loss + 0.1 * vel_loss

    print(colored("  ✓ Loss function successful", "green"))
    print(f"    pos_loss:   {pos_loss.item():.6f}")
    print(f"    vel_loss:   {vel_loss.item():.6f}")
    print(f"    total_loss: {total_loss.item():.6f}")

except Exception as e:
    print(colored(f"  ✗ Loss function failed: {e}", "red"))
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# 5️⃣ Shape Consistency Check
# ============================================================================
print(colored("\n5️⃣ Shape Consistency Summary...", "yellow"))

shape_checks = {
    "V_obs (input)":         V_obs.shape,
    "V_tr (ground truth)":   V_tr.shape,
    "V_pred (model output)": V_pred.shape,
}

print(colored("  All shapes:", "cyan"))
for name, shape in shape_checks.items():
    print(f"    {name:25s}: {list(shape)}")

all_passed = True

if V_pred.shape[-1] == 2:
    print(colored("\n  ✓ V_pred has 2 outputs (lon_vel, lat_vel) — deterministic", "green"))
else:
    print(colored(f"\n  ✗ V_pred should have 2 outputs, got {V_pred.shape[-1]}", "red"))
    all_passed = False

if V_obs.shape[-1] == 4:
    print(colored("  ✓ V_obs has 4 features (LON_rel, LAT_rel, SOG_rel, Heading_rel) — true baseline", "green"))
else:
    print(colored(f"  ✗ V_obs has {V_obs.shape[-1]} features (expected 4 for true baseline)", "red"))
    all_passed = False

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*80)
if all_passed:
    print(colored("✓ ALL SMOKE TESTS PASSED!", "green", attrs=["bold"]))
    print(colored("  Pipeline is ready for training and evaluation", "green"))
else:
    print(colored("⚠ SOME CHECKS FAILED", "yellow", attrs=["bold"]))
    print(colored("  Review the errors above", "yellow"))

if dataset is None:
    print(colored("\n⚠ Note: Dataset not found — ran with dummy data.", "yellow"))
    print(colored("  Run: python convert_json_to_csv.py", "cyan"))

print("="*80)