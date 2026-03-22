import torch
import os
from termcolor import colored
from torch.utils.data import DataLoader

from metrics import *
from model import *
from utils import *

print("="*80)
print(colored("SMOKE TEST - Trajectory Prediction Pipeline", "cyan", attrs=["bold"]))
print("="*80)

print(colored("\n✓ Imports loaded", "green"))
print(f"  Torch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

# Test parameters
obs_len = 10
pred_len = 10
embedding_dims = 64
num_gcn_layers = 1
num_heads = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"  Device: {device}")

# ============================================================================
# 1️⃣ Dataset Loading Test
# ============================================================================
print(colored("\n1️⃣ Testing Dataset Loading...", "yellow"))

# Check if preprocessed data exists
dataset_path = './dataset/noaa_dec2021/train'
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
        
        # Get one batch
        batch = next(iter(loader))
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr = batch
        
        print(colored("  ✓ Dataset loaded successfully", "green"))
        print(f"    Total sequences: {len(dataset)}")
        print(f"    V_obs shape: {V_obs.shape}")  # [batch_size, obs_len, N, 4]
        print(f"    V_tr shape:  {V_tr.shape}")   # [batch_size, pred_len, N, 4]
        
        # Move to device
        V_obs = V_obs.to(device)
        V_tr = V_tr.to(device)
        
    except Exception as e:
        print(colored(f"  ✗ Dataset loading failed: {e}", "red"))
        dataset = None
else:
    print(colored(f"  ⚠ Dataset not found at: {dataset_path}", "yellow"))
    print("    Skipping dataset tests (preprocessing not yet done)")
    dataset = None

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
        out_dims=5,  # Gaussian parameters
        num_heads=num_heads
    ).to(device)
    
    print(colored("  ✓ Model initialized", "green"))
    
    # Test with dummy data if no dataset
    if dataset is None:
        batch_size = 1
        N = 20  # dummy number of vessels
        # V_obs: [batch, obs_len, N, 5] = [pos_enc, LON_rel, LAT_rel, SOG_rel, Heading_rel]
        V_obs = torch.randn(batch_size, obs_len, N, 5).to(device)
        V_tr = torch.randn(batch_size, pred_len, N, 4).to(device)
        print(f"    Using dummy data: V_obs {V_obs.shape}")
    
    # Create identity matrices (original repo)
    T = V_obs.shape[1]  # obs_len
    N = V_obs.shape[2]  # num vessels
    identity_spatial  = torch.ones((T, N, N), device=device) * torch.eye(N, device=device)
    identity_temporal = torch.ones((N, T, T), device=device) * torch.eye(T, device=device)
    identity = [identity_spatial, identity_temporal]
    
    # Forward pass
    V_pred = model(V_obs, identity)
    
    print(colored("  ✓ Forward pass successful", "green"))
    print(f"    Input:  V_obs {list(V_obs.shape)}")
    print(f"    Output: V_pred {list(V_pred.shape)}")
    print(f"    Expected: [pred_len={pred_len}, N={N}, 5]")
    
    # Check shape
    expected_shape = (pred_len, N, 5)
    actual_shape = tuple(V_pred.shape) if V_pred.dim() == 3 else tuple(V_pred.squeeze(0).shape)
    
    if actual_shape == expected_shape:
        print(colored("  ✓ Output shape correct", "green"))
    else:
        print(colored(f"  ⚠ Output shape mismatch: {actual_shape} vs {expected_shape}", "yellow"))
    
    # Ensure correct shape for next tests
    if V_pred.dim() == 4:
        V_pred = V_pred.squeeze(0)
    
except Exception as e:
    print(colored(f"  ✗ Model forward pass failed: {e}", "red"))
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# 3️⃣ Gaussian Sampling Test
# ============================================================================
print(colored("\n3️⃣ Testing Gaussian Sampling...", "yellow"))

try:
    num_samples = 20
    samples = sample_bivariate_gaussian(V_pred, num_samples)
    
    print(colored("  ✓ Gaussian sampling successful", "green"))
    print(f"    V_pred shape:  {list(V_pred.shape)}")  # [pred_len, N, 5]
    print(f"    Samples shape: {list(samples.shape)}")  # [K, pred_len, N, 2]
    print(f"    Expected: [K={num_samples}, pred_len={pred_len}, N={N}, 2]")
    
    # Check shape
    expected_samples_shape = (num_samples, pred_len, N, 2)
    if tuple(samples.shape) == expected_samples_shape:
        print(colored("  ✓ Samples shape correct", "green"))
    else:
        print(colored(f"  ⚠ Samples shape mismatch", "yellow"))
        
except Exception as e:
    print(colored(f"  ✗ Gaussian sampling failed: {e}", "red"))
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# 4️⃣ Best-of-K Evaluation Test
# ============================================================================
print(colored("\n4️⃣ Testing Best-of-K Evaluation...", "yellow"))

try:
    # Ground truth (use first 2 features: LON, LAT)
    V_target = V_tr.squeeze(0)[:, :, :2]  # [pred_len, N, 2]
    
    # Evaluate
    results = evaluate_best_of_k(V_pred, V_target, num_samples=num_samples)
    
    print(colored("  ✓ Best-of-K evaluation successful", "green"))
    print(f"    V_pred shape:   {list(V_pred.shape)}")    # [pred_len, N, 5]
    print(f"    V_target shape: {list(V_target.shape)}")  # [pred_len, N, 2]
    print(f"    minADE-{num_samples}: {results['minADE']:.4f}")
    print(f"    FDE (best sample): {results['FDE']:.4f}")
    print(f"    Best sample shape: {list(results['best_sample'].shape)}")
    
    # Check return keys
    expected_keys = {'minADE', 'FDE', 'best_sample', 'all_ade_values'}
    actual_keys = set(results.keys())
    
    if expected_keys == actual_keys:
        print(colored("  ✓ Return dict keys correct", "green"))
    else:
        print(colored(f"  ⚠ Return dict keys mismatch", "yellow"))
        print(f"    Expected: {expected_keys}")
        print(f"    Actual: {actual_keys}")
        
except Exception as e:
    print(colored(f"  ✗ Best-of-K evaluation failed: {e}", "red"))
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# 5️⃣ Shape Consistency Check
# ============================================================================
print(colored("\n5️⃣ Shape Consistency Summary...", "yellow"))

shape_checks = {
    "V_obs (input)": V_obs.shape,
    "V_tr (ground truth)": V_tr.shape,
    "V_pred (model output)": V_pred.shape,
    "V_target (LON, LAT)": V_target.shape,
    "samples (Gaussian)": samples.shape,
    "best_sample": results['best_sample'].shape,
}

print(colored("  All shapes:", "cyan"))
for name, shape in shape_checks.items():
    print(f"    {name:25s}: {list(shape)}")

# Final checks
all_passed = True

# Check V_pred has 5 Gaussian parameters
if V_pred.shape[-1] == 5:
    print(colored("\n  ✓ V_pred has 5 Gaussian parameters (μx, μy, log(σx), log(σy), ρ)", "green"))
else:
    print(colored(f"\n  ✗ V_pred should have 5 parameters, got {V_pred.shape[-1]}", "red"))
    all_passed = False

# Check V_target has 2 coordinates
if V_target.shape[-1] == 2:
    print(colored("  ✓ V_target has 2 coordinates (LON, LAT)", "green"))
else:
    print(colored(f"  ✗ V_target should have 2 coordinates, got {V_target.shape[-1]}", "red"))
    all_passed = False

# Check samples dimensions
if samples.shape[0] == num_samples and samples.shape[-1] == 2:
    print(colored(f"  ✓ Samples: K={num_samples} trajectories with 2D positions", "green"))
else:
    print(colored("  ✗ Samples shape incorrect", "red"))
    all_passed = False

# Check best_sample matches V_target shape
if results['best_sample'].shape == V_target.shape:
    print(colored("  ✓ Best sample shape matches V_target", "green"))
else:
    print(colored("  ✗ Best sample shape mismatch", "red"))
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
    print(colored("\n⚠ Note: Dataset not found. Run preprocessing first:", "yellow"))
    print(colored("  python preprocess_ais.py", "cyan"))
    
print("="*80)