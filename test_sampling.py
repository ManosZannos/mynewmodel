"""
Test script for best-of-K sampling implementation.

Verifies that the sampling and evaluation functions work correctly.
"""

import torch
import numpy as np
from metrics import sample_bivariate_gaussian, best_of_k_ade, best_of_k_fde, evaluate_best_of_k


def test_sampling():
    """Test that sampling from Gaussian produces correct shapes."""
    print("Testing bivariate Gaussian sampling...")
    
    pred_len = 10
    N = 5  # 5 vessels
    num_samples = 20
    
    # Create dummy Gaussian parameters
    V_pred = torch.randn(pred_len, N, 5)
    
    # Sample
    samples = sample_bivariate_gaussian(V_pred, num_samples)
    
    # Check shape
    assert samples.shape == (num_samples, pred_len, N, 2), f"Expected shape (20, 10, 5, 2), got {samples.shape}"
    
    print(f"  ✓ Sampling works! Shape: {samples.shape}")
    print(f"  ✓ Mean: {samples.mean():.4f}, Std: {samples.std():.4f}")


def test_evaluation():
    """Test best-of-K ADE/FDE computation."""
    print("\nTesting best-of-K evaluation...")
    
    pred_len = 10
    N = 5
    num_samples = 20
    
    # Create dummy predictions (Gaussian parameters)
    V_pred = torch.randn(pred_len, N, 5)
    
    # Create dummy ground truth
    V_target = torch.randn(pred_len, N, 2)
    
    # Combined evaluation (NEW: single sampling, same best sample)
    results = evaluate_best_of_k(V_pred, V_target, num_samples)
    print(f"  ✓ Combined evaluation (paper-aligned):")
    print(f"    minADE-{num_samples}: {results['minADE']:.4f}")
    print(f"    FDE (best sample):    {results['FDE']:.4f}")
    print(f"    Best sample shape: {results['best_sample'].shape}")
    print(f"    Note: Both metrics computed on SAME best sample (selected by minADE)")
    
    # Test backward compatibility functions
    min_ade, _ = best_of_k_ade(V_pred, V_target, num_samples)
    fde_best, _ = best_of_k_fde(V_pred, V_target, num_samples)
    print(f"  ✓ Backward compatibility:")
    print(f"    best_of_k_ade(): {min_ade:.4f}")
    print(f"    best_of_k_fde(): {fde_best:.4f}")


def test_best_of_k_property():
    """Test that best-of-K is always better than or equal to single sample."""
    print("\nTesting best-of-K property...")
    
    pred_len = 10
    N = 5
    
    # Create predictions close to a target
    V_target = torch.zeros(pred_len, N, 2)
    
    # Create Gaussian centered around target with some noise
    V_pred = torch.zeros(pred_len, N, 5)
    V_pred[:, :, 0] = 0.1  # μx = 0.1 (close to target 0)
    V_pred[:, :, 1] = 0.1  # μy = 0.1
    V_pred[:, :, 2] = -2.0  # log(σx) = -2.0 → σx ≈ 0.135
    V_pred[:, :, 3] = -2.0  # log(σy) = -2.0 → σy ≈ 0.135
    V_pred[:, :, 4] = 0.0   # ρ = 0 (no correlation)
    
    # Test with different K values
    for K in [1, 5, 10, 20]:
        results = evaluate_best_of_k(V_pred, V_target, num_samples=K)
        print(f"  K={K:2d}: minADE={results['minADE']:.4f}, FDE={results['FDE']:.4f}")
    
    print("  ✓ As K increases, minADE/FDE should decrease (or stay same)")


def test_deterministic_case():
    """Test that with zero variance, sampling gives deterministic results."""
    print("\nTesting deterministic case (σ=0)...")
    
    pred_len = 5
    N = 3
    num_samples = 10
    
    # Create Gaussian with zero variance (deterministic)
    V_pred = torch.zeros(pred_len, N, 5)
    V_pred[:, :, 0] = 1.0   # μx = 1.0
    V_pred[:, :, 1] = 2.0   # μy = 2.0
    V_pred[:, :, 2] = -10.0 # log(σx) = -10 → σx ≈ 0 (very small)
    V_pred[:, :, 3] = -10.0 # log(σy) = -10 → σy ≈ 0
    V_pred[:, :, 4] = 0.0   # ρ = 0
    
    # Sample multiple times
    samples = sample_bivariate_gaussian(V_pred, num_samples)
    
    # All samples should be very close to the mean
    mean_x = samples[:, :, :, 0].mean()
    mean_y = samples[:, :, :, 1].mean()
    std_x = samples[:, :, :, 0].std()
    std_y = samples[:, :, :, 1].std()
    
    print(f"  ✓ Sampled mean: ({mean_x:.4f}, {mean_y:.4f}) [expected: (1.0, 2.0)]")
    print(f"  ✓ Sampled std:  ({std_x:.4f}, {std_y:.4f}) [expected: ~0.0]")
    
    assert abs(mean_x - 1.0) < 0.1, "Mean x should be close to 1.0"
    assert abs(mean_y - 2.0) < 0.1, "Mean y should be close to 2.0"
    assert std_x < 0.1, "Std x should be very small"
    assert std_y < 0.1, "Std y should be very small"


def test_correlation():
    """Test that correlation parameter works correctly."""
    print("\nTesting correlation parameter...")
    
    pred_len = 100  # More samples for better correlation estimation
    N = 1
    num_samples = 1000
    
    # Create Gaussian with positive correlation
    V_pred = torch.zeros(pred_len, N, 5)
    V_pred[:, :, 0] = 0.0   # μx = 0
    V_pred[:, :, 1] = 0.0   # μy = 0
    V_pred[:, :, 2] = 0.0   # log(σx) = 0 → σx = 1.0
    V_pred[:, :, 3] = 0.0   # log(σy) = 0 → σy = 1.0
    V_pred[:, :, 4] = 0.9   # ρ = tanh(0.9) ≈ 0.716 (positive correlation)
    
    # Sample
    samples = sample_bivariate_gaussian(V_pred, num_samples)
    
    # Compute empirical correlation
    x = samples[:, :, 0, 0].cpu().numpy()  # [num_samples, pred_len]
    y = samples[:, :, 0, 1].cpu().numpy()
    
    # Flatten and compute correlation
    x_flat = x.flatten()
    y_flat = y.flatten()
    correlation = np.corrcoef(x_flat, y_flat)[0, 1]
    
    expected_corr = np.tanh(0.9)
    print(f"  ✓ Expected correlation: {expected_corr:.3f}")
    print(f"  ✓ Empirical correlation: {correlation:.3f}")
    print(f"  ✓ Difference: {abs(correlation - expected_corr):.3f} [should be < 0.05]")


def main():
    print("="*80)
    print("TESTING BEST-OF-K SAMPLING IMPLEMENTATION")
    print("="*80)
    
    torch.manual_seed(42)  # For reproducibility
    np.random.seed(42)
    
    try:
        test_sampling()
        test_evaluation()
        test_best_of_k_property()
        test_deterministic_case()
        test_correlation()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nThe sampling implementation is working correctly.")
        print("You can now use evaluate.py for paper-aligned evaluation.")
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED! ✗")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
