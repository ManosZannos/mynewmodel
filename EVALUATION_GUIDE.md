# Evaluation Guide: Best-of-K Sampling

## Overview

This project now implements **paper-aligned evaluation** with best-of-K sampling, as described in the paper:

> "During the inference stage, 20 samples are extracted from the learned bivariate Gaussian distribution and the closest sample to the ground-truth is used to calculate the performance index of the model."

## Important Distinction

### Training/Validation Loss
- **What it measures**: Negative log-likelihood of the bivariate Gaussian distribution
- **Purpose**: Optimize model parameters during training
- **Computed in**: `train.py` (both training and validation loops)

### Evaluation Metrics (minADE, minFDE)
- **What it measures**: Displacement error using best-of-K sampled trajectories
- **Purpose**: Report final model performance (paper-aligned)
- **Computed in**: `evaluate.py` (separate evaluation script)

## Usage

### 1. Train the Model
```bash
python train.py --dataset noaa_dec2021 --obs_len 10 --pred_len 10 --batch_size 8 --num_epochs 100
```

This will save checkpoints to `checkpoints/AddGCN_10_10/noaa_dec2021/`:
- `train_best.pth` - Best model on training loss
- `val_best.pth` - Best model on validation loss

### 2. Evaluate with Best-of-K Sampling

#### Evaluate on Validation Set
```bash
python evaluate.py \
    --checkpoint checkpoints/AddGCN_10_10/noaa_dec2021/val_best.pth \
    --dataset noaa_dec2021 \
    --split val \
    --num_samples 20
```

#### Evaluate on Test Set
```bash
python evaluate.py \
    --checkpoint checkpoints/AddGCN_10_10/noaa_dec2021/val_best.pth \
    --dataset noaa_dec2021 \
    --split test \
    --num_samples 20
```

### 3. Output

The evaluation script will print:
```
================================================================================
EVALUATION RESULTS (Best-of-20)
================================================================================
Dataset: noaa_dec2021 (test split)
Checkpoint: checkpoints/AddGCN_10_10/noaa_dec2021/val_best.pth
Total sequences: 150

Metrics:
  minADE-20: 0.1234
  minFDE-20: 0.2456
================================================================================
```

Results are also saved to: `checkpoints/.../eval_results_test_K20.txt`

## How Best-of-K Sampling Works

### Algorithm
1. **Sample K trajectories** (K=20) from the predicted bivariate Gaussian distribution
   - Each trajectory is (pred_len × N × 2) for position (x, y)
   
2. **Compute errors** for each sample:
   - **ADE**: Average displacement across all timesteps
   - **FDE**: Final displacement at last timestep
   
3. **Select best sample**: Choose the trajectory with minimum error

4. **Report metric**: The minimum error across all K samples

### Mathematical Details

Given predicted Gaussian parameters:
- μx, μy: Mean positions
- σx, σy: Standard deviations (exp-transformed from log(σ))
- ρ: Correlation coefficient (tanh-transformed)

Sample using Cholesky decomposition:
```
z1, z2 ~ N(0, 1)
x = μx + σx * z1
y = μy + ρ*σy*z1 + σy*sqrt(1-ρ²)*z2
```

## Implementation Files

### New/Modified Files
1. **`metrics.py`** - Added functions:
   - `sample_bivariate_gaussian()` - Sample from Gaussian distribution
   - `best_of_k_ade()` - Compute minADE with sampling
   - `best_of_k_fde()` - Compute minFDE with sampling
   - `evaluate_best_of_k()` - Combined evaluation

2. **`evaluate.py`** - New evaluation script:
   - Loads trained model checkpoint
   - Runs inference with K=20 samples
   - Computes minADE-20 and minFDE-20
   - Saves detailed results

3. **`train.py`** - Updated documentation:
   - Added note in validation loop about using `evaluate.py` for final metrics

## Parameters

### Evaluation Script Arguments
```bash
--dataset        # Dataset name (e.g., noaa_dec2021)
--split          # Data split: train/val/test
--checkpoint     # Path to model checkpoint (.pth file)
--num_samples    # K for best-of-K (default: 20, as in paper)
--obs_len        # Observation length (default: 10)
--pred_len       # Prediction length (default: 10)
--device         # cuda or cpu
```

## Why This Matters

### Without Best-of-K Sampling
If you only use the mean trajectory (μx, μy):
- Ignores the stochastic nature of the model
- May not represent the best possible prediction
- Not comparable to paper results

### With Best-of-K Sampling
- Allows the model to explore multiple plausible futures
- Selects the best prediction (closest to ground truth)
- Paper-aligned evaluation
- Fair comparison with other stochastic models

## Expected Performance

Typical values (depends on dataset and preprocessing):
- **minADE-20**: 0.05 - 0.20 (normalized coordinates)
- **minFDE-20**: 0.10 - 0.40 (normalized coordinates)

**Note**: These are in normalized space (z-score). To convert back to real-world units (degrees, knots), you need to denormalize using the global statistics from `dataset/noaa_dec2021/global_stats.json`.

## Troubleshooting

### CUDA Out of Memory
If K=20 samples cause memory issues:
```bash
python evaluate.py --num_samples 10 ...  # Use K=10 instead
```

### Slow Evaluation
The sampling process is computationally intensive. For faster evaluation:
- Use GPU (`--device cuda`)
- Reduce `--num_samples` (but less paper-aligned)
- Evaluate on a subset of data

## Next Steps

After evaluation, you can:
1. **Compare different checkpoints** (train_best.pth vs val_best.pth)
2. **Try different K values** (5, 10, 20, 50) to see impact
3. **Visualize predictions** (extend evaluate.py to save sampled trajectories)
4. **Denormalize metrics** to report in real-world units

## References

This implementation follows the evaluation methodology from:
> [Paper citation - Vessel trajectory prediction paper with AddGCN model]

Key insight: Stochastic models should be evaluated by sampling multiple predictions and selecting the best one, not just using the mean prediction.
