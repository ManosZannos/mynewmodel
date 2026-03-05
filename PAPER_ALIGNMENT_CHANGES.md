# SMCHN Model - Paper Alignment Changes

## Overview
This document summarizes the changes made to align the implementation with the paper's SMCHN model description (Section 3.2).

---

## ✅ **IMPLEMENTED CHANGES**

### 1. **Positional Encoding for Temporal Graph** (Section 3.2.2, Eq. 6)

**Paper requirement:**
```
et = δ(Gt, Wt_e) + p
```
The temporal graph embedding should include positional encoding `p`.

**Implementation:**
- Added `positional_encoding()` method to `SelfAttention` class ([model.py](model.py#L110-L123))
- Uses sinusoidal positional encoding (standard Transformer approach)
- Modified `SelfAttention.forward()` to accept `add_positional_encoding` parameter
- Applied to temporal graph in `SparseWeightedAdjacency.forward()` ([model.py](model.py#L208))

**Code location:** [model.py](model.py#L95-L161)

---

### 2. **Hybrid Network for Feature Fusion** (Section 3.2.3, Eq. 18-19)

**Paper requirement:**
```
w = softmax(Why*tanh(Whid*[Hs ⊕ Ht] + bhid) + bhy)  (Eq. 18)
H = w1*Hs + w2*Ht  (Eq. 19)
```

**Implementation:**
- Created new `HybridNetwork` class implementing the two-layer MLP ([model.py](model.py#L248-L295))
- Takes concatenated features `[Hs ⊕ Ht]` as input
- Learns dynamic weights `w = [w1, w2]` via Tanh → Softmax
- Outputs fused features `H = w1*Hs + w2*Ht`

**Previous issue:** The `Mix` function was defined but commented out - now properly implemented as a class.

**Code location:** [model.py](model.py#L248-L295)

---

### 3. **Proper GCN Feature Fusion with Dynamic Weights** (Section 3.2.3, Eq. 16-17)

**Paper requirement:**
- **Spatial path** (Eq. 16): Â'st → GCN₁ → Â'ts → GCN₂ → **Hs**
- **Temporal path** (Eq. 17): Â'ts → GCN₁ → Â'st → GCN₂ → **Ht**
- **Fusion**: Use hybrid network to combine Hs and Ht

**Implementation:**
- Refactored `SparseGraphConvolution.forward()` to properly compute Hs and Ht ([model.py](model.py#L306-L350))
- Integrated `HybridNetwork` to fuse features with learned weights
- Removed old commented-out `Mix` function code
- Clear documentation of each path matching paper equations

**Previous issue:** Features were simply added without dynamic weighting.

**Code location:** [model.py](model.py#L306-L350)

---

### 4. **TCN Gating Mechanism Formula** (Section 3.2.4, Eq. 20)

**Paper requirement:**
```
H(l+1) = g(Wg * H(l)) ⊗ σ(Wf * H(l))
```
where `g` is tanh and `σ` is sigmoid.

**Implementation:**
- Updated `Encoder` class to match exact formula ([model.py](model.py#L380-L401))
- **Removed** residual connection (not mentioned in paper)
- Now: `output = tanh(Wg * H) ⊗ sigmoid(Wf * H)` (element-wise multiplication)

**Previous formula:** Had `residual + f*g` which added an extra residual connection.

**Code location:** [model.py](model.py#L380-L401)

---

## ⚠️ **DOCUMENTED BUT NOT YET CHANGED**

### 5. **Input Feature Dimensions** (Section 3.2.1)

**Paper specification:**
- State vector should be **4-dimensional**: `(x, y, s, h)`
  - `x` = longitude
  - `y` = latitude
  - `s` = speed over ground (SOG)
  - `h` = heading

**Current implementation:**
- Uses 2D coordinates `(x, y)` or adds random value as 3rd dimension
- Your preprocessing in `utils.py` **does** extract SOG and Heading from NOAA data

**Recommendation:**
Update data loading to include all 4 features. The preprocessing already extracts them:
```python
# In utils.py, the frame-format includes: frame_id, vessel_id, LON, LAT, SOG, Heading
# Modify TrajectoryDataset to use all 4 features instead of just x, y
```

**Documentation added to:** [model.py](model.py#L404-L419)

---

### 6. **Identity Matrix Initialization** (Section 3.2.1)

**Paper specification:**
- **Spatial graph**: "elements in En are initialized to 1" → **all-ones matrix**
- **Temporal graph**: "Et is initialized into an upper triangular matrix filled with 1"

**Current implementation:**
- Uses identity matrices (diagonal = 1, rest = 0) in [train.py](train.py#L112-L116)

**Why this may be acceptable:**
The initialization matrices are used as "self-connection" masks and then filtered by the sparse adjacency generation module. The sparse attention mechanism will learn the actual interactions regardless of initialization.

**Recommendation (if strict paper alignment desired):**
```python
# Spatial: all 1s
identity_spatial = torch.ones((V_obs.shape[1], V_obs.shape[2], V_obs.shape[2]), device='cuda')

# Temporal: upper triangular with 1s
identity_temporal = torch.triu(
    torch.ones((V_obs.shape[2], V_obs.shape[1], V_obs.shape[1]), device='cuda')
)
```

**Documentation added to:** [train.py](train.py#L112-L127) and [model.py](model.py#L412-L415)

---

## 📋 **COMPLETE CHANGE SUMMARY**

| Component | Paper Section | Status | File | Lines |
|-----------|---------------|--------|------|-------|
| Positional encoding for temporal graph | 3.2.2 (Eq. 6) | ✅ Implemented | model.py | 110-123, 136-146, 208 |
| Hybrid network (MLP) | 3.2.3 (Eq. 18) | ✅ Implemented | model.py | 248-295 |
| GCN feature fusion with dynamic weights | 3.2.3 (Eq. 16-19) | ✅ Implemented | model.py | 306-350 |
| TCN gating mechanism | 3.2.4 (Eq. 20) | ✅ Implemented | model.py | 380-401 |
| 4D input features (x,y,s,h) | 3.2.1 | ⚠️ Documented | model.py | 404-419 |
| Identity matrix initialization | 3.2.1 | ⚠️ Documented | train.py | 112-127 |

---

## 🔍 **VERIFICATION CHECKLIST**

### Before Training:
- [ ] **Test forward pass** with current data format (should work as-is)
- [ ] **Monitor hybrid network weights** during training (should learn w1, w2 ≠ 0.5)
- [ ] **Verify gradient flow** through all new modules

### Optional Enhancements (for strict paper compliance):
- [ ] Update data loading to use 4D features: (lon, lat, SOG, heading)
- [ ] Change identity matrix initialization to all-ones (spatial) and upper-triangular (temporal)
- [ ] Adjust model input dimensions if using 4D features

### Model Architecture Matches Paper:
- ✅ Multi-graph representation (spatial + temporal)
- ✅ Self-attention mechanism for dense interactions
- ✅ Spatial-temporal fusion with CNN (1×1 convolution)
- ✅ Asymmetric CNN for interaction masking
- ✅ Threshold truncation for sparsity
- ✅ Zero-Softmax normalization
- ✅ Two-layer GCN with cascading adjacency matrices
- ✅ Hybrid network for dynamic feature fusion
- ✅ TCN with gating mechanism (3 layers, kernel size 3)
- ✅ Bivariate Gaussian output distribution
- ✅ Negative log-likelihood loss function

---

## 🚀 **NEXT STEPS**

1. **Run smoke test** to ensure model trains without errors
   ```bash
   python smoke_test.py
   ```

2. **Train model** and monitor:
   - Loss convergence
   - Hybrid network weight distribution (should not be stuck at 0.5/0.5)
   - Gradient magnitudes

3. **If results differ from paper**, consider:
   - Switching to 4D input features (most impactful change)
   - Adjusting identity matrix initialization
   - Hyperparameter tuning (learning rate, threshold ξ, etc.)

4. **Compare with baseline** to validate improvements

---

## 📚 **PAPER REFERENCE**

Model described in Section 3.2:
- 3.2.1: Multi-graph representation
- 3.2.2: Generation of sparse adjacency matrix
- 3.2.3: Hybrid network for fusing spatial and temporal interaction
- 3.2.4: Temporal convolutional layer
- 3.3: Loss function

All equations (2-21) are now properly implemented or documented.

---

## ✏️ **NOTES**

- The implementation was already quite close to the paper
- Main issues were:
  1. Missing positional encoding
  2. Hybrid network not properly integrated (was commented out)
  3. TCN had extra residual connection
  
- The sparse adjacency generation, GCN structure, and loss function were already correctly implemented

- No changes were required to:
  - AsymmetricConvolution
  - InteractionMask
  - ZeroSoftmax
  - GraphConvolution
  - bivariate_loss function

---

*Document created: 2026-03-01*
*Model version: SMCHN (Sparse Multi-graph Convolutional Hybrid Network)*
