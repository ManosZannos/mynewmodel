# Data Dimensions Clarification

## ✅ CONCLUSION: The Implementation IS Correct!

The confusion was due to **misleading comments** in the code. The actual data flow and dimensions are correct and match the paper's specification.

---

## **Paper Specification (Section 3.2.1)**

> "the attribute of v^t_n is the state vector (x^t_n, y^t_n, s^t_n, h^t_n) of the nth vessel at time step t, where x^t_n, y^t_n, s^t_n and h^t_n represents the longitude, latitude, speed over ground and heading"

**Required: 4D state vector** = `(LON, LAT, SOG, Heading)`

---

## **Actual Data Flow** ✅

### 1. **Preprocessing** ([utils.py](utils.py))
```
Output CSV: frame_id, vessel_id, LON, LAT, SOG, Heading
           ↑ Column 0  ↑ Column 1  ↑ Columns 2-5 (4 features)
```
**Verified**: Your data files contain all 6 columns ✅

### 2. **Data Loading** ([utils.py](utils.py#L480-L622))
```python
data = np.asarray(data)[:, :6]  # Load all 6 columns
feat_seq = np.transpose(curr_ped_seq[:, 2:])  # Extract columns 2-5 (4 features)
curr_seq = np.zeros((len(peds), 4, seq_len))  # Shape: (N, 4, T)
```
**Result**: `self.obs_traj` has shape `(N, 4, obs_len)` with 4 features ✅

### 3. **Graph Construction** ([utils.py](utils.py#L436-L456))
```python
def seq_to_graph(seq_, seq_rel, pos_enc=False):
    V = np.zeros((seq_len, max_nodes, 4))  # 4 features
    for s in range(seq_len):
        step_rel = seq_rel[:, :, s]  # All 4 features
        V[s, h, :] = step_rel[h]     # Copy all 4 features
    
    if pos_enc:
        V = loc_pos(V)  # Adds positional index → shape becomes (seq_len, N, 5)
    
    return torch.from_numpy(V)
```
**Result**: `V_obs` has shape `(obs_len, N, 5)` = `[pos_enc, LON, LAT, SOG, Heading]` ✅

### 4. **Model Input** ([train.py](train.py))
```python
V_obs = model.v_obs[index]  # Shape: (obs_len, N, 5)
V_pred = model(V_obs, identity)
```

### 5. **Sparse Adjacency Matrix Generation** ([model.py](model.py#L220-L245))
```python
def forward(self, graph, identity):
    # graph shape: (obs_len, N, 5) = [pos_enc, LON, LAT, SOG, Heading]
    
    # SPATIAL GRAPH: Remove pos_enc to get 4D state vector
    spatial_graph = graph[:, :, 1:]  # → (obs_len, N, 4) = [LON, LAT, SOG, Heading] ✅
    
    # TEMPORAL GRAPH: Keep positional encoding
    temporal_graph = graph.permute(1, 0, 2)  # → (N, obs_len, 5) = [pos_enc, LON, LAT, SOG, Heading] ✅
```

### 6. **Self-Attention Mechanism** ([model.py](model.py#L189-L197))
```python
# Initialized with correct dimensions:
self.spatial_attention = SelfAttention(spa_in_dims=4, embedding_dims=64)   # 4D input ✅
self.temporal_attention = SelfAttention(tem_in_dims=5, embedding_dims=64)  # 5D input (with pos) ✅
```

### 7. **GCN Processing** ([model.py](model.py#L306-L350))
```python
# Removes pos_enc before GCN
graph = graph[:, :, :, 1:]  # → shape [batch, seq_len, N, 4] = [LON, LAT, SOG, Heading] ✅
```

---

## **Why The Confusion?**

### **Old Comments Were Misleading!**

❌ **Before** (incorrect comments):
```python
spatial_graph = graph[:, :, 1:]  # (T N 2) ← WRONG!
temporal_graph = graph.permute(1, 0, 2)  # (N T 3) ← WRONG!
```

✅ **After** (corrected):
```python
spatial_graph = graph[:, :, 1:]  # (T N 4) - 4D state vector [LON, LAT, SOG, Heading] ✅
temporal_graph = graph.permute(1, 0, 2)  # (N T 5) - with positional encoding ✅
```

The comments suggested only 2-3 features were being used, but the actual code was correctly using all 4 features!

---

## **Feature Dimensions Summary**

| Stage | Shape | Features | Notes |
|-------|-------|----------|-------|
| **CSV File** | (rows, 6) | `[frame_id, vessel_id, LON, LAT, SOG, Heading]` | Raw data |
| **TrajectoryDataset** | (N, 4, T) | `[LON, LAT, SOG, Heading]` | After loading |
| **seq_to_graph output** | (T, N, 5) | `[pos_enc, LON, LAT, SOG, Heading]` | pos_enc added |
| **Spatial graph input** | (T, N, 4) | `[LON, LAT, SOG, Heading]` | pos_enc removed |
| **Temporal graph input** | (N, T, 5) | `[pos_enc, LON, LAT, SOG, Heading]` | pos_enc kept |
| **GCN processing** | (*, *, 4) | `[LON, LAT, SOG, Heading]` | 4D state vector |

---

## **Why Positional Encoding is Added/Removed?**

### **For Spatial Graph:**
- Represents interactions between vessels **at each time step**
- Position in time is implicitly encoded by the time dimension
- Uses 4D state vector: `(LON, LAT, SOG, Heading)` ✅

### **For Temporal Graph:**
- Represents trajectory evolution **over time**
- Positional encoding helps model understand temporal order (as per Eq. 6 in paper)
- Uses 5D vector: `(pos_enc, LON, LAT, SOG, Heading)` ✅
- Positional encoding is then enhanced by our sinusoidal encoding in the attention mechanism

---

## **Verification**

You can verify the dimensions by adding print statements:

```python
# In model.py SparseWeightedAdjacency.forward():
print(f"Input graph shape: {graph.shape}")  # Should be (obs_len, N, 5)
print(f"Spatial graph shape: {spatial_graph.shape}")  # Should be (obs_len, N, 4)
print(f"Temporal graph shape: {temporal_graph.shape}")  # Should be (N, obs_len, 5)
```

---

## **Final Answer to Your Question**

> "Are you sure the part `spatial_graph = graph[:, :, 1:]` is ok with the paper?"

**YES, it is 100% correct!** ✅

The code properly implements the 4D state vector `(LON, LAT, SOG, Heading)` as specified in the paper. The confusion was entirely due to outdated comments that suggested only 2D coordinates were being used.

---

## **What Changed**

1. ✅ **Fixed misleading comments** to reflect actual 4D state vector
2. ✅ **Added documentation** explaining feature dimensions at each stage
3. ✅ **Verified data files** contain all 4 required features
4. ✅ **Confirmed model architecture** correctly processes 4D state vectors

**No code logic was changed** - only comments were corrected for clarity.

---

*Document created: 2026-03-01*
*Issue: Comments suggested 2D data, but implementation correctly uses 4D state vector*
