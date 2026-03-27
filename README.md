# SMCHN — Sparse Multi-graph Convolutional Hybrid Network for Vessel Trajectory Prediction

Faithful reconstruction of the SMCHN model proposed in:

> Wang et al., "Big data driven vessel trajectory prediction based on sparse multi-graph convolutional hybrid network with spatio-temporal awareness", *Ocean Engineering*, 2023.

---

## Results

| Metric | Paper | This implementation |
|--------|-------|-------------------|
| minADE-20 (10-min) | 0.0010° | 0.0015° |
| FDE (10-min) | 0.0012° | 0.0021° |

Dataset: AIS data, San Diego Harbor, December 2021 ([NOAA AIS Data](https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2021/index.html))

---

## Project Structure

```
├── model.py            # SMCHN architecture
├── train.py            # Training loop
├── evaluate.py         # Evaluation with best-of-20 sampling
├── utils.py            # Dataset, preprocessing utilities
├── metrics.py          # Loss function, sampling, metrics
├── preprocess_ais.py   # AIS data preprocessing pipeline
├── smoke_test.py       # Quick sanity check
├── check_dataset.py    # Dataset validation
└── experiments.md      # Experiment log
```

---

## Setup

```bash
pip install torch numpy pandas tqdm tensorboard termcolor
```

---

## Data Preprocessing

Download AIS data from [NOAA AIS Data Handler](https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2021/index.html) and place zip files in `data/raw/2021_12/`.

```bash
python preprocess_ais.py
```

This will create `dataset/noaa_dec2021/` with train/val/test splits (6:2:2).

---

## Training

```bash
python train.py \
  --dataset noaa_dec2021 \
  --tag SMCHN \
  --obs_len 10 \
  --pred_len 5 \
  --num_epochs 200 \
  --lr 0.00001 \
  --clip_grad 1.0
```

Monitor training:
```bash
grep "train_loss\|val_loss" Logs_train/$(ls -t Logs_train/ | head -1)
```

---

## Evaluation

```bash
python evaluate.py \
  --checkpoint checkpoints/SMCHN/noaa_dec2021/val_best.pth \
  --dataset noaa_dec2021 \
  --split test \
  --num_samples 20
```

---

## Model Architecture

```
V_obs [obs_len, N, 5]  (velocities + positional encoding)
        ↓
SparseWeightedAdjacency
  └── Multi-head Self-Attention (spatial + temporal)
  └── InteractionMask (AsymmetricConvolution × 2)
  └── ZeroSoftmax normalization
        ↓
SparseGraphConvolution
  └── Spatial GCN path + Temporal GCN path → addition fusion
        ↓
Gated TCN Encoder (Conv2d, 3 layers, residual connection)
        ↓
Linear → Bivariate Gaussian parameters [pred_len, N, 5]
```

**Key parameters:** embedding_dims=64, num_heads=4, gcn_hidden=16, TCN kernel=3, threshold ξ=0.5

---

## Inference

The model predicts velocity distributions. During evaluation:
1. Sample K=20 trajectories from the predicted Bivariate Gaussian
2. Convert velocities to absolute positions via cumulative sum
3. Denormalize to real-world coordinates (degrees)
4. Select the sample closest to ground truth (minADE criterion)

---

## Citation

```bibtex
@article{wang2023smchn,
  title={Big data driven vessel trajectory prediction based on sparse 
         multi-graph convolutional hybrid network with spatio-temporal awareness},
  author={Wang, Siwen and Li, Ying and Zhang, Zhaoyi and Xing, Hu},
  journal={Ocean Engineering},
  volume={287},
  pages={115695},
  year={2023},
  publisher={Elsevier}
}
```