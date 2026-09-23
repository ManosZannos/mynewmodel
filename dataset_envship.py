"""
dataset_envship.py — Data loader adapter for the EnvShip NOAA Track A benchmark.

Source: mark000071/envship_v2_datasets (HuggingFace), NOAA Track A subset.
    https://huggingface.co/datasets/mark000071/envship_v2_datasets

Track A: 10-min observation, 10-min prediction, 30+30 points @ 20s.
NOAA split sizes (from the dataset card): train=48,000 / val=6,000 / test=6,000.

=====================================================================
STATUS: PHASE 1 ONLY (trajectory-only core). Read this before using.
=====================================================================

CONFIRMED from the HuggingFace dataset card (safe to rely on):
  - Main CSV path pattern:
      track_a_short-term_Cross-domain_Datasets/noaa_track_v1/{split}/part-000.csv.gz
  - Columns present: hist_x_json, hist_y_json, fut_x_json, fut_y_json
      each a JSON list of floats, length 30, in a TARGET-VESSEL-CENTERED
      local East/North metric frame (already in meters — NOT lon/lat,
      NO geodetic conversion needed for this dataset).
  - Columns present: osm_temporal_consistent (bool-as-string 'true'/'false'),
      osm_max_inland_depth_m, osm_n_inland_points, osm_max_consec_inland_run.
      Paper-default filter: keep only osm_temporal_consistent == 'true'.

NOT YET CONFIRMED (do not trust until verified against a real downloaded file):
  - Exact column names for SOG / heading in the main CSV, if present at all.
    This loader currently DERIVES speed and heading from consecutive
    (x, y) position deltas instead of relying on a named column, which is
    a defensible fallback but should be checked against the real file —
    if an explicit SOG/heading (or COG) column exists, prefer that.
  - The neighbor/social context format under `context_v1/social/` (up to
    10 neighbors within 3km, with precomputed CPA/TCPA per the paper) —
    the dataset card describes this only at a high level ("compact AIS
    snapshots for neighbour lookup"), with no per-column schema and no
    confirmed join key back to the main CSV rows.
  - Without neighbor context, N=1 per sample and CPAFeatures/NeighborAggregation
    in model_cpagrn.py degenerate to a no-op (a single vessel has no valid
    neighbor to attend to). This Phase-1 loader is therefore only sufficient
    for a TRAJECTORY-ONLY comparison (vs. their LSTM/TCN/Transformer-NAR
    baselines) — NOT yet sufficient for a fair CPA-GRN vs. Gated-Neighbor-
    Attention comparison. That needs Phase 2.

NEXT STEP before writing Phase 2: download one real part-000.csv.gz plus
one real file from context_v1/social/, and run inspect_columns() below on
each. Report back the actual column names so Phase 2 can be written against
verified ground truth instead of guessed field names.
"""

from __future__ import annotations
import os
import json
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Schema constants — confirmed ones are marked; adjust the rest after
# inspect_columns() has been run on a real downloaded file.
# ---------------------------------------------------------------------------
HIST_X_COL = "hist_x_json"     # confirmed
HIST_Y_COL = "hist_y_json"     # confirmed
FUT_X_COL  = "fut_x_json"      # confirmed
FUT_Y_COL  = "fut_y_json"      # confirmed
OSM_FLAG_COL = "osm_temporal_consistent"  # confirmed

T_OBS_ENVSHIP  = 30   # 10 min @ 20s
T_PRED_ENVSHIP = 30   # 10 min @ 20s


def inspect_columns(csv_gz_path: str) -> None:
    """Run this FIRST on a real downloaded part-000.csv.gz before trusting
    anything else in this file. Prints every column name and a sample value,
    so the schema constants above (and the derived-heading fallback) can be
    checked or corrected."""
    df = pd.read_csv(csv_gz_path, nrows=5)
    print(f"\n{'='*60}\nColumns in {os.path.basename(csv_gz_path)}:\n{'='*60}")
    for col in df.columns:
        sample = df[col].iloc[0]
        sample_str = str(sample)[:80]
        print(f"  {col:<35} dtype={str(df[col].dtype):<10} sample={sample_str}")
    print(f"{'='*60}\n")


def _decode_xy(json_str: str) -> np.ndarray:
    return np.asarray(json.loads(json_str), dtype=np.float32)


def _derive_speed_heading(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: derive per-step speed (m per 20s step) and heading (radians)
    from consecutive position deltas. First step repeats the second step's
    value (no earlier delta available). Replace with real SOG/heading
    columns if inspect_columns() finds them in the source CSV."""
    deltas = np.diff(xy, axis=0, prepend=xy[[0]])  # [T, 2], first row = 0
    deltas[0] = deltas[1] if len(deltas) > 1 else deltas[0]
    speed   = np.linalg.norm(deltas, axis=-1)
    heading = np.arctan2(deltas[:, 1], deltas[:, 0])
    return speed, heading


class EnvShipNOAATrackA(Dataset):
    """
    PHASE 1 — trajectory-only. Each item is a SINGLE target vessel
    (no neighbors yet — see module docstring). Returned shapes match
    model_cpagrn.CPAGRN's expected input up to the N=1 degenerate case:

        obs   : [1, T_obs,  4]   (x, y, speed, heading) — NOT z-scored here;
                                  normalization stats are computed dataset-wide
                                  and applied in get_dataloaders_envship().
        pred  : [1, T_pred, 2]   (x, y) absolute, target-vessel-centered
        mask  : [1]              always True (single vessel per sample)

    NOTE: feeding this into CPAGRN as-is trains/evaluates only the temporal
    (GRU) pathway — the CPA graph attention has nothing to attend to with
    N=1. Valid for a trajectory-only ablation; NOT valid yet for comparing
    against EnvShip's Gated-Neighbor-Attention baseline. See Phase 2 TODO.
    """

    def __init__(self, data_dir: str, split: str, apply_osm_filter: bool = True):
        pattern = os.path.join(data_dir, split, "part-*.csv.gz")
        files = sorted(glob.glob(pattern))
        assert files, f"No files matched {pattern} — check data_dir/split."

        frames = []
        for f in files:
            df = pd.read_csv(f)
            if apply_osm_filter and OSM_FLAG_COL in df.columns:
                df = df[df[OSM_FLAG_COL].astype(str) == "true"].reset_index(drop=True)
            frames.append(df)
        self.df = pd.concat(frames, ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        hist_xy = np.column_stack([
            _decode_xy(row[HIST_X_COL]), _decode_xy(row[HIST_Y_COL])
        ])  # [T_obs, 2]
        fut_xy = np.column_stack([
            _decode_xy(row[FUT_X_COL]), _decode_xy(row[FUT_Y_COL])
        ])  # [T_pred, 2]

        speed, heading = _derive_speed_heading(hist_xy)  # fallback, see docstring

        obs = np.concatenate([
            hist_xy,
            speed[:, None],
            heading[:, None],
        ], axis=-1).astype(np.float32)  # [T_obs, 4]

        return obs, fut_xy.astype(np.float32)


def _collate_single_vessel(batch):
    """N=1 per sample -> stack into [B, 1, T, F] to match CPAGRN's [B, N, T, F]."""
    obs_list, fut_list = zip(*batch)
    obs = torch.from_numpy(np.stack(obs_list))[:, None, :, :]   # [B, 1, T_obs, 4]
    fut = torch.from_numpy(np.stack(fut_list))[:, None, :, :]   # [B, 1, T_pred, 2]
    mask = torch.ones(obs.shape[0], 1, dtype=torch.bool)        # [B, 1]
    return obs, fut, mask, None


def get_dataloaders_envship(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    """Mirrors dataset.get_dataloaders()'s call signature so
    train_cpagrn_envship.py can swap the import with minimal changes.
    obs_len/pred_len are fixed at 30/30 by the EnvShip Track A definition —
    they are NOT configurable here the way they are for our own dataset."""
    train_ds = EnvShipNOAATrackA(data_dir, "train")
    val_ds   = EnvShipNOAATrackA(data_dir, "val")
    test_ds  = EnvShipNOAATrackA(data_dir, "test")

    # Normalization stats from train split only (x, y, speed, heading)
    all_obs = np.stack([train_ds[i][0] for i in range(len(train_ds))])  # [N,T,4]
    stats = {
        "X":       {"mean": float(all_obs[..., 0].mean()), "std": float(all_obs[..., 0].std())},
        "Y":       {"mean": float(all_obs[..., 1].mean()), "std": float(all_obs[..., 1].std())},
        "SPEED":   {"mean": float(all_obs[..., 2].mean()), "std": float(all_obs[..., 2].std())},
        "HEADING": {"mean": float(all_obs[..., 3].mean()), "std": float(all_obs[..., 3].std())},
    }

    kwargs = dict(batch_size=batch_size, collate_fn=_collate_single_vessel,
                  num_workers=num_workers)
    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, stats


# ---------------------------------------------------------------------------
# PHASE 2 — NOT YET IMPLEMENTED. Do not call.
# ---------------------------------------------------------------------------
def load_neighbor_context(*args, **kwargs):
    raise NotImplementedError(
        "Phase 2 (neighbor/CPA context from context_v1/social/) is not yet "
        "written — the real column schema has not been verified. Run "
        "inspect_columns() on a real downloaded social-context file first, "
        "then this function can be implemented against confirmed field names "
        "instead of guessed ones."
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python dataset_envship.py <path-to-part-000.csv.gz>")
        sys.exit(1)
    inspect_columns(sys.argv[1])
