"""
check_dataset.py

Diagnostic script to validate the preprocessed dataset BEFORE training.
Run from project root:
    python check_dataset.py

Checks:
  1. Global stats (global_stats.json) — are mean/std reasonable?
  2. Per-split file counts and row counts
  3. Vessel counts per split
  4. Segment length distribution — how many segments are long enough for windows?
  5. Expected sequence count per split (what TrajectoryDataset will produce)
  6. Frame ID continuity — are there unexpected gaps inside segments?
  7. Neighbour density — how many vessels share the same frame_id on average?
"""

import os
import json
import glob
import numpy as np
import pandas as pd

DATASET_NAME = "noaa_dec2021"
DATASET_BASE = os.path.join("dataset", DATASET_NAME)
GLOBAL_STATS_PATH = os.path.join(DATASET_BASE, "global_stats.json")

OBS_LEN = 10
PRED_LEN = 10
SEQ_LEN = OBS_LEN + PRED_LEN
MAX_GAP_MINUTES = 1   # after resampling, gap > 1 min = new segment
MIN_NEIGHBOURS = 1    # minimum other vessels needed per window

SPLITS = ["train", "val", "test"]

# ============================================================================
print("=" * 70)
print("DATASET DIAGNOSTIC REPORT")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1. Global stats
# ----------------------------------------------------------------------------
print("\n[1] GLOBAL STATS (global_stats.json)")
print("-" * 40)

if not os.path.exists(GLOBAL_STATS_PATH):
    print(f"  ERROR: {GLOBAL_STATS_PATH} not found!")
    print("  Run preprocess_ais.py first.")
else:
    with open(GLOBAL_STATS_PATH) as f:
        stats = json.load(f)

    EXPECTED = {
        "LON": {"mean": (-120, -115), "std": (0.5, 5.0)},
        "LAT": {"mean": (30, 35),     "std": (0.5, 5.0)},
        "SOG": {"mean": (1, 22),      "std": (0.5, 10.0)},
        "Heading": {"mean": (0, 360), "std": (50, 150)},
    }

    all_ok = True
    for col, s in stats.items():
        mean, std = s["mean"], s["std"]
        exp = EXPECTED.get(col, {})
        mean_ok = exp.get("mean", (None, None))
        std_ok = exp.get("std", (None, None))

        mean_flag = "✓" if (mean_ok[0] is None or mean_ok[0] <= mean <= mean_ok[1]) else "✗ UNEXPECTED"
        std_flag  = "✓" if (std_ok[0]  is None or std_ok[0]  <= std  <= std_ok[1])  else "✗ UNEXPECTED"

        print(f"  {col:8s}: mean={mean:10.4f} {mean_flag}   std={std:8.4f} {std_flag}")
        if "UNEXPECTED" in mean_flag or "UNEXPECTED" in std_flag:
            all_ok = False

    if all_ok:
        print("  → All stats look reasonable.")
    else:
        print("  → WARNING: Some stats are outside expected ranges.")
        print("    This will cause poor normalization. Check preprocessing.")

# ----------------------------------------------------------------------------
# 2-7. Per-split analysis
# ----------------------------------------------------------------------------
total_expected_sequences = 0

for split in SPLITS:
    split_dir = os.path.join(DATASET_BASE, split)
    csv_files = sorted(glob.glob(os.path.join(split_dir, "day_*.csv")))

    print(f"\n[{SPLITS.index(split)+2}] SPLIT: {split.upper()}")
    print("-" * 40)

    if not csv_files:
        print(f"  No CSV files found in {split_dir}")
        continue

    print(f"  CSV files: {len(csv_files)}")

    # Load all CSVs for this split
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["_file"] = os.path.basename(f)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    total_rows = len(data)
    unique_vessels = data["vessel_id"].nunique()
    unique_frames = data["frame_id"].nunique()

    print(f"  Total rows:      {total_rows:,}")
    print(f"  Unique vessels:  {unique_vessels}")
    print(f"  Unique frame_ids:{unique_frames:,}")

    # ---- Segment analysis per vessel ----
    # A segment = consecutive frame_ids with gap <= MAX_GAP_MINUTES
    seg_lengths = []
    windows_per_vessel = []

    for vid, vdf in data.groupby("vessel_id"):
        fids = vdf["frame_id"].sort_values().values.astype(int)

        # Split into segments
        gaps = np.diff(fids)
        break_pts = np.where(gaps > MAX_GAP_MINUTES)[0] + 1
        seg_starts = np.concatenate([[0], break_pts])
        seg_ends   = np.concatenate([break_pts, [len(fids)]])

        v_windows = 0
        for s, e in zip(seg_starts, seg_ends):
            seg_len = e - s
            seg_lengths.append(seg_len)
            if seg_len >= SEQ_LEN:
                v_windows += (seg_len - SEQ_LEN) + 1  # skip=1
        windows_per_vessel.append(v_windows)

    seg_lengths = np.array(seg_lengths)
    windows_per_vessel = np.array(windows_per_vessel)

    n_segs = len(seg_lengths)
    n_segs_valid = (seg_lengths >= SEQ_LEN).sum()

    print(f"\n  Segments (gap>{MAX_GAP_MINUTES}min = new segment):")
    print(f"    Total segments:          {n_segs:,}")
    print(f"    Segments >= {SEQ_LEN} points:   {n_segs_valid:,}  ({100*n_segs_valid/max(1,n_segs):.1f}% usable)")
    print(f"    Segment length percentiles:")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        print(f"      {pct:3d}%: {np.percentile(seg_lengths, pct):.0f} points")

    # ---- Expected sequence count ----
    # This is the UPPER BOUND — actual count will be lower because
    # windows also need >= MIN_NEIGHBOURS other vessels at every timestep.
    raw_windows = int(windows_per_vessel.sum())
    print(f"\n  Expected sequences (upper bound, skip=1):")
    print(f"    Raw windows from segments: {raw_windows:,}")
    print(f"    (Actual will be lower after neighbour filter)")
    total_expected_sequences += raw_windows

    # ---- Neighbour density ----
    # Average vessels per frame_id (proxy for interaction density)
    vpf = data.groupby("frame_id")["vessel_id"].nunique()
    print(f"\n  Vessels per frame_id (interaction density):")
    print(f"    Mean:   {vpf.mean():.1f}")
    print(f"    Median: {vpf.median():.1f}")
    print(f"    Min:    {vpf.min()}")
    print(f"    Max:    {vpf.max()}")
    pct_lonely = (vpf < 2).mean() * 100
    print(f"    Frames with < 2 vessels: {pct_lonely:.1f}%  ← these windows lose neighbours")

    # ---- Frame ID gap check ----
    # Are there unexpected large gaps inside what should be continuous segments?
    all_fids = np.sort(data["frame_id"].unique())
    if len(all_fids) > 1:
        fid_gaps = np.diff(all_fids)
        large_gaps = (fid_gaps > 1).sum()
        print(f"\n  Frame_id continuity:")
        print(f"    Range: {all_fids[0]} → {all_fids[-1]}")
        print(f"    Gaps > 1 min between consecutive frame_ids: {large_gaps}")
        if large_gaps == 0:
            print(f"    → Frame_ids are fully continuous (expected for day-level CSVs).")
        else:
            print(f"    → Gaps exist (expected: multiple days or segment breaks).")

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Dataset: {DATASET_NAME}")
print(f"  obs_len={OBS_LEN}, pred_len={PRED_LEN}, seq_len={SEQ_LEN}")
print(f"  Total raw windows across all splits: {total_expected_sequences:,}")
print()
print("  VERDICT:")

issues = []

if os.path.exists(GLOBAL_STATS_PATH):
    with open(GLOBAL_STATS_PATH) as f:
        stats = json.load(f)
    if stats["LON"]["std"] < 0.1 or stats["LAT"]["std"] < 0.1:
        issues.append("Global stats std is suspiciously small → check normalization")

for split in SPLITS:
    split_dir = os.path.join(DATASET_BASE, split)
    csv_files = glob.glob(os.path.join(split_dir, "day_*.csv"))
    if not csv_files:
        issues.append(f"No CSV files in {split} split")

if total_expected_sequences < 1000:
    issues.append(f"Very few sequences ({total_expected_sequences}) — dataset may be too small to train well")
elif total_expected_sequences < 10000:
    issues.append(f"Moderate sequence count ({total_expected_sequences}) — training should work but paper had more")

if issues:
    for issue in issues:
        print(f"  ⚠  {issue}")
else:
    print("  ✓  Dataset looks healthy. Safe to proceed with training.")

print("=" * 70)
