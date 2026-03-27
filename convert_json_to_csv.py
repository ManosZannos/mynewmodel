"""
convert_json_to_csv.py

Converts METO-S2S JSON files to frame-format CSV files
compatible with TrajectoryDataset (frame_id, vessel_id, LON, LAT, SOG, Heading).

JSON format (per point):
  [timestamp_ms, lon_norm, lat_norm, sog_norm, heading_norm,
   distance, vessel_type, month, day, hour, lon_raw, lat_raw, mmsi]

Output CSV format (required by TrajectoryDataset):
  frame_id, vessel_id, LON, LAT, SOG, Heading

frame_id = timestamp_ms // (10 * 60 * 1000)  → 10-min units

Also computes global_stats.json from raw LON/LAT in train.json.

Usage (run from project root):
  python convert_json_to_csv.py
"""

import os
import json
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DATASET_DIR = "dataset/marinecadastre_2021"
SPLITS      = ["train", "val", "test"]

# JSON point field indices
IDX_TIMESTAMP = 0
IDX_LON_NORM  = 1
IDX_LAT_NORM  = 2
IDX_SOG_NORM  = 3
IDX_HEAD_NORM = 4
IDX_LON_RAW   = 10
IDX_LAT_RAW   = 11
IDX_MMSI      = 12


def load_json(split: str) -> list:
    path = os.path.join(DATASET_DIR, f"{split}.json")
    print(f"  Loading {path} ...")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  {len(data)} trajectories loaded")
    return data


def compute_global_stats(train_data: list) -> dict:
    """Compute mean/std from raw LON/LAT in train.json for denormalization."""
    lons, lats = [], []
    for traj in train_data:
        for point in traj:
            lons.append(point[IDX_LON_RAW])
            lats.append(point[IDX_LAT_RAW])

    lons = np.array(lons)
    lats = np.array(lats)

    stats = {
        "LON": {
            "mean": float(lons.mean()),
            "std":  float(lons.std()) if lons.std() > 0 else 1.0,
        },
        "LAT": {
            "mean": float(lats.mean()),
            "std":  float(lats.std()) if lats.std() > 0 else 1.0,
        },
        "SOG":     {"mean": 0.0, "std": 1.0},
        "Heading": {"mean": 0.0, "std": 1.0},
    }
    return stats


def json_to_csv(data: list, out_path: str):
    """
    Convert list of trajectories to frame-format CSV.

    frame_id = timestamp_ms // (10 * 60 * 1000)
    vessel_id = MMSI
    LON, LAT, SOG, Heading = normalized values (already from METO-S2S)
    """
    rows = []
    for traj in data:
        if len(traj) == 0:
            continue
        mmsi = traj[0][IDX_MMSI]
        for point in traj:
            frame_id = int(point[IDX_TIMESTAMP]) // (10 * 60 * 1000)
            rows.append({
                "frame_id":  frame_id,
                "vessel_id": int(mmsi),
                "LON":       point[IDX_LON_NORM],
                "LAT":       point[IDX_LAT_NORM],
                "SOG":       point[IDX_SOG_NORM],
                "Heading":   point[IDX_HEAD_NORM],
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["frame_id", "vessel_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"  Rows:        {len(df):,}")
    print(f"  Vessels:     {df['vessel_id'].nunique():,}")
    print(f"  Frame range: {df['frame_id'].min()} → {df['frame_id'].max()}")
    print(f"  Saved:       {out_path}")

    return df


def main():
    print("=" * 60)
    print("METO-S2S JSON → CSV Converter")
    print("=" * 60)

    # Step 1: Load train + compute global stats
    print("\n[1/4] Computing global stats from train.json ...")
    train_data = load_json("train")
    stats = compute_global_stats(train_data)

    stats_path = os.path.join(DATASET_DIR, "global_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  LON: mean={stats['LON']['mean']:.4f}, std={stats['LON']['std']:.4f}")
    print(f"  LAT: mean={stats['LAT']['mean']:.4f}, std={stats['LAT']['std']:.4f}")
    print(f"  Saved: {stats_path}")

    # Step 2: Convert each split
    print("\n[2/4] Converting splits ...")
    for split in SPLITS:
        print(f"\n  [{split.upper()}]")
        if split == "train":
            data = train_data
        else:
            data = load_json(split)
        out_csv = os.path.join(DATASET_DIR, split, f"day_{split}.csv")
        json_to_csv(data, out_csv)

    # Step 3: Summary
    print("\n[3/4] Final structure:")
    for split in SPLITS:
        csv_path = os.path.join(DATASET_DIR, split, f"day_{split}.csv")
        size_mb  = os.path.getsize(csv_path) / 1024 / 1024
        print(f"  {csv_path}  ({size_mb:.1f} MB)")
    print(f"  {stats_path}")

    print("\n[4/4] Done!")
    print("=" * 60)
    print("Next steps:")
    print("  python check_dataset.py")
    print("  python train.py --dataset marinecadastre_2021 \\")
    print("    --obs_len 10 --pred_len 5 --num_epochs 200 \\")
    print("    --lr 0.00001 --clip_grad 1.0")
    print("=" * 60)


if __name__ == "__main__":
    main()
