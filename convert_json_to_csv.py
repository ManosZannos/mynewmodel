"""
convert_json_to_csv.py

Converts METO-S2S JSON to frame-format CSV for TrajectoryDataset.

JSON point format (verified against max_min.json):
  [timestamp_ms, lat_norm, lon_norm, sog_norm, heading_norm,
   distance, vessel_type, month, day, hour, lon_raw, lat_raw, mmsi]

  NOTE: JSON[1] = LAT_norm, JSON[2] = LON_norm  (lat comes first!)

Normalization (from max_min.json):
  lat_norm = (lat - 20.90883) / 28.32044
  lon_norm = (lon - (-133.29703)) / 72.60811

Output CSV: frame_id, vessel_id, LON, LAT, SOG, Heading
  LON col = lon_norm (JSON[2])
  LAT col = lat_norm (JSON[1])

global_stats.json stores the denormalization constants:
  LON: mean=min=-133.29703, std=range=72.60811
  LAT: mean=min=20.90883,   std=range=28.32044

Usage (run from project root):
  python convert_json_to_csv.py
"""

import os
import json
import numpy as np
import pandas as pd

DATASET_DIR = "dataset/marinecadastre_2021"
SPLITS      = ["train", "val", "test"]

# JSON point field indices (verified)
IDX_TIMESTAMP = 0
IDX_LAT_NORM  = 1   # NOTE: lat comes first in JSON
IDX_LON_NORM  = 2
IDX_SOG_NORM  = 3
IDX_HEAD_NORM = 4
IDX_LON_RAW   = 10
IDX_LAT_RAW   = 11
IDX_MMSI      = 12

# Normalization constants from max_min.json
LAT_MIN   =  20.90883
LAT_RANGE =  28.32044
LON_MIN   = -133.29703
LON_RANGE =  72.60811


def load_json(split):
    path = os.path.join(DATASET_DIR, f"{split}.json")
    print(f"  Loading {path} ...")
    with open(path) as f:
        data = json.load(f)
    print(f"  {len(data)} trajectories")
    return data


def save_global_stats():
    """Save denormalization constants derived from max_min.json."""
    stats = {
        "LON": {"mean": LON_MIN,   "std": LON_RANGE},
        "LAT": {"mean": LAT_MIN,   "std": LAT_RANGE},
        "SOG": {"mean": 0.0,       "std": 30.695},
        "Heading": {"mean": 0.0,   "std": 3.141592653589793},
        "_note": "min-max normalization: actual = norm * std + mean"
    }
    path = os.path.join(DATASET_DIR, "global_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {path}")
    print(f"  LON: [{LON_MIN:.5f}, {LON_MIN+LON_RANGE:.5f}]")
    print(f"  LAT: [{LAT_MIN:.5f}, {LAT_MIN+LAT_RANGE:.5f}]")
    return stats


def json_to_csv(data, out_path):
    rows = []
    for traj in data:
        if not traj:
            continue
        mmsi = int(traj[0][IDX_MMSI])
        for point in traj:
            frame_id = int(point[IDX_TIMESTAMP]) // (10 * 60 * 1000)
            rows.append({
                "frame_id":  frame_id,
                "vessel_id": mmsi,
                "LON":       point[IDX_LON_NORM],   # JSON[2] = lon_norm
                "LAT":       point[IDX_LAT_NORM],   # JSON[1] = lat_norm
                "SOG":       point[IDX_SOG_NORM],
                "Heading":   point[IDX_HEAD_NORM],
            })

    df = pd.DataFrame(rows).sort_values(["frame_id", "vessel_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"  Rows:    {len(df):,}")
    print(f"  Vessels: {df['vessel_id'].nunique():,}")
    print(f"  Saved:   {out_path}")
    return df


def main():
    print("=" * 60)
    print("METO-S2S JSON → CSV Converter")
    print("=" * 60)

    print("\n[1/3] Saving global stats from max_min.json ...")
    save_global_stats()

    print("\n[2/3] Converting splits ...")
    for split in SPLITS:
        print(f"\n  [{split.upper()}]")
        data = load_json(split)
        out_csv = os.path.join(DATASET_DIR, split, f"day_{split}.csv")
        json_to_csv(data, out_csv)

    print("\n[3/3] Done!")
    print("=" * 60)
    print("Next steps:")
    print("  python check_dataset.py")
    print("  python train.py --dataset marinecadastre_2021 \\")
    print("    --obs_len 10 --pred_len 5 --num_epochs 200 \\")
    print("    --lr 0.00001 --clip_grad 1.0")
    print("=" * 60)


if __name__ == "__main__":
    main()