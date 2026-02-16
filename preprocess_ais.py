"""
NOAA AIS Data Preprocessing Script (paper-aligned)

Creates frame-format CSV files compatible with TrajectoryDataset:
  frame_id, vessel_id, LON, LAT, SOG, Heading

Usage:
  python preprocess_ais.py
"""

import os
import json
from utils import load_noaa_csv, preprocess_noaa_to_frames, save_frames_csv


def main():
    # ----------------------------
    # Input
    # ----------------------------
    # Prefer the .zip directly (saves disk space and avoids manual extraction)
    input_path = "data/raw/2020_12/AIS_2020_12_01.zip" 

    # If you pass a zip and want a specific inner csv name, set it; otherwise first csv is used
    inner_csv_name = None

    # Optional: for quick sanity runs, set nrows (e.g., 2_000_000). Use None for full day.
    nrows = None

    # ----------------------------
    # Paper preprocessing params
    # ----------------------------
    obs_len = 10
    pred_len = 30  # NOTE: used by TrajectoryDataset when training, not here

    lat_range = (30.0, 35.0)
    lon_range = (-120.0, -115.0)
    sog_range = (1.0, 22.0)
    heading_range = (0.0, 360.0)
    min_vessels_per_timestamp = 3
    do_zscore = True

    # ----------------------------
    # Output
    # ----------------------------
    dataset_name = "noaa_dec2020"
    out_dir = os.path.join("dataset", dataset_name, "train")
    out_csv = os.path.join(out_dir, "day_2020_12_01.csv")
    out_stats = os.path.join(out_dir, "day_2020_12_01_stats.json")

    os.makedirs(out_dir, exist_ok=True)

    print("Loading NOAA AIS data...")
    df_raw = load_noaa_csv(input_path, inner_csv_name=inner_csv_name, nrows=nrows)
    print(f"Loaded rows: {len(df_raw)}")

    print("Running paper-aligned preprocessing to frame format...")
    frames_df, stats = preprocess_noaa_to_frames(
        df_raw,
        lat_range=lat_range,
        lon_range=lon_range,
        sog_range=sog_range,
        heading_range=heading_range,
        min_vessels_per_timestamp=min_vessels_per_timestamp,
        do_zscore=do_zscore,
    )

    print(f"Processed frame rows: {len(frames_df)}")
    print("Saving CSV for TrajectoryDataset...")
    save_frames_csv(frames_df, out_csv)
    print(f"Saved: {out_csv}")

    if stats is not None:
        with open(out_stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved normalization stats: {out_stats}")

    print("\nDone.")
    print(f"Next: point your training args.dataset to '{dataset_name}' and run train.py.")


if __name__ == "__main__":
    main()
