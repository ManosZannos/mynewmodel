"""
NOAA AIS Data Preprocessing Script (paper-aligned)

Creates frame-format CSV files compatible with TrajectoryDataset:
  frame_id, vessel_id, LON, LAT, SOG, Heading

Processes all AIS files from a raw data folder and splits them into train/val/test.

TWO-PASS APPROACH (for dataset-level z-score normalization):
  Pass 1: Compute global statistics from train data only
  Pass 2: Apply those statistics to all splits (train/val/test)

Usage:
  python preprocess_ais.py
"""

import os
import json
import glob
import re
import numpy as np
import pandas as pd
from utils import (
    load_noaa_csv, 
    preprocess_noaa_to_frames, 
    save_frames_csv,
    clean_abnormal_data_noaa,
    resample_interpolate_1min,
    filter_timestamps_min_vessels,
)


def get_day_from_filename(filename):
    """Extract day number from AIS filename (e.g., AIS_2021_12_01.zip -> 1)"""
    match = re.search(r'AIS_\d{4}_\d{2}_(\d{2})\.zip', filename)
    if match:
        return int(match.group(1))
    return None


def get_date_str_from_filename(filename):
    """Extract full date string from AIS filename (e.g., AIS_2021_12_01.zip -> 2021_12_01)"""
    match = re.search(r'AIS_(\d{4}_\d{2}_\d{2})\.zip', filename)
    if match:
        return match.group(1)
    return None


def main():
    # ----------------------------
    # Input
    # ----------------------------
    raw_data_folder = "data/raw/2021_12"  # Folder with all AIS_*.zip files
    inner_csv_name = None  # Auto-detect first CSV in each zip
    nrows = None  # Use None for full processing, or set limit for testing (e.g., 2_000_000)

    # ----------------------------
    # Train/Val/Test Split (Paper: 6:2:2 ratio)
    # ----------------------------
    # Approximate 6:2:2 split over 31 days
    # Train: days 1-19 (19 days)
    # Val:   days 20-25 (6 days)
    # Test:  days 26-31 (6 days)
    
    train_days = list(range(1, 20))    # Days 1-19 → train/ (60%)
    val_days = list(range(20, 26))     # Days 20-25 → val/ (20%)
    test_days = list(range(26, 32))    # Days 26-31 → test/ (20%)

    #train_days = [1,2,3,4,5]
    #val_days = [6]
    #test_days = []

    # ----------------------------
    # Paper preprocessing params
    # ----------------------------
    lat_range = (30.0, 35.0)
    lon_range = (-120.0, -115.0)
    sog_range = (1.0, 22.0)
    heading_range = (0.0, 360.0)
    min_vessels_per_timestamp = 3  # Paper's value
    max_gap_minutes = 10           # Gaps > 10 min split vessel trajectory into segments
                                   # before resampling (prevents fake interpolation)

    # ----------------------------
    # Output
    # ----------------------------
    dataset_name = "noaa_dec2021"
    dataset_base = os.path.join("dataset", dataset_name)
    global_stats_path = os.path.join(dataset_base, "global_stats.json")

    # Find all AIS zip files
    zip_files = sorted(glob.glob(os.path.join(raw_data_folder, "AIS_*.zip")))
    
    if not zip_files:
        print(f"ERROR: No AIS_*.zip files found in {raw_data_folder}")
        return

    # Categorize files by split
    train_files = []
    val_files = []
    test_files = []
    
    for zip_path in zip_files:
        filename = os.path.basename(zip_path)
        day_num = get_day_from_filename(filename)
        if day_num is None:
            continue
        if day_num in train_days:
            train_files.append(zip_path)
        elif day_num in val_days:
            val_files.append(zip_path)
        elif day_num in test_days:
            test_files.append(zip_path)

    print(f"\n{'='*80}")
    print(f"DATASET SPLIT OVERVIEW")
    print(f"{'='*80}")
    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Test files:  {len(test_files)}")
    print(f"{'='*80}\n")

    # =========================================================================
    # PASS 1: Compute global statistics from TRAIN data only (streaming)
    # =========================================================================
    # Uses streaming/online statistics (Welford's algorithm) to avoid loading
    # all train data into memory simultaneously. This is memory-efficient for
    # large datasets (e.g., 19 days × 1-2M rows/day = 20-40M rows).
    #
    # Traditional approach: Keep all train data → pd.concat() → compute stats
    # Memory footprint: ~2-4GB for data + 2-4GB for concat = 4-8GB peak
    #
    # Streaming approach: Process one day at a time, update running statistics
    # Memory footprint: ~100-200MB per day (no accumulation)
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"PASS 1: Computing global statistics from TRAIN data (streaming)")
    print(f"{'='*80}\n")
    
    # Initialize streaming statistics accumulators (Welford's algorithm)
    # For each feature: count, running mean, M2 (sum of squared deviations)
    stats_cols = ["LON", "LAT", "SOG", "Heading"]
    streaming_stats = {col: {"count": 0, "mean": 0.0, "M2": 0.0} for col in stats_cols}
    
    total_train_rows = 0
    total_train_vessels = set()
    
    for zip_path in train_files:
        filename = os.path.basename(zip_path)
        day_num = get_day_from_filename(filename)
        date_str = get_date_str_from_filename(filename)
        
        if day_num is None or date_str is None:
            print(f"Skipping {filename} - couldn't parse date")
            continue

        print(f"[TRAIN] Processing day {day_num:02d}: {filename}")
        
        # Load and preprocess WITHOUT z-score normalization
        try:
            df_raw = load_noaa_csv(zip_path, inner_csv_name=inner_csv_name, nrows=nrows)
            print(f"  Loaded: {len(df_raw):,} rows")
            
            # Steps 1-3: clean, resample, filter (no z-score yet)
            df = clean_abnormal_data_noaa(df_raw, lat_range, lon_range, sog_range, heading_range)
            df = resample_interpolate_1min(df, freq="1min", rolling_window=5, max_gap_minutes=max_gap_minutes)
            df = filter_timestamps_min_vessels(df, min_vessels_per_timestamp)
            
            # Update streaming statistics (batch Welford's algorithm)
            # More efficient than per-value updates
            for col in stats_cols:
                values = df[col].values
                n_new = len(values)
                if n_new == 0:
                    continue
                
                # Compute statistics for this batch
                mean_new = float(values.mean())
                var_new = float(values.var(ddof=1)) if n_new > 1 else 0.0
                M2_new = var_new * (n_new - 1)
                
                # Combine with existing statistics (parallel Welford's)
                acc = streaming_stats[col]
                n_old = acc["count"]
                
                if n_old == 0:
                    # First batch
                    acc["count"] = n_new
                    acc["mean"] = mean_new
                    acc["M2"] = M2_new
                else:
                    # Combine two batches
                    n_combined = n_old + n_new
                    delta = mean_new - acc["mean"]
                    acc["mean"] = acc["mean"] + delta * n_new / n_combined
                    acc["M2"] = acc["M2"] + M2_new + delta**2 * n_old * n_new / n_combined
                    acc["count"] = n_combined
            
            total_train_rows += len(df)
            total_train_vessels.update(df["MMSI"].unique())
            print(f"  Processed: {len(df):,} rows\n")
            
            # Free memory immediately
            del df
            del df_raw
            
        except Exception as e:
            print(f"ERROR processing {filename}: {e}\n")
            continue
    
    # Check if we collected any data
    if streaming_stats["LON"]["count"] == 0:
        print("ERROR: No train data collected!")
        return
    
    print(f"\nTotal train data processed:")
    print(f"  Rows: {total_train_rows:,}")
    print(f"  Vessels: {len(total_train_vessels)}")
    
    # Compute final statistics from streaming accumulators
    print(f"\nComputing global statistics (LON, LAT, SOG, Heading)...")
    global_stats = {}
    for col in stats_cols:
        count = streaming_stats[col]["count"]
        mean = streaming_stats[col]["mean"]
        variance = streaming_stats[col]["M2"] / (count - 1) if count > 1 else 0.0
        std = np.sqrt(variance)
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        global_stats[col] = {"mean": float(mean), "std": float(std)}
        print(f"  {col}: μ={mean:.6f}, σ={std:.6f}")
    
    # Save global statistics
    os.makedirs(dataset_base, exist_ok=True)
    with open(global_stats_path, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, indent=2)
    print(f"\n✓ Saved global statistics: {global_stats_path}")
    
    # Clean up memory
    del streaming_stats
    del total_train_vessels
    
    # =========================================================================
    # PASS 2: Process all files with global statistics
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"PASS 2: Processing all files with global statistics")
    print(f"{'='*80}\n")
    
    all_files = [
        (zip_path, "train") for zip_path in train_files
    ] + [
        (zip_path, "val") for zip_path in val_files
    ] + [
        (zip_path, "test") for zip_path in test_files
    ]
    
    for zip_path, split in all_files:
        filename = os.path.basename(zip_path)
        day_num = get_day_from_filename(filename)
        date_str = get_date_str_from_filename(filename)
        
        if day_num is None or date_str is None:
            print(f"Skipping {filename} - couldn't parse date")
            continue

        # Output paths
        out_dir = os.path.join(dataset_base, split)
        out_csv = os.path.join(out_dir, f"day_{date_str}.csv")
        out_stats = os.path.join(out_dir, f"day_{date_str}_stats.json")
        os.makedirs(out_dir, exist_ok=True)

        print(f"[{split.upper()}] Processing day {day_num:02d}: {filename}")

        try:
            # Load raw data
            df_raw = load_noaa_csv(zip_path, inner_csv_name=inner_csv_name, nrows=nrows)
            print(f"  Loaded: {len(df_raw):,} rows")
            
            # Preprocess with GLOBAL statistics
            frames_df, _ = preprocess_noaa_to_frames(
                df_raw,
                lat_range=lat_range,
                lon_range=lon_range,
                sog_range=sog_range,
                heading_range=heading_range,
                min_vessels_per_timestamp=min_vessels_per_timestamp,
                max_gap_minutes=max_gap_minutes,
                do_zscore=True,
                zscore_stats=global_stats,
            )
            print(f"  Normalized: {len(frames_df):,} frame rows")
            
            # Save frame CSV
            save_frames_csv(frames_df, out_csv)
            print(f"  ✓ Saved: {out_csv}")
            
            # Save per-day stats (same as global for reference)
            with open(out_stats, "w", encoding="utf-8") as f:
                json.dump(global_stats, f, indent=2)
            print(f"  ✓ Saved stats: {out_stats}\n")
            
        except Exception as e:
            print(f"ERROR processing {filename}: {e}\n")
            continue

    print(f"{'='*80}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Location: {dataset_base}/")
    print(f"Global stats: {global_stats_path}")
    print(f"\nTrain files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Test files:  {len(test_files)}")
    print(f"\nNext: python train.py --dataset {dataset_name}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()