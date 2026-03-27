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
    # DualSTMA (Huang et al., 2024) — DOI: 10.3390/jmse12112031
    # Dataset: MarineCadastre.gov, AIS data 2021
    # Preprocessed dataset also available at:
    #   https://github.com/AIR-SkyForecast/METO-S2S/tree/main/dataset_json
    #
    # 4 US coastal regions (process each separately, then combine):
    #   Southwestern: LON 120°W–114°W, LAT 28°N–35°N
    #   Northeastern: LON 71°W–65°W,  LAT 41°N–46°N
    #   Southeastern: LON 81°W–75°W,  LAT 30°N–36°N
    #   Northwestern: LON 127°W–122°W, LAT 42°N–50°N
    #
    # Run this script once per region, changing the params below.
    # Then train on the combined dataset.
    #
    # NOTE: If using the preprocessed GitHub dataset, skip preprocessing
    # and go directly to training with obs_len=10, pred_len=5.

    # Select region (change as needed):
    REGION = "southwestern"  # Options: southwestern, northeastern, southeastern, northwestern

    REGION_PARAMS = {
        "southwestern": {"lon": (-120.0, -114.0), "lat": (28.0, 35.0)},
        "northeastern": {"lon": (-71.0,  -65.0),  "lat": (41.0, 46.0)},
        "southeastern": {"lon": (-81.0,  -75.0),  "lat": (30.0, 36.0)},
        "northwestern": {"lon": (-127.0, -122.0), "lat": (42.0, 50.0)},
    }

    raw_data_folder = f"data/raw/marinecadastre_2021_{REGION}"
    inner_csv_name = None
    nrows = None  # Use None for full processing, or set limit for testing

    # ----------------------------
    # Train/Val/Test Split (DualSTMA: 8:1:1 chronological)
    # ----------------------------
    # 2021 data — use full year or subset of months
    # Approximate 8:1:1 over available days:
    # If using monthly files (e.g., Jan–Dec 2021):
    #   Train: months 1–9  (Jan–Sep)
    #   Val:   month  10   (Oct)
    #   Test:  months 11–12 (Nov–Dec)
    # If using daily files within a month, adjust accordingly.
    # Here we use day-based split within available files (8:1:1):
    #   Train: days 1–25  (~80%)
    #   Val:   days 26–28 (~10%)
    #   Test:  days 29–31 (~10%)
    train_days = list(range(1, 26))    # days 1–25  → train (~80%)
    val_days   = list(range(26, 29))   # days 26–28 → val   (~10%)
    test_days  = list(range(29, 32))   # days 29–31 → test  (~10%)

    # ----------------------------
    # Paper preprocessing params (DualSTMA, Section 4.1.2)
    # ----------------------------
    lat_range     = REGION_PARAMS[REGION]["lat"]
    lon_range     = REGION_PARAMS[REGION]["lon"]
    sog_range     = (0.5, 40.0)        # Paper removes stationary vessels (SOG~0)
    heading_range = (0.0, 360.0)
    min_vessels_per_timestamp = 2      # Multi-vessel context needed
    max_gap_minutes = 10               # Gaps > 10 min → new segment before resampling
    resample_freq  = "10min"           # KEY: DualSTMA uses 10-min interval ← changed from 1min

    # ----------------------------
    # Output
    # ----------------------------
    dataset_name = "marinecadastre_2021"
    dataset_base = os.path.join("dataset", dataset_name)
    global_stats_path = os.path.join(dataset_base, "global_stats.json")

    # Find all AIS zip files
    zip_files = sorted(glob.glob(os.path.join(raw_data_folder, "AIS_*.zip")))

    if not zip_files:
        print(f"ERROR: No AIS_*.zip files found in {raw_data_folder}")
        print(f"Download from: https://marinecadastre.gov/ais/")
        print(f"Or use preprocessed data from: https://github.com/AIR-SkyForecast/METO-S2S/tree/main/dataset_json")
        return

    # Categorize files by split
    train_files = []
    val_files   = []
    test_files  = []

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
            df = resample_interpolate_1min(df, freq=resample_freq, rolling_window=5, max_gap_minutes=max_gap_minutes)
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
                resample_freq=resample_freq,
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
    print(f"Dataset:      {dataset_name}")
    print(f"Region:       {REGION} {REGION_PARAMS[REGION]}")
    print(f"Sampling:     {resample_freq} (DualSTMA-aligned)")
    print(f"Location:     {dataset_base}/")
    print(f"Global stats: {global_stats_path}")
    print(f"\nTrain files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Test files:  {len(test_files)}")
    print(f"\nNext steps (DualSTMA comparison):")
    print(f"  # obs_len=10 × 10min = 100min observation")
    print(f"  # pred_len=5  × 10min = 50min prediction (matches DualSTMA)")
    print(f"  python train.py --dataset {dataset_name} --tag SMCHN_dualstma \\")
    print(f"    --obs_len 10 --pred_len 5 --num_epochs 200 --lr 0.00001 --clip_grad 1.0")
    print(f"  python evaluate.py --dataset {dataset_name} \\")
    print(f"    --checkpoint checkpoints/SMCHN_dualstma/{dataset_name}/val_best.pth \\")
    print(f"    --split test --num_samples 20")
    print(f"\nComparison target: DualSTMA (Huang et al., 2024)")
    print(f"  DOI: 10.3390/jmse12112031")
    print(f"  ADE 10min: 0.000609°  FDE 10min: 0.000807°")
    print(f"  ADE 50min: 0.002436°  FDE 50min: 0.003946°")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()