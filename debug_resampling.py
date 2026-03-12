"""
Debug script: Analyze why vessels are lost during resampling.
Shows trajectory length distribution and resampling success rate.
"""

import pandas as pd
import numpy as np
from utils import load_noaa_csv, clean_abnormal_data_noaa

# Load one day of data
zip_path = "data/raw/2021_12/AIS_2021_12_01.zip"

print("Loading and cleaning data...")
df_raw = load_noaa_csv(zip_path, nrows=None)  # Full day
df = clean_abnormal_data_noaa(
    df_raw,
    lat_range=(30.0, 35.0),
    lon_range=(-120.0, -115.0),
    sog_range=(1.0, 22.0),
    heading_range=(0.0, 360.0),
)

print(f"\n{'='*60}")
print(f"VESSEL TRAJECTORY LENGTH ANALYSIS (BEFORE RESAMPLING)")
print(f"{'='*60}")

# Analyze points per vessel
vessel_counts = df.groupby("MMSI").size().sort_values(ascending=False)

print(f"\nTotal vessels: {len(vessel_counts)}")
print(f"Total points: {len(df):,}")
print(f"\nTrajectory length distribution:")
print(f"  Min:    {vessel_counts.min()} points")
print(f"  Max:    {vessel_counts.max()} points")
print(f"  Mean:   {vessel_counts.mean():.1f} points")
print(f"  Median: {vessel_counts.median():.0f} points")

print(f"\nPercentiles:")
for pct in [10, 25, 50, 75, 90, 95, 99]:
    val = vessel_counts.quantile(pct/100)
    print(f"  {pct:2d}%: {val:.0f} points")

print(f"\nVessels by trajectory length:")
bins = [0, 5, 10, 20, 50, 100, 200, 500, 1000, 10000]
for i in range(len(bins)-1):
    count = ((vessel_counts > bins[i]) & (vessel_counts <= bins[i+1])).sum()
    pct = 100 * count / len(vessel_counts)
    print(f"  {bins[i]+1:4d} - {bins[i+1]:4d} points: {count:4d} vessels ({pct:5.1f}%)")

# Now simulate resampling for each vessel
print(f"\n{'='*60}")
print(f"RESAMPLING SIMULATION")
print(f"{'='*60}")

df_sorted = df.copy().sort_values(["MMSI", "BaseDateTime"])

successful_vessels = []
failed_vessels = []

for mmsi, g in df_sorted.groupby("MMSI", sort=False):
    # Remove duplicates
    g = g.drop_duplicates(subset=["BaseDateTime"], keep="last").set_index("BaseDateTime")
    
    # Resample
    r = g.resample("1min").asfreq()
    
    # Try to interpolate
    r["LON"] = r["LON"].interpolate(method="time")
    r["LAT"] = r["LAT"].interpolate(method="time")
    
    # SOG/Heading with rolling mean
    for c in ["SOG", "Heading"]:
        s = r[c]
        s = s.fillna(s.rolling(window=5, min_periods=1, center=True).mean())
        r[c] = s.ffill().bfill()
    
    # Check if vessel survives (no NaN in critical fields)
    before_dropna = len(r)
    r_clean = r.dropna(subset=["LON", "LAT", "SOG", "Heading"])
    after_dropna = len(r_clean)
    
    if after_dropna > 0:
        successful_vessels.append({
            "MMSI": mmsi,
            "original_points": len(g),
            "resampled_points": before_dropna,
            "final_points": after_dropna,
            "survival_rate": after_dropna / before_dropna
        })
    else:
        failed_vessels.append({
            "MMSI": mmsi,
            "original_points": len(g),
            "resampled_points": before_dropna,
        })

print(f"\nResampling results:")
print(f"  Successful vessels: {len(successful_vessels)} ({100*len(successful_vessels)/len(vessel_counts):.1f}%)")
print(f"  Failed vessels:     {len(failed_vessels)} ({100*len(failed_vessels)/len(vessel_counts):.1f}%)")

if failed_vessels:
    failed_df = pd.DataFrame(failed_vessels)
    print(f"\nFailed vessels statistics:")
    print(f"  Original points - Min:    {failed_df['original_points'].min()}")
    print(f"  Original points - Max:    {failed_df['original_points'].max()}")
    print(f"  Original points - Mean:   {failed_df['original_points'].mean():.1f}")
    print(f"  Original points - Median: {failed_df['original_points'].median():.0f}")

if successful_vessels:
    success_df = pd.DataFrame(successful_vessels)
    print(f"\nSuccessful vessels statistics:")
    print(f"  Original points - Min:    {success_df['original_points'].min()}")
    print(f"  Original points - Max:    {success_df['original_points'].max()}")
    print(f"  Original points - Mean:   {success_df['original_points'].mean():.1f}")
    print(f"  Original points - Median: {success_df['original_points'].median():.0f}")
    print(f"\n  Final points - Min:       {success_df['final_points'].min()}")
    print(f"  Final points - Max:       {success_df['final_points'].max()}")
    print(f"  Final points - Mean:      {success_df['final_points'].mean():.1f}")

print(f"\n{'='*60}")
print(f"RECOMMENDATION:")
print(f"{'='*60}")

if failed_vessels and len(failed_vessels) > 0:
    failed_df = pd.DataFrame(failed_vessels)
    threshold = failed_df['original_points'].quantile(0.90)
    print(f"Most failed vessels have < {threshold:.0f} original points.")
    print(f"Consider adding a minimum trajectory length filter BEFORE resampling:")
    print(f"  df = df.groupby('MMSI').filter(lambda x: len(x) >= {max(10, int(threshold))})")
    print(f"\nThis will:")
    print(f"  - Remove vessels with very sparse data (can't interpolate)")
    print(f"  - Keep only vessels with continuous trajectories")
    print(f"  - Match paper's focus on vessels 'around San Diego Harbor'")
