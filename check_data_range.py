"""
Quick diagnostic to check actual LAT/LON range in raw NOAA data.
This will help identify why 99% of vessels are filtered out.
"""

import pandas as pd
from utils import load_noaa_csv

# Load one day of raw data
zip_path = "data/raw/2021_12/AIS_2021_12_01.zip"

print("Loading raw data...")
df = load_noaa_csv(zip_path, nrows=100_000)  # Sample first 100K rows

print(f"\nRAW DATA STATISTICS:")
print(f"Total rows: {len(df):,}")
print(f"Unique vessels (MMSI): {df['MMSI'].nunique():,}")

# Check LAT/LON distribution
print(f"\n{'='*60}")
print(f"LATITUDE (LAT) DISTRIBUTION:")
print(f"{'='*60}")
print(f"Min:    {df['LAT'].min():.4f}°")
print(f"Max:    {df['LAT'].max():.4f}°")
print(f"Mean:   {df['LAT'].mean():.4f}°")
print(f"Median: {df['LAT'].median():.4f}°")

print(f"\n{'='*60}")
print(f"LONGITUDE (LON) DISTRIBUTION:")
print(f"{'='*60}")
print(f"Min:    {df['LON'].min():.4f}°")
print(f"Max:    {df['LON'].max():.4f}°")
print(f"Mean:   {df['LON'].mean():.4f}°")
print(f"Median: {df['LON'].median():.4f}°")

# Check how many would pass the paper's geographic filter
paper_lat_range = (30.0, 35.0)
paper_lon_range = (-120.0, -115.0)

mask = (
    df['LAT'].between(paper_lat_range[0], paper_lat_range[1]) &
    df['LON'].between(paper_lon_range[0], paper_lon_range[1])
)

print(f"\n{'='*60}")
print(f"PAPER'S GEOGRAPHIC FILTER (San Diego area):")
print(f"{'='*60}")
print(f"LAT range: [{paper_lat_range[0]}, {paper_lat_range[1]}]")
print(f"LON range: [{paper_lon_range[0]}, {paper_lon_range[1]}]")
print(f"\nRows passing filter: {mask.sum():,} / {len(df):,} ({100*mask.sum()/len(df):.2f}%)")
print(f"Vessels in region: {df[mask]['MMSI'].nunique():,}")

# Show percentiles to understand data spread
print(f"\n{'='*60}")
print(f"LAT/LON PERCENTILES:")
print(f"{'='*60}")
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    lat_val = df['LAT'].quantile(pct/100)
    lon_val = df['LON'].quantile(pct/100)
    print(f"  {pct:2d}%: LAT={lat_val:7.2f}°, LON={lon_val:7.2f}°")

print(f"\n{'='*60}")
print(f"RECOMMENDATION:")
print(f"{'='*60}")
print("Update the geographic ranges in preprocess_ais.py to match your data:")
print(f"  lat_range = ({df['LAT'].quantile(0.01):.1f}, {df['LAT'].quantile(0.99):.1f})")
print(f"  lon_range = ({df['LON'].quantile(0.01):.1f}, {df['LON'].quantile(0.99):.1f})")
print("\nOr use a tighter filter (central 50% of data):")
print(f"  lat_range = ({df['LAT'].quantile(0.25):.1f}, {df['LAT'].quantile(0.75):.1f})")
print(f"  lon_range = ({df['LON'].quantile(0.25):.1f}, {df['LON'].quantile(0.75):.1f})")
