# AIS Data Preprocessing - Paper Implementation

## Overview

This preprocessing pipeline implements **exactly** the methodology described in the paper for vessel trajectory prediction.

## Paper Methodology (Section 4.1.1)

The paper defines 5 preprocessing steps for AIS data:

### 1. **Cleaning Abnormal Data**
> "MMSI is not a 9-bit data value; AIS attribute information contains a lot of null data; the LAT range of the trajectory point is set to [30°, 35°], the LON range is set to [−120°, −115°], the SOG range is set to [1.0–22.0], and the heading range is set to [0–360], and the data that is out of range is deleted. Then, we remove the AIS information of vessels not at sea, the timestamps with AIS information from moored/anchored vessels with recorded SOG values less than 1 knot, and the concurrent AIS information of less than three vessels under timestamps."

**Implementation:**
- ✓ MMSI must be exactly 9 digits
- ✓ Remove rows with null values in key fields
- ✓ Filter LAT ∈ [30°, 35°], LON ∈ [−120°, −115°]
- ✓ Filter SOG ∈ [1.0, 22.0] knots
- ✓ Filter Heading ∈ [0°, 360°] (511 = "not available" → NaN)
- ✓ Remove moored/anchored vessels (Status codes 1, 5) with SOG < 1
- ✓ Remove all SOG < 1 (vessels not at sea)
- ✓ Remove timestamps with < 3 concurrent vessels

### 2. **Data Interpolation**
> "For data with a time interval of more than one minute between consecutive trajectory points, the missing value of LON and LAT is supplemented by the linear interpolation method. We eliminate duplicate timestamps and resample the AIS data to a one-minute interval. Since SOG and heading are relatively stable over a short period of time, the average value is used for interpolation instead of SOG and heading values during the time period."

**Implementation:**
- ✓ Group by MMSI (each vessel independently)
- ✓ Remove duplicate timestamps (keep last)
- ✓ Resample to 1-minute intervals
- ✓ **LON/LAT**: Time-based linear interpolation
- ✓ **SOG/Heading**: Rolling average (5-minute window, center=True)
- ✓ Remove remaining NaN values at edges

### 3. **Data Sampling**
> "The raw AIS data is grouped by MMSI number, and the data of the same MMSI number (i.e. the same vessel) is sorted by time. In this work, vessel trajectory data is defined as T = {p₀, p₁, ..., pₙ}, where n is the number of valid samples of each vessel, and p = (x, y, s, h) represents the specific location information of the vessel at a certain time, where x, y, s and h are longitude, latitude, SOG and heading respectively."

**Implementation:**
- ✓ Data grouped by MMSI
- ✓ Sorted chronologically
- ✓ Trajectory T = {p₀, p₁, ..., pₙ}
- ✓ Point p = (x, y, s, h) = (LON, LAT, SOG, Heading)

### 4. **Data Standardization**
> "In order to make the data suitable for the input of deep neural network, every dynamic attribute vector in the whole dataset is normalized by z-score normalization: x̃ = (x − μ(x_dataset)) / σ(x_dataset), where x represents the vessel state attribute value. μ(⋅) denotes the sample mean of each feature attribute and σ(⋅) is the sample standard deviation of each feature attribute. This operation allows different features to have the same importance, even if the scale value of longitude is greater than that of latitude."

**Implementation:**
- ✓ Global statistics computed across entire dataset
- ✓ Z-score formula: x̃ = (x - μ) / σ
- ✓ Applied to all features: LON, LAT, SOG, Heading
- ✓ Statistics saved for denormalization

### 5. **Extracting Input and Target Data by Sliding Window**
> "The sliding window method is used to segment the corrected trajectory sequence data, and for the AIS trajectory data T = {p₀, p₁, ..., pₙ}, since the model needs to learn to predict the target sequence of length h based on the input sequence of length l, we extract all available time windows with length of l + h and slide forward in turn until the last trajectory point is reached."

**Implementation:**
- ✓ Frame-based format for sliding window extraction
- ✓ frame_id: minute index from start time
- ✓ TrajectoryDataset class handles sliding window extraction
- ✓ Observation length l=10, prediction length h=30 (configurable)

## Usage

### Quick Start

```bash
# Run preprocessing on raw AIS data
python preprocess_ais.py
```

This will:
1. Load `data/raw/2020_12/AIS_2020_12_01.zip`
2. Apply all 5 preprocessing steps
3. Save to `dataset/noaa_dec2020/train/day_2020_12_01.csv`
4. Save normalization stats to `day_2020_12_01_stats.json`

### Configuration

Edit `preprocess_ais.py` to customize:

```python
# Geographic range (San Diego Harbor area)
lat_range = (30.0, 35.0)
lon_range = (-120.0, -115.0)

# Dynamic ranges
sog_range = (1.0, 22.0)         # Speed Over Ground (knots)
heading_range = (0.0, 360.0)     # Heading (degrees)

# Minimum concurrent vessels per timestamp
min_vessels_per_timestamp = 3

# Enable/disable z-score normalization
do_zscore = True
```

### Output Format

**CSV Structure:**
```
frame_id,vessel_id,LON,LAT,SOG,Heading
0,241326000,-1.7206,-1.3999,-1.2901,-0.7779
0,367369720,-0.3153,0.6538,0.6723,-0.9861
1,367363650,1.1363,-0.5333,-0.6621,0.5705
...
```

**Normalization Stats (JSON):**
```json
{
  "LON": {"mean": -118.0662, "std": 0.7645},
  "LAT": {"mean": 33.1597, "std": 0.8266},
  "SOG": {"mean": 8.6742, "std": 5.0956},
  "Heading": {"mean": 209.4603, "std": 100.8540}
}
```

### Example Output

```
======================================================================
AIS DATA PREPROCESSING PIPELINE (Paper-Aligned)
======================================================================
Input data: 6,034,864 rows, 14744 vessels

[Step 1/5] Cleaning abnormal data...
  Initial rows: 6,034,864
  Final cleaned rows: 40,999 (5,993,865 total removed, 99.3% reduction)

[Step 2/5] Data interpolation and resampling...
  Output rows after resampling: 38,771 (-2,228 rows)
  Vessels after resampling: 68

[Step 3/5] Filtering timestamps by concurrent vessels...
  Timestamps kept: 1,439 (1 removed)
  Rows after filtering: 38,770

[Step 4/5] Data standardization (z-score normalization)...
  LON: μ=-118.0662, σ=0.7645
  LAT: μ=33.1597, σ=0.8266
  SOG: μ=8.6742, σ=5.0956
  Heading: μ=209.4603, σ=100.8540

[Step 5/5] Converting to frame format...
  Total frames (minutes): 1439
  Total vessels: 68
  Total data points: 38,770
======================================================================
```

## Integration with TrajectoryDataset

The preprocessed CSV is ready for use with `TrajectoryDataset`:

```python
from utils import TrajectoryDataset

# Load preprocessed data
train_dataset = TrajectoryDataset(
    data_dir="dataset/noaa_dec2020/train",
    obs_len=10,      # Observation sequence length (10 minutes)
    pred_len=30,     # Prediction sequence length (30 minutes)
    skip=1,          # Sliding window stride
    threshold=0.002, # Non-linearity threshold
    min_ped=1        # Minimum vessels per sequence
)
```

The dataset will:
1. Load frame-based CSV files
2. Apply sliding window extraction (obs_len + pred_len)
3. Create graph-based representations for the model
4. Generate (observation, prediction) pairs for training

## Denormalization

To convert predictions back to original scale:

```python
import json
import numpy as np

# Load normalization stats
with open("dataset/noaa_dec2020/train/day_2020_12_01_stats.json") as f:
    stats = json.load(f)

# Denormalize predictions
def denormalize(normalized_values, feature="LON"):
    mean = stats[feature]["mean"]
    std = stats[feature]["std"]
    return normalized_values * std + mean

# Example: denormalize LON predictions
lon_pred_normalized = model_output[:, :, 0]  # Assuming LON is first feature
lon_pred_original = denormalize(lon_pred_normalized, "LON")
```

## Data Statistics

Typical preprocessing results for 1 day of AIS data (San Diego Harbor):

- **Input**: ~6M raw AIS records
- **After cleaning**: ~41K records (99.3% reduction)
- **After resampling**: ~39K records at 1-minute intervals
- **Final output**: ~39K normalized data points
- **Vessels**: ~68 vessels (from ~15K total)
- **Time coverage**: 1,439 minutes (~24 hours)
- **Vessels per frame**: 4-43 (mean ~27)

## Notes

- **Heading value 511** indicates "not available" and is handled as missing data
- **Status codes**: 1=at anchor, 5=moored (removed if SOG < 1)
- **Rolling window** for SOG/Heading uses 5-minute window (center=True)
- **Z-score normalization** uses global statistics across entire dataset
- **Frame format** enables efficient sliding window extraction by TrajectoryDataset

## Paper Reference

This implementation follows the methodology described in Section 4.1.1 of the paper on vessel trajectory prediction using spatial-temporal network models.
