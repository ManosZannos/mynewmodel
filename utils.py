"""
utils.py

Includes:
1) NOAA AIS preprocessing (paper-aligned) -> frame-format CSV compatible with TrajectoryDataset:
   frame_id, vessel_id, LON, LAT, SOG, Heading

2) Original utility functions + TrajectoryDataset loader (cleaned, no non-English characters).
"""

import os
import math
import zipfile

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from tqdm import tqdm


# ============================================================================
# AIS / NOAA PREPROCESSING (paper-aligned)
# Output: frame_id, vessel_id, LON, LAT, SOG, Heading
# ============================================================================

NOAA_REQUIRED_COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "Heading", "Status"]


def _valid_mmsi_9digits(series: pd.Series) -> pd.Series:
    s = series.astype("Int64").astype(str)
    return s.str.fullmatch(r"\d{9}")


def load_noaa_csv(csv_or_zip_path: str, inner_csv_name: str | None = None, nrows: int | None = None) -> pd.DataFrame:
    """
    Load NOAA AIS daily CSV from either:
      - a .csv path, or
      - a .zip path containing a csv (optionally specify inner_csv_name)
    """
    if csv_or_zip_path.lower().endswith(".zip"):
        with zipfile.ZipFile(csv_or_zip_path) as z:
            if inner_csv_name is None:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not names:
                    raise ValueError("Zip does not contain a .csv file.")
                inner_csv_name = names[0]
            with z.open(inner_csv_name) as f:
                return pd.read_csv(f, nrows=nrows)
    return pd.read_csv(csv_or_zip_path, nrows=nrows)


def clean_abnormal_data_noaa(
    df: pd.DataFrame,
    lat_range=(30.0, 35.0),
    lon_range=(-120.0, -115.0),
    sog_range=(1.0, 22.0),
    heading_range=(0.0, 360.0),
) -> pd.DataFrame:
    """
    Paper Step 1: Cleaning abnormal data
    
    Quote from paper:
    "MMSI is not a 9-bit data value; AIS attribute information contains a lot of null data;
    the LAT range of the trajectory point is set to [30°, 35°], the LON range is set to
    [−120°, −115°], the SOG range is set to [1.0–22.0], and the heading range is set to [0–360],
    and the data that is out of range is deleted. Then, we remove the AIS information of vessels
    not at sea, the timestamps with AIS information from moored/anchored vessels with recorded
    SOG values less than 1 knot."
    
    Implementation:
      - MMSI must be 9 digits
      - Drop nulls in key fields (MMSI, BaseDateTime, LAT, LON, SOG, Heading, Status)
      - Parse BaseDateTime as UTC timestamp
      - Filter by geographic ranges: LAT [30°, 35°], LON [-120°, -115°]
      - Filter by SOG range: [1.0, 22.0] knots
      - Filter by Heading range: [0°, 360°] (511 = "not available", treated as missing)
      - Remove moored/anchored vessels (Status codes 1=at anchor, 5=moored) with SOG < 1
      - Remove all vessels with SOG < 1 ("not at sea")
    """
    missing = [c for c in NOAA_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    initial_count = len(df)
    print(f"\n[Step 1/5] Cleaning abnormal data...")
    print(f"  Initial rows: {initial_count:,}")

    df = df.copy()

    # Parse timestamps
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], errors="coerce", utc=True)
    df = df.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "Heading", "Status"])
    print(f"  After removing nulls: {len(df):,} rows ({initial_count - len(df):,} removed)")

    # Validate 9-digit MMSI
    before = len(df)
    df = df[_valid_mmsi_9digits(df["MMSI"])]
    print(f"  After MMSI validation (9-digit): {len(df):,} rows ({before - len(df):,} removed)")

    # Heading 511 = "not available"
    df.loc[df["Heading"] == 511, "Heading"] = np.nan

    # Filter by geographic and dynamic ranges
    before = len(df)
    df = df[
        df["LAT"].between(lat_range[0], lat_range[1]) &
        df["LON"].between(lon_range[0], lon_range[1]) &
        df["SOG"].between(sog_range[0], sog_range[1])
    ]
    print(f"  After range filtering (LAT/LON/SOG): {len(df):,} rows ({before - len(df):,} removed)")

    # Filter heading (allow NaN from 511 values)
    before = len(df)
    df = df[(df["Heading"].isna()) | (df["Heading"].between(heading_range[0], heading_range[1]))]
    print(f"  After Heading filtering: {len(df):,} rows ({before - len(df):,} removed)")

    # Remove moored/anchored vessels with SOG < 1
    before = len(df)
    anchored_or_moored = df["Status"].isin([1, 5])  # 1=at anchor, 5=moored
    df = df[~(anchored_or_moored & (df["SOG"] < 1.0))]
    print(f"  After removing moored/anchored (SOG<1): {len(df):,} rows ({before - len(df):,} removed)")

    # Remove all vessels not at sea (SOG < 1)
    before = len(df)
    df = df[df["SOG"] >= 1.0]
    print(f"  After removing SOG < 1.0: {len(df):,} rows ({before - len(df):,} removed)")
    print(f"  Final cleaned rows: {len(df):,} ({initial_count - len(df):,} total removed, {100*(1-len(df)/initial_count):.1f}% reduction)")

    return df


def resample_interpolate_1min(df: pd.DataFrame, freq: str = "1min", rolling_window: int = 5) -> pd.DataFrame:
    """
    Paper Step 2: Data interpolation
    
    Quote from paper:
    "For data with a time interval of more than one minute between consecutive trajectory points,
    the missing value of LON and LAT is supplemented by the linear interpolation method. We
    eliminate duplicate timestamps and resample the AIS data to a one-minute interval. Since SOG
    and heading are relatively stable over a short period of time, the average value is used for
    interpolation instead of SOG and heading values during the time period."
    
    Implementation:
      - Group by MMSI (each vessel independently)
      - Drop duplicate timestamps (keep last)
      - Resample to 1-minute intervals
      - LON/LAT: time-based linear interpolation
      - SOG/Heading: rolling average (window=5min, center=True) then forward/backward fill
      - Remove any remaining NaN values (edges that couldn't be interpolated)
    """
    initial_count = len(df)
    print(f"\n[Step 2/5] Data interpolation and resampling...")
    print(f"  Input rows: {initial_count:,}")
    
    df = df.copy().sort_values(["MMSI", "BaseDateTime"])
    n_vessels = df["MMSI"].nunique()
    print(f"  Vessels to process: {n_vessels}")

    out_parts = []
    for mmsi, g in df.groupby("MMSI", sort=False):
        # Remove duplicate timestamps
        g = g.drop_duplicates(subset=["BaseDateTime"], keep="last").set_index("BaseDateTime")

        # Resample to 1-minute frequency
        r = g.resample(freq).asfreq()

        # Linear interpolation for LON and LAT (position)
        r["LON"] = r["LON"].interpolate(method="time")
        r["LAT"] = r["LAT"].interpolate(method="time")

        # SOG and Heading: use average value over time period (rolling mean)
        # Paper: "Since SOG and heading are relatively stable over a short period of time,
        #         the average value is used for interpolation"
        for c in ["SOG", "Heading"]:
            s = r[c]
            # Rolling mean with center=True to get average of surrounding values
            s = s.fillna(s.rolling(window=rolling_window, min_periods=1, center=True).mean())
            # Fill any remaining NaN (at edges)
            r[c] = s.ffill().bfill()

        # Forward/backward fill Status
        r["Status"] = r["Status"].ffill().bfill()

        # Remove any remaining missing values (edges after resample)
        r = r.dropna(subset=["LON", "LAT", "SOG", "Heading"])

        r["MMSI"] = mmsi
        out_parts.append(r.reset_index())

    result = pd.concat(out_parts, ignore_index=True) if out_parts else df.iloc[0:0].copy()
    print(f"  Output rows after resampling: {len(result):,} ({len(result) - initial_count:+,} rows)")
    print(f"  Vessels after resampling: {result['MMSI'].nunique()}")
    
    return result


def filter_timestamps_min_vessels(df: pd.DataFrame, min_vessels_per_timestamp: int = 3) -> pd.DataFrame:
    """
    Paper Step 1 (final part): Filter timestamps by concurrent vessels
    
    Quote from paper:
    "...and the concurrent AIS information of less than three vessels under timestamps."
    
    Applied AFTER resampling to ensure we have timestamps where multiple vessels
    are spatially interacting (which is the focus of the model).
    
    Keeps only timestamps where >= min_vessels distinct vessels exist.
    """
    initial_count = len(df)
    initial_timestamps = df["BaseDateTime"].nunique()
    
    print(f"\n[Step 3/5] Filtering timestamps by concurrent vessels...")
    print(f"  Initial timestamps: {initial_timestamps:,}")
    print(f"  Minimum vessels per timestamp: {min_vessels_per_timestamp}")
    
    df = df.copy()
    counts = df.groupby("BaseDateTime")["MMSI"].nunique()
    keep_times = counts[counts >= min_vessels_per_timestamp].index
    df = df[df["BaseDateTime"].isin(keep_times)]
    
    print(f"  Timestamps kept: {len(keep_times):,} ({initial_timestamps - len(keep_times):,} removed)")
    print(f"  Rows after filtering: {len(df):,} ({initial_count - len(df):,} removed)")
    print(f"  Vessels remaining: {df['MMSI'].nunique()}")
    
    return df


def zscore_normalize_global(df: pd.DataFrame, cols=("LON", "LAT", "SOG", "Heading"), stats: dict | None = None):
    """
    Paper Step 4: Data standardization
    
    Quote from paper:
    "In order to make the data suitable for the input of deep neural network, every dynamic
    attribute vector in the whole dataset is normalized by z-score normalization."
    
    Formula: x̃ = (x - μ(x_dataset)) / σ(x_dataset)
    
    where:
      - x: vessel state attribute value
      - μ(·): sample mean of each feature attribute
      - σ(·): sample standard deviation of each feature attribute
      - x̃: normalized vessel state feature vector
    
    "This operation allows different features to have the same importance, even if the
    scale value of longitude is greater than that of latitude."
    
    Implementation:
      - Compute global mean and std for each feature across entire dataset
      - Apply z-score normalization to all values
      - Return normalized data and statistics (for denormalization later)
    """
    print(f"\n[Step 4/5] Data standardization (z-score normalization)...")
    
    df = df.copy()
    if stats is None:
        stats = {}
        print(f"  Computing global statistics for {len(cols)} features:")
        for c in cols:
            mean = float(df[c].mean())
            std = float(df[c].std(ddof=1))
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            stats[c] = {"mean": mean, "std": std}
            print(f"    {c}: μ={mean:.4f}, σ={std:.4f}")
    else:
        print(f"  Using provided statistics for normalization")

    # Apply z-score normalization
    for c in cols:
        df[c] = (df[c] - stats[c]["mean"]) / stats[c]["std"]
    
    print(f"  Normalization complete. All features now have μ≈0, σ≈1")

    return df, stats


def to_frame_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paper Step 5: Convert to frame format for sliding window extraction
    
    Quote from paper:
    "The raw AIS data is grouped by MMSI number, and the data of the same MMSI number
    (i.e. the same vessel) is sorted by time. In this work, vessel trajectory data is
    defined as T = {p0, p1, ..., pn}, where n is the number of valid samples of each vessel,
    and p = (x, y, s, h) represents the specific location information of the vessel at a
    certain time, where x, y, s and h are longitude, latitude, SOG and heading respectively."
    
    Output format for TrajectoryDataset:
      - frame_id: minute index from minimum timestamp (0, 1, 2, ...)
      - vessel_id: MMSI (unique vessel identifier)
      - LON, LAT, SOG, Heading: normalized trajectory features (x, y, s, h)
    
    This format allows TrajectoryDataset to apply sliding windows and extract
    observation/prediction sequences for training.
    """
    print(f"\n[Step 5/5] Converting to frame format...")
    
    df = df.copy().sort_values("BaseDateTime")
    t0 = df["BaseDateTime"].min()
    t_end = df["BaseDateTime"].max()
    
    # Create frame_id as minute index from t0
    df["frame_id"] = ((df["BaseDateTime"] - t0).dt.total_seconds() / 60.0).round().astype(int)
    df = df.rename(columns={"MMSI": "vessel_id"})
    
    result = df[["frame_id", "vessel_id", "LON", "LAT", "SOG", "Heading"]]
    
    print(f"  Time range: {t0} to {t_end}")
    print(f"  Total frames (minutes): {result['frame_id'].max() + 1}")
    print(f"  Total vessels: {result['vessel_id'].nunique()}")
    print(f"  Total data points: {len(result):,}")
    print(f"  Output columns: {list(result.columns)}")
    
    return result


def preprocess_noaa_to_frames(
    df_raw: pd.DataFrame,
    lat_range=(30.0, 35.0),
    lon_range=(-120.0, -115.0),
    sog_range=(1.0, 22.0),
    heading_range=(0.0, 360.0),
    min_vessels_per_timestamp: int = 3,
    do_zscore: bool = True,
):
    """
    Complete AIS preprocessing pipeline (paper-aligned)
    
    Implements all 5 steps from the paper:
    
    1. Cleaning abnormal data
       - MMSI validation (9 digits)
       - Remove nulls
       - Filter by ranges: LAT [30°,35°], LON [-120°,-115°], SOG [1.0,22.0], Heading [0°,360°]
       - Remove moored/anchored vessels (SOG < 1)
       - Remove timestamps with < 3 concurrent vessels
    
    2. Data interpolation
       - Resample to 1-minute intervals
       - Linear interpolation for LON/LAT
       - Rolling average for SOG/Heading
       - Remove duplicate timestamps
    
    3. Data sampling
       - Group by MMSI, sort by time
       - Define trajectory T = {p0, p1, ..., pn} where p = (x, y, s, h)
    
    4. Data standardization
       - Z-score normalization: x̃ = (x - μ) / σ
       - Global statistics across entire dataset
    
    5. Extract frame format
       - Output: frame_id, vessel_id, LON, LAT, SOG, Heading
       - Ready for TrajectoryDataset sliding window extraction
    
    Returns:
      - frames_df: DataFrame in frame format
      - stats: Dictionary with normalization statistics {feature: {mean, std}}
    """
    print("="*70)
    print("AIS DATA PREPROCESSING PIPELINE (Paper-Aligned)")
    print("="*70)
    print(f"Input data: {len(df_raw):,} rows, {df_raw['MMSI'].nunique()} vessels")
    
    # Step 1: Clean abnormal data
    df = clean_abnormal_data_noaa(
        df_raw,
        lat_range=lat_range,
        lon_range=lon_range,
        sog_range=sog_range,
        heading_range=heading_range,
    )
    
    # Step 2: Data interpolation and resampling
    df = resample_interpolate_1min(df, freq="1min", rolling_window=5)
    
    # Step 3: Filter timestamps with < min_vessels concurrent vessels
    df = filter_timestamps_min_vessels(df, min_vessels_per_timestamp=min_vessels_per_timestamp)

    # Step 4: Data standardization (z-score normalization)
    stats = None
    if do_zscore:
        df, stats = zscore_normalize_global(df, cols=("LON", "LAT", "SOG", "Heading"))

    # Step 5: Convert to frame format
    frames = to_frame_format(df)
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Output: {len(frames):,} data points across {frames['frame_id'].max()+1} frames")
    print(f"Vessels: {frames['vessel_id'].nunique()}")
    print(f"Ready for TrajectoryDataset with obs_len + pred_len sliding windows")
    print("="*70 + "\n")
    
    return frames, stats


def save_frames_csv(frames_df: pd.DataFrame, out_csv_path: str):
    """
    Save frame-format CSV for TrajectoryDataset.
    """
    out_dir = os.path.dirname(out_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    frames_df.to_csv(out_csv_path, index=False, header=True)


# ============================================================================
# ORIGINAL UTILITY FUNCTIONS
# ============================================================================

def anorm(p1, p2):
    norm = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if norm == 0:
        return 0
    return 1 / norm


def loc_pos(seq_):
    """
    Adds a simple positional index (1..seq_len) as an extra feature.
    Input:  seq_ shape (seq_len, N, F)
    Output: shape (seq_len, N, F+1)
    """
    seq_len = seq_.shape[0]
    num_nodes = seq_.shape[1]

    pos_seq = np.arange(1, seq_len + 1)[:, np.newaxis, np.newaxis]
    pos_seq = pos_seq.repeat(num_nodes, axis=1)
    return np.concatenate((pos_seq, seq_), axis=-1)


def seq_to_graph(seq_, seq_rel, pos_enc=False):
    """
    Builds node feature tensor V of shape (seq_len, N, 4) from absolute features.
    Paper-faithful: Uses absolute state (LON, LAT, SOG, Heading), not relative.
    If pos_enc=True, prepends a positional index feature.
    
    Input: seq_ shape (N, 4, seq_len) where N = number of vessels
    Output: V shape (seq_len, N, 4)
    """
    # Safety checks: ensure correct input shape
    assert seq_.dim() == 3, f"Expected seq_ (N, 4, T), got {seq_.shape}"
    assert seq_rel.dim() == 3, f"Expected seq_rel (N, 4, T), got {seq_rel.shape}"
    assert seq_.shape[1] == 4, f"Expected 4 features, got {seq_.shape[1]}"
    
    # Optimized: use permute instead of nested loops
    # Converts (N, 4, T) -> (T, N, 4) via axis permutation
    V = seq_.permute(2, 0, 1).contiguous()  # (seq_len, N, 4)

    if pos_enc:
        # loc_pos expects numpy array
        V_np = V.cpu().numpy()
        V_np = loc_pos(V_np)
        return torch.from_numpy(V_np).float().to(seq_.device)

    return V.float()


def poly_fit(traj, traj_len, threshold):
    """
    Determines whether a trajectory is non-linear using a 2nd-order polynomial fit.

    NOTE: If traj has more than 2 channels (e.g., LON, LAT, SOG, Heading),
    this function uses only the first two channels (LON, LAT).

    Input:
      - traj: numpy array shape (C, traj_len)
      - traj_len: length used for fitting
      - threshold: error threshold for non-linearity
    Output:
      - 1.0 if non-linear else 0.0
    """
    traj2 = traj[:2, :]
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj2[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj2[1, -traj_len:], 2, full=True)[1]
    return 1.0 if (res_x + res_y >= threshold) else 0.0


class TrajectoryDataset(Dataset):
    """Dataloader for trajectory datasets in frame format:
    frame_id, vessel_id, f1, f2, f3, f4
    This project uses 4 features (e.g., LON, LAT, SOG, Heading).
    """

    def __init__(self, data_dir, obs_len=10, pred_len=30, skip=1, threshold=0.002,
                 min_ped=1, delim="\t"):
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = obs_len + pred_len
        self.delim = delim

        # Load only CSV files from data_dir
        all_files = [
            os.path.join(self.data_dir, f) 
            for f in os.listdir(self.data_dir) 
            if f.lower().endswith('.csv')
        ]
        
        if not all_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []

        for path in all_files:
            data = pd.read_csv(path)
            data = np.asarray(data)[:, :6]  # frame_id, vessel_id, 4 features
            print("data_vessel:", data.shape)

            frames = np.unique(data[:, 0]).tolist()
            frame_to_idx = {frame: i for i, frame in enumerate(frames)}

            frame_data = [data[data[:, 0] == frame, :] for frame in frames]

            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))

                curr_seq = np.zeros((len(peds_in_curr_seq), 4, self.seq_len), dtype=np.float32)
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 4, self.seq_len), dtype=np.float32)
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len), dtype=np.float32)

                num_peds_considered = 0
                _non_linear_ped = []

                for ped_id in peds_in_curr_seq:
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                    pad_front = frame_to_idx[curr_ped_seq[0, 0]] - idx
                    pad_end = frame_to_idx[curr_ped_seq[-1, 0]] - idx + 1

                    if pad_end - pad_front != self.seq_len:
                        continue
                    if curr_ped_seq.shape[0] != self.seq_len:
                        continue

                    feat_seq = np.transpose(curr_ped_seq[:, 2:]).astype(np.float32)

                    rel_feat_seq = np.zeros_like(feat_seq)
                    rel_feat_seq[:, 1:] = feat_seq[:, 1:] - feat_seq[:, :-1]

                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = feat_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_feat_seq

                    _non_linear_ped.append(poly_fit(feat_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1.0

                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        if len(seq_list) == 0:
            raise ValueError(
                f"No valid sequences were created from {data_dir}. "
                "Check preprocessing filters and obs_len/pred_len."
            )

        self.num_seq = len(seq_list)

        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped, dtype=np.float32)

        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).float()
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).float()
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).float()
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).float()
        self.loss_mask = torch.from_numpy(loss_mask_list).float()
        self.non_linear_ped = torch.from_numpy(non_linear_ped).float()

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(s, e) for s, e in zip(cum_start_idx, cum_start_idx[1:])]

        self.v_obs = []
        self.v_pred = []
        print("Processing Data .....")

        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            start, end = self.seq_start_end[ss]

            v_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], False)
            self.v_obs.append(v_.clone())

            v_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], False)
            self.v_pred.append(v_.clone())

        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        return [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.v_pred[index]
        ]
