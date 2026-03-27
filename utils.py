"""
utils.py

Includes:
1) NOAA AIS preprocessing (paper-aligned) -> frame-format CSV compatible with TrajectoryDataset:
   frame_id, vessel_id, LON, LAT, SOG, Heading

2) TrajectoryDataset: trajectory-wise sliding window (paper-aligned).
   Each window is anchored to a single vessel's continuous trajectory segment.
   Interaction graph at each timestep includes ALL vessels present in those frames.
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
    Load NOAA AIS daily CSV from either a .csv or a .zip path.
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
    Paper Step 1: Cleaning abnormal data.
    - 9-digit MMSI validation
    - Drop nulls in key fields
    - Filter by geographic/dynamic ranges (LAT, LON, SOG, Heading)
    """
    missing = [c for c in NOAA_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    initial_count = len(df)
    print(f"\n[Step 1/5] Cleaning abnormal data...")
    print(f"  Initial rows: {initial_count:,}")

    df = df.copy()
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], errors="coerce", utc=True)
    df = df.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "Heading", "Status"])
    print(f"  After removing nulls: {len(df):,} rows")

    before = len(df)
    df = df[_valid_mmsi_9digits(df["MMSI"])]
    print(f"  After MMSI validation: {len(df):,} rows ({before - len(df):,} removed)")

    before = len(df)
    df = df[
        df["LAT"].between(lat_range[0], lat_range[1]) &
        df["LON"].between(lon_range[0], lon_range[1]) &
        df["SOG"].between(sog_range[0], sog_range[1]) &
        df["Heading"].between(heading_range[0], heading_range[1])
    ]
    print(f"  After range filtering: {len(df):,} rows ({before - len(df):,} removed)")
    print(f"  Final cleaned rows: {len(df):,} ({100*(1-len(df)/initial_count):.1f}% reduction)")

    return df


def resample_interpolate_1min(
    df: pd.DataFrame,
    freq: str = "1min",
    rolling_window: int = 5,
    max_gap_minutes: int = 10,
) -> pd.DataFrame:
    """
    Paper Step 2: Data interpolation (gap-aware).

    Quote from paper:
    "For data with a time interval of more than one minute between consecutive
    trajectory points, the missing value of LON and LAT is supplemented by the
    linear interpolation method."

    KEY FIX: before resampling, each vessel's trajectory is split into segments
    at gaps > max_gap_minutes. Only genuine short gaps (≤ max_gap_minutes) are
    interpolated. Large gaps (e.g. vessel left area for hours) are NOT bridged
    with fake linear stretches.

    Example without fix:
      10:00, 10:03, 15:40  →  340 artificial points across a 5.5-hour gap

    Example with fix (max_gap_minutes=10):
      Segment 1: 10:00, 10:01, 10:02, 10:03  (3 real + 1 interpolated point)
      Segment 2: 15:40                         (1 point, too short for windows)

    Each segment is resampled independently. The MMSI column is preserved so
    the downstream filter_timestamps_min_vessels() and TrajectoryDataset work
    correctly.
    """
    initial_count = len(df)
    print(f"\n[Step 2/5] Data interpolation and resampling (max_gap={max_gap_minutes}min)...")
    print(f"  Input rows: {initial_count:,}")

    df = df.copy().sort_values(["MMSI", "BaseDateTime"])
    n_vessels = df["MMSI"].nunique()
    print(f"  Vessels to process: {n_vessels}")

    max_gap = pd.Timedelta(minutes=max_gap_minutes)
    out_parts = []

    for mmsi, g in df.groupby("MMSI", sort=False):
        g = (
            g.drop_duplicates(subset=["BaseDateTime"], keep="last")
            .sort_values("BaseDateTime")
            .set_index("BaseDateTime")
        )
        if len(g) == 0:
            continue

        # ----------------------------------------------------------------
        # Split into continuous segments at large gaps
        # ----------------------------------------------------------------
        time_diffs = g.index.to_series().diff()
        # First row has NaT diff → treat as start of first segment
        gap_mask = time_diffs > max_gap
        segment_ids = gap_mask.cumsum()  # 0, 0, 0, 1, 1, 2, ...

        for _, seg in g.groupby(segment_ids):
            if len(seg) < 2:
                # Single isolated point — cannot interpolate or form windows
                continue

            # Resample only within this segment's time range
            time_range = pd.date_range(
                start=seg.index.min(),
                end=seg.index.max(),
                freq=freq,
            )
            r = seg.reindex(time_range)
            r.index.name = "BaseDateTime"

            # LON/LAT: time-based linear interpolation
            r["LON"] = r["LON"].interpolate(method="time")
            r["LAT"] = r["LAT"].interpolate(method="time")

            # SOG/Heading: rolling average (paper: "average value is used")
            for c in ["SOG", "Heading"]:
                s = r[c]
                s = s.fillna(
                    s.rolling(window=rolling_window, min_periods=1, center=True).mean()
                )
                r[c] = s.ffill().bfill()

            r["Status"] = r["Status"].ffill().bfill()
            r = r.dropna(subset=["LON", "LAT", "SOG", "Heading"])

            if len(r) == 0:
                continue

            r["MMSI"] = mmsi
            out_parts.append(r.reset_index())

    result = pd.concat(out_parts, ignore_index=True) if out_parts else df.iloc[0:0].copy()
    print(f"  Output rows after resampling: {len(result):,}")
    print(f"  Vessels after resampling: {result['MMSI'].nunique()}")

    return result


def filter_timestamps_min_vessels(df: pd.DataFrame, min_vessels_per_timestamp: int = 3) -> pd.DataFrame:
    """
    Paper Step 1 (final): Keep only timestamps where >= min_vessels vessels exist.
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

    print(f"  Timestamps kept: {len(keep_times):,}")
    print(f"  Rows after filtering: {len(df):,} ({initial_count - len(df):,} removed)")
    print(f"  Vessels remaining: {df['MMSI'].nunique()}")

    return df


def zscore_normalize_global(df: pd.DataFrame, cols=("LON", "LAT", "SOG", "Heading"), stats: dict | None = None):
    """
    Paper Step 4: Z-score normalization using global dataset statistics.
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

    for c in cols:
        df[c] = (df[c] - stats[c]["mean"]) / stats[c]["std"]

    print(f"  Normalization complete.")
    return df, stats


def to_frame_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paper Step 5: Convert to frame format.

    Output columns: frame_id, vessel_id, LON, LAT, SOG, Heading

    frame_id is a global minute index from the day's minimum timestamp.
    This is used by TrajectoryDataset to:
      1. Find all vessels present at a given frame (for interaction graph)
      2. Slide windows per vessel trajectory (paper-faithful)
    """
    print(f"\n[Step 5/5] Converting to frame format...")

    df = df.copy().sort_values("BaseDateTime")
    t0 = df["BaseDateTime"].min()
    t_end = df["BaseDateTime"].max()

    df["frame_id"] = ((df["BaseDateTime"] - t0).dt.total_seconds() / 60.0).round().astype(int)
    df = df.rename(columns={"MMSI": "vessel_id"})

    result = df[["frame_id", "vessel_id", "LON", "LAT", "SOG", "Heading"]]

    print(f"  Time range: {t0} to {t_end}")
    print(f"  Total frames (minutes): {result['frame_id'].max() + 1}")
    print(f"  Total vessels: {result['vessel_id'].nunique()}")
    print(f"  Total data points: {len(result):,}")

    return result


def preprocess_noaa_to_frames(
    df_raw: pd.DataFrame,
    lat_range=(30.0, 35.0),
    lon_range=(-120.0, -115.0),
    sog_range=(1.0, 22.0),
    heading_range=(0.0, 360.0),
    min_vessels_per_timestamp: int = 3,
    max_gap_minutes: int = 10,
    resample_freq: str = "1min",
    do_zscore: bool = True,
    zscore_stats: dict | None = None,
):
    """
    Complete AIS preprocessing pipeline (paper-aligned).
    Steps 1-5 as described in the paper.

    Args:
        max_gap_minutes: gaps larger than this break a vessel trajectory into
                         separate segments before resampling (default: 10 min).
                         Prevents fake linear interpolation across hour-long gaps.
    """
    print("=" * 70)
    print("AIS DATA PREPROCESSING PIPELINE (Paper-Aligned)")
    print("=" * 70)
    print(f"Input data: {len(df_raw):,} rows, {df_raw['MMSI'].nunique()} vessels")

    df = clean_abnormal_data_noaa(df_raw, lat_range, lon_range, sog_range, heading_range)
    df = resample_interpolate_1min(df, freq=resample_freq, rolling_window=5, max_gap_minutes=max_gap_minutes)
    df = filter_timestamps_min_vessels(df, min_vessels_per_timestamp=min_vessels_per_timestamp)

    stats = None
    if do_zscore:
        df, stats = zscore_normalize_global(df, cols=("LON", "LAT", "SOG", "Heading"), stats=zscore_stats)

    frames = to_frame_format(df)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Output: {len(frames):,} data points across {frames['frame_id'].max()+1} frames")
    print(f"Vessels: {frames['vessel_id'].nunique()}")
    print("=" * 70 + "\n")

    return frames, stats


def save_frames_csv(frames_df: pd.DataFrame, out_csv_path: str):
    """Save frame-format CSV for TrajectoryDataset."""
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
    Adds a positional index as an extra feature.
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
    Builds node feature tensor V from RELATIVE features (velocities).

    Original repo (paper-faithful):
      - V always contains seq_rel (velocities), NOT absolute positions
      - When pos_enc=True, prepends a positional index as extra feature
        → output shape: (seq_len, N, 5) with [pos_idx, LON_rel, LAT_rel, SOG_rel, Heading_rel]
      - When pos_enc=False:
        → output shape: (seq_len, N, 4) with [LON_rel, LAT_rel, SOG_rel, Heading_rel]

    Args:
        seq_:    (N, 4, seq_len) — absolute positions (used only for shape reference)
        seq_rel: (N, 4, seq_len) — velocities (differences between consecutive steps)
        pos_enc: if True, prepend positional index [1, 2, ..., seq_len]

    Returns:
        V: torch.FloatTensor of shape (seq_len, N, 5) if pos_enc else (seq_len, N, 4)
    """
    assert seq_rel.dim() == 3, f"Expected seq_rel (N, F, T), got {seq_rel.shape}"

    # Use all 4 features: LON_rel, LAT_rel, SOG_rel, Heading_rel
    # seq_rel shape: (N, 4, seq_len) → permute → (seq_len, N, 4)
    V = seq_rel.permute(2, 0, 1).contiguous()  # (seq_len, N, 4)

    if pos_enc:
        # Add positional index as first feature → (seq_len, N, 5)
        # [pos_idx, LON_rel, LAT_rel, SOG_rel, Heading_rel]
        V_np = V.cpu().numpy()
        V_np = loc_pos(V_np)  # (seq_len, N, 5)
        return torch.from_numpy(V_np).float()

    return V.float()


def poly_fit(traj, traj_len, threshold):
    """
    Determines whether a trajectory is non-linear using a 2nd-order polynomial fit.
    Input: traj shape (C, traj_len), uses only first 2 channels (LON, LAT).
    """
    traj2 = traj[:2, :]
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj2[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj2[1, -traj_len:], 2, full=True)[1]
    return 1.0 if (res_x + res_y >= threshold) else 0.0


# ============================================================================
# TRAJECTORY DATASET (PAPER-ALIGNED: day-level frame sliding window)
# ============================================================================

class TrajectoryDataset(Dataset):
    """
    Dataloader for AIS trajectory datasets in frame format:
      frame_id, vessel_id, LON, LAT, SOG, Heading

    PAPER-ALIGNED APPROACH (Section 3.2.1):
      "All vessels in the sea area are constructed as the graph at each time step"

    For each day CSV:
      1. Build list of unique frame_ids (minutes of the day)
      2. Slide a window of seq_len frames over the day
      3. For each window, keep only vessels present in ALL seq_len frames
      4. Require at least min_ped vessels per window

    This is fast (vectorized numpy) and matches the paper's scene-level graph:
    all co-present vessels form the spatial graph at each timestep.
    """

    def __init__(
        self,
        data_dir,
        obs_len=10,
        pred_len=5,
        skip=1,
        threshold=0.002,
        min_ped=1,
        delim=",",
    ):
        super(TrajectoryDataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.skip = skip
        self.max_peds_in_frame = 0

        all_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith('.csv')
        ])
        if not all_files:
            raise ValueError(f"No CSV files found in {data_dir}")

        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped_list = []

        for path in all_files:
            # Load CSV: frame_id, vessel_id, LON, LAT, SOG, Heading
            data = pd.read_csv(path)
            data_np = data[["frame_id", "vessel_id", "LON", "LAT", "SOG", "Heading"]].values.astype(np.float32)

            frames = np.unique(data_np[:, 0]).tolist()
            n_frames = len(frames)

            # Build frame_id → index map and frame_data list
            frame_to_idx = {frame: i for i, frame in enumerate(frames)}
            frame_data = []
            for frame in frames:
                frame_data.append(data_np[data_np[:, 0] == frame])

            vessel_ids_in_file = np.unique(data_np[:, 1])
            print(f"  {os.path.basename(path)}: {len(vessel_ids_in_file)} vessels, {n_frames} frames")

            num_sequences = max(0, (n_frames - self.seq_len) // self.skip + 1)

            for idx in range(0, num_sequences * self.skip, self.skip):
                if idx + self.seq_len > n_frames:
                    break

                # All rows in this window
                curr_seq_data = np.concatenate(frame_data[idx: idx + self.seq_len], axis=0)

                # Vessels that appear in this window
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))

                curr_seq     = np.zeros((len(peds_in_curr_seq), 4, self.seq_len), dtype=np.float32)
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 4, self.seq_len), dtype=np.float32)
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len), dtype=np.float32)

                num_peds_considered = 0
                _non_linear_ped = []

                for ped_id in peds_in_curr_seq:
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                    pad_front = frame_to_idx[curr_ped_seq[0, 0]] - idx
                    pad_end   = frame_to_idx[curr_ped_seq[-1, 0]] - idx + 1

                    # Vessel must be present in ALL frames of the window
                    if pad_end - pad_front != self.seq_len:
                        continue
                    if curr_ped_seq.shape[0] != self.seq_len:
                        continue

                    feat_seq = np.transpose(curr_ped_seq[:, 2:]).astype(np.float32)  # (4, seq_len)
                    rel_feat_seq = np.zeros_like(feat_seq)
                    rel_feat_seq[:, 1:] = feat_seq[:, 1:] - feat_seq[:, :-1]

                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end]     = feat_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_feat_seq
                    curr_loss_mask[_idx, pad_front:pad_end]  = 1.0

                    _non_linear_ped.append(poly_fit(feat_seq, pred_len, threshold))
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped_list += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        if not seq_list:
            raise ValueError(
                f"No valid sequences created from {data_dir}. "
                "Check preprocessing filters and obs_len/pred_len."
            )

        print(f"\nTotal sequences: {len(seq_list)}")
        self.num_seq = len(seq_list)

        seq_arr         = np.concatenate(seq_list, axis=0)
        seq_rel_arr     = np.concatenate(seq_list_rel, axis=0)
        loss_mask_arr   = np.concatenate(loss_mask_list, axis=0)
        non_linear_arr  = np.asarray(non_linear_ped_list, dtype=np.float32)

        self.obs_traj     = torch.from_numpy(seq_arr[:, :, :self.obs_len]).float()
        self.pred_traj    = torch.from_numpy(seq_arr[:, :, self.obs_len:]).float()
        self.obs_traj_rel = torch.from_numpy(seq_rel_arr[:, :, :self.obs_len]).float()
        self.pred_traj_rel= torch.from_numpy(seq_rel_arr[:, :, self.obs_len:]).float()
        self.loss_mask    = torch.from_numpy(loss_mask_arr).float()
        self.non_linear_ped = torch.from_numpy(non_linear_arr).float()

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(s, e) for s, e in zip(cum_start_idx, cum_start_idx[1:])]

        self.v_obs  = []
        self.v_pred = []
        print("Processing graph tensors...")

        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            start, end = self.seq_start_end[ss]
            # V_obs: velocities + positional encoding → (obs_len, N, 4)
            # matches original repo: seq_to_graph(..., pos_enc=True)
            v_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], True)
            self.v_obs.append(v_.clone())
            # V_pred: absolute positions → (pred_len, N, 4) used as loss target
            v_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], False)
            self.v_pred.append(v_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        return [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
            self.v_obs[index],
            self.v_pred[index],
        ]

# ============================================================================
# TRAJECTORY DATASET JSON (DualSTMA comparison — METO-S2S format)
# Reads train.json / val.json / test.json directly, no CSV conversion needed.
#
# JSON format (per trajectory, per point):
#   [timestamp_ms, lon_norm, lat_norm, sog_norm, heading_norm,
#    distance, vessel_type, month, day, hour, lon_raw, lat_raw, mmsi]
#
# global_stats.json is computed from lon_raw / lat_raw in train.json
# and is needed by evaluate.py for denormalization.
# ============================================================================

# Field indices in each JSON point
_J_TIMESTAMP  = 0
_J_LON_NORM   = 1
_J_LAT_NORM   = 2
_J_SOG_NORM   = 3
_J_HEAD_NORM  = 4
_J_LON_RAW    = 10
_J_LAT_RAW    = 11
_J_MMSI       = 12


def compute_global_stats_from_json(train_json_path: str) -> dict:
    """
    Compute global normalization stats (mean/std) from raw LON/LAT values
    in train.json. Saves to global_stats.json in the same folder.
    These stats are used by evaluate.py for denormalization.
    """
    import json as _json

    with open(train_json_path, "r") as f:
        data = _json.load(f)

    lons, lats = [], []
    for traj in data:
        for point in traj:
            lons.append(point[_J_LON_RAW])
            lats.append(point[_J_LAT_RAW])

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
        "SOG": {"mean": 0.0, "std": 1.0},
        "Heading": {"mean": 0.0, "std": 1.0},
    }

    stats_path = os.path.join(os.path.dirname(train_json_path), "global_stats.json")
    import json as _json2
    with open(stats_path, "w") as f:
        _json2.dump(stats, f, indent=2)

    print(f"global_stats.json saved to {stats_path}")
    print(f"  LON: mean={stats['LON']['mean']:.4f}, std={stats['LON']['std']:.4f}")
    print(f"  LAT: mean={stats['LAT']['mean']:.4f}, std={stats['LAT']['std']:.4f}")

    return stats


class TrajectoryDatasetJSON(Dataset):
    """
    Dataset loader for METO-S2S JSON format (DualSTMA comparison).

    Reads train.json / val.json / test.json directly.
    Each JSON file is a list of trajectories.
    Each trajectory is a list of points:
      [timestamp_ms, lon_norm, lat_norm, sog_norm, heading_norm,
       distance, vessel_type, month, day, hour, lon_raw, lat_raw, mmsi]

    Data is already normalized by METO-S2S preprocessing.
    Sampling interval: 10 min (DualSTMA-aligned).

    Sliding window per trajectory (not per frame like TrajectoryDataset):
      - obs_len=10 × 10min = 100min observation
      - pred_len=5  × 10min = 50min prediction

    Interaction graph: all vessels sharing the same timestamp window.
    Built by grouping trajectories by their timestamp overlap.
    """

    def __init__(
        self,
        json_path: str,
        obs_len: int = 10,
        pred_len: int = 5,
        skip: int = 1,
        threshold: float = 0.002,
        min_ped: int = 1,
    ):
        import json as _json

        super(TrajectoryDatasetJSON, self).__init__()

        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.seq_len  = obs_len + pred_len
        self.skip     = skip
        self.max_peds_in_frame = 0

        print(f"Loading {json_path} ...")
        with open(json_path, "r") as f:
            raw = _json.load(f)
        print(f"  {len(raw)} trajectories loaded")

        # ── Build frame_id → list of (vessel_idx, point_idx) ────────────
        # frame_id = timestamp_ms // (10 * 60 * 1000)  [10-min units]
        # We need to find windows where multiple vessels co-exist.

        # Step 1: convert each trajectory to numpy array and assign frame_ids
        trajs = []   # list of (mmsi, frames_array, feat_array)
        for traj in raw:
            if len(traj) < self.seq_len:
                continue
            mmsi   = traj[0][_J_MMSI]
            frames = np.array([p[_J_TIMESTAMP] // (10 * 60 * 1000) for p in traj], dtype=np.int64)
            feats  = np.array(
                [[p[_J_LON_NORM], p[_J_LAT_NORM], p[_J_SOG_NORM], p[_J_HEAD_NORM]] for p in traj],
                dtype=np.float32
            )  # (T, 4)
            trajs.append((mmsi, frames, feats))

        print(f"  {len(trajs)} trajectories with >= {self.seq_len} points")

        # Step 2: build frame → set of traj indices
        frame_to_trajs: dict = {}
        for ti, (mmsi, frames, feats) in enumerate(trajs):
            for fi, fid in enumerate(frames):
                if fid not in frame_to_trajs:
                    frame_to_trajs[fid] = []
                frame_to_trajs[fid].append((ti, fi))

        # Step 3: sliding window per trajectory
        num_peds_in_seq  = []
        seq_list         = []
        seq_list_rel     = []
        loss_mask_list   = []
        non_linear_ped_list = []

        for ti, (mmsi, frames, feats) in enumerate(trajs):
            T = len(frames)
            num_windows = max(0, (T - self.seq_len) // self.skip + 1)

            for w in range(0, num_windows * self.skip, self.skip):
                if w + self.seq_len > T:
                    break

                # Frame IDs for this window
                win_frames = frames[w: w + self.seq_len]

                # Check continuity (no gaps > 1 step in the anchor trajectory)
                if np.any(np.diff(win_frames) != 1):
                    continue

                # Find all vessels that are present in ALL frames of this window
                # (intersection of vessel sets across all window frame_ids)
                present_sets = [
                    set(idx for idx, _ in frame_to_trajs.get(fid, []))
                    for fid in win_frames
                ]
                common_traj_idxs = set.intersection(*present_sets) if present_sets else set()
                common_traj_idxs = sorted(common_traj_idxs)

                if len(common_traj_idxs) <= min_ped:
                    continue

                self.max_peds_in_frame = max(self.max_peds_in_frame, len(common_traj_idxs))

                N = len(common_traj_idxs)
                curr_seq      = np.zeros((N, 4, self.seq_len), dtype=np.float32)
                curr_seq_rel  = np.zeros((N, 4, self.seq_len), dtype=np.float32)
                curr_loss_mask = np.zeros((N, self.seq_len), dtype=np.float32)
                _non_linear   = []
                num_considered = 0

                for ni, cti in enumerate(common_traj_idxs):
                    _, c_frames, c_feats = trajs[cti]

                    # Find the point index in this trajectory for win_frames[0]
                    matches = np.where(c_frames == win_frames[0])[0]
                    if len(matches) == 0:
                        continue
                    start_fi = matches[0]
                    end_fi   = start_fi + self.seq_len

                    if end_fi > len(c_frames):
                        continue

                    # Check this vessel is also continuous
                    if np.any(np.diff(c_frames[start_fi:end_fi]) != 1):
                        continue

                    feat_seq = c_feats[start_fi:end_fi].T  # (4, seq_len)
                    rel_feat_seq = np.zeros_like(feat_seq)
                    rel_feat_seq[:, 1:] = feat_seq[:, 1:] - feat_seq[:, :-1]

                    curr_seq[num_considered]      = feat_seq
                    curr_seq_rel[num_considered]  = rel_feat_seq
                    curr_loss_mask[num_considered] = 1.0

                    _non_linear.append(poly_fit(feat_seq, pred_len, threshold))
                    num_considered += 1

                if num_considered <= min_ped:
                    continue

                non_linear_ped_list += _non_linear[:num_considered]
                num_peds_in_seq.append(num_considered)
                loss_mask_list.append(curr_loss_mask[:num_considered])
                seq_list.append(curr_seq[:num_considered])
                seq_list_rel.append(curr_seq_rel[:num_considered])

        if not seq_list:
            raise ValueError(
                f"No valid sequences from {json_path}. "
                "Check obs_len/pred_len and trajectory lengths."
            )

        print(f"  Total sequences: {len(seq_list)}")
        self.num_seq = len(seq_list)

        seq_arr        = np.concatenate(seq_list,       axis=0)
        seq_rel_arr    = np.concatenate(seq_list_rel,   axis=0)
        loss_mask_arr  = np.concatenate(loss_mask_list, axis=0)
        non_linear_arr = np.asarray(non_linear_ped_list, dtype=np.float32)

        self.obs_traj      = torch.from_numpy(seq_arr[:, :, :self.obs_len]).float()
        self.pred_traj     = torch.from_numpy(seq_arr[:, :, self.obs_len:]).float()
        self.obs_traj_rel  = torch.from_numpy(seq_rel_arr[:, :, :self.obs_len]).float()
        self.pred_traj_rel = torch.from_numpy(seq_rel_arr[:, :, self.obs_len:]).float()
        self.loss_mask     = torch.from_numpy(loss_mask_arr).float()
        self.non_linear_ped = torch.from_numpy(non_linear_arr).float()

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(s, e) for s, e in zip(cum_start_idx, cum_start_idx[1:])]

        self.v_obs  = []
        self.v_pred = []
        print("Processing graph tensors...")
        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            start, end = self.seq_start_end[ss]
            v_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], True)
            self.v_obs.append(v_.clone())
            v_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], False)
            self.v_pred.append(v_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        return [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
            self.v_obs[index],
            self.v_pred[index],
        ]