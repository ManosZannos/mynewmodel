"""
utils.py

Includes:
1) NOAA AIS preprocessing (paper-aligned) -> frame-format CSV compatible with TrajectoryDataset:
   frame_id, vessel_id, LON, LAT, SOG, Heading

2) TrajectoryDataset: trajectory-wise sliding window (paper-aligned).
   Each window is anchored to a single vessel's continuous trajectory segment.
   Interaction graph at each timestep includes ALL vessels present in those frames.

KEY FIXES vs original version:
  FIX 1 - resample_interpolate_1min():
    OLD: reindex over full vessel span (first→last timestamp) → massive artificial
         stretches across gaps of hours (e.g. 10:00, 10:03, 15:40 → 340 fake points).
    NEW: split each vessel trajectory into segments at gaps > MAX_GAP_MINUTES before
         resampling → only genuine 1-min gaps get interpolated, large gaps stay broken.

  FIX 2 - TrajectoryDataset:
    OLD: sliding window over day-level frames → a vessel survives only if present
         in EVERY frame of the window (~99% vessel dropout).
    NEW: sliding window per vessel trajectory segment → every vessel that has
         seq_len consecutive 1-min points produces windows, and neighbours are
         collected from the shared frame_id index at each timestep.
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
    df = resample_interpolate_1min(df, freq="1min", rolling_window=5, max_gap_minutes=max_gap_minutes)
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
    Builds node feature tensor V of shape (seq_len, N, 4) from absolute features.
    Input: seq_ shape (N, 4, seq_len)
    Output: V shape (seq_len, N, 4)
    """
    assert seq_.dim() == 3, f"Expected seq_ (N, 4, T), got {seq_.shape}"
    assert seq_rel.dim() == 3, f"Expected seq_rel (N, 4, T), got {seq_rel.shape}"
    assert seq_.shape[1] == 4, f"Expected 4 features, got {seq_.shape[1]}"

    V = seq_.permute(2, 0, 1).contiguous()  # (seq_len, N, 4)

    if pos_enc:
        V_np = V.cpu().numpy()
        V_np = loc_pos(V_np)
        return torch.from_numpy(V_np).float().to(seq_.device)

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
# TRAJECTORY DATASET (PAPER-ALIGNED: trajectory-wise sliding window)
# ============================================================================

class TrajectoryDataset(Dataset):
    """
    Dataloader for AIS trajectory datasets in frame format:
      frame_id, vessel_id, LON, LAT, SOG, Heading

    PAPER-ALIGNED APPROACH:
      - For each vessel, extract all continuous trajectory segments
        (consecutive 1-min timesteps; gaps > max_gap_minutes break a segment)
      - Slide a window of seq_len steps over each segment (skip=1 by default)
      - For each window (anchored to the focal vessel), collect ALL other vessels
        present at those exact frame_ids to build the interaction graph
      - This matches the paper: "trajectory data is defined as T={p0,...,pn}
        for each vessel, then sliding window sampling"

    OLD (WRONG) approach: sliding window over all day-level frames → vessel
    must appear in EVERY frame of the window, causing ~99% vessel dropout.
    """

    def __init__(
        self,
        data_dir,
        obs_len=10,
        pred_len=10,
        skip=1,
        threshold=0.002,
        min_ped=1,
        max_gap_minutes=1,   # gap > this breaks a trajectory into separate segments
        min_neighbours=1,    # minimum number of OTHER vessels needed in a window
    ):
        super(TrajectoryDataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.skip = skip
        self.data_dir = data_dir
        self.max_gap_minutes = max_gap_minutes
        self.min_neighbours = min_neighbours
        self.max_peds_in_frame = 0

        all_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith('.csv')
        ])
        if not all_files:
            raise ValueError(f"No CSV files found in {data_dir}")

        # ------------------------------------------------------------------
        # Collect all windows across all files
        # Each window entry: (file_idx, focal_vessel_id, [frame_ids of window])
        # ------------------------------------------------------------------
        # We store all data per file in a dict for fast frame lookup.
        # file_data[file_idx] = dict: frame_id -> array of rows (vessel_id, 4 feats)
        # ------------------------------------------------------------------

        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped_list = []

        for path in all_files:
            data = pd.read_csv(path)
            data = data[["frame_id", "vessel_id", "LON", "LAT", "SOG", "Heading"]].copy()

            # Build frame→rows lookup for fast neighbour retrieval
            # frame_lookup: frame_id (int) -> np.array shape (K, 6) [frame_id, vessel_id, 4 feats]
            data_np = data.values.astype(np.float64)
            all_frames = np.unique(data_np[:, 0]).astype(int)
            frame_lookup = {
                int(f): data_np[data_np[:, 0] == f]
                for f in all_frames
            }

            # For each vessel, find its continuous segments and slide windows
            vessel_ids = data["vessel_id"].unique()
            print(f"  {os.path.basename(path)}: {len(vessel_ids)} vessels, "
                  f"{len(all_frames)} frames")

            for vid in vessel_ids:
                vdata = data[data["vessel_id"] == vid].sort_values("frame_id")
                fids = vdata["frame_id"].values.astype(int)
                feats = vdata[["LON", "LAT", "SOG", "Heading"]].values  # (T_v, 4)

                if len(fids) < self.seq_len:
                    continue

                # Split into continuous segments (gap > max_gap_minutes breaks)
                gaps = np.diff(fids)
                break_points = np.where(gaps > self.max_gap_minutes)[0] + 1
                seg_starts = np.concatenate([[0], break_points])
                seg_ends = np.concatenate([break_points, [len(fids)]])

                for seg_s, seg_e in zip(seg_starts, seg_ends):
                    seg_fids = fids[seg_s:seg_e]
                    seg_feats = feats[seg_s:seg_e]  # (seg_len, 4)
                    seg_len = len(seg_fids)

                    if seg_len < self.seq_len:
                        continue

                    # Slide window over this segment
                    n_windows = (seg_len - self.seq_len) // self.skip + 1
                    for w in range(0, n_windows * self.skip, self.skip):
                        if w + self.seq_len > seg_len:
                            break

                        win_fids = seg_fids[w: w + self.seq_len]      # (seq_len,)
                        win_feats = seg_feats[w: w + self.seq_len]     # (seq_len, 4)

                        # --------------------------------------------------
                        # Collect ALL vessels present at these frame_ids
                        # This is the interaction scene for this window
                        # --------------------------------------------------
                        # vessel_feats: dict vessel_id -> array (seq_len, 4) or None
                        scene_vessels = {}  # vid2 -> (seq_len, 4) feat array

                        # Focal vessel always included
                        scene_vessels[vid] = win_feats

                        for t, fid in enumerate(win_fids):
                            if fid not in frame_lookup:
                                continue
                            rows = frame_lookup[fid]  # (K, 6): frame_id, vessel_id, 4 feats
                            for row in rows:
                                vid2 = int(row[1])
                                if vid2 == vid:
                                    continue
                                if vid2 not in scene_vessels:
                                    scene_vessels[vid2] = np.full((self.seq_len, 4), np.nan)
                                scene_vessels[vid2][t] = row[2:6]

                        # Keep only vessels present in ALL timesteps
                        # (partial vessels cause NaN; drop them for graph consistency)
                        valid_vessels = []
                        for vid2, feat_arr in scene_vessels.items():
                            if not np.any(np.isnan(feat_arr)):
                                valid_vessels.append((vid2, feat_arr))

                        # Need at least focal vessel + min_neighbours others
                        n_valid = len(valid_vessels)
                        if n_valid < 1 + self.min_neighbours:
                            continue

                        self.max_peds_in_frame = max(self.max_peds_in_frame, n_valid)

                        # --------------------------------------------------
                        # Build curr_seq: (N, 4, seq_len)
                        # Put focal vessel first, then neighbours
                        # --------------------------------------------------
                        # Sort: focal first, then by vessel_id for reproducibility
                        valid_vessels.sort(key=lambda x: (0 if x[0] == vid else 1, x[0]))

                        N = len(valid_vessels)
                        curr_seq = np.zeros((N, 4, self.seq_len), dtype=np.float32)
                        curr_seq_rel = np.zeros((N, 4, self.seq_len), dtype=np.float32)
                        curr_loss_mask = np.ones((N, self.seq_len), dtype=np.float32)

                        _non_linear = []
                        for i, (vid2, feat_arr) in enumerate(valid_vessels):
                            feat_T = feat_arr.T.astype(np.float32)  # (4, seq_len)
                            rel = np.zeros_like(feat_T)
                            rel[:, 1:] = feat_T[:, 1:] - feat_T[:, :-1]

                            curr_seq[i] = feat_T
                            curr_seq_rel[i] = rel
                            _non_linear.append(poly_fit(feat_T, pred_len, threshold))

                        num_peds_in_seq.append(N)
                        seq_list.append(curr_seq)
                        seq_list_rel.append(curr_seq_rel)
                        loss_mask_list.append(curr_loss_mask)
                        non_linear_ped_list.extend(_non_linear)

        if not seq_list:
            raise ValueError(
                f"No valid sequences created from {data_dir}. "
                "Check preprocessing filters and obs_len/pred_len."
            )

        print(f"\nTotal sequences: {len(seq_list)}")

        self.num_seq = len(seq_list)

        seq_arr = np.concatenate(seq_list, axis=0)          # (total_N, 4, seq_len)
        seq_rel_arr = np.concatenate(seq_list_rel, axis=0)
        loss_mask_arr = np.concatenate(loss_mask_list, axis=0)
        non_linear_arr = np.asarray(non_linear_ped_list, dtype=np.float32)

        self.obs_traj = torch.from_numpy(seq_arr[:, :, :self.obs_len]).float()
        self.pred_traj = torch.from_numpy(seq_arr[:, :, self.obs_len:]).float()
        self.obs_traj_rel = torch.from_numpy(seq_rel_arr[:, :, :self.obs_len]).float()
        self.pred_traj_rel = torch.from_numpy(seq_rel_arr[:, :, self.obs_len:]).float()
        self.loss_mask = torch.from_numpy(loss_mask_arr).float()
        self.non_linear_ped = torch.from_numpy(non_linear_arr).float()

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(s, e) for s, e in zip(cum_start_idx, cum_start_idx[1:])]

        self.v_obs = []
        self.v_pred = []
        print("Processing graph tensors...")

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
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
            self.v_obs[index],
            self.v_pred[index],
        ]