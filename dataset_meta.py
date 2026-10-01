"""
dataset_meta.py — AISDataset variant that ALSO records per-window metadata:
the vessel_id of every valid vessel, the source day (from the CSV file name)
and the first frame of the window.

Used only by dump_errors.py. It does NOT modify dataset.py and it produces the
SAME windows in the SAME order as AISDataset (the windowing logic below is a
verbatim copy of AISDataset._process_file plus three append() calls);
verify_against_parent() checks this explicitly.

vessel_id is a stable identifier across splits: preprocess_dec2021_merged.py
assigns it as the index of the sorted MMSI over the WHOLE month, before the
train/val/test split by day. That is what makes a "seen in train" audit possible.
"""

from __future__ import annotations
import glob
import os
import re

import numpy as np
import pandas as pd

from dataset import AISDataset


def _day_from_name(fpath: str) -> int:
    nums = re.findall(r'\d+', os.path.basename(fpath))
    return int(nums[-1]) if nums else -1


class AISDatasetMeta(AISDataset):

    def __init__(self, *args, **kwargs):
        # Must exist BEFORE the parent __init__, which calls self._process_file.
        self.vessel_ids_list = []     # per window: np.int64 [N_valid]
        self.day_list = []            # per window: int (day of month)
        self.frame_start_list = []    # per window: int (frame_id of first obs step)
        super().__init__(*args, **kwargs)

    def _process_file(self, fpath: str, stride: int) -> int:
        day = _day_from_name(fpath)

        df = pd.read_csv(fpath).dropna()
        if df.empty:
            return 0

        ts_vals = np.sort(df['frame_id'].unique())
        v_vals  = np.sort(df['vessel_id'].unique())

        T = len(ts_vals)
        N = len(v_vals)

        if T < self.win_len:
            return 0

        ts_idx = {t: i for i, t in enumerate(ts_vals)}
        v_idx  = {v: i for i, v in enumerate(v_vals)}

        arr = np.full((T, N, 4), np.nan, dtype=np.float32)

        ti = df['frame_id'].map(ts_idx).values
        vi = df['vessel_id'].map(v_idx).values
        arr[ti, vi, 0] = df['LON'].values
        arr[ti, vi, 1] = df['LAT'].values
        arr[ti, vi, 2] = df['SOG'].values
        arr[ti, vi, 3] = df['Heading'].values

        ts_diffs = np.diff(ts_vals)

        n_valid = 0
        for start in range(0, T - self.win_len + 1, stride):
            end = start + self.win_len

            if np.any(ts_diffs[start:end - 1] != 1):
                continue

            obs_arr  = arr[start : start + self.obs_len]
            pred_arr = arr[start + self.obs_len : end]

            present = ~np.isnan(obs_arr[:, :, :2]).any(axis=(0, 2))

            if present.sum() < self.min_vessels:
                continue

            pred_latlon = pred_arr[:, :, :2]
            has_pred = ~np.isnan(pred_latlon[:, present, :]).any(axis=(0, 2))

            if has_pred.sum() < self.min_vessels:
                continue

            valid_idx = np.where(present)[0][has_pred]

            if len(valid_idx) < self.min_vessels:
                continue

            self.obs_list.append(obs_arr[:, valid_idx, :].transpose(1, 0, 2))
            self.pred_list.append(pred_latlon[:, valid_idx, :].transpose(1, 0, 2))
            # ---- the only additions vs. AISDataset._process_file ----
            self.vessel_ids_list.append(v_vals[valid_idx].astype(np.int64))
            self.day_list.append(day)
            self.frame_start_list.append(int(ts_vals[start]))
            n_valid += 1

        return n_valid


def verify_against_parent(csv_dir: str, obs_len: int, pred_len: int, stride: int) -> None:
    """Assert AISDatasetMeta yields exactly the same windows as AISDataset."""
    a = AISDataset(csv_dir, obs_len, pred_len, stride=stride)
    b = AISDatasetMeta(csv_dir, obs_len, pred_len, stride=stride)
    assert len(a) == len(b), f'window count differs: {len(a)} vs {len(b)}'
    for i in range(len(a)):
        assert np.array_equal(a.obs_list[i], b.obs_list[i], equal_nan=True), f'obs differs at window {i}'
        assert np.array_equal(a.pred_list[i], b.pred_list[i], equal_nan=True), f'pred differs at window {i}'
        assert len(b.vessel_ids_list[i]) == a.obs_list[i].shape[0], f'vessel-id count differs at window {i}'
    print(f'[verify] AISDatasetMeta == AISDataset on {csv_dir}: {len(a)} windows identical')


def train_vessel_ids(data_dir: str) -> set:
    """vessel_ids that appear in ANY train CSV (upper bound of 'seen in training')."""
    ids = set()
    for f in sorted(glob.glob(os.path.join(data_dir, 'train', '*.csv'))):
        ids.update(pd.read_csv(f, usecols=['vessel_id'])['vessel_id'].unique().tolist())
    return ids
