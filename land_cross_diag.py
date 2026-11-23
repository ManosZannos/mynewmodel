#!/usr/bin/env python3
"""
land_cross_diag.py — διαγνωστικό, χωρίς εκπαίδευση: πόσες προβλέψεις πέφτουν σε ξηρά; Αποφασίζει αν αξίζει ένας
φυσικός περιορισμός μετά την πρόβλεψη (handoff §1). Ξηρά = raster SDF του stageE_arms.py (sdf < 0, κελί ~100 m).

Για τα παράθυρα της κύριας μετρικής και για την αλήθεια, το CV και κάθε checkpoint του phase3.py:
  any_land   ποσοστό παραθύρων με τουλάχιστον ένα από τα 10 σημεία σε ξηρά
  end_land   ποσοστό με το τελικό σημείο σε ξηρά
  depth_p50  διάμεση απόσταση μέσα στην ξηρά (m) των τελικών σημείων σε ξηρά
  FDE_share  μερίδιο του συνολικού FDE από τα παράθυρα any_land (άνω όριο αυτού που μπορεί να διορθώσει ένας περιορισμός)
Η γραμμή της αλήθειας είναι το «θόρυβος» του raster (σημεία που η ακτογραμμή βάζει σε ξηρά ενώ το πλοίο ήταν εκεί).

    python land_cross_diag.py dataset/houston_2025-03_7bfeb419 --eval_field_dir stageD_hab \
        --models reports/houston2_rtx/<m0>.pt reports/houston2_rtx/<m_habit+land>.pt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import baselines as BL
import land_diag as LDG
import loader as LD
import phase3 as P3
import stageE_arms as E


def sdf_at(r, d, xy):
    """Signed distance (m, < 0 = land) of local points xy (n, T, 2), in the frame of fut and pos (loader: origin = last report)."""
    m = d["meta"]
    c = m["last_avail"].to_numpy()
    lat0, lon0 = m["lat_last"].to_numpy(float), m["lon_last"].to_numpy(float)
    p_last = d["pos"][np.arange(len(c)), c]                      # last available point, same frame
    dx, dy = xy[..., 0] - p_last[:, None, 0], xy[..., 1] - p_last[:, None, 1]
    lat = lat0[:, None] + np.degrees(dy / LD.R_EARTH)
    lon = lon0[:, None] + np.degrees(dx / (LD.R_EARTH * np.cos(np.radians(lat0[:, None]))))
    big = 1e9                                                    # n=1 patch = the cell itself, no clipping
    v = E.sdf_patches(r, lat.ravel(), lon.ravel(), np.zeros(lat.size), n=1, half_m=0.0, clip_m=big)
    return v.reshape(lat.shape).astype(np.float64) * big


def stats(s, fde):
    """s: (n, T) signed distance, fde: (n,) -> row of the table."""
    on, end = s < 0, s[:, -1] < 0
    return {"any_land": on.any(1).mean(), "end_land": end.mean(),
            "depth_p50": float(np.median(-s[end, -1])) if end.any() else np.nan,
            "FDE_share": fde[on.any(1)].sum() / fde.sum() if fde is not None else np.nan}


def headline(d):
    """Only the headline windows; every per-window array is cut, every other key (features, ...) is kept."""
    h = d["meta"]["headline"].to_numpy() == 1
    return {k: (v[h].reset_index(drop=True) if k == "meta" else
                v[h] if isinstance(v, np.ndarray) and len(v) == len(h) else v) for k, v in d.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_dir")
    ap.add_argument("--manifest", default="vessel")
    ap.add_argument("--split", default="val")
    ap.add_argument("--models", nargs="*", default=[], help="checkpoints του phase3.py (όχι των σκελών)")
    ap.add_argument("--eval_field_dir", default=None)
    a = ap.parse_args(argv)
    d = LD.load_split(a.build_dir, a.manifest, a.split, x=LD.load_rows(a.build_dir, BL.shore_dir_of(a.build_dir)))
    d = headline(d)
    h = np.ones(len(d["meta"]), bool)
    r = E.load_sdf(a.build_dir)
    cv = P3.course(d["pos"], d["meta"]["last_avail"], d["meta"]["p_age_s"])[0]
    rows = {"αλήθεια": stats(sdf_at(r, d, d["fut"]), None), "CV": stats(sdf_at(r, d, cv), BL.errors(cv, d["fut"])[1])}
    if a.models:
        import torch
        device = BL.device_info(torch)["device"]
        for path in a.models:
            pred = LDG.model_pred(path, a.build_dir, d, device, a.eval_field_dir)
            rows[Path(path).stem] = stats(sdf_at(r, d, pred), BL.errors(pred, d["fut"])[1])
    print(f"[land_cross_diag] {a.build_dir} / {a.manifest} / {a.split} | κύρια μετρική n={int(h.sum()):,}")
    print(pd.DataFrame(rows).T.to_string(formatters={"any_land": "{:.2%}".format, "end_land": "{:.2%}".format,
                                                    "depth_p50": "{:.0f}".format, "FDE_share": "{:.2%}".format}))


if __name__ == "__main__":
    main()
