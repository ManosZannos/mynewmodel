#!/usr/bin/env python3
"""
stageE_arms.py — δεδομένα για τα ΕΦΕΔΡΙΚΑ σκέλη της Φάσης 3 (docs/phase3_handoff.md §6δ). Τρέχει μόνο αν το πολικό
πεδίο του σταδίου Δ δεν περάσει τις πύλες. Δεν αλλάζει κανένα άλλο στάδιο.

  nb    stageE/neighbours.parquet: ανά επιλέξιμο παράθυρο οι έως 8 πλησιέστεροι γείτονες σε 5 km τη στιγμή t0
        (ίδιοι υποψήφιοι και ίδιος κανόνας αιτιότητας με το στάδιο Δ), στο πλαίσιο της πορείας:
        σχετική θέση / R, σχετική ταχύτητα / 20 kn, απόσταση / R, και σημαία εγκυρότητας → phase3_pool.py
  sdf   stageE/sdf.npz: raster προσημασμένης απόστασης από την ακτή (m, θετική στο νερό) σε κάναβο ~100 m πάνω στο
        κουτί των δεδομένων, από το D_SHORE / ON_LAND του σταδίου Γ → phase3_sdf.py κόβει patches ανά παράθυρο

    python stageE_arms.py nb  dataset/socal_2025-03_b19c45d7
    python stageE_arms.py sdf dataset/socal_2025-03_b19c45d7 --osm_dir data/osm/2026-10-29
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baselines as BL
import loader as LD
import phase3 as P3
import stageD_field as D

ARMS_VERSION = "stageE-v1.0"
NB_K = 8
VEL_SCALE = 20.0 * D.KN            # m/min
SDF = {"step_m": 100.0, "pad_m": 6000.0, "n": 32, "half_m": 5000.0, "clip_m": 5000.0}


# ---------------------------------------------------------------------------
def neighbour_features(tp, tv, tcourse, tid, cp, cvel, cid, cfg, k=NB_K):
    """One minute: (m, k, 5) features of the k nearest candidates within the radius, in the course frame
    (+x = ahead, +y = port), and (m, k) validity."""
    m, R = len(tp), cfg["radius_m"]
    out, valid = np.zeros((m, k, 5)), np.zeros((m, k))
    kk = min(k, len(cp))
    if m == 0 or kk == 0:
        return out, valid
    r, rv = cp[None] - tp[:, None], cvel[None] - tv[:, None]
    dist = np.where(np.asarray(tid)[:, None] == np.asarray(cid)[None], np.inf, np.linalg.norm(r, axis=-1))
    order, rows = np.argsort(dist, axis=1, kind="stable")[:, :kk], np.arange(m)[:, None]
    d = dist[rows, order]
    ok = d <= R
    th = np.radians(90.0 - np.asarray(tcourse))
    f = np.concatenate([P3.rot(r[rows, order], th) / R, P3.rot(rv[rows, order], th) / VEL_SCALE,
                        (np.where(ok, d, 0.0) / R)[..., None]], axis=-1)
    out[:, :kk], valid[:, :kk] = np.where(ok[..., None], f, 0.0), ok
    return out, valid


def load_neighbours(build, window_ids):
    """(n, 8*5 + 8): flattened neighbour features followed by the validity flags."""
    f = pd.read_parquet(Path(build) / "stageE" / "neighbours.parquet").set_index("window_id")
    if not pd.Index(window_ids).isin(f.index).all():
        raise AssertionError("neighbours.parquet: λείπουν παράθυρα")
    return f.loc[window_ids].to_numpy(np.float32)


def sdf_grid(lat_min, lat_max, lon_min, lon_max, step_m):
    dlat = np.degrees(step_m / LD.R_EARTH)
    dlon = dlat / np.cos(np.radians(0.5 * (lat_min + lat_max)))
    return np.arange(lat_min, lat_max + dlat, dlat), np.arange(lon_min, lon_max + dlon, dlon), dlat, dlon


def sdf_raster(lats, lons, G, tr, lon0, log=None):
    """Signed distance to the shore (m, positive on water) on the lat × lon grid, from stage C's D_SHORE / ON_LAND."""
    import stageC_shore as SC
    sdf, zero = np.empty((len(lats), len(lons)), np.float16), np.zeros(len(lons))
    for i, la in enumerate(lats):                           # one raster row per call: no ray casting (cog_valid = 0)
        f = SC.features_latlon(np.full(len(lons), la), lons, zero, zero, G, tr, lon0, SC.DEFAULTS)
        sdf[i] = f["D_SHORE"].to_numpy(float) * np.where(f["ON_LAND"].to_numpy() == 1, -1.0, 1.0)
        if log and i % 200 == 0:
            log(i)
    return sdf


def load_sdf(build):
    z = np.load(Path(build) / "stageE" / "sdf.npz")
    return {k: z[k] for k in z.files}


def sdf_patches(r, lat, lon, th, n=SDF["n"], half_m=SDF["half_m"], clip_m=SDF["clip_m"], chunk=50_000):
    """(m, n*n) patches around (lat, lon), first axis = ahead (course th, rad from east), second = to port.
    Values clipped to ±clip_m and scaled to [-1, 1]. Nearest raster cell (raster ~100 m, patch cell ~300 m)."""
    u = (np.arange(n) + 0.5) / n * 2 * half_m - half_m
    fwd, left = np.meshgrid(u, u, indexing="ij")
    lat, lon, th = (np.asarray(v, float) for v in (lat, lon, th))
    out = np.empty((len(lat), n * n), np.float32)
    H, W = r["sdf"].shape
    for a in range(0, len(lat), chunk):
        la, lo, t = (v[a:a + chunk, None, None] for v in (lat, lon, th))
        east, north = np.cos(t) * fwd - np.sin(t) * left, np.sin(t) * fwd + np.cos(t) * left
        i = np.rint((la + np.degrees(north / LD.R_EARTH) - r["lat0"]) / r["dlat"]).astype(np.int64)
        j = np.rint((lo + np.degrees(east / (LD.R_EARTH * np.cos(np.radians(la)))) - r["lon0"]) / r["dlon"]).astype(np.int64)
        v = r["sdf"][np.clip(i, 0, H - 1), np.clip(j, 0, W - 1)].astype(np.float32)
        out[a:a + chunk] = (np.clip(v, -clip_m, clip_m) / clip_m).reshape(len(la), -1)
    return out


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["nb", "sdf"])
    ap.add_argument("build_dir")
    ap.add_argument("--osm_dir", default=None, help="για το sdf: data/osm/<YYYY-MM-DD>, όπως στο στάδιο Γ")
    a = ap.parse_args(argv)
    if a.what == "sdf" and not a.osm_dir:
        sys.exit("το sdf θέλει --osm_dir")
    t0, cfg = time.time(), dict(D.DEFAULTS)
    out = Path(a.build_dir) / "stageE"
    out.mkdir(parents=True, exist_ok=True)
    x = LD.load_rows(a.build_dir, BL.shore_dir_of(a.build_dir))
    meta = {"version": ARMS_VERSION, "what": a.what, "stageD_config": cfg, "phase3": P3.PHASE3_VERSION}
    if a.what == "nb":
        w = LD._read_tab(Path(a.build_dir) / "stageB" / "windows.parquet")
        w = w[w["eligible"] == 1].sort_values("window_id").reset_index(drop=True)
        s = D.window_state(x, w)
        feat, valid = np.zeros((len(s), NB_K, 5)), np.zeros((len(s), NB_K))
        for g, *pair in D.minute_pairs(x, s, cfg):
            feat[g], valid[g] = neighbour_features(*pair, cfg)
        f = pd.DataFrame(np.concatenate([feat.reshape(len(s), -1), valid], axis=1).astype(np.float32),
                         columns=[f"nb{i}_{c}" for i in range(NB_K) for c in ("x", "y", "vx", "vy", "d")]
                         + [f"nbv{i}" for i in range(NB_K)])
        f.insert(0, "window_id", s["window_id"].to_numpy())
        if len(f) != len(w) or not np.isfinite(f.to_numpy()).all():
            raise AssertionError("γείτονες: γραμμές ≠ επιλέξιμα παράθυρα ή μη πεπερασμένες τιμές")
        f.to_parquet(out / "neighbours.parquet", index=False)
        meta.update({"k": NB_K, "funnel": {"eligible_windows": int(len(w)), "rows_out": int(len(f))},
                     "with_neighbour": float(valid.any(1).mean()), "mean_neighbours": float(valid.sum(1).mean())})
        print(f"[stageE nb] {len(f):,} παράθυρα (= επιλέξιμα ✓) | με γείτονα {meta['with_neighbour']:.1%} | {time.time() - t0:.0f} s")
    else:
        G, tr, lon0 = D.geometry(x, a.osm_dir, cfg)
        pad = np.degrees(SDF["pad_m"] / LD.R_EARTH)
        lats, lons, dlat, dlon = sdf_grid(x["lat"].min() - pad, x["lat"].max() + pad,
                                          x["lon"].min() - 2 * pad, x["lon"].max() + 2 * pad, SDF["step_m"])
        sdf = sdf_raster(lats, lons, G, tr, lon0,
                         lambda i: print(f"    γραμμή {i:,} / {len(lats):,} | {time.time() - t0:.0f} s"))
        np.savez(out / "sdf.npz", sdf=sdf, lat0=lats[0], lon0=lons[0], dlat=dlat, dlon=dlon)
        meta.update({"sdf": SDF, "shape": list(sdf.shape), "land_share": float((sdf < 0).mean()), "osm_dir": a.osm_dir})
        print(f"[stageE sdf] raster {sdf.shape[0]:,} × {sdf.shape[1]:,} | ξηρά {meta['land_share']:.1%} | {time.time() - t0:.0f} s")
    meta["seconds"] = round(time.time() - t0, 1)
    (out / f"stageE_{a.what}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return meta


if __name__ == "__main__":
    main()
