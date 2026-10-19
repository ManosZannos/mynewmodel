#!/usr/bin/env python3
"""
v5_visual.py — V5 της προδιαγραφής: οπτικός έλεγχος του σταδίου Α.

Ξαναπερνά τον μήνα από τον καθαρισμό (μέχρι τη segmentation) καταγράφοντας ΠΟΙΕΣ
αναφορές αφαιρέθηκαν ως spikes και ΠΟΥ έγιναν τομές από άλματα, και φτιάχνει σελίδες
με εικόνες (raw αναφορές πάνω στο πλέγμα του build):

  despike_*.png   δείγμα αφαιρέσεων spikes (κόκκινο ×)
  cuts_*.png      δείγμα τομών από άλματα (πορτοκαλί)
  mmsiqc_*.png    τα MMSI με σημαία QC
  random_*.png    30 τυχαία κινούμενα segments
  extreme_*.png   20 ακραία segments (στροφές, GAP_IMP, QC_SOG)

και τυπώνει πώς κατανέμονται τα spikes και οι τομές ανά πλοίο και ανά μέρα.

    python v5_visual.py --raw_dir data/2025_03 --month 2025-03 --region socal \
        --build dataset/socal_2025-03_b87967f8 --out reports/v5_socal_2025-03
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import preprocess_noaa as P  # noqa: E402

GRID_COLS = ["mmsi", "t", "seg_id", "lat", "lon", "sog", "cog", "COG_VALID", "GAP_IMP",
             "QC_SOG", "STATIONARY", "vessel_cat", "transceiver"]


# ---------------------------------------------------------------------------
def load_raw(raw_dir, month, cfg, days=None):
    files = sorted(glob.glob(os.path.join(raw_dir, f"ais-{month}-*.csv.zst")))
    if days:
        files = [f for f in files if int(f[-10:-8]) in days]
    parts = []
    for f in files:
        d = pd.read_csv(f, usecols=lambda c: c in P.NOAA_COLS, low_memory=False)
        lat = pd.to_numeric(d["latitude"], errors="coerce")
        lon = pd.to_numeric(d["longitude"], errors="coerce")
        m = P.box_with_margin(pd.DataFrame({"lat": lat, "lon": lon}), cfg["region"], cfg["margin_km"])
        parts.append(d.loc[m.fillna(False)])
        print(f"  {os.path.basename(f)}: {m.sum():,}")
    return pd.concat(parts, ignore_index=True)


def scan(raw, cfg):
    """Καθαρισμός όπως στο run_stage_a, με καταγραφή spikes και τομών."""
    df = P.standardize(raw)
    df = df[df["t"].notna()]
    df = df[P.valid_mmsi(df["mmsi"])].copy()
    df["mmsi"] = df["mmsi"].astype(np.int64)
    df.loc[(df["cog"] >= 360) | (df["cog"] < 0), "cog"] = np.nan
    df.loc[(df["sog"] >= 102.2) | (df["sog"] < 0), "sog"] = np.nan
    vt = df.loc[df["vessel_type"] > 0].groupby("mmsi")["vessel_type"].agg(P._mode)
    cat = vt.map(P.vessel_category)
    caps = cfg["speed_caps_kn"]
    df["cap"] = df["mmsi"].map(cat).fillna("unknown").map(lambda c: caps.get(c, caps["global"])).astype(float)
    df.loc[df["sog"] > df["cap"], "sog"] = np.nan
    df["_n"] = df[["sog", "cog", "heading"]].notna().sum(axis=1)
    df = df.sort_values(["mmsi", "t", "_n"], ascending=[True, True, False], kind="mergesort")
    df = df[~df.duplicated(["mmsi", "t"], keep="first")]
    rows = []
    for mmsi, g in df.groupby("mmsi", sort=True):
        t = ((g["t"] - P.EPOCH) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
        lat, lon = g["lat"].to_numpy(float), g["lon"].to_numpy(float)
        sog, cog = g["sog"].to_numpy(float), g["cog"].to_numpy(float)
        cap = float(g["cap"].iloc[0])
        keep, _ = P.despike_vessel(lat, lon, t, sog, cog, cap, cfg)
        kept_idx = np.flatnonzero(keep)
        seg, _ = P.segment_vessel(lat[keep], lon[keep], t[keep], sog[keep], cap, cfg)
        cut = np.zeros(len(t), bool)
        if len(kept_idx) > 1:
            dt = np.diff(t[keep])
            jump_cut = (np.diff(seg) > 0) & (dt <= cfg["seg_gap_s"])
            cut[kept_idx[1:][jump_cut]] = True
        rows.append(pd.DataFrame({"mmsi": mmsi, "t": t, "lat": lat, "lon": lon, "sog": sog,
                                  "spike": ~keep, "cut_before": cut}))
    return pd.concat(rows, ignore_index=True)


def load_grid(build):
    files = sorted(glob.glob(f"{build}/day_*.parquet"))
    if files:
        x = pd.concat((pd.read_parquet(f, columns=GRID_COLS) for f in files), ignore_index=True)
    else:
        files = sorted(glob.glob(f"{build}/day_*.csv.gz"))
        x = pd.concat((pd.read_csv(f, usecols=GRID_COLS) for f in files), ignore_index=True)
        x["t"] = pd.to_datetime(x["t"], utc=True)
    x["ts"] = (x["t"] - P.EPOCH) // pd.Timedelta(seconds=1)
    return x


# ---------------------------------------------------------------------------
def _local_km(lat, lon, lat0, lon0):
    x = np.radians(np.asarray(lon) - lon0) * P.R_EARTH * math.cos(math.radians(lat0)) / 1000
    y = np.radians(np.asarray(lat) - lat0) * P.R_EARTH / 1000
    return x, y


def panel(ax, rep, grid, title):
    if rep.empty and grid.empty:
        ax.set_title(title + " (κενό)", fontsize=8)
        return
    ref = rep if not rep.empty else grid
    lat0, lon0 = float(ref["lat"].median()), float(ref["lon"].median())
    if not grid.empty:
        for _, g in grid.groupby("seg_id"):
            gx, gy = _local_km(g["lat"], g["lon"], lat0, lon0)
            ax.plot(gx, gy, "-", lw=0.8, color="tab:blue", zorder=2)
            imp = g["GAP_IMP"].to_numpy() == 1
            ax.plot(gx[imp], gy[imp], ".", ms=2, color="tab:cyan", zorder=3)
    if not rep.empty:
        rx, ry = _local_km(rep["lat"], rep["lon"], lat0, lon0)
        ok = ~rep["spike"].to_numpy()
        ax.plot(rx[ok], ry[ok], "o", ms=2.2, mfc="none", color="0.35", zorder=4)
        sp = rep["spike"].to_numpy()
        ax.plot(rx[sp], ry[sp], "x", ms=7, mew=1.6, color="red", zorder=6)
        ct = rep["cut_before"].to_numpy()
        ax.plot(rx[ct], ry[ct], "s", ms=5, mfc="none", mew=1.3, color="orange", zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=7)


def pages(cases, out_dir, stem, per_page=12):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for p in range(0, len(cases), per_page):
        chunk = cases[p:p + per_page]
        n = len(chunk)
        cols = 4
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), squeeze=False)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        for ax, (rep, grid, title) in zip(axes.ravel(), chunk):
            panel(ax, rep, grid, title)
        fig.suptitle(f"{stem}  — γκρι ○ raw αναφορές · μπλε πλέγμα (κυανό: GAP_IMP) · κόκκινο × spike · "
                     f"πορτοκαλί □ τομή από άλμα · άξονες σε km", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = os.path.join(out_dir, f"{stem}_{p // per_page + 1:02d}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths


def window(rep, grid, mmsi, t_center, half_s):
    r = rep[(rep["mmsi"] == mmsi) & (rep["t"].between(t_center - half_s, t_center + half_s))]
    g = grid[(grid["mmsi"] == mmsi) & (grid["ts"].between(t_center - half_s, t_center + half_s))]
    return r, g


def fmt_t(ts):
    return pd.to_datetime(int(ts), unit="s", utc=True).strftime("%d/%m %H:%M")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--month", required=True)
    ap.add_argument("--region", default="socal", choices=sorted(P.REGIONS))
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", default="reports/v5")
    ap.add_argument("--n_spikes", type=int, default=24)
    ap.add_argument("--n_cuts", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    meta = json.load(open(f"{a.build}/build_meta.json", encoding="utf-8"))
    cfg = meta["config"]
    print("[V5] ανάγνωση raw")
    raw = load_raw(a.raw_dir, a.month, cfg)
    print("[V5] καθαρισμός με καταγραφή")
    rep = scan(raw, cfg)
    del raw
    print("[V5] ανάγνωση πλέγματος")
    grid = load_grid(a.build)

    # --- στατιστικά spikes / τομών
    sp, ct = rep[rep["spike"]], rep[rep["cut_before"]]
    print(f"\nspikes: {len(sp):,} σε {sp['mmsi'].nunique()} πλοία | τομές από άλματα: {len(ct):,} σε {ct['mmsi'].nunique()} πλοία")
    for name, d in (("spikes", sp), ("τομές", ct)):
        if d.empty:
            continue
        per = d["mmsi"].value_counts()
        cat = grid.drop_duplicates("mmsi").set_index("mmsi")["vessel_cat"]
        top = per.head(10)
        print(f"  {name}: top-10 πλοία = {top.sum() / len(d):.0%} του συνόλου")
        print("   ", ", ".join(f"{m}({cat.get(m, '?')}):{n}" for m, n in top.items()))
        day = pd.to_datetime(d["t"], unit="s", utc=True).dt.day.value_counts().sort_index()
        print(f"  {name} ανά μέρα: min {day.min()}, median {int(day.median())}, max {day.max()} (μέρα {day.idxmax()})")

    out = a.out
    made = []
    half = 600
    # --- spikes
    if len(sp):
        smp = sp.sample(min(a.n_spikes, len(sp)), random_state=a.seed).sort_values(["mmsi", "t"])
        cases = [(*window(rep, grid, r.mmsi, r.t, half), f"spike {r.mmsi} {fmt_t(r.t)}") for r in smp.itertuples()]
        made += pages(cases, out, "despike")
    # --- τομές
    if len(ct):
        smp = ct.sample(min(a.n_cuts, len(ct)), random_state=a.seed).sort_values(["mmsi", "t"])
        cases = [(*window(rep, grid, r.mmsi, r.t, half), f"τομή {r.mmsi} {fmt_t(r.t)}") for r in smp.itertuples()]
        made += pages(cases, out, "cuts")
    # --- MMSI με QC: η μέρα με τις περισσότερες ανωμαλίες
    cases = []
    for m in meta["funnel"]["mmsi_qc_flagged"]:
        r = rep[rep["mmsi"] == m]
        if r.empty:
            continue
        an = r[r["spike"] | r["cut_before"]]
        day0 = (an if len(an) else r)["t"].floordiv(86400).mode().iloc[0] * 86400
        rr = r[r["t"].between(day0, day0 + 86400)]
        gg = grid[(grid["mmsi"] == m) & grid["ts"].between(day0, day0 + 86400)]
        cases.append((rr, gg, f"MMSI QC {m} {fmt_t(day0)} ({len(an)} ανωμαλίες)"))
    if cases:
        made += pages(cases, out, "mmsiqc")
    # --- τυχαία και ακραία segments (κινούμενα, ≥ 30 λεπτά)
    g2 = grid.sort_values(["seg_id", "ts"])
    dcog = g2.groupby("seg_id")["cog"].diff().abs()
    dcog = np.minimum(dcog, 360 - dcog).where(g2["COG_VALID"] == 1)
    g2["dcog"] = dcog
    seg = g2.groupby("seg_id").agg(n=("ts", "size"), mmsi=("mmsi", "first"), t0=("ts", "min"),
                                   stat=("STATIONARY", "mean"), gap=("GAP_IMP", "mean"),
                                   qc=("QC_SOG", "mean"), turn=("dcog", "mean"))
    seg = seg[(seg["n"] >= 30) & (seg["stat"] < 0.5)]

    def seg_case(sid, label):
        s = seg.loc[sid]
        t0 = int(s["t0"])
        t1 = t0 + min(int(s["n"]) * 60 + 600, 3 * 3600)
        r = rep[(rep["mmsi"] == s["mmsi"]) & rep["t"].between(t0 - 300, t1)]
        g = grid[(grid["seg_id"] == sid) & (grid["ts"] <= t1)]
        return r, g, f"{label} {sid} {fmt_t(t0)} n={int(s['n'])}"

    rnd = seg.sample(min(30, len(seg)), random_state=a.seed).index
    made += pages([seg_case(s, "τυχαίο") for s in rnd], out, "random")
    ext = list(seg["turn"].nlargest(7).index) + list(seg["gap"].nlargest(7).index) + list(seg["qc"].nlargest(6).index)
    labels = ["στροφές"] * 7 + ["GAP_IMP"] * 7 + ["QC_SOG"] * 6
    made += pages([seg_case(s, lab) for s, lab in zip(ext, labels)], out, "extreme")

    print(f"\n[V5] {len(made)} σελίδες στο {out}:")
    for p in made:
        print("  ", p)


if __name__ == "__main__":
    main()
