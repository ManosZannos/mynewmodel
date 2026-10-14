#!/usr/bin/env python3
"""
v3_holdout.py — V3 της προδιαγραφής: hold-out για τη μέθοδο παρεμβολής και το B.

Αφαιρείται ΜΙΑ αναφορά τη φορά (σε κινούμενα πλοία, μέσα στο ίδιο segment μετά τον
καθαρισμό του σταδίου Α), ανακατασκευάζεται η θέση της από τις γειτονικές, και το
σφάλμα μετριέται ανά bin του κενού που δημιουργείται (t[i+1] − t[i−1]).

Μέθοδοι:
  linear    γραμμική σε τοπικό επίπεδο (τρέχουσα του pipeline)
  hermite   κυβική Hermite με τις ταχύτητες SOG/COG των άκρων (κινηματική παρεμβολή)
  cubic4    κυβική Lagrange από 4 σημεία (i−2, i−1, i+1, i+2)

    python v3_holdout.py --raw_dir data/2025_03 --month 2025-03 --region socal --days 1 8 15 22

Η παρεμβολή δεν εξαρτάται από κανένα μοντέλο ή split, οπότε μπορεί να χρησιμοποιεί
οποιεσδήποτε μέρες χωρίς διαρροή.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

import preprocess_noaa as P

BINS = [60, 120, 180, 240, 360, 600]


def clean_reports(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Ίδια βήματα με το run_stage_a μέχρι και τη segmentation, χωρίς resampling."""
    df = P.standardize(raw)
    df = df[df["t"].notna() & df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
    df = df[P.box_with_margin(df, cfg["region"], cfg["margin_km"])]
    df = df[P.valid_mmsi(df["mmsi"])].copy()
    df["mmsi"] = df["mmsi"].astype(np.int64)
    df.loc[(df["heading"] >= 360) | (df["heading"] < 0), "heading"] = np.nan
    df.loc[(df["cog"] >= 360) | (df["cog"] < 0), "cog"] = np.nan
    df.loc[(df["sog"] >= 102.2) | (df["sog"] < 0), "sog"] = np.nan
    vt = df.loc[df["vessel_type"] > 0].groupby("mmsi")["vessel_type"].agg(P._mode)
    df["vessel_cat"] = df["mmsi"].map(vt.map(P.vessel_category)).fillna("unknown")
    trx = df.groupby("mmsi")["transceiver"].agg(P._mode)
    df["trx"] = df["mmsi"].map(trx)
    caps = cfg["speed_caps_kn"]
    df["cap"] = df["vessel_cat"].map(lambda c: caps.get(c, caps["global"])).astype(float)
    df.loc[df["sog"] > df["cap"], "sog"] = np.nan
    df["_n"] = df[["sog", "cog", "heading"]].notna().sum(axis=1)
    df = df.sort_values(["mmsi", "t", "_n"], ascending=[True, True, False], kind="mergesort")
    df = df[~df.duplicated(["mmsi", "t"], keep="first")]

    out = []
    for mmsi, g in df.groupby("mmsi", sort=True):
        t = ((g["t"] - P.EPOCH) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
        lat, lon = g["lat"].to_numpy(float), g["lon"].to_numpy(float)
        sog, cog = g["sog"].to_numpy(float), g["cog"].to_numpy(float)
        cap = float(g["cap"].iloc[0])
        keep, _ = P.despike_vessel(lat, lon, t, sog, cog, cap, cfg)
        t, lat, lon, sog, cog = t[keep], lat[keep], lon[keep], sog[keep], cog[keep]
        seg, _ = P.segment_vessel(lat, lon, t, sog, cap, cfg)
        out.append(pd.DataFrame({"mmsi": mmsi, "seg": seg, "t": t, "lat": lat, "lon": lon,
                                 "sog": sog, "cog": cog, "trx": g["trx"].iloc[0],
                                 "cat": g["vessel_cat"].iloc[0]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def holdout(rep: pd.DataFrame, min_sog: float) -> pd.DataFrame:
    rep = rep.sort_values(["mmsi", "seg", "t"], kind="mergesort").reset_index(drop=True)
    key = rep["mmsi"].astype(str) + "_" + rep["seg"].astype(str)
    same = lambda k: key.shift(k) == key  # noqa: E731
    ok = same(1) & same(-1) & same(2) & same(-2)
    lat0 = rep["lat"].to_numpy()
    lon0 = rep["lon"].to_numpy()
    R, KN = P.R_EARTH, P.KN
    coslat = np.cos(np.radians(lat0))

    def xy(k):  # θέση του σημείου i+k στο τοπικό επίπεδο με αρχή το i (θέση αλήθειας)
        la, lo = rep["lat"].shift(-k).to_numpy(), rep["lon"].shift(-k).to_numpy()
        return np.radians(lo - lon0) * R * coslat, np.radians(la - lat0) * R

    t = rep["t"].to_numpy(float)
    tm1, tp1 = rep["t"].shift(1).to_numpy(float), rep["t"].shift(-1).to_numpy(float)
    tm2, tp2 = rep["t"].shift(2).to_numpy(float), rep["t"].shift(-2).to_numpy(float)
    xm1, ym1 = xy(-1)
    xp1, yp1 = xy(1)
    xm2, ym2 = xy(-2)
    xp2, yp2 = xy(2)
    sm1, sp1 = rep["sog"].shift(1).to_numpy(), rep["sog"].shift(-1).to_numpy()
    cm1, cp1 = rep["cog"].shift(1).to_numpy(), rep["cog"].shift(-1).to_numpy()
    s_i = rep["sog"].to_numpy()

    gap = tp1 - tm1
    a = (t - tm1) / gap
    moving = (sm1 >= min_sog) & (sp1 >= min_sog) & (s_i >= min_sog)
    sel = ok.to_numpy() & moving & (gap > 0)

    # linear
    ex = (1 - a) * xm1 + a * xp1
    ey = (1 - a) * ym1 + a * yp1
    e_lin = np.hypot(ex, ey)  # η αλήθεια είναι η αρχή (0, 0)

    # hermite με ταχύτητες SOG/COG
    vxm, vym = sm1 * KN * np.sin(np.radians(cm1)), sm1 * KN * np.cos(np.radians(cm1))
    vxp, vyp = sp1 * KN * np.sin(np.radians(cp1)), sp1 * KN * np.cos(np.radians(cp1))
    h00, h10 = 2 * a**3 - 3 * a**2 + 1, a**3 - 2 * a**2 + a
    h01, h11 = -2 * a**3 + 3 * a**2, a**3 - a**2
    hx = h00 * xm1 + h10 * gap * vxm + h01 * xp1 + h11 * gap * vxp
    hy = h00 * ym1 + h10 * gap * vym + h01 * yp1 + h11 * gap * vyp
    hv = ~np.isnan(hx) & ~np.isnan(hy)
    e_her = np.where(hv, np.hypot(hx, hy), e_lin)

    # cubic Lagrange από 4 σημεία
    ts = [tm2, tm1, tp1, tp2]
    xs, ys = [xm2, xm1, xp1, xp2], [ym2, ym1, yp1, yp2]
    cx = np.zeros_like(t)
    cy = np.zeros_like(t)
    for j in range(4):
        L = np.ones_like(t)
        for k in range(4):
            if k != j:
                L *= (t - ts[k]) / (ts[j] - ts[k])
        cx += L * xs[j]
        cy += L * ys[j]
    e_cub = np.hypot(cx, cy)

    turn = P._ang_diff(cm1, cp1)
    res = pd.DataFrame({"gap": gap, "linear": e_lin, "hermite": e_her, "cubic4": e_cub,
                        "hermite_valid": hv, "trx": rep["trx"], "cat": rep["cat"],
                        "turn": turn, "sog": s_i})[sel]
    return res


def summarize(res: pd.DataFrame, by: str | None = None):
    res = res.copy()
    res["bin"] = pd.cut(res["gap"], BINS, right=True)
    keys = ["bin"] + ([by] if by else [])
    rows = []
    for k, g in res.groupby(keys, observed=True):
        k = k if isinstance(k, tuple) else (k,)
        row = {"bin": str(k[0])} | ({by: k[1]} if by else {}) | {"n": len(g)}
        for m in ("linear", "hermite", "cubic4"):
            row[f"{m} p50"] = round(float(g[m].median()), 1)
            row[f"{m} p90"] = round(float(g[m].quantile(0.9)), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--month", required=True)
    ap.add_argument("--region", default="socal", choices=sorted(P.REGIONS))
    ap.add_argument("--days", type=int, nargs="+", default=[1, 8, 15, 22])
    ap.add_argument("--min_sog", type=float, default=3.0)
    ap.add_argument("--limit_m", type=float, default=50.0,
                    help="όριο p90 σφάλματος για την επιλογή του B (προσωρινό κριτήριο)")
    a = ap.parse_args()

    cfg = P.load_config()
    cfg["region"] = P.REGIONS[a.region]
    files = [os.path.join(a.raw_dir, f"ais-{a.month}-{d:02d}.csv.zst") for d in a.days]
    files = [f for f in files if os.path.exists(f)] or sorted(glob.glob(os.path.join(a.raw_dir, f"ais-{a.month}-*.csv.zst")))[:4]
    raw = pd.concat((pd.read_csv(f, usecols=lambda c: c in P.NOAA_COLS, low_memory=False) for f in files),
                    ignore_index=True)
    print(f"[V3] {len(files)} μέρες, {len(raw):,} γραμμές")
    rep = clean_reports(raw, cfg)
    res = holdout(rep, a.min_sog)
    print(f"[V3] hold-out σε {len(res):,} αναφορές κινούμενων πλοίων (SOG ≥ {a.min_sog} kn)")
    print(f"     hermite με έγκυρες ταχύτητες: {res['hermite_valid'].mean():.1%}\n")

    pd.set_option("display.width", 200)
    print("ΣΦΑΛΜΑ ΘΕΣΗΣ (m) ανά bin κενού — όλα")
    s = summarize(res)
    print(s.to_string(index=False))
    print("\nανά class A/B")
    print(summarize(res, "trx").to_string(index=False))
    res["turn_bin"] = pd.cut(res["turn"], [-0.1, 10, 45, 180], labels=["<10°", "10–45°", ">45°"])
    print("\nανά στροφή (|ΔCOG| ανάμεσα στις γειτονικές αναφορές)")
    print(summarize(res, "turn_bin").to_string(index=False))

    best = s[["linear p90", "hermite p90", "cubic4 p90"]].mean().idxmin().replace(" p90", "")
    ok = s[s[f"{best} p90"] <= a.limit_m]
    edge = ok["bin"].iloc[-1] if len(ok) else "κανένα"
    print(f"\nΚαλύτερη μέθοδος (μέσο p90 στα bins): {best}")
    print(f"Μεγαλύτερο bin με p90 ≤ {a.limit_m:.0f} m για {best}: {edge}  (προσωρινό κριτήριο για το B)")


if __name__ == "__main__":
    main()
