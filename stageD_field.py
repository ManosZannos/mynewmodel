#!/usr/bin/env python3
"""
stageD_field.py — στάδιο Δ: πολικό πεδίο γύρω από το πλοίο τη στιγμή t0 (docs/phase3_handoff.md §6β).

Για κάθε επιλέξιμο παράθυρο του σταδίου Β, K τομείς ευθυγραμμισμένοι στην πορεία (τομέας 0 = μπροστά, δεξιόστροφα).
Όλες οι τιμές είναι στο [0, 1], 0 = «τίποτα εκεί», και υπολογίζονται μόνο από αναφορές ≤ t0 (R1 του spec §11):

  land_kk        1 − χρόνος/T_max ως το πρώτο εμπόδιο (ακτή, κατασκευή) στη διόπτευση του τομέα, με την τωρινή ταχύτητα
  vt_kk, vd_kk   ο πιο κρίσιμος γείτονας του τομέα: 1 − TCPA/T_max και 1 − DCPA/R (κρίσιμος = μέγιστο γινόμενο)
  hab_kk, hab_n  συνήθεια κίνησης: ποσοστό παραθύρων του train που από αυτό το κελί και με αυτή την πορεία βρέθηκαν
                 στον τομέα 10 λεπτά μετά, χωρίς το ίδιο το πλοίο και χωρίς τα πλοία του early stopping· hab_n = πλήθος
  conf, conf_n   αναδρομικός έλεγχος της συνήθειας στο ίδιο το πλοίο: η συνήθεια στην κατάσταση του πλοίου 10 λεπτά
                 πριν το t0 (θέση, πορεία), μερίδιο στον τομέα (±1) όπου βρέθηκε το πλοίο στο t0· conf_n = το hab_n εκεί.
                 «Ακολουθεί αυτό το πλοίο την κίνηση της περιοχής;» Και οι δύο καταστάσεις είναι εκτιμήσεις του
                 σταδίου στο t0 − 10 και στο t0 (αιτιατές)· 0 / 0 όπου δεν υπάρχει το προηγούμενο παράθυρο
  shore_d, shore_wf   βαθμωτές αποστάσεις του σταδίου Γ (σκέλος σύγκρισης), στο τελευταίο διαθέσιμο σημείο

Πορεία και ταχύτητα του στόχου: το αιτιατό CV του baselines.py (ίδια γωνία με την περιστροφή του phase3.py).
Γείτονες: τελευταία αναφορά ≤ t0 κάθε άλλου πλοίου (P_* της γραμμής του λεπτού t0), προωθημένη στο t0 με SOG/COG.
Ο πίνακας συνήθειας γράφεται στο build του train και διαβάζεται από τα άλλα builds με --prior_build.
Έλεγχος διαρροής: --habit_gap_days D αφαιρεί από τη συνήθεια κάθε παραθύρου τις συνεισφορές όλων των πλοίων από
μέρες (UTC) σε απόσταση ≤ D από τη μέρα του t0 (ταυτόχρονη κίνηση, π.χ. συνοδεία)· γράφεται σε --out_name.

    python stageD_field.py dataset/socal_2025-03_b19c45d7 --osm_dir data/osm/2026-10-29
    python stageD_field.py dataset/socal_2025-12_b19c45d7 --osm_dir data/osm/2026-10-29 --prior_build dataset/socal_2025-03_b19c45d7
    python stageD_field.py dataset/houston_2025-03_7bfeb419 --osm_dir data/osm/2026-10-29      # χωρίς συνήθεια: hab = 0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baselines as BL
import loader as LD
import phase3 as P3

DEFAULTS = {
    "stageD_version": "stageD-v1.2",   # v1.2: conf, conf_n; v1.1: day in the habit table, --habit_gap_days, --out_name
    "K": 16,                  # sectors
    "t_max_min": 20.0,        # twice the prediction horizon
    "radius_m": 5000.0,       # neighbour radius (spec §4.2)
    "land_cap_m": 20000.0,    # ray length = D_SHORE cap of stage C (inside its geometry margin)
    "cell_m": 250.0,          # habit map cell
    "count_cap": 50,          # hab_n saturates here
    "moving_kn": 0.5,         # neighbours below this are stationary (spec §8 default: excluded)
    "include_stationary": False,
    "on_land_tau_m": 25.0,    # same exclusion as stage B
    "chunk": 400_000,
}
KN = 1852.0 / 60.0            # knots -> m/min


# ---------------------------------------------------------------------------
def sector(bearing_deg, course_deg, K):
    """Index of the sector that contains a compass bearing, relative to the course (0 = ahead, clockwise)."""
    return np.rint(((np.asarray(bearing_deg) - course_deg) % 360.0) / (360.0 / K)).astype(np.int64) % K


def closeness(v, vmax):
    return np.clip(1.0 - np.asarray(v, float) / vmax, 0.0, 1.0)


def land_values(dist, speed, cfg):
    """dist (n, K) m along each sector bearing (>= land_cap = nothing), speed (n,) m/min -> closeness in time."""
    t = dist / np.maximum(speed, 1e-6)[:, None]
    return np.where(dist >= cfg["land_cap_m"], 0.0, closeness(t, cfg["t_max_min"]))


def land_distances(lat, lon, course_deg, G, tr, lon0, cfg):
    """(n, K) distance to the first obstacle along the centre bearing of each sector, with the ray caster of stage C.
    shortcut: one ray per sector, so an obstacle narrower than the gap between rays can be missed; cast several
    rays per sector and take the minimum if the land channel turns out noisy."""
    import stageC_shore as SC
    c = {**SC.DEFAULTS, "d_land_cog_cap_m": cfg["land_cap_m"]}
    K, one = cfg["K"], np.ones(len(lat))
    return np.stack([SC.features_latlon(lat, lon, course_deg + k * 360.0 / K, one, G, tr, lon0, c)["D_LAND_COG"]
                     .to_numpy(float) for k in range(K)], axis=1)


def vessel_values(tp, tv, tcourse, tid, cp, cvel, cid, cfg):
    """Targets (m) against candidates (c) of one minute, positions in m and velocities in m/min in one local frame.
    Returns (m, K) closeness in time and in distance of the most critical neighbour of each sector."""
    K, m = cfg["K"], len(tp)
    out_t, out_d = np.zeros((m, K)), np.zeros((m, K))
    if m == 0 or len(cp) == 0:
        return out_t, out_d
    r = cp[None] - tp[:, None]
    rv = cvel[None] - tv[:, None]
    tc = np.clip(-(r * rv).sum(-1) / np.maximum((rv * rv).sum(-1), 1e-9), 0.0, None)   # receding: closest now
    ct = closeness(tc, cfg["t_max_min"])
    cd = closeness(np.linalg.norm(r + tc[..., None] * rv, axis=-1), cfg["radius_m"])
    ok = (np.linalg.norm(r, axis=-1) <= cfg["radius_m"]) & (np.asarray(tid)[:, None] != np.asarray(cid)[None])
    score = np.where(ok, ct * cd, -1.0)
    sec = sector(np.degrees(np.arctan2(r[..., 0], r[..., 1])), np.asarray(tcourse)[:, None], K)
    rows = np.arange(m)
    for k in range(K):
        s = np.where(sec == k, score, -1.0)
        j = s.argmax(1)
        hit = s[rows, j] >= 0
        out_t[hit, k], out_d[hit, k] = ct[rows, j][hit], cd[rows, j][hit]
    return out_t, out_d


def habit_keys(lat, lon, course_deg, cfg):
    """Map cell (about cell_m on each side) and course bin."""
    dlat = np.degrees(cfg["cell_m"] / LD.R_EARTH)
    ci = np.floor(np.asarray(lat) / dlat).astype(np.int64)
    cj = np.floor(np.asarray(lon) * np.cos(np.radians(ci * dlat)) / dlat).astype(np.int64)
    return pd.DataFrame({"ci": ci, "cj": cj, "cb": sector(course_deg, 0.0, cfg["K"])})


def habit_table(keys, out_sec, mmsi, day):
    """Counts per (cell, course bin, sector reached, vessel, UTC day): per vessel, so a window can leave its own
    vessel out; per day, so a leakage check can leave out the traffic of the same days."""
    return keys.assign(sec=np.asarray(out_sec), mmsi=np.asarray(mmsi), day=np.asarray(day)).value_counts().rename("n").reset_index()


def habit_values(tab, keys, mmsi, cfg, day=None, gap_days=None):
    """Habit of each query: counts of the table without its own vessel; with gap_days, also without every vessel's
    counts from days within gap_days of the query's day (own vessel subtracted once)."""
    K, k3 = cfg["K"], ["ci", "cj", "cb"]

    def wide(by):
        return tab.groupby(by + ["sec"])["n"].sum().unstack("sec", fill_value=0).reindex(columns=range(K), fill_value=0)

    def get(q, by):
        return q.join(wide(by), on=by)[list(range(K))].fillna(0).to_numpy(float)
    q = keys.assign(mmsi=np.asarray(mmsi))
    h = get(q, k3) - get(q, k3 + ["mmsi"])
    if gap_days is not None:
        for o in range(-gap_days, gap_days + 1):
            qd = q.assign(day=np.asarray(day) + o)
            h -= get(qd, k3 + ["day"]) - get(qd, k3 + ["mmsi", "day"])
    n = h.sum(1)
    return h / np.maximum(n, 1.0)[:, None], np.minimum(n, cfg["count_cap"]) / cfg["count_cap"]


# ---------------------------------------------------------------------------
def window_state(x, w):
    """Per window, everything known at t0: position (last report moved to t0 with the CV velocity), velocity,
    course; and, for the habit table only, the sector reached 10 minutes later."""
    _, fut, pos, extra = LD.make_arrays(x, w)
    cv, th = P3.course(pos, extra["last_avail"], extra["p_age_s"])
    v = cv[:, 1] - cv[:, 0]                                   # m/min, east/north
    ix = LD.window_rows(x, w)
    t0r = ix[:, LD.OBS - 1]
    age = extra["p_age_s"] / 60.0
    plat, plon = x["P_LAT"].to_numpy(float)[t0r], x["P_LON"].to_numpy(float)[t0r]
    p0 = v * age[:, None]                                     # offset of the t0 position from the last report
    lat = plat + np.degrees(p0[:, 1] / LD.R_EARTH)
    lon = plon + np.degrees(p0[:, 0] / (LD.R_EARTH * np.cos(np.radians(plat))))
    course = (90.0 - np.degrees(th)) % 360.0
    d = fut[:, -1].astype(np.float64) - p0
    s = pd.DataFrame({"window_id": w["window_id"].to_numpy(), "mmsi": w["mmsi"].to_numpy(), "tm0": x["tm"].to_numpy()[t0r],
                      "lat": lat, "lon": lon, "vx": v[:, 0], "vy": v[:, 1], "speed": np.linalg.norm(v, axis=1),
                      "course": course})
    s["out_sec"] = sector(np.degrees(np.arctan2(d[:, 0], d[:, 1])), course, DEFAULTS["K"])
    s["row_last"] = ix[np.arange(len(w)), extra["last_avail"]]
    return s


def prev_state(x, w, shift=LD.PRED):
    """window_state of the same vessel shift minutes earlier (the habit horizon), where that window exists: same
    segment, >= 4 available observation points. Returns (rows of w that have it, their earlier states)."""
    r0 = w["row0"].to_numpy() - shift
    tm, mm, sg = x["tm"].to_numpy(), x["mmsi"].to_numpy(), x["seg_id"].to_numpy()
    end = r0 + LD.OBS + LD.PRED - 1
    ok = r0 >= 0
    ok[ok] = ((tm[r0[ok]] == w["t_start"].to_numpy()[ok] - shift) & (mm[r0[ok]] == mm[end[ok]])
              & (sg[r0[ok]] == sg[end[ok]]) & (mm[r0[ok]] == w["mmsi"].to_numpy()[ok]))
    idx = np.flatnonzero(ok)
    wp = pd.DataFrame({"window_id": w["window_id"].to_numpy()[idx], "mmsi": w["mmsi"].to_numpy()[idx],
                       "seg_id": sg[r0[idx]], "row0": r0[idx], "t_start": w["t_start"].to_numpy()[idx] - shift})
    keep = LD.available_mask(x, LD.window_rows(x, wp)).sum(1) >= 4
    return idx[keep], window_state(x, wp[keep].reset_index(drop=True))


def conform_values(tab, s, idx, sp, cfg, gap_days=None):
    """conf, conf_n per window (0 where idx has no earlier state): the habit at the earlier state, summed over the
    sector where the vessel was at t0 and its two neighbours (sector edges are noisy at 22.5 degrees)."""
    K, n = cfg["K"], len(s)
    lat, lon = s["lat"].to_numpy()[idx], s["lon"].to_numpy()[idx]
    plat, plon = sp["lat"].to_numpy(), sp["lon"].to_numpy()
    east = np.radians(lon - plon) * LD.R_EARTH * np.cos(np.radians(plat))
    north = np.radians(lat - plat) * LD.R_EARTH
    sec = sector(np.degrees(np.arctan2(east, north)), sp["course"].to_numpy(), K)
    h, hn = habit_values(tab, habit_keys(plat, plon, sp["course"].to_numpy(), cfg), sp["mmsi"].to_numpy(), cfg,
                         sp["tm0"].to_numpy() // 1440, gap_days)
    r = np.arange(len(idx))
    conf, conf_n = np.zeros(n), np.zeros(n)
    conf[idx] = h[r, sec] + h[r, (sec + 1) % K] + h[r, (sec - 1) % K]
    conf_n[idx] = hn
    return conf, conf_n


def vessels_channel(x, s, cfg):
    K = cfg["K"]
    vt, vd = np.zeros((len(s), K)), np.zeros((len(s), K))
    for g, *pair in minute_pairs(x, s, cfg):
        vt[g], vd[g] = vessel_values(*pair, cfg)
    return vt, vd


def minute_pairs(x, s, cfg):
    """For each minute that has targets, yield their rows in s and (tp, tv, tcourse, tid, cp, cvel, cid):
    targets and candidates (every vessel with a row at that minute) in one local frame, m and m/min."""
    c = x[["mmsi", "tm", "P_LAT", "P_LON", "P_SOG", "P_COG", "P_AGE_S"]].dropna()
    if "ON_LAND_DEPTH_M" in x:
        c = c[x.loc[c.index, "ON_LAND_DEPTH_M"].to_numpy() <= cfg["on_land_tau_m"]]
    if not cfg["include_stationary"]:
        c = c[c["P_SOG"] >= cfg["moving_kn"]]
    c = c.sort_values("tm", kind="mergesort")
    ctm = c["tm"].to_numpy()
    ca = {k: c[k].to_numpy(float) for k in ("P_LAT", "P_LON", "P_SOG", "P_COG", "P_AGE_S")}
    cid = c["mmsi"].to_numpy()
    for tm0, g in s.groupby("tm0").indices.items():
        a, b = np.searchsorted(ctm, [tm0, tm0 + 1])
        if b == a:
            continue
        lat0 = s["lat"].to_numpy()[g].mean()
        kx = LD.R_EARTH * np.cos(np.radians(lat0))             # one projection per minute: ~1% over the box, far less over 5 km

        def xy(lat, lon):
            return np.column_stack([np.radians(lon) * kx, np.radians(lat) * LD.R_EARTH])
        cr = np.radians(ca["P_COG"][a:b])
        cvel = (ca["P_SOG"][a:b] * KN)[:, None] * np.column_stack([np.sin(cr), np.cos(cr)])
        cp = xy(ca["P_LAT"][a:b], ca["P_LON"][a:b]) + cvel * (ca["P_AGE_S"][a:b] / 60.0)[:, None]
        t = s.iloc[g]
        yield (g, xy(t["lat"].to_numpy(), t["lon"].to_numpy()), t[["vx", "vy"]].to_numpy(),
               t["course"].to_numpy(), t["mmsi"].to_numpy(), cp, cvel, cid[a:b])


def geometry(x, osm_dir, cfg):
    """The prepared geometry of stage C for this build (same box, projection and cache file)."""
    import stageC_shore as SC
    sc = json.loads(json.dumps(SC.DEFAULTS))
    bb = [x["lon"].min(), x["lat"].min(), x["lon"].max(), x["lat"].max()]
    core_bb, geom_bb = SC.geo_bbox(bb, sc)
    cache = Path(osm_dir) / f"prepared_{SC.cache_key(core_bb, sc, osm_dir)}.parquet"
    G, tr, (_, lon0), _ = SC.prepare_geometry(osm_dir, core_bb, geom_bb, sc, cache)
    return G, tr, lon0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_dir")
    ap.add_argument("--osm_dir", default=None, help="data/osm/<YYYY-MM-DD>, όπως στο στάδιο Γ")
    ap.add_argument("--skip_land", action="store_true", help="μόνο για tests: κανάλι ξηράς = 0 (γράφεται στο meta)")
    ap.add_argument("--prior_build", default=None, help="build με πίνακα συνήθειας (το build του train)")
    ap.add_argument("--manifest", default="vessel", help="manifest του οποίου το train δίνει τον πίνακα συνήθειας")
    ap.add_argument("--habit_gap_days", type=int, default=None,
                    help="έλεγχος διαρροής: χωρίς τις συνεισφορές των ημερών σε απόσταση ≤ D από τη μέρα του παραθύρου")
    ap.add_argument("--out_name", default="stageD", help="φάκελος εξόδου μέσα στο build (π.χ. stageD_gap1)")
    a = ap.parse_args(argv)
    if a.habit_gap_days is not None and a.out_name == "stageD":
        sys.exit("--habit_gap_days θέλει άλλο --out_name, ώστε να μένει το κανονικό πεδίο")
    if not a.osm_dir and not a.skip_land:
        sys.exit("χρειάζεται --osm_dir (ή --skip_land για tests)")
    cfg, t_start = dict(DEFAULTS), time.time()
    K, sb = cfg["K"], Path(a.build_dir) / "stageB"
    out = Path(a.build_dir) / a.out_name
    out.mkdir(parents=True, exist_ok=True)
    shore_dir = BL.shore_dir_of(a.build_dir)
    print(f"[stageD] {a.build_dir} | στάδιο Γ: {shore_dir}")
    x = LD.load_rows(a.build_dir, shore_dir)
    w = LD._read_tab(sb / "windows.parquet")
    w = w[w["eligible"] == 1].sort_values("window_id").reset_index(drop=True)
    s = window_state(x, w)
    n = len(s)
    print(f"  επιλέξιμα παράθυρα: {n:,}")
    p_idx, sp = prev_state(x, w)                     # early, before the long land step
    print(f"  με κατάσταση {LD.PRED} λεπτά πριν (για το conf): {len(p_idx) / max(n, 1):.1%}")

    land = np.zeros((n, K))
    if not a.skip_land:
        G, tr, lon0 = geometry(x, a.osm_dir, cfg)
        for i in range(0, n, cfg["chunk"]):
            c = s.iloc[i:i + cfg["chunk"]]
            d = land_distances(c["lat"].to_numpy(), c["lon"].to_numpy(), c["course"].to_numpy(), G, tr, lon0, cfg)
            land[i:i + cfg["chunk"]] = land_values(d, c["speed"].to_numpy(), cfg)
        print(f"  ξηρά: {(land > 0).any(1).mean():.1%} των παραθύρων με εμπόδιο σε T_max | {time.time() - t_start:.0f} s")

    vt, vd = vessels_channel(x, s, cfg)
    print(f"  πλοία: {(vd > 0).any(1).mean():.1%} των παραθύρων με γείτονα σε {cfg['radius_m'] / 1000:.0f} km | {time.time() - t_start:.0f} s")

    keys, tab, prior = habit_keys(s["lat"], s["lon"], s["course"], cfg), None, None
    day = s["tm0"].to_numpy() // 1440                          # UTC day of t0
    tpath = sb / f"targets_{a.manifest}.parquet"
    t = LD._read_tab(tpath) if (tpath.exists() or tpath.with_name(tpath.stem + ".csv.gz").exists()) else None
    if a.prior_build:
        tab, prior = pd.read_parquet(Path(a.prior_build) / "stageD" / "habit_table.parquet"), a.prior_build
    elif t is not None and (t["split"] == "train").any():
        tr_ids = set(t.loc[t["split"] == "train", "window_id"])
        fit = s["window_id"].isin(tr_ids).to_numpy() & (BL._hash01(s["mmsi"].to_numpy()) >= 0.1)   # not the early-stopping vessels
        tab, prior = habit_table(keys[fit], s["out_sec"].to_numpy()[fit], s["mmsi"].to_numpy()[fit], day[fit]), a.build_dir
        tab.to_parquet(out / "habit_table.parquet", index=False)
    conf, conf_n = np.zeros(n), np.zeros(n)
    if tab is not None:
        hab, hab_n = habit_values(tab, keys, s["mmsi"].to_numpy(), cfg, day, a.habit_gap_days)
        conf, conf_n = conform_values(tab, s, p_idx, sp, cfg, a.habit_gap_days)
        print(f"  conf: {(conf_n > 0).mean():.1%} των παραθύρων με ιστορικό στην προηγούμενη κατάσταση, "
              f"διάμεσο {np.median(conf[conf_n > 0]) if (conf_n > 0).any() else float('nan'):.2f}")
        print(f"  συνήθεια (από {prior}): {(hab_n > 0).mean():.1%} των παραθύρων με ιστορικό | {time.time() - t_start:.0f} s")
    else:
        hab, hab_n = np.zeros((n, K)), np.zeros(n)
        print("  συνήθεια: κανένα train και κανένα --prior_build → hab = 0")

    f = pd.DataFrame({"window_id": s["window_id"].to_numpy()})
    for name, arr in (("land", land), ("vt", vt), ("vd", vd), ("hab", hab)):
        for k in range(K):
            f[f"{name}_{k:02d}"] = arr[:, k].astype(np.float32)
    f["hab_n"] = hab_n.astype(np.float32)
    f["conf"], f["conf_n"] = np.clip(conf, 0, 1).astype(np.float32), conf_n.astype(np.float32)
    if "D_SHORE" in x:
        rl = s["row_last"].to_numpy()
        f["shore_d"] = np.clip(x["D_SHORE"].to_numpy(float)[rl] / cfg["land_cap_m"], 0, 1).astype(np.float32)
        f["shore_wf"] = np.clip(x["D_WATERFRONT"].to_numpy(float)[rl] / 5000.0, 0, 1).astype(np.float32)
    vals = f.drop(columns="window_id").to_numpy()
    if len(f) != len(w) or not np.isfinite(vals).all() or vals.min() < 0 or vals.max() > 1:
        raise AssertionError("το πεδίο δεν κλείνει: γραμμές ≠ επιλέξιμα παράθυρα ή τιμές εκτός [0, 1]")
    f.to_parquet(out / "field.parquet", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        commit = None
    meta = {"config": cfg, "commit": commit, "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                                                         "pandas": pd.__version__, "loader": LD.LOADER_VERSION,
                                                         "phase3": P3.PHASE3_VERSION},
            "shore_dir": shore_dir, "osm_dir": a.osm_dir, "skip_land": a.skip_land, "habit_prior": prior,
            "habit_gap_days": a.habit_gap_days, "out_name": a.out_name,
            "funnel": {"eligible_windows": int(len(w)), "rows_out": int(len(f))},
            "summary": {"land_any": float((land > 0).any(1).mean()), "vessels_any": float((vd > 0).any(1).mean()),
                        "habit_any": float((hab_n > 0).mean()), "conf_any": float((conf_n > 0).mean()), "seconds": round(time.time() - t_start, 1)}}
    (out / "stageD_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[stageD] {len(f):,} παράθυρα (= επιλέξιμα ✓) → {out / 'field.parquet'} | {meta['summary']['seconds']:.0f} s")
    return meta


if __name__ == "__main__":
    main()
