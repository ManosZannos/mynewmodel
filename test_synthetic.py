#!/usr/bin/env python3
"""
test_synthetic.py — V1 της προδιαγραφής preprocessing (NOAA 2025).

Συνθετικές τροχιές με ΓΝΩΣΤΗ αλήθεια, «δειγματοληψία» τύπου AIS, πέρασμα από το
στάδιο Α και σύγκριση με την αλήθεια. Ένα test ανά κανόνα A2–A12 της προδιαγραφής.

Τρέξιμο (χωρίς pytest):
    python test_synthetic.py              # όλα τα tests
    python test_synthetic.py -k spike     # μόνο όσα περιέχουν "spike"
    python test_synthetic.py --selfcheck  # έλεγχος μόνο της γεννήτριας (χωρίς pipeline)
Με pytest:
    pytest -q test_synthetic.py

---------------------------------------------------------------------------
ΣΥΜΒΟΛΑΙΟ ΜΕ ΤΟ PIPELINE (το υλοποιεί το preprocess_noaa.py)
---------------------------------------------------------------------------
    from preprocess_noaa import run_stage_a, load_config
    cfg = load_config()                       # προεπιλογές της προδιαγραφής
    out, funnel = run_stage_a(raw, cfg)

raw: DataFrame στο σχήμα NOAA 2025 (mmsi, base_date_time, longitude, latitude,
     sog, cog, heading, vessel_type, status, transceiver). Όλος ο μήνας μαζί.

cfg (κλειδιά που διαβάζουν τα tests):
    region = {lat_min, lat_max, lon_min, lon_max}, margin_km
    grid_s, seg_gap_s, bridge_max_s, gap_imp_flag_s, min_segment_s
    speed_caps_kn = {global, cargo, tanker, passenger, fishing, tug}
    sog_interp_max_diff_kn, cog_valid_min_sog_kn, stationary_sog_kn
    qc_sog_rel, qc_sog_min_kn, mmsi_qc_max_anomaly_frac

out: μία γραμμή ανά (mmsi, t) στο πλέγμα. Στήλες: βλ. REQUIRED_COLS.
     t = datetime64[ns, UTC], ακέραιο λεπτό.
funnel: dict με
    raw_rows      (int)  γραμμές εισόδου
    kept_reports  (int)  αναφορές που έφτασαν στο resampling
    dropped       (dict) κανόνας -> γραμμές που αφαιρέθηκαν
                         (κλειδιά: invalid_mmsi, invalid_position, outside_box,
                          duplicate, despike, short_segment, ...)
    nulled        (dict) κανόνας -> τιμές που έγιναν NaN (π.χ. sog_implausible)
    mmsi_qc_flagged (list) MMSI με ποσοστό ανωμαλιών πάνω από το όριο
    raw_rows == kept_reports + sum(dropped.values())   (το funnel κλείνει)
"""
from __future__ import annotations

import inspect
import math
import sys
import traceback

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Σταθερές
# ---------------------------------------------------------------------------
R_EARTH = 6_371_000.0
KN = 0.514444  # m/s ανά κόμβο
REF_LAT, REF_LON = 33.30, -118.60   # μέσα στο κουτί Southern California, ανοιχτά
T0 = pd.Timestamp("2025-03-10 08:00:00", tz="UTC")

REQUIRED_COLS = [
    "mmsi", "t", "seg_id", "lat", "lon", "sog", "cog", "COG_VALID",
    "heading", "HDG_SRC", "vessel_cat", "transceiver", "GAP_IMP", "INTERP_SPAN",
    "STATIONARY", "QC_SOG", "QC_COG", "IN_CORE",
    "vx_sog", "vy_sog", "vx_pos", "vy_pos",
]


# ---------------------------------------------------------------------------
# Γεωμετρία
# ---------------------------------------------------------------------------
def xy_to_latlon(x, y, ref_lat=REF_LAT, ref_lon=REF_LON):
    lat = ref_lat + np.degrees(np.asarray(y) / R_EARTH)
    lon = ref_lon + np.degrees(np.asarray(x) / (R_EARTH * math.cos(math.radians(ref_lat))))
    return lat, lon


def latlon_to_xy(lat, lon, ref_lat=REF_LAT, ref_lon=REF_LON):
    x = np.radians(np.asarray(lon) - ref_lon) * R_EARTH * math.cos(math.radians(ref_lat))
    y = np.radians(np.asarray(lat) - ref_lat) * R_EARTH
    return x, y


def ang_diff(a, b):
    """Απόλυτη κυκλική διαφορά σε μοίρες."""
    d = (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0
    return np.abs(d)


# ---------------------------------------------------------------------------
# Γεννήτρια τροχιών με γνωστή αλήθεια
# ---------------------------------------------------------------------------
class Track:
    """Συνεχής τροχιά: ταχύτητα(t) [kn] και πορεία(t) [μοίρες, 0 = βορράς],
    ολοκληρωμένη με βήμα 1 s (σφάλμα ολοκλήρωσης αμελητέο)."""

    def __init__(self, dur_s, speed_kn, course_deg, x0=0.0, y0=0.0, t0=T0):
        self.t0 = t0
        self.ts = np.arange(0, int(dur_s) + 1, 1.0)
        sp = np.array([speed_kn(t) for t in self.ts], float) * KN
        co = np.radians([course_deg(t) for t in self.ts])
        vx, vy = sp * np.sin(co), sp * np.cos(co)
        # μέσο σημείο (trapezoid) για ακρίβεια
        x = x0 + np.concatenate([[0], np.cumsum((vx[1:] + vx[:-1]) / 2)])
        y = y0 + np.concatenate([[0], np.cumsum((vy[1:] + vy[:-1]) / 2)])
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.speed_kn_arr = sp / KN
        self.course_arr = np.degrees(co) % 360

    def at(self, t_s):
        t_s = np.asarray(t_s, float)
        return (np.interp(t_s, self.ts, self.x), np.interp(t_s, self.ts, self.y))

    def state(self, t_s):
        t_s = np.asarray(t_s, float)
        sog = np.interp(t_s, self.ts, self.speed_kn_arr)
        vx = np.interp(t_s, self.ts, self.vx)
        vy = np.interp(t_s, self.ts, self.vy)
        cog = np.degrees(np.arctan2(vx, vy)) % 360
        return sog, cog, vx, vy


def report_times(dur_s, lo=60, hi=80, phase=None, rng=None, gaps=()):
    """Χρόνοι αναφορών (s από t0). gaps: λίστα (αρχή, τέλος) χωρίς αναφορές."""
    rng = rng or np.random.default_rng(0)
    t = float(rng.uniform(0, 60) if phase is None else phase)
    out = []
    while t <= dur_s:
        if not any(a < t < b for a, b in gaps):
            out.append(t)
        t += float(rng.uniform(lo, hi)) if hi > lo else float(lo)
    return np.round(np.array(out))   # οι χρονοσφραγίδες AIS είναι ακέραια δευτερόλεπτα


def make_reports(track, times, mmsi=367000001, vessel_type=70, transceiver="A",
                 pos_noise_m=0.0, rng=None, heading_from_cog=True):
    rng = rng or np.random.default_rng(1)
    x, y = track.at(times)
    if pos_noise_m > 0:
        x = x + rng.normal(0, pos_noise_m, len(x))
        y = y + rng.normal(0, pos_noise_m, len(y))
    lat, lon = xy_to_latlon(x, y)
    sog, cog, _, _ = track.state(times)
    ts = [track.t0 + pd.Timedelta(seconds=float(s)) for s in times]
    return pd.DataFrame({
        "mmsi": mmsi,
        "base_date_time": [t.strftime("%Y-%m-%d %H:%M:%S") for t in ts],
        "longitude": lon,
        "latitude": lat,
        "sog": np.round(sog, 1),
        "cog": np.round(cog, 1),
        "heading": np.round(cog) if heading_from_cog and transceiver == "A" else np.nan,
        "vessel_type": vessel_type,
        "status": np.nan,
        "transceiver": transceiver,
        "_t_s": times,  # βοηθητική, το pipeline πρέπει να την αγνοεί
    })


def raw_only(df):
    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


def truth_on_grid(track, out_rows):
    """Αληθινή θέση (x, y) στις στιγμές του πλέγματος του out."""
    t_s = (out_rows["t"] - track.t0).dt.total_seconds().to_numpy()
    return track.at(t_s)


def pos_error_m(track, out_rows):
    tx, ty = truth_on_grid(track, out_rows)
    ox, oy = latlon_to_xy(out_rows["lat"].to_numpy(), out_rows["lon"].to_numpy())
    return np.hypot(ox - tx, oy - ty)


def straight(dur_s=3600, kn=10.0, course=45.0, **kw):
    return Track(dur_s, lambda t: kn, lambda t: course, **kw)


# ---------------------------------------------------------------------------
# Βοηθητικά για τα tests
# ---------------------------------------------------------------------------
_PIPE = None


def pipeline():
    global _PIPE
    if _PIPE is None:
        try:
            import preprocess_noaa as p
        except ImportError as e:
            raise RuntimeError(
                "Δεν βρέθηκε το preprocess_noaa.py (run_stage_a, load_config). "
                "Αυτό είναι αναμενόμενο πριν γραφτεί το pipeline."
            ) from e
        _PIPE = p
    return _PIPE


def run(raw, **overrides):
    p = pipeline()
    cfg = p.load_config()
    for k, v in overrides.items():
        cfg[k] = v
    out, funnel = p.run_stage_a(raw_only(raw), cfg)
    out = out.sort_values(["mmsi", "t"]).reset_index(drop=True)
    return out, funnel, cfg


def cfg_default():
    return pipeline().load_config()


def minutes_strictly_between(t0, a_s, b_s):
    a, b = t0 + pd.Timedelta(seconds=a_s), t0 + pd.Timedelta(seconds=b_s)
    first = a.ceil("min")
    if first == a:
        first += pd.Timedelta(minutes=1)
    return [m for m in pd.date_range(first, b, freq="min") if m < b]


# ---------------------------------------------------------------------------
# Tests: σχήμα και πλέγμα
# ---------------------------------------------------------------------------
def test_schema_and_grid():
    tr = straight(1800)
    raw = make_reports(tr, report_times(1800))
    out, funnel, _ = run(raw)
    missing = [c for c in REQUIRED_COLS if c not in out.columns]
    assert not missing, f"λείπουν στήλες: {missing}"
    assert str(out["t"].dt.tz) == "UTC", "το t πρέπει να είναι UTC"
    assert (out["t"].dt.second == 0).all(), "το t πρέπει να είναι ακέραιο λεπτό"
    assert not out.duplicated(["mmsi", "t"]).any(), "διπλό (mmsi, t)"
    assert out[["lat", "lon"]].notna().all().all(), "NaN σε θέση"
    for k in ("raw_rows", "kept_reports", "dropped", "nulled", "mmsi_qc_flagged"):
        assert k in funnel, f"λείπει το funnel['{k}']"


# ---------------------------------------------------------------------------
# A8: παρεμβολή
# ---------------------------------------------------------------------------
def test_straight_line_exact():
    tr = straight(7200, kn=10, course=45)
    times = report_times(7200, 60, 80)
    out, _, _ = run(make_reports(tr, times))
    err = pos_error_m(tr, out)
    assert err.max() < 1.0, f"ευθεία: μέγιστο σφάλμα {err.max():.2f} m (όριο 1 m)"
    expected = len(minutes_strictly_between(T0, times[0] - 1e-6, times[-1] + 1e-6))
    assert abs(len(out) - expected) <= 1, f"{len(out)} γραμμές, αναμένονταν ~{expected}"
    assert (out["GAP_IMP"] == 0).all()
    assert ang_diff(out["cog"], 45).max() < 1.0
    assert (out["sog"] - 10).abs().max() < 0.15


def test_turn_error_within_sagitta():
    # στροφή 90° σε 2 λεπτά στα 10 kn, ανάμεσα σε ευθείες
    omega = 90 / 120
    course = lambda t: 0 if t < 1200 else (min(90, (t - 1200) * omega))
    tr = Track(2400, lambda t: 10, course)
    times = report_times(2400, 60, 80)
    out, funnel, _ = run(make_reports(tr, times))
    dt_max = np.diff(times).max()
    radius = 10 * KN / math.radians(omega)
    sagitta = radius * (1 - math.cos(math.radians(omega * dt_max) / 2))
    err = pos_error_m(tr, out)
    assert err.max() <= sagitta * 1.05 + 1.0, \
        f"στροφή: σφάλμα {err.max():.1f} m > όριο {sagitta:.1f} m"
    assert funnel["dropped"].get("despike", 0) == 0, "η στροφή δεν πρέπει να θεωρηθεί spike"
    assert out["seg_id"].nunique() == 1, "η στροφή δεν πρέπει να κόψει το segment"


def test_tight_turn_recreational_not_cut():
    # σκάφος αναψυχής: 180° σε 60 s στα 6 kn, class B
    for lo, hi in [(30, 30), (60, 60)]:
        course = lambda t: 90 if t < 900 else (90 + min(180, (t - 900) * 3.0))
        tr = Track(1800, lambda t: 6, course)
        raw = make_reports(tr, report_times(1800, lo, hi, phase=13), vessel_type=37,
                           transceiver="B")
        out, funnel, _ = run(raw)
        assert funnel["dropped"].get("despike", 0) == 0, \
            f"κλειστή στροφή (αναφορές ανά {lo} s) θεωρήθηκε spike"
        assert out["seg_id"].nunique() == 1, f"κλειστή στροφή ({lo} s) έκοψε το segment"


def test_cog_wraparound():
    course = lambda t: 359.0 if t < 600 else 1.0
    tr = Track(1200, lambda t: 10, course)
    out, _, _ = run(make_reports(tr, report_times(1200, 60, 60, phase=17)))
    near = out[(out["t"] > T0 + pd.Timedelta(seconds=500)) &
               (out["t"] < T0 + pd.Timedelta(seconds=700))]
    assert ang_diff(near["cog"], 0).max() < 3.0, "η παρεμβολή του COG πρέπει να περνά από το 0°"


def test_sog_interp_large_jump_uses_nearest():
    speed = lambda t: 4 if t < 617 else 12
    tr = Track(1200, speed, lambda t: 90)
    out, _, cfg = run(make_reports(tr, report_times(1200, 60, 60, phase=17)))
    assert 12 - 4 > cfg["sog_interp_max_diff_kn"]
    # αναφορές στα 557 s (4 kn) και 617 s (12 kn) → λεπτό στα 600 s είναι πιο κοντά στο 617
    row = out[out["t"] == T0 + pd.Timedelta(seconds=600)]
    assert len(row) == 1
    assert abs(row["sog"].iloc[0] - 12) < 0.2, \
        f"SOG {row['sog'].iloc[0]:.1f}: έπρεπε να πάρει την πλησιέστερη αναφορά (12 kn)"


# ---------------------------------------------------------------------------
# A6 / A7: despike, τομές
# ---------------------------------------------------------------------------
def test_lateral_spikes_removed():
    for d in (200, 300, 500, 1000, 2000):
        tr = straight(2400, kn=10, course=0)
        times = report_times(2400, 60, 60, phase=17)
        raw = make_reports(tr, times)
        k = len(times) // 2
        x, y = latlon_to_xy(raw.loc[k, "latitude"], raw.loc[k, "longitude"])
        raw.loc[k, "latitude"], raw.loc[k, "longitude"] = xy_to_latlon(x + d, y)
        out, funnel, _ = run(raw)
        assert funnel["dropped"].get("despike", 0) >= 1, f"spike {d} m δεν αφαιρέθηκε"
        assert out["seg_id"].nunique() == 1, f"spike {d} m έκοψε το segment"
        err = pos_error_m(tr, out)
        assert err.max() < 5.0, f"spike {d} m: σφάλμα {err.max():.0f} m μένει στην έξοδο"


def test_longitudinal_spikes_removed():
    # spike κατά μήκος της πορείας: το κριτήριο της χορδής μετρά απόκλιση στον χρόνο, όχι μόνο πλευρικά
    for d in (200, 300, 500, 1000, 2000):
        tr = straight(2400, kn=10, course=0)
        times = report_times(2400, 60, 60, phase=17)
        raw = make_reports(tr, times)
        k = len(times) // 2
        x, y = latlon_to_xy(raw.loc[k, "latitude"], raw.loc[k, "longitude"])
        raw.loc[k, "latitude"], raw.loc[k, "longitude"] = xy_to_latlon(x, y + d)
        out, funnel, _ = run(raw)
        assert funnel["dropped"].get("despike", 0) >= 1, f"διαμήκες spike {d} m δεν αφαιρέθηκε"
        assert out["seg_id"].nunique() == 1, f"διαμήκες spike {d} m έκοψε το segment"
        assert pos_error_m(tr, out).max() < 5.0


def test_teleport_cuts_with_hole():
    tr1 = straight(1200, kn=10, course=0)
    tr2 = straight(1200, kn=10, course=0, x0=5000, y0=tr1.y[-1], t0=T0 + pd.Timedelta(seconds=1260))
    r1 = make_reports(tr1, report_times(1200, 60, 60, phase=17))
    r2 = make_reports(tr2, report_times(1200, 60, 60, phase=17))
    out, _, _ = run(pd.concat([r1, r2], ignore_index=True))
    assert out["seg_id"].nunique() == 2, "το άλμα 5 km πρέπει να κόψει το segment"
    g = out.groupby("seg_id")["t"]
    t_end, t_start = g.max().min(), g.min().max()
    assert t_start - t_end >= pd.Timedelta(minutes=2), \
        "πρέπει να μείνει τουλάχιστον ένα κενό λεπτό μετά την τομή"


def test_gaps_bridging_and_segmentation():
    cfg = cfg_default()
    gaps_len = [120, 180, 300, 480, cfg["seg_gap_s"] + 300]
    times, t, spans = [], 17.0, []
    for g in gaps_len:
        run_end = t + 1800
        times += list(np.arange(t, run_end + 1, 60.0))
        spans.append((run_end, run_end + g))
        t = run_end + g
    times += list(np.arange(t, t + 1801, 60.0))
    times = np.array(times)
    tr = straight(times[-1] + 1, kn=8, course=30)
    out, _, cfg = run(make_reports(tr, times))
    for (a, b), g in zip(spans, gaps_len):
        inside = minutes_strictly_between(T0, a, b)
        rows = out[out["t"].isin(inside)]
        if g <= cfg["bridge_max_s"]:
            assert len(rows) == len(inside), f"κενό {g} s έπρεπε να γεφυρωθεί"
            want = 1 if g > cfg["gap_imp_flag_s"] else 0
            assert (rows["GAP_IMP"] == want).all(), f"κενό {g} s: λάθος GAP_IMP"
        else:
            assert rows.empty, f"κενό {g} s: δεν επιτρέπονται σημεία μέσα"
        before = out[out["t"] < T0 + pd.Timedelta(seconds=a)]["seg_id"].iloc[-1]
        after = out[out["t"] > T0 + pd.Timedelta(seconds=b)]["seg_id"].iloc[0]
        if g > cfg["seg_gap_s"]:
            assert before != after, f"κενό {g} s έπρεπε να κόψει segment"
        else:
            assert before == after, f"κενό {g} s δεν έπρεπε να κόψει segment"


def test_short_segment_removed():
    cfg = cfg_default()
    short = cfg["min_segment_s"] - 120
    times = np.concatenate([
        np.arange(17, 1817, 60.0),
        np.arange(1817 + 1500, 1817 + 1500 + short + 1, 60.0),     # σύντομο κομμάτι
        np.arange(1817 + 3000 + short, 1817 + 4800 + short, 60.0),
    ])
    tr = straight(times[-1] + 1, kn=8, course=10)
    out, funnel, _ = run(make_reports(tr, times))
    lo = T0 + pd.Timedelta(seconds=1817 + 1400)
    hi = T0 + pd.Timedelta(seconds=1817 + 1600 + short)
    assert out[(out["t"] > lo) & (out["t"] < hi)].empty, "το σύντομο segment έπρεπε να αφαιρεθεί"
    assert funnel["dropped"].get("short_segment", 0) > 0


# ---------------------------------------------------------------------------
# A2–A5: εγκυρότητα, κουτί, sentinels, ταχύτητα, διπλότυπα
# ---------------------------------------------------------------------------
def test_invalid_mmsi_excluded():
    bad = [993123456, 12345678, 111234567, 3669999, 1234567890]
    parts = [make_reports(straight(1200), report_times(1200), mmsi=m) for m in bad]
    parts.append(make_reports(straight(1200), report_times(1200), mmsi=367123456))
    out, funnel, _ = run(pd.concat(parts, ignore_index=True))
    assert set(out["mmsi"].unique()) == {367123456}, "μη έγκυρα MMSI πέρασαν"
    assert funnel["dropped"].get("invalid_mmsi", 0) > 0


def test_box_margin_and_core_flag():
    cfg = cfg_default()
    reg = cfg["region"]
    lat0, lon0 = 33.0, reg["lon_min"] + 0.05
    x0, y0 = latlon_to_xy(lat0, lon0)
    tr = Track(4 * 3600, lambda t: 15, lambda t: 270, x0=x0, y0=y0)   # προς τα δυτικά
    out, funnel, _ = run(make_reports(tr, report_times(4 * 3600)))
    margin_deg = math.degrees(cfg["margin_km"] * 1000 / (R_EARTH * math.cos(math.radians(lat0))))
    assert (out["lon"] >= reg["lon_min"] - margin_deg - 1e-3).all(), "σημεία έξω από το περιθώριο"
    core = out["lon"] >= reg["lon_min"]
    assert (out.loc[core, "IN_CORE"] == 1).all() and (out.loc[~core, "IN_CORE"] == 0).all()
    assert (~core).any(), "αναμένονταν σημεία στο περιθώριο"
    assert funnel["dropped"].get("outside_box", 0) > 0


def test_sentinels_become_nan_rows_kept():
    tr = straight(1800, kn=10, course=60)
    raw = make_reports(tr, report_times(1800, 60, 60, phase=17))
    idx = raw.index[5:10]
    raw.loc[idx, "heading"] = 511
    raw.loc[idx, "cog"] = 360.0
    raw.loc[idx, "sog"] = 102.3
    out, funnel, _ = run(raw)
    assert (out["heading"].dropna() < 360).all()
    assert (out["cog"].dropna() < 360).all()
    assert (out["sog"].dropna() < 102.2).all()
    assert funnel["kept_reports"] == len(raw), "οι γραμμές με sentinels πρέπει να μείνουν"


def test_duplicates_keep_most_complete():
    tr = straight(1800, kn=10, course=60)
    raw = make_reports(tr, report_times(1800, 60, 60, phase=17))
    dup = raw.iloc[[10]].copy()
    raw.loc[10, ["sog", "cog", "heading"]] = np.nan   # η πρώτη εγγραφή ελλιπής
    raw = pd.concat([raw, dup], ignore_index=True)
    out, funnel, _ = run(raw)
    assert funnel["dropped"].get("duplicate", 0) == 1
    t10 = T0 + pd.Timedelta(seconds=float(raw.loc[10, "_t_s"]))
    row = out[out["t"] == t10.ceil("min")]
    assert row["sog"].notna().all(), "κρατήθηκε η ελλιπής εγγραφή αντί για την πλήρη"


def test_speed_plausibility_global_cap():
    cfg = cfg_default()
    cap = cfg["speed_caps_kn"]["global"]
    # SOG εξωπραγματικό (70 kn) ενώ το πλοίο κινείται με 8 → SOG άκυρο, τροχιά ακέραιη
    bad = make_reports(straight(1800, kn=8), report_times(1800), mmsi=367000052, vessel_type=52)
    bad["sog"] = 70.0
    # γρήγορα μικρά σκάφη με «λάθος» τύπο (V5, Μάρτιος 2025): sport fishing 35 kn, «ρυμουλκό» 50 kn
    fish = make_reports(straight(1800, kn=35), report_times(1800), mmsi=367010460, vessel_type=30,
                        transceiver="B")
    tug = make_reports(straight(1800, kn=50), report_times(1800), mmsi=368121190, vessel_type=31)
    hsc = make_reports(straight(1800, kn=35), report_times(1800), mmsi=367000040, vessel_type=40)
    out, funnel, _ = run(pd.concat([bad, fish, tug, hsc], ignore_index=True))
    assert 70 > cap
    b = out[out["mmsi"] == 367000052]
    assert not ((b["sog"] - 70).abs() < 1).any(), "το SOG 70 kn έπρεπε να ακυρωθεί"
    assert funnel["nulled"].get("sog_implausible", 0) >= len(bad)
    assert b["seg_id"].nunique() == 1
    for m, kn in [(367010460, 35), (368121190, 50), (367000040, 35)]:
        o = out[out["mmsi"] == m]
        assert len(o) > 20 and o["seg_id"].nunique() == 1, f"{m}: γρήγορο σκάφος ({kn} kn) κόπηκε"
        assert (o["sog"] - kn).abs().max() < 0.5, f"{m}: το SOG {kn} kn δεν έπρεπε να ακυρωθεί"
    assert funnel["dropped"].get("short_segment", 0) == 0


def test_vessel_category_from_mode():
    tr = straight(1800)
    raw = make_reports(tr, report_times(1800), vessel_type=70)
    raw.loc[raw.index[:3], "vessel_type"] = np.nan
    raw.loc[raw.index[3:5], "vessel_type"] = 0
    out, _, _ = run(raw)
    assert (out["vessel_cat"] == "cargo").all()


# ---------------------------------------------------------------------------
# A10: πορεία, ακίνητα, heading που λείπει
# ---------------------------------------------------------------------------
def test_stationary_classB_kept_without_heading():
    cfg = cfg_default()
    rng = np.random.default_rng(5)
    tr = Track(3600, lambda t: 0.0, lambda t: 0.0)
    raw = make_reports(tr, report_times(3600, 170, 190), transceiver="B", vessel_type=37,
                       pos_noise_m=8.0, rng=rng)
    raw["sog"] = 0.1
    raw["cog"] = rng.uniform(0, 360, len(raw)).round(1)
    raw["heading"] = np.nan
    out, funnel, _ = run(raw, bridge_max_s=max(cfg["bridge_max_s"], 200))
    assert len(out) > 30, "ακίνητο class B χωρίς heading δεν πρέπει να εξαφανίζεται"
    assert (out["STATIONARY"] == 1).mean() >= 0.95
    assert (out["COG_VALID"] == 0).all()
    assert out["seg_id"].nunique() <= 1
    assert funnel["dropped"].get("despike", 0) <= 0.02 * len(raw), "ο θόρυβος GPS κόπηκε ως spike"


def test_low_speed_cog_invalid():
    cfg = cfg_default()
    tr = straight(1800, kn=cfg["cog_valid_min_sog_kn"] * 0.5, course=90)
    out, _, _ = run(make_reports(tr, report_times(1800)))
    assert (out["COG_VALID"] == 0).all()


# ---------------------------------------------------------------------------
# A11 και ταχύτητα: έλεγχοι συνέπειας
# ---------------------------------------------------------------------------
def test_qc_sog_flags_inconsistent_speed():
    tr = straight(1800, kn=10, course=20)
    good = make_reports(tr, report_times(1800), mmsi=367000101)
    bad = make_reports(tr, report_times(1800), mmsi=367000102)
    bad["sog"] = (bad["sog"] * 1.5).round(1)
    out, _, _ = run(pd.concat([good, bad], ignore_index=True))
    assert out.loc[out["mmsi"] == 367000101, "QC_SOG"].mean() < 0.05
    assert out.loc[out["mmsi"] == 367000102, "QC_SOG"].mean() > 0.8


def test_velocity_vectors():
    tr = straight(1800, kn=10, course=0)
    out, _, _ = run(make_reports(tr, report_times(1800)))
    inner = out.iloc[2:-2]
    v = 10 * KN
    assert (inner["vy_sog"] - v).abs().max() < 0.1 and inner["vx_sog"].abs().max() < 0.1
    assert (inner["vy_pos"] - v).abs().max() < 0.1 and inner["vx_pos"].abs().max() < 0.1


# ---------------------------------------------------------------------------
# Περιπτώσεις από τον οπτικό έλεγχο (V5, Μάρτιος 2025)
# ---------------------------------------------------------------------------
def test_v5_wrong_low_sog_does_not_cut():
    # πλοίο που δηλώνει SOG 0 ενώ κινείται με 12 kn (V5: 338090233, 368121190)
    tr = Track(3600, lambda t: 12, lambda t: 30 + 0.01 * t)
    raw = make_reports(tr, report_times(3600, 60, 90), vessel_type=52)
    raw["sog"] = 0.0
    out, funnel, _ = run(raw)
    assert out["seg_id"].nunique() == 1, "λάθος χαμηλό SOG δεν πρέπει να κόβει την τροχιά"
    assert funnel["dropped"].get("short_segment", 0) == 0
    assert funnel["dropped"].get("despike", 0) == 0


def test_v5_corner_and_deceleration_not_spikes():
    # γωνία 60° στα 12 kn (V5: 368244550, 369140000) και ferry που κόβει 15 → 0 kn σε 2 λεπτά
    corner = Track(2400, lambda t: 12, lambda t: 0 if t < 1200 else min(60, (t - 1200) * 1.0))
    dec = Track(2400, lambda t: 15 if t < 1200 else max(0.0, 15 - (t - 1200) * 15 / 120),
                lambda t: 90)
    for name, tr in (("γωνία", corner), ("επιβράδυνση", dec)):
        out, funnel, _ = run(make_reports(tr, report_times(2400, 60, 80), pos_noise_m=5))
        assert funnel["dropped"].get("despike", 0) == 0, f"{name}: θεωρήθηκε spike"


def test_v5_no_long_chains_of_removals():
    # μόνιμη μετατόπιση 6 διαδοχικών αναφορών κατά 400 m (V5: 366991470)
    tr = straight(2400, kn=10, course=0)
    raw = make_reports(tr, report_times(2400, 60, 60, phase=17))
    k = len(raw) // 2
    for j in range(k, k + 6):
        x, y = latlon_to_xy(raw.loc[j, "latitude"], raw.loc[j, "longitude"])
        raw.loc[j, "latitude"], raw.loc[j, "longitude"] = xy_to_latlon(x + 400, y)
    _, funnel, cfg = run(raw)
    assert funnel["dropped"].get("despike", 0) <= 2 * cfg["despike"]["max_consecutive"], \
        "το despike έφαγε αλυσίδα αναφορών"


def test_v5_hermite_no_loops_at_low_speed():
    # σχεδόν ακίνητο σκάφος με θορυβώδες SOG (2–2.9 kn) και τυχαίο COG (V5: 367424810, 369914072).
    # Με Hermite για SOG ≥ 0.5 kn το σφάλμα έφτανε ~56 m· με το όριο των 3 kn ~13 m.
    rng = np.random.default_rng(11)
    tr = Track(3600, lambda t: 0.3, lambda t: 45)
    raw = make_reports(tr, report_times(3600, 150, 180, rng=rng), pos_noise_m=5, rng=rng)
    raw["cog"] = rng.uniform(0, 360, len(raw)).round(1)
    raw["sog"] = rng.uniform(2.0, 2.9, len(raw)).round(1)
    out, _, _ = run(raw)
    err = pos_error_m(tr, out)
    assert err.max() < 30, f"η παρεμβολή έφυγε {err.max():.0f} m από την τροχιά (βρόχος Hermite)"


def test_v5_placeholder_mmsi_excluded():
    parts = [make_reports(straight(1200), report_times(1200), mmsi=m) for m in (345000000, 366000000)]
    parts.append(make_reports(straight(1200), report_times(1200), mmsi=345000001))
    out, funnel, _ = run(pd.concat(parts, ignore_index=True))
    assert set(out["mmsi"].unique()) == {345000001}


def test_v5_qc_needs_min_reports():
    tr = straight(1500, kn=10, course=0)
    raw = make_reports(tr, report_times(1500, 70, 70, phase=5), mmsi=368297020)
    k = len(raw) // 2
    raw.loc[k, "latitude"] += 0.02
    _, funnel, cfg = run(raw)
    assert len(raw) < cfg["mmsi_qc_min_reports"]
    assert 368297020 not in set(funnel["mmsi_qc_flagged"]), "σημαία QC από 1 ανωμαλία σε λίγες αναφορές"


# ---------------------------------------------------------------------------
# Συνέχεια, ποιότητα ανά MMSI, funnel, ντετερμινισμός
# ---------------------------------------------------------------------------
def test_midnight_utc_continuity():
    t0 = pd.Timestamp("2025-03-10 23:40:00", tz="UTC")
    tr = straight(2400, kn=9, course=120, t0=t0)
    raw = make_reports(tr, report_times(2400, 60, 75))
    day1 = raw[raw["base_date_time"] < "2025-03-11"]
    day2 = raw[raw["base_date_time"] >= "2025-03-11"]
    out, _, _ = run(pd.concat([day1, day2], ignore_index=True))
    assert out["seg_id"].nunique() == 1, "τεχνητή τομή στα μεσάνυχτα UTC"
    assert (out["t"].diff().dropna() == pd.Timedelta(minutes=1)).all(), "κενό στα μεσάνυχτα"


def test_mmsi_clone_flagged():
    a = make_reports(straight(3600, kn=10, course=0), report_times(3600, 70, 70, phase=0), mmsi=367777777)
    b = make_reports(straight(3600, kn=10, course=0, x0=10_000), report_times(3600, 70, 70, phase=35),
                     mmsi=367777777)
    ok = make_reports(straight(3600, kn=10, course=0, x0=30_000), report_times(3600), mmsi=367000303)
    _, funnel, _ = run(pd.concat([a, b, ok], ignore_index=True))
    assert 367777777 in set(funnel["mmsi_qc_flagged"]), "ο κλώνος MMSI δεν σημάνθηκε"
    assert 367000303 not in set(funnel["mmsi_qc_flagged"])


def test_heading_not_filled_from_future():
    # αργό class B χωρίς heading· heading εμφανίζεται μόνο στο τέλος (V7: 11–12 πλοία)
    tr = Track(3600, lambda t: 0.2, lambda t: 0)
    raw = make_reports(tr, report_times(3600, 60, 80), transceiver="B", vessel_type=37)
    raw["heading"] = np.nan
    raw.loc[raw.index[-5:], "heading"] = 123.0
    out, _, _ = run(raw)
    t_first = T0 + pd.Timedelta(seconds=float(raw["_t_s"].iloc[-5]))
    early = out[out["t"] < t_first - pd.Timedelta(minutes=2)]
    assert len(early) > 30 and early["heading"].isna().all(), "το heading συμπληρώθηκε από το μέλλον"


def test_no_future_information():
    # V7: η έξοδος για το διάστημα [0, Τ − 30 λεπτά] δεν πρέπει να αλλάζει αν κόψουμε τα δεδομένα στο Τ
    raw = _mixed_scenario()
    raw = raw[raw["mmsi"] != 99912345]
    cut_s = 2400
    trunc = raw[raw["_t_s"] <= cut_s]
    full, _, _ = run(raw)
    part, _, _ = run(trunc)
    lim = T0 + pd.Timedelta(seconds=cut_s - 1800)
    cols = ["mmsi", "t", "lat", "lon", "sog", "cog", "COG_VALID", "heading", "HDG_SRC", "GAP_IMP",
            "INTERP_SPAN", "STATIONARY", "QC_SOG", "QC_COG", "vx_sog", "vy_sog", "vx_pos", "vy_pos"]
    a = full.loc[full["t"] <= lim, cols].reset_index(drop=True)
    b = part.loc[part["t"] <= lim, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def _mixed_scenario():
    rng = np.random.default_rng(42)
    parts = []
    for i in range(6):
        tr = Track(3600, lambda t, s=6 + 2 * i: s, lambda t, c=40 * i: (c + 0.02 * t) % 360,
                   x0=i * 3000.0)
        parts.append(make_reports(tr, report_times(3600, 60, 90, rng=rng, gaps=[(1500, 1500 + 60 * i)]),
                                  mmsi=367100000 + i, pos_noise_m=3.0, rng=rng))
    raw = pd.concat(parts, ignore_index=True)
    raw = pd.concat([raw, raw.iloc[[3, 50, 120]]], ignore_index=True)    # διπλότυπα
    raw.loc[7, "latitude"] += 0.02                                        # spike
    raw = pd.concat([raw, make_reports(straight(600), report_times(600), mmsi=99912345)])
    return raw.reset_index(drop=True)


def test_funnel_closes():
    raw = _mixed_scenario()
    _, funnel, _ = run(raw)
    total = funnel["kept_reports"] + sum(funnel["dropped"].values())
    assert funnel["raw_rows"] == len(raw)
    assert total == funnel["raw_rows"], \
        f"το funnel δεν κλείνει: {funnel['kept_reports']} + {sum(funnel['dropped'].values())} ≠ {funnel['raw_rows']}"


def test_deterministic_and_order_invariant():
    raw = _mixed_scenario()
    o1, _, _ = run(raw)
    o2, _, _ = run(raw)
    o3, _, _ = run(raw.sample(frac=1.0, random_state=7))
    pd.testing.assert_frame_equal(o1, o2)
    pd.testing.assert_frame_equal(o1, o3)


def test_output_invariants():
    raw = _mixed_scenario()
    out, _, cfg = run(raw)
    for _, g in out.groupby("seg_id"):
        x, y = latlon_to_xy(g["lat"].to_numpy(), g["lon"].to_numpy())
        dt = g["t"].diff().dt.total_seconds().to_numpy()[1:]
        v = np.hypot(np.diff(x), np.diff(y)) / dt / KN
        assert (v <= cfg["speed_caps_kn"]["global"]).all(), "βήμα πάνω από το γενικό όριο"
        assert (dt >= 60).all()


# ---------------------------------------------------------------------------
# Έλεγχος της ίδιας της γεννήτριας (δεν χρειάζεται pipeline)
# ---------------------------------------------------------------------------
def selfcheck():
    tr = straight(3600, kn=10, course=45)
    x, y = tr.at(3600)
    d = math.hypot(float(x), float(y))
    assert abs(d - 10 * KN * 3600) < 0.5, f"ευθεία: απόσταση {d:.1f}"
    lat, lon = xy_to_latlon(1234.0, -567.0)
    xb, yb = latlon_to_xy(lat, lon)
    assert abs(xb - 1234) < 1e-6 and abs(yb + 567) < 1e-6, "μετατροπή συντεταγμένων"
    sog, cog, _, _ = tr.state([100, 2000])
    assert np.allclose(sog, 10) and np.allclose(cog, 45)
    omega = 0.75
    turn = Track(120, lambda t: 10, lambda t: t * omega)
    r = 10 * KN / math.radians(omega)
    xe, ye = turn.at(120)
    assert abs(math.hypot(float(xe) - r, float(ye)) - r) < 0.5, "κυκλική στροφή"
    ts = report_times(3600, 60, 80, gaps=[(1000, 1500)])
    dts = np.diff(ts)
    assert dts.min() >= 60 - 1e-9 and (dts <= 80 + 1e-9).sum() >= len(dts) - 1
    assert not ((ts > 1000) & (ts < 1500)).any()
    raw = make_reports(tr, ts)
    assert set(["mmsi", "base_date_time", "longitude", "latitude", "sog", "cog", "heading",
                "vessel_type", "status", "transceiver"]).issubset(raw.columns)
    assert raw["base_date_time"].str.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$").all()
    assert len(minutes_strictly_between(T0, 17, 197)) == 3
    print("selfcheck: η γεννήτρια είναι συνεπής ✓")


# ---------------------------------------------------------------------------
# Runner χωρίς pytest
# ---------------------------------------------------------------------------
def _main(argv):
    if "--selfcheck" in argv:
        selfcheck()
        return 0
    pattern = argv[argv.index("-k") + 1] if "-k" in argv else ""
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and inspect.isfunction(f) and pattern in n]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"  ✗ {name}: {msg}")
            if "-v" in argv:
                traceback.print_exc()
            if isinstance(e, RuntimeError) and "preprocess_noaa" in str(e):
                break
    print(f"\n{passed} πέρασαν, {len(failed)} απέτυχαν από {len(tests)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))