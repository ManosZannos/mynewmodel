#!/usr/bin/env python3
"""test_stageB.py — tests του σταδίου Β σε συνθετικό build (python test_stageB.py)."""
import json, shutil, sys, tempfile, traceback
from pathlib import Path
import numpy as np, pandas as pd
import test_synthetic as T
import preprocess_noaa as P
import stageB_manifests as B

EAST = T.REF_LON + 0.05


def _build(tmp):
    rng = np.random.default_rng(0)
    parts = []
    for i in range(80):
        kn = rng.uniform(4, 16)
        turn = (i % 4 == 0)
        course0 = rng.uniform(0, 360)
        x0, y0 = rng.uniform(-15e3, 15e3), rng.uniform(-15e3, 15e3)
        tr = T.Track(6 * 3600, lambda t, k=kn: k,
                     (lambda t, c=course0: c + (0.3 * t if (t // 900) % 2 else 0)) if turn else (lambda t, c=course0: c),
                     x0=x0, y0=y0)
        parts.append(T.make_reports(tr, T.report_times(6 * 3600, 60, 80, rng=rng), mmsi=367100000 + i,
                                    vessel_type=70 if i % 3 else 37, rng=rng))
    for i in range(10):   # ακίνητα
        tr = T.Track(6 * 3600, lambda t: 0.0, lambda t: 0.0, x0=rng.uniform(-5e3, 5e3))
        r = T.make_reports(tr, T.report_times(6 * 3600, 60, 80, rng=rng), mmsi=367200000 + i, pos_noise_m=5, rng=rng)
        r["sog"] = 0.1
        parts.append(r)
    raw = T.raw_only(pd.concat(parts, ignore_index=True))
    out, _ = P.run_stage_a(raw, P.load_config())
    d = Path(tmp) / "build"
    d.mkdir()
    for day, g in out.groupby(out["t"].dt.strftime("%Y-%m-%d")):
        g.to_csv(d / f"day_{day}.csv.gz", index=False)
    return d, out


def _read(p):
    return pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)


def _out(d, name):
    o = d / "stageB"
    for ext in (".parquet", ".csv.gz"):
        f = o / f"{name}{ext}"
        if f.exists():
            return _read(f)
    raise FileNotFoundError(name)


TMP = tempfile.mkdtemp()
D, GRID = _build(TMP)
B.main([str(D), "--manifests", "vessel", "region", "time", "--east_lon", str(EAST)])
W = _out(D, "windows")


def test_windows_are_contiguous_same_segment():
    g = GRID.sort_values(["mmsi", "t"]).reset_index(drop=True)
    g["tm"] = ((g["t"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1)).astype(np.int64)
    for r in W.sample(300, random_state=0).itertuples():
        rows = g.iloc[r.row0:r.row0 + 20]
        assert rows["seg_id"].nunique() == 1 and (np.diff(rows["tm"]) == 1).all()
        assert rows["mmsi"].iloc[0] == r.mmsi and rows["tm"].iloc[0] == r.t_start


def test_stationary_not_eligible():
    st = W[W["mmsi"] >= 367200000]
    assert len(st) > 0 and (st["eligible"] == 0).all(), "ακίνητα πλοία έγιναν στόχοι"


def test_difficulty_straight_vs_turning():
    straight = W[(W["mmsi"] % 4 != 0) & (W["mmsi"] < 367200000) & (W["eligible"] == 1)]
    turning = W[(W["mmsi"] % 4 == 0) & (W["mmsi"] < 367200000) & (W["eligible"] == 1)]
    assert straight["difficulty"].median() < 2.0, f"ευθείες: διάμεση δυσκολία {straight['difficulty'].median():.1f}"
    assert turning["difficulty"].quantile(0.9) > 12, "οι στροφές δεν ξεχωρίζουν"


def test_cv_difficulty():
    el = W[(W["mmsi"] < 367200000) & (W["eligible"] == 1)]
    straight, turning = el[el["mmsi"] % 4 != 0], el[el["mmsi"] % 4 == 0]
    assert straight["cv_fde_m"].quantile(0.99) < 5.0, f"ευθεία: CV-FDE p99 {straight['cv_fde_m'].quantile(0.99):.1f} m"
    assert turning["cv_fde_m"].quantile(0.9) > 500, "οι στροφές δεν ξεχωρίζουν στο CV-FDE"
    m = json.load(open(D / "stageB" / "manifest_vessel.json"))
    assert m["cv_thresholds"][0] < m["cv_thresholds"][1]
    t = _out(D, "targets_vessel").merge(W[["window_id", "headline", "tier_cv"]], on="window_id")
    sh = t[(t["split"] == "train") & (t["headline"] == 1)]["tier_cv"].value_counts(normalize=True)
    assert abs(sh.get("easy", 0) - 0.35) < 0.05 and abs(sh.get("hard", 0) - 0.25) < 0.05, sh.to_dict()


def test_vessel_manifest_disjoint():
    t = _out(D, "targets_vessel")
    sets = {s: set(g["mmsi"]) for s, g in t.groupby("split")}
    assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["val"]) and not (sets["val"] & sets["test"])
    m = json.load(open(D / "stageB" / "manifest_vessel.json"))
    assert m["test_rows_from_train_mmsi"] == 0.0


def test_time_manifest_no_overlap():
    t = _out(D, "targets_time")
    g = B.DEFAULTS["gap_min"]
    end = t["t_start"] + 20
    tr, va, te = (t["split"] == s for s in ("train", "val", "test"))
    assert va.any() and te.any() and tr.any()
    assert t.loc[va, "t_start"].min() - end[tr].max() >= g, "κενό train→val < gap"
    assert t.loc[te, "t_start"].min() - end[va].max() >= g, "κενό val→test < gap"


def test_region_manifest_rule():
    t = _out(D, "targets_region").merge(W[["window_id", "lon_last", "lon_min", "lon_max"]], on="window_id")
    assert (t.loc[t["split"] == "test", "lon_last"] > EAST).all()
    assert (t.loc[t["split"] != "test", "lon_last"] <= EAST).all()
    assert not ((t["lon_min"] < EAST) & (t["lon_max"] > EAST)).any(), "παράθυρο που περνά το όριο"


def test_caps_flag_not_filter():
    caps = B.DEFAULTS["caps"]
    el = set(W.loc[W["eligible"] == 1, "window_id"])
    for name in ("vessel", "time"):
        t = _out(D, f"targets_{name}")
        if name == "vessel":
            assert set(t["window_id"]) == el, "τα όρια δεν πρέπει να αφαιρούν επιλέξιμα παράθυρα"
        c = t[t["capped"] == 1]
        assert 0 < len(c) <= len(t)
        for s, g in c.groupby("split"):
            assert g.groupby("mmsi").size().max() <= caps[s][0]
            assert g.groupby("seg_id").size().max() <= caps[s][1]


def test_difficulty_calibrated_on_train():
    m = json.load(open(D / "stageB" / "manifest_vessel.json"))
    thr = m["difficulty_thresholds"]
    t = _out(D, "targets_vessel").merge(W[["window_id", "difficulty", "headline", "tier"]], on="window_id")
    tr = t[(t["split"] == "train") & (t["headline"] == 1)]
    share = tr["tier"].value_counts(normalize=True)
    assert abs(share.get("easy", 0) - 0.35) < 0.05 and abs(share.get("hard", 0) - 0.25) < 0.05, share.to_dict()
    assert thr[0] < thr[1]


def test_deterministic():
    a = _out(D, "targets_vessel")
    B.main([str(D), "--manifests", "vessel", "--east_lon", str(EAST)])
    b = _out(D, "targets_vessel")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


# --- v1.3: funnel και seen σε μικρό χειροποίητο ευρετήριο (ανεξάρτητα από το build) ---
def _toy_windows(n_vessels=4, minutes=3000, stride=5):
    rows = [(v, s) for v in range(1, n_vessels + 1) for s in range(0, minutes, stride)]
    w = pd.DataFrame(rows, columns=["mmsi", "t_start"])
    w["window_id"] = np.arange(len(w))
    w["seg_id"] = w["mmsi"] * 10
    w["eligible"] = 1
    w["headline"] = (w["window_id"] % 2).astype(int)
    return w


def test_time_funnel_counts_gap_drops():
    """v1.2: τα παράθυρα στα κενά των ορίων χάνονταν χωρίς καταγραφή (1,405 στη SoCal Μαρτίου)."""
    w = _toy_windows()
    t, info = B.manifest_time(w, B.DEFAULTS)
    assert "dropped_gap" in info, "λείπει το dropped_gap από το manifest time"
    assert info["dropped_gap"] > 0 and len(t) + info["dropped_gap"] == int(w["eligible"].sum())
    f = B.check_funnel(w, t, info)
    assert f["closes"] and f["dropped_gap"] == info["dropped_gap"]


def test_funnel_detects_unexplained_loss():
    w = _toy_windows()
    t, info = B.manifest_time(w, B.DEFAULTS)
    try:
        B.check_funnel(w, t.iloc[1:], info)
    except AssertionError:
        return
    raise AssertionError("το check_funnel δεν εντόπισε ανεξήγητη απώλεια")


def test_alltest_seen_vessels_vs_windows():
    """v1.2: τυπωνόταν μόνο το ποσοστό παραθύρων (45.3%), ενώ το handoff ανέφερε 36.6% πλοία."""
    w = _toy_windows(n_vessels=1, minutes=450)                     # 90 παράθυρα, MMSI 1 (seen)
    others = _toy_windows(n_vessels=10, minutes=50)                # 10 παράθυρα ανά πλοίο
    others = others[others["mmsi"] > 1].assign(window_id=lambda d: d["window_id"] + 10_000)
    w = pd.concat([w, others], ignore_index=True)                  # 10 πλοία, 180 παράθυρα
    tmp = Path(tempfile.mkdtemp())
    pd.DataFrame({"window_id": [0], "mmsi": [1], "split": ["train"]}).to_csv(tmp / "targets_vessel.csv.gz", index=False)
    (tmp / "manifest_vessel.json").write_text("{}", encoding="utf-8")
    t, info = B.manifest_alltest(w, B.DEFAULTS, str(tmp / "manifest_vessel.json"))
    shutil.rmtree(tmp, ignore_errors=True)
    assert "seen_share" in info, "λείπει το seen_share από το manifest alltest"
    assert info["seen_share"]["vessels"] == 0.1 and info["seen_share"]["windows"] == 0.5, info["seen_share"]
    assert B.check_funnel(w, t, info)["closes"]


# --- v1.4: QC ακτής από το στάδιο Γ ---
def _toy_track(minutes=40, bad_minutes=()):
    t0 = pd.Timestamp("2025-03-01", tz="UTC")
    x = pd.DataFrame({"mmsi": 1, "t": t0 + pd.to_timedelta(np.arange(minutes), unit="min"), "seg_id": 10,
                      "lat": 33.0, "lon": -118.0 + np.arange(minutes) * 0.003, "sog": 10.0, "cog": 90.0,
                      "COG_VALID": 1, "GAP_IMP": 0, "IN_CORE": 1, "MMSI_QC": 0, "vessel_cat": "cargo", "transceiver": "A"})
    x["tm"] = ((x["t"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1)).astype(np.int64)
    x["LAND_BAD"] = np.isin(np.arange(minutes), bad_minutes).astype(np.int8)
    return x


def test_on_land_windows_excluded():
    """v1.3: παράθυρα με σημείο στη στεριά (τρέιλερ/ράμπες, V5-Γ) έμεναν επιλέξιμοι στόχοι."""
    w = B.build_windows(_toy_track(40, bad_minutes=[12]), B.DEFAULTS)
    assert list(w["t_start"] - w["t_start"].min()) == [0, 5, 10, 15, 20]
    assert "on_land" in w, "λείπει η σημαία on_land"
    assert list(w["on_land"]) == [1, 1, 1, 0, 0], list(w["on_land"])        # το λεπτό 12 ανήκει στα 0–19, 5–24, 10–29
    assert list(w["eligible_pre_shore"]) == [1] * 5 and list(w["eligible"]) == [0, 0, 0, 1, 1]


def test_attach_shore_tau_and_alignment():
    b = pd.DataFrame({"mmsi": [1, 1, 2, 2]})
    s = pd.DataFrame({"mmsi": [1, 1, 2, 2], "ON_LAND": [1, 1, 0, 1], "ON_LAND_DEPTH_M": [10.0, 30.0, 0.0, 25.0]})
    out = B.attach_shore([b], [s], 25.0)[0]
    assert list(out["LAND_BAD"]) == [0, 1, 0, 0], "τ: μόνο βάθος > 25 m"
    try:
        B.attach_shore([b], [s.iloc[::-1].reset_index(drop=True)], 25.0)
    except AssertionError:
        return
    raise AssertionError("η μη ευθυγράμμιση με το στάδιο Γ δεν εντοπίστηκε")


if __name__ == "__main__":
    k = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else ""
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f) and k in n]
    bad = 0
    for n, f in tests:
        try:
            f(); print(f"  ✓ {n}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"  ✗ {n}: {str(e).splitlines()[0] if str(e) else type(e).__name__}")
            if "-v" in sys.argv: traceback.print_exc()
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(tests) - bad} πέρασαν, {bad} απέτυχαν από {len(tests)}")
    sys.exit(1 if bad else 0)