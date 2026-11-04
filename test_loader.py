#!/usr/bin/env python3
"""test_loader.py — tests του loader και του CV baseline σε συνθετικά δεδομένα (python test_loader.py)."""
import shutil, sys, tempfile, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import loader as LD
import stageB_manifests as SB
import baselines as BL


def _track(mmsi, minutes, seg=1, lat0=33.0, lon0=-118.0, turn_deg_per_min=0.0, speed_kn=10.0, t0="2025-03-01"):
    hdg = 90.0 + turn_deg_per_min * np.arange(minutes)
    step = speed_kn * 1852 / 60.0
    lat, lon = [lat0], [lon0]
    for h in hdg[:-1]:
        lat.append(lat[-1] + step * np.cos(np.radians(h)) / 111_195.0)
        lon.append(lon[-1] + step * np.sin(np.radians(h)) / (111_195.0 * np.cos(np.radians(lat0))))
    return pd.DataFrame({"mmsi": mmsi, "t": pd.Timestamp(t0, tz="UTC") + pd.to_timedelta(np.arange(minutes), unit="min"),
                         "seg_id": seg, "lat": lat, "lon": lon, "sog": speed_kn, "cog": hdg % 360, "COG_VALID": 1,
                         "GAP_IMP": 0, "IN_CORE": 1, "MMSI_QC": 0, "vessel_cat": "cargo", "transceiver": "A",
                         "R_AFTER_S": 0, "P_AGE_S": 0, "P_LAT": lat, "P_LON": lon, "P_SOG": speed_kn, "P_COG": hdg % 360})


def _x():
    x = pd.concat([_track(1, 60), _track(2, 60, turn_deg_per_min=3.0, lat0=33.2),
                   _track(3, 45, seg=7, turn_deg_per_min=-2.0, speed_kn=6.0, lat0=32.8)], ignore_index=True)
    x = x.sort_values(["mmsi", "t"], kind="mergesort").reset_index(drop=True)
    x["tm"] = ((x["t"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1)).astype(np.int64)
    return x


def test_cv_matches_stageB():
    """Το CV μέσω του loader δίνει ακριβώς το cv_fde_m / cv_ade_m του σταδίου Β."""
    x = _x()
    w = SB.build_windows(x, SB.DEFAULTS)
    obs, fut, pos, _ = LD.make_arrays(x, w)
    ade, fde = BL.errors(BL.cv_predict(pos), fut)
    assert np.allclose(fde, w["cv_fde_m"], atol=0.05), np.abs(fde - w["cv_fde_m"]).max()
    assert np.allclose(ade, w["cv_ade_m"], atol=0.05), np.abs(ade - w["cv_ade_m"]).max()
    assert w["cv_fde_m"].max() > 50, "το συνθετικό χρειάζεται στροφές ώστε το CV να έχει σφάλμα"


def test_shapes_and_origin():
    x = _x()
    w = SB.build_windows(x, SB.DEFAULTS)
    obs, fut, pos, _ = LD.make_arrays(x, w)
    n = len(w)
    assert obs.shape == (n, LD.OBS + 1, len(LD.FEATURES)) and fut.shape == (n, LD.PRED, 2) and pos.shape == (n, LD.OBS, 2)
    assert np.allclose(pos[:, -1], 0) and np.allclose(obs[:, -1, :2], 0), "αρχή = τελευταίο σημείο παρατήρησης"
    assert np.isfinite(obs).all() and np.isfinite(fut).all()
    step = 10 * 1852 / 60
    assert np.allclose(np.linalg.norm(fut[w["mmsi"] == 1][:, 0], axis=-1), step, rtol=1e-3)


def test_no_future_leakage():
    """Αλλαγή στις γραμμές του μέλλοντος δεν αλλάζει τα χαρακτηριστικά παρατήρησης."""
    x = _x()
    w = SB.build_windows(x, SB.DEFAULTS).iloc[[0]]
    obs0, fut0, _, _ = LD.make_arrays(x, w)
    fr = w["row0"].iloc[0] + LD.OBS + np.arange(LD.PRED)
    x2 = x.copy()
    x2.loc[fr, ["lat", "lon"]] += 0.01
    x2.loc[fr, ["sog", "cog"]] = [25.0, 10.0]
    obs1, fut1, _, _ = LD.make_arrays(x2, w)
    assert np.array_equal(obs0, obs1), "τα χαρακτηριστικά παρατήρησης εξαρτώνται από το μέλλον"
    assert not np.allclose(fut0, fut1)


def test_misaligned_rows_detected():
    x = _x()
    w = SB.build_windows(x, SB.DEFAULTS)
    xs = x.sample(frac=1.0, random_state=0).reset_index(drop=True)
    try:
        LD.make_arrays(xs, w)
    except AssertionError:
        return
    raise AssertionError("δεν εντοπίστηκε αλλαγμένη σειρά γραμμών")


def test_scene_group():
    g = LD.scene_group(np.array([50.0, 50.0, 1500.0, 3000.0, 3000.0]), np.array([5000.0, 800.0, 5000.0, 5000.0, 1000.0]))
    assert list(g) == ["nearshore", "harbor", "coastal", "offshore", "harbor"], list(g)


def test_load_split_end_to_end_csv():
    """Build σε αρχεία → load_rows (ίδια σειρά με το stageB) → windows/targets → load_split."""
    d = Path(tempfile.mkdtemp())
    try:
        x = _x()
        for day, g in x.groupby(x["t"].dt.strftime("%Y-%m-%d")):
            g[SB.COLS + LD.CAUSAL_COLS].to_csv(d / f"day_{day}.csv.gz", index=False)
        xr = LD.load_rows(str(d))
        w = SB.build_windows(SB.load_build(str(d)), SB.DEFAULTS)
        w["tier_cv"] = "easy"
        (d / "stageB").mkdir()
        w.to_csv(d / "stageB" / "windows.csv.gz", index=False)
        t = w[w["eligible"] == 1][["window_id", "mmsi", "seg_id", "t_start"]].assign(split="val", capped=1)
        t.to_csv(d / "stageB" / "targets_vessel.csv.gz", index=False)
        out = LD.load_split(str(d), "vessel", "val", x=xr)
        assert len(out["obs"]) == len(t) and list(out["meta"]["window_id"]) == sorted(t["window_id"])
        ade, fde = BL.errors(BL.cv_predict(out["pos"]), out["fut"])
        ref = w.set_index("window_id").loc[out["meta"]["window_id"], "cv_fde_m"].to_numpy()
        assert np.allclose(fde, ref, atol=0.05)
        m = BL.metrics(ade, fde, out["meta"])
        assert m["headline"]["n"] == int(out["meta"]["headline"].sum())
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_rows_loaded_once_when_same_build():
    """v1.0: το build διαβαζόταν 2 φορές όταν εκπαίδευση και αξιολόγηση ήταν στο ίδιο build."""
    x = _x()
    orig = LD.load_rows
    LD.load_rows = lambda *a, **k: (_ for _ in ()).throw(AssertionError("δεύτερη φόρτωση του ίδιου build"))
    try:
        assert BL.train_rows("dataset/socal", "dataset/socal/", x) is x
    finally:
        LD.load_rows = orig


# --- loader v1.1 / baselines v1.2 / compare_runs ---
def test_nonoverlap_and_gap_frac():
    """v1.0: χωρίς stratum παρεμβολής και χωρίς μη επικαλυπτόμενο υποσύνολο."""
    x = _x()
    x.loc[x.index[25:28], "GAP_IMP"] = 1
    w = SB.build_windows(x, SB.DEFAULTS)
    _, _, _, extra = LD.make_arrays(x, w)
    assert "nonoverlap" in extra and "pred_gap_frac" in extra
    sel = w[extra["nonoverlap"] == 1]
    for _, g in sel.groupby("mmsi"):
        assert (g["t_start"].diff().dropna() >= LD.OBS + LD.PRED).all(), "επικαλυπτόμενα στο υποσύνολο nonoverlap"
    i = int(np.flatnonzero((w["mmsi"] == 1).to_numpy() & (w["row0"].to_numpy() == 15))[0])
    assert abs(extra["pred_gap_frac"][i] - 0.3) < 1e-9, extra["pred_gap_frac"][i]     # γραμμές 25–27 στα 25–34


def test_nonoverlap_coverage_unaligned_segment():
    """v1.1: σε segments που δεν ξεκινούν σε πολλαπλάσιο του 5 το υποσύνολο ήταν κενό (2,081 αντί ~9.6 χιλ.)."""
    x = _track(9, 100, t0="2025-03-01 00:07")
    x["tm"] = ((x["t"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1)).astype(np.int64)
    w = SB.build_windows(x, SB.DEFAULTS)
    nov = LD.make_arrays(x, w)[3]["nonoverlap"]
    assert nov.sum() == int(np.ceil(len(w) / 4)), (nov.sum(), len(w))


def test_resample_obs():
    x = _x()
    w = SB.build_windows(x, SB.DEFAULTS)
    obs, _, _, _ = LD.make_arrays(x, w, causal=False)
    o20, o120 = LD.resample_obs(obs, 20), LD.resample_obs(obs, 120)
    assert o20.shape[1] == 3 * (LD.OBS - 1) + 1 and o120.shape[1] == LD.OBS // 2
    assert np.allclose(o20[:, ::3, :2], obs[:, :, :2], atol=1e-6), "τα σημεία των 60 s πρέπει να διατηρούνται"
    assert np.allclose(o20[:, -1, :2], 0) and np.allclose(o120[:, -1, :2], 0), "τελευταίο σημείο = t0"
    assert np.allclose(o120[:, :, :2], obs[:, 1::2, :2])
    assert np.allclose(LD.resample_obs(obs, 60), obs)


def test_compare_runs():
    import compare_runs as CR
    r = CR.compare({0: 500.0, 1: 506.0, 2: 502.0}, {0: 490.0, 1: 497.0, 2: 491.0})
    assert r["n"] == 3 and abs(r["diff"] + 10.0) < 1e-9 and r["se"] is not None and r["z"] < -3
    r2 = CR.compare({0: 500.0, 1: 504.0}, {None: 642.4})
    assert r2["n"] == 2 and abs(r2["diff"] - 140.4) < 1e-9


# --- loader v1.3: causal observation ---
def _one_window(x):
    w = SB.build_windows(x, SB.DEFAULTS)
    i = int(np.flatnonzero((w["mmsi"] == 1).to_numpy() & (w["row0"].to_numpy() == 15))[0])
    return w.iloc[[i]].reset_index(drop=True), 15


def _last_report_at(x, r0, age_s):
    """Row t0 = r0 + 9: last report at t0 - age_s, on the straight track of MMSI 1 (exact position)."""
    a = age_s / 60.0
    for c in ("lat", "lon"):
        x.loc[r0 + 9, "P_" + c.upper()] = (1 - a) * x.loc[r0 + 9, c] + a * x.loc[r0 + 8, c]
    x.loc[r0 + 9, "P_AGE_S"] = age_s


def test_causal_mask_and_origin():
    """v1.2: the last obs points used reports after t0. v1.4: the origin was the last available grid point,
    which drops the last real report before t0 (§12.6). v1.5: extra token = last report, origin = that report."""
    x = _x()
    w, r0 = _one_window(x)
    x.loc[r0 + 9, "R_AFTER_S"] = 30       # needs a report 30 s after t0  -> unavailable
    x.loc[r0 + 8, "R_AFTER_S"] = 90       # report 30 s after t0          -> unavailable
    x.loc[r0 + 7, "R_AFTER_S"] = 100      # report 20 s before t0         -> available
    _last_report_at(x, r0, 20)            # that report is the last one at or before t0
    obs, fut, pos, extra = LD.make_arrays(x, w)
    i_av, i_dt = LD.FEATURES.index("available"), LD.FEATURES.index("dt_min")
    assert obs.shape[1] == LD.OBS + 1 and obs.shape[-1] == len(LD.FEATURES)
    assert list(obs[0, :, i_av]) == [1] * 8 + [0, 0] + [1], obs[0, :, i_av]
    assert np.allclose(obs[0, 8:10], 0) and extra["last_avail"][0] == 7
    assert np.allclose(obs[0, :8, i_dt], np.arange(9, 1, -1)) and abs(obs[0, -1, i_dt] - 20 / 60) < 1e-6
    assert np.allclose(obs[0, -1, :2], 0) and abs(extra["p_age_s"][0] - 20) < 1e-9, "αρχή = τελευταία αναφορά"
    step = 10 * 1852 / 60
    assert abs(np.linalg.norm(pos[0, 7]) - step * 100 / 60) < 0.5, "θέση 7 (t0 - 120 s) σε σχέση με την αναφορά t0 - 20 s"
    assert abs(np.linalg.norm(fut[0, 0]) - step * 80 / 60) < 0.5, "στόχος σε σχέση με την αναφορά"
    x2 = x.copy()
    x2.loc[[r0 + 8, r0 + 9], ["lat", "lon"]] += 0.01
    x2.loc[[r0 + 8, r0 + 9], ["sog", "cog"]] = [25.0, 10.0]
    obs2, fut2, _, _ = LD.make_arrays(x2, w)
    assert np.array_equal(obs, obs2) and np.array_equal(fut, fut2), "είσοδος/στόχος εξαρτώνται από σημεία μετά το t0"
    obs_nc = LD.make_arrays(x, w, causal=False)[0]
    assert obs_nc.shape[1] == LD.OBS and obs_nc[0, :, -1].all() and not np.allclose(obs_nc[0, 9], 0)


def test_causal_cv_from_last_report():
    """v1.4: CV extrapolated from the last available grid point, ignoring the later report (§12.6)."""
    x = _x()
    w, r0 = _one_window(x)                     # MMSI 1: straight line, constant speed
    x.loc[[r0 + 8, r0 + 9], "R_AFTER_S"] = [90, 30]
    _last_report_at(x, r0, 20)
    x.loc[r0 + 9, "P_LAT"] += 50 / 111_195.0  # report 50 m north of the track: CV must start from it
    _, fut, pos, extra = LD.make_arrays(x, w)
    pred = BL.cv_predict(pos, last=extra["last_avail"], age_s=extra["p_age_s"])
    ade, fde = BL.errors(pred, fut)
    assert abs(fde[0] - 50) < 0.5 and abs(ade[0] - 50) < 0.5, (ade, fde)   # parallel offset of 50 m


def test_last_report_before_last_available_rejected():
    """P_AGE_S older than the last available grid point means misaligned or inconsistent build columns."""
    x = _x()
    w, r0 = _one_window(x)
    x.loc[r0 + 9, "P_AGE_S"] = 120            # all grid points available, but last report 2 min old
    try:
        LD.make_arrays(x, w)
    except AssertionError:
        return
    raise AssertionError("ασυνεπές P_AGE_S έγινε δεκτό")


def test_missing_r_after_s_rejected_in_causal_mode():
    x = _x().drop(columns=["R_AFTER_S"])
    w, _ = _one_window(_x())
    try:
        LD.make_arrays(x, w)
    except AssertionError:
        LD.make_arrays(x, w, causal=False)
        return
    raise AssertionError("build χωρίς R_AFTER_S έγινε δεκτό σε αιτιατή λειτουργία")


def test_missing_last_report_rejected_in_causal_mode():
    """v1.5: a stage A < v3.5 build (no P_* columns) cannot give the causal observation."""
    x = _x().drop(columns=["P_AGE_S"])
    w, _ = _one_window(_x())
    try:
        LD.make_arrays(x, w)
    except AssertionError:
        LD.make_arrays(x, w, causal=False)
        return
    raise AssertionError("build χωρίς P_* έγινε δεκτό σε αιτιατή λειτουργία")


def test_cog_ignored_when_sog_missing():
    """v1.3: COG_VALID with missing SOG comes from central differences (next minute) -> not causal."""
    x = _x()
    w, r0 = _one_window(x)
    x.loc[r0 + 4, "sog"] = np.nan                     # COG_VALID stays 1 in the build
    obs = LD.make_arrays(x, w)[0]
    k = 4
    i_cv, i_sin, i_cos = (LD.FEATURES.index(f) for f in ("cog_valid", "sin_cog", "cos_cog"))
    assert obs[0, k, i_cv] == 0 and obs[0, k, i_sin] == 0 and obs[0, k, i_cos] == 0, obs[0, k]
    assert obs[0, k - 1, i_cv] == 1


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
    print(f"\n{len(tests) - bad} πέρασαν, {bad} απέτυχαν από {len(tests)}")
    sys.exit(1 if bad else 0)