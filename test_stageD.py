#!/usr/bin/env python3
"""test_stageD.py — tests του σταδίου Δ (πολικό πεδίο) σε συνθετικά δεδομένα (python test_stageD.py).
Το test με «shapely» στο όνομα θέλει shapely/pyproj (κανάλι ξηράς)· τα υπόλοιπα τρέχουν και χωρίς."""
import shutil, sys, tempfile, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import loader as LD
import stageB_manifests as SB
import baselines as BL
import stageD_field as D
from test_loader import _track

CFG = D.DEFAULTS
K = CFG["K"]


def test_sector():
    assert D.sector([0, 22.5, 90, 350, 180], 0.0, K).tolist() == [0, 1, 4, 0, 8]
    assert D.sector([100, 135, 80], 90.0, K).tolist() == [0, 2, 0], "σχετικά με την πορεία"


def test_land_values():
    """Time to reach at the current speed; a ray that hits nothing (>= cap) is 0 whatever the speed."""
    v = D.land_values(np.array([[600.0, CFG["land_cap_m"], 12000.0]]), np.array([300.0]), CFG)
    assert np.allclose(v, [[0.9, 0.0, 0.0]]), v
    fast = D.land_values(np.array([[CFG["land_cap_m"]]]), np.array([2000.0]), CFG)      # 20 km in 10 min, but no hit
    assert fast[0, 0] == 0.0


def test_vessels_channel_values():
    """Target at the origin heading north at 300 m/min."""
    tp, tv = np.zeros((1, 2)), np.array([[0.0, 300.0]])
    cp = np.array([[0.0, 3000.0],     # A ahead, head-on: TCPA 5 min, DCPA 0
                   [1000.0, 0.0],     # B abeam to starboard, same velocity: closest now, 1000 m
                   [-6000.0, 0.0],    # C outside the radius
                   [0.0, 0.0],        # D the target itself
                   [0.0, 4000.0]])    # E ahead, same velocity: closest now, 4000 m -> less critical than A
    cvel = np.array([[0.0, -300.0], [0.0, 300.0], [0.0, 300.0], [0.0, 300.0], [0.0, 300.0]])
    vt, vd = D.vessel_values(tp, tv, np.array([0.0]), np.array([7]), cp, cvel, np.array([1, 2, 3, 7, 5]), CFG)
    assert np.isclose(vt[0, 0], 0.75) and np.isclose(vd[0, 0], 1.0), (vt[0, 0], vd[0, 0])
    assert np.isclose(vt[0, 4], 1.0) and np.isclose(vd[0, 4], 0.8), (vt[0, 4], vd[0, 4])
    rest = [k for k in range(K) if k not in (0, 4)]
    assert not vt[0, rest].any() and not vd[0, rest].any(), "εκτός ακτίνας και ο ίδιος ο στόχος δεν μετρούν"
    # course east: the same neighbour to the north is now on the port beam (sector 12)
    vt2, _ = D.vessel_values(tp, np.array([[300.0, 0.0]]), np.array([90.0]), np.array([7]),
                             np.array([[0.0, 1000.0]]), np.array([[300.0, 0.0]]), np.array([1]), CFG)
    assert vt2[0, 12] == 1.0 and vt2[0].sum() == 1.0


def test_habit_leaves_own_vessel_out():
    lat, lon = np.full(5, 33.0), np.full(5, -118.0)
    keys = D.habit_keys(lat, lon, np.full(5, 90.0), CFG)
    tab = D.habit_table(keys, [0, 0, 0, 1, 1], [1, 1, 1, 2, 2], [0] * 5)
    q = D.habit_keys([33.0, 33.0, 33.5], [-118.0, -118.0, -118.0], [90.0, 90.0, 90.0], CFG)
    h, n = D.habit_values(tab, q, [1, 9, 9], CFG)
    assert np.allclose(h[0, :2], [0.0, 1.0]) and np.isclose(n[0], 2 / CFG["count_cap"]), "το πλοίο 1 βλέπει μόνο το 2"
    assert np.allclose(h[1, :2], [0.6, 0.4]) and np.isclose(n[1], 5 / CFG["count_cap"])
    assert not h[2].any() and n[2] == 0, "κελί χωρίς ιστορικό"
    other_course = D.habit_keys([33.0], [-118.0], [270.0], CFG)
    assert D.habit_values(tab, other_course, [9], CFG)[1][0] == 0, "άλλη πορεία στο ίδιο κελί"


def test_habit_gap_days_leaves_out_same_days():
    """Vessel 2 passes the cell on days 0 and 5, vessel 3 on day 1, vessel 1 (the query's own) on day 0.
    gap 0 for a query on day 0 drops only vessel 2's day-0 counts; gap 1 also vessel 3; the own vessel stays out."""
    keys = D.habit_keys(np.full(5, 33.0), np.full(5, -118.0), np.full(5, 90.0), CFG)
    tab = D.habit_table(keys, [0, 1, 2, 3, 4], [2, 2, 3, 1, 1], [0, 5, 1, 0, 0])
    q = D.habit_keys([33.0], [-118.0], [90.0], CFG)

    def n(g):
        return D.habit_values(tab, q, [1], CFG, [0], g)[1][0] * CFG["count_cap"]
    assert round(n(None)) == 3 and round(n(0)) == 2 and round(n(1)) == 1, (n(None), n(0), n(1))
    h = D.habit_values(tab, q, [1], CFG, [0], 1)[0][0]
    assert h[1] == 1.0 and h.sum() == 1.0, "μένει μόνο το πλοίο 2 της μέρας 5 (τομέας 1)"


def test_habit_cells_are_about_cell_m():
    step = np.degrees(CFG["cell_m"] / LD.R_EARTH)
    k = D.habit_keys([33.0, 33.0 + 3 * step, 33.0], [-118.0, -118.0, -118.0 + 3 * step / np.cos(np.radians(33.0))],
                     [0.0, 0.0, 0.0], CFG)
    assert k["ci"][1] - k["ci"][0] == 3 and k["cj"][2] - k["cj"][0] == 3, k


def _build(d, tracks, n_train=3):
    """Vessels with mmsi <= n_train are train. Training needs an early-stopping vessel in train (mmsi 8 is one)."""
    x = pd.concat(tracks, ignore_index=True)
    for day, g in x.groupby(x["t"].dt.strftime("%Y-%m-%d")):
        g[SB.COLS + LD.CAUSAL_COLS].to_csv(d / f"day_{day}.csv.gz", index=False)
    w = SB.build_windows(SB.load_build(str(d)), SB.DEFAULTS)
    w["tier_cv"] = "easy"
    (d / "stageB").mkdir()
    w.to_csv(d / "stageB" / "windows.csv.gz", index=False)
    t = w[w["eligible"] == 1][["window_id", "mmsi", "seg_id", "t_start"]].assign(capped=1)
    t["split"] = np.where(t["mmsi"] <= n_train, "train", "val")
    t.to_csv(d / "stageB" / "targets_vessel.csv.gz", index=False)
    return w[w["eligible"] == 1]


def test_end_to_end_skip_land():
    """Files -> field.parquet: one row per eligible window, values in [0, 1], neighbours in the right sector,
    habit table only from train vessels that are not early-stopping vessels."""
    d = Path(tempfile.mkdtemp())
    try:
        # 1 and 2 sail east side by side, 2 about 550 m north of 1; 3 and 4 are alone, far away
        w = _build(d, [_track(1, 60), _track(2, 60, lat0=33.005), _track(3, 60, lat0=34.0), _track(4, 60, lat0=35.0)])
        meta = D.main([str(d), "--skip_land"])
        f = pd.read_parquet(d / "stageD" / "field.parquet").set_index("window_id")
        assert meta["funnel"] == {"eligible_windows": len(w), "rows_out": len(f)} and sorted(f.index) == sorted(w["window_id"])
        assert len(f.columns) == 4 * K + 1 and not f.filter(like="land_").to_numpy().any()
        vd = f.filter(like="vd_").to_numpy()
        m = w.set_index("window_id").loc[f.index, "mmsi"].to_numpy()
        assert (vd[m == 1].argmax(1) == 12).all() and (vd[m == 2].argmax(1) == 4).all(), "2 αριστερά του 1, 1 δεξιά του 2"
        assert np.allclose(vd[m == 1].max(1), 1 - 556 / CFG["radius_m"], atol=0.01) and not vd[m >= 3].any()
        tab = pd.read_parquet(d / "stageD" / "habit_table.parquet")
        assert set(tab["mmsi"]) <= {1, 2, 3} and (BL._hash01(tab["mmsi"].to_numpy()) >= 0.1).all()
        hn = f["hab_n"].to_numpy()
        assert not hn[m == 4].any(), "πλοίο του val σε περιοχή χωρίς train: κανένα ιστορικό"
        own_only = [v for v in (1, 2, 3) if v in set(tab["mmsi"])]
        D.main([str(d), "--skip_land", "--habit_gap_days", "0", "--out_name", "stageD_gap0"])
        g0 = pd.read_parquet(d / "stageD_gap0" / "field.parquet").set_index("window_id").loc[f.index]
        assert not g0["hab_n"].any(), "όλα τα ίχνη είναι της ίδιας μέρας: χωρίς αυτή δεν μένει ιστορικό"
        assert np.array_equal(g0.filter(like="vd_").to_numpy(), f.filter(like="vd_").to_numpy())
        assert not hn[np.isin(m, own_only)].any(), "κάθε πλοίο του train είναι μόνο του στα κελιά του: αφαιρείται"
        # a second build reads the table of the first: the same vessels are now 'other' traffic
        d2 = Path(tempfile.mkdtemp())
        try:
            _build(d2, [_track(50, 60), _track(60, 60, lat0=34.0)])
            D.main([str(d2), "--skip_land", "--prior_build", str(d)])
            f2 = pd.read_parquet(d2 / "stageD" / "field.parquet")
            assert not (d2 / "stageD" / "habit_table.parquet").exists()
            if own_only:
                assert f2["hab_n"].max() > 0 and np.isclose(f2.loc[f2["hab_n"] > 0, "hab_00"].min(), 1.0), "ευθεία πορεία: τομέας 0"
        finally:
            shutil.rmtree(d2, ignore_errors=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_field_report():
    """err_share = share of the group's FDE inside the windows where the channel is non-zero (ceiling of its gain);
    end to end on a file build with the CV as reference."""
    import field_report as FR
    r = FR.report([100.0, 300.0, 600.0, 0.0], {"vessels": np.array([0, 1, 1, 0], bool)}, np.array(["a", "a", "b", "b"]))
    r = r.set_index("group")
    assert np.isclose(r.loc["all", "err_share"], 0.9) and np.isclose(r.loc["all", "covered"], 0.5)
    assert np.isclose(r.loc["a", "err_share"], 0.75) and r.loc["a", "FDE_in"] == 300 and r.loc["a", "FDE_out"] == 100
    assert r.loc["b", "err_share"] == 1.0 and r.loc["b", "n"] == 2
    d = Path(tempfile.mkdtemp())
    try:
        _build(d, [_track(1, 60), _track(2, 60, lat0=33.005), _track(3, 60, lat0=34.0),
                   _track(4, 60), _track(5, 60, lat0=35.0)])
        D.main([str(d), "--skip_land"])
        r = FR.main([str(d)]).set_index("channel")                  # val = vessels 4 (on the track of 1) and 5 (alone)
        assert r.loc["land", "covered"] == 0 and 0 < r.loc["vessels", "covered"] < 1, r
        assert r.loc["habit", "covered"] == r.loc["vessels", "covered"], "το 4 έχει γείτονες και ιστορικό, το 5 τίποτα"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_shapely_land_rays_follow_the_course():
    """Land 2 km to the east: sector 0 when heading east, sector 4 (starboard) when heading north."""
    import stageC_shore as SC
    from shapely import box
    lat0, lon0 = 33.0, -118.0
    tr = SC.make_transformer(lat0, lon0)
    G = SC.ShoreGeom.build(np.array([box(2000, -5000, 3000, 5000)], dtype=object), np.array([], dtype=object),
                           np.array([], dtype=object), SC.DEFAULTS)
    dist = D.land_distances(np.array([lat0, lat0]), np.array([lon0, lon0]), np.array([90.0, 0.0]), G, tr, lon0, CFG)
    assert abs(dist[0, 0] - 2000) < 5 and abs(dist[1, 4] - 2000) < 5, (dist[0, 0], dist[1, 4])
    assert dist[0, 8] == CFG["land_cap_m"] and dist[1, 12] == CFG["land_cap_m"], "προς τα δυτικά τίποτα"
    v = D.land_values(dist, np.array([300.0, 300.0]), CFG)
    assert np.isclose(v[0, 0], 1 - (2000 / 300) / CFG["t_max_min"], atol=0.01) and v[0, 8] == 0


def test_land_diag_groups_and_contrib():
    """land_diag.py (new): ahead bins from the 3 front sectors, 'narrow channel' = land on both beams,
    contributions of the groups add up to the total difference."""
    import land_diag as LDG
    f = pd.DataFrame(np.zeros((3, 16)), columns=[f"land_{k:02d}" for k in range(16)])
    f["hab_n"] = [0.0, 0.1, 0.6]                         # 0, 5 and 30 vessels
    f.loc[1, "land_15"] = 0.9                            # land ahead (port bow), no beams
    f.loc[2, ["land_04", "land_12", "land_01"]] = [0.8, 0.6, 0.2]
    g = LDG.land_groups(f)
    assert list(g["ahead"]) == ["0 (καμία)", "(2/3, 1]", "(0, 1/3]"], list(g["ahead"])
    assert g["channel"].tolist() == [False, False, True]
    assert list(g["habit"]) == ["0 πλοία", "1–5", "> 25"], list(g["habit"])
    assert list(g["front"]) == ["καμία κοντά", "μόνο πλάγια", "καμία κοντά"], list(g["front"])
    f.loc[0, "land_00"] = 0.7
    assert LDG.land_groups(f)["front"][0] == "ακριβώς μπροστά"
    t = LDG.table([9, 9, 9, 9], [10, 10, 10, 10], [10, 30, 10, 0], np.array(["x", "x", "y", "y"]))
    assert abs(t["contrib"].sum() - 1) < 1e-9 and t["diff"].tolist() == [10, -5], t


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
