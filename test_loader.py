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
                         "GAP_IMP": 0, "IN_CORE": 1, "MMSI_QC": 0, "vessel_cat": "cargo", "transceiver": "A"})


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
    assert obs.shape == (n, LD.OBS, len(LD.FEATURES)) and fut.shape == (n, LD.PRED, 2) and pos.shape == (n, LD.OBS, 2)
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
            g[SB.COLS].to_csv(d / f"day_{day}.csv.gz", index=False)
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
