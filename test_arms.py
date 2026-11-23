#!/usr/bin/env python3
"""test_arms.py — tests των εφεδρικών σκελών (stageE_arms.py, phase3_pool.py, phase3_sdf.py, phase3_sdf_habit.py) σε συνθετικά δεδομένα.
Τα tests με «torch» ή «shapely» στο όνομα θέλουν την αντίστοιχη βιβλιοθήκη· τα υπόλοιπα τρέχουν και χωρίς."""
import shutil, sys, tempfile, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import loader as LD
import phase3 as P3
import stageD_field as D
import stageE_arms as E
from test_loader import _track
from test_stageD import _build
from test_phase3 import _turns, _ARCH

CFG = D.DEFAULTS


def test_neighbour_features_course_frame():
    """Target heading east at 300 m/min: a vessel 1 km north is on the port side (+y), one 2 km east is ahead (+x).
    Nearest first, the target itself and anything beyond the radius are not neighbours, the rest is padding."""
    tp, tv = np.zeros((1, 2)), np.array([[300.0, 0.0]])
    cp = np.array([[2000.0, 0.0], [0.0, 1000.0], [0.0, 0.0], [0.0, -7000.0]])
    cvel = np.array([[300.0, 0.0], [300.0, 100.0], [300.0, 0.0], [0.0, 0.0]])
    f, v = E.neighbour_features(tp, tv, np.array([90.0]), np.array([7]), cp, cvel, np.array([1, 2, 7, 3]), CFG)
    assert f.shape == (1, E.NB_K, 5) and v[0].tolist() == [1, 1] + [0] * (E.NB_K - 2)
    assert np.allclose(f[0, 0], [0, 0.2, 0, 100 / E.VEL_SCALE, 0.2], atol=1e-9), f[0, 0]      # 1 km to port, closing sideways
    assert np.allclose(f[0, 1], [0.4, 0, 0, 0, 0.4], atol=1e-9), f[0, 1]                      # 2 km ahead, same velocity
    assert not f[0, 2:].any()


def test_sdf_patches_follow_the_course():
    """Raster with land east of lon0 + 0.02°: ahead (first rows of the patch) is land when heading east,
    the starboard half (second axis, low index = starboard) is land when heading north."""
    lats, lons, dlat, dlon = E.sdf_grid(32.9, 33.1, -118.1, -117.9, 100.0)
    sdf = np.where(lons[None, :] > -117.98, -3000.0, 3000.0) * np.ones((len(lats), 1))
    r = {"sdf": sdf.astype(np.float16), "lat0": lats[0], "lon0": lons[0], "dlat": dlat, "dlon": dlon}
    p = E.sdf_patches(r, [33.0, 33.0], [-118.0, -118.0], [0.0, np.pi / 2]).reshape(2, E.SDF["n"], E.SDF["n"])
    assert p.min() == -0.6 and p.max() == 0.6
    assert (p[0, -8:] < 0).all() and (p[0, :8] > 0).all(), "πορεία ανατολικά: ξηρά μπροστά, νερό πίσω"
    assert (p[1, :, :8] < 0).all() and (p[1, :, -8:] > 0).all(), "πορεία βόρεια: ξηρά δεξιά, νερό αριστερά"


def test_neighbours_end_to_end():
    d = Path(tempfile.mkdtemp())
    try:
        w = _build(d, [_track(1, 60), _track(2, 60, lat0=33.005), _track(3, 60, lat0=34.0)])
        meta = E.main(["nb", str(d)])
        assert meta["funnel"] == {"eligible_windows": len(w), "rows_out": len(w)}
        ids = np.sort(w["window_id"].to_numpy())
        c = E.load_neighbours(str(d), ids)
        m = w.set_index("window_id").loc[ids, "mmsi"].to_numpy()
        assert c.shape == (len(w), E.NB_K * 5 + E.NB_K)
        assert (c[m == 1, -E.NB_K] == 1).all() and not c[m == 3, -E.NB_K:].any(), "το 1 έχει γείτονα, το 3 κανέναν"
        assert np.allclose(c[m == 1, 1], 556 / CFG["radius_m"], atol=0.01), "το 2 είναι ~556 m αριστερά του 1"
        assert np.allclose(c[m == 2, 1], -556 / CFG["radius_m"], atol=0.01)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_shapely_sdf_raster_sign():
    """Island 1 km wide at the centre: negative inside, positive outside, about 500 m at its centre."""
    import stageC_shore as SC
    from shapely import box
    lat0, lon0 = 33.0, -118.0
    G = SC.ShoreGeom.build(np.array([box(-500, -500, 500, 500)], dtype=object), np.array([], dtype=object),
                           np.array([], dtype=object), SC.DEFAULTS)
    lats, lons, _, _ = E.sdf_grid(lat0 - 0.02, lat0 + 0.02, lon0 - 0.02, lon0 + 0.02, 100.0)
    sdf = E.sdf_raster(lats, lons, G, SC.make_transformer(lat0, lon0), lon0).astype(float)
    i, j = np.abs(lats - lat0).argmin(), np.abs(lons - lon0).argmin()
    assert -520 < sdf[i, j] < -400 and sdf[0, 0] > 1000 and abs((sdf < 0).mean() - (1000 / 4450) ** 2) < 0.02


def _reacts(module, c, c2):
    """Untrained arm = CV exactly; after perturbing the last layer the prediction depends on the context."""
    import torch
    p = P3.prepare(_turns([1, 2], 0))
    p = {k: v[:len(c)] for k, v in p.items()} | {"c": c}
    net = P3.make_net({**_ARCH, "n_ctx": c.shape[1]}, ctx_module=module)
    assert np.array_equal(P3.predict(P3.net_fn(net, "cpu"), p), p["cv"])
    torch.manual_seed(0)
    torch.nn.init.normal_(net.head[-1].weight)
    f = P3.net_fn(net, "cpu")
    return np.abs(P3.predict(f, p) - P3.predict(f, {**p, "c": c2})).max()


def test_torch_pool_arm():
    import phase3_pool as PP
    rng = np.random.default_rng(0)
    feat = rng.normal(size=(6, E.NB_K, 5)).astype(np.float32)
    valid = np.zeros((6, E.NB_K), np.float32)
    valid[:, :3] = 1
    c = np.concatenate([feat.reshape(6, -1), valid], 1)
    perm = [2, 0, 1] + list(range(3, E.NB_K))                    # same neighbours in another order
    c_perm = np.concatenate([feat[:, perm].reshape(6, -1), valid], 1)
    garbage = c.copy()
    garbage[:, 15:E.NB_K * 5] = 99.0                             # values in padded slots must not matter
    none = np.zeros_like(c)
    assert _reacts(PP.ctx_module, c, c_perm) < 1e-3, "το pooling εξαρτάται από τη σειρά των γειτόνων"
    assert _reacts(PP.ctx_module, c, garbage) < 1e-3, "μη έγκυρες θέσεις επηρεάζουν την έξοδο"
    assert _reacts(PP.ctx_module, c, none) > 1.0, "οι γείτονες αγνοούνται"


def test_torch_sdf_arm():
    import phase3_sdf as PS
    rng = np.random.default_rng(0)
    c = rng.uniform(-1, 1, size=(6, PS.N * PS.N)).astype(np.float32)
    assert _reacts(PS.ctx_module, c, -c) > 1.0, "το patch αγνοείται"


def test_torch_arms_end_to_end():
    """Both arms through the shared main(): data products -> training -> JSON; a checkpoint of one arm is refused by the other."""
    import phase3_pool as PP
    import phase3_sdf as PS
    d = Path(tempfile.mkdtemp())
    try:
        rng = np.random.default_rng(1)
        _build(d, [_track(m, 60, turn_deg_per_min=float(rng.uniform(-3, 3)), lat0=33.0 + 0.004 * m) for m in range(1, 13)], n_train=8)
        E.main(["nb", str(d)])
        lats, lons, dlat, dlon = E.sdf_grid(32.8, 33.3, -118.3, -117.5, 200.0)
        (d / "stageE").mkdir(exist_ok=True)
        np.savez(d / "stageE" / "sdf.npz", sdf=np.outer(np.linspace(-4000, 4000, len(lats)), np.ones(len(lons))).astype(np.float16),
                 lat0=lats[0], lon0=lons[0], dlat=dlat, dlon=dlon)
        common = ["--train_build", str(d), "--out_dir", str(d / "out"), "--hidden", "16", "--epochs", "2", "--batch", "16"]
        rp, rs = PP.main(common), PS.main(common)
        assert rp["arch"]["channels"] == ["pool"] and rp["arch"]["n_ctx"] == E.NB_K * 6 and rp["tag"].startswith("m_pool_")
        assert rs["arch"]["channels"] == ["sdf"] and rs["arch"]["n_ctx"] == PS.N ** 2 and rs["tag"].startswith("m_sdf_")
        assert PP.main(common + ["--load_model", rp["model_path"], "--tag", "reload"])["metrics"] == rp["metrics"]
        try:
            PS.main(common + ["--load_model", rp["model_path"]])
        except (SystemExit, RuntimeError):
            return
        raise AssertionError("checkpoint του pool φορτώθηκε από το sdf")
    finally:
        shutil.rmtree(d, ignore_errors=True)



def _sdf_habit_build(d):
    """Synthetic build with stageD field (habit, no land) and a stageE SDF raster."""
    rng = np.random.default_rng(1)
    _build(d, [_track(m, 60, turn_deg_per_min=float(rng.uniform(-3, 3)), lat0=33.0 + 0.004 * m) for m in range(1, 13)], n_train=8)
    D.main([str(d), "--skip_land"])
    lats, lons, dlat, dlon = E.sdf_grid(32.8, 33.3, -118.3, -117.5, 200.0)
    (d / "stageE").mkdir(exist_ok=True)
    np.savez(d / "stageE" / "sdf.npz", sdf=np.outer(np.linspace(-4000, 4000, len(lats)), np.ones(len(lons))).astype(np.float16),
             lat0=lats[0], lon0=lons[0], dlat=dlat, dlon=dlon)


def test_sdf_habit_context_layout():
    """Context = [SDF patch | habit of the given field dir]; another field dir changes only the habit part."""
    import phase3_sdf_habit as SH
    d = Path(tempfile.mkdtemp())
    try:
        _sdf_habit_build(d)
        ev = LD.load_split(str(d), "vessel", "val")
        c = SH.ctx_loader(str(d), ev)
        assert c.shape == (len(ev["meta"]), SH.N2 + SH.NH) and c.dtype == np.float32
        f = pd.read_parquet(d / "stageD" / "field.parquet")
        f[[n for n in f.columns if n.startswith("hab_")]] = 0.5
        (d / "stageD_alt").mkdir()
        f.to_parquet(d / "stageD_alt" / "field.parquet")
        c2 = SH.ctx_loader(str(d), ev, "stageD_alt")
        assert np.array_equal(c2[:, :SH.N2], c[:, :SH.N2]) and (c2[:, SH.N2:] == 0.5).all()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_torch_sdf_habit_arm():
    """Untrained = CV; the prediction reacts to the patch and to the habit separately."""
    import phase3_sdf_habit as SH
    rng = np.random.default_rng(0)
    c = rng.uniform(-1, 1, size=(6, SH.N2 + SH.NH)).astype(np.float32)
    c_sdf, c_hab = c.copy(), c.copy()
    c_sdf[:, :SH.N2] *= -1
    c_hab[:, SH.N2:] = rng.uniform(0, 1, size=(6, SH.NH))
    assert _reacts(SH.ctx_module, c, c_sdf) > 1.0, "το patch αγνοείται"
    assert _reacts(SH.ctx_module, c, c_hab) > 1.0, "η συνήθεια αγνοείται"


def test_torch_sdf_habit_end_to_end_eval_field_dir():
    """phase3 v0.7: --eval_field_dir reaches the arm's ctx_loader. Old code (v0.6) called ctx_loader(build, d), so the
    reload read stageD, which is removed here, and failed."""
    import phase3_sdf_habit as SH
    d = Path(tempfile.mkdtemp())
    try:
        _sdf_habit_build(d)
        common = ["--train_build", str(d), "--out_dir", str(d / "out"), "--hidden", "16", "--epochs", "2", "--batch", "16"]
        r = SH.main(common)
        assert r["arch"]["channels"] == ["sdf+habit"] and r["arch"]["n_ctx"] == SH.N2 + SH.NH and r["tag"].startswith("m_sdf+habit_")
        (d / "stageD").rename(d / "stageD_alt")
        rh = SH.main(common + ["--load_model", r["model_path"], "--eval_field_dir", "stageD_alt"])
        assert rh["metrics"] == r["metrics"] and rh["tag"].endswith("_evstageD_alt")
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
