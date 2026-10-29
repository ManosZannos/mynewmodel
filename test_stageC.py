#!/usr/bin/env python3
"""test_stageC.py — V1-Γ: tests του σταδίου Γ σε συνθετική γεωμετρία, χωρίς OSM (python test_stageC.py)."""
import math, sys, traceback
import numpy as np
import shapely
from shapely import box, LineString
import stageC_shore as C

CFG = C.DEFAULTS
CAP_SH, CAP_WF, CAP_LC = CFG["d_shore_cap_m"], CFG["d_waterfront_cap_m"], CFG["d_land_cog_cap_m"]


def _G(land, water=(), wf=()):
    return C.ShoreGeom.build(np.array(list(land), dtype=object), np.array(list(water), dtype=object),
                             np.array(list(wf), dtype=object), CFG)


def _f(G, pts, az=None, cogv=None):
    p = np.asarray(pts, float)
    az = np.zeros(len(p)) if az is None else np.asarray(az, float)
    cogv = np.ones(len(p), int) if cogv is None else np.asarray(cogv)
    return C.compute_features(p[:, 0], p[:, 1], az, cogv, G, CFG)


ISLAND = _G([box(-500, -500, 500, 500)])


def test_distance_and_direction_island():
    f = _f(ISLAND, [(1500, 0), (0, 2000), (-900, -900)])
    assert np.allclose(f["d_shore"], [1000, 1500, 400 * math.sqrt(2)], atol=1e-6), f["d_shore"]
    u = f["v_shore"] / np.linalg.norm(f["v_shore"], axis=1, keepdims=True)
    assert np.allclose(u[0], [-1, 0]) and np.allclose(u[1], [0, -1]), u
    assert not f["inside"].any()


def test_on_land_depth():
    f = _f(ISLAND, [(0, 0), (100, 0), (499, 0)])
    assert f["inside"].all()
    assert np.allclose(f["d_shore"], [500, 400, 1], atol=1e-6), f["d_shore"]


def test_land_cog_ray():
    pts = [(1500, 0), (1500, 0), (1500, 0), (0, 0), (8000, 0), (1500, 0)]
    az = [270, 90, 270, 0, 270, 280]
    cogv = [1, 1, 0, 1, 1, 1]
    d = _f(ISLAND, pts, az, cogv)["d_land_cog"]
    assert abs(d[0] - 1000) < 1e-6, d[0]                 # κατευθείαν προς το νησί
    assert d[1] == CAP_LC, d[1]                           # αντίθετη κατεύθυνση
    assert np.isnan(d[2])                                 # άκυρο COG
    assert d[3] == 0.0                                    # μέσα στην ξηρά
    assert d[4] == CAP_LC                                 # ξηρά πέρα από 5 km
    assert abs(d[5] - 1000 / math.sin(math.radians(80))) < 1e-6, d[5]   # πλάγια ακτίνα


def test_strait_not_blocked():
    G = _G([box(-1000, -500, -75, 500), box(75, -500, 1000, 500)])
    d = _f(G, [(0, -3000), (500, -3000)], [0, 0])["d_land_cog"]
    assert d[0] == CAP_LC, f"η ακτίνα δεν περνά το στενό 150 m: {d[0]}"
    assert abs(d[1] - 2500) < 1e-6, d[1]


def test_concave_harbor():
    land = shapely.difference(box(-1000, -1000, 1000, 1000), box(-200, -1000, 200, 500))   # κόλπος ανοιχτός νότια
    G = _G([land])
    f = _f(G, [(0, 0)], [0])
    assert not f["inside"][0]
    assert abs(f["d_shore"][0] - 200) < 1e-6, f["d_shore"]
    assert abs(abs(f["v_shore"][0, 0]) - 200) < 1e-6 and abs(f["v_shore"][0, 1]) < 1e-6, f["v_shore"]
    assert abs(f["d_land_cog"][0] - 500) < 1e-6, f["d_land_cog"]     # κεφαλή του κόλπου


def test_water_subtracted_channel():
    G = _G([box(-5000, -5000, 5000, 5000)], water=[box(-100, -6000, 100, 6000)])
    f = _f(G, [(0, 0), (300, 0)], [0, 0])
    assert not f["inside"][0], "κανάλι (πολύγωνο νερού) μετρήθηκε ως ξηρά"
    assert abs(f["d_shore"][0] - 100) < 1e-6, f["d_shore"]
    assert f["d_land_cog"][0] == CAP_LC, "ακτίνα κατά μήκος του καναλιού δεν πρέπει να «χτυπά» ξηρά"
    assert f["inside"][1] and abs(f["d_shore"][1] - 200) < 1e-6


def test_split_seams_are_not_shore():
    G = _G([box(-1000, -1000, 0, 1000), box(0, -1000, 1000, 1000)])      # όπως τα split land polygons
    f = _f(G, [(10, 0), (0, 1200)])
    assert f["inside"][0] and abs(f["d_shore"][0] - 990) < 1e-6, f"η ραφή μετρήθηκε ως ακτή: {f['d_shore'][0]}"
    assert abs(f["d_shore"][1] - 200) < 1e-6


def test_waterfront_distance_and_caps():
    G = _G([box(50_000, 0, 51_000, 1000)], wf=[LineString([(0, 0), (0, 300)])])
    f = _f(G, [(100, 150), (10_000, 0)])
    assert abs(f["d_wf"][0] - 100) < 1e-6 and f["d_wf"][1] == CAP_WF, f["d_wf"]
    assert (f["d_shore"] == CAP_SH).all(), "πέρα από το όριο D_SHORE = cap"


def test_cache_roundtrip_same_features():
    G2 = C.ShoreGeom(shapely.from_wkb(shapely.to_wkb(ISLAND.land)), ISLAND.wf, CFG)
    a, b = _f(ISLAND, [(1500, 0), (0, 0)], [270, 0]), _f(G2, [(1500, 0), (0, 0)], [270, 0])
    for k in ("d_shore", "d_land_cog", "inside"):
        assert np.array_equal(a[k], b[k]), k


def test_classify_tags():
    c = lambda t: C.classify_tags(t, CFG)
    assert c({"natural": "water"}) == {"water"}
    assert c({"waterway": "riverbank"}) == {"water"}
    assert c({"waterway": "canal"}) == set(), "το waterway=canal είναι άξονας, όχι όχθη"
    assert c({"man_made": "pier"}) == {"waterfront"} and c({"landuse": "port"}) == {"waterfront"}
    assert c({"natural": "coastline"}) == set()


def test_projection_scale_and_convergence():
    from pyproj import Geod
    lat0, lon0 = 32.5, -117.5
    tr, geod = C.make_transformer(lat0, lon0), Geod(ellps="WGS84")
    for lat, lon in ((33.5, -115.0), (30.2, -120.2), (34.9, -117.5)):
        gam = C.convergence_deg(lat, lon, lon0)
        x1, y1 = tr.transform(lon, lat)
        for A in (0, 45, 90, 200, 315):
            lo2, la2, _ = geod.fwd(lon, lat, A, 3000.0)
            x2, y2 = tr.transform(lo2, la2)
            az_grid = math.degrees(math.atan2(x2 - x1, y2 - y1))
            err = (az_grid - (A - gam) + 180) % 360 - 180
            assert abs(err) < 0.05, f"σύγκλιση: ({lat},{lon}) A={A} σφάλμα {err:.3f}°"
            d = math.hypot(x2 - x1, y2 - y1) / C.scale_k(x1)
            assert abs(d - 3000) / 3000 < 1e-4, f"κλίμακα: ({lat},{lon}) {d:.2f} m"


def test_features_latlon_end_to_end():
    lat0, lon0 = 32.5, -117.5
    tr = C.make_transformer(lat0, lon0)
    G = _G([box(1000, -200, 1400, 200)])                                  # νησί 1 km ανατολικά του κέντρου
    out = C.features_latlon(np.array([lat0, lat0]), np.array([lon0, lon0]), np.array([90.0, 0.0]),
                            np.array([1, 0]), G, tr, lon0, CFG)
    assert list(out.columns) == C.OUT_COLS
    r = out.iloc[0]
    assert abs(r["D_SHORE"] - 1000) < 0.5 and abs(r["D_LAND_COG"] - 1000) < 0.5, r.to_dict()
    assert abs(r["SHORE_DX"] - 1) < 1e-3 and abs(r["SHORE_DY"]) < 1e-3 and r["ON_LAND"] == 0
    assert np.isnan(out.iloc[1]["D_LAND_COG"])
    assert not out[["D_SHORE", "ON_LAND_DEPTH_M", "SHORE_DX", "SHORE_DY"]].isna().any().any()


def test_cache_key_changes_with_osm_inputs():
    """v1.0: η cache αγνοούσε νέο .pbf (π.χ. Λουιζιάνα) → σιωπηλά παλιά γεωμετρία."""
    import tempfile, shutil, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    try:
        (d / "land-polygons-split-4326.zip").write_bytes(b"x" * 10)
        (d / "texas-latest.osm.pbf").write_bytes(b"x" * 20)
        bb = (-97.5, 26.0, -91.5, 32.0)
        k1 = C.cache_key(bb, CFG, str(d))
        assert C.cache_key(bb, CFG, str(d)) == k1, "μη ντετερμινιστικό κλειδί"
        (d / "prepared_dummy.parquet").write_bytes(b"y")            # η ίδια η cache δεν επηρεάζει το κλειδί
        assert C.cache_key(bb, CFG, str(d)) == k1
        (d / "louisiana-latest.osm.pbf").write_bytes(b"x" * 30)
        assert C.cache_key(bb, CFG, str(d)) != k1, "νέο pbf δεν άλλαξε το κλειδί της cache"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_bench_breakdown_moving_vs_stationary():
    import pandas as pd
    df = pd.DataFrame({"sog": [0.0, 0.1, 5.0, 8.0]})
    feat = pd.DataFrame({"ON_LAND": [1, 0, 1, 0], "ON_LAND_DEPTH_M": [12.0, 0, 40.0, 0]})
    b = C.bench_breakdown(df, feat)
    assert b["κινούμενα"]["ON_LAND"] == 0.5 and b["ακίνητα"]["ON_LAND"] == 0.5
    assert b["κινούμενα"]["βάθος p50/p90/p99 (m)"][0] == 40.0


def test_same_region_same_geometry_across_months():
    """v1.1: SoCal Μαρ. (έως −115.07°) και Δεκ. (έως −115.65°) έπαιρναν διαφορετικό κουτί/προβολή."""
    mar = [-120.109, 29.911, -115.065, 34.708]
    dec = [-120.109, 29.910, -115.651, 34.978]
    assert C.geo_bbox(mar, CFG)[0] == C.geo_bbox(dec, CFG)[0], (C.geo_bbox(mar, CFG)[0], C.geo_bbox(dec, CFG)[0])


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