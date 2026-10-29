#!/usr/bin/env python3
"""
stageC_shore.py — Στάδιο Γ της προδιαγραφής: χαρακτηριστικά ακτογραμμής ανά γραμμή του build.

Ανεξάρτητο από μοντέλο και από τα στάδια Α/Β. Γράφει στο <build>/stageC/shore_<osm-date>/:
  day_YYYY-MM-DD.parquet   (mmsi, t) + ON_LAND, ON_LAND_DEPTH_M, D_SHORE, SHORE_DX, SHORE_DY,
                           D_WATERFRONT, D_LAND_COG
  stageC_meta.json         config, εκδόσεις, αρχεία OSM + sha256, προβολή, bbox, commit, σύνοψη

Γεωμετρία (OSM, ένα snapshot ανά φάκελο data/osm/<YYYY-MM-DD>/):
  ξηρά       = land polygons (osmdata, από natural=coastline) ΜΕΙΟΝ πολύγωνα νερού
               (natural=water, waterway=riverbank|dock) από τα extracts .pbf του Geofabrik
  ακτή       = όριο της ξηράς
  waterfront = man_made=pier|breakwater|groyne|quay, landuse=port (όπως EnvShip, για συγκρισιμότητα)

Μέθοδος: ακριβής διανυσματική (shapely 2, STRtree) σε τοπική Transverse Mercator με κέντρο το κουτί.
  D_SHORE      απόσταση από την ακτή (m), με άνω όριο d_shore_cap_m
  SHORE_DX/DY  μοναδιαίο διάνυσμα προς το πλησιέστερο σημείο ακτής (ανατολή, βορράς, αληθής βορράς)
  D_LAND_COG   απόσταση μέχρι την ξηρά κατά μήκος του COG (τομή ακτίνας-ακτής), άνω όριο 5 km,
               NaN όταν COG_VALID = 0, 0 όταν το σημείο είναι μέσα στην ξηρά
  ON_LAND      σημείο μέσα στην ξηρά· ON_LAND_DEPTH_M = βάθος μέσα στην ξηρά (ανοχή = κατώφλι αργότερα)
Οι αποστάσεις διορθώνονται για τον συντελεστή κλίμακας της προβολής και οι γωνίες για τη σύγκλιση
μεσημβρινών (σφάλμα < 0.01% και < 0.05° μέσα στο κουτί).

    python stageC_shore.py dataset/socal_2025-03_210b4127 --osm_dir data/osm/2026-10-30 --bench 1000000
    python stageC_shore.py dataset/socal_2025-03_210b4127 --osm_dir data/osm/2026-10-30 --workers 8
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

R_EARTH = 6_371_000.0

DEFAULTS = {
    "stageC_version": "stageC-v1.0",
    "d_shore_cap_m": 20_000.0,       # ακριβής έως εδώ (το περιθώριο γεωμετρίας το εξασφαλίζει)
    "d_waterfront_cap_m": 5_000.0,
    "d_land_cog_cap_m": 5_000.0,
    "bbox_round_deg": 0.5,           # το κουτί δεδομένων στρογγυλεύεται προς τα έξω → κοινή cache Μαρ./Δεκ.
    "geom_margin_extra_m": 2_000.0,  # περιθώριο γεωμετρίας = d_shore_cap + αυτό
    "piece_vertices": 32,            # τμήματα γραμμών για το STRtree
    "chunk_rows": 400_000,
    "water_tags": {"natural": ["water"], "waterway": ["riverbank", "dock"]},
    "waterfront_tags": {"man_made": ["pier", "breakwater", "groyne", "quay"], "landuse": ["port"]},
}
OUT_COLS = ["ON_LAND", "ON_LAND_DEPTH_M", "D_SHORE", "SHORE_DX", "SHORE_DY", "D_WATERFRONT", "D_LAND_COG"]


# ---------------------------------------------------------------------------
# Προβολή
def tm_proj4(lat0, lon0):
    return f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"


def make_transformer(lat0, lon0):
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4326", tm_proj4(lat0, lon0), always_xy=True)


def convergence_deg(lat, lon, lon0):
    """Γωνία από τον αληθή στον κάναβο βορρά (θετική ανατολικά του κεντρικού μεσημβρινού)."""
    return np.degrees(np.arctan(np.tan(np.radians(np.asarray(lon) - lon0)) * np.sin(np.radians(lat))))


def scale_k(x):
    """Συντελεστής κλίμακας της TM (σφαίρα) στην τετμημένη x: απόσταση_αληθής = απόσταση_κανάβου / k."""
    return np.cosh(np.asarray(x) / R_EARTH)


def project_geoms(geoms, transformer):
    def f(c):
        x, y = transformer.transform(c[:, 0], c[:, 1])
        return np.column_stack([x, y])
    return shapely.transform(geoms, f)


# ---------------------------------------------------------------------------
# Γεωμετρία
def split_lines(geoms, k):
    """Γραμμές/όρια → μικρά LineString έως k κορυφών (αποδοτικό STRtree)."""
    parts = shapely.get_parts(np.atleast_1d(geoms))
    parts = parts[~shapely.is_empty(parts)]
    if len(parts) == 0:
        return np.array([], dtype=object)
    coords, idx = shapely.get_coordinates(parts, return_index=True)
    starts = np.r_[0, np.flatnonzero(np.diff(idx)) + 1, len(idx)]
    out_c, out_i, piece = [], [], 0
    for a, b in zip(starts[:-1], starts[1:]):
        for s in range(a, b - 1, k - 1):
            e = min(s + k, b)
            out_c.append(coords[s:e])
            out_i.append(np.full(e - s, piece))
            piece += 1
    return shapely.linestrings(np.concatenate(out_c), indices=np.concatenate(out_i))


class ShoreGeom:
    """Ξηρά (πολύγωνο), τμήματα ακτής και waterfront σε μέτρα της προβολής."""

    def __init__(self, land, waterfront_lines, cfg):
        self.land = land
        shapely.prepare(self.land)
        k = cfg["piece_vertices"]
        self.shore = split_lines(shapely.boundary(land), k) if not shapely.is_empty(land) else np.array([], dtype=object)
        self.wf = split_lines(waterfront_lines, k) if len(np.atleast_1d(waterfront_lines)) else np.array([], dtype=object)
        self.shore_tree = STRtree(self.shore)
        self.wf_tree = STRtree(self.wf)

    @classmethod
    def build(cls, land_polys, water_polys, waterfront_lines, cfg):
        land = shapely.union_all(shapely.make_valid(np.atleast_1d(land_polys)))   # ενώνει τις ραφές του split
        if len(np.atleast_1d(water_polys)):
            water = shapely.union_all(shapely.make_valid(np.atleast_1d(water_polys)))
            land = shapely.difference(land, water)
        return cls(land, np.atleast_1d(waterfront_lines), cfg)


def _nearest(tree, pieces, pts, cap):
    n = len(pts)
    d = np.full(n, cap, dtype=np.float64)
    v = np.zeros((n, 2), dtype=np.float64)
    if len(pieces) == 0 or n == 0:
        return d, v
    (ii, jj), dist = tree.query_nearest(pts, max_distance=cap, return_distance=True, all_matches=False)
    d[ii] = dist
    c = shapely.get_coordinates(shapely.shortest_line(pts[ii], pieces[jj])).reshape(-1, 2, 2)
    v[ii] = c[:, 1] - c[:, 0]
    return d, v


def compute_features(x, y, az_grid_deg, cog_valid, G, cfg):
    """Χαρακτηριστικά σε μέτρα κανάβου (χωρίς διόρθωση κλίμακας/σύγκλισης). az_grid: COG στον κάναβο."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    pts = shapely.points(x, y)
    inside = shapely.contains_xy(G.land, x, y) if not shapely.is_empty(G.land) else np.zeros(n, bool)

    d_sh, v_sh = _nearest(G.shore_tree, G.shore, pts, cfg["d_shore_cap_m"])
    d_wf, _ = _nearest(G.wf_tree, G.wf, pts, cfg["d_waterfront_cap_m"])

    cap = cfg["d_land_cog_cap_m"]
    valid = np.asarray(cog_valid) == 1
    d_lc = np.full(n, np.nan)
    d_lc[valid] = cap
    cand = np.flatnonzero(valid & ~inside & (d_sh < cap))
    if len(cand) and len(G.shore):
        az = np.radians(np.asarray(az_grid_deg, float)[cand])
        x0, y0 = x[cand], y[cand]
        seg = np.stack([np.column_stack([x0, y0]), np.column_stack([x0 + cap * np.sin(az), y0 + cap * np.cos(az)])], axis=1)
        rays = shapely.linestrings(seg)
        ri, pj = G.shore_tree.query(rays, predicate="intersects")
        res = np.full(len(cand), cap)
        if len(ri):
            inter = shapely.intersection(rays[ri], G.shore[pj])
            dd = shapely.distance(shapely.points(x0[ri], y0[ri]), inter)
            ok = np.isfinite(dd)
            np.minimum.at(res, ri[ok], dd[ok])
        d_lc[cand] = res
    d_lc[valid & inside] = 0.0

    return {"inside": inside, "d_shore": d_sh, "v_shore": v_sh, "d_wf": d_wf, "d_land_cog": d_lc}


def features_latlon(lat, lon, cog, cog_valid, G, transformer, lon0, cfg):
    """Πλήρης υπολογισμός από lat/lon/COG: προβολή, σύγκλιση μεσημβρινών, συντελεστής κλίμακας."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    x, y = transformer.transform(lon, lat)
    x, y = np.asarray(x), np.asarray(y)
    gam = convergence_deg(lat, lon, lon0)
    az_grid = np.asarray(cog, float) - gam
    f = compute_features(x, y, az_grid, cog_valid, G, cfg)
    k = scale_k(x)
    d_sh = f["d_shore"] / k
    d_sh = np.where(f["d_shore"] >= cfg["d_shore_cap_m"], cfg["d_shore_cap_m"], d_sh)
    d_wf = np.where(f["d_wf"] >= cfg["d_waterfront_cap_m"], cfg["d_waterfront_cap_m"], f["d_wf"] / k)
    d_lc = f["d_land_cog"]
    d_lc = np.where(np.isfinite(d_lc) & (d_lc < cfg["d_land_cog_cap_m"]), d_lc / k, d_lc)
    vx, vy = f["v_shore"][:, 0], f["v_shore"][:, 1]
    norm = np.hypot(vx, vy)
    az_true = np.radians(np.degrees(np.arctan2(vx, vy)) + gam)
    has = norm > 0
    dxu = np.where(has, np.sin(az_true), 0.0)
    dyu = np.where(has, np.cos(az_true), 0.0)
    inside = f["inside"]
    return pd.DataFrame({
        "ON_LAND": inside.astype(np.int8),
        "ON_LAND_DEPTH_M": np.where(inside, d_sh, 0.0).astype(np.float32),
        "D_SHORE": d_sh.astype(np.float32),
        "SHORE_DX": dxu.astype(np.float32), "SHORE_DY": dyu.astype(np.float32),
        "D_WATERFRONT": d_wf.astype(np.float32),
        "D_LAND_COG": d_lc.astype(np.float32),
    })


# ---------------------------------------------------------------------------
# OSM
def classify_tags(tags, cfg):
    """→ σύνολο από {'water', 'waterfront'}· tags: dict ή TagList του pyosmium (αρκεί .get)."""
    out = set()
    for key, vals in cfg["water_tags"].items():
        if tags.get(key) in vals:
            out.add("water")
    for key, vals in cfg["waterfront_tags"].items():
        if tags.get(key) in vals:
            out.add("waterfront")
    return out


def _bbox_hit(g, bb):
    b = shapely.bounds(g)
    return not (b[2] < bb[0] or b[0] > bb[2] or b[3] < bb[1] or b[1] > bb[3])


def _pbf_box(path):
    try:
        import osmium
        r = osmium.io.Reader(str(path), osmium.osm.osm_entity_bits.NOTHING)
        b = r.header().box()
        r.close()
        if b.valid():
            return (b.bottom_left.lon, b.bottom_left.lat, b.top_right.lon, b.top_right.lat)
    except Exception:  # noqa: BLE001
        pass
    return None


def read_pbf(path, bb, cfg):
    """Πολύγωνα νερού και γραμμές waterfront (WGS84) μέσα στο bbox, με pyosmium."""
    import osmium
    keys = sorted(set(cfg["water_tags"]) | set(cfg["waterfront_tags"]))
    wkb = osmium.geom.WKBFactory()
    fp = (osmium.FileProcessor(str(path)).with_locations()
          .with_areas(osmium.filter.KeyFilter(*keys)).with_filter(osmium.filter.KeyFilter(*keys)))
    water, wf, n_way, n_bad = [], [], 0, 0
    for o in fp:
        if o.is_area():
            cls = classify_tags(o.tags, cfg)
            if not cls:
                continue
            try:
                g = shapely.from_wkb(wkb.create_multipolygon(o))
            except Exception:  # noqa: BLE001
                continue
            if not _bbox_hit(g, bb):
                continue
            if "water" in cls:
                water.append(g)
            if "waterfront" in cls and not o.from_way():
                wf.append(shapely.boundary(g))
        elif o.is_way():
            cls = classify_tags(o.tags, cfg)
            if "waterfront" not in cls:
                continue
            n_way += 1
            try:
                g = shapely.from_wkb(wkb.create_linestring(o))
            except Exception:  # noqa: BLE001
                n_bad += 1
                continue
            if _bbox_hit(g, bb):
                wf.append(g)
    if n_way and n_bad / n_way > 0.5:
        raise RuntimeError(f"{path}: {n_bad}/{n_way} ways χωρίς θέσεις κόμβων — έλεγξε το with_locations")
    return water, wf, {"waterfront_ways": n_way, "ways_bad_location": n_bad}


def find_land_shp(osm_dir):
    shp = glob.glob(f"{osm_dir}/**/land_polygons.shp", recursive=True)
    if shp:
        return shp[0]
    for z in sorted(glob.glob(f"{osm_dir}/land-polygons-split-4326*.zip")):
        inner = [n for n in zipfile.ZipFile(z).namelist() if n.endswith("land_polygons.shp")]
        if inner:
            return f"/vsizip/{os.path.abspath(z)}/{inner[0]}"
    raise FileNotFoundError(f"land_polygons.shp δεν βρέθηκε στο {osm_dir}")


def sha256(path, bs=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def geo_bbox(data_bb, cfg):
    r = cfg["bbox_round_deg"]
    lo0, la0 = math.floor(data_bb[0] / r) * r, math.floor(data_bb[1] / r) * r
    lo1, la1 = math.ceil(data_bb[2] / r) * r, math.ceil(data_bb[3] / r) * r
    m = cfg["d_shore_cap_m"] + cfg["geom_margin_extra_m"]
    dlat = m / 111_320.0
    dlon = m / (111_320.0 * math.cos(math.radians(max(abs(la0), abs(la1)) + dlat)))
    core = (lo0, la0, lo1, la1)
    return core, (lo0 - dlon, la0 - dlat, lo1 + dlon, la1 + dlat)


def prepare_geometry(osm_dir, core_bb, geom_bb, cfg, cache_path):
    """Φορτώνει ή φτιάχνει την προβεβλημένη γεωμετρία (cache Parquet με WKB)."""
    lat0, lon0 = 0.5 * (core_bb[1] + core_bb[3]), 0.5 * (core_bb[0] + core_bb[2])
    tr = make_transformer(lat0, lon0)
    if cache_path.exists():
        c = pd.read_parquet(cache_path)
        land = shapely.from_wkb(c.loc[c["kind"] == "land", "wkb"].iloc[0])
        wf = shapely.from_wkb(c.loc[c["kind"] == "wf", "wkb"].to_numpy())
        info = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
        return ShoreGeom(land, wf, cfg), tr, (lat0, lon0), info
    import pyogrio
    t0 = time.time()
    shp = find_land_shp(osm_dir)
    _, _, geom_wkb, _ = pyogrio.raw.read(shp, bbox=geom_bb, columns=[])
    land_geo = shapely.clip_by_rect(shapely.from_wkb(geom_wkb), *geom_bb)
    water, wf, pbf_info = [], [], {}
    for pbf in sorted(glob.glob(f"{osm_dir}/*.osm.pbf")):
        box = _pbf_box(pbf)
        if box is not None and (box[2] < geom_bb[0] or box[0] > geom_bb[2] or box[3] < geom_bb[1] or box[1] > geom_bb[3]):
            pbf_info[Path(pbf).name] = "εκτός κουτιού"
            continue
        w, f, inf = read_pbf(pbf, geom_bb, cfg)
        water += w
        wf += f
        pbf_info[Path(pbf).name] = {"water_polys": len(w), "waterfront": len(f), **inf}
    water_geo = shapely.clip_by_rect(np.array(water, dtype=object), *geom_bb) if water else np.array([], dtype=object)
    wf_geo = shapely.clip_by_rect(np.array(wf, dtype=object), *geom_bb) if wf else np.array([], dtype=object)
    G = ShoreGeom.build(project_geoms(land_geo, tr), project_geoms(water_geo, tr) if len(water_geo) else water_geo,
                        project_geoms(wf_geo, tr) if len(wf_geo) else wf_geo, cfg)
    pd.DataFrame({"kind": ["land"] + ["wf"] * len(G.wf),
                  "wkb": [shapely.to_wkb(G.land)] + list(shapely.to_wkb(G.wf))}).to_parquet(cache_path, index=False)
    info = {"land_shp": shp, "land_pieces_read": int(len(land_geo)), "pbf": pbf_info,
            "shore_pieces": int(len(G.shore)), "waterfront_pieces": int(len(G.wf)),
            "prepare_seconds": round(time.time() - t0, 1)}
    cache_path.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return G, tr, (lat0, lon0), info


# ---------------------------------------------------------------------------
# Εκτέλεση ανά build
READ_COLS = ["mmsi", "t", "lat", "lon", "cog", "COG_VALID"]
OPT_COLS = ["sog", "IN_CORE", "vessel_cat"]
_G = _TR = _LON0 = _CFG = _OUT = None


def _read_day(f, cols):
    if f.endswith(".parquet"):
        import pyarrow.parquet as pq
        have = set(pq.read_schema(f).names)
        return pd.read_parquet(f, columns=[c for c in cols if c in have])
    df = pd.read_csv(f)
    return df[[c for c in cols if c in df.columns]]


def _features_df(df):
    parts = []
    for a in range(0, len(df), _CFG["chunk_rows"]):
        d = df.iloc[a:a + _CFG["chunk_rows"]]
        parts.append(features_latlon(d["lat"].to_numpy(), d["lon"].to_numpy(), d["cog"].fillna(0).to_numpy(),
                                     d["COG_VALID"].fillna(0).to_numpy(), _G, _TR, _LON0, _CFG))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=OUT_COLS)


def _process_day(f):
    df = _read_day(f, READ_COLS + OPT_COLS)
    feat = _features_df(df)
    out = pd.concat([df[["mmsi", "t"]].reset_index(drop=True), feat], axis=1)
    name = Path(f).name.replace(".csv.gz", ".parquet")
    out.to_parquet(_OUT / name, index=False)
    mv = np.ones(len(df), bool)
    if "sog" in df:
        mv &= df["sog"].to_numpy() >= 0.5
    if "IN_CORE" in df:
        mv &= df["IN_CORE"].to_numpy() == 1
    land = feat["ON_LAND"].to_numpy() == 1
    cat = df["vessel_cat"].astype(str).to_numpy() if "vessel_cat" in df else np.full(len(df), "all")
    cells = pd.Series(list(zip(np.floor(df["lat"].to_numpy()[land & mv] * 100) / 100,
                               np.floor(df["lon"].to_numpy()[land & mv] * 100) / 100))).value_counts()
    ds = feat["D_SHORE"].to_numpy()[mv]
    return {"rows": len(df), "moving": int(mv.sum()), "moving_on_land": int((land & mv).sum()),
            "by_cat": pd.DataFrame({"cat": cat[mv], "land": land[mv]}).groupby("cat")["land"].agg(["sum", "count"]),
            "cells": cells, "d_shore_hist": np.histogram(ds, bins=[0, 100, 250, 500, 900, 2200, 5000, 20000, np.inf])[0],
            "nan_out": int(feat[["D_SHORE", "ON_LAND_DEPTH_M"]].isna().to_numpy().sum())}


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def _versions():
    v = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "shapely": shapely.__version__}
    for m in ("pyproj", "pyogrio"):
        try:
            v[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            pass
    try:
        import osmium.version as ov
        v["osmium"] = ov.pyosmium_release
    except Exception:  # noqa: BLE001
        pass
    return v


def main(argv=None):
    global _G, _TR, _LON0, _CFG, _OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_dir")
    ap.add_argument("--osm_dir", required=True, help="data/osm/<YYYY-MM-DD> με land polygons (.zip) και .osm.pbf")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bench", type=int, default=0, help="μόνο χρονομέτρηση σε τυχαίο δείγμα N γραμμών")
    ap.add_argument("--prepare_only", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(json.dumps(DEFAULTS))
    _CFG = cfg
    osm_date = Path(a.osm_dir.rstrip("/")).name

    files = sorted(glob.glob(f"{a.build_dir}/day_*.parquet")) or sorted(glob.glob(f"{a.build_dir}/day_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"κανένα day_* στο {a.build_dir}")
    print(f"[stageC] {a.build_dir} | {len(files)} μέρες | OSM {osm_date}")
    bb = [180.0, 90.0, -180.0, -90.0]
    n_rows = 0
    for f in files:
        d = _read_day(f, ["lat", "lon"])
        n_rows += len(d)
        bb = [min(bb[0], d["lon"].min()), min(bb[1], d["lat"].min()), max(bb[2], d["lon"].max()), max(bb[3], d["lat"].max())]
    core_bb, geom_bb = geo_bbox(bb, cfg)
    print(f"  δεδομένα bbox {np.round(bb, 3).tolist()} → κουτί {core_bb}, γεωμετρία {np.round(geom_bb, 3).tolist()}")

    key = hashlib.sha1(json.dumps({"bb": core_bb, "cfg": {k: cfg[k] for k in cfg if k != "chunk_rows"}},
                                  sort_keys=True).encode()).hexdigest()[:8]
    cache = Path(a.osm_dir) / f"prepared_{key}.parquet"
    t0 = time.time()
    cached = cache.exists()
    _G, _TR, (lat0, lon0), ginfo = prepare_geometry(a.osm_dir, core_bb, geom_bb, cfg, cache)
    _LON0 = lon0
    print(f"  γεωμετρία ({'cache ' + cache.name if cached else 'νέα → ' + cache.name}): "
          f"{ginfo['shore_pieces']:,} τμήματα ακτής, {ginfo['waterfront_pieces']:,} waterfront | {time.time() - t0:.0f} s")
    print(f"  pbf: {ginfo.get('pbf')}")
    if a.prepare_only:
        return

    if a.bench:
        rng = np.random.default_rng(0)
        pool_df = pd.concat([_read_day(f, READ_COLS) for f in rng.choice(files, size=min(len(files), 8), replace=False)],
                            ignore_index=True)
        smp = pool_df.sample(n=min(a.bench, len(pool_df)), random_state=0).reset_index(drop=True)
        t1 = time.time()
        feat = _features_df(smp)
        dt = time.time() - t1
        rate = len(smp) / dt
        print(f"  [bench] {len(smp):,} γραμμές σε {dt:.1f} s → {rate:,.0f} γραμμές/s ανά πυρήνα | "
              f"εκτίμηση για {n_rows:,}: {n_rows / rate / 60:.1f} λεπτά σε 1 πυρήνα, "
              f"{n_rows / rate / 60 / max(a.workers, 1):.1f} με {a.workers}")
        print(f"  [bench] D_SHORE p50/p90 {np.nanpercentile(feat['D_SHORE'], [50, 90]).round(0).tolist()} m | "
              f"ON_LAND {feat['ON_LAND'].mean():.3%} | D_LAND_COG < cap {(feat['D_LAND_COG'] < cfg['d_land_cog_cap_m']).mean():.1%}")
        return

    _OUT = Path(a.build_dir) / "stageC" / f"shore_{osm_date}"
    _OUT.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    if a.workers > 1:
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = pool.map(_process_day, files, chunksize=1)
    else:
        res = [_process_day(f) for f in files]
    rows = sum(r["rows"] for r in res)
    mv = sum(r["moving"] for r in res)
    ml = sum(r["moving_on_land"] for r in res)
    by_cat = pd.concat([r["by_cat"] for r in res]).groupby(level=0).sum() if res else None
    if by_cat is not None:
        by_cat["share"] = (by_cat["sum"] / by_cat["count"]).round(5)
    cells = pd.concat([r["cells"] for r in res]).groupby(level=0).sum().sort_values(ascending=False) if res else pd.Series()
    hist = np.sum([r["d_shore_hist"] for r in res], axis=0)
    bins = ["<100", "100-250", "250-500", "500-900", "900-2200", "2200-5000", "5000-20000", ">=20000"]
    nan_out = sum(r["nan_out"] for r in res)
    summary = {"rows": rows, "moving_core_rows": mv, "moving_on_land": ml, "moving_on_land_share": round(ml / max(mv, 1), 6),
               "on_land_by_cat_moving": by_cat.to_dict(orient="index") if by_cat is not None else {},
               "on_land_hotspots_0.01deg": [{"lat": float(k[0]), "lon": float(k[1]), "rows": int(v)} for k, v in cells.head(20).items()],
               "d_shore_moving_hist": dict(zip(bins, hist.astype(int).tolist())), "nan_in_outputs": nan_out,
               "seconds": round(time.time() - t1, 1)}
    if nan_out:
        raise AssertionError(f"NaN σε D_SHORE/ON_LAND_DEPTH_M: {nan_out}")
    if rows != n_rows:
        raise AssertionError(f"γραμμές εξόδου {rows} ≠ εισόδου {n_rows}")
    osm_files = sorted(glob.glob(f"{a.osm_dir}/*.zip") + glob.glob(f"{a.osm_dir}/*.osm.pbf"))
    meta = {"config": cfg, "commit": _git(), "versions": _versions(), "osm_date": osm_date,
            "osm_files": {Path(p).name: sha256(p) for p in osm_files}, "projection": tm_proj4(lat0, lon0),
            "data_bbox": [float(v) for v in bb], "core_bbox": core_bb, "geom_bbox": list(geom_bb),
            "geometry": ginfo, "summary": summary}
    (_OUT / "stageC_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"  γραμμές {rows:,} (= είσοδος ✓) | κινούμενες IN_CORE {mv:,} | ON_LAND {ml:,} ({ml / max(mv, 1):.3%})")
    print(f"  D_SHORE (κινούμενες): {summary['d_shore_moving_hist']}")
    if by_cat is not None:
        print("  ON_LAND ανά κατηγορία (κινούμενες):")
        print(by_cat.sort_values("count", ascending=False).to_string())
    print("  hotspots ON_LAND (κελιά 0.01°):")
    for h in summary["on_land_hotspots_0.01deg"][:10]:
        print(f"    {h['lat']:.2f}, {h['lon']:.2f}: {h['rows']:,}")
    print(f"[stageC] έξοδος: {_OUT} | {summary['seconds']:.0f} s")


if __name__ == "__main__":
    main()
