#!/usr/bin/env python3
"""
v5c_visual.py — V5-Γ: οπτικός έλεγχος του σταδίου Γ (ακτογραμμή).

Για ένα build με έξοδο σταδίου Γ γράφει στο reports/v5c_<build>/:
  hotspot_XX.png      τα top N κελιά ON_LAND (κινούμενα, IN_CORE): ξηρά/ακτή/waterfront του OSM,
                      σημεία κινούμενα (μπλε), κινούμενα ON_LAND (κόκκινα), ακίνητα ON_LAND (ροζ x),
                      και ακτίνες D_LAND_COG (πράσινες) ως έλεγχος της κατεύθυνσης/απόστασης
  depth_hist.png      βάθος μέσα στην ξηρά, κινούμενα έναντι ακίνητων, με υποψήφιες ανοχές τ
  summary.json        ποσοστό ON_LAND ανά ανοχή τ, ανά hotspot: σημεία, πλοία, κατηγορίες, βάθος

    python v5c_visual.py dataset/houston_2025-03_6e340c21 --osm_dir data/osm/2026-10-29
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

import stageC_shore as C

TAUS = [0, 10, 25, 50, 100, 250]


def load_rows(build, shore_dir):
    """Build (lat, lon, sog, cog, COG_VALID, IN_CORE, vessel_cat, mmsi) + στήλες σταδίου Γ, ευθυγραμμισμένα."""
    parts = []
    for f in sorted(glob.glob(f"{shore_dir}/day_*.parquet")):
        name = Path(f).name
        src = Path(build) / name
        if not src.exists():
            src = Path(build) / name.replace(".parquet", ".csv.gz")
        b = C._read_day(str(src), ["mmsi", "t", "lat", "lon", "sog", "cog", "COG_VALID", "IN_CORE", "vessel_cat"])
        s = pd.read_parquet(f)
        if len(b) != len(s) or not (b["mmsi"].to_numpy() == s["mmsi"].to_numpy()).all():
            raise AssertionError(f"{name}: το στάδιο Γ δεν ευθυγραμμίζεται με το build")
        parts.append(pd.concat([b.reset_index(drop=True), s.drop(columns=["mmsi", "t"])], axis=1))
    return pd.concat(parts, ignore_index=True)


def _poly_patch(geom, **kw):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    verts, codes = [], []
    for p in shapely.get_parts(geom):
        if shapely.get_type_id(p) != 3:          # μόνο Polygon
            continue
        p = shapely.orient_polygons(p) if hasattr(shapely, "orient_polygons") else p
        for ring in [p.exterior, *p.interiors]:
            c = np.asarray(ring.coords)[:, :2]
            verts += c.tolist()
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(c) - 2) + [MPath.CLOSEPOLY]
    return PathPatch(MPath(verts, codes), **kw) if verts else None


def plot_hotspot(ax, G, tr, lon0, df_win, cell, half_m, rng, cfg):
    cx, cy = tr.transform(cell[1] + 0.005, cell[0] + 0.005)
    win = shapely.box(cx - half_m, cy - half_m, cx + half_m, cy + half_m)
    land = shapely.intersection(G.land, win)
    patch = _poly_patch(land, facecolor="#d9d4c7", edgecolor="#6b6457", lw=0.6, zorder=1)
    if patch is not None:
        ax.add_patch(patch)
    if len(G.wf):
        for g in G.wf[G.wf_tree.query(win)]:
            c = shapely.get_coordinates(g)
            ax.plot(c[:, 0], c[:, 1], color="#e07b00", lw=1.0, zorder=2)
    x, y = tr.transform(df_win["lon"].to_numpy(), df_win["lat"].to_numpy())
    x, y = np.asarray(x), np.asarray(y)
    mv = df_win["sog"].to_numpy() >= 0.5
    on = df_win["ON_LAND"].to_numpy() == 1
    bg = np.flatnonzero(mv & ~on)
    if len(bg) > 20_000:
        bg = rng.choice(bg, 20_000, replace=False)
    ax.scatter(x[bg], y[bg], s=1, c="#3b75af", alpha=0.35, lw=0, zorder=3, label="κινούμενα")
    ax.scatter(x[~mv & on], y[~mv & on], s=6, c="#e8a0b4", marker="x", lw=0.6, zorder=4, label="ακίνητα ON_LAND")
    ax.scatter(x[mv & on], y[mv & on], s=4, c="#c62828", lw=0, zorder=5, label="κινούμενα ON_LAND")
    d = df_win["D_LAND_COG"].to_numpy()
    ray = np.flatnonzero(mv & ~on & np.isfinite(d) & (d < cfg["d_land_cog_cap_m"]))
    if len(ray):
        ray = rng.choice(ray, min(25, len(ray)), replace=False)
        gam = C.convergence_deg(df_win["lat"].to_numpy()[ray], df_win["lon"].to_numpy()[ray], lon0)
        az = np.radians(df_win["cog"].to_numpy()[ray] - gam)
        L = d[ray] * C.scale_k(x[ray])
        for i in range(len(ray)):
            ax.plot([x[ray][i], x[ray][i] + L[i] * np.sin(az[i])], [y[ray][i], y[ray][i] + L[i] * np.cos(az[i])],
                    color="#2e7d32", lw=0.7, zorder=6)
            ax.plot(x[ray][i] + L[i] * np.sin(az[i]), y[ray][i] + L[i] * np.cos(az[i]), "o", ms=2, color="#2e7d32", zorder=6)
    ax.set_xlim(cx - half_m, cx + half_m)
    ax.set_ylim(cy - half_m, cy + half_m)
    ax.set_aspect("equal")
    ticks = np.linspace(-half_m, half_m, 5)
    ax.set_xticks(cx + ticks, [f"{t / 1000:.1f}" for t in ticks])
    ax.set_yticks(cy + ticks, [f"{t / 1000:.1f}" for t in ticks])
    ax.set_xlabel("km (ανατολή)")
    ax.set_ylabel("km (βορράς)")


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_dir")
    ap.add_argument("--osm_dir", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--half_km", type=float, default=1.5, help="μισό πλάτος παραθύρου εικόνας")
    a = ap.parse_args(argv)
    osm_date = Path(a.osm_dir.rstrip("/")).name
    shore_dir = Path(a.build_dir) / "stageC" / f"shore_{osm_date}"
    meta = json.loads((shore_dir / "stageC_meta.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    core_bb, geom_bb = tuple(meta["core_bbox"]), tuple(meta["geom_bbox"])
    cache = Path(a.osm_dir) / f"prepared_{C.cache_key(core_bb, cfg, a.osm_dir)}.parquet"
    if not cache.exists():
        raise FileNotFoundError(f"{cache}: η cache γεωμετρίας δεν αντιστοιχεί στο stageC_meta (άλλη έκδοση ή αρχεία OSM;)")
    G, tr, (lat0, lon0), _ = C.prepare_geometry(a.osm_dir, core_bb, geom_bb, cfg, cache)
    out = Path("reports") / f"v5c_{Path(a.build_dir.rstrip('/')).name}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[v5c] {a.build_dir} → {out}")
    df = load_rows(a.build_dir, shore_dir)
    core = df["IN_CORE"].to_numpy() == 1 if "IN_CORE" in df else np.ones(len(df), bool)
    mv = (df["sog"].to_numpy() >= 0.5) & core
    st = (df["sog"].to_numpy() < 0.5) & core
    on = df["ON_LAND"].to_numpy() == 1
    depth = df["ON_LAND_DEPTH_M"].to_numpy()

    tau_tab = {lab: {f"tau_{t}m": round(float((on[m] & (depth[m] > t)).mean()), 6) for t in TAUS}
               for lab, m in (("κινούμενα", mv), ("ακίνητα", st))}
    print("  ON_LAND ανά ανοχή τ (ποσοστό γραμμών IN_CORE):")
    for lab, r in tau_tab.items():
        print(f"    {lab:<10} " + " | ".join(f"τ={k[4:]}: {v:.4%}" for k, v in r.items()))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(0, np.log10(max(depth[on].max(), 10)), 50) if on.any() else np.logspace(0, 3, 50)
    for lab, m, c in (("κινούμενα", mv, "#c62828"), ("ακίνητα", st, "#3b75af")):
        ax.hist(depth[m & on], bins=bins, histtype="step", lw=1.5, color=c, label=f"{lab} (n={int((m & on).sum()):,})")
    for t in (25, 50, 100):
        ax.axvline(t, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("βάθος μέσα στην ξηρά (m)")
    ax.set_ylabel("γραμμές")
    ax.set_title(f"{Path(a.build_dir).name}: ON_LAND, βάθος")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "depth_hist.png", dpi=150)
    plt.close(fig)

    cells = pd.DataFrame({"clat": np.floor(df["lat"].to_numpy()[mv & on] * 100) / 100,
                          "clon": np.floor(df["lon"].to_numpy()[mv & on] * 100) / 100})
    top = cells.value_counts().head(a.top)
    rng = np.random.default_rng(0)
    half_m = a.half_km * 1000
    deg = a.half_km / 111.0 * 1.6
    hs = []
    for i, ((clat, clon), n) in enumerate(top.items(), start=1):
        w = df[(df["lat"].between(clat + 0.005 - deg, clat + 0.005 + deg)) &
               (df["lon"].between(clon + 0.005 - deg * 1.3, clon + 0.005 + deg * 1.3)) & core]
        wm = (w["sog"] >= 0.5) & (w["ON_LAND"] == 1)
        info = {"cell": [float(clat), float(clon)], "moving_on_land_rows": int(n),
                "vessels_on_land": int(w.loc[wm, "mmsi"].nunique()),
                "cats_on_land": w.loc[wm, "vessel_cat"].astype(str).value_counts().head(4).to_dict(),
                "depth_p50_p90_m": np.percentile(w.loc[wm, "ON_LAND_DEPTH_M"], [50, 90]).round(0).tolist() if wm.any() else None}
        hs.append(info)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        plot_hotspot(ax, G, tr, lon0, w, (clat, clon), half_m, rng, cfg)
        ax.set_title(f"#{i} ({clat:.2f}, {clon:.2f}) · κινούμενα ON_LAND {n:,} σε {info['vessels_on_land']} πλοία · "
                     f"βάθος p50/p90 {info['depth_p50_p90_m']} m\n{info['cats_on_land']}", fontsize=8)
        ax.legend(loc="lower left", fontsize=7, markerscale=3)
        fig.tight_layout()
        fig.savefig(out / f"hotspot_{i:02d}.png", dpi=150)
        plt.close(fig)
        print(f"  #{i} {info}")

    (out / "summary.json").write_text(json.dumps({"build": a.build_dir, "osm_date": osm_date, "tau": tau_tab, "hotspots": hs},
                                                 indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[v5c] {len(hs)} εικόνες + depth_hist.png στο {out}")


if __name__ == "__main__":
    main()
