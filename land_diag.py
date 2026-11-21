#!/usr/bin/env python3
"""
land_diag.py — διαγνωστικό, ΜΕΤΑ τα test: πού κερδίζει ή χάνει ένα μοντέλο με context (Β) έναντι του M0 (Α),
ανά τιμή του καναλιού ξηράς. Δεν επιλέγει τίποτα· εξηγεί ένα αποτέλεσμα που έχει ήδη αναφερθεί (handoff §1, Houston).

Για τα παράθυρα της κύριας μετρικής, ανά ομάδα:
  ahead    εγγύτητα της ξηράς στους 3 τομείς μπροστά (max του land_kk: 0 = καμία μέσα σε T_max, 1 = τώρα)
  front    ξηρά κοντά (> 2/3) ακριβώς μπροστά (τομέας 0) ή μόνο στους πλάγιους μπροστινούς (1, K−1)
  channel  ξηρά και στις δύο πλευρές (τομείς K/4 και 3K/4, εγγύτητα > 0.5): «στενό κανάλι»
  tier_cv, scene_group
στήλες: share (των παραθύρων), FDE του CV / Α / Β, diff = Β − Α, contrib = μερίδιο της συνολικής διαφοράς
(άθροισμα 1 ανά ομαδοποίηση).

    python land_diag.py dataset/houston_2025-03_7bfeb419 --manifest alltest --split test \
        --model_a reports/phase3_rtx/<m0 tag>.pt --model_b reports/phase3_rtx/<m_land tag>.pt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import baselines as BL
import loader as LD
import phase3 as P3

AHEAD_BINS = [-np.inf, 0, 1 / 3, 2 / 3, 1]
AHEAD_LABELS = ["0 (καμία)", "(0, 1/3]", "(1/3, 2/3]", "(2/3, 1]"]


def land_groups(f):
    """f: rows of stageD/field.parquet -> {'ahead': labels, 'front': labels, 'channel': bool} per window."""
    land = f.filter(like="land_").to_numpy()
    K = land.shape[1]
    ahead = land[:, [0, 1, K - 1]].max(1)
    dead, flank = land[:, 0] > 2 / 3, land[:, [1, K - 1]].max(1) > 2 / 3
    return {"ahead": pd.cut(ahead, AHEAD_BINS, labels=AHEAD_LABELS).astype(str),
            "front": np.where(dead, "ακριβώς μπροστά", np.where(flank, "μόνο πλάγια", "καμία κοντά")),
            "channel": (land[:, K // 4] > 0.5) & (land[:, 3 * K // 4] > 0.5)}


def table(fde_cv, fde_a, fde_b, keys):
    a, b, cv = (np.asarray(v, float) for v in (fde_a, fde_b, fde_cv))
    total, rows = (b - a).sum(), []
    for g in np.unique(keys):
        m = np.asarray(keys) == g
        rows.append({"group": g, "n": int(m.sum()), "share": m.mean(), "FDE_cv": cv[m].mean(), "FDE_a": a[m].mean(),
                     "FDE_b": b[m].mean(), "diff": (b[m] - a[m]).mean(), "contrib": (b[m] - a[m]).sum() / total})
    return pd.DataFrame(rows)


def model_fde(path, build, d, device):
    model, ck = P3.load_checkpoint(path, device)
    arch = ck["arch"]
    c = P3.load_field(Path(build) / arch.get("field_dir", "stageD") / "field.parquet",
                      d["meta"]["window_id"].to_numpy(), arch["channels"]) if arch["channels"] else None
    pred = P3.predict(P3.net_fn(model, device), P3.prepare(d, not arch["no_rotate"], c), not arch["no_cv"])
    return BL.errors(pred, d["fut"])[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_dir")
    ap.add_argument("--manifest", default="vessel")
    ap.add_argument("--split", default="val")
    ap.add_argument("--model_a", required=True, help="checkpoint αναφοράς (M0)")
    ap.add_argument("--model_b", required=True, help="checkpoint με context")
    a = ap.parse_args(argv)
    import torch
    device = BL.device_info(torch)["device"]
    d = LD.load_split(a.build_dir, a.manifest, a.split, x=LD.load_rows(a.build_dir, BL.shore_dir_of(a.build_dir)))
    m = d["meta"]
    h = m["headline"].to_numpy() == 1
    cv = BL.errors(P3.course(d["pos"], m["last_avail"], m["p_age_s"])[0], d["fut"])[1]
    fa, fb = model_fde(a.model_a, a.build_dir, d, device), model_fde(a.model_b, a.build_dir, d, device)
    f = pd.read_parquet(Path(a.build_dir) / "stageD" / "field.parquet").set_index("window_id").loc[m["window_id"]]
    g = land_groups(f)
    keys = {"ahead": g["ahead"], "front": g["front"], "channel": np.where(g["channel"], "στενό", "όχι"),
            "tier_cv": m["tier_cv"].to_numpy(), "scene_group": m["scene_group"].to_numpy()}
    print(f"[land_diag] {a.build_dir} / {a.manifest} / {a.split} | κύρια μετρική n={int(h.sum()):,}\n"
          f"  Α {a.model_a}: FDE {fa[h].mean():.1f}\n  Β {a.model_b}: FDE {fb[h].mean():.1f}")
    fmt = {"share": "{:.1%}".format, "contrib": "{:+.1%}".format,
           **{k: "{:.1f}".format for k in ("FDE_cv", "FDE_a", "FDE_b")}, "diff": "{:+.1f}".format}
    for k, v in keys.items():
        print(f"\n== {k}")
        print(table(cv[h], fa[h], fb[h], np.asarray(v)[h]).to_string(index=False, formatters=fmt))
    # ahead × channel: is the loss in narrow channels with land ahead?
    print("\n== ahead × channel")
    both = np.char.add(np.char.add(np.asarray(g["ahead"], str), " | "), keys["channel"].astype(str))
    print(table(cv[h], fa[h], fb[h], both[h]).to_string(index=False, formatters=fmt))


if __name__ == "__main__":
    main()
