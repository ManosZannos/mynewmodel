#!/usr/bin/env python3
"""
field_report.py — πόσο σφάλμα «βλέπει» κάθε κανάλι του πολικού πεδίου, ΠΡΙΝ εκπαιδευτεί μοντέλο με context.

Για τα παράθυρα της κύριας μετρικής ενός split, ανά scene_group και ανά κανάλι:
  covered    ποσοστό παραθύρων όπου το κανάλι έχει τιμή ≠ 0
  FDE_in/out μέσο FDE της αναφοράς μέσα / έξω από αυτά τα παράθυρα
  err_share  μερίδιο του συνολικού FDE της ομάδας που πέφτει μέσα = ΑΝΩ ΟΡΙΟ του κέρδους του καναλιού
             (αν μηδένιζε όλο το σφάλμα εκεί)· κανάλι με μικρό err_share δεν μπορεί να περάσει την πύλη του 1%
Αναφορά: το αιτιατό CV (χωρίς torch) ή, με --load_model, ένα checkpoint του M0. Μόνο train / val.
Κανάλια: land (ξηρά σε κάποιον τομέα), land_ahead (στους 3 τομείς μπροστά), vessels, habit (υπάρχει ιστορικό).

    python field_report.py dataset/socal_2025-03_b19c45d7
    python field_report.py dataset/socal_2025-03_b19c45d7 --load_model reports/phase3_dgx/<m0 tag>.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import baselines as BL
import loader as LD
import phase3 as P3


def flags(f):
    """f: rows of stageD/field.parquet -> channel -> (n,) bool, 'the channel says something here'."""
    land = f.filter(like="land_")
    ahead = [c for c in land if int(c[-2:]) in (0, 1, land.shape[1] - 1)]
    return {"land": land.to_numpy().max(1) > 0, "land_ahead": land[ahead].to_numpy().max(1) > 0,
            "vessels": f.filter(like="vd_").to_numpy().max(1) > 0, "habit": f["hab_n"].to_numpy() > 0}


def report(fde, fl, groups):
    fde, g, rows = np.asarray(fde, float), np.asarray(groups), []
    for name in ["all"] + sorted(set(g) - {"all"}):
        m = np.ones(len(fde), bool) if name == "all" else g == name
        for ch, v in fl.items():
            a, b = m & v, m & ~v
            rows.append({"group": name, "channel": ch, "n": int(m.sum()), "covered": a.sum() / m.sum(),
                         "FDE_in": fde[a].mean() if a.any() else np.nan, "FDE_out": fde[b].mean() if b.any() else np.nan,
                         "err_share": fde[a].sum() / fde[m].sum()})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_dir")
    ap.add_argument("--manifest", default="vessel")
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--load_model", default=None, help="checkpoint του M0 (χωρίς context)· προεπιλογή: CV")
    a = ap.parse_args(argv)
    d = LD.load_split(a.build_dir, a.manifest, a.split, x=LD.load_rows(a.build_dir, BL.shore_dir_of(a.build_dir)))
    m = d["meta"]
    if a.load_model:
        import torch
        device = BL.device_info(torch)["device"]
        model, ck = P3.load_checkpoint(a.load_model, device)
        if ck["arch"]["channels"]:
            sys.exit("η αναφορά πρέπει να είναι μοντέλο χωρίς context (M0)")
        pred = P3.predict(P3.net_fn(model, device), P3.prepare(d, not ck["arch"]["no_rotate"]), not ck["arch"]["no_cv"])
    else:
        pred = P3.course(d["pos"], m["last_avail"], m["p_age_s"])[0]
    fde = BL.errors(pred, d["fut"])[1]
    f = pd.read_parquet(Path(a.build_dir) / "stageD" / "field.parquet").set_index("window_id").loc[m["window_id"]]
    h = m["headline"].to_numpy() == 1
    g = m["scene_group"].to_numpy() if "scene_group" in m else np.full(len(m), "all")
    r = report(fde[h], {k: v[h] for k, v in flags(f).items()}, g[h])
    print(f"[field_report] {a.build_dir} / {a.manifest} / {a.split} | αναφορά: {a.load_model or 'CV'} | "
          f"κύρια μετρική n={int(h.sum()):,} | FDE {fde[h].mean():.1f} m")
    print(r.to_string(index=False, formatters={"covered": "{:.1%}".format, "err_share": "{:.1%}".format,
                                               "FDE_in": "{:.1f}".format, "FDE_out": "{:.1f}".format}))
    return r


if __name__ == "__main__":
    main()
