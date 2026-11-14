#!/usr/bin/env python3
"""
scorecard.py — ετυμηγορία για μια δοκιμή της Φάσης 3 (κανόνες: docs/phase3_handoff.md §6β).

Συγκρίνει δύο ομάδες JSON (Α = αναφορά, Β = δοκιμή) με τη ζευγαρωτή διαφορά ανά seed του compare_runs.py.

  επιτυχία        κύρια μετρική FDE: z ≤ −3 και κέρδος ≥ 1%, και όλες οι δικλείδες περνούν
  αδιευκρίνιστο   |z| < 3 με < 6 seeds → τρέξε 6 seeds
  χωρίς effect    |z| < 3 με ≥ 6 seeds, ή καθαρό κέρδος < 1% → το κομμάτι δεν μπαίνει
  αποτυχία        z ≥ 3 (χειρότερο) ή παραβίαση δικλείδας

Δικλείδες: nonoverlap με το ίδιο πρόσημο· κανένα stratum με n ≥ 1000 χειρότερο κατά ≥ 3 ΤΣ· κάθε --target
(το stratum που στοχεύει το κανάλι) καλύτερο κατά ≥ 3 ΤΣ· παράμετροι Β/Α ≤ 1.5 όταν καταγράφονται και στα δύο.

    python scorecard.py --a "reports/conv_v37/lstm_*_vessel_val_s?.json" --b "reports/phase3/m0_*_vessel_val_s?.json"
    python scorecard.py --a "...m0..." --b "...land..." --target by_scene_group.harbor.FDE
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import compare_runs as CR

GROUPS = ("by_tier_cv", "by_scene_group", "by_vessel_cat")


def _first(pattern):
    return json.loads(Path(sorted(glob.glob(pattern))[0]).read_text(encoding="utf-8"))


def judge(a, b, targets=(), min_n=1000, min_gain_pct=1.0):
    def cmp(metric):
        return CR.compare(CR.load(a, metric), CR.load(b, metric))

    main, guards = cmp("headline.FDE"), []
    ja, jb = _first(a), _first(b)
    r = cmp("headline_nonoverlap.FDE")
    guards.append({"guard": "nonoverlap: ίδιο πρόσημο", "ok": r["diff"] < 0, **r})
    for g in GROUPS:
        for v, s in jb["metrics"].get(g, {}).items():
            if s["n"] >= min_n:
                r = cmp(f"{g}.{v}.FDE")
                guards.append({"guard": f"{g}.{v} όχι χειρότερο", "ok": not (r["z"] is not None and r["z"] >= 3), **r})
    for t in targets:
        r = cmp(t)
        guards.append({"guard": f"στόχος {t}", "ok": r["z"] is not None and r["z"] <= -3, **r})
    if "n_params" in ja and "n_params" in jb:
        ratio = jb["n_params"] / ja["n_params"]
        guards.append({"guard": "παράμετροι Β/Α ≤ 1.5", "ok": ratio <= 1.5, "ratio": round(ratio, 2)})

    z = main["z"]
    if z is None:
        verdict = "ένα ζεύγος: χωρίς ΤΣ"
    elif z >= 3:
        verdict = "αποτυχία"
    elif z > -3:
        verdict = "χωρίς effect" if main["n"] >= 6 else "αδιευκρίνιστο"
    elif main["rel_pct"] > -min_gain_pct:
        verdict = "χωρίς effect"
    elif not all(g["ok"] for g in guards):
        verdict = "αποτυχία"
    else:
        verdict = "επιτυχία"
    return {"verdict": verdict, "main": main, "guards": guards}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="glob αρχείων αναφοράς")
    ap.add_argument("--b", required=True, help="glob αρχείων της δοκιμής")
    ap.add_argument("--target", action="append", default=[], help="μετρική όπου περιμένουμε το κέρδος (επαναλαμβάνεται)")
    a = ap.parse_args(argv)
    r = judge(a.a, a.b, a.target)
    m = r["main"]
    print(f"[scorecard] {r['verdict'].upper()} | headline.FDE: Α {m['mean_a']} → Β {m['mean_b']} | "
          f"{m['diff']:+} ± {m['se']} | {m['rel_pct']:+}% | z {m['z']} | n={m['n']}")
    for g in r["guards"]:
        d = f"{g['diff']:+} ± {g['se']} (z {g['z']})" if "diff" in g else f"λόγος {g['ratio']}"
        print(f"   {'✓' if g['ok'] else '✗'} {g['guard']}: {d}")
    return r


if __name__ == "__main__":
    main()
