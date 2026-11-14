#!/usr/bin/env python3
"""
compare_runs.py — ζευγαρωτή σύγκριση δύο ομάδων αποτελεσμάτων (JSON του baselines.py) ανά seed.

Πρακτική «Protocol before progress» (Raisi & Nazarzehi Had, 2026, §5.1, με βάση Bouthillier et al., 2021):
κάθε διαφορά συγκρίνεται με το τυπικό σφάλμα της ΖΕΥΓΑΡΩΤΗΣ διαφοράς ανά seed, όχι με την απλή διασπορά.
Αν η Β έχει ένα μόνο αποτέλεσμα χωρίς seed (π.χ. CV), συγκρίνεται με κάθε seed της Α.
Κανόνας: 3 seeds ως βάση· 5–6 όταν |διαφορά| < 3 τυπικά σφάλματα.

    python compare_runs.py --a "reports/v6/lstm_*_vessel_val_s?.json" --b "reports/v6/lstm_*_vessel_val_s?_obs20s.json"
    python compare_runs.py --a "..." --b "..." --metric by_scene_group.harbor.FDE
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path


def get(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def load(pattern, metric):
    out = {}
    for f in sorted(glob.glob(pattern)):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        # every trained model has a seed (lstm, phase3); only the deterministic CV has none
        seed = r.get("args", {}).get("seed") if r.get("args", {}).get("model") != "cv" else None
        if seed in out:
            raise ValueError(f"διπλό seed {seed} στο {pattern}")
        out[seed] = float(get(r["metrics"], metric))
    if not out:
        raise FileNotFoundError(f"κανένα αρχείο για {pattern}")
    return out


def compare(a, b):
    """a, b: {seed: τιμή}. Επιστρέφει μέση διαφορά (b − a), τυπικό σφάλμα, n."""
    if list(b) == [None]:
        pairs = [(a[s], b[None]) for s in a]
    elif list(a) == [None]:
        pairs = [(a[None], b[s]) for s in b]
    else:
        common = sorted(set(a) & set(b))
        if not common:
            raise ValueError("κανένα κοινό seed")
        pairs = [(a[s], b[s]) for s in common]
    d = [y - x for x, y in pairs]
    n = len(d)
    mean = sum(d) / n
    se = (math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1)) / math.sqrt(n)) if n > 1 else float("nan")
    base = sum(x for x, _ in pairs) / n
    return {"n": n, "mean_a": round(base, 2), "mean_b": round(base + mean, 2), "diff": round(mean, 2),
            "se": round(se, 2) if se == se else None, "rel_pct": round(100 * mean / base, 2),
            "z": round(mean / se, 1) if se and se == se and se > 0 else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="glob αρχείων αναφοράς")
    ap.add_argument("--b", required=True, help="glob αρχείων σύγκρισης")
    ap.add_argument("--metric", default="headline.FDE", help="π.χ. headline.ADE, headline_nonoverlap.FDE, by_tier_cv.hard.FDE")
    a = ap.parse_args(argv)
    r = compare(load(a.a, a.metric), load(a.b, a.metric))
    verdict = ("καθαρή διαφορά (≥ 3 ΤΣ)" if r["z"] is not None and abs(r["z"]) >= 3 else
               "μέσα στον θόρυβο — περισσότερα seeds (5–6)" if r["z"] is not None else "ένα ζεύγος: χωρίς ΤΣ")
    print(f"[compare] {a.metric}: Α {r['mean_a']} → Β {r['mean_b']} | διαφορά {r['diff']:+} ± {r['se']} (ΤΣ) "
          f"| {r['rel_pct']:+}% | n={r['n']} | {verdict}")
    return r


if __name__ == "__main__":
    main()
