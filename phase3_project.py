#!/usr/bin/env python3
"""
phase3_project.py — αξιολόγηση ενός checkpoint του phase3.py με φυσικό περιορισμό «όχι σε ξηρά» μετά την πρόβλεψη.

Χωρίς εκπαίδευση. Κάθε προβλεπόμενο σημείο με SDF < 0 (raster του stageE_arms.py) μετακινείται κατά μήκος της κλίσης
του SDF ώστε να βρεθεί MARGIN_M μέσα στο νερό (έως ITERS επαναλήψεις, κλίση με κεντρικές διαφορές ενός κελιού).
Τα σημεία στο νερό δεν αλλάζουν. MARGIN_M και ITERS ορίστηκαν πριν από κάθε run και δεν ρυθμίζονται (handoff §1).
Όλα τα υπόλοιπα (ορίσματα, μετρικές, JSON) είναι του phase3.py· το tag παίρνει _proj.

    python phase3_project.py --train_build dataset/socal_2025-03_b19c45d7 --load_model reports/phase3_rtx/<tag>.pt
"""
import sys

import numpy as np

import land_cross_diag as LC
import phase3 as P3
import stageE_arms as E

MARGIN_M, ITERS, STEP_M = 100.0, 3, 100.0


def project(r, d, xy):
    """xy (n, T, 2) in the frame of fut -> same shape, points on land moved MARGIN_M into the water."""
    xy = np.array(xy, np.float64)
    ex, ey = np.array([STEP_M, 0.0]), np.array([0.0, STEP_M])
    s = LC.sdf_at(r, d, xy)
    land = s < 0                                             # only these points are ever moved
    for _ in range(ITERS):
        bad = land & (s < MARGIN_M / 2)                      # still on land, or short of the margin after a step
        if not bad.any():
            break
        g = np.stack([LC.sdf_at(r, d, xy + ex) - LC.sdf_at(r, d, xy - ex),
                      LC.sdf_at(r, d, xy + ey) - LC.sdf_at(r, d, xy - ey)], -1) / (2 * STEP_M)
        norm = np.linalg.norm(g, axis=-1)
        n1 = np.where(norm > 1e-6, norm, 1.0)
        step = ((MARGIN_M - s) / n1 ** 2)[..., None] * g       # distance (MARGIN − s) / |g| along g / |g|
        xy = np.where((bad & (norm > 1e-6))[..., None], xy + step, xy)
        s = LC.sdf_at(r, d, xy)
    return xy.astype(np.float32)


def post(pred, build, d):
    return project(E.load_sdf(build), d, pred)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--load_model" not in argv:
        sys.exit("ο περιορισμός εφαρμόζεται μόνο στην αξιολόγηση: θέλει --load_model")
    return P3.main(argv, post=("proj", post))


if __name__ == "__main__":
    main()
