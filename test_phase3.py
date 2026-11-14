#!/usr/bin/env python3
"""test_phase3.py — tests του phase3.py και του scorecard.py σε συνθετικά δεδομένα (python test_phase3.py).
Τα tests με «torch» στο όνομα θέλουν torch· τα υπόλοιπα τρέχουν και χωρίς."""
import json, shutil, sys, tempfile, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import loader as LD
import stageB_manifests as SB
import baselines as BL
import compare_runs as CR
import phase3 as P3
import scorecard as SC
from test_loader import _track


def _split(tracks):
    """Synthetic tracks -> a loader-style split (all windows, headline = 1)."""
    x = pd.concat(tracks, ignore_index=True).sort_values(["mmsi", "t"], kind="mergesort").reset_index(drop=True)
    x["tm"] = ((x["t"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1)).astype(np.int64)
    w = SB.build_windows(x, SB.DEFAULTS)
    obs, fut, pos, extra = LD.make_arrays(x, w)
    return {"obs": obs, "fut": fut, "pos": pos, "features": LD.FEATURES,
            "meta": pd.DataFrame({"mmsi": w["mmsi"].to_numpy(), "headline": 1, **extra})}


def _turns(mmsis, seed):
    rng = np.random.default_rng(seed)
    return _split([_track(m, 60, turn_deg_per_min=float(rng.uniform(-4, 4)), speed_kn=float(rng.uniform(6, 14)),
                          lat0=33.0 + 0.01 * i) for i, m in enumerate(mmsis)])


def _rotated(d, alpha):
    """The same windows seen with the compass turned by alpha: every (east, north) vector rotated."""
    th = np.full(len(d["obs"]), -alpha)
    obs = d["obs"].astype(np.float64)
    for p in P3.VEC:
        obs[..., p] = P3.rot(obs[..., p], th)
    return {**d, "obs": obs.astype(np.float32), "pos": P3.rot(d["pos"].astype(np.float64), th),
            "fut": P3.rot(d["fut"].astype(np.float64), th)}


def _fake_net(x):
    """Any fixed function of the input stands in for the network."""
    return np.tanh(x[:, :, :2].sum(1))[:, None, :] * np.arange(1, LD.PRED + 1)[None, :, None] * 0.05


def test_zero_output_is_cv():
    """Residual formulation: a network that outputs zeros reproduces baselines.cv_predict exactly."""
    d = _turns([1, 2, 3], 0)
    p = P3.prepare(d)
    ref = BL.cv_predict(d["pos"], last=d["meta"]["last_avail"].to_numpy(), age_s=d["meta"]["p_age_s"].to_numpy())
    assert np.array_equal(P3.predict(lambda x: np.zeros((len(x), LD.PRED, 2)), p), ref)
    assert BL.errors(ref, d["fut"])[1].max() > 50, "το συνθετικό χρειάζεται στροφές"


def test_target_roundtrip():
    """A network that outputs target() gives the true future, with and without the CV term."""
    p = P3.prepare(_turns([1, 2, 3], 1))
    for use_cv in (True, False):
        out = P3.predict(lambda x: P3.target(p, use_cv), p, use_cv)
        assert np.abs(out - p["fut"]).max() < 0.05, (use_cv, np.abs(out - p["fut"]).max())


def test_rotation_equivariance():
    """Turning the compass by alpha leaves the network input unchanged and turns the prediction by alpha.
    Without rotation (--no_rotate) the input changes, so the property is not trivially true."""
    d, alpha = _turns([1, 2, 3], 2), np.radians(73.0)
    d2 = _rotated(d, alpha)
    p, p2 = P3.prepare(d), P3.prepare(d2)
    assert np.abs(p["x"] - p2["x"]).max() < 1e-4, np.abs(p["x"] - p2["x"]).max()
    a, b = P3.predict(_fake_net, p), P3.predict(_fake_net, p2)
    assert np.abs(P3.rot(a, np.full(len(a), -alpha)) - b).max() < 0.05
    assert np.abs(P3.prepare(d, rotate=False)["x"] - P3.prepare(d2, rotate=False)["x"]).max() > 0.1


def test_training_on_test_split_refused():
    """Protocol: a training run evaluates on val only; test sets need --load_model."""
    try:
        P3.main(["--train_build", "does_not_exist", "--eval_split", "test"])
    except SystemExit as e:
        assert "val" in str(e)
        return
    raise AssertionError("εκπαίδευση με αξιολόγηση σε test έγινε δεκτή")


# --- scorecard ---
def _runs(d, name, fde, strata=None, n_params=None, model=None):
    for s, v in enumerate(fde):
        st = {k: {"n": n, "FDE": v + off + 0.3 * s} for k, (n, off) in (strata or {}).items()}
        r = {"args": {"seed": s, **({"model": model} if model else {})},
             "metrics": {"headline": {"FDE": v}, "headline_nonoverlap": {"FDE": v + 20}, "by_scene_group": st}}
        if n_params:
            r["n_params"] = n_params
        (d / f"{name}_s{s}.json").write_text(json.dumps(r), encoding="utf-8")
    return str(d / f"{name}_s?.json")


def test_compare_runs_reads_phase3_seeds():
    """compare_runs took the seed only from args.model == 'lstm': phase3 results collided on seed None."""
    d = Path(tempfile.mkdtemp())
    try:
        a = _runs(d, "lstm", [570.0, 573.0, 571.0], model="lstm")
        b = _runs(d, "m0", [550.0, 554.0, 551.0])
        assert sorted(CR.load(b, "headline.FDE")) == [0, 1, 2]
        r = CR.compare(CR.load(a, "headline.FDE"), CR.load(b, "headline.FDE"))
        assert r["n"] == 3 and r["z"] < -3
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_scorecard_verdicts():
    d = Path(tempfile.mkdtemp())
    try:
        st = {"harbor": (5000, 400.0), "nearshore": (300, 900.0)}
        a = _runs(d, "a", [570.0, 573.0, 571.0], st, n_params=200_000)
        ok = _runs(d, "ok", [550.0, 554.0, 551.0], st, n_params=250_000)
        assert SC.judge(a, ok)["verdict"] == "επιτυχία"
        assert SC.judge(a, ok, targets=["by_scene_group.harbor.FDE"])["verdict"] == "επιτυχία"
        assert SC.judge(a, _runs(d, "noise", [565.0, 580.0, 568.0], st))["verdict"] == "αδιευκρίνιστο"
        six = _runs(d, "a6", [570.0, 573.0, 571.0, 569.0, 572.0, 570.0], st)
        assert SC.judge(six, _runs(d, "n6", [565.0, 580.0, 568.0, 571.0, 566.0, 575.0], st))["verdict"] == "χωρίς effect"
        assert SC.judge(a, _runs(d, "small", [567.0, 570.1, 568.0], st))["verdict"] == "χωρίς effect"   # clear but < 1%
        assert SC.judge(a, _runs(d, "worse", [590.0, 594.0, 591.0], st))["verdict"] == "αποτυχία"
        # headline better, but the large stratum (harbor, n >= 1000) clearly worse -> guardrail
        r = SC.judge(a, _runs(d, "hurt", [550.0, 554.0, 551.0], {"harbor": (5000, 460.0), "nearshore": (300, 900.0)}))
        assert r["verdict"] == "αποτυχία" and not all(g["ok"] for g in r["guards"])
        # the same damage in a small stratum (n < 1000) is not a guardrail
        assert SC.judge(a, _runs(d, "tiny", [550.0, 554.0, 551.0], {"harbor": (5000, 400.0), "nearshore": (300, 990.0)}))["verdict"] == "επιτυχία"
        big = SC.judge(a, _runs(d, "big", [550.0, 554.0, 551.0], st, n_params=400_000))
        assert big["verdict"] == "αποτυχία" and [g for g in big["guards"] if "παράμετροι" in g["guard"]][0]["ratio"] == 2.0
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --- torch ---
def _args(**kw):
    from types import SimpleNamespace as NS
    return NS(**{"seed": 0, "lr": 3e-3, "epochs": 200, "batch": 64, "patience": 8, "lr_patience": 3, "no_cv": False, **kw})


_ARCH = {"encoder": "tcn", "hidden": 32, "no_cv": False, "no_rotate": False}


def test_torch_new_net_is_cv():
    """Zero-initialised last layer: an untrained network predicts exactly the CV baseline, for both encoders."""
    d = _turns([1, 2, 3], 0)
    p = P3.prepare(d)
    for enc in ("tcn", "gru"):
        out = P3.predict(P3.net_fn(P3.make_net({**_ARCH, "encoder": enc}), "cpu"), p)
        assert np.array_equal(out, p["cv"]), enc


def test_torch_stop_reason():
    tr, es = P3.prepare(_turns(range(1, 9), 0)), P3.prepare(_turns(range(20, 24), 1))
    _, info = P3.train(P3.make_net(_ARCH), tr, es, _args(lr=0.0, patience=2), "cpu")     # lr 0: nothing improves
    assert info["stop"] == "patience" and info["epochs_run"] == 3 and info["best_epoch"] == 1, info
    _, info = P3.train(P3.make_net(_ARCH), tr, es, _args(epochs=2, patience=50), "cpu")
    assert info["stop"] == "max_epochs" and info["epochs_run"] == 2 and len(info["curve"]) == 2, info


def test_torch_learns_turns():
    """Constant turn rate is visible in the observation: the model must end far below the CV error."""
    tr, es = P3.prepare(_turns(range(1, 41), 0)), P3.prepare(_turns(range(100, 112), 1))
    cv = BL.errors(es["cv"], es["fut"])[0].mean()
    _, info = P3.train(P3.make_net(_ARCH), tr, es, _args(), "cpu")
    assert info["stop"] == "patience", info["stop"]
    assert info["best_early_stop_ADE"] < 0.3 * cv, (info["best_early_stop_ADE"], cv)


def test_torch_end_to_end_and_checkpoint_arch():
    """main(): train on a file build -> JSON + checkpoint; evaluation from the checkpoint rebuilds the saved
    architecture (GRU, hidden 16) although the command line asks for the default TCN, and gives the same numbers."""
    d = Path(tempfile.mkdtemp())
    try:
        rng = np.random.default_rng(0)
        x = pd.concat([_track(m, 60, turn_deg_per_min=float(rng.uniform(-3, 3)), lat0=33.0 + 0.01 * m)
                       for m in range(1, 31)], ignore_index=True)
        for day, g in x.groupby(x["t"].dt.strftime("%Y-%m-%d")):
            g[SB.COLS + LD.CAUSAL_COLS].to_csv(d / f"day_{day}.csv.gz", index=False)
        w = SB.build_windows(SB.load_build(str(d)), SB.DEFAULTS)
        w["tier_cv"] = "easy"
        (d / "stageB").mkdir()
        w.to_csv(d / "stageB" / "windows.csv.gz", index=False)
        t = w[w["eligible"] == 1][["window_id", "mmsi", "seg_id", "t_start"]].assign(capped=1)
        t["split"] = np.where(t["mmsi"] <= 24, "train", "val")
        t.to_csv(d / "stageB" / "targets_vessel.csv.gz", index=False)
        common = ["--train_build", str(d), "--out_dir", str(d / "out")]
        r = P3.main(common + ["--encoder", "gru", "--hidden", "16", "--epochs", "2", "--batch", "32", "--seed", "3"])
        assert r["train"]["stop"] == "max_epochs" and r["arch"]["encoder"] == "gru" and r["args"]["seed"] == 3
        assert r["n_params"] > 0 and r["metrics"]["headline"]["n"] == int((t["split"] == "val").sum())
        r2 = P3.main(common + ["--load_model", r["model_path"], "--tag", "reload"])
        assert r2["arch"] == r["arch"] and r2["n_params"] == r["n_params"] and r2["args"]["seed"] == 3
        assert r2["metrics"] == r["metrics"] and "train" not in r2
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
