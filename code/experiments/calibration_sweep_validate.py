"""calibration_sweep_validate.py -- stage 2: do the survivors hold up?

STAGE 1 RECAP (single fit, HGB, full clean test)

    bare                     0.8986
    sigmoid cv=3             0.8968
    bag-only cv=3            0.9061
    sigmoid cv=5 PRODUCTION  0.9032
    bag-only cv=5            0.9053
    sigmoid cv=10            0.9059     Brier 0.0882 (better than production's 0.0893)
    bag-only cv=10           0.9049
    sigmoid cv=20            0.9061     Brier 0.0889

A CORRECTION TO THE FIRST READING OF THESE NUMBERS

Seeing only cv=3, it looked like the per-fold sigmoid destroys ranking
information: sigmoid 0.8968 vs bag-only 0.9061, a 0.0093 gap. That does NOT
generalise. At cv=10 the sigmoid arm is slightly AHEAD of bag-only
(0.9059 vs 0.9049), and at cv=20 it matches the best arm outright. The cv=3
sigmoid is the outlier, not the rule -- with three folds each per-fold sigmoid
is fit on a third of the data and is simply noisy. The honest summary is that
the AVERAGING supplies the gain and the number of folds is what matters;
the sigmoid is roughly neutral once it has enough data to be stable.

WHY THIS STAGE EXISTS

Every number above is a single fit. This project measured, two experiments
ago, that a single training draw carries about +/-0.005 of arbitrary variation
and that paired deltas have sd ~0.0024 -- larger than the differences being
compared here. Stage 1 is therefore a landscape scan and nothing more. This
re-fits every survivor AND the production baseline on the same bootstrap
resamples of the training rows and reports the delta distribution.

Parallelised across resamples rather than within a fit: HGB is deterministic
given data, so a resample is a self-contained unit of work, and OMP_NUM_THREADS
is pinned to 1 to avoid the nested-parallelism thrashing recorded in
ENVIRONMENT_NOTES.md.
"""
import os
import sys
import json
import time
import importlib.util
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import roc_auc_score  # exact drop-in, ~23x faster inside bootstraps

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "calibration_sweep_validate_results.json")

SEED = 42
N_RESAMPLES = 8
N_BOOT = 1500
N_WORKERS = 6

# baseline first; survivors are everything stage 1 put above production
BASELINE = ("sigmoid cv=5 [PRODUCTION]", "sigmoid", 5)
SURVIVORS = [
    ("bag-only cv=3", "bag", 3),
    ("bag-only cv=5", "bag", 5),
    ("sigmoid cv=10", "sigmoid", 10),
    ("bag-only cv=10", "bag", 10),
    ("sigmoid cv=20", "sigmoid", 20),
]


class BagOnly:
    """k models on the CalibratedClassifierCV folds, raw probabilities averaged,
    no calibration -- isolates averaging from the sigmoid."""

    def __init__(self, base, cv=5, seed=SEED):
        self.base, self.cv, self.seed = base, cv, seed

    def fit(self, X, y):
        self.models_ = []
        skf = StratifiedKFold(self.cv, shuffle=True, random_state=self.seed)
        for tr_idx, _ in skf.split(X, y):
            m = clone(self.base)
            m.fit(X.iloc[tr_idx], y[tr_idx])
            self.models_.append(m)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def make(kind, k, bare):
    if kind == "sigmoid":
        return CalibratedClassifierCV(clone(bare), cv=k, method="sigmoid")
    return BagOnly(clone(bare), cv=k)


def paired_boot(y, pa, pb, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
    d = np.asarray(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


_G = {}


def _init():
    m05 = _m05()
    df = pd.read_csv(TRAINING)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    tr, te = m05.split_by_host(df)
    tr, te = np.asarray(tr), np.asarray(te)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce")
    te2 = te & (((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy())
    prod = joblib.load(PROD)
    _G.update(X=X, y=y, tr=tr, te=te, te2=te2,
              bare=clone(getattr(prod, "estimator", prod)))


def run_resample(rep):
    if not _G:
        _init()
    X, y, tr, te, te2, bare = (_G["X"], _G["y"], _G["tr"], _G["te"],
                               _G["te2"], _G["bare"])
    Xtr, ytr = X[tr], y[tr]
    n = len(ytr)
    rng = np.random.RandomState(1000 + rep)
    idx = rng.randint(0, n, n)
    Xb, yb = Xtr.iloc[idx], ytr[idx]

    def fit_eval(kind, k):
        m = make(kind, k, bare).fit(Xb, yb)
        return (m.predict_proba(X[te])[:, 1], m.predict_proba(X[te2])[:, 1])

    bF, b2 = fit_eval(BASELINE[1], BASELINE[2])
    row = {"rep": rep,
           "baseline_full": float(roc_auc_score(y[te], bF)),
           "baseline_2min": float(roc_auc_score(y[te2], b2)),
           "baseline_brier": float(brier_score_loss(y[te], bF)),
           "arms": {}}
    for name, kind, k in SURVIVORS:
        pF, p2 = fit_eval(kind, k)
        dF, loF, hiF = paired_boot(y[te], bF, pF)
        d2, lo2, hi2 = paired_boot(y[te2], b2, p2)
        row["arms"][name] = {
            "auc_full": float(roc_auc_score(y[te], pF)),
            "auc_2min": float(roc_auc_score(y[te2], p2)),
            "brier": float(brier_score_loss(y[te], pF)),
            "delta_full": dF, "ci_full": [loF, hiF], "clears_full": bool(loF > 0),
            "delta_2min": d2, "ci_2min": [lo2, hi2], "clears_2min": bool(lo2 > 0)}
    return row


def main():
    print("=" * 100)
    print("STAGE 2 -- resample validation of the arms that beat production on a single fit")
    print("=" * 100)
    print(f"  baseline: {BASELINE[0]}")
    print(f"  survivors: {', '.join(n for n, _, _ in SURVIVORS)}")
    print(f"  {N_RESAMPLES} bootstrap resamples of the TRAINING rows; test set frozen")
    print(f"  every arm refit on the SAME resample as the baseline it is compared to\n")

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_resample, r): r for r in range(N_RESAMPLES)}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            print(f"  resample {futs[f]} done ({i}/{N_RESAMPLES}, "
                  f"{(time.time()-t0)/60:.1f} min elapsed)", flush=True)
    rows.sort(key=lambda r: r["rep"])

    print("\n" + "=" * 100)
    print("DISTRIBUTION ACROSS RESAMPLES -- the number that decides")
    print("=" * 100)
    base_f = np.array([r["baseline_full"] for r in rows])
    print(f"  production baseline: mean AUC {base_f.mean():.4f} "
          f"(sd {base_f.std():.4f}) -- note the spread, this is why stage 1 alone "
          f"is not evidence\n")
    print(f"  {'arm':<20}{'mean d_full':>13}{'sd':>8}{'pos':>7}{'clears':>9}"
          f"{'mean d_2min':>13}{'pos':>7}{'clears':>9}{'mean Brier':>12}")
    out = {"n_resamples": len(rows), "baseline": BASELINE[0],
           "baseline_mean_auc": float(base_f.mean()),
           "baseline_sd_auc": float(base_f.std()), "rows": rows, "summary": {}}
    bb = np.array([r["baseline_brier"] for r in rows])
    for name, _, _ in SURVIVORS:
        dF = np.array([r["arms"][name]["delta_full"] for r in rows])
        d2 = np.array([r["arms"][name]["delta_2min"] for r in rows])
        cF = sum(r["arms"][name]["clears_full"] for r in rows)
        c2 = sum(r["arms"][name]["clears_2min"] for r in rows)
        br = np.array([r["arms"][name]["brier"] for r in rows])
        print(f"  {name:<20}{dF.mean():>+13.4f}{dF.std():>8.4f}"
              f"{int((dF>0).sum()):>4}/{len(rows)}{cF:>6}/{len(rows)}"
              f"{d2.mean():>+13.4f}{int((d2>0).sum()):>4}/{len(rows)}"
              f"{c2:>6}/{len(rows)}{br.mean():>12.4f}")
        out["summary"][name] = {
            "delta_full_mean": float(dF.mean()), "delta_full_sd": float(dF.std()),
            "n_positive_full": int((dF > 0).sum()), "n_clearing_full": cF,
            "delta_2min_mean": float(d2.mean()), "n_positive_2min": int((d2 > 0).sum()),
            "n_clearing_2min": c2, "mean_brier": float(br.mean()),
            "baseline_mean_brier": float(bb.mean()), "n": len(rows)}

    print(f"\n  production mean Brier {bb.mean():.4f} (lower is better)")
    print("\n" + "=" * 100)
    best = max(out["summary"].items(), key=lambda kv: kv[1]["delta_full_mean"])
    robust = [k for k, v in out["summary"].items()
              if v["n_positive_full"] == len(rows) and v["n_positive_2min"] == len(rows)]
    if robust:
        print(f"POSITIVE ON EVERY RESAMPLE, BOTH POPULATIONS: {', '.join(robust)}")
    else:
        print("NO ARM IS POSITIVE ON EVERY RESAMPLE ON BOTH POPULATIONS")
    print(f"largest mean gain: {best[0]} {best[1]['delta_full_mean']:+.4f} full")
    print("=" * 100)
    out["robust_arms"] = robust

    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
