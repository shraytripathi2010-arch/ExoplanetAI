"""calibration_sweep_validate_remaining.py -- the two arms stage 2 skipped.

WHY THERE IS A GAP TO CLOSE

The stage-2 survivor list was frozen while stage 1 was still running, so two
HGB arms that beat production never got resampled:

    bag-only cv=20    0.9056 single-fit  (+0.0024 vs production)
    isotonic cv=10    0.9051 single-fit  (+0.0019 vs production, best Brier 0.0881)

They were dismissed by reasoning -- neither tops the AUC table, and
`isotonic cv=10` is dominated by `sigmoid cv=10` (0.9059) at essentially equal
Brier -- but reasoning is not measurement, and every other single-fit gain in
this sweep evaporated on resampling. Closing the gap costs ~20 minutes.

Identical harness to `calibration_sweep_validate.py`: same baseline
(production, HGB + sigmoid + cv=5, refit per resample), same RNG seeds
(1000+rep), same paired bootstrap, same 8 resamples. Results are therefore
directly comparable to the five arms already reported rather than merely
similar.

`isotonic` needs the extra branch that the original harness lacked -- it only
knew "sigmoid" and "bag" -- which is the mechanical reason these two could not
simply be appended to that run.
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
RESULTS = os.path.join(SCRIPT_DIR, "calibration_sweep_remaining_results.json")

SEED = 42
N_RESAMPLES = 8
N_BOOT = 1500
N_WORKERS = 6

ARMS = [
    ("bag-only cv=20", "bag", 20),
    ("isotonic cv=10", "isotonic", 10),
]


class BagOnly:
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
    # frozen manifest test subset -- the post-freeze allocation changed today,
    # so pin evaluation to the same stars the earlier stage 2 used.
    frozen = m05.frozen_test_mask(df)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    prod = joblib.load(PROD)
    _G.update(X=X, y=y, tr=tr, te=frozen, te2=frozen & is2,
              hgb=clone(getattr(prod, "estimator", prod)))


def build(kind, k, bare):
    if kind == "sigmoid":
        return CalibratedClassifierCV(clone(bare), cv=k, method="sigmoid")
    if kind == "isotonic":
        return CalibratedClassifierCV(clone(bare), cv=k, method="isotonic")
    return BagOnly(clone(bare), cv=k)


def run_resample(rep):
    if not _G:
        _init()
    X, y, tr, te, te2, bare = (_G["X"], _G["y"], _G["tr"], _G["te"],
                               _G["te2"], _G["hgb"])
    Xtr, ytr = X[tr], y[tr]
    rng = np.random.RandomState(1000 + rep)      # SAME seeds as the first run
    idx = rng.randint(0, len(ytr), len(ytr))
    Xb, yb = Xtr.iloc[idx], ytr[idx]

    def ev(est):
        m = est.fit(Xb, yb)
        return m.predict_proba(X[te])[:, 1], m.predict_proba(X[te2])[:, 1]

    bF, b2 = ev(build("sigmoid", 5, bare))       # production baseline
    row = {"rep": rep, "baseline_full": float(roc_auc_score(y[te], bF)),
           "baseline_2min": float(roc_auc_score(y[te2], b2)),
           "baseline_brier": float(brier_score_loss(y[te], bF)), "arms": {}}
    for name, kind, k in ARMS:
        pF, p2 = ev(build(kind, k, bare))
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
    print("STAGE 2 -- the two arms the first survivor list missed")
    print("=" * 100)
    print("  baseline: HGB sigmoid cv=5 (production), refit per resample")
    print(f"  arms: {', '.join(n for n, _, _ in ARMS)}")
    print(f"  {N_RESAMPLES} resamples, seeds 1000..1007 -- identical to the")
    print("  first stage 2, so these numbers sit alongside those five directly.")
    print("  evaluated on the FROZEN manifest test subset.\n")

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_resample, r): r for r in range(N_RESAMPLES)}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            print(f"  resample {futs[f]} done ({i}/{N_RESAMPLES}, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
    rows.sort(key=lambda r: r["rep"])

    bf = np.array([r["baseline_full"] for r in rows])
    bb = np.array([r["baseline_brier"] for r in rows])
    print("\n" + "=" * 100)
    print("DISTRIBUTION ACROSS RESAMPLES")
    print("=" * 100)
    print(f"  production baseline: mean AUC {bf.mean():.4f} (sd {bf.std():.4f}), "
          f"mean Brier {bb.mean():.4f}\n")
    print(f"  {'arm':<20}{'mean d_full':>13}{'sd':>8}{'pos':>7}{'clears':>9}"
          f"{'mean d_2min':>13}{'pos':>7}{'clears':>9}{'Brier':>9}")
    out = {"n_resamples": len(rows), "baseline": "HGB sigmoid cv=5 (production)",
           "baseline_mean_auc": float(bf.mean()),
           "baseline_mean_brier": float(bb.mean()), "rows": rows, "summary": {}}
    for name, _, _ in ARMS:
        dF = np.array([r["arms"][name]["delta_full"] for r in rows])
        d2 = np.array([r["arms"][name]["delta_2min"] for r in rows])
        cF = sum(r["arms"][name]["clears_full"] for r in rows)
        c2 = sum(r["arms"][name]["clears_2min"] for r in rows)
        br = np.array([r["arms"][name]["brier"] for r in rows])
        print(f"  {name:<20}{dF.mean():>+13.4f}{dF.std():>8.4f}"
              f"{int((dF>0).sum()):>4}/{len(rows)}{cF:>6}/{len(rows)}"
              f"{d2.mean():>+13.4f}{int((d2>0).sum()):>4}/{len(rows)}"
              f"{c2:>6}/{len(rows)}{br.mean():>9.4f}")
        out["summary"][name] = {
            "delta_full_mean": float(dF.mean()), "delta_full_sd": float(dF.std()),
            "delta_full_min": float(dF.min()), "delta_full_max": float(dF.max()),
            "n_positive_full": int((dF > 0).sum()), "n_clearing_full": cF,
            "delta_2min_mean": float(d2.mean()), "n_positive_2min": int((d2 > 0).sum()),
            "n_clearing_2min": c2, "mean_brier": float(br.mean()), "n": len(rows)}

    print("\n" + "=" * 100)
    robust = [k for k, v in out["summary"].items()
              if v["n_clearing_full"] >= 0.9 * len(rows)
              and v["n_clearing_2min"] >= 0.9 * len(rows)]
    print("CLEARS ON >=90% OF RESAMPLES: " + (", ".join(robust) if robust else "none"))
    print("=" * 100)
    out["robust_arms"] = robust

    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
