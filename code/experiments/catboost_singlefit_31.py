"""catboost_singlefit_31.py -- PART 1 only: what does a single CatBoost fit do
against the CURRENT 31-feature baseline?

CatBoost has never been tested against crowding or variability. Both the
original CatBoost result (+0.0085 single fit, 0/10 on replication) and the
retrieved cross-family ensemble (+0.0077, 8/12) were measured on the
24-feature/0.9031 model, two deployments ago. Neither says anything about now.

Reported before any ensemble is built, per the task.

HYPERPARAMETERS. The original stress test carried an explicit caveat: it used
"fixed mid-range CatBoost hyperparameters, not the search-selected config
(those were lost when the main run crashed before writing its JSON)", so part of
the +0.0085 -> +0.0013 gap might have been tuning rather than seed. Those
parameters were RECOVERED in `gbm_ensemble_results.json`:

    learning_rate 0.05, depth 8, l2_leaf_reg 9.0, iterations 500

so this run closes that caveat by using the actual search-selected config.

BOTH ARMS USE PRODUCTION'S WRAPPER -- CalibratedClassifierCV(cv=5, sigmoid) --
so AUC, Brier and ECE are all directly comparable to the deployed 0.9300 /
0.0832 / 0.0365 rather than needing a caveat.
"""
import os
import sys
import json
import time
import importlib.util
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import fast_auc  # noqa: E402

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
OUT = os.path.join(SCRIPT_DIR, "catboost_singlefit_31_results.json")

SEED, N_BOOT = 42, 2000
CAT_PARAMS = dict(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=9.0)


def ece(y, p, bins=15):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = np.linspace(0, 1, bins + 1)
    i = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    return float(sum((i == b).mean() * abs(y[i == b].mean() - p[i == b].mean())
                     for b in range(bins) if (i == b).any()))


def paired_boot(y, pa, pb, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        yi = y[i]
        if yi.sum() in (0, len(yi)):
            continue
        d.append(fast_auc(yi, pb[i]) - fast_auc(yi, pa[i]))
    d = np.asarray(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def catboost_pipe(seed):
    from catboost import CatBoostClassifier
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", CatBoostClassifier(
                         verbose=0, random_seed=seed,
                         auto_class_weights="Balanced",
                         allow_writing_files=False, **CAT_PARAMS))])


def main():
    print("=" * 92)
    print("PART 1 -- single-fit CatBoost vs the CURRENT 31-feature baseline")
    print("=" * 92)
    m05 = _m05()
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 31
    df = pd.read_csv(TRAINING)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    tr, _ = m05.split_by_host(df)
    tr = np.asarray(tr)
    te = m05.frozen_test_mask(df)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    te2 = te & is2
    print(f"  train {int(tr.sum())} | frozen test {int(te.sum())} | 2-min {int(te2.sum())}")
    print(f"  CatBoost params (search-selected, recovered): {CAT_PARAMS}\n")

    prod = joblib.load(PROD)
    hgb = clone(getattr(prod, "estimator", prod))

    res = {}
    preds = {}
    for name, base in [("HGB (production recipe)", hgb),
                       ("CatBoost", catboost_pipe(SEED))]:
        t0 = time.time()
        est = CalibratedClassifierCV(clone(base), cv=5, method="sigmoid")
        est.fit(X[tr], y[tr])
        pF = est.predict_proba(X.loc[te])[:, 1]
        p2 = est.predict_proba(X.loc[te2])[:, 1]
        preds[name] = (pF, p2)
        a = roc_auc_score(y[te], pF)
        res[name] = {"auc": float(a), "auc_2min": float(roc_auc_score(y[te2], p2)),
                     "brier": float(brier_score_loss(y[te], pF)),
                     "ece": ece(y[te], pF), "fit_s": round(time.time() - t0, 1)}
        print(f"  {name:<26} AUC {a:.4f}  2min {res[name]['auc_2min']:.4f}  "
              f"Brier {res[name]['brier']:.4f}  ECE {res[name]['ece']:.4f}  "
              f"({res[name]['fit_s']}s)")

    d, lo, hi = paired_boot(y[te], preds["HGB (production recipe)"][0], preds["CatBoost"][0])
    d2, lo2, hi2 = paired_boot(y[te2], preds["HGB (production recipe)"][1], preds["CatBoost"][1])
    print(f"\n  paired bootstrap, CatBoost - HGB:")
    print(f"    full test : {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  clears={lo>0}")
    print(f"    2-min     : {d2:+.4f}  95% CI [{lo2:+.4f}, {hi2:+.4f}]  clears={lo2>0}")
    print(f"\n  deployed production reference: AUC 0.9300, Brier 0.0832, ECE 0.0365")
    res["delta"] = {"full": d, "ci_lo": lo, "ci_hi": hi, "clears": bool(lo > 0),
                    "2min": d2, "ci_lo_2min": lo2, "clears_2min": bool(lo2 > 0)}
    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
