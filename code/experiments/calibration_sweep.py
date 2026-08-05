"""calibration_sweep.py -- is production's +0.0046 from CALIBRATION or from BAGGING?

THE OBSERVATION THIS STARTS FROM

    HGB       bare 0.8986 -> CalibratedClassifierCV(cv=5, sigmoid) 0.9032   +0.0046
    CatBoost  bare 0.9113 -> CalibratedClassifierCV(cv=5, sigmoid) 0.9089   -0.0024

+0.0046 is larger than any of the 28 feature/architecture experiments produced,
and it arrived as a side effect of a calibration wrapper nobody was treating as
a modelling choice.

A SIGMOID CANNOT DO THIS, AND THAT IS THE POINT

ROC-AUC is invariant under any monotone transform of the scores. Applying one
sigmoid to one model's outputs cannot move AUC by even 0.0001. So the +0.0046
is NOT calibration -- it is arithmetically impossible for it to be calibration.
`CalibratedClassifierCV(cv=k)` quietly does two things:

    1. fits k models, each on (k-1)/k of the training data
    2. applies a per-fold sigmoid, then AVERAGES the k calibrated outputs

Only step 2's averaging can change AUC, and it does so because an average of k
monotone transforms of k DIFFERENT models is not a monotone transform of any
one of them. Production has therefore been running a 5-model bagging ensemble
since the day it shipped, and reporting it as "calibration".

WHAT THIS SWEEP MEASURES

  * does more averaging help? cv = 3, 5 (production), 10, 20
  * does the per-fold sigmoid contribute anything beyond the averaging?
    -> BAG-ONLY arms average the k models' RAW probabilities with no sigmoid.
       If bag-only matches sigmoid, calibration is doing nothing for AUC and
       the whole effect is bagging.
  * isotonic vs sigmoid
  * does the same lever work for CatBoost, which currently LOSES from it?

METHODOLOGY, APPLYING THE LESSON FROM THE BASELINE AUDIT

A single fit is not evidence. The leave-one-out audit measured sd(delta)=0.0024
between paired arms, with the unperturbed draw landing outside the perturbed
distribution's centre. Stage 2 therefore re-runs the surviving arms over
bootstrap resamples of the TRAINING ROWS and reports the distribution. Stage 1
is a cheap landscape scan only, and is labelled as such rather than reported as
a result.
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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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
RESULTS = os.path.join(SCRIPT_DIR, "calibration_sweep_results.json")

SEED = 42
N_BOOT = 1500
N_RESAMPLES = 10
CAT_PARAMS = dict(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=9.0)


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


class BagOnly:
    """Average k models fit on the same CV folds CalibratedClassifierCV uses,
    with NO calibration applied. This is the control that separates 'averaging'
    from 'sigmoid' -- without it the two are confounded in every measurement
    this project has ever made of the production wrapper."""

    def __init__(self, base, cv=5, seed=SEED):
        self.base, self.cv, self.seed = base, cv, seed

    def fit(self, X, y):
        self.models_ = []
        skf = StratifiedKFold(self.cv, shuffle=True, random_state=self.seed)
        for tr_idx, _ in skf.split(X, y):
            m = clone(self.base)
            m.fit(X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx], y[tr_idx])
            self.models_.append(m)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)


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


def build_arms(bare):
    """(name, estimator-factory). Production is cv=5 sigmoid."""
    arms = [("bare (no wrapper)", lambda: clone(bare))]
    for k in (3, 5, 10, 20):
        tag = " [PRODUCTION]" if k == 5 else ""
        arms.append((f"sigmoid cv={k}{tag}",
                     lambda k=k: CalibratedClassifierCV(clone(bare), cv=k,
                                                        method="sigmoid")))
        arms.append((f"bag-only cv={k}", lambda k=k: BagOnly(clone(bare), cv=k)))
    for k in (5, 10):
        arms.append((f"isotonic cv={k}",
                     lambda k=k: CalibratedClassifierCV(clone(bare), cv=k,
                                                        method="isotonic")))
    return arms


def main():
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
    hgb_bare = clone(getattr(prod, "estimator", prod))
    from catboost import CatBoostClassifier
    cat_bare = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("clf", CatBoostClassifier(
                             verbose=0, random_seed=SEED,
                             auto_class_weights="Balanced",
                             allow_writing_files=False, **CAT_PARAMS))])

    out = {"note": "AUC is invariant to monotone transforms; a single sigmoid "
                   "cannot move it. Any AUC change from CalibratedClassifierCV "
                   "is the k-fold AVERAGING, not the calibration."}

    print("=" * 96)
    print("STAGE 1 -- LANDSCAPE SCAN (single fit; NOT a result, see stage 2)")
    print("=" * 96)
    print(f"  train {int(tr.sum())} | test {int(te.sum())} | 2-min {int(te2.sum())}")

    stage1 = {}
    for mdl_name, bare in (("HGB", hgb_bare), ("CatBoost", cat_bare)):
        print(f"\n  --- {mdl_name} ---")
        print(f"  {'arm':<26}{'full AUC':>10}{'2-min AUC':>11}{'Brier':>9}{'fit s':>8}")
        stage1[mdl_name] = {}
        for name, factory in build_arms(bare):
            t0 = time.time()
            try:
                m = factory().fit(X[tr], y[tr])
                pf = m.predict_proba(X[te])[:, 1]
                p2 = m.predict_proba(X[te2])[:, 1]
                aF, a2 = roc_auc_score(y[te], pf), roc_auc_score(y[te2], p2)
                br = brier_score_loss(y[te], pf)
                stage1[mdl_name][name] = {"auc_full": float(aF), "auc_2min": float(a2),
                                          "brier": float(br),
                                          "fit_s": round(time.time() - t0, 1)}
                print(f"  {name:<26}{aF:>10.4f}{a2:>11.4f}{br:>9.4f}"
                      f"{time.time()-t0:>8.0f}", flush=True)
            except Exception as e:
                stage1[mdl_name][name] = {"error": f"{type(e).__name__}: {str(e)[:90]}"}
                print(f"  {name:<26}  FAILED {type(e).__name__}: {str(e)[:60]}")
    out["stage1"] = stage1

    # --------------------------------------------------------------- stage 2
    base_full = stage1["HGB"].get("sigmoid cv=5 [PRODUCTION]", {}).get("auc_full")
    cands = []
    for mdl_name in ("HGB", "CatBoost"):
        for name, v in stage1[mdl_name].items():
            if "auc_full" in v and base_full and v["auc_full"] > base_full + 0.001:
                cands.append((mdl_name, name, v["auc_full"]))
    cands.sort(key=lambda r: -r[2])
    print("\n" + "=" * 96)
    print("ARMS BEATING PRODUCTION ON THE SINGLE FIT (candidates for stage 2)")
    print("=" * 96)
    if not cands:
        print("  none -- nothing to validate")
    for mdl_name, name, a in cands:
        print(f"  {mdl_name:<10}{name:<26}{a:.4f}  (+{a-base_full:.4f} vs production)")
    out["stage1_candidates"] = [{"model": m, "arm": n, "auc_full": a}
                                for m, n, a in cands]

    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {RESULTS}")
    print("\nStage 2 (resample validation of the survivors) runs separately --")
    print("see calibration_sweep_validate.py.")


if __name__ == "__main__":
    main()
