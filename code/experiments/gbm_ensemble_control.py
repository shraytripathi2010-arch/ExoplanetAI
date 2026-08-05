"""gbm_ensemble_control.py -- the control that gbm_ensemble.py needed.

WHAT THE FIRST RUN ACTUALLY SHOWED, AND WHY IT IS NOT AN ENSEMBLE RESULT

Averaging HGB + CatBoost + LightGBM + XGBoost beat the production HGB by
+0.0077 (full test) and +0.0096 (2-min), positive on 12/12 training-data
resamples. That looks like the first real win in 28 experiments. Decomposed,
it is not:

    ensemble mean AUC       0.9034
    CatBoost ALONE mean     0.9032     <- averaging adds +0.0002
    HGB (production) mean   0.8958     <- the whole gap is here

So the effect is entirely "CatBoost scores higher than HGB". The averaging
contributes nothing, which is consistent with the stacking result (families
too correlated to add independent signal) and means the finding must be
judged as a MODEL SWAP, not an ensemble.

THE CONFOUND IN THE ORIGINAL DESIGN -- MINE, NOT THE DATA'S

CatBoost, LightGBM and XGBoost each got a RandomizedSearchCV pass on the train
split. HGB did not: it used the deployed production configuration as-is. So
the comparison was TUNED challengers against an UNTUNED incumbent, and a
tuning advantage is a perfectly good explanation for +0.0075 that has nothing
to do with model family.

This script removes that asymmetry: HGB gets the same search budget over the
same grid the original bake-off used, and everything is re-measured on the
same 12 resamples. If the gap survives, it is a family difference. If it
closes, the first run measured hyperparameter tuning and nothing else.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import roc_auc_score  # exact drop-in, ~23x faster inside bootstraps

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
PRIOR = os.path.join(SCRIPT_DIR, "gbm_ensemble_results.json")
RESULTS = os.path.join(SCRIPT_DIR, "gbm_ensemble_control_results.json")

SEED = 42
N_RESAMPLES = 12
N_BOOT = 1500
TUNE_ITER = 10
TUNE_CV = 3


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def imp(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", est)])


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
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    te2 = te & is2

    prior = json.load(open(PRIOR))
    cat_params = prior["tuned_params"]["CatBoost"]

    print("=" * 88)
    print("CONTROL: was the ensemble gain a MODEL-FAMILY effect or a TUNING effect?")
    print("=" * 88)
    print("  first run, mean AUC across 12 resamples (full test):")
    print(f"    ensemble           0.9034")
    print(f"    CatBoost alone     {prior['per_model_mean_auc_full']['CatBoost']:.4f}"
          f"   <- averaging added +0.0002")
    print(f"    HGB (production)   {prior['per_model_mean_auc_full']['HGB']:.4f}"
          f"   <- UNTUNED, unlike the others")

    prod = joblib.load(PROD)
    hgb_prod = clone(getattr(prod, "estimator", prod))

    print(f"\nTuning HGB with the SAME budget the GBMs got "
          f"(n_iter={TUNE_ITER}, cv={TUNE_CV})...")
    grid = {"clf__max_iter": [300, 500, 700],
            "clf__max_leaf_nodes": [31, 63, 127],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__l2_regularization": [0.0, 0.5, 1.0]}
    cv = StratifiedKFold(TUNE_CV, shuffle=True, random_state=SEED)
    t0 = time.time()
    s = RandomizedSearchCV(
        imp(HistGradientBoostingClassifier(class_weight="balanced",
                                           random_state=SEED)),
        grid, n_iter=TUNE_ITER, cv=cv, scoring="roc_auc",
        random_state=SEED, n_jobs=1)
    s.fit(X[tr], y[tr])
    hgb_tuned = s.best_estimator_
    print(f"  HGB tuned  cv_auc={s.best_score_:.4f}  ({time.time()-t0:.0f}s)")
    print(f"    {dict(s.best_params_)}")
    print(f"  (CatBoost's tuned cv_auc was 0.9244 on the same folds)")

    from catboost import CatBoostClassifier
    cat = imp(CatBoostClassifier(verbose=0, random_seed=SEED,
                                 auto_class_weights="Balanced",
                                 allow_writing_files=False))
    cat.set_params(**cat_params)

    out = {"hgb_tuned_params": {k: v for k, v in s.best_params_.items()},
           "hgb_tuned_cv_auc": float(s.best_score_),
           "catboost_params": cat_params}

    print(f"\nRe-running {N_RESAMPLES} resamples: HGB-production, HGB-TUNED, CatBoost")
    print(f"  {'rep':>4}{'HGB prod':>10}{'HGB tuned':>11}{'CatBoost':>10}"
          f"{'cat-tuned':>11}{'ci_lo':>9}{'clr':>5}"
          f"{'cat-tuned 2m':>14}{'ci_lo':>9}{'clr':>5}")

    Xtr, ytr = X[tr], y[tr]
    n = len(ytr)
    rng = np.random.RandomState(7)          # SAME seed as the first run
    rows = []
    for rep in range(N_RESAMPLES):
        idx = rng.randint(0, n, n)
        Xb, yb = Xtr.iloc[idx], ytr[idx]
        if len(np.unique(yb)) < 2:
            continue
        mp = clone(hgb_prod).fit(Xb, yb)
        mt = clone(hgb_tuned).fit(Xb, yb)
        mc = clone(cat).fit(Xb, yb)
        pp_f, pt_f, pc_f = (m.predict_proba(X[te])[:, 1] for m in (mp, mt, mc))
        pp_2, pt_2, pc_2 = (m.predict_proba(X[te2])[:, 1] for m in (mp, mt, mc))
        aP = roc_auc_score(y[te], pp_f)
        aT = roc_auc_score(y[te], pt_f)
        aC = roc_auc_score(y[te], pc_f)
        d, lo, hi = paired_boot(y[te], pt_f, pc_f)          # cat vs TUNED hgb
        d2, lo2, hi2 = paired_boot(y[te2], pt_2, pc_2)
        rows.append({"rep": rep, "hgb_prod": float(aP), "hgb_tuned": float(aT),
                     "catboost": float(aC),
                     "hgb_tuned_2min": float(roc_auc_score(y[te2], pt_2)),
                     "catboost_2min": float(roc_auc_score(y[te2], pc_2)),
                     "delta_cat_vs_tuned_full": d, "ci_full": [lo, hi],
                     "clears_full": bool(lo > 0),
                     "delta_cat_vs_tuned_2min": d2, "ci_2min": [lo2, hi2],
                     "clears_2min": bool(lo2 > 0)})
        print(f"  {rep:>4}{aP:>10.4f}{aT:>11.4f}{aC:>10.4f}{d:>+11.4f}{lo:>+9.4f}"
              f"{'Y' if lo > 0 else 'n':>5}{d2:>+14.4f}{lo2:>+9.4f}"
              f"{'Y' if lo2 > 0 else 'n':>5}", flush=True)

    r = pd.DataFrame(rows)
    out["rows"] = rows

    print("\n" + "=" * 88)
    print("WHERE DID THE GAIN GO?")
    print("=" * 88)
    print(f"  mean AUC across resamples (full test):")
    print(f"    HGB production (untuned)  {r.hgb_prod.mean():.4f}")
    print(f"    HGB TUNED                 {r.hgb_tuned.mean():.4f}"
          f"   (tuning alone: {r.hgb_tuned.mean()-r.hgb_prod.mean():+.4f})")
    print(f"    CatBoost                  {r.catboost.mean():.4f}"
          f"   (family, over tuned HGB: {r.catboost.mean()-r.hgb_tuned.mean():+.4f})")
    out["mean_hgb_prod"] = float(r.hgb_prod.mean())
    out["mean_hgb_tuned"] = float(r.hgb_tuned.mean())
    out["mean_catboost"] = float(r.catboost.mean())
    out["tuning_effect"] = float(r.hgb_tuned.mean() - r.hgb_prod.mean())
    out["family_effect"] = float(r.catboost.mean() - r.hgb_tuned.mean())

    for pop, dcol, ccol in (("full clean test", "delta_cat_vs_tuned_full", "clears_full"),
                            ("2-min-only test", "delta_cat_vs_tuned_2min", "clears_2min")):
        d = r[dcol]
        print(f"\n  CatBoost vs TUNED HGB, {pop}:")
        print(f"    delta mean {d.mean():+.4f}  sd {d.std():.4f}  "
              f"min {d.min():+.4f}  max {d.max():+.4f}")
        print(f"    positive on {int((d > 0).sum())}/{len(r)} resamples;  "
              f"clearing ci_lo > 0 on {int(r[ccol].sum())}/{len(r)}")
        out.setdefault("summary", {})[pop] = {
            "delta_mean": float(d.mean()), "delta_sd": float(d.std()),
            "delta_min": float(d.min()), "delta_max": float(d.max()),
            "n_positive": int((d > 0).sum()), "n_clearing": int(r[ccol].sum()),
            "n_resamples": int(len(r))}

    tune_share = (out["tuning_effect"] /
                  (out["tuning_effect"] + out["family_effect"])
                  if (out["tuning_effect"] + out["family_effect"]) else float("nan"))
    print("\n" + "=" * 88)
    print(f"  tuning accounts for {100*tune_share:.0f}% of the original gap; "
          f"model family for {100*(1-tune_share):.0f}%")
    print("=" * 88)
    out["tuning_share_of_gap"] = float(tune_share)

    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
