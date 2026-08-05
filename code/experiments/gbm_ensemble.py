"""gbm_ensemble.py -- probability averaging across HGB / CatBoost / LightGBM /
XGBoost, measured on the axis that actually matters.

WHY THIS IS NOT THE STACKING EXPERIMENT AGAIN
The small-lift trio already tested a META-LEARNER stack (HGB+RF+LR) and it
failed for a diagnosed reason: hgb-rf test correlation 0.941, and the
meta-learner assigned LR a NEGATIVE coefficient. This is plain probability
averaging across four families, no meta-learner, no fitted combination weights
on the primary arm.

WHY BOOTSTRAP RESAMPLES AND NOT SEEDS
The pseudo-labelling work established this the hard way. HGB is DETERMINISTIC
at this dataset size -- `early_stopping='auto'` is off below 10,000 rows and
binning does not subsample -- so varying `random_state` for it changes nothing
and a seed sweep returns sd exactly 0.0000. CatBoost, separately, IS
stochastic and was shown to swing across seeds (sd 0.0024), which is how its
single-fit clear dissolved at 0/10. The axis that captures both is the
TRAINING DATA DRAW: resample the training rows, refit everything on that
resample, and compare against a baseline fit on the SAME resample. The test
set is never resampled -- it stays the frozen real-label set.

HYPERPARAMETERS ARE TUNED ONCE AND FROZEN
Tuning inside every resample would conflate tuning variance with ensemble
variance and cost 12x more. The three GBMs get one RandomizedSearchCV pass on
the train split, over the same grids as the original bake-off; the chosen
values are recorded in the output. HGB uses the PRODUCTION configuration
(cloned from the deployed artifact), so the baseline is the real thing rather
than a re-tuned lookalike. This does mean the GBM hyperparameters saw the full
training split -- stated plainly because it biases in the ENSEMBLE's favour,
and it still has to clear.

ENVIRONMENT
LightGBM and XGBoost need OpenMP. The working fix is the rpath one documented
in ENVIRONMENT_NOTES.md: torch's libomp copied to
/opt/homebrew/opt/libomp/lib/. The ctypes preload is NOT used and must not be
-- combined with the rpath fix it loads a second OpenMP runtime and segfaults
the process (exit 139, silent, no traceback).
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import roc_auc_score  # exact drop-in, ~23x faster inside bootstraps

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "gbm_ensemble_results.json")

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


def candidates():
    """Probe-first, exactly as ENVIRONMENT_NOTES.md prescribes. Anything that
    cannot be imported is reported, never silently dropped."""
    out, missing = {}, {}
    try:
        from catboost import CatBoostClassifier
        out["CatBoost"] = (imp(CatBoostClassifier(
            verbose=0, random_seed=SEED, auto_class_weights="Balanced",
            allow_writing_files=False)),
            {"clf__iterations": [300, 500, 800], "clf__depth": [4, 6, 8],
             "clf__learning_rate": [0.03, 0.05, 0.1],
             "clf__l2_leaf_reg": [1.0, 3.0, 9.0]})
    except Exception as e:
        missing["CatBoost"] = f"{type(e).__name__}: {str(e)[:120]}"
    try:
        import lightgbm as lgb
        out["LightGBM"] = (imp(lgb.LGBMClassifier(
            class_weight="balanced", verbose=-1, random_state=SEED)),
            {"clf__n_estimators": [300, 500, 800], "clf__num_leaves": [15, 31, 63],
             "clf__learning_rate": [0.03, 0.05, 0.1],
             "clf__min_child_samples": [10, 20, 40]})
    except Exception as e:
        missing["LightGBM"] = f"{type(e).__name__}: {str(e)[:120]}"
    try:
        import xgboost as xgb
        out["XGBoost"] = (imp(xgb.XGBClassifier(
            random_state=SEED, eval_metric="logloss", tree_method="hist")),
            {"clf__n_estimators": [300, 500, 800], "clf__max_depth": [3, 5, 7],
             "clf__learning_rate": [0.03, 0.05, 0.1],
             "clf__scale_pos_weight": [1.0, 0.26], "clf__reg_lambda": [1.0, 5.0]})
    except Exception as e:
        missing["XGBoost"] = f"{type(e).__name__}: {str(e)[:120]}"
    return out, missing


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

    prod = joblib.load(PROD)
    hgb_proto = clone(getattr(prod, "estimator", prod))

    out = {"n_resamples": N_RESAMPLES}
    models, missing = candidates()
    print("=" * 88)
    print("GBM AVERAGING ENSEMBLE")
    print("=" * 88)
    print(f"  available : HGB (production config) + {', '.join(models)}")
    if missing:
        for k, v in missing.items():
            print(f"  UNAVAILABLE: {k} -- {v}")
    out["unavailable"] = missing
    print(f"  train {int(tr.sum())} | test {int(te.sum())} | "
          f"2-min-only test {int(te2.sum())}")

    # ---------------------------------------------------- tune once, then freeze
    print(f"\nTuning {len(models)} GBMs once on the train split "
          f"(RandomizedSearchCV, n_iter={TUNE_ITER}, cv={TUNE_CV})...")
    tuned, params, cv_auc = {}, {}, {}
    cv = StratifiedKFold(TUNE_CV, shuffle=True, random_state=SEED)
    for name, (pipe, grid) in models.items():
        t0 = time.time()
        s = RandomizedSearchCV(pipe, grid, n_iter=TUNE_ITER, cv=cv,
                               scoring="roc_auc", random_state=SEED, n_jobs=1)
        s.fit(X[tr], y[tr])
        tuned[name] = s.best_estimator_
        params[name] = {k: v for k, v in s.best_params_.items()}
        cv_auc[name] = float(s.best_score_)
        print(f"  {name:<10} cv_auc={s.best_score_:.4f}  ({time.time()-t0:.0f}s)")
        print(f"    {params[name]}")
    out["tuned_params"] = params
    # Was previously written as a zip against a list of zeros, which recorded
    # 0.0 for every family while the printed values were correct. Cosmetic in
    # the log, wrong in the JSON -- exactly the kind of silent discrepancy that
    # makes a saved result untrustworthy later.
    out["tuned_cv_auc"] = cv_auc

    protos = {"HGB": hgb_proto}
    protos.update({k: clone(v) for k, v in tuned.items()})

    # ------------------------------------------------- resample distribution
    print(f"\nRunning {N_RESAMPLES} training-data bootstrap resamples "
          f"({len(protos)} models each)...")
    print(f"  {'rep':>4}{'HGB base':>10}{'ens full':>10}{'d_full':>9}"
          f"{'ci_lo':>9}{'clr':>5}{'HGB 2min':>10}{'ens 2min':>10}"
          f"{'d_2min':>9}{'ci_lo':>9}{'clr':>5}")

    Xtr, ytr = X[tr], y[tr]
    n = len(ytr)
    rng = np.random.RandomState(7)
    rows = []
    for rep in range(N_RESAMPLES):
        idx = rng.randint(0, n, n)
        Xb, yb = Xtr.iloc[idx], ytr[idx]
        if len(np.unique(yb)) < 2:
            continue
        pf, p2 = {}, {}
        for name, proto in protos.items():
            m = clone(proto).fit(Xb, yb)
            pf[name] = m.predict_proba(X[te])[:, 1]
            p2[name] = m.predict_proba(X[te2])[:, 1]
        ens_f = np.mean([pf[k] for k in protos], axis=0)
        ens_2 = np.mean([p2[k] for k in protos], axis=0)

        aF0 = roc_auc_score(y[te], pf["HGB"])
        a20 = roc_auc_score(y[te2], p2["HGB"])
        aF1 = roc_auc_score(y[te], ens_f)
        a21 = roc_auc_score(y[te2], ens_2)
        dF, loF, hiF = paired_boot(y[te], pf["HGB"], ens_f)
        d2, lo2, hi2 = paired_boot(y[te2], p2["HGB"], ens_2)
        rows.append({"rep": rep, "hgb_full": float(aF0), "ens_full": float(aF1),
                     "delta_full": dF, "ci_lo_full": loF, "ci_hi_full": hiF,
                     "clears_full": bool(loF > 0),
                     "hgb_2min": float(a20), "ens_2min": float(a21),
                     "delta_2min": d2, "ci_lo_2min": lo2, "ci_hi_2min": hi2,
                     "clears_2min": bool(lo2 > 0),
                     "per_model_full": {k: float(roc_auc_score(y[te], pf[k]))
                                        for k in protos}})
        print(f"  {rep:>4}{aF0:>10.4f}{aF1:>10.4f}{dF:>+9.4f}{loF:>+9.4f}"
              f"{'Y' if loF > 0 else 'n':>5}"
              f"{a20:>10.4f}{a21:>10.4f}{d2:>+9.4f}{lo2:>+9.4f}"
              f"{'Y' if lo2 > 0 else 'n':>5}", flush=True)

    r = pd.DataFrame(rows)
    out["rows"] = rows

    print("\n" + "=" * 88)
    print("RESAMPLE DISTRIBUTION -- the whole point; a single fit is one draw")
    print("=" * 88)
    for pop, dcol, ccol in (("full clean test", "delta_full", "clears_full"),
                            ("2-min-only test", "delta_2min", "clears_2min")):
        d = r[dcol]
        print(f"  {pop}")
        print(f"    delta: mean {d.mean():+.4f}  sd {d.std():.4f}  "
              f"min {d.min():+.4f}  max {d.max():+.4f}")
        print(f"    resamples with a POSITIVE delta : "
              f"{int((d > 0).sum())}/{len(r)}")
        print(f"    resamples CLEARING ci_lo > 0    : "
              f"{int(r[ccol].sum())}/{len(r)}")
        out.setdefault("summary", {})[pop] = {
            "delta_mean": float(d.mean()), "delta_sd": float(d.std()),
            "delta_min": float(d.min()), "delta_max": float(d.max()),
            "n_positive": int((d > 0).sum()),
            "n_clearing": int(r[ccol].sum()), "n_resamples": int(len(r))}

    print("\n  mean per-model AUC across resamples (full test):")
    pm = pd.DataFrame([x["per_model_full"] for x in rows])
    for k in pm.columns:
        print(f"    {k:<10} {pm[k].mean():.4f}  (sd {pm[k].std():.4f})")
    out["per_model_mean_auc_full"] = {k: float(pm[k].mean()) for k in pm.columns}
    out["per_model_sd_auc_full"] = {k: float(pm[k].std()) for k in pm.columns}

    clears_maj = (out["summary"]["full clean test"]["n_clearing"] >= 0.9 * len(r)
                  and out["summary"]["2-min-only test"]["n_clearing"] >= 0.9 * len(r))
    print("\n" + "=" * 88)
    print("VERDICT: " + ("CLEARS ROBUSTLY -- >=90% of resamples clear on BOTH "
                         "populations" if clears_maj else
                         "DOES NOT CLEAR ROBUSTLY under the resample distribution"))
    print("=" * 88)
    out["clears_robustly"] = bool(clears_maj)

    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
