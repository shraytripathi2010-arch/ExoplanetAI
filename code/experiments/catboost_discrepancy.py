"""catboost_discrepancy.py -- resolve seed-check 0/10 vs training-draw 11/12.

THE TWO RESULTS

  bakeoff_followup.py  seed stress   10 seeds   0/10 cleared, deltas ~0/negative
                                                (e.g. seed 1000: cat 0.9011,
                                                 hgb 0.9036, delta -0.0024)
  gbm_ensemble_control.py  resamples 12 draws  11/12 positive, mean +0.0073

Both cannot be describing the same quantity. Reading the two scripts, they
differ in THREE ways at once, so the earlier conclusion ("seeds vs training
draws") was a guess at which one mattered:

  1. CATBOOST HYPERPARAMETERS
       seed check : iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0
                    -- hardcoded, NOT the tuned values
       control    : iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=9.0
                    -- from RandomizedSearchCV
  2. THE HGB BASELINE IT WAS COMPARED AGAINST
       seed check reported hgb_full ~0.9036; the same production config refit
       today scores 0.8986. A 0.0050 gap that should not exist if HGB is
       deterministic and the data is unchanged.
  3. THE RANDOMISATION AXIS
       seeds (fit stochasticity) vs bootstrap resamples (training draw).

This runs the full factorial -- {seed-check params, tuned params} x {seed axis,
resample axis} -- against one shared, freshly measured HGB baseline, so each
cause can be read off independently instead of inferred.

It also directly tests the determinism claim that the whole "seeds are the
wrong axis" argument rests on.
"""
import os
import sys
import json
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import roc_auc_score  # exact drop-in, ~23x faster inside bootstraps

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "catboost_discrepancy_results.json")

N = 10                      # seeds AND resamples, so the counts are comparable
N_BOOT = 1500
SEED = 42

SEEDCHECK_PARAMS = dict(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0)
TUNED_PARAMS = dict(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=9.0)


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def imp(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", est)])


def boot(y, pa, pb, n=N_BOOT, seed=SEED):
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


def cb(params, seed):
    from catboost import CatBoostClassifier
    return imp(CatBoostClassifier(verbose=0, random_seed=seed,
                                  auto_class_weights="Balanced",
                                  allow_writing_files=False, **params))


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
    hgb_params = getattr(prod, "estimator", prod).named_steps["clf"].get_params()
    out = {"n": N, "seedcheck_params": SEEDCHECK_PARAMS, "tuned_params": TUNED_PARAMS}

    print("=" * 92)
    print("RESOLVING: seed check 0/10 vs training-draw 11/12")
    print("=" * 92)
    print(f"  train {int(tr.sum())} | test {int(te.sum())} | 2-min {int(te2.sum())}")

    # ---- CAUSE 2 first: is the HGB baseline reproducible, and deterministic? --
    print("\n" + "-" * 92)
    print("CAUSE 2 -- the HGB baseline. Seed check reported 0.9036; what is it today?")
    print("-" * 92)
    hgb_aucs = []
    for sd in (1000, 1001, 1002):
        hp = dict(hgb_params); hp["random_state"] = sd
        m = imp(HistGradientBoostingClassifier(**hp)).fit(X[tr], y[tr])
        a = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        hgb_aucs.append(float(a))
        print(f"  HGB production config, random_state={sd}: full-test AUC {a:.6f}")
    det = (max(hgb_aucs) - min(hgb_aucs)) < 1e-12
    print(f"\n  deterministic across seeds: {det}  "
          f"(spread {max(hgb_aucs)-min(hgb_aucs):.2e})")
    print(f"  today's value {hgb_aucs[0]:.4f} vs the seed check's reported 0.9036 "
          f"-> gap {hgb_aucs[0]-0.9036:+.4f}")
    out["hgb_today"] = hgb_aucs[0]
    out["hgb_seedcheck_reported"] = 0.9036
    out["hgb_deterministic"] = bool(det)
    out["hgb_gap_vs_seedcheck"] = float(hgb_aucs[0] - 0.9036)

    hgb_fixed = imp(HistGradientBoostingClassifier(**hgb_params)).fit(X[tr], y[tr])
    ph_f = hgb_fixed.predict_proba(X[te])[:, 1]
    ph_2 = hgb_fixed.predict_proba(X[te2])[:, 1]

    # ---- CAUSE 1: do the two CatBoost configurations differ? -----------------
    print("\n" + "-" * 92)
    print("CAUSE 1 -- CatBoost hyperparameters, single fit, same data, same baseline")
    print("-" * 92)
    single = {}
    for nm, prm in (("seed-check (depth 6, l2 3)", SEEDCHECK_PARAMS),
                    ("tuned      (depth 8, l2 9)", TUNED_PARAMS)):
        m = cb(prm, SEED).fit(X[tr], y[tr])
        pf = m.predict_proba(X[te])[:, 1]
        p2 = m.predict_proba(X[te2])[:, 1]
        d, lo, hi = boot(y[te], ph_f, pf)
        d2, lo2, hi2 = boot(y[te2], ph_2, p2)
        aF, a2 = roc_auc_score(y[te], pf), roc_auc_score(y[te2], p2)
        print(f"  {nm}  full {aF:.4f}  d {d:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{'CLEARS' if lo > 0 else 'no':<7} | 2min {a2:.4f} d {d2:+.4f} "
              f"[{lo2:+.4f},{hi2:+.4f}] {'CLEARS' if lo2 > 0 else 'no'}")
        single[nm] = {"auc_full": float(aF), "auc_2min": float(a2),
                      "delta_full": d, "ci_full": [lo, hi], "clears_full": bool(lo > 0),
                      "delta_2min": d2, "ci_2min": [lo2, hi2], "clears_2min": bool(lo2 > 0)}
    out["single_fit_by_params"] = single

    # ---- CAUSE 3: the factorial ---------------------------------------------
    print("\n" + "-" * 92)
    print(f"CAUSE 3 -- axis. {N} SEEDS vs {N} RESAMPLES, each param set, same baseline")
    print("-" * 92)
    print(f"  {'params':<12}{'axis':<11}{'mean d_full':>13}{'sd':>9}"
          f"{'pos':>7}{'clears':>9}{'mean d_2min':>13}{'pos':>7}{'clears':>9}")

    Xtr, ytr = X[tr], y[tr]
    n_tr = len(ytr)
    grid = {}
    for pname, prm in (("seed-check", SEEDCHECK_PARAMS), ("tuned", TUNED_PARAMS)):
        for axis in ("seeds", "resamples"):
            rng = np.random.RandomState(7)
            dF, d2, cF, c2 = [], [], 0, 0
            for i in range(N):
                if axis == "seeds":
                    Xb, yb, sd = Xtr, ytr, 1000 + i
                    bF, b2 = ph_f, ph_2          # HGB is deterministic; fixed
                else:
                    idx = rng.randint(0, n_tr, n_tr)
                    Xb, yb, sd = Xtr.iloc[idx], ytr[idx], SEED
                    hb = imp(HistGradientBoostingClassifier(**hgb_params)).fit(Xb, yb)
                    bF = hb.predict_proba(X[te])[:, 1]
                    b2 = hb.predict_proba(X[te2])[:, 1]
                m = cb(prm, sd).fit(Xb, yb)
                pf = m.predict_proba(X[te])[:, 1]
                p2 = m.predict_proba(X[te2])[:, 1]
                a, lo, _ = boot(y[te], bF, pf)
                b, lo2, _ = boot(y[te2], b2, p2)
                dF.append(a); d2.append(b)
                cF += int(lo > 0); c2 += int(lo2 > 0)
            dF, d2 = np.array(dF), np.array(d2)
            print(f"  {pname:<12}{axis:<11}{dF.mean():>+13.4f}{dF.std():>9.4f}"
                  f"{int((dF>0).sum()):>4}/{N}{cF:>6}/{N}"
                  f"{d2.mean():>+13.4f}{int((d2>0).sum()):>4}/{N}{c2:>6}/{N}", flush=True)
            grid[f"{pname}|{axis}"] = {
                "delta_full_mean": float(dF.mean()), "delta_full_sd": float(dF.std()),
                "n_positive_full": int((dF > 0).sum()), "n_clearing_full": cF,
                "delta_2min_mean": float(d2.mean()), "n_positive_2min": int((d2 > 0).sum()),
                "n_clearing_2min": c2, "n": N}
    out["factorial"] = grid

    print("\n" + "=" * 92)
    print("READ-OFF")
    print("=" * 92)
    pe = (grid["tuned|seeds"]["delta_full_mean"]
          - grid["seed-check|seeds"]["delta_full_mean"])
    ae = (grid["seed-check|resamples"]["delta_full_mean"]
          - grid["seed-check|seeds"]["delta_full_mean"])
    print(f"  hyperparameter effect (tuned - seed-check, seeds axis) : {pe:+.4f}")
    print(f"  axis effect (resamples - seeds, seed-check params)     : {ae:+.4f}")
    print(f"  HGB baseline gap vs the seed check's reported figure   : "
          f"{out['hgb_gap_vs_seedcheck']:+.4f}")
    out["hyperparameter_effect"] = float(pe)
    out["axis_effect"] = float(ae)

    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
