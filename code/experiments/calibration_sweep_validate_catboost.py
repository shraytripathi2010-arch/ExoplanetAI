"""calibration_sweep_validate_catboost.py -- stage 2 for the CatBoost arms.

WHY THIS IS A SEPARATE RUN

Stage 2 on the HGB arms dissolved every single-fit gain: +0.0027..+0.0029
became +0.0001..+0.0007, and the best-looking arm reversed sign. CatBoost's
single-fit arms sit at +0.0071..+0.0092 over production -- larger, but produced
by exactly the same one-draw procedure that just proved unreliable. They get
the same treatment before anything is claimed.

ARMS, chosen to answer three separate questions rather than to survey

  CatBoost bare            is the family advantage real at all, stripped of
                           any wrapper? (1 fit)
  CatBoost sigmoid cv=5    the LIKE-FOR-LIKE comparison. Production is
                           HGB+sigmoid+cv=5; this is the same wrapper with the
                           model swapped, which is what the promotion gate
                           would actually see. (5 fits)
  CatBoost bag-only cv=10  best single-fit arm overall, 0.9124. (10 fits)
  CatBoost sigmoid cv=20   best CALIBRATED arm, 0.9120, and the one that would
                           actually be deployable since it keeps probability
                           quality. (20 fits)

Omitted deliberately: cv=3 arms (already shown harmful in both families),
isotonic (dominated by sigmoid at equal fold count), and bag-only cv=3/5/20
(bracketed by the arms above at ~3x the compute).

BASELINE is production itself -- HGB + sigmoid + cv=5 -- refit on each
resample, never a stored constant. A challenger must beat what is deployed.
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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "calibration_sweep_catboost_results.json")

SEED = 42
N_RESAMPLES = 8
N_BOOT = 1500
N_WORKERS = 6
CAT_PARAMS = dict(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=9.0)

ARMS = [
    ("CatBoost bare", "bare", None),
    ("CatBoost sigmoid cv=5 (like-for-like)", "sigmoid", 5),
    ("CatBoost bag-only cv=10", "bag", 10),
    ("CatBoost sigmoid cv=20", "sigmoid", 20),
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
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce")
    te2 = te & (((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy())
    prod = joblib.load(PROD)
    from catboost import CatBoostClassifier
    cat_bare = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("clf", CatBoostClassifier(
                             verbose=0, random_seed=SEED,
                             auto_class_weights="Balanced",
                             allow_writing_files=False, **CAT_PARAMS))])
    _G.update(X=X, y=y, tr=tr, te=te, te2=te2,
              hgb=clone(getattr(prod, "estimator", prod)), cat=cat_bare)


def run_resample(rep):
    if not _G:
        _init()
    X, y, tr, te, te2 = _G["X"], _G["y"], _G["tr"], _G["te"], _G["te2"]
    Xtr, ytr = X[tr], y[tr]
    n = len(ytr)
    rng = np.random.RandomState(1000 + rep)     # SAME seeds as the HGB stage 2
    idx = rng.randint(0, n, n)
    Xb, yb = Xtr.iloc[idx], ytr[idx]

    def ev(est):
        m = est.fit(Xb, yb)
        return m.predict_proba(X[te])[:, 1], m.predict_proba(X[te2])[:, 1]

    # baseline: production exactly as deployed
    bF, b2 = ev(CalibratedClassifierCV(clone(_G["hgb"]), cv=5, method="sigmoid"))
    row = {"rep": rep,
           "baseline_full": float(roc_auc_score(y[te], bF)),
           "baseline_2min": float(roc_auc_score(y[te2], b2)),
           "baseline_brier": float(brier_score_loss(y[te], bF)),
           "arms": {}}

    for name, kind, k in ARMS:
        base = clone(_G["cat"])
        if kind == "bare":
            est = base
        elif kind == "sigmoid":
            est = CalibratedClassifierCV(base, cv=k, method="sigmoid")
        else:
            est = BagOnly(base, cv=k)
        pF, p2 = ev(est)
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
    print("=" * 104)
    print("STAGE 2 -- CatBoost arms vs PRODUCTION (HGB + sigmoid + cv=5), resampled")
    print("=" * 104)
    print(f"  {N_RESAMPLES} bootstrap resamples of the training rows, same seeds")
    print(f"  as the HGB stage 2 so the two runs are directly comparable.")
    print(f"  arms: {', '.join(n for n, _, _ in ARMS)}\n")

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
    print("\n" + "=" * 104)
    print("DISTRIBUTION ACROSS RESAMPLES")
    print("=" * 104)
    print(f"  production baseline: mean AUC {bf.mean():.4f} (sd {bf.std():.4f}), "
          f"mean Brier {bb.mean():.4f}\n")
    print(f"  {'arm':<40}{'mean d_full':>13}{'sd':>8}{'pos':>7}{'clears':>9}"
          f"{'mean d_2min':>13}{'pos':>7}{'clears':>9}{'Brier':>9}")
    out = {"n_resamples": len(rows), "baseline": "HGB sigmoid cv=5 (production)",
           "baseline_mean_auc": float(bf.mean()),
           "baseline_sd_auc": float(bf.std()),
           "baseline_mean_brier": float(bb.mean()),
           "rows": rows, "summary": {}}
    for name, _, _ in ARMS:
        dF = np.array([r["arms"][name]["delta_full"] for r in rows])
        d2 = np.array([r["arms"][name]["delta_2min"] for r in rows])
        cF = sum(r["arms"][name]["clears_full"] for r in rows)
        c2 = sum(r["arms"][name]["clears_2min"] for r in rows)
        br = np.array([r["arms"][name]["brier"] for r in rows])
        print(f"  {name:<40}{dF.mean():>+13.4f}{dF.std():>8.4f}"
              f"{int((dF>0).sum()):>4}/{len(rows)}{cF:>6}/{len(rows)}"
              f"{d2.mean():>+13.4f}{int((d2>0).sum()):>4}/{len(rows)}"
              f"{c2:>6}/{len(rows)}{br.mean():>9.4f}")
        out["summary"][name] = {
            "delta_full_mean": float(dF.mean()), "delta_full_sd": float(dF.std()),
            "delta_full_min": float(dF.min()), "delta_full_max": float(dF.max()),
            "n_positive_full": int((dF > 0).sum()), "n_clearing_full": cF,
            "delta_2min_mean": float(d2.mean()), "delta_2min_sd": float(d2.std()),
            "n_positive_2min": int((d2 > 0).sum()), "n_clearing_2min": c2,
            "mean_brier": float(br.mean()), "n": len(rows)}

    print("\n" + "=" * 104)
    robust = [k for k, v in out["summary"].items()
              if v["n_clearing_full"] >= 0.9 * len(rows)
              and v["n_clearing_2min"] >= 0.9 * len(rows)]
    allpos = [k for k, v in out["summary"].items()
              if v["n_positive_full"] == len(rows)
              and v["n_positive_2min"] == len(rows)]
    if robust:
        print(f"CLEARS ci_lo>0 ON >=90% OF RESAMPLES, BOTH POPULATIONS: "
              f"{', '.join(robust)}")
    else:
        print("NO ARM CLEARS ci_lo>0 ON >=90% OF RESAMPLES")
    if allpos:
        print(f"positive on EVERY resample, both populations: {', '.join(allpos)}")
    print("=" * 104)
    out["robust_arms"], out["all_positive_arms"] = robust, allpos

    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
