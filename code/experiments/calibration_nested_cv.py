"""calibration_nested_cv.py -- is the CatBoost>HGB ordering a test-set artifact?

WHY

Every number in this sweep comes from one 1,098-star test set. Resampling the
TRAINING rows tests sensitivity to the training draw; it cannot test whether
the held-out stars themselves are a lucky panel for CatBoost. Nested CV is the
complement: the outer folds rotate WHICH stars are evaluated, so a ranking that
survives it is not a property of one particular held-out set.

DESIGN

  outer: 5-fold stratified over the 4,386 TRAINING rows only. The frozen test
         set is never touched -- not read, not scored. This is deliberate:
         the frozen split's whole value is that it has not been optimised
         against, and using it here would spend that.
  inner: each arm's own calibration CV, fit strictly inside the outer-train
         fold. That is what makes it nested rather than a flat CV.

Training has one row per star, so a stratified row split is a star split and
no star straddles an outer boundary.

ARMS -- the decision-relevant ones only, not a survey:
  HGB sigmoid cv=5     production, exactly as deployed
  CatBoost sigmoid cv=20   best resampled arm that keeps calibrated outputs
  CatBoost bare        is the family advantage present without any wrapper?
  HGB bare             the matched control for that question

Reported per arm: mean/sd AUC across outer folds, plus a paired bootstrap on
the POOLED out-of-fold predictions (every training star predicted exactly once,
by a model that never saw it).
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
RESULTS = os.path.join(SCRIPT_DIR, "calibration_nested_cv_results.json")

SEED = 42
N_OUTER = 5
N_BOOT = 2000
N_WORKERS = 5
N_ECE_BINS = 15
CAT_PARAMS = dict(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=9.0)

ARMS = [
    ("HGB sigmoid cv=5 (production)", "HGB", "sigmoid", 5),
    ("CatBoost sigmoid cv=20", "CatBoost", "sigmoid", 20),
    ("CatBoost bare", "CatBoost", None, None),
    ("HGB bare", "HGB", None, None),
]
BASELINE = "HGB sigmoid cv=5 (production)"


def fast_auc(y, p):
    n = p.shape[0]
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    newgrp = np.empty(n, bool)
    newgrp[0] = True
    np.not_equal(sp[1:], sp[:-1], out=newgrp[1:])
    gid = np.cumsum(newgrp) - 1
    avg = (np.bincount(gid, weights=np.arange(1, n + 1, dtype=np.float64))
           / np.bincount(gid))
    r = np.empty(n, np.float64)
    r[order] = avg[gid]
    n1 = int(y.sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * (n - n1))


def ece(y, p, bins=N_ECE_BINS):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def paired_boot(y, pa, pb, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        yi = y[i]
        s = yi.sum()
        if s == 0 or s == len(yi):
            continue
        d.append(fast_auc(yi, pb[i]) - fast_auc(yi, pa[i]))
    d = np.asarray(d)
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


_G = {}


def _init():
    m05 = _m05()
    df = pd.read_csv(TRAINING)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    tr, _ = m05.split_by_host(df)
    tr = np.asarray(tr)
    # 2-min cadence mask, carried through so every arm can also be reported on
    # the population the model is actually deployed on.
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    prod = joblib.load(PROD)
    from catboost import CatBoostClassifier
    cat = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("clf", CatBoostClassifier(
                        verbose=0, random_seed=SEED,
                        auto_class_weights="Balanced",
                        allow_writing_files=False, **CAT_PARAMS))])
    # TRAINING ROWS ONLY -- the frozen test set is never read here.
    _G.update(X=X[tr].reset_index(drop=True), y=y[tr], is2=is2[tr],
              models={"HGB": clone(getattr(prod, "estimator", prod)),
                      "CatBoost": cat})


def run_outer_fold(fold, tr_idx, va_idx):
    if not _G:
        _init()
    X, y = _G["X"], _G["y"]
    Xt, yt = X.iloc[tr_idx], y[tr_idx]
    Xv, yv = X.iloc[va_idx], y[va_idx]
    out = {"fold": fold, "n_train": len(yt), "n_val": len(yv), "arms": {}}
    for label, fam, method, k in ARMS:
        base = clone(_G["models"][fam])
        est = base if method is None else CalibratedClassifierCV(
            base, cv=k, method=method)
        t0 = time.time()
        est.fit(Xt, yt)
        p = est.predict_proba(Xv)[:, 1]
        out["arms"][label] = {"auc": float(roc_auc_score(yv, p)),
                              "brier": float(brier_score_loss(yv, p)),
                              "p": p.tolist(), "idx": va_idx.tolist(),
                              "fit_s": round(time.time() - t0, 1)}
    return out


def main():
    print("=" * 104)
    print("NESTED CV -- does the ranking survive rotating WHICH stars are held out?")
    print("=" * 104)
    print(f"  outer: {N_OUTER}-fold stratified, TRAINING ROWS ONLY "
          f"(frozen test set untouched)")
    print("  inner: each arm's own calibration CV, fit inside the outer-train fold")
    print(f"  arms: {len(ARMS)}\n")

    _init()
    X, y = _G["X"], _G["y"]
    print(f"  {len(y)} training rows, {int(y.sum())} positive\n")

    skf = StratifiedKFold(N_OUTER, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_outer_fold, i, tr, va): i
                for i, (tr, va) in enumerate(folds)}
        for n, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            print(f"  outer fold {futs[f]} done ({n}/{N_OUTER}, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
    rows.sort(key=lambda r: r["fold"])

    # pooled out-of-fold predictions: every training star scored exactly once
    pooled = {}
    for label, _, _, _ in ARMS:
        p = np.empty(len(y))
        for r in rows:
            p[np.asarray(r["arms"][label]["idx"])] = r["arms"][label]["p"]
        pooled[label] = p

    print("\n" + "=" * 104)
    print("PER-FOLD AUC")
    print("=" * 104)
    print(f"  {'arm':<34}" + "".join(f"{'f'+str(r['fold']):>9}" for r in rows)
          + f"{'mean':>9}{'sd':>8}")
    out = {"n_outer": N_OUTER, "n_train_rows": int(len(y)),
           "baseline": BASELINE, "folds": [], "summary": {}}
    for label, _, _, _ in ARMS:
        a = np.array([r["arms"][label]["auc"] for r in rows])
        print(f"  {label:<34}" + "".join(f"{v:>9.4f}" for v in a)
              + f"{a.mean():>9.4f}{a.std():>8.4f}")
        out["summary"][label] = {"fold_aucs": a.tolist(),
                                 "mean_auc": float(a.mean()),
                                 "sd_auc": float(a.std())}

    is2 = _G["is2"]
    y2 = y[is2]
    print("\n" + "=" * 104)
    print("POOLED OUT-OF-FOLD (paired bootstrap vs production)")
    print("=" * 104)
    print(f"  n = {len(y)} full, {int(is2.sum())} 2-min\n")
    print(f"  {'arm':<34}{'AUC':>9}{'Brier':>9}{'ECE':>8}"
          f"{'delta':>10}{'ci_lo':>9}{'ci_hi':>9}{'clr':>5}"
          f"{'AUC2m':>9}{'d_2min':>10}{'ci_lo2':>9}{'clr2':>6}")
    base_p = pooled[BASELINE]
    for label, _, _, _ in ARMS:
        p = pooled[label]
        d, lo, hi = paired_boot(y, base_p, p)
        d2, lo2, hi2 = paired_boot(y2, base_p[is2], p[is2])
        clears, clears2 = lo > 0, lo2 > 0
        print(f"  {label:<34}{fast_auc(y, p):>9.4f}"
              f"{brier_score_loss(y, p):>9.4f}{ece(y, p):>8.4f}"
              f"{d:>+10.4f}{lo:>+9.4f}{hi:>+9.4f}"
              f"{('yes' if clears else 'no'):>5}"
              f"{fast_auc(y2, p[is2]):>9.4f}{d2:>+10.4f}{lo2:>+9.4f}"
              f"{('yes' if clears2 else 'no'):>6}")
        out["summary"][label].update(
            pooled_auc=float(fast_auc(y, p)),
            pooled_brier=float(brier_score_loss(y, p)),
            pooled_ece=ece(y, p), delta_vs_production=d,
            ci=[lo, hi], clears=bool(clears),
            pooled_auc_2min=float(fast_auc(y2, p[is2])),
            pooled_ece_2min=ece(y2, p[is2]),
            delta_2min=d2, ci_2min=[lo2, hi2], clears_2min=bool(clears2))
    out["n_2min"] = int(is2.sum())

    out["folds"] = [{"fold": r["fold"], "n_train": r["n_train"],
                     "n_val": r["n_val"],
                     "arms": {k: {kk: vv for kk, vv in v.items()
                                  if kk not in ("p", "idx")}
                              for k, v in r["arms"].items()}} for r in rows]
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
