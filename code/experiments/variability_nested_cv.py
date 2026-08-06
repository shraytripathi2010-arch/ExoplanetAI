"""variability_nested_cv.py -- the missing gate before any promotion decision.

WHAT THIS TESTS, AND WHY IT IS NOT REDUNDANT WITH WHAT WE ALREADY HAVE

The resampled result (+0.0101 headline, +0.0098 sky-controlled, 12/12 clearing)
comes from ONE 1,098-star frozen test set, with the TRAINING rows bootstrapped.
That tests sensitivity to the training draw. It cannot test whether those
particular held-out stars are a lucky panel for the new features.

Nested CV is the complement: the outer folds rotate WHICH stars are evaluated,
so an effect that survives is not a property of one held-out set. The two
answer different questions and both are reported; neither replaces the other.

DESIGN -- identical to `calibration_nested_cv.py`, this project's established
nested-CV methodology, so the numbers are comparable to prior runs:

  outer: 5-fold stratified over the TRAINING rows only. The frozen test set is
         never read here. That is deliberate -- its value is that it has not
         been optimised against, and spending it here would destroy that.
  inner: production's own CalibratedClassifierCV(cv=5, sigmoid), fit strictly
         inside each outer-train fold. That is what makes it nested.

Training has one row per star, so a stratified row split is a star split and no
star straddles an outer boundary.

ARMS
  26 production          exactly as deployed
  31 + variability       the promotion candidate
  30 + variability-4     drops var_oot_rms (|r| 0.967 vs chi2red_min), so the
                         result is not resting on a near-duplicate column

Reported per arm: mean/sd AUC across outer folds, and a paired bootstrap on the
POOLED out-of-fold predictions -- every training star predicted exactly once by
a model that never saw it. Also reported on the 2-min cadence subset, the
population the model is actually deployed on.
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
from sklearn.metrics import roc_auc_score, brier_score_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import fast_auc  # noqa: E402

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
VFEAT = os.path.join(SCRIPT_DIR, "stellar_variability_features.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "variability_nested_cv_results.json")

SEED, N_OUTER, N_BOOT, N_WORKERS, N_ECE_BINS = 42, 5, 2000, 5, 15
VAR5 = ["var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]
VAR4 = ["var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]
BASELINE = "26 production"


def ece(y, p, bins=N_ECE_BINS):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = np.linspace(0.0, 1.0, bins + 1)
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


_G = {}


def _init():
    m05 = _m05()
    df = pd.read_csv(TRAINING)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    y = np.asarray(y)

    v = pd.read_csv(VFEAT)
    merged = df[["host"]].merge(v[["host"] + VAR5], on="host", how="left")
    for c in VAR5:
        X[c] = pd.to_numeric(merged[c], errors="coerce").to_numpy()

    tr, _ = m05.split_by_host(df)
    tr = np.asarray(tr)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    prod = joblib.load(PROD)
    base = list(m05.FEATURE_COLUMNS)
    assert len(base) == 26, f"expected 26 production features, got {len(base)}"
    _G.update(X=X[tr].reset_index(drop=True), y=y[tr], is2=is2[tr],
              base=base, hgb=clone(getattr(prod, "estimator", prod)),
              arms=[("26 production", base),
                    ("31 + variability", base + VAR5),
                    ("30 + variability-4", base + VAR4)])


def run_outer_fold(fold, tr_idx, va_idx):
    if not _G:
        _init()
    X, y = _G["X"], _G["y"]
    Xt, yt = X.iloc[tr_idx], y[tr_idx]
    Xv, yv = X.iloc[va_idx], y[va_idx]
    out = {"fold": fold, "n_train": len(yt), "n_val": len(yv), "arms": {}}
    for label, cols in _G["arms"]:
        exp = {"26 production": 26, "31 + variability": 31,
               "30 + variability-4": 30}[label]
        assert len(cols) == exp, f"{label}: expected {exp} cols, got {len(cols)}"
        est = CalibratedClassifierCV(clone(_G["hgb"]), cv=5, method="sigmoid")
        t0 = time.time()
        est.fit(Xt[cols], yt)
        p = est.predict_proba(Xv[cols])[:, 1]
        out["arms"][label] = {"auc": float(roc_auc_score(yv, p)),
                              "brier": float(brier_score_loss(yv, p)),
                              "p": p.tolist(), "idx": va_idx.tolist(),
                              "fit_s": round(time.time() - t0, 1)}
    return out


def main():
    print("=" * 104)
    print("NESTED CV -- does the variability gain survive rotating WHICH stars are held out?")
    print("=" * 104)
    _init()
    X, y, is2 = _G["X"], _G["y"], _G["is2"]
    print(f"  outer: {N_OUTER}-fold stratified, TRAINING ROWS ONLY (frozen test set untouched)")
    print("  inner: production CalibratedClassifierCV(cv=5, sigmoid) inside each outer-train fold")
    print(f"  {len(y)} training rows, {int(y.sum())} positive, {int(is2.sum())} 2-min\n")

    skf = StratifiedKFold(N_OUTER, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_outer_fold, i, tr, va): i
                for i, (tr, va) in enumerate(folds)}
        for n, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            print(f"  fold {futs[f]} done ({n}/{len(folds)}, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
    rows.sort(key=lambda r: r["fold"])

    # pool out-of-fold predictions: every training star predicted exactly once
    names = [a[0] for a in _G["arms"]]
    pooled = {n: np.full(len(y), np.nan) for n in names}
    for r in rows:
        for n in names:
            pooled[n][np.array(r["arms"][n]["idx"])] = np.array(r["arms"][n]["p"])
    for n in names:
        assert np.isfinite(pooled[n]).all(), f"{n}: some rows never predicted"

    out = {"n_outer": N_OUTER, "n_train_rows": int(len(y)), "folds": [
        {k: v for k, v in r.items() if k != "arms"} |
        {"arms": {n: {kk: vv for kk, vv in r["arms"][n].items()
                      if kk not in ("p", "idx")} for n in names}}
        for r in rows], "summary": {}}

    print("\n" + "=" * 104)
    print(f"  {'arm':<22}{'foldAUC mean':>13}{'sd':>8}{'pooledAUC':>11}"
          f"{'d vs prod':>11}{'95% CI':>20}{'clears':>8}{'Brier':>9}{'ECE':>8}{'d 2min':>9}")
    for n in names:
        fa = np.array([r["arms"][n]["auc"] for r in rows])
        pa = pooled[n]
        pooled_auc = roc_auc_score(y, pa)
        if n == BASELINE:
            d = lo = hi = 0.0
            d2 = 0.0
            clears = False
        else:
            d, lo, hi = paired_boot(y, pooled[BASELINE], pa)
            d2, lo2, _ = paired_boot(y[is2], pooled[BASELINE][is2], pa[is2])
            clears = lo > 0
        br = brier_score_loss(y, pa)
        ec = ece(y, pa)
        print(f"  {n:<22}{fa.mean():>13.4f}{fa.std():>8.4f}{pooled_auc:>11.4f}"
              f"{d:>+11.4f}{f'[{lo:+.4f},{hi:+.4f}]':>20}{str(clears):>8}"
              f"{br:>9.4f}{ec:>8.4f}{d2:>+9.4f}")
        out["summary"][n] = {
            "fold_auc_mean": float(fa.mean()), "fold_auc_sd": float(fa.std()),
            "fold_aucs": [float(x) for x in fa],
            "pooled_oof_auc": float(pooled_auc), "delta_vs_prod": float(d),
            "ci_lo": float(lo), "ci_hi": float(hi), "clears": bool(clears),
            "pooled_brier": float(br), "pooled_ece": float(ec),
            "delta_2min": float(d2)}
    print("=" * 104)
    json.dump(out, open(RESULTS, "w"), indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
