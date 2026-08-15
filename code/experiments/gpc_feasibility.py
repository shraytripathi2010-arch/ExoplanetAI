"""gpc_feasibility.py -- PART 0: measure GPC cost at 33 features BEFORE
committing to a validation protocol.

WHY THIS IS NOT A FRESH QUESTION, AND WHAT IS ACTUALLY NEW
-----------------------------------------------------------
A real GPC was already built: `gp_classifier.py` / `gp_results.json`, sklearn
`GaussianProcessClassifier` (Laplace), `ConstantKernel * RBF`, test AUC 0.8673
against a 0.9032 baseline, fit time 267.9 s on 4,392 rows. It was also already
stacked (Part C: HGB+GP+CNN -> 0.9018 vs 0.9016 alone). The bootstrap CI on
GP-vs-classical was [0.017, 0.054], entirely above zero -- the tree model won by
a real margin.

Four things about that run are stale, which is why a re-test is legitimate:
  1. 24 features, not the current 33 (no crowding / variability / Gaia)
  2. a random `train_test_split`, NOT the frozen host split -- possible host
     leakage, which would have FLATTERED the GP
  3. never calibrated -- no Brier/ECE on record, and calibration is the entire
     stated purpose of this proposal
  4. single fit, never resampled, against a baseline now 0.9454 not 0.9032

COST IS THE GATING QUESTION
---------------------------
Exact GP classification is O(n^3) in training rows. Every other model family
tested in this project is effectively linear-ish at this scale, so the usual
"just run 12 bootstraps" reflex does not transfer. This script measures the
real curve at 33 features and extrapolates BEFORE anything expensive is
launched, because this project has repeatedly underestimated wall-clock on
expensive methods.

Also measured: two kernel-approximation fallbacks (Nystroem and RBFSampler
feeding a logistic regression), so that if exact GPC is impractical the
substitution is a REPORTED tradeoff with numbers, not a silent swap.
"""
import os
import sys
import json
import time
import importlib.util
import warnings
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT = os.path.join(HERE, "gpc_feasibility.json")

SEED = 42
SIZES = [400, 800, 1600, 2400]
N_BOOT_TARGET = 12


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def gpc_pipe(kernel):
    """Median impute + StandardScaler is mandatory for a GP: an RBF kernel is a
    distance in feature space, so unscaled inputs let `period` (days, ~1e2)
    dominate `odd_even_mismatch` (~1e0) entirely."""
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", GaussianProcessClassifier(kernel=kernel, random_state=SEED))])


def main():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 33, len(cols)

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    tr_mask, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)
    tr_idx = np.where(tr_mask)[0]
    n_full = len(tr_idx)
    Xte, yte = X[te], y[te]
    print(f"train {n_full}  frozen test {int(te.sum())}  features {len(cols)}")
    print(f"prior run (24 feat, random split, uncalibrated): 267.9 s @ 4,392 rows, "
          f"AUC 0.8673\n")

    rng = np.random.default_rng(SEED)
    res = {"n_train_full": int(n_full), "sizes": [], "prior": {
        "fit_time_s": 267.889, "n_train": 4392, "auc": 0.8673, "n_features": 24,
        "split": "random train_test_split (NOT frozen host split)",
        "calibrated": False}}

    # ---------- 1. exact GPC scaling curve ----------
    print("=== 1. EXACT GPC SCALING (RBF, uncalibrated, single fit) ===")
    print(f"  {'n':>6}{'fit s':>10}{'test AUC':>10}{'Brier':>9}{'ECE':>8}")
    for n in SIZES:
        sub = rng.choice(tr_idx, size=n, replace=False)
        k = ConstantKernel(1.0) * RBF(length_scale=1.0)
        t0 = time.time()
        mo = gpc_pipe(k).fit(X.iloc[sub], y[sub])
        el = time.time() - t0
        p = mo.predict_proba(Xte)[:, 1]
        a, b, e = roc_auc_score(yte, p), brier_score_loss(yte, p), ece(yte, p)
        print(f"  {n:>6}{el:>10.1f}{a:>10.4f}{b:>9.4f}{e:>8.4f}", flush=True)
        res["sizes"].append({"n": n, "fit_s": round(el, 2), "auc": float(a),
                             "brier": float(b), "ece": float(e)})

    # ---------- 2. fit the cubic and extrapolate ----------
    ns = np.array([r["n"] for r in res["sizes"]], float)
    ts = np.array([r["fit_s"] for r in res["sizes"]], float)
    # t = c * n^p ; fit in log space so the exponent is measured, not assumed
    p_exp, log_c = np.polyfit(np.log(ns), np.log(ts), 1)
    c = float(np.exp(log_c))
    def predict_t(n):
        return c * n ** p_exp
    print(f"\n=== 2. MEASURED SCALING: t = {c:.3e} * n^{p_exp:.2f} "
          f"(theory says n^3) ===")
    t_full = predict_t(n_full)
    t_cv = predict_t(n_full * 4 / 5)
    print(f"  extrapolated single fit @ {n_full} rows      {t_full/60:.1f} min")
    print(f"  one CalibratedClassifierCV(cv=5) fit        {5*t_cv/60:.1f} min "
          f"(5 sub-fits on {int(n_full*4/5)} rows each)")
    res["scaling"] = {"exponent": float(p_exp), "coef": c,
                      "pred_single_fit_s_full": float(t_full),
                      "pred_calibrated_fit_s": float(5 * t_cv)}

    # ---------- 3. project the full protocol ----------
    per_boot_gpc = 5 * t_cv
    # meta-learner needs out-of-fold GPC probabilities -> a second cv=5 pass
    per_boot_meta = 5 * t_cv
    total_h = N_BOOT_TARGET * (per_boot_gpc + per_boot_meta) / 3600
    print(f"\n=== 3. PROJECTED BUDGET for {N_BOOT_TARGET} bootstraps ===")
    print(f"  GPC arm only                    {N_BOOT_TARGET*per_boot_gpc/3600:.1f} h")
    print(f"  + out-of-fold GPC for the meta  {N_BOOT_TARGET*per_boot_meta/3600:.1f} h")
    print(f"  GPC-attributable TOTAL          {total_h:.1f} h  "
          f"(HGB/CatBoost arms are minutes by comparison)")
    res["projected_hours_12_boot"] = float(total_h)

    # ---------- 4. approximation fallbacks, measured not assumed ----------
    print(f"\n=== 4. KERNEL-APPROXIMATION FALLBACKS @ full {n_full} rows ===")
    print(f"  {'method':<28}{'fit s':>9}{'test AUC':>10}{'Brier':>9}{'ECE':>8}")
    res["approx"] = {}
    for name, mk in (
        ("Nystroem(300)+LR", lambda: Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("approx", Nystroem(kernel="rbf", n_components=300, random_state=SEED)),
            ("clf", LogisticRegression(max_iter=2000))])),
        ("Nystroem(1000)+LR", lambda: Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("approx", Nystroem(kernel="rbf", n_components=1000, random_state=SEED)),
            ("clf", LogisticRegression(max_iter=2000))])),
        ("RBFSampler(1000)+LR", lambda: Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("approx", RBFSampler(n_components=1000, random_state=SEED)),
            ("clf", LogisticRegression(max_iter=2000))]))):
        t0 = time.time()
        mo = mk().fit(X.iloc[tr_idx], y[tr_idx])
        el = time.time() - t0
        p = mo.predict_proba(Xte)[:, 1]
        a, b, e = roc_auc_score(yte, p), brier_score_loss(yte, p), ece(yte, p)
        print(f"  {name:<28}{el:>9.1f}{a:>10.4f}{b:>9.4f}{e:>8.4f}", flush=True)
        res["approx"][name] = {"fit_s": round(el, 2), "auc": float(a),
                               "brier": float(b), "ece": float(e)}

    # ---------- 5. kernel choice, at a tractable size ----------
    print(f"\n=== 5. KERNEL CHOICE (n=1600, exact GPC) ===")
    print(f"  {'kernel':<28}{'fit s':>9}{'test AUC':>10}{'Brier':>9}{'ECE':>8}")
    sub = rng.choice(tr_idx, size=1600, replace=False)
    res["kernels"] = {}
    for name, k in (("RBF", ConstantKernel(1.0) * RBF(length_scale=1.0)),
                    ("Matern nu=1.5", ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)),
                    ("Matern nu=2.5", ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5))):
        t0 = time.time()
        mo = gpc_pipe(k).fit(X.iloc[sub], y[sub])
        el = time.time() - t0
        p = mo.predict_proba(Xte)[:, 1]
        a, b, e = roc_auc_score(yte, p), brier_score_loss(yte, p), ece(yte, p)
        print(f"  {name:<28}{el:>9.1f}{a:>10.4f}{b:>9.4f}{e:>8.4f}", flush=True)
        res["kernels"][name] = {"fit_s": round(el, 2), "auc": float(a),
                                "brier": float(b), "ece": float(e)}

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
