"""giant_star_fix.py -- three candidate fixes for the giant-star blind spot,
chosen from what the diagnosis actually says rather than from the premise.

WHAT THE DIAGNOSIS FOUND (giant_star_diagnose.py, deployed 0.9208 model)

    population          n    planets%   AUC      err@0.5   err@best-thr   ECE
    dwarfs  <1.5      833      86.3   0.9017       9.0          8.9      0.0204
    giants >=1.5      229      55.0   0.9013      19.2         16.2      0.0867
      subgiants 1.5-3 176      51.1   0.8548      23.9         19.9      0.1044
      giants   >=3     53      67.9   0.9935       3.8          3.8      0.0787

Three things follow, and they change what is worth trying:

1. **Giants are NOT ranked worse.** AUC 0.9013 vs 0.9017, a gap of -0.0004.
   The model orders giants as well as dwarfs. A feature aimed at "giants are
   hard to rank" would be aimed at a problem that is not there.
2. **The error-rate gap is mostly base rate and calibration.** Giants are 55%
   planets versus 86% for dwarfs, so a FIXED 0.5 threshold necessarily
   misclassifies more of them. Re-thresholding alone takes 19.2% -> 16.2%.
   And giant ECE is 0.0867 against 0.0204 for dwarfs -- **4.2x worse
   calibrated**, which is a real, fixable defect that AUC cannot see.
3. **The genuine ranking deficit is narrow**: subgiants 1.5-3 at AUC 0.8548
   (-0.0469 vs dwarfs), 16% of the test set. Giants proper (>=3) are at 0.9935,
   BETTER than dwarfs.

ARMS

  A. st_rad interactions -- the simplest thing, tested first per the brief even
     though st_rad is already the single most important feature in the deployed
     model (permutation importance +0.04454, rank 1/26), so a tree can already
     split on it directly. Included to measure that, not to assume it.
  B. giant upweighting -- sample_weight boosting st_rad>=1.5 rows during
     training. Distinct from the closed class-weighting experiment, which
     reweighted by LABEL; this reweights by SUBPOPULATION.
  C. stratified calibration -- one base model (unchanged ranking), but the
     Platt sigmoid fitted SEPARATELY for giants and dwarfs from cross-fitted
     out-of-fold predictions. This targets the defect the diagnosis actually
     identified. Within-group AUC is mathematically unchanged (a per-group
     monotone map); overall AUC can move because cross-group ordering changes,
     and ECE should improve materially.

Every arm reports OVERALL and GIANT-SUBPOPULATION deltas for AUC, Brier and
ECE, over 12 training bootstraps, against production refit on the same rows.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import fast_auc  # noqa: E402

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "giant_star_fix_results.json")

SEED, N_RESAMPLES, N_BOOT, N_WORKERS, N_ECE_BINS = 42, 12, 1500, 6, 15
RAD_CUT, MDE = 1.5, 0.0097
GIANT_WEIGHT = 3.0


def ece(y, p, bins=N_ECE_BINS):
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
        s = yi.sum()
        if s == 0 or s == len(yi):
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
    rad = pd.to_numeric(df["st_rad"], errors="coerce").to_numpy()
    giant = np.isfinite(rad) & (rad >= RAD_CUT)

    Xi = X.copy()                       # arm A feature set
    Xi["giant_flag"] = giant.astype(float)
    Xi["depth_x_rad"] = pd.to_numeric(X["depth"], errors="coerce") * rad
    Xi["dur_over_rad"] = pd.to_numeric(X["duration"], errors="coerce") / np.where(
        np.isfinite(rad) & (rad > 0), rad, np.nan)

    tr, _ = m05.split_by_host(df)
    frozen = m05.frozen_test_mask(df)
    prod = joblib.load(PROD)
    _G.update(X=X, Xi=Xi, y=y, giant=giant, tr=np.asarray(tr), te=frozen,
              cols=list(m05.FEATURE_COLUMNS),
              cols_i=list(m05.FEATURE_COLUMNS) + ["giant_flag", "depth_x_rad",
                                                  "dur_over_rad"],
              hgb=clone(getattr(prod, "estimator", prod)))


def _prod_fit(Xb, yb, cols, w=None):
    est = CalibratedClassifierCV(clone(_G["hgb"]), cv=5, method="sigmoid")
    if w is None:
        return est.fit(Xb[cols], yb)
    return est.fit(Xb[cols], yb, **{"sample_weight": w})


def _stratified_calibration(Xb, yb, gb, cols, Xte, gte):
    """Arm C: one base model, per-group Platt sigmoid from OOF predictions."""
    base = clone(_G["hgb"])
    oof = np.zeros(len(yb))
    for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xb, yb):
        mm = clone(base).fit(Xb.iloc[tr_i][cols], yb[tr_i])
        oof[va_i] = mm.predict_proba(Xb.iloc[va_i][cols])[:, 1]
    full = clone(base).fit(Xb[cols], yb)
    raw_te = full.predict_proba(Xte[cols])[:, 1]

    out = np.empty(len(raw_te))
    eps = 1e-6
    for flag in (False, True):
        m_tr, m_te = (gb == flag), (gte == flag)
        if m_te.sum() == 0:
            continue
        if m_tr.sum() < 30 or len(np.unique(yb[m_tr])) < 2:
            out[m_te] = raw_te[m_te]      # too few to calibrate; leave as-is
            continue
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        z = np.log(np.clip(oof[m_tr], eps, 1 - eps)
                   / (1 - np.clip(oof[m_tr], eps, 1 - eps))).reshape(-1, 1)
        lr.fit(z, yb[m_tr])
        zt = np.log(np.clip(raw_te[m_te], eps, 1 - eps)
                    / (1 - np.clip(raw_te[m_te], eps, 1 - eps))).reshape(-1, 1)
        out[m_te] = lr.predict_proba(zt)[:, 1]
    return out


def run_resample(rep):
    if not _G:
        _init()
    X, Xi, y, giant, tr, te = (_G["X"], _G["Xi"], _G["y"], _G["giant"],
                               _G["tr"], _G["te"])
    cols, cols_i = _G["cols"], _G["cols_i"]
    Xtr, ytr, gtr = X[tr], y[tr], giant[tr]
    Xitr = Xi[tr]
    rng = np.random.RandomState(1000 + rep)
    idx = rng.randint(0, len(ytr), len(ytr))
    Xb, yb, gb = Xtr.iloc[idx], ytr[idx], gtr[idx]
    Xib = Xitr.iloc[idx]
    yte, gte = y[te], giant[te]

    base_p = _prod_fit(Xb, yb, cols).predict_proba(X.loc[te, cols])[:, 1]
    arms = {
        "A: +st_rad interactions": _prod_fit(Xib, yb, cols_i).predict_proba(
            Xi.loc[te, cols_i])[:, 1],
        "B: giant upweight x3": _prod_fit(
            Xb, yb, cols, w=np.where(gb, GIANT_WEIGHT, 1.0)).predict_proba(
            X.loc[te, cols])[:, 1],
        "C: stratified calibration": _stratified_calibration(
            Xb, yb, gb, cols, X.loc[te], gte),
    }

    row = {"rep": rep,
           "base_auc": float(roc_auc_score(yte, base_p)),
           "base_auc_giant": float(fast_auc(yte[gte], base_p[gte])),
           "base_brier": float(brier_score_loss(yte, base_p)),
           "base_ece": ece(yte, base_p),
           "base_ece_giant": ece(yte[gte], base_p[gte]), "arms": {}}
    for name, p in arms.items():
        d, lo, hi = paired_boot(yte, base_p, p)
        dg, log, hig = paired_boot(yte[gte], base_p[gte], p[gte])
        row["arms"][name] = {
            "auc": float(roc_auc_score(yte, p)), "delta": d, "clears": bool(lo > 0),
            "auc_giant": float(fast_auc(yte[gte], p[gte])),
            "delta_giant": dg, "clears_giant": bool(log > 0),
            "brier": float(brier_score_loss(yte, p)), "ece": ece(yte, p),
            "ece_giant": ece(yte[gte], p[gte])}
    return row


def main():
    print("=" * 108)
    print("GIANT-STAR TARGETED FIX -- three arms vs the deployed 0.9208 recipe")
    print("=" * 108)
    _init()
    print(f"  giants (st_rad >= {RAD_CUT}) in test: {int(_G['giant'][_G['te']].sum())}"
          f" / {int(_G['te'].sum())}\n")
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_resample, r): r for r in range(N_RESAMPLES)}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            json.dump(rows, open(RESULTS + ".partial", "w"), default=float)
            print(f"  resample {futs[f]} done ({i}/{N_RESAMPLES}, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
    rows.sort(key=lambda r: r["rep"])

    ba = np.array([r["base_auc"] for r in rows])
    bg = np.array([r["base_auc_giant"] for r in rows])
    be = np.array([r["base_ece"] for r in rows])
    beg = np.array([r["base_ece_giant"] for r in rows])
    bb = np.array([r["base_brier"] for r in rows])
    print("\n" + "=" * 108)
    print(f"  baseline: AUC {ba.mean():.4f}  giantAUC {bg.mean():.4f}  "
          f"Brier {bb.mean():.4f}  ECE {be.mean():.4f}  giantECE {beg.mean():.4f}\n")
    print(f"  {'arm':<28}{'d_overall':>11}{'sd':>8}{'pos':>7}{'clr':>6}{'>=MDE':>7}"
          f"{'d_GIANT':>10}{'pos':>7}{'clr':>6}{'Brier':>9}{'ECE':>8}{'gECE':>8}")
    out = {"n_resamples": len(rows), "baseline_auc": float(ba.mean()),
           "baseline_auc_giant": float(bg.mean()),
           "baseline_ece": float(be.mean()),
           "baseline_ece_giant": float(beg.mean()),
           "baseline_brier": float(bb.mean()), "rows": rows, "summary": {}}
    for name in rows[0]["arms"]:
        d = np.array([r["arms"][name]["delta"] for r in rows])
        dg = np.array([r["arms"][name]["delta_giant"] for r in rows])
        c = sum(r["arms"][name]["clears"] for r in rows)
        cg = sum(r["arms"][name]["clears_giant"] for r in rows)
        br = np.mean([r["arms"][name]["brier"] for r in rows])
        ec = np.mean([r["arms"][name]["ece"] for r in rows])
        eg = np.mean([r["arms"][name]["ece_giant"] for r in rows])
        print(f"  {name:<28}{d.mean():>+11.4f}{d.std():>8.4f}"
              f"{int((d>0).sum()):>4}/{len(rows)}{c:>3}/{len(rows)}"
              f"{int((d>=MDE).sum()):>4}/{len(rows)}"
              f"{dg.mean():>+10.4f}{int((dg>0).sum()):>4}/{len(rows)}"
              f"{cg:>3}/{len(rows)}{br:>9.4f}{ec:>8.4f}{eg:>8.4f}")
        out["summary"][name] = {
            "delta_mean": float(d.mean()), "delta_sd": float(d.std()),
            "delta_min": float(d.min()), "delta_max": float(d.max()),
            "n_positive": int((d > 0).sum()), "n_clearing": c,
            "n_at_or_above_mde": int((d >= MDE).sum()),
            "delta_giant_mean": float(dg.mean()), "delta_giant_sd": float(dg.std()),
            "n_positive_giant": int((dg > 0).sum()), "n_clearing_giant": cg,
            "mean_brier": float(br), "mean_ece": float(ec),
            "mean_ece_giant": float(eg),
            "ece_giant_vs_baseline": float(eg - beg.mean())}
    print("=" * 108)
    json.dump(out, open(RESULTS, "w"), indent=2, default=float)
    if os.path.exists(RESULTS + ".partial"):
        os.remove(RESULTS + ".partial")
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
