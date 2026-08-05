"""promote_crowding_retrain.py -- build and validate the production successor.

LIKE-FOR-LIKE BY CONSTRUCTION. The challenger is `clone()`d from the deployed
estimator, so hyperparameters and the CalibratedClassifierCV configuration come
from production itself rather than being retyped. The ONLY difference is that
`FEATURE_COLUMNS` now contains two more entries. Same frozen split, same rows,
same recipe.

Runs the full suite at real scale on the complete backfilled dataset:
  * headline test ROC-AUC, old vs new
  * 12 training-data bootstrap resamples (the standing rule)
  * nested CV, 5 outer folds on training rows only
  * calibration: Brier and ECE
  * 2-min subset alongside the full clean test set

Writes the new artifact to a STAGING path. It does not touch
models/best_model.joblib -- deployment is a separate, explicit step.
"""
import os
import sys
import json
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
PROD = os.path.join(ROOT, "models", "best_model.joblib")
STAGED = os.path.join(ROOT, "models", "best_model_crowding_staged.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "promote_crowding_results.json")

SEED, N_RESAMPLES, N_BOOT, N_WORKERS, N_ECE_BINS, N_OUTER = 42, 12, 2000, 6, 15, 5
NEW = ["crowd_flux_ratio_max", "crowd_nearest_arcsec"]


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
    frozen = m05.frozen_test_mask(df)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    prod = joblib.load(PROD)
    _G.update(X=X, y=y, tr=np.asarray(tr), te=frozen, te2=frozen & is2,
              cols=list(m05.FEATURE_COLUMNS),
              old=[c for c in m05.FEATURE_COLUMNS if c not in NEW],
              hgb=clone(getattr(prod, "estimator", prod)))


def _fit(cols, Xb, yb):
    est = CalibratedClassifierCV(clone(_G["hgb"]), cv=5, method="sigmoid")
    return est.fit(Xb[cols], yb)


def run_resample(rep):
    if not _G:
        _init()
    X, y, tr, te, te2 = _G["X"], _G["y"], _G["tr"], _G["te"], _G["te2"]
    Xtr, ytr = X[tr], y[tr]
    rng = np.random.RandomState(1000 + rep)
    idx = rng.randint(0, len(ytr), len(ytr))
    Xb, yb = Xtr.iloc[idx], ytr[idx]
    o = _fit(_G["old"], Xb, yb)
    n = _fit(_G["cols"], Xb, yb)
    po = o.predict_proba(X.loc[te, _G["old"]])[:, 1]
    pn = n.predict_proba(X.loc[te, _G["cols"]])[:, 1]
    po2 = o.predict_proba(X.loc[te2, _G["old"]])[:, 1]
    pn2 = n.predict_proba(X.loc[te2, _G["cols"]])[:, 1]
    d, lo, hi = paired_boot(y[te], po, pn)
    d2, lo2, hi2 = paired_boot(y[te2], po2, pn2)
    return {"rep": rep, "old_auc": float(roc_auc_score(y[te], po)),
            "new_auc": float(roc_auc_score(y[te], pn)),
            "old_brier": float(brier_score_loss(y[te], po)),
            "new_brier": float(brier_score_loss(y[te], pn)),
            "old_ece": ece(y[te], po), "new_ece": ece(y[te], pn),
            "delta": d, "clears": bool(lo > 0),
            "delta_2min": d2, "clears_2min": bool(lo2 > 0)}


def outer_fold(fold, tr_idx, va_idx):
    if not _G:
        _init()
    X, y = _G["X"][_G["tr"]].reset_index(drop=True), _G["y"][_G["tr"]]
    Xt, yt, Xv, yv = X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], y[va_idx]
    o = _fit(_G["old"], Xt, yt)
    n = _fit(_G["cols"], Xt, yt)
    return {"fold": fold,
            "old": float(roc_auc_score(yv, o.predict_proba(Xv[_G["old"]])[:, 1])),
            "new": float(roc_auc_score(yv, n.predict_proba(Xv[_G["cols"]])[:, 1]))}


def main():
    print("=" * 100)
    print("PRODUCTION SUCCESSOR -- retrain with crowding, full suite at real scale")
    print("=" * 100)
    _init()
    X, y, tr, te, te2 = _G["X"], _G["y"], _G["tr"], _G["te"], _G["te2"]
    print(f"  rows {len(y)}   train {int(tr.sum())}   frozen test {int(te.sum())}"
          f"   2-min {int(te2.sum())}")
    print(f"  features: {len(_G['old'])} -> {len(_G['cols'])}  (added {NEW})")
    for c in NEW:
        print(f"    {c:<26} non-null {X[c].notna().mean()*100:.2f}%")

    # ---- headline single fit on the real full dataset ---------------------
    old_m = _fit(_G["old"], X[tr], y[tr])
    new_m = _fit(_G["cols"], X[tr], y[tr])
    po = old_m.predict_proba(X.loc[te, _G["old"]])[:, 1]
    pn = new_m.predict_proba(X.loc[te, _G["cols"]])[:, 1]
    auc_o, auc_n = roc_auc_score(y[te], po), roc_auc_score(y[te], pn)
    d, lo, hi = paired_boot(y[te], po, pn)
    print("\n" + "-" * 100)
    print("HEADLINE (single fit, full real dataset)")
    print(f"  old (24 features): AUC {auc_o:.4f}  Brier {brier_score_loss(y[te], po):.4f}"
          f"  ECE {ece(y[te], po):.4f}")
    print(f"  new (26 features): AUC {auc_n:.4f}  Brier {brier_score_loss(y[te], pn):.4f}"
          f"  ECE {ece(y[te], pn):.4f}")
    print(f"  delta {d:+.4f}  ci [{lo:+.4f}, {hi:+.4f}]  clears={lo > 0}")

    # ---- 12 bootstraps ----------------------------------------------------
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_resample, r): r for r in range(N_RESAMPLES)}
        for f in as_completed(futs):
            rows.append(f.result())
    rows.sort(key=lambda r: r["rep"])
    dd = np.array([r["delta"] for r in rows])
    print("\n" + "-" * 100)
    print(f"RESAMPLED ({N_RESAMPLES} training bootstraps)")
    print(f"  old AUC {np.mean([r['old_auc'] for r in rows]):.4f}   "
          f"new AUC {np.mean([r['new_auc'] for r in rows]):.4f}")
    print(f"  delta {dd.mean():+.4f} (sd {dd.std():.4f}, range {dd.min():+.4f}..{dd.max():+.4f})"
          f"  positive {int((dd>0).sum())}/{len(rows)}  clears "
          f"{sum(r['clears'] for r in rows)}/{len(rows)}")
    print(f"  2-min delta {np.mean([r['delta_2min'] for r in rows]):+.4f}   "
          f"clears {sum(r['clears_2min'] for r in rows)}/{len(rows)}")
    print(f"  Brier {np.mean([r['old_brier'] for r in rows]):.4f} -> "
          f"{np.mean([r['new_brier'] for r in rows]):.4f}   "
          f"ECE {np.mean([r['old_ece'] for r in rows]):.4f} -> "
          f"{np.mean([r['new_ece'] for r in rows]):.4f}")

    # ---- nested CV --------------------------------------------------------
    Xtr_only = X[tr].reset_index(drop=True)
    skf = StratifiedKFold(N_OUTER, shuffle=True, random_state=SEED)
    folds = list(skf.split(Xtr_only, y[tr]))
    ncv = []
    with ProcessPoolExecutor(max_workers=N_OUTER) as ex:
        futs = {ex.submit(outer_fold, i, a, b): i for i, (a, b) in enumerate(folds)}
        for f in as_completed(futs):
            ncv.append(f.result())
    ncv.sort(key=lambda r: r["fold"])
    no = np.array([r["old"] for r in ncv])
    nn = np.array([r["new"] for r in ncv])
    print("\n" + "-" * 100)
    print(f"NESTED CV ({N_OUTER} outer folds, training rows only)")
    print(f"  old {no.mean():.4f} (sd {no.std():.4f})   new {nn.mean():.4f} (sd {nn.std():.4f})"
          f"   delta {nn.mean()-no.mean():+.4f}   new wins {int((nn>no).sum())}/{N_OUTER}")

    joblib.dump(new_m, STAGED)
    import hashlib
    md5 = hashlib.md5(open(STAGED, "rb").read()).hexdigest()
    print("\n" + "=" * 100)
    print(f"STAGED artifact: {os.path.relpath(STAGED, ROOT)}")
    print(f"  md5 {md5}")
    print("  models/best_model.joblib NOT modified -- deployment is a separate step")
    print("=" * 100)

    json.dump({"n_rows": int(len(y)), "n_train": int(tr.sum()), "n_test": int(te.sum()),
               "features_old": len(_G["old"]), "features_new": len(_G["cols"]),
               "headline_old_auc": float(auc_o), "headline_new_auc": float(auc_n),
               "headline_delta": d, "headline_ci": [lo, hi],
               "resampled_delta_mean": float(dd.mean()),
               "resampled_delta_sd": float(dd.std()),
               "resampled_positive": int((dd > 0).sum()),
               "resampled_clears": sum(r["clears"] for r in rows),
               "resampled_old_auc": float(np.mean([r["old_auc"] for r in rows])),
               "resampled_new_auc": float(np.mean([r["new_auc"] for r in rows])),
               "brier_old": float(np.mean([r["old_brier"] for r in rows])),
               "brier_new": float(np.mean([r["new_brier"] for r in rows])),
               "ece_old": float(np.mean([r["old_ece"] for r in rows])),
               "ece_new": float(np.mean([r["new_ece"] for r in rows])),
               "nested_cv_old": float(no.mean()), "nested_cv_new": float(nn.mean()),
               "staged_md5": md5, "resamples": rows, "nested_cv_folds": ncv},
              open(RESULTS, "w"), indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
