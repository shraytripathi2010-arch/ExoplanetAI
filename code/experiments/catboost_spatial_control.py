"""catboost_spatial_control.py -- is CatBoost's +0.0080 partly positional?

WHY THIS ONE NEEDS A RUN AND THE OTHERS DO NOT

The cheap diagnostic showed no production feature is strongly correlated with
sky position (max |r| = 0.192 vs |galactic b|, 0.213 vs |ecliptic latitude|),
and |ecliptic latitude| has no standalone signal at all (AUC 0.4920). That
settles the feature-level experiments by inspection.

It does not settle a MODEL-family change. CatBoost could be extracting more
from the same weakly-position-correlated features than HGB does, in which case
part of its +0.0080 would be the spatial artifact rather than better modelling.
Correlations cannot answer that; only refitting can.

THE TEST. Two paired comparisons over the same 12 bootstraps:

    CatBoost - HGB, neither seeing position
    CatBoost - HGB, BOTH seeing |galactic b|

If the gap is unchanged, CatBoost's advantage is not positional. If it shrinks,
part of the previously reported +0.0080 was the confound.

Like-for-like configuration (HGB+sigmoid+cv=5 vs CatBoost+sigmoid+cv=5) so the
wrapper is held constant and only the model family varies -- and because cv=5
is ~4x cheaper than the cv=20 arm while measuring the same family question.
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
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import fast_auc  # noqa: E402

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "catboost_spatial_control_results.json")

SEED, N_RESAMPLES, N_BOOT, N_WORKERS = 42, 12, 1500, 6
MDE = 0.0097
CAT_PARAMS = dict(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=9.0)


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
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from catboost import CatBoostClassifier
    m05 = _m05()
    df = pd.read_csv(TRAINING)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy()
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy()
    gb = np.full(len(df), np.nan)
    ok = np.isfinite(ra) & np.isfinite(dec)
    gb[ok] = np.abs(SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg).galactic.b.deg)
    X["sky_abs_galactic_b"] = gb

    tr, _ = m05.split_by_host(df)
    frozen = m05.frozen_test_mask(df)
    prod = joblib.load(PROD)
    cat = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("clf", CatBoostClassifier(
                        verbose=0, random_seed=SEED,
                        auto_class_weights="Balanced",
                        allow_writing_files=False, **CAT_PARAMS))])
    _G.update(X=X, y=y, tr=np.asarray(tr), te=frozen,
              base_cols=list(m05.FEATURE_COLUMNS),
              hgb=clone(getattr(prod, "estimator", prod)), cat=cat)


def _fit(model_key, cols, Xb, yb, X, te):
    est = CalibratedClassifierCV(clone(_G[model_key]), cv=5, method="sigmoid")
    est.fit(Xb[cols], yb)
    return est.predict_proba(X.loc[te, cols])[:, 1]


def run_resample(rep):
    if not _G:
        _init()
    X, y, tr, te = _G["X"], _G["y"], _G["tr"], _G["te"]
    base = _G["base_cols"]
    withb = base + ["sky_abs_galactic_b"]
    Xtr, ytr = X[tr], y[tr]
    rng = np.random.RandomState(1000 + rep)
    idx = rng.randint(0, len(ytr), len(ytr))
    Xb, yb = Xtr.iloc[idx], ytr[idx]

    h0 = _fit("hgb", base, Xb, yb, X, te)
    c0 = _fit("cat", base, Xb, yb, X, te)
    h1 = _fit("hgb", withb, Xb, yb, X, te)
    c1 = _fit("cat", withb, Xb, yb, X, te)

    d0, lo0, _ = paired_boot(y[te], h0, c0)
    d1, lo1, _ = paired_boot(y[te], h1, c1)
    return {"rep": rep,
            "cat_minus_hgb_no_position": d0, "clears_no_position": bool(lo0 > 0),
            "cat_minus_hgb_with_position": d1, "clears_with_position": bool(lo1 > 0),
            "hgb_auc_base": float(roc_auc_score(y[te], h0)),
            "cat_auc_base": float(roc_auc_score(y[te], c0))}


def main():
    print("=" * 96)
    print("SPATIAL CONTROL ON CATBOOST -- is the family advantage positional?")
    print("=" * 96)
    print("  like-for-like: HGB+sigmoid+cv=5 vs CatBoost+sigmoid+cv=5")
    print(f"  {N_RESAMPLES} training bootstraps; both models refit on identical rows\n")
    _init()
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_resample, r): r for r in range(N_RESAMPLES)}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            json.dump(rows, open(RESULTS + ".partial", "w"), default=float)
            print(f"  resample {futs[f]} done ({i}/{N_RESAMPLES}, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
    rows.sort(key=lambda r: r["rep"])

    d0 = np.array([r["cat_minus_hgb_no_position"] for r in rows])
    d1 = np.array([r["cat_minus_hgb_with_position"] for r in rows])
    c0 = sum(r["clears_no_position"] for r in rows)
    c1 = sum(r["clears_with_position"] for r in rows)
    print("\n" + "=" * 96)
    print(f"  {'comparison':<42}{'mean':>10}{'sd':>8}{'pos':>8}{'clears':>9}")
    print(f"  {'CatBoost - HGB, no position':<42}{d0.mean():>+10.4f}{d0.std():>8.4f}"
          f"{int((d0>0).sum()):>5}/{len(rows)}{c0:>6}/{len(rows)}")
    print(f"  {'CatBoost - HGB, BOTH given |b|':<42}{d1.mean():>+10.4f}{d1.std():>8.4f}"
          f"{int((d1>0).sum()):>5}/{len(rows)}{c1:>6}/{len(rows)}")
    shift = d1.mean() - d0.mean()
    print(f"\n  change when position is available to both: {shift:+.4f}")
    if abs(shift) < 0.002:
        v = "CatBoost's advantage is NOT positional -- unchanged when both see |b|"
    elif shift < 0:
        v = f"CatBoost's advantage SHRINKS by {abs(shift):.4f} once position is modelled"
    else:
        v = f"CatBoost's advantage GROWS by {shift:.4f} once position is modelled"
    print(f"  reading: {v}")
    print("=" * 96)
    json.dump({"n_resamples": len(rows),
               "cat_minus_hgb_no_position_mean": float(d0.mean()),
               "cat_minus_hgb_with_position_mean": float(d1.mean()),
               "shift": float(shift), "n_clearing_no_position": c0,
               "n_clearing_with_position": c1, "verdict": v, "rows": rows},
              open(RESULTS, "w"), indent=2, default=float)
    if os.path.exists(RESULTS + ".partial"):
        os.remove(RESULTS + ".partial")
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
