"""catboost_seed_ensemble.py -- does averaging CatBoost against ITSELF across
seeds turn the unstable advantage into a provable one?

BACKGROUND. On the 24-feature model a single CatBoost fit gave +0.0085 and
CLEARED -- the first arm in 23 experiments to do so on both populations -- then
collapsed to +0.0013 (sd 0.0024) with 0/10 seeds clearing. The instability was
seed variance. Averaging many seeds is the textbook fix for exactly that, and it
has never been tried here.

PART 1 (already run, `catboost_singlefit_31.py`): on the CURRENT 31 features a
single CatBoost fit gives 0.9342 vs HGB 0.9300, delta +0.0042, CI
[-0.0013, +0.0100] -- does NOT clear. The edge has roughly halved from the
24-feature era, consistent with crowding and variability having absorbed part of
what CatBoost was exploiting.

TWO LEVELS, kept strictly separate because they answer different questions:

  LEVEL 1 -- SEED STABILITY. Fit N members on the SAME training split, varying
  only `random_seed`, average their probabilities, and see how AUC and its
  spread behave as N grows (10, 20, 40). This shows whether averaging removes
  the seed variance that killed the original result. It is NECESSARY but NOT
  SUFFICIENT: it holds the training draw fixed, so it cannot say whether the
  gain survives a different sample of stars.

  LEVEL 2 -- DATA-DRAW ROBUSTNESS. Re-run the whole ensemble inside 12
  TRAINING-DATA bootstraps against the production recipe refit on identical
  rows. This is the project's actual promotability bar (`ci_lo > 0`).

MEMBER RECIPE. Each member is CalibratedClassifierCV(CatBoost, cv=3, sigmoid).
Calibrated members are used so Brier/ECE are directly comparable to production's
0.0832 / 0.0365 rather than needing a caveat; cv=3 rather than production's cv=5
is a cost concession with precedent in this repo -- `09_build_bootstrap_ensemble`
documents the same trade for the same reason. The baseline arm is always the
FULL production recipe, cv=5, unmodified.
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import fast_auc  # noqa: E402

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
OUT = os.path.join(SCRIPT_DIR, "catboost_seed_ensemble_results.json")

SEED, N_BOOT, N_WORKERS = 42, 2000, 6
N_LEVELS = [10, 20, 40]
N_LEVEL2 = 20          # ensemble size carried into the data-draw test
N_RESAMPLES = 12
MDE = 0.0097
CAT_PARAMS = dict(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=9.0)


def ece(y, p, bins=15):
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


def cat_member(seed):
    from catboost import CatBoostClassifier
    return CalibratedClassifierCV(
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", CatBoostClassifier(
                      verbose=0, random_seed=seed, auto_class_weights="Balanced",
                      allow_writing_files=False, thread_count=1, **CAT_PARAMS))]),
        cv=3, method="sigmoid")


_G = {}


def _init():
    m05 = _m05()
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 31
    df = pd.read_csv(TRAINING)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    tr, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    prod = joblib.load(PROD)
    _G.update(X=X, y=y, tr=np.asarray(tr), te=te, te2=te & is2,
              hgb=clone(getattr(prod, "estimator", prod)))


def fit_member_level1(seed):
    """One member on the FULL training split (Level 1)."""
    if not _G:
        _init()
    X, y, tr, te, te2 = _G["X"], _G["y"], _G["tr"], _G["te"], _G["te2"]
    m = cat_member(seed)
    m.fit(X[tr], y[tr])
    return (seed, m.predict_proba(X.loc[te])[:, 1], m.predict_proba(X.loc[te2])[:, 1])


def run_resample(rep):
    """One training-data bootstrap: production baseline vs N-member ensemble."""
    if not _G:
        _init()
    X, y, tr, te, te2 = _G["X"], _G["y"], _G["tr"], _G["te"], _G["te2"]
    tr_idx = np.where(tr)[0]
    rng = np.random.RandomState(1000 + rep)
    boot = tr_idx[rng.randint(0, len(tr_idx), len(tr_idx))]
    Xb, yb = X.iloc[boot], y[boot]

    base = CalibratedClassifierCV(clone(_G["hgb"]), cv=5, method="sigmoid")
    base.fit(Xb, yb)
    bF, b2 = base.predict_proba(X.loc[te])[:, 1], base.predict_proba(X.loc[te2])[:, 1]

    accF = np.zeros(int(te.sum()))
    acc2 = np.zeros(int(te2.sum()))
    for k in range(N_LEVEL2):
        m = cat_member(10_000 * (rep + 1) + k)
        m.fit(Xb, yb)
        accF += m.predict_proba(X.loc[te])[:, 1]
        acc2 += m.predict_proba(X.loc[te2])[:, 1]
    eF, e2 = accF / N_LEVEL2, acc2 / N_LEVEL2

    d, lo, hi = paired_boot(y[te], bF, eF)
    d2, lo2, _ = paired_boot(y[te2], b2, e2)
    return {"rep": rep,
            "base_auc": float(roc_auc_score(y[te], bF)),
            "ens_auc": float(roc_auc_score(y[te], eF)),
            "delta": d, "ci_lo": lo, "ci_hi": hi, "clears": bool(lo > 0),
            "delta_2min": d2, "clears_2min": bool(lo2 > 0),
            "base_brier": float(brier_score_loss(y[te], bF)),
            "ens_brier": float(brier_score_loss(y[te], eF)),
            "base_ece": ece(y[te], bF), "ens_ece": ece(y[te], eF)}


def main():
    print("=" * 104)
    print("CATBOOST SEED-ENSEMBLE vs the CURRENT 31-feature production baseline")
    print("=" * 104)
    _init()
    X, y, tr, te, te2 = _G["X"], _G["y"], _G["tr"], _G["te"], _G["te2"]
    res = {"cat_params": CAT_PARAMS, "n_level2": N_LEVEL2}

    # ---------------- LEVEL 1 ----------------
    print(f"\nLEVEL 1 -- SEED STABILITY (same training split, {max(N_LEVELS)} members)")
    print("  necessary but NOT sufficient: the training draw is held fixed\n")
    t0 = time.time()
    members = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(fit_member_level1, s): s for s in range(max(N_LEVELS))}
        for i, f in enumerate(as_completed(futs), 1):
            s, pF, p2 = f.result()
            members[s] = (pF, p2)
            if i % 10 == 0:
                print(f"    {i}/{max(N_LEVELS)} members ({(time.time()-t0)/60:.1f} min)",
                      flush=True)

    ind = np.array([roc_auc_score(y[te], members[s][0]) for s in sorted(members)])
    print(f"\n  individual member AUC: mean {ind.mean():.4f}  sd {ind.std():.4f}  "
          f"min {ind.min():.4f}  max {ind.max():.4f}")
    res["member_auc"] = {"mean": float(ind.mean()), "sd": float(ind.std()),
                         "min": float(ind.min()), "max": float(ind.max())}

    base = CalibratedClassifierCV(clone(_G["hgb"]), cv=5, method="sigmoid")
    base.fit(X[tr], y[tr])
    bF = base.predict_proba(X.loc[te])[:, 1]
    b2 = base.predict_proba(X.loc[te2])[:, 1]
    print(f"  production baseline (same split): AUC {roc_auc_score(y[te], bF):.4f}")

    print(f"\n  {'N':>4}{'ens AUC':>10}{'2min':>9}{'delta':>10}{'95% CI':>22}"
          f"{'clears':>8}{'Brier':>9}{'ECE':>8}")
    res["level1"] = {}
    for N in N_LEVELS:
        sel = sorted(members)[:N]
        eF = np.mean([members[s][0] for s in sel], axis=0)
        e2 = np.mean([members[s][1] for s in sel], axis=0)
        a = roc_auc_score(y[te], eF)
        d, lo, hi = paired_boot(y[te], bF, eF)
        br, ec = brier_score_loss(y[te], eF), ece(y[te], eF)
        print(f"  {N:>4}{a:>10.4f}{roc_auc_score(y[te2], e2):>9.4f}{d:>+10.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>22}{str(lo>0):>8}{br:>9.4f}{ec:>8.4f}")
        res["level1"][str(N)] = {"auc": float(a), "auc_2min": float(roc_auc_score(y[te2], e2)),
                                 "delta": d, "ci_lo": lo, "ci_hi": hi,
                                 "clears": bool(lo > 0), "brier": float(br), "ece": float(ec)}

    # ---------------- LEVEL 2 ----------------
    print(f"\nLEVEL 2 -- DATA-DRAW ROBUSTNESS ({N_RESAMPLES} training bootstraps, "
          f"N={N_LEVEL2} members each)")
    print("  this is the actual promotability bar\n")
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_resample, r): r for r in range(N_RESAMPLES)}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            print(f"    resample {futs[f]} done ({i}/{N_RESAMPLES}, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
    rows.sort(key=lambda r: r["rep"])
    d = np.array([r["delta"] for r in rows])
    d2 = np.array([r["delta_2min"] for r in rows])
    clears = sum(r["clears"] for r in rows)
    clears2 = sum(r["clears_2min"] for r in rows)
    print(f"\n  baseline AUC {np.mean([r['base_auc'] for r in rows]):.4f}  ->  "
          f"ensemble {np.mean([r['ens_auc'] for r in rows]):.4f}")
    print(f"  {'arm':<26}{'mean d':>9}{'sd':>8}{'min':>9}{'max':>9}{'pos':>7}"
          f"{'clr':>6}{'>=MDE':>7}{'d_2min':>9}{'Brier':>9}{'ECE':>8}")
    print(f"  {'CatBoost seed-ensemble':<26}{d.mean():>+9.4f}{d.std():>8.4f}"
          f"{d.min():>+9.4f}{d.max():>+9.4f}{int((d>0).sum()):>4}/{len(d)}"
          f"{clears:>3}/{len(d)}{int((d>=MDE).sum()):>4}/{len(d)}{d2.mean():>+9.4f}"
          f"{np.mean([r['ens_brier'] for r in rows]):>9.4f}"
          f"{np.mean([r['ens_ece'] for r in rows]):>8.4f}")
    print(f"  {'(production baseline)':<26}{'--':>9}{'--':>8}{'--':>9}{'--':>9}"
          f"{'--':>7}{'--':>6}{'--':>7}{'--':>9}"
          f"{np.mean([r['base_brier'] for r in rows]):>9.4f}"
          f"{np.mean([r['base_ece'] for r in rows]):>8.4f}")
    res["level2"] = {
        "n_resamples": len(rows), "delta_mean": float(d.mean()),
        "delta_sd": float(d.std()), "delta_min": float(d.min()),
        "delta_max": float(d.max()), "n_positive": int((d > 0).sum()),
        "n_clearing": int(clears), "n_at_or_above_mde": int((d >= MDE).sum()),
        "delta_2min_mean": float(d2.mean()), "n_clearing_2min": int(clears2),
        "base_auc": float(np.mean([r["base_auc"] for r in rows])),
        "ens_auc": float(np.mean([r["ens_auc"] for r in rows])),
        "base_brier": float(np.mean([r["base_brier"] for r in rows])),
        "ens_brier": float(np.mean([r["ens_brier"] for r in rows])),
        "base_ece": float(np.mean([r["base_ece"] for r in rows])),
        "ens_ece": float(np.mean([r["ens_ece"] for r in rows])),
        "rows": rows}
    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
