"""promote_variability_retrain.py -- stage the 31-feature model and run the
full validation suite at real scale.

This does NOT deploy. It writes models/best_model_variability_staged.joblib and
a results JSON; `deploy_variability_model.py` performs the swap only if the
numbers here justify it.

WHY FEATURE_COLUMNS IS NOT READ HERE. The deployed artifact and
`05_train_models.FEATURE_COLUMNS` must change together -- a 26-feature model
raises ValueError on a 31-column matrix and vice versa, so editing either alone
breaks the scheduler's next retrain tick. This script therefore builds the
31-column list EXPLICITLY from the 26 production columns plus the five
variability columns, so it can run while production is still 26. The edit to
FEATURE_COLUMNS happens in the same commit as the artifact swap.

RECIPE: production's exact object, cloned from the deployed artifact --
CalibratedClassifierCV(Pipeline([SimpleImputer(median), HGB(...)]), cv=5,
method="sigmoid"). Cloning rather than reconstructing keeps the imputer and
every tuned hyperparameter identical.

VALIDATION, all at real scale on the full backfilled dataset:
  * headline single fit on the frozen test set (26 vs 31)
  * 12 training bootstraps: AUC delta, Brier, ECE, on the full clean test set
    AND the 2-min-only subset
  * nested CV, 5-fold, training rows only
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
PROD = os.path.join(ROOT, "models", "best_model.joblib")
STAGED = os.path.join(ROOT, "models", "best_model_variability_staged.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "promote_variability_results.json")

SEED, N_RESAMPLES, N_BOOT, N_WORKERS, N_ECE_BINS, N_OUTER = 42, 12, 2000, 6, 15, 5
MDE = 0.0097
VAR5 = ["var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]


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

    missing = [c for c in VAR5 if c not in df.columns]
    assert not missing, f"training.csv is not backfilled: missing {missing}"
    for c in VAR5:
        X[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()

    base = list(m05.FEATURE_COLUMNS)
    assert len(base) == 26, f"expected 26 production features, got {len(base)}"
    new = base + VAR5
    assert len(new) == 31, f"expected 31, got {len(new)}"

    tr, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    prod = joblib.load(PROD)
    _G.update(X=X, y=y, tr=np.asarray(tr), te=te, te2=te & is2,
              base=base, new=new, hgb=clone(getattr(prod, "estimator", prod)))


def _fit(cols, Xb, yb, expect):
    assert len(cols) == expect, f"expected {expect} columns, got {len(cols)}"
    est = CalibratedClassifierCV(clone(_G["hgb"]), cv=5, method="sigmoid")
    est.fit(Xb[cols], yb)
    return est


def run_resample(rep):
    if not _G:
        _init()
    X, y, tr, te, te2 = _G["X"], _G["y"], _G["tr"], _G["te"], _G["te2"]
    Xtr, ytr = X[tr], y[tr]
    rng = np.random.RandomState(1000 + rep)
    idx = rng.randint(0, len(ytr), len(ytr))
    Xb, yb = Xtr.iloc[idx], ytr[idx]
    out = {"rep": rep}
    for label, cols, n in [("old", _G["base"], 26), ("new", _G["new"], 31)]:
        est = _fit(cols, Xb, yb, n)
        pF = est.predict_proba(X.loc[te, cols])[:, 1]
        p2 = est.predict_proba(X.loc[te2, cols])[:, 1]
        out[label] = {"auc": float(roc_auc_score(y[te], pF)),
                      "brier": float(brier_score_loss(y[te], pF)),
                      "ece": ece(y[te], pF), "pF": pF.tolist(), "p2": p2.tolist()}
    d, lo, hi = paired_boot(y[te], np.array(out["old"]["pF"]), np.array(out["new"]["pF"]))
    d2, lo2, _ = paired_boot(y[te2], np.array(out["old"]["p2"]), np.array(out["new"]["p2"]))
    out["delta"], out["ci_lo"], out["ci_hi"], out["clears"] = d, lo, hi, bool(lo > 0)
    out["delta_2min"], out["clears_2min"] = d2, bool(lo2 > 0)
    for k in ("old", "new"):
        del out[k]["pF"], out[k]["p2"]
    return out


def run_outer_fold(args):
    fold, tr_idx, va_idx = args
    if not _G:
        _init()
    X, y, tr = _G["X"], _G["y"], _G["tr"]
    Xt = X[tr].reset_index(drop=True)
    yt = y[tr]
    out = {"fold": fold}
    for label, cols, n in [("old", _G["base"], 26), ("new", _G["new"], 31)]:
        est = _fit(cols, Xt.iloc[tr_idx], yt[tr_idx], n)
        p = est.predict_proba(Xt.iloc[va_idx][cols])[:, 1]
        out[label] = {"auc": float(roc_auc_score(yt[va_idx], p)),
                      "p": p.tolist(), "idx": va_idx.tolist()}
    return out


def main():
    print("=" * 100)
    print("STAGE 31-FEATURE MODEL + FULL VALIDATION SUITE AT REAL SCALE")
    print("=" * 100)
    _init()
    X, y, tr, te, te2 = _G["X"], _G["y"], _G["tr"], _G["te"], _G["te2"]
    print(f"  training rows {int(tr.sum())}, frozen test {int(te.sum())}, "
          f"2-min subset {int(te2.sum())}")
    print(f"  26 -> 31 features (+{', '.join(VAR5)})\n")
    res = {}

    # ---- headline single fit, full training split ----
    print("[1] headline single fit on the frozen test set")
    heads = {}
    for label, cols, n in [("old", _G["base"], 26), ("new", _G["new"], 31)]:
        est = _fit(cols, X[tr], y[tr], n)
        p = est.predict_proba(X.loc[te, cols])[:, 1]
        heads[label] = (est, p)
        print(f"    {label:3s} {n} feat: AUC {roc_auc_score(y[te], p):.4f}  "
              f"Brier {brier_score_loss(y[te], p):.4f}  ECE {ece(y[te], p):.4f}")
    d, lo, hi = paired_boot(y[te], heads["old"][1], heads["new"][1])
    print(f"    paired bootstrap: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"clears={lo > 0}")
    res["headline_old_auc"] = float(roc_auc_score(y[te], heads["old"][1]))
    res["headline_new_auc"] = float(roc_auc_score(y[te], heads["new"][1]))
    res["headline_delta"], res["headline_ci_lo"], res["headline_ci_hi"] = d, lo, hi
    res["brier_old"] = float(brier_score_loss(y[te], heads["old"][1]))
    res["brier_new"] = float(brier_score_loss(y[te], heads["new"][1]))
    res["ece_old"] = ece(y[te], heads["old"][1])
    res["ece_new"] = ece(y[te], heads["new"][1])

    joblib.dump(heads["new"][0], STAGED)
    import hashlib
    staged_md5 = hashlib.md5(open(STAGED, "rb").read()).hexdigest()
    res["staged_md5"] = staged_md5
    print(f"    staged -> {os.path.relpath(STAGED, ROOT)}  md5 {staged_md5}")

    # ---- 12 training bootstraps ----
    print(f"\n[2] {N_RESAMPLES} training bootstraps")
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_resample, r): r for r in range(N_RESAMPLES)}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            print(f"    resample {futs[f]} ({i}/{N_RESAMPLES}, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
    rows.sort(key=lambda r: r["rep"])
    dd = np.array([r["delta"] for r in rows])
    d2 = np.array([r["delta_2min"] for r in rows])
    clears = sum(r["clears"] for r in rows)
    clears2 = sum(r["clears_2min"] for r in rows)
    print(f"    delta {dd.mean():+.4f} (sd {dd.std():.4f}), positive "
          f"{int((dd>0).sum())}/{len(dd)}, clears {clears}/{len(dd)}, "
          f">=MDE {int((dd>=MDE).sum())}/{len(dd)}")
    print(f"    2-min delta {d2.mean():+.4f}, clears {clears2}/{len(d2)}")
    print(f"    Brier {np.mean([r['old']['brier'] for r in rows]):.4f} -> "
          f"{np.mean([r['new']['brier'] for r in rows]):.4f}   "
          f"ECE {np.mean([r['old']['ece'] for r in rows]):.4f} -> "
          f"{np.mean([r['new']['ece'] for r in rows]):.4f}")
    res.update(resampled_delta_mean=float(dd.mean()), resampled_delta_sd=float(dd.std()),
               resampled_positive=int((dd > 0).sum()), resampled_clears=int(clears),
               resampled_at_mde=int((dd >= MDE).sum()),
               resampled_delta_2min=float(d2.mean()), resampled_clears_2min=int(clears2),
               resampled_brier_old=float(np.mean([r["old"]["brier"] for r in rows])),
               resampled_brier_new=float(np.mean([r["new"]["brier"] for r in rows])),
               resampled_ece_old=float(np.mean([r["old"]["ece"] for r in rows])),
               resampled_ece_new=float(np.mean([r["new"]["ece"] for r in rows])),
               resample_rows=[{k: v for k, v in r.items()} for r in rows])

    # ---- nested CV ----
    print(f"\n[3] nested CV, {N_OUTER}-fold, training rows only")
    Xt = X[tr].reset_index(drop=True)
    yt = y[tr]
    skf = StratifiedKFold(N_OUTER, shuffle=True, random_state=SEED)
    folds = [(i, a, b) for i, (a, b) in enumerate(skf.split(Xt, yt))]
    t0, frows = time.time(), []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for r in ex.map(run_outer_fold, folds):
            frows.append(r)
            print(f"    fold {r['fold']} ({(time.time()-t0)/60:.1f} min)", flush=True)
    frows.sort(key=lambda r: r["fold"])
    pooled = {k: np.full(len(yt), np.nan) for k in ("old", "new")}
    for r in frows:
        for k in ("old", "new"):
            pooled[k][np.array(r[k]["idx"])] = np.array(r[k]["p"])
    for k in ("old", "new"):
        assert np.isfinite(pooled[k]).all()
    ncv_old = roc_auc_score(yt, pooled["old"])
    ncv_new = roc_auc_score(yt, pooled["new"])
    nd, nlo, nhi = paired_boot(yt, pooled["old"], pooled["new"])
    print(f"    pooled OOF AUC {ncv_old:.4f} -> {ncv_new:.4f}  "
          f"delta {nd:+.4f} CI [{nlo:+.4f}, {nhi:+.4f}] clears={nlo > 0}")
    print(f"    fold AUC mean {np.mean([r['old']['auc'] for r in frows]):.4f} -> "
          f"{np.mean([r['new']['auc'] for r in frows]):.4f}")
    res.update(nested_cv_old=float(ncv_old), nested_cv_new=float(ncv_new),
               nested_cv_delta=float(nd), nested_cv_ci_lo=float(nlo),
               nested_cv_ci_hi=float(nhi),
               nested_fold_old=[float(r["old"]["auc"]) for r in frows],
               nested_fold_new=[float(r["new"]["auc"]) for r in frows])

    json.dump(res, open(RESULTS, "w"), indent=2, default=float)
    print("\n" + "=" * 100)
    print(f"Saved {RESULTS}")
    print(f"Staged artifact md5 {staged_md5} -- NOT deployed. "
          "Run deploy_variability_model.py to swap.")


if __name__ == "__main__":
    main()
