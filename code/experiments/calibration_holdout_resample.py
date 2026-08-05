"""calibration_holdout_resample.py -- >=10 training bootstraps for clearing arms.

Reads `calibration_holdout_results.json` and resamples EVERY arm that cleared
`ci_lo > 0` on the single fit. Nothing from a single fit is treated as a
finding in this project: stage 2 of the earlier sweep turned +0.0029 into
+0.0001 and flipped the best-looking arm's sign, so this step is mandatory
rather than confirmatory.

Resampling is of the TRAINING ROWS (regenerating every fit), not `random_state`.
HGB is deterministic at this data size -- early stopping is off and binning does
not subsample -- so the seed axis is literally inert for it and would fabricate
a reassuring zero variance. CatBoost is genuinely stochastic, and the training
draw dominates its seed variance anyway (measured: axis effect -0.0004 against
a -0.0050 one-row baseline shift).

12 resamples, seeds 1000..1011. The first 8 coincide with the earlier stage-2
seeds so those runs remain directly comparable; the brief asks for >=10.

If no arm cleared, this exits without fitting anything and says so.
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
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
SWEEP = os.path.join(SCRIPT_DIR, "calibration_holdout_results.json")
RESULTS = os.path.join(SCRIPT_DIR, "calibration_holdout_resample_results.json")

SEED = 42
N_RESAMPLES = 12
N_BOOT = 1500
N_WORKERS = 8          # 8 physical cores; every fit is OMP_NUM_THREADS=1
N_ECE_BINS = 15
CAT_PARAMS = dict(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=9.0)


def ece(y, p, bins=N_ECE_BINS):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def fast_auc(y, p):
    """Exact ROC-AUC by rank-sum, averaged ranks for ties. See the sweep script:
    verified to 1e-12 against roc_auc_score on 400 tie-heavy cases, and ~18x
    faster, which matters because isotonic arms produce many tied probabilities.
    """
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
    hgb = clone(getattr(prod, "estimator", prod))
    from catboost import CatBoostClassifier
    cat = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("clf", CatBoostClassifier(
                        verbose=0, random_seed=SEED,
                        auto_class_weights="Balanced",
                        allow_writing_files=False, **CAT_PARAMS))])
    _G.update(X=X, y=y, tr=np.asarray(tr), te=frozen, te2=frozen & is2,
              models={"HGB": hgb, "CatBoost": cat})


def fit_and_predict(spec, Xb, yb, X, te, te2):
    base = clone(_G["models"][spec["family"]])
    if spec["kind"] == "bare":
        est = base.fit(Xb, yb)
    elif spec["kind"] == "crossfit":
        est = CalibratedClassifierCV(
            base, cv=int(spec["param"]), method=spec["method"]).fit(Xb, yb)
    else:
        Xh_tr, Xh_cal, yh_tr, yh_cal = train_test_split(
            Xb, yb, test_size=spec["param"], stratify=yb, random_state=SEED)
        base.fit(Xh_tr, yh_tr)
        est = CalibratedClassifierCV(
            FrozenEstimator(base), method=spec["method"]).fit(Xh_cal, yh_cal)
    return est.predict_proba(X[te])[:, 1], est.predict_proba(X[te2])[:, 1]


def run_resample(rep, specs):
    if not _G:
        _init()
    X, y, tr, te, te2 = _G["X"], _G["y"], _G["tr"], _G["te"], _G["te2"]
    Xtr, ytr = X[tr], y[tr]
    rng = np.random.RandomState(1000 + rep)
    idx = rng.randint(0, len(ytr), len(ytr))
    Xb, yb = Xtr.iloc[idx], ytr[idx]

    # production, refit on the same resampled rows
    bF, b2 = fit_and_predict(
        {"family": "HGB", "kind": "crossfit", "method": "sigmoid", "param": 5},
        Xb, yb, X, te, te2)
    row = {"rep": rep,
           "baseline_full": float(roc_auc_score(y[te], bF)),
           "baseline_2min": float(roc_auc_score(y[te2], b2)),
           "baseline_brier": float(brier_score_loss(y[te], bF)),
           "baseline_ece": ece(y[te], bF), "arms": {}}

    for spec in specs:
        pF, p2 = fit_and_predict(spec, Xb, yb, X, te, te2)
        dF, loF, hiF = paired_boot(y[te], bF, pF)
        d2, lo2, hi2 = paired_boot(y[te2], b2, p2)
        row["arms"][spec["label"]] = {
            "auc_full": float(roc_auc_score(y[te], pF)),
            "auc_2min": float(roc_auc_score(y[te2], p2)),
            "brier": float(brier_score_loss(y[te], pF)),
            "ece": ece(y[te], pF),
            "delta_full": dF, "clears_full": bool(loF > 0),
            "delta_2min": d2, "clears_2min": bool(lo2 > 0)}
    return row


def main():
    sweep = json.load(open(SWEEP))
    clearing = sweep["single_fit_clearing"]
    by_key = {(r["family"], r["arm"]): r for r in sweep["rows"]}

    if not clearing:
        print("No arm cleared ci_lo>0 on the single fit -- nothing to resample.")
        print("That is the result; no fits were run.")
        json.dump({"n_clearing_single_fit": 0, "resampled": []},
                  open(RESULTS, "w"), indent=2)
        return

    specs = []
    for c in clearing:
        r = by_key[(c["family"], c["arm"])]
        specs.append({"label": f"{r['family']} {r['arm']}", "family": r["family"],
                      "kind": r["kind"], "method": r["method"], "param": r["param"]})

    print("=" * 110)
    print(f"RESAMPLE VALIDATION -- {len(specs)} arm(s) that cleared on a single fit")
    print("=" * 110)
    for s in specs:
        print(f"    {s['label']}")
    print(f"\n  {N_RESAMPLES} training-data bootstraps (seeds 1000..{999+N_RESAMPLES});")
    print("  the first 8 match the earlier stage-2 seeds. Baseline is production")
    print("  (HGB sigmoid cv=5) refit on the SAME resampled rows each time.\n")

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_resample, r, specs): r for r in range(N_RESAMPLES)}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            # checkpoint every resample -- this run is ~90 min of fitting and
            # must not be lost to a bug in the summary code below.
            json.dump(rows, open(RESULTS + ".partial", "w"), default=float)
            print(f"  resample {futs[f]} done ({i}/{N_RESAMPLES}, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
    rows.sort(key=lambda r: r["rep"])

    bf = np.array([r["baseline_full"] for r in rows])
    print("\n" + "=" * 110)
    print("DISTRIBUTION")
    print("=" * 110)
    print(f"  production baseline: mean AUC {bf.mean():.4f} (sd {bf.std():.4f})\n")
    print(f"  {'arm':<34}{'mean':>9}{'sd':>8}{'min':>9}{'max':>9}"
          f"{'pos':>8}{'clears':>9}{'d_2min':>9}{'clears2':>9}")
    out = {"n_clearing_single_fit": len(specs), "n_resamples": len(rows),
           "baseline_mean_auc": float(bf.mean()), "baseline_sd_auc": float(bf.std()),
           "rows": rows, "summary": {}}
    for s in specs:
        k = s["label"]
        dF = np.array([r["arms"][k]["delta_full"] for r in rows])
        d2 = np.array([r["arms"][k]["delta_2min"] for r in rows])
        cF = sum(r["arms"][k]["clears_full"] for r in rows)
        c2 = sum(r["arms"][k]["clears_2min"] for r in rows)
        print(f"  {k:<34}{dF.mean():>+9.4f}{dF.std():>8.4f}{dF.min():>+9.4f}"
              f"{dF.max():>+9.4f}{int((dF>0).sum()):>5}/{len(rows)}"
              f"{cF:>6}/{len(rows)}{d2.mean():>+9.4f}{c2:>6}/{len(rows)}")
        out["summary"][k] = {
            "delta_full_mean": float(dF.mean()), "delta_full_sd": float(dF.std()),
            "delta_full_min": float(dF.min()), "delta_full_max": float(dF.max()),
            "n_positive_full": int((dF > 0).sum()), "n_clearing_full": cF,
            "delta_2min_mean": float(d2.mean()), "n_clearing_2min": c2,
            "mean_brier": float(np.mean([r["arms"][k]["brier"] for r in rows])),
            "mean_ece": float(np.mean([r["arms"][k]["ece"] for r in rows])),
            "n": len(rows)}

    robust = [k for k, v in out["summary"].items()
              if v["n_clearing_full"] >= 0.9 * len(rows)
              and v["n_clearing_2min"] >= 0.9 * len(rows)]
    print("\n" + "=" * 110)
    print("ROBUSTLY CLEARS (>=90% of resamples, both populations): "
          + (", ".join(robust) if robust else "none"))
    print("=" * 110)
    out["robust_arms"] = robust
    json.dump(out, open(RESULTS, "w"), indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
