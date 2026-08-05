"""timeseries_stats_validate.py -- resampled validation of the two new
time-series statistics against the deployed 0.9208 / 26-feature model.

WHY THIS ONE EARNED A MODEL FIT. The pre-model checks came back stronger than
anything else tested recently, and in the direction predicted BEFORE measuring:

    ts_var_ratio   AUC 0.4195   FP > planets, as predicted
    ts_acf_lag1    AUC 0.4131   FP > planets, as predicted
    ts_acf_1hr     AUC 0.4491   FP > planets, as predicted

with clean missingness (availability AUC 0.4917), no redundancy (max |r| 0.413
against the 26 production features) and no spatial exposure (|r| <= 0.073
against |galactic b|). Coverage 96.8%.

ARMS
  A  + ts_var_ratio                 the second moment the closed flux-stats
                                    work never compared
  B  + ts_acf_lag1, ts_acf_1hr      residual temporal correlation, untested
                                    anywhere in this project
  C  all three                      combined

THE BUG THIS SCRIPT ASSERTS AGAINST. The closed flux-distribution experiment
first reported "base 24 -> with new 24": `build_feature_matrix` selects only
`FEATURE_COLUMNS`, so merging new columns into the dataframe never put them in
X, and it would have reported a perfectly null result for a comparison that
never differed. Every arm here therefore ASSERTS its column count before
fitting, and the run aborts rather than producing a reassuring zero.
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
from sklearn.metrics import roc_auc_score, brier_score_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import fast_auc  # noqa: E402

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
TSFEAT = os.path.join(SCRIPT_DIR, "timeseries_stats_features.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "timeseries_stats_validate_results.json")

SEED, N_RESAMPLES, N_BOOT, N_WORKERS, N_ECE_BINS = 42, 12, 1500, 6, 15
MDE = 0.0097
VAR = ["ts_var_ratio"]
ACF = ["ts_acf_lag1", "ts_acf_1hr"]
ARMS = [("A: +var_ratio", VAR), ("B: +residual ACF", ACF), ("C: all three", VAR + ACF)]


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
    ts = pd.read_csv(TSFEAT)
    merged = df[["host"]].merge(ts[["host"] + VAR + ACF], on="host", how="left")
    for c in VAR + ACF:
        X[c] = pd.to_numeric(merged[c], errors="coerce").to_numpy()
    tr, _ = m05.split_by_host(df)
    frozen = m05.frozen_test_mask(df)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    prod = joblib.load(PROD)
    _G.update(X=X, y=y, tr=np.asarray(tr), te=frozen, te2=frozen & is2,
              base=list(m05.FEATURE_COLUMNS),
              hgb=clone(getattr(prod, "estimator", prod)))


def _fit(cols, Xb, yb, X, te, te2, expect):
    # guard against the silent-null bug: the column list must actually differ
    assert len(cols) == expect, f"expected {expect} columns, got {len(cols)}"
    est = CalibratedClassifierCV(clone(_G["hgb"]), cv=5, method="sigmoid")
    est.fit(Xb[cols], yb)
    return (est.predict_proba(X.loc[te, cols])[:, 1],
            est.predict_proba(X.loc[te2, cols])[:, 1])


def run_resample(rep):
    if not _G:
        _init()
    X, y, tr, te, te2 = _G["X"], _G["y"], _G["tr"], _G["te"], _G["te2"]
    base = _G["base"]
    Xtr, ytr = X[tr], y[tr]
    rng = np.random.RandomState(1000 + rep)
    idx = rng.randint(0, len(ytr), len(ytr))
    Xb, yb = Xtr.iloc[idx], ytr[idx]

    bF, b2 = _fit(base, Xb, yb, X, te, te2, len(base))
    row = {"rep": rep, "base_auc": float(roc_auc_score(y[te], bF)),
           "base_brier": float(brier_score_loss(y[te], bF)),
           "base_ece": ece(y[te], bF), "arms": {}}
    for name, extra in ARMS:
        cols = base + extra
        pF, p2 = _fit(cols, Xb, yb, X, te, te2, len(base) + len(extra))
        d, lo, hi = paired_boot(y[te], bF, pF)
        d2, lo2, hi2 = paired_boot(y[te2], b2, p2)
        row["arms"][name] = {
            "n_features": len(cols), "auc": float(roc_auc_score(y[te], pF)),
            "delta": d, "clears": bool(lo > 0),
            "delta_2min": d2, "clears_2min": bool(lo2 > 0),
            "brier": float(brier_score_loss(y[te], pF)), "ece": ece(y[te], pF)}
    return row


def main():
    print("=" * 106)
    print("TIME-SERIES STATISTICS -- resampled validation vs the deployed 0.9208")
    print("=" * 106)
    _init()
    print(f"  base features: {len(_G['base'])}")
    for n, e in ARMS:
        print(f"    {n:<20} -> {len(_G['base'])+len(e)} features  {e}")
    print(f"  {N_RESAMPLES} bootstraps, production recipe refit on identical rows\n")

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
    bb = np.array([r["base_brier"] for r in rows])
    be = np.array([r["base_ece"] for r in rows])
    print("\n" + "=" * 106)
    print(f"  baseline: AUC {ba.mean():.4f} (sd {ba.std():.4f})  "
          f"Brier {bb.mean():.4f}  ECE {be.mean():.4f}\n")
    print(f"  {'arm':<20}{'nfeat':>7}{'mean d':>9}{'sd':>8}{'min':>9}{'max':>9}"
          f"{'pos':>7}{'clr':>6}{'>=MDE':>7}{'d_2min':>9}{'Brier':>9}{'ECE':>8}")
    out = {"n_resamples": len(rows), "baseline_auc": float(ba.mean()),
           "baseline_brier": float(bb.mean()), "baseline_ece": float(be.mean()),
           "rows": rows, "summary": {}}
    for name, _ in ARMS:
        d = np.array([r["arms"][name]["delta"] for r in rows])
        d2 = np.array([r["arms"][name]["delta_2min"] for r in rows])
        c = sum(r["arms"][name]["clears"] for r in rows)
        c2 = sum(r["arms"][name]["clears_2min"] for r in rows)
        br = np.mean([r["arms"][name]["brier"] for r in rows])
        ec = np.mean([r["arms"][name]["ece"] for r in rows])
        nf = rows[0]["arms"][name]["n_features"]
        print(f"  {name:<20}{nf:>7}{d.mean():>+9.4f}{d.std():>8.4f}{d.min():>+9.4f}"
              f"{d.max():>+9.4f}{int((d>0).sum()):>4}/{len(rows)}{c:>3}/{len(rows)}"
              f"{int((d>=MDE).sum()):>4}/{len(rows)}{d2.mean():>+9.4f}{br:>9.4f}{ec:>8.4f}")
        out["summary"][name] = {
            "n_features": nf, "delta_mean": float(d.mean()), "delta_sd": float(d.std()),
            "delta_min": float(d.min()), "delta_max": float(d.max()),
            "n_positive": int((d > 0).sum()), "n_clearing": c,
            "n_at_or_above_mde": int((d >= MDE).sum()),
            "delta_2min_mean": float(d2.mean()), "n_clearing_2min": c2,
            "mean_brier": float(br), "mean_ece": float(ec),
            "brier_vs_base": float(br - bb.mean()), "ece_vs_base": float(ec - be.mean())}
    print("=" * 106)
    json.dump(out, open(RESULTS, "w"), indent=2, default=float)
    if os.path.exists(RESULTS + ".partial"):
        os.remove(RESULTS + ".partial")
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
