"""weak_secondary_validate.py -- leakage check, then retrain, for the
noise-normalised weak-secondary feature.

ORDER MATTERS AND IS NOT NEGOTIABLE: the leakage/redundancy checks run first
and are reported whatever the retrain does. A feature that turns out to be a
0.80+ correlate of something already in the model has not added information
even if the AUC ticks up, and this project has been caught by exactly that
before (the engineered ratios "measured what they were designed to measure and
the model already knew it").

Baselines are computed IN-RUN for both the bare Pipeline and the calibrated
production wrapper, because the two differ by ~0.0035 and mixing them
manufactures a fake result -- the trap caught during the K2 work. Every delta
below compares like with like.
"""
import os
import sys
import json
import importlib.util
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
SECONDARY = os.path.join(SCRIPT_DIR, "weak_secondary_features.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "weak_secondary_results.json")

N_BOOT = 2000
SEED = 42
REDUNDANCY_CUTOFF = 0.80
NEW = ["sec_significance", "sec_depth_windowed"]


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def paired_boot(y, pa, pb, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
    d = np.asarray(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def ece(y, p, bins=10):
    e, n = 0.0, len(y)
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


def main():
    m05 = _m05()
    df = pd.read_csv(TRAINING)
    sec = pd.read_csv(SECONDARY)
    df["host"] = df["host"].astype(str)
    sec["host"] = sec["host"].astype(str)
    df = df.merge(sec[["host"] + NEW], on="host", how="left")

    X0, y = m05.build_feature_matrix(df)
    X0 = X0.reset_index(drop=True)
    y = np.asarray(y)
    tr, te = m05.split_by_host(df)
    tr, te = np.asarray(tr), np.asarray(te)

    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    te2 = te & is2

    out = {}

    # ---------------------------------------------------------- leakage first
    print("=" * 80)
    print("LEAKAGE / REDUNDANCY CHECK -- run before any model result")
    print("=" * 80)
    print(f"  {'feature':<24}{'NaN% pos':>10}{'NaN% neg':>10}{'single-feat AUC':>18}")
    lk = {}
    for f in NEW:
        v = pd.to_numeric(df[f], errors="coerce")
        nan_p = float(v[y == 1].isna().mean() * 100)
        nan_n = float(v[y == 0].isna().mean() * 100)
        m = v.notna().to_numpy()
        auc = float(roc_auc_score(y[m], v[m])) if m.sum() > 50 else np.nan
        print(f"  {f:<24}{nan_p:>10.1f}{nan_n:>10.1f}{auc:>18.3f}")
        lk[f] = {"nan_pct_pos": nan_p, "nan_pct_neg": nan_n,
                 "single_feature_auc": auc}
    print("\n  NaN-rate parity matters: a feature missing at different rates by")
    print("  class leaks the label through its own missingness.")

    print(f"\n  redundancy vs the existing {len(X0.columns)} features "
          f"(|r| >= {REDUNDANCY_CUTOFF} is this project's cutoff):")
    worst = {}
    for f in NEW:
        v = pd.to_numeric(df[f], errors="coerce")
        cors = {}
        for c0 in X0.columns:
            u = pd.to_numeric(X0[c0], errors="coerce")
            m = v.notna() & u.notna()
            if m.sum() > 100:
                r = float(np.corrcoef(v[m], u[m])[0, 1])
                if np.isfinite(r):
                    cors[c0] = r
        top = sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:3]
        worst[f] = top
        flag = "REDUNDANT" if abs(top[0][1]) >= REDUNDANCY_CUTOFF else "ok"
        print(f"    {f:<24} max |r| = {abs(top[0][1]):.3f} vs {top[0][0]}  [{flag}]")
        for c0, r in top[1:]:
            print(f"      {'':<22} {r:+.3f} vs {c0}")
        lk[f]["top_correlations"] = top
    out["leakage"] = lk

    # ------------------------------------------------------------------ arms
    prod = joblib.load(PROD)
    bare = clone(getattr(prod, "estimator", prod))
    cal = clone(prod)

    Xn = X0.copy()
    for f in NEW:
        Xn[f] = pd.to_numeric(df[f], errors="coerce").to_numpy()

    print("\n" + "=" * 80)
    print("RETRAIN -- like compared with like (bare vs bare, calibrated vs calibrated)")
    print("=" * 80)
    print(f"  train {int(tr.sum())} | test {int(te.sum())} | "
          f"2-min-only test {int(te2.sum())}")
    print(f"\n  {'arm':<34}{'full test':>11}{'delta [95% CI]':>26}"
          f"{'2-min test':>12}{'delta [95% CI]':>26}")

    res = {}
    for tag, proto in (("bare", bare), ("calibrated", cal)):
        mb = clone(proto).fit(X0[tr], y[tr])
        pb_f = mb.predict_proba(X0[te])[:, 1]
        pb_2 = mb.predict_proba(X0[te2])[:, 1]
        mn = clone(proto).fit(Xn[tr], y[tr])
        pn_f = mn.predict_proba(Xn[te])[:, 1]
        pn_2 = mn.predict_proba(Xn[te2])[:, 1]

        aF0, a20 = roc_auc_score(y[te], pb_f), roc_auc_score(y[te2], pb_2)
        aF1, a21 = roc_auc_score(y[te], pn_f), roc_auc_score(y[te2], pn_2)
        dF, loF, hiF = paired_boot(y[te], pb_f, pn_f)
        d2, lo2, hi2 = paired_boot(y[te2], pb_2, pn_2)

        print(f"  {tag+' baseline (24 feat)':<34}{aF0:>11.4f}{'--':>26}"
              f"{a20:>12.4f}{'--':>26}")
        print(f"  {tag+' + weak secondary (26)':<34}{aF1:>11.4f}"
              f"{f'{dF:+.4f} [{loF:+.4f},{hiF:+.4f}]':>26}"
              f"{a21:>12.4f}{f'{d2:+.4f} [{lo2:+.4f},{hi2:+.4f}]':>26}")
        res[tag] = {"baseline_full": float(aF0), "baseline_2min": float(a20),
                    "new_full": float(aF1), "new_2min": float(a21),
                    "delta_full": dF, "ci_full": [loF, hiF],
                    "clears_full": bool(loF > 0),
                    "delta_2min": d2, "ci_2min": [lo2, hi2],
                    "clears_2min": bool(lo2 > 0)}
        if tag == "calibrated":
            res[tag]["brier_baseline"] = float(brier_score_loss(y[te], pb_f))
            res[tag]["brier_new"] = float(brier_score_loss(y[te], pn_f))
            res[tag]["ece_baseline"] = ece(y[te], pb_f)
            res[tag]["ece_new"] = ece(y[te], pn_f)
    out["arms"] = res

    print(f"\n  calibration (calibrated arm): "
          f"Brier {res['calibrated']['brier_baseline']:.4f} -> "
          f"{res['calibrated']['brier_new']:.4f}   "
          f"ECE {res['calibrated']['ece_baseline']:.4f} -> "
          f"{res['calibrated']['ece_new']:.4f}")

    # ------------------------------------------------------------- nested CV
    print("\n  nested CV (5-fold, train split only, bare pipeline):")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for nm, XX in (("24 features", X0), ("26 features", Xn)):
        p = cross_val_predict(clone(bare), XX[tr], y[tr], cv=cv,
                              method="predict_proba")[:, 1]
        a = roc_auc_score(y[tr], p)
        print(f"    {nm:<14} {a:.4f}")
        out.setdefault("nested_cv", {})[nm] = float(a)

    clears = any(v["clears_full"] or v["clears_2min"] for v in res.values())
    print("\n" + "=" * 80)
    print("VERDICT: " + ("CLEARS ci_lo > 0 somewhere -- inspect which population"
                         if clears else
                         "NO ARM CLEARS ci_lo > 0 on either population"))
    print("=" * 80)
    out["clears_anywhere"] = bool(clears)

    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
