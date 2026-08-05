"""retrain_with_centroid_multisector.py -- re-tests centroid displacement as a
classifier feature at the coverage the multi-sector fix actually delivers.

The original test (retrain_with_centroid.py) measured shift_pixels on only
53.3% of training stars, because the check examined a single sector and gave up
if that sector's depth didn't match the ephemeris. Median-imputing the other
46.7% diluted whatever signal the feature carries, so its negative result was
confounded: "centroid displacement carries no signal" could not be separated
from "we only measured it on half the stars". Trying up to 6 sectors raises
real coverage to 77.6%, which is what this script re-tests.

TWO DIFFERENCES FROM THE ORIGINAL SCRIPT, both deliberate:

1. It uses m05.split_by_host (the frozen manifest) instead of
   train_test_split(df.index, ...). The original predates the Phase 1
   contamination fix and still used a positional split, which silently
   reshuffles whenever training.csv grows -- and training.csv HAS grown since.
   A positional split here would compare the two feature sets on different
   test stars than every other number in this project, and could leak training
   stars into the test set exactly as it did before.

2. Missing shift_pixels is still median-imputed via the same SimpleImputer
   convention, so base-vs-centroid stays a clean single-variable comparison.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import roc_auc_score  # exact drop-in, ~23x faster inside bootstraps

TRAINING_CSV = os.path.join(CODE_DIR, "..", "data", "training_dataset", "training.csv")
CENTROID_CSV = os.path.join(SCRIPT_DIR, "training_centroid_results_multisector.csv")
PRIOR_CENTROID_CSV = os.path.join(SCRIPT_DIR, "training_centroid_results.csv")
OUT_CSV = os.path.join(SCRIPT_DIR, "training_with_centroid_multisector.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "retrain_centroid_multisector_results.json")

RANDOM_SEED = 42
N_CV_FOLDS = 5
N_BOOTSTRAP = 2000

BASE_FEATURE_COLUMNS = [
    "SDE", "SDE_raw", "FAP", "period", "period_uncertainty", "duration", "depth",
    "depth_mean", "depth_mean_std", "depth_mean_even", "depth_mean_odd",
    "odd_even_mismatch", "rp_rs", "snr", "transit_count", "distinct_transit_count",
    "empty_transit_count",
    "st_rad", "st_teff",
    "chi2red_min", "depth_consistency_std", "secondary_eclipse_depth", "transit_shape_ratio",
    "depth_duration_ratio",
]
CENTROID_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ["shift_pixels"]


def _load_m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def build_matrix(df, columns):
    # Identical to retrain_with_centroid.py's build_feature_matrix. Both steps
    # matter: TLS emits inf for some ratio features, which SimpleImputer
    # rejects outright, and a missing FAP means "no false-alarm probability
    # could be computed", which this project encodes as 1.0 (worst) rather
    # than imputing it to the median of stars that did get one.
    X = df[columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X["FAP"] = X["FAP"].fillna(1.0)
    return X


def build_hgb_pipeline():
    # Same hyperparameters as the original centroid experiment and as
    # production. Using sklearn defaults here instead would change base AUC
    # too, so the base-vs-centroid difference would no longer be comparable
    # with the single-sector result this script exists to re-test.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.05,
            class_weight="balanced", random_state=RANDOM_SEED,
        )),
    ])


def paired_bootstrap_auc_diff(y_test, proba_a, proba_b, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y_test)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], proba_b[idx]) - roc_auc_score(y[idx], proba_a[idx]))
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    print("=" * 72)
    print("Centroid feature at multi-sector coverage: honest retrain-and-compare")
    print("=" * 72)

    df = pd.read_csv(TRAINING_CSV)
    cen = pd.read_csv(CENTROID_CSV)
    cen = cen[cen["status"] == "completed"][["host", "shift_pixels"]]
    df = df.merge(cen, on="host", how="left")
    df.to_csv(OUT_CSV, index=False)

    n_cov = df["shift_pixels"].notna().sum()
    print(f"Rows: {len(df)} ({(df['label']==1).sum()} positive, {(df['label']==0).sum()} negative)")
    print(f"shift_pixels available: {n_cov}/{len(df)} = {100*n_cov/len(df):.1f}%")

    prior = pd.read_csv(PRIOR_CENTROID_CSV)
    prior_cov = (prior["status"] == "completed").sum()
    print(f"  (single-sector run measured {prior_cov}/{len(prior)} = "
          f"{100*prior_cov/len(prior):.1f}%)")

    # ---- leakage re-check at the new coverage -------------------------------
    print("\n--- LEAKAGE CHECK ---")
    miss = df["shift_pixels"].isna()
    for lbl, name in [(1, "positive"), (0, "negative")]:
        sel = df["label"] == lbl
        print(f"  missing rate, {name} class: {100*miss[sel].mean():.1f}%")
    # Does MISSINGNESS alone predict the label? (the dangerous shortcut)
    auc_missing = roc_auc_score(df["label"], (~miss).astype(int))
    print(f"  AUC of the missingness indicator alone: {auc_missing:.3f}")
    have = df[~miss]
    auc_raw = roc_auc_score(have["label"], have["shift_pixels"])
    print(f"  single-feature AUC of raw shift_pixels (rows that have it): {auc_raw:.3f}")

    # ---- frozen split, same rows for both models ----------------------------
    m05 = _load_m05()
    train_mask, test_mask = m05.split_by_host(df)
    y = df["label"]
    X_base = build_matrix(df, BASE_FEATURE_COLUMNS)
    X_cent = build_matrix(df, CENTROID_FEATURE_COLUMNS)

    overlap = set(df.loc[train_mask, "host"]) & set(df.loc[test_mask, "host"])
    print(f"\nSplit: {train_mask.sum()} train / {test_mask.sum()} test; "
          f"hosts on both sides = {len(overlap)} (must be 0)")

    Xtr_b, Xte_b = X_base[train_mask], X_base[test_mask]
    Xtr_c, Xte_c = X_cent[train_mask], X_cent[test_mask]
    ytr, yte = y[train_mask], y[test_mask]

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cvb = cross_validate(build_hgb_pipeline(), Xtr_b, ytr, cv=cv, scoring="roc_auc")
    cvc = cross_validate(build_hgb_pipeline(), Xtr_c, ytr, cv=cv, scoring="roc_auc")
    print(f"\n5-fold CV (train only): base = {cvb['test_score'].mean():.4f} "
          f"+/- {cvb['test_score'].std():.4f}, "
          f"with centroid = {cvc['test_score'].mean():.4f} +/- {cvc['test_score'].std():.4f}")

    mb = build_hgb_pipeline().fit(Xtr_b, ytr)
    pb = mb.predict_proba(Xte_b)[:, 1]
    ab = roc_auc_score(yte, pb)
    mc = build_hgb_pipeline().fit(Xtr_c, ytr)
    pc = mc.predict_proba(Xte_c)[:, 1]
    ac = roc_auc_score(yte, pc)
    print(f"\nHeld-out test ROC-AUC: base = {ab:.4f}, with centroid = {ac:.4f}")

    mean_d, lo, hi = paired_bootstrap_auc_diff(yte, pb, pc)
    print(f"Paired bootstrap ({N_BOOTSTRAP} resamples) on the difference:")
    print(f"  mean {mean_d:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
    clears = lo > 0
    print(f"\nPROMOTION BAR (CI entirely above zero): {'CLEARS' if clears else 'DOES NOT CLEAR'}")

    out = {
        "coverage_multisector": int(n_cov),
        "coverage_multisector_pct": float(100 * n_cov / len(df)),
        "coverage_single_sector": int(prior_cov),
        "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        "hosts_on_both_sides": len(overlap),
        "missingness_auc": float(auc_missing),
        "raw_shift_pixels_auc": float(auc_raw),
        "cv_auc_base_mean": float(cvb["test_score"].mean()),
        "cv_auc_centroid_mean": float(cvc["test_score"].mean()),
        "test_auc_base": float(ab), "test_auc_centroid": float(ac),
        "bootstrap_mean_diff": mean_d,
        "bootstrap_ci_lower": lo, "bootstrap_ci_upper": hi,
        "clears_promotion_bar": bool(clears),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
