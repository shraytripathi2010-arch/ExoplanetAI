"""
retrain_with_centroid.py -- Step 4: honest retrain-and-compare for the
centroid-displacement feature, tested as an actual classifier input for
the first time (previously only ever displayed as evidence text, never
fed into the model). Same methodology as every other experiment this
session (retrain_with_augmentation.py): same train/test split as
production, same HGB hyperparameters, paired bootstrap CI on the
difference -- only promote if it clears the same bar as everything else.

Missing shift_pixels values are median-imputed via the same SimpleImputer
pipeline every other feature already uses (see leakage-check notes: raw
single-feature AUC was 0.532, barely above chance, and the missingness
itself skews slightly AWAY from positive class, not toward it -- no
red flag, so no special missing-data handling beyond the existing
convention is justified).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import roc_auc_score  # exact drop-in, ~23x faster inside bootstraps

TRAINING_WITH_CENTROID_CSV = os.path.join(SCRIPT_DIR, "training_with_centroid.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "retrain_centroid_results.json")
MODELS_DIR = os.path.join(CODE_DIR, "..", "models")
PRODUCTION_METADATA_PATH = os.path.join(MODELS_DIR, "best_model_metadata.json")

RANDOM_SEED = 42
TEST_SIZE = 0.2
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


def build_feature_matrix(df, columns):
    X = df[columns].copy()
    y = df["label"].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X["FAP"] = X["FAP"].fillna(1.0)
    return X, y


def build_hgb_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.05,
            class_weight="balanced", random_state=RANDOM_SEED,
        )),
    ])


def paired_bootstrap_auc_diff(y_test, proba_a, proba_b, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    y_arr = np.asarray(y_test)
    n = len(y_arr)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        y_b = y_arr[idx]
        if len(np.unique(y_b)) < 2:
            continue
        diffs.append(roc_auc_score(y_b, proba_b[idx]) - roc_auc_score(y_b, proba_a[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def main():
    print("=" * 70)
    print("Centroid-displacement feature: honest retrain-and-compare")
    print("=" * 70)

    df = pd.read_csv(TRAINING_WITH_CENTROID_CSV)
    print(f"Total rows: {len(df)} ({(df['label']==1).sum()} positive, {(df['label']==0).sum()} negative)")
    print(f"shift_pixels available for {df['shift_pixels'].notna().sum()}/{len(df)} rows "
          f"({100*df['shift_pixels'].notna().mean():.1f}%)")

    # SAME split for both models -- identical rows in train/test regardless
    # of feature set, so the comparison is apples-to-apples.
    X_base, y = build_feature_matrix(df, BASE_FEATURE_COLUMNS)
    X_centroid, _ = build_feature_matrix(df, CENTROID_FEATURE_COLUMNS)
    idx_train, idx_test = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    X_train_base, X_test_base = X_base.loc[idx_train], X_base.loc[idx_test]
    X_train_centroid, X_test_centroid = X_centroid.loc[idx_train], X_centroid.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    # 5-fold CV comparison (model-selection-style check, same N_CV_FOLDS
    # this project always uses) in addition to the single held-out test,
    # since a single split can be unrepresentative.
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_base = cross_validate(build_hgb_pipeline(), X_train_base, y_train, cv=cv, scoring="roc_auc")
    cv_centroid = cross_validate(build_hgb_pipeline(), X_train_centroid, y_train, cv=cv, scoring="roc_auc")
    print(f"\n5-fold CV (train split only): base = {cv_base['test_score'].mean():.4f} +/- "
          f"{cv_base['test_score'].std():.4f}, with centroid = {cv_centroid['test_score'].mean():.4f} "
          f"+/- {cv_centroid['test_score'].std():.4f}")

    model_base = build_hgb_pipeline()
    model_base.fit(X_train_base, y_train)
    proba_base = model_base.predict_proba(X_test_base)[:, 1]
    auc_base = roc_auc_score(y_test, proba_base)

    model_centroid = build_hgb_pipeline()
    model_centroid.fit(X_train_centroid, y_train)
    proba_centroid = model_centroid.predict_proba(X_test_centroid)[:, 1]
    auc_centroid = roc_auc_score(y_test, proba_centroid)

    print(f"\nHeld-out test ROC-AUC: base (no centroid) = {auc_base:.4f}, "
          f"with centroid feature = {auc_centroid:.4f}")

    production_auc = None
    if os.path.exists(PRODUCTION_METADATA_PATH):
        with open(PRODUCTION_METADATA_PATH) as f:
            production_auc = json.load(f).get("test_roc_auc")
        print(f"Production best_model.joblib recorded test ROC-AUC: {production_auc:.4f} "
              f"(sanity check vs reproduced base: diff = {abs(auc_base - production_auc):.4f})")

    mean_diff, ci_lo, ci_hi = paired_bootstrap_auc_diff(y_test, proba_base, proba_centroid)
    print(f"\nPaired bootstrap ({N_BOOTSTRAP} resamples) on (with_centroid - base) AUC difference:")
    print(f"  mean diff: {mean_diff:+.4f}   95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    clears_bar = ci_lo > 0
    if clears_bar:
        verdict = "Centroid feature measurably helps (95% CI entirely above zero) -- clears the promotion bar."
    elif ci_hi < 0:
        verdict = "Centroid feature measurably HURTS (95% CI entirely below zero)."
    else:
        verdict = ("Centroid feature is within noise (CI includes zero) -- does NOT clear the same "
                   "statistical-significance bar every other experiment in this project has had to clear.")
    print(f"\nVERDICT: {verdict}")

    results = {
        "n_total": len(df), "n_with_centroid": int(df["shift_pixels"].notna().sum()),
        "n_train": len(idx_train), "n_test": len(idx_test),
        "cv_auc_base_mean": float(cv_base["test_score"].mean()), "cv_auc_base_std": float(cv_base["test_score"].std()),
        "cv_auc_centroid_mean": float(cv_centroid["test_score"].mean()),
        "cv_auc_centroid_std": float(cv_centroid["test_score"].std()),
        "test_auc_base": float(auc_base), "test_auc_centroid": float(auc_centroid),
        "production_recorded_auc": production_auc,
        "bootstrap_mean_diff": mean_diff, "bootstrap_ci_lower": ci_lo, "bootstrap_ci_upper": ci_hi,
        "clears_promotion_bar": bool(clears_bar), "verdict": verdict,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")
    print("\nProduction model NOT touched by this script regardless of outcome "
          "-- promotion (if warranted) is a separate, explicit step.")


if __name__ == "__main__":
    main()
