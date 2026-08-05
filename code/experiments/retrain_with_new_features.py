"""
retrain_with_new_features.py -- the closing retrain-and-validate step for
the classical model (per explicit user framing: this is the last feature
experiment, not an open-ended search). Tests multi_transit_depth_chi2red,
power_ratio_half_period, and power_ratio_double_period -- both together
and individually -- against the same production model, same methodology
as every prior experiment this session.
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

TRAINING_CSV = os.path.join(SCRIPT_DIR, "training_with_new_features.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "retrain_new_features_results.json")
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
NEW_FEATURES = ["multi_transit_depth_chi2red", "power_ratio_half_period", "power_ratio_double_period"]


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


def evaluate_variant(name, feature_cols, X_train_base, X_test_base, y_train, y_test,
                      proba_base, auc_base, df):
    X_full, y_full = build_feature_matrix(df, feature_cols)
    X_train, X_test = X_full.loc[X_train_base.index], X_full.loc[X_test_base.index]

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_validate(build_hgb_pipeline(), X_train, y_train, cv=cv, scoring="roc_auc")

    model = build_hgb_pipeline()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    mean_diff, ci_lo, ci_hi = paired_bootstrap_auc_diff(y_test, proba_base, proba)
    clears_bar = ci_lo > 0
    print(f"\n--- {name} ---")
    print(f"5-fold CV: {cv_scores['test_score'].mean():.4f} +/- {cv_scores['test_score'].std():.4f}")
    print(f"Held-out test ROC-AUC: {auc:.4f} (base: {auc_base:.4f})")
    print(f"Paired bootstrap vs base: mean diff {mean_diff:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    verdict = ("clears the promotion bar" if clears_bar
               else "within noise -- does NOT clear the bar" if ci_hi >= 0
               else "measurably HURTS")
    print(f"Verdict: {verdict}")
    return {
        "name": name, "cv_auc_mean": float(cv_scores["test_score"].mean()),
        "cv_auc_std": float(cv_scores["test_score"].std()), "test_auc": float(auc),
        "bootstrap_mean_diff": mean_diff, "bootstrap_ci_lower": ci_lo, "bootstrap_ci_upper": ci_hi,
        "clears_promotion_bar": bool(clears_bar), "verdict": verdict,
    }


def main():
    print("=" * 70)
    print("Closing feature experiment: multi-transit consistency + frequency-domain features")
    print("=" * 70)

    df = pd.read_csv(TRAINING_CSV)
    print(f"Total rows: {len(df)} ({(df['label']==1).sum()} positive, {(df['label']==0).sum()} negative)")
    for col in NEW_FEATURES:
        print(f"{col}: available for {df[col].notna().sum()}/{len(df)} rows "
              f"({100*df[col].notna().mean():.1f}%)")

    X_base, y = build_feature_matrix(df, BASE_FEATURE_COLUMNS)
    idx_train, idx_test = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    X_train_base, X_test_base = X_base.loc[idx_train], X_base.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    model_base = build_hgb_pipeline()
    model_base.fit(X_train_base, y_train)
    proba_base = model_base.predict_proba(X_test_base)[:, 1]
    auc_base = roc_auc_score(y_test, proba_base)

    production_auc = None
    if os.path.exists(PRODUCTION_METADATA_PATH):
        with open(PRODUCTION_METADATA_PATH) as f:
            production_auc = json.load(f).get("test_roc_auc")
    print(f"\nBase (no new features) test ROC-AUC: {auc_base:.4f}"
          + (f" (production recorded: {production_auc:.4f}, sanity-check diff "
             f"{abs(auc_base-production_auc):.4f})" if production_auc else ""))

    results = []
    variants = {
        "multi_transit_depth_chi2red only": BASE_FEATURE_COLUMNS + ["multi_transit_depth_chi2red"],
        "power_ratio features only": BASE_FEATURE_COLUMNS + ["power_ratio_half_period", "power_ratio_double_period"],
        "all three new features combined": BASE_FEATURE_COLUMNS + NEW_FEATURES,
    }
    for name, cols in variants.items():
        results.append(evaluate_variant(name, cols, X_train_base, X_test_base, y_train, y_test,
                                         proba_base, auc_base, df))

    any_clears = any(r["clears_promotion_bar"] for r in results)
    print("\n" + "=" * 70)
    if any_clears:
        print("At least one variant clears the promotion bar -- see results for which one.")
    else:
        print("None of the variants clear the promotion bar. Combined with the eight prior "
              "experiments this session (real data expansion, synthetic augmentation, CNN, "
              "GP, ensemble, Kepler pilot, centroid displacement), the ~0.90 ROC-AUC ceiling "
              "for this feature family is now treated as final.")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"auc_base": float(auc_base), "production_recorded_auc": production_auc,
                    "variants": results, "any_clears_bar": bool(any_clears)}, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")
    print("Production model NOT touched regardless of outcome.")


if __name__ == "__main__":
    main()
