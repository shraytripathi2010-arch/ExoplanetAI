"""
retrain_with_new_features_full_suite.py -- corrects a real gap: the first
pass at this closing experiment (retrain_with_new_features.py) only ran
plain CV + bootstrap CI, not the full suite the user explicitly asked for
by name: "nested CV, calibration, bootstrap CI". This version adds the
two that were missing, reusing this project's own established methodology
(05b_model_analysis.py's nested_cv_tuning / calibration_check patterns)
rather than inventing a new approach.

Idea 1 (multi-transit consistency) closes as depth-only --
multi_transit_depth_chi2red -- after per-transit duration extraction was
attempted and found methodologically unsound (validated broken on both a
shallow- and deep-transit real example; per explicit user decision, not
pursued further with a more complex method).

Only the two variants worth carrying into full validation: base, and
base + all three already-validated new features (chi2red, half_ratio,
double_ratio) combined -- the single/pairwise breakdowns were already
shown negative in the first pass and don't need repeating under heavier
methodology to reach the same conclusion.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(SCRIPT_DIR, "training_with_new_features.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "retrain_new_features_full_suite_results.json")
MODELS_DIR = os.path.join(CODE_DIR, "..", "models")
PRODUCTION_METADATA_PATH = os.path.join(MODELS_DIR, "best_model_metadata.json")

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_BOOTSTRAP = 2000
N_RANDOM_SEARCH_ITER = 30

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

HGB_PARAM_DIST = {
    "clf__max_iter": [100, 200, 300, 500],
    "clf__max_depth": [3, 4, 5, 6, None],
    "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "clf__l2_regularization": [0.0, 0.1, 0.5, 1.0, 2.0],
    "clf__max_leaf_nodes": [15, 31, 63, 127],
}


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


def nested_cv_auc(X_train, y_train):
    """Same pattern as 05b_model_analysis.py's nested_cv_tuning: outer
    5-fold gives an unbiased performance estimate of 'tune then predict',
    inner 3-fold RandomizedSearchCV does the actual hyperparameter search
    -- the outer test fold never influences its own fold's tuning."""
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    outer_scores = []
    for train_idx, test_idx in outer_cv.split(X_train, y_train):
        X_out_train, y_out_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_out_test, y_out_test = X_train.iloc[test_idx], y_train.iloc[test_idx]
        search = RandomizedSearchCV(
            build_hgb_pipeline(), HGB_PARAM_DIST, n_iter=N_RANDOM_SEARCH_ITER, scoring="roc_auc",
            cv=inner_cv, random_state=RANDOM_SEED, n_jobs=-1,
        )
        search.fit(X_out_train, y_out_train)
        y_proba = search.predict_proba(X_out_test)[:, 1]
        outer_scores.append(roc_auc_score(y_out_test, y_proba))
    return float(np.mean(outer_scores)), float(np.std(outer_scores))


def calibration_check(pipeline, X_train, y_train, X_test, y_test):
    """Same pattern as 05b_model_analysis.py's calibration_check: raw vs
    sigmoid vs isotonic, Brier score is the metric that matters here (ROC-
    AUC is a monotonic-invariant ranking metric, calibration doesn't change
    it) -- reported anyway for completeness."""
    y_proba_raw = pipeline.predict_proba(X_test)[:, 1]
    brier_raw = brier_score_loss(y_test, y_proba_raw)

    results = {"raw": {"brier": float(brier_raw), "auc": float(roc_auc_score(y_test, y_proba_raw))}}
    for method in ["sigmoid", "isotonic"]:
        calibrated = CalibratedClassifierCV(pipeline, method=method, cv=5)
        calibrated.fit(X_train, y_train)
        y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
        results[method] = {"brier": float(brier_score_loss(y_test, y_proba_cal)),
                            "auc": float(roc_auc_score(y_test, y_proba_cal))}
    best_method = min(["sigmoid", "isotonic"], key=lambda m: results[m]["brier"])
    calibration_helped = results[best_method]["brier"] < brier_raw - 0.005
    results["best_method"] = best_method
    results["calibration_meaningfully_helped"] = bool(calibration_helped)
    return results


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
    print("Full-suite validation (nested CV + calibration + bootstrap CI) --")
    print("correcting the gap from the first pass at this closing experiment")
    print("=" * 70)

    df = pd.read_csv(TRAINING_CSV)
    X_base, y = build_feature_matrix(df, BASE_FEATURE_COLUMNS)
    X_new, _ = build_feature_matrix(df, BASE_FEATURE_COLUMNS + NEW_FEATURES)
    idx_train, idx_test = train_test_split(df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
    X_train_base, X_test_base = X_base.loc[idx_train], X_base.loc[idx_test]
    X_train_new, X_test_new = X_new.loc[idx_train], X_new.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    results = {}

    print("\n--- 1. Nested CV (outer 5-fold / inner 3-fold RandomizedSearchCV, 30 iters) ---")
    nested_base_mean, nested_base_std = nested_cv_auc(X_train_base, y_train)
    print(f"Base: {nested_base_mean:.4f} +/- {nested_base_std:.4f}")
    nested_new_mean, nested_new_std = nested_cv_auc(X_train_new, y_train)
    print(f"Base + new features: {nested_new_mean:.4f} +/- {nested_new_std:.4f}")
    results["nested_cv"] = {
        "base_mean": nested_base_mean, "base_std": nested_base_std,
        "new_mean": nested_new_mean, "new_std": nested_new_std,
    }

    print("\n--- 2. Held-out test + probability calibration ---")
    model_base = build_hgb_pipeline()
    model_base.fit(X_train_base, y_train)
    proba_base = model_base.predict_proba(X_test_base)[:, 1]
    auc_base = roc_auc_score(y_test, proba_base)

    model_new = build_hgb_pipeline()
    model_new.fit(X_train_new, y_train)
    proba_new = model_new.predict_proba(X_test_new)[:, 1]
    auc_new = roc_auc_score(y_test, proba_new)

    print(f"Base test ROC-AUC: {auc_base:.4f}")
    print(f"Base + new features test ROC-AUC: {auc_new:.4f}")

    cal_base = calibration_check(model_base, X_train_base, y_train, X_test_base, y_test)
    cal_new = calibration_check(model_new, X_train_new, y_train, X_test_new, y_test)
    print(f"Base calibration: raw Brier={cal_base['raw']['brier']:.4f}, "
          f"best calibrated Brier={cal_base[cal_base['best_method']]['brier']:.4f} "
          f"({cal_base['best_method']}), meaningfully helped: {cal_base['calibration_meaningfully_helped']}")
    print(f"Base+new calibration: raw Brier={cal_new['raw']['brier']:.4f}, "
          f"best calibrated Brier={cal_new[cal_new['best_method']]['brier']:.4f} "
          f"({cal_new['best_method']}), meaningfully helped: {cal_new['calibration_meaningfully_helped']}")
    results["calibration"] = {"base": cal_base, "new_features": cal_new}

    print("\n--- 3. Paired bootstrap CI (2000 resamples) ---")
    mean_diff, ci_lo, ci_hi = paired_bootstrap_auc_diff(y_test, proba_base, proba_new)
    print(f"(base+new - base) AUC diff: mean {mean_diff:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    clears_bar = ci_lo > 0
    verdict = ("clears the promotion bar" if clears_bar
               else "within noise -- does NOT clear the bar" if ci_hi >= 0
               else "measurably HURTS")
    print(f"Verdict: {verdict}")

    production_auc = None
    if os.path.exists(PRODUCTION_METADATA_PATH):
        with open(PRODUCTION_METADATA_PATH) as f:
            production_auc = json.load(f).get("test_roc_auc")

    results.update({
        "test_auc_base": float(auc_base), "test_auc_new_features": float(auc_new),
        "production_recorded_auc": production_auc,
        "bootstrap_mean_diff": mean_diff, "bootstrap_ci_lower": ci_lo, "bootstrap_ci_upper": ci_hi,
        "clears_promotion_bar": bool(clears_bar), "verdict": verdict,
    })
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    if clears_bar:
        print("Clears the promotion bar under full validation -- see results for details.")
    else:
        print("Does NOT clear the promotion bar under the FULL validation suite (nested CV, "
              "calibration, bootstrap CI) -- same conclusion as the simpler first pass, now on "
              "solid methodological ground. Combined with the eight prior experiments this "
              "session, the ~0.90 ROC-AUC ceiling for this feature family is treated as final.")
    print(f"Saved to {RESULTS_PATH}")
    print("Production model NOT touched.")


if __name__ == "__main__":
    main()
