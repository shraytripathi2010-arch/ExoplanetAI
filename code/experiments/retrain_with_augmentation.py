"""
retrain_with_augmentation.py -- Part B's honest retrain-and-compare step,
run once augment_classical_dataset.py's full-TLS-search synthetic rows exist.

Trains the SAME HistGradientBoosting model (same hyperparameters, same
FEATURE_COLUMNS, same random_state) 05_train_models.py trains, twice:
  1. "real-only": on the exact same train split production's best_model
     was trained on -- a sanity check this reproduces production's own
     test ROC-AUC before trusting any comparison.
  2. "real+synthetic": on that same real training split PLUS the augmented
     rows (label carried through from injection.py: 1=transit, 0=EB).

Both are evaluated ONLY on the real held-out test set (synthetic rows never
enter the test set) -- same honest-comparison rule already used for the
CNN (real-only vs real+synthetic) and the GP/ensemble experiments this
session. A paired bootstrap (same resampled test-set indices for both
models each iteration) reports the CI on the AUC difference.

Per the project's standing rule: best_model.joblib is NOT overwritten
unless real+synthetic beats real-only by a margin whose 95% CI is entirely
above zero -- the same bar every other experiment this session had to
clear, and none did until now (if this one does, that's the headline).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import roc_auc_score  # exact drop-in, ~23x faster inside bootstraps

TRAINING_CSV = os.path.join(CODE_DIR, "..", "data", "training_dataset", "training.csv")
AUGMENTED_CSV = os.path.join(SCRIPT_DIR, "augmented_classical_dataset.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "retrain_augmentation_results.json")
MODELS_DIR = os.path.join(CODE_DIR, "..", "models")
PRODUCTION_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
PRODUCTION_METADATA_PATH = os.path.join(MODELS_DIR, "best_model_metadata.json")

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_BOOTSTRAP = 2000

FEATURE_COLUMNS = [
    "SDE", "SDE_raw", "FAP", "period", "period_uncertainty", "duration", "depth",
    "depth_mean", "depth_mean_std", "depth_mean_even", "depth_mean_odd",
    "odd_even_mismatch", "rp_rs", "snr", "transit_count", "distinct_transit_count",
    "empty_transit_count",
    "st_rad", "st_teff",
    "chi2red_min", "depth_consistency_std", "secondary_eclipse_depth", "transit_shape_ratio",
    "depth_duration_ratio",
]


def build_feature_matrix(df):
    X = df[FEATURE_COLUMNS].copy()
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
    """Same resampled indices used for both models each iteration (paired),
    so the CI is on the DIFFERENCE, not two independent noisy estimates --
    matches the GP-vs-classical comparison already done this session."""
    rng = np.random.RandomState(seed)
    y_test_arr = np.asarray(y_test)
    n = len(y_test_arr)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        y_b = y_test_arr[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_a = roc_auc_score(y_b, proba_a[idx])
        auc_b = roc_auc_score(y_b, proba_b[idx])
        diffs.append(auc_b - auc_a)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float(diffs.mean()), "ci_lower_2.5": float(lo), "ci_upper_97.5": float(hi)}


def main():
    print("=" * 70)
    print("Part B: honest retrain-and-compare -- real-only vs real+synthetic")
    print("=" * 70)

    if not os.path.exists(AUGMENTED_CSV):
        print(f"ERROR: {AUGMENTED_CSV} not found -- run augment_classical_dataset.py first.")
        return

    real_df = pd.read_csv(TRAINING_CSV)
    aug_df_raw = pd.read_csv(AUGMENTED_CSV)
    aug_df = aug_df_raw[aug_df_raw["status"] == "Success"].copy()
    print(f"Real training rows: {len(real_df)} ({(real_df['label']==1).sum()} positive, "
          f"{(real_df['label']==0).sum()} negative)")
    print(f"Synthetic rows available: {len(aug_df)}/{len(aug_df_raw)} attempted "
          f"({(aug_df['label']==1).sum()} positive/transit, {(aug_df['label']==0).sum()} negative/EB) "
          f"-- rest failed the same all-features-finite bar real candidates must clear.")
    if len(aug_df) < 20:
        print("\nTOO FEW usable synthetic rows to draw any honest conclusion from -- stopping "
              "rather than reporting a number that would just be noise. Let the augmentation "
              "run continue and re-run this script once more rows exist.")
        return

    X_real, y_real = build_feature_matrix(real_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real, test_size=TEST_SIZE, stratify=y_real, random_state=RANDOM_SEED
    )

    X_aug, y_aug = build_feature_matrix(aug_df)
    X_train_plus = pd.concat([X_train, X_aug], ignore_index=True)
    y_train_plus = pd.concat([y_train, y_aug], ignore_index=True)

    print(f"\nTrain set: {len(X_train)} real rows -> {len(X_train_plus)} real+synthetic rows "
          f"({len(X_aug)} synthetic added)")
    print(f"Test set: {len(X_test)} real rows (held out, synthetic-free, identical across both models)")

    model_real_only = build_hgb_pipeline()
    model_real_only.fit(X_train, y_train)
    proba_real_only = model_real_only.predict_proba(X_test)[:, 1]
    auc_real_only = roc_auc_score(y_test, proba_real_only)

    model_augmented = build_hgb_pipeline()
    model_augmented.fit(X_train_plus, y_train_plus)
    proba_augmented = model_augmented.predict_proba(X_test)[:, 1]
    auc_augmented = roc_auc_score(y_test, proba_augmented)

    print(f"\nreal-only (reproduced) test ROC-AUC:     {auc_real_only:.4f}")
    print(f"real+synthetic test ROC-AUC:              {auc_augmented:.4f}")

    production_auc = None
    if os.path.exists(PRODUCTION_METADATA_PATH):
        with open(PRODUCTION_METADATA_PATH) as f:
            production_meta = json.load(f)
        production_auc = production_meta.get("test_roc_auc")
        print(f"Production best_model.joblib's recorded test ROC-AUC: {production_auc:.4f} "
              f"(sanity check: real-only reproduction should match closely -- "
              f"diff = {abs(auc_real_only - production_auc):.4f})")

    ci = paired_bootstrap_auc_diff(y_test, proba_real_only, proba_augmented)
    print(f"\nPaired bootstrap ({N_BOOTSTRAP} resamples) on (augmented - real-only) AUC difference:")
    print(f"  mean diff: {ci['mean_diff']:+.4f}   95% CI: [{ci['ci_lower_2.5']:+.4f}, {ci['ci_upper_97.5']:+.4f}]")

    beats_production = ci["ci_lower_2.5"] > 0
    if beats_production:
        verdict = ("real+synthetic measurably beats real-only by more than noise "
                   "(95% CI entirely above zero) -- meets this project's bar to promote a new model.")
    else:
        verdict = ("does NOT measurably beat real-only (CI includes zero or is negative) -- "
                   "production model is NOT touched, matching the standard used for every "
                   "other experiment this session (GP, ensemble, CNN all failed the same bar).")
    print(f"\nVERDICT: {verdict}")

    results = {
        "n_real_train": len(X_train), "n_synthetic_added": len(X_aug), "n_real_test": len(X_test),
        "auc_real_only": float(auc_real_only), "auc_augmented": float(auc_augmented),
        "production_recorded_auc": production_auc,
        "bootstrap_ci_on_diff": ci, "beats_production_bar": bool(beats_production),
        "verdict": verdict,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")

    if beats_production:
        print("\nPromoting: saving the real+synthetic model as the new best_model.joblib "
              "(previous production model backed up first).")
        import joblib
        import shutil
        backup_path = os.path.join(MODELS_DIR, "best_model_pre_augmentation_backup.joblib")
        if os.path.exists(PRODUCTION_MODEL_PATH) and not os.path.exists(backup_path):
            shutil.copy(PRODUCTION_MODEL_PATH, backup_path)
        joblib.dump(model_augmented, PRODUCTION_MODEL_PATH)
        new_metadata = {
            "model_name": "HistGradientBoosting_real_plus_synthetic",
            "feature_columns": FEATURE_COLUMNS,
            "test_roc_auc": float(auc_augmented),
            "random_seed": RANDOM_SEED,
            "training_rows": len(X_train_plus),
            "test_rows": len(X_test),
            "n_synthetic_rows_added": len(X_aug),
            "promoted_over_real_only_ci": ci,
        }
        with open(PRODUCTION_METADATA_PATH, "w") as f:
            json.dump(new_metadata, f, indent=2)
        print("Done -- best_model.joblib and best_model_metadata.json updated.")
    else:
        print("\nProduction model left untouched (models/best_model.joblib unchanged).")


if __name__ == "__main__":
    main()
