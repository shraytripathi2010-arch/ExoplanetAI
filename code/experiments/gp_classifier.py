"""
gp_classifier.py -- Part C item 1: Gaussian Process classifier on the SAME
feature set/split as the production classical model, for direct, honest
comparability. Referenced in this project's literature review (Armstrong
et al. used GPs for this exact problem).

A same-family ensemble (RF+HGB+LR) was already tested and found not to
help -- this is a genuinely different model family/mechanism, so it's a
real, distinct question, not a repeat of that test.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import roc_auc_score

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                          "data", "training_dataset", "training.csv")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gp_results.json")

FEATURE_COLUMNS = [
    "SDE", "SDE_raw", "FAP", "period", "period_uncertainty", "duration", "depth",
    "depth_mean", "depth_mean_std", "depth_mean_even", "depth_mean_odd",
    "odd_even_mismatch", "rp_rs", "snr", "transit_count", "distinct_transit_count",
    "empty_transit_count",
    "st_rad", "st_teff",
    "chi2red_min", "depth_consistency_std", "secondary_eclipse_depth", "transit_shape_ratio",
    "depth_duration_ratio",
]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS].copy()
    X = X.replace([np.inf, -np.inf], np.nan)  # same handling as the production model
    X["FAP"] = X["FAP"].fillna(1.0)  # same convention as the production model
    y = df["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    # GPs scale poorly (O(n^3)) -- 5,491 rows is on the edge of practical for
    # exact GP inference. Using sklearn's GaussianProcessClassifier (Laplace
    # approximation) directly on ~4,392 training rows; if this proves too
    # slow, that itself is a real, honest scope finding worth reporting, not
    # something to force through with a subsample that changes the question.
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("gp", GaussianProcessClassifier(kernel=kernel, random_state=RANDOM_SEED, max_iter_predict=100)),
    ])

    print(f"Training GP classifier on {len(X_train)} rows, {len(FEATURE_COLUMNS)} features...")
    import time
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"Fit time: {fit_time:.1f}s")

    test_pred = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    print(f"Test ROC-AUC: {test_auc:.4f}")

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    t0 = time.time()
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_time = time.time() - t0
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f} (cv time: {cv_time:.1f}s)")

    results = {
        "test_roc_auc": float(test_auc),
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist(),
        "fit_time_s": fit_time,
        "n_train": len(X_train), "n_test": len(X_test),
        "classical_model_baseline_test_roc_auc": 0.9031559838011451,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")
    print(f"For comparison, classical model test ROC-AUC: 0.9032")


if __name__ == "__main__":
    main()
