"""
09_build_bootstrap_ensemble.py

Pre-computes a bootstrap ensemble used ONLY to attach an uncertainty band to
each candidate's probability. Run once; the saved members are then reused for
every scoring call, so no candidate lookup ever retrains anything.

WHY BOOTSTRAP RATHER THAN ENSEMBLE VARIANCE
-------------------------------------------
Two candidate approaches were considered:

  * Ensemble variance across model versions -- NOT POSSIBLE here. The version
    history is empty: models/versions/ has zero files and model_versions has
    zero rows, because nothing has ever cleared the promotion gate. There is
    exactly one model in this project, so there is no version spread to
    measure. (This may change now that the promotion gate's own test-set
    contamination is fixed, but it is not available today.)

  * Bootstrap resampling -- what this script does. The production classifier
    is a HistGradientBoosting pipeline wrapped in CalibratedClassifierCV. It
    is not a Bayesian model and exposes no native predictive variance the way
    a Gaussian Process would, so the only way to get a real uncertainty for an
    individual prediction is to perturb what the model was trained on and see
    how much that prediction moves.

WHAT THE BAND ACTUALLY MEANS
----------------------------
Epistemic uncertainty from training-set sampling: "if we had drawn a different
training set of the same size from the same population, how much would this
candidate's probability change?" It does NOT capture error in the input
features themselves (a noisy TLS depth), and it does NOT capture whether the
labels are right. A tight band means the model is stable here, not that the
answer is correct.

TWO DELIBERATE CHOICES, BOTH LOAD-BEARING
-----------------------------------------
1. Members are fit ONLY on the frozen training split (data/training_dataset/
   split_manifest.json). The held-out test stars are never touched by any
   member, so the band can be validated on the test set without circularity,
   and the frozen split stays clean.

2. Members replicate the PRODUCTION estimator's actual tuned hyperparameters,
   cloned from best_model.joblib itself -- NOT 05_train_models.build_models(),
   which returns the untuned defaults (max_iter=300, max_depth=4) rather than
   what is actually deployed (max_iter=500, max_leaf_nodes=63,
   l2_regularization=0.5). Bootstrapping a different architecture than the one
   in production would produce a band that describes a model nobody uses.

The one concession to cost: the internal calibration uses cv=3 instead of
production's cv=5. That makes each member ~40% cheaper and affects the
calibration curve's own stability slightly, not the resample spread this is
measuring.

Author: Ray's Exoplanet AI Project
"""
import argparse
import importlib.util
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "train_models", os.path.join(SCRIPT_DIR, "05_train_models.py"))
train_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_models)

MODELS_DIR = train_models.MODELS_DIR
ENSEMBLE_DIR = os.path.join(MODELS_DIR, "bootstrap_ensemble")
PRODUCTION_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
MANIFEST_PATH = os.path.join(ENSEMBLE_DIR, "ensemble_manifest.json")

N_MEMBERS = 32
CALIBRATION_CV = 3


def _fit_one(i, X_train, y_train, estimator):
    """One bootstrap member: resample the training rows with replacement,
    refit the production architecture on that resample."""
    rng = np.random.RandomState(1000 + i)
    idx = rng.choice(len(X_train), len(X_train), replace=True)
    Xb, yb = X_train.iloc[idx], y_train.iloc[idx]
    # A resample can, by chance, contain a single class -- rare at this class
    # balance but not impossible, and it would raise deep inside calibration.
    if yb.nunique() < 2:
        return None
    member = CalibratedClassifierCV(clone(estimator), cv=CALIBRATION_CV, method="sigmoid")
    member.fit(Xb, yb)
    path = os.path.join(ENSEMBLE_DIR, f"member_{i:03d}.joblib")
    joblib.dump(member, path, compress=3)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-members", type=int, default=N_MEMBERS)
    ap.add_argument("--n-jobs", type=int, default=6,
                    help="parallel fits; each member is single-threaded via OMP_NUM_THREADS=1")
    args = ap.parse_args()

    os.makedirs(ENSEMBLE_DIR, exist_ok=True)

    production = joblib.load(PRODUCTION_PATH)
    estimator = production.estimator if hasattr(production, "estimator") else production
    print(f"Cloning the deployed estimator: {estimator}")

    df = train_models.load_and_report_class_balance()
    X, y = train_models.build_feature_matrix(df)
    train_mask, test_mask = train_models.split_by_host(df)
    X_train, y_train = X[train_mask], y[train_mask]
    print(f"Fitting {args.n_members} members on the TRAINING SPLIT ONLY "
          f"({len(X_train)} stars); the {int(test_mask.sum())} held-out test stars are untouched.")

    from joblib import Parallel, delayed
    t0 = time.time()
    paths = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(_fit_one)(i, X_train, y_train, estimator) for i in range(args.n_members))
    paths = [p for p in paths if p]

    manifest = {
        "n_members": len(paths),
        "built_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "calibration_cv": CALIBRATION_CV,
        "n_training_rows": int(len(X_train)),
        "trained_on": "frozen training split only (split_manifest.json); test stars never seen",
        "production_model_sha_size": os.path.getsize(PRODUCTION_PATH),
        "feature_columns": list(X.columns),
        "purpose": "epistemic uncertainty (training-resample variance) for individual predictions",
        "refresh_policy": "rebuild whenever the production model is replaced, on the same "
                          "cadence as normal retraining -- a stale ensemble describes a model "
                          "that is no longer deployed",
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nBuilt {len(paths)} members in {(time.time()-t0)/60:.1f} min -> {ENSEMBLE_DIR}")
    print(f"Total size: {sum(os.path.getsize(p) for p in paths)/1e6:.1f} MB")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
