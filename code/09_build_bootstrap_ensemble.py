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

3. Each member is a clone of the WHOLE production model -- CalibratedClassifierCV
   with production's own cv=5 / sigmoid -- refit on a bootstrap resample.
   CHANGED 2026-10-11: members used cv=3 "to save cost", on the assumption that it
   only affected calibration stability slightly. Measured, the cv=3 refit moved
   scores by median 0.019 and up to 0.18 from production, and production fell
   outside the members' own 16-84% range for 85 of 294 candidates.

4. The members must be trained on the SAME rows as production. Pass the
   training.csv snapshot production was fit on (--training-csv; for fe3fa82f, the
   copy committed in 3329ad18). Rows appended later are assigned to train/test by
   a host hash, so today's file gives a different split. Before any member is
   built, refitting production's config with no resampling must reproduce
   production's predictions exactly, or the build stops.

5. Members are built in a staging folder and swapped in only when all of them
   succeed. The manifest records the md5 of the production model they describe
   (target_model_md5) and of the training snapshot; web/backfill_uncertainty.py
   refuses to write bands when target_model_md5 is not the deployed model.

Author: Ray's Exoplanet AI Project
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
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
REPRODUCTION_TOL = 1e-12


def _fit_one(i, X_train, y_train, production, out_dir):
    """One bootstrap member: resample the training rows with replacement and
    refit an exact clone of the production model (calibration included)."""
    rng = np.random.RandomState(1000 + i)
    idx = rng.choice(len(X_train), len(X_train), replace=True)
    Xb, yb = X_train.iloc[idx], y_train.iloc[idx]
    # A resample can, by chance, contain a single class -- rare at this class
    # balance but not impossible, and it would raise deep inside calibration.
    if yb.nunique() < 2:
        return None
    member = clone(production)
    member.fit(Xb, yb)
    path = os.path.join(out_dir, f"member_{i:03d}.joblib")
    joblib.dump(member, path, compress=3)
    return path


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-members", type=int, default=N_MEMBERS)
    ap.add_argument("--n-jobs", type=int, default=6,
                    help="parallel fits; each member is single-threaded via OMP_NUM_THREADS=1")
    ap.add_argument("--training-csv", required=True,
                    help="the training.csv snapshot the production model was fit on")
    args = ap.parse_args()

    with open(PRODUCTION_PATH, "rb") as f:
        blob = f.read()
    target_md5 = hashlib.md5(blob).hexdigest()
    production = joblib.load(io.BytesIO(blob))
    if not isinstance(production, CalibratedClassifierCV):
        raise SystemExit(f"production model is a {type(production).__name__}, not a CalibratedClassifierCV")
    print(f"Target: production model {target_md5} -- members clone it whole "
          f"(cv={production.cv}, method={production.method})")

    df = pd.read_csv(args.training_csv)
    X, y = train_models.build_feature_matrix(df)
    train_mask, test_mask = train_models.split_by_host(df)
    X_train, y_train = X[train_mask], y[train_mask]

    # GATE: same rows, same recipe -> same model. If this fails, the snapshot or
    # the split is not the one production was trained on, and every band would
    # describe a different model.
    t0 = time.time()
    refit = clone(production).fit(X_train, y_train)
    worst = float(np.max(np.abs(refit.predict_proba(X)[:, 1] - production.predict_proba(X)[:, 1])))
    print(f"Reproduction gate: refit on {len(X_train)} training rows vs production over all "
          f"{len(X)} rows: max |diff| {worst:.2e} ({time.time() - t0:.0f}s)")
    if worst > REPRODUCTION_TOL:
        raise SystemExit("STOPPING: the refit does not reproduce production -- wrong snapshot or split. "
                         "Nothing was built.")

    stage = os.path.join(ENSEMBLE_DIR, f".build-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    os.makedirs(stage)
    print(f"Fitting {args.n_members} members on the TRAINING SPLIT ONLY "
          f"({len(X_train)} stars); the {int(test_mask.sum())} held-out test stars are untouched.")
    from joblib import Parallel, delayed
    t0 = time.time()
    paths = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(_fit_one)(i, X_train, y_train, production, stage) for i in range(args.n_members))
    paths = [p for p in paths if p]
    if len(paths) != args.n_members:
        raise SystemExit(f"STOPPING: only {len(paths)} of {args.n_members} members were built; "
                         f"the deployed ensemble is untouched (staging left at {stage}).")

    manifest = {
        "n_members": len(paths),
        "built_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "target_model_md5": target_md5,
        "calibration_cv": production.cv,
        "calibration_method": production.method,
        "n_training_rows": int(len(X_train)),
        "training_csv_md5": _md5(args.training_csv),
        "reproduction_max_abs_diff": worst,
        "trained_on": "frozen training split only (split_manifest.json); test stars never seen",
        "feature_columns": list(X.columns),
        "purpose": "spread of production-recipe models refit on bootstrap resamples of the training data",
        "refresh_policy": "rebuild whenever the production model is replaced; backfill_uncertainty.py "
                          "refuses to write bands when target_model_md5 is not the deployed model",
    }
    # Swap: the old members go (they are backed up outside the repo by the
    # operator), the new ones move in, and the manifest is written LAST so a
    # reader never sees a manifest describing members that are not there yet.
    for f in os.listdir(ENSEMBLE_DIR):
        if f.startswith("member_") or f == "ensemble_manifest.json":
            os.remove(os.path.join(ENSEMBLE_DIR, f))
    for p in paths:
        shutil.move(p, os.path.join(ENSEMBLE_DIR, os.path.basename(p)))
    os.rmdir(stage)
    with open(MANIFEST_PATH + ".tmp", "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(MANIFEST_PATH + ".tmp", MANIFEST_PATH)
    print(f"\nBuilt {len(paths)} members in {(time.time()-t0)/60:.1f} min -> {ENSEMBLE_DIR}")
    print(f"Total size: {sum(os.path.getsize(os.path.join(ENSEMBLE_DIR, os.path.basename(p))) for p in paths)/1e6:.1f} MB")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
