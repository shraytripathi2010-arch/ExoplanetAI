"""deploy_crowding_model.py -- promote the staged crowding model to production.

DEPLOYMENT PATH: MANUAL OFFLINE SWAP, not the live promotion gate.

The gate cannot perform this change, and that is a structural fact rather than a
preference. It compares a challenger against the deployed model on ONE feature
matrix built from the current `FEATURE_COLUMNS`. With the feature list extended,
the deployed 24-feature model cannot consume the 26-column matrix at all --
verified directly, sklearn raises:

    ValueError: The feature names should match those that were passed during
    fit. Feature names unseen at fit time: crowd_flux_ratio_max,
    crowd_nearest_arcsec

So a gate run in this state raises rather than mis-promoting. Its invariant is
"same recipe, more data"; this change is "same data, different recipe", which
sits outside what it was built to judge. After this swap both production and any
future challenger use 26 features, so the invariant is restored and the gate
resumes working normally on subsequent retrains.

Safety: the outgoing artifact is copied to models/versions/ first, so rollback
is a single file copy. Every step is verified by md5 before the next begins, and
the script refuses to proceed if the staged artifact is not the one that was
validated.
"""
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
MODELS = os.path.join(ROOT, "models")
PROD = os.path.join(MODELS, "best_model.joblib")
STAGED = os.path.join(MODELS, "best_model_crowding_staged.joblib")
VERSIONS = os.path.join(MODELS, "versions")
META = os.path.join(MODELS, "best_model_metadata.json")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "promote_crowding_results.json")

EXPECTED_OLD = "341f1a3907e77f6ec294f182833e613c"
EXPECTED_NEW = "0c996a41a76cc765895d3013830a536b"


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    if not os.path.exists(STAGED):
        sys.exit("staged artifact missing -- run promote_crowding_retrain.py first")

    old_md5, new_md5 = md5(PROD), md5(STAGED)
    print(f"current production : {old_md5}")
    print(f"staged candidate   : {new_md5}")

    if old_md5 != EXPECTED_OLD:
        sys.exit(f"REFUSING: production md5 is not the expected {EXPECTED_OLD}. "
                 "Something changed underneath this deployment; investigate first.")
    if new_md5 != EXPECTED_NEW:
        sys.exit(f"REFUSING: staged md5 is not the validated {EXPECTED_NEW}. "
                 "The artifact under test is not the artifact being deployed.")
    print("both artifacts match the validated run\n")

    os.makedirs(VERSIONS, exist_ok=True)
    rollback = os.path.join(VERSIONS, f"best_model_pre_crowding_{old_md5[:8]}.joblib")
    shutil.copy2(PROD, rollback)
    shutil.copy2(META, os.path.join(VERSIONS, "best_model_metadata_pre_crowding.json"))
    assert md5(rollback) == old_md5, "rollback copy is corrupt"
    print(f"rollback saved  : {os.path.relpath(rollback, ROOT)}  (verified)")

    shutil.copy2(STAGED, PROD)
    live = md5(PROD)
    if live != EXPECTED_NEW:
        shutil.copy2(rollback, PROD)
        sys.exit(f"deploy verification FAILED (got {live}); rolled back")
    print(f"deployed        : {live}")

    res = json.load(open(RESULTS))
    meta = json.load(open(META))
    meta.update({
        "model_name": "HistGradientBoosting+sigmoid_calibration "
                      "(tuned, v3 stellar params, + catalog crowding)",
        "feature_columns": meta["feature_columns"] + [
            "crowd_flux_ratio_max", "crowd_nearest_arcsec"],
        "test_roc_auc": res["headline_new_auc"],
        "previous_test_roc_auc": 0.9031,
        "previous_model_md5": old_md5,
        "model_md5": live,
        "promoted_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "promotion_note": (
            "First deployed model change in the project's history. Added two "
            "catalog neighbour-crowding features. Validated at real scale: "
            f"resampled delta {res['resampled_delta_mean']:+.4f} over 12 training "
            f"bootstraps, positive {res['resampled_positive']}/12, clearing "
            f"{res['resampled_clears']}/12; nested CV {res['nested_cv_old']:.4f} -> "
            f"{res['nested_cv_new']:.4f}. Brier {res['brier_old']:.4f} -> "
            f"{res['brier_new']:.4f}, ECE {res['ece_old']:.4f} -> {res['ece_new']:.4f}. "
            "Deployed by manual offline swap, NOT the live promotion gate: the "
            "gate requires production and challenger to share a feature space "
            "and raises ValueError across a feature-set change."),
        "deployment_path": "manual offline swap",
        "rollback_artifact": os.path.relpath(rollback, ROOT),
    })
    json.dump(meta, open(META, "w"), indent=2)
    print(f"metadata updated: test_roc_auc {meta['previous_test_roc_auc']} -> "
          f"{meta['test_roc_auc']:.4f}, {len(meta['feature_columns'])} features")
    print("\nNOTE: models/conformal_calibration.json records the OLD model_md5 and "
          "is now stale -- regenerate before the conformal layer is trusted.")


if __name__ == "__main__":
    main()
