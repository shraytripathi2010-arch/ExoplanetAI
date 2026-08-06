"""deploy_variability_model.py -- promote the staged 31-feature model.

DEPLOYMENT PATH: MANUAL OFFLINE SWAP, not the live promotion gate -- the same
structural reason as the crowding promotion. The gate compares a challenger
against the deployed model on ONE feature matrix built from the current
`FEATURE_COLUMNS`. Across a feature-set change the deployed model cannot
consume the new matrix at all; sklearn raises

    ValueError: The feature names should match those that were passed during
    fit. Feature names unseen at fit time: var_excess, var_ls_amp, ...

so a gate run in this state raises rather than mis-promoting. Its invariant is
"same recipe, more data"; this is "same data, different recipe". After the swap
both production and any future challenger use 31 features and the gate resumes
working normally.

Safety, identical to the crowding deployment:
  * refuses unless the outgoing artifact is exactly the expected production md5
  * refuses unless the staged artifact is exactly the one that was validated
  * copies the outgoing artifact and metadata to models/versions/ FIRST and
    verifies the copy by md5, so rollback is one file copy
  * re-verifies after the swap and rolls back automatically on mismatch
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
STAGED = os.path.join(MODELS, "best_model_variability_staged.joblib")
VERSIONS = os.path.join(MODELS, "versions")
META = os.path.join(MODELS, "best_model_metadata.json")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "promote_variability_results.json")

EXPECTED_OLD = "0c996a41a76cc765895d3013830a536b"
VAR5 = ["var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    if not os.path.exists(STAGED):
        sys.exit("staged artifact missing -- run promote_variability_retrain.py first")
    res = json.load(open(RESULTS))
    expected_new = res["staged_md5"]

    old_md5, new_md5 = md5(PROD), md5(STAGED)
    print(f"current production : {old_md5}")
    print(f"staged candidate   : {new_md5}")
    print(f"validated staged   : {expected_new}")

    if old_md5 != EXPECTED_OLD:
        sys.exit(f"REFUSING: production md5 is not the expected {EXPECTED_OLD}. "
                 "Something changed underneath this deployment; investigate first.")
    if new_md5 != expected_new:
        sys.exit("REFUSING: staged md5 differs from the validated artifact. "
                 "The artifact under test is not the artifact being deployed.")
    print("both artifacts match the validated run\n")

    os.makedirs(VERSIONS, exist_ok=True)
    rollback = os.path.join(VERSIONS, f"best_model_pre_variability_{old_md5[:8]}.joblib")
    shutil.copy2(PROD, rollback)
    shutil.copy2(META, os.path.join(VERSIONS, "best_model_metadata_pre_variability.json"))
    assert md5(rollback) == old_md5, "rollback copy is corrupt"
    print(f"rollback saved  : {os.path.relpath(rollback, ROOT)}  (verified)")

    shutil.copy2(STAGED, PROD)
    live = md5(PROD)
    if live != expected_new:
        shutil.copy2(rollback, PROD)
        sys.exit(f"deploy verification FAILED (got {live}); rolled back")
    print(f"deployed        : {live}")

    meta = json.load(open(META))
    old_auc = meta["test_roc_auc"]
    meta.update({
        "model_name": "HistGradientBoosting+sigmoid_calibration "
                      "(tuned, v3 stellar params, + catalog crowding, "
                      "+ stellar variability)",
        "feature_columns": meta["feature_columns"] + VAR5,
        "test_roc_auc": res["headline_new_auc"],
        "test_brier_score": res["brier_new"],
        "previous_test_roc_auc": old_auc,
        "previous_model_md5": old_md5,
        "model_md5": live,
        "promoted_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "promotion_note": (
            "Second deployed model change. Added five stellar variability / "
            "activity features computed from the RAW (pre-flatten) light "
            f"curve. Frozen test AUC {old_auc:.4f} -> {res['headline_new_auc']:.4f}. "
            f"Resampled over 12 training bootstraps: {res['resampled_delta_mean']:+.4f} "
            f"(sd {res['resampled_delta_sd']:.4f}), positive "
            f"{res['resampled_positive']}/12, clearing {res['resampled_clears']}/12, "
            f"at/above MDE {res['resampled_at_mde']}/12; 2-min subset "
            f"{res['resampled_delta_2min']:+.4f}. Nested CV pooled out-of-fold "
            f"{res['nested_cv_old']:.4f} -> {res['nested_cv_new']:.4f} "
            f"(delta {res['nested_cv_delta']:+.4f}, CI [{res['nested_cv_ci_lo']:+.4f}, "
            f"{res['nested_cv_ci_hi']:+.4f}]). Brier "
            f"{res['resampled_brier_old']:.4f} -> {res['resampled_brier_new']:.4f}, "
            f"ECE {res['resampled_ece_old']:.4f} -> {res['resampled_ece_new']:.4f}. "
            "Controls passed: sky held constant +0.0098 (12/12), missingness-only "
            "indicator +0.0000 (0/12), and a combined worst-case arm holding sky, "
            "availability and population all constant at once +0.0087 (12/12 "
            "positive, 9/12 clearing). Deployed by manual offline swap, NOT the "
            "live promotion gate, which raises ValueError across a feature-set "
            "change."),
        "deployment_path": "manual offline swap",
        "rollback_artifact": os.path.relpath(rollback, ROOT),
        "raw_lightcurve_dependency": (
            "var_* features are computed from RAW pre-flatten light curves "
            "(data/known_lightcurves*, data/unknown_lightcurves*, "
            "data/retrain_pipeline/raw), NOT data/processed*. 02_preprocess.py "
            "savgol-flattens over ~13.4 h, which removes the multi-day "
            "rotational signal these features measure. See "
            "code/experiments/variability_features.py."),
    })
    json.dump(meta, open(META, "w"), indent=2)
    print(f"metadata updated: test_roc_auc {old_auc:.4f} -> {meta['test_roc_auc']:.4f}, "
          f"{len(meta['feature_columns'])} features")
    print("\nNEXT: regenerate models/conformal_calibration.json (records model_md5) "
          "and restart the app.")


if __name__ == "__main__":
    main()
