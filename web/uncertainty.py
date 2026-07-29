"""
uncertainty.py -- per-candidate uncertainty from the pre-built bootstrap ensemble.

The ensemble is built ONCE by code/09_build_bootstrap_ensemble.py and reused
for every scoring call from here. Nothing in this module ever fits a model, so
a candidate lookup costs one load (cached for the process lifetime) plus N
cheap predict_proba calls.

WHAT IS AND IS NOT BEING REPORTED
---------------------------------
The point estimate stays the PRODUCTION model's probability, unchanged. That
is the number validated at test ROC-AUC 0.9032, and swapping in the bootstrap
mean would silently restate every score in the project against a figure that
was never validated. The ensemble contributes the BAND around it, nothing else.

The band is epistemic uncertainty from training-set resampling: how much this
candidate's probability moves when the model is refit on a different draw of
the same training data. It does not describe error in the input features, and
a narrow band is a statement about model stability, not about being right.

`bootstrap_mean` is stored alongside so the two can be compared. If the
production probability sits far outside the ensemble's own spread, that is
worth knowing rather than hiding -- see `disagreement_sigma`.
"""
import json
import os
import threading

import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
ENSEMBLE_DIR = os.path.join(MODELS_DIR, "bootstrap_ensemble")
MANIFEST_PATH = os.path.join(ENSEMBLE_DIR, "ensemble_manifest.json")

_members = None
_manifest = None
_load_lock = threading.Lock()


def ensemble_available():
    return os.path.exists(MANIFEST_PATH)


def _load():
    """Loads every member once per process. Guarded by a lock because the
    Flask app is threaded and two concurrent requests could otherwise both
    start loading the ensemble."""
    global _members, _manifest
    if _members is not None:
        return _members, _manifest
    with _load_lock:
        if _members is not None:
            return _members, _manifest
        if not ensemble_available():
            _members, _manifest = [], None
            return _members, _manifest
        import joblib
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        paths = sorted(p for p in os.listdir(ENSEMBLE_DIR) if p.startswith("member_"))
        members = [joblib.load(os.path.join(ENSEMBLE_DIR, p)) for p in paths]
        _members, _manifest = members, manifest
        return _members, _manifest


def predict_with_uncertainty(X, production_probability=None):
    """X: a DataFrame of one or more rows with the model's feature columns.

    Returns a list of dicts (one per row). Percentile bounds come from the
    member distribution directly rather than mean +/- k*std, so an asymmetric
    spread -- which is normal near 0 or 1, where probabilities are bounded --
    is reported as it actually is instead of being forced symmetric.
    """
    members, manifest = _load()
    if not members:
        return [None] * len(X)

    # (n_members, n_rows)
    P = np.vstack([m.predict_proba(X)[:, 1] for m in members])
    out = []
    for j in range(P.shape[1]):
        col = P[:, j]
        std = float(np.std(col, ddof=1))
        rec = {
            "bootstrap_mean": float(np.mean(col)),
            "uncertainty_std": std,
            "uncertainty_p16": float(np.percentile(col, 16)),
            "uncertainty_p84": float(np.percentile(col, 84)),
            "uncertainty_n_members": int(len(members)),
        }
        if production_probability is not None:
            prod = float(np.asarray(production_probability).ravel()[j])
            # How far the deployed model's answer sits from the ensemble's own
            # centre, in units of the ensemble spread. Large values mean the
            # band should NOT be read as a confidence interval around the
            # displayed number -- they disagree about where the answer is.
            rec["disagreement_sigma"] = (
                float(abs(prod - rec["bootstrap_mean"]) / std) if std > 0 else 0.0)
        out.append(rec)
    return out


def format_band(probability, std):
    if probability is None or std is None:
        return None
    return f"{probability:.3f} +/- {std:.3f}"
