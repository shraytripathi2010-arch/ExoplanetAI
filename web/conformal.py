"""conformal.py -- split conformal prediction sets for the web app.

WHAT THIS ADDS, AND WHAT IT DOES NOT

It does not change any prediction. The model's probability is unchanged and
still shown. This converts that probability into a SET of labels that is still
plausible at a stated confidence level, so "0.62" becomes the more honest
"could be either" rather than a number a reader will round to "probably a
planet".

CALIBRATION vs CONFORMAL -- related, different, both needed
  CalibratedClassifierCV (already in production) makes probabilities mean what
  they say ON AVERAGE: of all candidates scored 0.7, about 70% are planets. It
  says nothing about any single candidate, and it carries no guarantee -- it is
  a fitted transformation that can be wrong.
  Conformal prediction adds a FINITE-SAMPLE, DISTRIBUTION-FREE guarantee about
  a per-prediction OUTPUT SET: over exchangeable data, the true label is in the
  set at least (1-alpha) of the time, for any model, any sample size, with no
  assumption that the probabilities were any good.

METHOD: Mondrian (class-conditional) LAC. Plain marginal LAC was measured first
and rejected: it hits its overall target while covering the negative class only
62.8% of the time at the nominal 90% level, because the calibration set is 79%
planets. Mondrian uses a separate threshold per class and delivers 90.4% /
91.3%. See RESULTS_SUMMARY.

THE LIMITS THAT MATTER, MEASURED RATHER THAN ASSUMED
The guarantee requires the calibration data and the scored point to be
exchangeable. Both measured limits are read from models/conformal_diagnostics.json
(code/experiments/conformal_diagnostics.py), never hard-coded here:
  * domain_shift -- how well a classifier separates the calibration stars from
    the candidate pools (AUC 0.9820 for fe3fa82f on 2026-10-11; it was a
    hard-coded 0.9763, measured with the old 24 features). Above 0.90 the
    guarantee does not transfer to candidates, and the page says so.
  * fresh_label_coverage -- coverage of the deployed thresholds on labelled
    stars added after the calibration snapshot, where planets were
    under-covered (0.831 at 90%).
The page calls the figures stale when the diagnostics' model_md5 is not the
deployed model.
"""
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.join(SCRIPT_DIR, "..", "models", "conformal_calibration.json")
DIAGNOSTICS = os.path.join(SCRIPT_DIR, "..", "models", "conformal_diagnostics.json")
PRODUCTION_MODEL = os.path.join(SCRIPT_DIR, "..", "models", "best_model.joblib")
TRANSFER_AUC_CEILING = 0.90      # conformal_prediction.py's own verdict line

_ART = None


def _load():
    """Parse the calibration artifact once. Never raises -- a missing artifact
    degrades to 'unavailable' rather than breaking a candidate page."""
    global _ART
    if _ART is not None:
        return _ART
    try:
        with open(ARTIFACT) as f:
            _ART = json.load(f)
    except Exception:
        _ART = {}
    return _ART


def available():
    return bool(_load().get("thresholds"))


_md5_cache = {}


def _production_md5():
    """md5 of the deployed model, recomputed only when the file changes."""
    import hashlib
    try:
        st = os.stat(PRODUCTION_MODEL)
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    if key not in _md5_cache:
        with open(PRODUCTION_MODEL, "rb") as f:
            _md5_cache.clear()
            _md5_cache[key] = hashlib.md5(f.read()).hexdigest()
    return _md5_cache[key]


def _diagnostics():
    """The measured limits, or {} if the artifact is missing. Re-read on each
    call (it is tiny) so a regenerated file is picked up without a restart."""
    try:
        with open(DIAGNOSTICS) as f:
            return json.load(f)
    except Exception:
        return {}


def prediction_set(p, alpha=0.05):
    """Conformal set for one probability, at one confidence level.

    Returns a dict shaped for direct template use. The four-valued output is
    the point: a raw probability cannot say "I don't know", and a set can.
    """
    art = _load()
    t = (art.get("thresholds") or {}).get(str(alpha))
    if t is None or p is None:
        return None
    q_neg, q_pos = t["q_neg"], t["q_pos"]
    inc_neg = p <= q_neg              # "could this be a false positive?"
    inc_pos = (1.0 - p) <= q_pos      # "could this be a planet?"

    if inc_pos and not inc_neg:
        key, label = "planet", "Planet"
        plain = ("At this confidence level the only status still consistent "
                 "with the model's evidence is PLANET CANDIDATE.")
    elif inc_neg and not inc_pos:
        key, label = "not_planet", "Not a planet"
        plain = ("At this confidence level the only status still consistent "
                 "with the model's evidence is NOT A PLANET.")
    elif inc_pos and inc_neg:
        key, label = "ambiguous", "Planet or not a planet"
        plain = ("Both statuses remain consistent with the evidence at this "
                 "confidence level. The model cannot separate them for this "
                 "star -- this is a genuine 'not sure', not a weak yes.")
    else:
        key, label = "empty", "Neither status fits"
        plain = ("Neither status fits the calibration data at this confidence "
                 "level. That usually means this star looks unlike anything "
                 "the model was calibrated on.")

    # The surprise case: {not planet} for a probability above 0.5. Thresholds
    # are per class, fitted on a calibration set that is ~79% planets, so at
    # this level "planet" survives only once the score reaches 1 - q_pos.
    planet_floor = 1.0 - q_pos
    why = None
    if key == "not_planet" and p > 0.5:
        why = (f"Why \u201cnot a planet\u201d at a score above 0.5: thresholds are set separately "
               f"for each class, and at {int(round((1 - alpha) * 100))}% a score must reach "
               f"{planet_floor:.3f} before \u201cplanet\u201d stays in the set.")
    return {
        "alpha": alpha,
        "confidence_pct": int(round((1 - alpha) * 100)),
        "planet_floor": planet_floor,
        "why": why,
        "set_key": key,
        "set_label": label,
        "set_notation": {"planet": "{Planet}", "not_planet": "{Not a planet}",
                         "ambiguous": "{Planet, Not a planet}",
                         "empty": "{ }"}[key],
        "is_ambiguous": key == "ambiguous",
        "is_empty": key == "empty",
        "plain_language": plain,
        "q_neg": q_neg, "q_pos": q_pos,
    }


def summary(p, alphas=(0.10, 0.05, 0.01)):
    """Everything a candidate page needs, including the honesty flags.

    Returns None when the artifact is missing or the candidate has no
    probability, so the template can simply skip the panel.
    """
    if p is None or not available():
        return None
    art = _load()
    sets = [s for s in (prediction_set(p, a) for a in alphas) if s]
    if not sets:
        return None
    return {
        "probability": p,
        "sets": sets,
        "n_calibration": art.get("n_calibration"),
        "n_calibration_pos": art.get("n_calibration_pos"),
        "n_calibration_neg": art.get("n_calibration_neg"),
        "method": art.get("method"),
        "model_md5": art.get("model_md5"),
        "thresholds_stale": art.get("model_md5") != _production_md5(),
        **_diagnostics_view(alphas),
        # Empty sets are structurally impossible at the current thresholds,
        # because q_neg + q_pos > 1 for every alpha in the artifact. Stated
        # rather than left as a mystery branch: the honest distribution-shift
        # signal for this app is the domain AUC above, not an empty set.
        "empty_possible": any(s["q_neg"] + s["q_pos"] < 1.0 for s in sets),
    }


def _diagnostics_view(alphas):
    """The measured limits for the template, from conformal_diagnostics.json."""
    d = _diagnostics()
    if not d:
        return {"diagnostics_available": False, "guarantee_transfers": False}
    auc = (d.get("domain_shift") or {}).get("domain_auc")
    fresh = d.get("fresh_label_coverage") or {}
    rows = []
    for a in alphas:
        v = (fresh.get("by_alpha") or {}).get(str(a))
        if v:
            tgt = v.get("target", 1 - a)
            rows.append({"confidence_pct": int(round((1 - a) * 100)), "planet": v.get("planet"),
                         "not_planet": v.get("not_planet"),
                         "planet_below": v.get("planet") is not None and v["planet"] < tgt,
                         "not_planet_below": v.get("not_planet") is not None and v["not_planet"] < tgt})
    return {
        "diagnostics_available": True,
        "domain_auc": auc,
        "domain_n_features": d.get("n_features"),
        "domain_measured_at": d.get("generated_at"),
        "diagnostics_model_md5": d.get("model_md5"),
        "diagnostics_stale": d.get("model_md5") != _production_md5(),
        "guarantee_transfers": auc is not None and auc <= TRANSFER_AUC_CEILING,
        "guarantee_scope": d.get("guarantee_scope"),
        "fresh_coverage": rows,
        "fresh_n_planet": fresh.get("n_planet"),
        "fresh_n_not_planet": fresh.get("n_not_planet"),
    }
