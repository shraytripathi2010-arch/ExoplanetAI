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

THE LIMIT THAT MATTERS, MEASURED RATHER THAN ASSUMED
The guarantee requires the calibration data and the scored point to be
exchangeable. A domain classifier separates the calibration stars from this
app's unknown candidates at **AUC 0.9763** -- they are demonstrably not from
the same distribution. So on unknown candidates this is a well-calibrated
indication, NOT a guarantee, and `guarantee_transfers` is False so the UI can
say so instead of overclaiming.
"""
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.join(SCRIPT_DIR, "..", "models", "conformal_calibration.json")

# Measured in conformal_prediction.py; see the module docstring.
DOMAIN_AUC_VS_CANDIDATES = 0.9763
GUARANTEE_TRANSFERS = False

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

    return {
        "alpha": alpha,
        "confidence_pct": int(round((1 - alpha) * 100)),
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
        "guarantee_transfers": GUARANTEE_TRANSFERS,
        "domain_auc": DOMAIN_AUC_VS_CANDIDATES,
        # Empty sets are structurally impossible at the current thresholds,
        # because q_neg + q_pos > 1 for every alpha in the artifact. Stated
        # rather than left as a mystery branch: the honest distribution-shift
        # signal for this app is the domain AUC above, not an empty set.
        "empty_possible": any(s["q_neg"] + s["q_pos"] < 1.0 for s in sets),
    }
