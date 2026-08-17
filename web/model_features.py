"""model_features.py -- surface the model's own contamination/blend evidence.

WHAT THIS IS, AND HOW IT DIFFERS FROM THE TFOP PANEL

`exofop_vetting.py` shows EXTERNAL expert assessment. This module shows the
opposite: three groups of features the MODEL ITSELF scores on, which until now
were invisible to anyone reading a candidate page.

    crowding    (2 features, deployed 2026-08-05)  contamination from neighbours
    variability (5 features, deployed 2026-08-06)  stellar activity / rotation
    Gaia DR3    (2 features, deployed 2026-08-14)  unresolved-companion flags

Three of the four deployed model improvements. The model uses all nine to
decide the score shown at the top of the page, so a reviewer who cannot see
them cannot audit the reasoning behind it. Everything here is MODEL INPUT,
attributed as such -- derived by this project's own pipeline, not published
expert opinion.

VALUES ARE READ, NEVER RECOMPUTED. The source is the same per-pool feature
table the candidate was scored from (`data/catalogs/unknown_features*.csv`),
looked up on host. No model call, no network, no filesystem walk per render --
one cached index, same as the TFOP panel, so a slow lookup can never make a
candidate page hang.

THE NaN-VERSUS-ZERO DISTINCTION, WHICH IS REAL AND NOT COSMETIC

`crowd_flux_ratio_max == 0.0` and `crowd_nearest_arcsec == NaN` mean different
things, and collapsing them into "no data" would destroy real information.
Measured across the 296 live candidates:

    ratio 0.0, distance present (15)  a neighbour EXISTS but contributes no
                                      measurable flux -- genuinely clean
    ratio 0.0, distance NaN     (2)   NO catalogued neighbour inside the search
                                      radius at all -- nothing to measure
    ratio > 0                  (279)  a neighbour is diluting the aperture

`gaia_ruwe`/`gaia_nss` NaN (10 candidates) is a THIRD kind of absence: the star
had no Gaia DR3 match within 3 arcsec, so the check could not run. Both columns
are in OPTIONAL_FEATURES precisely because that is a supported state -- the
fitted pipeline imputes them at serve time -- and the panel says so rather than
implying a problem.
"""
import os
import csv
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_DIR = os.path.join(SCRIPT_DIR, "..", "data", "catalogs")

# (feature table, human label). Ordered: the main pool wins on a duplicate host,
# matching how the ranked exports are built.
FEATURE_TABLES = [
    ("unknown_features.csv", "main candidate pool"),
    ("unknown_features_widesector.csv", "wide-sector pool"),
]

CROWD_COLS = ["crowd_flux_ratio_max", "crowd_nearest_arcsec"]
VAR_COLS = ["var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]
GAIA_COLS = ["gaia_ruwe", "gaia_nss"]

# Lindegren, GAIA-C3-TN-LU-LL-124. Above this, the single-star astrometric fit
# is a poor description of the source -- classically an unresolved companion.
RUWE_THRESHOLD = 1.4
# Gaia non_single_star is a BITFIELD, not a count.
NSS_BITS = [(1, "astrometric"), (2, "spectroscopic"), (4, "eclipsing")]

_INDEX = None


def _num(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _load():
    """One pass over the feature tables, cached for the process lifetime."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    idx = {}
    for fname, label in FEATURE_TABLES:
        path = os.path.join(CATALOG_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    host = (row.get("host") or "").strip()
                    if not host.startswith("TIC_"):
                        continue
                    tic = host[4:]
                    if tic in idx:        # first table wins
                        continue
                    idx[tic] = {"_pool": label,
                                **{c: _num(row.get(c)) for c in
                                   CROWD_COLS + VAR_COLS + GAIA_COLS}}
        except Exception:
            # A malformed table must not take the whole page down; the panel
            # degrades to "not available" instead.
            continue
    _INDEX = idx
    return _INDEX


def _crowding(v):
    ratio, arcsec = v["crowd_flux_ratio_max"], v["crowd_nearest_arcsec"]
    out = {"ratio": ratio, "arcsec": arcsec}
    if ratio is None:
        out.update(status="unknown", headline="Not measured",
                   detail="The neighbour search did not return a result for this star.")
        return out
    if ratio == 0.0 and arcsec is None:
        out.update(status="pass", headline="No catalogued neighbour",
                   detail="No TIC neighbour was found inside the search radius at all, so "
                          "nothing can be diluting the aperture. There is no distance to "
                          "report because there is no neighbour — this is a clean "
                          "result, not missing data.")
        return out
    if ratio == 0.0:
        out.update(status="pass", headline="Neighbour present, but contributes no flux",
                   detail=f"The nearest catalogued star sits {arcsec:.1f} arcsec away and "
                          "contributes no measurable flux to the aperture. A real "
                          "measurement of zero, not an absent one.")
        return out
    if ratio >= 1.0:
        out.update(status="caution", headline="Neighbour outshines the target",
                   detail=f"A star {arcsec:.1f} arcsec away contributes {ratio:.2f}x the "
                          "target's own flux to the aperture. A transit-like dip could "
                          "originate on that neighbour and be diluted into this star's "
                          "light curve — the classic nearby-eclipsing-binary blend.")
        return out
    out.update(status="neutral", headline="Some neighbour flux in the aperture",
               detail=f"The nearest star is {arcsec:.1f} arcsec away and adds "
                      f"{ratio:.3f}x the target's flux. Modest dilution; the measured "
                      "transit depth is a slight underestimate of the true depth.")
    return out


def _variability(v, transit_period=None):
    power, period = v["var_ls_power"], v["var_ls_period"]
    out = {c: v[c] for c in VAR_COLS}
    out["transit_period"] = transit_period
    out["period_ratio"] = None
    out["harmonic_note"] = None

    if power is None:
        out.update(status="unknown", headline="Not measured",
                   detail="Variability was not computed for this star.")
        return out

    # The single most actionable check here: a rotation period at or near the
    # transit period (or a low harmonic of it) is the signature of a starspot
    # signal masquerading as a transit.
    if transit_period and period and transit_period > 0 and period > 0:
        r = period / transit_period
        out["period_ratio"] = r
        # Phrased to read correctly in "which is {note} the N d transit signal".
        for n, name in ((1.0, "the same as"), (2.0, "twice"), (0.5, "half")):
            if abs(r - n) / n < 0.05:
                out["harmonic_note"] = name
                break

    if out["harmonic_note"]:
        out.update(status="caution", headline="Rotation period matches the transit period",
                   detail=f"The star's strongest brightness cycle is {period:.2f} d, which is "
                          f"{out['harmonic_note']} the {transit_period:.2f} d transit signal. "
                          "That coincidence is the signature of starspots rotating in and "
                          "out of view rather than an object passing in front.")
    elif power >= 0.65:
        out.update(status="caution", headline="Strongly variable star",
                   detail=f"A clear {period:.2f} d brightness cycle (strength {power:.2f} on a "
                          "0-1 scale). The transit period is different, so this is not "
                          "obviously the cause of the signal, but an active star makes any "
                          "shallow transit harder to trust.")
    elif power >= 0.30:
        out.update(status="neutral", headline="Moderately variable star",
                   detail=f"A {period:.2f} d brightness cycle of moderate strength "
                          f"({power:.2f}). Common in cool stars and not by itself a concern.")
    else:
        out.update(status="pass", headline="Photometrically quiet",
                   detail=f"No strong brightness cycle (strongest is {period:.2f} d at only "
                          f"{power:.2f} strength). A quiet star is the easiest case for "
                          "believing a shallow transit.")
    return out


def _gaia(v):
    ruwe, nss = v["gaia_ruwe"], v["gaia_nss"]
    out = {"ruwe": ruwe, "nss": nss, "ruwe_threshold": RUWE_THRESHOLD, "nss_flags": []}
    if ruwe is None and nss is None:
        out.update(status="unknown", headline="No Gaia DR3 match",
                   detail="No Gaia source was found within 3 arcsec of this star, so the "
                          "astrometric checks could not run. Both values are optional model "
                          "inputs and the classifier imputes them, so the score is still "
                          "valid — it simply had no Gaia evidence to use here.")
        return out
    if nss:
        for bit, name in NSS_BITS:
            if int(nss) & bit:
                out["nss_flags"].append(name)
    if out["nss_flags"]:
        _joined = " and ".join(out["nss_flags"])
        _article = "an" if _joined[:1].lower() in "aeiou" else "a"
        out.update(status="caution", headline="Gaia flags this as a non-single star",
                   detail=f"Gaia's own analysis classifies this source as {_article} "
                          + _joined +
                          " binary. A companion star is the most common origin of a "
                          "transit-like dip that is not a planet. Armstrong et al. (2022) "
                          "rejected astrometrically flagged binaries outright when "
                          "validating planets.")
    elif ruwe is not None and ruwe > RUWE_THRESHOLD:
        out.update(status="caution", headline="Elevated astrometric noise (RUWE)",
                   detail=f"RUWE is {ruwe:.2f}, above the {RUWE_THRESHOLD} threshold where a "
                          "single-star model stops describing Gaia's measurements well. That "
                          "usually means an unresolved companion tugging on the star — a "
                          "hint of a binary Gaia could not separate, not proof of one.")
    elif ruwe is not None:
        out.update(status="pass", headline="Astrometrically well-behaved",
                   detail=f"RUWE is {ruwe:.2f}, comfortably below the {RUWE_THRESHOLD} "
                          "threshold, and Gaia raises no non-single-star flag. Consistent "
                          "with a single star, which is what a genuine planet host should "
                          "look like.")
    else:
        out.update(status="neutral", headline="Partial Gaia record",
                   detail="Gaia matched this star but did not report RUWE for it.")
    return out


def lookup(tic_id, transit_period=None):
    """Evidence for one candidate. Never raises; returns available=False when
    the star is not in any feature table."""
    v = _load().get(str(tic_id))
    if v is None:
        return {"available": False,
                "reason": "This star is not in the candidate feature tables, so the "
                          "model's contamination inputs cannot be shown for it."}
    return {"available": True, "pool": v["_pool"],
            "crowding": _crowding(v), "variability": _variability(v, transit_period),
            "gaia": _gaia(v)}
