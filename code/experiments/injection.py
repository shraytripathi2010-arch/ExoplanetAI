"""
injection.py -- synthetic transit injection into REAL processed light curves.

Part B of the multi-data-source/model-architecture effort. Injects physically
realistic transit signals (via batman, already installed -- no new dependency)
into real, already-detrended negative-class light curves (data/processed_negative),
so the injected signal sits in genuine TESS noise/systematics, not synthetic
noise. Parameters (period, depth, duration) are drawn by empirical resampling
from the REAL positive-class training data (data/training_dataset/training.csv),
not an assumed parametric distribution -- matches how this project has
consistently preferred real data over convenient assumptions.

Two injectors:
  - inject_transit(): a physically realistic single/multi-transit planet signal.
  - inject_eclipsing_binary(): an EB-like signal (V-shaped via high impact
    parameter, plus a secondary eclipse) as a synthetic NEGATIVE example --
    this is a real astrophysical false-positive shape, not a random negative.

Every synthetic example this module produces must carry is_synthetic=True
through to any downstream training/reporting -- this is enforced by
returning it explicitly in the result dict, not left to the caller to infer.
"""
import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROCESSED_NEGATIVE_DIR = os.path.join(PROJECT_ROOT, "data", "processed_negative")
TRAINING_CSV = os.path.join(PROJECT_ROOT, "data", "training_dataset", "training.csv")

AU_IN_R_SUN = 215.032
R_SUN_IN_R_EARTH = 109.076

# Typical quadratic limb-darkening coefficients for a Sun-like star (TESS
# bandpass) -- a simplification (real coefficients vary with Teff/logg), but
# reasonable for injection-recovery purposes since TLS's own detection
# statistic is not strongly sensitive to the exact LD law, only to the
# overall transit shape being physically plausible. Documented here rather
# than silently assumed.
DEFAULT_LIMB_DARKENING = [0.3, 0.2]


def _load_real_param_distributions():
    """Empirical (real-data) distributions to sample injection parameters
    from, rather than an assumed parametric form. Cached at module level
    since this is read many times during a completeness-curve run."""
    df = pd.read_csv(TRAINING_CSV)
    pos = df[df["label"] == 1]
    periods = pos["period"].dropna()
    periods = periods[(periods > 0) & (periods < 14)].to_numpy()
    depths_ppm = ((1 - pos["depth"]).dropna() * 1e6)
    depths_ppm = depths_ppm[(depths_ppm > 50) & (depths_ppm < 50000)].to_numpy()
    durations = pos["duration"].dropna()
    durations = durations[(durations > 0) & (durations < 0.5)].to_numpy()
    return periods, depths_ppm, durations


_REAL_PERIODS, _REAL_DEPTHS_PPM, _REAL_DURATIONS = None, None, None


def sample_real_params(rng):
    """Draws one (period_days, depth_ppm, duration_days) triple by resampling
    from the real positive-class training data's own empirical distribution."""
    global _REAL_PERIODS, _REAL_DEPTHS_PPM, _REAL_DURATIONS
    if _REAL_PERIODS is None:
        _REAL_PERIODS, _REAL_DEPTHS_PPM, _REAL_DURATIONS = _load_real_param_distributions()
    period = float(rng.choice(_REAL_PERIODS))
    depth_ppm = float(rng.choice(_REAL_DEPTHS_PPM))
    duration = float(rng.choice(_REAL_DURATIONS))
    return period, depth_ppm, duration


def list_real_negative_lightcurves():
    return sorted(f for f in os.listdir(PROCESSED_NEGATIVE_DIR) if f.endswith(".csv"))


def load_real_lightcurve(fname):
    df = pd.read_csv(os.path.join(PROCESSED_NEGATIVE_DIR, fname))
    return df["time"].to_numpy(), df["flux"].to_numpy(), df["flux_err"].to_numpy()


def _a_over_rstar(period_days, duration_days, rp_rs, impact_param=0.3):
    """Derives the scaled semi-major axis a/R* directly from the transit
    duration, period, and Rp/R* -- avoids needing a separate, possibly-wrong
    stellar-mass assumption. Standard circular-orbit transit geometry
    (Winn 2010 eq. 14, solved for a/R*). Falls back to a minimum sane value
    if the algebra would otherwise go negative/undefined for an
    unphysical parameter combination (can happen with resampled reals
    combined arbitrarily) -- clamped, not silently NaN'd."""
    phase_frac = duration_days / period_days
    sin_arg = np.sin(np.pi * phase_frac)
    inner = ((1 + rp_rs) ** 2 - impact_param ** 2)
    if inner <= 0 or sin_arg <= 0:
        return 20.0  # a reasonable generic fallback, rarely hit
    a_rs = np.sqrt(inner) / sin_arg
    return float(np.clip(a_rs, 2.0, 500.0))


def inject_transit(time, flux, period_days, depth_ppm, duration_days, rng,
                    t0=None, impact_param=None):
    """Injects one physically realistic transiting-planet signal (via batman)
    into a real flux array. Returns (injected_flux, params_dict)."""
    import batman

    depth_frac = depth_ppm / 1e6
    rp_rs = float(np.sqrt(np.clip(depth_frac, 1e-8, 0.9)))
    if impact_param is None:
        impact_param = float(rng.uniform(0.0, 0.6))  # favor non-grazing, realistic geometry
    a_rs = _a_over_rstar(period_days, duration_days, rp_rs, impact_param)
    inc = float(np.degrees(np.arccos(np.clip(impact_param / a_rs, -1, 1))))

    if t0 is None:
        t0 = float(rng.uniform(time.min(), time.min() + period_days))

    params = batman.TransitParams()
    params.t0 = t0
    params.per = period_days
    params.rp = rp_rs
    params.a = a_rs
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.limb_dark = "quadratic"
    params.u = DEFAULT_LIMB_DARKENING

    model = batman.TransitModel(params, time)
    model_flux = model.light_curve(params)
    injected_flux = flux * model_flux  # multiplicative: preserves real noise's relative scale

    return injected_flux, {
        "period_days": period_days, "depth_ppm": depth_ppm, "duration_days": duration_days,
        "rp_rs": rp_rs, "a_rs": a_rs, "inc": inc, "t0": t0, "impact_param": impact_param,
        "is_synthetic": True, "synthetic_kind": "transit",
    }


def inject_eclipsing_binary(time, flux, period_days, depth_ppm, duration_days, rng, t0=None):
    """Injects an EB-like signal as a synthetic NEGATIVE example: a
    grazing/high-impact-parameter primary (naturally V-shaped, unlike a
    planet's flatter-bottomed transit at low impact parameter) plus a
    secondary eclipse at half depth -- a real, common TESS false-positive
    shape, not an arbitrary negative. depth_ppm here is deliberately allowed
    to reach EB-scale depths (parts-per-thousand), unlike the planet
    injector's realistic-planet range."""
    import batman

    impact_param = float(rng.uniform(0.85, 0.98))  # grazing -> V-shaped
    depth_frac = depth_ppm / 1e6
    rp_rs = float(np.sqrt(np.clip(depth_frac, 1e-8, 0.98)))
    a_rs = _a_over_rstar(period_days, duration_days, rp_rs, impact_param)
    inc = float(np.degrees(np.arccos(np.clip(impact_param / a_rs, -1, 1))))
    if t0 is None:
        t0 = float(rng.uniform(time.min(), time.min() + period_days))

    params = batman.TransitParams()
    params.t0 = t0
    params.per = period_days
    params.rp = rp_rs
    params.a = a_rs
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.limb_dark = "quadratic"
    params.u = DEFAULT_LIMB_DARKENING

    primary_model = batman.TransitModel(params, time)
    primary_flux = primary_model.light_curve(params)

    # Secondary eclipse: same geometry, half the depth (asymmetric primary/
    # secondary, a real EB signature TLS's odd/even and secondary-depth
    # features are specifically designed to catch), centered at phase 0.5.
    sec_params = batman.TransitParams()
    for k in ("per", "a", "inc", "ecc", "w", "limb_dark", "u"):
        setattr(sec_params, k, getattr(params, k))
    sec_params.t0 = t0 + period_days / 2.0
    sec_params.rp = rp_rs * 0.6  # shallower secondary -> asymmetric depths, an EB signature
    sec_model = batman.TransitModel(sec_params, time)
    secondary_flux = sec_model.light_curve(sec_params)

    injected_flux = flux * primary_flux * secondary_flux

    return injected_flux, {
        "period_days": period_days, "depth_ppm": depth_ppm, "duration_days": duration_days,
        "rp_rs": rp_rs, "a_rs": a_rs, "inc": inc, "t0": t0, "impact_param": impact_param,
        "is_synthetic": True, "synthetic_kind": "eclipsing_binary",
    }
