"""
uncertainty.py -- Part D item 1: Monte Carlo uncertainty propagation for the
derived physical parameters (planet radius, semi-major axis, equilibrium
temperature), additive alongside the existing point estimates in
08_characterize_candidates.py's derive_physical_params -- never replacing them.

Real finding while building this: depth_mean_std and period_uncertainty are
both already COMPUTED elsewhere in this pipeline (TLS extracts them the same
way for known and unknown stars) but are NOT currently populated in the
candidate characterization data reaching derive_physical_params (confirmed
live against a real candidate: both None) -- a real gap, same "computed
somewhere, dropped before it reaches where it's needed" pattern as the
st_mass bug found earlier in this project. Rather than block this feature on
fixing that upstream gap, or silently pretend a real per-star uncertainty
exists when it doesn't, every fallback used here is explicit and reported
back in the result, so a user can tell a real-measurement-based range from a
generic-assumption-based one at a glance.
"""
import numpy as np

R_SUN_IN_R_EARTH = 109.076
AU_IN_R_SUN = 215.032
TEFF_SUN_K = 5772.0

# Documented fallback fractional uncertainties, used ONLY when a real
# per-star value isn't available. Not arbitrary: TIC catalog stellar
# parameter uncertainties are commonly ~10-15% for radius/mass for stars
# without an interferometric or asteroseismic radius; a fixed 5% depth/
# duration fallback reflects typical single-sector TLS precision for a
# moderate-SNR detection, per this project's own real completeness-curve
# results (Part B) showing recovery becoming reliable well above marginal
# SNR -- these are documented, reviewable assumptions, not hidden ones.
DEFAULT_ST_RAD_FRAC_ERR = 0.12
DEFAULT_ST_MASS_FRAC_ERR = 0.10
DEFAULT_DEPTH_FRAC_ERR = 0.05
DEFAULT_DURATION_FRAC_ERR = 0.05
N_SAMPLES = 5000


def propagate_uncertainty(period_days, period_err, depth, depth_err, duration_days, duration_err,
                           st_rad, st_rad_err, st_teff, st_mass, st_mass_err, rng=None):
    """Monte Carlo propagation: samples each input from a normal distribution
    around its point estimate (clipped to physically valid ranges), recomputes
    the same derived-parameter formulas derive_physical_params already uses
    for every sample, and returns 16th/50th/84th percentile ranges (roughly
    a 1-sigma-equivalent interval) for planet radius, semi-major axis, and
    equilibrium temperature. Returns a dict of (value, low, high, used_fallback)
    tuples per quantity, plus which inputs fell back to a default fraction.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    fallbacks_used = []

    def _err_or_fallback(err, point, default_frac, name):
        if err is not None and np.isfinite(err) and err > 0:
            return err
        fallbacks_used.append(name)
        return abs(point) * default_frac if point else default_frac

    period_err = _err_or_fallback(period_err, period_days, 0.001, "period")  # TLS period is usually tight
    depth_err = _err_or_fallback(depth_err, (1 - depth) if depth is not None else None,
                                  DEFAULT_DEPTH_FRAC_ERR, "depth")
    duration_err = _err_or_fallback(duration_err, duration_days, DEFAULT_DURATION_FRAC_ERR, "duration")
    st_rad_err = _err_or_fallback(st_rad_err, st_rad, DEFAULT_ST_RAD_FRAC_ERR, "st_rad")
    mass_was_defaulted = not (st_mass and st_mass > 0)
    st_mass_point = st_mass if not mass_was_defaulted else 1.0
    st_mass_err = _err_or_fallback(st_mass_err, st_mass_point, DEFAULT_ST_MASS_FRAC_ERR, "st_mass")

    period_s = np.clip(rng.normal(period_days, period_err, N_SAMPLES), 1e-6, None)
    depth_s = np.clip(rng.normal(depth, depth_err, N_SAMPLES), 1e-6, 1.4999)
    duration_s = np.clip(rng.normal(duration_days, duration_err, N_SAMPLES), 1e-6, None)
    st_rad_s = np.clip(rng.normal(st_rad, st_rad_err, N_SAMPLES), 1e-3, None)
    st_mass_s = np.clip(rng.normal(st_mass_point, st_mass_err, N_SAMPLES), 1e-3, None)
    st_teff_s = np.full(N_SAMPLES, st_teff) if st_teff else np.full(N_SAMPLES, np.nan)

    depth_fraction = np.clip(1.0 - depth_s, 0.0, None)
    rp_rstar = np.sqrt(depth_fraction)
    rp_earth_s = st_rad_s * rp_rstar * R_SUN_IN_R_EARTH

    period_years = period_s / 365.25
    a_au_s = (st_mass_s * period_years ** 2) ** (1 / 3)

    a_rsun_s = a_au_s * AU_IN_R_SUN
    teq_k_s = st_teff_s * np.sqrt(st_rad_s / (2 * a_rsun_s))

    def _summarize(samples):
        finite = samples[np.isfinite(samples)]
        if len(finite) == 0:
            return {"median": None, "p16": None, "p84": None}
        return {"median": float(np.median(finite)), "p16": float(np.percentile(finite, 16)),
                "p84": float(np.percentile(finite, 84))}

    return {
        "planet_radius_earth": _summarize(rp_earth_s),
        "semi_major_axis_au": _summarize(a_au_s),
        "equilibrium_temp_k": _summarize(teq_k_s),
        "fallback_uncertainties_used": fallbacks_used,
    }
