"""flux_trend_phase.py -- PART 2 of the phase-folded flux-trend assessment.

The proposal: "variance of out-of-transit data on the candidate's period" and
"fitted baseline slope ... on the candidate's period".

PART 1 routed both, and only one residue survived:

  * The VARIANCE half is `var_oot_rms`, DEPLOYED. Its out-of-transit mask is
    already computed BY PHASE-FOLDING on the candidate's ephemeris
    (`variability_features.py`: ph = ((t-t0+0.5P) % P)/P - 0.5;
    oot = |ph| > duration/period). Beyond that, a variance is a function of the
    multiset of flux values alone, so reordering the points -- which is all a
    fold does -- cannot change it. `ft_var_unfolded` / `ft_var_folded` are
    computed here only to DEMONSTRATE that identity on real data rather than
    assert it.

  * The SLOPE half, folded at the FULL period, is not `trend_slope_ppm_day`
    (a monotonic slope in TIME over the whole sector) and it is not any stored
    ellipsoidal column. `ellipsoidal_features._harmonic_fit` fits
    [1, cos, sin, cos2, sin2] but returns only a1 = hypot(c1,s1),
    a2 = hypot(c2,s2) and the SIGNED c2 -- the signed FIRST-harmonic components
    c1 and s1 are computed and discarded.

    A least-squares line in phase over [-0.5, 0.5] is, analytically, a
    projection onto the ODD part of the fold:
        m = 12 * <phi * f>  ->  -(6/pi)*s1 + (3/pi)*s2
    i.e. it is dominated by s1, the SINE (odd) component at the FULL period.
    That is the one projection of the folded trend this project has never kept.
    It is the exact full-period analogue of `ell_c2_signed`, which was the only
    ellipsoidal quantity to come back genuinely non-redundant (|rho| 0.124).

    Physics: a real reflection / phase-curve signal is brightest at secondary
    eclipse, i.e. EVEN about mid-transit -- pure c1 < 0, s1 = 0, slope = 0.
    A non-zero phase slope therefore means the full-period modulation is NOT
    phase-locked the way an orbiting companion's would be. It is a falsification
    statistic, not an amplitude.

Everything else -- cleaning, masking, the wrong-period null control -- is
`ellipsoidal_features.py` step for step, so every correlation against a deployed
or previously-tested column is like-for-like.
"""
import os
import warnings
import importlib.util
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CAT = os.path.join(ROOT, "data", "catalogs")
OUT = os.path.join(HERE, "flux_trend_phase_features.csv")

TRAIN_RAW_DIRS = [os.path.join(ROOT, "data", "known_lightcurves"),
                  os.path.join(ROOT, "data", "known_lightcurves_negative"),
                  os.path.join(ROOT, "data", "retrain_pipeline", "raw")]
POOLS = [("main", "unknown_features.csv",
          os.path.join(ROOT, "data", "unknown_lightcurves")),
         ("widesector", "unknown_features_widesector.csv",
          os.path.join(ROOT, "data", "unknown_lightcurves_widesector"))]

SIGMA = 5
MIN_POINTS = 200
MIN_FIT_POINTS = 100
CTRL_MULT = 0.7137          # identical to the ellipsoidal null control

COLUMNS = ["ft_slope_phase", "ft_slope_ctrl", "ft_slope_over_ctrl",
           "ft_c1_signed", "ft_s1_signed", "ft_slope_snr",
           "ft_var_unfolded", "ft_var_folded", "ft_n_oot", "ft_sector_first_t"]

_ELL = None


def _ell():
    global _ELL
    if _ELL is None:
        spec = importlib.util.spec_from_file_location(
            "ellfeat", os.path.join(HERE, "ellipsoidal_features.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _ELL = m
    return _ELL


def _phase_slope(ph_cycles, f):
    """Signed OLS slope of flux on phase-in-cycles, plus its standard error.

    Phase is centred, so the slope is exactly 12*<phi*f> for a uniformly
    covered fold -- the odd projection described in the module docstring.
    """
    X = np.column_stack([np.ones_like(ph_cycles), ph_cycles])
    try:
        beta, *_ = np.linalg.lstsq(X, f, rcond=None)
        resid = f - X @ beta
        dof = max(len(f) - 2, 1)
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.pinv(X.T @ X)
        return float(beta[1]), float(np.sqrt(max(cov[1, 1], 0.0)))
    except Exception:
        return np.nan, np.nan


def _first_harmonic_signed(phi, f):
    """c1 and s1, SIGNED, from the same 5-term design ellipsoidal_features uses."""
    X = np.column_stack([np.ones_like(phi), np.cos(phi), np.sin(phi),
                         np.cos(2 * phi), np.sin(2 * phi)])
    try:
        beta, *_ = np.linalg.lstsq(X, f, rcond=None)
        return float(beta[1]), float(beta[2])
    except Exception:
        return np.nan, np.nan


def flux_trend_for_raw(raw_path, period=np.nan, t0=np.nan, duration=np.nan):
    out = {c: np.nan for c in COLUMNS}
    out["ft_status"] = "ok"
    if not raw_path or not os.path.exists(raw_path):
        out["ft_status"] = "no raw light curve"
        return out
    if not (np.isfinite(period) and np.isfinite(t0) and np.isfinite(duration)
            and period > 0 and duration > 0):
        out["ft_status"] = "no usable ephemeris"
        return out
    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        out["ft_status"] = f"read error: {type(e).__name__}"
        return out
    try:
        PRE = _ell()._vf()._pre()
        if PRE.validate_schema(df) is not None:
            out["ft_status"] = "non-standard schema"
            return out
        flux, ferr, _ = PRE.choose_flux_columns(df)
        if flux is None:
            out["ft_status"] = "no usable flux"
            return out
        t = df["time"].to_numpy(); q = df["quality"].to_numpy()
        ok = ~np.isnan(t) & ~np.isnan(flux) & ~np.isnan(ferr)
        t, flux, q = t[ok], flux[ok], q[ok]
        g = q == 0
        t, flux = t[g], flux[g]
        if len(t) < MIN_POINTS:
            out["ft_status"] = f"only {len(t)} points"
            return out
        o = np.argsort(t, kind="stable")
        t, flux = t[o], flux[o]
        from astropy.stats import sigma_clip
        m = sigma_clip(flux, sigma=SIGMA, stdfunc="mad_std", maxiters=5,
                       masked=True).mask
        t, flux = t[~m], flux[~m]
        med = np.median(flux)
        if not np.isfinite(med) or med == 0:
            out["ft_status"] = "degenerate median flux"
            return out
        f = flux / med
        out["ft_sector_first_t"] = float(t.min())

        ph = ((t - t0 + 0.5 * period) % period) / period - 0.5
        half_dur = duration / period
        keep = (np.abs(ph) > half_dur) & (np.abs(np.abs(ph) - 0.5) > half_dur)
        if keep.sum() < MIN_FIT_POINTS:
            out["ft_status"] = f"only {int(keep.sum())} usable points"
            return out
        pk, fk = ph[keep], f[keep]
        out["ft_n_oot"] = int(keep.sum())

        # --- PART 1 DEMONSTRATION: a variance cannot notice a fold -----------
        # identical point set, two orderings: time order and phase order.
        order = np.argsort(pk, kind="stable")
        out["ft_var_unfolded"] = float(1.4826 * np.median(np.abs(fk - np.median(fk))))
        ffold = fk[order]
        out["ft_var_folded"] = float(1.4826 * np.median(np.abs(ffold - np.median(ffold))))

        # --- the residue: signed odd projection at the FULL period -----------
        slope, slope_se = _phase_slope(pk, fk)
        out["ft_slope_phase"] = slope
        out["ft_slope_snr"] = float(slope / slope_se) if slope_se and slope_se > 0 else np.nan
        c1, s1 = _first_harmonic_signed(2.0 * np.pi * pk, fk)
        out["ft_c1_signed"] = c1
        out["ft_s1_signed"] = s1

        # --- NULL CONTROL: same slope at a deliberately WRONG period ---------
        pc = ((t[keep] - t0) % (period * CTRL_MULT)) / (period * CTRL_MULT) - 0.5
        slope_c, _ = _phase_slope(pc, fk)
        out["ft_slope_ctrl"] = slope_c
        if np.isfinite(slope) and np.isfinite(slope_c) and abs(slope_c) > 0:
            out["ft_slope_over_ctrl"] = float(abs(slope) / abs(slope_c))
    except Exception as e:
        out["ft_status"] = f"error: {type(e).__name__}: {e}"
    return out


def worker(job):
    host, path, per, t0, dur = job
    r = flux_trend_for_raw(path, per, t0, dur)
    r["host"] = host
    return r


def main():
    E = _ell()
    tr = pd.read_csv(TRAINING); tr["host"] = tr.host.astype(str)
    jobs = [("train", j) for j in E._jobs(tr, TRAIN_RAW_DIRS)]
    for tag, ff, d in POOLS:
        p = pd.read_csv(os.path.join(CAT, ff)); p["host"] = p.host.astype(str)
        if "status" in p.columns:
            p = p[p.status.astype(str).str.startswith("Success")]
        jobs += [(tag, j) for j in E._jobs(p, [d])]
    print(f"{len(jobs)} light curves", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for n, (setname, r) in enumerate(
                zip([s for s, _ in jobs],
                    ex.map(worker, [j for _, j in jobs], chunksize=8)), 1):
            r["_set"] = setname
            rows.append(r)
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"saved {OUT}  {len(out)} rows")
    print(out.groupby("_set").ft_status.value_counts())


if __name__ == "__main__":
    main()
