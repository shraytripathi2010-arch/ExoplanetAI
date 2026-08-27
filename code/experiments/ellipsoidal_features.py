"""ellipsoidal_features.py -- PART 2 of the ellipsoidal-variation assessment.

Ellipsoidal variation is tidal distortion of a star by a close companion. The
distorted star presents more projected surface area at quadrature than at
conjunction, so the light curve carries TWO maxima per orbit -- a sinusoid at
HALF the orbital period, phase-locked to the transit ephemeris.

That is a targeted measurement at a KNOWN frequency, which is what makes it
different from everything already in this project:

  * the closed `trend_slope_ppm_day` / `trend_amp_frac` are a median-of-halves
    LINEAR slope over the whole sector -- monotonic, aperiodic, not phase-locked
  * `var_ls_*` report whatever periodicity is STRONGEST (argmax), which is the
    star's rotation, at no particular relation to the transit ephemeris
  * `ls_period_match` compares that argmax peak's period to harmonics of the
    transit period -- a detect-then-compare, which only ever looks at P_tr/2 when
    the star's dominant peak happens to already be there (2.59% of training rows)

The model fitted here is the standard BEER decomposition, on out-of-transit flux
phased to the candidate's own ephemeris:

    f(phi) = 1 + c1*cos(phi) + s1*sin(phi) + c2*cos(2*phi) + s2*sin(2*phi)

  a1 = hypot(c1, s1)   first harmonic, at P     -- reflection / Doppler beaming
  a2 = hypot(c2, s2)   second harmonic, at P/2  -- ELLIPSOIDAL
  c2 signed            ellipsoidal has MINIMA at both conjunctions, so a real
                       ellipsoidal signal has c2 < 0. The sign is a free
                       falsification test the amplitude alone does not give.

BOTH conjunctions are masked (+/- one duration at phase 0 AND phase 0.5). Two
reasons, and the second is the important one:
  1. the transit must not inflate its own vetting statistic
  2. an eclipsing binary's SECONDARY ECLIPSE sits at phase 0.5, and folding at
     P/2 maps phase 0.5 exactly onto phase 0 -- so without that mask this
     feature would be a secondary-eclipse detector, i.e. a restatement of the
     DEPLOYED `secondary_eclipse_depth`. Masking both conjunctions is what makes
     it specifically ellipsoidal. They map to the same place in the P/2 fold, so
     one contiguous gap is removed and the quadrature maxima are fully retained.

Two internal-validity controls, both requested and both necessary:
  * the same statistic evaluated at the FULL period (a1) -- ellipsoidal should
    concentrate in a2, not a1
  * the same statistic evaluated at a deliberately WRONG, incommensurate period
    (P * 0.7137) -- this is the null. If a2 at the true ephemeris does not beat
    a2 at the control period, the feature is measuring noise, not physics.

Cleaning is `variability_features.py`'s, step for step, so every correlation
against a deployed feature is like-for-like. Reads raw light curves only.
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
OUT = os.path.join(HERE, "ellipsoidal_features.csv")

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
N_PHASE_BINS = 25
CTRL_MULT = 0.7137          # incommensurate with 1, 1/2, 1/3, 2, 3

COLUMNS = ["ell_a1", "ell_a2", "ell_c2_signed", "ell_a2_snr", "ell_a2_frac",
           "ell_a2_ctrl", "ell_a2_snr_ctrl", "ell_a2_over_ctrl",
           "ell_pp_half", "ell_pp_full", "ell_pp_ratio", "ell_phase_cov"]

_VF = None


def _vf():
    global _VF
    if _VF is None:
        spec = importlib.util.spec_from_file_location(
            "varfeat", os.path.join(HERE, "variability_features.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _VF = m
    return _VF


def _harmonic_fit(phi, f):
    """Least squares on [1, cos, sin, cos2, sin2]. Returns a1, a2, c2, sigma."""
    X = np.column_stack([np.ones_like(phi), np.cos(phi), np.sin(phi),
                         np.cos(2 * phi), np.sin(2 * phi)])
    try:
        beta, *_ = np.linalg.lstsq(X, f, rcond=None)
        resid = f - X @ beta
        dof = max(len(f) - X.shape[1], 1)
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.pinv(X.T @ X)
        a1 = float(np.hypot(beta[1], beta[2]))
        a2 = float(np.hypot(beta[3], beta[4]))
        sig = float(np.sqrt(max(0.5 * (cov[3, 3] + cov[4, 4]), 0.0)))
        return a1, a2, float(beta[3]), sig
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def _folded_pp(ph01, f, nbins=N_PHASE_BINS):
    """Peak-to-peak of the binned folded curve; robust (bin medians)."""
    idx = np.clip((ph01 * nbins).astype(int), 0, nbins - 1)
    g = pd.Series(f).groupby(idx).median()
    if len(g) < nbins // 2:
        return np.nan, float(len(g)) / nbins
    return float(g.max() - g.min()), float(len(g)) / nbins


def ellipsoidal_for_raw(raw_path, period=np.nan, t0=np.nan, duration=np.nan):
    out = {c: np.nan for c in COLUMNS}
    out["ell_status"] = "ok"
    if not raw_path or not os.path.exists(raw_path):
        out["ell_status"] = "no raw light curve"
        return out
    if not (np.isfinite(period) and np.isfinite(t0) and np.isfinite(duration)
            and period > 0 and duration > 0):
        out["ell_status"] = "no usable ephemeris"
        return out
    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        out["ell_status"] = f"read error: {type(e).__name__}"
        return out
    try:
        PRE = _vf()._pre()
        if PRE.validate_schema(df) is not None:
            out["ell_status"] = "non-standard schema"
            return out
        flux, ferr, _ = PRE.choose_flux_columns(df)
        if flux is None:
            out["ell_status"] = "no usable flux"
            return out
        t = df["time"].to_numpy(); q = df["quality"].to_numpy()
        ok = ~np.isnan(t) & ~np.isnan(flux) & ~np.isnan(ferr)
        t, flux, ferr, q = t[ok], flux[ok], ferr[ok], q[ok]
        g = q == 0
        t, flux = t[g], flux[g]
        if len(t) < MIN_POINTS:
            out["ell_status"] = f"only {len(t)} points"
            return out
        o = np.argsort(t, kind="stable")
        t, flux = t[o], flux[o]
        from astropy.stats import sigma_clip
        m = sigma_clip(flux, sigma=SIGMA, stdfunc="mad_std", maxiters=5,
                       masked=True).mask
        t, flux = t[~m], flux[~m]
        med = np.median(flux)
        if not np.isfinite(med) or med == 0:
            out["ell_status"] = "degenerate median flux"
            return out
        f = flux / med

        # phase in [-0.5, 0.5), 0 = mid-transit
        ph = ((t - t0 + 0.5 * period) % period) / period - 0.5
        half_dur = duration / period
        # mask BOTH conjunctions -- see the module docstring
        keep = (np.abs(ph) > half_dur) & (np.abs(np.abs(ph) - 0.5) > half_dur)
        if keep.sum() < MIN_FIT_POINTS:
            out["ell_status"] = f"only {int(keep.sum())} usable points"
            return out
        pk, fk = ph[keep], f[keep]

        phi = 2.0 * np.pi * pk
        a1, a2, c2, sig = _harmonic_fit(phi, fk)
        out["ell_a1"] = a1
        out["ell_a2"] = a2
        out["ell_c2_signed"] = c2
        out["ell_a2_snr"] = float(a2 / sig) if sig and sig > 0 else np.nan
        if np.isfinite(a1) and np.isfinite(a2) and (a1 + a2) > 0:
            out["ell_a2_frac"] = float(a2 / (a1 + a2))

        # NULL CONTROL: identical fit at an incommensurate period
        pc = ((t[keep] - t0) % (period * CTRL_MULT)) / (period * CTRL_MULT)
        _, a2c, _, sigc = _harmonic_fit(2.0 * np.pi * (pc - 0.5), fk)
        out["ell_a2_ctrl"] = a2c
        out["ell_a2_snr_ctrl"] = float(a2c / sigc) if sigc and sigc > 0 else np.nan
        if np.isfinite(a2) and np.isfinite(a2c) and a2c > 0:
            out["ell_a2_over_ctrl"] = float(a2 / a2c)

        # fold-and-bin peak-to-peak, at HALF the period and at the FULL period
        pp_h, cov_h = _folded_pp((2.0 * (pk + 0.5)) % 1.0, fk)
        pp_f, _ = _folded_pp((pk + 0.5) % 1.0, fk)
        out["ell_pp_half"] = pp_h
        out["ell_pp_full"] = pp_f
        out["ell_phase_cov"] = cov_h
        if np.isfinite(pp_h) and np.isfinite(pp_f) and pp_f > 0:
            out["ell_pp_ratio"] = float(pp_h / pp_f)
    except Exception as e:
        out["ell_status"] = f"error: {type(e).__name__}: {e}"
    return out


def worker(job):
    host, path, per, t0, dur = job
    r = ellipsoidal_for_raw(path, per, t0, dur)
    r["host"] = host
    return r


def _find(host, dirs):
    for d in dirs:
        p = os.path.join(d, str(host) + ".csv")
        if os.path.exists(p):
            return p
    return None


def _jobs(df, dirs):
    return [(str(r["host"]), _find(r["host"], dirs),
             pd.to_numeric(r.get("period"), errors="coerce"),
             pd.to_numeric(r.get("T0", np.nan), errors="coerce"),
             pd.to_numeric(r.get("duration"), errors="coerce"))
            for _, r in df.iterrows()]


def main():
    tr = pd.read_csv(TRAINING); tr["host"] = tr.host.astype(str)
    jobs = [("train", j) for j in _jobs(tr, TRAIN_RAW_DIRS)]
    for tag, ff, d in POOLS:
        p = pd.read_csv(os.path.join(CAT, ff)); p["host"] = p.host.astype(str)
        if "status" in p.columns:
            p = p[p.status.astype(str).str.startswith("Success")]
        jobs += [(tag, j) for j in _jobs(p, [d])]
    print(f"{len(jobs)} light curves", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
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
    print(out.groupby("_set").ell_status.value_counts())


if __name__ == "__main__":
    main()
