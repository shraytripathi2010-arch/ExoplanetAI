"""harmonic_band_power_features.py -- the two formulations the periodogram
investigation did NOT compute.

WHAT WAS ALREADY DONE, read from the code and not from memory
-------------------------------------------------------------
`ls_secondary_peak_features._peaks` is entirely PEAK-BASED: `argmax` for the
primary, then the highest remaining LOCAL MAXIMUM for the secondary. Its `_nh`
variant excludes harmonics of **f1, the primary peak's own frequency** -- NOT of
the transit frequency. There is no evaluation at externally-specified
frequencies anywhere, and no summation or integration of power (`grep` for
`integrat|band|trapz|.sum()` returns only the `_nh` comment).

So a peak SEARCH finds whatever is locally strongest. This module instead
EVALUATES the periodogram at frequencies fixed in advance by the candidate's own
ephemeris, and INTEGRATES power over a band. Different operations.

RELATIONSHIP TO THE ELLIPSOIDAL WORK -- partial overlap, stated up front
-----------------------------------------------------------------------
`ellipsoidal_features` already fitted sinusoids at exactly 1x and 2x the transit
frequency (`ell_a1`, `ell_a2`), and both were REDUNDANT with `var_ls_amp`
(|rho| 0.866 / 0.852) and spatially unstable (spread 0.291 / 0.279).

But those are ABSOLUTE amplitudes, and that is precisely why they collapsed onto
`var_ls_amp`. Lomb-Scargle power under `normalization="standard"` is
`1 - chi2(f)/chi2_ref` -- the FRACTION OF VARIANCE the sinusoid at f explains,
i.e. amplitude normalised by the star's own scatter. Dividing out the scatter is
the exact operation that separated redundant from non-redundant in both the
secondary-peak and the ellipsoidal investigations. So the normalised form is a
genuinely different quantity, and 0.5x and 3x were never evaluated at all.

FEATURES
  hp_p05x hp_p1x hp_p2x hp_p3x   LS power at periods 0.5P, 1P, 2P, 3P
  hp_max hp_sum                  max / sum over those four
  hp_frac                        mean power AT the transit harmonics divided by
                                 the mean power everywhere -- dimensionless and
                                 scale-free; >1 means the transit harmonics run
                                 hotter than the star's typical frequency
  bp_band                        power integrated over |f - 1/P| < W
  bp_band_frac                   bp_band / total integrated power
  bp_band_ratio                  mean power in the band / mean power outside it

W is set to BAND_WIDTHS independent peak widths (1/span in frequency), the same
resolution scale `ls_secondary_peak_features` used to separate peaks.

Both the FULL-flux and OOT-MASKED inputs are computed. On full flux the transit
itself injects power at its own harmonics, which would make `hp_p1x` a transit
detector restating SDE/snr -- the OOT arm isolates the star. Separating the two
axes is what made the secondary-peak result interpretable.
"""
import os
import warnings
import importlib.util
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# numpy 2 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CAT = os.path.join(ROOT, "data", "catalogs")
OUT = os.path.join(HERE, "harmonic_band_power_features.csv")

TRAIN_RAW_DIRS = [os.path.join(ROOT, "data", "known_lightcurves"),
                  os.path.join(ROOT, "data", "known_lightcurves_negative"),
                  os.path.join(ROOT, "data", "retrain_pipeline", "raw")]
POOLS = [("main", "unknown_features.csv",
          os.path.join(ROOT, "data", "unknown_lightcurves")),
         ("widesector", "unknown_features_widesector.csv",
          os.path.join(ROOT, "data", "unknown_lightcurves_widesector"))]

SIGMA = 5
MIN_POINTS = 200
MIN_P, MAX_P = 0.2, 13.0
BIN_MINUTES = 10.0
BAND_WIDTHS = 3.0
PERIOD_MULTIPLES = (0.5, 1.0, 2.0, 3.0)
TAGS = ("p05x", "p1x", "p2x", "p3x")

BASE = ([f"hp_{t}" for t in TAGS] + ["hp_max", "hp_sum", "hp_frac",
        "bp_band", "bp_band_frac", "bp_band_ratio"])
COLUMNS = [f"{p}{c}" for p in ("f_", "o_") for c in BASE]

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


def _one_series(tb, fb, period, prefix, out):
    span = tb.max() - tb.min() if len(tb) > 2 else 0.0
    if len(tb) <= 50 or span <= 2 * MIN_P:
        return
    from astropy.timeseries import LombScargle
    ls = LombScargle(tb, fb)
    fmin = 1.0 / min(MAX_P, span / 2.0)
    fmax = 1.0 / MIN_P
    fr, pw = ls.autopower(minimum_frequency=fmin, maximum_frequency=fmax,
                          normalization="standard", samples_per_peak=5)
    if not len(pw):
        return
    total = float(_trapz(pw, fr))
    if not np.isfinite(total) or total <= 0:
        return

    # --- harmonic-specific EVALUATION, not a peak search ---
    freqs = np.array([1.0 / (mu * period) for mu in PERIOD_MULTIPLES], dtype=float)
    inb = (freqs >= fmin) & (freqs <= fmax)
    vals = np.full(len(freqs), np.nan)
    if inb.any():
        # ONE evaluation for all in-band harmonics -- four separate ls.power
        # calls cost ~4x this
        vals[inb] = ls.power(freqs[inb], normalization="standard")
    for v, tag in zip(vals, TAGS):
        out[f"{prefix}hp_{tag}"] = float(v) if np.isfinite(v) else np.nan
    if np.isfinite(vals).any():
        out[f"{prefix}hp_max"] = float(np.nanmax(vals))
        out[f"{prefix}hp_sum"] = float(np.nansum(vals))
        # dimensionless: mean power AT the transit harmonics relative to the
        # mean power everywhere. >1 means the harmonics are hotter than typical.
        mp = float(np.mean(pw))
        if mp > 0:
            out[f"{prefix}hp_frac"] = float(np.nanmean(vals) / mp)

    # --- BAND-INTEGRATED power around the candidate frequency ---
    f0 = 1.0 / period
    w = BAND_WIDTHS / span
    m = np.abs(fr - f0) <= w
    if m.sum() >= 3:
        band = float(_trapz(pw[m], fr[m]))
        out[f"{prefix}bp_band"] = band
        out[f"{prefix}bp_band_frac"] = float(band / total)
        outside = ~m
        if outside.sum() > 10:
            out[f"{prefix}bp_band_ratio"] = float(
                np.mean(pw[m]) / np.mean(pw[outside]))


def features_for_raw(raw_path, period=np.nan, t0=np.nan, duration=np.nan):
    out = {c: np.nan for c in COLUMNS}
    out["hb_status"] = "ok"
    if not raw_path or not os.path.exists(raw_path):
        out["hb_status"] = "no raw light curve"
        return out
    if not (np.isfinite(period) and period > 0):
        out["hb_status"] = "no usable period"
        return out
    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        out["hb_status"] = f"read error: {type(e).__name__}"
        return out
    try:
        PRE = _vf()._pre()
        if PRE.validate_schema(df) is not None:
            out["hb_status"] = "non-standard schema"
            return out
        flux, ferr, _ = PRE.choose_flux_columns(df)
        if flux is None:
            out["hb_status"] = "no usable flux"
            return out
        t = df["time"].to_numpy(); q = df["quality"].to_numpy()
        ok = ~np.isnan(t) & ~np.isnan(flux) & ~np.isnan(ferr)
        t, flux, q = t[ok], flux[ok], q[ok]
        g = q == 0
        t, flux = t[g], flux[g]
        if len(t) < MIN_POINTS:
            out["hb_status"] = f"only {len(t)} points"
            return out
        o = np.argsort(t, kind="stable")
        t, flux = t[o], flux[o]
        from astropy.stats import sigma_clip
        m = sigma_clip(flux, sigma=SIGMA, stdfunc="mad_std", maxiters=5,
                       masked=True).mask
        t, flux = t[~m], flux[~m]
        med = np.median(flux)
        if not np.isfinite(med) or med == 0:
            out["hb_status"] = "degenerate median flux"
            return out
        f = flux / med
        bin_ = _vf()._acf_bin

        tb, fb = bin_(t, f, BIN_MINUTES)
        _one_series(tb, fb, period, "f_", out)

        if np.isfinite(t0) and np.isfinite(duration) and duration > 0:
            ph = ((t - t0 + 0.5 * period) % period) / period - 0.5
            oot = np.abs(ph) > (duration / period)
        else:
            oot = np.ones(len(t), bool)
        if oot.sum() >= MIN_POINTS:
            tb2, fb2 = bin_(t[oot], f[oot], BIN_MINUTES)
            _one_series(tb2, fb2, period, "o_", out)
    except Exception as e:
        out["hb_status"] = f"error: {type(e).__name__}: {e}"
    return out


def worker(job):
    host, path, per, t0, dur = job
    r = features_for_raw(path, per, t0, dur)
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
    print(out.groupby("_set").hb_status.value_counts())


if __name__ == "__main__":
    main()
