"""variability_features.py -- production computation of the five stellar
variability / activity features.

=============================================================================
READ THIS BEFORE CHANGING THE PREPROCESSING STAGE
=============================================================================
These five features are the ONLY model inputs in this pipeline computed from
the RAW light curve rather than from `data/processed/`.

`02_preprocess.py` flattens with `savgol_filter` using a window capped at
MAX_FLATTEN_WINDOW = 401 points (~13.4 h at 2-minute cadence). That is a
HIGH-PASS FILTER: it removes variability on timescales LONGER than the window.
Stellar rotation -- the dominant activity signal these features measure -- has
periods of days. Feeding a processed light curve into this module would
therefore measure the detrending residual instead of the star, producing
plausible-looking numbers that are silently wrong, with no error raised.

The raw path is therefore an EXPLICIT ARGUMENT here, never inferred from a
module-level directory list. Each caller states where its raw data lives:

    06_download_unknown.py  ->  data/unknown_lightcurves/<host>.csv
    web/retrain_pipeline.py ->  data/retrain_pipeline/raw/<host>.csv
    backfill (training)     ->  data/known_lightcurves[_negative]/<host>.csv

A previous draft kept the directory in module state and set it from the parent
process; under `spawn` the workers re-imported the module and silently reverted
to the default, which surfaced as BrokenProcessPool. Passing the path per call
makes that failure mode unreachable.
=============================================================================

Cleaning matches `02_preprocess.py` step for step -- schema validation, flux
column choice, NaN drop, quality==0 filter, time sort, 5-sigma MAD clip,
median normalisation -- and then STOPS, omitting only the flatten. The
pipeline's own parsers are imported rather than reimplemented, because that
file documents eight distinct real schemas among the downloaded files.

FEATURES
  var_oot_rms    robust scatter (1.4826 x MAD) of out-of-transit flux
  var_excess     var_oot_rms / median photometric error -- scatter ABOVE
                 photon noise, the metric that isolates real activity from
                 a merely faint star
  var_ls_amp     amplitude of the dominant Lomb-Scargle peak, 0.2-13 d
  var_ls_power   normalised power of that peak (0-1)
  var_ls_period  period of that peak, days

Transits are masked at 2x duration before measuring, so the signal being vetted
cannot inflate its own vetting statistic. Single sector by construction.

Never raises: any failure returns NaNs with a status string, which
HistGradientBoosting handles natively via the median imputer.
"""
import os
import warnings
import importlib.util

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

VARIABILITY_COLUMNS = ["var_oot_rms", "var_excess", "var_ls_amp",
                       "var_ls_power", "var_ls_period"]

SIGMA = 5
MIN_POINTS = 200
MIN_P, MAX_P = 0.2, 13.0
BIN_MINUTES = 10.0

_PRE = None


def _pre():
    """02_preprocess.py's own parsers. Module name starts with a digit, so a
    normal import statement is not syntactically possible."""
    global _PRE
    if _PRE is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "02_preprocess.py")
        spec = importlib.util.spec_from_file_location("preprocess02", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _PRE = m
    return _PRE


def _acf_bin(t, f, minutes):
    if len(t) == 0:
        return t, f
    w = minutes / (24.0 * 60.0)
    idx = np.floor((t - t.min()) / w).astype(int)
    g = pd.DataFrame({"i": idx, "t": t, "f": f}).groupby("i").mean()
    return g["t"].to_numpy(), g["f"].to_numpy()


def variability_for_raw(raw_path, period=np.nan, t0=np.nan, duration=np.nan):
    """Five variability features from ONE raw light-curve CSV.

    `raw_path` must be a RAW file (pre-flatten). See the module docstring.
    Returns a dict with the five columns plus `var_status`; never raises.
    """
    out = {c: np.nan for c in VARIABILITY_COLUMNS}
    out["var_status"] = "ok"

    if not raw_path or not os.path.exists(raw_path):
        out["var_status"] = "no raw light curve"
        return out
    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        out["var_status"] = f"read error: {type(e).__name__}"
        return out

    try:
        PRE = _pre()
        if PRE.validate_schema(df) is not None:
            out["var_status"] = "non-standard schema"
            return out
        flux, ferr, _src = PRE.choose_flux_columns(df)
        if flux is None:
            out["var_status"] = "no usable flux"
            return out

        t = df["time"].to_numpy()
        q = df["quality"].to_numpy()
        ok = ~np.isnan(t) & ~np.isnan(flux) & ~np.isnan(ferr)
        t, flux, ferr, q = t[ok], flux[ok], ferr[ok], q[ok]
        g = q == 0
        t, flux, ferr = t[g], flux[g], ferr[g]
        if len(t) < MIN_POINTS:
            out["var_status"] = f"only {len(t)} points"
            return out
        o = np.argsort(t, kind="stable")
        t, flux, ferr = t[o], flux[o], ferr[o]

        from astropy.stats import sigma_clip
        m = sigma_clip(flux, sigma=SIGMA, stdfunc="mad_std", maxiters=5,
                       masked=True).mask
        t, flux, ferr = t[~m], flux[~m], ferr[~m]
        med = np.median(flux)
        if not np.isfinite(med) or med == 0:
            out["var_status"] = "degenerate median flux"
            return out
        f = flux / med
        fe = ferr / med

        # mask the transit so it cannot inflate its own vetting statistic
        if all(np.isfinite([period, t0, duration])) and period > 0 and duration > 0:
            ph = ((t - t0 + 0.5 * period) % period) / period - 0.5
            oot = np.abs(ph) > (duration / period)
        else:
            oot = np.ones(len(t), bool)
        if oot.sum() < MIN_POINTS:
            out["var_status"] = f"only {int(oot.sum())} out-of-transit points"
            return out
        to, fo, feo = t[oot], f[oot], fe[oot]

        rms = 1.4826 * np.median(np.abs(fo - np.median(fo)))
        out["var_oot_rms"] = float(rms)
        mede = float(np.median(feo))
        out["var_excess"] = float(rms / mede) if mede > 0 else np.nan

        tb, fb = _acf_bin(to, fo, BIN_MINUTES)
        span = tb.max() - tb.min() if len(tb) > 2 else 0.0
        if len(tb) > 50 and span > 2 * MIN_P:
            from astropy.timeseries import LombScargle
            ls = LombScargle(tb, fb)
            fr, pw = ls.autopower(
                minimum_frequency=1.0 / min(MAX_P, span / 2.0),
                maximum_frequency=1.0 / MIN_P,
                normalization="standard", samples_per_peak=5)
            if len(pw):
                i = int(np.argmax(pw))
                out["var_ls_power"] = float(pw[i])
                out["var_ls_period"] = float(1.0 / fr[i])
                out["var_ls_amp"] = float(
                    np.std(ls.model(tb, fr[i]) - np.mean(fb)) * np.sqrt(2))
    except Exception as e:
        out["var_status"] = f"error: {type(e).__name__}: {e}"
    return out


def worker(job):
    """Top-level, picklable worker for ProcessPoolExecutor.

    `job` = (host, raw_path, period, t0, duration). The raw path travels WITH
    the job rather than living in module state, so `spawn` workers cannot
    silently fall back to a different directory.
    """
    host, raw_path, period, t0, duration = job
    r = variability_for_raw(raw_path, period, t0, duration)
    r["host"] = host
    return r


def find_raw(host, raw_dirs):
    for d in raw_dirs:
        p = os.path.join(d, str(host) + ".csv")
        if os.path.exists(p):
            return p
    return None
