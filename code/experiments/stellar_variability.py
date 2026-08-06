"""stellar_variability.py -- out-of-transit variability / activity metrics.

WHY THIS CANNOT USE data/processed/, WHICH IS THE OBVIOUS SHORTCUT

`02_preprocess.py` applies `savgol_filter` with a window capped at
MAX_FLATTEN_WINDOW = 401 points -- about 13.4 h at 2-minute cadence. That is a
high-pass filter: it removes variability on timescales LONGER than the window.
Stellar rotation, the dominant activity signal, has periods of days. So the
processed light curves have had precisely the signal this feature is about
removed by construction, and computing variability from them would measure the
detrending residual rather than the star.

This script therefore reads the RAW light curves and reuses
`02_preprocess.py`'s own `validate_schema` and `choose_flux_columns` -- that
file documents EIGHT distinct real schemas among the downloaded files (CDIPS
flux in magnitudes, single-letter string quality codes, sentinel fill values
like -9.2e18). Re-implementing that parsing would be a fresh source of silent
corruption, so it is imported, not rewritten. Every cleaning step is kept
EXCEPT the flatten.

Single sector per star by construction -- one raw file per star -- so the
ephemeris-precision constraint that closed the multi-sector work does not
apply. No stacking is attempted.

THE METRICS, and why each exists

  var_oot_rms    Robust scatter (1.4826 x MAD) of out-of-transit normalized
                 flux. The simplest standard variability index. Transits are
                 masked using the stored ephemeris so the signal being vetted
                 does not contribute to its own vetting statistic.

  var_excess     var_oot_rms divided by the median normalized flux error.
                 THE ONE THAT MATTERS. Raw RMS conflates a genuinely active
                 star with a faint noisy one; dividing by the star's own
                 photometric error isolates variability ABOVE photon noise.
                 ~1 means "scatter consistent with noise", >>1 means real
                 astrophysical variability.

  var_ls_amp     Amplitude of the dominant Lomb-Scargle peak over periods
                 0.2-13 d (spot modulation and pulsation live here; the upper
                 bound is set by a 27-day sector, where anything beyond ~half
                 the baseline is unconstrained).

  var_ls_power   Normalized power of that peak, 0-1. Distinguishes coherent
                 periodic variability from broadband scatter.

PHYSICAL HYPOTHESIS, stated before measuring. Spotted/active and pulsating
stars produce quasi-periodic dips that mimic or obscure transits, and
eclipsing binaries add ellipsoidal modulation and reflection between eclipses.
Both push variability UP for false positives. PREDICTION: false positives >
planets, i.e. single-feature AUC BELOW 0.5 for all four.

That prediction carries a known risk worth naming in advance: this project's
positive class is dominated by bright, well-studied confirmed hosts and the
negative class by TOI false positives, so a brightness//noise difference could
produce the same direction for an uninteresting reason. `var_excess` is the
metric designed to be robust to that, and the two are compared.
"""
import os
import sys
import warnings
import importlib.util

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from astropy.timeseries import LombScargle
from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
RAW_DIRS = [os.path.join(ROOT, "data", "known_lightcurves"),
            os.path.join(ROOT, "data", "known_lightcurves_negative")]
OUT = os.path.join(SCRIPT_DIR, "stellar_variability_features.csv")

SIGMA = 5
MIN_POINTS = 200
MIN_P, MAX_P = 0.2, 13.0
BIN_MINUTES = 10.0
N_WORKERS = 6
FEATURES = ["var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]


def _load_pipeline():
    """Import 02_preprocess.py's parsers. The module name starts with a digit,
    so a normal import statement is not syntactically possible."""
    p = os.path.join(ROOT, "code", "02_preprocess.py")
    spec = importlib.util.spec_from_file_location("preprocess02", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PRE = None


def _find(host):
    for d in RAW_DIRS:
        p = os.path.join(d, str(host) + ".csv")
        if os.path.exists(p):
            return p
    return None


def _bin(t, f, minutes):
    if len(t) == 0:
        return t, f
    w = minutes / (24.0 * 60.0)
    idx = np.floor((t - t.min()) / w).astype(int)
    df = pd.DataFrame({"i": idx, "t": t, "f": f}).groupby("i").mean()
    return df["t"].to_numpy(), df["f"].to_numpy()


def compute_one(args):
    global PRE
    if PRE is None:
        PRE = _load_pipeline()
    host, period, t0, duration = args
    out = {"host": host, "var_status": "ok"}
    for k in FEATURES:
        out[k] = np.nan

    path = _find(host)
    if path is None:
        out["var_status"] = "no raw light curve"
        return out
    try:
        df = pd.read_csv(path)
    except Exception as e:
        out["var_status"] = f"read error: {type(e).__name__}"
        return out

    # --- identical cleaning to the pipeline, MINUS the flatten ---
    if PRE.validate_schema(df) is not None:
        out["var_status"] = "non-standard schema"
        return out
    flux, ferr, src = PRE.choose_flux_columns(df)
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

    m = sigma_clip(flux, sigma=SIGMA, stdfunc="mad_std", maxiters=5, masked=True).mask
    t, flux, ferr = t[~m], flux[~m], ferr[~m]
    med = np.median(flux)
    if not np.isfinite(med) or med == 0:
        out["var_status"] = "degenerate median flux"
        return out
    f = flux / med
    fe = ferr / med

    # --- mask the transit itself, so it cannot inflate its own vetting stat ---
    if all(np.isfinite([period, t0, duration])) and period > 0 and duration > 0:
        ph = ((t - t0 + 0.5 * period) % period) / period - 0.5
        oot = np.abs(ph) > (duration / period)      # 2x duration, generous
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

    tb, fb = _bin(to, fo, BIN_MINUTES)
    span = tb.max() - tb.min() if len(tb) > 2 else 0.0
    if len(tb) > 50 and span > 2 * MIN_P:
        try:
            ls = LombScargle(tb, fb)
            fr, pw = ls.autopower(minimum_frequency=1.0 / min(MAX_P, span / 2.0),
                                  maximum_frequency=1.0 / MIN_P,
                                  normalization="standard",
                                  samples_per_peak=5)
            if len(pw):
                i = int(np.argmax(pw))
                out["var_ls_power"] = float(pw[i])
                out["var_ls_period"] = float(1.0 / fr[i])
                out["var_ls_amp"] = float(np.std(ls.model(tb, fr[i]) - np.mean(fb)) * np.sqrt(2))
        except Exception:
            pass
    return out


def main():
    tr = pd.read_csv(TRAINING)
    jobs = []
    for _, r in tr.iterrows():
        jobs.append((r["host"], pd.to_numeric(r.get("period"), errors="coerce"),
                     pd.to_numeric(r.get("T0"), errors="coerce"),
                     pd.to_numeric(r.get("duration"), errors="coerce")))
    print(f"{len(jobs)} stars, {N_WORKERS} workers", flush=True)

    rows = []
    import time as _t
    t0 = _t.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, r in enumerate(ex.map(compute_one, jobs, chunksize=16), 1):
            rows.append(r)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}  ({(_t.time()-t0)/60:.1f} min)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print(out["var_status"].value_counts().to_string())
    for f in FEATURES:
        print(f"  {f:15s} non-null {out[f].notna().sum()} ({out[f].notna().mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
