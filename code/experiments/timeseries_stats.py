"""timeseries_stats.py -- two time-series statistics TLS never computes and
this project has never tested.

PART 0 ESTABLISHED WHAT IS AND IS NOT NEW

The closed medium-lift "phase-folded flux distribution statistics" experiment
computed exactly nine columns, confirmed from its own output file:

    in_skew, in_kurt, out_skew, out_kurt, skew_diff, kurt_diff,
    wavelet_e1, wavelet_e2, wavelet_e3

Those are the THIRD and FOURTH moments, in and out of transit, plus their
differences, plus Haar detail energies. Result: -0.0072, CI [-0.0153, +0.0010],
nested CV flat. So skewness/kurtosis of the phase-folded profile is a DUPLICATE
and is not recomputed here.

What that experiment never touched:

  * the SECOND moment. It differenced skew and kurt but never compared
    variances. `n_in`/`n_out` in its output are point COUNTS, not variances.
  * any temporal correlation structure. A repo-wide grep for
    autocorr/acf/lag_1/variance_ratio returns nothing.

THE TWO FEATURES, WITH THEIR PREDICTED DIRECTIONS STATED BEFORE MEASUREMENT

  ts_var_ratio = var(in-transit flux) / var(out-of-transit flux)

    A clean, flat-bottomed transit spends most of the in-transit window at
    roughly constant depth, so its in-transit scatter is close to the
    out-of-transit noise and the ratio sits near 1. A V-shaped grazing eclipse
    or a blended binary has flux changing steeply throughout the window, which
    inflates in-transit variance. Instrumental systematics do likewise.
    PREDICTION: false positives > planets, i.e. AUC below 0.5.

  ts_acf_lag1, ts_acf_1hr = autocorrelation of RESIDUALS at two lags

    Residual is defined precisely as flux minus the star's own phase-folded
    binned profile, interpolated back to each cadence. That removes the
    coherent transit signal -- whatever its shape -- and leaves noise plus
    anything the fold does not explain.

    A clean detection leaves near-white residuals, so ACF ~ 0. Eclipsing
    binaries (ellipsoidal variation, imperfect fold) and instrumental
    systematics leave temporally CORRELATED residuals.
    PREDICTION: false positives > planets, i.e. AUC below 0.5.

    Two lags only, both physically chosen, not a lag bank: lag-1 is the
    cadence-to-cadence correlation that detrending is supposed to remove, and
    ~1 hour is the timescale of TESS scattered-light and pointing systematics.
    The small-lift round already showed that many weakly-motivated columns just
    buy extra chances to pass by luck.

No TLS re-run and no downloads: the fold is reconstructed from the stored
period/T0 against light curves already on disk.
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT = os.path.join(SCRIPT_DIR, "timeseries_stats_features.csv")
PROC_DIRS = [os.path.join(ROOT, "data", "processed"),
             os.path.join(ROOT, "data", "processed_negative"),
             os.path.join(ROOT, "data", "processed_unknown"),
             os.path.join(ROOT, "data", "processed_unknown_widesector")]

MIN_BASELINE = 100
MIN_IN, MIN_OUT = 10, 30
N_BINS = 200
LAG_HOURS = 1.0
N_WORKERS = 6
FEATURES = ["ts_var_ratio", "ts_acf_lag1", "ts_acf_1hr"]


def _find(host):
    for d in PROC_DIRS:
        p = os.path.join(d, str(host) + ".csv")
        if os.path.exists(p):
            return p
    return None


def _acf_at(x, k):
    """Autocorrelation of x at integer lag k. NaN if too short."""
    n = len(x)
    if k < 1 or n <= k + 5:
        return np.nan
    a, b = x[:-k], x[k:]
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def compute_one(args):
    host, period, t0, duration = args
    out = {"host": host, "ts_var_ratio": np.nan, "ts_acf_lag1": np.nan,
           "ts_acf_1hr": np.nan, "ts_n_in": 0, "ts_n_out": 0, "ts_status": "ok"}
    path = _find(host)
    if path is None:
        out["ts_status"] = "no light curve"
        return out
    if not (np.isfinite(period) and period > 0 and np.isfinite(t0)
            and np.isfinite(duration) and duration > 0 and duration < period):
        out["ts_status"] = "no usable ephemeris"
        return out
    try:
        d = pd.read_csv(path)
        t = d["time"].to_numpy(float)
        f = d["flux"].to_numpy(float)
    except Exception as e:
        out["ts_status"] = f"read error: {type(e).__name__}"
        return out
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if len(t) < MIN_BASELINE:
        out["ts_status"] = f"too few points ({len(t)})"
        return out

    order = np.argsort(t)
    t, f = t[order], f[order]
    ph = ((t - t0) / period) % 1.0
    ph = np.where(ph > 0.5, ph - 1.0, ph)          # signed, in [-0.5, 0.5]
    dur_ph = duration / period

    in_tr = np.abs(ph) < (dur_ph / 2.0)
    out_tr = (np.abs(ph) > dur_ph) & (np.abs(np.abs(ph) - 0.5) > dur_ph)
    out["ts_n_in"], out["ts_n_out"] = int(in_tr.sum()), int(out_tr.sum())
    if in_tr.sum() < MIN_IN or out_tr.sum() < MIN_OUT:
        out["ts_status"] = f"insufficient in/out points ({in_tr.sum()}/{out_tr.sum()})"
        return out

    # ---- second moment comparison -------------------------------------
    v_in, v_out = float(np.var(f[in_tr])), float(np.var(f[out_tr]))
    if v_out > 0:
        out["ts_var_ratio"] = v_in / v_out

    # ---- residuals: flux minus the star's own folded profile ----------
    edges = np.linspace(-0.5, 0.5, N_BINS + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, N_BINS - 1)
    prof = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        sel = idx == b
        if sel.sum() >= 3:
            prof[b] = np.median(f[sel])
    good = np.isfinite(prof)
    if good.sum() < N_BINS // 4:
        out["ts_status"] = "folded profile too sparse"
        return out
    centres = 0.5 * (edges[:-1] + edges[1:])
    model = np.interp(ph, centres[good], prof[good])
    resid = f - model

    out["ts_acf_lag1"] = _acf_at(resid, 1)
    dt = np.median(np.diff(t))
    if np.isfinite(dt) and dt > 0:
        k = int(round((LAG_HOURS / 24.0) / dt))
        out["ts_acf_1hr"] = _acf_at(resid, max(k, 1))
    return out


def main():
    df = pd.read_csv(TRAINING)
    args = list(zip(df["host"],
                    pd.to_numeric(df["period"], errors="coerce"),
                    pd.to_numeric(df["T0"], errors="coerce"),
                    pd.to_numeric(df["duration"], errors="coerce")))
    print(f"computing time-series statistics for {len(args)} stars "
          f"({N_WORKERS} workers)")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        rows = list(ex.map(compute_one, args, chunksize=32))
    o = pd.DataFrame(rows)
    o.to_csv(OUT, index=False)
    for c in FEATURES:
        print(f"  {c:<16} coverage {o[c].notna().mean()*100:5.1f}%")
    bad = o["ts_status"] != "ok"
    if bad.any():
        print("top failure reasons:")
        for r, n in o.loc[bad, "ts_status"].value_counts().head(4).items():
            print(f"  {n:>5}  {str(r)[:60]}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
