"""oddeven_timing.py -- odd/even TIMING offset, the half-period EB signature
that depth-based tests cannot see.

WHAT THIS MEASURES, AND WHY IT IS NOT THE DEPLOYED odd_even_mismatch

`odd_even_mismatch` (deployed) compares the DEPTHS of odd- and even-numbered
transits. It catches a half-period binary whose two eclipses have different
depths -- i.e. components of different surface brightness. It is blind to a
binary whose eclipses happen to match in depth.

This measures WHERE the two groups fall in phase. Fold at the detected period,
split events by epoch parity, and measure each group's flux-weighted phase
centroid. A single periodic source puts both groups at the same phase. A
half-period binary with a non-circular orbit puts the secondary away from the
midpoint, so the two groups separate in phase even when their depths agree.

    offset_frac = |centroid_even - centroid_odd| / (duration / period)

expressed as a fraction of the transit duration so it is comparable across
stars with wildly different periods.

WHY GROUP FOLDING RATHER THAN PER-TRANSIT TIMES. This project already hit a
wall measuring individual transits: "simple threshold-crossing on raw
per-cadence photometry doesn't reliably measure a single transit's duration --
validated broken on both a shallow-transit star and a deep one". Individual
mid-times face exactly that problem. Folding each parity group together first
recovers the SNR the individual events lack, the same trick the weak-secondary
work used. No TLS re-run: the fold comes from the stored period/T0.

RELATIONSHIP TO `power_ratio_half_period` (already tested, negative). That
asks a frequency-domain question -- is there periodogram power at P/2. This
asks a time-domain one. They target the same physical phenomenon by different
routes, and the residual population this addresses is narrow: eccentric
binaries whose eclipse depths happen to match. Small expected headroom, stated
up front.
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
OUT = os.path.join(SCRIPT_DIR, "oddeven_timing_features.csv")
PROC_DIRS = [os.path.join(ROOT, "data", "processed"),
             os.path.join(ROOT, "data", "processed_negative"),
             os.path.join(ROOT, "data", "processed_unknown"),
             os.path.join(ROOT, "data", "processed_unknown_widesector")]

MIN_BASELINE = 100
MIN_EPOCHS_PER_GROUP = 2      # need >=2 distinct epochs per parity to compare
MIN_PTS_PER_GROUP = 8
N_WORKERS = 6
FEATURES = ["oe_timing_offset_frac", "oe_timing_significance"]


def _find(host):
    for d in PROC_DIRS:
        p = os.path.join(d, str(host) + ".csv")
        if os.path.exists(p):
            return p
    return None


def _group_centroid(ph, fl, oot_level):
    """Flux-weighted phase centroid of a dip, plus its standard error.

    Weights are the DEPTH below the out-of-transit level, clipped at zero, so
    only points that actually dim contribute. Returns (centroid, sigma, n).
    """
    w = np.clip(oot_level - fl, 0.0, None)
    if w.sum() <= 0:
        return np.nan, np.nan, 0
    c = float(np.sum(w * ph) / np.sum(w))
    # weighted scatter -> standard error of the weighted mean
    var = float(np.sum(w * (ph - c) ** 2) / np.sum(w))
    n_eff = float(np.sum(w) ** 2 / np.sum(w ** 2))     # Kish effective n
    sig = np.sqrt(var / n_eff) if n_eff > 1 else np.nan
    return c, sig, int(len(ph))


def measure(args):
    host, period, t0, duration = args
    out = {"host": host, "oe_timing_offset_frac": np.nan,
           "oe_timing_significance": np.nan, "oe_n_epochs_odd": 0,
           "oe_n_epochs_even": 0, "oe_status": "ok"}
    path = _find(host)
    if path is None:
        out["oe_status"] = "no light curve"
        return out
    if not (np.isfinite(period) and period > 0 and np.isfinite(t0)
            and np.isfinite(duration) and duration > 0 and duration < period):
        out["oe_status"] = "no usable ephemeris"
        return out
    try:
        d = pd.read_csv(path)
        t = d["time"].to_numpy(float)
        f = d["flux"].to_numpy(float)
    except Exception as e:
        out["oe_status"] = f"read error: {type(e).__name__}"
        return out
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if len(t) < MIN_BASELINE:
        out["oe_status"] = f"too few points ({len(t)})"
        return out

    cyc = (t - t0) / period
    epoch = np.round(cyc).astype(int)
    ph = cyc - epoch                      # signed phase in [-0.5, 0.5]
    dur_ph = duration / period

    # in-transit window: 1.0x the duration, centred; out-of-transit reference
    # taken well away from both the primary and the phase-0.5 secondary.
    in_tr = np.abs(ph) < (dur_ph / 2.0)
    oot = (np.abs(ph) > dur_ph) & (np.abs(np.abs(ph) - 0.5) > dur_ph)
    if oot.sum() < 20:
        out["oe_status"] = "insufficient out-of-transit baseline"
        return out
    oot_level = float(np.median(f[oot]))

    odd = in_tr & (epoch % 2 != 0)
    even = in_tr & (epoch % 2 == 0)
    n_ep_odd = len(np.unique(epoch[odd]))
    n_ep_even = len(np.unique(epoch[even]))
    out["oe_n_epochs_odd"], out["oe_n_epochs_even"] = n_ep_odd, n_ep_even
    if (n_ep_odd < MIN_EPOCHS_PER_GROUP or n_ep_even < MIN_EPOCHS_PER_GROUP
            or odd.sum() < MIN_PTS_PER_GROUP or even.sum() < MIN_PTS_PER_GROUP):
        out["oe_status"] = (f"too few epochs (odd={n_ep_odd}, even={n_ep_even})")
        return out

    c_o, s_o, _ = _group_centroid(ph[odd], f[odd], oot_level)
    c_e, s_e, _ = _group_centroid(ph[even], f[even], oot_level)
    if not (np.isfinite(c_o) and np.isfinite(c_e)):
        out["oe_status"] = "no measurable dip in one parity group"
        return out

    offset = abs(c_e - c_o)
    out["oe_timing_offset_frac"] = float(offset / dur_ph)
    if np.isfinite(s_o) and np.isfinite(s_e) and (s_o > 0 or s_e > 0):
        out["oe_timing_significance"] = float(offset / np.sqrt(s_o ** 2 + s_e ** 2))
    return out


def main():
    df = pd.read_csv(TRAINING)
    args = list(zip(df["host"],
                    pd.to_numeric(df["period"], errors="coerce"),
                    pd.to_numeric(df["T0"], errors="coerce"),
                    pd.to_numeric(df["duration"], errors="coerce")))
    print(f"measuring odd/even timing offset for {len(args)} stars "
          f"({N_WORKERS} workers)")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        rows = list(ex.map(measure, args, chunksize=32))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    ok = out["oe_timing_offset_frac"].notna()
    print(f"\ncoverage: {ok.sum()}/{len(out)} = {ok.mean()*100:.1f}%")
    print("top failure reasons:")
    for r, n in out.loc[~ok, "oe_status"].value_counts().head(5).items():
        print(f"  {n:>5}  {str(r)[:70]}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
