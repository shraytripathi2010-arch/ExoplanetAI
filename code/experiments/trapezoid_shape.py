"""trapezoid_shape.py -- a properly duration-scaled transit shape metric, to
replace the structurally broken `transit_shape_ratio`.

WHY A NEW FEATURE RATHER THAN A DUPLICATE

The deployed `transit_shape_ratio` measures two windows fixed in PHASE:

    center = |phase| < 0.005      edge = 0.005 <= |phase| < 0.015
    shape_ratio = (1 - median(flux[edge])) / (1 - median(flux[center]))

Those windows never scale to the transit's actual duration. Measured against
each star's real half-duration in phase units: for 36.9% of stars the entire
transit fits inside the "center" window (so the edge window samples
out-of-transit flux and the ratio collapses toward 0), and for 18.2% both
windows sit inside the flat bottom (both depths ~ full depth, ratio -> 1). For
55% of stars the value is set by window geometry, not by the transit profile.
Its near-zero importance (+0.0001) is therefore evidence about the
implementation, not about whether shape information helps.

The duty-cycle-proxy hypothesis was checked and REJECTED: |r| = 0.018 against
duration/period. It is not a proxy for anything; it is mostly noise.

THE MODEL: A TRAPEZOID, SCALED TO EACH STAR'S OWN FITTED DURATION

Everything is parameterised in units of the star's TLS duration, so no window
is fixed in phase and the 55% geometry failure cannot recur by construction.

    f(phi) = baseline                                   x >= T14/2
           = baseline - depth                           x <= T23/2
           = linear ramp between the two                otherwise
    where x = |phi - phi0|,  T23 = T14 * (1 - w)

Five free parameters: baseline, depth, T14 (total duration, first to fourth
contact), w (the shape metric itself), phi0 (a small centring correction for
ephemeris drift). Parameterising the flat duration as T23 = T14*(1-w) rather
than fitting T23 directly bounds the metric to [0, 1] and makes the invalid
region T23 > T14 unreachable, instead of relying on the optimiser to avoid it.

THE SHAPE METRIC

    trap_vshape = w = (T14 - T23) / T14

the fraction of the total transit spent in ingress plus egress.

    w -> 0   flat-bottomed / boxy: the occulter is small relative to the star
             and fully inside the disc for most of the event. Planetary.
    w -> 1   V-shaped: no flat bottom at all, the event is all ingress and
             egress. This is what a GRAZING eclipse looks like (the occulter
             never fully enters the disc) and what an EQUAL-SIZE binary looks
             like (the occulter is comparable to the star, so ingress occupies
             a large fraction of the event).

For a central transit, ingress duration / T14 ~ Rp/Rs, so a Jupiter around a
Sun (Rp/Rs ~ 0.1) gives w ~ 0.2 while a stellar companion (Rp/Rs ~ 0.5-1)
gives w ~ 0.6-1.0. That is the discriminator, and it is LEO-Vetter's approach.

WHY A TRAPEZOID AND NOT batman, WHICH IS ALREADY INSTALLED

`batman` is a real dependency here (`injection.py` uses it) so this was a live
option, and it was rejected for three specific reasons:

  1. It is a LIMB-DARKENED PLANETARY model, parameterised by Rp/Rs, a/Rs, inc
     and limb-darkening coefficients. Fitting it to a grazing eclipsing binary
     forces a planet-shaped curve onto exactly the profiles this feature exists
     to flag. The fit could report how badly a star fits a planet model, but
     not what shape the transit actually has -- which is the question.
  2. a/Rs and inc are near-degenerate at single-sector TESS SNR. The trapezoid
     asks directly for the one geometric quantity wanted and nothing else.
  3. ~100x the per-star cost, over 5,486 stars, for a worse-posed fit.

The trapezoid is the model-agnostic parameterisation and the one professional
vetting pipelines use for this specific test.

NO TLS RE-RUN AND NO DOWNLOADS: the fold is reconstructed from the stored
period/T0/duration against light curves already on disk, the same pattern as
`timeseries_stats.py` and `oddeven_timing.py`. Fitting is
`scipy.optimize.least_squares` (scipy is already a dependency; `curve_fit` is
used in `learning_curve_extrapolation.py`).

CONVERGENCE AND LOW SNR ARE HANDLED EXPLICITLY, NOT ASSUMED AWAY

A shallow, noisy transit cannot constrain a shape parameter no matter how well
the optimiser converges -- the ingress is simply below the noise. Every fit
therefore carries a parameter uncertainty from the Jacobian at the solution,
and `trap_vshape_err` is reported alongside the value so the usable-coverage
threshold is chosen with the distribution visible rather than guessed at.
"""
import os
import sys
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CATALOGS = os.path.join(ROOT, "data", "catalogs")
PROC_DIRS = [os.path.join(ROOT, "data", "processed"),
             os.path.join(ROOT, "data", "processed_negative"),
             os.path.join(ROOT, "data", "processed_unknown"),
             os.path.join(ROOT, "data", "processed_unknown_widesector")]

SOURCES = {
    "training": (TRAINING, os.path.join(SCRIPT_DIR, "trapezoid_shape_features.csv")),
    "pool": (os.path.join(CATALOGS, "unknown_features.csv"),
             os.path.join(SCRIPT_DIR, "trapezoid_shape_pool.csv")),
    "widesector": (os.path.join(CATALOGS, "unknown_features_widesector.csv"),
                   os.path.join(SCRIPT_DIR, "trapezoid_shape_widesector.csv")),
}

MIN_BASELINE = 100          # raw cadences before anything is attempted
BINS_PER_DURATION = 12      # bin width = T14_tls / 12 -- duration-scaled
WINDOW_DURATIONS = 2.5      # fit window is +-2.5 x T14_tls around phase 0
MIN_PTS_PER_BIN = 3
MIN_BINS_TOTAL = 25
MIN_BINS_IN_TRANSIT = 6
N_WORKERS = 6

FEATURES = ["trap_vshape", "trap_t14_ratio"]
DIAGNOSTICS = ["trap_vshape_err", "trap_depth", "trap_depth_snr", "trap_rmse",
               "trap_nbins", "trap_status"]


def _find(host):
    for d in PROC_DIRS:
        p = os.path.join(d, str(host) + ".csv")
        if os.path.exists(p):
            return p
    return None


def trapezoid(phi, base, depth, t14, w, phi0):
    """Symmetric trapezoid. w is the ingress+egress fraction of T14, so the
    flat bottom is T14*(1-w) wide. The +1e-12 keeps w=0 (a perfect box) a
    valid limit rather than a division by zero."""
    x = np.abs(phi - phi0)
    half14 = t14 / 2.0
    ramp = t14 * w / 2.0 + 1e-12
    frac = np.clip((half14 - x) / ramp, 0.0, 1.0)
    return base - depth * frac


def fit_one(args):
    host, period, t0, duration, depth_lvl = args
    out = {"host": host}
    for c in FEATURES + DIAGNOSTICS:
        out[c] = np.nan
    out["trap_nbins"] = 0
    out["trap_status"] = "ok"

    path = _find(host)
    if path is None:
        out["trap_status"] = "no light curve"
        return out
    if not (np.isfinite(period) and period > 0 and np.isfinite(t0)
            and np.isfinite(duration) and duration > 0 and duration < period):
        out["trap_status"] = "no usable ephemeris"
        return out
    try:
        d = pd.read_csv(path)
        t = d["time"].to_numpy(float)
        f = d["flux"].to_numpy(float)
    except Exception as e:
        out["trap_status"] = f"read error: {type(e).__name__}"
        return out
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if len(t) < MIN_BASELINE:
        out["trap_status"] = f"too few points ({len(t)})"
        return out

    ph = ((t - t0) / period) % 1.0
    ph = np.where(ph > 0.5, ph - 1.0, ph)
    dur_ph = duration / period
    if dur_ph <= 0 or dur_ph > 0.4:
        out["trap_status"] = "duration implausible vs period"
        return out

    # ---- duration-scaled binning: the fix for the fixed-phase-window bug ----
    win = min(WINDOW_DURATIONS * dur_ph, 0.5)
    sel = np.abs(ph) <= win
    if sel.sum() < 30:
        out["trap_status"] = f"too few in-window points ({int(sel.sum())})"
        return out
    p_sel, f_sel = ph[sel], f[sel]
    bw = dur_ph / BINS_PER_DURATION
    nb = max(int(round(2 * win / bw)), MIN_BINS_TOTAL)
    edges = np.linspace(-win, win, nb + 1)
    idx = np.clip(np.digitize(p_sel, edges) - 1, 0, nb - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    bx, by, bn = [], [], []
    for b in range(nb):
        s = idx == b
        n = int(s.sum())
        if n >= MIN_PTS_PER_BIN:
            bx.append(centres[b])
            by.append(np.median(f_sel[s]))
            bn.append(n)
    bx, by, bn = np.asarray(bx), np.asarray(by), np.asarray(bn, float)
    out["trap_nbins"] = int(len(bx))
    if len(bx) < MIN_BINS_TOTAL:
        out["trap_status"] = f"too few usable bins ({len(bx)})"
        return out
    if int((np.abs(bx) < dur_ph / 2).sum()) < MIN_BINS_IN_TRANSIT:
        out["trap_status"] = "too few in-transit bins"
        return out

    # bin medians average down as ~1/sqrt(n); weight the residuals accordingly
    wts = np.sqrt(bn)

    oot = np.abs(bx) > dur_ph
    base0 = float(np.median(by[oot])) if oot.sum() >= 3 else float(np.median(by))
    intr = np.abs(bx) < dur_ph / 2
    depth0 = float(base0 - np.min(by[intr])) if intr.any() else np.nan
    if not np.isfinite(depth0) or depth0 <= 0:
        depth0 = max(float(1.0 - depth_lvl) if np.isfinite(depth_lvl) else 1e-4, 1e-6)
    depth0 = float(np.clip(depth0, 1e-6, 0.5))

    p0 = [base0, depth0, dur_ph, 0.4, 0.0]
    lo = [base0 - 0.05, 1e-7, 0.3 * dur_ph, 0.0, -0.5 * dur_ph]
    hi = [base0 + 0.05, 0.5, 2.5 * dur_ph, 1.0, 0.5 * dur_ph]
    p0 = [float(np.clip(v, l + 1e-12, h - 1e-12)) for v, l, h in zip(p0, lo, hi)]

    def resid(p):
        return (trapezoid(bx, *p) - by) * wts

    try:
        r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=2000)
    except Exception as e:
        out["trap_status"] = f"fit error: {type(e).__name__}"
        return out
    if r.status <= 0:
        out["trap_status"] = "did not converge"
        return out

    base_f, depth_f, t14_f, w_f, _ = r.x
    resid_w = r.fun
    dof = max(len(bx) - 5, 1)
    s2 = float(2 * r.cost / dof)
    try:
        cov = s2 * np.linalg.inv(r.jac.T @ r.jac)
        perr = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        perr = np.full(5, np.nan)

    out["trap_vshape"] = float(w_f)
    out["trap_vshape_err"] = float(perr[3])
    out["trap_t14_ratio"] = float(t14_f / dur_ph)
    out["trap_depth"] = float(depth_f)
    out["trap_depth_snr"] = (float(depth_f / perr[1])
                             if np.isfinite(perr[1]) and perr[1] > 0 else np.nan)
    out["trap_rmse"] = float(np.sqrt(np.mean((resid_w / wts) ** 2)))

    # a fit that ran to a duration bound has not measured a duration, and its
    # shape parameter is whatever the bound implies -- flag, don't silently keep
    if t14_f <= lo[2] * 1.01 or t14_f >= hi[2] * 0.99:
        out["trap_status"] = "T14 hit bound"
    return out


def run(source):
    path, outpath = SOURCES[source]
    df = pd.read_csv(path)
    depth_col = pd.to_numeric(df["depth"], errors="coerce") if "depth" in df else pd.Series(np.nan, index=df.index)
    args = list(zip(df["host"],
                    pd.to_numeric(df["period"], errors="coerce"),
                    pd.to_numeric(df["T0"], errors="coerce"),
                    pd.to_numeric(df["duration"], errors="coerce"),
                    depth_col))
    print(f"\n[{source}] fitting trapezoids for {len(args)} stars ({N_WORKERS} workers)")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        rows = list(ex.map(fit_one, args, chunksize=16))
    o = pd.DataFrame(rows)
    o.to_csv(outpath, index=False)

    conv = o["trap_vshape"].notna()
    print(f"  fit converged:        {conv.sum():>5}/{len(o)} = {conv.mean()*100:5.1f}%")
    print("  failure reasons:")
    bad = o.loc[o["trap_status"] != "ok", "trap_status"]
    for r_, n in bad.value_counts().head(8).items():
        print(f"    {n:>5}  {str(r_)[:62]}")
    if conv.any():
        for q in (0.1, 0.5, 0.9):
            print(f"  vshape_err q{int(q*100):<2}      {o.loc[conv,'trap_vshape_err'].quantile(q):.3f}"
                  f"    depth_snr q{int(q*100):<2} {o.loc[conv,'trap_depth_snr'].quantile(q):8.2f}")
    print(f"  saved {outpath}")
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all",
                    choices=["all", "training", "pool", "widesector"])
    a = ap.parse_args()
    for s in (["training", "pool", "widesector"] if a.source == "all" else [a.source]):
        run(s)


if __name__ == "__main__":
    main()
