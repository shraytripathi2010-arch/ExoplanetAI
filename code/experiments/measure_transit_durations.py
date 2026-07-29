"""
measure_transit_durations.py -- genuinely new per-transit duration
extraction, since TLS's own result object only exposes ONE global
box-fit duration, not a per-transit array (confirmed live earlier this
session). Independently measures each individual transit's actual
in-transit time span directly from the real light curve, using only
data already on disk (period/T0/duration/depth from training.csv,
processed light curves already downloaded) -- no TLS rerun needed.

Method: for each predicted transit epoch (T0 + n*period within the light
curve's time range), take the local window of real flux points, find
points below a half-depth threshold, and measure the time span of the
CONTIGUOUS in-transit block closest to the predicted epoch (guards
against a stray low point far from the real transit inflating the
measured span). At least 3 in-transit points required per transit,
otherwise that epoch is skipped (real data gap or noise, not guessed).

duration_consistency_cv: coefficient of variation (std/mean) of the
per-transit measured durations for a star -- analogous in spirit to the
existing depth-consistency features, but for duration, and empirically
measured rather than reused from the global box fit.
"""
import numpy as np


def measure_transit_durations(time, flux, period, t0, duration, depth):
    if not (period > 0 and duration > 0) or not np.isfinite(depth):
        return []
    depth_frac = 1.0 - depth
    if depth_frac <= 0:
        return []
    threshold = 1.0 - depth_frac * 0.5  # half-depth crossing

    t_min, t_max = float(np.min(time)), float(np.max(time))
    n_start = int(np.floor((t_min - t0) / period)) - 1
    n_end = int(np.ceil((t_max - t0) / period)) + 1

    durations = []
    for n in range(n_start, n_end + 1):
        t_epoch = t0 + n * period
        window_half = duration * 2.5
        mask = np.abs(time - t_epoch) < window_half
        if mask.sum() < 5:
            continue
        t_local, f_local = time[mask], flux[mask]
        order = np.argsort(t_local)
        t_local, f_local = t_local[order], f_local[order]

        # BUG FOUND LIVE: taking min/max over EVERY below-threshold point in
        # a loose window let one stray noisy point far from the real dip
        # inflate the "duration" to roughly the window width itself --
        # measured values came out a consistent ~3x the known fitted
        # duration, not noise, a real bug. Fix: find the actual CONTIGUOUS
        # run of below-threshold points (by index adjacency after sorting
        # by time), keep only the run closest to the predicted epoch, and
        # measure just that run's span.
        below = f_local < threshold
        if below.sum() < 3:
            continue
        runs = []
        run_start = None
        for i, b in enumerate(below):
            if b and run_start is None:
                run_start = i
            elif not b and run_start is not None:
                runs.append((run_start, i - 1))
                run_start = None
        if run_start is not None:
            runs.append((run_start, len(below) - 1))
        runs = [r for r in runs if (r[1] - r[0] + 1) >= 3]
        if not runs:
            continue
        run_centers = [(t_local[r[0]] + t_local[r[1]]) / 2 for r in runs]
        best_run = runs[int(np.argmin(np.abs(np.array(run_centers) - t_epoch)))]
        if abs(((t_local[best_run[0]] + t_local[best_run[1]]) / 2) - t_epoch) > duration * 1.0:
            continue  # closest run still isn't near the predicted epoch -- skip, don't guess
        measured = float(t_local[best_run[1]] - t_local[best_run[0]])
        if measured > 0:
            durations.append(measured)
    return durations


def duration_consistency_cv(durations):
    if len(durations) < 2:
        return np.nan
    arr = np.asarray(durations)
    mean = arr.mean()
    if mean <= 0:
        return np.nan
    return float(arr.std() / mean)
