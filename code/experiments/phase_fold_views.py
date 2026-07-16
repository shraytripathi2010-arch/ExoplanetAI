"""
phase_fold_views.py -- builds AstroNet-style local+global phase-folded views
from a light curve given a period/T0/duration, for CNN training (Part A).

Sized smaller than AstroNet's original (2001/201 bins) on purpose: this
project's real training set (~5,491 rows) is roughly a third of AstroNet's
(~15,000), so the view resolution is scaled down to keep the CNN's first-
layer parameter count proportionate to available data -- a deliberate
scope adjustment, not an oversight.
"""
import numpy as np

GLOBAL_BINS = 201
LOCAL_BINS = 61
LOCAL_VIEW_DURATIONS = 4.0  # local view spans +/- this many transit durations around phase 0


def _phase_fold(time, flux, period, t0):
    phase = np.mod((time - t0) / period, 1.0)
    phase = np.where(phase > 0.5, phase - 1.0, phase)  # centered on [-0.5, 0.5]
    order = np.argsort(phase)
    return phase[order], flux[order]


def _median_bin(phase, flux, bin_edges):
    """Median-bin flux into fixed phase bins; empty bins filled with the
    global median (a flat baseline) rather than NaN or zero -- avoids
    injecting a spurious sharp discontinuity the CNN could latch onto."""
    n_bins = len(bin_edges) - 1
    binned = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask.sum() > 0:
            binned[i] = np.median(flux[mask])
    global_median = np.nanmedian(flux) if np.any(~np.isnan(binned)) else 1.0
    binned = np.where(np.isnan(binned), global_median, binned)
    return binned


def make_global_view(time, flux, period, t0, n_bins=GLOBAL_BINS):
    phase, flux_sorted = _phase_fold(time, flux, period, t0)
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    return _median_bin(phase, flux_sorted, edges)


def make_local_view(time, flux, period, t0, duration, n_bins=LOCAL_BINS,
                     window_durations=LOCAL_VIEW_DURATIONS):
    phase, flux_sorted = _phase_fold(time, flux, period, t0)
    half_window = (duration / period) * window_durations / 2.0
    half_window = min(half_window, 0.5)  # can't exceed the full phase range
    edges = np.linspace(-half_window, half_window, n_bins + 1)
    return _median_bin(phase, flux_sorted, edges)


def make_views(time, flux, period, t0, duration):
    """Returns (global_view, local_view) as float32 arrays, each normalized
    to zero-median (subtracting 1.0, since flux is already normalized to a
    baseline of 1.0 by preprocessing) -- makes the CNN's input scale
    consistent regardless of a star's absolute brightness."""
    g = make_global_view(time, flux, period, t0).astype(np.float32) - 1.0
    l = make_local_view(time, flux, period, t0, duration).astype(np.float32) - 1.0
    return g, l
