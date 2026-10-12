"""
folded_plot.py -- the candidate page's folded light-curve figure.

Two panels in one PNG, folded on the period and T0 already stored in the
candidate's characterization (nothing is refitted here):

  top     the transit itself: about +/-3 transit durations around phase 0,
          raw points faint, binned medians prominent with standard-error bars;
  bottom  the whole orbit, binned medians only, with phase 0.5 marked so a
          secondary eclipse or an odd shape is visible.

The previous figure was a single full-phase scatter of every point. The dip
was a median 14 px wide on its axis and, for 211 of 296 candidates, shallower
than the point-to-point scatter, so it was not visible
(audit/roadmap_scoping.md, H6).

PLOT_FORMAT_VERSION is part of the cached filename. Bump it whenever the
figure changes, so a PNG drawn in an older format is never served as the
current one.
"""
import math

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

PLOT_FORMAT_VERSION = 2

REQUIRED_FIELDS = ("period", "T0", "transit_duration_hours")

ZOOM_HALF_WIDTH_DURATIONS = 3.0
# Bin-width rules, both derived from the transit duration rather than fixed.
# Zoom panel: duration / 5, i.e. five bins across the transit. That is enough
# to show the flat bottom and the ingress/egress slopes, and each bin still
# averages points from every transit observed.
ZOOM_BINS_PER_DURATION = 5
# Full-orbit panel: half a duration, so a secondary eclipse (the same duration
# for a circular orbit) spans at least two bins. Clamped to between 50 and 400
# bins per orbit: fewer hides structure, more is mostly noise.
FULL_BINS_PER_DURATION = 2
FULL_MIN_BINS, FULL_MAX_BINS = 50, 400
MIN_POINTS_PER_BIN = 3


def _finite_positive(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0


def inputs_missing(char):
    """The stored fields this figure needs and the candidate lacks. period and
    duration must be positive; T0 only has to be a number."""
    char = char or {}
    missing = []
    for k in REQUIRED_FIELDS:
        v = char.get(k)
        if k == "T0":
            ok = v is not None and not (isinstance(v, float) and v != v)
        else:
            ok = _finite_positive(v)
        if not ok:
            missing.append(k)
    return missing


def plot_filename(host):
    return f"{host}_folded_v{PLOT_FORMAT_VERSION}.png"


def layout(period_days, duration_hours):
    """Window and bin widths, in phase and in time units, for one candidate."""
    period_days = float(period_days)
    duration_hours = float(duration_hours)
    dur_phase = duration_hours / 24.0 / period_days          # hours -> days -> phase
    half_window = min(ZOOM_HALF_WIDTH_DURATIONS * dur_phase, 0.5)
    zoom_bin = dur_phase / ZOOM_BINS_PER_DURATION
    full_bin = min(max(dur_phase / FULL_BINS_PER_DURATION, 1.0 / FULL_MAX_BINS), 1.0 / FULL_MIN_BINS)
    to_hours = period_days * 24.0
    return {
        "period_days": period_days, "duration_hours": duration_hours, "dur_phase": dur_phase,
        "half_window": half_window, "window_clamped": half_window < ZOOM_HALF_WIDTH_DURATIONS * dur_phase,
        "zoom_bin": zoom_bin, "full_bin": full_bin,
        "half_window_hours": half_window * to_hours,
        "zoom_bin_minutes": zoom_bin * to_hours * 60.0,
        "full_bin_minutes": full_bin * to_hours * 60.0,
    }


def caption(char):
    """What the figure actually shows, in words, for the page under it."""
    lay = layout(char["period"], char["transit_duration_hours"])
    if lay["window_clamped"]:
        window = (f"the whole orbit, because three transit durations "
                  f"({ZOOM_HALF_WIDTH_DURATIONS * lay['duration_hours']:.2f} h) exceed half the period")
    else:
        window = (f"±{lay['half_window_hours']:.2f} h around mid-transit "
                  f"(±{ZOOM_HALF_WIDTH_DURATIONS:.0f} transit durations of {lay['duration_hours']:.2f} h)")
    return (
        f"Top: {window}. Raw TESS points are drawn faint; the dark points are medians in "
        f"{lay['zoom_bin_minutes']:.1f}-minute bins (a fifth of the transit duration), with error "
        f"bars of the scatter within each bin divided by √n. The y-axis is scaled to the binned "
        f"points, so raw points beyond it are cut off. "
        f"Bottom: the whole orbit, binned medians only, in {lay['full_bin_minutes']:.1f}-minute bins; "
        f"the dashed line at phase 0.5 is where a secondary eclipse would fall, and the shaded band "
        f"is the top panel's window. Both are folded on the stored period "
        f"({lay['period_days']:.5f} d) and epoch; nothing is refitted.")


def fold(time, t0, period):
    """Phase in [-0.5, 0.5), transit at 0."""
    return np.mod((np.asarray(time, float) - t0) / period + 0.5, 1.0) - 0.5


def bin_medians(x, y, lo, hi, width, min_points=MIN_POINTS_PER_BIN):
    """Median, standard error (std / sqrt(n)) and count per bin over [lo, hi).
    Bins with fewer than min_points points are dropped."""
    n_bins = max(1, int(round((hi - lo) / width)))
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.searchsorted(edges, x, side="right") - 1
    keep = (idx >= 0) & (idx < n_bins)
    idx, y = idx[keep], y[keep]
    order = np.argsort(idx, kind="stable")
    idx, y = idx[order], y[order]
    starts = np.searchsorted(idx, np.arange(n_bins), side="left")
    stops = np.searchsorted(idx, np.arange(n_bins), side="right")
    centers, med, err, cnt = [], [], [], []
    for b, (s, e) in enumerate(zip(starts, stops)):
        n = e - s
        if n < min_points:
            continue
        seg = y[s:e]
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        med.append(float(np.median(seg)))
        err.append(float(np.std(seg, ddof=1) / math.sqrt(n)))
        cnt.append(n)
    return np.array(centers), np.array(med), np.array(err), np.array(cnt)


def load_curve(csv_path):
    df = pd.read_csv(csv_path, usecols=["time", "flux"]).dropna()
    return df["time"].to_numpy(float), df["flux"].to_numpy(float)


def zoom_stats(phase, flux, lay):
    """Binned transit depth against bin-to-bin scatter in the zoom window.

    In-transit bins have centres within half a duration of phase 0. Baseline
    bins are at least one full duration away, which skips ingress and egress.
    depth = median of the baseline medians - median of the in-transit medians;
    scatter = 1.4826 * MAD of the baseline medians. Same bins as the figure."""
    hw, d = lay["half_window"], lay["dur_phase"]
    sel = np.abs(phase) < hw
    c, m, _, _ = bin_medians(phase[sel], flux[sel], -hw, hw, lay["zoom_bin"])
    inside, base = m[np.abs(c) <= d / 2], m[np.abs(c) >= d]
    if len(inside) == 0 or len(base) < 3:
        return {"depth": None, "scatter": None, "ratio": None, "n_in": len(inside), "n_base": len(base)}
    depth = float(np.median(base) - np.median(inside))
    scatter = float(1.4826 * np.median(np.abs(base - np.median(base))))
    return {"depth": depth, "scatter": scatter, "ratio": depth / scatter if scatter > 0 else None,
            "n_in": len(inside), "n_base": len(base)}


def _ylim(med, err):
    lo, hi = float(np.min(med - 2 * err)), float(np.max(med + 2 * err))
    pad = max(0.25 * (hi - lo), 1e-5)
    return lo - pad, hi + pad


def render(host, csv_path, period, t0, duration_hours, out_path):
    time, flux = load_curve(csv_path)
    period, t0 = float(period), float(t0)
    lay = layout(period, duration_hours)
    phase = fold(time, t0, period)
    to_hours = period * 24.0

    fig = Figure(figsize=(8, 6.4), dpi=120, layout="constrained")
    FigureCanvasAgg(fig)
    top, bottom = fig.subplots(2, 1, height_ratios=[3, 2])

    hw = lay["half_window"]
    sel = np.abs(phase) < hw
    zc, zm, ze, _ = bin_medians(phase[sel], flux[sel], -hw, hw, lay["zoom_bin"])
    top.scatter(phase[sel] * to_hours, flux[sel], s=2, alpha=0.18, color="#7a8794", linewidths=0,
                rasterized=True, label="TESS points")
    if len(zm):
        top.errorbar(zc * to_hours, zm, yerr=ze, fmt="o", ms=3.5, color="#17404f", ecolor="#17404f",
                     elinewidth=1, capsize=0, label=f"median, {lay['zoom_bin_minutes']:.1f}-min bins")
        top.set_ylim(*_ylim(zm, ze))
    half_dur_h = lay["duration_hours"] / 2
    top.axvspan(-half_dur_h, half_dur_h, color="#b8791f", alpha=0.10, lw=0,
                label=f"transit duration ({lay['duration_hours']:.2f} h)")
    top.set_xlim(-hw * to_hours, hw * to_hours)
    top.set_xlabel("Hours from mid-transit")
    top.set_ylabel("Normalized flux")
    top.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize=8, frameon=False)

    full_phase = np.where(phase < -0.25, phase + 1.0, phase)     # [-0.25, 0.75): 0 and 0.5 both inside
    fc, fm, fe, _ = bin_medians(full_phase, flux, -0.25, 0.75, lay["full_bin"])
    if len(fm):
        bottom.errorbar(fc, fm, yerr=fe, fmt="o", ms=2.5, color="#17404f", ecolor="#17404f",
                        elinewidth=0.8, capsize=0)
        bottom.set_ylim(*_ylim(fm, fe))
    bottom.axvspan(-hw, hw, color="#b8791f", alpha=0.10, lw=0)
    bottom.axvline(0.5, color="#a83f2e", ls="--", lw=1)
    bottom.text(0.5, 0.97, " phase 0.5", transform=bottom.get_xaxis_transform(), color="#a83f2e",
                fontsize=8, va="top", ha="left")
    bottom.set_xlim(-0.25, 0.75)
    bottom.set_xlabel(f"Orbital phase (whole orbit, {lay['full_bin_minutes']:.1f}-min bins)")
    bottom.set_ylabel("Normalized flux")

    for ax in (top, bottom):
        ax.ticklabel_format(axis="y", useOffset=False)   # absolute flux, not "+1" offsets

    fig.suptitle(f"{host}, folded at P = {period:.5f} d\nUnexamined candidate, not a confirmed planet",
                 fontsize=11)
    fig.savefig(out_path, dpi=120, format="png")
    return lay
