"""completeness_snr.py -- adds the SNR axis to the injection system's
completeness validation.

`completeness_curve.py` measured recovery against injected DEPTH and PERIOD.
Depth alone is not the quantity that determines detectability: a 500 ppm
transit is trivial on a quiet bright star and invisible on a noisy faint one.
The detection-relevant quantity is the total signal-to-noise accumulated over
every in-transit cadence,

    SNR = (depth / sigma_point) * sqrt(N_in_transit)

which is the standard expression used for transit-search completeness (and the
quantity TLS's own SDE tracks). Recovery should rise steeply through SNR ~7-10
and saturate above it. If it doesn't, the injector is not producing detectable
signals the way real transits are, and nothing downstream of it can be trusted.

This needs no TLS re-run. Every trial in `completeness_curve_results.csv`
records its source light curve, and `completeness_curve.py` fixed the transit
duration at 5% of the period (`FIXED_DURATION_FRACTION`), so N_in_transit and
sigma_point are both recoverable after the fact from the real file.

sigma_point is estimated from the point-to-point difference
(std(diff)/sqrt(2)) rather than the plain standard deviation. The light curves
are detrended but not perfectly flat, and any residual low-frequency
variability inflates a plain std while leaving the differenced estimate --
which is what actually limits a short-duration transit -- alone.
"""
import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
PROCESSED_NEGATIVE = os.path.join(ROOT, "data", "processed_negative")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "completeness_curve_results.csv")
OUT_CSV = os.path.join(SCRIPT_DIR, "completeness_snr_results.csv")

FIXED_DURATION_FRACTION = 0.05   # must match completeness_curve.py


def point_noise_and_n(fname):
    path = os.path.join(PROCESSED_NEGATIVE, fname)
    if not os.path.exists(path):
        return np.nan, np.nan, np.nan
    d = pd.read_csv(path)
    f = pd.to_numeric(d["flux"], errors="coerce").to_numpy(float)
    f = f[np.isfinite(f)]
    if len(f) < 50:
        return np.nan, np.nan, np.nan
    sigma = float(np.std(np.diff(f)) / np.sqrt(2.0))
    t = pd.to_numeric(d["time"], errors="coerce").to_numpy(float)
    t = t[np.isfinite(t)]
    baseline = float(t.max() - t.min()) if len(t) > 1 else np.nan
    return sigma, len(f), baseline


def main():
    df = pd.read_csv(RESULTS_CSV)
    print(f"{len(df)} completeness trials")

    cache = {}
    sig, npts, base = [], [], []
    for fn in df["source_file"]:
        if fn not in cache:
            cache[fn] = point_noise_and_n(fn)
        s, n, b = cache[fn]
        sig.append(s); npts.append(n); base.append(b)
    df["sigma_point"] = sig
    df["n_points"] = npts
    df["baseline_days"] = base

    # Fraction of all cadences that fall in transit: duration/period is fixed
    # at 5% by construction, so N_in = 0.05 * N_total for every trial. Written
    # out explicitly rather than hardcoded to 0.05 so the relationship stays
    # visible if the generating script's constant ever changes.
    duration_days = df["injected_period"] * FIXED_DURATION_FRACTION
    in_transit_frac = duration_days / df["injected_period"]
    df["n_in_transit"] = df["n_points"] * in_transit_frac

    depth_frac = df["injected_depth_ppm"] / 1e6
    df["expected_snr"] = (depth_frac / df["sigma_point"]) * np.sqrt(df["n_in_transit"])

    ok = df[np.isfinite(df["expected_snr"])].copy()
    print(f"{len(ok)} trials with a computable SNR "
          f"({len(df)-len(ok)} source files missing/unreadable)")
    print(f"median point noise: {ok['sigma_point'].median()*1e6:.0f} ppm | "
          f"median cadences: {ok['n_points'].median():.0f}")

    bins = [0, 5, 10, 20, 40, 80, np.inf]
    labels = ["<5", "5-10", "10-20", "20-40", "40-80", ">80"]
    ok["snr_bin"] = pd.cut(ok["expected_snr"], bins=bins, labels=labels)

    print("\n=== recovery rate vs expected SNR ===")
    g = ok.groupby("snr_bin", observed=True)["recovered"].agg(["mean", "count"])
    g["mean"] = (g["mean"] * 100).round(1)
    g.columns = ["recovery_%", "n_trials"]
    print(g.to_string())

    print("\n=== recovery rate vs injected depth (for reference) ===")
    g2 = ok.groupby("injected_depth_ppm")["recovered"].agg(["mean", "count"])
    g2["mean"] = (g2["mean"] * 100).round(1)
    g2.columns = ["recovery_%", "n_trials"]
    print(g2.to_string())

    # Where does the curve cross 50%? The standard "is this injector sane"
    # summary number, comparable to published transit-search completeness.
    rec = ok.sort_values("expected_snr")
    win = max(5, len(rec) // 10)
    roll = rec["recovered"].rolling(win, center=True, min_periods=3).mean()
    cross = rec.loc[roll[roll >= 0.5].index[:1], "expected_snr"]
    if len(cross):
        print(f"\n50% recovery reached around SNR ~ {float(cross.iloc[0]):.1f} "
              f"(rolling window of {win} trials)")

    ok.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
