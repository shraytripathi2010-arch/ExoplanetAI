"""weak_secondary.py -- the weak-secondary-eclipse test, done properly.

WHAT IS ALREADY IN THE MODEL, AND WHY THIS IS NOT A REPEAT

`secondary_eclipse_depth` has been a production feature since the v2 round. It
is a RAW DEPTH with no noise normalisation at all:

    secondary_eclipse_depth = 1 - median(flux[0.45 < phase < 0.55])
                                                (05c_extract_new_features.py:111)

Its single-feature AUC was 0.550 -- the pilot note that shipped with it said
"weak, likely needs a better estimator". The small-lift trio then tested
`secondary_ratio = secondary_eclipse_depth / depth_mean`, which normalises by
the PRIMARY depth, and the whole engineered-ratio arm returned +0.0001
[-0.0055, +0.0057].

Neither is the published test. Kepler DV and ExoMiner use a weak-secondary
statistic normalised by the LOCAL NOISE, not by the primary depth:

    sigma_eff  = 1.4826 * MAD(out-of-eclipse flux) / sqrt(N_in_window)
    significance = secondary_depth / sigma_eff

That distinction is the entire point. A 200 ppm dip is meaningless on a noisy
star and decisive on a quiet one; a depth ratio cannot tell those apart,
because it divides by a quantity that carries no information about the noise.
So the raw depth is in the model, the depth ratio has been tested and failed,
and the noise-normalised version -- the one the literature actually uses -- has
not been built. That is what this computes.

NO TLS RE-RUN. The folded phase is reconstructed from `period` and `T0`, both
already stored in training.csv, applied to the detrended light curves already
on disk. 05c had to re-run TLS only because it needed TLS's own internal
folded arrays; recomputing the fold from stored ephemerides costs milliseconds.

METHOD NOTES, so the numbers are interpretable
  * The secondary window is centred on phase 0.5 and is HALF THE PRIMARY
    DURATION wide (converted from the stored `duration`, in days, to phase
    units per star), rather than a fixed 0.45-0.55 slab. A fixed slab is 10%
    of the orbit, which for a 0.5% duty-cycle transit is ~20x too wide and
    dilutes any real dip toward zero -- a plausible reason the raw feature
    scored only 0.550.
  * The baseline excludes both the primary (phase near 0/1) and the secondary
    window itself, so the noise estimate is not inflated by the very signal
    being measured.
  * MAD, not standard deviation, because a handful of un-clipped outliers
    would otherwise dominate sigma on exactly the noisy stars where this test
    matters most.
"""
import os
import sys
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
PROC_DIRS = [os.path.join(ROOT, "data", "processed"),
             os.path.join(ROOT, "data", "processed_negative")]
OUT_CSV = os.path.join(SCRIPT_DIR, "weak_secondary_features.csv")

N_WORKERS = 8
MIN_IN_WINDOW = 5
MIN_BASELINE = 50


def _find(host):
    for d in PROC_DIRS:
        p = os.path.join(d, str(host) + ".csv")
        if os.path.exists(p):
            return p
    return None


def measure(args):
    """Weak-secondary statistics for one star. Returns NaNs rather than raising
    -- a star with too little data is a legitimate non-detection, not an error,
    and conflating the two would bias the class-rate comparison."""
    host, period, t0, duration = args
    out = {"host": host, "sec_depth_windowed": np.nan,
           "sec_significance": np.nan, "sec_sigma_eff": np.nan,
           "sec_n_in_window": 0, "sec_status": "ok"}
    path = _find(host)
    if path is None:
        out["sec_status"] = "no light curve"
        return out
    if not (np.isfinite(period) and period > 0 and np.isfinite(t0)):
        out["sec_status"] = "no ephemeris"
        return out
    try:
        df = pd.read_csv(path)
        t = df["time"].to_numpy(float)
        f = df["flux"].to_numpy(float)
    except Exception as e:
        out["sec_status"] = f"read error: {type(e).__name__}"
        return out
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if len(t) < MIN_BASELINE:
        out["sec_status"] = f"too few points ({len(t)})"
        return out

    phase = ((t - t0) / period) % 1.0

    # Window width from the star's own duration; fall back to a 1% duty cycle
    # when duration is missing, which is the median of this dataset.
    dur_phase = (duration / period) if (np.isfinite(duration) and duration > 0
                                        and duration < period) else 0.01
    half = max(0.5 * dur_phase, 0.002)          # never narrower than ~0.2% phase

    sec = np.abs(phase - 0.5) < half
    prim = (phase < half * 1.5) | (phase > 1.0 - half * 1.5)
    base = ~sec & ~prim

    out["sec_n_in_window"] = int(sec.sum())
    if sec.sum() < MIN_IN_WINDOW or base.sum() < MIN_BASELINE:
        out["sec_status"] = "insufficient window/baseline coverage"
        return out

    base_level = float(np.median(f[base]))
    sec_level = float(np.median(f[sec]))
    depth = base_level - sec_level

    mad = float(np.median(np.abs(f[base] - base_level)))
    sigma_pt = 1.4826 * mad
    if not np.isfinite(sigma_pt) or sigma_pt <= 0:
        out["sec_status"] = "degenerate noise estimate"
        return out
    sigma_eff = sigma_pt / np.sqrt(sec.sum())

    out["sec_depth_windowed"] = depth
    out["sec_sigma_eff"] = sigma_eff
    out["sec_significance"] = depth / sigma_eff
    return out


def main():
    df = pd.read_csv(TRAINING)
    jobs = list(zip(df["host"].astype(str),
                    pd.to_numeric(df["period"], errors="coerce"),
                    pd.to_numeric(df["T0"], errors="coerce"),
                    pd.to_numeric(df["duration"], errors="coerce")))
    print(f"Computing weak-secondary statistics for {len(jobs)} stars "
          f"across {N_WORKERS} workers (no TLS)...")

    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(measure, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 78)
    print("COVERAGE")
    print("=" * 78)
    for s, n in r["sec_status"].value_counts().items():
        print(f"  {n:>5}  {s}")

    m = df[["host", "label"]].copy()
    m["host"] = m["host"].astype(str)
    j = m.merge(r, on="host", how="left")
    ok = j["sec_significance"].notna()
    print(f"\n  usable: {int(ok.sum())}/{len(j)} ({100*ok.mean():.1f}%)")

    print("\n" + "=" * 78)
    print("CLASS-RATE CHECK -- raw distribution, before any model")
    print("=" * 78)
    print(f"  {'':<12}{'n':>7}{'median sig':>12}{'p90 sig':>10}"
          f"{'% >3 sig':>10}{'% >5 sig':>10}")
    for lab, name in [(1.0, "planets"), (0.0, "false pos")]:
        s = j.loc[ok & (j["label"] == lab), "sec_significance"]
        print(f"  {name:<12}{len(s):>7}{np.median(s):>12.2f}"
              f"{np.percentile(s, 90):>10.2f}"
              f"{100*(s > 3).mean():>10.1f}{100*(s > 5).mean():>10.1f}")
    print("\n  (a real planet should essentially never show a significant")
    print("   secondary; an eclipsing binary often should)")
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
