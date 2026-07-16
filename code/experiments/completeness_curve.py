"""
completeness_curve.py -- validates the injection system (injection.py) the
same way Hippke & Heller validated TLS itself in the original paper this
pipeline is built on: inject known synthetic signals into real light curves,
run the SAME real TLS invocation the production pipeline uses, and measure
the fraction correctly recovered as a function of depth/period/SNR.

This is Part B item 3 -- must pass before trusting the injector for anything
downstream (CNN training data, classical-model augmentation).
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import injection as inj

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "completeness_curve_results.csv")

# Grid chosen to span the real parameter ranges found in the actual positive
# training data (see injection.py's _load_real_param_distributions), not
# arbitrary values.
GRID_PERIODS = [1.0, 3.0, 5.0, 8.0, 12.0]
GRID_DEPTHS_PPM = [200, 500, 1000, 3000, 8000]
N_REPEATS = 4
FIXED_DURATION_FRACTION = 0.05  # duration as a fraction of period -- typical real ratio

PERIOD_MATCH_TOLERANCE = 0.01  # 1% -- matches the tolerance style used elsewhere in this project


def _period_recovered(injected_period, recovered_period, tol=PERIOD_MATCH_TOLERANCE):
    """Real signals can be recovered at an alias (half/double/etc. the true
    period) -- Hippke & Heller's own validation accounted for this, so this
    does too, but reports which case happened rather than merging them."""
    if recovered_period is None or not np.isfinite(recovered_period) or recovered_period <= 0:
        return False, None
    for n, name in [(1, "exact"), (2, "double"), (0.5, "half"), (3, "triple"), (1/3, "third")]:
        target = injected_period * n
        if abs(recovered_period - target) / target < tol:
            return True, name
    return False, None


def _run_one(args):
    """One injection + real TLS recovery attempt. Must be a top-level
    function (not a closure) to be picklable for ProcessPoolExecutor."""
    period, depth_ppm, repeat_idx, seed = args
    from transitleastsquares import transitleastsquares

    rng = np.random.default_rng(seed)
    files = inj.list_real_negative_lightcurves()
    fname = files[rng.integers(0, len(files))]
    time_arr, flux_arr, err_arr = inj.load_real_lightcurve(fname)

    duration_days = period * FIXED_DURATION_FRACTION
    injected_flux, params = inj.inject_transit(time_arr, flux_arr, period, depth_ppm, duration_days, rng)

    t0 = time.monotonic()
    try:
        model = transitleastsquares(time_arr, injected_flux, err_arr)
        r = model.power(use_threads=1, oversampling_factor=1, duration_grid_step=1.1, show_progress_bar=False)
        recovered_period = float(r.period)
        recovered_sde = float(r.SDE)
    except Exception as e:
        recovered_period, recovered_sde = None, None
        print(f"TLS error for {fname} P={period} depth={depth_ppm}: {e}", file=sys.stderr)
    elapsed = time.monotonic() - t0

    recovered, alias = _period_recovered(period, recovered_period)
    return {
        "injected_period": period, "injected_depth_ppm": depth_ppm, "repeat": repeat_idx,
        "source_file": fname, "recovered_period": recovered_period, "recovered_sde": recovered_sde,
        "recovered": recovered, "alias": alias, "elapsed_s": elapsed,
    }


def main():
    jobs = []
    seed_counter = 1000
    for period in GRID_PERIODS:
        for depth in GRID_DEPTHS_PPM:
            for rep in range(N_REPEATS):
                jobs.append((period, depth, rep, seed_counter))
                seed_counter += 1

    print(f"Running {len(jobs)} injection+recovery trials across 8 workers...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_run_one, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            results.append(res)
            if i % 10 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] done ({time.time()-t0:.0f}s elapsed)")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nTotal wall time: {time.time()-t0:.0f}s for {len(jobs)} trials")
    print(f"Saved raw results to {RESULTS_PATH}")

    print("\n=== Overall recovery rate ===")
    print(f"{df['recovered'].mean()*100:.1f}% ({df['recovered'].sum()}/{len(df)})")

    print("\n=== Recovery rate by depth (ppm) ===")
    print(df.groupby("injected_depth_ppm")["recovered"].agg(["mean", "count"]))

    print("\n=== Recovery rate by period (days) ===")
    print(df.groupby("injected_period")["recovered"].agg(["mean", "count"]))

    print("\n=== Alias breakdown (among recovered) ===")
    print(df[df["recovered"]]["alias"].value_counts())


if __name__ == "__main__":
    main()
