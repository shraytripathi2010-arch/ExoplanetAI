"""
compute_new_features.py -- the closing feature experiment for the
classical model (per explicit user framing: exactly one more round after
eight prior experiments already converged on a ~0.90 ceiling).

Two genuinely different-information features, both requiring a fresh TLS
rerun since transit_search_results.csv only ever saved scalar summary
columns, not the per-transit arrays or full periodogram these need:

1. multi_transit_depth_chi2red -- uncertainty-weighted consistency of
   individual transit depths (TLS's own transit_depths +
   transit_depths_uncertainties arrays, confirmed live to exist and be
   populated -- neither was ever used before; the existing
   depth_consistency_std feature is a RAW std that ignores per-transit
   noise, this is a proper reduced chi-square against each transit's own
   measurement uncertainty). A real planet's transit-to-transit depth
   variation should be consistent with measurement noise (chi2red ~ 1);
   contamination/blends often show excess real scatter (chi2red >> 1).

2. power_ratio_half_period / power_ratio_double_period -- TLS's full
   periodogram (periods/power arrays) evaluated at exactly half and
   double the detected period, relative to power at the detected period
   itself. The classic aliasing/eclipsing-binary signature: a real
   periodic box-shaped dip should dominate its own period; strong power
   at P/2 or 2P suggests the true signal period differs from what was
   reported (e.g. an EB where TLS locked onto twice the true orbital
   period because primary/secondary eclipses look similar).

Reruns TLS on already-processed light curves (no new downloads needed --
100% coverage confirmed live for both classes). Parallelized + checkpointed,
matching this project's established pattern for multi-hour batch compute.
"""
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
TRAINING_CSV = os.path.join(PROJECT_ROOT, "data", "training_dataset", "training.csv")
PROCESSED_POS_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PROCESSED_NEG_DIR = os.path.join(PROJECT_ROOT, "data", "processed_negative")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "new_features_results.csv")

N_WORKERS = 8
PER_JOB_TIMEOUT_S = 300  # backstop, same lesson as the centroid batch


def _nearest_power(periods, power, target_period):
    """Nearest-grid-point power value at target_period. Returns NaN if
    target_period falls outside the searched grid range -- genuinely
    missing, not guessed."""
    if target_period < periods.min() or target_period > periods.max():
        return np.nan
    idx = np.argmin(np.abs(periods - target_period))
    return float(power[idx])


def _worker(args):
    host, label, path, st_rad, st_mass = args
    t0 = time.time()
    try:
        from transitleastsquares import transitleastsquares

        lc = pd.read_csv(path)
        r_star = st_rad if pd.notna(st_rad) and st_rad > 0 else 1.0
        m_star = st_mass if pd.notna(st_mass) and st_mass > 0 else 1.0
        r_star_min, r_star_max = min(0.13, r_star * 0.5), max(3.5, r_star * 1.5)
        m_star_min, m_star_max = min(0.1, m_star * 0.5), max(1.0, m_star * 1.5)

        model = transitleastsquares(lc["time"].to_numpy(), lc["flux"].to_numpy(), lc["flux_err"].to_numpy())
        r = model.power(
            use_threads=1, oversampling_factor=1, duration_grid_step=1.1, show_progress_bar=False,
            R_star=r_star, R_star_min=r_star_min, R_star_max=r_star_max,
            M_star=m_star, M_star_min=m_star_min, M_star_max=m_star_max,
        )

        # Feature 1: uncertainty-weighted depth consistency (chi2red)
        depths = np.asarray(r.transit_depths, dtype=float)
        depth_errs = np.asarray(r.transit_depths_uncertainties, dtype=float)
        valid = np.isfinite(depths) & np.isfinite(depth_errs) & (depth_errs > 0)
        depths, depth_errs = depths[valid], depth_errs[valid]
        if len(depths) >= 2:
            weights = 1.0 / depth_errs ** 2
            weighted_mean = np.sum(depths * weights) / np.sum(weights)
            chi2 = np.sum(((depths - weighted_mean) ** 2) / (depth_errs ** 2))
            chi2red = float(chi2 / (len(depths) - 1))
        else:
            chi2red = np.nan

        # Feature 2: periodogram power at period/2 and period*2, relative
        # to power at the detected best period.
        periods = np.asarray(r.periods, dtype=float)
        power = np.asarray(r.power, dtype=float)
        best_period = float(r.period)
        power_at_best = _nearest_power(periods, power, best_period)
        power_at_half = _nearest_power(periods, power, best_period / 2.0)
        power_at_double = _nearest_power(periods, power, best_period * 2.0)
        ratio_half = (power_at_half / power_at_best) if (power_at_best and power_at_best > 0) else np.nan
        ratio_double = (power_at_double / power_at_best) if (power_at_best and power_at_best > 0) else np.nan

        return {
            "host": host, "label": label, "status": "Success", "elapsed_s": time.time() - t0,
            "multi_transit_depth_chi2red": chi2red, "n_transits_used": len(depths),
            "power_ratio_half_period": ratio_half, "power_ratio_double_period": ratio_double,
        }
    except Exception as e:
        return {"host": host, "label": label, "status": f"failed: {e}", "elapsed_s": time.time() - t0,
                "multi_transit_depth_chi2red": None, "n_transits_used": None,
                "power_ratio_half_period": None, "power_ratio_double_period": None}


def main():
    df = pd.read_csv(TRAINING_CSV)
    jobs = []
    for _, row in df.iterrows():
        host = row["host"]
        folder = PROCESSED_POS_DIR if row["label"] == 1 else PROCESSED_NEG_DIR
        path = os.path.join(folder, host + ".csv")
        if not os.path.exists(path):
            continue
        jobs.append((host, int(row["label"]), path, row.get("st_rad"), row.get("st_mass")))

    results = []
    already_done = set()
    if os.path.exists(RESULTS_PATH):
        results = pd.read_csv(RESULTS_PATH).to_dict("records")
        already_done = {r["host"] for r in results}
        print(f"{len(already_done)} already done -- resuming.")
    jobs = [j for j in jobs if j[0] not in already_done]

    print(f"{len(jobs)} jobs to run across {N_WORKERS} workers "
          f"(calibrated ~14.6s/star average, so expect ~{len(jobs)*14.6/N_WORKERS/60:.0f} min)...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                res = future.result(timeout=PER_JOB_TIMEOUT_S)
            except FutureTimeoutError:
                res = {"host": job[0], "label": job[1], "status": "failed: timed out",
                       "elapsed_s": PER_JOB_TIMEOUT_S, "multi_transit_depth_chi2red": None,
                       "n_transits_used": None, "power_ratio_half_period": None,
                       "power_ratio_double_period": None}
            except Exception as e:
                res = {"host": job[0], "label": job[1], "status": f"failed: worker crashed: {e}",
                       "elapsed_s": None, "multi_transit_depth_chi2red": None,
                       "n_transits_used": None, "power_ratio_half_period": None,
                       "power_ratio_double_period": None}
            results.append(res)
            if i % 50 == 0 or i == len(jobs):
                elapsed = time.time() - t0
                n_ok = sum(1 for r in results if r["status"] == "Success")
                print(f"  [{i}/{len(jobs)}] done ({elapsed:.0f}s elapsed, "
                      f"~{elapsed/i*(len(jobs)-i):.0f}s remaining, {n_ok}/{len(results)} usable)", flush=True)
                pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)

    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    df_r = pd.DataFrame(results)
    print(f"\nTotal wall time: {time.time()-t0:.0f}s for {len(jobs)} jobs")
    print(f"{(df_r['status']=='Success').sum()}/{len(df_r)} succeeded")
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
