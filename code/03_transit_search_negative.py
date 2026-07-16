"""
03_transit_search_negative.py

Run Transit Least Squares (TLS) on the cleaned negative-class (TOI false
positive) light curves from 02_preprocess_negative.py. Identical search
logic and settings to 03_transit_search.py -- see that script's module
docstring for the full reasoning (TLS's internal multiprocessing and why
use_threads=1 is required, why batching is for stuck-star containment not
JIT amortization, why futures are awaited in submission order for a real
per-star timeout, etc.). Not re-derived here.

IMPORTANT: OVERSAMPLING_FACTOR/DURATION_GRID_STEP/R_star/M_star are kept
IDENTICAL to 03_transit_search.py on purpose. If the positive and negative
classes were searched with different settings, any difference in the
resulting SDE/period distributions between label=1 and label=0 could be an
artifact of methodology rather than a real astrophysical difference --
which would quietly corrupt whatever a classifier learns from these
features. Keep these two scripts' settings in sync if you ever tune one.

Author: Ray's Exoplanet AI Project
"""

import os
import time
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
import pandas as pd
from tqdm import tqdm
from transitleastsquares import transitleastsquares

warnings.filterwarnings("ignore")

# =====================================
# SETTINGS -- kept in sync with 03_transit_search.py, see module docstring.
# =====================================
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)
TLS_THREADS_PER_WORKER = 1
BATCH_SIZE = 20
PER_STAR_TIMEOUT = 900
ETA_WINDOW = 30
PROCESS_LIMIT = None

OVERSAMPLING_FACTOR = 1
DURATION_GRID_STEP = 1.1

DEFAULT_R_STAR = 1.0
DEFAULT_M_STAR = 1.0

# =====================================
# LARGE-STAR BINNING
#
# A real run showed 78/1,141 stars (~7%) have 50k-114k points, and TLS cost
# scales with point count -- these few outliers were dominating wall-clock
# time (17+ hours projected) while the other 93% of stars were fast.
#
# We deliberately do NOT lower OVERSAMPLING_FACTOR/DURATION_GRID_STEP to fix
# this: those already match the completed positive-class run
# (03_transit_search.py), and changing them here only would make the two
# classes' SDE/period values methodologically incomparable -- a much worse
# problem than a slow run. Instead, only stars ABOVE MAX_POINTS_BEFORE_BINNING
# get block-averaged down to roughly TARGET_POINTS_AFTER_BINNING points. This
# leaves the ~93% of typical-sized stars in both classes completely
# untouched, and only mildly coarsens time resolution for the rare huge
# multi-sector stars, whose sheer point density was providing diminishing
# returns anyway (still comfortably above what's needed to resolve any
# realistic transit duration).
#
# flux_err is combined in quadrature per bin (correct error propagation for
# averaging independent measurements), not just averaged.
# =====================================
MAX_POINTS_BEFORE_BINNING = 30000
TARGET_POINTS_AFTER_BINNING = 15000


def bin_lightcurve(time_arr, flux_arr, flux_err_arr):
    n = len(time_arr)
    if n <= MAX_POINTS_BEFORE_BINNING:
        return time_arr, flux_arr, flux_err_arr, False

    bin_factor = int(np.ceil(n / TARGET_POINTS_AFTER_BINNING))
    n_bins = n // bin_factor
    trimmed = n_bins * bin_factor

    t_binned = time_arr[:trimmed].reshape(n_bins, bin_factor).mean(axis=1)
    f_binned = flux_arr[:trimmed].reshape(n_bins, bin_factor).mean(axis=1)
    e_binned = np.sqrt((flux_err_arr[:trimmed].reshape(n_bins, bin_factor) ** 2).sum(axis=1)) / bin_factor

    return t_binned, f_binned, e_binned, True


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "processed_negative")
CATALOG_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "catalogs")
RESULTS_PATH = os.path.join(CATALOG_FOLDER, "transit_search_results_negative.csv")
LOG_PATH = os.path.join(CATALOG_FOLDER, "transit_search_log_negative.csv")

os.makedirs(CATALOG_FOLDER, exist_ok=True)


def validate_input(df):
    if not {"time", "flux", "flux_err"}.issubset(df.columns):
        return "Missing expected columns (time/flux/flux_err)"
    if len(df) < 50:
        return f"Too few points ({len(df)}) for a meaningful search"
    if not np.all(np.isfinite(df["time"])) or not np.all(np.isfinite(df["flux"])):
        return "Non-finite values in time or flux"
    return None


def search_one_star(csv_path):
    host = os.path.splitext(os.path.basename(csv_path))[0]
    t0 = time.monotonic()

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"host": host, "status": f"Read error: {e}", "elapsed_s": time.monotonic() - t0}

    problem = validate_input(df)
    if problem:
        return {"host": host, "status": f"Skipped: {problem}", "elapsed_s": time.monotonic() - t0}

    t_arr, f_arr, e_arr, was_binned = bin_lightcurve(
        df["time"].to_numpy(), df["flux"].to_numpy(), df["flux_err"].to_numpy()
    )

    try:
        model = transitleastsquares(t_arr, f_arr, e_arr)
        r = model.power(
            use_threads=TLS_THREADS_PER_WORKER,
            oversampling_factor=OVERSAMPLING_FACTOR,
            duration_grid_step=DURATION_GRID_STEP,
            R_star=DEFAULT_R_STAR,
            M_star=DEFAULT_M_STAR,
            show_progress_bar=False,
        )
    except Exception as e:
        return {"host": host, "status": f"TLS error: {e}", "elapsed_s": time.monotonic() - t0}

    result = {
        "host": host,
        "status": "Success",
        "n_points": len(t_arr),
        "n_points_original": len(df),
        "was_binned": was_binned,
        "SDE": r.SDE,
        "SDE_raw": r.SDE_raw,
        "FAP": r.FAP,
        "period": r.period,
        "period_uncertainty": r.period_uncertainty,
        "T0": r.T0,
        "duration": r.duration,
        "depth": r.depth,
        "depth_mean": r.depth_mean[0],
        "depth_mean_std": r.depth_mean[1],
        "depth_mean_even": r.depth_mean_even[0],
        "depth_mean_odd": r.depth_mean_odd[0],
        "odd_even_mismatch": r.odd_even_mismatch,
        "rp_rs": r.rp_rs,
        "snr": r.snr,
        "transit_count": r.transit_count,
        "distinct_transit_count": r.distinct_transit_count,
        "empty_transit_count": r.empty_transit_count,
        "elapsed_s": time.monotonic() - t0,
    }
    return result


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def save_progress(log_rows, result_rows):
    log_df = pd.DataFrame(log_rows)
    if os.path.exists(LOG_PATH):
        old_log = pd.read_csv(LOG_PATH)
        if set(old_log.columns) == set(log_df.columns):
            log_df = pd.concat([old_log, log_df], ignore_index=True).drop_duplicates(subset="host", keep="last")
        else:
            print(f"WARNING: existing {LOG_PATH} has a different column schema -- replacing rather than merging.")
    log_df.to_csv(LOG_PATH, index=False)

    if result_rows:
        results_df = pd.DataFrame(result_rows).drop(columns=["status"])
        if os.path.exists(RESULTS_PATH):
            old_results = pd.read_csv(RESULTS_PATH)
            if set(old_results.columns) == set(results_df.columns):
                results_df = pd.concat([old_results, results_df], ignore_index=True).drop_duplicates(subset="host", keep="last")
            else:
                print(f"WARNING: existing {RESULTS_PATH} has a different column schema -- replacing rather than merging.")
        results_df.to_csv(RESULTS_PATH, index=False)


def main():
    print(f"Settings: OVERSAMPLING_FACTOR={OVERSAMPLING_FACTOR}, DURATION_GRID_STEP={DURATION_GRID_STEP}, "
          f"MAX_WORKERS={MAX_WORKERS}, BATCH_SIZE={BATCH_SIZE}")

    all_files = sorted(f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv"))
    print(f"{len(all_files)} cleaned light curves found in {INPUT_FOLDER}")

    already_done = set()
    if os.path.exists(LOG_PATH):
        old_log = pd.read_csv(LOG_PATH)
        already_done = set(old_log.loc[old_log["status"] == "Success", "host"])
        print(f"{len(already_done)} stars already have a successful TLS result -- skipping. "
              f"(Timed-out/errored stars from a previous run will be retried.)")

    work_list = [
        os.path.join(INPUT_FOLDER, f) for f in all_files
        if os.path.splitext(f)[0] not in already_done
    ]

    if PROCESS_LIMIT is not None:
        work_list = work_list[:PROCESS_LIMIT]
        print(f"PROCESS_LIMIT is set -- only running on {len(work_list)} stars (timing test).")

    print(f"\n{len(work_list)} stars to search with {MAX_WORKERS} worker processes "
          f"(TLS use_threads={TLS_THREADS_PER_WORKER} per worker), in batches of {BATCH_SIZE}...\n")

    recent_durations = deque(maxlen=ETA_WINDOW)
    all_status_rows = []

    progress = tqdm(total=len(work_list), desc="Transit search")
    stars_done = 0

    for batch in chunked(work_list, BATCH_SIZE):
        batch_log_rows = []
        batch_results = []

        executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
        try:
            futures = [(executor.submit(search_one_star, path), path) for path in batch]

            for future, path in futures:
                host = os.path.splitext(os.path.basename(path))[0]
                try:
                    r = future.result(timeout=PER_STAR_TIMEOUT)
                except FutureTimeoutError:
                    r = {"host": host, "status": f"Timed out after {PER_STAR_TIMEOUT}s", "elapsed_s": PER_STAR_TIMEOUT}
                except Exception as e:
                    r = {"host": host, "status": f"Worker error: {e}", "elapsed_s": None}

                batch_log_rows.append(r)
                all_status_rows.append(r)
                if r["status"] == "Success":
                    batch_results.append(r)
                if r.get("elapsed_s") is not None:
                    recent_durations.append(r["elapsed_s"])

                stars_done += 1
                progress.update(1)
                if recent_durations and stars_done % 10 == 0:
                    sd = sorted(recent_durations)
                    median = sd[len(sd) // 2]
                    remaining = len(work_list) - stars_done
                    eta_seconds = (median * remaining) / MAX_WORKERS
                    progress.set_postfix({
                        "median_s": f"{median:.1f}",
                        "eta": f"{eta_seconds/60:.0f}m",
                    })
        finally:
            executor.shutdown(wait=False)

        save_progress(batch_log_rows, batch_results)

    progress.close()

    log_df = pd.DataFrame(all_status_rows)
    status_counts = log_df["status"].apply(
        lambda s: "Success" if s == "Success"
        else ("Skipped" if str(s).startswith("Skipped")
        else ("Timed out" if str(s).startswith("Timed out") else "Error"))
    ).value_counts()

    print("\n===================================")
    print("Finished")
    print("===================================")
    for status, count in status_counts.items():
        print(f"{status}: {count}")
    print(f"Results: {RESULTS_PATH}")
    print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
