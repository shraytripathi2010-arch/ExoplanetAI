"""
03_transit_search.py

Run Transit Least Squares (TLS) on the cleaned light curves from
02_preprocess.py to search for periodic transit signals.

IMPORTANT STRUCTURAL NOTE ABOUT TLS AND CONCURRENCY (read before changing
MAX_WORKERS or TLS_THREADS_PER_WORKER):

  1. TLS's own `.power()` defaults `use_threads` to `multiprocessing.cpu_count()`
     -- i.e. by default, a SINGLE call spawns its own internal multiprocessing
     Pool using ALL cores. Wrapping that in an outer process pool would cause
     severe oversubscription (N outer workers x cpu_count() inner processes
     each). We avoid this by always calling TLS with use_threads=1 inside our
     own outer ProcessPoolExecutor -- confirmed by reading TLS's source
     (transitleastsquares/main.py) before writing this script, not assumed.

  2. We initially suspected repeated numba JIT compilation (TLS's core
     functions have no `cache=True`, confirmed by grep) was inflating
     per-star cost, and that long-lived workers would amortize it. Tested
     this directly: two back-to-back `.power()` calls on the same
     ~15,700-point light curve, same process, use_threads=1 -- call 1 took
     158.1s, call 2 took 168.2s. Essentially identical, so JIT warmup is NOT
     the dominant cost here; the periodogram search itself is just genuinely
     this slow. Keeping workers long-lived across a batch is still a
     reasonable default (there's no cost to it), but don't expect it to be a
     meaningful speedup -- the real lever is OVERSAMPLING_FACTOR /
     DURATION_GRID_STEP below.

  3. Despite (2), a worker pool can't live forever: if a single pathological
     star (this dataset has some with 180k+ points) hangs or runs
     extremely long, that worker would be stuck for the rest of the run.
     We bound this by processing stars in BATCHES, each with its own
     ProcessPoolExecutor -- a stuck star can only cost you the rest of its
     own batch, not the whole remaining dataset, and the next batch starts
     with a clean, unstuck pool.

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
# SETTINGS
# =====================================
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)  # leave one core free
TLS_THREADS_PER_WORKER = 1        # MUST stay 1 -- see note (1) above
BATCH_SIZE = 20                    # stars per worker-pool lifetime -- see note (2)/(3).
                                   # Lowered from 150 after a real crash showed batch-level
                                   # checkpointing was too coarse (up to ~1 hour of unsaved
                                   # work lost). Since the numba-JIT-amortization rationale for
                                   # large batches was already disproven (see note 2), there's
                                   # no real downside to smaller batches -- at ~163s/star median
                                   # with 7 workers, 20 stars is ~8 minutes per checkpoint
                                   # instead of ~58 minutes.
PER_STAR_TIMEOUT = 900            # seconds; safety net for pathological stars, not the typical case
ETA_WINDOW = 30                   # rolling window (stars) for the median-based ETA
PROCESS_LIMIT = None              # set to e.g. 5 for a quick timing test before the full run

# TLS search settings. Lowered from TLS's published defaults (oversampling_factor=3,
# duration_grid_step=1.1) for speed, after the full run proved impractically slow
# (~27h) at defaults. oversampling_factor=1 is still within TLS's documented
# acceptable range, trading finer period-grid resolution for roughly a 3x speedup.
# NOTE: this was changed after ~44 stars had already completed under the old
# (denser) settings -- their SDE/period values used a finer grid than everything
# search afterward. Not perfectly consistent across the full dataset, but the
# affected stars are a small, identifiable fraction (check transit_search_log.csv
# for anything logged before this change) if it matters for later analysis.
OVERSAMPLING_FACTOR = 1
DURATION_GRID_STEP = 1.1

# TLS uses these to build a physically-informed period/duration grid. The
# NASA Exoplanet Archive catalog fetched in stage 1 only has hostname/ra/dec
# -- no stellar radius or mass -- so every star uses TLS's solar defaults
# here. This is a known simplification, not an oversight: for host stars far
# from solar (M dwarfs, giants), a per-star R_star/M_star from the archive's
# st_rad/st_mass columns would give a more accurate grid. Left as a follow-up.
DEFAULT_R_STAR = 1.0
DEFAULT_M_STAR = 1.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "processed")
CATALOG_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "catalogs")
RESULTS_PATH = os.path.join(CATALOG_FOLDER, "transit_search_results.csv")
LOG_PATH = os.path.join(CATALOG_FOLDER, "transit_search_log.csv")

os.makedirs(CATALOG_FOLDER, exist_ok=True)


def validate_input(df):
    """Defensive check -- stage 2 already guarantees this schema, but this
    project has repeatedly found it worth verifying rather than assuming."""
    if not {"time", "flux", "flux_err"}.issubset(df.columns):
        return "Missing expected columns (time/flux/flux_err)"
    if len(df) < 50:
        return f"Too few points ({len(df)}) for a meaningful search"
    if not np.all(np.isfinite(df["time"])) or not np.all(np.isfinite(df["flux"])):
        return "Non-finite values in time or flux"
    return None


# =====================================
# PER-STAR WORKER
#
# Top-level function so it can be pickled to worker processes. use_threads=1
# is critical here -- see the module docstring's note (1).
# =====================================
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

    try:
        model = transitleastsquares(df["time"].to_numpy(), df["flux"].to_numpy(), df["flux_err"].to_numpy())
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
        "n_points": len(df),
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
    """Merge this batch's outcomes into the on-disk log/results CSVs.

    Called after EVERY batch, not just once at the end of main(). A run here
    can take on the order of a day; only persisting at the very end would
    mean an interrupted or crashed run loses all progress, not just the
    in-flight batch. This makes each batch (currently 150 stars, a few
    minutes to ~1 hour of work) the actual unit of resumability."""
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


# =====================================
# MAIN DRIVER
# =====================================
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
    all_status_rows = []  # every outcome across the whole run, for the final summary print

    progress = tqdm(total=len(work_list), desc="Transit search")
    stars_done = 0

    for batch in chunked(work_list, BATCH_SIZE):
        batch_log_rows = []
        batch_results = []

        # A fresh executor per batch: bounds the damage of a stuck star (note 3 in
        # the module docstring) to at most this batch. (We do NOT recreate this
        # per-batch for numba JIT amortization -- tested and ruled out, see note 2.)
        executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
        try:
            # Submit the whole batch up front, then wait on futures in SUBMISSION
            # order (not as_completed). This is deliberate: stage 1 of this
            # project had a bug where future.result(timeout=...) was called
            # after as_completed() had already resolved the future, making the
            # timeout a no-op that could never fire. Waiting in submission order
            # means we call .result(timeout=...) on futures that may still be
            # running, so the timeout is real -- and since the pool keeps all
            # MAX_WORKERS workers busy pulling from the same queue regardless of
            # which future we happen to be waiting on, this costs no concurrency.
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
            # wait=False: if a star in this batch is still stuck when we move on,
            # don't block the whole script waiting for it to finish or error out.
            executor.shutdown(wait=False)

        # Persist after every batch (see save_progress's docstring for why this
        # can't wait until the whole run finishes).
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

# =====================================
# HONEST RUNTIME NOTE (measured, not guessed)
#
# Two direct benchmarks on real data, single-threaded (use_threads=1), at the
# OVERSAMPLING_FACTOR/DURATION_GRID_STEP defaults above:
#   - Smallest star in the dataset (762 points):        20.9s
#   - Median-sized star (15,764 points, dataset median
#     is ~16,700):                                       158-168s (two runs, consistent)
#
# Extrapolating from the median-star number: ~4,100 stars x ~163s / 7 workers
# (MAX_WORKERS on an 8-core machine) ~= 95,000s ~= ~26 HOURS for the full
# dataset. This is a real estimate, not a hedge -- but it's driven by the
# MEDIAN star; this dataset has some stars with 100k-180k points (per stage
# 2's QC summary), and those will individually take much longer, pushing the
# tail of the distribution well past 26 hours of wall time for the run as a
# whole.
#
# If ~26+ hours isn't practical, in order of effectiveness:
#   1. Lower OVERSAMPLING_FACTOR (3 -> 1 or 2): roughly proportional speedup,
#      since it directly shrinks the period grid size.
#   2. Raise DURATION_GRID_STEP (1.1 -> ~1.4-2.0): coarser duration grid.
#   3. Constrain period_min/period_max in the model.power() call if you only
#      care about a specific period range (e.g. short-period hot Jupiters).
# Any of these trade search sensitivity/resolution for speed -- that's a
# scientific call, not one to make silently, which is why the defaults above
# are TLS's own published values rather than pre-loosened.
#
# STRONGLY RECOMMENDED: set PROCESS_LIMIT = 5 and run once first to get a
# real number for your machine (check elapsed_s in transit_search_log.csv)
# before committing to the full dataset overnight (or longer).
# =====================================
