"""
02_preprocess.py

Clean raw TESS light curve CSVs (from 01_download_known.py) before running
Transit Least Squares on them: pick the best flux column, drop bad-quality
points, remove outliers, flatten stellar variability, and normalize.

This is CPU/disk-bound (no network calls), so it uses ProcessPoolExecutor
for real parallelism across files -- see the "WHY ProcessPoolExecutor" note
near the bottom for the reasoning and how this differs from stage 1's
network-timeout problem.

Author: Ray's Exoplanet AI Project
"""

import os
import time
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import savgol_filter
from astropy.stats import sigma_clip

# =====================================
# SETTINGS
# =====================================
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)   # leave one core free for the OS/UI
SIGMA_CLIP_THRESHOLD = 5          # MAD-based sigma threshold for outlier removal
MAX_FLATTEN_WINDOW = 401          # cap on the Savitzky-Golay window, in points
SAVGOL_POLYORDER = 2              # quadratic trend -- smooth, won't chase individual points
MIN_POINTS_FOR_FLATTEN = 50       # below this, flattening is unreliable; skip and log
ETA_WINDOW = 100                  # rolling window (files) used for the median-based ETA
PROCESS_LIMIT = None              # set to e.g. 10 for a quick dry run before the full 4,700

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "known_lightcurves")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "processed")
CATALOG_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "catalogs")
LOG_PATH = os.path.join(CATALOG_FOLDER, "preprocess_log.csv")
QC_PATH = os.path.join(CATALOG_FOLDER, "preprocess_qc_summary.csv")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CATALOG_FOLDER, exist_ok=True)

# =====================================
# SCHEMA VALIDATION
#
# A real survey of the ~4,530 files currently downloaded found 8 distinct
# column schemas, not one: alongside the standard SPOC pipeline (pdcsap_flux,
# numeric quality bitmask), there are CDIPS files (flux in MAGNITUDES, not
# normalized flux; quality is a single-letter STRING code like 'G' -- a naive
# `quality != 0` filter on a string column is True for every row in pandas,
# which would silently zero out the whole star with no error), QLP files,
# and several custom/FFI pipelines with raw counts or sentinel fill values
# (e.g. -9.2e18) mixed into an otherwise-real flux column.
#
# Rather than trying to normalize every one of these incompatible unit
# systems into one pipeline (high risk of exactly the kind of silent
# corruption this project already got burned by once), we only process the
# standard SPOC schema and explicitly skip+log everything else with the
# detected reason. This was a deliberate, confirmed decision, not a
# corner-cut -- see conversation history for the full schema breakdown.
# =====================================
REQUIRED_COLUMNS = {"time", "flux", "flux_err", "quality", "pdcsap_flux", "pdcsap_flux_err"}


def validate_schema(df):
    """Returns None if df matches the expected SPOC schema, else a reason string."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return f"Non-standard schema (missing columns: {sorted(missing)})"
    if df["quality"].dtype == object:
        return "Non-standard schema (quality column is non-numeric, e.g. CDIPS-style flags)"
    if len(df) == 0:
        return "Empty file (zero rows)"
    return None


def choose_flux_columns(df):
    """Prefer pdcsap_flux (NASA's instrumentally-corrected flux); fall back to flux
    if pdcsap_flux is missing or entirely NaN for this star. Returns (flux, flux_err,
    source_name) as numpy arrays, or (None, None, reason) if nothing usable exists."""
    if df["pdcsap_flux"].notna().any():
        return df["pdcsap_flux"].to_numpy(), df["pdcsap_flux_err"].to_numpy(), "pdcsap_flux"
    if df["flux"].notna().any():
        return df["flux"].to_numpy(), df["flux_err"].to_numpy(), "flux"
    return None, None, "No usable flux data (both pdcsap_flux and flux are entirely NaN)"


def choose_savgol_window(n_points, max_window=MAX_FLATTEN_WINDOW, polyorder=SAVGOL_POLYORDER):
    """Adaptive window: capped at max_window, but must stay odd and STRICTLY smaller
    than the number of points. (A prior draft allowed window == n_points for small
    stars, which scipy accepts but silently degrades into fitting one global
    polynomial over the whole light curve instead of a local smoothing filter --
    this formula rules that out structurally.) Returns None if there aren't enough
    points left to flatten meaningfully."""
    window = min(max_window, n_points - 1)
    if window % 2 == 0:
        window -= 1
    if window < polyorder + 2 or window < 5:
        return None
    return window


def output_is_valid(path):
    """Resume check: verify the saved file actually has the expected columns AND
    at least one data row -- mirrors the stage-1 lesson that a file existing on
    disk is not proof it was written correctly."""
    try:
        df = pd.read_csv(path)
        return {"time", "flux", "flux_err"}.issubset(df.columns) and len(df) > 0
    except Exception:
        return False


# =====================================
# PER-FILE WORKER
#
# Must be a top-level function (not a closure/lambda/method) so it can be
# pickled and sent to worker processes by ProcessPoolExecutor. All imports
# used here are at module level for the same reason.
# =====================================
def process_one_file(csv_path):
    filename = os.path.splitext(os.path.basename(csv_path))[0]
    t0 = time.monotonic()
    result = {
        "host": filename, "status": None,
        "n_original": None, "n_after_quality": None, "n_after_outliers": None,
        "n_final": None, "pct_removed": None, "pre_norm_median_flux": None,
        "flux_source": None, "elapsed_s": None,
    }

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        result["status"] = f"Read error: {e}"
        result["elapsed_s"] = time.monotonic() - t0
        return result

    schema_problem = validate_schema(df)
    if schema_problem:
        result["status"] = f"Skipped: {schema_problem}"
        result["elapsed_s"] = time.monotonic() - t0
        return result

    n_original = len(df)
    result["n_original"] = n_original

    flux, flux_err, source = choose_flux_columns(df)
    if flux is None:
        result["status"] = f"Skipped: {source}"
        result["elapsed_s"] = time.monotonic() - t0
        return result
    result["flux_source"] = source

    time_arr = df["time"].to_numpy()
    quality = df["quality"].to_numpy()

    # Step 1: remove NaNs (time, flux, or flux_err) -- do this before the quality
    # filter so "after_quality" reflects both cleaning steps together, matching
    # how these numbers are actually used downstream (as a running "what's left" count).
    valid = ~np.isnan(time_arr) & ~np.isnan(flux) & ~np.isnan(flux_err)
    time_arr, flux, flux_err, quality = time_arr[valid], flux[valid], flux_err[valid], quality[valid]

    # Step 2: quality flag filter. quality == 0 means no instrumental flag raised;
    # anything else means a cosmic ray hit, thruster fire, safe mode, etc.
    good_quality = quality == 0
    time_arr, flux, flux_err = time_arr[good_quality], flux[good_quality], flux_err[good_quality]
    n_after_quality = len(flux)
    result["n_after_quality"] = n_after_quality

    if n_after_quality < MIN_POINTS_FOR_FLATTEN:
        result["status"] = f"Skipped: only {n_after_quality} points survived quality filtering"
        result["n_after_outliers"] = n_after_quality
        result["n_final"] = 0
        result["elapsed_s"] = time.monotonic() - t0
        return result

    # Step 3: sort by time. Should already be sorted (verified true for a sample of
    # real files), but don't assume -- a single out-of-order chunk (e.g. from a
    # stitched multi-sector download) would silently corrupt both the sigma clip's
    # notion of "neighboring" points and the Savitzky-Golay trend.
    order = np.argsort(time_arr, kind="stable")
    time_arr, flux, flux_err = time_arr[order], flux[order], flux_err[order]

    # Step 4: MAD-based sigma clipping. Using astropy's sigma_clip with
    # stdfunc="mad_std" rather than hand-rolling the MAD math -- it's a
    # well-tested implementation, and this project has already been burned once
    # by a subtle hand-written bug, so lean on a library where possible.
    with warnings.catch_warnings():
        # Scoped to just this call: astropy warns on degenerate (e.g. constant)
        # input, which is expected/harmless here. We deliberately do NOT suppress
        # warnings globally (a prior draft did), since a RuntimeWarning elsewhere
        # -- e.g. divide-by-zero producing inf during flattening below -- is a
        # real signal worth seeing, not noise to silence.
        warnings.simplefilter("ignore")
        clip_result = sigma_clip(flux, sigma=SIGMA_CLIP_THRESHOLD, stdfunc="mad_std", maxiters=5, masked=True)
    keep = ~clip_result.mask
    time_arr, flux, flux_err = time_arr[keep], flux[keep], flux_err[keep]
    n_after_outliers = len(flux)
    result["n_after_outliers"] = n_after_outliers

    if n_after_outliers < MIN_POINTS_FOR_FLATTEN:
        result["status"] = f"Skipped: only {n_after_outliers} points survived outlier removal"
        result["n_final"] = 0
        result["elapsed_s"] = time.monotonic() - t0
        return result

    # QC stat: flux level in its original physical units, captured *before*
    # flattening/normalization -- lets you later tell a faint/noisy star apart
    # from a bright one purely from the QC summary, without re-opening the file.
    pre_norm_median_flux = float(np.median(flux))
    result["pre_norm_median_flux"] = pre_norm_median_flux

    # Step 5: detrend with Savitzky-Golay, then normalize to ~1.0.
    # The window adapts to how many points this star has (capped at
    # MAX_FLATTEN_WINDOW) so we don't over-smooth short light curves or
    # under-smooth long ones. Dividing by the trend both flattens long-term
    # stellar variability AND normalizes the flux near 1.0 in one step -- and
    # flux_err is divided by the same trend to keep the relative uncertainty
    # meaningful (this mirrors what lightkurve's own .flatten() does).
    window = choose_savgol_window(n_after_outliers)
    if window is None:
        result["status"] = f"Skipped: {n_after_outliers} points too few for a valid flatten window"
        result["n_final"] = 0
        result["elapsed_s"] = time.monotonic() - t0
        return result

    trend = savgol_filter(flux, window_length=window, polyorder=SAVGOL_POLYORDER, mode="interp")
    flat_flux = flux / trend
    flat_err = flux_err / trend

    if not np.all(np.isfinite(flat_flux)):
        # A zero or near-zero trend value would produce inf/nan here -- this is
        # exactly the kind of silent corruption the RuntimeWarning suppression in
        # the previous draft could have hidden. Fail loudly instead of writing it.
        result["status"] = "Skipped: non-finite values in flattened flux (degenerate trend)"
        result["n_final"] = 0
        result["elapsed_s"] = time.monotonic() - t0
        return result

    # Residual normalization: the trend division should already put us near 1.0,
    # this just removes any small leftover offset so downstream TLS code can
    # assume a clean out-of-transit baseline of exactly 1.0.
    median_flat = np.median(flat_flux)
    flat_flux = flat_flux / median_flat
    flat_err = flat_err / median_flat

    n_final = len(flat_flux)
    result["n_final"] = n_final
    result["pct_removed"] = 100.0 * (1 - n_final / n_original) if n_original else None

    out_df = pd.DataFrame({"time": time_arr, "flux": flat_flux, "flux_err": flat_err})
    out_path = os.path.join(OUTPUT_FOLDER, filename + ".csv")
    out_df.to_csv(out_path, index=False)

    result["status"] = "Success"
    result["elapsed_s"] = time.monotonic() - t0
    return result


# =====================================
# MAIN DRIVER
#
# Wrapped in `if __name__ == "__main__":` because ProcessPoolExecutor on macOS
# uses the "spawn" start method, which re-imports this file fresh in every
# worker process. Without the guard, each worker would re-run the entire
# driver (rebuild the work queue, spawn its own pool, ...) recursively.
# =====================================
def main():
    all_files = sorted(f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv"))
    print(f"{len(all_files)} raw light curve files found in {INPUT_FOLDER}")

    work_list = []
    for f in all_files:
        out_path = os.path.join(OUTPUT_FOLDER, f)
        if not (os.path.exists(out_path) and output_is_valid(out_path)):
            work_list.append(os.path.join(INPUT_FOLDER, f))

    already_done = len(all_files) - len(work_list)
    if already_done:
        print(f"{already_done} files already correctly processed (valid time/flux/flux_err columns) -- skipping.")

    if PROCESS_LIMIT is not None:
        work_list = work_list[:PROCESS_LIMIT]
        print(f"PROCESS_LIMIT is set -- only running on {len(work_list)} files (dry run).")

    print(f"\n{len(work_list)} files to process with {MAX_WORKERS} worker processes...\n")

    log_rows = []
    qc_rows = []
    recent_durations = deque(maxlen=ETA_WINDOW)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_file, path): path for path in work_list}

        # NOTE on concurrency correctness: stage 1 had a bug where
        # future.result(timeout=...) was called after as_completed() had
        # already resolved the future, making the timeout a no-op that could
        # never fire. We deliberately do NOT add a per-file timeout wrapper
        # here at all, for a different reason than that bug: this is local
        # CPU/disk work with no network round-trip, so there's no hang risk
        # to guard against in the first place (measured worst-case on the
        # largest real file in this dataset: ~0.5s). Adding a timeout here
        # would just reintroduce the same footgun for no benefit.
        progress = tqdm(total=len(futures), desc="Preprocessing")
        for i, future in enumerate(as_completed(futures)):
            path = futures[future]
            host = os.path.splitext(os.path.basename(path))[0]
            try:
                r = future.result()
            except Exception as e:
                r = {"host": host, "status": f"Worker error: {e}", "elapsed_s": None}

            log_rows.append(r)
            if r["status"] == "Success":
                qc_rows.append(r)
            if r.get("elapsed_s") is not None:
                recent_durations.append(r["elapsed_s"])

            progress.update(1)
            if recent_durations and (i + 1) % 100 == 0:
                sd = sorted(recent_durations)
                median = sd[len(sd) // 2]
                p90 = sd[int(len(sd) * 0.9)]
                remaining = len(futures) - (i + 1)
                eta_seconds = (median * remaining) / MAX_WORKERS
                progress.set_postfix({
                    "median_s": f"{median:.3f}",
                    "p90_s": f"{p90:.3f}",
                    "eta": f"{eta_seconds:.0f}s",
                })
        progress.close()

    # =====================================
    # SAVE LOG + QC SUMMARY
    # =====================================
    log_df = pd.DataFrame(log_rows)
    if os.path.exists(LOG_PATH):
        old_log = pd.read_csv(LOG_PATH)
        if set(old_log.columns) == set(log_df.columns):
            # Keep the newest result per star. Without this, a star that failed on
            # an earlier run (e.g. before stage 1's time-column bug was fixed) and
            # then succeeds on a later run would end up with two contradictory rows
            # instead of the log reflecting current reality.
            log_df = pd.concat([old_log, log_df], ignore_index=True).drop_duplicates(subset="host", keep="last")
        else:
            # Old log is from a different (incompatible) schema version -- e.g. a
            # legacy "File"/"Status" format. Concatenating mismatched schemas
            # produces a malformed CSV with a union of both column sets, and
            # pandas' drop_duplicates treats separate NaNs as equal, which can
            # silently collapse thousands of unrelated legacy rows into one
            # arbitrary survivor. Safer to just start fresh: the old log's content
            # was already stale (pre-dates the stage-1 fix) and had no diagnostic
            # value anyway.
            print(f"WARNING: existing {LOG_PATH} has an incompatible column schema "
                  f"(old: {sorted(old_log.columns)}, new: {sorted(log_df.columns)}). "
                  f"Replacing it instead of merging to avoid a malformed log.")
    log_df.to_csv(LOG_PATH, index=False)

    if qc_rows:
        qc_df = pd.DataFrame(qc_rows)[
            ["host", "n_original", "n_after_quality", "n_after_outliers", "n_final",
             "pct_removed", "pre_norm_median_flux", "flux_source"]
        ]
        if os.path.exists(QC_PATH):
            old_qc = pd.read_csv(QC_PATH)
            qc_df = pd.concat([old_qc, qc_df], ignore_index=True).drop_duplicates(subset="host", keep="last")
        qc_df.to_csv(QC_PATH, index=False)

    status_counts = log_df["status"].apply(
        lambda s: "Success" if s == "Success" else ("Skipped" if str(s).startswith("Skipped") else "Error")
    ).value_counts()

    print("\n===================================")
    print("Finished")
    print("===================================")
    for status, count in status_counts.items():
        print(f"{status}: {count}")
    print(f"Processed files saved to: {OUTPUT_FOLDER}")
    print(f"Per-file log: {LOG_PATH}")
    print(f"QC summary: {QC_PATH}")


if __name__ == "__main__":
    main()

# =====================================
# WHY ProcessPoolExecutor (not ThreadPoolExecutor or serial)
#
# This work is CPU-bound: reading a CSV, sigma-clipping, and running
# Savitzky-Golay are all real computation with no network/disk waiting to
# overlap. Python's GIL means threads don't give true parallelism for CPU-bound
# code (unlike stage 1's network I/O, which released the GIL while waiting).
# ProcessPoolExecutor sidesteps the GIL by using separate processes, each with
# its own interpreter -- the right tool here, at the cost of some
# process-startup and pickling overhead per file (small relative to the
# ~0.1s of real work per file measured below).
#
# WHY NO TIMEOUT WRAPPER
# See the comment inline above the as_completed() loop.
#
# MEASURED RUNTIME (on this machine, M1 MacBook Air, 8 cores):
# - Dataset is ~26 GB across 4,530 files (median 4.5 MB, largest ~62 MB) --
#   bigger than the "several GB" estimate in the original ask.
# - Benchmarked end-to-end (read + clean + flatten + write) on a random
#   30-file sample: median 0.09s/file, p90 0.12s/file. Even the largest file
#   in the dataset (206k rows) took 0.55s total.
# - Serial estimate for ~4,358 valid SPOC-schema files: ~7 minutes.
# - With MAX_WORKERS=7 (leaving one core free): a few minutes, realistically
#   2-5 minutes once you account for process-pool startup and disk I/O
#   contention from 7 processes reading simultaneously (not a clean /7
#   division in practice).
# - SUSPICIOUS if it runs longer than ~20-30 minutes: that would suggest
#   something is wrong -- e.g. accidentally running single-process, a corrupt
#   file causing repeated retries, or disk contention from something else
#   running concurrently. Unlike stage 1, there is no legitimate reason for
#   this stage to take hours: there's no external service to wait on.
# =====================================
