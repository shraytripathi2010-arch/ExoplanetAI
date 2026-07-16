"""
02_preprocess_negative.py

Clean raw TESS light curve CSVs for the negative class (TOI false positives,
from 01_download_negative.py) using the exact same logic as
02_preprocess.py -- pick the best flux column, drop bad-quality points,
remove outliers, flatten, normalize. Only the input/output paths differ.

See 02_preprocess.py for the full reasoning behind each step (schema
validation against the 8 real distinct schemas found in this project,
the savgol window formula that structurally avoids a degenerate global-fit
bug, the scoped (not global) warning suppression, etc.) -- not re-derived
here since it's identical logic, just pointed at a different folder.

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
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)
SIGMA_CLIP_THRESHOLD = 5
MAX_FLATTEN_WINDOW = 401
SAVGOL_POLYORDER = 2
MIN_POINTS_FOR_FLATTEN = 50
ETA_WINDOW = 100
PROCESS_LIMIT = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "known_lightcurves_negative")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "processed_negative")
CATALOG_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "catalogs")
LOG_PATH = os.path.join(CATALOG_FOLDER, "preprocess_log_negative.csv")
QC_PATH = os.path.join(CATALOG_FOLDER, "preprocess_qc_summary_negative.csv")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CATALOG_FOLDER, exist_ok=True)

REQUIRED_COLUMNS = {"time", "flux", "flux_err", "quality", "pdcsap_flux", "pdcsap_flux_err"}


def validate_schema(df):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return f"Non-standard schema (missing columns: {sorted(missing)})"
    if df["quality"].dtype == object:
        return "Non-standard schema (quality column is non-numeric, e.g. CDIPS-style flags)"
    if len(df) == 0:
        return "Empty file (zero rows)"
    return None


def choose_flux_columns(df):
    if df["pdcsap_flux"].notna().any():
        return df["pdcsap_flux"].to_numpy(), df["pdcsap_flux_err"].to_numpy(), "pdcsap_flux"
    if df["flux"].notna().any():
        return df["flux"].to_numpy(), df["flux_err"].to_numpy(), "flux"
    return None, None, "No usable flux data (both pdcsap_flux and flux are entirely NaN)"


def choose_savgol_window(n_points, max_window=MAX_FLATTEN_WINDOW, polyorder=SAVGOL_POLYORDER):
    window = min(max_window, n_points - 1)
    if window % 2 == 0:
        window -= 1
    if window < polyorder + 2 or window < 5:
        return None
    return window


def output_is_valid(path):
    try:
        df = pd.read_csv(path)
        return {"time", "flux", "flux_err"}.issubset(df.columns) and len(df) > 0
    except Exception:
        return False


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

    valid = ~np.isnan(time_arr) & ~np.isnan(flux) & ~np.isnan(flux_err)
    time_arr, flux, flux_err, quality = time_arr[valid], flux[valid], flux_err[valid], quality[valid]

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

    order = np.argsort(time_arr, kind="stable")
    time_arr, flux, flux_err = time_arr[order], flux[order], flux_err[order]

    with warnings.catch_warnings():
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

    pre_norm_median_flux = float(np.median(flux))
    result["pre_norm_median_flux"] = pre_norm_median_flux

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
        result["status"] = "Skipped: non-finite values in flattened flux (degenerate trend)"
        result["n_final"] = 0
        result["elapsed_s"] = time.monotonic() - t0
        return result

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

    log_df = pd.DataFrame(log_rows)
    if os.path.exists(LOG_PATH):
        old_log = pd.read_csv(LOG_PATH)
        if set(old_log.columns) == set(log_df.columns):
            log_df = pd.concat([old_log, log_df], ignore_index=True).drop_duplicates(subset="host", keep="last")
        else:
            print(f"WARNING: existing {LOG_PATH} has an incompatible column schema -- replacing rather than merging.")
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
