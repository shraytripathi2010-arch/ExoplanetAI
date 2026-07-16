"""
05c_extract_new_features.py

Re-run TLS on every star (both classes) to extract fields that were never
saved in the original stage-3 run: chi2red_min (goodness of transit-model
fit), depth_consistency_std (variability of depth across individual
transits), secondary_eclipse_depth, and transit_shape_ratio (V vs U shape).

WHY A FULL RE-RUN IS NEEDED (not cheap): tried restricting TLS's
period_min/period_max to a narrow window around each star's already-known
best period, hoping to skip the expensive full periodogram search -- TLS
still searched its full default period range regardless (verified on a real
star: same ~53s cost, wrong period range honored). So this costs the same
as the original stage-3 run, per star.

PILOT VALIDATION (80 stars, 40/40 split) BEFORE COMMITTING TO THIS FULL RUN:
  chi2red_min: AUC=0.703 (promising)
  depth_consistency_std: AUC=0.635 (moderate)
  secondary_eclipse_depth: AUC=0.550 (weak, likely needs a better estimator)
  transit_shape_ratio: AUC=0.708 but 30-45% NaN (promising but noisy/small-n)
User explicitly approved the full-scale run given this pilot signal.

Reuses the concurrency/timeout/incremental-save patterns from
03_transit_search.py (real per-star timeout via submission-order iteration,
batch-based checkpointing, use_threads=1 to avoid TLS's own internal
multiprocessing oversubscription) -- see that script for the full reasoning.

Author: Ray's Exoplanet AI Project
"""

import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
import pandas as pd
from tqdm import tqdm
from transitleastsquares import transitleastsquares

warnings.filterwarnings("ignore")

MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)
BATCH_SIZE = 20
PER_STAR_TIMEOUT = 900
OVERSAMPLING_FACTOR = 1       # matches 03_transit_search.py / 03_transit_search_negative.py
DURATION_GRID_STEP = 1.1      # for consistency with the already-saved results

# Large-star binning (SAME fix, same reasoning, as 03_transit_search_negative.py) --
# forgot to carry this over on the first version of this script, which let a
# 92,954-point star run unbinned and blew the ETA out to 254 hours. Re-added here.
MAX_POINTS_BEFORE_BINNING = 30000
TARGET_POINTS_AFTER_BINNING = 15000


def bin_lightcurve(time_arr, flux_arr, flux_err_arr):
    n = len(time_arr)
    if n <= MAX_POINTS_BEFORE_BINNING:
        return time_arr, flux_arr, flux_err_arr

    bin_factor = int(np.ceil(n / TARGET_POINTS_AFTER_BINNING))
    n_bins = n // bin_factor
    trimmed = n_bins * bin_factor

    t_binned = time_arr[:trimmed].reshape(n_bins, bin_factor).mean(axis=1)
    f_binned = flux_arr[:trimmed].reshape(n_bins, bin_factor).mean(axis=1)
    e_binned = np.sqrt((flux_err_arr[:trimmed].reshape(n_bins, bin_factor) ** 2).sum(axis=1)) / bin_factor

    return t_binned, f_binned, e_binned

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_POS_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")
PROCESSED_NEG_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed_negative")
CATALOG_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "catalogs")
TRAINING_PATH = os.path.join(SCRIPT_DIR, "..", "data", "training_dataset", "training.csv")
OUTPUT_PATH = os.path.join(CATALOG_FOLDER, "new_tls_features.csv")

os.makedirs(CATALOG_FOLDER, exist_ok=True)


def compute_new_features(csv_path, host, label):
    t0 = time.monotonic()
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"host": host, "label": label, "status": f"Read error: {e}", "elapsed_s": time.monotonic() - t0}

    if len(df) < 50:
        return {"host": host, "label": label, "status": f"Too few points ({len(df)})", "elapsed_s": time.monotonic() - t0}

    t_arr, f_arr, e_arr = bin_lightcurve(
        df["time"].to_numpy(), df["flux"].to_numpy(), df["flux_err"].to_numpy()
    )

    try:
        model = transitleastsquares(t_arr, f_arr, e_arr)
        r = model.power(
            use_threads=1, oversampling_factor=OVERSAMPLING_FACTOR,
            duration_grid_step=DURATION_GRID_STEP, show_progress_bar=False,
        )
    except Exception as e:
        return {"host": host, "label": label, "status": f"TLS error: {e}", "elapsed_s": time.monotonic() - t0}

    try:
        phase = r.folded_phase
        flux = r.folded_y

        # secondary eclipse: median flux in a window around phase 0.5, as a depth
        # relative to the out-of-transit baseline (1.0)
        sec_mask = (phase > 0.45) & (phase < 0.55)
        secondary_depth = float(1.0 - np.median(flux[sec_mask])) if sec_mask.sum() > 5 else np.nan

        # transit shape: depth near the transit center vs near its edges. Genuine
        # (box/U-shaped) transits stay near-constant depth across the transit;
        # grazing/V-shaped ones are shallower toward the edges.
        primary_mask = (phase < 0.02) | (phase > 0.98)
        if primary_mask.sum() > 5:
            in_transit_phase = np.where(phase > 0.5, phase - 1, phase)[primary_mask]
            center_mask = np.abs(in_transit_phase) < 0.005
            edge_mask = (np.abs(in_transit_phase) >= 0.005) & (np.abs(in_transit_phase) < 0.015)
            center_depth = float(1.0 - np.median(flux[primary_mask][center_mask])) if center_mask.sum() > 2 else np.nan
            edge_depth = float(1.0 - np.median(flux[primary_mask][edge_mask])) if edge_mask.sum() > 2 else np.nan
            shape_ratio = edge_depth / center_depth if (center_depth and center_depth > 0) else np.nan
        else:
            shape_ratio = np.nan

        depths = r.transit_depths
        depths = depths[~np.isnan(depths)] if isinstance(depths, np.ndarray) else np.array([])
        depth_std = float(np.std(depths)) if len(depths) > 1 else np.nan

        result = {
            "host": host, "label": label, "status": "Success",
            "chi2red_min": float(r.chi2red_min),
            "depth_consistency_std": depth_std,
            "secondary_eclipse_depth": secondary_depth,
            "transit_shape_ratio": shape_ratio,
            "elapsed_s": time.monotonic() - t0,
        }
    except Exception as e:
        result = {"host": host, "label": label, "status": f"Post-processing error: {e}", "elapsed_s": time.monotonic() - t0}

    return result


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def save_progress(rows):
    df = pd.DataFrame(rows)
    if os.path.exists(OUTPUT_PATH):
        old = pd.read_csv(OUTPUT_PATH)
        if set(old.columns) == set(df.columns):
            df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset="host", keep="last")
        else:
            print(f"WARNING: {OUTPUT_PATH} has a different column schema -- replacing rather than merging.")
    df.to_csv(OUTPUT_PATH, index=False)


def main():
    training = pd.read_csv(TRAINING_PATH)
    already_done = set()
    if os.path.exists(OUTPUT_PATH):
        old = pd.read_csv(OUTPUT_PATH)
        already_done = set(old.loc[old["status"] == "Success", "host"])
        print(f"{len(already_done)} stars already have new-feature results -- skipping.")

    work_list = []
    for _, row in training.iterrows():
        host = row["host"]
        if host in already_done:
            continue
        subdir = PROCESSED_POS_DIR if row["label"] == 1 else PROCESSED_NEG_DIR
        path = os.path.join(subdir, host + ".csv")
        if os.path.exists(path):
            work_list.append((path, host, int(row["label"])))

    print(f"{len(work_list)} stars to process with {MAX_WORKERS} workers, batches of {BATCH_SIZE}...")

    all_rows = []
    progress = tqdm(total=len(work_list), desc="Extracting new features")

    for batch in chunked(work_list, BATCH_SIZE):
        batch_rows = []
        executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
        try:
            futures = [(executor.submit(compute_new_features, path, host, label), host)
                       for path, host, label in batch]
            for future, host in futures:
                try:
                    r = future.result(timeout=PER_STAR_TIMEOUT)
                except FutureTimeoutError:
                    r = {"host": host, "status": f"Timed out after {PER_STAR_TIMEOUT}s", "elapsed_s": PER_STAR_TIMEOUT}
                except Exception as e:
                    r = {"host": host, "status": f"Worker error: {e}", "elapsed_s": None}
                batch_rows.append(r)
                all_rows.append(r)
                progress.update(1)
        finally:
            executor.shutdown(wait=False)
        save_progress(batch_rows)

    progress.close()
    status_counts = pd.DataFrame(all_rows)["status"].value_counts()
    print("\nFinished this run:")
    print(status_counts)
    print(f"Results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
