"""
05d_recompute_stellar_ttv.py

Re-run TLS on every star using each star's ACTUAL stellar radius/mass
(from training.csv's st_rad/st_mass, falling back to solar 1.0/1.0 only
when missing) instead of the blanket solar defaults used in
03_transit_search.py / 03_transit_search_negative.py / 05c_extract_new_features.py.

WHY THIS MATTERS: TLS uses R_star/M_star to build its period/duration search
grid (via Kepler's third law). Assuming every star is Sun-like when 63% of
this dataset's stars deviate from solar by more than 20% (checked on real
data before committing to this run -- some up to 100x solar radius) means
the grid searched for those stars may not be well-matched to what's
physically plausible for them, potentially affecting how well TLS's period/
duration/depth results reflect the true signal.

BUNDLED IN THE SAME RUN (to avoid paying for a full re-run twice):
  - Everything 05c_extract_new_features.py computed (chi2red_min,
    depth_consistency_std, secondary_eclipse_depth, transit_shape_ratio)
  - NEW: a Transit Timing Variation (TTV) metric. TLS's transit_times field
    (individual transit epochs) was never extracted before. Fit a linear
    ephemeris (transit time vs transit number) and take the residual scatter
    -- real planets swept up in an n-body system show TTVs; this is a
    distinct signal from anything else in the feature set. Needs >=3
    transits to be meaningful (2 points always fit a line with 0 residual).

Reuses the same concurrency/timeout/incremental-save/binning patterns as
05c_extract_new_features.py -- see that script for the detailed reasoning.

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
OVERSAMPLING_FACTOR = 1
DURATION_GRID_STEP = 1.1
DEFAULT_R_STAR = 1.0
DEFAULT_M_STAR = 1.0

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
OUTPUT_PATH = os.path.join(CATALOG_FOLDER, "v3_stellar_ttv_features.csv")

os.makedirs(CATALOG_FOLDER, exist_ok=True)


def compute_ttv(transit_times, period):
    """Fit a linear ephemeris to observed transit times and return the
    residual scatter (std of observed-minus-calculated times, in days).
    Needs >=3 transits -- with 2, a line fits perfectly (0 residual) by
    construction, which would look like a "perfect non-TTV" signal for
    every 2-transit star, a spurious floor rather than real information."""
    times = np.asarray(transit_times, dtype=float)
    times = times[~np.isnan(times)]
    if len(times) < 3 or period <= 0:
        return np.nan, np.nan

    transit_number = np.round((times - times[0]) / period)
    if len(np.unique(transit_number)) < 3:
        return np.nan, np.nan

    coeffs = np.polyfit(transit_number, times, 1)
    predicted = np.polyval(coeffs, transit_number)
    residuals = times - predicted
    return float(np.std(residuals)), float(np.max(residuals) - np.min(residuals))


def compute_features(csv_path, host, label, r_star, m_star):
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

    r_star = r_star if (r_star and not np.isnan(r_star) and r_star > 0) else DEFAULT_R_STAR
    m_star = m_star if (m_star and not np.isnan(m_star) and m_star > 0) else DEFAULT_M_STAR

    # TLS's own default R_star_max/M_star_max (3.5 / 1.0 solar) are too narrow for
    # this dataset -- found via a real crash on a giant star (R_star=13.76) before
    # committing to the full run. Scale min/max bounds around each star's actual
    # value (with margin for measurement uncertainty) instead of relying on TLS's
    # solar-centric defaults, which would reject any non-Sun-like star outright.
    r_star_min, r_star_max = min(0.13, r_star * 0.5), max(3.5, r_star * 1.5)
    m_star_min, m_star_max = min(0.1, m_star * 0.5), max(1.0, m_star * 1.5)

    try:
        model = transitleastsquares(t_arr, f_arr, e_arr)
        r = model.power(
            use_threads=1, oversampling_factor=OVERSAMPLING_FACTOR,
            duration_grid_step=DURATION_GRID_STEP, show_progress_bar=False,
            R_star=r_star, R_star_min=r_star_min, R_star_max=r_star_max,
            M_star=m_star, M_star_min=m_star_min, M_star_max=m_star_max,
        )
    except Exception as e:
        return {"host": host, "label": label, "status": f"TLS error: {e}", "elapsed_s": time.monotonic() - t0}

    try:
        phase = r.folded_phase
        flux = r.folded_y

        sec_mask = (phase > 0.45) & (phase < 0.55)
        secondary_depth = float(1.0 - np.median(flux[sec_mask])) if sec_mask.sum() > 5 else np.nan

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

        ttv_std, ttv_amplitude = compute_ttv(r.transit_times, r.period)

        result = {
            "host": host, "label": label, "status": "Success",
            "r_star_used": r_star, "m_star_used": m_star,
            "period_v3": float(r.period), "duration_v3": float(r.duration),
            "SDE_v3": float(r.SDE), "depth_v3": float(r.depth),
            "chi2red_min": float(r.chi2red_min),
            "depth_consistency_std": depth_std,
            "secondary_eclipse_depth": secondary_depth,
            "transit_shape_ratio": shape_ratio,
            "ttv_std": ttv_std,
            "ttv_amplitude": ttv_amplitude,
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
        print(f"{len(already_done)} stars already have v3 results -- skipping.")

    n_non_solar = ((training["st_rad"] - 1.0).abs() > 0.2).sum()
    print(f"{n_non_solar}/{len(training)} stars deviate >20% from solar radius -- this is why "
          f"per-star R_star/M_star matters here, not just a cosmetic change.")

    work_list = []
    for _, row in training.iterrows():
        host = row["host"]
        if host in already_done:
            continue
        subdir = PROCESSED_POS_DIR if row["label"] == 1 else PROCESSED_NEG_DIR
        path = os.path.join(subdir, host + ".csv")
        if os.path.exists(path):
            work_list.append((path, host, int(row["label"]), row.get("st_rad"), row.get("st_mass")))

    print(f"{len(work_list)} stars to process with {MAX_WORKERS} workers, batches of {BATCH_SIZE}...")

    all_rows = []
    progress = tqdm(total=len(work_list), desc="Recomputing with real stellar params + TTV")

    for batch in chunked(work_list, BATCH_SIZE):
        batch_rows = []
        executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
        try:
            futures = [(executor.submit(compute_features, path, host, label, r_star, m_star), host)
                       for path, host, label, r_star, m_star in batch]
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
