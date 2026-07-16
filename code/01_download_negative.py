"""
01_download_negative.py

Download TESS light curves for TOI (TESS Object of Interest) false positives
-- the negative class (label=0) for the training set, as opposed to
01_download_known.py's confirmed planet hosts (label=1).

"False positive" here means the TFOPWG (TESS Follow-up Observing Program
Working Group) vetted the signal and disposition-ed it "FP" -- NOT an
unexamined/unconfirmed candidate. This distinction was deliberate: an
"unconfirmed" TOI just means nobody's checked it yet, which tells you nothing
about whether it's a real planet. An "FP" has actually been examined and
rejected (e.g. found to be an eclipsing binary, background source, or
instrumental artifact), so it's a genuine negative example for training,
not a "maybe."

KEY DIFFERENCE FROM 01_download_known.py: the TOI table gives us a TIC ID
(tid) for every entry, not just a star name. Searching lightkurve by
"TIC <id>" is unambiguous, unlike the fuzzy name-based search the confirmed-
planet pipeline needed (which caused real, verified problems: name
sanitization inconsistencies across runs produced 24 duplicate stars under
different filenames in the positive-class pipeline). Using the TIC ID as
both the search key AND the filename avoids that whole class of bug here.

Otherwise mirrors 01_download_known.py's structure and fixes exactly
(network timeout config, real per-star timeout via submission-order
iteration, honest rolling-median progress) -- see that script's comments
for the full reasoning; not re-derived here.

Author: Ray's Exoplanet AI Project
"""

import os
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import pandas as pd
from tqdm import tqdm
from lightkurve import search_lightcurve

warnings.filterwarnings("ignore", category=UserWarning, module="lightkurve")
warnings.filterwarnings("ignore", category=UserWarning, module="astropy")

# =====================================
# SETTINGS
# =====================================
MAX_DOWNLOADS = 1500      # comfortably above the ~1,241 unique FP TIC IDs found in the archive
PER_STAR_TIMEOUT = 45
NETWORK_TIMEOUT = 30
MAX_WORKERS = 8
ETA_WINDOW = 100
RATE_LIMIT_WINDOW = 50
RATE_LIMIT_THRESHOLD = 0.2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "known_lightcurves_negative")
CATALOG_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "catalogs")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CATALOG_FOLDER, exist_ok=True)


def configure_network_timeouts(seconds):
    applied = []
    try:
        from astropy.utils.data import conf as astropy_conf
        astropy_conf.remote_timeout = seconds
        applied.append("astropy.utils.data.conf.remote_timeout")
    except Exception:
        pass
    try:
        from astroquery.mast import conf as mast_conf
        mast_conf.timeout = seconds
        applied.append("astroquery.mast.conf.timeout")
    except Exception:
        pass
    try:
        import astroquery
        astroquery.conf.timeout = seconds
        applied.append("astroquery.conf.timeout")
    except Exception:
        pass
    return applied


applied_timeouts = configure_network_timeouts(NETWORK_TIMEOUT)
if applied_timeouts:
    print(f"Network timeout set to {NETWORK_TIMEOUT}s via: {', '.join(applied_timeouts)}")
else:
    print(f"WARNING: could not set any network timeout config -- hangs longer than "
          f"{PER_STAR_TIMEOUT}s are still possible.")

RATE_LIMIT_SIGNS = ("429", "too many requests", "503", "service unavailable", "throttl")

# =====================================
# DOWNLOAD TOI TABLE, FILTER TO FALSE POSITIVES
#
# Verified live against the archive before writing this: the "toi" table has
# a tfopwg_disp column with values PC/FP/CP/KP/APC/FA. FP = 1,244 rows / 1,241
# unique TIC IDs (a handful of TICs have more than one FP TOI, mirroring the
# multi-planet-host situation in the positive-class catalog).
# =====================================
print("Downloading TOI table from NASA Exoplanet Archive...")
url = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+tid,toi,tfopwg_disp,ra,dec,st_rad,st_teff+from+toi"
    "&format=csv"
)
toi = pd.read_csv(url)
toi_fp = toi[toi["tfopwg_disp"] == "FP"].copy()
catalog_path = os.path.join(CATALOG_FOLDER, "toi_false_positives.csv")
toi_fp.to_csv(catalog_path, index=False)
print(f"TOI table has {len(toi)} rows; {len(toi_fp)} are disposition='FP'.")

tics_df = (
    toi_fp.dropna(subset=["tid"])
    .drop_duplicates(subset=["tid"])
    .sort_values("tid")
)
tics_df["tid"] = tics_df["tid"].astype("int64")
print(f"{len(tics_df)} unique false-positive TIC IDs found.")

# =====================================
# RESUME SUPPORT (same content-verified pattern as 01_download_known.py)
# =====================================
def file_has_time_column(path):
    try:
        header = pd.read_csv(path, nrows=0).columns
        return "time" in header
    except Exception:
        return False


already_downloaded = set()
candidate_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".csv")]
if candidate_files:
    print(f"Checking {len(candidate_files)} existing files for a valid time column...")
    for f in tqdm(candidate_files, desc="Validating existing files"):
        full_path = os.path.join(OUTPUT_FOLDER, f)
        if file_has_time_column(full_path):
            already_downloaded.add(os.path.splitext(f)[0])
    skipped_bad = len(candidate_files) - len(already_downloaded)
    print(f"{len(already_downloaded)} existing files are valid and will be skipped.")
    if skipped_bad > 0:
        print(f"{skipped_bad} existing files were missing 'time' and will be re-downloaded.")


def try_search(tid, ra, dec):
    """TIC ID search first (unambiguous); coordinate fallback only if that
    somehow fails (e.g. a TIC ID not yet in MAST's TESS product listing).

    Tries author='SPOC' first, then falls back to "whatever's first" --
    same fix and same caveat as 01_download_known.py's try_search(): checked
    against real data, only ~8% of stars currently excluded for non-standard
    schema actually have an available SPOC product this recovers; most are
    genuinely SPOC-unavailable. Worth doing, modest expected impact."""
    search = search_lightcurve(f"TIC {tid}", mission="TESS", author="SPOC")
    if len(search) > 0:
        return search, "tic_id_spoc"
    if pd.notna(ra) and pd.notna(dec):
        search = search_lightcurve(f"{ra} {dec}", mission="TESS", author="SPOC")
        if len(search) > 0:
            return search, "coords_spoc"

    search = search_lightcurve(f"TIC {tid}", mission="TESS")
    if len(search) > 0:
        return search, "tic_id"
    if pd.notna(ra) and pd.notna(dec):
        search = search_lightcurve(f"{ra} {dec}", mission="TESS")
        if len(search) > 0:
            return search, "coords"
    return search, "none"


def download_one_star(tid, ra, dec, filename):
    timings = {"t_search": 0.0, "t_download": 0.0, "t_write": 0.0, "search_method": "none"}

    t0 = time.monotonic()
    search, method = try_search(tid, ra, dec)
    timings["t_search"] = time.monotonic() - t0
    timings["search_method"] = method

    if len(search) == 0:
        return {"host": filename, "status": "No TESS Data", **timings}

    t1 = time.monotonic()
    lc = search[0].download()
    timings["t_download"] = time.monotonic() - t1

    if lc is None:
        return {"host": filename, "status": "Download Failed", **timings}

    t2 = time.monotonic()
    df_out = lc.to_pandas().reset_index()
    if "time" not in df_out.columns:
        return {"host": filename, "status": "Missing 'time' column after reset_index", **timings}

    csv_path = os.path.join(OUTPUT_FOLDER, filename + ".csv")
    df_out.to_csv(csv_path, index=False)
    timings["t_write"] = time.monotonic() - t2

    return {"host": filename, "status": "Success", **timings}


# =====================================
# BUILD WORK QUEUE
# =====================================
work_items = []
for _, row in tics_df.iterrows():
    tid = row["tid"]
    filename = f"TIC_{tid}"
    if filename not in already_downloaded:
        work_items.append((tid, row.get("ra"), row.get("dec"), filename))

work_items = work_items[:MAX_DOWNLOADS]
print(f"\n{len(work_items)} stars to download with {MAX_WORKERS} concurrent workers...\n")

# =====================================
# CONCURRENT DOWNLOAD LOOP -- identical pattern to 01_download_known.py.
# See that script's comments for why submission-order iteration (not
# as_completed) is required for the per-star timeout to actually work.
# =====================================
downloaded = 0
failed = 0
timed_out = 0
log = []
recent_durations = deque(maxlen=ETA_WINDOW)
recent_outcomes = deque(maxlen=RATE_LIMIT_WINDOW)
rate_limit_warned = False

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
futures = [
    (executor.submit(download_one_star, tid, ra, dec, filename), filename)
    for tid, ra, dec, filename in work_items
]

progress = tqdm(total=len(futures), desc="Downloading")
start_time = time.monotonic()

for i, (future, filename) in enumerate(futures):
    star_start = time.monotonic()
    try:
        result = future.result(timeout=PER_STAR_TIMEOUT)
        elapsed = time.monotonic() - star_start
        recent_durations.append(elapsed)

        status = result["status"]
        if status == "Success":
            downloaded += 1
            recent_outcomes.append(False)
        else:
            failed += 1
            is_rate_limit = any(sign in status.lower() for sign in RATE_LIMIT_SIGNS)
            recent_outcomes.append(is_rate_limit)
        log.append([filename, status, result["t_search"], result["t_download"],
                     result["t_write"], result["search_method"]])

    except FutureTimeoutError:
        timed_out += 1
        failed += 1
        recent_durations.append(PER_STAR_TIMEOUT)
        recent_outcomes.append(False)
        log.append([filename, f"Timed out after {PER_STAR_TIMEOUT}s (abandoned, may finish in background)",
                     None, None, None, None])

    except Exception as e:
        failed += 1
        msg = str(e)
        is_rate_limit = any(sign in msg.lower() for sign in RATE_LIMIT_SIGNS)
        recent_outcomes.append(is_rate_limit)
        log.append([filename, msg, None, None, None, None])

    if not rate_limit_warned and len(recent_outcomes) >= RATE_LIMIT_WINDOW:
        rate = sum(recent_outcomes) / len(recent_outcomes)
        if rate > RATE_LIMIT_THRESHOLD:
            tqdm.write(
                f"\nWARNING: {rate:.0%} of the last {RATE_LIMIT_WINDOW} outcomes look like "
                f"rate-limiting (429/503/throttle messages). Consider lowering MAX_WORKERS.\n"
            )
            rate_limit_warned = True

    progress.update(1)
    if recent_durations and (i + 1) % 25 == 0:
        sorted_durations = sorted(recent_durations)
        median = sorted_durations[len(sorted_durations) // 2]
        p90 = sorted_durations[int(len(sorted_durations) * 0.9)]
        remaining = len(futures) - (i + 1)
        eta_seconds = (median * remaining) / MAX_WORKERS
        progress.set_postfix({
            "median_s": f"{median:.1f}",
            "p90_s": f"{p90:.1f}",
            "eta": f"{eta_seconds / 60:.0f}m",
        })

progress.close()
executor.shutdown(wait=False)

# =====================================
# SAVE LOG
# =====================================
log_df = pd.DataFrame(
    log,
    columns=["Host Star", "Status", "search_seconds", "download_seconds", "write_seconds", "search_method"],
)
log_path = os.path.join(CATALOG_FOLDER, "download_log_negative.csv")

if os.path.exists(log_path):
    old_log = pd.read_csv(log_path)
    log_df = pd.concat([old_log, log_df], ignore_index=True)

log_df.to_csv(log_path, index=False)

total_elapsed = time.monotonic() - start_time
# The data (log CSV) is already saved above -- everything from here is just a
# human-readable summary. Wrapped in try/except because under some execution
# contexts (observed when run with stdout redirected to a file, e.g. `python3
# script.py > log.txt`), a leftover ThreadPoolExecutor worker thread (still
# running after shutdown(wait=False) since we deliberately don't block on
# stragglers) can hit a "ValueError: I/O operation on closed file" race
# against the main thread's own print calls. Cosmetic only -- no data is lost
# either way -- but there's no reason to let it produce a scary traceback and
# non-zero exit code when the actual work already completed successfully.
try:
    print("\n===================================")
    print("Finished")
    print("===================================")
    print(f"Downloaded this run: {downloaded}")
    print(f"Failed this run: {failed}")
    print(f"  (of which timed out/abandoned: {timed_out})")
    print(f"Total files in output folder: {len(os.listdir(OUTPUT_FOLDER))}")
    print(f"Total wall time: {total_elapsed / 60:.1f} min for {len(futures)} stars")
    print(f"Files saved to: {OUTPUT_FOLDER}")
    print(f"Per-star timing breakdown logged to: {log_path}")
except ValueError:
    pass
