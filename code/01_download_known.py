"""
01_download_known.py

Download TESS light curves for confirmed exoplanet host stars.
Runs multiple downloads concurrently for speed, with a per-star timeout
so a single hung request can't block the others.

Author: Ray's Exoplanet AI Project
"""

import os
import re
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
MAX_DOWNLOADS = 4732
PER_STAR_TIMEOUT = 45     # wrapper deadline: stop *waiting* on a star after this long
NETWORK_TIMEOUT = 30      # actual socket/HTTP timeout for MAST calls (see note below)
MAX_WORKERS = 8           # this is I/O-bound work; see note at bottom of file
ETA_WINDOW = 100          # how many recent per-star durations to use for the ETA estimate
RATE_LIMIT_WINDOW = 50    # how many recent outcomes to scan for rate-limit signals
RATE_LIMIT_THRESHOLD = 0.2  # warn if >20% of the recent window looks like throttling

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "known_lightcurves")
CATALOG_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "catalogs")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CATALOG_FOLDER, exist_ok=True)


# =====================================
# NETWORK TIMEOUT CONFIG
#
# By default astroquery.mast's timeout is 600s and astropy's file-download
# timeout is 10s. The 600s figure is almost certainly why a single star was
# able to hang for 25+ minutes with no error: the MAST search/query call had
# no reason to give up. We lower both here so a stalled call actually raises
# an exception within NETWORK_TIMEOUT seconds instead of hanging indefinitely.
# This is the *real* fix for hangs -- the ThreadPoolExecutor wrapper timeout
# below is only a second line of defense, since Python cannot forcibly kill
# a thread blocked in a network call.
# =====================================
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
    print(
        f"WARNING: could not set any network timeout config (checked astropy/astroquery "
        f"attributes). Hangs longer than {PER_STAR_TIMEOUT}s are still possible; the "
        f"ThreadPoolExecutor wrapper timeout below will limit how long the *script* waits, "
        f"but stuck threads may linger in the background."
    )

# Strings that typically indicate MAST is throttling/rate-limiting us, as opposed to a
# generic network hiccup. Used only for the warning heuristic below -- not exhaustive.
RATE_LIMIT_SIGNS = ("429", "too many requests", "503", "service unavailable", "throttl")


# =====================================
# DOWNLOAD NASA CATALOG
# =====================================
print("Downloading NASA Exoplanet Archive catalog...")
url = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+hostname,pl_name,ra,dec,st_rad,st_teff,st_mass+from+pscomppars"
    "&format=csv"
)
catalog = pd.read_csv(url)
catalog_path = os.path.join(CATALOG_FOLDER, "confirmed_planets.csv")
catalog.to_csv(catalog_path, index=False)
print(f"Catalog contains {len(catalog)} confirmed planets.")

hosts_df = (
    catalog.dropna(subset=["hostname"])
    .drop_duplicates(subset=["hostname"])
    .sort_values("hostname")
)
print(f"{len(hosts_df)} unique host stars found.")

# =====================================
# RESUME SUPPORT
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


def safe_filename(name):
    # Keep it filesystem-safe: letters, digits, dash, underscore, dot only.
    name = name.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-.]", "", name)


def strip_component_suffix(name):
    return re.sub(r"\s+[A-Za-z]$", "", name)


def try_search(host, ra, dec):
    """Returns (search_result, method_used) so callers can log which lookup path fired.

    Tries author='SPOC' first at each name/coord variant, THEN falls back to
    "whatever's first" for that same variant. Verified against real data
    before changing this: a naive "first result" search was grabbing whatever
    pipeline MAST returned first (CDIPS, QLP, custom -- 8 distinct schemas
    found across this project's downloaded set), and stage 2 has to reject
    ~172-274 stars for non-standard schema as a result. Checked a 50-star
    sample of those rejects: only 8% actually had an available SPOC product
    that this fix would have caught -- most are genuinely SPOC-unavailable,
    not just mis-selected. Worth doing since it's a real, if modest, fix, but
    don't expect it to reclaim most of the excluded population."""
    variants = [host]
    stripped = strip_component_suffix(host)
    if stripped != host:
        variants.append(stripped)
    if pd.notna(ra) and pd.notna(dec):
        variants.append(f"{ra} {dec}")
    variant_labels = ["name", "stripped_name", "coords"][:len(variants)]

    for target, label in zip(variants, variant_labels):
        search = search_lightcurve(target, mission="TESS", author="SPOC")
        if len(search) > 0:
            return search, f"{label}_spoc"

    for target, label in zip(variants, variant_labels):
        search = search_lightcurve(target, mission="TESS")
        if len(search) > 0:
            return search, label
    return search, "none"


def download_one_star(host, ra, dec, filename):
    """The actual work for a single star, run inside a worker thread.

    Returns a dict with a status plus per-phase timings so bottlenecks
    (search vs download vs write) are visible in the log, not guessed at.
    """
    timings = {"t_search": 0.0, "t_download": 0.0, "t_write": 0.0, "search_method": "none"}

    t0 = time.monotonic()
    search, method = try_search(host, ra, dec)
    timings["t_search"] = time.monotonic() - t0
    timings["search_method"] = method

    if len(search) == 0:
        return {"host": host, "status": "No TESS Data", **timings}

    t1 = time.monotonic()
    lc = search[0].download()
    timings["t_download"] = time.monotonic() - t1

    if lc is None:
        return {"host": host, "status": "Download Failed", **timings}

    t2 = time.monotonic()
    df_out = lc.to_pandas().reset_index()
    if "time" not in df_out.columns:
        # Same silent bug we already got burned by once -- verify at write time too,
        # not just when checking previously-downloaded files during resume.
        return {"host": host, "status": "Missing 'time' column after reset_index", **timings}

    csv_path = os.path.join(OUTPUT_FOLDER, filename + ".csv")
    df_out.to_csv(csv_path, index=False)
    timings["t_write"] = time.monotonic() - t2

    return {"host": host, "status": "Success", **timings}


# =====================================
# BUILD WORK QUEUE (skip already-valid files up front)
# =====================================
work_items = []
for _, row in hosts_df.iterrows():
    host = row["hostname"]
    filename = safe_filename(host)
    if filename not in already_downloaded:
        work_items.append((host, row.get("ra"), row.get("dec"), filename))

work_items = work_items[:MAX_DOWNLOADS]
print(f"\n{len(work_items)} stars to download with {MAX_WORKERS} concurrent workers...\n")

# =====================================
# CONCURRENT DOWNLOAD LOOP WITH A REAL PER-STAR TIMEOUT
#
# IMPORTANT: the previous version wrapped as_completed(...) with
# future.result(timeout=...). as_completed() only yields a future *after* it
# has already finished, so that timeout could never fire -- it's a no-op.
# Instead we iterate futures in submission order and call .result(timeout=...)
# on each one *before* it's necessarily done. Since ThreadPoolExecutor keeps
# MAX_WORKERS threads busy pulling from the same queue, waiting on futures in
# submission order still gets full concurrency -- we're just choosing to
# observe/report results in submission order rather than completion order.
#
# If a future doesn't finish in time we stop waiting on it and log it as
# abandoned. The underlying thread may still be running in the background;
# with NETWORK_TIMEOUT now configured, it should raise on its own shortly and
# its result is simply discarded. We use shutdown(wait=False) at the end so a
# straggler can't block the whole script from exiting.
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
    (executor.submit(download_one_star, host, ra, dec, filename), host)
    for host, ra, dec, filename in work_items
]

progress = tqdm(total=len(futures), desc="Downloading")
start_time = time.monotonic()

for i, (future, host) in enumerate(futures):
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
        log.append([host, status, result["t_search"], result["t_download"],
                     result["t_write"], result["search_method"]])

    except FutureTimeoutError:
        timed_out += 1
        failed += 1
        recent_durations.append(PER_STAR_TIMEOUT)
        recent_outcomes.append(False)
        log.append([host, f"Timed out after {PER_STAR_TIMEOUT}s (abandoned, may finish in background)",
                     None, None, None, None])

    except Exception as e:
        failed += 1
        msg = str(e)
        is_rate_limit = any(sign in msg.lower() for sign in RATE_LIMIT_SIGNS)
        recent_outcomes.append(is_rate_limit)
        log.append([host, msg, None, None, None, None])

    # Rate-limit heuristic: warn once if a large fraction of recent outcomes look
    # like throttling. Not exhaustive -- just a signal to lower MAX_WORKERS.
    if not rate_limit_warned and len(recent_outcomes) >= RATE_LIMIT_WINDOW:
        rate = sum(recent_outcomes) / len(recent_outcomes)
        if rate > RATE_LIMIT_THRESHOLD:
            tqdm.write(
                f"\nWARNING: {rate:.0%} of the last {RATE_LIMIT_WINDOW} outcomes look like "
                f"rate-limiting (429/503/throttle messages). Consider lowering MAX_WORKERS.\n"
            )
            rate_limit_warned = True

    # Honest progress reporting: use the rolling median of *actual* completions
    # instead of tqdm's default rate smoothing, which swings wildly here because
    # individual star durations are highly variable (cache hits, retries, big vs
    # small files, occasional timeouts).
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
log_path = os.path.join(CATALOG_FOLDER, "download_log.csv")

if os.path.exists(log_path):
    old_log = pd.read_csv(log_path)
    log_df = pd.concat([old_log, log_df], ignore_index=True)

log_df.to_csv(log_path, index=False)

total_elapsed = time.monotonic() - start_time
# The data (log CSV) is already saved above -- everything from here is just a
# human-readable summary. Wrapped in try/except: under some execution contexts
# (observed with stdout redirected to a file), a leftover ThreadPoolExecutor
# worker thread (still running after shutdown(wait=False), since we
# deliberately don't block on stragglers) can hit a "ValueError: I/O operation
# on closed file" race against the main thread's own print calls. Cosmetic
# only -- no data is lost either way.
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

# =====================================
# NOTES ON SPEED / WORKER COUNT
#
# - ThreadPoolExecutor is the right tool here: this is I/O-bound (waiting on
#   MAST's network response), and Python releases the GIL during socket I/O,
#   so threads give real concurrency without the pickling/process-startup
#   overhead a ProcessPoolExecutor would add for no benefit.
# - There's no published official MAST concurrency limit. Start at MAX_WORKERS=8,
#   watch the rate-limit warning above, and back off if it fires. Going much
#   above ~16 is unlikely to help since MAST's server-side response time, not
#   your client, becomes the bottleneck.
# - Hard floor: each star needs at least one MAST search round-trip (roughly
#   0.5-3s typically) plus a FITS download (TESS 2-min light curves are usually
#   a few MB, adding another 1-5s depending on your connection and MAST load).
#   With realistic per-star cost around 3-8s and 8 workers running truly
#   concurrently, ~2,900 stars is roughly (2900 * 5s) / 8 =~ 30 minutes in the
#   best case, but expect 1.5-4 hours in practice once you account for slower
#   stars, retries, and MAST-side load -- NOT the 15-20+ hours you were seeing,
#   which was a symptom of the as_completed/timeout bug silently degrading your
#   effective concurrency over the course of the run, not an inherent limit.
# =====================================
