"""
06_download_unknown.py

Find TESS stars that have NEVER been examined/flagged by anyone -- not in
the confirmed-planet catalog, not in the TOI table under ANY disposition
(PC/CP/KP/APC/FP/FA) -- and score them with the trained model to produce a
ranked candidate list for human review.

===============================================================================
READ THIS FIRST -- WHAT "UNKNOWN" ACTUALLY MEANS HERE
===============================================================================
TESS observes and publicly releases light curve data for nearly all its
targets automatically. "Unknown" in this script means NOT YET EXAMINED OR
FLAGGED BY ANYONE -- it does NOT mean "data that doesn't exist yet" and it
is NOT a guarantee of novelty. Recency (preferring the newest TESS sectors)
matters only because newer sectors have had less time for professional
pipelines, citizen scientists, or published papers to look at them closely
-- that is the actual mechanism for a better shot at something genuinely
unexamined, not any guarantee. Most stars in even the newest sector are
still going to be boring (noise, non-astrophysical variability, or genuine
non-detections) -- that's expected and fine; the point is to surface the
FEW worth a qualified human's attention, not to promise discoveries.

A high predicted probability from this script means "worth a closer look
by someone qualified to vet it" -- it does NOT mean "this is a planet."
Real confirmation of any candidate requires far more than a light-curve
classifier can do: spectroscopic follow-up, multiple independent
observations, and ruling out instrumental/astrophysical false-positive
scenarios (background eclipsing binaries, etc.) that a light-curve-only
model fundamentally cannot see. This disclaimer is repeated in the script's
own output, not just this docstring, since it matters that whoever reads
the ranked CSV sees it too.

===============================================================================
PIPELINE STAGES (each with resume support -- checkpointed CSVs throughout,
matching this project's established pattern for long-running downloads)
===============================================================================
A. Build the exclusion set: confirmed-planet TIC IDs (from the archive's
   pscomppars.tic_id column directly -- not fuzzy name matching, which
   caused real bugs earlier in this project) UNION all TOI TIC IDs under
   ANY disposition (not just the FP/FA used as label=0 in training).
B. Build the candidate pool from the MOST RECENT TESS sectors with public
   SPOC light curve data (checked live against MAST, not assumed), working
   backward to older sectors only if needed to hit the target sample size.
C. Batch-fetch stellar parameters (radius, mass, Teff) from the TIC catalog.
D. Download light curves -- same SPOC-preference search, per-star timeout
   (submission-order iteration, the correct fix), and content-verified
   resume support as 01_download_known.py / 01_download_negative.py.
E. Preprocess -- identical cleaning logic to 02_preprocess.py (same
   function, ported verbatim: schema validation, quality filter, sigma
   clipping, Savitzky-Golay flatten+normalize).
F. TLS feature extraction with each star's REAL stellar radius/mass (not
   solar defaults) -- same approach as 05d_recompute_stellar_ttv.py,
   producing every feature in models/best_model_metadata.json's
   feature_columns list. A star is EXCLUDED (not silently zero-filled) if
   any required feature can't be computed -- a mismatched feature set
   would make its prediction meaningless.
G. Load the saved model, predict on every successfully-featured star, save
   a full ranked table.
H. For the top N candidates: plain-language explanations (via permutation
   importance recomputed fresh on the training set) + folded light curve
   plots for visual sanity-checking.

Usage:
    python3 06_download_unknown.py --sample-size 300 --top-n 20
"""

import argparse
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError as FutureTimeoutError

import joblib
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import savgol_filter
from astropy.stats import sigma_clip

warnings.filterwarnings("ignore", category=UserWarning, module="lightkurve")
warnings.filterwarnings("ignore", category=UserWarning, module="astropy")

# A leftover ThreadPoolExecutor worker thread (still running after
# shutdown(wait=False), since we deliberately don't block on stragglers so a
# single slow download can't hold up the whole pipeline) can race the main
# thread's stdout under some execution contexts -- seen throughout this
# project whenever stdout is redirected. It only ever affects console output,
# never the data itself (which is always written to disk before this can
# happen). This isn't confined to print() -- tqdm calls sys.stdout.flush()
# directly, so patching print alone isn't enough; wrap the actual stdout
# object so ANY write/flush after it's gone stale is swallowed instead of
# crashing the whole pipeline over what is, at worst, a garbled progress bar.
import sys as _sys


class _SafeStdout:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def write(self, *args, **kwargs):
        try:
            return self._real.write(*args, **kwargs)
        except (ValueError, OSError):
            return 0

    def flush(self, *args, **kwargs):
        try:
            return self._real.flush(*args, **kwargs)
        except (ValueError, OSError):
            pass


_sys.stdout = _SafeStdout(_sys.stdout)

# ROOT CAUSE FOUND (not just "a leftover thread races stdout" as previously
# documented -- confirmed by reading lightkurve's own source): every call to
# a SearchResult's `.download()` is wrapped by lightkurve's own
# `suppress_stdout` decorator, which does:
#     old_out = sys.stdout; sys.stdout = devnull; ...; sys.stdout = old_out
# This mutates the GLOBAL, process-wide `sys.stdout` with no lock. With
# DOWNLOAD_WORKERS threads calling `.download()` concurrently, two threads'
# swap/restore cycles interleave: thread B captures thread A's `devnull` as
# its own "old_out", and when B's `with open(...) as devnull` block exits,
# that devnull gets closed -- but it may now be the value LEFT installed as
# the real `sys.stdout`, permanently replacing the _SafeStdout wrapper above
# (not just writing through it). Once that happens, _SafeStdout's own
# write()/flush() protection is bypassed entirely, because `sys.stdout` no
# longer IS that wrapper -- the next print() ANYWHERE in the process (often
# something totally unrelated, like main()'s opening banner) raises "I/O
# operation on closed file" for real. This is why retries alone (internal
# or external/fresh-process) don't reliably help once a batch is large
# enough to make the race likely: a fresh process just reduces the odds per
# run, it doesn't remove the race. The actual fix: never let two threads be
# inside a `.download()` call at the same time.
import threading as _threading
_DOWNLOAD_LOCK = _threading.Lock()


def _safe_download(search_result):
    """Serializes every lightkurve `.download()` call against this lock --
    see the root-cause note above. Also defensively re-wraps sys.stdout in
    _SafeStdout afterward in case a stale devnull object slipped through
    from some other, unlocked caller in this process (e.g. before this fix
    was applied elsewhere, or a future call site that forgets to use this
    helper) -- cheap insurance on top of the real fix, not a substitute for it."""
    with _DOWNLOAD_LOCK:
        try:
            return search_result.download()
        finally:
            if not isinstance(_sys.stdout, _SafeStdout):
                _sys.stdout = _SafeStdout(_sys.stdout)


# =====================================
# SETTINGS
# =====================================
PER_STAR_DOWNLOAD_TIMEOUT = 45
NETWORK_TIMEOUT = 30
DOWNLOAD_WORKERS = 8
TLS_WORKERS = max(1, (os.cpu_count() or 4) - 1)
TLS_PER_STAR_TIMEOUT = 900
TLS_BATCH_SIZE = 20
SECTORS_TO_TRY = 6          # how many recent sectors (newest-first) to draw candidates from before giving up
MAX_POINTS_BEFORE_BINNING = 30000
TARGET_POINTS_AFTER_BINNING = 15000
MIN_POINTS_FOR_FLATTEN = 50
MAX_FLATTEN_WINDOW = 401
SAVGOL_POLYORDER = 2
SIGMA_CLIP_THRESHOLD = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
CATALOG_FOLDER = os.path.join(PROJECT_ROOT, "data", "catalogs")
MODELS_FOLDER = os.path.join(PROJECT_ROOT, "models")
TRAINING_PATH = os.path.join(PROJECT_ROOT, "data", "training_dataset", "training.csv")

# --multi-sector runs get their own folders/files (a "_widesector" tag) so a
# direct before/after comparison against the original single-sector pilot is
# possible without one run clobbering the other's saved data. Read directly
# off sys.argv here (not argparse, which only runs inside main()) since these
# paths are needed at module level before main() is called.
_RUN_TAG = "_widesector" if "--multi-sector" in _sys.argv else ""

RAW_FOLDER = os.path.join(PROJECT_ROOT, "data", f"unknown_lightcurves{_RUN_TAG}")
PROCESSED_FOLDER = os.path.join(PROJECT_ROOT, "data", f"processed_unknown{_RUN_TAG}")
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, "results", f"unknown_candidates{_RUN_TAG}")

CANDIDATE_LIST_PATH = os.path.join(CATALOG_FOLDER, f"unknown_candidate_list{_RUN_TAG}.csv")
DOWNLOAD_LOG_PATH = os.path.join(CATALOG_FOLDER, f"unknown_download_log{_RUN_TAG}.csv")
FEATURES_PATH = os.path.join(CATALOG_FOLDER, f"unknown_features{_RUN_TAG}.csv")
RANKED_OUTPUT_PATH = os.path.join(RESULTS_FOLDER, "ranked_candidates.csv")
EXPLANATIONS_PATH = os.path.join(RESULTS_FOLDER, "top_candidate_explanations.txt")

for d in (RAW_FOLDER, PROCESSED_FOLDER, RESULTS_FOLDER, CATALOG_FOLDER):
    os.makedirs(d, exist_ok=True)


def configure_network_timeouts(seconds):
    try:
        from astropy.utils.data import conf as astropy_conf
        astropy_conf.remote_timeout = seconds
    except Exception:
        pass
    try:
        from astroquery.mast import conf as mast_conf
        mast_conf.timeout = seconds
    except Exception:
        pass


configure_network_timeouts(NETWORK_TIMEOUT)


def canonical_key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# =====================================
# STAGE A: EXCLUSION SET
# =====================================
def build_exclusion_set():
    print("=" * 60)
    print("STAGE A: building the exclusion set (confirmed planets + ALL TOI dispositions)")
    print("=" * 60)

    confirmed_url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+hostname,tic_id+from+pscomppars&format=csv"
    )
    confirmed = pd.read_csv(confirmed_url)
    confirmed_tics = set(
        confirmed["tic_id"].dropna().str.replace("TIC ", "", regex=False).astype("int64")
    )
    print(f"Confirmed planet hosts: {confirmed['hostname'].nunique()} unique hostnames, "
          f"{len(confirmed_tics)} resolvable to a TIC ID directly.")

    toi_url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+tid,tfopwg_disp+from+toi&format=csv"
    )
    toi = pd.read_csv(toi_url)
    toi_tics = set(toi["tid"].dropna().astype("int64"))
    print(f"TOI table: {len(toi)} rows, {len(toi_tics)} unique TIC IDs across ALL dispositions "
          f"({toi['tfopwg_disp'].value_counts().to_dict()}).")

    excluded = confirmed_tics | toi_tics
    print(f"\nTotal excluded TIC IDs (union, already flagged by someone): {len(excluded)}")
    return excluded


# =====================================
# STAGE B: CANDIDATE POOL FROM RECENT SECTORS
# =====================================
def find_recent_sectors():
    """Live check against MAST for the most recent sectors with public SPOC
    light curve data -- never assumed. Uses a t_min window rather than
    querying the entire TESS archive (which is enormous and slow)."""
    from astropy.time import Time
    from astroquery.mast import Observations

    now_mjd = Time.now().mjd
    window_start = now_mjd - 150   # generous window to safely catch the newest few sectors
    obs = Observations.query_criteria(
        obs_collection="TESS", dataproduct_type="timeseries", calib_level=3,
        t_min=[window_start, now_mjd],
    )
    df = obs.to_pandas()
    sectors = sorted(df["sequence_number"].dropna().unique(), reverse=True)
    print(f"Live MAST check: sectors with public SPOC data in the last ~150 days: {sectors}")
    return sectors, df


def build_candidate_pool(excluded_tics, target_size):
    print("\n" + "=" * 60)
    print("STAGE B: building candidate pool from the newest available TESS sectors")
    print("=" * 60)

    if os.path.exists(CANDIDATE_LIST_PATH):
        existing = pd.read_csv(CANDIDATE_LIST_PATH)
        if len(existing) >= target_size:
            print(f"{CANDIDATE_LIST_PATH} already has {len(existing)} candidates (>= target "
                  f"{target_size}) -- reusing the tic_id/sector list (resume support). Stellar "
                  f"params are re-fetched fresh below regardless -- this file may already have "
                  f"them baked in from a prior run's save, and re-merging onto that would collide "
                  f"column names instead of cleanly refreshing them.")
            # BUG FIXED (found by a clean-clone reproduction test): this
            # returned the ENTIRE existing list, silently ignoring
            # target_size. The repo ships a 2,000-row
            # unknown_candidate_list.csv, so a new user running the
            # documented quick-start (`--sample-size 10`) got a 2,000-star,
            # ~25-hour download instead of the few minutes the flag implies.
            # Resume support should reuse the cached list, not override the
            # size the user explicitly asked for.
            return existing[["tic_id", "sector"]].head(target_size)

    sectors, sector_obs_df = find_recent_sectors()
    sectors_used = sectors[:SECTORS_TO_TRY]

    candidates = []
    seen_tics = set()
    for sector in sectors_used:
        sector_rows = sector_obs_df[sector_obs_df["sequence_number"] == sector]
        tic_ids = sector_rows["target_name"].dropna().unique()
        n_before = len(candidates)
        for tic_str in tic_ids:
            try:
                tic_id = int(tic_str)
            except (ValueError, TypeError):
                continue
            if tic_id in excluded_tics or tic_id in seen_tics:
                continue
            seen_tics.add(tic_id)
            candidates.append({"tic_id": tic_id, "sector": int(sector)})
            if len(candidates) >= target_size:
                break
        print(f"  Sector {int(sector)}: {len(tic_ids)} targets observed, "
              f"{len(candidates) - n_before} new unflagged candidates added "
              f"(running total: {len(candidates)}/{target_size})")
        if len(candidates) >= target_size:
            break

    if len(candidates) < target_size:
        print(f"WARNING: only found {len(candidates)}/{target_size} unflagged candidates within "
              f"the {len(sectors_used)} most recent sectors tried. Proceeding with what was found "
              f"rather than silently padding the list -- increase SECTORS_TO_TRY to search further back.")

    df = pd.DataFrame(candidates)
    df.to_csv(CANDIDATE_LIST_PATH, index=False)
    print(f"\nCandidate pool: {len(df)} stars, saved to {CANDIDATE_LIST_PATH}")
    return df


def build_candidate_pool_multi_sector(excluded_tics, target_size, n_sectors=3):
    """Pulls candidates from the N most recent sectors (not just the newest
    one), prioritizing MULTI-SECTOR coverage first, then falling back to
    single-sector stars newest-first. Rationale: a star observed in only
    sector 101 isn't meaningfully more "examined" than one observed only in
    103 -- both are single-sector and face the same multi-transit-feature
    bottleneck. The real lever for improving yield is stars TESS happened to
    re-observe across consecutive sectors (giving a longer stitched baseline,
    enough for more transits of longer-period signals), and those stars are
    still very recent (their most recent observation is still within this
    same N-sector window) -- so prioritizing multi-sector coverage doesn't
    meaningfully sacrifice the recency goal. Priority order:
      1. Observed in ALL N sectors considered (best baseline, still recent)
      2. Observed in N-1, N-2, ... sectors (more sectors = higher priority)
      3. Tie-broken by whether the newest sector (highest sector number) is
         among the ones observed, then by single-sector recency
    """
    print("\n" + "=" * 60)
    print(f"STAGE B: building candidate pool from a {n_sectors}-sector window "
          f"(multi-sector coverage prioritized)")
    print("=" * 60)

    if os.path.exists(CANDIDATE_LIST_PATH):
        existing = pd.read_csv(CANDIDATE_LIST_PATH)
        if len(existing) >= target_size:
            print(f"{CANDIDATE_LIST_PATH} already has {len(existing)} candidates (>= target "
                  f"{target_size}) -- reusing the tic_id/sector list (resume support).")
            return existing[["tic_id", "sector", "n_sectors_observed", "sectors_observed"]]

    sectors, sector_obs_df = find_recent_sectors()
    sectors_used = sectors[:n_sectors]
    print(f"Using sectors: {sectors_used}")

    # map each TIC ID to the set of sectors (within sectors_used) it was observed in
    tic_to_sectors = {}
    for sector in sectors_used:
        sector_rows = sector_obs_df[sector_obs_df["sequence_number"] == sector]
        for tic_str in sector_rows["target_name"].dropna().unique():
            try:
                tic_id = int(tic_str)
            except (ValueError, TypeError):
                continue
            if tic_id in excluded_tics:
                continue
            tic_to_sectors.setdefault(tic_id, set()).add(int(sector))

    print(f"{len(tic_to_sectors)} unflagged unique TIC IDs observed across these {len(sectors_used)} sectors")

    records = []
    for tic_id, obs_sectors in tic_to_sectors.items():
        records.append({
            "tic_id": tic_id,
            "n_sectors_observed": len(obs_sectors),
            "sectors_observed": ",".join(str(s) for s in sorted(obs_sectors, reverse=True)),
            "sector": max(obs_sectors),   # most recent sector observed, kept for backward-compat display
        })
    pool = pd.DataFrame(records)

    # multi-sector coverage first (descending), then most-recent-sector-observed
    # as the tiebreaker (descending) -- implements the priority order above
    pool = pool.sort_values(["n_sectors_observed", "sector"], ascending=[False, False]).reset_index(drop=True)

    multi_sector_available = int((pool["n_sectors_observed"] > 1).sum())
    print(f"Of these, {multi_sector_available} have multi-sector coverage (observed in >1 of "
          f"the {len(sectors_used)} sectors considered) -- these are prioritized first.")

    selected = pool.head(target_size)
    n_multi_selected = int((selected["n_sectors_observed"] > 1).sum())
    print(f"Selected {len(selected)} candidates: {n_multi_selected} multi-sector, "
          f"{len(selected) - n_multi_selected} single-sector.")

    selected.to_csv(CANDIDATE_LIST_PATH, index=False)
    print(f"Candidate pool saved to {CANDIDATE_LIST_PATH}")
    return selected


# =====================================
# STAGE C: STELLAR PARAMETERS
# =====================================
def fetch_stellar_params(candidates_df):
    print("\n" + "=" * 60)
    print("STAGE C: fetching stellar parameters (radius, mass, Teff) from the TIC catalog")
    print("=" * 60)
    from astroquery.mast import Catalogs

    tic_ids = candidates_df["tic_id"].tolist()
    chunks = [tic_ids[i:i + 500] for i in range(0, len(tic_ids), 500)]
    rows = []
    for chunk in tqdm(chunks, desc="Querying TIC catalog"):
        r = Catalogs.query_criteria(catalog="Tic", ID=chunk)
        rows.append(r[["ID", "ra", "dec", "rad", "e_rad", "mass", "e_mass", "Teff"]].to_pandas())
    stellar = pd.concat(rows, ignore_index=True)
    stellar["ID"] = stellar["ID"].astype("int64")
    # e_rad/e_mass (Part D: uncertainty propagation) are additive alongside
    # the existing point estimates -- feed the Monte Carlo propagation in
    # 08_characterize_candidates.py, never used for the TLS search grid or
    # any point-estimate calculation itself.
    stellar = stellar.rename(columns={"ID": "tic_id", "rad": "st_rad", "e_rad": "st_rad_err",
                                       "mass": "st_mass", "e_mass": "st_mass_err", "Teff": "st_teff"})
    stellar = stellar.drop_duplicates(subset="tic_id")

    merged = candidates_df.merge(stellar, on="tic_id", how="left")
    n_missing = merged["st_rad"].isna().sum()
    print(f"Stellar params found for {len(merged) - n_missing}/{len(merged)} candidates "
          f"({n_missing} missing -- will fall back to solar defaults for the TLS search grid only, "
          f"same as this project's established handling elsewhere).")
    return merged


# =====================================
# STAGE D: DOWNLOAD (reuses 01_download_known.py's proven pattern exactly)
# =====================================
def file_has_time_column(path):
    try:
        header = pd.read_csv(path, nrows=0).columns
        return "time" in header
    except Exception:
        return False


def try_search(tic_id):
    from lightkurve import search_lightcurve
    search = search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
    if len(search) > 0:
        return search, "tic_id_spoc"
    search = search_lightcurve(f"TIC {tic_id}", mission="TESS")
    if len(search) > 0:
        return search, "tic_id"
    return search, "none"


def download_one_star(tic_id, filename, target_sectors=None):
    """target_sectors: if given (multi-sector mode), fetch EVERY matching
    product for those specific sectors and concatenate them into one
    stitched raw light curve, rather than only the first search result --
    otherwise the multi-sector-aware candidate SELECTION is pointless, since
    only a single sector's worth of data would ever actually get downloaded
    (a real bug found and fixed: the original version of this function did
    exactly that, silently defeating the entire point of --multi-sector)."""
    timings = {"t_search": 0.0, "t_download": 0.0, "search_method": "none"}
    t0 = time.monotonic()
    search, method = try_search(tic_id)
    timings["t_search"] = time.monotonic() - t0
    timings["search_method"] = method
    if len(search) == 0:
        return {"host": filename, "status": "No TESS Data", **timings}

    t1 = time.monotonic()
    if target_sectors:
        sector_col = list(search.table["sequence_number"])
        seen_sectors = set()
        indices = []
        for i, s in enumerate(sector_col):
            if s in target_sectors and s not in seen_sectors:
                indices.append(i)
                seen_sectors.add(s)
        matching = search[indices] if indices else search[:1]

        frames = []
        for i in range(len(matching)):
            lc = _safe_download(matching[i])
            if lc is not None:
                frames.append(lc.to_pandas().reset_index())
        timings["t_download"] = time.monotonic() - t1
        timings["n_sectors_downloaded"] = len(frames)
        if not frames:
            return {"host": filename, "status": "Download Failed", **timings}
        df_out = pd.concat(frames, ignore_index=True)
        if "time" in df_out.columns:
            df_out = df_out.sort_values("time").reset_index(drop=True)
    else:
        lc = _safe_download(search[0])
        timings["t_download"] = time.monotonic() - t1
        if lc is None:
            return {"host": filename, "status": "Download Failed", **timings}
        df_out = lc.to_pandas().reset_index()

    if "time" not in df_out.columns:
        return {"host": filename, "status": "Missing 'time' column after reset_index", **timings}
    csv_path = os.path.join(RAW_FOLDER, filename + ".csv")
    df_out.to_csv(csv_path, index=False)
    return {"host": filename, "status": "Success", **timings}


def download_candidates(candidates_df):
    print("\n" + "=" * 60)
    print("STAGE D: downloading light curves")
    print("=" * 60)

    already_downloaded = set()
    if os.path.isdir(RAW_FOLDER):
        for f in os.listdir(RAW_FOLDER):
            if f.endswith(".csv") and file_has_time_column(os.path.join(RAW_FOLDER, f)):
                already_downloaded.add(os.path.splitext(f)[0])
    print(f"{len(already_downloaded)} files already downloaded and valid -- skipping (resume support).")

    has_sector_info = "sectors_observed" in candidates_df.columns
    work_items = []
    for _, row in candidates_df.iterrows():
        tic_id = int(row["tic_id"])
        filename = f"TIC_{tic_id}"
        if filename in already_downloaded:
            continue
        target_sectors = None
        if has_sector_info and pd.notna(row.get("sectors_observed")):
            target_sectors = set(int(s) for s in str(row["sectors_observed"]).split(","))
        work_items.append((tic_id, filename, target_sectors))

    if not work_items:
        print("Nothing new to download.")
        return

    print(f"{len(work_items)} stars to download with {DOWNLOAD_WORKERS} workers"
          f"{' (multi-sector: fetching + stitching all matching sectors per star)' if has_sector_info else ''}...")
    log = []
    executor = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
    futures = [(executor.submit(download_one_star, tic_id, filename, target_sectors), filename)
               for tic_id, filename, target_sectors in work_items]

    downloaded, failed = 0, 0
    progress = tqdm(total=len(futures), desc="Downloading")
    for future, filename in futures:
        try:
            result = future.result(timeout=PER_STAR_DOWNLOAD_TIMEOUT)
            status = result["status"]
            if status == "Success":
                downloaded += 1
            else:
                failed += 1
            log.append(result)
        except FutureTimeoutError:
            failed += 1
            log.append({"host": filename, "status": f"Timed out after {PER_STAR_DOWNLOAD_TIMEOUT}s"})
        except Exception as e:
            failed += 1
            log.append({"host": filename, "status": str(e)})
        progress.update(1)
    progress.close()
    # REVERTED to wait=False (tried wait=True to chase the stdout-closure
    # crash below, but that was WRONG and much worse: a download thread's
    # underlying network call can hang with no low-level timeout of its own
    # -- future.result(timeout=...) above only bounds how long WE wait, not
    # how long the thread itself keeps blocking on a stuck socket read -- so
    # wait=True can block the entire pipeline indefinitely (confirmed: saw a
    # ~4hr hang with almost no CPU time used, i.e. stuck on I/O). A rare
    # cosmetic crash after data is already safely saved (handled by the
    # process-level retry loop the caller wraps this script in) is a far
    # better tradeoff than risking an unbounded hang.
    executor.shutdown(wait=False)

    log_df = pd.DataFrame(log)
    if os.path.exists(DOWNLOAD_LOG_PATH):
        old = pd.read_csv(DOWNLOAD_LOG_PATH)
        log_df = pd.concat([old, log_df], ignore_index=True)
    log_df.to_csv(DOWNLOAD_LOG_PATH, index=False)
    # Data is already saved above -- this is just a summary print. A leftover
    # ThreadPoolExecutor worker thread (still running after shutdown(wait=False),
    # since we deliberately don't block on stragglers) can race the main thread's
    # own print here under some execution contexts (seen throughout this project
    # whenever stdout is redirected). Cosmetic only; wrap so it can't kill the
    # rest of the pipeline the way it did before this fix.
    try:
        print(f"Downloaded: {downloaded}, Failed: {failed}")
    except ValueError:
        pass


# =====================================
# STAGE E: PREPROCESSING (ported verbatim from 02_preprocess.py)
# =====================================
# PHASE 3 CHANGE: pdcsap_flux/pdcsap_flux_err are no longer REQUIRED columns.
#
# Measured cost of requiring them (Phase 2 blind validation): of 89 known TOI
# false positives absent from the training set, 88 were rejected here -- every
# one of them downloaded fine, then failed schema validation purely because
# the product had no pdcsap_flux COLUMN. Those targets have no SPOC 2-minute
# data at all, only FFI-derived products (QLP, GSFC-ELEANOR-LITE, TARS). The
# same requirement also cost 1 of 10 genuinely-new confirmed planets.
#
# The population this excluded was not random: those 89 stars are a median
# 2.19 mag FAINTER than the 1,155 FPs that did make it into training (Tmag
# 12.73 vs 10.55). So the pipeline had a systematic blind spot toward fainter
# targets, and -- because the training negative class was built through this
# same gate -- that blind spot was invisible from inside the training data.
#
# choose_flux_columns() ALREADY falls back to the generic `flux` column; it
# simply never got the chance, because validation rejected the file first.
# For QLP products `flux` is the KSPSAP detrended photometry (verified live:
# fully populated, no NaNs), not raw SAP, so this is a reasonable input rather
# than a desperate one. It is still a DIFFERENT photometry pipeline than
# PDCSAP, which is why choose_flux_columns reports which source was used --
# see flux_source_note() -- so any downstream result can be traced back to it.
REQUIRED_COLUMNS = {"time", "flux", "flux_err", "quality"}


def validate_schema(df):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return f"Non-standard schema (missing columns: {sorted(missing)})"
    if df["quality"].dtype == object:
        return "Non-standard schema (quality column is non-numeric)"
    if len(df) == 0:
        return "Empty file (zero rows)"
    return None


def choose_flux_columns(df):
    """Preference order is unchanged -- PDCSAP first, always. The only change
    is that a MISSING pdcsap column is now treated the same as an all-NaN one
    (fall through) instead of being impossible to reach because validation
    rejected the file first."""
    if "pdcsap_flux" in df.columns and df["pdcsap_flux"].notna().any():
        return df["pdcsap_flux"].to_numpy(), df["pdcsap_flux_err"].to_numpy(), "pdcsap_flux"
    if df["flux"].notna().any():
        return df["flux"].to_numpy(), df["flux_err"].to_numpy(), "flux"
    return None, None, "No usable flux data"


def choose_savgol_window(n_points, max_window=MAX_FLATTEN_WINDOW, polyorder=SAVGOL_POLYORDER):
    window = min(max_window, n_points - 1)
    if window % 2 == 0:
        window -= 1
    if window < polyorder + 2 or window < 5:
        return None
    return window


def clean_light_curve(csv_path):
    """Identical cleaning logic to 02_preprocess.py's process_one_file(), minus
    the QC-stat bookkeeping this script doesn't need. Returns a cleaned
    DataFrame (time, flux, flux_err) or None with a reason."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, f"Read error: {e}"

    schema_problem = validate_schema(df)
    if schema_problem:
        return None, schema_problem

    flux, flux_err, source = choose_flux_columns(df)
    if flux is None:
        return None, source

    time_arr = df["time"].to_numpy()
    quality = df["quality"].to_numpy()

    valid = ~np.isnan(time_arr) & ~np.isnan(flux) & ~np.isnan(flux_err)
    time_arr, flux, flux_err, quality = time_arr[valid], flux[valid], flux_err[valid], quality[valid]

    good_quality = quality == 0
    time_arr, flux, flux_err = time_arr[good_quality], flux[good_quality], flux_err[good_quality]
    if len(flux) < MIN_POINTS_FOR_FLATTEN:
        return None, f"Only {len(flux)} points survived quality filtering"

    order = np.argsort(time_arr, kind="stable")
    time_arr, flux, flux_err = time_arr[order], flux[order], flux_err[order]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clip_result = sigma_clip(flux, sigma=SIGMA_CLIP_THRESHOLD, stdfunc="mad_std", maxiters=5, masked=True)
    keep = ~clip_result.mask
    time_arr, flux, flux_err = time_arr[keep], flux[keep], flux_err[keep]
    if len(flux) < MIN_POINTS_FOR_FLATTEN:
        return None, f"Only {len(flux)} points survived outlier removal"

    window = choose_savgol_window(len(flux))
    if window is None:
        return None, f"{len(flux)} points too few for a valid flatten window"

    trend = savgol_filter(flux, window_length=window, polyorder=SAVGOL_POLYORDER, mode="interp")
    flat_flux = flux / trend
    flat_err = flux_err / trend
    if not np.all(np.isfinite(flat_flux)):
        return None, "Non-finite values in flattened flux (degenerate trend)"

    median_flat = np.median(flat_flux)
    flat_flux = flat_flux / median_flat
    flat_err = flat_err / median_flat

    # Carry the photometry source through with the data rather than dropping
    # it here. Now that non-PDCSAP products are accepted, "which pipeline
    # produced this flux" is a real caveat attached to every downstream
    # number, and it must not be silently lost between preprocessing and the
    # candidate table.
    out = pd.DataFrame({"time": time_arr, "flux": flat_flux, "flux_err": flat_err})
    # A real column, not df.attrs: this DataFrame gets written to CSV and
    # re-read later by compute_all_features, and attrs do not survive that
    # round-trip. compute_all_features only reads time/flux/flux_err, so the
    # extra column is inert -- it exists so the provenance is still attached
    # to the data when someone looks at a processed file months later.
    out["flux_source"] = source
    status = "Success" if source == "pdcsap_flux" else f"Success (flux source: {source})"
    return out, status


def preprocess_candidates():
    print("\n" + "=" * 60)
    print("STAGE E: preprocessing (identical logic to 02_preprocess.py)")
    print("=" * 60)

    raw_files = [f for f in os.listdir(RAW_FOLDER) if f.endswith(".csv")]
    already_processed = set()
    for f in os.listdir(PROCESSED_FOLDER) if os.path.isdir(PROCESSED_FOLDER) else []:
        if f.endswith(".csv"):
            already_processed.add(os.path.splitext(f)[0])

    n_success, n_skipped = 0, 0
    for f in tqdm(raw_files, desc="Preprocessing"):
        host = os.path.splitext(f)[0]
        if host in already_processed:
            continue
        cleaned, status = clean_light_curve(os.path.join(RAW_FOLDER, f))
        if cleaned is not None:
            cleaned.to_csv(os.path.join(PROCESSED_FOLDER, host + ".csv"), index=False)
            n_success += 1
        else:
            n_skipped += 1
    print(f"Preprocessed this run: {n_success} success, {n_skipped} skipped "
          f"(+{len(already_processed)} already done from a prior run)")


# =====================================
# STAGE F: TLS FEATURE EXTRACTION (real stellar params, same as 05d)
# =====================================
FEATURE_METADATA_PATH = os.path.join(MODELS_FOLDER, "best_model_metadata.json")

# Features allowed to be absent without disqualifying a star (see the long
# note in compute_all_features for why only these two). Both are undefined
# -- not merely noisy -- when a light curve has too few transits to
# characterise, which is a property of the observing window rather than of
# the star.
OPTIONAL_FEATURES = {"transit_shape_ratio", "FAP"}


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


def compute_all_features(csv_path, host, r_star, m_star, required_columns):
    """Runs TLS once with real stellar params and extracts every scalar TLS
    field the model needs PLUS the v2 feature set (chi2red_min,
    depth_consistency_std, secondary_eclipse_depth, transit_shape_ratio,
    depth_duration_ratio) -- one TLS call, not two, mirroring 05d's approach.
    Returns (feature_dict_or_None, status_string). A star only proceeds to
    scoring if EVERY required feature is present and finite -- this function
    fails loudly (returns None) rather than silently leaving gaps."""
    from transitleastsquares import transitleastsquares

    t0 = time.monotonic()
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, f"Read error: {e}"
    if len(df) < 50:
        return None, f"Too few points ({len(df)})"

    t_arr, f_arr, e_arr = bin_lightcurve(df["time"].to_numpy(), df["flux"].to_numpy(), df["flux_err"].to_numpy())

    r_star = r_star if (r_star and not np.isnan(r_star) and r_star > 0) else 1.0
    m_star = m_star if (m_star and not np.isnan(m_star) and m_star > 0) else 1.0
    r_star_min, r_star_max = min(0.13, r_star * 0.5), max(3.5, r_star * 1.5)
    m_star_min, m_star_max = min(0.1, m_star * 0.5), max(1.0, m_star * 1.5)

    try:
        model = transitleastsquares(t_arr, f_arr, e_arr)
        r = model.power(
            use_threads=1, oversampling_factor=1, duration_grid_step=1.1, show_progress_bar=False,
            R_star=r_star, R_star_min=r_star_min, R_star_max=r_star_max,
            M_star=m_star, M_star_min=m_star_min, M_star_max=m_star_max,
        )
    except Exception as e:
        return None, f"TLS error: {e}"

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

        depths = np.asarray(r.transit_depths, dtype=float)
        depths = depths[~np.isnan(depths)]
        depth_std = float(np.std(depths)) if len(depths) > 1 else np.nan

        feats = {
            "SDE": float(r.SDE), "SDE_raw": float(r.SDE_raw), "FAP": float(r.FAP),
            "period": float(r.period), "period_uncertainty": float(r.period_uncertainty),
            "T0": float(r.T0), "duration": float(r.duration), "depth": float(r.depth),
            "depth_mean": float(r.depth_mean[0]), "depth_mean_std": float(r.depth_mean[1]),
            "depth_mean_even": float(r.depth_mean_even[0]), "depth_mean_odd": float(r.depth_mean_odd[0]),
            "odd_even_mismatch": float(r.odd_even_mismatch), "rp_rs": float(r.rp_rs), "snr": float(r.snr),
            "transit_count": float(r.transit_count), "distinct_transit_count": float(r.distinct_transit_count),
            "empty_transit_count": float(r.empty_transit_count),
            "chi2red_min": float(r.chi2red_min), "depth_consistency_std": depth_std,
            "secondary_eclipse_depth": secondary_depth, "transit_shape_ratio": shape_ratio,
        }
        feats["depth_duration_ratio"] = feats["depth"] / feats["duration"] if feats["duration"] else np.nan
    except Exception as e:
        return None, f"Post-processing error: {e}"

    missing_or_bad = [
        c for c in required_columns
        if c not in ("st_rad", "st_teff")   # these come from the catalog, not TLS -- checked separately
        and (c not in feats or feats[c] is None or (isinstance(feats[c], float) and not np.isfinite(feats[c])))
    ]

    # PHASE 3 CHANGE: a small set of features may now be MISSING rather than
    # disqualifying. Measured cost of the old all-or-nothing rule (Phase 2):
    # 7 of 10 genuinely-new confirmed planets were discarded here, every one
    # of them on transit_shape_ratio, four also on FAP. Those were real
    # planets with real detections -- periods 14-180 days, whose transits are
    # simply too few in a single TESS sector for an edge-vs-centre shape
    # measurement to exist. Signal quality was never the issue.
    #
    # Only these two are optional, and for a specific reason: both are
    # UNDEFINED for sparse transits rather than merely noisy, so their absence
    # carries information ("too few transits to characterise") instead of
    # indicating a broken star. Everything else stays mandatory -- a missing
    # period or depth means the detection itself failed, and no imputation can
    # honestly stand in for that.
    #
    # IMPORTANT CAVEAT, and the reason the status string says so out loud:
    # the classifier's imputer fills these with the TRAINING MEDIAN, and it
    # was fit on data where they were always present. A candidate scored this
    # way is being judged partly on a stand-in value, not a measurement, so
    # its probability is less trustworthy than one from a complete feature
    # vector. See the measured effect of this in
    # results/tables/optional_feature_impact.csv -- it is not negligible.
    optional = [c for c in missing_or_bad if c in OPTIONAL_FEATURES]
    blocking = [c for c in missing_or_bad if c not in OPTIONAL_FEATURES]
    if blocking:
        return None, f"Required feature(s) not computable: {blocking}"
    for c in optional:
        feats[c] = np.nan
    if optional:
        return feats, f"Success (imputed, not measured: {sorted(optional)})"

    feats["elapsed_s"] = time.monotonic() - t0
    return feats, "Success"


def _tls_worker(args):
    csv_path, host, r_star, m_star, required_columns = args
    feats, status = compute_all_features(csv_path, host, r_star, m_star, required_columns)
    return host, feats, status


def extract_features(candidates_df, required_columns):
    print("\n" + "=" * 60)
    print("STAGE F: TLS feature extraction with REAL per-star stellar params")
    print("=" * 60)

    already_done = set()
    if os.path.exists(FEATURES_PATH):
        old = pd.read_csv(FEATURES_PATH)
        already_done = set(old["host"])
        print(f"{len(already_done)} stars already have features -- resuming from there.")

    work_list = []
    for _, row in candidates_df.iterrows():
        host = f"TIC_{int(row['tic_id'])}"
        if host in already_done:
            continue
        path = os.path.join(PROCESSED_FOLDER, host + ".csv")
        if os.path.exists(path):
            work_list.append((path, host, row.get("st_rad"), row.get("st_mass"), required_columns))

    print(f"{len(work_list)} stars to run TLS on with {TLS_WORKERS} workers, "
          f"saving progress every {TLS_BATCH_SIZE} stars...")

    # Saves after every batch (not just once at the end) -- this stage is the
    # most expensive in the pipeline, and this environment has shown it can
    # crash for reasons unrelated to the computation itself (a stdout race);
    # without incremental checkpointing, a crash partway through would lose
    # everything computed so far, same lesson as 05d_recompute_stellar_ttv.py.
    progress = tqdm(total=len(work_list), desc="TLS feature extraction")
    for batch_start in range(0, len(work_list), TLS_BATCH_SIZE):
        batch = work_list[batch_start:batch_start + TLS_BATCH_SIZE]
        batch_rows = []
        executor = ProcessPoolExecutor(max_workers=TLS_WORKERS)
        try:
            futures = [(executor.submit(_tls_worker, item), item[1]) for item in batch]
            for future, host in futures:
                try:
                    h, feats, status = future.result(timeout=TLS_PER_STAR_TIMEOUT)
                except FutureTimeoutError:
                    h, feats, status = host, None, f"Timed out after {TLS_PER_STAR_TIMEOUT}s"
                except Exception as e:
                    h, feats, status = host, None, f"Worker error: {e}"
                row = {"host": h, "status": status}
                if feats:
                    row.update(feats)
                batch_rows.append(row)
                progress.update(1)
        finally:
            executor.shutdown(wait=False)

        if batch_rows:
            new_df = pd.DataFrame(batch_rows)
            if os.path.exists(FEATURES_PATH):
                old = pd.read_csv(FEATURES_PATH)
                combined = pd.concat([old, new_df], ignore_index=True).drop_duplicates(subset="host", keep="last")
            else:
                combined = new_df
            combined.to_csv(FEATURES_PATH, index=False)
    progress.close()

    final = pd.read_csv(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else pd.DataFrame()
    n_success = (final["status"] == "Success").sum() if len(final) else 0
    print(f"Feature extraction: {n_success}/{len(final)} stars succeeded with every required feature computable.")
    return final


# =====================================
# STAGE G: SCORE + RANK
# =====================================
def score_candidates(features_df, candidates_df, feature_columns):
    print("\n" + "=" * 60)
    print("STAGE G: scoring with the saved model")
    print("=" * 60)

    model = joblib.load(os.path.join(MODELS_FOLDER, "best_model.joblib"))

    # BUG FIXED: st_mass was fetched from the TIC catalog (fetch_stellar_params)
    # and even used to run the actual TLS fit above (see _tls_worker/work_list),
    # but was dropped here and never made it into the saved ranked-candidates
    # CSV -- so every downstream consumer (07, 08) saw a missing st_mass for
    # every candidate regardless of what TIC actually had on file. Found via
    # 08's new "stellar_mass_was_defaulted_to_solar" badge showing 105/105
    # defaulted, which shouldn't happen when ~48% of the underlying pool has
    # a real catalog mass.
    # st_rad_err/st_mass_err (Part D: uncertainty propagation) -- same drop-at-
    # the-merge risk the st_mass bug above already exposed once, so they're
    # included explicitly here rather than assumed to ride along automatically.
    optional_cols = [c for c in ("n_sectors_observed", "sectors_observed", "ra", "dec",
                                  "st_rad_err", "st_mass_err") if c in candidates_df.columns]
    # startswith, not equality: compute_all_features now also returns
    # "Success (imputed, not measured: [...])" for stars that cleared every
    # blocking feature but had an optional one undefined. An exact-match test
    # here would silently discard exactly the stars the optional-feature
    # change was made to rescue.
    success = features_df[features_df["status"].astype(str).str.startswith("Success")].copy()
    success = success.merge(
        candidates_df.assign(host=lambda d: "TIC_" + d["tic_id"].astype("int64").astype(str))[
            ["host", "st_rad", "st_teff", "st_mass", "sector"] + optional_cols],
        on="host", how="left",
    )

    if len(success) == 0:
        print("No candidates succeeded feature extraction this run -- nothing to score "
              "(not a schema error, just an empty result for this batch).")
        return pd.DataFrame()

    missing_required = [c for c in feature_columns if c not in success.columns]
    if missing_required:
        raise SystemExit(f"FATAL: required feature columns missing entirely from the feature table: "
                          f"{missing_required} -- refusing to score with a mismatched feature set.")

    # A NaN in a BLOCKING feature still disqualifies a star -- the original
    # reasoning holds: better no score than a meaningless one. But NaNs in
    # OPTIONAL_FEATURES are now expected and deliberate, so they must not
    # re-trigger the same exclusion one stage later.
    blocking_cols = [c for c in feature_columns if c not in OPTIONAL_FEATURES]
    still_bad = success[blocking_cols].isna().any(axis=1)
    if still_bad.any():
        print(f"Excluding {still_bad.sum()} stars with a NaN in a required feature "
              f"(st_rad/st_teff missing from the TIC catalog) -- not scoring these rather than "
              f"silently imputing and presenting a meaningless probability.")
        success = success[~still_bad]

    # Flag -- per star -- whether any feature behind its probability was
    # imputed rather than measured, so this cannot be lost between here and
    # the candidate table the user actually reads.
    imputed_mask = success[list(OPTIONAL_FEATURES & set(feature_columns))].isna()
    success["imputed_features"] = imputed_mask.apply(
        lambda r: ",".join(sorted(c for c, v in r.items() if v)) or "", axis=1)
    n_imputed = (success["imputed_features"] != "").sum()
    if n_imputed:
        print(f"{n_imputed} star(s) scored with at least one IMPUTED (not measured) feature -- "
              f"their probabilities carry more uncertainty than the model's headline metrics imply; "
              f"see the imputed_features column.")

    if len(success) == 0:
        print("No candidates have a complete feature set -- nothing to score.")
        return pd.DataFrame()

    X = success[feature_columns]
    proba = model.predict_proba(X)[:, 1]
    success["predicted_probability"] = proba
    ranked = success.sort_values("predicted_probability", ascending=False).reset_index(drop=True)

    ranked.to_csv(RANKED_OUTPUT_PATH, index=False)
    print(f"Ranked {len(ranked)} candidates, saved to {RANKED_OUTPUT_PATH}")
    return ranked


# =====================================
# STAGE G.5: OUT-OF-DISTRIBUTION SAFEGUARD
#
# Found during the first 300-star run: several top-ranked candidates had
# st_rad/st_teff values exceeding anything in the training set (e.g.
# st_rad=230 vs. a training max of 88.5), and since st_rad/st_teff are the
# model's top two importance-weighted features, this can inflate a
# candidate's score without reflecting genuine transit-signal quality --
# tree-based models (HistGradientBoosting/RandomForest) have no learned
# behavior beyond the range of values they were trained on; past the max
# they've ever seen, they just keep following whichever split direction was
# last learned. MIN/MAX is used here (not the 5th/95th percentile band)
# because the concept being flagged is specifically EXTRAPOLATION beyond the
# model's training support, not general "unusualness" -- a percentile band
# would by construction flag ~10% of the training data itself, which
# measures something different (rarity, not model extrapolation risk).
# Percentile position is still included in the explanation text as softer
# context for how unusual a value is even when it's technically in-range.
# =====================================
FEATURE_RANGES_PATH = os.path.join(MODELS_FOLDER, "training_feature_ranges.json")


def compute_training_feature_ranges(feature_columns):
    """Computes min/max/p5/p95/mean/std for every model feature from the
    training set and saves it to disk. Reusable across runs of this script --
    only needs to be regenerated when the model itself is retrained on
    different/updated training data, not on every unknown-star run."""
    df = pd.read_csv(TRAINING_PATH)
    ranges = {}
    for col in feature_columns:
        if col not in df.columns:
            continue
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) == 0:
            continue
        ranges[col] = {
            "min": float(series.min()), "max": float(series.max()),
            "p5": float(series.quantile(0.05)), "p95": float(series.quantile(0.95)),
            "mean": float(series.mean()), "std": float(series.std()),
            "n": int(len(series)),
        }
    with open(FEATURE_RANGES_PATH, "w") as f:
        json.dump({
            "source": TRAINING_PATH, "training_rows": len(df),
            "note": "min/max used as the out-of-distribution flagging criterion (extrapolation "
                    "beyond training support); p5/p95 included as softer 'how unusual' context only.",
            "ranges": ranges,
        }, f, indent=2)
    print(f"Computed training feature ranges from {len(df)} rows, saved to {FEATURE_RANGES_PATH}")
    return ranges


def load_or_compute_feature_ranges(feature_columns, force_recompute=False):
    if not force_recompute and os.path.exists(FEATURE_RANGES_PATH):
        with open(FEATURE_RANGES_PATH) as f:
            saved = json.load(f)
        print(f"Reusing cached training feature ranges from {FEATURE_RANGES_PATH} "
              f"(computed from {saved.get('training_rows')} training rows). Delete this file "
              f"or pass --recompute-ranges if the model has since been retrained on different data.")
        return saved["ranges"]
    return compute_training_feature_ranges(feature_columns)


def flag_out_of_distribution(ranked_df, feature_columns, ranges):
    """Adds 'in_distribution_univariate' (bool) and 'ood_features' (human-
    readable detail string, empty if none) columns. A candidate is flagged
    if ANY single feature falls outside the training set's observed min/max.
    This catches extrapolation on individual features, but NOT implausible
    COMBINATIONS of individually in-range values (e.g. a near-substellar
    radius paired with an O-star temperature) -- see flag_multivariate_ood()
    below for that."""
    in_distribution_flags = []
    ood_detail_strings = []

    for _, row in ranked_df.iterrows():
        violations = []
        for feat in feature_columns:
            if feat not in ranges or feat not in row or pd.isna(row[feat]):
                continue
            val = row[feat]
            r = ranges[feat]
            if val < r["min"]:
                pct_over = (r["min"] - val) / (abs(r["min"]) + 1e-12) * 100
                violations.append(f"{feat}={val:.4g} (below training min {r['min']:.4g}, "
                                   f"{pct_over:.0f}% under)")
            elif val > r["max"]:
                pct_over = (val - r["max"]) / (abs(r["max"]) + 1e-12) * 100
                violations.append(f"{feat}={val:.4g} (above training max {r['max']:.4g}, "
                                   f"{pct_over:.0f}% over)")
        in_distribution_flags.append(len(violations) == 0)
        ood_detail_strings.append("; ".join(violations))

    ranked_df = ranked_df.copy()
    ranked_df["in_distribution_univariate"] = in_distribution_flags
    ranked_df["ood_features"] = ood_detail_strings
    return ranked_df


# =====================================
# MULTIVARIATE OOD CHECK (Isolation Forest)
#
# The univariate min/max check above misses implausible COMBINATIONS of
# individually in-range values -- found via TIC_421999342 (st_rad=0.18,
# st_teff=31000K: each value alone sits within the training min/max, but a
# near-substellar radius paired with an O-star temperature almost certainly
# never occurred together in training). Isolation Forest is used here rather
# than Mahalanobis distance (which assumes an elliptical/Gaussian feature
# space -- a poor fit for this dataset's heavily skewed features like
# chi2red_min, which spans ~8 orders of magnitude) or hand-coded domain
# rules (which would reliably catch only the ONE combination already found,
# not generalize to unknown implausible combinations among the other 22
# features). Isolation Forest makes no distributional assumption and
# naturally captures multi-feature interactions -- anomalies get isolated
# in fewer random tree partitions than typical points, by construction.
# Tradeoff: it's a black box relative to the univariate check -- it flags
# THAT something is unusual without cleanly attributing WHICH feature pair
# drove it. The threshold is calibrated against real training data (not
# assumed) so its false-positive rate is measured, not guessed.
# =====================================
MULTIVARIATE_OOD_MODEL_PATH = os.path.join(MODELS_FOLDER, "multivariate_ood_detector.joblib")
MULTIVARIATE_OOD_META_PATH = os.path.join(MODELS_FOLDER, "multivariate_ood_meta.json")
ISOLATION_FOREST_CONTAMINATION = 0.02   # nominal target; actual resulting training flag-rate is measured and reported, not assumed


def compute_multivariate_ood_detector(feature_columns):
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer

    df = pd.read_csv(TRAINING_PATH)
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    if "FAP" in X.columns:
        X["FAP"] = X["FAP"].fillna(1.0)   # same domain-specific fill as the main model's build_feature_matrix()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    iso_forest = IsolationForest(
        n_estimators=200, contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=42, n_jobs=-1,
    )
    iso_forest.fit(X_imputed)

    # Calibrate/measure against the training data itself, rather than trusting
    # the contamination parameter's nominal value -- sklearn's contamination
    # setting is an approximate target, not a guarantee.
    train_scores = iso_forest.score_samples(X_imputed)   # higher = more normal
    threshold = float(np.percentile(train_scores, ISOLATION_FOREST_CONTAMINATION * 100))
    actual_flagged_frac = float((train_scores < threshold).mean())

    joblib.dump({"imputer": imputer, "iso_forest": iso_forest, "feature_columns": feature_columns},
                MULTIVARIATE_OOD_MODEL_PATH)
    meta = {
        "source": TRAINING_PATH, "training_rows": len(df),
        "contamination_target": ISOLATION_FOREST_CONTAMINATION,
        "threshold_score": threshold,
        "actual_training_flagged_fraction": actual_flagged_frac,
        "note": "actual_training_flagged_fraction is the measured false-positive baseline: this is "
                "the fraction of REAL, legitimate training examples (confirmed planets + confirmed "
                "false positives) that this detector would ALSO flag if scored against itself -- "
                "the honest false-positive rate to expect, not the nominal contamination target.",
    }
    with open(MULTIVARIATE_OOD_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Multivariate OOD detector fit on {len(df)} training rows. Measured false-positive "
          f"baseline: {actual_flagged_frac:.1%} of real training examples would themselves be "
          f"flagged (target was {ISOLATION_FOREST_CONTAMINATION:.1%}).")
    return {"imputer": imputer, "iso_forest": iso_forest, "threshold": threshold}


def load_or_compute_multivariate_detector(feature_columns, force_recompute=False):
    if not force_recompute and os.path.exists(MULTIVARIATE_OOD_MODEL_PATH) and os.path.exists(MULTIVARIATE_OOD_META_PATH):
        bundle = joblib.load(MULTIVARIATE_OOD_MODEL_PATH)
        with open(MULTIVARIATE_OOD_META_PATH) as f:
            meta = json.load(f)
        print(f"Reusing cached multivariate OOD detector from {MULTIVARIATE_OOD_MODEL_PATH} "
              f"(measured training false-positive rate: {meta['actual_training_flagged_fraction']:.1%}). "
              f"Delete this file or pass --recompute-ranges if the model has since been retrained.")
        return {"imputer": bundle["imputer"], "iso_forest": bundle["iso_forest"], "threshold": meta["threshold_score"]}
    return compute_multivariate_ood_detector(feature_columns)


def flag_multivariate_ood(ranked_df, feature_columns, detector):
    X = ranked_df[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    if "FAP" in X.columns:
        X["FAP"] = X["FAP"].fillna(1.0)
    X_imputed = detector["imputer"].transform(X)
    scores = detector["iso_forest"].score_samples(X_imputed)

    ranked_df = ranked_df.copy()
    ranked_df["multivariate_ood_score"] = scores
    ranked_df["multivariate_ood_flag"] = scores < detector["threshold"]
    return ranked_df


def combine_ood_flags(ranked_df):
    """Overall in_distribution requires passing BOTH checks -- they catch
    genuinely different failure modes (a single feature out of range vs. an
    implausible combination of in-range values), so either one failing is
    reason enough to flag a candidate for extra scrutiny."""
    ranked_df = ranked_df.copy()
    ranked_df["in_distribution"] = (
        ranked_df["in_distribution_univariate"] & ~ranked_df["multivariate_ood_flag"]
    )

    def _combine_detail(row):
        parts = []
        if row["ood_features"]:
            parts.append(row["ood_features"])
        if row["multivariate_ood_flag"]:
            parts.append(f"MULTIVARIATE: unusual combination of feature values "
                          f"(anomaly score={row['multivariate_ood_score']:.3f}, below the "
                          f"training-calibrated threshold -- individual features may each be "
                          f"in-range, but this specific combination is atypical)")
        return "; ".join(parts)

    ranked_df["ood_features"] = ranked_df.apply(_combine_detail, axis=1)
    return ranked_df


def split_and_rerank(ranked_df):
    """Splits into in-distribution (trustworthy shortlist) and
    out-of-distribution (flagged, needs independent stellar-param
    verification) tables, each re-sorted on its own by predicted probability
    -- not just the original list with a warning column bolted on."""
    in_dist = ranked_df[ranked_df["in_distribution"]].sort_values(
        "predicted_probability", ascending=False).reset_index(drop=True)
    out_dist = ranked_df[~ranked_df["in_distribution"]].sort_values(
        "predicted_probability", ascending=False).reset_index(drop=True)

    in_dist_path = os.path.join(RESULTS_FOLDER, "ranked_candidates_in_distribution.csv")
    out_dist_path = os.path.join(RESULTS_FOLDER, "ranked_candidates_out_of_distribution.csv")
    in_dist.to_csv(in_dist_path, index=False)
    out_dist.to_csv(out_dist_path, index=False)

    print("\n" + "=" * 60)
    print("OUT-OF-DISTRIBUTION IMPACT SUMMARY")
    print("=" * 60)
    print(f"Total ranked candidates: {len(ranked_df)}")
    print(f"In-distribution (trustworthy shortlist): {len(in_dist)} -- saved to {in_dist_path}")
    print(f"Out-of-distribution (needs stellar-param verification first): {len(out_dist)} -- "
          f"saved to {out_dist_path}")

    if len(ranked_df) > 0:
        original_top_hosts = ranked_df.head(min(10, len(ranked_df)))["host"].tolist()
        new_top_hosts = in_dist.head(min(10, len(in_dist)))["host"].tolist()
        if original_top_hosts == new_top_hosts:
            print("\nTop-10 is UNCHANGED after removing OOD candidates -- the original ranking's "
                  "top of the list was not distorted by this issue for this run.")
        else:
            n_dropped_from_top10 = len(set(original_top_hosts) - set(new_top_hosts))
            print(f"\nTop-10 CHANGED after removing OOD candidates -- {n_dropped_from_top10} of the "
                  f"original top-10 were OOD-flagged and are no longer in the top-10 once set aside. "
                  f"This means the OOD issue WAS materially distorting the original top of the list, "
                  f"not just a minor edge case.")

    return in_dist, out_dist


# =====================================
# STAGE H: EXPLANATIONS + PLOTS FOR TOP-N
# =====================================
def compute_reference_importance(feature_columns):
    """Fresh permutation importance on the training set (same technique used
    throughout this project's validation), so explanations are grounded in
    what actually drives THIS model, not an assumption."""
    from sklearn.inspection import permutation_importance
    import importlib.util

    spec = importlib.util.spec_from_file_location("tm", os.path.join(SCRIPT_DIR, "05_train_models.py"))
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)

    df = pd.read_csv(TRAINING_PATH)
    X, y = tm.build_feature_matrix(df)
    # BUG FIXED: was a positional train_test_split, which drifts as
    # training.csv grows -- meaning permutation importance (and therefore the
    # per-candidate explanations built from it) could be measured on stars the
    # model was trained on, overstating how much each feature really matters
    # on unseen data. Same stable star-ID split as everywhere else now.
    train_mask, test_mask = tm.split_by_host(df)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    model = joblib.load(os.path.join(MODELS_FOLDER, "best_model.joblib"))
    result = permutation_importance(model, X_test, y_test, n_repeats=15, random_state=tm.RANDOM_SEED,
                                     scoring="roc_auc", n_jobs=-1)
    importances = pd.Series(result.importances_mean, index=feature_columns).sort_values(ascending=False)

    # positive-class reference distribution, for percentile framing in explanations
    pos_df = df[df["label"] == 1]
    return importances, pos_df


def plain_language_explanation(row, importances, pos_df, top_k=4):
    top_feats = importances.head(top_k).index.tolist()
    parts = []
    for feat in top_feats:
        if feat not in row or pd.isna(row[feat]):
            continue
        val = row[feat]
        ref = pos_df[feat].dropna()
        if len(ref) == 0:
            continue
        percentile = (ref < val).mean() * 100
        parts.append(f"{feat}={val:.4g} ({percentile:.0f}th percentile among confirmed planets)")
    return "Flagged primarily due to: " + "; ".join(parts) if parts else "Flagged (feature comparison unavailable)"


def plot_folded_light_curve(host, csv_path, period, t0, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    phase = np.mod((df["time"].to_numpy() - t0) / period, 1.0)
    phase = np.where(phase > 0.5, phase - 1, phase)
    order = np.argsort(phase)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(phase[order], df["flux"].to_numpy()[order], s=2, alpha=0.4, color="steelblue")
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("Normalized flux")
    ax.set_title(f"{host} -- folded at period={period:.4f}d (UNEXAMINED CANDIDATE, not a confirmed planet)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def generate_top_n_output(in_dist_df, out_dist_df, feature_columns, top_n):
    print("\n" + "=" * 60)
    print(f"STAGE H: explanations + folded light curve plots for the top {top_n} (in-distribution)")
    print("=" * 60)

    if len(in_dist_df) == 0 and len(out_dist_df) == 0:
        print("Nothing to explain -- ranked lists are empty.")
        return

    importances, pos_df = compute_reference_importance(feature_columns)
    plots_dir = os.path.join(RESULTS_FOLDER, "folded_light_curves")
    os.makedirs(plots_dir, exist_ok=True)

    in_dist_explanations = []
    top_in_dist = in_dist_df.head(top_n)
    for i, (_, row) in enumerate(top_in_dist.iterrows()):
        host = row["host"]
        explanation = plain_language_explanation(row, importances, pos_df)
        in_dist_explanations.append(
            f"#{i+1}  {host}  (p={row['predicted_probability']:.3f}, sector {row.get('sector')})\n"
            f"    {explanation}\n"
        )
        if i < 10:
            csv_path = os.path.join(PROCESSED_FOLDER, host + ".csv")
            try:
                plot_folded_light_curve(host, csv_path, row["period"], row["T0"],
                                         os.path.join(plots_dir, f"{host}_folded.png"))
            except Exception as e:
                print(f"  Plot failed for {host}: {e}")

    # OOD candidates get a DIFFERENT explanation style: which feature(s) are
    # out of the training range and by how much, not the usual importance-based
    # framing -- a reviewer needs to know THIS is why the score may be
    # unreliable, before anything else about the candidate.
    ood_explanations = []
    top_ood = out_dist_df.head(top_n)
    for i, (_, row) in enumerate(top_ood.iterrows()):
        host = row["host"]
        ood_explanations.append(
            f"#{i+1}  {host}  (p={row['predicted_probability']:.3f}, sector {row.get('sector')})\n"
            f"    OUT OF DISTRIBUTION -- score may be unreliable. {row['ood_features']}\n"
        )

    with open(EXPLANATIONS_PATH, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("UNEXAMINED CANDIDATES RANKED FOR HUMAN REVIEW -- NOT CONFIRMED DETECTIONS\n")
        f.write("A high probability means 'worth a closer look by someone qualified to vet\n"
                "it', not 'this is a planet'. Real confirmation requires spectroscopic\n"
                "follow-up, multiple independent observations, and ruling out instrumental/\n"
                "astrophysical false-positive scenarios this light-curve-only model cannot see.\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"IN-DISTRIBUTION CANDIDATES (top {len(top_in_dist)} of {len(in_dist_df)} total) --\n")
        f.write("every feature value falls within the training data's observed range. This is\n")
        f.write("the trustworthy shortlist.\n\n")
        f.writelines(in_dist_explanations)
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"OUT-OF-DISTRIBUTION CANDIDATES (top {len(top_ood)} of {len(out_dist_df)} total) --\n")
        f.write("at least one feature value falls outside the training data's observed min/max.\n")
        f.write("These may be interesting, but the model has no learned behavior for feature\n")
        f.write("values this extreme -- verify the stellar characterization independently\n")
        f.write("(e.g. SIMBAD, Gaia) before trusting the score at all.\n\n")
        f.writelines(ood_explanations)

    print(f"Explanations saved to {EXPLANATIONS_PATH}")
    print(f"Folded light curve plots (top 10 in-distribution) saved to {plots_dir}")


# =====================================
# MAIN
# =====================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--recompute-ranges", action="store_true",
                         help="Force-recompute training_feature_ranges.json instead of reusing the "
                              "cached version. Only needed if the model has been retrained on "
                              "different/updated training data since the file was last generated.")
    parser.add_argument("--multi-sector", action="store_true",
                         help="Pull candidates from the N most recent sectors (see --n-sectors) "
                              "instead of just the single newest one, prioritizing multi-sector "
                              "coverage first. Uses separate '_widesector'-tagged files so it doesn't "
                              "clobber a single-sector run's saved data.")
    parser.add_argument("--n-sectors", type=int, default=3,
                         help="Number of most recent sectors to draw from when --multi-sector is set.")
    args = parser.parse_args()

    print("#" * 70)
    print("# 06_download_unknown.py -- UNEXAMINED CANDIDATE DISCOVERY")
    print("#")
    print("# 'Unknown' = not yet examined/flagged by anyone (absent from the")
    print("# confirmed-planet catalog AND the TOI table under every disposition).")
    print("# NOT a guarantee of novelty, NOT 'newly created data'. Recency is used")
    print("# only because newer sectors have had less time for anyone to look at")
    print("# them closely. A high score means 'worth qualified human review',")
    print("# never 'confirmed planet'.")
    print("#" * 70)

    with open(FEATURE_METADATA_PATH) as f:
        metadata = json.load(f)
    feature_columns = metadata["feature_columns"]

    excluded_tics = build_exclusion_set()
    if args.multi_sector:
        candidates_df = build_candidate_pool_multi_sector(excluded_tics, args.sample_size, args.n_sectors)
    else:
        candidates_df = build_candidate_pool(excluded_tics, args.sample_size)
    candidates_df = fetch_stellar_params(candidates_df)
    candidates_df.to_csv(CANDIDATE_LIST_PATH, index=False)

    download_candidates(candidates_df)
    preprocess_candidates()
    features_df = extract_features(candidates_df, feature_columns)
    ranked = score_candidates(features_df, candidates_df, feature_columns)

    if len(ranked) == 0:
        print("\nNothing was successfully scored this run -- skipping OOD flagging/ranking "
              "(nothing to flag or rank).")
        in_dist, out_dist = pd.DataFrame(), pd.DataFrame()
    else:
        ranges = load_or_compute_feature_ranges(feature_columns, force_recompute=args.recompute_ranges)
        ranked_flagged = flag_out_of_distribution(ranked, feature_columns, ranges)

        detector = load_or_compute_multivariate_detector(feature_columns, force_recompute=args.recompute_ranges)
        ranked_flagged = flag_multivariate_ood(ranked_flagged, feature_columns, detector)
        ranked_flagged = combine_ood_flags(ranked_flagged)

        in_dist, out_dist = split_and_rerank(ranked_flagged)
    generate_top_n_output(in_dist, out_dist, feature_columns, args.top_n)

    print("\n" + "#" * 70)
    print("# REMINDER: these are UNEXAMINED CANDIDATES ranked for human review.")
    print("# Not confirmed detections. Not discoveries. A high score means")
    print("# 'worth a closer look', not 'this is a planet'.")
    print("#" * 70)


if __name__ == "__main__":
    # Every stage above is idempotent/resumable (checkpointed CSVs, content-
    # verified resume checks), so if the known stdout race (a leftover
    # ThreadPoolExecutor thread racing some library's cached stdout/stderr
    # reference -- confirmed NOT fully caught by the SafeStdout wrapper above,
    # despite that wrapper handling the common case) still escapes and kills
    # the run, the robust fix is to just re-invoke main() -- it picks up
    # exactly where it left off rather than losing any completed work.
    _MAX_ATTEMPTS = 10
    for _attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            main()
            break
        except ValueError as e:
            if "closed file" not in str(e) or _attempt == _MAX_ATTEMPTS:
                raise
            _sys.stderr.write(f"\n[06_download_unknown.py] Hit the known stdout race on attempt "
                               f"{_attempt}/{_MAX_ATTEMPTS} -- resuming via re-invocation "
                               f"(all completed work is checkpointed, nothing is lost).\n")
