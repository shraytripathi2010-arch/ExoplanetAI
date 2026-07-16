"""
job_runner.py -- background orchestration for the web app.

Everything that actually touches TESS data, runs TLS, scores the
classifier, or does external verification is 100% unmodified code imported
directly from src/06_download_unknown.py, src/07_search_unknown.py, and
src/08_characterize_candidates.py. This file only adds:
  1. Running the full pipeline as background subprocesses (the "Update"
     button) with log-tailing for live progress, so the Flask request that
     starts it returns immediately instead of blocking for hours.
  2. A single-candidate re-verification action that calls 08's own check
     functions directly (imported, not subprocessed) for just one TIC ID.
  3. On-demand light curve plot generation using 06's own
     plot_folded_light_curve, unmodified.
"""
import importlib
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

import db
import sync

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC_DIR = os.path.join(PROJECT_ROOT, "code")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Matches the pipeline's own existing progress lines, e.g.
# "  [42/105] TIC_403153811: clean -- Rp=1.48 R_earth, Teq=1285K"
# -- reused as the progress signal rather than modifying 06/07/08 to emit
# anything new, exactly per the "wrap, don't rewrite" instruction.
PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\]\s+(\S+)")


KNOWN_STDOUT_RACE_SIGNATURE = "I/O operation on closed file"


def _retry_on_stdout_race(func, *args, max_attempts=3, **kwargs):
    """The on-demand per-candidate checks (multi-sector, centroid) run as
    IN-PROCESS threads, not subprocesses -- so _run_stage's subprocess-level
    retry doesn't cover them. Confirmed live: running two of these checks
    concurrently in the same process can trigger the exact same known
    stdout race (a library's internal print/progress-bar racing another
    thread's stdout) inside TLS itself. Since this is a plain function call
    in-process, not a subprocess, the fix here is a simple retry of the call
    itself rather than re-invoking a whole external process."""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if KNOWN_STDOUT_RACE_SIGNATURE not in str(e) or attempt == max_attempts:
                raise
            last_exc = e
    raise last_exc


def _run_stage_once(run_id, stage_name, cmd, log_path, attempt_note=""):
    """Runs the subprocess with stdout/stderr redirected DIRECTLY to a real
    log file (like the `> logfile 2>&1` bash redirection every earlier
    manual run of this pipeline used), not piped through a Python
    read-loop. This isn't just a style choice: piping through
    `subprocess.PIPE` and reading it line-by-line in this process turned
    out to make 06's known stdout race (a leftover ThreadPoolExecutor
    worker thread racing the main thread's stdout) hit on EVERY external
    attempt instead of intermittently, because the pipe's line-buffered
    Python-side reader is a different stdout plumbing scenario than a
    direct OS-level file redirect. Progress is now tracked by polling the
    log file's tail on a timer instead of reading from the pipe."""
    db.update_run_progress(run_id, current_stage=stage_name, progress_text=f"starting...{attempt_note}")
    with open(log_path, "a") as logf:
        proc = subprocess.Popen(cmd, cwd=SRC_DIR, stdout=logf, stderr=subprocess.STDOUT)
        last_size = 0
        while proc.poll() is None:
            time.sleep(1.0)
            try:
                # BUG FIXED: this polling read used strict UTF-8 (the
                # default), and some external library occasionally writes a
                # stray non-UTF-8 byte to stdout (seen live: a Windows-1252
                # en-dash/smart-quote from a catalog query response) --
                # crashing THIS polling code with UnicodeDecodeError, which
                # then got caught by _update_job_body's broad except and
                # misreported as if the pipeline stage itself had failed,
                # even though the actual subprocess was running fine.
                # errors="replace" makes log-tailing robust to that without
                # ever affecting the real data on disk (this only reads the
                # log file back for progress display).
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(last_size)
                    new_text = f.read()
                    last_size = f.tell()
            except OSError:
                continue
            for line in new_text.splitlines():
                m = PROGRESS_RE.search(line)
                if m:
                    db.update_run_progress(
                        run_id, progress_text=f"{stage_name}: [{m.group(1)}/{m.group(2)}] {m.group(3)}{attempt_note}"
                    )
        proc.wait()

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(max(0, last_size - 4000))
        tail = f.read()
    return proc.returncode, tail


def _run_stage(run_id, stage_name, cmd, log_path, max_attempts=1):
    """Runs one pipeline stage as a subprocess. 06_download_unknown.py uses
    ThreadPoolExecutor/ProcessPoolExecutor internally and is subject to a
    known, already-diagnosed stdout race (a leftover worker thread racing
    the main thread's stdout, causing a spurious "I/O operation on closed
    file" ValueError). 06 already retries this internally up to 10 times,
    but once the underlying file descriptor is truly closed, only a FRESH
    process invocation clears it -- in-process retries alone are not
    sufficient, which is exactly why every earlier manual run of this
    pipeline in this project used an external bash retry loop, not just
    06's own internal one. This reproduces that same external-retry
    wrapping in code so the web app's Update button is just as robust as
    a manually-run pipeline, instead of silently failing the whole Update
    the first time this known, harmless race is hit.
    All of 06's stages checkpoint to disk independently, so re-invoking the
    same command is always safe -- it resumes rather than redoing work."""
    for attempt in range(1, max_attempts + 1):
        note = f" (attempt {attempt}/{max_attempts})" if max_attempts > 1 else ""
        rc, tail = _run_stage_once(run_id, stage_name, cmd, log_path, attempt_note=note)
        if rc == 0:
            return rc, tail, attempt
        if KNOWN_STDOUT_RACE_SIGNATURE not in tail or attempt == max_attempts:
            return rc, tail, attempt
        # Known, harmless race -- all completed work is checkpointed to disk,
        # so simply re-invoking the process picks up where it left off.
    return rc, tail, attempt


# Known failure signatures, checked in order -- gives the website a specific,
# actionable message instead of a bare "exited 1", and tells the user
# whether the failure is a known/likely-transient category or something
# that needs real attention. Add new signatures here as they're identified,
# rather than leaving future failures equally generic.
_FAILURE_CATEGORIES = [
    (KNOWN_STDOUT_RACE_SIGNATURE,
     "Known stdout race in a third-party library (lightkurve's .download() is not "
     "thread-safe against concurrent calls) -- this is normally prevented by a lock "
     "added to serialize downloads; seeing this means either an unusually large batch "
     "hit it anyway or the lock didn't cover every download path. All completed work "
     "up to this point is saved to disk regardless."),
    ("HTTPError", "A network/HTTP error from an external service (MAST, Gaia, ExoFOP, etc.) "
                  "-- often transient rate-limiting or a temporary outage."),
    ("ConnectionError", "A network connection failure -- likely transient, check your internet connection "
                        "if it keeps recurring."),
    ("TimeoutError", "A network or per-star operation timed out -- the configured timeouts "
                     "(30s network, 45s per-star download) are working as designed; this star "
                     "was skipped rather than hanging the whole run."),
    ("MemoryError", "Ran out of memory -- consider a smaller sample size."),
]


def _categorize_failure(tail):
    for signature, explanation in _FAILURE_CATEGORIES:
        if signature in tail:
            return explanation
    # Show the actual last real line of output (not blank) so an unrecognized
    # failure is still informative rather than a bare exit code.
    lines = [l for l in tail.strip().splitlines() if l.strip()]
    last_line = lines[-1] if lines else "(no output captured)"
    return f"Unrecognized failure -- last output line: {last_line}"


def _update_job_body(run_id, sample_size):
    try:
        rc, tail, attempts = _run_stage(run_id, "06_download_unknown",
                         [sys.executable, "06_download_unknown.py", "--sample-size", str(sample_size)],
                         os.path.join(LOG_DIR, f"run_{run_id}_06.log"), max_attempts=5)
        if rc != 0:
            db.finish_run(run_id, "failed",
                          error_message=f"Download stage failed after {attempts} attempt(s): {_categorize_failure(tail)}")
            return

        rc, tail, attempts = _run_stage(run_id, "07_search_unknown", [sys.executable, "07_search_unknown.py"],
                         os.path.join(LOG_DIR, f"run_{run_id}_07.log"))
        if rc != 0:
            db.finish_run(run_id, "failed",
                          error_message=f"Stellar-verification stage failed: {_categorize_failure(tail)}")
            return

        rc, tail, attempts = _run_stage(run_id, "08_characterize_candidates",
                         [sys.executable, "08_characterize_candidates.py"],
                         os.path.join(LOG_DIR, f"run_{run_id}_08.log"))
        if rc != 0:
            db.finish_run(run_id, "failed",
                          error_message=f"Characterization stage failed: {_categorize_failure(tail)}")
            return

        db.update_run_progress(run_id, current_stage="sync", progress_text="syncing results to database...")
        result = sync.sync_from_csvs(run_id=run_id)

        # Phase 1 of the multi-data-source effort: pixel-level centroid
        # checks now run automatically for every combined-filter-passing
        # candidate as part of the Update itself, not just on-demand one at
        # a time. Deliberately NOT allowed to fail the whole Update -- the
        # core pipeline result (new candidates synced) is already valid at
        # this point regardless of what happens here, and a bug in this
        # batch shouldn't erase that. Per-candidate failures are already
        # handled inside _centroid_body itself and don't raise.
        try:
            _run_centroid_batch(run_id)
        except Exception as e:
            print(f"Centroid batch failed (Update result is unaffected): {e}")

        db.finish_run(run_id, "completed", n_new_candidates=result["new"])
    except Exception as e:
        db.finish_run(run_id, "failed", error_message=f"Unexpected error: {e}")


def _run_centroid_batch(run_id):
    """Runs the same _centroid_body used by the on-demand button, but
    automatically and sequentially for every candidate that passes the
    combined filter and has never had a centroid attempt (see
    db.get_candidates_needing_centroid). Real measured cost per candidate:
    ~5.5s and ~47MB (confirmed live before building this), so even a few
    dozen new passers per Update adds only a minute or two -- negligible
    next to the multi-hour download/TLS stages already accepted."""
    candidates = db.get_candidates_needing_centroid()
    n = len(candidates)
    if n == 0:
        return
    for i, candidate in enumerate(candidates, start=1):
        char = candidate["characterization"]
        tic_id = candidate["tic_id"]
        db.update_run_progress(
            run_id, current_stage="centroid_check",
            progress_text=f"Pixel-level centroid check: {i}/{n} combined-filter-passing candidates "
                          f"({candidate['host']})",
        )
        db.start_centroid_check(tic_id)
        _centroid_body(
            tic_id, candidate["host"], char.get("ra"), char.get("dec"),
            char.get("period_days"), char.get("epoch_bjd"), char.get("transit_duration_hours"),
            char.get("transit_depth_ppm"),
        )


def start_update_job(sample_size):
    run_id = db.create_run(sample_size)
    thread = threading.Thread(target=_update_job_body, args=(run_id, sample_size), daemon=True)
    thread.start()
    return run_id


# ---- in-process scheduler ----
#
# Deliberately reuses start_update_job -- a scheduled run is not a different
# code path from a manually-clicked one, just a different trigger. This only
# fires while THIS Flask process is alive: closing the terminal running
# app.py, or the Mac sleeping, means no scheduled run happens. For a
# schedule that survives that, see the launchd option documented in
# web/README_SCHEDULING.md -- an OS-level mechanism that can wake the
# machine and ensure app.py itself is running, hitting the same
# /jobs/update endpoint this scheduler calls internally.
SCHEDULER_POLL_SECONDS = 60


# Continuous-retraining tick (Item 2, Part B) runs on its own, much coarser
# interval than the 60s Update-due check above -- it queries the live
# archive and can trigger real downloads/TLS runs, so checking it every 60s
# would be wasteful and noisy. Once/day is enough for something whose whole
# point is "don't go stale over weeks/months," not "react within a minute."
RETRAIN_TICK_INTERVAL_HOURS = 24


def _retrain_tick_due():
    last = db.get_last_retrain_tick_at()
    if not last:
        return True
    last_dt = datetime.fromisoformat(last.replace(" UTC", "+00:00").replace(" ", "T"))
    return datetime.now(timezone.utc) - last_dt >= timedelta(hours=RETRAIN_TICK_INTERVAL_HOURS)


def _scheduler_loop():
    while True:
        time.sleep(SCHEDULER_POLL_SECONDS)
        try:
            cfg = db.get_scheduler_config()
            if cfg["enabled"] and cfg["next_run_at"]:
                next_run = datetime.fromisoformat(cfg["next_run_at"].replace(" UTC", "+00:00").replace(" ", "T"))
                if datetime.now(timezone.utc) >= next_run and not db.get_running_run():
                    start_update_job(cfg["sample_size"])
                    new_next = datetime.now(timezone.utc) + timedelta(days=cfg["interval_days"])
                    db.set_scheduler_next_run(
                        new_next.strftime("%Y-%m-%d %H:%M:%S UTC"), db.now_iso()
                    )
        except Exception:
            pass  # scheduler thread must never die from one bad tick

        try:
            if _retrain_tick_due():
                import retrain_pipeline
                retrain_pipeline.scheduler_tick()
                db.set_last_retrain_tick_at(db.now_iso())
        except Exception as e:
            print(f"Continuous-retraining tick failed (does not affect Update jobs): {e}")


def start_scheduler_thread():
    """Starts the always-on background thread that checks, once a minute,
    whether a scheduled Update is due. Safe to call once at app startup --
    if scheduling is disabled it just polls and does nothing."""
    thread = threading.Thread(target=_scheduler_loop, daemon=True)
    thread.start()


# ---- single-candidate re-verify (Phase 3) ----

def _load_08():
    return importlib.import_module("08_characterize_candidates")


REVERIFY_CALL_TIMEOUT = 30


def _call_with_timeout(fn, *args, timeout=REVERIFY_CALL_TIMEOUT, default=None):
    """Bounds one external check call to `timeout` seconds.

    ROOT CAUSE (found via live-testing the reverify button, which hung at
    "running" for 90+ seconds with no error): fetch_fresh_exclusion_data()
    called plain pd.read_csv(url), which has no timeout of its own -- fixed
    at the source too, but astroquery's Vizier/Gaia calls used by
    check_vsx/check_blending don't reliably expose or honor a timeout knob
    either. Rather than trust each library's internal timeout handling,
    every external call in the reverify path is bounded uniformly here, the
    same "don't trust an unbounded network call" principle already applied
    to per-star downloads via PER_STAR_DOWNLOAD_TIMEOUT."""
    import concurrent.futures
    # NOT a `with` block: ThreadPoolExecutor.__exit__ calls shutdown(wait=True),
    # which blocks until the submitted call actually finishes -- defeating the
    # timeout below the instant future.result() raises. Same class of bug as
    # the documented executor.shutdown(wait=False) choice in 06_download_unknown.py.
    # The orphaned thread is abandoned (daemon-like) if it never returns.
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = ex.submit(fn, *args)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return default
    finally:
        ex.shutdown(wait=False)


def _reverify_body(candidate):
    """Re-runs just the external checks (ExoFOP, arXiv, VSX, blend) for one
    TIC ID, reusing 08's own check functions directly rather than running
    the whole batch script. Logs the result and bumps last_verified_date;
    updates current_status if the candidate has newly been flagged.

    BUG FIXED: this used to run synchronously inline in the Flask request
    handler -- unlike multi-sector/centroid, which were already proper
    background jobs with status polling. A slow live archive/ExoFOP/VSX
    round-trip (or a network hiccup) would block the browser tab with zero
    progress feedback, indistinguishable from the page being frozen. Now
    runs as a background thread with the same status-polling pattern as
    the other two on-demand checks."""
    tic_id = candidate["tic_id"]
    try:
        m08 = _load_08()
        char = candidate["characterization"]
        ra, dec = char.get("ra"), char.get("dec")

        exclusion_result = _call_with_timeout(m08.fetch_fresh_exclusion_data,
                                               default=(set(), set(), set()))
        confirmed_tics, toi_tics, exofop_tics = exclusion_result
        newly_flagged = tic_id in confirmed_tics or tic_id in toi_tics or tic_id in exofop_tics
        exofop_has_page, exofop_note, exofop_code = _call_with_timeout(
            m08.check_exofop_target_page, tic_id,
            default=(None, "ExoFOP check timed out", "ERROR"))
        arxiv_status, arxiv_links, arxiv_code = _call_with_timeout(
            m08.check_arxiv, tic_id,
            default=("arXiv check timed out", "", "ERROR"))
        vsx_status, vsx_detail, vsx_code = _call_with_timeout(
            m08.check_vsx, ra, dec, default=("VSX check timed out", "", "ERROR")
        ) if ra and dec else ("No RA/Dec on file", None, "SKIPPED")
        blend_status, n_tier1, n_tier2, blend_code = _call_with_timeout(
            m08.check_blending, ra, dec, default=("Blend check timed out", None, None, "ERROR")
        ) if ra and dec else ("No RA/Dec on file", None, None, "SKIPPED")

        # Same graceful-degradation pattern as a full 08 run: only attempt ADS
        # when a key is actually configured, otherwise leave arXiv's result
        # (already in `char` from the last full run/characterization) as-is.
        ads_api_key = os.environ.get("ADS_API_KEY")
        if ads_api_key:
            ads_status, ads_links, ads_code = _call_with_timeout(
                m08.check_ads, tic_id, ads_api_key,
                default=("ADS check timed out", "", "ERROR"))
            char["ads_status"], char["ads_links"], char["ads_code"] = ads_status, ads_links, ads_code

        changes = []
        if newly_flagged != bool(candidate.get("current_status") != "unknown_candidate"):
            changes.append(f"exclusion-set status changed (newly_flagged={newly_flagged})")
        if exofop_code == "HIT":
            changes.append(f"ExoFOP: {exofop_note}")
        if vsx_code == "HIT":
            changes.append(f"VSX match: {vsx_detail}")
        if blend_status != candidate.get("blending_status"):
            changes.append(f"blend status changed: {blend_status}")
        if ads_api_key and char.get("ads_code") == "HIT":
            changes.append(f"ADS: {char.get('ads_status')}")

        summary = "; ".join(changes) if changes else "no change since last check"

        status = "flagged_in_archive_or_toi" if newly_flagged else "unknown_candidate"
        db.upsert_candidate(
            tic_id=tic_id, host=candidate["host"], status=status,
            predicted_probability=candidate.get("predicted_probability"),
            confidence_tier=candidate.get("confidence_tier"),
            combined_filter_pass=candidate.get("combined_filter_pass"),
            combined_filter_tier=candidate.get("combined_filter_tier"),
            needs_manual_review=candidate.get("needs_manual_review"),
            radius_plausible=candidate.get("radius_plausible"),
            blending_status=blend_status,
            vsx_code=vsx_code,
            characterization_dict=char,
            note=summary,
        )
        db.log_verification_event(tic_id, "single_reverify", summary)
        db.save_reverify_result(tic_id, summary)
    except Exception as e:
        db.fail_reverify(tic_id, str(e))


def start_reverify_check(candidate):
    tic_id = candidate["tic_id"]
    db.start_reverify_status(tic_id)
    thread = threading.Thread(target=_reverify_body, args=(candidate,), daemon=True)
    thread.start()


# ---- on-demand light curve plots (Phase 4) ----

def _load_06():
    return importlib.import_module("06_download_unknown")


def generate_plot_for_candidate(candidate, out_path):
    m06 = _load_06()
    char = candidate["characterization"]
    host = candidate["host"]
    period = char.get("period")
    t0 = char.get("T0")
    if period is None or t0 is None:
        return False, "missing period/T0 in stored characterization"

    csv_path = os.path.join(PROJECT_ROOT, "data", "processed_unknown", f"{host}.csv")
    if not os.path.exists(csv_path):
        return False, f"processed light curve not found at {csv_path}"

    try:
        m06.plot_folded_light_curve(host, csv_path, period, t0, out_path)
        return True, None
    except Exception as e:
        return False, str(e)


# ---- multi-sector strengthening (Item 1) ----
#
# Reuses 06_download_unknown.py's own functions unmodified: try_search (MAST
# sector search), download_one_star (the already-validated multi-sector
# fetch+concatenate-by-time pattern -- NOT lightkurve's .stitch(), which
# this project doesn't use anywhere; this IS the actual validated approach),
# clean_light_curve (per-file detrend/normalize), and compute_all_features
# (the TLS run itself). The result is stored as SUPPLEMENTARY evidence only
# -- it is never fed back through the classifier or the OOD checks, since
# widening the baseline shifts period/transit_count outside what the
# CURRENT classifier was trained on (a real, already-learned lesson in this
# project) and would cause a false OOD flag if treated as a new model input.

def _multi_sector_body(tic_id, host, st_rad, st_mass):
    import json
    m06 = _load_06()
    try:
        search, method = _retry_on_stdout_race(m06.try_search, tic_id)
        if len(search) == 0:
            db.fail_multi_sector_check(tic_id, "No TESS light curve data found via MAST")
            return

        all_sectors = sorted(set(int(s) for s in search.table["sequence_number"]))
        ms_filename = f"{host}_multisector"

        dl_result = _retry_on_stdout_race(
            m06.download_one_star, tic_id, ms_filename, target_sectors=set(all_sectors)
        )
        if dl_result.get("status") != "Success":
            db.fail_multi_sector_check(tic_id, f"Download failed: {dl_result.get('status')}")
            return

        raw_path = os.path.join(m06.RAW_FOLDER, ms_filename + ".csv")
        cleaned_df, clean_status = m06.clean_light_curve(raw_path)
        if cleaned_df is None:
            db.fail_multi_sector_check(tic_id, f"Preprocessing failed: {clean_status}")
            return

        os.makedirs(m06.PROCESSED_FOLDER, exist_ok=True)
        processed_path = os.path.join(m06.PROCESSED_FOLDER, ms_filename + ".csv")
        cleaned_df.to_csv(processed_path, index=False)

        with open(m06.FEATURE_METADATA_PATH) as f:
            feature_columns = json.load(f)["feature_columns"]

        feats, tls_status = _retry_on_stdout_race(
            m06.compute_all_features, processed_path, ms_filename, st_rad, st_mass, feature_columns
        )
        if feats is None:
            db.fail_multi_sector_check(tic_id, f"TLS re-run failed: {tls_status}")
            return

        # BUG FIXED (found during a later pipeline-wide audit): TLS's `depth`
        # field is the relative FLUX LEVEL at transit bottom, not the size of
        # the dip -- confirmed against training_feature_ranges.json (depth
        # ranges from 0.525 to 1.0, mean 0.994) and 08_characterize_candidates.py's
        # own formula (`depth_ppm = (1.0 - depth) * 1e6`, "Rp/R_star =
        # sqrt(1 - depth)"). This function originally used `feats["depth"] *
        # 1e6` directly, which is why a normal, real transit (bottom flux
        # ~0.9995, i.e. a genuine ~550 ppm dip) was being reported and stored
        # as an absurd "999,452 ppm" (99.9%) depth. That earlier result was
        # MISDIAGNOSED as a TLS/binning artifact from the larger multi-sector
        # dataset -- it was actually just this unit-conversion bug; TLS's
        # actual output was fine. Fixed to use the same (1 - depth) *1e6
        # convention as the rest of the project.
        fractional_dip = 1.0 - feats["depth"]
        depth_ppm = fractional_dip * 1e6

        # A genuine defensive check is still worth keeping now that the
        # formula itself is correct: ground it in the real data anyway, in
        # case a future edge case (e.g. a truly degenerate TLS fit) produces
        # a dip deeper than the light curve itself ever shows.
        observed_min_flux = float(cleaned_df["flux"].min())
        max_plausible_dip = (1.0 - observed_min_flux) * 1.5
        if fractional_dip > max_plausible_dip and fractional_dip > 0.05:
            plausible = False
            plausibility_note = (
                f"Reported depth ({depth_ppm:.0f} ppm) exceeds what the light curve itself ever "
                f"shows (deepest real dip: {(1 - observed_min_flux) * 1e6:.0f} ppm) -- likely a "
                f"degenerate TLS fit, not real evidence. Treat this specific re-run's numbers "
                f"with skepticism."
            )
        else:
            plausible = True
            plausibility_note = None

        db.save_multi_sector_result(
            tic_id,
            sectors_used=",".join(str(s) for s in all_sectors),
            n_sectors=len(all_sectors),
            period_days=feats["period"],
            t0_bjd=feats["T0"],
            duration_hours=feats["duration"] * 24.0,
            depth_ppm=depth_ppm,
            sde=feats["SDE"],
            transit_count=feats["distinct_transit_count"],
            plausible=plausible,
            plausibility_note=plausibility_note,
        )
    except Exception as e:
        db.fail_multi_sector_check(tic_id, str(e))


def start_multi_sector_check(candidate):
    tic_id = candidate["tic_id"]
    char = candidate["characterization"]
    db.start_multi_sector_check(tic_id)
    thread = threading.Thread(
        target=_multi_sector_body,
        args=(tic_id, candidate["host"], char.get("st_rad"), char.get("st_mass")),
        daemon=True,
    )
    thread.start()


# ---- difference-image centroid vetting (Item 5) ----
#
# Standard technique used in professional TESS vetting reports: split a
# target pixel file's cadences into in-transit vs out-of-transit, take the
# median flux image of each, subtract to get a difference image (positive
# where flux dropped during transit -- i.e. where the signal actually comes
# from), and compute that difference image's flux-weighted centroid. If it
# lands away from the target's own catalog position (converted to pixel
# coordinates via the TPF's own WCS), the transit is probably coming from a
# nearby contaminant, not the target star. This tests something genuinely
# different from the existing Gaia-proximity blend check (catalog distance
# vs. where the light is actually coming from), so both stay visible as
# separate, independent evidence rather than one replacing the other.
CENTROID_CONSISTENT_THRESHOLD_PIX = 0.3
CENTROID_BORDERLINE_THRESHOLD_PIX = 0.5


def _centroid_body(tic_id, host, ra, dec, period_days, t0_bjd, duration_hours, expected_depth_ppm):
    import numpy as np
    from scipy.ndimage import center_of_mass

    tpf = None
    try:
        if ra is None or dec is None:
            db.fail_centroid_check(tic_id, "No RA/Dec on file for this candidate")
            return
        if period_days is None or t0_bjd is None or duration_hours is None:
            db.fail_centroid_check(tic_id, "Missing period/epoch/duration -- run characterization first")
            return

        import lightkurve as lk
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        search = lk.search_targetpixelfile(f"TIC {tic_id}", mission="TESS", author="SPOC")
        if len(search) == 0:
            search = lk.search_targetpixelfile(f"TIC {tic_id}", mission="TESS")
        if len(search) == 0:
            db.fail_centroid_check(tic_id, "No target pixel file available via MAST for this star")
            return

        # BUG FIXED (found while scoping running this at scale): a bare
        # search[0] silently picks whichever product MAST lists first, which
        # is the 20-second fast-cadence TPF -- not the standard 2-minute one
        # -- whenever a target has both (common for extended-mission
        # sectors). Measured live: 250-280MB per sector vs. 43-47MB for the
        # equivalent 2-minute product, a ~6x cost difference that only shows
        # up once this runs across many candidates instead of one at a time.
        # The difference-image technique doesn't need 20s cadence at all, so
        # explicitly prefer the standard cadence product.
        try:
            standard_mask = search.table["exptime"] == 120.0
            standard = search[standard_mask]
        except Exception:
            standard = search[:0]
        chosen = standard[0] if len(standard) > 0 else search[0]

        # Root cause (found during a later reliability audit): lightkurve's
        # own `.download()` is decorated with a non-thread-safe global
        # sys.stdout swap -- if this centroid check's download happens to
        # overlap with ANY other concurrent lightkurve download in this
        # process (e.g. a multi-sector check running at the same time),
        # sys.stdout can get permanently corrupted, crashing an unrelated
        # print() later. m06._DOWNLOAD_LOCK is the SAME lock instance
        # 06_download_unknown.py's own worker threads use (shared because
        # `m06` is a cached, single module object within this process), so
        # this serializes against every other download path in the app, not
        # just against itself. _retry_on_stdout_race stays on as a backstop.
        m06 = _load_06()
        with m06._DOWNLOAD_LOCK:
            tpf = _retry_on_stdout_race(chosen.download)
        if tpf is None:
            db.fail_centroid_check(tic_id, "Target pixel file download failed")
            return
        sector_used = int(tpf.sector) if hasattr(tpf, "sector") else None

        time_arr = tpf.time.value
        flux_cube = tpf.flux.value.astype(float)  # (n_cadences, ny, nx)

        duration_days = duration_hours / 24.0
        phase = np.mod((time_arr - t0_bjd) / period_days, 1.0)
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        half_dur_phase = (duration_days / period_days) / 2.0

        in_transit = np.abs(phase) < (half_dur_phase * 1.5)
        out_of_transit = np.abs(phase) > 0.25

        if in_transit.sum() < 5:
            db.fail_centroid_check(tic_id, f"Only {int(in_transit.sum())} in-transit cadences in this "
                                            f"sector -- not enough for a reliable difference image")
            return
        if out_of_transit.sum() < 5:
            db.fail_centroid_check(tic_id, "Not enough out-of-transit cadences in this sector")
            return

        with np.errstate(invalid="ignore"):
            in_transit_img = np.nanmedian(flux_cube[in_transit], axis=0)
            out_of_transit_img = np.nanmedian(flux_cube[out_of_transit], axis=0)
        diff_img = out_of_transit_img - in_transit_img
        diff_img_clipped = np.clip(diff_img, 0, None)
        diff_img_clipped = np.nan_to_num(diff_img_clipped, nan=0.0)

        if diff_img_clipped.sum() <= 0:
            db.fail_centroid_check(tic_id, "Difference image has no positive flux -- transit signal "
                                            "too weak in this sector's pixel data to localize")
            return

        row_com, col_com = center_of_mass(diff_img_clipped)

        target_coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
        col_expected, row_expected = tpf.wcs.world_to_pixel(target_coord)

        # Plausibility guard found necessary via this same audit: this
        # function picks whichever sector `search[0]` happens to return
        # first from MAST -- with no guarantee it's the SAME sector the
        # stored period/T0 were actually derived from. If it isn't, and the
        # ephemeris doesn't propagate cleanly across the gap (accumulated
        # period uncertainty, real TTVs, or just enough elapsed cycles), the
        # in-transit/out-of-transit phase mask silently folds on the WRONG
        # part of this sector's data -- producing a confident-looking but
        # meaningless shift_pixels number with no error or warning. Ground
        # the check in the data actually available: measure the dip
        # directly at the target's own expected pixel and compare it to
        # the already-known transit depth from the single-sector
        # characterization. If they don't roughly agree, the phase-fold
        # likely isn't landing on a real transit in THIS sector's data, and
        # the resulting shift is not trustworthy.
        row_int, col_int = int(round(float(row_expected))), int(round(float(col_expected)))
        ny, nx = diff_img.shape
        r0, r1 = max(0, row_int - 1), min(ny, row_int + 2)
        c0, c1 = max(0, col_int - 1), min(nx, col_int + 2)
        if r0 >= r1 or c0 >= c1:
            db.fail_centroid_check(tic_id, "Target's expected pixel position falls outside this TPF's cutout")
            return

        target_out_flux = float(np.nansum(out_of_transit_img[r0:r1, c0:c1]))
        target_in_flux = float(np.nansum(in_transit_img[r0:r1, c0:c1]))
        if target_out_flux <= 0:
            db.fail_centroid_check(tic_id, "No usable flux at the target's own pixel position in this sector")
            return

        observed_depth_ppm = (1.0 - target_in_flux / target_out_flux) * 1e6

        if expected_depth_ppm is not None and expected_depth_ppm > 0:
            ratio = observed_depth_ppm / expected_depth_ppm
            if not (0.1 <= ratio <= 10) or observed_depth_ppm <= 0:
                db.fail_centroid_check(
                    tic_id,
                    f"Depth measured at the target's own pixel in this sector ({observed_depth_ppm:.0f} ppm) "
                    f"doesn't match the known transit depth ({expected_depth_ppm:.0f} ppm) -- this sector's "
                    f"data likely doesn't correspond to the same ephemeris (e.g. a different sector than "
                    f"where period/epoch were derived, or period aliasing). The pixel shift below would not "
                    f"be trustworthy, so this check was not completed with a verdict."
                )
                return

        shift_pixels = float(np.hypot(row_com - row_expected, col_com - col_expected))

        if shift_pixels < CENTROID_CONSISTENT_THRESHOLD_PIX:
            verdict = f"Photocenter consistent with target ({shift_pixels:.2f} px from catalog position)"
        elif shift_pixels < CENTROID_BORDERLINE_THRESHOLD_PIX:
            verdict = f"Borderline: photocenter {shift_pixels:.2f} px from target -- weak evidence of a nearby contaminant"
        else:
            verdict = f"Photocenter shifted {shift_pixels:.2f} px from target -- signal likely from a nearby contaminant, not the target star"

        db.save_centroid_result(tic_id, sector_used, shift_pixels, verdict)
        _fold_centroid_into_evidence_writeup(tic_id, shift_pixels, verdict)
    except Exception as e:
        db.fail_centroid_check(tic_id, str(e))
    finally:
        if tpf is not None:
            try:
                os.remove(tpf.path)
            except OSError:
                pass


def _fold_centroid_into_evidence_writeup(tic_id, shift_pixels, verdict):
    """Makes the centroid result first-class evidence in the same
    supporting/doubting-evidence writeup 08_characterize_candidates.py
    already produces, instead of leaving it visible only in its own
    separate section. Purely additive -- appends one sentence to existing
    text, never touches predicted_probability or confidence_tier (same
    "additive evidence only, never re-scored" rule already applied to the
    multi-sector and Gaia-blend checks). Guards against duplicate sentences
    if this ever runs twice for the same candidate (manual re-run after an
    automatic run, or vice versa)."""
    candidate = db.get_candidate(tic_id)
    if candidate is None:
        return
    char = candidate["characterization"]
    marker = "Pixel-level centroid check:"
    supporting = char.get("supporting_evidence") or ""
    doubting = char.get("doubting_evidence") or ""
    if marker in supporting or marker in doubting:
        return  # already folded in once; don't accumulate duplicate sentences

    sentence = f"{marker} {verdict}"
    if shift_pixels < CENTROID_CONSISTENT_THRESHOLD_PIX:
        char["supporting_evidence"] = (supporting + "; " if supporting else "") + sentence
    else:
        # Both the borderline and clear-shift cases are doubt, not support --
        # a confirmed nearby contaminant is genuine, strong evidence against
        # the target star being the transit source, not a minor caveat.
        char["doubting_evidence"] = (doubting + "; " if doubting else "") + sentence

    db.upsert_candidate(
        tic_id=tic_id, host=candidate["host"], status=candidate["current_status"],
        predicted_probability=candidate.get("predicted_probability"),
        confidence_tier=candidate.get("confidence_tier"),
        combined_filter_pass=candidate.get("combined_filter_pass"),
        combined_filter_tier=candidate.get("combined_filter_tier"),
        needs_manual_review=candidate.get("needs_manual_review"),
        radius_plausible=candidate.get("radius_plausible"),
        blending_status=char.get("blending_status"),
        vsx_code=char.get("vsx_code"),
        characterization_dict=char,
        note="pixel-level centroid check folded into evidence writeup",
    )


def start_centroid_check(candidate):
    tic_id = candidate["tic_id"]
    char = candidate["characterization"]
    db.start_centroid_check(tic_id)
    thread = threading.Thread(
        target=_centroid_body,
        args=(tic_id, candidate["host"], char.get("ra"), char.get("dec"),
              char.get("period_days"), char.get("epoch_bjd"), char.get("transit_duration_hours"),
              char.get("transit_depth_ppm")),
        daemon=True,
    )
    thread.start()
