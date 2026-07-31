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
import scheduler_log
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

# BUG FIXED (found by watching a real Update run against the live archive):
# PROGRESS_RE alone matched only the pipeline's own "[42/105] TIC_x" lines.
# But 06's two longest stages -- the TIC catalog query and the bulk light
# curve download -- report progress through a tqdm bar instead:
#
#   Downloading:  29%|##9       | 167/569 [03:05<12:14,  1.83s/it]
#
# which that regex never matches. Confirmed live: the dashboard sat on
# "starting..." for over four minutes while the download stage was in fact
# 167 stars in and progressing normally. That is precisely the state a user
# cannot distinguish from a hung job -- the Update button's own progress
# display was the worst offender of everything audited here.
#
# tqdm also gives us its own measured ETA, which is a far more honest
# "how long will this take" than any estimate hard-coded here, since it is
# computed from this run's actual observed rate.
TQDM_PROGRESS_RE = re.compile(
    r"([A-Za-z][\w \-]*?):\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^<\]]+)<([^,\]]+)")


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
            # Only the LAST progress signal in this chunk matters -- a tqdm
            # bar emits dozens of updates per poll interval and writing each
            # one is dozens of pointless DB writes to display a value that is
            # already stale by the time it lands.
            latest = None
            for line in new_text.splitlines():
                m = PROGRESS_RE.search(line)
                if m:
                    latest = f"[{m.group(1)}/{m.group(2)}] {m.group(3)}"
                    continue
                t = TQDM_PROGRESS_RE.search(line)
                if t:
                    label, done, total, _elapsed, remaining = t.groups()
                    remaining = remaining.strip()
                    # tqdm prints "?" until it has enough samples to estimate;
                    # showing "about ? remaining" would be worse than silence.
                    eta = f" (about {remaining} remaining)" if remaining != "?" else ""
                    latest = f"{label.strip()} {done}/{total}{eta}"
            if latest:
                # No stage_name prefix: the banner already renders
                # current_stage next to this text, so including it here
                # produced "06_download_unknown -- 06_download_unknown: ...".
                db.update_run_progress(run_id, progress_text=f"{latest}{attempt_note}")
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
        # Only surface the attempt counter once we are actually RETRYING.
        # Showing "(attempt 1/5)" on a normal, healthy first run reads as if
        # something has already gone wrong.
        note = f" (retry {attempt} of {max_attempts})" if attempt > 1 else ""
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

# Named so scheduler_is_alive() can distinguish "the scheduler thread is
# running" from "the process is running" -- see the /health endpoint.
_SCHEDULER_THREAD_NAME = "exoplanetai-scheduler"


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


def _safe_label_count():
    """Progress toward the retrain threshold, in the liveness line itself, so
    the log answers 'is it alive AND is it getting anywhere' in one grep.
    Never raises -- a counter read must not be able to kill the scheduler."""
    try:
        return db.count_processed_watch_labels_since("2000-01-01 00:00:00 UTC")
    except Exception:
        return "?"


def _scheduler_loop():
    log = scheduler_log.get_logger()
    log.info("SCHEDULER  thread started (poll=%ss, retrain tick every %sh)",
             SCHEDULER_POLL_SECONDS, RETRAIN_TICK_INTERVAL_HOURS)
    scheduler_log.write_heartbeat(event="thread_started")
    tick = 0

    while True:
        time.sleep(SCHEDULER_POLL_SECONDS)
        tick += 1
        update_status = "idle"
        retrain_status = "not_due"

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
                    update_status = "started"
                    log.info("UPDATE     scheduled Update job started (sample_size=%s); "
                             "next run %s", cfg["sample_size"], new_next)
                elif db.get_running_run():
                    update_status = "skipped_run_in_progress"
            else:
                update_status = "disabled"
        except Exception:
            # WAS `except Exception: pass`. A failing due-check -- an
            # unparseable next_run_at, a locked DB -- vanished with no record
            # anywhere, so scheduled Updates could stop firing forever and the
            # only symptom was silence. exception() logs the full traceback.
            update_status = "error"
            log.exception("UPDATE     due-check raised; scheduler continues")

        try:
            if _retrain_tick_due():
                import retrain_pipeline
                log.info("RETRAIN    tick due -- querying archives, processing labels")
                retrain_pipeline.scheduler_tick()
                db.set_last_retrain_tick_at(db.now_iso())
                retrain_status = "ran"
                try:
                    n = db.count_processed_watch_labels_since("2000-01-01 00:00:00 UTC")
                    log.info("RETRAIN    tick complete -- %s processed watch labels "
                             "(threshold %s to trigger a retrain attempt)",
                             n, getattr(retrain_pipeline, "RETRAIN_THRESHOLD", "?"))
                except Exception:
                    log.exception("RETRAIN    post-tick counter read failed")
        except Exception:
            retrain_status = "error"
            log.exception("RETRAIN    tick raised (does not affect Update jobs)")

        # Heartbeat EVERY tick, including boring ones. A log that only records
        # interesting events cannot tell "healthy and idle" from "dead", which
        # is the one question this whole mechanism exists to answer.
        scheduler_log.write_heartbeat(
            tick=tick, update=update_status, retrain=retrain_status)
        # Every 5 minutes, not hourly. The heartbeat file records EVERY tick,
        # but the stated requirement is that the LOG ALONE shows whether the
        # scheduler is alive and when it last ticked -- and an hourly line
        # leaves up to 59 minutes where the log cannot answer that. At 60s
        # polling this is 288 lines/day, which rotation absorbs easily.
        if tick % 5 == 0:
            log.info("SCHEDULER  alive -- tick %s, update=%s, retrain=%s, "
                     "processed_labels=%s",
                     tick, update_status, retrain_status,
                     _safe_label_count())


def start_scheduler_thread():
    """Starts the always-on background thread that checks, once a minute,
    whether a scheduled Update is due. Safe to call once at app startup --
    if scheduling is disabled it just polls and does nothing."""
    thread = threading.Thread(target=_scheduler_loop, daemon=True,
                              name=_SCHEDULER_THREAD_NAME)
    thread.start()
    return thread


def scheduler_is_alive():
    """True only if the scheduler THREAD is running -- not merely that the
    process is up. Those differ, and the difference is what made the
    hibernation freeze invisible: the process looked fine throughout."""
    for t in threading.enumerate():
        if t.name == _SCHEDULER_THREAD_NAME and t.is_alive():
            return True
    return False


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


_EXCLUSION_CACHE_TTL_SECONDS = 30 * 60
_exclusion_cache = {"data": None, "fetched_at": 0.0}
_exclusion_cache_lock = threading.Lock()


def cached_exclusion_data(force_refresh=False):
    """fetch_fresh_exclusion_data() re-downloads THREE full catalogs (the
    archive's confirmed-planet table, the archive's TOI table, and ExoFOP's
    entire TOI CSV) on every call. Measured live: 24.5s, and it is by far
    the single largest cost in a per-candidate re-verify -- more than the
    other four external checks put together.

    None of those three tables is candidate-specific: the exact same
    download is repeated verbatim for every star. They also change on a
    daily-ish cadence at best (new TOIs are published in batches, not
    continuously), so a 30-minute in-process cache costs nothing in
    freshness while removing ~24s from every re-verify after the first.
    This is a real speedup, not a cosmetic one -- the data genuinely does
    not need re-fetching, and the checks themselves are unchanged.

    `force_refresh` bypasses the cache for callers that must see live data
    (currently none; kept so a future "force fresh" control has a real
    mechanism rather than needing this function rewritten)."""
    now = time.time()
    with _exclusion_cache_lock:
        cached = _exclusion_cache["data"]
        age = now - _exclusion_cache["fetched_at"]
        if cached is not None and not force_refresh and age < _EXCLUSION_CACHE_TTL_SECONDS:
            return cached, age

    m08 = _load_08()
    data = _call_with_timeout(m08.fetch_fresh_exclusion_data, timeout=60,
                              default=(set(), set(), set()))
    # Never cache an empty/failed result -- otherwise one transient network
    # failure would silently make every candidate look "not flagged" for the
    # next 30 minutes, which is a false negative dressed up as a real answer.
    if all(len(s) == 0 for s in data):
        return data, 0.0
    with _exclusion_cache_lock:
        _exclusion_cache["data"] = data
        _exclusion_cache["fetched_at"] = time.time()
    return data, 0.0


def _spawn(fn, *args):
    """Starts one external check immediately and returns (executor, future)
    so several can be in flight at once. Deliberately one executor per call
    rather than a shared pool: _collect must be able to walk away from a
    hung call without waiting on it (see _call_with_timeout's note on
    shutdown(wait=True)), and a shared pool's worker would stay occupied by
    the hung call and stall unrelated checks behind it."""
    import concurrent.futures
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    return ex, ex.submit(fn, *args)


def _collect(pending, timeout=REVERIFY_CALL_TIMEOUT, default=None):
    """Waits for one _spawn'd call, with the same per-call timeout and
    same-shaped default as _call_with_timeout. An exception inside the call
    also yields the default rather than killing the whole re-verify -- one
    external service being broken should degrade that one line of evidence,
    not the action."""
    ex, future = pending
    try:
        return future.result(timeout=timeout)
    except Exception:
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

        # SPEED: these checks hit five completely independent external
        # services (NASA archive, ExoFOP, arXiv, VSX/Vizier, Gaia/Vizier)
        # and none of them consumes another's output. Running them
        # sequentially made the user wait for the SUM of five network
        # round-trips (measured live: 24.5 + 2.2 + 15.3 + 0.8 + 2.3 = ~45s)
        # when the real floor is the SLOWEST one. Submitting them together
        # makes the wait the max instead of the sum. Each still carries its
        # own independent timeout and its own default value, so one slow or
        # down service degrades exactly as before rather than dragging the
        # others with it -- this changes only scheduling, never what any
        # check does or how its result is interpreted.
        ads_api_key = os.environ.get("ADS_API_KEY")
        pending = {
            "exclusion": _spawn(cached_exclusion_data),
            "exofop": _spawn(m08.check_exofop_target_page, tic_id),
            "arxiv": _spawn(m08.check_arxiv, tic_id),
        }
        if ra and dec:
            pending["vsx"] = _spawn(m08.check_vsx, ra, dec)
            pending["blend"] = _spawn(m08.check_blending, ra, dec)
        if ads_api_key:
            pending["ads"] = _spawn(m08.check_ads, tic_id, ads_api_key)

        # Longer ceiling than the other checks on purpose: this one pulls
        # three whole catalogs (measured ~24.5s cold), so the standard 30s
        # per-check timeout would sit right on top of its normal runtime and
        # turn ordinary slowness into a spurious "timed out".
        (confirmed_tics, toi_tics, exofop_tics), _age = _collect(
            pending["exclusion"], timeout=90, default=((set(), set(), set()), 0.0))
        newly_flagged = tic_id in confirmed_tics or tic_id in toi_tics or tic_id in exofop_tics
        exofop_has_page, exofop_note, exofop_code = _collect(
            pending["exofop"], default=(None, "ExoFOP check timed out", "ERROR"))
        arxiv_status, arxiv_links, arxiv_code = _collect(
            pending["arxiv"], default=("arXiv check timed out", "", "ERROR"))
        vsx_status, vsx_detail, vsx_code = _collect(
            pending["vsx"], default=("VSX check timed out", "", "ERROR")
        ) if "vsx" in pending else ("No RA/Dec on file", None, "SKIPPED")
        blend_status, n_tier1, n_tier2, blend_code = _collect(
            pending["blend"], default=("Blend check timed out", None, None, "ERROR")
        ) if "blend" in pending else ("No RA/Dec on file", None, None, "SKIPPED")

        # Same graceful-degradation pattern as a full 08 run: only attempt ADS
        # when a key is actually configured, otherwise leave arXiv's result
        # (already in `char` from the last full run/characterization) as-is.
        if ads_api_key:
            ads_status, ads_links, ads_code = _collect(
                pending["ads"], default=("ADS check timed out", "", "ERROR"))
            char["ads_status"], char["ads_links"], char["ads_code"] = ads_status, ads_links, ads_code

        # BUG FIXED (caught by running three checks concurrently against
        # real services): every check's timeout/error DEFAULT was written
        # straight into the candidate's stored record. A single slow Gaia
        # round-trip left blending_status literally equal to the string
        # "Blend check timed out" and vsx_code equal to "ERROR", replacing
        # the star's real, previously-measured values -- and the next
        # re-verify then compared against that placeholder and announced a
        # spurious "blend status changed". A transient network hiccup was
        # silently, permanently degrading real data while looking like a
        # successful re-verify.
        #
        # A check that did not complete has produced no information, so it
        # must leave the stored value alone and say plainly that it didn't
        # complete. Only a real answer overwrites a real answer.
        failed_checks = []
        if blend_code in ("ERROR", None):
            blend_status = candidate.get("blending_status")
            failed_checks.append("blend/Gaia")
        if vsx_code == "ERROR":
            vsx_code = candidate.get("vsx_code")
            failed_checks.append("VSX")
        if exofop_code == "ERROR":
            failed_checks.append("ExoFOP")
        if arxiv_code == "ERROR":
            failed_checks.append("arXiv")
        if ads_api_key and char.get("ads_code") == "ERROR":
            # Same rule for ADS: drop the placeholder rather than storing it.
            char.pop("ads_status", None), char.pop("ads_links", None), char.pop("ads_code", None)
            failed_checks.append("ADS")
        exclusion_failed = not (confirmed_tics or toi_tics or exofop_tics)
        if exclusion_failed:
            failed_checks.append("archive/TOI exclusion list")

        changes = []
        # Only trust a status flip when the exclusion lists actually loaded --
        # empty sets from a failed fetch would otherwise read as "nothing
        # flags this star", which is a false all-clear.
        if not exclusion_failed and newly_flagged != bool(
                candidate.get("current_status") != "unknown_candidate"):
            changes.append(f"exclusion-set status changed (newly_flagged={newly_flagged})")
        if exofop_code == "HIT":
            changes.append(f"ExoFOP: {exofop_note}")
        if vsx_code == "HIT":
            changes.append(f"VSX match: {vsx_detail}")
        if blend_code not in ("ERROR", None) and blend_status != candidate.get("blending_status"):
            changes.append(f"blend status changed: {blend_status}")
        if ads_api_key and char.get("ads_code") == "HIT":
            changes.append(f"ADS: {char.get('ads_status')}")

        summary = "; ".join(changes) if changes else "no change since last check"
        if failed_checks:
            summary += (f" [{', '.join(failed_checks)} did not complete this time "
                        f"(timeout or service error) -- those previous values were kept "
                        f"unchanged rather than overwritten]")

        # Same rule applied to the status itself. If the exclusion lists
        # didn't load, `newly_flagged` is False purely because the sets are
        # empty -- writing "unknown_candidate" on that basis would silently
        # UN-flag a star that really is a known TOI, turning a network
        # failure into a false all-clear on the most consequential field
        # on the page.
        if exclusion_failed:
            status = candidate.get("current_status") or "unknown_candidate"
        else:
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


EXOFOP_REFRESH_WATCHDOG_SECONDS = 3 * 60


def _exofop_refresh_body(candidate):
    """Answers one narrow question live: has THIS star picked up an ExoFOP
    entry / TOI designation since we last looked?

    Reuses 08's own already-validated ExoFOP logic unchanged -- both halves
    of the pattern that's already proven correct in the full pipeline:
      * the BULK ExoFOP TOI CSV (via cached_exclusion_data ->
        fetch_fresh_exclusion_data), which answers "is this TIC in the TOI
        list at all", and
      * the PER-TARGET page scrape (check_exofop_target_page), which is the
        only thing that yields an actual TOI-<number>. 08's own docstring
        records why: a target page EXISTING is not a usable signal, only a
        real TOI-NNNN string in it is. No new ExoFOP-querying code here.

    Both halves run concurrently -- they hit different endpoints and
    neither needs the other's answer.

    Like the full re-verify, this bumps the candidate's last_verified_date
    and rewrites current_status through the same db.upsert_candidate path,
    so a newly-flagged star stops being presented as an unknown candidate.
    It deliberately does NOT touch predicted_probability, confidence_tier,
    or any other check's result -- it only knows about ExoFOP."""
    tic_id = candidate["tic_id"]
    try:
        m08 = _load_08()
        pending_bulk = _spawn(cached_exclusion_data)
        pending_page = _spawn(m08.check_exofop_target_page, tic_id)

        (confirmed_tics, toi_tics, exofop_tics), age = _collect(
            pending_bulk, timeout=90, default=((set(), set(), set()), 0.0))
        _has_page, page_note, page_code = _collect(
            pending_page, default=(None, "ExoFOP target-page check timed out", "ERROR"))

        if page_code == "ERROR":
            db.fail_exofop_refresh(tic_id, page_note)
            return

        # None, not False, when the list failed to load -- "we don't know"
        # must not be recorded as "confirmed absent from the TOI list".
        in_bulk = None if not exofop_tics else (tic_id in exofop_tics)
        toi_designation = None
        if page_code == "HIT":
            import re as _re
            m = _re.search(r"TOI-\d+", page_note or "")
            toi_designation = m.group(0) if m else None

        # What the candidate looked like BEFORE this check, so "changed"
        # means something real rather than just "we ran again".
        #
        # BUG FIXED (caught in the first live positive-path test): this read
        # the row back with `prev = db.get_exofop_refresh(...)` and treated
        # any non-None row as a previous result. But start_exofop_refresh()
        # has already INSERTed this candidate's row (status='running', every
        # result column NULL) by the time the body runs, so `prev` is never
        # None -- not even on a first-ever check. The very first check on a
        # star therefore compared "TOI-700" against a NULL and announced
        # "This is a CHANGE since the last ExoFOP check" when there had
        # never been one. `computed_at` is the honest marker of a genuine
        # prior result, since only save_exofop_refresh_result sets it.
        row = db.get_exofop_refresh(tic_id)
        prev = row if (row and row.get("computed_at")) else None
        prev_toi = prev.get("toi_designation") if prev else None
        prev_in_bulk = bool(prev.get("in_bulk_toi_list")) if prev else None
        was_unknown = candidate.get("current_status") == "unknown_candidate"

        if toi_designation:
            summary = f"ExoFOP now lists this target as {toi_designation}."
        elif in_bulk:
            summary = ("This TIC appears in ExoFOP's TOI list, but its target page shows no "
                       "TOI-<number> designation yet.")
        else:
            summary = "Still not listed on ExoFOP as a TOI, and no TOI designation on its target page."
        if age > 0:
            summary += f" (TOI list from a cached copy {int(age // 60)} min old; target page checked live.)"

        changed = prev is not None and (
            toi_designation != prev_toi or in_bulk != prev_in_bulk)
        if changed:
            summary += " This is a CHANGE since the last ExoFOP check on this candidate."

        # Same exclusion-set logic the full re-verify uses, so the two can
        # never disagree about what counts as "flagged" -- including the
        # same refusal to treat an empty (failed-fetch) exclusion list as
        # evidence that nothing flags this star.
        exclusion_failed = not (confirmed_tics or toi_tics or exofop_tics)
        newly_flagged = tic_id in confirmed_tics or tic_id in toi_tics or tic_id in exofop_tics
        if exclusion_failed:
            status = candidate.get("current_status") or "unknown_candidate"
            summary += (" [the bulk archive/TOI list did not load this time, so only the "
                        "target page was used and the stored status was left unchanged]")
        else:
            status = "flagged_in_archive_or_toi" if newly_flagged else "unknown_candidate"
            if was_unknown and newly_flagged:
                changed = True

        char = candidate["characterization"]

        # `last_verified_unknown_utc` is the timestamp the CTOI summary quotes
        # on its "Not previously flagged by ExoFOP/TOI/archive as of:" line.
        # It is a CLAIM ABOUT A CHECK -- "we asked the archives at this moment
        # and this star was not flagged" -- so it may only advance when a
        # check actually ran and actually came back clean. Advancing it
        # because a button was pressed, or when the exclusion lists failed to
        # load, or for a star that IS now flagged, would put a verification
        # into a submission-ready document that never happened.
        if not exclusion_failed and not newly_flagged and page_code != "ERROR":
            char["last_verified_unknown_utc"] = db.now_iso()

        db.upsert_candidate(
            tic_id=tic_id, host=candidate["host"], status=status,
            predicted_probability=candidate.get("predicted_probability"),
            confidence_tier=candidate.get("confidence_tier"),
            combined_filter_pass=candidate.get("combined_filter_pass"),
            combined_filter_tier=candidate.get("combined_filter_tier"),
            needs_manual_review=candidate.get("needs_manual_review"),
            radius_plausible=candidate.get("radius_plausible"),
            blending_status=candidate.get("blending_status"),
            vsx_code=candidate.get("vsx_code"),
            characterization_dict=char,
            note=summary,
        )
        db.log_verification_event(tic_id, "exofop_refresh", summary)
        db.save_exofop_refresh_result(tic_id, in_bulk, toi_designation, summary, changed)
    except Exception as e:
        db.fail_exofop_refresh(tic_id, str(e))


def start_exofop_refresh(candidate):
    tic_id = candidate["tic_id"]
    db.start_exofop_refresh(tic_id)
    body = lambda: _exofop_refresh_body(candidate)
    thread = threading.Thread(
        target=_with_watchdog,
        args=(body, tic_id, db.fail_exofop_refresh, EXOFOP_REFRESH_WATCHDOG_SECONDS,
              "ExoFOP re-check"),
        daemon=True,
    )
    thread.start()


# Hard ceilings for the two on-demand checks that do bulk network I/O.
# Measured normal runtimes: multi-sector ~170s (14s download + ~120s TLS),
# centroid ~5s (a ~45MB TPF download). The ceilings are deliberately several
# times those so ordinary slowness never trips them.
MULTI_SECTOR_WATCHDOG_SECONDS = 20 * 60
CENTROID_WATCHDOG_SECONDS = 10 * 60


def _with_watchdog(body, tic_id, fail_fn, timeout, label):
    """BUG FIXED: _multi_sector_body and _centroid_body had no overall time
    bound at all. Each does a MAST download through lightkurve/astroquery,
    whose own socket timeouts are not reliably exposed -- exactly the
    unbounded-network-call problem already fixed for the re-verify path via
    _call_with_timeout. If such a call wedged, the body never returned, its
    DB row stayed 'running' forever, the page's poll loop spun forever on a
    static spinner, and the route's "already running" guard then rejected
    every future click on that candidate with a 409 -- permanently, until
    the whole app was restarted. From the user's side that is
    indistinguishable from the button being broken, with no error ever
    shown. This wraps each body so an overrun is recorded as an honest,
    retryable failure with a real message instead.

    The worker thread itself cannot be killed (Python has no safe thread
    kill), so it is abandoned as a daemon; the row is unlocked either way,
    and the body's own writes are last-write-wins on an already-failed row."""
    import concurrent.futures
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = ex.submit(body)
    try:
        future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        spent = f"{timeout // 60} minutes" if timeout >= 60 else f"{timeout} seconds"
        fail_fn(tic_id, f"{label} exceeded {spent} and was abandoned -- "
                        f"most likely a stalled download from an external archive. "
                        f"Nothing was saved; click again to retry.")
    except Exception as e:
        fail_fn(tic_id, str(e))
    finally:
        ex.shutdown(wait=False)


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

    def progress(text):
        db.set_check_progress("multi_sector_evidence", tic_id, text)

    try:
        progress("Searching MAST for every sector of this star...")
        search, method = _retry_on_stdout_race(m06.try_search, tic_id)
        if len(search) == 0:
            db.fail_multi_sector_check(tic_id, "No TESS light curve data found via MAST")
            return

        all_sectors = sorted(set(int(s) for s in search.table["sequence_number"]))
        ms_filename = f"{host}_multisector"

        progress(f"Downloading {len(all_sectors)} sector(s) "
                 f"({', '.join(str(s) for s in all_sectors)}) from MAST -- about 15s...")
        dl_result = _retry_on_stdout_race(
            m06.download_one_star, tic_id, ms_filename, target_sectors=set(all_sectors)
        )
        if dl_result.get("status") != "Success":
            db.fail_multi_sector_check(tic_id, f"Download failed: {dl_result.get('status')}")
            return

        raw_path = os.path.join(m06.RAW_FOLDER, ms_filename + ".csv")
        progress("Detrending and cleaning the combined light curve...")
        cleaned_df, clean_status = m06.clean_light_curve(raw_path)
        if cleaned_df is None:
            db.fail_multi_sector_check(tic_id, f"Preprocessing failed: {clean_status}")
            return

        os.makedirs(m06.PROCESSED_FOLDER, exist_ok=True)
        processed_path = os.path.join(m06.PROCESSED_FOLDER, ms_filename + ".csv")
        cleaned_df.to_csv(processed_path, index=False)

        with open(m06.FEATURE_METADATA_PATH) as f:
            feature_columns = json.load(f)["feature_columns"]

        # The long pole, and an honest number rather than a vague "a while":
        # measured 118s on a real 3-sector, 272k-point light curve. Running
        # the search across multiple CPU threads was tested directly and is
        # NOT a speedup here -- it ran >16 minutes before being abandoned,
        # against 132s single-threaded -- so this is a real compute floor,
        # not a missing optimization. The honest response is to tell the
        # user what it's doing and roughly how long, not to fake progress.
        progress(f"Re-running the transit search across {len(all_sectors)} combined sector(s) "
                 f"({len(cleaned_df):,} points) -- this is the slow part, typically ~2 minutes.")
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
    body = lambda: _multi_sector_body(tic_id, candidate["host"], char.get("st_rad"), char.get("st_mass"))
    thread = threading.Thread(
        target=_with_watchdog,
        args=(body, tic_id, db.fail_multi_sector_check, MULTI_SECTOR_WATCHDOG_SECONDS,
              "Multi-sector re-analysis"),
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

# Ceiling on how many sectors to try before giving up on a star. Each attempt
# is a ~45MB download plus a few seconds of numpy, and a handful of targets
# have a dozen-plus sectors -- without a cap, one pathological star could turn
# a ~5s check into several minutes and blow through the watchdog. Six is well
# above the number of sectors most of these targets have at all, so in
# practice this bounds the worst case without changing the common one.
MAX_CENTROID_SECTORS_TO_TRY = 6


def _centroid_from_tpf(tpf, ra, dec, period_days, t0_bjd, duration_hours, expected_depth_ppm):
    """Runs the difference-image centroid measurement on ONE target pixel file.

    Returns (result_dict, None) if this TPF yields a trustworthy photocenter,
    or (None, reason) if it does not. Split out of _centroid_body so several
    sectors can be tried in turn -- every rejection reason below is a reason
    to try the NEXT sector, not to give up on the star.
    """
    import numpy as np
    from scipy.ndimage import center_of_mass
    from astropy.coordinates import SkyCoord
    import astropy.units as u

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
        return None, f"only {int(in_transit.sum())} in-transit cadences"
    if out_of_transit.sum() < 5:
        return None, "not enough out-of-transit cadences"

    with np.errstate(invalid="ignore"):
        in_transit_img = np.nanmedian(flux_cube[in_transit], axis=0)
        out_of_transit_img = np.nanmedian(flux_cube[out_of_transit], axis=0)
    diff_img = out_of_transit_img - in_transit_img
    diff_img_clipped = np.nan_to_num(np.clip(diff_img, 0, None), nan=0.0)

    if diff_img_clipped.sum() <= 0:
        return None, "difference image has no positive flux (signal too weak here)"

    row_com, col_com = center_of_mass(diff_img_clipped)
    target_coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    col_expected, row_expected = tpf.wcs.world_to_pixel(target_coord)

    # Plausibility guard: measure the dip at the target's OWN pixel and compare
    # it to the transit depth already known from characterization. If they
    # disagree, the phase-fold is not landing on a real transit in THIS
    # sector's data (different sector than the ephemeris came from, accumulated
    # period error, TTVs, or aliasing) and the resulting shift would be a
    # confident-looking meaningless number. Rejecting here is what lets the
    # caller move on and try a sector where the ephemeris does line up.
    row_int, col_int = int(round(float(row_expected))), int(round(float(col_expected)))
    ny, nx = diff_img.shape
    r0, r1 = max(0, row_int - 1), min(ny, row_int + 2)
    c0, c1 = max(0, col_int - 1), min(nx, col_int + 2)
    if r0 >= r1 or c0 >= c1:
        return None, "target's expected pixel falls outside this cutout"

    target_out_flux = float(np.nansum(out_of_transit_img[r0:r1, c0:c1]))
    target_in_flux = float(np.nansum(in_transit_img[r0:r1, c0:c1]))
    if target_out_flux <= 0:
        return None, "no usable flux at the target's pixel"

    observed_depth_ppm = (1.0 - target_in_flux / target_out_flux) * 1e6
    if expected_depth_ppm is not None and expected_depth_ppm > 0:
        ratio = observed_depth_ppm / expected_depth_ppm
        if not (0.1 <= ratio <= 10) or observed_depth_ppm <= 0:
            return None, (f"measured depth {observed_depth_ppm:.0f} ppm vs expected "
                          f"{expected_depth_ppm:.0f} ppm -- ephemeris doesn't line up here")

    shift_pixels = float(np.hypot(row_com - row_expected, col_com - col_expected))
    return {"shift_pixels": shift_pixels, "sector_used": sector_used,
            "observed_depth_ppm": observed_depth_ppm}, None


def _centroid_body(tic_id, host, ra, dec, period_days, t0_bjd, duration_hours, expected_depth_ppm):
    tpf = None
    try:
        if ra is None or dec is None:
            db.fail_centroid_check(tic_id, "No RA/Dec on file for this candidate")
            return
        if period_days is None or t0_bjd is None or duration_hours is None:
            db.fail_centroid_check(tic_id, "Missing period/epoch/duration -- run characterization first")
            return

        import lightkurve as lk

        db.set_check_progress("centroid_evidence", tic_id, "Searching MAST for a target pixel file...")
        search = lk.search_targetpixelfile(f"TIC {tic_id}", mission="TESS", author="SPOC")
        if len(search) == 0:
            search = lk.search_targetpixelfile(f"TIC {tic_id}", mission="TESS")
        if len(search) == 0:
            db.fail_centroid_check(tic_id, "No target pixel file available via MAST for this star")
            return

        # BUG FIXED -- root cause of EVERY centroid failure: 34 of 34 on the
        # candidate list, and 2,284 of 2,377 on the 5,086-star training run.
        #
        # This used to download exactly ONE product (`standard[0]`, whichever
        # MAST happened to list first) and give up if its data didn't match
        # the stored ephemeris. But the stored period/T0 come from one
        # specific sector, and which product MAST lists first is arbitrary.
        # Pick a different sector and the phase-fold lands on the wrong part
        # of the light curve, so the depth measured at the target's own pixel
        # disagrees with the known depth and the guard correctly refuses a
        # verdict. The guard was never the bug -- the single blind pick was.
        #
        # Evidence this is sector selection and not a data limitation: 28 of
        # those 34 failures measured a NEGATIVE depth (a brightening, not a
        # dip) at the target pixel, which is exactly what folding on
        # out-of-transit data looks like; and failures skewed to shallow,
        # short transits (median 197 ppm vs 340 ppm for successes) -- the
        # signals least tolerant of a mis-aligned fold.
        #
        # Now: try each available sector, keep the first whose measured depth
        # actually agrees with the known transit depth. 120s products first
        # (~6x smaller than the 20s ones, and the technique doesn't need fast
        # cadence). Each TPF is deleted before the next is fetched, so peak
        # disk stays at one file however many sectors get examined.
        try:
            standard = search[search.table["exptime"] == 120.0]
        except Exception:
            standard = search[:0]
        products = [standard[i] for i in range(len(standard))] or \
                   [search[i] for i in range(len(search))]
        sector_of = []
        try:
            table = (standard if len(standard) else search).table
            sector_of = [int(s) for s in table["sequence_number"]]
        except Exception:
            sector_of = [None] * len(products)
        products = products[:MAX_CENTROID_SECTORS_TO_TRY]

        m06 = _load_06()
        rejections = []
        shift_pixels = sector_used = None
        for i, product in enumerate(products):
            label = f"sector {sector_of[i]}" if i < len(sector_of) and sector_of[i] else f"product {i+1}"
            db.set_check_progress(
                "centroid_evidence", tic_id,
                f"Checking {label} ({i+1} of {len(products)}): downloading ~45MB "
                f"and testing it against this candidate's ephemeris...")
            with m06._DOWNLOAD_LOCK:
                tpf = _retry_on_stdout_race(product.download)
            if tpf is None:
                rejections.append(f"{label}: download failed")
                continue
            result, reason = _centroid_from_tpf(
                tpf, ra, dec, period_days, t0_bjd, duration_hours, expected_depth_ppm)
            try:
                os.remove(tpf.path)
            except OSError:
                pass
            tpf = None
            if result is not None:
                shift_pixels = result["shift_pixels"]
                sector_used = result["sector_used"]
                break
            rejections.append(f"{label}: {reason}")

        if shift_pixels is None:
            db.fail_centroid_check(
                tic_id,
                f"Checked {len(products)} available sector(s); none gave a trustworthy "
                f"photocenter. " + "; ".join(rejections[:4])
                + ("; ..." if len(rejections) > 4 else "")
                + ". Usually this means the transit is too shallow to localize in pixel data, "
                  "or the stored ephemeris doesn't propagate cleanly to any sector that has "
                  "pixel data available.")
            return

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
    body = lambda: _centroid_body(
        tic_id, candidate["host"], char.get("ra"), char.get("dec"),
        char.get("period_days"), char.get("epoch_bjd"), char.get("transit_duration_hours"),
        char.get("transit_depth_ppm"))
    thread = threading.Thread(
        target=_with_watchdog,
        args=(body, tic_id, db.fail_centroid_check, CENTROID_WATCHDOG_SECONDS,
              "Centroid check"),
        daemon=True,
    )
    thread.start()
