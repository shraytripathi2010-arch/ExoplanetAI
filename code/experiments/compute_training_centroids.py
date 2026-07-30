"""
compute_training_centroids.py -- runs the REAL, unmodified difference-image
centroid check (job_runner.py's _centroid_body) across all 5,491 training
stars, to test centroid displacement as an actual classifier FEATURE for
the first time (previously only ever displayed as evidence on candidate
pages, never fed into the model).

Reuses _centroid_body byte-for-byte via monkeypatching db.start_centroid_check
/save_centroid_result/fail_centroid_check to write to a local, isolated
results list instead of the production SQLite DB -- this must never mix
into web/exoplanet_candidates.db (that table's FK ties centroid_evidence to
the unknown-candidate `candidates` table, which training stars aren't part
of, and semantically this is a different kind of data: training-feature
computation, not live-candidate evidence). _fold_centroid_into_evidence_writeup
is untouched -- it already no-ops safely via `db.get_candidate() is None`
for any TIC ID not in the candidates table, confirmed by reading the code.

Parallelized across workers (I/O-bound TPF downloads), matching this
project's established pattern (e.g. augment_classical_dataset.py).
Checkpoints every N completions -- this is real, multi-hour, unattended
compute, and this project has been burned before by not checkpointing.
"""
import os
import sys
import time
import warnings
from concurrent.futures import (ProcessPoolExecutor, as_completed, FIRST_COMPLETED,
                                wait as concurrent_wait,
                                TimeoutError as FutureTimeoutError)


class _Stalled(Exception):
    """Raised when no job completes within STALL_GAP_S -- a real stall, as
    opposed to as_completed()'s whole-iteration deadline that previously
    masqueraded as one."""

warnings.filterwarnings("ignore")

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
WEB_DIR = os.path.join(SCRIPT_DIR, "..", "..", "web")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, WEB_DIR)

TRAINING_CSV = os.path.join(SCRIPT_DIR, "..", "..", "data", "training_dataset", "training.csv")
TIC_MAP_CSV = os.path.join(SCRIPT_DIR, "positive_class_tic_ids.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "training_centroid_results_multisector.csv")
# The single-sector run's output, kept for the before/after comparison. Writing
# the multi-sector run to a NEW file matters for a second reason: the resume
# logic below skips any host already present, so pointing at the old file would
# skip all 5,086 jobs and silently "finish" in seconds having done nothing.
PRIOR_RESULTS_PATH = os.path.join(SCRIPT_DIR, "training_centroid_results.csv")

N_WORKERS = 8
# BUG FOUND LIVE: the first full run hung for ~21 hours with zero progress
# and no crash -- _centroid_body's TPF download has no per-call network
# timeout (unlike 06_download_unknown.py's own downloads, which learned
# this lesson already), and the run spanned the machine going to sleep
# overnight, which silently kills in-flight connections without raising --
# a hung worker just sits forever, and ProcessPoolExecutor.as_completed()
# has no way to know. PER_JOB_TIMEOUT_S bounds this: a job that exceeds it
# is abandoned (counted as a timeout failure, not silently lost) so the
# whole batch can't freeze on one bad connection again.
# RAISED from 180 for the multi-sector re-run. _centroid_body now tries up to
# MAX_CENTROID_SECTORS_TO_TRY (6) sectors, downloading a TPF for each until one
# gives a trustworthy photocenter. A single-sector job averaged 37s; six
# attempts can legitimately take several minutes. At 180s this re-run would
# have aborted exactly the jobs that succeed on the 4th/5th/6th sector -- the
# entire population the fix exists to recover -- and the recovered-coverage
# number would have come out near zero for a purely artificial reason.
PER_JOB_TIMEOUT_S = 900

# SEPARATE BUG, found while preparing this re-run: PER_JOB_TIMEOUT_S above was
# added to stop the ~21-hour silent hang, but it never actually fired.
# as_completed() only yields futures that have ALREADY finished, so the
# future.result(timeout=...) inside the loop returns instantly and its timeout
# is dead code. A worker hung in a network call simply never gets yielded, and
# as_completed() -- given no timeout of its own -- blocks forever. That is
# precisely the original failure mode, still live.
#
# STALL_GAP_S is a gap between completions, enforced with concurrent.futures
# .wait() in a loop.
#
# It was originally passed to as_completed(fs, timeout=...) instead, which was
# simply wrong: that argument is a deadline for the ENTIRE iteration, not a gap
# between items. The result was a hard 20-minute cap on the whole run. Two full
# runs "stalled" at exactly 1201s -- 155 and 203 jobs in -- and both times the
# batch was perfectly healthy and progressing at ~6s/job; the guard meant to
# protect the run was the only thing killing it. Identical total wall times on
# two runs of different sizes were the giveaway.
STALL_GAP_S = 1200

CHECKPOINT_EVERY = 5

# THE ACTUAL ROOT-CAUSE FIX. Everything above only bounds how long the PARENT
# waits; none of it stops a worker from hanging, and the smoke test proved the
# hang is common rather than exotic -- 4 of 12 jobs never returned at all.
#
# Setting astropy/astroquery's remote_timeout in each worker (below) is not
# sufficient: lightkurve's product.download() goes through code paths that
# don't honour it, so a stalled socket read blocks the worker indefinitely and
# the pool loses that slot permanently. With 8 slots and a ~33% hang rate, the
# pool starves long before 5,086 jobs finish -- which is exactly the "~21-hour
# run with zero progress" originally blamed on the machine sleeping.
#
# SIGALRM alone was NOT enough, proven by the first full run: it stalled after
# 167 jobs with all 8 workers wedged and not one "worker self-aborted" row in
# the output. The reason is a limitation I got wrong: the kernel does deliver
# SIGALRM, but CPython only *raises* the resulting exception when the
# interpreter next executes bytecode. A thread parked in a C-level socket read
# never gets there, so the alarm sits pending forever and the worker stays
# wedged exactly as before.
#
# WORKER_SOCKET_TIMEOUT_S is the fix that actually reaches the blocking call:
# socket.setdefaulttimeout() makes every socket the worker subsequently creates
# raise on a stalled read at the socket layer itself, which returns control to
# Python and lets the exception propagate. The lightkurve/astroquery download
# path ultimately pulls from stpubdata.s3.amazonaws.com over ordinary sockets,
# so this covers it where astropy's own remote_timeout did not.
#
# SIGALRM is kept as a second line of defence for a hang that is NOT a socket
# read (e.g. a pathological loop in FITS parsing), where it does work.
WORKER_SOCKET_TIMEOUT_S = 90
WORKER_ALARM_S = 420


def _worker(args):
    """Top-level/picklable. Monkeypatches db's centroid-write functions to
    capture results locally instead of hitting the real, FK-constrained
    production table."""
    host, tic_id, ra, dec, period_days, t0_bjd, duration_hours, depth_ppm = args

    # REAL fix for the hang (PER_JOB_TIMEOUT_S above is only the backstop):
    # bound the actual network calls, same as 06_download_unknown.py's own
    # configure_network_timeouts -- each spawned worker process needs this
    # set independently, since it's process-local astropy/astroquery config.
    try:
        from astropy.utils.data import conf as astropy_conf
        astropy_conf.remote_timeout = 60
    except Exception:
        pass
    try:
        from astroquery.mast import conf as mast_conf
        mast_conf.timeout = 60
    except Exception:
        pass
    # The one that actually unwedges a stalled download -- see the comment on
    # WORKER_SOCKET_TIMEOUT_S. Must be set before any socket is created, i.e.
    # before the first astroquery/lightkurve network call in this process.
    try:
        import socket
        socket.setdefaulttimeout(WORKER_SOCKET_TIMEOUT_S)
    except Exception:
        pass

    import db
    import job_runner as jr

    captured = {"status": "never_run", "sector_used": None, "shift_pixels": None,
                "verdict": None, "error_message": None}

    def _fake_start(tic_id_arg):
        captured["status"] = "running"

    def _fake_save(tic_id_arg, sector_used, shift_pixels, verdict):
        captured.update(status="completed", sector_used=sector_used,
                         shift_pixels=shift_pixels, verdict=verdict)

    def _fake_fail(tic_id_arg, error_message):
        captured.update(status="failed", error_message=error_message)

    def _fake_get_candidate(tic_id_arg):
        return None  # correctly no-ops _fold_centroid_into_evidence_writeup

    db.start_centroid_check = _fake_start
    db.save_centroid_result = _fake_save
    db.fail_centroid_check = _fake_fail
    db.get_candidate = _fake_get_candidate

    import signal

    def _alarm(signum, frame):
        raise TimeoutError(f"worker self-aborted after {WORKER_ALARM_S}s")

    t0 = time.time()
    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(WORKER_ALARM_S)
    except Exception:
        pass  # non-POSIX; parent-side guards still apply
    try:
        jr._centroid_body(tic_id, host, ra, dec, period_days, t0_bjd, duration_hours, depth_ppm)
    except TimeoutError as e:
        captured.update(status="failed", error_message=str(e))
    except Exception as e:
        captured.update(status="failed", error_message=f"Unhandled: {e}")
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass
    elapsed = time.time() - t0

    return {"host": host, "tic_id": tic_id, "elapsed_s": elapsed, **captured}


def main():
    df = pd.read_csv(TRAINING_CSV)
    tic_map = pd.read_csv(TIC_MAP_CSV)
    tic_map = tic_map.dropna(subset=["tic_id"]).set_index("host")["tic_id"].astype("int64").to_dict()

    jobs = []
    skipped_no_tic, skipped_no_ephemeris = 0, 0
    for _, row in df.iterrows():
        host = row["host"]
        if row["label"] == 0:
            tic_id = int(str(host).replace("TIC_", ""))
        else:
            tic_id = tic_map.get(host)
            if tic_id is None:
                skipped_no_tic += 1
                continue
        ra, dec = row.get("ra"), row.get("dec")
        period, t0, duration = row.get("period"), row.get("T0"), row.get("duration")
        depth = row.get("depth")
        if pd.isna(ra) or pd.isna(dec) or pd.isna(period) or pd.isna(t0) or pd.isna(duration):
            skipped_no_ephemeris += 1
            continue
        duration_hours = duration * 24.0
        depth_ppm = (1.0 - depth) * 1e6 if pd.notna(depth) else None
        jobs.append((host, tic_id, ra, dec, period, t0, duration_hours, depth_ppm))

    print(f"{len(df)} total training rows. {skipped_no_tic} positive-class skipped (no resolved TIC ID), "
          f"{skipped_no_ephemeris} skipped (missing ra/dec/period/T0/duration). "
          f"{len(jobs)} jobs to run across {N_WORKERS} workers.")

    results = []
    already_done = set()
    if os.path.exists(RESULTS_PATH):
        prior = pd.read_csv(RESULTS_PATH)
        # Rows abandoned when a previous run hit the stall guard were never
        # actually attempted -- they are bookkeeping, not results. Treating them
        # as "done" would permanently poison the resume: the first stall would
        # mark every remaining star complete-with-no-data, and re-running would
        # cheerfully report a finished run with 2% coverage.
        abandoned = prior["error_message"].fillna("").str.contains(
            "Abandoned: batch stalled", regex=False)
        if abandoned.any():
            print(f"Discarding {int(abandoned.sum())} row(s) abandoned by a previous "
                  f"stall so they get retried.")
        prior = prior[~abandoned]
        results = prior.to_dict("records")
        already_done = {r["host"] for r in results}
        print(f"{len(already_done)} genuinely attempted already -- resuming.")
    jobs = [j for j in jobs if j[0] not in already_done]

    # --limit N: smoke-test the real code path on a handful of stars before
    # committing to an unattended multi-hour run.
    if "--limit" in sys.argv:
        n = int(sys.argv[sys.argv.index("--limit") + 1])
        jobs = jobs[:n]
        print(f"--limit {n}: running only {len(jobs)} job(s).")

    t0 = time.time()
    # NOT a `with` block on purpose: ProcessPoolExecutor.__exit__ calls
    # shutdown(wait=True), which would block forever on the very hung worker
    # the stall guard exists to escape -- the same trap 06_download_unknown.py
    # already documents for ThreadPoolExecutor.
    executor = ProcessPoolExecutor(max_workers=N_WORKERS)
    futures = {executor.submit(_worker, job): job for job in jobs}
    pending = dict(futures)
    i = 0
    stalled = False
    try:
        waiting = set(futures)
        while waiting:
            # wait()'s timeout IS a per-call gap, unlike as_completed()'s
            # whole-iteration deadline. An empty `done` therefore means nothing
            # finished in STALL_GAP_S -- a genuine stall -- while a healthy but
            # slow batch simply keeps looping.
            done, waiting = concurrent_wait(
                waiting, timeout=STALL_GAP_S, return_when=FIRST_COMPLETED)
            if not done:
                pending = {f: futures[f] for f in waiting}
                raise _Stalled()
            for future in done:
                job = pending.pop(future, futures[future])
                i += 1
                try:
                    res = future.result(timeout=PER_JOB_TIMEOUT_S)
                except FutureTimeoutError:
                    res = {"host": job[0], "tic_id": job[1], "elapsed_s": PER_JOB_TIMEOUT_S,
                           "status": "failed", "sector_used": None, "shift_pixels": None,
                           "verdict": None, "error_message": f"Timed out after {PER_JOB_TIMEOUT_S}s "
                                                             f"(likely a hung network call)"}
                except Exception as e:
                    res = {"host": job[0], "tic_id": job[1], "elapsed_s": None,
                           "status": "failed", "sector_used": None, "shift_pixels": None,
                           "verdict": None, "error_message": f"Worker crashed: {e}"}
                results.append(res)
            # CHECKPOINT_EVERY was 20, which meant a 12-job smoke run saved
            # nothing at all until the very end -- and when the cleanup path
            # below then crashed, all 8 finished results were lost. Save often
            # enough that no plausible crash can discard more than a few jobs.
            if i % CHECKPOINT_EVERY == 0 or i == len(jobs):
                elapsed = time.time() - t0
                n_completed = sum(1 for r in results if r["status"] == "completed")
                print(f"  [{i}/{len(jobs)}] done ({elapsed:.0f}s elapsed, "
                      f"~{elapsed/i*(len(jobs)-i):.0f}s remaining, "
                      f"{n_completed}/{len(results)} usable so far)", flush=True)
                pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    except _Stalled:
        stalled = True
        print(f"\nSTALLED: no job completed in {STALL_GAP_S}s with {len(pending)} "
              f"still outstanding. Recording them as timeouts and stopping rather than "
              f"hanging. Re-run to resume -- completed hosts are checkpointed.", flush=True)
        for future, job in pending.items():
            future.cancel()
            results.append({"host": job[0], "tic_id": job[1], "elapsed_s": None,
                            "status": "failed", "sector_used": None, "shift_pixels": None,
                            "verdict": None,
                            "error_message": f"Abandoned: batch stalled >{STALL_GAP_S}s"})
    finally:
        # Save BEFORE any teardown. The first version of this block called
        # shutdown() and then read executor._processes -- which shutdown() sets
        # to None, so it raised AttributeError inside `finally`, propagated out
        # of main(), and skipped the final to_csv entirely. Eight completed jobs
        # were thrown away by cleanup code whose only job was to tidy up.
        try:
            pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
        except Exception as e:
            print(f"WARNING: could not checkpoint results: {e}", flush=True)

        # Grab the process handles while they still exist, then shut down.
        procs = list((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        if stalled:
            # shutdown(wait=False) does not kill a process blocked in a socket
            # read, and those children would keep this script alive forever --
            # and, left orphaned, would accumulate exactly like the 24 stale
            # workers found on this machine from earlier abandoned runs.
            for proc in procs:
                try:
                    proc.terminate()
                except Exception:
                    pass

    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    df_r = pd.DataFrame(results)
    print(f"\nTotal wall time: {time.time()-t0:.0f}s for {len(jobs)} jobs")
    print(df_r["status"].value_counts())
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
