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
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

warnings.filterwarnings("ignore")

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
WEB_DIR = os.path.join(SCRIPT_DIR, "..", "..", "web")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, WEB_DIR)

TRAINING_CSV = os.path.join(SCRIPT_DIR, "..", "..", "data", "training_dataset", "training.csv")
TIC_MAP_CSV = os.path.join(SCRIPT_DIR, "positive_class_tic_ids.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "training_centroid_results.csv")

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
PER_JOB_TIMEOUT_S = 180


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

    t0 = time.time()
    try:
        jr._centroid_body(tic_id, host, ra, dec, period_days, t0_bjd, duration_hours, depth_ppm)
    except Exception as e:
        captured.update(status="failed", error_message=f"Unhandled: {e}")
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
        results = pd.read_csv(RESULTS_PATH).to_dict("records")
        already_done = {r["host"] for r in results}
        print(f"{len(already_done)} already done -- resuming.")
    jobs = [j for j in jobs if j[0] not in already_done]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            job = futures[future]
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
            if i % 20 == 0 or i == len(jobs):
                elapsed = time.time() - t0
                n_completed = sum(1 for r in results if r["status"] == "completed")
                print(f"  [{i}/{len(jobs)}] done ({elapsed:.0f}s elapsed, "
                      f"~{elapsed/i*(len(jobs)-i):.0f}s remaining, "
                      f"{n_completed}/{len(results)} usable so far)", flush=True)
                pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)

    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    df_r = pd.DataFrame(results)
    print(f"\nTotal wall time: {time.time()-t0:.0f}s for {len(jobs)} jobs")
    print(df_r["status"].value_counts())
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
