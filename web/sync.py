"""
sync.py -- reads the pipeline's already-computed output CSVs and upserts
them into the persistent SQLite DB (db.py). This is the piece that gives
the site permanent history: the CSVs from 06/07/08 are each a snapshot of
the latest run only and get overwritten every run; this module is what
turns those snapshots into an accumulating record with first-found/last-
verified dates and a status-change audit trail.

Deliberately does NOT recompute anything -- it only reads columns 08 already
wrote to disk (characterized_candidates.csv, final_best_candidates.csv,
needs_manual_blend_review.csv) and stores them as-is.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import db

RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                               "results", "unknown_candidates")
CHARACTERIZED_PATH = os.path.join(RESULTS_FOLDER, "characterized_candidates.csv")
BEST_CANDIDATES_PATH = os.path.join(RESULTS_FOLDER, "final_best_candidates.csv")
NEEDS_REVIEW_PATH = os.path.join(RESULTS_FOLDER, "needs_manual_blend_review.csv")


def _status_for_row(row):
    """The underlying pipeline data only distinguishes 'flagged somewhere in
    the archive/TOI/ExoFOP tables' vs. not -- it does NOT separately record
    whether that flag came from the confirmed-planet table specifically vs.
    a TOI/community-candidate entry (08's `newly_flagged` boolean already
    combines all three sources). Rather than invent a false distinction the
    data can't support, this uses two honest states, with the ExoFOP note
    (which does include the real TOI-#### designation when there is one)
    surfaced as the human-readable detail."""
    if bool(row.get("newly_flagged_in_archive_or_exofop")):
        return "flagged_in_archive_or_toi"
    return "unknown_candidate"


def sync_from_csvs(run_id=None):
    """Reads the latest characterized_candidates.csv (plus the combined-
    filter and needs-review lists it feeds) and upserts every row into the
    candidates table. Returns a summary dict for the caller (e.g. the job
    runner) to report back.

    08's output CSV is cumulative (every candidate ever characterized, not
    just this run's), so linking EVERY row to run_id would make every run's
    "batch" balloon to include all historical candidates -- defeating the
    point of batch history. Only candidates that are actually NEW this sync
    (is_new) get linked to run_id, so each run's batch in the dashboard is
    genuinely "what this Update found," not a repeat of everything."""
    db.ensure_schema()

    if run_id is None:
        run_id = db.get_or_create_initial_run()

    if not os.path.exists(CHARACTERIZED_PATH):
        return {"synced": 0, "new": 0, "status_changed": 0, "error": "characterized_candidates.csv not found"}

    full = pd.read_csv(CHARACTERIZED_PATH)
    best_tics = set()
    best_tier = {}
    if os.path.exists(BEST_CANDIDATES_PATH):
        best = pd.read_csv(BEST_CANDIDATES_PATH)
        best_tics = set(best["host"])
        best_tier = dict(zip(best["host"], best["combined_tier"]))
    review_tics = set()
    if os.path.exists(NEEDS_REVIEW_PATH):
        review = pd.read_csv(NEEDS_REVIEW_PATH)
        review_tics = set(review["host"])

    n_new, n_status_changed = 0, 0
    for _, row in full.iterrows():
        host = row["host"]
        tic_id = int(host.replace("TIC_", ""))
        status = _status_for_row(row)
        note = row.get("exofop_note") if status != "unknown_candidate" else None

        is_new, status_changed = db.upsert_candidate(
            tic_id=tic_id,
            host=host,
            status=status,
            predicted_probability=row.get("predicted_probability"),
            confidence_tier=row.get("confidence_tier"),
            combined_filter_pass=host in best_tics,
            combined_filter_tier=best_tier.get(host),
            needs_manual_review=host in review_tics,
            radius_plausible=row.get("radius_plausible"),
            blending_status=row.get("blending_status"),
            vsx_code=row.get("vsx_code"),
            characterization_dict=row.where(pd.notna(row), None).to_dict(),
            note=note,
        )
        if is_new:
            n_new += 1
            db.link_run_candidate(run_id, tic_id, is_new=1)
        if status_changed:
            n_status_changed += 1

    return {"synced": len(full), "new": n_new, "status_changed": n_status_changed,
            "run_id": run_id, "error": None}


if __name__ == "__main__":
    result = sync_from_csvs()
    print(f"Synced {result['synced']} candidates -- {result['new']} new, "
          f"{result['status_changed']} status changes (run_id={result['run_id']}).")
    if result["error"]:
        print(f"ERROR: {result['error']}")
