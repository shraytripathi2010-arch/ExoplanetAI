"""
db.py -- SQLite persistence layer for the exoplanet candidate web app.

This is the ONE genuinely new piece of state the pipeline never had: every
CSV produced by 06/07/08 is a snapshot of the latest run only. This module
gives every candidate a permanent identity (by TIC ID) that survives across
runs, tracks when it was first found vs. last re-verified, and records a
full status-change history (e.g. "unknown -> flagged in TOI table on
[date]") instead of silently losing that fact the next time a CSV gets
overwritten.

Deliberately raw sqlite3, not an ORM: four small tables, all queries are
simple, and a thin hand-written layer is easier to reason about than
pulling in SQLAlchemy for this scale.
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exoplanet_candidates.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    tic_id INTEGER PRIMARY KEY,
    host TEXT NOT NULL,
    first_found_date TEXT NOT NULL,
    last_verified_date TEXT NOT NULL,
    current_status TEXT NOT NULL,
    predicted_probability REAL,
    confidence_tier TEXT,
    combined_filter_pass INTEGER NOT NULL DEFAULT 0,
    combined_filter_tier TEXT,
    needs_manual_review INTEGER NOT NULL DEFAULT 0,
    radius_plausible INTEGER,
    blending_status TEXT,
    vsx_code TEXT,
    characterization_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS status_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tic_id INTEGER NOT NULL REFERENCES candidates(tic_id),
    old_status TEXT,
    new_status TEXT NOT NULL,
    changed_date TEXT NOT NULL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    current_stage TEXT,
    progress_text TEXT,
    sample_size INTEGER,
    n_new_candidates INTEGER,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS verification_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tic_id INTEGER NOT NULL REFERENCES candidates(tic_id),
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    result_summary TEXT
);

-- Links every candidate to every Update run that touched it, so the
-- dashboard can show "batch from run on [date]" history instead of just
-- one flat candidate list. A candidate can appear under many runs (it gets
-- re-verified each time); is_new marks the run where it was first seen.
CREATE TABLE IF NOT EXISTS run_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    tic_id INTEGER NOT NULL REFERENCES candidates(tic_id),
    is_new INTEGER NOT NULL DEFAULT 0,
    UNIQUE(run_id, tic_id)
);

CREATE INDEX IF NOT EXISTS idx_status_history_tic ON status_history(tic_id);
CREATE INDEX IF NOT EXISTS idx_verification_events_tic ON verification_events(tic_id);
CREATE INDEX IF NOT EXISTS idx_run_candidates_run ON run_candidates(run_id);
CREATE INDEX IF NOT EXISTS idx_run_candidates_tic ON run_candidates(tic_id);

-- Single-row config for the in-process Update scheduler. Only fires while
-- the Flask process itself is running (see job_runner.py's scheduler
-- thread) -- there is no separate OS-level daemon backing this.
CREATE TABLE IF NOT EXISTS scheduler_config (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    enabled INTEGER NOT NULL DEFAULT 0,
    interval_days REAL NOT NULL DEFAULT 7,
    sample_size INTEGER NOT NULL DEFAULT 300,
    next_run_at TEXT,
    last_scheduled_run_at TEXT
);

-- On-demand, per-candidate multi-sector re-analysis (Item 1). Stored
-- SEPARATELY from the candidate's main classifier-derived fields on
-- purpose: this is supplementary evidence from a longer baseline, never
-- re-fed through the classifier/OOD checks (widening the baseline shifts
-- period/transit_count outside the training distribution, which would
-- cause false OOD flags if treated as a new classifier input).
CREATE TABLE IF NOT EXISTS multi_sector_evidence (
    tic_id INTEGER PRIMARY KEY REFERENCES candidates(tic_id),
    status TEXT NOT NULL DEFAULT 'never_run',
    sectors_used TEXT,
    n_sectors INTEGER,
    period_days REAL,
    t0_bjd REAL,
    duration_hours REAL,
    depth_ppm REAL,
    sde REAL,
    transit_count REAL,
    plausible INTEGER,
    plausibility_note TEXT,
    computed_at TEXT,
    error_message TEXT
);

-- On-demand, per-candidate difference-image centroid check (Item 5).
-- Independent of, and stored separately from, the existing Gaia-proximity
-- blend check -- they test related but different things (catalog
-- proximity/magnitude vs. where the actual transit signal's photocenter
-- falls), so both are kept visible rather than one replacing the other.
CREATE TABLE IF NOT EXISTS centroid_evidence (
    tic_id INTEGER PRIMARY KEY REFERENCES candidates(tic_id),
    status TEXT NOT NULL DEFAULT 'never_run',
    sector_used INTEGER,
    shift_pixels REAL,
    verdict TEXT,
    computed_at TEXT,
    error_message TEXT
);

-- Tracks the single-candidate re-verify action's own background-job state.
-- BUG FIXED: this action used to run synchronously inline in the Flask
-- request handler (unlike multi-sector/centroid, which were already
-- background jobs with polling) -- meaning the button click blocked the
-- browser tab for however long the external checks took, with zero
-- progress feedback, indistinguishable from the page being frozen. Now
-- follows the same pattern as the other two on-demand checks.
CREATE TABLE IF NOT EXISTS reverify_status (
    tic_id INTEGER PRIMARY KEY REFERENCES candidates(tic_id),
    status TEXT NOT NULL DEFAULT 'never_run',
    result_summary TEXT,
    computed_at TEXT,
    error_message TEXT
);

-- Continuous retraining infrastructure. Tracks every newly-published
-- confirmed planet / newly-dispositioned TOI false positive found by the
-- scheduler's periodic archive re-query, from first discovery through to
-- whether it made it into a training row.
CREATE TABLE IF NOT EXISTS label_watch_queue (
    host TEXT PRIMARY KEY,
    label INTEGER NOT NULL,
    source TEXT NOT NULL,
    discovered_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    n_points INTEGER,
    error_message TEXT,
    processed_at TEXT
);

-- Every retrain attempt the scheduler triggers, whether or not it got
-- promoted -- this project has always preferred showing negative results
-- over hiding them, and a "the scheduler tried and didn't promote" history
-- is exactly that kind of result.
CREATE TABLE IF NOT EXISTS retrain_attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    triggered_at TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    n_new_examples INTEGER NOT NULL,
    n_training_rows INTEGER,
    test_roc_auc REAL,
    production_roc_auc REAL,
    bootstrap_ci_lower REAL,
    bootstrap_ci_upper REAL,
    promoted INTEGER NOT NULL DEFAULT 0,
    model_version_id INTEGER,
    reasoning TEXT,
    error_message TEXT
);

-- One row per model version ever trained by the continuous-retraining
-- pipeline (not the original hand-run 05_train_models.py history, which
-- predates this table) -- versioned .joblib files live in
-- models/versions/, this table is the index + the "which one is live" flag.
CREATE TABLE IF NOT EXISTS model_versions (
    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at TEXT NOT NULL,
    model_path TEXT NOT NULL,
    is_live INTEGER NOT NULL DEFAULT 0,
    n_training_rows INTEGER,
    n_positive INTEGER,
    n_negative INTEGER,
    test_roc_auc REAL,
    bootstrap_ci_lower REAL,
    bootstrap_ci_upper REAL,
    retrain_attempt_id INTEGER REFERENCES retrain_attempts(attempt_id)
);

-- Single-row baseline recording the real class counts at the time of the
-- last full CNN-vs-classical architecture comparison (Part A). Retrains
-- compare the current dataset's class counts against this baseline to
-- flag "worth re-evaluating CNN viability" -- NOT to auto-build a CNN,
-- just to surface that real data has grown enough that the original
-- CNN-underperforms-classical finding might no longer hold. Update this
-- row by hand (db.update_architecture_baseline) whenever the comparison
-- is actually re-run, so the flag resets against the new baseline.
CREATE TABLE IF NOT EXISTS architecture_baseline (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    n_positive INTEGER NOT NULL,
    n_negative INTEGER NOT NULL,
    compared_at TEXT NOT NULL,
    note TEXT
);
"""


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# Columns added to tables after they were first created. `CREATE TABLE IF
# NOT EXISTS` is a no-op once a table already exists on disk -- it does NOT
# retroactively add new columns, which is a real bug this project hit
# directly (a DB already had `multi_sector_evidence` from before the
# `plausible`/`plausibility_note` columns were added, so new rows failed
# with "no such column: plausible" until this migration ran). Every future
# additive schema change should be listed here, not just added to SCHEMA.
MIGRATIONS = {
    "multi_sector_evidence": [
        ("plausible", "INTEGER"),
        ("plausibility_note", "TEXT"),
    ],
    "scheduler_config": [
        ("last_retrain_tick_at", "TEXT"),
    ],
    "candidates": [
        # Deliberately separate from `current_status` -- sync_from_csvs()
        # recomputes current_status fresh from the pipeline's own output on
        # EVERY sync (a real behavior, not a bug: it needs to catch a star
        # picking up a new archive/TOI flag between runs). A human review
        # decision stored in current_status itself would get silently
        # overwritten by the very next scheduled Update. This column is
        # never touched by upsert_candidate/sync, so it survives.
        ("manual_review_status", "TEXT"),
        ("manual_review_note", "TEXT"),
        ("manual_review_date", "TEXT"),
    ],
    "retrain_attempts": [
        ("cnn_reevaluation_flag", "INTEGER"),
        ("growth_pct_positive", "REAL"),
        ("growth_pct_negative", "REAL"),
    ],
}


def _run_migrations(conn):
    for table, columns in MIGRATIONS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for col_name, col_type in columns:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")


def _reset_orphaned_running_jobs(conn):
    """BUG FIXED: every background-job table (runs, multi_sector_evidence,
    centroid_evidence, reverify_status) marks a row 'running' when its
    thread starts, but nothing ever cleared that if the Flask process was
    killed/restarted mid-job (e.g. during development, or a crash) -- the
    thread dies with the process, but the DB row stays 'running' forever.
    Every route that checks "already running" then permanently 409s on that
    candidate/run, which looks from the outside exactly like "the button
    doesn't work" or "it's stuck" with no error ever shown. Since this
    function only runs at process startup, any row still 'running' at this
    point cannot belong to a thread in the just-started process -- it's
    necessarily orphaned from a previous process, so it's safe to fail it."""
    msg = "Interrupted by an app restart (no result was recorded) -- click again to retry."
    conn.execute("UPDATE runs SET status = 'failed', error_message = ? WHERE status = 'running'", (msg,))
    for table in ("multi_sector_evidence", "centroid_evidence", "reverify_status"):
        conn.execute(f"UPDATE {table} SET status = 'failed', error_message = ? WHERE status = 'running'", (msg,))


def ensure_schema():
    """Creates tables/columns if missing. Safe to call at any time, including
    mid-run (e.g. from sync.py during an in-progress Update) -- unlike
    init_db(), this never touches existing row data, so it can't stomp on a
    run that's still legitimately 'running'."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _run_migrations(conn)


def init_db():
    """Full startup initialization: schema + migrations + orphaned-job
    cleanup. BUG FIXED: this used to also be called from sync.py on every
    Update's sync step, which meant _reset_orphaned_running_jobs() ran while
    the very run being synced was itself still legitimately 'running' --
    it would immediately mark that in-progress run 'failed' with
    "Interrupted by an app restart", even though its background thread was
    alive and correctly continuing (e.g. into the automatic centroid-check
    batch). Only call this once, at actual process startup (app.py) -- use
    ensure_schema() everywhere else that just needs the tables to exist."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _run_migrations(conn)
        _reset_orphaned_running_jobs(conn)


def upsert_candidate(tic_id, host, status, predicted_probability, confidence_tier,
                      combined_filter_pass, combined_filter_tier, needs_manual_review,
                      radius_plausible, blending_status, vsx_code, characterization_dict,
                      note=None):
    """Insert a new candidate or update an existing one. Returns
    (is_new, status_changed) so the caller (sync.py) can log history."""
    now = now_iso()
    char_json = json.dumps(characterization_dict, default=str)

    with get_conn() as conn:
        existing = conn.execute(
            "SELECT current_status FROM candidates WHERE tic_id = ?", (tic_id,)
        ).fetchone()

        if existing is None:
            conn.execute(
                """INSERT INTO candidates
                   (tic_id, host, first_found_date, last_verified_date, current_status,
                    predicted_probability, confidence_tier, combined_filter_pass,
                    combined_filter_tier, needs_manual_review, radius_plausible,
                    blending_status, vsx_code, characterization_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tic_id, host, now, now, status, predicted_probability, confidence_tier,
                 int(combined_filter_pass), combined_filter_tier, int(needs_manual_review),
                 int(radius_plausible) if radius_plausible is not None else None,
                 blending_status, vsx_code, char_json),
            )
            return True, False

        old_status = existing["current_status"]
        status_changed = old_status != status
        conn.execute(
            """UPDATE candidates SET
                 host = ?, last_verified_date = ?, current_status = ?,
                 predicted_probability = ?, confidence_tier = ?, combined_filter_pass = ?,
                 combined_filter_tier = ?, needs_manual_review = ?, radius_plausible = ?,
                 blending_status = ?, vsx_code = ?, characterization_json = ?
               WHERE tic_id = ?""",
            (host, now, status, predicted_probability, confidence_tier,
             int(combined_filter_pass), combined_filter_tier, int(needs_manual_review),
             int(radius_plausible) if radius_plausible is not None else None,
             blending_status, vsx_code, char_json, tic_id),
        )
        if status_changed:
            conn.execute(
                """INSERT INTO status_history (tic_id, old_status, new_status, changed_date, note)
                   VALUES (?, ?, ?, ?, ?)""",
                (tic_id, old_status, status, now, note),
            )
        return False, status_changed


def list_candidates(sort_by="predicted_probability", ascending=False, tier_filter=None,
                     combined_only=False):
    allowed_sort = {"predicted_probability", "confidence_tier", "first_found_date",
                     "last_verified_date", "tic_id", "current_status"}
    if sort_by not in allowed_sort:
        sort_by = "predicted_probability"
    order = "ASC" if ascending else "DESC"

    query = "SELECT * FROM candidates WHERE 1=1"
    params = []
    if tier_filter:
        query += " AND confidence_tier = ?"
        params.append(tier_filter)
    if combined_only:
        query += " AND combined_filter_pass = 1"
    query += f" ORDER BY {sort_by} {order}"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_candidate(tic_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM candidates WHERE tic_id = ?", (tic_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["characterization"] = json.loads(d["characterization_json"])
        return d


def set_manual_review_decision(tic_id, status, note):
    """status: 'rejected' or 'accepted'. Deliberately a separate column from
    current_status -- see MIGRATIONS' comment for why. Also logs to
    status_history so it shows up in that existing audit trail, without
    changing current_status itself (the pipeline's own honest computed
    state stays visible and distinct from the human curation decision)."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE candidates SET manual_review_status = ?, manual_review_note = ?,
               manual_review_date = ? WHERE tic_id = ?""",
            (status, note, now_iso(), tic_id),
        )
        conn.execute(
            """INSERT INTO status_history (tic_id, old_status, new_status, changed_date, note)
               VALUES (?, ?, ?, ?, ?)""",
            (tic_id, "needs_manual_review", f"manually_{status}", now_iso(), note),
        )


def get_manual_review_decision(tic_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT manual_review_status, manual_review_note, manual_review_date "
            "FROM candidates WHERE tic_id = ?", (tic_id,)
        ).fetchone()
        return dict(row) if row else None


def get_status_history(tic_id):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM status_history WHERE tic_id = ? ORDER BY changed_date ASC", (tic_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_verification_events(tic_id):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM verification_events WHERE tic_id = ? ORDER BY timestamp DESC", (tic_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def log_verification_event(tic_id, event_type, result_summary):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO verification_events (tic_id, event_type, timestamp, result_summary)
               VALUES (?, ?, ?, ?)""",
            (tic_id, event_type, now_iso(), result_summary),
        )


def touch_last_verified(tic_id):
    with get_conn() as conn:
        conn.execute(
            "UPDATE candidates SET last_verified_date = ? WHERE tic_id = ?", (now_iso(), tic_id)
        )


def summary_counts():
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM candidates").fetchone()["c"]
        by_tier = {r["confidence_tier"]: r["c"] for r in conn.execute(
            "SELECT confidence_tier, COUNT(*) c FROM candidates GROUP BY confidence_tier")}
        combined = conn.execute(
            "SELECT COUNT(*) c FROM candidates WHERE combined_filter_pass = 1").fetchone()["c"]
        needs_review = conn.execute(
            "SELECT COUNT(*) c FROM candidates WHERE needs_manual_review = 1").fetchone()["c"]
        flagged = conn.execute(
            "SELECT COUNT(*) c FROM candidates WHERE current_status != 'unknown_candidate'"
        ).fetchone()["c"]
        return {
            "total": total, "by_tier": by_tier, "combined_filter_pass": combined,
            "needs_manual_review": needs_review, "flagged_since_found": flagged,
        }


# ---- runs table (background Update jobs) ----

def create_run(sample_size):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO runs (started_at, status, current_stage, progress_text, sample_size)
               VALUES (?, 'running', 'starting', '', ?)""",
            (now_iso(), sample_size),
        )
        return cur.lastrowid


def get_running_run():
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE status = 'running' ORDER BY run_id DESC LIMIT 1").fetchone()
        return dict(row) if row else None


def get_run(run_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def update_run_progress(run_id, current_stage=None, progress_text=None):
    sets, params = [], []
    if current_stage is not None:
        sets.append("current_stage = ?")
        params.append(current_stage)
    if progress_text is not None:
        sets.append("progress_text = ?")
        params.append(progress_text)
    if not sets:
        return
    params.append(run_id)
    with get_conn() as conn:
        conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE run_id = ?", params)


def finish_run(run_id, status, n_new_candidates=None, error_message=None):
    with get_conn() as conn:
        conn.execute(
            """UPDATE runs SET status = ?, finished_at = ?, n_new_candidates = ?, error_message = ?
               WHERE run_id = ?""",
            (status, now_iso(), n_new_candidates, error_message, run_id),
        )


def list_runs(limit=20):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY run_id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_or_create_initial_run():
    """Every candidate synced before the web app's Update job existed (the
    original 105 from the CSV-only pipeline) still needs a batch to belong
    to for the run-history view. Creates one retroactive 'initial import'
    run the first time this is called; reuses it on subsequent calls."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT run_id FROM runs WHERE current_stage = 'initial_import' LIMIT 1"
        ).fetchone()
        if row:
            return row["run_id"]
        cur = conn.execute(
            """INSERT INTO runs (started_at, finished_at, status, current_stage, progress_text, sample_size)
               VALUES (?, ?, 'completed', 'initial_import', 'Candidates present before the web app existed', NULL)""",
            (now_iso(), now_iso()),
        )
        return cur.lastrowid


# ---- run <-> candidate linkage (batch history) ----

def link_run_candidate(run_id, tic_id, is_new):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO run_candidates (run_id, tic_id, is_new) VALUES (?, ?, ?)
               ON CONFLICT(run_id, tic_id) DO UPDATE SET is_new = excluded.is_new""",
            (run_id, tic_id, int(is_new)),
        )


def get_candidates_for_run(run_id):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT c.*, rc.is_new FROM run_candidates rc
               JOIN candidates c ON c.tic_id = rc.tic_id
               WHERE rc.run_id = ?
               ORDER BY c.predicted_probability DESC""",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_run_candidate_counts(run_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) total, SUM(is_new) new FROM run_candidates WHERE run_id = ?", (run_id,)
        ).fetchone()
        return {"total": row["total"] or 0, "new": row["new"] or 0}


def backfill_unlinked_candidates_to_initial_run():
    """One-off migration helper: any candidate that predates the
    run_candidates table (synced before this feature existed) has no batch
    to belong to yet. Links every such orphan to the retroactive 'initial
    import' run so the dashboard's batch history is complete instead of
    silently missing the earliest candidates."""
    run_id = get_or_create_initial_run()
    with get_conn() as conn:
        orphans = conn.execute(
            """SELECT tic_id FROM candidates
               WHERE tic_id NOT IN (SELECT tic_id FROM run_candidates)"""
        ).fetchall()
    for row in orphans:
        link_run_candidate(run_id, row["tic_id"], is_new=1)
    return len(orphans)


# ---- scheduler config (single row) ----

def get_scheduler_config():
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM scheduler_config WHERE id = 1").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO scheduler_config (id, enabled, interval_days, sample_size) "
                "VALUES (1, 0, 7, 300)"
            )
            row = conn.execute("SELECT * FROM scheduler_config WHERE id = 1").fetchone()
        return dict(row)


def update_scheduler_config(enabled, interval_days, sample_size, next_run_at=None):
    get_scheduler_config()  # ensures the row exists
    with get_conn() as conn:
        conn.execute(
            """UPDATE scheduler_config SET enabled = ?, interval_days = ?, sample_size = ?,
               next_run_at = COALESCE(?, next_run_at) WHERE id = 1""",
            (int(enabled), interval_days, sample_size, next_run_at),
        )


def set_scheduler_next_run(next_run_at, last_scheduled_run_at=None):
    with get_conn() as conn:
        conn.execute(
            "UPDATE scheduler_config SET next_run_at = ?, "
            "last_scheduled_run_at = COALESCE(?, last_scheduled_run_at) WHERE id = 1",
            (next_run_at, last_scheduled_run_at),
        )


def get_last_retrain_tick_at():
    get_scheduler_config()
    with get_conn() as conn:
        row = conn.execute("SELECT last_retrain_tick_at FROM scheduler_config WHERE id = 1").fetchone()
        return row["last_retrain_tick_at"] if row else None


def set_last_retrain_tick_at(timestamp):
    with get_conn() as conn:
        conn.execute("UPDATE scheduler_config SET last_retrain_tick_at = ? WHERE id = 1", (timestamp,))


# ---- multi-sector strengthening evidence (Item 1) ----

def get_multi_sector_evidence(tic_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM multi_sector_evidence WHERE tic_id = ?", (tic_id,)).fetchone()
        return dict(row) if row else None


def start_multi_sector_check(tic_id):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO multi_sector_evidence (tic_id, status) VALUES (?, 'running')
               ON CONFLICT(tic_id) DO UPDATE SET status = 'running', error_message = NULL""",
            (tic_id,),
        )


def save_multi_sector_result(tic_id, sectors_used, n_sectors, period_days, t0_bjd,
                              duration_hours, depth_ppm, sde, transit_count,
                              plausible, plausibility_note):
    with get_conn() as conn:
        conn.execute(
            """UPDATE multi_sector_evidence SET
                 status = 'completed', sectors_used = ?, n_sectors = ?, period_days = ?,
                 t0_bjd = ?, duration_hours = ?, depth_ppm = ?, sde = ?, transit_count = ?,
                 plausible = ?, plausibility_note = ?, computed_at = ?, error_message = NULL
               WHERE tic_id = ?""",
            (sectors_used, n_sectors, period_days, t0_bjd, duration_hours, depth_ppm,
             sde, transit_count, int(plausible), plausibility_note, now_iso(), tic_id),
        )


def fail_multi_sector_check(tic_id, error_message):
    with get_conn() as conn:
        conn.execute(
            "UPDATE multi_sector_evidence SET status = 'failed', error_message = ? WHERE tic_id = ?",
            (error_message, tic_id),
        )


# ---- difference-image centroid evidence (Item 5) ----

def get_centroid_evidence(tic_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM centroid_evidence WHERE tic_id = ?", (tic_id,)).fetchone()
        return dict(row) if row else None


def get_candidates_needing_centroid():
    """Combined-filter-passing candidates that have never had a centroid
    check attempted at all -- the automatic Update-driven batch (Phase 1 of
    the multi-data-source effort) only runs for these, not for ones that
    already failed (e.g. "no TPF available" is normally a permanent fact
    about the star, so retrying it on every future Update forever would
    just waste time) -- a failed one can still be manually retried from its
    own detail page exactly as before."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT c.* FROM candidates c
               LEFT JOIN centroid_evidence ce ON ce.tic_id = c.tic_id
               WHERE c.combined_filter_pass = 1 AND ce.tic_id IS NULL"""
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["characterization"] = json.loads(d["characterization_json"])
            result.append(d)
        return result


def start_centroid_check(tic_id):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO centroid_evidence (tic_id, status) VALUES (?, 'running')
               ON CONFLICT(tic_id) DO UPDATE SET status = 'running', error_message = NULL""",
            (tic_id,),
        )


def save_centroid_result(tic_id, sector_used, shift_pixels, verdict):
    with get_conn() as conn:
        conn.execute(
            """UPDATE centroid_evidence SET
                 status = 'completed', sector_used = ?, shift_pixels = ?, verdict = ?,
                 computed_at = ?, error_message = NULL
               WHERE tic_id = ?""",
            (sector_used, shift_pixels, verdict, now_iso(), tic_id),
        )


def fail_centroid_check(tic_id, error_message):
    with get_conn() as conn:
        conn.execute(
            "UPDATE centroid_evidence SET status = 'failed', error_message = ? WHERE tic_id = ?",
            (error_message, tic_id),
        )


def get_reverify_status(tic_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM reverify_status WHERE tic_id = ?", (tic_id,)).fetchone()
        return dict(row) if row else None


def start_reverify_status(tic_id):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO reverify_status (tic_id, status) VALUES (?, 'running')
               ON CONFLICT(tic_id) DO UPDATE SET status = 'running', error_message = NULL""",
            (tic_id,),
        )


def save_reverify_result(tic_id, result_summary):
    with get_conn() as conn:
        conn.execute(
            """UPDATE reverify_status SET status = 'completed', result_summary = ?,
               computed_at = ?, error_message = NULL WHERE tic_id = ?""",
            (result_summary, now_iso(), tic_id),
        )


def fail_reverify(tic_id, error_message):
    with get_conn() as conn:
        conn.execute(
            "UPDATE reverify_status SET status = 'failed', error_message = ? WHERE tic_id = ?",
            (error_message, tic_id),
        )


def delete_run(run_id):
    """Removes a run's log entry (e.g. a noisy failed/0-candidate test run)
    from the dashboard's history. Only clears the run_candidates BATCH
    linkage for this run, not the underlying candidates themselves -- a
    candidate found by this run stays in the main candidate list either
    way, it just won't show up grouped under this run's batch anymore.
    Refuses to delete a run that's currently in progress (its background
    thread is still writing to it)."""
    with get_conn() as conn:
        row = conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return False, "not found"
        if row["status"] == "running":
            return False, "cannot delete a run that is currently in progress"
        conn.execute("DELETE FROM run_candidates WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        return True, None


# ---- continuous retraining: label watch queue ----

def get_known_watch_hosts():
    with get_conn() as conn:
        rows = conn.execute("SELECT host FROM label_watch_queue").fetchall()
        return {r["host"] for r in rows}


def add_watched_labels(rows):
    """rows: list of dicts with host, label, source. Ignores hosts already
    queued (ON CONFLICT DO NOTHING) -- this function is safe to call every
    scheduler tick with the full live re-query result, not just the diff."""
    if not rows:
        return 0
    now = now_iso()
    with get_conn() as conn:
        n = 0
        for r in rows:
            cur = conn.execute(
                """INSERT INTO label_watch_queue (host, label, source, discovered_at, status)
                   VALUES (?, ?, ?, ?, 'pending')
                   ON CONFLICT(host) DO NOTHING""",
                (r["host"], r["label"], r["source"], now),
            )
            n += cur.rowcount
        return n


def get_pending_watch_labels():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM label_watch_queue WHERE status = 'pending'").fetchall()
        return [dict(r) for r in rows]


def mark_watch_label_processed(host, n_points):
    with get_conn() as conn:
        conn.execute(
            """UPDATE label_watch_queue SET status = 'processed', n_points = ?,
               processed_at = ? WHERE host = ?""",
            (n_points, now_iso(), host),
        )


def mark_watch_label_failed(host, error_message):
    with get_conn() as conn:
        conn.execute(
            """UPDATE label_watch_queue SET status = 'failed', error_message = ?,
               processed_at = ? WHERE host = ?""",
            (error_message, now_iso(), host),
        )


def count_processed_watch_labels_since(since_iso):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) c FROM label_watch_queue WHERE status = 'processed' AND processed_at > ?",
            (since_iso,),
        ).fetchone()
        return row["c"]


# ---- continuous retraining: model versions + retrain attempts ----

def log_retrain_attempt(trigger_reason, n_new_examples, n_training_rows=None, test_roc_auc=None,
                         production_roc_auc=None, bootstrap_ci=None, promoted=False,
                         model_version_id=None, reasoning=None, error_message=None,
                         cnn_reevaluation_flag=False, growth_pct_positive=None,
                         growth_pct_negative=None):
    ci_lower, ci_upper = (bootstrap_ci if bootstrap_ci else (None, None))
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO retrain_attempts
               (triggered_at, trigger_reason, n_new_examples, n_training_rows, test_roc_auc,
                production_roc_auc, bootstrap_ci_lower, bootstrap_ci_upper, promoted,
                model_version_id, reasoning, error_message, cnn_reevaluation_flag,
                growth_pct_positive, growth_pct_negative)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now_iso(), trigger_reason, n_new_examples, n_training_rows, test_roc_auc,
             production_roc_auc, ci_lower, ci_upper, int(promoted), model_version_id,
             reasoning, error_message, int(cnn_reevaluation_flag), growth_pct_positive,
             growth_pct_negative),
        )
        return cur.lastrowid


def list_retrain_attempts(limit=50):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM retrain_attempts ORDER BY attempt_id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def add_model_version(model_path, n_training_rows, n_positive, n_negative, test_roc_auc,
                       bootstrap_ci, retrain_attempt_id, is_live=False):
    ci_lower, ci_upper = bootstrap_ci
    with get_conn() as conn:
        if is_live:
            conn.execute("UPDATE model_versions SET is_live = 0")
        cur = conn.execute(
            """INSERT INTO model_versions
               (trained_at, model_path, is_live, n_training_rows, n_positive, n_negative,
                test_roc_auc, bootstrap_ci_lower, bootstrap_ci_upper, retrain_attempt_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now_iso(), model_path, int(is_live), n_training_rows, n_positive, n_negative,
             test_roc_auc, ci_lower, ci_upper, retrain_attempt_id),
        )
        return cur.lastrowid


def list_model_versions(limit=50):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM model_versions ORDER BY version_id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_live_model_version():
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM model_versions WHERE is_live = 1 LIMIT 1").fetchone()
        return dict(row) if row else None


# ---- architecture-comparison baseline (single row) ----
# Real numbers from code/experiments/cnn_dataset.npz's real (non-synthetic)
# examples at the time of the Part A CNN-vs-classical comparison -- not a
# placeholder default, the actual measured class counts then.
_ARCHITECTURE_BASELINE_DEFAULT = {
    "n_positive": 4223, "n_negative": 1155, "compared_at": "2026-07-17",
    "note": "Seeded from code/experiments/cnn_dataset.npz real-example counts "
            "(Part A CNN vs. classical comparison, real_only test ROC-AUC 0.6964).",
}


def get_architecture_baseline():
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM architecture_baseline WHERE id = 1").fetchone()
        if row is None:
            d = _ARCHITECTURE_BASELINE_DEFAULT
            conn.execute(
                "INSERT INTO architecture_baseline (id, n_positive, n_negative, compared_at, note) "
                "VALUES (1, ?, ?, ?, ?)",
                (d["n_positive"], d["n_negative"], d["compared_at"], d["note"]),
            )
            row = conn.execute("SELECT * FROM architecture_baseline WHERE id = 1").fetchone()
        return dict(row)


def update_architecture_baseline(n_positive, n_negative, note=None):
    """Call this by hand after actually re-running the CNN-vs-classical
    comparison, so future growth is measured against the new baseline
    instead of re-flagging against a stale one forever."""
    get_architecture_baseline()  # ensures the row exists
    with get_conn() as conn:
        conn.execute(
            "UPDATE architecture_baseline SET n_positive = ?, n_negative = ?, "
            "compared_at = ?, note = ? WHERE id = 1",
            (n_positive, n_negative, now_iso(), note),
        )
