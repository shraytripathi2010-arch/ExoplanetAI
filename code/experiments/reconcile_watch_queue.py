"""reconcile_watch_queue.py -- one-time: give every label-watch queue row whose
star is represented in training.csv by a DIFFERENT row an explicit status.

  processed, own row gone, same star (by TIC) present with the same label
      -> superseded_by_dedup, superseded_by = the surviving training host
         (the 2026-08-02 TIC dedup removed 137 appended duplicates and left
         their queue rows saying 'processed')
  failed, star already present under another name
      -> already_present, superseded_by = that host
         (queued 2026-07-19, before membership was checked by TIC id)

Star identity is confirmed_hosts.py's rule. A processed row whose star appears
only with the OPPOSITE label is a conflict: the script lists it and stops.

DRY RUN by default: prints the exact rows and writes nothing.
--execute:
  1. backs up the DB with the sqlite backup API to --backup-dir and
     integrity-checks the copy;
  2. adds the new columns (db.ensure_schema; existing rows' other fields are
     untouched);
  3. applies every change in ONE transaction, through db.mark_queue_superseded
     (the same helper every future training-row removal calls).
Idempotent: only rows still in their old status are touched, so a second run
changes nothing.

Usage: python3 code/experiments/reconcile_watch_queue.py [--execute] [--db PATH] [--training PATH]
"""
import argparse
import os
import sqlite3
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "web"))
import confirmed_hosts  # noqa: E402
import db  # noqa: E402


def plan(db_path, training_path):
    resolve = confirmed_hosts.tic_resolver()
    tr = pd.read_csv(training_path, usecols=["host", "label"])
    tr["host"] = tr["host"].astype(str)
    own = set(tr["host"])
    by_tic = {}
    for h, lab in zip(tr["host"], tr["label"]):
        t = resolve(h)
        if t is not None:
            by_tic.setdefault(int(t), []).append((h, int(lab)))
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    q = pd.read_sql("SELECT host, label, status, processed_at FROM label_watch_queue", con)
    con.close()
    superseded, present, conflicts = [], [], []
    for r in q.itertuples():
        if r.host in own:
            continue
        rows = by_tic.get(int(r.host.replace("TIC_", "")), [])
        if not rows:
            continue
        same = [h for h, lab in rows if lab == int(r.label)]
        if r.status == "processed":
            (superseded if same else conflicts).append((r.host, same[0] if same else rows, int(r.label), r.processed_at))
        elif r.status == "failed":
            present.append((r.host, rows[0][0], int(r.label), r.processed_at))
    return superseded, present, conflicts


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--db", default=os.path.join(ROOT, "web", "exoplanet_candidates.db"))
    ap.add_argument("--training", default=os.path.join(ROOT, "data", "training_dataset", "training.csv"))
    ap.add_argument("--backup-dir", default=os.path.expanduser("~/exoplanetai-db-backup"))
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()

    superseded, present, conflicts = plan(a.db, a.training)
    print(f"{len(superseded)} processed -> superseded_by_dedup | {len(present)} failed -> already_present | "
          f"{len(conflicts)} label conflict(s)")
    for host, surv, lab, at in superseded:
        print(f"  superseded_by_dedup  {host:<15} label {lab}  survivor {surv:<22} processed_at {at}")
    for host, surv, lab, at in present:
        print(f"  already_present      {host:<15} label {lab}  present as {surv:<20} failed_at {at}")
    for host, rows, lab, at in conflicts:
        print(f"  CONFLICT             {host:<15} label {lab}  training has {rows}")
    if conflicts:
        sys.exit("STOPPING: a processed row's star is in training only with the opposite label. Nothing written.")
    if not a.execute:
        print("DRY RUN -- nothing written. Re-run with --execute.")
        return
    if not (superseded or present):
        print("Nothing to change.")
        return

    os.makedirs(a.backup_dir, exist_ok=True)
    backup = os.path.join(a.backup_dir, f"exoplanet_candidates_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_pre_queue_reconcile.db")
    src = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    dst = sqlite3.connect(backup)
    src.backup(dst)
    dst.close(); src.close()
    ok = sqlite3.connect(f"file:{backup}?mode=ro", uri=True).execute("PRAGMA integrity_check").fetchone()[0]
    if ok != "ok":
        sys.exit(f"STOPPING: backup integrity check returned {ok!r}. Nothing written.")
    print(f"backup: {backup} (integrity ok)")

    db.DB_PATH = a.db
    db.ensure_schema()
    conn = sqlite3.connect(a.db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        n_sup = db.mark_queue_superseded({h: s for h, s, _, _ in superseded}, conn=conn)
        now = db.now_iso()
        n_pre = sum(conn.execute(
            "UPDATE label_watch_queue SET status = 'already_present', superseded_by = ?, resolved_at = ? "
            "WHERE host = ? AND status = 'failed'", (s, now, h)).rowcount for h, s, _, _ in present)
        if (n_sup, n_pre) != (len(superseded), len(present)):
            raise RuntimeError(f"expected {len(superseded)}/{len(present)} updates, got {n_sup}/{n_pre}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    print(f"APPLIED in one transaction: {n_sup} superseded_by_dedup, {n_pre} already_present")


if __name__ == "__main__":
    main()
