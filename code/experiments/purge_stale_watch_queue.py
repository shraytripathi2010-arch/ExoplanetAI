"""purge_stale_watch_queue.py -- retire watch-queue entries for stars that are
already in training.csv under a different name.

The dedupe fix in `retrain_pipeline.find_new_labeled_examples` stops NEW
duplicate entries being queued, but 4,243 bad entries were already written
before the fix. Left alone, the scheduler would work through them on its
24h tick and re-add every one of those stars a second time -- exactly the
behaviour that produced 144 duplicated stars and 56 split-straddling leaks
from just the first 137 it processed.

Entries are marked `skipped_duplicate`, not deleted, so the queue keeps an
auditable record of what was retired and why. Anything that genuinely is not
in training.csv is left `pending` and untouched.

Backs up the affected rows to CSV before writing. Dry-run by default; pass
--apply to actually update.
"""
import os
import re
import sys
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
WEB = os.path.join(ROOT, "web")
sys.path.insert(0, WEB)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
ARCHIVE_CSV = os.path.join(SCRIPT_DIR, "archive_all_confirmed_tic.csv")
BACKUP = os.path.join(SCRIPT_DIR, "watch_queue_purged_rows.csv")
DB_PATH = os.path.join(WEB, "exoplanet_candidates.db")


def _pid(v):
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else None


def training_tics():
    """Same identity resolution the fixed pipeline uses."""
    hosts = pd.read_csv(TRAINING_CSV)["host"].astype(str)
    known = set(pd.to_numeric(hosts.str.extract(r"^TIC_(\d+)", expand=False),
                              errors="coerce").dropna().astype("int64"))
    a = pd.read_csv(ARCHIVE_CSV)
    a["_tic"] = a["tic_id"].map(_pid)
    name_to_tic = {}
    for h, t in zip(a["hostname"].astype(str), a["_tic"]):
        if pd.isna(t):
            continue
        t = int(t)
        name_to_tic.setdefault(h, t)
        name_to_tic.setdefault(h.replace(" ", "_"), t)
    for h in hosts:
        t = name_to_tic.get(h)
        if t is not None:
            known.add(t)
    return known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually update the DB (default is dry run)")
    args = ap.parse_args()

    import sqlite3
    con = sqlite3.connect(DB_PATH)
    q = pd.read_sql("select rowid,host,label,status from label_watch_queue "
                    "where status='pending'", con)
    print(f"pending watch-queue rows: {len(q)}")
    if q.empty:
        print("nothing pending")
        return

    known = training_tics()
    q["tic"] = pd.to_numeric(
        q["host"].astype(str).str.extract(r"^TIC_(\d+)", expand=False),
        errors="coerce")
    q["already_present"] = q["tic"].isin(known)

    stale = q[q.already_present]
    keep = q[~q.already_present]
    print(f"  already in training.csv by TIC (STALE -> retire): {len(stale)}")
    print(f"  genuinely not in training.csv (KEEP pending):     {len(keep)}")
    print(f"  unparseable host / no TIC:                        {int(q.tic.isna().sum())}")
    print("\n  by label:")
    print(stale.groupby("label").size().to_string() if len(stale) else "   (none)")

    if not len(stale):
        print("\nnothing to purge")
        return

    stale.to_csv(BACKUP, index=False)
    print(f"\nbacked up the rows to be retired -> {BACKUP}")

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to update.")
        return

    cur = con.cursor()
    cur.executemany("update label_watch_queue set status='skipped_duplicate' "
                    "where rowid=?", [(int(r),) for r in stale["rowid"]])
    con.commit()
    after = pd.read_sql("select status,count(*) n from label_watch_queue "
                        "group by status", con)
    print("\nAPPLIED. queue status now:")
    print(after.to_string(index=False))


if __name__ == "__main__":
    main()
