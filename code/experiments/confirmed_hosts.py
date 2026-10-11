"""confirmed_hosts.py -- star identity by TIC id, and the set of TIC ids that
host a confirmed planet.

WHY THIS EXISTS. A TOI disposition belongs to one CANDIDATE signal; this
project's labels belong to a STAR. A star can carry an FP or FA candidate and
still host confirmed planets: TOI-712.03 is a false alarm, TOI-712 b/c/d are
confirmed. Six such stars sit in data/processed_negative/ (RESULTS_SUMMARY.md
register D3). A confirmed host must never be a negative, and the only identity
that joins a hostname-named positive ("TOI-712") to a TIC-named negative
("TIC_150151262") is the TIC id -- names cannot.

IDENTITY, in the same priority order as dedupe_training_by_tic.py, so the two
can never disagree about which star a row is:
  1. a host named TIC_<id>... -- parsed directly
  2. the archive's own hostname -> tic_id map (archive_all_confirmed_tic.csv)
  3. the coordinate cross-match (positive_class_tic_ids.csv; only rows that
     resolved carry a tic_id -- ambiguous and no-match rows are NaN)
Hosts that resolve to nothing (microlensing events, KMT-*-BLG-*) have no TIC
entry and cannot collide with a TESS target.

CONFIRMED-HOST SET, the union of:
  archive_all_confirmed_tic.csv   committed 2026-08-02, pscomppars hostname/tic_id
  positive_class_tic_ids.csv      committed 2026-07-16, the training positives'
                                   resolved TICs
  label_watch_queue               web/exoplanet_candidates.db, rows with source
                                   'pscomppars_confirmed', from the scheduler's
                                   live archive re-queries (2026-07-19 onward).
                                   Not committed: in a checkout without the
                                   database, the set is the two files only, and
                                   the provenance says so.
"""
import os
import re
import sqlite3

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
ARCHIVE_CSV = os.path.join("code", "experiments", "archive_all_confirmed_tic.csv")
COORD_CSV = os.path.join("code", "experiments", "positive_class_tic_ids.csv")
QUEUE_DB = os.path.join("web", "exoplanet_candidates.db")


def _pid(v):
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else None


def _read(rel, root):
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is required to establish star identity")
    return pd.read_csv(path)


def tic_resolver(root=None):
    """Returns resolve(host) -> TIC id or None."""
    root = ROOT if root is None else root
    amap = {}
    a = _read(ARCHIVE_CSV, root)
    for h, t in zip(a["hostname"].astype(str), a["tic_id"].map(_pid)):
        if t is None:
            continue
        amap.setdefault(h.replace(" ", "_"), t)
        amap.setdefault(h, t)
    o = _read(COORD_CSV, root).dropna(subset=["tic_id"])
    omap = o.set_index("host")["tic_id"].astype("int64").to_dict()

    def resolve(h):
        h = str(h)
        m = re.match(r"^TIC_(\d+)", h)
        if m:
            return int(m.group(1))
        return amap.get(h) or omap.get(h)

    return resolve


def confirmed_host_tics(root=None):
    """Returns (set of TIC ids, provenance dict)."""
    root = ROOT if root is None else root
    a = _read(ARCHIVE_CSV, root)
    archive = {t for t in a["tic_id"].map(_pid) if t is not None}
    o = _read(COORD_CSV, root).dropna(subset=["tic_id"])
    coord = set(o["tic_id"].astype("int64"))

    db_path = os.path.join(root, QUEUE_DB)
    queue, queue_note = set(), "not available (no database in this checkout)"
    if os.path.exists(db_path):
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT host FROM label_watch_queue "
                               "WHERE source = 'pscomppars_confirmed'").fetchall()
        finally:
            con.close()
        queue = {t for (h,) in rows if (t := _pid(h)) is not None}
        queue_note = f"{len(queue)} TICs"

    tics = archive | coord | queue
    prov = {"n_confirmed_host_tics": len(tics),
            "archive_all_confirmed_tic": len(archive),
            "positive_class_tic_ids": len(coord),
            "label_watch_queue": queue_note,
            "only_in_queue": len(queue - archive - coord)}
    return tics, prov


def summary(prov):
    return (f"confirmed-host TICs: {prov['n_confirmed_host_tics']} "
            f"(archive_all_confirmed_tic {prov['archive_all_confirmed_tic']}, "
            f"positive_class_tic_ids {prov['positive_class_tic_ids']}, "
            f"label_watch_queue {prov['label_watch_queue']}; "
            f"{prov['only_in_queue']} only in the queue)")
