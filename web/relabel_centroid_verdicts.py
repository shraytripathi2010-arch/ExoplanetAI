"""Recompute the stored centroid verdict text from each stored offset, with the calibrated
threshold (job_runner.centroid_verdict). Offline: no download, the measurement and its date are
kept. Dry run by default; --apply writes. --db points at another database (scratch tests)."""
import argparse, os, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
import job_runner

ap = argparse.ArgumentParser(); ap.add_argument("--db"); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
if a.db:
    db.DB_PATH = a.db
con = sqlite3.connect(f"file:{db.DB_PATH}?mode=ro", uri=True)
rows = con.execute("SELECT tic_id, shift_pixels, verdict FROM centroid_evidence WHERE status = 'completed' AND shift_pixels IS NOT NULL").fetchall()
n_fail = con.execute("SELECT count(*) FROM centroid_evidence WHERE status != 'completed'").fetchone()[0]
con.close()
new = {t: job_runner.centroid_verdict(s) for t, s, _ in rows}
changed = [t for t, s, v in rows if v != new[t]]
above = [t for t, s, _ in rows if s >= job_runner.CENTROID_CALIBRATED_THRESHOLD_PIX]
print(f"completed {len(rows)} (not completed {n_fail}, left as is) | verdicts to change {len(changed)} | above {job_runner.CENTROID_CALIBRATED_THRESHOLD_PIX} px: {len(above)} {above}")
print(f"old verdicts mentioning a contaminant: {sum('contaminant' in (v or '').lower() for _, _, v in rows)} -> after: {sum('contaminant' in v.lower() for v in new.values())}")
if a.apply:
    for t in changed:
        db.set_centroid_verdict(t, new[t])
    print(f"applied: {len(changed)} verdicts relabelled")
else:
    print("dry run: nothing written")
