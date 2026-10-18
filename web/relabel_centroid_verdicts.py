"""Recompute the stored centroid verdict text (and the masked warning flag) from the stored
measurements: the masked offset and its noise null where present (job_runner.centroid_masked_verdict,
centroid_masked_level), else the whole-cutout offset against the calibrated threshold
(job_runner.centroid_verdict). Offline: no download, the measurement and its date are kept. Dry
run by default; --apply writes. --db points at another database (scratch tests)."""
import argparse, os, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
import job_runner

ap = argparse.ArgumentParser(); ap.add_argument("--db"); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
if a.db:
    db.DB_PATH = a.db
con = sqlite3.connect(f"file:{db.DB_PATH}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute(
    "SELECT tic_id, shift_pixels, verdict, masked_shift_pixels, masked_warn, masked_null_p95, masked_null_frac_below "
    "FROM centroid_evidence WHERE status = 'completed' AND shift_pixels IS NOT NULL")]
n_fail = con.execute("SELECT count(*) FROM centroid_evidence WHERE status != 'completed'").fetchone()[0]
con.close()
new, level = {}, {}
for r in rows:
    t = r["tic_id"]
    if r["masked_shift_pixels"] is not None:
        level[t] = job_runner.centroid_masked_level(r["masked_shift_pixels"], r["masked_null_p95"])
        new[t] = (job_runner.centroid_masked_verdict(r), int(level[t] == "warn"))
    else:
        level[t] = "legacy"
        new[t] = (job_runner.centroid_verdict(r["shift_pixels"]), r["masked_warn"])
changed = [r["tic_id"] for r in rows if (r["verdict"], r["masked_warn"]) != new[r["tic_id"]]]
above = [r["tic_id"] for r in rows if r["shift_pixels"] >= job_runner.CENTROID_CALIBRATED_THRESHOLD_PIX]
print(f"completed {len(rows)} (not completed {n_fail}, left as is) | verdicts/flags to change {len(changed)} | above {job_runner.CENTROID_CALIBRATED_THRESHOLD_PIX} px: {len(above)} {above}")
print(f"masked warnings: {sum(bool(r['masked_warn']) for r in rows)} -> {sum(w for _, w in new.values() if w)}")
for lv in ("warn", "info_noise", "info_nonull", "below", "legacy"):
    ts = sorted(t for t in level if level[t] == lv)
    print(f"  {lv:<12} {len(ts):>3}" + (f": {ts}" if lv not in ("below", "legacy") else ""))
print(f"old verdicts mentioning a contaminant: {sum('contaminant' in (r['verdict'] or '').lower() for r in rows)} -> after: {sum('contaminant' in v.lower() for v, _ in new.values())}")
if a.apply:
    for t in changed:
        db.set_centroid_verdict(t, *new[t])
    print(f"applied: {len(changed)} verdicts/flags relabelled")
else:
    print("dry run: nothing written")
