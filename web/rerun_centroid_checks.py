"""Re-run the centroid check (masked estimator + noise null) for candidates that have a centroid
record, in two steps so nothing live is written until the results have been looked at.

  python3 rerun_centroid_checks.py --compute out.csv [--shard i/n] [--limit k]
        Network (MAST target pixel files into the lightkurve cache, each deleted after use).
        Uses job_runner.measure_centroid() with each candidate's stored ephemeris, exactly as
        the page's "Re-run" button does. Writes only out.csv.
  python3 rerun_centroid_checks.py --apply out.csv [--db path]
        Writes each completed result to centroid_evidence (verdict, legacy offset, masked
        fields). A re-run that FAILED (e.g. a transient MAST error) leaves the stored record
        as it is, and is listed.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
import job_runner

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--compute"); g.add_argument("--apply")
ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int); ap.add_argument("--db")
a = ap.parse_args()
if a.db:
    db.DB_PATH = a.db

if a.compute:
    import csv, sqlite3
    i, n = map(int, a.shard.split("/"))
    con = sqlite3.connect(f"file:{db.DB_PATH}?mode=ro", uri=True)
    tics = [t for (t,) in con.execute("SELECT tic_id FROM centroid_evidence ORDER BY tic_id")][i::n]
    con.close()
    if a.limit:
        tics = tics[:a.limit]
    with open(a.compute, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tic_id", "result_json"])
        for k, tic in enumerate(tics):
            ch = db.get_candidate(tic)["characterization"]
            try:
                m = job_runner.measure_centroid(tic, ch.get("ra"), ch.get("dec"), ch.get("period_days"), ch.get("epoch_bjd"),
                                                ch.get("transit_duration_hours"), ch.get("transit_depth_ppm"))
            except Exception as e:
                m = {"status": "failed", "error": f"{type(e).__name__}: {str(e)[:200]}"}
            w.writerow([tic, json.dumps(m)]); f.flush()
            ms = (m.get("masked") or {})
            print(f"  {k + 1}/{len(tics)} TIC {tic}: {m['status']} | legacy {m.get('shift_pixels')} | masked {ms.get('masked_shift_pixels')} "
                  f"warn {ms.get('masked_warn')} | {m.get('error', '')[:80]}", flush=True)
else:
    import csv
    db.ensure_schema()   # adds the masked-estimator columns if this DB predates them
    rows = list(csv.DictReader(open(a.apply)))
    done = failed = 0; kept = []
    for r in rows:
        m = json.loads(r["result_json"]); tic = int(r["tic_id"])
        if m["status"] != "completed":
            kept.append((tic, m.get("error", "")[:100])); failed += 1; continue
        db.save_centroid_result(tic, m["sector_used"], m["shift_pixels"], m["verdict"])
        db.save_centroid_masked(tic, m.get("masked"))
        done += 1
    print(f"applied {done} completed results; {failed} failed re-runs left as stored: {kept[:12]}")
