"""multisector_consistency.py -- ITEM 1: cross-sector consistency of the
transit signal.

FEASIBILITY (multisector_feasibility.py, 400-star stratified sample) cleared
both gates before this was written:
  - 97.8% of stars have >1 sector, median 7. Not a minority feature.
  - n_sectors single-feature AUC 0.479, Mann-Whitney p=0.459 -- observation
    history does NOT predict the label, so a multi-sector feature does not
    inherit a shortcut. This was the real risk and it did not materialise.

DESIGN DECISION THAT MAKES THIS AFFORDABLE

The obvious implementation -- re-run TLS per sector and compare recovered
parameters -- is not affordable and is not necessary. The full single-pass TLS
rerun earlier in this project took ~16.9 hours for one search per star; at a
median 7 sectors that is roughly 118 hours.

It is also the wrong tool. TLS exists to FIND an unknown period. The period,
T0 and duration are already known for every training star. Measuring whether
the transit repeats consistently only requires folding each sector at the
known ephemeris and measuring the depth -- no search, no periodogram. That is
milliseconds of compute per sector, and the cost collapses to the download.

Storage is handled the same way the centroid check already does it: download
one sector, measure, delete before fetching the next, so peak disk stays at a
few files rather than the several hundred GB a full multi-sector archive would
need (92 GB free on this machine).

FEATURES
  sector_depth_frac_scatter  std(depth) / mean(depth) across sectors.
      The core quantity. A real planet occults the same fraction of the same
      star every orbit, so depth should repeat within noise. An instrumental
      artefact or a blended eclipsing binary drifting in and out of the
      aperture produces depths that vary sector to sector. Scale-free, so it
      is comparable between a 200 ppm and a 20,000 ppm signal.

  sector_depth_chi2red       Uncertainty-weighted reduced chi-square of the
      per-sector depths about their mean.
      The scatter above cannot tell "genuinely inconsistent" from "measured
      badly in a noisy sector". Weighting each sector's depth by its own
      standard error asks whether the variation EXCEEDS what the measurement
      noise alone explains. ~1 means consistent, >>1 means real disagreement.

  n_sectors_measured         How many sectors produced a usable depth.
      Retained because the feasibility check specifically established it is
      not label-correlated (AUC 0.479); it also lets the model discount the
      two features above when they rest on few sectors.

MISSINGNESS: single-sector stars get NaN for the two consistency features and
1 for n_sectors_measured. NaN is correct rather than 0 -- a single-sector star
has no measured consistency, and 0 scatter would assert perfect consistency,
which is a claim the data cannot support. This follows the convention already
established here: median-impute via the same SimpleImputer every other feature
uses, with the earlier native-NaN/add_indicator work showing neither
alternative treatment clears the bar.

Resumable and checkpointed, matching compute_training_centroids.py.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait as cwait

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
TIC_MAP_CSV = os.path.join(SCRIPT_DIR, "positive_class_tic_ids.csv")
OUT_CSV = os.path.join(SCRIPT_DIR, "multisector_consistency.csv")

N_WORKERS = 6
MAX_SECTORS = 5          # enough to measure scatter; caps download cost
CHECKPOINT_EVERY = 25
STALL_GAP_S = 1800
WORKER_SOCKET_TIMEOUT_S = 90


class _Stalled(Exception):
    pass


def _measure_depth(t, f, period, t0, duration):
    """Depth and its standard error at a KNOWN ephemeris. No search."""
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if len(t) < 100:
        return None
    phase = ((t - t0 + 0.5 * period) % period) / period - 0.5
    half = (duration / period) / 2.0
    if not np.isfinite(half) or half <= 0 or half > 0.4:
        return None
    intr = np.abs(phase) <= half
    outr = np.abs(phase) >= (3 * half)
    if intr.sum() < 10 or outr.sum() < 30:
        return None
    base = np.median(f[outr])
    depth = base - np.median(f[intr])
    # standard error of the in-transit median, scaled for median vs mean
    se = 1.253 * np.std(f[intr]) / np.sqrt(intr.sum())
    return float(depth), float(se), int(intr.sum())


def compute_one(args):
    host, tic_id, period, t0, duration = args
    out = {"host": host, "tic_id": tic_id}
    try:
        import socket
        socket.setdefaulttimeout(WORKER_SOCKET_TIMEOUT_S)
        from astropy.utils.data import conf as aconf
        aconf.remote_timeout = 60
    except Exception:
        pass
    try:
        import lightkurve as lk
        if not all(np.isfinite([period, t0, duration])) or period <= 0:
            return {**out, "status": "no ephemeris"}

        search = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS")
        if search is None or len(search) == 0:
            return {**out, "status": "no data"}
        # one product per sector, preferring the first listed for that sector
        seen, picks = set(), []
        try:
            seqs = list(search.table["sequence_number"])
        except Exception:
            seqs = [None] * len(search)
        for i, s in enumerate(seqs):
            if s is not None and s not in seen:
                seen.add(s)
                picks.append((i, int(s)))
            if len(picks) >= MAX_SECTORS:
                break

        depths, ses, sectors = [], [], []
        for i, sec in picks:
            try:
                lc = search[i].download()
                if lc is None:
                    continue
                d = lc.remove_nans().normalize()
                t = np.asarray(d.time.value, dtype=float)
                f = np.asarray(d.flux.value, dtype=float)
                m = _measure_depth(t, f, period, t0, duration)
                if m is not None:
                    depths.append(m[0]); ses.append(m[1]); sectors.append(sec)
                try:
                    if getattr(lc, "path", None) and os.path.exists(lc.path):
                        os.remove(lc.path)   # bounded storage -- see module docstring
                except Exception:
                    pass
            except Exception:
                continue

        n = len(depths)
        out["n_sectors_measured"] = n
        out["sectors"] = ",".join(map(str, sectors))
        if n == 0:
            return {**out, "status": "no sector yielded a depth"}
        depths = np.array(depths); ses = np.array(ses)
        out["mean_depth"] = float(depths.mean())
        if n >= 2:
            mu = depths.mean()
            out["sector_depth_frac_scatter"] = (
                float(depths.std(ddof=1) / abs(mu)) if mu != 0 else np.nan)
            w = 1.0 / np.clip(ses, 1e-12, None) ** 2
            wmu = float((w * depths).sum() / w.sum())
            out["sector_depth_chi2red"] = float(
                (w * (depths - wmu) ** 2).sum() / (n - 1))
        else:
            out["sector_depth_frac_scatter"] = np.nan
            out["sector_depth_chi2red"] = np.nan
        out["status"] = "ok"
        return out
    except Exception as e:
        # Include the MESSAGE, not just the type. The first full run logged
        # bare "error: TypeError" for ~4% of stars, which is unactionable --
        # the same information-destroying handler pattern this project has
        # been bitten by before.
        return {**out, "status": f"error: {type(e).__name__}: {str(e)[:120]}"}


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    df = pd.read_csv(TRAINING_CSV)
    tmap = pd.read_csv(TIC_MAP_CSV).dropna(subset=["tic_id"])
    tmap = tmap.set_index("host")["tic_id"].astype("int64").to_dict()

    rows, done = [], set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        # Only completed rows count as done. Failures are RETRIED on resume --
        # a transient network error must not permanently mark a star as having
        # no multi-sector data. Marking failures as done is exactly the
        # poisoned-resume bug found in the label-watch queue earlier.
        ok_prev = prev[prev["status"].astype(str) == "ok"]
        rows = prev.to_dict("records")
        done = set(ok_prev["host"])
        n_retry = len(prev) - len(ok_prev)
        print(f"resuming: {len(done)} completed; {n_retry} prior failure(s) will be retried")
        if n_retry:
            rows = ok_prev.to_dict("records")

    jobs = []
    for _, r in df.iterrows():
        host = r["host"]
        if host in done:
            continue
        tic = (int(str(host).replace("TIC_", "")) if r["label"] == 0
               else tmap.get(host))
        if tic is None:
            continue
        jobs.append((host, int(tic), r.get("period"), r.get("T0"), r.get("duration")))
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} hosts to process, up to {MAX_SECTORS} sectors each, "
          f"{N_WORKERS} workers")
    if not jobs:
        print("nothing to do")
        return

    ex = ProcessPoolExecutor(max_workers=N_WORKERS)
    futures = {ex.submit(compute_one, j): j for j in jobs}
    waiting = set(futures)
    i, t0 = 0, time.time()
    stalled = False
    try:
        while waiting:
            got, waiting = cwait(waiting, timeout=STALL_GAP_S, return_when=FIRST_COMPLETED)
            if not got:
                raise _Stalled()
            for fut in got:
                i += 1
                try:
                    rows.append(fut.result())
                except Exception as e:
                    rows.append({"host": futures[fut][0], "status": f"crash: {e}"})
                if i % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                    el = time.time() - t0
                    print(f"  {i}/{len(jobs)} ({el:.0f}s, ~{el/i*(len(jobs)-i):.0f}s left)",
                          flush=True)
    except _Stalled:
        stalled = True
        print(f"STALLED: nothing completed in {STALL_GAP_S}s -- saving partial results")
    finally:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        procs = list((getattr(ex, "_processes", None) or {}).values())
        ex.shutdown(wait=False, cancel_futures=True)
        if stalled:
            for p in procs:
                try:
                    p.terminate()
                except Exception:
                    pass

    res = pd.DataFrame(rows)
    print(f"\ndone: {len(res)} rows")
    print(res["status"].value_counts().head(8).to_string())
    if "n_sectors_measured" in res:
        n = pd.to_numeric(res["n_sectors_measured"], errors="coerce")
        print(f"multi-sector (>=2 measured): {int((n >= 2).sum())}/{len(res)}")
    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
