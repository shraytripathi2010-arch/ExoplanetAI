"""momentum_dump_schedule.py -- build a per-sector TESS momentum-dump time table.

WHY THIS IS NOT BLOCKED BY THE PRIOR CLOSURE
--------------------------------------------
The recorded closure said momentum-dump recovery "would need a full re-download
of all 5,494 training stars plus both candidate pools with quality_bitmask=0".
That is correct for a PER-STAR flag feature. It is wrong for a time-since-dump
feature, because **a momentum dump is a spacecraft event, not a stellar one**:
every star observed in a given sector shares the same dump times. So the schedule
costs ONE reference download per sector -- ~105 downloads, not ~9,200.

Verified before writing this: a `quality_bitmask=0` download of TIC 261136679
sector 1 returns 70 bit-32 (Desat) cadences forming 10 distinct events at
BTJD 1327.843 ... 1352.185, spaced 2.500 d -- exactly the interval the Sector 1
Data Release Notes document.

METHOD
  1. scan every raw light curve on disk for its (t_min, t_max)
  2. bin those into ~27.4 d sector-width bins
  3. for each bin, take one host that lives in it, search its SPOC 120 s
     products, and download ONLY the product whose time range overlaps the bin,
     with quality_bitmask=0
  4. extract bit-32 cadences, group cadences within 0.05 d into single dump
     EVENTS, and record the event times

Writes one JSON. Downloads nothing per-star and touches no production path.
"""
import os
import re
import glob
import json
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
OUT = os.path.join(HERE, "momentum_dump_schedule.json")
RAW_DIRS = ["data/known_lightcurves", "data/known_lightcurves_negative",
            "data/retrain_pipeline/raw", "data/unknown_lightcurves",
            "data/unknown_lightcurves_widesector"]
BIN = 27.4
DESAT = 32
EVENT_GAP = 0.05          # cadences closer than this are one dump event
BTJD_MAX = 100000.0       # above this the file is in full BJD, a known quirk


def scan_one(p):
    try:
        t = pd.read_csv(p, usecols=lambda c: c == "time", low_memory=False)["time"]
        t = pd.to_numeric(t, errors="coerce").dropna()
        if len(t) < 50:
            return None
        return (os.path.splitext(os.path.basename(p))[0], float(t.min()), float(t.max()))
    except Exception:
        return None


def main():
    files = []
    for d in RAW_DIRS:
        files += sorted(glob.glob(os.path.join(ROOT, d, "*.csv")))
    print(f"scanning {len(files)} raw light curves for time ranges...", flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = [r for r in ex.map(scan_one, files) if r]
    df = pd.DataFrame(rows, columns=["host", "tmin", "tmax"])
    btjd = df[df.tmin < BTJD_MAX]
    print(f"  {len(df)} usable, {len(btjd)} in BTJD "
          f"({len(df) - len(btjd)} in full BJD, excluded)", flush=True)

    lo = np.floor(btjd.tmin.min())
    btjd = btjd.assign(bin=((btjd.tmin - lo) // BIN).astype(int))
    bins = sorted(btjd.bin.unique())
    print(f"  {len(bins)} sector-width bins spanning "
          f"BTJD {btjd.tmin.min():.0f}-{btjd.tmax.max():.0f}", flush=True)

    import lightkurve as lk
    sched, failed = {}, []
    for k, b in enumerate(bins, 1):
        sub = btjd[btjd.bin == b]
        b0, b1 = lo + b * BIN, lo + (b + 1) * BIN
        got = False
        # try up to 4 hosts from this bin before giving up on it
        for host in sub.host.sample(min(4, len(sub)), random_state=b).tolist():
            name = re.sub(r"^TIC_", "TIC ", host).replace("_", " ")
            try:
                sr = lk.search_lightcurve(name, mission="TESS", author="SPOC",
                                          cadence=120)
                if not len(sr):
                    continue
                tstart = np.asarray(sr.table["t_min"], dtype=float)
                # MAST t_min is MJD; BTJD = BJD-2457000 = MJD+2400000.5-2457000
                tb = tstart + 2400000.5 - 2457000.0
                j = int(np.argmin(np.abs(tb - sub.tmin.median())))
                if abs(tb[j] - sub.tmin.median()) > 5.0:
                    continue
                lc = sr[j].download(quality_bitmask=0)
                q = np.asarray(lc.quality)
                t = np.asarray(lc.time.value)
                m = (q & DESAT).astype(bool) & np.isfinite(t)
                sec = int(lc.meta.get("SECTOR", -1))
                if not m.sum():
                    sched[str(sec)] = {"sector": sec, "bin": int(b),
                                       "t0": float(np.nanmin(t)), "t1": float(np.nanmax(t)),
                                       "dumps": [], "n_desat_cadences": 0,
                                       "source": host}
                    print(f"  [{k}/{len(bins)}] bin {b} -> sector {sec}: "
                          f"0 dump cadences ({host})", flush=True)
                    got = True
                    break
                dt = np.sort(t[m])
                ev, cur = [], [dt[0]]
                for x in dt[1:]:
                    if x - cur[-1] <= EVENT_GAP:
                        cur.append(x)
                    else:
                        ev.append(float(np.mean(cur))); cur = [x]
                ev.append(float(np.mean(cur)))
                sched[str(sec)] = {"sector": sec, "bin": int(b),
                                   "t0": float(np.nanmin(t)), "t1": float(np.nanmax(t)),
                                   "dumps": ev, "n_desat_cadences": int(m.sum()),
                                   "source": host}
                iv = np.diff(ev)
                print(f"  [{k}/{len(bins)}] bin {b} -> sector {sec}: {len(ev)} dumps, "
                      f"median interval {np.median(iv) if len(iv) else float('nan'):.3f} d "
                      f"({host})", flush=True)
                got = True
                break
            except Exception as e:
                continue
        if not got:
            failed.append(int(b))
            print(f"  [{k}/{len(bins)}] bin {b} (BTJD {b0:.0f}-{b1:.0f}): NO SCHEDULE",
                  flush=True)
    json.dump({"bin_width": BIN, "t_origin": float(lo), "sectors": sched,
               "failed_bins": failed}, open(OUT, "w"), indent=2)
    nd = sum(len(v["dumps"]) for v in sched.values())
    print(f"\nsaved {OUT}: {len(sched)} sectors, {nd} dump events, "
          f"{len(failed)} bins with no schedule")


if __name__ == "__main__":
    main()
