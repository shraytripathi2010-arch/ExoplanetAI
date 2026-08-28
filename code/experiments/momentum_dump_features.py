"""momentum_dump_features.py -- "time since last momentum dump" at the transit
epoch, built from the per-sector dump schedule.

PHYSICAL HYPOTHESIS, stated before measurement
----------------------------------------------
A momentum dump fires the thrusters to unload reaction-wheel momentum. Pointing
jitter and residual attitude error are elevated for some minutes-to-hours
afterwards, which injects photometric systematics. A transit-like signal whose
events cluster shortly AFTER dumps is therefore more likely to be instrumental
than astrophysical -> `md_*` proximity should be enriched in the NEGATIVE class.

The sharper version of the same idea, and the one with a real mechanism behind
it: if the candidate's PERIOD is commensurate with the dump INTERVAL (2.5 d in
S1-4, 3.0-3.375 d in S5-13, 4.0+ d later), then a periodic instrumental
artefact can masquerade as a transit at that period. `md_period_ratio` targets
exactly that, and it is the feature most likely to carry signal if any does.

FEATURES
  md_min_dt          min over this star's transits of (t_transit - t_last_dump)
  md_median_dt       median of the same
  md_frac_near_6h    fraction of transits within 6 h after a dump
  md_frac_near_24h   fraction within 24 h after a dump
  md_n_dumps         dumps in the star's sector (a SECTOR-EPOCH proxy: watch it)
  md_dump_interval   median dump interval in the sector (same caveat)
  md_period_ratio    |log(P_transit / (n * dump_interval))| minimised over
                     n in {1, 2, 1/2, 3, 1/3} -- 0 = exactly commensurate

`md_n_dumps` and `md_dump_interval` are included deliberately AND flagged: they
are functions of sector alone, so they double as the built-in confound probe.
If they carry the signal and the proximity features do not, the "feature" is an
observation-epoch proxy, not a systematics one.
"""
import os
import json
import glob
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CAT = os.path.join(ROOT, "data", "catalogs")
SCHED = os.path.join(HERE, "momentum_dump_schedule.json")
OUT = os.path.join(HERE, "momentum_dump_features.csv")

TRAIN_RAW_DIRS = [os.path.join(ROOT, "data", "known_lightcurves"),
                  os.path.join(ROOT, "data", "known_lightcurves_negative"),
                  os.path.join(ROOT, "data", "retrain_pipeline", "raw")]
POOLS = [("main", "unknown_features.csv",
          os.path.join(ROOT, "data", "unknown_lightcurves")),
         ("widesector", "unknown_features_widesector.csv",
          os.path.join(ROOT, "data", "unknown_lightcurves_widesector"))]

HARMONICS = [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]
BTJD_MAX = 100000.0
COLUMNS = ["md_min_dt", "md_median_dt", "md_frac_near_6h", "md_frac_near_24h",
           "md_n_dumps", "md_dump_interval", "md_period_ratio"]

_SCHED = None


def sched():
    global _SCHED
    if _SCHED is None:
        _SCHED = json.load(open(SCHED))
    return _SCHED


def _sector_for(tmin, tmax):
    """Pick the schedule entry whose time range overlaps this light curve most."""
    best, bov = None, 0.0
    for v in sched()["sectors"].values():
        ov = min(tmax, v["t1"]) - max(tmin, v["t0"])
        if ov > bov:
            best, bov = v, ov
    return best if bov > 1.0 else None


def dump_features_for(raw_path, period, t0, duration):
    out = {c: np.nan for c in COLUMNS}
    out["md_status"] = "ok"
    if not raw_path or not os.path.exists(raw_path):
        out["md_status"] = "no raw light curve"
        return out
    if not (np.isfinite(period) and np.isfinite(t0) and period > 0):
        out["md_status"] = "no usable ephemeris"
        return out
    try:
        t = pd.read_csv(raw_path, usecols=lambda c: c == "time",
                        low_memory=False)["time"]
        t = pd.to_numeric(t, errors="coerce").dropna().to_numpy()
        if len(t) < 50:
            out["md_status"] = "too few points"
            return out
        tmin, tmax = float(t.min()), float(t.max())
        if tmin > BTJD_MAX:
            out["md_status"] = "full-BJD time system"
            return out
        s = _sector_for(tmin, tmax)
        if s is None:
            out["md_status"] = "no dump schedule for this sector"
            return out
        dumps = np.asarray(s["dumps"], dtype=float)
        out["md_n_dumps"] = float(len(dumps))
        out["md_dump_interval"] = float(np.median(np.diff(dumps))) if len(dumps) > 1 else np.nan
        if len(dumps) == 0:
            out["md_status"] = "sector has no dumps"
            return out

        # transit mid-times inside this light curve's span
        k0 = int(np.ceil((tmin - t0) / period))
        k1 = int(np.floor((tmax - t0) / period))
        if k1 < k0:
            out["md_status"] = "no transits in span"
            return out
        tt = t0 + period * np.arange(k0, k1 + 1)
        if len(tt) == 0:
            out["md_status"] = "no transits in span"
            return out

        # time since the most recent PRECEDING dump, per transit
        idx = np.searchsorted(dumps, tt, side="right") - 1
        ok = idx >= 0
        if not ok.any():
            out["md_status"] = "all transits precede first dump"
        else:
            dt = tt[ok] - dumps[idx[ok]]
            out["md_min_dt"] = float(dt.min())
            out["md_median_dt"] = float(np.median(dt))
            out["md_frac_near_6h"] = float((dt < 0.25).mean())
            out["md_frac_near_24h"] = float((dt < 1.0).mean())

        iv = out["md_dump_interval"]
        if np.isfinite(iv) and iv > 0:
            out["md_period_ratio"] = float(min(
                abs(np.log(period / (n * iv))) for n in HARMONICS))
    except Exception as e:
        out["md_status"] = f"error: {type(e).__name__}: {e}"
    return out


def worker(job):
    host, path, per, t0, dur = job
    r = dump_features_for(path, per, t0, dur)
    r["host"] = host
    return r


def _find(host, dirs):
    for d in dirs:
        p = os.path.join(d, str(host) + ".csv")
        if os.path.exists(p):
            return p
    return None


def _jobs(df, dirs):
    return [(str(r["host"]), _find(r["host"], dirs),
             pd.to_numeric(r.get("period"), errors="coerce"),
             pd.to_numeric(r.get("T0", np.nan), errors="coerce"),
             pd.to_numeric(r.get("duration"), errors="coerce"))
            for _, r in df.iterrows()]


def main():
    tr = pd.read_csv(TRAINING); tr["host"] = tr.host.astype(str)
    jobs = [("train", j) for j in _jobs(tr, TRAIN_RAW_DIRS)]
    for tag, ff, d in POOLS:
        p = pd.read_csv(os.path.join(CAT, ff)); p["host"] = p.host.astype(str)
        if "status" in p.columns:
            p = p[p.status.astype(str).str.startswith("Success")]
        jobs += [(tag, j) for j in _jobs(p, [d])]
    print(f"{len(jobs)} candidates; schedule covers "
          f"{len(sched()['sectors'])} sectors", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for n, (setname, r) in enumerate(
                zip([s for s, _ in jobs],
                    ex.map(worker, [j for _, j in jobs])), 1):
            r["_set"] = setname
            rows.append(r)
            if n % 1000 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"saved {OUT}  {len(out)} rows")
    print(out.groupby("_set").md_status.value_counts())


if __name__ == "__main__":
    main()
