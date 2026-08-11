"""
eightsector_build_pool.py -- assemble a real 8-consecutive-sector TESS sample
for the period-ceiling extension.

Stages A-D (selection, download, preprocess, host features). Stage E is the
injection-recovery grid, in eightsector_run.py.

WHY 8 AND NOT 13
----------------
Measured live: of 827 SPOC 120 s targets near the south ecliptic pole, only 1
has a run of >= 13 CONSECUTIVE sectors, 3 have >= 10, but 135 have >= 8 and 333
have >= 6. TESS revisits the CVZ yearly, so high TOTAL sector counts are split
by year-long gaps -- and a year gap gives TLS a huge period grid over sparse
data, the pathology already filtered out of the K2 pool. So 8 consecutive
(~215 d) is the realistic ceiling, not 13.

WHY NOT JUST RUN 06_download_unknown.py --n-sectors 8
-----------------------------------------------------
That would run the whole discovery pipeline -- scoring, OOD flagging, ranking,
top-N candidate output -- writing candidate files and feeding the web DB. This
is a characterisation task; it must not manufacture candidate records. So this
reuses 06's FUNCTIONS (try_search, download_one_star, the preprocess step,
crowding, variability) without its main().

Everything is written under *_8sector paths so nothing here can collide with
production candidate data.
"""
import os
import re
import sys
import json
import time
import warnings
import collections
import importlib.util
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(HERE, "..")
ROOT = os.path.join(HERE, "..", "..")
RAW_DIR = os.path.join(ROOT, "data", "raw_8sector")
PROC_DIR = os.path.join(ROOT, "data", "processed_8sector")
POOL_CSV = os.path.join(HERE, "eightsector_host_pool.csv")
TARGETS_CSV = os.path.join(HERE, "eightsector_targets.csv")

N_CONSECUTIVE = 8
N_TARGETS = 45            # aim for ~40 usable after losses
SEP_RA, SEP_DEC = [85, 95], [-70, -63]


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CODE_DIR, fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def stage_a_select():
    """Find targets with a run of >= N_CONSECUTIVE sectors, and record WHICH
    sectors form that run (not just the count) -- only those get downloaded."""
    if os.path.exists(TARGETS_CSV):
        print(f"[A] reusing {TARGETS_CSV}")
        return pd.read_csv(TARGETS_CSV)
    from astroquery.mast import Observations
    print("[A] querying MAST for SPOC 120s near south ecliptic pole...", flush=True)
    obs = Observations.query_criteria(
        obs_collection="TESS", dataproduct_type="timeseries",
        provenance_name="SPOC", t_exptime=120, s_ra=SEP_RA, s_dec=SEP_DEC)
    by = collections.defaultdict(set)
    for r in obs:
        m = re.search(r"-s(\d{4})-", str(r["obs_id"]))
        if m:
            by[str(r["target_name"])].add(int(m.group(1)))
    print(f"    targets: {len(by)}", flush=True)

    rows = []
    for tic, secs in by.items():
        s = sorted(secs)
        best, run, bstart, start = 0, 1, None, s[0]
        for i in range(1, len(s)):
            if s[i] == s[i - 1] + 1:
                run += 1
            else:
                run, start = 1, s[i]
            if run > best:
                best, bstart = run, start
        if best == 1 and s:
            best, bstart = 1, s[0]
        if best >= N_CONSECUTIVE:
            # take the LAST N_CONSECUTIVE of the run (most recent, best calibrated)
            runsecs = list(range(bstart, bstart + best))[-N_CONSECUTIVE:]
            rows.append({"tic_id": int(tic), "longest_run": best,
                         "sectors_observed": ",".join(str(x) for x in runsecs)})
    df = pd.DataFrame(rows).sort_values("longest_run", ascending=False).head(N_TARGETS)
    df.to_csv(TARGETS_CSV, index=False)
    print(f"[A] {len(df)} targets with >= {N_CONSECUTIVE} consecutive sectors -> {TARGETS_CSV}", flush=True)
    return df


def stage_b_download(targets):
    """Reuses 06's download_one_star multi-sector branch: fetches each named
    sector and concatenates time-ordered. NO folding at a stored ephemeris --
    TLS does its own blind period search on the concatenated array."""
    m06 = _load("m06", "06_download_unknown.py")
    os.makedirs(RAW_DIR, exist_ok=True)
    m06.RAW_FOLDER = RAW_DIR
    from concurrent.futures import ThreadPoolExecutor, as_completed

    todo = []
    for _, r in targets.iterrows():
        fn = f"TIC_{int(r.tic_id)}"
        if os.path.exists(os.path.join(RAW_DIR, fn + ".csv")):
            continue
        todo.append((int(r.tic_id), fn, set(int(s) for s in str(r.sectors_observed).split(","))))
    print(f"[B] downloading {len(todo)} stars x {N_CONSECUTIVE} sectors "
          f"({len(targets)-len(todo)} already present)", flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(m06.download_one_star, t, f, s): f for t, f, s in todo}
        for fut in as_completed(futs):
            res = fut.result(); done += 1
            if done % 5 == 0 or done == len(todo):
                print(f"    [{done}/{len(todo)}] {(time.time()-t0)/60:.1f} min  "
                      f"last={res.get('host')} status={res.get('status')} "
                      f"sectors={res.get('n_sectors_downloaded')}", flush=True)
    return sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".csv"))


def stage_c_preprocess():
    """Same detrend/normalise/sigma-clip production applies to unknown stars."""
    m06 = _load("m06", "06_download_unknown.py")
    os.makedirs(PROC_DIR, exist_ok=True)
    m06.RAW_FOLDER, m06.PROCESSED_FOLDER = RAW_DIR, PROC_DIR
    print("[C] preprocessing...", flush=True)
    m06.preprocess_candidates()
    out = sorted(f for f in os.listdir(PROC_DIR) if f.endswith(".csv"))
    print(f"[C] processed {len(out)} files", flush=True)
    return out


def stage_d_features(targets):
    """st_rad/st_teff from TIC, crowding from TIC id, variability from the RAW
    (pre-flatten) curve -- the same sources production uses."""
    m06 = _load("m06", "06_download_unknown.py")
    files = [f[:-4] for f in os.listdir(PROC_DIR) if f.endswith(".csv")]
    df = pd.DataFrame({"host": files})
    df["tic_id"] = df.host.str.replace("TIC_", "", regex=False).astype("int64")

    print("[D] stellar params...", flush=True)
    sp = m06.fetch_stellar_params(pd.DataFrame({"tic_id": df.tic_id.values}))
    df = df.merge(sp, on="tic_id", how="left")

    print("[D] crowding...", flush=True)
    df = m06.add_crowding_features(df)

    print("[D] variability (from RAW)...", flush=True)
    df = m06.add_variability_features(df, raw_dir=RAW_DIR)

    HOST = ["st_rad", "st_teff", "crowd_flux_ratio_max", "crowd_nearest_arcsec",
            "var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]
    have = [c for c in HOST if c in df.columns]
    print(f"[D] host feature coverage:\n{df[have].notna().mean().round(3).to_string()}", flush=True)
    df.to_csv(POOL_CSV, index=False)
    print(f"[D] -> {POOL_CSV} ({len(df)} rows, "
          f"{int(df[have].notna().all(axis=1).sum())} with all 9 finite)", flush=True)
    return df


def main():
    t = stage_a_select()
    stage_b_download(t)
    stage_c_preprocess()
    d = stage_d_features(t)

    # continuity report -- the thing that must hold for the baseline to be real
    rows = []
    for f in sorted(os.listdir(PROC_DIR)):
        if not f.endswith(".csv"):
            continue
        arr = np.sort(pd.read_csv(os.path.join(PROC_DIR, f), usecols=["time"])["time"].to_numpy())
        if len(arr) < 100:
            continue
        g = np.diff(arr)
        rows.append((f[:-4], arr.max() - arr.min(), g.max(),
                     1 - g[g > 0.5].sum() / (arr.max() - arr.min()), len(arr)))
    c = pd.DataFrame(rows, columns=["host", "baseline", "max_gap", "duty", "n"])
    print("\n[continuity] baseline/max_gap/duty/n:")
    print(c[["baseline", "max_gap", "duty", "n"]].describe().round(3).loc[
        ["count", "mean", "50%", "min", "max"]].to_string())
    print(f"\nPREDICTED period_max = baseline/2 = {c.baseline.median()/2:.1f} d "
          f"(single sector 12.7, 3 sectors 38.2)")


if __name__ == "__main__":
    main()
