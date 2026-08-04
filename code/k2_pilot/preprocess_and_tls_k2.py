"""preprocess_and_tls_k2.py -- K2 pilot steps 2/3: preprocess + TLS features.

Reuses 02_preprocess.process_one_file and 06_download_unknown.compute_all_features
rather than reimplementing them, so the K2 rows are produced by the SAME code
that produced every TESS row. That is a precondition for the domain-separability
check meaning anything: if the two populations went through different feature
code, a discriminator would be detecting the code path, not the data.

WHAT WAS CHANGED FOR K2, AND WHY -- exactly one thing

  MAX_FLATTEN_WINDOW: 401 points -> 27 points

This is not a tuning choice, it is a units bug waiting to happen. The
Savitzky-Golay detrending window is specified in POINTS, so its physical
duration depends entirely on cadence:

    TESS 2-min : 401 pts x  2.0 min = 13.4 hours
    K2   30-min: 401 pts x 29.4 min = 196.5 hours = 8.2 DAYS

Left alone, K2 stars would be detrended over an 8-day window -- wide enough to
leave stellar variability and K2's ~6-hour thruster-drift sawtooth entirely
intact, while the same nominal setting removes them for TESS. The domain
classifier would then be detecting a preprocessing artefact rather than a real
distribution difference, and the whole experiment would be measuring its own
bug.

Holding the window constant in TIME instead: 13.4 h / 29.4 min = 27.3 -> 27
points (odd, as savgol requires). Every other parameter is left alone and
verified as still appropriate:

  MIN_POINTS_FOR_FLATTEN = 50     K2 delivers ~3,500 pts/campaign -- never binds
  SIGMA_CLIP_THRESHOLD   = 5      cadence-independent, unchanged
  MAX_POINTS_BEFORE_BINNING = 30k K2 is ~3.5k -- never triggers
  TLS period grid                 left at defaults: K2's ~82-day baseline is 3x
                                  TESS's 27 days, not the 4-YEAR baseline that
                                  forced the Kepler pilot's binning cap

Writes only into code/k2_pilot/. Never touches data/processed/ or training.csv.
"""
import os
import sys
import time
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(PILOT_DIR, "..")
sys.path.insert(0, CODE_DIR)

RAW_DIR = os.path.join(PILOT_DIR, "raw")
PROCESSED_DIR = os.path.join(PILOT_DIR, "processed")
SAMPLE_PATH = os.path.join(PILOT_DIR, "k2_pilot_sample.csv")
FEATURES_PATH = os.path.join(PILOT_DIR, "k2_pilot_features.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

N_WORKERS = 8
K2_FLATTEN_WINDOW = 27      # 13.4 h at 29.4-min cadence; see module docstring


def _patch_window(m02, window=K2_FLATTEN_WINDOW):
    """Force the savgol window to the cadence-corrected value.

    Rebinding m02.MAX_FLATTEN_WINDOW alone does NOT work: choose_savgol_window
    binds it as a DEFAULT ARGUMENT at def time, so the old 401 is already
    captured. The function itself has to be replaced.

    The original is captured FIRST and called by the wrapper. Assigning a
    wrapper that looks the name up again at call time would resolve to itself
    and recurse -- a bug this project has already hit once.
    """
    original = m02.choose_savgol_window

    def cadence_corrected(n_points, max_window=window,
                          polyorder=m02.SAVGOL_POLYORDER):
        return original(n_points, max_window=max_window, polyorder=polyorder)

    m02.choose_savgol_window = cadence_corrected
    m02.MAX_FLATTEN_WINDOW = window
    return original


def preprocess_one(host):
    """process_one_file writes into the REAL data/processed/; move the output
    into the pilot folder immediately so the pilot cannot contaminate the
    training corpus even transiently."""
    import importlib
    m02 = importlib.import_module("02_preprocess")
    _patch_window(m02)
    raw_path = os.path.join(RAW_DIR, host + ".csv")
    if not os.path.exists(raw_path):
        return {"status": "No raw file"}
    result = m02.process_one_file(raw_path)
    real_out = os.path.join(m02.OUTPUT_FOLDER, host + ".csv")
    pilot_out = os.path.join(PROCESSED_DIR, host + ".csv")
    if os.path.exists(real_out):
        os.replace(real_out, pilot_out)
    return result


def _tls_worker(args):
    host, label, r_star, m_star = args
    import importlib
    m06 = importlib.import_module("06_download_unknown")
    m05 = importlib.import_module("05_train_models")
    path = os.path.join(PROCESSED_DIR, host + ".csv")
    t0 = time.time()
    feats, status = m06.compute_all_features(path, host, r_star, m_star,
                                             m05.FEATURE_COLUMNS)
    el = time.time() - t0
    if feats is None:
        return {"host": host, "label": label, "status": f"TLS: {status}",
                "elapsed_s": el}
    r = dict(feats)
    r.update({"host": host, "label": label, "status": "Success",
              "st_rad": r_star, "st_mass": m_star, "elapsed_s": el})
    return r


def main():
    sample = pd.read_csv(SAMPLE_PATH)
    have_raw = {f[:-4] for f in os.listdir(RAW_DIR) if f.endswith(".csv")}
    sample = sample[sample["host"].isin(have_raw)].reset_index(drop=True)
    print(f"{len(sample)} stars with raw data\n")

    # ---- cadence sanity check: prove the window change is warranted --------
    cad = []
    for h in sample["host"].head(20):
        try:
            t = pd.read_csv(os.path.join(RAW_DIR, h + ".csv"))["time"].to_numpy()
            d = np.median(np.diff(np.sort(t))) * 24 * 60
            if np.isfinite(d):
                cad.append(d)
        except Exception:
            pass
    med_cad = float(np.median(cad)) if cad else float("nan")
    print(f"measured median cadence: {med_cad:.1f} min "
          f"(TESS 2-min baseline: 2.0)")
    print(f"savgol window: {K2_FLATTEN_WINDOW} pts = "
          f"{K2_FLATTEN_WINDOW*med_cad/60:.1f} h "
          f"(TESS: 401 pts = {401*2.0/60:.1f} h)\n")

    print("Preprocessing (serial)...")
    pre = {}
    for _, r in sample.iterrows():
        host = r["host"]
        if os.path.exists(os.path.join(PROCESSED_DIR, host + ".csv")):
            pre[host] = "Success"
            continue
        try:
            pre[host] = preprocess_one(host)["status"]
        except Exception as e:
            pre[host] = f"Exception: {type(e).__name__}: {e}"
    ok = sum(1 for v in pre.values() if v == "Success")
    print(f"  {ok}/{len(sample)} preprocessed OK")
    for s, n in pd.Series(list(pre.values())).value_counts().items():
        if s != "Success":
            print(f"    {n:>3}  {s}")

    results, done = [], set()
    if os.path.exists(FEATURES_PATH):
        results = pd.read_csv(FEATURES_PATH).to_dict("records")
        done = {r["host"] for r in results}
        print(f"\n{len(done)} already have features -- resuming")

    jobs = []
    for _, r in sample.iterrows():
        if r["host"] in done:
            continue
        if pre.get(r["host"]) != "Success":
            results.append({"host": r["host"], "label": r["label"],
                            "status": f"Preprocess: {pre.get(r['host'])}"})
            continue
        jobs.append((r["host"], int(r["label"]),
                     r.get("st_rad"), r.get("st_mass")))

    print(f"\nRunning TLS on {len(jobs)} stars across {N_WORKERS} workers...")
    t0 = time.time()
    if jobs:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(_tls_worker, j): j[0] for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({"host": futs[fut],
                                    "status": f"Worker crash: {type(e).__name__}: {e}"})
                if i % 10 == 0 or i == len(jobs):
                    el = time.time() - t0
                    print(f"  [{i}/{len(jobs)}] {el/60:.1f} min elapsed, "
                          f"{el/i:.0f}s/star", flush=True)
                    pd.DataFrame(results).to_csv(FEATURES_PATH, index=False)

    df = pd.DataFrame(results)
    df.to_csv(FEATURES_PATH, index=False)

    print("\n" + "=" * 78)
    print("PILOT YIELD -- failure reasons in full")
    print("=" * 78)
    vc = df["status"].fillna("(none)").value_counts()
    for s, n in vc.items():
        print(f"  {n:>4}  {s}")
    good = df[df["status"] == "Success"]
    print(f"\n  YIELD: {len(good)}/{len(df)} = {100*len(good)/max(1,len(df)):.1f}%")
    if len(good):
        print(f"  class balance: {int((good.label==1).sum())} pos / "
              f"{int((good.label==0).sum())} neg")
        if "elapsed_s" in good:
            print(f"  median TLS time: {good['elapsed_s'].median():.0f}s/star")
    print(f"\nSaved {FEATURES_PATH}")


if __name__ == "__main__":
    main()
