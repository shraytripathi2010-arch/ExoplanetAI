"""
test_binning_fix.py -- quick, parallelized, unbuffered-output test of the
binning-cap fix (MAX_POINTS_BEFORE_BINNING/TARGET_POINTS_AFTER_BINNING)
against the 8 known-failing (transit_shape_ratio not computable) pilot
stars. Writes each result to a CSV as it completes, so progress is visible
without waiting on stdout buffering.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(PILOT_DIR, "..")
sys.path.insert(0, CODE_DIR)

import pandas as pd

RESULTS_PATH = os.path.join(PILOT_DIR, "binning_fix_test_results.csv")
N_WORKERS = 8


def _worker(args):
    host, st_rad, st_mass = args
    t0 = time.time()
    import importlib
    m06 = importlib.import_module("06_download_unknown")
    m05 = importlib.import_module("05_train_models")
    m06.MAX_POINTS_BEFORE_BINNING = 100000
    m06.TARGET_POINTS_AFTER_BINNING = 70000

    path = os.path.join(PILOT_DIR, "processed", host + ".csv")
    feat, status = m06.compute_all_features(path, host, st_rad, st_mass, m05.FEATURE_COLUMNS)
    elapsed = time.time() - t0
    if feat is None:
        return {"host": host, "status": status, "elapsed_s": elapsed}
    return {"host": host, "status": "Success", "elapsed_s": elapsed,
            "transit_shape_ratio": feat.get("transit_shape_ratio")}


def main():
    sample = pd.read_csv(os.path.join(PILOT_DIR, "koi_pilot_sample.csv"))
    feats = pd.read_csv(os.path.join(PILOT_DIR, "pilot_features.csv"))
    failing = feats[feats["status"].str.contains("transit_shape_ratio", na=False)]["host"].tolist()[:8]
    print(f"Testing {len(failing)} known-failing stars: {failing}", flush=True)

    jobs = []
    for host in failing:
        kepid = int(host.replace("KIC_", ""))
        row = sample[sample["kepid"] == kepid].iloc[0]
        jobs.append((host, row.get("koi_srad"), row.get("koi_smass")))

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker, job): job for job in jobs}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
            print(f"  {res['host']}: {res['status']} ({res['elapsed_s']:.0f}s, "
                  f"{time.time()-t0:.0f}s total elapsed)", flush=True)

    df = pd.DataFrame(results)
    n_ok = (df["status"] == "Success").sum()
    print(f"\n{n_ok}/{len(df)} now succeed with the raised binning cap.", flush=True)
    print(f"Saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
