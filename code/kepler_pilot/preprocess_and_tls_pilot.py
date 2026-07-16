"""
preprocess_and_tls_pilot.py -- Kepler pilot steps 2/3: preprocess + TLS
feature extraction, reusing 02_preprocess.py's process_one_file and
06_download_unknown.py's compute_all_features UNCHANGED (imported, not
reimplemented) -- exactly matching this project's established pattern of
reusing proven pipeline code rather than parallel-implementing it.

REAL COST FINDING (measured live, not assumed): a single Kepler star's TLS
run took 661.5s -- ~18x slower than TESS's typical ~36s -- because Kepler's
~4-year observing baseline (vs TESS's ~27 days) makes TLS search periods
out past 700 days, exploding the period grid (51,151 periods vs TESS's
low-thousands). Serial processing of 150 stars would be ~27hr; this script
parallelizes TLS across 8 workers (the same ProcessPoolExecutor pattern
this project already uses for its own TLS batches), bringing wall time to
a tractable few hours for the approved pilot scope.

Deliberately does NOT touch the real data/processed/ folder or
data/training_dataset/training.csv -- this is a PILOT, its own isolated
output, until/unless a full build is explicitly approved.
"""
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import pandas as pd

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(PILOT_DIR, "..")
sys.path.insert(0, CODE_DIR)

RAW_DIR = os.path.join(PILOT_DIR, "raw")
PROCESSED_DIR = os.path.join(PILOT_DIR, "processed")
KOI_SAMPLE_PATH = os.path.join(PILOT_DIR, "koi_pilot_sample.csv")
FEATURES_PATH = os.path.join(PILOT_DIR, "pilot_features.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

N_WORKERS = 8


def preprocess_one(host):
    """Reuses 02_preprocess.py's process_one_file UNCHANGED, but redirects
    its output away from the real data/processed/ folder -- this pilot must
    never mix into the real dataset until/unless a full build is approved."""
    import importlib
    m02 = importlib.import_module("02_preprocess")
    raw_path = os.path.join(RAW_DIR, host + ".csv")
    result = m02.process_one_file(raw_path)
    if result["status"] != "Success":
        return result
    real_output_path = os.path.join(m02.OUTPUT_FOLDER, host + ".csv")
    pilot_output_path = os.path.join(PROCESSED_DIR, host + ".csv")
    if os.path.exists(real_output_path):
        os.replace(real_output_path, pilot_output_path)
    return result


def _tls_worker(args):
    """Top-level, picklable -- must import inside the worker since this
    runs in a fresh spawned process (macOS 'spawn' start method)."""
    host, label, st_rad, st_teff, st_mass = args
    import importlib
    m06 = importlib.import_module("06_download_unknown")
    m05 = importlib.import_module("05_train_models")
    processed_path = os.path.join(PROCESSED_DIR, host + ".csv")
    t0 = time.time()
    feats, status = m06.compute_all_features(processed_path, host, st_rad, st_mass, m05.FEATURE_COLUMNS)
    elapsed = time.time() - t0
    if feats is None:
        return {"host": host, "label": label, "status": f"TLS: {status}", "elapsed_s": elapsed}
    r = dict(feats)
    r.update({"host": host, "label": label, "status": "Success",
              "st_rad": st_rad, "st_teff": st_teff, "st_mass": st_mass, "elapsed_s": elapsed})
    return r


def main():
    sample = pd.read_csv(KOI_SAMPLE_PATH)

    # Step 1: preprocess (fast, ~0.2s/star measured) -- done serially, no
    # need to parallelize a sub-second-per-star operation.
    print("Preprocessing (serial, fast)...")
    preprocess_status = {}
    for _, row in sample.iterrows():
        host = f"KIC_{int(row['kepid'])}"
        pilot_output_path = os.path.join(PROCESSED_DIR, host + ".csv")
        if os.path.exists(pilot_output_path):
            preprocess_status[host] = "Success"
            continue
        result = preprocess_one(host)
        preprocess_status[host] = result["status"]
    n_preprocessed_ok = sum(1 for s in preprocess_status.values() if s == "Success")
    print(f"{n_preprocessed_ok}/{len(sample)} preprocessed successfully.")

    # Step 2: TLS feature extraction, parallelized across N_WORKERS -- the
    # expensive step, per the real per-star cost measured above.
    already_done = set()
    results = []
    if os.path.exists(FEATURES_PATH):
        results = pd.read_csv(FEATURES_PATH).to_dict("records")
        already_done = {r["host"] for r in results}
        print(f"{len(already_done)} already have TLS features -- resuming.")

    jobs = []
    for _, row in sample.iterrows():
        host = f"KIC_{int(row['kepid'])}"
        if host in already_done:
            continue
        if preprocess_status.get(host) != "Success":
            results.append({"host": host, "label": int(row["label"]),
                             "status": f"Preprocess failed: {preprocess_status.get(host)}"})
            continue
        jobs.append((host, int(row["label"]), row.get("koi_srad"), row.get("koi_steff"), row.get("koi_smass")))

    print(f"Running TLS on {len(jobs)} stars across {N_WORKERS} workers "
          f"(real measured cost: ~11min/star single-threaded, so expect "
          f"~{len(jobs) * 661.5 / N_WORKERS / 60:.0f} min wall time)...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_tls_worker, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            results.append(res)
            pd.DataFrame(results).to_csv(FEATURES_PATH, index=False)
            elapsed = time.time() - t0
            print(f"  [{i}/{len(jobs)}] {res['host']}: {res['status']} "
                  f"({elapsed:.0f}s elapsed, ~{elapsed/i*(len(jobs)-i):.0f}s remaining)", flush=True)

    df = pd.DataFrame(results)
    n_ok = (df["status"] == "Success").sum()
    print(f"\n{n_ok}/{len(df)} produced usable features.")
    print(df.groupby("label")["status"].apply(lambda s: (s == "Success").sum()))


if __name__ == "__main__":
    main()
