"""
augment_classical_dataset.py -- Part B's deprioritized "expensive path",
now actually run: injects synthetic transit/EB signals into real negative-
class light curves, runs the SAME full TLS feature extraction the classical
model's real training data goes through (06_download_unknown.py's
compute_all_features -- one TLS call, full v2 feature set), and saves the
result as classical-model-compatible training rows (same FEATURE_COLUMNS as
05_train_models.py). Only after this exists can the classical model
honestly be retrained on real+synthetic and compared to the real-only
production model, exactly as Part A's CNN comparison already was.

Every row carries is_synthetic=True, is_synthetic_kind ("transit" or
"eclipsing_binary"), and source_file (which real negative light curve the
signal was injected into) -- so downstream retraining/reporting can always
separate real from synthetic and never blur the two silently.

Reuses the real stellar radius/Teff of the SPECIFIC real star each signal is
injected into (joined from data/training_dataset/training.csv by host), not
a resampled/mismatched value -- a real (rp_rs, R_star) pairing stays
internally consistent for any downstream physical-parameter derivation.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import injection as inj
import importlib
m06 = importlib.import_module("06_download_unknown")

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
TRAINING_CSV = os.path.join(PROJECT_ROOT, "data", "training_dataset", "training.csv")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "augmented_classical_dataset.csv")
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_augment_lc")

# TLS feature columns 05_train_models.py needs (minus st_rad/st_teff, added
# separately per-row below from the real host's own catalog values).
REQUIRED_TLS_COLUMNS = [
    "SDE", "SDE_raw", "FAP", "period", "period_uncertainty", "duration", "depth",
    "depth_mean", "depth_mean_std", "depth_mean_even", "depth_mean_odd",
    "odd_even_mismatch", "rp_rs", "snr", "transit_count", "distinct_transit_count",
    "empty_transit_count", "chi2red_min", "depth_consistency_std",
    "secondary_eclipse_depth", "transit_shape_ratio", "depth_duration_ratio",
]

N_POSITIVE = 800   # synthetic planet-transit examples (label=1)
N_NEGATIVE = 800   # synthetic EB-like examples (label=0)
# Calibrated live on this machine (60-job sample): 15% yield (9/60 passed the
# same all-features-finite bar production candidates must clear), ~3290s
# wall time across 8 workers -- much slower than the original completeness-
# curve's ~14s/job estimate because THIS script (correctly) uses each real
# star's actual stellar radius, and several real negative-class hosts are
# giants (R_star up to ~35 R_sun), which blows up TLS's period/duration
# search grid by orders of magnitude for those stars. At this rate, 1600
# total jobs is ~24hr wall time and an expected ~240 usable rows -- sized to
# the "~1 day, mostly unattended" budget, not a guess.
FIXED_DURATION_FRACTION = None  # duration sampled from real distribution, not fixed


def _load_negative_stellar_params():
    df = pd.read_csv(TRAINING_CSV)
    neg = df[df["label"] == 0][["host", "st_rad", "st_teff"]].dropna()
    return {row["host"]: (row["st_rad"], row["st_teff"]) for _, row in neg.iterrows()}


def _run_one(args):
    """One injection + REAL TLS feature-extraction pass (not just recovery
    check) -- must be top-level/picklable for ProcessPoolExecutor."""
    kind, seed, idx = args
    rng = np.random.default_rng(seed)

    stellar_params = _load_negative_stellar_params()
    fname = inj.list_real_negative_lightcurves()[rng.integers(0, len(inj.list_real_negative_lightcurves()))]
    host = fname.replace(".csv", "")
    time_arr, flux_arr, err_arr = inj.load_real_lightcurve(fname)

    period, depth_ppm, duration = inj.sample_real_params(rng)
    st_rad, st_teff = stellar_params.get(host, (1.0, 5778.0))
    # BUG FIXED: an earlier version of this function passed st_teff (Kelvin,
    # e.g. 3495) into compute_all_features's m_star argument (solar masses)
    # -- caught live via TLS's own "M_star was set to 1000 for period_grid
    # (was unphysical: 3495.0)" warning during a calibration run. st_mass is
    # 100% NaN for every negative-class host in training.csv (the TOI
    # archive table this project's negative class comes from has no
    # stellar-mass column at all -- a genuine data limitation, documented in
    # 04_build_training_dataset.py), so there is no real per-star mass to
    # pass here. NaN is passed explicitly instead of guessing a value --
    # compute_all_features already has its own documented fallback (defaults
    # to 1.0 solar mass when NaN/invalid), the same graceful path a real
    # production candidate with unknown mass goes through.
    st_mass = np.nan

    if kind == "transit":
        injected_flux, params = inj.inject_transit(time_arr, flux_arr, period, depth_ppm, duration, rng)
        label = 1
    else:
        injected_flux, params = inj.inject_eclipsing_binary(time_arr, flux_arr, period, depth_ppm, duration, rng)
        label = 0

    tmp_path = os.path.join(TMP_DIR, f"aug_{kind}_{idx}_{os.getpid()}.csv")
    pd.DataFrame({"time": time_arr, "flux": injected_flux, "flux_err": err_arr}).to_csv(tmp_path, index=False)

    t0 = time.monotonic()
    try:
        feats, status = m06.compute_all_features(tmp_path, host, st_rad, st_mass, REQUIRED_TLS_COLUMNS)
    except Exception as e:
        feats, status = None, f"Exception: {e}"
    elapsed = time.monotonic() - t0
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    if feats is None:
        return {"status": status, "label": label, "is_synthetic": True,
                "synthetic_kind": kind, "source_file": fname, "elapsed_s": elapsed}

    row = dict(feats)
    row.update({
        "label": label, "is_synthetic": True, "synthetic_kind": kind,
        "source_file": fname, "st_rad": st_rad, "st_teff": st_teff,
        "injected_period": period, "injected_depth_ppm": depth_ppm,
        "injected_duration": duration, "status": "Success", "elapsed_s": elapsed,
    })
    return row


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    jobs = [("transit", 5000 + i, i) for i in range(N_POSITIVE)]
    jobs += [("eclipsing_binary", 9000 + i, i) for i in range(N_NEGATIVE)]

    print(f"Running {len(jobs)} injection + full-TLS-feature-extraction trials "
          f"({N_POSITIVE} positive, {N_NEGATIVE} negative) across 8 workers...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_run_one, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            results.append(res)
            if i % 20 == 0 or i == len(jobs):
                elapsed = time.time() - t0
                n_ok_so_far = sum(1 for r in results if r.get("status") == "Success")
                print(f"  [{i}/{len(jobs)}] done ({elapsed:.0f}s elapsed, "
                      f"~{elapsed / i * (len(jobs) - i):.0f}s remaining, "
                      f"{n_ok_so_far}/{i} usable so far)", flush=True)
                # Checkpoint every 20 -- this is a multi-hour unattended run;
                # never lose partial progress to a crash, same pattern as
                # 06_download_unknown.py's own batch checkpointing.
                pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    n_ok = (df["status"] == "Success").sum()
    print(f"\nTotal wall time: {time.time()-t0:.0f}s for {len(jobs)} trials")
    print(f"{n_ok}/{len(jobs)} produced usable feature rows (rest failed TLS/feature "
          f"computation on the synthetic signal -- kept in the CSV with their status "
          f"for transparency, excluded from training).")
    print(f"Saved to {RESULTS_PATH}")

    try:
        os.rmdir(TMP_DIR)
    except OSError:
        pass


if __name__ == "__main__":
    main()
