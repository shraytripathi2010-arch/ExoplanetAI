"""
build_cnn_dataset.py -- builds the CNN's local+global view training set:
  1. REAL examples: every star in training.csv, folded at its own real
     period/T0/duration (already known -- no TLS re-run needed).
  2. SYNTHETIC examples: injected transits (positive) and injected
     eclipsing-binary-like signals (negative), folded at the KNOWN injected
     period/T0 -- also no TLS search needed, which is what makes this path
     cheap (see the Part B feasibility report).

Every row carries is_synthetic explicitly so real and synthetic results can
always be reported separately, never silently blended.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import injection as inj
import phase_fold_views as pfv

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
TRAINING_CSV = os.path.join(PROJECT_ROOT, "data", "training_dataset", "training.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PROCESSED_NEG_DIR = os.path.join(PROJECT_ROOT, "data", "processed_negative")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_dataset.npz")

N_SYNTHETIC_POSITIVE = 2000
N_SYNTHETIC_NEGATIVE = 2000
RANDOM_SEED = 42


def _real_lightcurve_path(host):
    p1 = os.path.join(PROCESSED_DIR, f"{host}.csv")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(PROCESSED_NEG_DIR, f"{host}.csv")
    if os.path.exists(p2):
        return p2
    return None


def build_real_examples():
    df = pd.read_csv(TRAINING_CSV)
    globals_, locals_, labels, hosts, synth_flags = [], [], [], [], []
    n_missing = 0
    for _, row in df.iterrows():
        path = _real_lightcurve_path(row["host"])
        if path is None:
            n_missing += 1
            continue
        if pd.isna(row["period"]) or pd.isna(row["T0"]) or pd.isna(row["duration"]) or row["period"] <= 0:
            n_missing += 1
            continue
        lc = pd.read_csv(path)
        try:
            g, l = pfv.make_views(lc["time"].to_numpy(), lc["flux"].to_numpy(),
                                   row["period"], row["T0"], row["duration"])
        except Exception:
            n_missing += 1
            continue
        globals_.append(g)
        locals_.append(l)
        labels.append(int(row["label"]))
        hosts.append(row["host"])
        synth_flags.append(False)
    print(f"Real examples built: {len(labels)} ({n_missing} skipped -- missing file or params)")
    return globals_, locals_, labels, hosts, synth_flags


def build_synthetic_examples():
    rng = np.random.default_rng(RANDOM_SEED)
    neg_files = inj.list_real_negative_lightcurves()
    globals_, locals_, labels, hosts, synth_flags = [], [], [], [], []

    for i in range(N_SYNTHETIC_POSITIVE):
        fname = neg_files[rng.integers(0, len(neg_files))]
        time_arr, flux_arr, _ = inj.load_real_lightcurve(fname)
        period, depth_ppm, duration = inj.sample_real_params(rng)
        injected_flux, params = inj.inject_transit(time_arr, flux_arr, period, depth_ppm, duration, rng)
        g, l = pfv.make_views(time_arr, injected_flux, params["period_days"], params["t0"], params["duration_days"])
        globals_.append(g); locals_.append(l); labels.append(1)
        hosts.append(f"SYNTH_POS_{i}_{fname}"); synth_flags.append(True)
        if (i + 1) % 500 == 0:
            print(f"  synthetic positive {i+1}/{N_SYNTHETIC_POSITIVE}")

    for i in range(N_SYNTHETIC_NEGATIVE):
        fname = neg_files[rng.integers(0, len(neg_files))]
        time_arr, flux_arr, _ = inj.load_real_lightcurve(fname)
        period, depth_ppm, duration = inj.sample_real_params(rng)
        # EB-like negatives skew toward deeper, longer-duration signals than
        # typical planets -- real EB false positives are usually not subtle.
        depth_ppm = float(depth_ppm * rng.uniform(2, 8))
        injected_flux, params = inj.inject_eclipsing_binary(time_arr, flux_arr, period, depth_ppm, duration, rng)
        g, l = pfv.make_views(time_arr, injected_flux, params["period_days"], params["t0"], params["duration_days"])
        globals_.append(g); locals_.append(l); labels.append(0)
        hosts.append(f"SYNTH_NEG_{i}_{fname}"); synth_flags.append(True)
        if (i + 1) % 500 == 0:
            print(f"  synthetic negative {i+1}/{N_SYNTHETIC_NEGATIVE}")

    return globals_, locals_, labels, hosts, synth_flags


def main():
    rg, rl, rlab, rhost, rsynth = build_real_examples()
    sg, sl, slab, shost, ssynth = build_synthetic_examples()

    globals_arr = np.stack(rg + sg).astype(np.float32)
    locals_arr = np.stack(rl + sl).astype(np.float32)
    labels_arr = np.array(rlab + slab, dtype=np.int64)
    hosts_arr = np.array(rhost + shost)
    synth_arr = np.array(rsynth + ssynth, dtype=bool)

    np.savez_compressed(OUT_PATH, global_view=globals_arr, local_view=locals_arr,
                         label=labels_arr, host=hosts_arr, is_synthetic=synth_arr)
    print(f"\nSaved {len(labels_arr)} total examples to {OUT_PATH}")
    print(f"  Real: {(~synth_arr).sum()} (pos={labels_arr[~synth_arr].sum()}, neg={(~synth_arr).sum() - labels_arr[~synth_arr].sum()})")
    print(f"  Synthetic: {synth_arr.sum()} (pos={labels_arr[synth_arr].sum()}, neg={synth_arr.sum() - labels_arr[synth_arr].sum()})")


if __name__ == "__main__":
    main()
