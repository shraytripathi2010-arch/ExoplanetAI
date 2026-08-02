"""augment_train_only.py -- synthetic augmentation rows generated ONLY from
train-split source light curves.

WHY THIS EXISTS (a real leakage bug in the original Part B run)

`augment_classical_dataset.py` picks the real light curve to inject into
uniformly at random from `data/processed_negative/`, with no reference to the
train/test split. That was written before the split was frozen. Checked against
the frozen manifest today:

    953 usable synthetic rows
    163 of them (17%) were injected into a star in the HELD-OUT TEST SET
    111 distinct test-split stars were used as injection hosts

Training on those rows leaks the test set. Not the label -- the label is the
injected signal, which is genuinely synthetic -- but everything else about the
star: its actual noise realisation, its systematics, and its real `st_rad` /
`st_teff`, which `_run_one` deliberately (and correctly, for realism) copies
from the host star's own catalog entry. A model that has seen a test star's
noise and stellar parameters during training is not being evaluated cleanly on
that star.

The effect on the original result is unknown in sign: leakage usually flatters
a model, and that run reported synthetic data HURTING (-0.0131), so the true
penalty may be worse rather than better. Either way the number is not
trustworthy, so it is being regenerated rather than reinterpreted.

This script fixes it by construction: the candidate pool of source light curves
is intersected with the train split BEFORE any sampling, so a test-split star
can never be selected. Everything else -- the injectors, the parameter
resampling, the full TLS feature extraction, the all-features-finite bar -- is
byte-for-byte the same code path as the original, imported rather than copied,
so this batch stays comparable to the decontaminated remainder of the old one.

Output goes to a NEW file. The original `augmented_classical_dataset.csv` is
left untouched as the record of what was actually run before.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait as cwait

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

import injection as inj  # noqa: E402

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT_CSV = os.path.join(SCRIPT_DIR, "augmented_train_only.csv")
TMP_DIR = os.path.join(SCRIPT_DIR, "_tmp_augment_train_only")

N_POSITIVE = 900
N_NEGATIVE = 900
N_WORKERS = 8
CHECKPOINT_EVERY = 20
STALL_GAP_S = 1800

# Seeds offset well clear of augment_classical_dataset.py's 5000/9000 bases so
# this batch is independent of the original rather than a partial replay of it.
SEED_BASE_POS = 41000
SEED_BASE_NEG = 77000

REQUIRED_TLS_COLUMNS = [
    "SDE", "SDE_raw", "FAP", "period", "period_uncertainty", "duration", "depth",
    "depth_mean", "depth_mean_std", "depth_mean_even", "depth_mean_odd",
    "odd_even_mismatch", "rp_rs", "snr", "transit_count", "distinct_transit_count",
    "empty_transit_count", "chi2red_min", "depth_consistency_std",
    "secondary_eclipse_depth", "transit_shape_ratio", "depth_duration_ratio",
]

_STELLAR = None
_POOL = None


class _Stalled(Exception):
    pass


def _train_split_negative_files():
    """Source-file pool: negative-class processed light curves whose host is on
    the TRAIN side of the frozen split. Intersected before sampling, so a
    test-split star cannot be drawn at all."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m05
    spec.loader.exec_module(m05)

    df = pd.read_csv(TRAINING_CSV)
    tr, te = m05.split_by_host(df)
    train_hosts = set(df.loc[tr, "host"].astype(str))
    test_hosts = set(df.loc[te, "host"].astype(str))

    available = inj.list_real_negative_lightcurves()
    pool = [f for f in available if f.replace(".csv", "") in train_hosts]
    dropped = [f for f in available if f.replace(".csv", "") in test_hosts]
    print(f"source pool: {len(pool)} train-split light curves "
          f"({len(dropped)} test-split excluded, "
          f"{len(available)-len(pool)-len(dropped)} not in training.csv)")
    if not pool:
        raise SystemExit("no train-split negative light curves found -- refusing "
                         "to fall back to the unfiltered pool, which is the bug "
                         "this script exists to fix")
    return pool


def _stellar_params():
    global _STELLAR
    if _STELLAR is None:
        df = pd.read_csv(TRAINING_CSV)
        neg = df[df["label"] == 0][["host", "st_rad", "st_teff"]].dropna()
        _STELLAR = {r["host"]: (r["st_rad"], r["st_teff"]) for _, r in neg.iterrows()}
    return _STELLAR


def _run_one(args):
    """One injection + full TLS feature extraction. The source file is chosen
    by the PARENT from the train-only pool and passed in, rather than sampled
    inside the worker from the unfiltered directory listing -- that inversion
    is the whole fix."""
    kind, seed, idx, fname = args
    rng = np.random.default_rng(seed)
    host = fname.replace(".csv", "")

    import importlib
    m06 = importlib.import_module("06_download_unknown")

    try:
        time_arr, flux_arr, err_arr = inj.load_real_lightcurve(fname)
        period, depth_ppm, duration = inj.sample_real_params(rng)
        st_rad, st_teff = _stellar_params().get(host, (1.0, 5778.0))
        # st_mass is 100% NaN for negative-class hosts (the TOI archive table
        # has no stellar-mass column). Pass NaN explicitly and let
        # compute_all_features use its own documented 1.0-Msun fallback --
        # the same path a real candidate with unknown mass takes. Do NOT pass
        # st_teff here; an earlier version did, and TLS caught it live with
        # "M_star was set to 1000 (was unphysical: 3495.0)".
        st_mass = np.nan

        if kind == "transit":
            injected_flux, _ = inj.inject_transit(
                time_arr, flux_arr, period, depth_ppm, duration, rng)
            label = 1
        else:
            injected_flux, _ = inj.inject_eclipsing_binary(
                time_arr, flux_arr, period, depth_ppm, duration, rng)
            label = 0

        tmp_path = os.path.join(TMP_DIR, f"aug_{kind}_{idx}_{os.getpid()}.csv")
        pd.DataFrame({"time": time_arr, "flux": injected_flux,
                      "flux_err": err_arr}).to_csv(tmp_path, index=False)

        t0 = time.monotonic()
        try:
            feats, status = m06.compute_all_features(
                tmp_path, host, st_rad, st_mass, REQUIRED_TLS_COLUMNS)
        except Exception as e:
            feats, status = None, f"Exception: {type(e).__name__}: {str(e)[:120]}"
        elapsed = time.monotonic() - t0
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        base = {"label": label, "is_synthetic": True, "synthetic_kind": kind,
                "source_file": fname, "source_split": "train", "elapsed_s": elapsed}
        if feats is None:
            return {**base, "status": status}
        row = dict(feats)
        row.update({**base, "st_rad": st_rad, "st_teff": st_teff,
                    "injected_period": period, "injected_depth_ppm": depth_ppm,
                    "injected_duration": duration, "status": "Success"})
        return row
    except Exception as e:
        return {"status": f"error: {type(e).__name__}: {str(e)[:120]}",
                "label": 1 if kind == "transit" else 0, "is_synthetic": True,
                "synthetic_kind": kind, "source_file": fname, "source_split": "train"}


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    pool = _train_split_negative_files()

    rng = np.random.default_rng(20260802)
    jobs = [("transit", SEED_BASE_POS + i, i, pool[rng.integers(0, len(pool))])
            for i in range(N_POSITIVE)]
    jobs += [("eclipsing_binary", SEED_BASE_NEG + i, i, pool[rng.integers(0, len(pool))])
             for i in range(N_NEGATIVE)]

    rows = []
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        rows = prev.to_dict("records")
        print(f"resuming: {len(rows)} rows already present; "
              f"generating {len(jobs)-len(rows)} more")
        jobs = jobs[len(rows):]
    if not jobs:
        print("nothing to do")
        return

    print(f"{len(jobs)} injection + full-TLS jobs across {N_WORKERS} workers")
    ex = ProcessPoolExecutor(max_workers=N_WORKERS)
    futures = {ex.submit(_run_one, j): j for j in jobs}
    waiting = set(futures)
    i, t0 = 0, time.time()
    stalled = False
    try:
        while waiting:
            got, waiting = cwait(waiting, timeout=STALL_GAP_S,
                                 return_when=FIRST_COMPLETED)
            if not got:
                raise _Stalled()
            for fut in got:
                i += 1
                try:
                    rows.append(fut.result())
                except Exception as e:
                    rows.append({"status": f"crash: {type(e).__name__}: {e}"})
                if i % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                    el = time.time() - t0
                    n_ok = sum(1 for r in rows if r.get("status") == "Success")
                    print(f"  [{i}/{len(jobs)}] {el:.0f}s elapsed, "
                          f"~{el/i*(len(jobs)-i):.0f}s left, {n_ok} usable",
                          flush=True)
    except _Stalled:
        stalled = True
        print(f"STALLED: nothing completed in {STALL_GAP_S}s -- saving partial")
    finally:
        # Save BEFORE teardown. shutdown() sets _processes to None, so reading
        # it afterwards raises inside finally and skips the save -- exactly the
        # bug that destroyed a completed centroid run earlier in this project.
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        procs = list((getattr(ex, "_processes", None) or {}).values())
        ex.shutdown(wait=False, cancel_futures=True)
        if stalled:
            for p in procs:
                try:
                    p.terminate()
                except Exception:
                    pass

    df = pd.DataFrame(rows)
    n_ok = int((df["status"] == "Success").sum())
    print(f"\ndone: {len(df)} attempted, {n_ok} usable ({100*n_ok/len(df):.1f}%)")
    if n_ok:
        okd = df[df["status"] == "Success"]
        print(f"  positive/transit {int((okd['label']==1).sum())} | "
              f"negative/EB {int((okd['label']==0).sum())}")
    print(df["status"].value_counts().head(6).to_string())
    print(f"Saved to {OUT_CSV}")
    try:
        os.rmdir(TMP_DIR)
    except OSError:
        pass


if __name__ == "__main__":
    main()
