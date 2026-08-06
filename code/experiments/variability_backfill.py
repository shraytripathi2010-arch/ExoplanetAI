"""variability_backfill.py -- write the five variability columns into
training.csv and both candidate pools, with integrity verification.

RAW LIGHT CURVES, NOT PROCESSED. This is the one feature family in this
pipeline that does NOT read data/processed/. `02_preprocess.py` savgol-flattens
with a window capped at 401 points (~13.4 h), which is a high-pass filter and
removes exactly the multi-day rotational signal these features measure. Reading
the processed files would silently produce the detrending residual instead of
the star's variability, with no error. Any future change to the preprocessing
stage must NOT be assumed to apply here.

INTEGRITY. training.csv is the input to every retrain, so this refuses to write
unless:
  * row count is unchanged
  * host column is unchanged, in order, element by element
  * every pre-existing column is byte-identical to what was there before
Only then are the five new columns appended. A backup is written first.
"""
import os
import sys
import shutil
import hashlib
import importlib.util
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
VFEAT = os.path.join(SCRIPT_DIR, "stellar_variability_features.csv")
POOLS = [
    (os.path.join(ROOT, "data", "catalogs", "unknown_features.csv"),
     [os.path.join(ROOT, "data", "unknown_lightcurves")]),
    (os.path.join(ROOT, "data", "catalogs", "unknown_features_widesector.csv"),
     [os.path.join(ROOT, "data", "unknown_lightcurves_widesector")]),
]
VAR5 = ["var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]


def _sv():
    spec = importlib.util.spec_from_file_location(
        "sv", os.path.join(SCRIPT_DIR, "stellar_variability.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["sv"] = m
    spec.loader.exec_module(m)
    return m


def col_digest(df, cols):
    h = hashlib.md5()
    for c in cols:
        h.update(df[c].to_csv(index=False).encode())
    return h.hexdigest()


def backfill_training():
    print("=" * 70)
    print("TRAINING SET")
    print("=" * 70)
    t = pd.read_csv(TRAINING)
    v = pd.read_csv(VFEAT)
    orig_cols = list(t.columns)
    orig_rows = len(t)
    orig_hosts = t["host"].astype(str).tolist()
    before = col_digest(t, orig_cols)

    already = [c for c in VAR5 if c in t.columns]
    if already:
        print(f"  already present, nothing to do: {already}")
        return

    assert len(v) == orig_rows, f"row mismatch: {len(v)} vs {orig_rows}"
    assert v["host"].astype(str).tolist() == orig_hosts, "host order differs"

    for c in VAR5:
        t[c] = pd.to_numeric(v[c], errors="coerce").to_numpy()

    assert len(t) == orig_rows, "row count changed"
    assert t["host"].astype(str).tolist() == orig_hosts, "host order changed"
    assert col_digest(t, orig_cols) == before, "a pre-existing column changed"
    assert list(t.columns) == orig_cols + VAR5, "column layout unexpected"

    bak = TRAINING + ".pre_variability.bak"
    if not os.path.exists(bak):
        shutil.copy2(TRAINING, bak)
    t.to_csv(TRAINING, index=False)
    print(f"  rows {orig_rows} unchanged, hosts identical, "
          f"{len(orig_cols)} existing columns byte-identical")
    print(f"  appended {len(VAR5)} columns -> {len(t.columns)} total")
    for c in VAR5:
        print(f"    {c:15s} non-null {t[c].notna().sum()}/{len(t)} "
              f"({t[c].notna().mean()*100:.1f}%)")
    print(f"  backup: {os.path.relpath(bak, ROOT)}")


def backfill_pool(path, raw_dirs):
    print("\n" + "=" * 70)
    print(f"POOL {os.path.basename(path)}")
    print("=" * 70)
    d = pd.read_csv(path)
    orig_cols = list(d.columns)
    orig_rows = len(d)
    orig_hosts = d["host"].astype(str).tolist()
    before = col_digest(d, orig_cols)

    if all(c in d.columns for c in VAR5):
        print("  already present, nothing to do")
        return

    # Use the PRODUCTION module with an explicit per-job raw path. An earlier
    # draft set a module-level RAW_DIRS from the parent; under `spawn` the
    # workers re-imported the module, reverted to the default directory, and
    # the pool broke. The path now travels with the job, so that is unreachable.
    sys.path.insert(0, SCRIPT_DIR)
    import variability_features as vf
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(r["host"], vf.find_raw(r["host"], raw_dirs),
             pd.to_numeric(r.get("period"), errors="coerce"),
             pd.to_numeric(r.get("T0"), errors="coerce"),
             pd.to_numeric(r.get("duration"), errors="coerce"))
            for _, r in d.iterrows()]
    missing = sum(1 for j in jobs if j[1] is None)
    print(f"  {len(jobs)} rows, raw light curve found for {len(jobs)-missing}, "
          f"missing {missing}")
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, res in enumerate(ex.map(vf.worker, jobs, chunksize=8), 1):
            rows.append(res)
            if i % 500 == 0:
                print(f"    {i}/{len(jobs)}", flush=True)
    out = pd.DataFrame(rows)
    assert out["host"].astype(str).tolist() == orig_hosts, "pool host order drifted"

    for c in VAR5:
        d[c] = pd.to_numeric(out[c], errors="coerce").to_numpy()

    assert len(d) == orig_rows, "row count changed"
    assert col_digest(d, orig_cols) == before, "a pre-existing column changed"

    bak = path + ".pre_variability.bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    d.to_csv(path, index=False)
    print(f"  rows {orig_rows} unchanged, {len(orig_cols)} existing columns byte-identical")
    for c in VAR5:
        print(f"    {c:15s} non-null {d[c].notna().sum()}/{len(d)} "
              f"({d[c].notna().mean()*100:.1f}%)")
    # the rows that actually get scored are those with a real detection
    P = pd.to_numeric(d.get("period"), errors="coerce")
    T = pd.to_numeric(d.get("duration"), errors="coerce")
    det = (P > 0) & (T > 0)
    if det.any():
        cov = d.loc[det, "var_excess"].notna().mean() * 100
        print(f"  scorable rows (period&duration finite): {int(det.sum())}; "
              f"var_excess present on {cov:.1f}% of them")


def main():
    backfill_training()
    for p, dirs in POOLS:
        backfill_pool(p, dirs)
    print("\nBACKFILL COMPLETE")


if __name__ == "__main__":
    main()
