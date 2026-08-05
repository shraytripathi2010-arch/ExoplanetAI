"""backfill_crowding_columns.py -- add the two crowding columns to training.csv.

PURELY ADDITIVE. This step deliberately does NOT touch FEATURE_COLUMNS, the
model, or anything the running scheduler reads. `build_feature_matrix` selects
only the columns named in FEATURE_COLUMNS, so extra columns in training.csv are
invisible to the live system until that list is edited -- which is a later,
separately-gated step.

Verifies, and refuses to write if any of these fail:
  * row count unchanged
  * every pre-existing column still present, with identical values
  * host order unchanged (the frozen split is keyed to host)
  * the two new columns are the ONLY additions

Missing values: the 33 stars whose nearest catalogued TIC source is >15 arcsec
away could not be resolved to a target and get NaN, which HistGradientBoosting
handles natively. That is distinct from a resolved star with no neighbour in the
aperture, which gets flux_ratio_max = 0.0 -- a real measurement of zero
contaminating flux, not missing data.
"""
import os
import shutil
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CROWD = os.path.join(SCRIPT_DIR, "crowding_per_star.csv")
BACKUP = os.path.join(ROOT, "data", "training_dataset",
                      "training_backup_before_crowding_backfill.csv")

NEW_COLS = ["crowd_flux_ratio_max", "crowd_nearest_arcsec"]


def main():
    before = pd.read_csv(TRAINING)
    crowd = pd.read_csv(CROWD)
    print(f"training.csv BEFORE : {len(before)} rows, {len(before.columns)} columns")
    print(f"crowding source     : {len(crowd)} rows")

    if any(c in before.columns for c in NEW_COLS):
        print("ERROR: crowding columns already present -- refusing to double-apply")
        sys.exit(1)

    merged = before.merge(crowd[["host"] + NEW_COLS], on="host", how="left")

    # ---- integrity gates -------------------------------------------------
    fail = []
    if len(merged) != len(before):
        fail.append(f"row count changed {len(before)} -> {len(merged)}")
    if not merged["host"].equals(before["host"]):
        fail.append("host column/order changed -- frozen split is keyed to host")
    added = [c for c in merged.columns if c not in before.columns]
    if sorted(added) != sorted(NEW_COLS):
        fail.append(f"unexpected column changes: {added}")
    for c in before.columns:
        if not before[c].equals(merged[c]):
            fail.append(f"pre-existing column altered: {c}")
    if fail:
        print("\nINTEGRITY CHECK FAILED -- nothing written:")
        for f in fail:
            print("   -", f)
        sys.exit(1)

    print("\nintegrity checks passed:")
    print(f"  rows unchanged           {len(before)} == {len(merged)}")
    print(f"  host order identical     yes")
    print(f"  pre-existing columns     all {len(before.columns)} byte-identical")
    print(f"  columns added            {added}")

    cov = {c: merged[c].notna().mean() * 100 for c in NEW_COLS}
    print("\ncoverage on the full backfill:")
    for c in NEW_COLS:
        n_missing = int(merged[c].isna().sum())
        print(f"  {c:<26} {cov[c]:5.2f}%  ({n_missing} missing)")
    zero_flux = int((merged["crowd_flux_ratio_max"] == 0).sum())
    print(f"  of which, resolved stars with NO neighbour in aperture "
          f"(flux ratio exactly 0.0): {zero_flux}")

    shutil.copy2(TRAINING, BACKUP)
    merged.to_csv(TRAINING, index=False)
    after = pd.read_csv(TRAINING)
    print(f"\nbackup written : {os.path.relpath(BACKUP, ROOT)}")
    print(f"training.csv AFTER  : {len(after)} rows, {len(after.columns)} columns")
    print("done -- FEATURE_COLUMNS deliberately NOT touched at this stage")


if __name__ == "__main__":
    main()
