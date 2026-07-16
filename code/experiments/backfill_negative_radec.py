"""
backfill_negative_radec.py -- fixes the real gap found while scoping
centroid-as-feature work: negative-class training rows (1,155 TIC-ID-keyed
TOI false positives) are missing ra/dec 98.8% of the time (vs 0.14% for
positive class), purely because 04_build_training_dataset.py never joined
it in for that class. Left uncorrected, this would make "has centroid data
at all" almost perfectly predict the label -- a leakage trap manufactured
by a pipeline gap, not real astrophysics.

Batch-queries the TIC catalog (same 500-per-chunk pattern already used in
06_download_unknown.py's fetch_stellar_params) and writes ra/dec back into
training.csv IN PLACE for negative-class rows only -- this is a genuine
data-completeness fix, not a new column; positive class is untouched.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from tqdm import tqdm

TRAINING_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                             "data", "training_dataset", "training.csv")


def main():
    from astroquery.mast import Catalogs

    df = pd.read_csv(TRAINING_CSV)
    neg = df[df["label"] == 0].copy()
    missing_mask = neg["ra"].isna()
    tic_ids = neg.loc[missing_mask, "host"].str.replace("TIC_", "", regex=False).astype("int64").tolist()
    print(f"{len(tic_ids)} negative-class rows missing ra/dec -- querying TIC catalog live...")

    chunks = [tic_ids[i:i + 500] for i in range(0, len(tic_ids), 500)]
    rows = []
    for chunk in tqdm(chunks, desc="Querying TIC catalog"):
        r = Catalogs.query_criteria(catalog="Tic", ID=chunk)
        rows.append(r[["ID", "ra", "dec"]].to_pandas())
    stellar = pd.concat(rows, ignore_index=True)
    stellar["ID"] = stellar["ID"].astype("int64")
    stellar = stellar.drop_duplicates(subset="ID")
    radec_map = {row["ID"]: (row["ra"], row["dec"]) for _, row in stellar.iterrows()}

    n_filled = 0
    for idx, row in df[df["label"] == 0].iterrows():
        if pd.isna(row["ra"]):
            tic_id = int(row["host"].replace("TIC_", ""))
            if tic_id in radec_map:
                ra, dec = radec_map[tic_id]
                df.at[idx, "ra"] = ra
                df.at[idx, "dec"] = dec
                n_filled += 1

    print(f"Backfilled ra/dec for {n_filled}/{len(tic_ids)} negative-class rows "
          f"({len(tic_ids) - n_filled} not found in TIC catalog).")

    n_neg_total = (df["label"] == 0).sum()
    n_neg_still_missing = df[df["label"] == 0]["ra"].isna().sum()
    print(f"Negative-class ra/dec coverage now: {n_neg_total - n_neg_still_missing}/{n_neg_total} "
          f"({100*(1 - n_neg_still_missing/n_neg_total):.1f}%)")

    df.to_csv(TRAINING_CSV, index=False)
    print(f"Saved to {TRAINING_CSV}")


if __name__ == "__main__":
    main()
