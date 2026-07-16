"""
04_build_training_dataset.py

Merge per-star TLS features + preprocessing QC stats + catalog metadata into
one labeled table for ML training.

IMPORTANT ASSUMPTIONS -- read before using the output downstream:

1. ONE ROW PER STAR, NOT ONE ROW PER PLANET. 1,054 stars in the catalog have
   multiple confirmed planets (e.g. TRAPPIST-1 has 7). TLS run on the full
   light curve finds one dominant periodic signal, not one row per known
   planet, so multi-planet hosts collapse to a single row here. If you later
   need one row per planet, this script does not do that -- you'd have to
   duplicate each host's row per pl_name, and all duplicated rows would share
   identical TLS features (since TLS was only run once per star).

2. STELLAR PARAMETERS: st_rad/st_teff are joined for BOTH classes (positive
   from confirmed_planets.csv, negative from toi_false_positives.csv -- the
   negative-class join was a found-and-fixed bug: the data was fetched back
   in 01_download_negative.py but never actually wired in until now). Each
   has its own NaN rate from the archive itself (~5%), independent of this
   pipeline. st_mass IS positive-class only, but that's a genuine data
   limitation, not a bug: the TOI archive table has no stellar mass column
   at all (checked live against the archive).

3. NEGATIVE-CLASS QC STATS ARE NOT MERGED. Only one preprocess_qc_summary.csv
   exists (positive-class only, from 02_preprocess.py). If/when a
   negative-class preprocessing run produces its own QC file, this script
   needs a matching --negative-qc argument added -- not built yet since no
   such file exists to test against and it wasn't part of the original ask.

4. HOST NAME JOINS USE A CANONICAL KEY, NOT THE RAW STRING. Verified against
   the real data that a naive `hostname == host` join only matches 75% of
   stars (1,552 of 6,316 catalog hostnames contain spaces/special characters
   that got sanitized differently -- and inconsistently -- across this
   project's history when they became filenames). canonical_key() strips
   everything but lowercase alphanumerics on both sides, which gave 100%
   match coverage with zero key collisions when checked against the real
   dataset.

5. THIS DATASET HAS DUPLICATE STARS UNDER DIFFERENT FILENAMES. 24 stars in
   transit_search_results.csv are the same physical star processed twice
   under two different sanitized names (e.g. "BD+14_4559" and "BD14_4559").
   Verified these are near-identical duplicate computations (same n_points,
   SDE, period), not two different observations. load_tls_results() detects
   and deduplicates these by canonical key, and PRINTS which ones it merged
   -- never silently.

Usage:
    python3 04_build_training_dataset.py
    python3 04_build_training_dataset.py --negative-results path/to/negative_results.csv
"""

import argparse
import os
import re
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "catalogs")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "training_dataset")

RESULTS_PATH = os.path.join(CATALOG_FOLDER, "transit_search_results.csv")
QC_PATH = os.path.join(CATALOG_FOLDER, "preprocess_qc_summary.csv")
CATALOG_PATH = os.path.join(CATALOG_FOLDER, "confirmed_planets.csv")
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "training.csv")

# TLS feature columns carried into the final table. Deliberately excludes
# elapsed_s from transit_search_results.csv -- that's runtime bookkeeping
# metadata from stage 3, not a real astrophysical feature.
TLS_FEATURE_COLUMNS = [
    "n_points", "SDE", "SDE_raw", "FAP", "period", "period_uncertainty", "T0",
    "duration", "depth", "depth_mean", "depth_mean_std", "depth_mean_even",
    "depth_mean_odd", "odd_even_mismatch", "rp_rs", "snr", "transit_count",
    "distinct_transit_count", "empty_transit_count",
]

REQUIRED_CATALOG_STELLAR_COLUMNS = ["st_rad", "st_teff", "st_mass"]


def canonical_key(name):
    """Normalize a star name for joining across files. See module docstring
    point 4 -- this is not a style choice, it's required for correctness."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_tls_results(path, label):
    """Load one TLS results CSV (stage-3 schema) and tag it with a label.
    Detects and deduplicates same-star-different-filename rows -- see
    module docstring point 5."""
    df = pd.read_csv(path)
    if "host" not in df.columns:
        sys.exit(f"ERROR: {path} has no 'host' column -- is this really a TLS results file?")

    df["canon"] = df["host"].apply(canonical_key)

    dupe_mask = df.duplicated(subset="canon", keep=False)
    if dupe_mask.any():
        dupes = df.loc[dupe_mask, ["host", "canon"]].sort_values("canon")
        print(f"WARNING: {dupe_mask.sum()} rows in {os.path.basename(path)} are the same star "
              f"under different sanitized filenames -- keeping one row per star:")
        for canon, group in dupes.groupby("canon"):
            kept = sorted(group["host"])[0]
            print(f"  {canon}: {group['host'].tolist()} -- keeping '{kept}'")
        df = df.sort_values("host").drop_duplicates(subset="canon", keep="first")

    df["label"] = label
    return df


def _first_non_null(series):
    """first-non-null rather than plain 'first': stellar params (st_rad etc.)
    describe the host star, not the individual planet, so if one planet's row
    in the archive happens to have a NaN for it but a sibling planet row for
    the SAME star has a real value, we shouldn't throw that value away."""
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) > 0 else float("nan")


def load_catalog(path):
    """Load the confirmed-planets catalog and collapse multi-planet hosts to
    one row per star. See module docstring points 1 and 2."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: catalog file not found: {path}")
    cat = pd.read_csv(path)

    missing_stellar = [c for c in REQUIRED_CATALOG_STELLAR_COLUMNS if c not in cat.columns]
    if missing_stellar:
        print(f"NOTE: {path} is missing {missing_stellar} -- proceeding WITHOUT stellar "
              f"parameters in this training set. Re-query the NASA Exoplanet Archive to add "
              f"them, then extend load_catalog() below, when ready.")
    stellar_cols = [c for c in REQUIRED_CATALOG_STELLAR_COLUMNS if c in cat.columns]

    cat["canon"] = cat["hostname"].apply(canonical_key)

    # ra/dec should be identical across planets of the same host, so plain
    # "first" is fine there. pl_name is aggregated into a semicolon-joined
    # list so the multi-planet information isn't silently lost, just
    # collapsed to one row. Stellar params use _first_non_null (see above).
    agg_dict = {"hostname": "first", "ra": "first", "dec": "first"}
    agg_dict.update({c: _first_non_null for c in stellar_cols})
    grouped = cat.groupby("canon").agg(agg_dict)
    grouped["pl_names"] = cat.groupby("canon")["pl_name"].apply(lambda s: "; ".join(sorted(s)))
    grouped["n_planets_in_catalog"] = cat.groupby("canon")["pl_name"].nunique()
    return grouped.reset_index(), stellar_cols


def load_negative_stellar_params(path):
    """Stellar params for the negative (TOI false-positive) class. These were
    already fetched in 01_download_negative.py's TOI query (st_rad, st_teff)
    but never joined in here -- found and fixed after the fact, since the
    data existed the whole time, just unused. NOTE: the TOI archive table has
    no stellar MASS column at all (checked live against the archive), so
    st_mass stays positive-class-only -- a genuine data limitation, not
    something this join can fix.

    Negative-class hosts are filenamed "TIC_<tid>" (see 01_download_negative.py),
    so we rebuild that same identifier from the TOI table's tid column and run
    it through the same canonical_key() used everywhere else in this script."""
    if not os.path.exists(path):
        print(f"NOTE: {path} not found -- negative-class rows will have no stellar params.")
        return pd.DataFrame(columns=["canon", "st_rad", "st_teff"])

    toi = pd.read_csv(path)
    toi["host"] = "TIC_" + toi["tid"].astype("int64").astype(str)
    toi["canon"] = toi["host"].apply(canonical_key)

    available = [c for c in ["st_rad", "st_teff"] if c in toi.columns]
    if not available:
        return pd.DataFrame(columns=["canon", "st_rad", "st_teff"])

    agg_dict = {c: _first_non_null for c in available}
    grouped = toi.groupby("canon").agg(agg_dict)
    return grouped.reset_index()


def main():
    parser = argparse.ArgumentParser(
        description="Merge TLS features + QC stats + catalog metadata into one labeled training table."
    )
    parser.add_argument(
        "--negative-results", default=None,
        help="Path to a TLS results CSV (same schema as transit_search_results.csv) for "
             "label=0 (TOI false-positive) stars. Omit if you don't have a negative class yet."
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ---- load positive class (required) ----
    if not os.path.exists(RESULTS_PATH):
        sys.exit(f"ERROR: {RESULTS_PATH} not found -- run 03_transit_search.py first.")
    positive = load_tls_results(RESULTS_PATH, label=1)
    print(f"Loaded {len(positive)} positive-class (confirmed planet host) stars from {RESULTS_PATH}")

    # ---- load negative class (optional) ----
    frames = [positive]
    if args.negative_results:
        if not os.path.exists(args.negative_results):
            sys.exit(f"ERROR: --negative-results path not found: {args.negative_results}")
        negative = load_tls_results(args.negative_results, label=0)
        print(f"Loaded {len(negative)} negative-class (false positive) stars from {args.negative_results}")

        overlap = set(positive["canon"]) & set(negative["canon"])
        if overlap:
            print(f"WARNING: {len(overlap)} stars appear in BOTH positive and negative sets -- "
                  f"keeping the positive-class label for these (a confirmed planet host takes "
                  f"precedence over a false-positive TOI disposition for the same star): "
                  f"{sorted(overlap)}")
            negative = negative[~negative["canon"].isin(overlap)]

        frames.append(negative)
    else:
        print("No --negative-results provided -- building a positive-class-only dataset (label=1 for all rows).")

    tls_df = pd.concat(frames, ignore_index=True)

    # ---- join catalog metadata (pl_names/n_planets_in_catalog are inherently
    # positive-class only -- TOI false positives were never confirmed planets,
    # so there's nothing to join there. ra/dec/stellar params, however, DO
    # exist for the negative class too, from its own TOI catalog -- joined in
    # as a second step below rather than left NaN.) ----
    catalog, stellar_cols = load_catalog(CATALOG_PATH)
    catalog_join_cols = ["canon", "pl_names", "n_planets_in_catalog", "ra", "dec"] + stellar_cols
    merged = tls_df.merge(catalog[catalog_join_cols], on="canon", how="left")

    # ---- fill in negative-class ra/dec/stellar params from the TOI catalog.
    # These were fetched back in 01_download_negative.py but never joined in
    # until now -- the data existed the whole time, this was a bug, not a
    # missing-data limitation (st_mass is the one genuine exception: the TOI
    # archive table has no stellar mass column at all, checked live). ----
    # Stellar params for the negative class can come from more than one TOI
    # disposition catalog now (FP, and optionally FA if that expansion has been
    # run) -- merge whichever catalogs exist, FP first then filling any gaps
    # from FA, since a given TIC ID only appears in one disposition's catalog.
    neg_catalog_paths = [
        os.path.join(CATALOG_FOLDER, "toi_false_positives.csv"),
        os.path.join(CATALOG_FOLDER, "toi_false_alarms.csv"),
    ]
    for neg_catalog_path in neg_catalog_paths:
        neg_stellar = load_negative_stellar_params(neg_catalog_path)
        if len(neg_stellar) > 0:
            merged = merged.merge(neg_stellar, on="canon", how="left", suffixes=("", "_neg"))
            for col in ["ra", "dec", "st_rad", "st_teff"]:
                neg_col = f"{col}_neg"
                if neg_col in merged.columns:
                    merged[col] = merged[col].fillna(merged[neg_col])
                    merged = merged.drop(columns=[neg_col])
    n_filled = merged.loc[merged["label"] == 0, "st_rad"].notna().sum()
    print(f"Filled in st_rad/st_teff for {n_filled} negative-class stars from the TOI catalog(s) "
          f"(previously all NaN due to a join gap -- the data existed, it just wasn't used).")

    # ---- join preprocessing QC stats ----
    if not os.path.exists(QC_PATH):
        sys.exit(f"ERROR: {QC_PATH} not found -- run 02_preprocess.py first.")
    qc = pd.read_csv(QC_PATH)
    qc["canon"] = qc["host"].apply(canonical_key)
    if qc.duplicated(subset="canon", keep=False).any():
        qc = qc.sort_values("host").drop_duplicates(subset="canon", keep="first")
    qc = qc.rename(columns={"pct_removed": "points_removed_pct"})
    qc_cols = ["canon", "n_original", "n_after_quality", "n_after_outliers", "n_final",
               "points_removed_pct", "pre_norm_median_flux", "flux_source"]
    merged = merged.merge(qc[qc_cols], on="canon", how="left")

    # ---- final column selection ----
    final_cols = (
        ["host", "label"] + TLS_FEATURE_COLUMNS +
        ["pl_names", "n_planets_in_catalog", "ra", "dec"] + stellar_cols +
        ["n_original", "n_after_quality", "n_after_outliers", "n_final",
         "points_removed_pct", "pre_norm_median_flux", "flux_source"]
    )
    final_cols = [c for c in final_cols if c in merged.columns]
    final = merged[final_cols].reset_index(drop=True)

    # =====================================
    # SANITY CHECKS -- printed clearly, nothing silently dropped or fixed
    # =====================================
    print("\n===================================")
    print("Sanity checks")
    print("===================================")

    n_negative = len(tls_df) - len(positive)
    print(f"Row count: {len(final)} final rows "
          f"({len(positive)} positive + {n_negative} negative TLS results, post-dedup)")

    # Scoped to genuine feature columns -- pl_names/ra/dec/n_planets_in_catalog
    # are EXPECTED to be NaN for negative-class rows (no catalog match), so
    # they're deliberately excluded from this check.
    feature_cols = [c for c in TLS_FEATURE_COLUMNS + [
        "points_removed_pct", "n_original", "n_after_quality", "n_after_outliers",
        "n_final", "pre_norm_median_flux",
    ] if c in final.columns]
    nan_report = final[feature_cols].isna().sum()
    nan_report = nan_report[nan_report > 0]
    if len(nan_report) > 0:
        print(f"WARNING: NaN values found in feature columns:\n{nan_report}")
        affected = final.loc[final[feature_cols].isna().any(axis=1), "host"].tolist()
        print(f"Affected stars ({len(affected)}): {affected}")
    else:
        print("No NaN values in feature columns. OK")

    dupe_count = final["host"].duplicated().sum()
    print(f"Duplicate star rows: {dupe_count} "
          f"{'OK' if dupe_count == 0 else '-- WARNING: dedup logic may have a gap, investigate before training'}")

    label_values = set(final["label"].unique())
    if label_values <= {0, 1}:
        print(f"Label column OK -- values present: {sorted(label_values)}")
    else:
        print(f"WARNING: label column has unexpected values: {label_values}")

    print(f"\nSaving to {OUTPUT_PATH}")
    final.to_csv(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
