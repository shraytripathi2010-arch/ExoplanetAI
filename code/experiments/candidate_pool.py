"""candidate_pool.py -- the scored candidate pool, rebuilt from the cumulative
files instead of read from ranked_candidates.csv.

ranked_candidates.csv is rewritten by every 06 run with only that run's stars:
it went 254 -> 44 on 2026-08-29 (run 32), and any analysis reading it as "the
pool" then silently describes a biased 44-star slice
(audit/cumulative_read_traps.md).

This applies 06's own selection rule (score_candidates in
06_download_unknown.py) to the files 06 selects FROM:

  data/catalogs/unknown_features{tag}.csv        append-only across runs; the
                                                  same source, and the same
                                                  host dedupe, as
                                                  web/backfill_uncertainty.py
  data/catalogs/unknown_candidate_list{tag}.csv  st_rad/st_teff, and ra/dec
                                                  for the main pool (the
                                                  widesector list has none)

A row is in the pool when its status starts with "Success" and no model
feature outside 06's OPTIONAL_FEATURES is NaN -- the rows 06 scores against the
full candidate list. Checked against the last full ranked exports (6311b244):
widesector 54/54 identical; main 253 vs 254, the one difference
(TIC_207109256) has no crowd_nearest_arcsec and was ranked before crowding
became a required feature, so 06 would exclude it today.
"""
import ast
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
CATALOGS = os.path.join(ROOT, "data", "catalogs")
META_PATH = os.path.join(ROOT, "models", "best_model_metadata.json")
M06_PATH = os.path.join(ROOT, "code", "06_download_unknown.py")

POOLS = {
    "main": ("unknown_features.csv", "unknown_candidate_list.csv"),
    "widesector": ("unknown_features_widesector.csv", "unknown_candidate_list_widesector.csv"),
}


def _optional_features():
    """06's OPTIONAL_FEATURES, read from its source rather than copied (importing
    06 costs ~4 s and rebinds sys.stdout)."""
    tree = ast.parse(open(M06_PATH).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "OPTIONAL_FEATURES" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise KeyError(f"OPTIONAL_FEATURES not found in {M06_PATH}")


def load_scored_pool(pool):
    """Returns (frame, provenance). provenance records the row count and the
    files read, for the caller to store in its own output."""
    feat_name, list_name = POOLS[pool]
    feat_path = os.path.join(CATALOGS, feat_name)
    list_path = os.path.join(CATALOGS, list_name)
    with open(META_PATH) as f:
        feature_columns = json.load(f)["feature_columns"]
    blocking = [c for c in feature_columns if c not in _optional_features()]

    feats = pd.read_csv(feat_path)
    feats["host"] = feats["host"].astype(str)
    feats = feats.drop_duplicates(subset="host", keep="last")
    feats = feats[feats["status"].astype(str).str.startswith("Success")]

    cand = pd.read_csv(list_path)
    cand["host"] = "TIC_" + cand["tic_id"].astype("int64").astype(str)
    cand = cand.drop_duplicates(subset="host", keep="last")
    extra = [c for c in ("st_rad", "st_teff", "ra", "dec") if c in cand.columns]

    d = feats.drop(columns=[c for c in extra if c in feats.columns]).merge(
        cand[["host"] + extra], on="host", how="left")
    absent = [c for c in blocking if c not in d.columns]
    if absent:
        raise KeyError(f"{pool}: required feature column(s) {absent} are in neither "
                       f"{feat_name} nor {list_name}")
    d = d[d[blocking].notna().all(axis=1)].reset_index(drop=True)

    prov = {"pool": pool, "n_rows": int(len(d)),
            "sources": [os.path.relpath(feat_path, ROOT), os.path.relpath(list_path, ROOT)],
            "rule": "status Success, no NaN in a non-optional model feature (06 score_candidates)"}
    return d, prov
