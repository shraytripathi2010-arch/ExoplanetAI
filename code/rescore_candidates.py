"""rescore_candidates.py -- score every characterized candidate with the current
production model, in place, and recompute tiers and the two derived lists.

Needs the migrated (canonical, no _dup) results files; 08's refuse_dup_columns
rejects anything else. Features are rebuilt exactly as 06's score_candidates
builds them: unknown_features.csv (latest row per host, status Success) plus
st_rad/st_teff/st_mass from unknown_candidate_list.csv. A star missing from
today's candidate list (it is rewritten per run) takes st_rad/st_teff from
characterized_candidates.csv -- the values 06 scored it with -- licensed only by
gate 1b. A star with a NaN in a required feature is NOT scored (06's own rule):
it keeps its old probability and provenance, so the app labels it unscored.

GATES -- any failure exits non-zero before anything is written:
  0   models/best_model.joblib must be fe3fa82f (the bytes scored are the bytes hashed)
  1   the 29 probabilities already stamped fe3fa82f must be reproduced to 1e-9
  1b  st_rad/st_teff present in BOTH the candidate list and the characterized
      file must be identical
  2   final_best and needs-review membership must not change

Usage: python3 code/rescore_candidates.py
"""
import hashlib
import importlib.util
import io
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results", "unknown_candidates")
CHAR = os.path.join(RES, "characterized_candidates.csv")
BEST = os.path.join(RES, "final_best_candidates.csv")
REVIEW = os.path.join(RES, "needs_manual_blend_review.csv")
FEATURES = os.path.join(ROOT, "data", "catalogs", "unknown_features.csv")
CAND_LIST = os.path.join(ROOT, "data", "catalogs", "unknown_candidate_list.csv")
MODEL = os.path.join(ROOT, "models", "best_model.joblib")
META = os.path.join(ROOT, "models", "best_model_metadata.json")
EXPECTED_MD5 = "fe3fa82f36cc978396c68be07d6057f9"
TOL = 1e-9


def fail(msg):
    print(f"GATE FAILED: {msg}\nNothing was written.", file=sys.stderr)
    sys.exit(3)


def load_08():
    spec = importlib.util.spec_from_file_location("m08", os.path.join(HERE, "08_characterize_candidates.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    with open(MODEL, "rb") as f:
        blob = f.read()
    md5 = hashlib.md5(blob).hexdigest()
    if md5 != EXPECTED_MD5:
        fail(f"gate 0: models/best_model.joblib md5 is {md5}, expected {EXPECTED_MD5}")
    model = joblib.load(io.BytesIO(blob))
    print(f"gate 0 PASS: production model {md5}")

    m08 = load_08()
    for p in (CHAR, BEST, REVIEW):
        m08.refuse_dup_columns(p)
    char = pd.read_csv(CHAR)
    best0, review0 = pd.read_csv(BEST), pd.read_csv(REVIEW)
    feature_columns = json.load(open(META))["feature_columns"]
    sys.path.insert(0, os.path.join(HERE, "experiments"))
    from candidate_pool import _optional_features
    optional = _optional_features()
    blocking = [c for c in feature_columns if c not in optional]

    # ---- features, as 06 builds them
    feats = pd.read_csv(FEATURES)
    feats["host"] = feats["host"].astype(str)
    feats = feats.drop_duplicates(subset="host", keep="last")
    feats = feats[feats["status"].astype(str).str.startswith("Success") & feats["host"].isin(char["host"])]
    cl = pd.read_csv(CAND_LIST)
    cl["host"] = "TIC_" + cl["tic_id"].astype("int64").astype(str)
    cl = cl.drop_duplicates(subset="host", keep="last")
    X = feats.merge(cl[["host", "st_rad", "st_teff", "st_mass"]], on="host", how="left")
    ch = char.set_index("host")

    # gate 1b: the two sources must agree wherever both have the values
    both = X.set_index("host")[["st_rad", "st_teff"]].dropna().join(
        ch[["st_rad", "st_teff"]].dropna(), rsuffix="_char", how="inner")
    diff = both[(np.abs(both.st_rad - both.st_rad_char) > 0) | (np.abs(both.st_teff - both.st_teff_char) > 0)]
    if len(diff):
        fail(f"gate 1b: {len(diff)} of {len(both)} stars differ in st_rad/st_teff between the candidate "
             f"list and the characterized file, e.g. {list(diff.index[:5])}")
    print(f"gate 1b PASS: st_rad/st_teff identical for all {len(both)} stars present in both sources")
    from_char = X["st_rad"].isna() & X["host"].isin(ch.index)
    for c in ("st_rad", "st_teff"):
        X.loc[from_char, c] = X.loc[from_char, "host"].map(ch[c])
    src = np.where(from_char, "characterized_candidates.csv", "unknown_candidate_list.csv")
    print(f"st_rad/st_teff: {int((~from_char).sum())} from the candidate list, "
          f"{int(from_char.sum())} from the characterized file")

    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        fail(f"required feature columns absent: {missing}")
    bad = X[blocking].isna().any(axis=1)
    unscored = sorted(set(char["host"]) - set(X.loc[~bad, "host"]))
    for h in unscored:
        why = [c for c in blocking if h in set(X.host) and pd.isna(X.set_index("host").loc[h, c])] or ["no Success feature row"]
        print(f"NOT SCORED (keeps its old probability, labelled unscored): {h} -- NaN in {why}")
    X, src = X[~bad].copy(), src[~bad.values]
    X["new_p"] = model.predict_proba(X[feature_columns])[:, 1]

    # gate 1: reproduce the probabilities already scored by this model
    cur = ch.loc[ch["scored_model_md5"] == EXPECTED_MD5, "predicted_probability"]
    rep = X.set_index("host").loc[cur.index.intersection(X.host), "new_p"]
    if len(rep) != len(cur):
        fail(f"gate 1: only {len(rep)} of the {len(cur)} fe3fa82f-stamped stars could be rescored")
    worst = float(np.max(np.abs(rep.values - cur.loc[rep.index].values)))
    if worst > TOL:
        fail(f"gate 1: fe3fa82f-stamped probabilities not reproduced (max |diff| {worst:.3e} > {TOL})")
    print(f"gate 1 PASS: {len(rep)} fe3fa82f probabilities reproduced, max |diff| {worst:.2e}")

    # ---- write in place via 08's own update path, then stamp every rescored star
    scored = pd.DataFrame({"host": X["host"], "predicted_probability": X["new_p"],
                           "scored_model_md5": EXPECTED_MD5, "scored_at": ts})
    full = m08.apply_scored_fields(char, scored, ts)
    rows = full.index[full["host"].isin(set(X["host"]))]
    srcmap = dict(zip(X["host"], src))
    full.loc[rows, "scored_model_md5"] = EXPECTED_MD5
    full.loc[rows, "scored_at"] = ts
    full.loc[rows, "scored_source"] = [
        f"rescore {ts}: production model {EXPECTED_MD5[:8]}; features unknown_features.csv; "
        f"st_rad/st_teff from {srcmap[h]}" for h in full.loc[rows, "host"]]
    t = [m08.confidence_tier(r) for _, r in full.iterrows()]
    full["confidence_tier"] = [x[0] for x in t]
    full["supporting_evidence"] = [x[1] for x in t]
    full["doubting_evidence"] = [x[2] for x in t]

    # ---- derived lists: 08's own masks (08_characterize_candidates.py, combined filter)
    not_flagged = ~full["newly_flagged_in_archive_or_exofop"].astype(bool) & (full["exofop_code"] != "HIT")
    not_vsx = full["vsx_code"] != "HIT"
    plausible = full["radius_plausible"].fillna(False).astype(bool)
    blend_unknown = full["blending_status"].str.contains("UNKNOWN", na=False)
    blend_ok_strict = full["blending_status"].str.contains("LOW", na=False) & ~full["blending_status"].str.contains("LOW-MODERATE", na=False)
    blend_ok_lenient = ~full["blending_status"].str.contains("HIGH|UNKNOWN", na=False, regex=True)
    base = not_flagged & not_vsx & plausible
    strict_mask, lenient_mask, review_mask = base & blend_ok_strict, base & blend_ok_lenient, base & blend_unknown
    combined = full[lenient_mask].copy()
    combined["combined_tier"] = np.where(strict_mask.loc[combined.index], "strict", "moderate")
    combined = combined.sort_values(["confidence_tier", "predicted_probability"],
                                    key=lambda s: s.map({"High": 2, "Medium": 1, "Low": 0}) if s.name == "confidence_tier" else s,
                                    ascending=[False, False])
    needs_review = full[review_mask].copy()
    needs_review["review_reason"] = "Gaia did not resolve the target star -- blend risk unknown, not verified low"
    needs_review = needs_review.sort_values("predicted_probability", ascending=False)

    # gate 2: membership unchanged
    for name, new, old in (("final_best", combined, best0), ("needs_review", needs_review, review0)):
        if set(new["host"]) != set(old["host"]):
            fail(f"gate 2: {name} membership changed: +{sorted(set(new.host) - set(old.host))[:5]} "
                 f"-{sorted(set(old.host) - set(new.host))[:5]}")
    ct_old = dict(zip(best0.host, best0.combined_tier))
    if any(ct_old[h] != t for h, t in zip(combined.host, combined.combined_tier)):
        fail("gate 2: combined_tier (strict/moderate) changed for some final_best star")
    print(f"gate 2 PASS: final_best {len(combined)} and needs_review {len(needs_review)} -- same members")

    for frame, ref, path in ((full, char, CHAR), (combined, best0, BEST), (needs_review, review0, REVIEW)):
        if set(frame.columns) != set(ref.columns):
            fail(f"{os.path.basename(path)}: column set changed: +{sorted(set(frame.columns) - set(ref.columns))} "
                 f"-{sorted(set(ref.columns) - set(frame.columns))}")
    for frame, ref, path in ((full, char, CHAR), (combined, best0, BEST), (needs_review, review0, REVIEW)):
        tmp = path + ".rescore.tmp"
        frame[list(ref.columns)].to_csv(tmp, index=False)
        os.replace(tmp, path)
    print(f"wrote {len(rows)} rescored rows ({len(unscored)} kept unscored) to "
          f"{', '.join(os.path.relpath(p, ROOT) for p in (CHAR, BEST, REVIEW))}")


if __name__ == "__main__":
    main()
