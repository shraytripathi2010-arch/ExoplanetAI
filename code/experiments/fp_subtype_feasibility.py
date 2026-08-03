"""fp_subtype_feasibility.py -- do false-positive SUBTYPE labels exist at all?

Gate for the multi-task/multi-label idea. The whole experiment depends on a
prerequisite that may simply not be there: a real, externally-assigned reason
each negative-class star is a false positive (eclipsing binary, blend/nearby
EB, instrumental systematic, stellar variability, ...).

SOURCES CHECKED, in order of authority:
  1. NASA archive `toi` table -- 90 columns, exactly ONE disposition field
     (`tfopwg_disp`, values FP/PC/KP/CP/APC/FA). No comment, note, subtype,
     classification or flag column exists. Confirmed by querying
     TAP_SCHEMA.columns. This source cannot supply subtypes.
  2. ExoFOP-TESS TOI export -- carries a free-text `Comments` field written by
     the TFOP working groups, plus per-SG disposition columns (SG1A, SG1B,
     SG2, SG3, SG4, SG5). This is the only realistic source.

WHAT THIS SCRIPT DOES NOT DO
It does NOT construct subtypes from the 24 model features. Deriving auxiliary
labels from the same features the auxiliary task is supposed to improve is
circular, and would manufacture a result rather than measure one. Subtypes here
come only from human-written external vetting text. If that text is absent or
uninformative, the experiment does not exist and this script says so.

Keyword matching over the comment field is a MEASUREMENT of how much real
subtype information the humans recorded -- deliberately conservative, reporting
matched, unmatched and ambiguous counts rather than forcing every row into a
bucket.

LEAKAGE NOTE (recorded here because it decides validity, not just coverage):
many TFOP dispositions and comments are written AFTER ground-based follow-up --
seeing-limited photometry, reconnaissance spectroscopy, speckle imaging. Those
observations are not available to this pipeline at prediction time. Any
subtype derived from them encodes post-hoc knowledge the model could never
have. The script therefore also counts how many commented negatives reference
follow-up observations.
"""
import os
import re
import sys
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
EXOFOP_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
CACHE = os.path.join(SCRIPT_DIR, "exofop_toi_export.csv")
RESULTS = os.path.join(SCRIPT_DIR, "fp_subtype_feasibility.json")

# Subtype vocabulary, drawn from how TFOP actually writes these notes.
# Ordered: earlier patterns win when several match, most-specific first.
SUBTYPE_PATTERNS = [
    ("nearby_eb_blend", r"\bNEB\b|nearby eclipsing binary|neighbou?r.{0,20}\bEB\b|"
                        r"blended? (?:with )?(?:nearby )?EB|contaminat"),
    ("eclipsing_binary", r"\bEB\b|eclipsing binary|\bSB1\b|\bSB2\b|double.lined|"
                         r"secondary eclipse|v-shaped|grazing"),
    ("stellar_variability", r"variab|puls|rotat|spot|flare|\bRR Lyr|delta scuti|"
                            r"gamma dor|heartbeat"),
    ("systematic_artifact", r"systematic|artifact|artefact|momentum dump|scattered light|"
                            r"detrend|instrument|spacecraft|asteroid|cosmic ray|"
                            r"data (?:gap|quality)"),
    ("centroid_offset", r"centroid|offset|photocenter|difference image"),
    ("odd_even", r"odd.even|odd/even"),
    ("stellar_companion", r"companion|binary star|double star|speckle|"
                          r"adaptive optics|\bAO\b imaging"),
]

# Comment text implying ground-based / spectroscopic follow-up the pipeline
# cannot see at prediction time.
FOLLOWUP_PATTERNS = r"spectroscop|\bRV\b|radial velocit|speckle|adaptive optics|\bAO\b|" \
                    r"ground.based|seeing.limited|LCO|SG1|SG2|SG3|SG4|TRES|CHIRON|" \
                    r"follow.?up|on.?off|photometric follow"


def load_exofop():
    if os.path.exists(CACHE):
        print(f"using cached ExoFOP export: {CACHE}")
        return pd.read_csv(CACHE, low_memory=False)
    print("downloading ExoFOP TOI export...")
    d = pd.read_csv(EXOFOP_URL, low_memory=False)
    d.to_csv(CACHE, index=False)
    return d


def classify(text):
    if not isinstance(text, str) or not text.strip():
        return None
    for name, pat in SUBTYPE_PATTERNS:
        if re.search(pat, text, flags=re.I):
            return name
    return "other_uncategorised"


def main():
    res = {}
    tr = pd.read_csv(TRAINING_CSV)
    neg = tr[tr["label"] == 0].copy()
    neg["tic"] = pd.to_numeric(
        neg["host"].astype(str).str.extract(r"^TIC_(\d+)", expand=False),
        errors="coerce")
    neg = neg.dropna(subset=["tic"])
    neg["tic"] = neg["tic"].astype("int64")
    print("=" * 78)
    print("PART 1 GATE: do FP subtype labels exist?")
    print("=" * 78)
    print(f"negative-class stars in training.csv: {len(neg)}")
    res["n_negatives"] = int(len(neg))

    ex = load_exofop()
    ex.columns = [c.strip() for c in ex.columns]
    ex["tic"] = pd.to_numeric(ex["TIC ID"], errors="coerce")
    ex = ex.dropna(subset=["tic"])
    ex["tic"] = ex["tic"].astype("int64")
    print(f"ExoFOP TOI export rows: {len(ex)} | unique TIC: {ex.tic.nunique()}")

    # one row per TIC: prefer a row that actually has a comment
    ex["_has_comment"] = ex["Comments"].astype(str).str.strip().ne("").fillna(False) \
        & ex["Comments"].notna()
    ex = ex.sort_values("_has_comment", ascending=False).drop_duplicates("tic")

    m = neg.merge(ex[["tic", "Comments", "TFOPWG Disposition", "TESS Disposition",
                      "SG1A", "SG1B", "SG2", "SG3", "SG4", "SG5"]],
                  on="tic", how="left")
    matched = m["TFOPWG Disposition"].notna()
    print(f"\nnegatives matched to an ExoFOP TOI row: {int(matched.sum())} "
          f"({100*matched.mean():.1f}%)")
    res["n_matched_exofop"] = int(matched.sum())

    has_comment = m["Comments"].notna() & m["Comments"].astype(str).str.strip().ne("")
    print(f"negatives with a NON-EMPTY Comments field: {int(has_comment.sum())} "
          f"({100*has_comment.mean():.1f}%)")
    res["n_with_comment"] = int(has_comment.sum())

    # ---- SG working-group disposition columns ----
    print("\n" + "-" * 78)
    print("per-working-group disposition coverage (SG1A..SG5)")
    print("-" * 78)
    sg_cov = {}
    for c in ["SG1A", "SG1B", "SG2", "SG3", "SG4", "SG5"]:
        n = int(m[c].notna().sum())
        sg_cov[c] = n
        print(f"  {c:5s} non-null for {n:5d} / {len(m)} negatives "
              f"({100*n/len(m):5.1f}%)")
    res["sg_coverage"] = sg_cov

    # ---- subtype extraction from comments ----
    print("\n" + "-" * 78)
    print("subtype extracted from human vetting comments")
    print("-" * 78)
    m["subtype"] = m["Comments"].map(classify)
    vc = m["subtype"].value_counts(dropna=False)
    total_labelled = int(m["subtype"].notna().sum())
    usable = int(m.loc[m["subtype"].notna() & (m["subtype"] != "other_uncategorised")].shape[0])
    for k, v in vc.items():
        print(f"  {str(k):24s} {v:5d}")
    print(f"\n  any subtype assigned:      {total_labelled}")
    print(f"  RECOGNISED subtype (excl. other_uncategorised): {usable}")
    res["subtype_counts"] = {str(k): int(v) for k, v in vc.items()}
    res["n_any_subtype"] = total_labelled
    res["n_recognised_subtype"] = usable

    # ---- leakage: do comments reference follow-up the pipeline cannot see? ----
    print("\n" + "-" * 78)
    print("LEAKAGE CHECK: comments referencing post-hoc follow-up")
    print("-" * 78)
    fu = m["Comments"].astype(str).str.contains(FOLLOWUP_PATTERNS, case=False,
                                                na=False, regex=True)
    n_fu = int((fu & has_comment).sum())
    print(f"  commented negatives referencing follow-up observations: {n_fu}"
          + (f"  ({100*n_fu/max(int(has_comment.sum()),1):.1f}% of commented)"
             if has_comment.sum() else ""))
    print("  (spectroscopy / RV / speckle / AO / ground-based photometry are NOT")
    print("   available to this pipeline at prediction time -- any subtype resting")
    print("   on them encodes knowledge the model could never have)")
    res["n_comments_referencing_followup"] = n_fu

    if has_comment.sum():
        print("\n  sample comments:")
        for s in m.loc[has_comment, "Comments"].astype(str).head(6):
            print(f"    - {s[:150]}")

    # ---- LEAKAGE-ADJUSTED COUNTS ----
    # A subtype is unusable if determining it required information the pipeline
    # cannot have at prediction time. Three ways that happens:
    #   explicit  -- the comment names follow-up (spectroscopy, speckle, AO, ...)
    #   spectro   -- SB1/SB2/RV designations, which are spectroscopic by definition
    #   inherent  -- NEB ("nearby eclipsing binary"). Deciding the eclipse is on a
    #                NEIGHBOUR rather than the target means spatially resolving
    #                them. TFOP's "retired as NEB" is seeing-limited photometry
    #                from SG1. TESS difference imaging can sometimes reach the
    #                same conclusion, and this pipeline does compute a centroid,
    #                so this is the ambiguous case -- excluded CONSERVATIVELY,
    #                and the count is reported separately so the call is visible
    #                rather than buried.
    print("\n" + "-" * 78)
    print("LEAKAGE-ADJUSTED SUBTYPE COUNTS")
    print("-" * 78)
    inherent = m["Comments"].astype(str).str.contains(
        r"retired as NEB|\bNEB\b", case=False, na=False)
    explicit = m["Comments"].astype(str).str.contains(
        FOLLOWUP_PATTERNS, case=False, na=False, regex=True)
    spectro = m["Comments"].astype(str).str.contains(
        r"\bSB1\b|\bSB2\b|spectroscop|radial velocit", case=False, na=False)
    m["leaky"] = inherent | explicit | spectro
    rec = m[m["subtype"].notna() & (m["subtype"] != "other_uncategorised")]
    print(f"  {'subtype':<24}{'total':>7}{'leaky':>7}{'USABLE':>9}")
    lk = {}
    for s, g in rec.groupby("subtype"):
        u = int((~g["leaky"]).sum())
        lk[s] = {"total": int(len(g)), "leaky": int(g["leaky"].sum()), "usable": u}
        print(f"  {s:<24}{len(g):>7}{int(g['leaky'].sum()):>7}{u:>9}")
    n_usable_clean = int((~rec["leaky"]).sum())
    print(f"  {'TOTAL':<24}{len(rec):>7}{int(rec['leaky'].sum()):>7}{n_usable_clean:>9}")
    res["subtype_leakage_breakdown"] = lk
    res["n_usable_after_leakage"] = n_usable_clean

    # ---- GATE ----
    print("\n" + "=" * 78)
    print("GATE DECISION (threshold: >=400-500 negatives with reliable subtypes,")
    print("and no single category swamping the rest)")
    print("=" * 78)
    if usable:
        sub = m.loc[m["subtype"].notna() & (m["subtype"] != "other_uncategorised"),
                    "subtype"].value_counts()
        dom = 100 * sub.iloc[0] / sub.sum()
        print(f"  usable subtype labels: {usable}")
        print(f"  distinct subtypes:     {len(sub)}")
        print(f"  dominant category:     {sub.index[0]} at {dom:.1f}%")
        res["dominant_subtype_pct"] = float(dom)
        res["n_distinct_subtypes"] = int(len(sub))
    else:
        print("  usable subtype labels: 0")
        res["dominant_subtype_pct"] = None
        res["n_distinct_subtypes"] = 0

    raw_passes = usable >= 400
    clean_passes = n_usable_clean >= 400
    res["gate_passes_raw"] = bool(raw_passes)
    res["gate_passes_after_leakage"] = bool(clean_passes)
    if n_usable_clean:
        cs = rec.loc[~rec["leaky"], "subtype"].value_counts()
        top2 = 100 * cs.iloc[:2].sum() / cs.sum()
        print(f"\n  after removing leaky labels: {n_usable_clean} usable, "
              f"{len(cs)} subtypes, top-2 hold {top2:.0f}%")
        print(f"    sizes: {cs.to_dict()}")
        res["clean_subtype_sizes"] = {str(k): int(v) for k, v in cs.items()}
        res["clean_top2_share_pct"] = float(top2)
    print(f"\n  raw gate (>=400):              {'CLEARS' if raw_passes else 'FAILS'}")
    print(f"  leakage-adjusted gate (>=400): {'CLEARS' if clean_passes else 'FAILS'}")
    print("\n  The leakage-adjusted number is the real one. Subtypes that exist only")
    print("  because a human looked through a ground-based telescope are not")
    print("  learnable auxiliary targets for a pipeline that never gets that look.")

    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
