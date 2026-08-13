"""cadence_class_confound.py -- PART 0 GATE for the cadence-aware detrending fix.

THE PRECEDENT THIS EXISTS TO HONOUR
-----------------------------------
Training-side multi-sector reprocessing was PERMANENTLY EXCLUDED because
eligibility was class-correlated (72.5% of positives vs 41.4% of negatives,
Fisher p=0.0034, OR 3.74): reprocessing only the eligible stars would have
applied an "improvement" differentially by class and injected ~0.19 SD of
artificial class signal.

The cadence fix has exactly the same shape. It changes TLS-derived features for
the ~18% of stars that are NOT at 2-min cadence and leaves the other 82%
untouched. If cadence correlates with class, the fix would manufacture
separation the same way -- a different mechanism, an identical failure.

So this is a hard gate, run BEFORE any reprocessing, and the mechanism being
different is not a reason to assume it is clean.

Cadence is measured DIRECTLY from each star's processed light curve (median
time difference), not read from any cached table, so the numbers cannot be
stale relative to training.csv.
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT_CSV = os.path.join(HERE, "cadence_class_confound.csv")
OUT_JSON = os.path.join(HERE, "cadence_class_confound.json")

CURVE_DIRS = [os.path.join(ROOT, "data", "processed"),
              os.path.join(ROOT, "data", "processed_negative")]

# buckets, matching the GP-pilot entry's breakdown
EDGES = [0, 1.0, 2.6, 11.0, 31.0, 1e9]
LABELS = ["20-sec", "2-min", "10-min", "30-min", ">30-min"]
TARGET_HOURS = 401 * 2.0 / 60.0          # the 2-min design intent, 13.3667 h


def measure_cadence(host):
    for d in CURVE_DIRS:
        p = os.path.join(d, host + ".csv")
        if os.path.exists(p):
            try:
                t = pd.read_csv(p, usecols=["time"])["time"].to_numpy()
            except Exception:
                return np.nan, 0
            t = np.sort(t[np.isfinite(t)])
            if len(t) < 100:
                return np.nan, len(t)
            return float(np.median(np.diff(t)) * 1440.0), len(t)
    return np.nan, 0


def main():
    df = pd.read_csv(TRAINING)
    df["host"] = df.host.astype(str)
    print(f"training.csv: {len(df)} rows, "
          f"{int((df.label==1).sum())} positive / {int((df.label==0).sum())} negative")

    rows = []
    for i, (h, lab) in enumerate(zip(df.host, df.label), 1):
        cad, n = measure_cadence(h)
        rows.append({"host": h, "label": int(lab), "cadence_min": cad, "n_points": n})
        if i % 1000 == 0:
            print(f"  measured {i}/{len(df)}", flush=True)
    c = pd.DataFrame(rows)
    c.to_csv(OUT_CSV, index=False)

    have = c[c.cadence_min.notna()].copy()
    print(f"\ncadence measured for {len(have)}/{len(c)} training stars "
          f"({len(c)-len(have)} have no readable processed curve)")
    have["bucket"] = pd.cut(have.cadence_min, EDGES, labels=LABELS)
    # the window each star ACTUALLY gets today vs what the design intends
    have["window_pts_now"] = np.minimum(401, have.n_points - 1)
    have["protected_h_now"] = have.window_pts_now * have.cadence_min / 60.0

    print("\n=== CADENCE x CLASS ===")
    ct = pd.crosstab(have.bucket, have.label)
    ct.columns = ["negative", "positive"]
    pct = pd.crosstab(have.bucket, have.label, normalize="columns") * 100
    pct.columns = ["negative_%", "positive_%"]
    tab = ct.join(pct.round(2))
    tab["median_protected_h"] = have.groupby("bucket").protected_h_now.median().round(2)
    print(tab.to_string())

    # THE gate: is "affected by the fix" (i.e. NOT 2-min) class-correlated?
    have["affected"] = have.bucket != "2-min"
    a = pd.crosstab(have.affected, have.label)
    n_pos = int((have.label == 1).sum()); n_neg = int((have.label == 0).sum())
    aff_pos = int(((have.label == 1) & have.affected).sum())
    aff_neg = int(((have.label == 0) & have.affected).sum())
    print("\n=== THE GATE: is 'affected by the fix' class-correlated? ===")
    print(f"  positives affected: {aff_pos}/{n_pos} = {100*aff_pos/n_pos:.2f}%")
    print(f"  negatives affected: {aff_neg}/{n_neg} = {100*aff_neg/n_neg:.2f}%")

    from scipy.stats import fisher_exact, chi2_contingency
    table = [[aff_pos, n_pos - aff_pos], [aff_neg, n_neg - aff_neg]]
    orr, fp = fisher_exact(table)
    chi2, cp, dof, _ = chi2_contingency(ct.values)
    print(f"  Fisher exact: OR = {orr:.3f}, p = {fp:.6f}")
    print(f"  chi-square across ALL cadence buckets: chi2 = {chi2:.2f}, "
          f"dof = {dof}, p = {cp:.3e}")

    # the multi-sector precedent's own yardstick: how much artificial class
    # signal would a differential treatment inject, in SD units
    p1, p0 = aff_pos / n_pos, aff_neg / n_neg
    pooled = (aff_pos + aff_neg) / (n_pos + n_neg)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_pos + 1 / n_neg))
    z = (p1 - p0) / se if se > 0 else np.nan
    print(f"  difference in affected-rate: {100*(p1-p0):+.2f} pp, z = {z:+.2f}")
    print(f"  (multi-sector precedent, which was DISQUALIFIED: 72.5% vs 41.4%, "
          f"OR 3.74, p 0.0034)")

    CLEAN = fp >= 0.05
    print("\n" + "=" * 70)
    if CLEAN:
        print("GATE PASSED: cadence is NOT significantly class-correlated.")
        print("This is the load-bearing assumption for everything downstream.")
    else:
        print("*** GATE FAILED: cadence IS class-correlated. ***")
        print("Same disqualifying pattern as training-side multi-sector "
              "reprocessing.\nDO NOT reprocess training data with the new window.")
    print("=" * 70)

    json.dump({"n_training": int(len(df)), "n_measured": int(len(have)),
               "affected_positive": aff_pos, "n_positive": n_pos,
               "affected_negative": aff_neg, "n_negative": n_neg,
               "affected_pct_positive": float(100 * p1),
               "affected_pct_negative": float(100 * p0),
               "fisher_or": float(orr), "fisher_p": float(fp),
               "chi2": float(chi2), "chi2_p": float(cp), "z": float(z),
               "gate_passed": bool(CLEAN),
               "bucket_counts": {str(k): int(v) for k, v in
                                 have.bucket.value_counts().items()}},
              open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved {OUT_JSON} and {OUT_CSV}")


if __name__ == "__main__":
    main()
