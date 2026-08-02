"""multisector_feasibility.py -- ITEM 1, STEP 1: is a cross-sector consistency
feature even worth building?

Two questions decide it, and both must be answered BEFORE any extraction code
is written, because either can kill the idea outright:

  1. COVERAGE. What fraction of training stars actually have >1 TESS sector? A
     feature populated for a small minority is mostly imputation, and the
     experiment ends up measuring the imputer rather than the physics.

  2. LEAKAGE. Does the SECTOR COUNT ITSELF predict the label? This is the
     serious risk and it is not hypothetical. The positive class is confirmed
     planets -- objects that were confirmed partly BECAUSE they were observed
     repeatedly -- while the negative class is TOI false positives, often
     dispositioned from a single sector. If n_sectors alone separates the
     classes, then any feature derived from multi-sector data inherits that
     separation, and a model would "learn" observation history rather than
     astrophysics. That would look like a win on this dataset and fail
     completely on new stars, which is the exact failure mode the frozen split
     and every leakage check in this project exist to prevent.

Measured on a class-stratified random sample rather than all 5,631 stars: each
star costs a live MAST search, and a few hundred is ample to estimate a
proportion and to detect a label correlation large enough to matter.
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
TIC_MAP_CSV = os.path.join(SCRIPT_DIR, "positive_class_tic_ids.csv")
OUT_CSV = os.path.join(SCRIPT_DIR, "multisector_feasibility_sample.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "multisector_feasibility_results.json")

SAMPLE_PER_CLASS = 200
N_WORKERS = 8
RANDOM_SEED = 42


def sector_count(tic_id):
    """Number of distinct TESS sectors with light curve data for this TIC.
    Counts SECTORS, not products: the same sector often appears two or three
    times (120s and 20s cadence, SPOC and QLP), and counting products would
    inflate 'multi-sector' with duplicates of one observation."""
    import lightkurve as lk
    try:
        s = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS")
        if s is None or len(s) == 0:
            return 0, []
        try:
            secs = sorted({int(x) for x in s.table["sequence_number"]})
        except Exception:
            return len(s), []
        return len(secs), secs
    except Exception:
        return -1, []   # -1 = query failed, distinct from 0 = genuinely none


def main():
    df = pd.read_csv(TRAINING_CSV)
    tmap = pd.read_csv(TIC_MAP_CSV).dropna(subset=["tic_id"])
    tmap = tmap.set_index("host")["tic_id"].astype("int64").to_dict()

    rows = []
    for _, r in df.iterrows():
        host, label = r["host"], r["label"]
        tic = (int(str(host).replace("TIC_", "")) if label == 0
               else tmap.get(host))
        if tic is not None:
            rows.append({"host": host, "label": int(label), "tic_id": int(tic)})
    pool = pd.DataFrame(rows)
    print(f"{len(df)} training rows; {len(pool)} have a resolvable TIC ID "
          f"({int((pool.label==1).sum())} positive / {int((pool.label==0).sum())} negative)")

    rng = np.random.RandomState(RANDOM_SEED)
    parts = []
    for lbl in (1, 0):
        sub = pool[pool.label == lbl]
        n = min(SAMPLE_PER_CLASS, len(sub))
        parts.append(sub.iloc[rng.choice(len(sub), n, replace=False)])
    sample = pd.concat(parts).reset_index(drop=True)
    print(f"sampling {len(sample)} stars ({int((sample.label==1).sum())} pos / "
          f"{int((sample.label==0).sum())} neg) with {N_WORKERS} workers...\n")

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(sector_count, r.tic_id): r for r in sample.itertuples()}
        for i, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            try:
                n, secs = f.result()
            except Exception:
                n, secs = -1, []
            results.append({"host": r.host, "label": r.label, "tic_id": r.tic_id,
                            "n_sectors": n, "sectors": ",".join(map(str, secs))})
            if i % 50 == 0:
                print(f"  {i}/{len(sample)} ({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(results)
    res.to_csv(OUT_CSV, index=False)
    ok = res[res.n_sectors >= 0]
    failed = int((res.n_sectors < 0).sum())

    print("\n" + "=" * 72)
    print("COVERAGE")
    print("=" * 72)
    print(f"  queries failed: {failed}/{len(res)}")
    multi = ok.n_sectors > 1
    print(f"  stars with >1 sector: {int(multi.sum())}/{len(ok)} = {100*multi.mean():.1f}%")
    print(f"  median sectors: {ok.n_sectors.median():.0f}   mean: {ok.n_sectors.mean():.2f}")
    print("\n  sector-count distribution:")
    for k, v in ok.n_sectors.value_counts().sort_index().head(12).items():
        print(f"    {k:>3} sector(s): {v}")

    print("\n" + "=" * 72)
    print("LEAKAGE -- does sector count alone predict the label?")
    print("=" * 72)
    out = {"n_failed": failed, "n_ok": int(len(ok)),
           "pct_multisector": float(100 * multi.mean()),
           "median_sectors": float(ok.n_sectors.median())}
    for lbl, name in ((1, "positive"), (0, "negative")):
        s = ok[ok.label == lbl]
        if len(s):
            print(f"  {name}: n={len(s)}, mean sectors {s.n_sectors.mean():.2f}, "
                  f"median {s.n_sectors.median():.0f}, >1 sector {100*(s.n_sectors>1).mean():.1f}%")
            out[f"{name}_mean_sectors"] = float(s.n_sectors.mean())
            out[f"{name}_pct_multi"] = float(100 * (s.n_sectors > 1).mean())

    if len(ok.label.unique()) > 1:
        from sklearn.metrics import roc_auc_score
        from scipy.stats import mannwhitneyu
        auc = roc_auc_score(ok.label, ok.n_sectors)
        u, p = mannwhitneyu(ok[ok.label == 1].n_sectors, ok[ok.label == 0].n_sectors)
        out["n_sectors_single_feature_auc"] = float(auc)
        out["mannwhitney_p"] = float(p)
        print(f"\n  single-feature AUC of n_sectors alone: {auc:.3f}")
        print(f"  Mann-Whitney U p-value: {p:.3g}")
        verdict = ("SERIOUS LEAKAGE RISK" if auc > 0.65 or auc < 0.35 else
                   "moderate, needs care" if auc > 0.58 or auc < 0.42 else
                   "no meaningful label correlation")
        out["leakage_verdict"] = verdict
        print(f"  -> {verdict}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH} and {OUT_CSV}")


if __name__ == "__main__":
    main()
