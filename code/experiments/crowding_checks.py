"""crowding_checks.py -- Part 1 coverage + Part 3 leakage/validity, before modelling.

Five things, in the order that can kill the experiment fastest:

  1. COVERAGE / SPARSITY at full scale -- what fraction of stars actually have
     a catalogued neighbour, and how many.
  2. MISSINGNESS BY CLASS -- the canary. If a feature's mere availability
     predicts the label, the model can learn label provenance rather than
     astrophysics, and HistGradientBoosting handles NaN natively so it will.
  3. SINGLE-FEATURE AUC for each new feature.
  4. CORRELATION against the existing 24, flagging |r| > 0.80, the redundancy
     threshold this project has already hit once.
  5. PRODUCTION AVAILABILITY -- the features must compute for unknown
     candidates from the `TIC_<id>` key alone, with no ra/dec and no label.
     Tested live against real rows from the unknown pool, not asserted.
"""
import os
import sys
import importlib.util
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import fast_auc  # noqa: E402

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CROWD = os.path.join(SCRIPT_DIR, "crowding_per_star.csv")
UNKNOWN = os.path.join(ROOT, "data", "catalogs", "unknown_features.csv")

NEW = ["crowd_contratio", "crowd_numcont",
       "crowd_flux_ratio_max", "crowd_nearest_arcsec"]


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def safe_auc(y, v):
    m = np.isfinite(np.asarray(v, dtype=float))
    y2, v2 = np.asarray(y)[m], np.asarray(v, dtype=float)[m]
    if len(y2) < 30 or len(np.unique(y2)) < 2:
        return np.nan, int(m.sum())
    return fast_auc(y2, v2), int(m.sum())


def main():
    m05 = _m05()
    df = pd.read_csv(TRAINING)
    crowd = pd.read_csv(CROWD)
    d = df.merge(crowd, on="host", how="left")
    y = d["label"].to_numpy()

    print("=" * 100)
    print("PART 1 -- COVERAGE AND SPARSITY (full scale)")
    print("=" * 100)
    print(f"  training stars: {len(df)}   crowding rows fetched: {len(crowd)}")
    ok = d["crowd_ok"] == 1
    print(f"  resolved against TIC: {int(ok.sum())} ({ok.mean()*100:.1f}%)")
    if (~ok).any():
        print("  unresolved reasons:")
        for note, n in d.loc[~ok, "crowd_note"].fillna("(not fetched)").apply(
                lambda s: str(s).split(";")[0][:60]).value_counts().head(4).items():
            print(f"      {n:>5}  {note}")

    print("\n  neighbour counts (catalogued TIC sources, excluding the target):")
    for col, lab in [("crowd_n_nb_21", 'within 21" (1 TESS px)'),
                     ("crowd_n_nb_42", 'within 42" (2 TESS px)'),
                     ("crowd_n_nb_63", 'within 63" (3 TESS px)')]:
        if col not in d.columns:
            continue
        s = d.loc[ok, col].dropna()
        print(f"    {lab:<26} >=1 neighbour: {(s > 0).mean()*100:5.1f}%   "
              f"median {s.median():.0f}   p90 {s.quantile(.9):.0f}   max {s.max():.0f}")

    print("\n" + "=" * 100)
    print("PART 3.1 -- MISSINGNESS BY CLASS (the leakage canary)")
    print("=" * 100)
    print(f"  {'feature':<26}{'planet':>10}{'false-pos':>12}{'gap':>9}"
          f"{'AUC of availability':>22}")
    canary = {}
    for f in NEW:
        if f not in d.columns:
            continue
        present = d[f].notna()
        p1 = present[y == 1].mean() * 100
        p0 = present[y == 0].mean() * 100
        a, _ = safe_auc(y, present.astype(float))
        canary[f] = {"present_planet_pct": p1, "present_fp_pct": p0,
                     "availability_auc": None if np.isnan(a) else float(a)}
        flag = "  <-- PROVENANCE RISK" if abs(0.5 - (a if not np.isnan(a) else .5)) > 0.05 else ""
        print(f"  {f:<26}{p1:>9.1f}%{p0:>11.1f}%{p1-p0:>+9.1f}"
              f"{(a if not np.isnan(a) else float('nan')):>22.4f}{flag}")

    print("\n" + "=" * 100)
    print("PART 3.1b -- SINGLE-FEATURE AUC (values, where present)")
    print("=" * 100)
    sf = {}
    for f in NEW:
        if f not in d.columns:
            continue
        a, n = safe_auc(y, d[f])
        sf[f] = {"auc": None if np.isnan(a) else float(a), "n": n}
        print(f"  {f:<26} n={n:>5}  AUC={a:.4f}   |0.5-AUC|={abs(0.5-a):.4f}")

    print("\n" + "=" * 100)
    print("PART 3.2 -- CORRELATION WITH THE EXISTING 24 (flagging |r| > 0.80)")
    print("=" * 100)
    X, _ = m05.build_feature_matrix(df)
    worst = {}
    for f in NEW:
        if f not in d.columns:
            continue
        v = pd.to_numeric(d[f], errors="coerce")
        cors = {c: abs(v.corr(pd.to_numeric(X[c], errors="coerce")))
                for c in X.columns}
        cors = {k: x for k, x in cors.items() if np.isfinite(x)}
        top = sorted(cors.items(), key=lambda kv: -kv[1])[:3]
        worst[f] = [{"feature": k, "abs_r": float(x)} for k, x in top]
        flag = "  <-- REDUNDANT" if top and top[0][1] > 0.80 else ""
        print(f"  {f:<26} max |r| = {top[0][1]:.3f} vs {top[0][0]}{flag}")
        print(f"       next: " + ", ".join(f"{k} {x:.3f}" for k, x in top[1:]))

    print("\n" + "=" * 100)
    print("PART 3.3 -- AVAILABLE AT PREDICTION TIME? (live, unknown pool)")
    print("=" * 100)
    from crowding_features import crowding_for
    unk = pd.read_csv(UNKNOWN)
    sample = unk["host"].head(5).tolist()
    print(f"  unknown pool: {len(unk)} rows, keyed like {sample[0]!r}")
    print("  computing from the TIC key alone -- no ra/dec, no label:\n")
    got = 0
    for h in sample:
        r = crowding_for(h)                      # NOTE: no ra/dec passed
        vals = " ".join(f"{k.replace('crowd_',''):}={r[k]:.4g}" if
                        r[k] == r[k] else f"{k.replace('crowd_','')}=nan"
                        for k in NEW)
        print(f"    {h:<18} ok={r['crowd_ok']}  {vals}")
        got += r["crowd_ok"]
    print(f"\n  {got}/{len(sample)} unknown candidates resolved from the TIC key alone")

    out = {"coverage_resolved_pct": float(ok.mean() * 100),
           "missingness_by_class": canary, "single_feature_auc": sf,
           "top_correlations": worst,
           "unknown_pool_resolved": f"{got}/{len(sample)}"}
    import json
    json.dump(out, open(os.path.join(SCRIPT_DIR, "crowding_checks.json"), "w"),
              indent=2, default=float)
    print("\nSaved crowding_checks.json")


if __name__ == "__main__":
    main()
