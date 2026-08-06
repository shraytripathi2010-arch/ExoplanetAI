"""synthetic_gap_attribution.py -- WHERE does the 0.9654 realism gap live?

PART 0 DIAGNOSTIC. Before building a second, more elaborate injector, establish
what the first one actually got wrong. The proposal assumes the gap is
simulation FIDELITY (too few scenario types, no TTVs, no stellar activity) and
that better simulation closes it. That is testable with data already on disk,
and it should be tested before any code is written.

WHAT THE FIRST ATTEMPT ALREADY DID, read from injection.py rather than assumed:
  * injected into REAL processed light curves, so the signal already sits in
    genuine TESS noise, real systematics and real stellar activity
  * drew period/depth/duration by EMPIRICAL RESAMPLING from real positives,
    not from a parametric guess
  * already produced a grazing/V-shaped EB with a secondary eclipse

So "use real light curves as hosts" -- the single largest realism upgrade the
proposal offers -- was already spent, and the result was still 0.9654.

THE TEST. Split the 31 production features into two groups:

  DETECTION statistics  SDE, SDE_raw, FAP, snr, chi2red_min, transit_count,
                        distinct_transit_count, empty_transit_count
                        -- properties of HOW the signal was found

  EVERYTHING ELSE       shape, depth, stellar, crowding, variability
                        -- properties of WHAT the signal looks like

Then train the domain discriminator (real vs synthetic) three ways: all
features, detection-only, and shape-only. If the separability is carried by
the DETECTION group, the gap is a selection-function mismatch -- real positives
are the ones that were discovered and confirmed by a heterogeneous human
process, injected ones are simply recovered -- and no amount of extra scenario
realism addresses it, because the mismatch is not in the astrophysics.

If instead the shape group carries it, the gap IS a fidelity problem and a
better injector is worth building.
"""
import os
import sys
import json
import importlib.util
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
SYNTH = os.path.join(SCRIPT_DIR, "augmented_train_only.csv")
OUT = os.path.join(SCRIPT_DIR, "synthetic_gap_attribution.json")

SEED = 42
DETECTION = ["SDE", "SDE_raw", "FAP", "snr", "chi2red_min",
             "transit_count", "distinct_transit_count", "empty_transit_count"]


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def domain_auc(X, y, cols, seed=SEED):
    """Cross-validated real-vs-synthetic AUC on a column subset."""
    # FAP is reported as 0 by TLS for very strong signals and the pipeline
    # stores inf/-inf in a few columns; the imputer rejects non-finite values.
    # Replace with NaN so the median imputer handles them, rather than
    # dropping the rows (which would silently change the comparison).
    Xs = X[cols].replace([np.inf, -np.inf], np.nan)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, va in skf.split(Xs, y):
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", HistGradientBoostingClassifier(
                             random_state=seed, max_iter=300))])
        pipe.fit(Xs.iloc[tr], y[tr])
        oof[va] = pipe.predict_proba(Xs.iloc[va])[:, 1]
    return float(roc_auc_score(y, oof)), oof


def main():
    m05 = _m05()
    prod = list(m05.FEATURE_COLUMNS)
    real = pd.read_csv(TRAINING)
    syn = pd.read_csv(SYNTH)

    shared = [c for c in prod if c in real.columns and c in syn.columns]
    detection = [c for c in DETECTION if c in shared]
    shape = [c for c in shared if c not in detection]

    print("=" * 88)
    print("WHERE DOES THE SYNTHETIC-vs-REAL GAP LIVE?")
    print("=" * 88)
    print(f"  production features {len(prod)}; present in BOTH real and "
          f"synthetic frames: {len(shared)}")
    print(f"  detection group ({len(detection)}): {detection}")
    print(f"  shape/other group ({len(shape)}): {shape}")
    missing = [c for c in prod if c not in shared]
    if missing:
        print(f"  NOT comparable (absent from the synthetic frame): {missing}")

    R = real[shared].apply(pd.to_numeric, errors="coerce")
    S = syn[shared].apply(pd.to_numeric, errors="coerce")
    X = pd.concat([R, S], ignore_index=True)
    y = np.r_[np.zeros(len(R), int), np.ones(len(S), int)]
    print(f"\n  {len(R)} real rows vs {len(S)} synthetic rows\n")

    res = {"n_real": int(len(R)), "n_synth": int(len(S)),
           "detection_cols": detection, "shape_cols": shape}

    print(f"  {'feature group':<26}{'n':>4}{'domain AUC':>13}")
    for label, cols in [("ALL shared features", shared),
                        ("DETECTION statistics only", detection),
                        ("SHAPE / everything else", shape)]:
        a, _ = domain_auc(X, y, cols)
        res[label] = a
        print(f"  {label:<26}{len(cols):>4}{a:>13.4f}")

    # per-feature standardized mean difference, to see which columns move
    print(f"\n  per-feature standardized mean difference (synthetic - real), "
          f"|SMD| >= 0.30:")
    smd = {}
    for c in shared:
        r = R[c].replace([np.inf,-np.inf],np.nan).dropna()
        s = S[c].replace([np.inf,-np.inf],np.nan).dropna()
        if len(r) < 30 or len(s) < 30:
            continue
        sd = np.sqrt((r.var() + s.var()) / 2.0)
        if sd > 0:
            smd[c] = float((s.mean() - r.mean()) / sd)
    res["smd"] = smd
    for c, v in sorted(smd.items(), key=lambda kv: -abs(kv[1])):
        if abs(v) >= 0.30:
            grp = "DETECTION" if c in detection else "shape"
            print(f"    {c:<26}{v:>+8.2f}   [{grp}]")

    # single-feature domain AUC for the worst offenders
    print(f"\n  single-feature domain AUC (how separable on that column alone):")
    singles = {}
    for c in sorted(smd, key=lambda k: -abs(smd[k]))[:8]:
        a, _ = domain_auc(X, y, [c])
        singles[c] = a
        grp = "DETECTION" if c in detection else "shape"
        print(f"    {c:<26}{a:>8.4f}   [{grp}]")
    res["single_feature_domain_auc"] = singles

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
