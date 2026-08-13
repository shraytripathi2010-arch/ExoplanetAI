"""shape_ratio_importance.py -- is `transit_shape_ratio` load-bearing?

WHY THIS EXISTS
---------------
`secondary_eclipse_depth` was found to be computed at the WRONG PHASE (TLS's
`folded_phase` puts the PRIMARY at 0.5, not 0) and to carry single-feature AUC
0.4935 -- chance. It was then measured at permutation-importance **rank 9/31**:
buggy but load-bearing, which is why both fixing it and retiring it made things
worse.

`transit_shape_ratio` was confirmed to have the SAME phase-convention bug, and
its permutation importance was NEVER measured. So nobody knows whether it is
the same "buggy but load-bearing" case or genuinely dead weight. This answers
that, and nothing else -- no promotion, no retirement, no model change.

PROTOCOL
--------
Identical to `secondary_retire_validate.py`, so the two numbers are directly
comparable: production's exact recipe, frozen manifest test mask,
SEED = 20260812, n_repeats = 10, scoring roc_auc. Both columns are ranked in
the SAME run, which is the only way the comparison is fair -- the earlier 9/31
was measured on training.csv before the 2026-08-13 label recovery (+8 rows).

`transit_shape_ratio` is in OPTIONAL_FEATURES and is ~30% NaN, so its
importance is also a statement about how much the model leans on a column it
often does not have.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..")
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT = os.path.join(HERE, "shape_ratio_importance.json")

SEED = 20260812
N_REPEATS = 10
PRIOR_SECONDARY_RANK = 9   # from secondary_retire_results.json


def _m05():
    spec = importlib.util.spec_from_file_location("m05", os.path.join(CODE, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["m05"] = m
    spec.loader.exec_module(m); return m


def model():
    return CalibratedClassifierCV(
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", HistGradientBoostingClassifier(random_state=42))]),
        cv=5, method="sigmoid")


def main():
    m05 = _m05()
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 31
    for c in ("transit_shape_ratio", "secondary_eclipse_depth"):
        assert c in cols, c

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    tr_mask, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)
    print(f"training.csv {len(df)} rows; train {int(tr_mask.sum())}, "
          f"frozen test {int(te.sum())}")
    print(f"transit_shape_ratio NaN rate: {X.transit_shape_ratio.isna().mean():.1%}  "
          f"secondary_eclipse_depth NaN rate: {X.secondary_eclipse_depth.isna().mean():.1%}")

    mo = model(); mo.fit(X[tr_mask], y[tr_mask])
    p = mo.predict_proba(X[te])[:, 1]
    print(f"baseline frozen-test AUC: {roc_auc_score(y[te], p):.4f}")

    pi = permutation_importance(mo, X[te], y[te], n_repeats=N_REPEATS,
                                random_state=SEED, scoring="roc_auc", n_jobs=1)
    order = np.argsort(pi.importances_mean)[::-1]
    rank = {cols[k]: i + 1 for i, k in enumerate(order)}

    print("\nFULL PERMUTATION-IMPORTANCE RANKING (frozen test, roc_auc)")
    print(f"{'rank':>5}  {'feature':<26}{'mean':>10}{'sd':>9}")
    for i, k in enumerate(order, 1):
        mark = "  <==" if cols[k] in ("transit_shape_ratio", "secondary_eclipse_depth") else ""
        print(f"{i:>5}  {cols[k]:<26}{pi.importances_mean[k]:>+10.5f}"
              f"{pi.importances_std[k]:>9.5f}{mark}")

    res = {}
    for c in ("transit_shape_ratio", "secondary_eclipse_depth"):
        k = cols.index(c)
        res[c] = {"rank": rank[c], "mean": float(pi.importances_mean[k]),
                  "sd": float(pi.importances_std[k]),
                  "nan_rate": float(X[c].isna().mean())}
    print("\n" + "=" * 60)
    # A single "load-bearing / not" flag flattens a real difference: both
    # columns clear 2 sd, but one costs 3.7x more than the other and neither
    # costs as much as the ~0.0097 MDE. Report the band, not a binary.
    MDE = 0.0097
    for c, r in res.items():
        if r["mean"] <= 2 * r["sd"]:
            verdict = "INDISTINGUISHABLE FROM NOISE -- safe to retire"
        elif r["mean"] >= MDE:
            verdict = "LOAD-BEARING above the MDE -- a change here is provable"
        else:
            verdict = (f"NON-ZERO BUT SUB-MDE ({r['mean']/MDE:.0%} of {MDE}) -- "
                       f"real contribution, too small to prove a change either way")
        print(f"{c:<26} rank {r['rank']:>2}/31  {r['mean']:+.5f} +/- {r['sd']:.5f}  "
              f"NaN {r['nan_rate']:.0%}\n{'':<26}   -> {verdict}")
    print(f"\n(secondary_eclipse_depth was rank {PRIOR_SECONDARY_RANK}/31 in the "
          f"pre-recovery run; both are re-ranked here in ONE run so they are "
          f"comparable to each other.)")

    json.dump({"seed": SEED, "n_repeats": N_REPEATS, "n_rows": int(len(df)),
               "n_frozen_test": int(te.sum()),
               "baseline_frozen_auc": float(roc_auc_score(y[te], p)),
               "features": res,
               "full_ranking": [{"rank": i + 1, "feature": cols[k],
                                 "mean": float(pi.importances_mean[k]),
                                 "sd": float(pi.importances_std[k])}
                                for i, k in enumerate(order)]},
              open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
