"""
09b_validate_uncertainty.py -- sanity-checks the bootstrap uncertainty bands.

The specific thing being tested: uncertainty should be WIDEST near the
decision boundary (p ~ 0.5) and NARROWEST at confident extremes (p near 0 or
1). That is not a stylistic preference -- it is what training-resample
variance has to look like if the implementation is correct. Probabilities are
bounded at 0 and 1, so members cannot disagree much about a candidate every
one of them scores 0.99; near 0.5 they are free to land on either side.

If that pattern does NOT hold, something is wrong -- e.g. members that are
near-identical (bootstrap not actually resampling), or a band dominated by
calibration noise rather than by resample variance.

Run on the HELD-OUT TEST SPLIT, which no ensemble member was trained on, so
the check is not measuring memorised training rows.
"""
import importlib.util
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "web"))
import uncertainty  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_models", os.path.join(SCRIPT_DIR, "05_train_models.py"))
train_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_models)


def main():
    if not uncertainty.ensemble_available():
        raise SystemExit("No ensemble -- run 09_build_bootstrap_ensemble.py first.")

    production = joblib.load(os.path.join(train_models.MODELS_DIR, "best_model.joblib"))
    df = train_models.load_and_report_class_balance()
    X, y = train_models.build_feature_matrix(df)
    _, test_mask = train_models.split_by_host(df)
    Xte, yte = X[test_mask], y[test_mask]

    prod_p = production.predict_proba(Xte)[:, 1]
    recs = uncertainty.predict_with_uncertainty(Xte, production_probability=prod_p)
    std = np.array([r["uncertainty_std"] for r in recs])
    mean = np.array([r["bootstrap_mean"] for r in recs])
    disagree = np.array([r["disagreement_sigma"] for r in recs])

    print("\n" + "=" * 68)
    print("UNCERTAINTY VALIDATION -- held-out test split "
          f"({len(Xte)} stars, never seen by any ensemble member)")
    print("=" * 68)
    print(f"ensemble members: {recs[0]['uncertainty_n_members']}")
    print(f"band width (std): median {np.median(std):.4f}, "
          f"min {std.min():.4f}, max {std.max():.4f}")

    print("\n1. Does uncertainty widen near the decision boundary?")
    print("   (distance from 0.5 vs mean band width)\n")
    dist = np.abs(prod_p - 0.5)
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
    prev = None
    monotonic = True
    for lo, hi in bins:
        m = (dist >= lo) & (dist < hi)
        if m.sum() == 0:
            continue
        w = std[m].mean()
        print(f"   |p-0.5| in [{lo:.1f},{hi:.1f})  n={m.sum():4d}   mean band +/- {w:.4f}")
        if prev is not None and w > prev + 1e-9:
            monotonic = False
        prev = w
    print(f"\n   -> band narrows monotonically as confidence grows: "
          f"{'YES (expected)' if monotonic else 'NO -- INVESTIGATE'}")

    corr = np.corrcoef(dist, std)[0, 1]
    print(f"   -> correlation(distance from 0.5, band width) = {corr:+.3f} "
          f"({'expected: strongly negative' if corr < -0.3 else 'WEAKER THAN EXPECTED -- INVESTIGATE'})")

    print("\n2. Do the production score and the ensemble centre agree?")
    print(f"   median |production - bootstrap mean| = {np.median(np.abs(prod_p-mean)):.4f}")
    print(f"   candidates >3 sigma apart: {int((disagree > 3).sum())}/{len(disagree)}")

    print("\n3. Is the band informative about being WRONG?")
    pred = (prod_p >= 0.5).astype(int)
    correct = pred == yte.to_numpy()
    print(f"   mean band on CORRECT predictions:   +/- {std[correct].mean():.4f}")
    print(f"   mean band on INCORRECT predictions: +/- {std[~correct].mean():.4f}")
    ratio = std[~correct].mean() / std[correct].mean()
    print(f"   -> wrong predictions carry a {ratio:.2f}x wider band "
          f"({'useful signal' if ratio > 1.15 else 'little discriminating power -- report honestly'})")

    out = os.path.join(train_models.TABLES_DIR, "uncertainty_validation.csv")
    pd.DataFrame({"production_p": prod_p, "bootstrap_mean": mean, "std": std,
                  "label": yte.to_numpy(), "correct": correct}).to_csv(out, index=False)
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
