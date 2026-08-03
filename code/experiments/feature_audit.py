"""feature_audit.py -- what is the production model actually using?

Descriptive half of the feature-selection experiment. Useful regardless of
whether pruning helps, because it says which of the 24 features carry signal,
which are redundant with each other, and which have coverage problems.

Three views:
  1. PERMUTATION importance on the frozen clean test set, measured on the real
     deployed artifact. Permutation rather than impurity/gain: impurity
     importance is biased toward high-cardinality continuous features and is
     computed on training data, so it rewards a feature for being splittable
     rather than for being useful. Permutation measures the actual drop in test
     AUC when a column is shuffled.
  2. PAIRWISE correlation, to find redundancy. Reported as the worst offenders
     rather than a 24x24 wall.
  3. COVERAGE -- missing rate overall and by class, since a feature that is
     absent for a third of stars carries less than its importance suggests, and
     class-asymmetric missingness is the failure mode that already produced one
     artefact in this project (the multi-sector result).

Read-only: loads the deployed model, never refits or writes it.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "feature_audit_results.json")

RANDOM_SEED = 42


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def main():
    res = {}
    m05 = _m05()
    df = pd.read_csv(TRAINING_CSV)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    tr, te = m05.split_by_host(df)
    tr, te = np.asarray(tr), np.asarray(te)
    Xte, yte = X[te], y[te]

    prod = joblib.load(PROD)
    base_auc = roc_auc_score(yte, prod.predict_proba(Xte)[:, 1])
    print("=" * 78)
    print(f"FEATURE AUDIT -- production model, clean test set (n={te.sum()})")
    print("=" * 78)
    print(f"baseline test ROC-AUC: {base_auc:.4f}\n")
    res["baseline_test_auc"] = float(base_auc)

    print("computing permutation importance (10 repeats)...")
    pi = permutation_importance(prod, Xte, yte, n_repeats=10,
                                random_state=RANDOM_SEED, scoring="roc_auc",
                                n_jobs=-1)
    imp = pd.DataFrame({"feature": X.columns,
                        "drop_in_auc": pi.importances_mean,
                        "std": pi.importances_std}).sort_values(
        "drop_in_auc", ascending=False)

    # coverage
    miss, miss_p, miss_n = {}, {}, {}
    for c in X.columns:
        v = pd.to_numeric(X[c], errors="coerce")
        m = v.isna().to_numpy()
        miss[c] = 100 * m.mean()
        miss_p[c] = 100 * m[y == 1].mean()
        miss_n[c] = 100 * m[y == 0].mean()
    imp["missing_pct"] = imp["feature"].map(miss)
    imp["miss_pos"] = imp["feature"].map(miss_p)
    imp["miss_neg"] = imp["feature"].map(miss_n)

    # single-feature AUC, for context on direction/strength
    sfa = {}
    for c in X.columns:
        v = pd.to_numeric(X[c], errors="coerce")
        mk = v.notna().to_numpy() & te
        if mk.sum() > 30 and len(np.unique(y[mk])) > 1:
            sfa[c] = float(roc_auc_score(y[mk], v[mk]))
    imp["single_auc"] = imp["feature"].map(sfa)

    print(f"\n  {'feature':<24}{'perm drop':>11}{'+/-':>8}{'miss%':>8}"
          f"{'miss pos/neg':>14}{'1-feat AUC':>12}")
    for _, r in imp.iterrows():
        sa = f"{r.single_auc:.3f}" if pd.notna(r.single_auc) else "  --"
        print(f"  {r.feature:<24}{r.drop_in_auc:>+11.4f}{r['std']:>8.4f}"
              f"{r.missing_pct:>8.1f}{f'{r.miss_pos:.0f}/{r.miss_neg:.0f}':>14}{sa:>12}")
    res["permutation_importance"] = imp.to_dict("records")

    n_zero = int((imp["drop_in_auc"] <= 0).sum())
    print(f"\n  features with ZERO or NEGATIVE permutation importance: {n_zero} of {len(imp)}")
    print(f"  top-5 carry {100*imp['drop_in_auc'].head(5).sum()/imp['drop_in_auc'].clip(lower=0).sum():.0f}%"
          f" of total positive importance")
    res["n_zero_or_negative_importance"] = n_zero

    # ---- correlations ----
    print("\n" + "=" * 78)
    print("PAIRWISE CORRELATION (train split, |r| >= 0.80)")
    print("=" * 78)
    Xtr = X[tr].apply(pd.to_numeric, errors="coerce")
    corr = Xtr.corr(method="spearman").abs()
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if np.isfinite(r) and r >= 0.80:
                pairs.append((cols[i], cols[j], float(r)))
    pairs.sort(key=lambda t: -t[2])
    if pairs:
        for a, b, r in pairs:
            print(f"  {r:.3f}   {a}  <->  {b}")
    else:
        print("  none at |r| >= 0.80")
    print(f"\n  redundant pairs found: {len(pairs)}")
    res["correlated_pairs"] = [{"a": a, "b": b, "spearman_abs": r} for a, b, r in pairs]

    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
