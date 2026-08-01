"""small_lift_combined.py -- the one combined run warranted by small_lift_trio.py.

WHICH ARMS ARE COMBINED, AND WHY NOT THE OTHERS

Individual results (delta vs the production config refit on the same split):

    stacked              +0.0039  CI [-0.0022, +0.0106]
    engineered_features  +0.0001  CI [-0.0055, +0.0057]
    weight_none          -0.0022  CI [-0.0083, +0.0039]
    weight_sqrt_inverse  -0.0021  CI [-0.0072, +0.0033]

Only the first two have a positive mean, so only those are combined. Both
weighting variants made things measurably worse, and folding a change that
hurts into a combination in the hope the total clears is not a combination
test -- it is a search for a passing number.

A NOTE ON THE GATE IN small_lift_trio.py, which was too loose: it advanced any
arm whose ci_upper > 0, which is satisfied by an arm with a NEGATIVE mean
(weight_none, mean -0.0022, upper +0.0039). Selecting on the mean is the
correct filter and is what is applied here.

This is ONE run. If it does not clear, that is the answer -- the small-lift
tier is exhausted, and iterating further would be the p-hacking this project
has avoided throughout.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, SCRIPT_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "small_lift_combined_results.json")

RANDOM_SEED = 42
N_BOOTSTRAP = 2000
N_CV_FOLDS = 5


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def hgb():
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        class_weight="balanced", random_state=RANDOM_SEED)


def pipe(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])


def paired_bootstrap(y, p_a, p_b, n=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    return float(sum((( idx == b).sum() / len(y)) * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def main():
    m05 = _load("m05", os.path.join(CODE_DIR, "05_train_models.py"))
    trio = _load("trio", os.path.join(SCRIPT_DIR, "small_lift_trio.py"))

    df = pd.read_csv(TRAINING_CSV)
    X, y = m05.build_feature_matrix(df)
    tr, te = m05.split_by_host(df)
    y = np.asarray(y)
    Xe = trio.add_engineered(X, df)

    print("=" * 76)
    print("COMBINED: stacking (HGB + RF + LR) ON the engineered feature set")
    print("=" * 76)
    print(f"train {tr.sum()} test {te.sum()} | features {X.shape[1]} -> {Xe.shape[1]}")

    ytr, yte = y[tr], y[te]
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # baseline: production config, original features
    base = pipe(hgb()).fit(X[tr], ytr)
    p_base = base.predict_proba(X[te])[:, 1]
    auc_base = roc_auc_score(yte, p_base)
    print(f"\nbaseline (production config, 24 features): test ROC-AUC {auc_base:.4f}")

    bases = {
        "hgb": pipe(hgb()),
        "rf": pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                          random_state=RANDOM_SEED, n_jobs=-1)),
        "lr": Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                                   random_state=RANDOM_SEED))]),
    }
    oof = np.column_stack([
        cross_val_predict(m, Xe[tr], ytr, cv=cv, method="predict_proba")[:, 1]
        for m in bases.values()])
    test_stack = np.column_stack([
        m.fit(Xe[tr], ytr).predict_proba(Xe[te])[:, 1] for m in bases.values()])
    meta = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED).fit(oof, ytr)
    p = meta.predict_proba(test_stack)[:, 1]
    auc = roc_auc_score(yte, p)

    for i, k in enumerate(bases):
        print(f"  base {k:4s} test AUC {roc_auc_score(yte, test_stack[:, i]):.4f}")
    print(f"  meta coefficients: {dict(zip(bases, meta.coef_[0].round(3).tolist()))}")
    print(f"\ncombined: test ROC-AUC {auc:.4f}  Brier {brier_score_loss(yte,p):.4f}  "
          f"ECE {ece(yte,p):.4f}")

    m_, lo, hi = paired_bootstrap(yte, p_base, p)
    clears = lo > 0
    print(f"vs baseline: {m_:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"\nPROMOTION BAR (ci_lo > 0): {'CLEARS' if clears else 'DOES NOT CLEAR'}")

    out = {"auc_base": float(auc_base), "auc_combined": float(auc),
           "brier": float(brier_score_loss(yte, p)), "ece": ece(yte, p),
           "mean_delta": m_, "ci_lower": lo, "ci_upper": hi,
           "clears_bar": bool(clears),
           "meta_coefficients": dict(zip(bases, meta.coef_[0].round(3).tolist()))}
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
