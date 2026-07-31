"""native_nan_vs_imputer.py -- does median-imputing before HistGradientBoosting
throw away real signal?

MOTIVATION, straight from 05_train_models.build_models():

    ("impute", SimpleImputer(strategy="median")),  # HGB can handle NaN natively, but
    # imputing here anyway keeps every model seeing identical input data for a fair comparison

That was the correct call for the three-way model bake-off -- LogisticRegression
and RandomForest cannot take NaN, so imputing everywhere kept the comparison
honest. But HGB won, and the DEPLOYED pipeline still carries the imputer that
only ever existed to be fair to two models that are no longer used.

Why this might matter: HGB's native NaN handling learns, per split, which
direction missing values should go -- so missingness itself becomes usable
signal. Median-imputing erases that, and in this dataset missingness is not
random. FAP is absent when TLS could not compute a false-alarm probability;
transit_shape_ratio is absent when the shape fit failed. Both are properties of
weak or ambiguous detections, i.e. genuinely correlated with the label.

Three arms, identical rows, frozen split, same hyperparameters:
  A. current   : SimpleImputer(median) -> HGB
  B. native    : HGB alone, NaN passed through
  C. indicator : SimpleImputer(median, add_indicator=True) -> HGB

Promotion bar is the project standard and is NOT relaxed: a paired bootstrap CI
on the AUC difference must lie entirely above zero.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, CODE_DIR)
TRAINING_CSV = os.path.join(CODE_DIR, "..", "data", "training_dataset", "training.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "native_nan_results.json")

RANDOM_SEED = 42
N_BOOTSTRAP = 2000
N_CV_FOLDS = 5


def _load_m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def hgb():
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        class_weight="balanced", random_state=RANDOM_SEED)


ARMS = {
    "A_current_imputer": lambda: Pipeline([("impute", SimpleImputer(strategy="median")),
                                           ("clf", hgb())]),
    "B_native_nan":      lambda: Pipeline([("clf", hgb())]),
    "C_add_indicator":   lambda: Pipeline([("impute", SimpleImputer(strategy="median",
                                                                    add_indicator=True)),
                                           ("clf", hgb())]),
}


def paired_bootstrap(y, pa, pb, n=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    print("=" * 72)
    print("Native-NaN vs median-imputation for the production HGB")
    print("=" * 72)

    m05 = _load_m05()
    df = pd.read_csv(TRAINING_CSV)
    X, y = m05.build_feature_matrix(df)
    train_mask, test_mask = m05.split_by_host(df)

    overlap = set(df.loc[train_mask, "host"]) & set(df.loc[test_mask, "host"])
    print(f"rows {len(df)} | split {train_mask.sum()} train / {test_mask.sum()} test | "
          f"hosts on both sides {len(overlap)} (must be 0)")
    nan_rate = X.isna().mean().sort_values(ascending=False)
    print("\nfeatures with the most missing values:")
    for c, v in nan_rate.head(6).items():
        print(f"  {c:28s} {100*v:5.1f}% missing")

    Xtr, Xte = X[train_mask], X[test_mask]
    ytr, yte = y[train_mask], y[test_mask]

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    probas, aucs, cvs = {}, {}, {}
    for name, build in ARMS.items():
        cvres = cross_validate(build(), Xtr, ytr, cv=cv, scoring="roc_auc")
        cvs[name] = (cvres["test_score"].mean(), cvres["test_score"].std())
        model = build().fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        probas[name] = p
        aucs[name] = roc_auc_score(yte, p)
        print(f"\n{name:20s} CV {cvs[name][0]:.4f} +/- {cvs[name][1]:.4f} | "
              f"held-out test ROC-AUC {aucs[name]:.4f}")

    base = "A_current_imputer"
    out = {"n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
           "hosts_on_both_sides": len(overlap),
           "cv": {k: {"mean": v[0], "std": v[1]} for k, v in cvs.items()},
           "test_auc": {k: float(v) for k, v in aucs.items()},
           "comparisons": {}}

    print("\n--- paired bootstrap vs the current production arm ---")
    for name in ARMS:
        if name == base:
            continue
        m, lo, hi = paired_bootstrap(yte, probas[base], probas[name])
        clears = lo > 0
        out["comparisons"][name] = {"mean_diff": m, "ci_lower": lo, "ci_upper": hi,
                                    "clears_promotion_bar": bool(clears)}
        print(f"  {name:20s} mean {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"-> {'CLEARS' if clears else 'does not clear'}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
