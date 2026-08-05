"""validate_multisector.py -- ITEM 1 validation: leakage checks then the full
comparison, on the frozen split.

n_sectors_measured is deliberately included as a feature. The feasibility check
established that sector COUNT does not predict the label (single-feature AUC
0.479, Mann-Whitney p=0.459), which is what makes it safe to include; it lets
the model discount the two consistency features when they rest on few sectors.
The leakage table below re-checks that on the full population rather than
trusting the 400-star sample.

Same methodology as every other feature experiment in this project: frozen
split via m05.split_by_host (never regenerated), production hyperparameters,
nested CV, calibration, and a paired bootstrap CI that must have ci_lo > 0.
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
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     RandomizedSearchCV)
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import roc_auc_score  # exact drop-in, ~23x faster inside bootstraps

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
FEATURES_CSV = os.path.join(SCRIPT_DIR, "multisector_consistency.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "multisector_consistency_results.json")

RANDOM_SEED = 42
N_BOOTSTRAP = 2000
N_CV = 5
NEW = ["sector_depth_frac_scatter", "sector_depth_chi2red", "n_sectors_measured"]


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


def pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", hgb())])


def boot(y, pa, pb, n=N_BOOTSTRAP, seed=RANDOM_SEED):
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


def ece(y, p, bins=10):
    idx = np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1])
    return float(sum(((idx == b).sum() / len(y)) * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def main():
    res = {}
    m05 = _load_m05()
    df = pd.read_csv(TRAINING_CSV)
    feats = pd.read_csv(FEATURES_CSV)
    ok = feats[feats["status"] == "ok"]
    print(f"feature extraction: {len(ok)}/{len(feats)} usable "
          f"({100*len(ok)/len(feats):.1f}%)")
    res["extraction_ok"] = int(len(ok))
    res["extraction_total"] = int(len(feats))

    df = df.merge(ok[["host"] + NEW], on="host", how="left")
    # BUG FIXED: build_feature_matrix selects only m05.FEATURE_COLUMNS, so
    # merging the new columns into df does NOT put them in X. The first run
    # printed "base 24 -> with new 24" and would have reported a perfectly
    # null result for a comparison that never actually differed. The new
    # columns must be concatenated onto X explicitly.
    X, y = m05.build_feature_matrix(df)
    y = np.asarray(y)
    extra = df[NEW].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = pd.concat([X.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    tr, te = m05.split_by_host(df)
    overlap = set(df.loc[tr, "host"]) & set(df.loc[te, "host"])
    print(f"split: {tr.sum()} train / {te.sum()} test | hosts both sides {len(overlap)}")
    res["hosts_both_sides"] = len(overlap)

    # ---------------- LEAKAGE ----------------
    print("\n" + "=" * 74)
    print("LEAKAGE CHECKS")
    print("=" * 74)
    print(f"  {'feature':<16}{'single-AUC':>12}{'missing%':>10}{'miss(pos)':>11}{'miss(neg)':>11}")
    res["leakage"] = {}
    for f in NEW:
        v = pd.to_numeric(df[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        m = v.notna().to_numpy()
        a = roc_auc_score(y[m], v[m]) if m.sum() > 30 and len(np.unique(y[m])) > 1 else np.nan
        mp = 100 * (~m[y == 1]).mean()
        mn = 100 * (~m[y == 0]).mean()
        flag = "  <-- CHECK" if (a > 0.85 or a < 0.15) else ""
        print(f"  {f:<16}{a:>12.3f}{100*(~m).mean():>10.1f}{mp:>11.1f}{mn:>11.1f}{flag}")
        res["leakage"][f] = {"single_auc": float(a), "pct_missing": float(100*(~m).mean()),
                             "pct_missing_pos": float(mp), "pct_missing_neg": float(mn)}
    # is MISSINGNESS itself predictive? (the dangerous shortcut)
    present = df[NEW[0]].notna().astype(int)
    a_miss = roc_auc_score(y, present)
    res["leakage"]["missingness_indicator_auc"] = float(a_miss)
    print(f"\n  AUC of the missingness indicator alone: {a_miss:.3f} "
          f"({'SAFE direction' if a_miss < 0.5 else 'check: presence predicts positive'})")

    Xb = X.drop(columns=NEW, errors="ignore")
    print(f"\nfeature counts: base {Xb.shape[1]} -> with new {X.shape[1]}")

    # ---------------- MAIN COMPARISON ----------------
    print("\n" + "=" * 74)
    print("COMPARISON (frozen split, production hyperparameters)")
    print("=" * 74)
    cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RANDOM_SEED)
    out = {}
    probs = {}
    for name, Xv in (("base", Xb), ("with_multisector", X)):
        c = cross_validate(pipe(), Xv[tr], y[tr], cv=cv, scoring="roc_auc")
        mdl = pipe().fit(Xv[tr], y[tr])
        p = mdl.predict_proba(Xv[te])[:, 1]
        probs[name] = p
        out[name] = {"cv_mean": float(c["test_score"].mean()),
                     "cv_std": float(c["test_score"].std()),
                     "test_auc": float(roc_auc_score(y[te], p)),
                     "brier": float(brier_score_loss(y[te], p)),
                     "ece": ece(y[te], p)}
        print(f"  {name:18s} CV {c['test_score'].mean():.4f}+/-{c['test_score'].std():.4f}"
              f"  test {out[name]['test_auc']:.4f}  Brier {out[name]['brier']:.4f}"
              f"  ECE {out[name]['ece']:.4f}")
    m_, lo, hi = boot(y[te], probs["base"], probs["with_multisector"])
    clears = lo > 0
    out["delta"] = {"mean": m_, "ci_lower": lo, "ci_upper": hi, "clears_bar": bool(clears)}
    print(f"\n  delta vs base: {m_:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"-> {'CLEARS' if clears else 'DOES NOT CLEAR'}")
    res["comparison"] = out

    # ---------------- NESTED CV ----------------
    print("\n" + "=" * 74)
    print("NESTED CV (outer 5-fold / inner 3-fold RandomizedSearchCV, 15 iters)")
    print("=" * 74)
    grid = {"clf__max_iter": [200, 300, 400], "clf__max_depth": [3, 4, 5, 6],
            "clf__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "clf__min_samples_leaf": [10, 20, 40]}
    res["nested_cv"] = {}
    for name, Xv in (("base", Xb), ("with_multisector", X)):
        inner = StratifiedKFold(3, shuffle=True, random_state=RANDOM_SEED)
        outer = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
        search = RandomizedSearchCV(pipe(), grid, n_iter=15, scoring="roc_auc",
                                    cv=inner, random_state=RANDOM_SEED, n_jobs=-1)
        sc = cross_validate(search, Xv[tr], y[tr], cv=outer, scoring="roc_auc")
        res["nested_cv"][name] = {"mean": float(sc["test_score"].mean()),
                                  "std": float(sc["test_score"].std())}
        print(f"  {name:18s} {sc['test_score'].mean():.4f} +/- {sc['test_score'].std():.4f}")

    # ---------------- CALIBRATION ----------------
    print("\n" + "=" * 74)
    print("CALIBRATION (sigmoid-wrapped, as production deploys it)")
    print("=" * 74)
    res["calibrated"] = {}
    for name, Xv in (("base", Xb), ("with_multisector", X)):
        cal = CalibratedClassifierCV(pipe(), method="sigmoid", cv=3)
        cal.fit(Xv[tr], y[tr])
        p = cal.predict_proba(Xv[te])[:, 1]
        res["calibrated"][name] = {"test_auc": float(roc_auc_score(y[te], p)),
                                   "brier": float(brier_score_loss(y[te], p)),
                                   "ece": ece(y[te], p)}
        print(f"  {name:18s} AUC {roc_auc_score(y[te], p):.4f}  "
              f"Brier {brier_score_loss(y[te], p):.4f}  ECE {ece(y[te], p):.4f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
