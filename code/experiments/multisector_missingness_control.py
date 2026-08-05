"""multisector_missingness_control.py -- the control test that decides whether
Item 1's result is physics or bookkeeping.

WHY THIS EXISTS

The multi-sector consistency features cleared the promotion bar (+0.0094, CI
[+0.0008, +0.0177]). But their missingness is strongly asymmetric: 19.1% of
positive-class stars lack the features versus 3.5% of negatives, and the
missingness indicator alone scores AUC 0.422. That asymmetry is not
astrophysical -- it comes from 494 positive-class stars having no resolvable
TIC ID, a property of this project's name-to-TIC cross-match, not of the stars.

So there are two competing explanations for the gain:

  (A) PHYSICS. Depths that disagree across sectors beyond measurement noise
      indicate a false positive, and sector_depth_chi2red (single-feature AUC
      0.312) is measuring exactly that.

  (B) BOOKKEEPING. The model learns "these features are missing -> probably a
      confirmed planet", which is an artefact of TIC resolution and would
      transfer to no new star whatsoever.

These are distinguishable. Fit a model with ONLY a binary missingness
indicator -- no measured values at all -- and see how much of the +0.0094 it
reproduces. If most of it survives on the indicator alone, the result is (B)
and must not be promoted. If the indicator alone gains little while the real
values gain a lot, the result is (A).

A third arm isolates it further: real feature VALUES with the missingness
pattern held constant, by restricting to rows where the features exist.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import roc_auc_score  # exact drop-in, ~23x faster inside bootstraps

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
FEATURES_CSV = os.path.join(SCRIPT_DIR, "multisector_consistency.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "multisector_missingness_control.json")

RANDOM_SEED = 42
N_BOOTSTRAP = 2000
NEW = ["sector_depth_frac_scatter", "sector_depth_chi2red", "n_sectors_measured"]


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", HistGradientBoostingClassifier(
                         max_iter=300, max_depth=4, learning_rate=0.05,
                         class_weight="balanced", random_state=RANDOM_SEED))])


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


def main():
    res = {}
    m05 = _m05()
    df = pd.read_csv(TRAINING_CSV)
    feats = pd.read_csv(FEATURES_CSV)
    ok = feats[feats["status"] == "ok"]
    df = df.merge(ok[["host"] + NEW], on="host", how="left")

    Xb, y = m05.build_feature_matrix(df)
    y = np.asarray(y)
    tr, te = m05.split_by_host(df)
    vals = df[NEW].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Xb = Xb.reset_index(drop=True)
    vals = vals.reset_index(drop=True)

    present = vals["sector_depth_chi2red"].notna().astype(int)
    X_ind = pd.concat([Xb, present.rename("multisector_present")], axis=1)
    X_full = pd.concat([Xb, vals], axis=1)

    print("=" * 74)
    print("CONTROL: is the gain physics, or the missingness pattern?")
    print("=" * 74)
    print(f"  missing: {100*(1-present.mean()):.1f}% overall | "
          f"positives {100*(1-present[y==1].mean()):.1f}% | "
          f"negatives {100*(1-present[y==0].mean()):.1f}%")
    print(f"  missingness-indicator single-feature AUC: {roc_auc_score(y, present):.3f}")

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    probs, aucs = {}, {}
    for name, Xv in (("base", Xb), ("indicator_only", X_ind), ("full_features", X_full)):
        c = cross_validate(pipe(), Xv[tr], y[tr], cv=cv, scoring="roc_auc")
        p = pipe().fit(Xv[tr], y[tr]).predict_proba(Xv[te])[:, 1]
        probs[name], aucs[name] = p, roc_auc_score(y[te], p)
        print(f"\n  {name:16s} CV {c['test_score'].mean():.4f}  test {aucs[name]:.4f}")
        if name != "base":
            m_, lo, hi = boot(y[te], probs["base"], p)
            res[name] = {"test_auc": float(aucs[name]), "mean_delta": m_,
                         "ci_lower": lo, "ci_upper": hi, "clears": bool(lo > 0)}
            print(f"     vs base: {m_:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"-> {'CLEARS' if lo > 0 else 'does not clear'}")
    res["base_auc"] = float(aucs["base"])

    # ---- restricted population: values only, missingness held constant ----
    print("\n" + "-" * 74)
    print("RESTRICTED to stars that HAVE the features (missingness constant)")
    print("-" * 74)
    has = present.to_numpy().astype(bool)
    tr2, te2 = tr & has, te & has
    print(f"  train {tr2.sum()} / test {te2.sum()} "
          f"({int(y[te2].sum())} positive / {int((1-y[te2]).sum())} negative)")
    sub = {}
    for name, Xv in (("base", Xb), ("full_features", X_full)):
        p = pipe().fit(Xv[tr2], y[tr2]).predict_proba(Xv[te2])[:, 1]
        sub[name] = p
        print(f"  {name:16s} test {roc_auc_score(y[te2], p):.4f}")
    m_, lo, hi = boot(y[te2], sub["base"], sub["full_features"])
    res["restricted"] = {"n_test": int(te2.sum()), "mean_delta": m_,
                         "ci_lower": lo, "ci_upper": hi, "clears": bool(lo > 0)}
    print(f"  delta on this population: {m_:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"-> {'CLEARS' if lo > 0 else 'does not clear'}")

    print("\n" + "=" * 74)
    ind_d = res["indicator_only"]["mean_delta"]
    full_d = res["full_features"]["mean_delta"]
    frac = (ind_d / full_d * 100) if full_d else float("nan")
    print(f"indicator alone reproduces {frac:.0f}% of the full-feature gain "
          f"({ind_d:+.4f} of {full_d:+.4f})")
    res["indicator_fraction_of_gain_pct"] = float(frac)
    if frac > 60:
        verdict = ("MOSTLY BOOKKEEPING -- the gain is largely the missingness "
                   "pattern, not the measurements. Do not promote.")
    elif frac > 30:
        verdict = ("MIXED -- a substantial part of the gain is the missingness "
                   "pattern. Treat with caution.")
    else:
        verdict = ("MOSTLY PHYSICS -- the measured values carry the gain, not "
                   "the missingness pattern.")
    res["verdict"] = verdict
    print(verdict)
    print("=" * 74)

    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
