"""small_lift_trio.py -- three "small lift" ideas tested independently against
the production configuration, then combined if any shows signal.

Nothing here is a new data source: class weighting is a fit parameter, the
engineered features are ratios of the existing 24, and stacking recombines
models already in the codebase.

TWO PREMISES CORRECTED BEFORE ANY OF THIS RAN
---------------------------------------------
1. Class weighting is NOT unaddressed. 05_train_models.build_models() already
   passes class_weight="balanced" to all three models, and the deployed
   artifact confirms it (CalibratedClassifierCV -> Pipeline -> HGB with
   class_weight='balanced'). So arm 1 is not "add weighting"; it is "is
   'balanced' the right choice, or is it overcorrecting a 3.5:1 imbalance?"
   Tested against no weighting and against sqrt-inverse-frequency.
   (sklearn's HistGradientBoostingClassifier supports class_weight natively
   since 1.2 -- no sample_weight workaround is required.)

2. A stacked ensemble has been tried before (RESULTS_SUMMARY.md Part C): HGB +
   GP + CNN with a logistic meta-learner gave 0.9018 vs 0.9016 for HGB alone
   -- noise -- and the meta-learner's coefficients showed HGB dominating
   (4.13 vs 2.18 vs 2.33). This arm uses a different roster (HGB + RF + LR),
   so it is not a duplicate, but the prior result is the relevant prior:
   RF is MORE correlated with HGB than GP or CNN were, so if anything it has
   less independent signal to contribute. Stated up front rather than
   discovered afterwards.

METHODOLOGY
-----------
The frozen split is used via m05.split_by_host and is never regenerated.
Baseline is the production configuration REFIT on the same training split, so
base and challenger differ by exactly one thing. The deployed artifact's own
test AUC is also reported for context, but it was trained on an older
training.csv snapshot, so it is not the apples-to-apples comparator.

Bar is the project standard, unchanged: paired bootstrap on the ROC-AUC delta
must have ci_lo > 0. Each arm gets ONE honest attempt -- no iterating until
something clears, which would be the p-hacking this project has explicitly
avoided elsewhere.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
MODEL_PATH = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "small_lift_trio_results.json")

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


def hgb(class_weight="balanced"):
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        class_weight=class_weight, random_state=RANDOM_SEED)


def pipe(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])


def paired_bootstrap(y, p_a, p_b, n=N_BOOTSTRAP, seed=RANDOM_SEED):
    """CI on AUC(challenger) - AUC(base). Positive = challenger better."""
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
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(tot)


# ---------------------------------------------------------------- features
def add_engineered(X, df):
    """Four ratios, each with a stated physical reason. Deliberately few --
    'dozens' would be fishing, and every extra column is another chance for
    one to clear the bar by luck.

    1. duty_cycle = duration / period
       Fraction of the orbit spent in transit. Geometry ties this to a/R*: a
       planet on a wide orbit occults its star briefly. An anomalously long
       duty cycle is characteristic of an eclipsing binary, especially on a
       giant -- which is exactly the regime the error analysis found the model
       least reliable in (21% error vs 5.8%). duration and period are both
       features already, but their RATIO is the physically meaningful quantity
       and a tree has to spend splits to approximate it.

    2. duration_vs_expected = duration / (period**(1/3) * st_rad)
       For a central transit, T is proportional to P^(1/3) * R_star / M^(1/3).
       This asks whether the observed duration matches what the orbit and star
       imply. A large value means the event lasts far longer than geometry
       allows for a planet -- a grazing binary or a systematic.

    3. secondary_ratio = secondary_eclipse_depth / depth_mean
       A planet reflects almost nothing, so its secondary eclipse is ~0
       relative to the primary. A stellar companion produces a substantial
       one. Both depths are already features; the ratio is what separates the
       classes, and it is scale-free where the raw depths are not.

    4. odd_even_significance = odd_even_mismatch / depth_mean_std
       Turns a raw odd-vs-even depth difference into a significance by
       dividing by the measured scatter. An EB detected at half its true
       period shows alternating depths; the raw mismatch alone cannot say
       whether the difference exceeds noise, and this can.
    """
    X = X.copy()
    period = df["period"].replace(0, np.nan)
    X["duty_cycle"] = df["duration"] / period
    X["duration_vs_expected"] = df["duration"] / (
        np.cbrt(period) * df["st_rad"].replace(0, np.nan))
    X["secondary_ratio"] = df["secondary_eclipse_depth"] / df["depth_mean"].replace(0, np.nan)
    X["odd_even_significance"] = df["odd_even_mismatch"] / df["depth_mean_std"].replace(0, np.nan)
    return X.replace([np.inf, -np.inf], np.nan)


NEW_FEATURES = ["duty_cycle", "duration_vs_expected", "secondary_ratio",
                "odd_even_significance"]


def main():
    res = {}
    m05 = _load_m05()
    df = pd.read_csv(TRAINING_CSV)
    X, y = m05.build_feature_matrix(df)
    tr, te = m05.split_by_host(df)
    y = np.asarray(y)

    overlap = set(df.loc[tr, "host"]) & set(df.loc[te, "host"])
    print("=" * 76)
    print("SMALL-LIFT TRIO -- class weighting / engineered features / stacking")
    print("=" * 76)
    print(f"rows {len(df)} | train {tr.sum()} test {te.sum()} | "
          f"hosts on both sides {len(overlap)} (must be 0)")
    print(f"class balance: {int(y.sum())} positive / {int((1-y).sum())} negative "
          f"= {y.sum()/(1-y).sum():.2f}:1")
    res["n_train"], res["n_test"] = int(tr.sum()), int(te.sum())
    res["hosts_both_sides"] = len(overlap)

    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    def evaluate(name, model, Xtr_, Xte_, store):
        cvres = cross_validate(model, Xtr_, ytr, cv=cv, scoring="roc_auc")
        fitted = model.fit(Xtr_, ytr)
        p = fitted.predict_proba(Xte_)[:, 1]
        auc = roc_auc_score(yte, p)
        store[name] = {"cv_mean": float(cvres["test_score"].mean()),
                       "cv_std": float(cvres["test_score"].std()),
                       "test_auc": float(auc), "brier": float(brier_score_loss(yte, p)),
                       "ece": ece(yte, p)}
        print(f"  {name:26s} CV {cvres['test_score'].mean():.4f}+/-{cvres['test_score'].std():.4f}"
              f"  test {auc:.4f}  Brier {brier_score_loss(yte,p):.4f}  ECE {ece(yte,p):.4f}")
        return p

    # deployed artifact, for context only
    deployed = joblib.load(MODEL_PATH)
    p_deployed = deployed.predict_proba(Xte)[:, 1]
    print(f"\ndeployed artifact test ROC-AUC: {roc_auc_score(yte, p_deployed):.4f} "
          f"(trained on an older training.csv -- context, not the comparator)")
    res["deployed_auc"] = float(roc_auc_score(yte, p_deployed))

    # ---------------- BASELINE ----------------
    print("\n" + "-" * 76)
    print("BASELINE = production config refit on this split (HGB, class_weight='balanced')")
    print("-" * 76)
    res["arms"] = {}
    p_base = evaluate("base_balanced", pipe(hgb("balanced")), Xtr, Xte, res["arms"])

    def compare(name, p):
        m, lo, hi = paired_bootstrap(yte, p_base, p)
        clears = lo > 0
        res["arms"][name].update({"mean_delta": m, "ci_lower": lo, "ci_upper": hi,
                                  "clears_bar": bool(clears)})
        print(f"     vs base: {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"-> {'CLEARS' if clears else 'does not clear'}")

    # ---------------- 1. CLASS WEIGHTING ----------------
    print("\n" + "=" * 76)
    print("ITEM 1: CLASS WEIGHTING  (production ALREADY uses 'balanced')")
    print("=" * 76)
    p_none = evaluate("weight_none", pipe(hgb(None)), Xtr, Xte, res["arms"])
    compare("weight_none", p_none)

    n_pos, n_neg = ytr.sum(), (1 - ytr).sum()
    sqrt_w = {0: float(np.sqrt(len(ytr) / (2 * n_neg))),
              1: float(np.sqrt(len(ytr) / (2 * n_pos)))}
    print(f"  sqrt-inverse-frequency weights: {{0: {sqrt_w[0]:.3f}, 1: {sqrt_w[1]:.3f}}} "
          f"(vs balanced {{0: {len(ytr)/(2*n_neg):.3f}, 1: {len(ytr)/(2*n_pos):.3f}}})")
    p_sqrt = evaluate("weight_sqrt_inverse", pipe(hgb(sqrt_w)), Xtr, Xte, res["arms"])
    compare("weight_sqrt_inverse", p_sqrt)

    # ---------------- 2. ENGINEERED FEATURES ----------------
    print("\n" + "=" * 76)
    print("ITEM 2: ENGINEERED INTERACTION FEATURES (4, each physically motivated)")
    print("=" * 76)
    Xe = add_engineered(X, df)
    print("  leakage check -- single-feature AUC on rows where the feature exists")
    print("  (a value near 0.5 is fine; near 1.0 would mean a label proxy):")
    res["feature_leakage"] = {}
    for f in NEW_FEATURES:
        v = Xe[f]
        m = v.notna().to_numpy()
        a = roc_auc_score(y[m], v[m]) if len(np.unique(y[m])) > 1 else float("nan")
        res["feature_leakage"][f] = {"single_feature_auc": float(a),
                                     "pct_missing": float(100 * (~m).mean())}
        flag = "  <-- CHECK" if (a > 0.85 or a < 0.15) else ""
        print(f"    {f:24s} AUC {a:.3f}   missing {100*(~m).mean():4.1f}%{flag}")
    p_feat = evaluate("engineered_features", pipe(hgb("balanced")), Xe[tr], Xe[te], res["arms"])
    compare("engineered_features", p_feat)

    # ---------------- 3. STACKING ----------------
    print("\n" + "=" * 76)
    print("ITEM 3: STACKED ENSEMBLE (HGB + RF + LR, LR meta-learner on OOF preds)")
    print("=" * 76)
    bases = {
        "hgb": pipe(hgb("balanced")),
        "rf": pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                          random_state=RANDOM_SEED, n_jobs=-1)),
        "lr": Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                                   random_state=RANDOM_SEED))]),
    }
    # OOF predictions on TRAIN feed the meta-learner -- never in-sample, or the
    # meta-learner learns to trust a base model that has memorised the row.
    oof = np.column_stack([
        cross_val_predict(mdl, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        for mdl in bases.values()])
    test_stack = np.column_stack([
        mdl.fit(Xtr, ytr).predict_proba(Xte)[:, 1] for mdl in bases.values()])
    for i, k in enumerate(bases):
        print(f"  base {k:4s} OOF AUC {roc_auc_score(ytr, oof[:, i]):.4f}   "
              f"test AUC {roc_auc_score(yte, test_stack[:, i]):.4f}")
    corr = np.corrcoef(test_stack.T)
    print(f"  base-model correlation on test: hgb-rf {corr[0,1]:.3f}, "
          f"hgb-lr {corr[0,2]:.3f}, rf-lr {corr[1,2]:.3f}")
    meta = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED).fit(oof, ytr)
    p_stack = meta.predict_proba(test_stack)[:, 1]
    auc_stack = roc_auc_score(yte, p_stack)
    res["arms"]["stacked"] = {"test_auc": float(auc_stack),
                              "brier": float(brier_score_loss(yte, p_stack)),
                              "ece": ece(yte, p_stack),
                              "meta_coefficients": dict(zip(bases, meta.coef_[0].round(3).tolist())),
                              "base_correlations": {"hgb_rf": float(corr[0,1]),
                                                    "hgb_lr": float(corr[0,2]),
                                                    "rf_lr": float(corr[1,2])}}
    print(f"  meta-learner coefficients: "
          f"{dict(zip(bases, meta.coef_[0].round(3).tolist()))}")
    print(f"  {'stacked':26s} test {auc_stack:.4f}  "
          f"Brier {brier_score_loss(yte,p_stack):.4f}  ECE {ece(yte,p_stack):.4f}")
    compare("stacked", p_stack)

    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")

    # ---------------- COMBINED, only if warranted ----------------
    promising = [k for k, v in res["arms"].items()
                 if k != "base_balanced" and v.get("ci_upper", -1) > 0]
    print("\n" + "=" * 76)
    if promising:
        print(f"arms whose CI even reaches above zero: {promising}")
        print("-> a combined run is justified; see small_lift_combined.py")
    else:
        print("NO arm's confidence interval even reaches above zero.")
        print("-> a combined version is not justified: combining changes that each")
        print("   measurably fail cannot produce a real gain, and running it anyway")
        print("   until something clears is exactly the p-hacking to avoid.")
    print("=" * 76)


if __name__ == "__main__":
    main()
