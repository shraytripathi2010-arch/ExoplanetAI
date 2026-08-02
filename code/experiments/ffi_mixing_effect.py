"""ffi_mixing_effect.py -- does mixing in COARSE-cadence (FFI-like) data help
or hurt? Answered with data already on disk.

`cadence_audit.py` found the training set is already mixed-cadence. A first
pass at this script then made a real error, corrected here and recorded rather
than quietly fixed:

    It grouped everything that was not 2-min into one "non-2-min" bucket and
    treated that as the FFI proxy. 401 of those 632 rows are SUB-MINUTE
    (20-second SPOC) -- FINER cadence than 2-min, i.e. higher quality, the
    opposite of FFI. The genuinely coarse population is only 231 rows. Every
    conclusion drawn from the mixed bucket was therefore about 20-second data
    as much as about FFI.

CORRECTED GROUPING (measured from the light curves themselves, not filenames):

    FINE   (< 1 min)        401 rows   20s SPOC -- better than 2-min
    2-min  (1.0-2.6 min)  4,852 rows   the standard product
    COARSE (> 2.6 min)      231 rows   200s / 10-min / 30-min, FFI-derived

COARSE is strongly class-asymmetric: 14.0% of negatives vs 1.6% of positives.

THE NATURAL EXPERIMENT
Those 231 COARSE rows are real stars with real labels already run through the
identical pipeline. If including them helps, FFI is a lever. If not, adding
thousands more of the same kind will not help either.

ARMS, all on the frozen clean split:
  A. train on 2-min + COARSE          (FINE rows excluded from every arm, so
  B. train on 2-min only               the comparison isolates COARSE alone)
  C. A with COARSE down-weighted 0.25

SECOND CORRECTION: arm C originally passed sample_weight to a
CalibratedClassifierCV wrapping a Pipeline. sklearn warned that it "does not
appear to accept sample_weight [so] sample weights will only be used for the
calibration itself" -- the down-weighting never reached the classifier and arm
C was silently identical to arm A. All arms now fit the BARE production
Pipeline (SimpleImputer + HGB with production hyperparameters), which does
accept `clf__sample_weight`. Sigmoid calibration is monotonic, so dropping the
wrapper does not affect AUC comparisons.

Evaluated on two populations: the full clean test set, and its 2-min subset --
the population we actually care about, where distribution damage would show up
undiluted rather than being masked by the model simply learning to serve the
coarse rows in the test set.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE_CSV = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "ffi_mixing_results.json")

RANDOM_SEED = 42
N_BOOT = 2000
COARSE_MIN = 2.6      # minutes; above this is FFI-derived
FINE_MAX = 1.0        # minutes; below this is 20s SPOC


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def boot(y, pa, pb, n=N_BOOT, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
    d = np.array(d)
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


def production_bare_pipeline():
    """Production's hyperparameters WITHOUT the calibration wrapper, so
    sample_weight actually reaches the classifier."""
    prod = joblib.load(PROD)
    inner = getattr(prod, "estimator", prod)
    if hasattr(inner, "named_steps"):
        return clone(inner)
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", HistGradientBoostingClassifier(
                         max_iter=500, max_leaf_nodes=63, l2_regularization=0.5,
                         class_weight="balanced", random_state=RANDOM_SEED))])


def main():
    res = {}
    m05 = _m05()
    df = pd.read_csv(TRAINING_CSV).merge(
        pd.read_csv(CADENCE_CSV), on="host", how="left")
    c = pd.to_numeric(df["cadence_min"], errors="coerce")
    is_fine = (c < FINE_MAX).to_numpy()
    is_coarse = (c > COARSE_MIN).to_numpy()
    is_2min = ~is_fine & ~is_coarse          # includes unknown -> default product

    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True); y = np.asarray(y)
    tr, te = m05.split_by_host(df)
    tr = np.asarray(tr); te = np.asarray(te)

    print("=" * 78)
    print("CADENCE GROUPS (corrected)")
    print("=" * 78)
    for nm, m in (("FINE (20s SPOC)", is_fine), ("2-min SPOC", is_2min),
                  ("COARSE (FFI-like)", is_coarse)):
        print(f"  {nm:20s} {m.sum():5d} rows | "
              f"{int((y[m]==1).sum()):5d} pos / {int((y[m]==0).sum()):5d} neg")
    res["n_fine"] = int(is_fine.sum()); res["n_2min"] = int(is_2min.sum())
    res["n_coarse"] = int(is_coarse.sum())

    # ---- domain classifier, COARSE vs 2-min only ----
    print("\n" + "=" * 78)
    print("DOMAIN CLASSIFIER: 2-min vs COARSE (the number that decides this)")
    print("=" * 78)
    dom_mask = is_2min | is_coarse
    dom = is_coarse[dom_mask].astype(int)
    pipe = Pipeline([("i", SimpleImputer(strategy="median")),
                     ("c", HistGradientBoostingClassifier(
                         max_iter=300, max_depth=4, learning_rate=0.05,
                         class_weight="balanced", random_state=RANDOM_SEED))])
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    pdom = cross_val_predict(pipe, X[dom_mask], dom, cv=cv,
                             method="predict_proba")[:, 1]
    dauc = roc_auc_score(dom, pdom)
    print(f"  AUC {dauc:.4f}   ({int((dom==0).sum())} 2-min vs {int(dom.sum())} COARSE)")
    print("  reference: synthetic-vs-real 0.9654 (mixing HURT); 0.5 = indistinguishable")
    res["domain_auc_2min_vs_coarse"] = float(dauc)

    smd = {}
    for col in X.columns:
        a = X.loc[is_2min, col].astype(float); b = X.loc[is_coarse, col].astype(float)
        sd = np.sqrt((a.var() + b.var()) / 2)
        if np.isfinite(sd) and sd > 0:
            smd[col] = float((b.mean() - a.mean()) / sd)
    top = sorted(smd.items(), key=lambda kv: -abs(kv[1]))[:8]
    print("\n  largest standardized mean differences (COARSE - 2min):")
    for k, v in top:
        print(f"    {k:24s} {v:+.2f}")
    print(f"  features shifted > 0.5 SD: {sum(1 for v in smd.values() if abs(v) > 0.5)}"
          f" of {len(smd)}")
    res["standardized_mean_diff"] = smd
    res["n_features_shifted_gt_0.5sd"] = int(sum(1 for v in smd.values() if abs(v) > 0.5))

    # ---- arms: FINE excluded everywhere so COARSE is isolated ----
    base = ~is_fine
    trA = tr & base
    trB = tr & base & ~is_coarse
    print("\n" + "=" * 78)
    print("ARMS (FINE rows excluded from all arms, isolating COARSE)")
    print("=" * 78)
    print(f"  A train: {trA.sum()} (incl {int((trA & is_coarse).sum())} COARSE)")
    print(f"  B train: {trB.sum()} (2-min only)")
    res["n_train_A"] = int(trA.sum()); res["n_train_B"] = int(trB.sum())
    res["n_coarse_train"] = int((trA & is_coarse).sum())

    proto = production_bare_pipeline()

    def fit(mask, w=None):
        m = clone(proto)
        kw = {"clf__sample_weight": w} if w is not None else {}
        m.fit(X[mask], y[mask], **kw)
        return m

    mA = fit(trA)
    mB = fit(trB)
    wC = np.where(is_coarse[trA], 0.25, 1.0)
    mC = fit(trA, w=wC)

    res["arms"] = {}
    for pop_name, pop in (("full_clean_test", te & base),
                          ("2min_only_test", te & is_2min)):
        yy = y[pop]
        pA = mA.predict_proba(X[pop])[:, 1]
        pB = mB.predict_proba(X[pop])[:, 1]
        pC = mC.predict_proba(X[pop])[:, 1]
        print("\n" + "-" * 78)
        print(f"  evaluated on {pop_name}: n={pop.sum()} "
              f"({int(yy.sum())} pos / {int((1-yy).sum())} neg, "
              f"{int((pop & is_coarse).sum())} COARSE)")
        print("-" * 78)
        print(f"    A. 2-min + COARSE            AUC {roc_auc_score(yy, pA):.4f}")
        print(f"    B. 2-min only                AUC {roc_auc_score(yy, pB):.4f}")
        print(f"    C. COARSE down-weighted .25  AUC {roc_auc_score(yy, pC):.4f}")
        m_, lo, hi = boot(yy, pB, pA)
        m2, lo2, hi2 = boot(yy, pB, pC)
        print(f"\n    adding COARSE (B->A):        {m_:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]"
              f"  -> {'CLEARS' if lo > 0 else 'does not clear'}")
        print(f"    adding down-weighted (B->C): {m2:+.4f}  CI [{lo2:+.4f}, {hi2:+.4f}]"
              f"  -> {'CLEARS' if lo2 > 0 else 'does not clear'}")
        res["arms"][pop_name] = {
            "n": int(pop.sum()), "n_coarse": int((pop & is_coarse).sum()),
            "auc_A_all": float(roc_auc_score(yy, pA)),
            "auc_B_2min_only": float(roc_auc_score(yy, pB)),
            "auc_C_downweighted": float(roc_auc_score(yy, pC)),
            "delta_add_coarse": {"mean": m_, "ci_lower": lo, "ci_upper": hi,
                                 "clears": bool(lo > 0)},
            "delta_add_coarse_downweighted": {"mean": m2, "ci_lower": lo2,
                                              "ci_upper": hi2, "clears": bool(lo2 > 0)},
        }

    # ---- duration / sampling ----
    print("\n" + "=" * 78)
    print("TRANSIT SAMPLING -- can coarse cadence resolve these transits?")
    print("=" * 78)
    dur_h = pd.to_numeric(df["duration"], errors="coerce") * 24.0
    for nm, m in (("2-min", is_2min), ("COARSE", is_coarse)):
        cad = pd.to_numeric(df["cadence_min"], errors="coerce").fillna(2.0)
        pts = (dur_h[m] * 60.0) / cad[m]
        s = dur_h[m].dropna()
        print(f"  {nm:8s} duration median {s.median():.2f}h | "
              f"in-transit samples median {pts.median():.1f} | "
              f"under 5 samples: {100*(pts < 5).mean():.1f}%")
        res.setdefault("sampling", {})[nm] = {
            "median_duration_h": float(s.median()),
            "median_in_transit_points": float(pts.median()),
            "pct_under_5_points": float(100 * (pts < 5).mean()),
        }

    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
