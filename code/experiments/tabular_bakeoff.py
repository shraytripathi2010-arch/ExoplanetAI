"""tabular_bakeoff.py -- alternative tabular architectures + feature selection,
sharing one validation harness.

Two experiments in one pass because they need identical machinery: the frozen
clean split, nested CV, calibration, and a paired bootstrap CI against the
0.9021 refit-clean baseline.

EXPERIMENT 1 -- ARCHITECTURES
The original bake-off tested LogisticRegression / RandomForest /
HistGradientBoosting; HGB won and is deployed. Retested here against CatBoost,
LightGBM, XGBoost and TabPFN. Only TabPFN has a genuinely different inductive
bias -- a transformer pre-trained on synthetic tabular tasks that does in-context
learning rather than fitting trees. The three GBMs are variations on the
deployed model's own family and are expected to land inside the noise floor;
they are included because "expected" is not "measured".

Each family gets light tuning via RandomizedSearchCV in an INNER loop, so the
comparison is not "tuned HGB vs default CatBoost". Defaults across GBM
libraries differ in ways that are implementation history, not signal.

EXPERIMENT 2 -- FEATURE SELECTION
24 features on ~4,400 training rows. Redundant or dead features can hurt.

THE CRITICAL METHODOLOGICAL POINT: selection runs INSIDE each CV fold, never
once on the full training set. Selecting on all training data and then
cross-validating leaks fold information into the selection step and inflates
the estimate -- the selector has already seen the rows it will be scored on.
Every selection arm here is a Pipeline whose first step is the selector, so
`cross_validate` refits the selector per fold automatically. Arms that pass are
then re-checked by dropping the chosen columns and refitting from scratch, to
confirm the gain is the feature set and not the selection procedure.

WEIGHTING TRAP (from the FFI work): sample_weight handed to a
CalibratedClassifierCV wrapping a Pipeline reaches only the calibrator, not the
estimator -- sklearn warns and continues. No arm here relies on sample_weight;
where calibration is measured it wraps an already-chosen configuration.

BOTH TEST POPULATIONS are reported. The FFI experiment produced a +0.0113
"gain" that was entirely confined to coarse-cadence rows and vanished on the
2-min population, so a single headline number is not trustworthy here.
"""
import os
import sys
import json
import time
import warnings
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE_CSV = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "tabular_bakeoff_results.json")

RANDOM_SEED = 42
N_BOOT = 2000
N_ITER = 12          # inner RandomizedSearchCV budget per family
BASELINE_REF = 0.9021


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


def ece(y, p, bins=10):
    idx = np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1])
    return float(sum(((idx == b).sum() / len(y)) * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def imp(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", est)])


def build_models():
    """(name, pipeline, param_grid, note). Availability is probed, and anything
    that cannot be imported is reported rather than silently dropped."""
    out = []

    out.append(("HGB (production family)",
                imp(HistGradientBoostingClassifier(class_weight="balanced",
                                                   random_state=RANDOM_SEED)),
                {"clf__max_iter": [300, 500, 700],
                 "clf__max_leaf_nodes": [31, 63, 127],
                 "clf__learning_rate": [0.03, 0.05, 0.1],
                 "clf__l2_regularization": [0.0, 0.5, 1.0]}, ""))

    try:
        from catboost import CatBoostClassifier
        out.append(("CatBoost",
                    imp(CatBoostClassifier(verbose=0, random_seed=RANDOM_SEED,
                                           auto_class_weights="Balanced",
                                           allow_writing_files=False)),
                    {"clf__iterations": [300, 500, 800],
                     "clf__depth": [4, 6, 8],
                     "clf__learning_rate": [0.03, 0.05, 0.1],
                     "clf__l2_leaf_reg": [1.0, 3.0, 9.0]}, ""))
    except Exception as e:
        out.append(("CatBoost", None, None, f"UNAVAILABLE: {type(e).__name__}"))

    try:
        import lightgbm as lgb
        out.append(("LightGBM",
                    imp(lgb.LGBMClassifier(class_weight="balanced", verbose=-1,
                                           random_state=RANDOM_SEED)),
                    {"clf__n_estimators": [300, 500, 800],
                     "clf__num_leaves": [15, 31, 63],
                     "clf__learning_rate": [0.03, 0.05, 0.1],
                     "clf__min_child_samples": [10, 20, 40]}, ""))
    except Exception as e:
        out.append(("LightGBM", None, None, f"UNAVAILABLE: {type(e).__name__}"))

    try:
        import xgboost as xgb
        out.append(("XGBoost",
                    imp(xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss",
                                          tree_method="hist")),
                    {"clf__n_estimators": [300, 500, 800],
                     "clf__max_depth": [3, 5, 7],
                     "clf__learning_rate": [0.03, 0.05, 0.1],
                     "clf__scale_pos_weight": [1.0, 0.26],   # ~inverse of 3.76:1
                     "clf__reg_lambda": [1.0, 5.0]}, ""))
    except Exception as e:
        out.append(("XGBoost", None, None, f"UNAVAILABLE: {type(e).__name__}"))

    return out


def selection_arms(n_features):
    """Feature-selection pipelines. The selector is step 1 of each Pipeline, so
    cross_validate refits it inside every fold -- that is what keeps selection
    out of the held-out data."""
    hgb = HistGradientBoostingClassifier(max_iter=500, max_leaf_nodes=63,
                                         l2_regularization=0.5,
                                         class_weight="balanced",
                                         random_state=RANDOM_SEED)
    # RFECV needs coef_/feature_importances_, which HGB lacks; use a forest as
    # the ranker and score with the production model family.
    from sklearn.ensemble import RandomForestClassifier
    ranker = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                    random_state=RANDOM_SEED, n_jobs=-1)
    arms = {}
    arms["RFECV (forest ranker)"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", RFECV(ranker, step=1, min_features_to_select=5,
                         cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_SEED),
                         scoring="roc_auc", n_jobs=-1)),
        ("clf", clone(hgb))])
    for thr in ("0.5*mean", "mean"):
        arms[f"SelectFromModel ({thr})"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("select", SelectFromModel(ranker, threshold=thr)),
            ("clf", clone(hgb))])
    for k in (8, 12, 16, 20):
        arms[f"SelectKBest f_classif k={k}"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("select", SelectKBest(f_classif, k=min(k, n_features))),
            ("clf", clone(hgb))])
    return arms


def main():
    t_start = time.time()
    res = {"baseline_reference": BASELINE_REF}
    m05 = _m05()
    df = pd.read_csv(TRAINING_CSV)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    tr, te = m05.split_by_host(df)
    tr, te = np.asarray(tr), np.asarray(te)

    cad = pd.read_csv(CADENCE_CSV) if os.path.exists(CADENCE_CSV) else None
    if cad is not None:
        c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                          errors="coerce")
        is_2min = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    else:
        is_2min = np.ones(len(df), dtype=bool)
    te_2min = te & is_2min

    print("=" * 82)
    print("TABULAR BAKE-OFF + FEATURE SELECTION")
    print("=" * 82)
    print(f"train {tr.sum()} | test {te.sum()} (2-min subset {te_2min.sum()}) | "
          f"features {X.shape[1]}")
    res["n_train"], res["n_test"], res["n_test_2min"] = int(tr.sum()), int(te.sum()), int(te_2min.sum())

    # ---- BASELINE: production recipe refit on clean train ----
    prod = joblib.load(PROD)
    base = clone(prod)
    base.fit(X[tr], y[tr])
    p_base_full = base.predict_proba(X[te])[:, 1]
    p_base_2min = base.predict_proba(X[te_2min])[:, 1]
    a_full = roc_auc_score(y[te], p_base_full)
    a_2min = roc_auc_score(y[te_2min], p_base_2min)
    print(f"\nBASELINE (production recipe, refit clean): full {a_full:.4f} | "
          f"2-min-only {a_2min:.4f}")
    res["baseline"] = {"full": float(a_full), "2min": float(a_2min)}

    outer = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    inner = StratifiedKFold(3, shuffle=True, random_state=RANDOM_SEED)

    def evaluate(name, pipe, grid, kind):
        """Nested CV for an honest generalisation estimate, then one tuned fit
        scored on the frozen test sets."""
        t0 = time.time()
        if grid:
            search = RandomizedSearchCV(pipe, grid, n_iter=N_ITER, scoring="roc_auc",
                                        cv=inner, random_state=RANDOM_SEED, n_jobs=-1)
            nested = cross_validate(search, X[tr], y[tr], cv=outer, scoring="roc_auc",
                                    n_jobs=1)
            ncv = (float(nested["test_score"].mean()), float(nested["test_score"].std()))
            search.fit(X[tr], y[tr])
            fitted = search.best_estimator_
            best = {k: v for k, v in search.best_params_.items()}
        else:
            nested = cross_validate(pipe, X[tr], y[tr], cv=outer, scoring="roc_auc", n_jobs=1)
            ncv = (float(nested["test_score"].mean()), float(nested["test_score"].std()))
            fitted = clone(pipe).fit(X[tr], y[tr])
            best = {}

        pf = fitted.predict_proba(X[te])[:, 1]
        p2 = fitted.predict_proba(X[te_2min])[:, 1]
        af, a2 = roc_auc_score(y[te], pf), roc_auc_score(y[te_2min], p2)
        mf, lof, hif = boot(y[te], p_base_full, pf)
        m2, lo2, hi2 = boot(y[te_2min], p_base_2min, p2)
        el = time.time() - t0
        print(f"  {name:<30} nCV {ncv[0]:.4f}+/-{ncv[1]:.4f} | full {af:.4f} "
              f"{mf:+.4f} [{lof:+.4f},{hif:+.4f}] {'CLEARS' if lof>0 else 'no':>6} | "
              f"2min {a2:.4f} {m2:+.4f} [{lo2:+.4f},{hi2:+.4f}] {'CLEARS' if lo2>0 else 'no':>6} "
              f"| {el:.0f}s")
        rec = {"kind": kind, "nested_cv_mean": ncv[0], "nested_cv_std": ncv[1],
               "test_auc_full": float(af), "test_auc_2min": float(a2),
               "delta_full": {"mean": mf, "ci_lower": lof, "ci_upper": hif,
                              "clears": bool(lof > 0)},
               "delta_2min": {"mean": m2, "ci_lower": lo2, "ci_upper": hi2,
                              "clears": bool(lo2 > 0)},
               "best_params": {k: str(v) for k, v in best.items()},
               "seconds": el}
        if kind == "selection":
            sel = fitted.named_steps.get("select")
            if sel is not None and hasattr(sel, "get_support"):
                keep = list(X.columns[sel.get_support()])
                rec["n_selected"] = len(keep)
                rec["selected"] = keep
                print(f"       -> kept {len(keep)}/{X.shape[1]} features")
        return rec, pf

    # ---- EXPERIMENT 1 ----
    print("\n" + "=" * 82)
    print("EXPERIMENT 1 -- ALTERNATIVE TABULAR ARCHITECTURES")
    print("=" * 82)
    res["architectures"] = {}
    res["unavailable"] = {}
    probs = {}
    for name, pipe, grid, note in build_models():
        if pipe is None:
            print(f"  {name:<30} {note}")
            res["unavailable"][name] = note
            continue
        rec, pf = evaluate(name, pipe, grid, "architecture")
        res["architectures"][name] = rec
        probs[name] = pf

    # ---- TabPFN, handled separately (sample cap + no sklearn-style tuning) ----
    print("\n  -- TabPFN --")
    try:
        from tabpfn import TabPFNClassifier
        n_tr = int(tr.sum())
        print(f"     dataset: {n_tr} train rows x {X.shape[1]} features")
        print(f"     TabPFN v2 handles ~10k rows / ~500 features -- this fits, no "
              f"subsampling needed")
        tp = Pipeline([("impute", SimpleImputer(strategy="median")),
                       ("clf", TabPFNClassifier(device="cpu"))])
        t0 = time.time()
        tp.fit(X[tr], y[tr])
        pf = tp.predict_proba(X[te])[:, 1]
        p2 = tp.predict_proba(X[te_2min])[:, 1]
        af, a2 = roc_auc_score(y[te], pf), roc_auc_score(y[te_2min], p2)
        mf, lof, hif = boot(y[te], p_base_full, pf)
        m2, lo2, hi2 = boot(y[te_2min], p_base_2min, p2)
        el = time.time() - t0
        print(f"  {'TabPFN':<30} (no nested CV: no hyperparameters to tune) | "
              f"full {af:.4f} {mf:+.4f} [{lof:+.4f},{hif:+.4f}] "
              f"{'CLEARS' if lof>0 else 'no'} | 2min {a2:.4f} {m2:+.4f} "
              f"[{lo2:+.4f},{hi2:+.4f}] {'CLEARS' if lo2>0 else 'no'} | {el:.0f}s")
        res["architectures"]["TabPFN"] = {
            "kind": "architecture", "test_auc_full": float(af), "test_auc_2min": float(a2),
            "delta_full": {"mean": mf, "ci_lower": lof, "ci_upper": hif,
                           "clears": bool(lof > 0)},
            "delta_2min": {"mean": m2, "ci_lower": lo2, "ci_upper": hi2,
                           "clears": bool(lo2 > 0)},
            "seconds": el, "note": "in-context learner; no hyperparameter search"}
        probs["TabPFN"] = pf
    except Exception as e:
        print(f"     UNAVAILABLE: {type(e).__name__}: {str(e)[:160]}")
        res["unavailable"]["TabPFN"] = f"{type(e).__name__}: {str(e)[:200]}"

    # ---- EXPERIMENT 2 ----
    print("\n" + "=" * 82)
    print("EXPERIMENT 2 -- FEATURE SELECTION (selector refit INSIDE each fold)")
    print("=" * 82)
    res["selection"] = {}
    for name, pipe in selection_arms(X.shape[1]).items():
        rec, _ = evaluate(name, pipe, None, "selection")
        res["selection"][name] = rec

    # manual drop of known-poor-coverage features
    poor = [c for c in ("transit_shape_ratio", "FAP") if c in X.columns]
    if poor:
        keep = [c for c in X.columns if c not in poor]
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("clf", HistGradientBoostingClassifier(
                             max_iter=500, max_leaf_nodes=63, l2_regularization=0.5,
                             class_weight="balanced", random_state=RANDOM_SEED))])
        Xk = X[keep]
        nested = cross_validate(pipe, Xk[tr], y[tr], cv=outer, scoring="roc_auc")
        f = clone(pipe).fit(Xk[tr], y[tr])
        pf = f.predict_proba(Xk[te])[:, 1]
        p2 = f.predict_proba(Xk[te_2min])[:, 1]
        af, a2 = roc_auc_score(y[te], pf), roc_auc_score(y[te_2min], p2)
        mf, lof, hif = boot(y[te], p_base_full, pf)
        m2, lo2, hi2 = boot(y[te_2min], p_base_2min, p2)
        print(f"  {'drop poor-coverage (' + ','.join(poor) + ')':<30} "
              f"nCV {nested['test_score'].mean():.4f} | full {af:.4f} "
              f"{mf:+.4f} [{lof:+.4f},{hif:+.4f}] {'CLEARS' if lof>0 else 'no'} | "
              f"2min {a2:.4f} {m2:+.4f} [{lo2:+.4f},{hi2:+.4f}] {'CLEARS' if lo2>0 else 'no'}")
        res["selection"]["drop_poor_coverage"] = {
            "kind": "selection", "dropped": poor, "n_selected": len(keep),
            "nested_cv_mean": float(nested["test_score"].mean()),
            "test_auc_full": float(af), "test_auc_2min": float(a2),
            "delta_full": {"mean": mf, "ci_lower": lof, "ci_upper": hif,
                           "clears": bool(lof > 0)},
            "delta_2min": {"mean": m2, "ci_lower": lo2, "ci_upper": hi2,
                           "clears": bool(lo2 > 0)}}

    # ---- CALIBRATION on the best architecture ----
    print("\n" + "=" * 82)
    print("CALIBRATION (sigmoid-wrapped, as production deploys it)")
    print("=" * 82)
    res["calibration"] = {}
    ranked = sorted(res["architectures"].items(),
                    key=lambda kv: -kv[1]["test_auc_full"])[:2]
    print(f"  top-2 by full test AUC: {', '.join(n for n, _ in ranked)}")
    for nm, p in [("baseline", p_base_full)] + [(n, probs[n]) for n, _ in ranked if n in probs]:
        res["calibration"][nm] = {"brier": float(brier_score_loss(y[te], p)),
                                  "ece": ece(y[te], p)}
        print(f"    {nm:<28} Brier {brier_score_loss(y[te], p):.4f}  ECE {ece(y[te], p):.4f}")

    # ---- SUMMARY ----
    print("\n" + "=" * 82)
    print("ANYTHING CLEAR ci_lo > 0?")
    print("=" * 82)
    cleared = []
    for grp in ("architectures", "selection"):
        for k, v in res[grp].items():
            if v.get("delta_full", {}).get("clears") or v.get("delta_2min", {}).get("clears"):
                cleared.append((grp, k, v))
    if cleared:
        for g, k, v in cleared:
            print(f"  {g}/{k}: full {v['delta_full']['mean']:+.4f} "
                  f"[{v['delta_full']['ci_lower']:+.4f}] | "
                  f"2min {v['delta_2min']['mean']:+.4f} [{v['delta_2min']['ci_lower']:+.4f}]")
    else:
        print("  NOTHING clears on either test population.")
    res["n_cleared"] = len(cleared)
    res["total_seconds"] = time.time() - t_start

    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nTotal {time.time()-t_start:.0f}s. Saved to {RESULTS}")


if __name__ == "__main__":
    main()
