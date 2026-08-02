"""validate_challenger_config.py -- proves the retrain promotion gate's
challenger now matches production's configuration, and measures what the old
mismatch actually cost.

OFFLINE AND READ-ONLY. This never calls maybe_trigger_retrain(), never writes
to models/, and never touches the database. It rebuilds the same comparison
the gate performs, using the same frozen split, so the numbers are directly
comparable to what a real attempt would produce -- without consuming a retrain
attempt or risking a promotion.

THE BUG

The incumbent every challenger had to beat is the tuned, calibration-wrapped
production model. The challenger was built from
m05.build_models()["HistGradientBoosting"] -- the untuned default. So each
retrain attempt opened with a deficit that had nothing to do with whether the
new labelled data helped, and the promotion gate is exactly the place where a
silent handicap does the most damage.

WHAT THIS MEASURES

  A. old challenger  -- m05 default config, unwrapped
  B. new challenger  -- clone(production), i.e. production's own recipe
  C. production      -- the deployed artifact, as-is

All three scored on the identical frozen test set. B vs A isolates the
handicap. B vs C is what the fixed gate would actually decide on today's data.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
PRODUCTION_MODEL_PATH = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "challenger_config_validation.json")

RANDOM_SEED = 42
N_BOOTSTRAP = 2000


def _load_m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


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
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


def inner_clf(est):
    """Reach the HGB inside whatever wrapper the estimator uses."""
    e = getattr(est, "estimator", est)          # CalibratedClassifierCV -> Pipeline
    if hasattr(e, "named_steps"):
        return e.named_steps.get("clf", e)
    return e


def main():
    res = {}
    m05 = _load_m05()
    df = pd.read_csv(TRAINING_CSV)
    X, y = m05.build_feature_matrix(df)
    tr, te = m05.split_by_host(df)
    X_train, X_test = X[tr], X[te]
    y_train, y_test = y[tr], y[te]
    y_test = np.asarray(y_test)
    print(f"frozen split: {tr.sum()} train / {te.sum()} test\n")
    res["n_train"], res["n_test"] = int(tr.sum()), int(te.sum())

    prod = joblib.load(PRODUCTION_MODEL_PATH)
    old_challenger = m05.build_models()["HistGradientBoosting"]
    new_challenger = clone(prod)

    # ---------------- CONFIG COMPARISON ----------------
    print("=" * 78)
    print("HYPERPARAMETER COMPARISON")
    print("=" * 78)
    p_prod = inner_clf(prod).get_params()
    p_old = inner_clf(old_challenger).get_params()
    p_new = inner_clf(new_challenger).get_params()

    print(f"  wrapper: production {type(prod).__name__} | "
          f"old challenger {type(old_challenger).__name__} | "
          f"new challenger {type(new_challenger).__name__}")

    diffs_old = {k: (p_old.get(k), p_prod.get(k))
                 for k in sorted(p_prod) if p_old.get(k) != p_prod.get(k)}
    diffs_new = {k: (p_new.get(k), p_prod.get(k))
                 for k in sorted(p_prod) if p_new.get(k) != p_prod.get(k)}

    print(f"\n  OLD challenger vs production -- {len(diffs_old)} differing hyperparameter(s):")
    for k, (a, b) in diffs_old.items():
        print(f"    {k:22s} challenger={a!r:>8}   production={b!r}")
    print(f"\n  NEW challenger vs production -- {len(diffs_new)} differing hyperparameter(s)"
          + ("" if diffs_new else "  <- MATCHES"))
    for k, (a, b) in diffs_new.items():
        print(f"    {k:22s} challenger={a!r:>8}   production={b!r}")

    wrapper_match = type(new_challenger).__name__ == type(prod).__name__
    print(f"\n  wrapper class matches: {wrapper_match}")
    res["old_challenger_param_diffs"] = {k: [repr(a), repr(b)] for k, (a, b) in diffs_old.items()}
    res["new_challenger_param_diffs"] = {k: [repr(a), repr(b)] for k, (a, b) in diffs_new.items()}
    res["new_challenger_wrapper_matches"] = bool(wrapper_match)
    res["new_challenger_matches_production"] = bool(not diffs_new and wrapper_match)

    # ---------------- PERFORMANCE ----------------
    print("\n" + "=" * 78)
    print("PERFORMANCE ON THE FROZEN TEST SET")
    print("=" * 78)
    prod_proba = prod.predict_proba(X_test)[:, 1]
    prod_auc = roc_auc_score(y_test, prod_proba)

    old_challenger.fit(X_train, y_train)
    old_proba = old_challenger.predict_proba(X_test)[:, 1]
    old_auc = roc_auc_score(y_test, old_proba)

    new_challenger.fit(X_train, y_train)
    new_proba = new_challenger.predict_proba(X_test)[:, 1]
    new_auc = roc_auc_score(y_test, new_proba)

    print(f"  C. production (deployed, as-is)      {prod_auc:.4f}")
    print(f"  A. OLD challenger (m05 defaults)     {old_auc:.4f}")
    print(f"  B. NEW challenger (clone of prod)    {new_auc:.4f}")
    res["production_auc"] = float(prod_auc)
    res["old_challenger_auc"] = float(old_auc)
    res["new_challenger_auc"] = float(new_auc)

    m, lo, hi = paired_bootstrap(y_test, old_proba, new_proba)
    print(f"\n  THE HANDICAP (new - old challenger): {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
    res["handicap"] = {"mean": m, "ci_lower": lo, "ci_upper": hi}

    m2, lo2, hi2 = paired_bootstrap(y_test, prod_proba, old_proba)
    print(f"  OLD gate decision  (old - production): {m2:+.4f}  CI [{lo2:+.4f}, {hi2:+.4f}]"
          f"  -> {'PROMOTE' if lo2 > 0 else 'no promotion'}")
    m3, lo3, hi3 = paired_bootstrap(y_test, prod_proba, new_proba)
    print(f"  NEW gate decision  (new - production): {m3:+.4f}  CI [{lo3:+.4f}, {hi3:+.4f}]"
          f"  -> {'PROMOTE' if lo3 > 0 else 'no promotion'}")
    res["old_gate"] = {"mean": m2, "ci_lower": lo2, "ci_upper": hi2, "would_promote": bool(lo2 > 0)}
    res["new_gate"] = {"mean": m3, "ci_lower": lo3, "ci_upper": hi3, "would_promote": bool(lo3 > 0)}

    print("\n" + "=" * 78)
    print("Note: the new challenger is production's recipe refit on MORE data, so")
    print("a near-zero delta is the expected and correct outcome -- it means the")
    print("gate is now measuring the effect of data alone, which is the point.")
    print("=" * 78)

    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
