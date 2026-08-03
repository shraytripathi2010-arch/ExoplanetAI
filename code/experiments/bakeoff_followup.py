"""bakeoff_followup.py -- finish the two arms SIP blocked, and stress-test the
one result that cleared.

WHY THIS EXISTS

`tabular_bakeoff.py` reported LightGBM and XGBoost as UNAVAILABLE even though
both import fine in an interactive shell. Cause, confirmed directly: macOS
System Integrity Protection strips `DYLD_*` environment variables when exec'ing
a protected system binary, and the run was launched through `/usr/bin/caffeinate`.
The variable was set, survived to caffeinate, and was purged before python
started -- `DYLD_LIBRARY_PATH` read as None inside the child.

Fix without env vars: dlopen torch's bundled libomp with RTLD_GLOBAL before
importing either library, so the OpenMP symbols are already resolved in-process.

CATBOOST CLEARED, AND THAT IS WHY IT GETS STRESS-TESTED

CatBoost was the first arm in this project's 22 experiments to clear ci_lo > 0
on BOTH test populations (+0.0085 [+0.0004, +0.0171] full; +0.0105 [+0.0014,
+0.0201] 2-min-only). Both lower bounds sit a hair above zero. This project has
already seen one result clear the bar and then dissolve under a control (the
multi-sector feature, where an indicator-only model reproduced 108% of the
gain), so a barely-clearing first positive earns more scrutiny, not less.

The decisive question for a tree ensemble is SEED SENSITIVITY. A single fit is
one draw from a stochastic procedure; if the delta swings across seeds and its
lower bound crosses zero, the single-seed CI is measuring the seed, not the
model. Ten seeds for CatBoost, ten matched seeds for the HGB baseline, then the
delta distribution.
"""
import os
import sys
import json
import ctypes
import time
import warnings
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE_CSV = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "bakeoff_followup_results.json")

RANDOM_SEED = 42
N_BOOT = 2000
N_SEEDS = 10


def preload_libomp():
    """Make libomp available to LightGBM/XGBoost, but ONLY if they cannot
    already load it themselves.

    Three failures got us here, recorded because each was a different kind:

    1. Neither library ships an OpenMP runtime and this machine has no
       Homebrew, so `libomp.dylib` was simply absent.
    2. Setting DYLD_LIBRARY_PATH fixed it in an interactive shell but NOT in
       the actual run: macOS SIP strips DYLD_* when exec'ing a protected binary,
       and the job was launched through /usr/bin/caffeinate. The variable read
       as None inside the child.
    3. ctypes-preloading torch's libomp did not help either -- the dylibs
       declare a hard `@rpath/libomp.dylib` dependency, which a preloaded
       symbol does not satisfy. The rpaths are baked to
       /opt/homebrew/opt/libomp/lib and /opt/local/lib/libomp, so the real fix
       was creating the first of those (writable without sudo) and putting
       libomp there.

    And then the preload became actively harmful: with the rpath satisfied,
    preloading torch's copy meant TWO OpenMP runtimes in one process, which
    segfaulted the run (SIGSEGV, exit 139) after the header printed and before
    any arm reported -- no traceback, because a segfault is not an exception.

    So: probe first, preload only as a fallback. Returns a human-readable note.
    """
    try:
        import lightgbm  # noqa: F401
        import xgboost   # noqa: F401
        return "not needed -- both libraries load libomp via their own rpath"
    except Exception:
        pass
    import glob
    cands = glob.glob(os.path.join(os.path.dirname(os.__file__),
                                   "site-packages", "torch", "lib", "libomp*.dylib"))
    for p in cands:
        try:
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            return f"fallback preload from {p}"
        except OSError:
            continue
    return "unavailable -- no libomp found"


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


def imp(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", est)])


def main():
    res = {}
    lib = preload_libomp()
    print(f"libomp: {lib}")
    res["libomp_path"] = lib

    m05 = _m05()
    df = pd.read_csv(TRAINING_CSV)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True); y = np.asarray(y)
    tr, te = m05.split_by_host(df)
    tr, te = np.asarray(tr), np.asarray(te)
    cad = pd.read_csv(CADENCE_CSV)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is_2min = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    te2 = te & is_2min

    prod = joblib.load(PROD)
    base = clone(prod).fit(X[tr], y[tr])
    pb_f = base.predict_proba(X[te])[:, 1]
    pb_2 = base.predict_proba(X[te2])[:, 1]
    print(f"baseline: full {roc_auc_score(y[te], pb_f):.4f} | "
          f"2min {roc_auc_score(y[te2], pb_2):.4f}\n")

    outer = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    inner = StratifiedKFold(3, shuffle=True, random_state=RANDOM_SEED)

    def run(name, pipe, grid):
        t0 = time.time()
        s = RandomizedSearchCV(pipe, grid, n_iter=12, scoring="roc_auc", cv=inner,
                               random_state=RANDOM_SEED, n_jobs=-1)
        ncv = cross_validate(s, X[tr], y[tr], cv=outer, scoring="roc_auc", n_jobs=1)
        s.fit(X[tr], y[tr])
        f = s.best_estimator_
        pf = f.predict_proba(X[te])[:, 1]; p2 = f.predict_proba(X[te2])[:, 1]
        af, a2 = roc_auc_score(y[te], pf), roc_auc_score(y[te2], p2)
        mf, lof, hif = boot(y[te], pb_f, pf)
        m2, lo2, hi2 = boot(y[te2], pb_2, p2)
        print(f"  {name:<12} nCV {ncv['test_score'].mean():.4f} | full {af:.4f} "
              f"{mf:+.4f} [{lof:+.4f},{hif:+.4f}] {'CLEARS' if lof>0 else 'no'} | "
              f"2min {a2:.4f} {m2:+.4f} [{lo2:+.4f},{hi2:+.4f}] "
              f"{'CLEARS' if lo2>0 else 'no'} | {time.time()-t0:.0f}s")
        return {"nested_cv": float(ncv["test_score"].mean()),
                "test_auc_full": float(af), "test_auc_2min": float(a2),
                "delta_full": {"mean": mf, "ci_lower": lof, "ci_upper": hif,
                               "clears": bool(lof > 0)},
                "delta_2min": {"mean": m2, "ci_lower": lo2, "ci_upper": hi2,
                               "clears": bool(lo2 > 0)}}

    print("=" * 80)
    print("THE TWO ARMS SIP BLOCKED")
    print("=" * 80)
    res["arms"] = {}
    try:
        import lightgbm as lgb
        res["arms"]["LightGBM"] = run("LightGBM", imp(lgb.LGBMClassifier(
            class_weight="balanced", verbose=-1, random_state=RANDOM_SEED)),
            {"clf__n_estimators": [300, 500, 800], "clf__num_leaves": [15, 31, 63],
             "clf__learning_rate": [0.03, 0.05, 0.1],
             "clf__min_child_samples": [10, 20, 40]})
    except Exception as e:
        print(f"  LightGBM STILL UNAVAILABLE: {type(e).__name__}: {str(e)[:120]}")
        res["arms"]["LightGBM"] = {"unavailable": str(e)[:200]}
    try:
        import xgboost as xgb
        res["arms"]["XGBoost"] = run("XGBoost", imp(xgb.XGBClassifier(
            random_state=RANDOM_SEED, eval_metric="logloss", tree_method="hist")),
            {"clf__n_estimators": [300, 500, 800], "clf__max_depth": [3, 5, 7],
             "clf__learning_rate": [0.03, 0.05, 0.1],
             "clf__scale_pos_weight": [1.0, 0.26], "clf__reg_lambda": [1.0, 5.0]})
    except Exception as e:
        print(f"  XGBoost STILL UNAVAILABLE: {type(e).__name__}: {str(e)[:120]}")
        res["arms"]["XGBoost"] = {"unavailable": str(e)[:200]}

    if "--arms-only" in sys.argv:
        # Re-run just the two arms once the libomp rpath is satisfied. The
        # stress test and calibration are already recorded from the first pass
        # and are not seed-dependent on these arms.
        with open(RESULTS.replace(".json", "_arms.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"\n[--arms-only] saved to {RESULTS.replace('.json', '_arms.json')}")
        return

    # ---------- CatBoost seed-sensitivity stress test ----------
    print("\n" + "=" * 80)
    print(f"CATBOOST STRESS TEST -- {N_SEEDS} seeds, matched HGB baseline seeds")
    print("=" * 80)
    from catboost import CatBoostClassifier
    hgb_p = getattr(prod, "estimator", prod).named_steps["clf"].get_params()

    rows = []
    for i in range(N_SEEDS):
        sd = 1000 + i
        cb = imp(CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                    l2_leaf_reg=3.0, verbose=0, random_seed=sd,
                                    auto_class_weights="Balanced",
                                    allow_writing_files=False)).fit(X[tr], y[tr])
        hp = dict(hgb_p); hp["random_state"] = sd
        hb = imp(HistGradientBoostingClassifier(**hp)).fit(X[tr], y[tr])
        pc_f = cb.predict_proba(X[te])[:, 1]; ph_f = hb.predict_proba(X[te])[:, 1]
        pc_2 = cb.predict_proba(X[te2])[:, 1]; ph_2 = hb.predict_proba(X[te2])[:, 1]
        mf, lof, _ = boot(y[te], ph_f, pc_f)
        m2, lo2, _ = boot(y[te2], ph_2, pc_2)
        rows.append({"seed": sd,
                     "cat_full": float(roc_auc_score(y[te], pc_f)),
                     "hgb_full": float(roc_auc_score(y[te], ph_f)),
                     "delta_full": mf, "ci_lo_full": lof, "clears_full": bool(lof > 0),
                     "delta_2min": m2, "ci_lo_2min": lo2, "clears_2min": bool(lo2 > 0)})
        print(f"  seed {sd}: cat {rows[-1]['cat_full']:.4f} hgb {rows[-1]['hgb_full']:.4f} "
              f"| full {mf:+.4f} (lo {lof:+.4f}) {'C' if lof>0 else '.'} "
              f"| 2min {m2:+.4f} (lo {lo2:+.4f}) {'C' if lo2>0 else '.'}")

    r = pd.DataFrame(rows)
    print("\n  " + "-" * 74)
    print(f"  delta_full : mean {r.delta_full.mean():+.4f}  sd {r.delta_full.std():.4f}  "
          f"min {r.delta_full.min():+.4f}  max {r.delta_full.max():+.4f}")
    print(f"  delta_2min : mean {r.delta_2min.mean():+.4f}  sd {r.delta_2min.std():.4f}  "
          f"min {r.delta_2min.min():+.4f}  max {r.delta_2min.max():+.4f}")
    print(f"  seeds clearing on full test    : {int(r.clears_full.sum())}/{N_SEEDS}")
    print(f"  seeds clearing on 2-min-only   : {int(r.clears_2min.sum())}/{N_SEEDS}")
    res["seed_stress"] = {
        "rows": rows,
        "delta_full_mean": float(r.delta_full.mean()),
        "delta_full_sd": float(r.delta_full.std()),
        "delta_2min_mean": float(r.delta_2min.mean()),
        "delta_2min_sd": float(r.delta_2min.std()),
        "n_clearing_full": int(r.clears_full.sum()),
        "n_clearing_2min": int(r.clears_2min.sum()),
        "n_seeds": N_SEEDS}

    # ---------- calibration (lost when the main run crashed on a dead line) ----------
    print("\n" + "=" * 80)
    print("CALIBRATION (sigmoid-wrapped, as production deploys it)")
    print("=" * 80)

    def ece(yy, p, bins=10):
        idx = np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1])
        return float(sum(((idx == b).sum() / len(yy)) * abs(p[idx == b].mean() - yy[idx == b].mean())
                         for b in range(bins) if (idx == b).any()))

    from sklearn.calibration import CalibratedClassifierCV
    res["calibration"] = {}
    cb_best = imp(CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                     l2_leaf_reg=3.0, verbose=0, random_seed=RANDOM_SEED,
                                     auto_class_weights="Balanced",
                                     allow_writing_files=False))
    for nm, est in (("baseline (production, as deployed)", None),
                    ("CatBoost + sigmoid", CalibratedClassifierCV(cb_best, method="sigmoid", cv=3))):
        if est is None:
            p = pb_f
        else:
            est.fit(X[tr], y[tr])
            p = est.predict_proba(X[te])[:, 1]
        res["calibration"][nm] = {"auc": float(roc_auc_score(y[te], p)),
                                  "brier": float(brier_score_loss(y[te], p)),
                                  "ece": ece(y[te], p)}
        print(f"  {nm:<36} AUC {roc_auc_score(y[te], p):.4f}  "
              f"Brier {brier_score_loss(y[te], p):.4f}  ECE {ece(y[te], p):.4f}")

    frac = r.clears_full.mean()
    print("\n" + "=" * 80)
    if frac >= 0.9:
        v = ("ROBUST -- CatBoost beats the production family on essentially every "
             "seed. The single-seed result was not a lucky draw.")
    elif frac >= 0.5:
        v = ("PARTIALLY ROBUST -- clears on some seeds and not others. The effect "
             "is real in direction but sits close enough to the bar that a single "
             "fit cannot settle it.")
    else:
        v = ("NOT ROBUST -- the original clear was substantially a seed draw. Do "
             "not treat it as a genuine improvement.")
    print(v)
    res["seed_verdict"] = v
    print("=" * 80)

    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
