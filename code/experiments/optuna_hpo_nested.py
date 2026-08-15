"""optuna_hpo_nested.py -- PART 1a: a proper Bayesian (TPE) hyperparameter search
on production's HGB, under NESTED cross-validation.

WHY THIS IS NOT A DUPLICATE
---------------------------
Every hyperparameter search in this project's history is `RandomizedSearchCV`:
`05b_model_analysis.py` (n_iter=30), `tabular_bakeoff.py` (n_iter=12),
`gbm_ensemble.py`, `validate_multisector.py` and
`retrain_with_new_features_full_suite.py` (n_iter=15-30). Random search draws
i.i.d. from the prior and never conditions on what it has already seen. TPE
builds a density model of good vs bad configurations and samples from the ratio.
"We did some tuning" is true; "we did a thorough Bayesian search" was not.

Precisely: `05b_model_analysis.py` searched a 5-dimensional discrete HGB grid
(max_iter, max_depth, learning_rate, l2_regularization, max_leaf_nodes -- 2,000
combinations) with 30 draws, i.e. 1.5% coverage, and never varied
`min_samples_leaf` or `class_weight` at all. The space searched here is
continuous in learning_rate and l2, and adds both of those.

WHY NESTED CV -- and a correction to the framing that prompted this
-------------------------------------------------------------------
Nested CV is NOT new here: `05b_model_analysis.py` already ran outer-5 / inner-3
with the search inside the inner loop, which is the correct structure. What
changes is that the guard now matters much more. With 30 random draws the
selection bias from taking the best inner score is small; with 120 TPE trials it
is not, because TPE actively concentrates sampling on whatever the inner CV
happens to reward, noise included. Its own best score is optimistic BY
CONSTRUCTION. The search here runs entirely inside each outer fold's training
portion and never sees that fold's held-out rows, so the outer score is an
honest estimate of the SEARCH PROCEDURE, and the gap between the two is
reported as the size of that optimism.

That is still not this project's bar. The nested-CV number answers "would
searching help?"; `optuna_hpo_validate.py` then puts the single selected config
through 12 training bootstraps against the frozen test, which is what every
other result here had to clear.

COST CONTROL, stated because it is a real design choice
--------------------------------------------------------
The inner objective scores the BARE pipeline, not the calibrated wrapper. AUC
is invariant under each fold's sigmoid, so the wrapper's only AUC effect is
5-model averaging -- which the calibration sweep already characterised and which
does not interact strongly with the base learner's hyperparameters. This is also
what the historical `RandomizedSearchCV(pipe, ...)` calls did. The OUTER
evaluation uses production's full calibrated recipe, so the reported number is
for the configuration as it would actually be deployed.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys
import json
import time
import importlib.util
import warnings
import numpy as np
import pandas as pd
import optuna
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT = os.path.join(HERE, "optuna_hpo_nested.json")

N_TRIALS = 120
OUTER_K = 5
INNER_K = 3
SEED = 42

# What production actually carries today (sklearn defaults + random_state).
PROD_PARAMS = {"random_state": 42}
# The configuration every model version before the Gaia swap carried.
LEGACY_TUNED = {"random_state": 42, "learning_rate": 0.1, "max_iter": 500,
                "max_leaf_nodes": 63, "max_depth": None, "l2_regularization": 0.5,
                "min_samples_leaf": 20, "class_weight": "balanced"}


def pipe(params):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", HistGradientBoostingClassifier(**params))])


def calibrated(params):
    return CalibratedClassifierCV(pipe(params), cv=5, method="sigmoid")


def suggest(trial):
    md = trial.suggest_categorical("max_depth", ["none", 2, 3, 4, 6, 8, 12])
    return {"random_state": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 50, 500, step=25),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128, log=True),
            "max_depth": None if md == "none" else md,
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["none", "balanced"])}


def _clean(params):
    p = dict(params)
    if p.get("class_weight") == "none":
        p["class_weight"] = None
    return p


def run_study(X, y, seed, n_trials, tag):
    """TPE search scored by inner CV of the bare pipeline, with median pruning
    across inner folds so hopeless configurations do not pay for all three."""
    inner = list(StratifiedKFold(INNER_K, shuffle=True, random_state=seed).split(X, y))

    def objective(trial):
        params = _clean(suggest(trial))
        scores = []
        for step, (i_tr, i_va) in enumerate(inner):
            m = pipe(params).fit(X.iloc[i_tr], y[i_tr])
            scores.append(roc_auc_score(y[i_va], m.predict_proba(X.iloc[i_va])[:, 1]))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=20),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=1))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    el = time.time() - t0
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  [{tag}] {len(done)} complete / {len(pruned)} pruned in {el/60:.1f} min, "
          f"best inner AUC {study.best_value:.4f}", flush=True)
    return study, el


def outer_fold(k, i_tr, i_te, X, y):
    Xtr, ytr, Xte, yte = X.iloc[i_tr], y[i_tr], X.iloc[i_te], y[i_te]
    study, el = run_study(Xtr, ytr, SEED + k, N_TRIALS, f"outer {k+1}")
    # rebuild the concrete params from the winning trial's raw suggestions
    p = study.best_params
    best = _clean({"random_state": 42, "learning_rate": p["learning_rate"],
                   "max_iter": p["max_iter"], "max_leaf_nodes": p["max_leaf_nodes"],
                   "max_depth": None if p["max_depth"] == "none" else p["max_depth"],
                   "min_samples_leaf": p["min_samples_leaf"],
                   "l2_regularization": p["l2_regularization"],
                   "class_weight": p["class_weight"]})
    res = {"fold": k, "inner_best_auc": float(study.best_value), "best_params": best,
           "search_seconds": round(el, 1),
           "n_complete": sum(1 for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE)}
    for lab, params in (("searched", best), ("prod", PROD_PARAMS), ("legacy", LEGACY_TUNED)):
        mo = calibrated(params).fit(Xtr, ytr)
        res[f"outer_{lab}"] = float(roc_auc_score(yte, mo.predict_proba(Xte)[:, 1]))
    print(f"  [outer {k+1}] OUTER searched {res['outer_searched']:.4f}  "
          f"prod {res['outer_prod']:.4f}  legacy {res['outer_legacy']:.4f}  "
          f"(inner best {res['inner_best_auc']:.4f})", flush=True)
    return res


def main():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 33, len(cols)

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    tr_mask, _ = m05.split_by_host(df)
    Xtr = X[tr_mask].reset_index(drop=True)
    ytr = y[tr_mask]
    print(f"search population: {len(ytr)} training rows, {len(cols)} features, "
          f"{ytr.mean():.4f} positive")
    print(f"nested CV: outer {OUTER_K}-fold / inner {INNER_K}-fold TPE, "
          f"{N_TRIALS} trials per outer fold\n")

    t_all = time.time()
    folds = list(StratifiedKFold(OUTER_K, shuffle=True, random_state=SEED).split(Xtr, ytr))
    res = Parallel(n_jobs=OUTER_K, backend="loky")(
        delayed(outer_fold)(k, i_tr, i_te, Xtr, ytr) for k, (i_tr, i_te) in enumerate(folds))
    nested_s = time.time() - t_all

    R = pd.DataFrame(res)
    d_search = (R.outer_searched - R.outer_prod).values
    d_legacy = (R.outer_legacy - R.outer_prod).values
    print("\n" + "=" * 78)
    print("NESTED CV (outer folds are held out from the search entirely)")
    print(f"  production config      {R.outer_prod.mean():.4f}  "
          f"(sd {R.outer_prod.std():.4f})")
    print(f"  Optuna-searched        {R.outer_searched.mean():.4f}  "
          f"(sd {R.outer_searched.std():.4f})   delta {d_search.mean():+.4f}  "
          f"{(d_search>0).sum()}/{OUTER_K} folds")
    print(f"  legacy tuned config    {R.outer_legacy.mean():.4f}  "
          f"(sd {R.outer_legacy.std():.4f})   delta {d_legacy.mean():+.4f}  "
          f"{(d_legacy>0).sum()}/{OUTER_K} folds")
    print(f"  inner best (OPTIMISTIC){R.inner_best_auc.mean():.4f}  -- "
          f"{R.inner_best_auc.mean()-R.outer_searched.mean():+.4f} vs its own outer score")

    # ---- final study on ALL training rows -> the one deployable config ----
    print(f"\nfinal study on all {len(ytr)} training rows...", flush=True)
    study, el_final = run_study(Xtr, ytr, SEED, N_TRIALS, "final")
    p = study.best_params
    final = _clean({"random_state": 42, "learning_rate": p["learning_rate"],
                    "max_iter": p["max_iter"], "max_leaf_nodes": p["max_leaf_nodes"],
                    "max_depth": None if p["max_depth"] == "none" else p["max_depth"],
                    "min_samples_leaf": p["min_samples_leaf"],
                    "l2_regularization": p["l2_regularization"],
                    "class_weight": p["class_weight"]})
    print("  selected config:")
    for k2 in sorted(final):
        print(f"    {k2:<20} {final[k2]}")

    out = {"n_trials": N_TRIALS, "outer_k": OUTER_K, "inner_k": INNER_K,
           "prod_params": PROD_PARAMS, "legacy_params": LEGACY_TUNED,
           "final_params": final, "final_inner_auc": float(study.best_value),
           "nested": res,
           "nested_summary": {
               "prod": float(R.outer_prod.mean()),
               "searched": float(R.outer_searched.mean()),
               "legacy": float(R.outer_legacy.mean()),
               "delta_search": float(d_search.mean()),
               "delta_legacy": float(d_legacy.mean()),
               "search_positive_folds": int((d_search > 0).sum()),
               "inner_best_mean": float(R.inner_best_auc.mean())},
           "wall_clock": {"nested_s": round(nested_s, 1),
                          "final_study_s": round(el_final, 1),
                          "total_s": round(time.time() - t_all, 1)}}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwall clock: nested {nested_s/60:.1f} min (5 folds in parallel), "
          f"final study {el_final/60:.1f} min, total {(time.time()-t_all)/60:.1f} min")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
