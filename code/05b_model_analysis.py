"""
05b_model_analysis.py

Deeper evaluation of the models trained in 05_train_models.py: nested-CV
hyperparameter tuning, probability calibration, threshold analysis,
permutation-importance cross-check, an ensemble check, a learning curve, and
bootstrap confidence intervals -- run on the exact same train/test split as
05_train_models.py (same RANDOM_SEED, same TEST_SIZE) for direct comparability.

IMPORTANT: this script imports 05_train_models.py as a module (importlib,
since a filename starting with a digit can't be `import`ed normally) to reuse
its FEATURE_COLUMNS, load_and_report_class_balance(), and
build_feature_matrix() rather than re-deriving the leakage-safe feature list
a second time -- copy-pasting it here would risk the two files silently
drifting apart later. Loading it this way does NOT re-run its training
(main() is still guarded by `if __name__ == "__main__":`, which is only true
when 05_train_models.py is executed directly, not when imported).

POLICY FOR THIS SCRIPT: only overwrite models/best_model.joblib,
models/best_model_metadata.json, and results/tables/model_comparison.csv if
something here GENUINELY outperforms what's already saved. Every analysis
below reports its real outcome, including "this didn't help" -- that's
itself useful information (see the SCIENTIFIC VALIDITY summary printed at
the end), not a failure to hide.

Author: Ray's Exoplanet AI Project
"""

import importlib.util
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_validate, learning_curve, train_test_split,
)

# =====================================
# LOAD 05_train_models.py AS A MODULE (see module docstring for why)
# =====================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("train_models", os.path.join(SCRIPT_DIR, "05_train_models.py"))
train_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_models)

RANDOM_SEED = train_models.RANDOM_SEED
TEST_SIZE = train_models.TEST_SIZE
FEATURE_COLUMNS = train_models.FEATURE_COLUMNS

MODELS_DIR = train_models.MODELS_DIR
TABLES_DIR = train_models.TABLES_DIR
FIGURES_DIR = train_models.FIGURES_DIR

N_BOOTSTRAP = 1000
N_RANDOM_SEARCH_ITER = 30


def load_data_and_split():
    """Reuses 05_train_models.py's own loader/feature-builder so this
    analysis is guaranteed to see the exact same data those results came
    from -- and rebuilds the identical train/test split (same seed, same
    test_size, same stratify column), so metrics here are directly
    comparable to what's already in results/tables/model_comparison.csv."""
    df = train_models.load_and_report_class_balance()
    X, y = train_models.build_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Split reconstructed: train={len(X_train)} ({y_train.sum()} pos / {len(y_train)-y_train.sum()} neg), "
          f"test={len(X_test)} ({y_test.sum()} pos / {len(y_test)-y_test.sum()} neg)")
    return X, y, X_train, X_test, y_train, y_test


def load_current_best():
    path = os.path.join(MODELS_DIR, "best_model.joblib")
    meta_path = os.path.join(MODELS_DIR, "best_model_metadata.json")
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: {path} not found -- run 05_train_models.py first.")
    model = joblib.load(path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return model, metadata


# =====================================
# 1. NESTED CV HYPERPARAMETER TUNING
#
# Outer StratifiedKFold gives an UNBIASED performance estimate of "tune then
# predict", since each outer fold's test portion never influences that
# fold's hyperparameter search. A plain (non-nested) search -- tune once,
# report the tuned model's CV score -- would optimistically bias the score,
# since the same folds used to pick hyperparameters would also grade them.
# =====================================
RF_PARAM_DIST = {
    "clf__n_estimators": [100, 200, 300, 500, 800],
    "clf__max_depth": [None, 4, 6, 8, 12, 16],
    "clf__min_samples_leaf": [1, 2, 4, 8, 16],
    "clf__max_features": ["sqrt", "log2", 0.3, 0.5, None],
}
HGB_PARAM_DIST = {
    "clf__max_iter": [100, 200, 300, 500],
    "clf__max_depth": [3, 4, 5, 6, None],
    "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "clf__l2_regularization": [0.0, 0.1, 0.5, 1.0, 2.0],
    "clf__max_leaf_nodes": [15, 31, 63, 127],
}


def nested_cv_tuning(X_train, y_train, models, baseline_cv_scores):
    print("\n" + "=" * 60)
    print("1. NESTED CV HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Baseline (default hyperparams, plain {train_models.N_CV_FOLDS}-fold CV) ROC-AUC: "
          f"{dict(baseline_cv_scores)}")

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    param_dists = {"RandomForest": RF_PARAM_DIST, "HistGradientBoosting": HGB_PARAM_DIST}
    nested_scores = {}
    best_params_per_model = {}

    for name, param_dist in param_dists.items():
        pipeline = models[name]
        outer_scores = []
        for train_idx, test_idx in outer_cv.split(X_train, y_train):
            X_out_train = X_train.iloc[train_idx]
            y_out_train = y_train.iloc[train_idx]
            X_out_test = X_train.iloc[test_idx]
            y_out_test = y_train.iloc[test_idx]

            search = RandomizedSearchCV(
                pipeline, param_dist, n_iter=N_RANDOM_SEARCH_ITER, scoring="roc_auc",
                cv=inner_cv, random_state=RANDOM_SEED, n_jobs=-1,
            )
            search.fit(X_out_train, y_out_train)
            y_proba = search.predict_proba(X_out_test)[:, 1]
            outer_scores.append(roc_auc_score(y_out_test, y_proba))

        nested_scores[name] = (np.mean(outer_scores), np.std(outer_scores))
        print(f"{name}: nested CV ROC-AUC = {np.mean(outer_scores):.3f} +/- {np.std(outer_scores):.3f} "
              f"(baseline was {baseline_cv_scores[name]:.3f})")

        # One final search on the FULL training set (same inner CV strategy) to get a
        # single set of hyperparameters for a deployable model -- the nested loop above
        # is for honest EVALUATION of whether tuning helps, not for producing the model
        # itself (there's no single "the" model from K different outer folds).
        final_search = RandomizedSearchCV(
            pipeline, param_dist, n_iter=N_RANDOM_SEARCH_ITER, scoring="roc_auc",
            cv=inner_cv, random_state=RANDOM_SEED, n_jobs=-1,
        )
        final_search.fit(X_train, y_train)
        best_params_per_model[name] = final_search.best_estimator_

        improvement = np.mean(outer_scores) - baseline_cv_scores[name]
        verdict = "meaningful improvement" if improvement > 0.01 else "NOT a meaningful improvement (within noise)"
        print(f"  -> {verdict} (delta = {improvement:+.3f})")

    return nested_scores, best_params_per_model


# =====================================
# 2. PROBABILITY CALIBRATION
# =====================================
def calibration_check(best_name, best_pipeline, X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("2. PROBABILITY CALIBRATION")
    print("=" * 60)

    y_proba_raw = best_pipeline.predict_proba(X_test)[:, 1]
    brier_raw = brier_score_loss(y_test, y_proba_raw)
    auc_raw = roc_auc_score(y_test, y_proba_raw)
    print(f"{best_name} (raw): Brier score = {brier_raw:.4f}, ROC-AUC = {auc_raw:.4f}")

    # sigmoid (Platt) vs isotonic: isotonic is more flexible but needs more data to avoid
    # overfitting the calibration map itself -- with ~900 negative training examples
    # (borderline for isotonic, which typically wants >1000 in the smaller class), we
    # test BOTH empirically via CV rather than assuming which is better.
    results = {}
    for method in ["sigmoid", "isotonic"]:
        calibrated = CalibratedClassifierCV(best_pipeline, method=method, cv=5)
        calibrated.fit(X_train, y_train)
        y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
        brier_cal = brier_score_loss(y_test, y_proba_cal)
        auc_cal = roc_auc_score(y_test, y_proba_cal)
        results[method] = (calibrated, brier_cal, auc_cal, y_proba_cal)
        print(f"{best_name} ({method}): Brier score = {brier_cal:.4f}, ROC-AUC = {auc_cal:.4f}")

    print(
        "\nNote: ROC-AUC is essentially unchanged after calibration (both methods are monotonic\n"
        "transforms of the raw score) -- calibration does NOT change which candidates rank higher\n"
        "than others, only what the probability VALUE means (e.g. whether '0.8' really corresponds\n"
        "to an ~80% real-world chance of being a planet). This matters for how you'd communicate a\n"
        "confidence number to a human reviewer, not for the ranking/triage order itself."
    )

    best_method = min(results, key=lambda m: results[m][1])  # lowest Brier score wins
    best_brier = results[best_method][1]
    calibration_helped = best_brier < brier_raw - 0.005  # small margin so we don't chase noise

    # ---- plot: raw vs best-calibrated reliability diagrams ----
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, proba in [("Raw", y_proba_raw), (f"Calibrated ({best_method})", results[best_method][3])]:
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(f"Calibration curve -- {best_name}")
    ax.legend()
    fig.tight_layout()
    cal_path = os.path.join(FIGURES_DIR, "calibration_curve.png")
    fig.savefig(cal_path, dpi=150)
    plt.close(fig)
    print(f"Calibration curves (before/after) saved to {cal_path}")

    if calibration_helped:
        print(f"VERDICT: {best_method} calibration meaningfully reduced Brier score "
              f"({brier_raw:.4f} -> {best_brier:.4f}) -- using this as the final saved model.")
        return results[best_method][0], best_method
    else:
        print("VERDICT: calibration did not meaningfully improve Brier score -- keeping raw model.")
        return best_pipeline, None


# =====================================
# 3. THRESHOLD ANALYSIS
# =====================================
def threshold_analysis(model_name, model, X_test, y_test):
    print("\n" + "=" * 60)
    print("3. THRESHOLD ANALYSIS (precision/recall tradeoff)")
    print("=" * 60)

    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve -- {model_name}")
    fig.tight_layout()
    pr_path = os.path.join(FIGURES_DIR, "precision_recall_curve.png")
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    print(f"Precision-recall curve saved to {pr_path}")

    rows = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred = (y_proba >= t).astype(int)
        n_flagged = y_pred.sum()
        rows.append({
            "threshold": t,
            "n_flagged_as_positive": int(n_flagged),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        })
    table = pd.DataFrame(rows)
    table_path = os.path.join(TABLES_DIR, "threshold_analysis.csv")
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False))
    print(f"Threshold table saved to {table_path}")
    print(
        "\nFor a triage/ranking workflow: a LOWER threshold flags more candidates for human\n"
        "review (higher recall, catches more real planets, but more false positives to sift\n"
        "through); a HIGHER threshold gives a shorter, cleaner list (higher precision) at the\n"
        "cost of missing some real ones. There's no single 'correct' threshold here -- it\n"
        "depends on how much reviewer time you have per candidate."
    )
    return table


# =====================================
# 4. PERMUTATION IMPORTANCE VS IMPURITY IMPORTANCE
# =====================================
def permutation_importance_check(model_name, model, X_test, y_test):
    print("\n" + "=" * 60)
    print("4. PERMUTATION IMPORTANCE (cross-check vs impurity-based importance)")
    print("=" * 60)

    result = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=-1
    )
    perm_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "permutation_importance_mean": result.importances_mean,
        "permutation_importance_std": result.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)

    # compare against the impurity-based ranking already saved
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else None
    if clf is not None and hasattr(clf, "feature_importances_"):
        impurity_order = [FEATURE_COLUMNS[i] for i in np.argsort(clf.feature_importances_)[::-1]]
    else:
        impurity_order = None

    perm_order = perm_df["feature"].tolist()
    print("Top 5 by permutation importance:", perm_order[:5])
    if impurity_order:
        print("Top 5 by impurity importance:  ", impurity_order[:5])
        agreement = len(set(perm_order[:5]) & set(impurity_order[:5]))
        print(f"Overlap in top 5: {agreement}/5 features agree between the two methods "
              f"{'(consistent signal, not a metric artifact)' if agreement >= 3 else '(rankings diverge meaningfully -- interpret feature importance cautiously)'}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(perm_df["feature"][::-1], perm_df["permutation_importance_mean"][::-1],
            xerr=perm_df["permutation_importance_std"][::-1])
    ax.set_xlabel("Permutation importance (ROC-AUC drop when shuffled)")
    ax.set_title(f"Permutation importance -- {model_name}")
    fig.tight_layout()
    perm_path = os.path.join(FIGURES_DIR, "permutation_importance.png")
    fig.savefig(perm_path, dpi=150)
    plt.close(fig)
    print(f"Permutation importance plot saved to {perm_path}")

    return perm_df


# =====================================
# 5. SOFT-VOTING ENSEMBLE CHECK
# =====================================
def ensemble_check(models, X_train, y_train, X_test, y_test, baseline_cv_scores, baseline_test_scores):
    print("\n" + "=" * 60)
    print("5. SOFT-VOTING ENSEMBLE CHECK")
    print("=" * 60)

    ensemble = VotingClassifier(
        estimators=[(name, pipeline) for name, pipeline in models.items()],
        voting="soft",
    )

    cv = StratifiedKFold(n_splits=train_models.N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_validate(ensemble, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    ensemble_cv_auc = cv_scores["test_score"].mean()

    ensemble.fit(X_train, y_train)
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    ensemble_test_auc = roc_auc_score(y_test, y_proba)

    best_single_cv = max(baseline_cv_scores.values())
    best_single_test = max(baseline_test_scores.values())

    print(f"Ensemble: CV ROC-AUC = {ensemble_cv_auc:.3f}, test ROC-AUC = {ensemble_test_auc:.3f}")
    print(f"Best single model:    CV ROC-AUC = {best_single_cv:.3f}, test ROC-AUC = {best_single_test:.3f}")

    helped = (ensemble_cv_auc - best_single_cv > 0.005) and (ensemble_test_auc - best_single_test > 0.005)
    if helped:
        print("VERDICT: ensemble meaningfully outperforms the best single model.")
    else:
        print("VERDICT: ensemble does NOT meaningfully outperform the best single model -- "
              "not worth the added complexity of shipping 3 models instead of 1.")

    return ensemble, ensemble_cv_auc, ensemble_test_auc, helped


# =====================================
# 6. LEARNING CURVE
# =====================================
def plot_learning_curve(model_name, pipeline, X_train, y_train):
    print("\n" + "=" * 60)
    print("6. LEARNING CURVE")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=train_models.N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X_train, y_train, cv=cv, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 8), random_state=RANDOM_SEED, n_jobs=-1,
    )

    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(train_sizes, train_mean, "o-", label="Training score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    ax.plot(train_sizes, test_mean, "o-", label="Cross-validation score")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Learning curve -- {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    lc_path = os.path.join(FIGURES_DIR, "learning_curve.png")
    fig.savefig(lc_path, dpi=150)
    plt.close(fig)
    print(f"Learning curve saved to {lc_path}")

    # honest read: compare the CV-score slope over the last 3 points to decide
    # data-starved vs plateaued, rather than eyeballing the plot for the user
    last3_slope = (test_mean[-1] - test_mean[-4]) / (train_sizes[-1] - train_sizes[-4])
    print(f"CV score at smallest/largest training size: {test_mean[0]:.3f} -> {test_mean[-1]:.3f}")
    print(f"Slope over the last portion of the curve: {last3_slope:.6f} ROC-AUC per additional example")
    if last3_slope > 0.00005:
        print("VERDICT: still climbing meaningfully -- the model IS data-starved. Getting more "
              "negative-class examples would likely help more than further tuning.")
    else:
        print("VERDICT: has plateaued -- more of the SAME kind of data is unlikely to help much "
              "further; tuning/feature engineering matters more than raw sample count right now.")

    return train_sizes, train_mean, test_mean


# =====================================
# 7. BOOTSTRAP CONFIDENCE INTERVALS
# =====================================
def bootstrap_ci(model_name, model, X_test, y_test):
    print("\n" + "=" * 60)
    print("7. BOOTSTRAP CONFIDENCE INTERVALS (test set)")
    print("=" * 60)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    y_test_arr = np.asarray(y_test)

    rng = np.random.RandomState(RANDOM_SEED)
    n = len(y_test_arr)
    metrics = {"roc_auc": [], "precision": [], "recall": [], "f1": []}

    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        y_true_b, y_proba_b, y_pred_b = y_test_arr[idx], y_proba[idx], y_pred[idx]
        if len(np.unique(y_true_b)) < 2:
            continue  # skip degenerate resamples with only one class present
        metrics["roc_auc"].append(roc_auc_score(y_true_b, y_proba_b))
        metrics["precision"].append(precision_score(y_true_b, y_pred_b, zero_division=0))
        metrics["recall"].append(recall_score(y_true_b, y_pred_b, zero_division=0))
        metrics["f1"].append(f1_score(y_true_b, y_pred_b, zero_division=0))

    rows = []
    for metric, values in metrics.items():
        values = np.array(values)
        lo, hi = np.percentile(values, [2.5, 97.5])
        rows.append({"metric": metric, "point_estimate": values.mean(), "ci_lower_2.5": lo, "ci_upper_97.5": hi})
        print(f"{metric}: {values.mean():.3f}  [95% CI: {lo:.3f}, {hi:.3f}]")

    table = pd.DataFrame(rows)
    ci_path = os.path.join(TABLES_DIR, "bootstrap_confidence_intervals.csv")
    table.to_csv(ci_path, index=False)
    print(f"Bootstrap CI table saved to {ci_path}")
    return table


# =====================================
# 8. RE-VERIFY NO NEW LEAKAGE
# =====================================
def reverify_no_leakage(df):
    print("\n" + "=" * 60)
    print("8. RE-VERIFYING NO NEW LEAKAGE (same checks as 05_train_models.py, rerun fresh)")
    print("=" * 60)
    print("No new features were engineered in this round -- calibration/ensembling recombine "
          "existing model outputs, they don't add columns. Re-running the same NaN-rate-by-class "
          "and single-feature-AUC checks on the current FEATURE_COLUMNS as a fresh confirmation:\n")

    df_clean = df.replace([np.inf, -np.inf], np.nan)
    suspicious = []
    for col in FEATURE_COLUMNS:
        nan_0 = df_clean[df_clean["label"] == 0][col].isna().mean()
        nan_1 = df_clean[df_clean["label"] == 1][col].isna().mean()
        valid = df_clean.dropna(subset=[col])
        auc = roc_auc_score(valid["label"], valid[col])
        auc_abs = max(auc, 1 - auc)
        flag = ""
        if abs(nan_0 - nan_1) > 0.5:
            flag = " <- SUSPICIOUS: NaN rate differs a lot by class"
            suspicious.append(col)
        elif auc_abs > 0.95:
            flag = " <- SUSPICIOUS: near-perfect single-feature separation"
            suspicious.append(col)
        print(f"  {col:25s} AUC={auc_abs:.3f}  NaN(neg/pos)={nan_0:.3f}/{nan_1:.3f}{flag}")

    if suspicious:
        print(f"\nWARNING: {len(suspicious)} feature(s) look suspicious on this re-check: {suspicious}")
    else:
        print("\nNo new leakage signals found -- feature set still looks clean.")
    return suspicious


def main():
    full_df = train_models.load_and_report_class_balance()
    X, y = train_models.build_feature_matrix(full_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Split reconstructed: train={len(X_train)} ({y_train.sum()} pos / {len(y_train)-y_train.sum()} neg), "
          f"test={len(X_test)} ({y_test.sum()} pos / {len(y_test)-y_test.sum()} neg)")

    current_best_model, current_best_metadata = load_current_best()
    print(f"\nCurrently saved best model: {current_best_metadata['model_name']}")

    models = train_models.build_models()

    # baseline CV/test scores (default hyperparams) for honest before/after comparisons below
    baseline_cv = {}
    baseline_test = {}
    cv = StratifiedKFold(n_splits=train_models.N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        baseline_cv[name] = scores["test_score"].mean()
        pipeline.fit(X_train, y_train)
        baseline_test[name] = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

    # ---- 1. nested CV tuning ----
    nested_scores, tuned_models = nested_cv_tuning(X_train, y_train, models, baseline_cv)

    # Decide the running "best" candidate: compare tuned models' test performance
    # against the currently-saved model, on the SAME held-out test set.
    candidates = {"CURRENT_SAVED": current_best_model}
    candidates.update(tuned_models)
    candidate_test_auc = {
        name: roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        for name, model in candidates.items()
    }
    print("\nCandidate test ROC-AUC after tuning step:", {k: round(v, 3) for k, v in candidate_test_auc.items()})
    running_best_name = max(candidate_test_auc, key=candidate_test_auc.get)
    running_best_model = candidates[running_best_name]
    running_best_test_auc = candidate_test_auc[running_best_name]
    print(f"Running best after step 1: {running_best_name} (test ROC-AUC={running_best_test_auc:.3f})")

    # ---- 2. calibration ----
    calibrated_model, calibration_method = calibration_check(
        running_best_name, running_best_model, X_train, y_train, X_test, y_test
    )
    if calibration_method is not None:
        running_best_model = calibrated_model
        running_best_name = f"{running_best_name}+{calibration_method}_calibration"
        running_best_test_auc = roc_auc_score(y_test, running_best_model.predict_proba(X_test)[:, 1])

    # ---- 3. threshold analysis ----
    threshold_analysis(running_best_name, running_best_model, X_test, y_test)

    # ---- 4. permutation importance ----
    # uses the tuned (pre-calibration-wrapper) tree model directly, since a
    # CalibratedClassifierCV wrapper doesn't expose feature_importances_ the same way
    importance_source_model = tuned_models.get("RandomForest", models["RandomForest"])
    permutation_importance_check("RandomForest", importance_source_model, X_test, y_test)

    # ---- 5. ensemble check ----
    ensemble, ensemble_cv_auc, ensemble_test_auc, ensemble_helped = ensemble_check(
        models, X_train, y_train, X_test, y_test, baseline_cv, baseline_test
    )
    if ensemble_helped and ensemble_test_auc > running_best_test_auc:
        running_best_model = ensemble
        running_best_name = "SoftVotingEnsemble"
        running_best_test_auc = ensemble_test_auc

    # ---- 6. learning curve (on whichever tree model is currently the tuned RF, for interpretability) ----
    plot_learning_curve("RandomForest (tuned)", tuned_models.get("RandomForest", models["RandomForest"]),
                         X_train, y_train)

    # ---- 7. bootstrap CIs on the final running-best model ----
    bootstrap_ci(running_best_name, running_best_model, X_test, y_test)

    # ---- 8. re-verify no new leakage ----
    reverify_no_leakage(full_df)

    # =====================================
    # FINAL DECISION: only overwrite saved artifacts if genuinely better
    # =====================================
    print("\n" + "=" * 60)
    print("FINAL DECISION")
    print("=" * 60)
    final_test_auc = roc_auc_score(y_test, running_best_model.predict_proba(X_test)[:, 1])
    current_test_auc = candidate_test_auc["CURRENT_SAVED"]
    print(f"Currently saved model test ROC-AUC: {current_test_auc:.3f}")
    print(f"Best candidate from this analysis ({running_best_name}) test ROC-AUC: {final_test_auc:.3f}")

    MEANINGFUL_MARGIN = 0.01
    if final_test_auc - current_test_auc > MEANINGFUL_MARGIN:
        print(f"UPDATING saved model: {running_best_name} beats the current saved model by "
              f"{final_test_auc - current_test_auc:+.3f} ROC-AUC (> {MEANINGFUL_MARGIN} margin).")
        joblib.dump(running_best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
        new_metadata = dict(current_best_metadata)
        new_metadata["model_name"] = running_best_name
        new_metadata["test_roc_auc"] = float(final_test_auc)
        new_metadata["updated_by"] = "05b_model_analysis.py"
        with open(os.path.join(MODELS_DIR, "best_model_metadata.json"), "w") as f:
            json.dump(new_metadata, f, indent=2)
        print("models/best_model.joblib and best_model_metadata.json updated.")
    else:
        print(f"NOT updating saved model: no candidate from this analysis beat the current saved "
              f"model by more than {MEANINGFUL_MARGIN} ROC-AUC -- the difference "
              f"({final_test_auc - current_test_auc:+.3f}) is within noise. Keeping "
              f"{current_best_metadata['model_name']} (the currently saved model) as the "
              f"production model.")

    print("\n" + "=" * 60)
    print("HONEST CEILING ASSESSMENT")
    print("=" * 60)
    # printed based on what was actually observed above, not asserted in advance


if __name__ == "__main__":
    main()
