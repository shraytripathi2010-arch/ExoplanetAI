"""
05_train_models.py

Train and compare classical ML models (Logistic Regression, Random Forest,
HistGradientBoosting) on the labeled dataset from 04_build_training_dataset.py, to
distinguish confirmed exoplanet hosts (label=1) from confirmed TESS false
positives (label=0, TOI disposition='FP').

VERIFIED BEFORE WRITING THIS SCRIPT (not assumed):
- Real class balance: 4,333 positive / 1,141 negative (3.8:1) -- imbalanced,
  handled below via class weighting + stratified splits/folds, NOT ignored.
- Negative-class provenance confirmed: every label=0 row is TIC-ID-named and
  has no confirmed-planet catalog match (pl_names is NaN for all of them) --
  these are genuinely TOI false positives, not just "unconfirmed" stars.

FEATURE LIST -- WHY SEVERAL REAL COLUMNS ARE DELIBERATELY EXCLUDED:

This dataset has real, verified data-leakage traps that would let a model
achieve a great-looking score for entirely the wrong reason. Checked every
candidate column's NaN-rate-by-class and single-feature AUC against the
actual data before finalizing this list -- not guessed.

EXCLUDED, and why:
  - st_rad, st_teff, st_mass, points_removed_pct, n_original, n_after_quality,
    n_after_outliers, n_final, pre_norm_median_flux: 100% NaN for every
    label=0 row (these were only ever joined in for the positive class in
    stage 4/2). A model would trivially learn "NaN in this column -> false
    positive", which is a pipeline artifact, not astrophysics.
  - n_points: single-feature AUC 0.622 -- looks informative, but it's
    leakage from OUR OWN pipeline: the negative-class TLS run
    (03_transit_search_negative.py) bins any star over 30,000 points down to
    ~15-19k for speed, while the positive-class run never did this (some
    positive stars go up to 183,114 points). The resulting size difference
    reflects a speed optimization we applied asymmetrically, not a real
    difference between planets and false positives.
  - T0: an absolute observation timestamp (when the transit was seen), not a
    physical property of the signal. Also showed mild leakage (AUC 0.565),
    plausibly reflecting the two classes being observed in different TESS
    campaigns -- excluded on both physical-relevance and leakage grounds.
  - host, pl_names, n_planets_in_catalog, ra, dec, flux_source: identifiers/
    metadata, not predictive features (ra/dec are also 100% NaN for label=0
    for the same reason as the stellar params above).

KEPT: SDE, SDE_raw, FAP, period, period_uncertainty, duration, depth,
depth_mean, depth_mean_std, depth_mean_even, depth_mean_odd,
odd_even_mismatch, rp_rs, snr, transit_count, distinct_transit_count,
empty_transit_count. Single-feature AUCs for these (0.50-0.75) look like real
signal, not artifacts -- SDE (0.749) and SNR (0.715) being the strongest
matches domain expectation (detection strength should differ between real
transits and false positives), not a red flag.

DATA QUALITY ISSUES HANDLED:
  - period_uncertainty has 68 infinite values, odd_even_mismatch has 1 --
    occurs in both classes at a low, comparable rate (not a leakage pattern,
    just a TLS numerical edge case). Converted to NaN and imputed like any
    other missing value.
  - FAP is NaN for 55% of label=1 rows vs 17% of label=0 rows (TLS returns
    NaN when SDE is below its calibration floor, i.e. a weak/null
    detection). This asymmetry is plausibly a REAL signal, not a bug: the
    negative class is TOI candidates, which by definition had SOME
    detectable signal to be catalogued in the first place, whereas the
    positive class includes confirmed hosts with genuinely subtle transits
    our re-search sometimes can't recover at OVERSAMPLING_FACTOR=1. Kept as
    a feature; imputed with 1.0 (max false-alarm probability), since a
    NaN here specifically means "not even significant enough to calibrate".

WHY HistGradientBoostingClassifier, NOT XGBOOST/LIGHTGBM: initially picked
XGBoost over LightGBM for lower macOS install friction (LightGBM needs
Homebrew's libomp) -- except XGBoost's prebuilt wheel ALSO dynamically links
libomp on macOS and failed to load with no Homebrew installed on this
machine. Rather than asking you to install Homebrew for one library, used
scikit-learn's own HistGradientBoostingClassifier: same family of algorithm
(histogram-based gradient boosting, the same idea LightGBM popularized),
zero external dependencies since it ships with sklearn (already installed
and working), and it natively supports class_weight like the other two
models here. At this dataset size (~5,500 rows, 17 features) there's no
meaningful performance gap between this and XGBoost/LightGBM -- the
difference that mattered here was which one actually ran on this machine
without extra setup.

HOW TRUSTWORTHY ARE THESE RESULTS? Read the printed caveat at the end of a
run, or search "SCIENTIFIC VALIDITY" in this file -- short version:
preliminary/directional, not a validated discovery pipeline, primarily
because of the n_points-style leakage risk pattern (we caught the ones we
could verify, but that doesn't guarantee there are no more subtle ones) and
the modest, imbalanced sample size.

Author: Ray's Exoplanet AI Project
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # headless plotting -- no GUI backend needed/available
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =====================================
# SETTINGS
# =====================================
RANDOM_SEED = 42  # used everywhere below (split, CV folds, model init) for reproducibility
TEST_SIZE = 0.2

# Fraction of newly-labelled stars sent to TEST once the frozen manifest is
# full. Was 0.2 (matching the original split); raised to 0.5 on 2026-08-04.
#
# WHY IT WAS RAISED
# The classifier is ceiling-bound and the binding constraint moved from the
# training set to the test set. Measured:
#   * training side: +262 stars on ~4,386 predicts ~+0.0011 AUC from the fitted
#     learning curve -- unmeasurable, well under the noise floor.
#   * test side: the smallest effect a 1,098-star test set can certify at
#     ci_lo>0 is ~0.0097. The best real candidate found (CatBoost, +0.0080,
#     positive on 8/8 resamples on both populations) sits BELOW that and so
#     cannot be promoted however real it is.
#
# WHY 0.5 AND NOT 1.0
# Routing everything to test grows resolution fastest but starves the
# continuous-retraining pipeline completely -- challengers would have no new
# data and would simply reproduce the incumbent, suspending Phase 3 Item 2's
# whole purpose. 0.5 still gives test growth 2.5x the old rate while keeping
# training fed, which matters because the ~150-180 labels/year arriving is a
# multi-year stream and the pipeline needs to stay exercised over it.
#
# Assignment stays a deterministic md5 of the host name, so a given star lands
# on the same side no matter when or how often it is seen -- that property is
# what makes the split reproducible and must not be traded away for balance.
#
# Set back toward 0.2 once the test set reaches TEST_GROWTH_TARGET, the size
# that can certify a +0.0080 effect.
POST_FREEZE_TEST_FRACTION = 0.5
TEST_GROWTH_TARGET = 1900  # documentation, not enforced -- see the note above
N_CV_FOLDS = 5
CONFUSION_MATRIX_THRESHOLD = 0.5  # see note near the confusion-matrix code for why this
                                   # is just one reference operating point, not necessarily
                                   # how you'd actually use this model's output

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "training_dataset", "training.csv")
SPLIT_MANIFEST_PATH = os.path.join(SCRIPT_DIR, "..", "data", "training_dataset",
                                    "split_manifest.json")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
TABLES_DIR = os.path.join(SCRIPT_DIR, "..", "results", "tables")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "results", "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# See the module docstring's "FEATURE LIST" section for why each excluded
# column is excluded -- this isn't an arbitrary subset.
#
# v2 ADDITIONS (all verified on the full 5,474-star dataset, not just a pilot
# sample, before being added -- see module docstring for the full leakage
# check results):
#   st_rad, st_teff: previously unusable for the negative class due to a join
#     bug in 04_build_training_dataset.py (data existed, was never wired in).
#     Fixed. Single-feature AUC 0.70-0.72 -- stronger than most existing
#     features. st_mass is NOT included: the TOI archive table has no
#     stellar mass column at all (checked live), so it stays 100% NaN for
#     the negative class -- a genuine data limitation, not a fixable bug.
#   chi2red_min, depth_consistency_std, secondary_eclipse_depth,
#     transit_shape_ratio: newly extracted from a full TLS re-run (these
#     fields/derivations were never saved in the original stage-3 run).
#     AUCs 0.53-0.65 on the full dataset -- real but modest; a small-sample
#     pilot (80 stars) had overestimated transit_shape_ratio's AUC at 0.708,
#     the full run corrected that to 0.570. Kept anyway since none show
#     leakage and weak-but-real signal can still help tree models in
#     combination with other features.
#   depth_duration_ratio: engineered from existing depth/duration (transit
#     geometry consistency check -- a known vetting heuristic). AUC 0.649.
FEATURE_COLUMNS = [
    "SDE", "SDE_raw", "FAP", "period", "period_uncertainty", "duration", "depth",
    "depth_mean", "depth_mean_std", "depth_mean_even", "depth_mean_odd",
    "odd_even_mismatch", "rp_rs", "snr", "transit_count", "distinct_transit_count",
    "empty_transit_count",
    "st_rad", "st_teff",
    "chi2red_min", "depth_consistency_std", "secondary_eclipse_depth", "transit_shape_ratio",
    "depth_duration_ratio",
]


def load_and_report_class_balance():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"ERROR: {DATA_PATH} not found -- run 04_build_training_dataset.py first.")
    df = pd.read_csv(DATA_PATH)

    if "label" not in df.columns:
        raise SystemExit("ERROR: no 'label' column in training.csv -- check 04_build_training_dataset.py's output.")

    counts = df["label"].value_counts()
    n_pos = int(counts.get(1, 0))
    n_neg = int(counts.get(0, 0))

    print("=" * 50)
    print("CLASS BALANCE (verify this matches what you expect)")
    print("=" * 50)
    print(f"Positive (label=1, confirmed planet):     {n_pos}")
    print(f"Negative (label=0, confirmed FP):          {n_neg}")
    print(f"Total:                                     {len(df)}")

    if n_neg == 0:
        raise SystemExit(
            "\nSTOPPING: there are ZERO negative-class (label=0) examples in training.csv.\n"
            "A classifier trained on this would trivially predict 'planet' for every input\n"
            "and report a meaningless ~100% accuracy -- it would have learned nothing, since\n"
            "there's nothing to distinguish FROM. Go build the negative-class pipeline\n"
            "(01_download_negative.py -> 02_preprocess_negative.py -> "
            "03_transit_search_negative.py -> 04_build_training_dataset.py --negative-results)\n"
            "before running this script."
        )
    if n_neg < 30:
        print(
            f"\nWARNING: only {n_neg} negative examples. Stratified CV and train/test splits "
            f"below will still run, but with this few, expect high variance between CV folds "
            f"and treat every metric here as a rough signal, not a precise estimate."
        )

    print(f"\nImbalance ratio: {n_pos / n_neg:.1f} : 1 (positive:negative)")
    print("Handled via class_weight='balanced' (all three models) below,")
    print("plus stratified splits/folds throughout so this ratio is preserved in every subset.\n")

    return df


def build_feature_matrix(df):
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise SystemExit(
            f"ERROR: expected feature columns not found in training.csv: {missing_cols}\n"
            f"The actual columns are: {list(df.columns)}\n"
            f"training.csv's schema may have changed -- update FEATURE_COLUMNS above to match."
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].copy()

    # Infinite values (period_uncertainty, odd_even_mismatch -- see module
    # docstring) aren't handled by SimpleImputer's NaN-based logic, so
    # convert them first. Same treatment as any other missing value.
    X = X.replace([np.inf, -np.inf], np.nan)

    # FAP specifically: NaN means "SDE too low to calibrate a false-alarm
    # probability", which is itself informative (see module docstring) --
    # fill with 1.0 (max FAP) rather than a generic median, since the
    # generic median would misrepresent what a NaN here actually means.
    X["FAP"] = X["FAP"].fillna(1.0)

    return X, y


def build_models():
    """Each model is wrapped in a Pipeline so imputation (and scaling, for
    LR) is fit ONLY on each CV fold's training portion -- fitting it on the
    whole dataset up front would leak test-fold statistics into training.

    All three models use class_weight='balanced' for consistent, comparable
    imbalance handling (each computes it slightly differently internally,
    but the intent -- upweight the minority label=0 class -- is the same)."""
    models = {
        "LogisticRegression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),  # LR is sensitive to feature scale; tree models below aren't
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
            )),
        ]),
        "RandomForest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("impute", SimpleImputer(strategy="median")),  # HGB can handle NaN natively, but
            # imputing here anyway keeps every model seeing identical input data for a fair comparison
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, max_depth=4, learning_rate=0.05,
                class_weight="balanced", random_state=RANDOM_SEED,
            )),
        ]),
    }
    return models


def run_cross_validation(models, X_train, y_train):
    """Stratified k-fold CV on the TRAINING split only, for model comparison.
    With ~900 negative examples in the training set (80% of 1,141), a single
    train/test split could easily land an unrepresentative mix in either
    side -- CV averages over N_CV_FOLDS different splits to reduce that risk,
    per the user's explicit request not to trust a single split here."""
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    cv_rows = []
    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric in scoring:
            row[f"cv_{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"cv_{metric}_std"] = scores[f"test_{metric}"].std()
        cv_rows.append(row)
        print(f"{name}: CV ROC-AUC = {row['cv_roc_auc_mean']:.3f} +/- {row['cv_roc_auc_std']:.3f} "
              f"(across {N_CV_FOLDS} folds)")

    return pd.DataFrame(cv_rows)


def evaluate_on_test_set(models, X_train, y_train, X_test, y_test):
    """Fit each model on the FULL training set, evaluate ONCE on the held-out
    test set. This is the honest generalization estimate -- CV above is for
    comparing/selecting models, this is for reporting what you'd actually
    expect on genuinely new data.

    NOTE ON THE CONFUSION MATRIX / 0.5 THRESHOLD: this pipeline's real use
    case (per the original ask) is RANKING candidate stars by probability,
    not handing out hard yes/no verdicts -- you'd look at the top of a
    sorted list of unknown stars, not apply a rigid cutoff. A confusion
    matrix needs SOME threshold to exist at all, so 0.5 is used below as one
    reference operating point for comparison across models, not a claim that
    0.5 is the "right" cutoff for how you'd use this in practice. If you
    care more about not missing real planets (recall) than about avoiding
    false alarms (precision), or vice versa, the probability scores
    themselves (not this table) are what to act on -- move the threshold
    down for higher recall/lower precision, or up for the reverse."""
    test_rows = []
    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= CONFUSION_MATRIX_THRESHOLD).astype(int)

        row = {
            "model": name,
            "test_roc_auc": roc_auc_score(y_test, y_proba),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
        }
        test_rows.append(row)
        print(f"{name}: test ROC-AUC={row['test_roc_auc']:.3f}  "
              f"precision={row['test_precision']:.3f}  recall={row['test_recall']:.3f}  "
              f"f1={row['test_f1']:.3f}")

    return pd.DataFrame(test_rows), fitted_models


def plot_confusion_matrices(fitted_models, X_test, y_test, path):
    fig, axes = plt.subplots(1, len(fitted_models), figsize=(5 * len(fitted_models), 4.5))
    for ax, (name, pipeline) in zip(axes, fitted_models.items()):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= CONFUSION_MATRIX_THRESHOLD).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["False Positive", "Confirmed Planet"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model_name, pipeline, path, X_test=None, y_test=None):
    """Native impurity-based importance (RF) or coefficient magnitude (LR)
    when available; falls back to permutation importance (on held-out test
    data) otherwise -- needed for HistGradientBoostingClassifier, which
    exposes neither feature_importances_ nor coef_. A prior version of this
    function silently no-op'd for such models while main() still claimed a
    plot was saved -- fixed both the missing fallback and that misleading
    print."""
    clf = pipeline.named_steps["clf"]
    xlabel = "Importance"
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])  # magnitude only -- sign/direction isn't "importance"
    elif X_test is not None and y_test is not None:
        result = permutation_importance(
            pipeline, X_test, y_test, n_repeats=30, random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=-1
        )
        importances = result.importances_mean
        xlabel = "Permutation importance (ROC-AUC drop when shuffled)"
    else:
        print(f"NOTE: {model_name} has no feature_importances_/coef_ attribute and no test "
              f"data was given for a permutation-importance fallback -- skipping importance plot.")
        return False

    order = np.argsort(importances)[::-1]
    sorted_features = [FEATURE_COLUMNS[i] for i in order]
    sorted_importances = importances[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(sorted_features[::-1], sorted_importances[::-1])
    ax.set_xlabel(xlabel)
    ax.set_title(f"Feature importance -- {model_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"\nTop 5 features for {model_name} (why a candidate gets flagged):")
    for feat, imp in zip(sorted_features[:5], sorted_importances[:5]):
        print(f"  {feat}: {imp:.4f}")
    return True


def split_by_host(df):
    """Splits train/test by STABLE STAR ID, not by row position.

    BUG FIXED (found by an audit that compared the split's actual membership
    across dataset versions): this used to be
    `train_test_split(X, y, test_size=0.2, random_state=42)`, which assigns
    rows by POSITION. A fixed seed makes that reproducible only while the row
    count and row order never change -- and they do change, because the
    continuous-retraining label watcher appends newly-confirmed planets to
    training.csv. Measured directly: after 15 rows were appended
    (5,491 -> 5,506), only 1,010 of the original 1,099 test stars were still
    in the test set. 89 stars the deployed model had been TRAINED on had
    silently moved INTO the test set, inflating its apparent test ROC-AUC
    from the true 0.9032 to 0.9113.

    Membership now comes from data/training_dataset/split_manifest.json,
    which records the exact host IDs of the original 4,392/1,099 split that
    models/best_model.joblib was really trained and evaluated on. Stars in
    the manifest keep their original side forever, so every future retrain is
    measured on the same held-out stars and the numbers stay comparable.

    Stars NOT in the manifest (i.e. added after it was frozen) are assigned by
    a deterministic hash of the host name -- stable across runs, processes and
    machines, unlike Python's builtin hash(), which is salted per-process for
    strings. **The split of those new stars changed on 2026-08-04 from 20% to
    50% test** -- see POST_FREEZE_TEST_FRACTION for the measurements behind it.
    Manifest stars are untouched by that change and keep their original side
    forever, so every number ever measured on the frozen split stays
    comparable.

    Use `frozen_test_mask(df)` to evaluate on the original 1,099 manifest test
    hosts only. Any figure meant to be compared against the project's history
    -- the 0.9031 headline above all -- must use that mask, not the growing
    test set, or it is comparing different populations.
    """
    import hashlib

    with open(SPLIT_MANIFEST_PATH) as f:
        manifest = json.load(f)
    test_hosts = set(manifest["test_hosts"])
    train_hosts = set(manifest["train_hosts"])

    def side(host):
        if host in test_hosts:
            return "test"
        if host in train_hosts:
            return "train"
        digest = hashlib.md5(str(host).encode("utf-8")).hexdigest()
        cut = int(POST_FREEZE_TEST_FRACTION * 10000)
        return "test" if (int(digest[:8], 16) % 10000) < cut else "train"

    sides = df["host"].map(side)
    n_new = int((~df["host"].isin(test_hosts | train_hosts)).sum())
    is_test = (sides == "test").to_numpy()
    policy = (f"stable hash, {POST_FREEZE_TEST_FRACTION:.0%} to test")
    print(f"Split by star ID (frozen manifest): {int((~is_test).sum())} train / "
          f"{int(is_test.sum())} test"
          + (f"  [{n_new} post-manifest star(s) {policy}]" if n_new else ""))
    return ~is_test, is_test


def frozen_test_mask(df):
    """The ORIGINAL 1,099 manifest test hosts only, ignoring anything added
    since. Every historical figure in this project -- 0.9031 included -- was
    measured on these stars.

    Exists because the post-freeze allocation now grows the test set: comparing
    a new measurement on the grown set against an old one on the frozen set
    would be comparing different populations, which is exactly the class of
    error the frozen split was created to prevent. Report both, and label
    which is which.
    """
    with open(SPLIT_MANIFEST_PATH) as f:
        manifest = json.load(f)
    return df["host"].isin(set(manifest["test_hosts"])).to_numpy()


def main():
    df = load_and_report_class_balance()
    X, y = build_feature_matrix(df)

    train_mask, test_mask = split_by_host(df)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"Train: {len(X_train)} ({y_train.sum()} positive, {len(y_train) - y_train.sum()} negative)")
    print(f"Test:  {len(X_test)} ({y_test.sum()} positive, {len(y_test) - y_test.sum()} negative)\n")

    models = build_models()

    print("=" * 50)
    print(f"CROSS-VALIDATION ({N_CV_FOLDS}-fold, stratified, on training set only)")
    print("=" * 50)
    cv_results = run_cross_validation(models, X_train, y_train)

    print("\n" + "=" * 50)
    print("HELD-OUT TEST SET EVALUATION")
    print("=" * 50)
    test_results, fitted_models = evaluate_on_test_set(models, X_train, y_train, X_test, y_test)

    comparison = cv_results.merge(test_results, on="model")
    comparison_path = os.path.join(TABLES_DIR, "model_comparison.csv")
    comparison.to_csv(comparison_path, index=False)
    print(f"\nComparison table saved to {comparison_path}")

    # Best model chosen by TEST-set ROC-AUC (the honest generalization
    # estimate), not CV score -- CV is for sanity-checking model choice is
    # stable, but the held-out test set is the number that should actually
    # decide which model you trust.
    best_model_name = test_results.loc[test_results["test_roc_auc"].idxmax(), "model"]
    best_pipeline = fitted_models[best_model_name]
    print(f"\nBest model (by test ROC-AUC): {best_model_name}")

    cm_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plot_confusion_matrices(fitted_models, X_test, y_test, cm_path)
    print(f"Confusion matrices (all models) saved to {cm_path}")

    fi_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plot_saved = plot_feature_importance(best_model_name, best_pipeline, fi_path, X_test=X_test, y_test=y_test)
    if plot_saved:
        print(f"Feature importance plot saved to {fi_path}")

    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"\nBest model ({best_model_name}) saved to {model_path}")

    # A later script (candidate ranking) needs to know EXACTLY which
    # features to compute and in what order -- save this alongside the
    # model rather than hardcoding it a second time somewhere else, which
    # is exactly the kind of duplication that drifts out of sync silently.
    metadata = {
        "model_name": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "random_seed": RANDOM_SEED,
        "confusion_matrix_threshold": CONFUSION_MATRIX_THRESHOLD,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "class_balance": {"positive": int(y.sum()), "negative": int(len(y) - y.sum())},
    }
    metadata_path = os.path.join(MODELS_DIR, "best_model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to {metadata_path}")

    print("\n" + "=" * 50)
    print("SCIENTIFIC VALIDITY -- read before presenting these results")
    print("=" * 50)
    print(
        "Treat this as PRELIMINARY / DIRECTIONAL, not a validated discovery pipeline:\n"
        "1. ~1,141 negative examples is a real, curated sample, but still modest for a\n"
        "   17-feature classifier -- expect meaningful metric swings if the negative-class\n"
        "   population grows later (e.g. if you add TOI 'FA' dispositions).\n"
        "2. We caught several real leakage sources during feature selection (n_points,\n"
        "   T0, stellar params, QC columns -- see module docstring) by explicitly checking\n"
        "   NaN-rate-by-class and single-feature AUC against the real data. That process\n"
        "   caught what we checked for, not everything that could possibly be wrong --\n"
        "   don't treat a high ROC-AUC as proof the model learned real astrophysics rather\n"
        "   than some remaining artifact we didn't think to check.\n"
        "3. Both classes were searched with identical TLS settings for comparability, but\n"
        "   they come from different underlying MAST pipelines/observing strategies\n"
        "   (confirmed hosts vs. TOI candidates) -- some residual systematic difference\n"
        "   between the two populations, unrelated to being a real planet, is plausible\n"
        "   and would inflate the reported metrics without you being able to see it here.\n"
        "Bottom line: fine for ranking/triaging candidates for human follow-up, not fine\n"
        "for a standalone claim of 'X% accurate exoplanet detection'."
    )


if __name__ == "__main__":
    main()
