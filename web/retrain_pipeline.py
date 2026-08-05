"""
retrain_pipeline.py -- continuous retraining infrastructure (Item 2, Part B).

Independent of the Kepler pilot (Part A) -- this only ever touches TESS
data, reusing this project's existing real/negative pipeline exactly as it
already works, just triggered periodically instead of by hand.

Three responsibilities, matching the three-step request:
  1. find_new_labeled_examples() -- live re-query of the NASA Exoplanet
     Archive (confirmed planets) and TOI table (false positives), diffed
     against what's already a row in data/training_dataset/training.csv.
  2. process_and_append_new_examples() -- runs each newly-found star
     through the EXACT SAME download -> preprocess -> TLS feature
     extraction functions the original pipeline uses (imported, not
     rewritten), then appends a matching row to training.csv.
  3. maybe_trigger_retrain() -- once enough new rows have accumulated,
     retrains via 05_train_models.py's own model-building code, compares
     to the current production model via the same paired-bootstrap CI
     standard used throughout this project, and ONLY promotes if it wins
     by more than noise. Every attempt (promoted or not) is logged.

Never auto-replaces best_model.joblib outside of this explicit gate.
"""
import os
import sys
import time
import json
import shutil
import importlib

import numpy as np
import pandas as pd
from sklearn.base import clone

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(WEB_DIR, "..", "code")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, WEB_DIR)
import db

m06 = importlib.import_module("06_download_unknown")
m02 = importlib.import_module("02_preprocess")
m05 = importlib.import_module("05_train_models")

PROJECT_ROOT = os.path.join(WEB_DIR, "..")
TRAINING_CSV = os.path.join(PROJECT_ROOT, "data", "training_dataset", "training.csv")
RETRAIN_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "retrain_pipeline", "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_VERSIONS_DIR = os.path.join(MODELS_DIR, "versions")
PRODUCTION_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
PRODUCTION_METADATA_PATH = os.path.join(MODELS_DIR, "best_model_metadata.json")

os.makedirs(RETRAIN_RAW_DIR, exist_ok=True)
os.makedirs(MODEL_VERSIONS_DIR, exist_ok=True)

# Threshold reasoning: current training set is 5,491 rows. 50 new examples
# is ~0.9% growth -- small enough that a retrain won't be forced on some
# arbitrary calendar schedule regardless of whether anything meaningful
# changed, large enough to be a real sample (this project's own prior
# experiment, the TOI FA expansion, tested a comparable-order change of
# +98 real stars and found that WAS enough data to draw an honest
# before/after conclusion from, even though the conclusion was negative).
RETRAIN_THRESHOLD = 50
N_BOOTSTRAP = 2000
RANDOM_SEED = 42


def _read_csv_url(url):
    return pd.read_csv(url)


def _training_tic_ids(confirmed):
    """Every star already in training.csv, as a set of TIC ids.

    BUG FIXED (2026-08-02): this used to compare the archive's `TIC_<id>` name
    against training.csv's `host` column directly, on the stated assumption
    that host "is always 'TIC_<id>'". That is true of the negative class and of
    rows this pipeline itself added, but NOT of the original positive class,
    which `01_download_known.py` names by HOSTNAME (`11_Com`, `Kepler-142`).

    So every confirmed planet already in the training set under its hostname
    looked brand new, got re-queued as `TIC_<id>`, and was re-downloaded and
    appended as a SECOND row for the same star. Measured from only 137
    processed watch labels: 144 stars duplicated, and 56 of them landed on
    opposite sides of the frozen split -- the model training on a star and
    then being scored on it. 4,243 labels were still queued behind that.

    The resolution uses the `confirmed` frame this function already downloads,
    which carries hostname alongside tic_id, so the authoritative
    hostname -> TIC mapping is free. Names are matched both raw and with
    spaces replaced by underscores, since that is the transform
    01_download_known.py applies when it builds file/host names.

    Stars that resolve to no TIC at all (microlensing events, KMT-*-BLG-*)
    cannot collide with a TESS target and are simply absent from this set.
    """
    if not os.path.exists(TRAINING_CSV):
        return set()
    hosts = pd.read_csv(TRAINING_CSV)["host"].astype(str)

    known = set(
        pd.to_numeric(hosts.str.extract(r"^TIC_(\d+)", expand=False),
                      errors="coerce").dropna().astype("int64")
    )

    name_to_tic = {}
    for h, t in zip(confirmed["hostname"].astype(str),
                    confirmed["_tic"]):
        if pd.isna(t):
            continue
        t = int(t)
        name_to_tic.setdefault(h, t)
        name_to_tic.setdefault(h.replace(" ", "_"), t)
    for h in hosts:
        t = name_to_tic.get(h)
        if t is not None:
            known.add(t)
    return known


def find_new_labeled_examples():
    """Live re-query, same URL pattern already validated in
    08_characterize_candidates.py's fetch_fresh_exclusion_data. Returns a
    list of {host, label, source} dicts for stars not already in training.csv.

    Membership is decided by TIC ID (see _training_tic_ids), NOT by host
    string -- the positive class is hostname-named, so a string comparison
    silently re-queues stars that are already present.
    """
    already_watched = db.get_known_watch_hosts()

    confirmed_url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
                      "query=select+hostname,tic_id+from+pscomppars&format=csv")
    confirmed = _read_csv_url(confirmed_url)
    confirmed["_tic"] = pd.to_numeric(
        confirmed["tic_id"].astype(str).str.replace("TIC ", "", regex=False),
        errors="coerce")
    confirmed_tics = set(confirmed["_tic"].dropna().astype("int64"))

    toi_url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
               "query=select+tid,tfopwg_disp+from+toi&format=csv")
    toi = _read_csv_url(toi_url)
    fp_tics = set(toi[toi["tfopwg_disp"] == "FP"]["tid"].dropna().astype("int64"))

    existing_tics = _training_tic_ids(confirmed)

    new_rows = []
    skipped_already_present = 0
    for tic_id in confirmed_tics:
        host = f"TIC_{tic_id}"
        if tic_id in existing_tics:
            skipped_already_present += 1
            continue
        if host not in already_watched:
            new_rows.append({"host": host, "label": 1, "source": "pscomppars_confirmed"})
    for tic_id in fp_tics:
        host = f"TIC_{tic_id}"
        if tic_id in existing_tics:
            skipped_already_present += 1
            continue
        if host not in already_watched:
            new_rows.append({"host": host, "label": 0, "source": "toi_false_positive"})

    n_added = db.add_watched_labels(new_rows)
    print(f"find_new_labeled_examples: {len(confirmed_tics)} confirmed, {len(fp_tics)} FP live "
          f"(archive); {len(existing_tics)} stars already in training.csv by TIC, "
          f"{skipped_already_present} archive entries skipped as already present; "
          f"{len(new_rows)} genuinely new, {n_added} newly queued.")
    return new_rows


def _fetch_stellar_params_for_host(tic_id):
    """Real per-star st_rad/st_teff/st_mass from the TIC catalog -- same
    astroquery.mast.Catalogs call 06_download_unknown.py's
    fetch_stellar_params already uses, just for a single star at a time
    here since this runs one new label at a time, not in bulk."""
    from astroquery.mast import Catalogs
    r = Catalogs.query_criteria(catalog="Tic", ID=[tic_id])
    if len(r) == 0:
        return None, None, None
    row = r[0]
    return (float(row["rad"]) if row["rad"] is not None else None,
            float(row["Teff"]) if row["Teff"] is not None else None,
            float(row["mass"]) if row["mass"] is not None else None)


def process_and_append_new_examples(max_new=None):
    """Runs every pending watch-queue entry through download -> preprocess
    -> TLS feature extraction, reusing 06_download_unknown.py's
    try_search/_safe_download and 02_preprocess.py's process_one_file
    UNCHANGED, then appends a matching row to training.csv. Resumable:
    each star's status is tracked in label_watch_queue, so a crash mid-run
    just leaves some stars 'pending' for the next tick to pick up."""
    pending = db.get_pending_watch_labels()
    if max_new:
        pending = pending[:max_new]
    if not pending:
        return 0

    print(f"process_and_append_new_examples: {len(pending)} pending stars to process...")
    appended = 0
    for item in pending:
        host, label = item["host"], item["label"]
        tic_id = int(host.replace("TIC_", ""))
        try:
            search, method = m06.try_search(tic_id)
            if len(search) == 0:
                db.mark_watch_label_failed(host, "No TESS data found via MAST")
                continue
            lc = m06._safe_download(search[0])
            if lc is None:
                db.mark_watch_label_failed(host, "Download failed")
                continue
            raw_path = os.path.join(RETRAIN_RAW_DIR, host + ".csv")
            lc.to_pandas().reset_index().to_csv(raw_path, index=False)

            result = m02.process_one_file(raw_path)
            if result["status"] != "Success":
                db.mark_watch_label_failed(host, f"Preprocess: {result['status']}")
                continue
            # process_one_file writes directly to its own OUTPUT_FOLDER
            # (data/processed/, the SAME canonical folder the rest of this
            # project's positive-class pipeline uses) -- not a
            # retrain-pipeline-specific copy, so newly-appended stars'
            # processed light curves are genuinely part of the same real
            # dataset going forward, not a parallel shadow copy.
            processed_path = os.path.join(m02.OUTPUT_FOLDER, host + ".csv")
            if not os.path.exists(processed_path):
                db.mark_watch_label_failed(host, "Preprocessed output not found where expected")
                continue

            st_rad, st_teff, st_mass = _fetch_stellar_params_for_host(tic_id)
            feats, status = m06.compute_all_features(
                processed_path, host, st_rad, st_mass, m05.FEATURE_COLUMNS
            )
            if feats is None:
                db.mark_watch_label_failed(host, f"TLS/feature extraction: {status}")
                continue

            row = dict(feats)
            row.update({"host": host, "label": label, "st_rad": st_rad, "st_teff": st_teff,
                        "st_mass": st_mass})
            # BUG FIXED (caught live): appending with header=False writes
            # values POSITIONALLY -- a naive pd.DataFrame([row]) has whatever
            # key order dict(feats) happened to produce, which does NOT
            # match training.csv's real column order, so every value landed
            # in the wrong column (host became a period value, label became
            # an SDE value, etc.) until this reindex was added. Any column
            # this row doesn't populate (ra/dec, pl_names, QC-stage counts
            # that only exist for the original bulk-downloaded stars) is
            # correctly left NaN, not misaligned.
            existing_columns = pd.read_csv(TRAINING_CSV, nrows=0).columns if os.path.exists(TRAINING_CSV) else None
            df_row = pd.DataFrame([row])
            if existing_columns is not None:
                df_row = df_row.reindex(columns=existing_columns)
                df_row.to_csv(TRAINING_CSV, mode="a", header=False, index=False)
            else:
                df_row.to_csv(TRAINING_CSV, index=False)

            db.mark_watch_label_processed(host, len(pd.read_csv(processed_path)))
            appended += 1
        except Exception as e:
            db.mark_watch_label_failed(host, f"Unexpected error: {e}")

    print(f"process_and_append_new_examples: {appended}/{len(pending)} appended to training.csv.")
    return appended


def _paired_bootstrap_auc_diff(y_test, proba_a, proba_b, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    # `fast_auc` is the same statistic as sklearn's roc_auc_score computed by
    # rank-sum (verified to 1e-12, ties averaged). sklearn's version costs
    # ~25 ms/call at this test-set size, nearly all input validation, which
    # would put these 2000 iterations x 2 calls at ~90 s inside the retrain
    # path. See code/fast_auc.py and ENVIRONMENT_NOTES section 9.
    from fast_auc import fast_auc
    rng = np.random.RandomState(seed)
    y_arr = np.asarray(y_test)
    n = len(y_arr)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        y_b = y_arr[idx]
        n_pos = int((y_b == 1).sum())
        if n_pos == 0 or n_pos == n:
            continue
        diffs.append(fast_auc(y_b, proba_b[idx]) - fast_auc(y_b, proba_a[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def _build_challenger(prod_model):
    """Builds the unfitted challenger for a retrain attempt.

    The challenger must be production's OWN recipe, so that a promotion means
    "the same model, trained on more data, is genuinely better" rather than
    "one configuration beat a different configuration". `clone` copies the
    estimator's class and hyperparameters without any fitted state, so this
    tracks production automatically if its configuration ever changes.

    Returns (unfitted_estimator, human_readable_source).

    FIRST-EVER-RETRAIN CASE: with no production model on disk there is nothing
    to clone, so this falls back to m05's HistGradientBoosting config -- the
    same model family the project has always deployed. That fallback is close
    to inert in practice: the caller refuses to promote at all when there is no
    production baseline to compare against ("No existing production model to
    compare against -- not auto-promoting"), so the fallback config only
    determines what gets LOGGED for that first attempt, never what ships. It
    matters more now than before the fix only in the sense that it is now the
    single remaining path that can produce a config mismatch, and it can only
    do so when a mismatch is impossible to define.
    """
    if prod_model is None:
        return (m05.build_models()["HistGradientBoosting"],
                "m05.build_models()['HistGradientBoosting'] (no production model to clone)")
    return clone(prod_model), f"clone of production {type(prod_model).__name__}"


def maybe_trigger_retrain(threshold=RETRAIN_THRESHOLD, dry_run=False):
    """Checks how many NEW rows have been appended since the last retrain
    attempt (any attempt, promoted or not -- re-testing after every batch
    of new data is the point, not just after a promoted one). If over
    threshold, retrains + validates + promotes-or-not, and ALWAYS logs the
    attempt regardless of outcome.

    dry_run=True runs the exact same retrain/compare/decide logic but never
    writes to PRODUCTION_MODEL_PATH/PRODUCTION_METADATA_PATH or the DB,
    regardless of what the promotion decision would have been -- for
    verifying this pipeline works without any risk of a noise-driven
    promotion on too little data (e.g. testing with a near-zero threshold)."""
    from sklearn.metrics import roc_auc_score

    last_attempts = db.list_retrain_attempts(limit=1)
    since = last_attempts[0]["triggered_at"] if last_attempts else "2000-01-01 00:00:00 UTC"
    n_new = db.count_processed_watch_labels_since(since)
    if n_new < threshold:
        print(f"maybe_trigger_retrain: {n_new}/{threshold} new examples since last attempt -- not yet.")
        return None

    print(f"maybe_trigger_retrain: {n_new} new examples since last attempt -- triggering retrain.")
    df = pd.read_csv(TRAINING_CSV)

    # CNN-reevaluation growth check (independent of promotion): compares the
    # CURRENT full dataset's class counts against the baseline recorded at
    # the last real CNN-vs-classical architecture comparison. This never
    # triggers building a CNN -- it only flags the attempt so a human knows
    # it may be worth manually re-running that comparison, since the
    # original "CNN underperforms with real_only data" finding was itself
    # conditioned on the dataset size at that time.
    baseline = db.get_architecture_baseline()
    n_pos_now = int((df["label"] == 1).sum())
    n_neg_now = int((df["label"] == 0).sum())
    growth_pos = ((n_pos_now - baseline["n_positive"]) / baseline["n_positive"]
                  if baseline["n_positive"] else 0.0)
    growth_neg = ((n_neg_now - baseline["n_negative"]) / baseline["n_negative"]
                  if baseline["n_negative"] else 0.0)
    cnn_flag = growth_pos >= 0.20 or growth_neg >= 0.20

    # BUG FIXED: this used m05's old positional train_test_split, which
    # reshuffles whenever training.csv grows -- and this pipeline is the very
    # thing that makes it grow. The consequence was specific and serious:
    # three lines below, the CURRENT PRODUCTION MODEL is scored on X_test to
    # get the baseline the challenger must beat. Once the split had drifted,
    # that test set contained stars the production model was TRAINED on (89
    # of 1,102, measured), inflating its baseline AUC by ~+0.008 and making
    # the promotion gate systematically harder for challengers to clear --
    # a silent bias against ever promoting anything, in the one comparison
    # that decides what model users actually get.
    #
    # split_by_host keys membership to stable star IDs from the frozen
    # manifest, so the production model is only ever measured on stars it was
    # genuinely held out from, and every retrain attempt is scored on the
    # same test stars as every other.
    X, y = m05.build_feature_matrix(df)
    train_mask, test_mask = m05.split_by_host(df)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # BUG FIXED (2026-08-02): the challenger used to be built from
    # m05.build_models()["HistGradientBoosting"] -- the UNTUNED default config
    # -- while the incumbent it had to beat is the TUNED, calibration-wrapped
    # production model. Measured side by side on the same frozen test set,
    # five hyperparameters differed (l2_regularization 0.0 vs 0.5,
    # learning_rate 0.05 vs 0.1, max_depth 4 vs None, max_iter 300 vs 500,
    # max_leaf_nodes 31 vs 63) plus the CalibratedClassifierCV(cv=5) wrapper,
    # which is not merely monotonic -- it averages five sub-models, so it acts
    # as an ensemble and moves AUC in its own right.
    #
    # The effect was the same KIND of silent bias as the split-drift bug
    # documented above: every retrain attempt started in a hole that had
    # nothing to do with whether the new data helped. A challenger had to
    # out-run a hyperparameter handicap before it could even begin to
    # demonstrate a data-driven improvement.
    #
    # The fix is structural rather than a copied literal. sklearn.base.clone
    # reproduces the production estimator's class and every parameter, leaving
    # it unfitted -- so the challenger IS production's recipe by construction,
    # retrained on more data. If production's configuration ever changes (a
    # future tuning promotion, a different wrapper, even a different model
    # class), the challenger follows automatically and this bug cannot silently
    # come back. The joblib artifact is the single source of truth here;
    # best_model_metadata.json does NOT record hyperparameters, so it cannot
    # serve that role.
    production_auc = None
    production_proba = None
    prod_model = None
    if os.path.exists(PRODUCTION_MODEL_PATH):
        import joblib
        prod_model = joblib.load(PRODUCTION_MODEL_PATH)
        production_proba = prod_model.predict_proba(X_test)[:, 1]
        production_auc = roc_auc_score(y_test, production_proba)

    hgb_pipeline, challenger_source = _build_challenger(prod_model)
    hgb_pipeline.fit(X_train, y_train)
    new_proba = hgb_pipeline.predict_proba(X_test)[:, 1]
    new_auc = roc_auc_score(y_test, new_proba)

    if production_proba is not None:
        mean_diff, ci_lo, ci_hi = _paired_bootstrap_auc_diff(y_test, production_proba, new_proba)
        promoted = ci_lo > 0
        reasoning = (
            f"Challenger built as {challenger_source}. "
            f"New model test ROC-AUC {new_auc:.4f} vs production {production_auc:.4f}. "
            f"Paired bootstrap on (new - production): mean {mean_diff:+.4f}, "
            f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]. "
            + ("Promoted: CI entirely above zero." if promoted
               else "NOT promoted: CI includes zero or is negative -- doesn't clear the "
                    "same statistical-significance bar every other experiment in this "
                    "project has had to clear.")
        )
    else:
        ci_lo, ci_hi = None, None
        promoted = False
        reasoning = "No existing production model to compare against -- not auto-promoting " \
                    "without a baseline comparison."

    if cnn_flag:
        reasoning += (
            f" [FLAG: real training data has grown {growth_pos:+.0%} positive / "
            f"{growth_neg:+.0%} negative since the last CNN-vs-classical architecture "
            f"comparison ({baseline['compared_at']}) -- worth manually re-evaluating CNN "
            f"viability now that more real data exists. Not auto-building a CNN.]"
        )

    if dry_run:
        reasoning = "[DRY RUN -- nothing written] " + reasoning
        print(f"maybe_trigger_retrain (dry run): {reasoning}")
        return {"attempt_id": None, "promoted": promoted, "reasoning": reasoning, "dry_run": True,
                "cnn_reevaluation_flag": cnn_flag}

    version_id = None
    model_path = None
    if promoted:
        import joblib
        timestamp = db.now_iso().replace(":", "").replace(" ", "_").replace("-", "")
        model_path = os.path.join(MODEL_VERSIONS_DIR, f"model_{timestamp}.joblib")
        joblib.dump(hgb_pipeline, model_path)
        shutil.copy(model_path, PRODUCTION_MODEL_PATH)
        new_metadata = {
            "model_name": "HistGradientBoosting_continuous_retrain",
            "feature_columns": m05.FEATURE_COLUMNS, "test_roc_auc": float(new_auc),
            "random_seed": m05.RANDOM_SEED, "training_rows": len(X_train), "test_rows": len(X_test),
        }
        with open(PRODUCTION_METADATA_PATH, "w") as f:
            json.dump(new_metadata, f, indent=2)

    version_id = db.add_model_version(
        model_path or "(not promoted -- no file saved)", len(X_train),
        int((y_train == 1).sum()), int((y_train == 0).sum()), float(new_auc),
        (ci_lo or 0.0, ci_hi or 0.0), retrain_attempt_id=None, is_live=promoted,
    )
    attempt_id = db.log_retrain_attempt(
        trigger_reason=f"{n_new} new labeled examples processed since last attempt",
        n_new_examples=n_new, n_training_rows=len(df), test_roc_auc=float(new_auc),
        production_roc_auc=production_auc, bootstrap_ci=(ci_lo, ci_hi), promoted=promoted,
        model_version_id=version_id, reasoning=reasoning, cnn_reevaluation_flag=cnn_flag,
        growth_pct_positive=growth_pos, growth_pct_negative=growth_neg,
    )
    print(f"maybe_trigger_retrain: attempt #{attempt_id} logged. {reasoning}")
    return {"attempt_id": attempt_id, "promoted": promoted, "reasoning": reasoning,
            "cnn_reevaluation_flag": cnn_flag}


# BUG AVOIDED (found live, not guessed): the very first live run of
# find_new_labeled_examples() queued 4,485 "new" stars -- NOT because that
# many were genuinely just published, but because most confirmed planets in
# the archive were discovered by other missions (RV, ground-based, Kepler)
# and never had TESS data at all -- the same thing the ORIGINAL one-time
# 01_download_known.py bulk run already had to filter through (most of its
# candidate list also failed "No TESS data" via the same try_search call).
# Processing all 4,485 in one scheduler tick would mean hours of MAST
# download attempts triggered by a single background poll -- capped per-tick
# below so this stays a genuinely incremental, low-footprint background
# process, not an accidental re-run of the original bulk gather.
PER_TICK_MAX_NEW = 25


def scheduler_tick():
    """Called from job_runner.py's existing scheduler loop -- one full
    cycle: find new labels, process a bounded batch of what's pending, maybe
    retrain. Deliberately NOT run on the same 60s cadence as the Update-job
    due-check -- see job_runner.py's _scheduler_loop for the coarser
    interval this is actually called on."""
    try:
        find_new_labeled_examples()
        process_and_append_new_examples(max_new=PER_TICK_MAX_NEW)
        maybe_trigger_retrain()
    except Exception as e:
        print(f"retrain_pipeline.scheduler_tick error (does not affect Update jobs): {e}")
