"""deploy_optuna_model.py -- STEP 3: the live swap.

Hyperparameter-only change; the feature set is UNCHANGED at 33 columns, so
unlike the crowding/variability/Gaia deployments this one cannot break the
promotion gate on a column-count mismatch. It is still done as a manual offline
swap, because the gate is structurally incapable of performing it: the gate's
challenger is `clone(prod_model)`, i.e. production's OWN configuration refit on
more data, so it can never propose a DIFFERENT configuration. A hyperparameter
change has to come from outside the gate by construction.

Order of operations, so a failure at any point leaves production intact:
  1. verify the staged artifact loads, predicts, and carries the exact
     hyperparameters from the saved study
  2. verify the CURRENT production artifact matches its recorded md5
  3. copy the current model + metadata to models/versions/ as rollback
  4. verify the rollback copies are byte-identical to their sources
  5. atomically swap model and metadata together (os.replace)
  6. re-verify the live artifact's md5 and hyperparameters after the swap

Nothing is deleted. Run with --dry-run to rehearse every check without writing.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
MODELS = os.path.join(ROOT, "models")
VERSIONS = os.path.join(MODELS, "versions")
PROD_MODEL = os.path.join(MODELS, "best_model.joblib")
PROD_META = os.path.join(MODELS, "best_model_metadata.json")
STAGED_MODEL = os.path.join(MODELS, "staged_best_model_optuna33.joblib")
STAGED_META = os.path.join(MODELS, "staged_best_model_optuna33_metadata.json")
STUDY = os.path.join(HERE, "optuna_hpo_nested.json")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")

EXPECTED_PREV_MD5 = "c37f9f4bdb252d52b8c1c5487dad9e6d"


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def hp_of(model):
    est = getattr(model, "estimator", None) or getattr(model, "base_estimator", None)
    return est.named_steps["clf"].get_params()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tag = "[DRY RUN] " if args.dry_run else ""

    # ---- 1. staged artifact sanity ----
    print("=== 1. STAGED ARTIFACT ===")
    for p in (STAGED_MODEL, STAGED_META):
        if not os.path.exists(p):
            sys.exit(f"FATAL: missing {p}")
    staged = joblib.load(STAGED_MODEL)
    smeta = json.load(open(STAGED_META))
    smd5 = md5(STAGED_MODEL)
    print(f"  md5           {smd5}")
    if smd5 != smeta["model_md5"]:
        sys.exit(f"FATAL: staged md5 {smd5} != metadata {smeta['model_md5']}")
    print("  md5 matches its own metadata                      OK")

    want = json.load(open(STUDY))["final_params"]
    got = hp_of(staged)
    for k, v in want.items():
        if got.get(k) != v:
            sys.exit(f"FATAL: staged {k}={got.get(k)!r}, study says {v!r}")
    print("  hyperparameters match the saved study exactly     OK")
    for k in sorted(want):
        print(f"    {k:<20} {want[k]}")

    # ---- live prediction smoke test on real rows ----
    sys.path.insert(0, os.path.join(ROOT, "code"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols = list(m05.FEATURE_COLUMNS)
    if list(smeta["feature_columns"]) != cols:
        sys.exit("FATAL: staged feature_columns != m05.FEATURE_COLUMNS")
    print(f"  feature_columns == m05.FEATURE_COLUMNS ({len(cols)})       OK")
    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    te = m05.frozen_test_mask(df)
    p = staged.predict_proba(X[te])[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(np.asarray(y)[te], p)
    print(f"  live frozen-test AUC {auc:.4f} vs metadata "
          f"{smeta['test_roc_auc']:.4f}")
    if abs(auc - smeta["test_roc_auc"]) > 1e-9:
        sys.exit("FATAL: staged model does not reproduce its recorded AUC")
    print("  reproduces its recorded AUC                       OK")

    # ---- 2. current production integrity ----
    print("\n=== 2. CURRENT PRODUCTION ===")
    cur_md5 = md5(PROD_MODEL)
    cmeta = json.load(open(PROD_META))
    print(f"  md5           {cur_md5}")
    if cur_md5 != EXPECTED_PREV_MD5:
        sys.exit(f"FATAL: production md5 changed under us (expected {EXPECTED_PREV_MD5})")
    if cur_md5 != cmeta.get("model_md5"):
        sys.exit("FATAL: production md5 != its own metadata")
    print("  matches expected md5 and its own metadata         OK")
    cur_hp = hp_of(joblib.load(PROD_MODEL))
    print(f"  outgoing hyperparameters: lr={cur_hp['learning_rate']} "
          f"iter={cur_hp['max_iter']} leaves={cur_hp['max_leaf_nodes']} "
          f"l2={cur_hp['l2_regularization']} cw={cur_hp['class_weight']}")

    # ---- 3-4. rollback copies ----
    print("\n=== 3. ROLLBACK PRESERVATION ===")
    os.makedirs(VERSIONS, exist_ok=True)
    rb_model = os.path.join(VERSIONS, f"best_model_pre_optuna_{cur_md5[:8]}.joblib")
    rb_meta = os.path.join(VERSIONS, f"best_model_pre_optuna_{cur_md5[:8]}_metadata.json")
    print(f"  {tag}{os.path.relpath(rb_model, ROOT)}")
    print(f"  {tag}{os.path.relpath(rb_meta, ROOT)}")
    if not args.dry_run:
        shutil.copy2(PROD_MODEL, rb_model)
        shutil.copy2(PROD_META, rb_meta)
        if md5(rb_model) != cur_md5:
            sys.exit("FATAL: rollback copy is not byte-identical -- ABORTING BEFORE SWAP")
        if json.load(open(rb_meta))["model_md5"] != cur_md5:
            sys.exit("FATAL: rollback metadata mismatch -- ABORTING BEFORE SWAP")
        print("  rollback copies verified byte-identical           OK")

    # ---- 5. atomic swap ----
    print("\n=== 4. SWAP ===")
    if args.dry_run:
        print("  [DRY RUN] no files written")
        print(f"\n{cur_md5} -> {smd5} (would be)")
        return
    for src, dst in ((STAGED_MODEL, PROD_MODEL), (STAGED_META, PROD_META)):
        fd, tmp = tempfile.mkstemp(dir=MODELS, suffix=".swap")
        os.close(fd)
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    print("  model + metadata replaced atomically              OK")

    # ---- 6. post-swap verification ----
    print("\n=== 5. POST-SWAP VERIFICATION ===")
    new_md5 = md5(PROD_MODEL)
    nmeta = json.load(open(PROD_META))
    print(f"  live md5      {new_md5}")
    assert new_md5 == smd5, "live md5 != staged md5"
    assert new_md5 == nmeta["model_md5"], "live md5 != live metadata"
    assert nmeta["previous_model_md5"] == cur_md5, "previous_model_md5 wrong"
    live_hp = hp_of(joblib.load(PROD_MODEL))
    for k, v in want.items():
        assert live_hp.get(k) == v, f"live {k}={live_hp.get(k)!r} != {v!r}"
    print("  live md5 == staged == metadata                    OK")
    print("  live hyperparameters == saved study                OK")
    print(f"  rollback available: {os.path.relpath(rb_model, ROOT)}")
    print(f"\nDEPLOYED  {cur_md5} -> {new_md5}")
    print(f"AUC       {cmeta['test_roc_auc']:.4f} -> {nmeta['test_roc_auc']:.4f}")
    print(f"RULE      {nmeta['promotion_rule_status']}")


if __name__ == "__main__":
    main()
