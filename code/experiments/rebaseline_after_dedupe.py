"""rebaseline_after_dedupe.py -- what the model actually scores once the
leaking rows are gone.

56 stars appeared on BOTH sides of the frozen split under two different names
(see dedupe_training_by_tic.py). Every AUC this project has quoted recently was
therefore measured on a test set containing stars the model had trained on. The
contamination is small (~5% of test rows) but it is one-directional: leakage
inflates. This re-measures on the cleaned data so the project has an honest
number to compare future work against.

FOUR MEASUREMENTS, all on the production configuration:

  A. production artifact on the CONTAMINATED test set   (the number on record)
  B. production artifact on the CLEAN test set          (same model, honest set)
  C. production artifact on the LEAKED ROWS ONLY        (isolates the inflation)
  D. refit on clean train, scored on clean test         (the go-forward baseline)

A vs B is the headline correction: the same deployed model, same weights,
scored two ways. C shows directly how the model behaves on stars it memorised.

The production model is NEVER modified. This only measures.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, brier_score_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, SCRIPT_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CLEAN_CSV = os.path.join(ROOT, "data", "training_dataset", "training_deduped.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "rebaseline_after_dedupe.json")

RANDOM_SEED = 42
N_BOOT = 2000


def load_m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def boot_ci(y, p, n=N_BOOT, seed=RANDOM_SEED):
    """CI on a single model's AUC (not a paired difference) -- the two test
    sets here differ in membership, so a paired bootstrap is not defined."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    a = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        a.append(roc_auc_score(y[i], p[i]))
    a = np.array(a)
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    res = {}
    m05 = load_m05()
    import dedupe_training_by_tic as D
    resolve = D.build_resolver()

    raw = pd.read_csv(TRAINING_CSV)
    clean = pd.read_csv(CLEAN_CSV)
    prod = joblib.load(PROD)

    Xr, yr = m05.build_feature_matrix(raw)
    trr, ter = m05.split_by_host(raw)
    Xr = Xr.reset_index(drop=True); yr = np.asarray(yr); ter = np.asarray(ter)

    Xc, yc = m05.build_feature_matrix(clean)
    trc, tec = m05.split_by_host(clean)
    Xc = Xc.reset_index(drop=True); yc = np.asarray(yc)
    trc = np.asarray(trc); tec = np.asarray(tec)

    print("=" * 78)
    print("RE-BASELINE AFTER DEDUPE")
    print("=" * 78)
    print(f"contaminated: {trr.sum()} train / {ter.sum()} test")
    print(f"clean:        {trc.sum()} train / {tec.sum()} test")
    res["contaminated_split"] = {"train": int(trr.sum()), "test": int(ter.sum())}
    res["clean_split"] = {"train": int(trc.sum()), "test": int(tec.sum())}

    # ---- A: production on the contaminated test set ----
    pa = prod.predict_proba(Xr[ter])[:, 1]
    auc_a = roc_auc_score(yr[ter], pa)
    ma, la, ha = boot_ci(yr[ter], pa)
    print(f"\n  A. production, CONTAMINATED test  AUC {auc_a:.4f}  "
          f"CI [{la:.4f}, {ha:.4f}]   <- the number on record")

    # ---- B: production on the clean test set ----
    pb = prod.predict_proba(Xc[tec])[:, 1]
    auc_b = roc_auc_score(yc[tec], pb)
    mb, lb, hb = boot_ci(yc[tec], pb)
    print(f"  B. production, CLEAN test         AUC {auc_b:.4f}  "
          f"CI [{lb:.4f}, {hb:.4f}]   <- honest")
    print(f"     correction: {auc_b - auc_a:+.4f}")

    # ---- C: production on the leaked rows only ----
    raw_side = np.where(ter, "test", "train")
    rw = raw.copy(); rw["tic"] = rw["host"].map(resolve); rw["side"] = raw_side
    kept = set(clean["host"].astype(str))
    leaked_mask = ter & (~raw["host"].astype(str).isin(kept)).to_numpy()
    n_leak = int(leaked_mask.sum())
    print(f"\n  C. leaked test rows (dropped by dedupe): {n_leak}")
    if n_leak >= 20 and len(np.unique(yr[leaked_mask])) > 1:
        pc = prod.predict_proba(Xr[leaked_mask])[:, 1]
        auc_c = roc_auc_score(yr[leaked_mask], pc)
        print(f"     production AUC on JUST those rows  {auc_c:.4f}")
        res["leaked_rows_auc"] = float(auc_c)
    elif n_leak:
        pc = prod.predict_proba(Xr[leaked_mask])[:, 1]
        print(f"     single-class or too few for AUC; mean predicted prob "
              f"{pc.mean():.4f} vs true positive rate {yr[leaked_mask].mean():.4f}")
        res["leaked_rows_auc"] = None
    res["n_leaked_test_rows"] = n_leak

    # ---- D: refit on clean train ----
    ch = clone(prod)
    ch.fit(Xc[trc], yc[trc])
    pd_ = ch.predict_proba(Xc[tec])[:, 1]
    auc_d = roc_auc_score(yc[tec], pd_)
    md, ld, hd = boot_ci(yc[tec], pd_)
    print(f"\n  D. refit on clean train, clean test  AUC {auc_d:.4f}  "
          f"CI [{ld:.4f}, {hd:.4f}]   <- go-forward baseline")
    print(f"     Brier {brier_score_loss(yc[tec], pd_):.4f}")

    res["A_production_contaminated"] = {"auc": float(auc_a), "ci": [la, ha]}
    res["B_production_clean"] = {"auc": float(auc_b), "ci": [lb, hb]}
    res["D_refit_clean"] = {"auc": float(auc_d), "ci": [ld, hd],
                            "brier": float(brier_score_loss(yc[tec], pd_))}
    res["contamination_correction"] = float(auc_b - auc_a)

    print("\n" + "=" * 78)
    if auc_b < auc_a:
        print(f"Leakage was INFLATING the reported AUC by {auc_a - auc_b:.4f}.")
    else:
        print(f"Removing the leaked rows did NOT lower the score "
              f"({auc_b - auc_a:+.4f}) -- the contamination was too small, or the "
              f"leaked stars were not ones the model got right.")
    print("Production model unchanged; this script only measures.")
    print("=" * 78)

    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
