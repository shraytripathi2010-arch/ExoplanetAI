"""merge_and_validate_k2.py -- Part 3: merge the K2 pilot and measure honestly.

WHAT THIS DOES NOT DO
It never writes data/training_dataset/training.csv, never regenerates the
frozen split, never touches models/ or the promotion gate. The merge happens
in memory; the on-disk corpus is unchanged.

THE SOURCE COLUMN IS THE POINT OF THIS SCRIPT
Cadence was an invisible confound in this project for months -- the FFI work
found a "+0.0113 gain" that was entirely an artefact of coarse rows sitting in
the test set, and it was only visible once rows could be grouped by cadence
after the fact. Every row here therefore carries an explicit `source`:

    tess_2min   1.0-2.6 min   the population the model is actually deployed on
    tess_fine   < 1.0 min     20-second SPOC; FINER than 2-min, not coarser
    tess_coarse > 2.6 min     FFI
    k2          ~29.4 min     this pilot

THE EVALUATION RULE THAT PREVENTS THE FFI ARTEFACT RECURRING
Every arm is scored on BOTH the full test set and the tess_2min-only subset,
and both are printed side by side. A gain that appears on the full set and
vanishes on tess_2min is not a gain -- it is the model getting credit for
classifying rows of a kind it will never be asked about in production.

sample_weight goes to the BARE pipeline via clf__sample_weight. Passing it to
CalibratedClassifierCV reaches only the calibrator, which sklearn warns about
and which silently made an earlier arm meaningless.
"""
import os
import sys
import json
import hashlib
import importlib.util
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(PILOT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
EXP_DIR = os.path.join(CODE_DIR, "experiments")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, EXP_DIR)
from fast_auc import roc_auc_score  # exact drop-in, ~23x faster inside bootstraps

FEATURES = os.path.join(PILOT_DIR, "k2_pilot_features.csv")
SAMPLE = os.path.join(PILOT_DIR, "k2_pilot_sample.csv")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(EXP_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
MANIFEST = os.path.join(ROOT, "data", "training_dataset", "split_manifest.json")
RESULTS = os.path.join(PILOT_DIR, "k2_merge_results.json")

N_BOOT = 2000
SEED = 42
BASELINE_REF = 0.9021        # refit-clean -> clean; what new work competes with


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def paired_boot(y, pa, pb, n=N_BOOT, seed=SEED):
    """Paired bootstrap on the ROC-AUC delta. Same resampled indices for both
    models, so the pairing removes test-set variance rather than adding it."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
    d = np.asarray(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def tag_sources(tess):
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(tess.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce").to_numpy()
    s = np.full(len(tess), "tess_2min", dtype=object)
    s[c < 1.0] = "tess_fine"
    s[c > 2.6] = "tess_coarse"
    s[~np.isfinite(c)] = "tess_2min"        # unmeasured: default to the majority
    return s


def side_of(host, manifest):
    if host in set(manifest["test_hosts"]):
        return "test"
    if host in set(manifest["train_hosts"]):
        return "train"
    h = int(hashlib.md5(str(host).encode()).hexdigest(), 16)
    return "test" if (h % 100) < 20 else "train"


def main():
    m05 = _m05()
    with open(MANIFEST) as f:
        manifest = json.load(f)

    tess = pd.read_csv(TRAINING)
    tess["source"] = tag_sources(tess)

    k2 = pd.read_csv(FEATURES)
    k2 = k2[k2["status"] == "Success"].reset_index(drop=True)
    # st_teff is a model feature the TLS worker never carried (it only needs
    # r_star/m_star for the period grid); take the archive's values for all
    # three stellar params from the sample file.
    smp = pd.read_csv(SAMPLE)[["host", "tic_id", "disposition",
                               "st_rad", "st_mass", "st_teff"]]
    k2 = k2.drop(columns=[c for c in ("st_rad", "st_mass", "st_teff")
                          if c in k2.columns]).merge(smp, on="host", how="left")
    k2["source"] = "k2"

    print("=" * 84)
    print("MERGE -- identity and split integrity, checked by TIC")
    print("=" * 84)

    # --- duplicate check by TIC, the stable identifier -----------------------
    import re
    tess_tic = pd.to_numeric(
        tess["host"].astype(str).str.extract(r"^TIC_(\d+)", expand=False),
        errors="coerce")
    # hostname-named TESS rows were resolved during selection; reuse that set
    sel_tics = set(pd.read_csv(SAMPLE)["tic_id"].dropna().astype("int64"))
    known = set(tess_tic.dropna().astype("int64"))
    collide = sel_tics & known
    print(f"  K2 pilot stars whose TIC already appears in training.csv: "
          f"{len(collide)}  {'OK' if not collide else 'DUPLICATE RISK'}")

    merged = pd.concat([tess, k2], ignore_index=True)
    merged["_side"] = [side_of(h, manifest) for h in merged["host"]]
    dup = merged["host"].duplicated().sum()
    straddle = (merged.groupby("host")["_side"].nunique() > 1).sum()
    print(f"  duplicated host rows after merge : {dup}")
    print(f"  hosts straddling the split       : {straddle}")
    print(f"\n  source composition:")
    for s, n in merged["source"].value_counts().items():
        sub = merged[merged["source"] == s]
        print(f"    {s:<12} {n:>5}   train {int((sub._side=='train').sum()):>4} "
              f"/ test {int((sub._side=='test').sum()):>4}")
    k2_side = merged[merged["source"] == "k2"]["_side"].value_counts().to_dict()
    print(f"\n  K2 rows landed: {k2_side} (assigned by the manifest's md5 hash rule)")

    # --- leakage suite ------------------------------------------------------
    X, y = m05.build_feature_matrix(merged)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    src = merged["source"].to_numpy()
    is_k2 = (src == "k2")

    print("\n" + "=" * 84)
    print("LEAKAGE SUITE")
    print("=" * 84)
    print(f"  {'feature':<26}{'NaN% K2':>9}{'NaN% TESS':>11}"
          f"{'label-AUC':>11}{'source-AUC':>12}")
    rows = []
    for c in X.columns:
        v = pd.to_numeric(X[c], errors="coerce")
        nan_k2 = float(v[is_k2].isna().mean() * 100)
        nan_te = float(v[~is_k2].isna().mean() * 100)
        m = v.notna().to_numpy()
        la = sa = np.nan
        if m.sum() > 50 and len(np.unique(y[m])) > 1:
            la = roc_auc_score(y[m], v[m])
        if m.sum() > 50 and len(np.unique(is_k2[m])) > 1:
            sa = roc_auc_score(is_k2[m], v[m])
        rows.append({"feature": c, "nan_pct_k2": nan_k2, "nan_pct_tess": nan_te,
                     "label_auc": None if np.isnan(la) else float(la),
                     "source_auc": None if np.isnan(sa) else float(sa)})
    rr = pd.DataFrame(rows).sort_values(
        "source_auc", key=lambda s: -(s - 0.5).abs())
    for _, r in rr.head(10).iterrows():
        print(f"  {r['feature']:<26}{r['nan_pct_k2']:>9.1f}{r['nan_pct_tess']:>11.1f}"
              f"{(r['label_auc'] or np.nan):>11.3f}{(r['source_auc'] or np.nan):>12.3f}")

    ok = rr.dropna(subset=["label_auc", "source_auc"])
    corr = float(np.corrcoef((ok["label_auc"] - 0.5).abs(),
                             (ok["source_auc"] - 0.5).abs())[0, 1])
    print(f"\n  correlation |label-AUC-0.5| vs |source-AUC-0.5| : {corr:+.3f}")
    print("  (cadence audit reported -0.247 for the analogous quantity;")
    print("   POSITIVE and large would mean the features that separate the")
    print("   sources are the same ones carrying the label -- entangled.)")

    # --- arms ---------------------------------------------------------------
    tr = (merged["_side"] == "train").to_numpy()
    te = (merged["_side"] == "test").to_numpy()
    te_tess2 = te & (src == "tess_2min")
    te_full_tess = te & (~is_k2)

    prod = joblib.load(PROD)
    bare = clone(getattr(prod, "estimator", prod))

    print("\n" + "=" * 84)
    print("ARMS -- evaluated on the TESS test set, never on a mixed one")
    print("=" * 84)
    print(f"  test rows: all-TESS {int(te_full_tess.sum())} | "
          f"tess_2min-only {int(te_tess2.sum())}")
    print(f"  train rows: baseline {int((tr & ~is_k2).sum())} | "
          f"pooled {int(tr.sum())}  (+{int((tr & is_k2).sum())} K2)\n")

    def fit_eval(mask_tr, w=None):
        mdl = clone(bare)
        if w is None:
            mdl.fit(X[mask_tr], y[mask_tr])
        else:
            mdl.fit(X[mask_tr], y[mask_tr], clf__sample_weight=w)
        return (mdl.predict_proba(X[te_full_tess])[:, 1],
                mdl.predict_proba(X[te_tess2])[:, 1])

    base_tr = tr & ~is_k2
    pA_full, pA_2 = fit_eval(base_tr)
    pooled = tr
    pB_full, pB_2 = fit_eval(pooled)
    wC = np.where(is_k2[pooled], 0.25, 1.0)
    pC_full, pC_2 = fit_eval(pooled, w=wC)

    yF, y2 = y[te_full_tess], y[te_tess2]
    out = {"arms": {}, "leakage": rows, "corr_label_vs_source": corr,
           "n_k2_success": int(len(k2)), "duplicates": int(dup),
           "straddling": int(straddle)}

    print(f"  {'arm':<28}{'all-TESS test':>15}{'delta [95% CI]':>26}"
          f"{'tess_2min test':>16}{'delta [95% CI]':>26}")
    base_auc_F = roc_auc_score(yF, pA_full)
    base_auc_2 = roc_auc_score(y2, pA_2)
    print(f"  {'A. baseline (TESS only)':<28}{base_auc_F:>15.4f}{'--':>26}"
          f"{base_auc_2:>16.4f}{'--':>26}")
    for name, pf, p2 in [("B. pooled (TESS + K2)", pB_full, pB_2),
                         ("C. K2 down-weighted 0.25", pC_full, pC_2)]:
        aF, a2 = roc_auc_score(yF, pf), roc_auc_score(y2, p2)
        dF, loF, hiF = paired_boot(yF, pA_full, pf)
        d2, lo2, hi2 = paired_boot(y2, pA_2, p2)
        print(f"  {name:<28}{aF:>15.4f}"
              f"{f'{dF:+.4f} [{loF:+.4f},{hiF:+.4f}]':>26}"
              f"{a2:>16.4f}{f'{d2:+.4f} [{lo2:+.4f},{hi2:+.4f}]':>26}")
        out["arms"][name] = {
            "auc_all_tess": float(aF), "delta_all_tess": dF,
            "ci_all_tess": [loF, hiF], "clears_all_tess": bool(loF > 0),
            "auc_tess2min": float(a2), "delta_tess2min": d2,
            "ci_tess2min": [lo2, hi2], "clears_tess2min": bool(lo2 > 0)}
    out["arms"]["A. baseline"] = {"auc_all_tess": float(base_auc_F),
                                  "auc_tess2min": float(base_auc_2)}

    print(f"\n  reference: refit-clean baseline for new work = {BASELINE_REF}")
    print("  bar: ci_lo > 0 on the paired bootstrap delta. Noise floor ~+/-0.003.")

    any_clear = any(v.get("clears_tess2min") for v in out["arms"].values()
                    if isinstance(v, dict))
    print("\n" + "=" * 84)
    print("VERDICT: " + ("at least one arm clears on tess_2min"
                         if any_clear else
                         "NO ARM CLEARS ci_lo > 0 on the tess_2min population"))
    print("=" * 84)

    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
