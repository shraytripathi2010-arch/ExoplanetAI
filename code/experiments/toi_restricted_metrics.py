"""toi_restricted_metrics.py -- SUPPLEMENTARY MEASUREMENT: how does the
deployed model score if evaluation is restricted to TOI-flagged candidates?

Nothing is trained, tuned or modified. This defines a different EVALUATION
POPULATION and reports what changes.

WHY THE RESTRICTION EXISTS
AstroNet, ExoMiner and RAVEN solve a narrower task than this project: they VET
candidates a mission pipeline already flagged as plausible. This project runs
TLS on raw light curves and does its own detection first, then classifies. Those
are different problems and their headline numbers are not comparable. Restricting
evaluation to TOI-flagged stars approximates the narrower task and gives a
context point against that literature -- nothing more.

HOW "WAS TOI-FLAGGED" IS DEFINED, AND WHY NOT BY DISPOSITION
Membership in the NASA archive's `toi` table is assigned when SPOC's pipeline
raises a threshold-crossing event and the object is alerted -- BEFORE any human
disposition. So "appears in the TOI table at all, under any disposition
including the still-undispositioned PC/APC" is a pre-label property. Using the
disposition itself (KP/CP vs FP/FA) would be circular, because disposition IS
the label this project trains on.

THE ASYMMETRY THAT GOVERNS THIS ENTIRE MEASUREMENT
The restriction is NOT a random subsample. Measured on the frozen test set:

    negatives that are TOIs : 231 / 231  = 100.0%
    positives that are TOIs : 269 / 867  =  31.0%

The negative class was SOURCED from the TOI false-positive list, so every
negative is a TOI by construction. The restriction therefore removes 598
positives and zero negatives. It is a one-sided filter on the positive class,
and it shifts prevalence from 79.0% to 53.8%. Any metric that depends on class
balance (precision@k, PR-AUC) moves for that reason alone, independent of task
difficulty. ROC-AUC is prevalence-invariant and is the only clean comparison.

ATTRIBUTION
Two effects could move the number in opposite directions:
  (a) TOI pre-filtering removes marginal detections -> easier task -> AUC up;
  (b) the removed rows are all POSITIVES, and if those non-TOI positives are
      EASY ones (bright, long-known planets found by other surveys) then
      removing them makes the remaining problem harder -> AUC down.
This script measures both directly by comparing model scores and key features
between TOI and non-TOI positives, rather than guessing which dominates.
"""
import os
import re
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
TOI_CSV = os.path.join(SCRIPT_DIR, "archive_toi_dispositions.csv")
ARCHIVE_CSV = os.path.join(SCRIPT_DIR, "archive_all_confirmed_tic.csv")
OLD_TIC_MAP = os.path.join(SCRIPT_DIR, "positive_class_tic_ids.csv")
RESULTS = os.path.join(SCRIPT_DIR, "toi_restricted_metrics.json")

KS = [10, 20, 50, 100, 200]
N_BOOT = 2000
SEED = 42


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def _pid(v):
    mm = re.search(r"(\d+)", str(v))
    return int(mm.group(1)) if mm else None


def resolver():
    amap = {}
    a = pd.read_csv(ARCHIVE_CSV)
    for h, t in zip(a["hostname"].astype(str), a["tic_id"].map(_pid)):
        if t:
            amap.setdefault(h, t)
            amap.setdefault(h.replace(" ", "_"), t)
    om = (pd.read_csv(OLD_TIC_MAP).dropna(subset=["tic_id"])
          .set_index("host")["tic_id"].astype("int64").to_dict())

    def r(h):
        h = str(h)
        mm = re.match(r"^TIC_(\d+)", h)
        return int(mm.group(1)) if mm else (amap.get(h) or om.get(h))
    return r


def boot_metric(y, s, fn, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); s = np.asarray(s)
    v = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        v.append(fn(y[i], s[i]))
    v = np.array(v)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def prec_rec_at_k(y, s, k):
    k = min(k, len(y))
    idx = np.argsort(-s)[:k]
    h = y[idx].sum()
    return h / k, h / max(y.sum(), 1)


def main():
    res = {}
    m05 = _m05()
    df = pd.read_csv(TRAINING_CSV)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True); y = np.asarray(y)
    tr, te = m05.split_by_host(df)
    te = np.asarray(te)

    rid = resolver()
    tic = df["host"].map(rid)
    toi_tics = set(pd.to_numeric(pd.read_csv(TOI_CSV)["tid"], errors="coerce")
                   .dropna().astype("int64"))
    is_toi = tic.isin(toi_tics).to_numpy()

    prod = joblib.load(PROD)
    p_all = prod.predict_proba(X)[:, 1]

    full = te
    restr = te & is_toi

    print("=" * 84)
    print("SUPPLEMENTARY: TOI-RESTRICTED EVALUATION OF THE DEPLOYED MODEL")
    print("=" * 84)
    print("Nothing retrained. This changes the EVALUATION POPULATION only.\n")

    # ---- population definition ----
    ct = pd.crosstab(pd.Series(is_toi[te], name="in TOI table"),
                     pd.Series(y[te], name="label"))
    print("test-set composition by TOI membership:")
    print(ct.to_string())
    n_unres = int(tic[te].isna().sum())
    print(f"\nstars whose TIC could not be resolved (counted as non-TOI): {n_unres}")
    print(f"  negatives that are TOIs: {100*is_toi[te & (y==1) == False].mean() if False else 100*is_toi[te][y[te]==0].mean():.1f}%"
          f"   <- negative class was SOURCED from the TOI FP list")
    print(f"  positives that are TOIs: {100*is_toi[te][y[te]==1].mean():.1f}%")
    print(f"\n  the restriction removes {int((te & ~is_toi & (y==1)).sum())} positives "
          f"and {int((te & ~is_toi & (y==0)).sum())} negatives -- a ONE-SIDED filter")

    res["composition"] = {
        "full_n": int(full.sum()), "full_pos": int(y[full].sum()),
        "full_neg": int((1 - y[full]).sum()), "full_prevalence": float(y[full].mean()),
        "restricted_n": int(restr.sum()), "restricted_pos": int(y[restr].sum()),
        "restricted_neg": int((1 - y[restr]).sum()),
        "restricted_prevalence": float(y[restr].mean()),
        "pct_negatives_that_are_toi": float(100 * is_toi[te][y[te] == 0].mean()),
        "pct_positives_that_are_toi": float(100 * is_toi[te][y[te] == 1].mean()),
        "n_unresolved_tic": n_unres,
        "removed_positives": int((te & ~is_toi & (y == 1)).sum()),
        "removed_negatives": int((te & ~is_toi & (y == 0)).sum())}

    # ---- metrics side by side ----
    print("\n" + "=" * 84)
    print("METRIC COMPARISON")
    print("=" * 84)
    res["metrics"] = {}
    print(f"  {'':<26}{'FULL test':>22}{'TOI-RESTRICTED':>22}")
    rows = {}
    for name, mask in (("full", full), ("restricted", restr)):
        yy, pp = y[mask], p_all[mask]
        prev = yy.mean()
        roc, rlo, rhi = boot_metric(yy, pp, roc_auc_score)
        ap, alo, ahi = boot_metric(yy, pp, average_precision_score)
        rows[name] = {
            "n": int(mask.sum()), "prevalence": float(prev),
            "roc_auc": float(roc_auc_score(yy, pp)), "roc_ci": [rlo, rhi],
            "pr_auc": float(average_precision_score(yy, pp)), "pr_ci": [alo, ahi],
            "pr_noskill": float(prev), "at_k": []}
        for k in KS:
            if k <= len(yy):
                pk, rk = prec_rec_at_k(yy, pp, k)
                rows[name]["at_k"].append({"k": k, "precision": float(pk),
                                           "recall": float(rk),
                                           "random_baseline": float(prev)})
    f_, r_ = rows["full"], rows["restricted"]
    print(f"  {'n':<26}{f_['n']:>22}{r_['n']:>22}")
    print(f"  {'positive prevalence':<26}{f_['prevalence']:>22.3f}{r_['prevalence']:>22.3f}")
    print(f"  {'ROC-AUC':<26}{f_['roc_auc']:>22.4f}{r_['roc_auc']:>22.4f}")
    print(f"  {'  95% CI':<26}{'[%.4f, %.4f]'%tuple(f_['roc_ci']):>22}"
          f"{'[%.4f, %.4f]'%tuple(r_['roc_ci']):>22}")
    print(f"  {'PR-AUC':<26}{f_['pr_auc']:>22.4f}{r_['pr_auc']:>22.4f}")
    print(f"  {'  no-skill baseline':<26}{f_['pr_noskill']:>22.3f}{r_['pr_noskill']:>22.3f}")
    print(f"  {'  lift over no-skill':<26}"
          f"{f_['pr_auc']/f_['pr_noskill']:>21.2f}x{r_['pr_auc']/r_['pr_noskill']:>21.2f}x")
    print()
    for i, k in enumerate(KS):
        fa = f_["at_k"][i] if i < len(f_["at_k"]) else None
        ra = r_["at_k"][i] if i < len(r_["at_k"]) else None
        if fa and ra:
            print(f"  {'P@'+str(k)+' (random)':<26}"
                  f"{'%.3f (%.3f)'%(fa['precision'],fa['random_baseline']):>22}"
                  f"{'%.3f (%.3f)'%(ra['precision'],ra['random_baseline']):>22}")
    res["metrics"] = rows
    d_roc = r_["roc_auc"] - f_["roc_auc"]
    print(f"\n  ROC-AUC delta (restricted - full): {d_roc:+.4f}")
    res["roc_delta"] = float(d_roc)

    # ---- attribution ----
    print("\n" + "=" * 84)
    print("ATTRIBUTION: why did it move?")
    print("=" * 84)
    pos_toi = te & is_toi & (y == 1)
    pos_non = te & ~is_toi & (y == 1)
    print(f"  model score on POSITIVES:")
    print(f"    TOI-flagged     n={int(pos_toi.sum()):4d}  mean {p_all[pos_toi].mean():.4f}"
          f"  median {np.median(p_all[pos_toi]):.4f}")
    print(f"    NOT TOI-flagged n={int(pos_non.sum()):4d}  mean {p_all[pos_non].mean():.4f}"
          f"  median {np.median(p_all[pos_non]):.4f}")
    easier = "EASIER" if p_all[pos_non].mean() > p_all[pos_toi].mean() else "HARDER"
    print(f"    -> the positives REMOVED by the restriction were {easier} for the model")
    res["attribution"] = {
        "pos_toi_mean_score": float(p_all[pos_toi].mean()),
        "pos_non_toi_mean_score": float(p_all[pos_non].mean()),
        "removed_positives_were": easier}

    print(f"\n  feature comparison (test-set positives):")
    print(f"    {'feature':<16}{'TOI-flagged':>16}{'not TOI':>14}{'diff (SD)':>12}")
    for feat in ("SDE", "snr", "depth", "st_rad", "chi2red_min"):
        if feat not in X.columns:
            continue
        a = pd.to_numeric(X.loc[pos_toi, feat], errors="coerce").dropna()
        b = pd.to_numeric(X.loc[pos_non, feat], errors="coerce").dropna()
        if not len(a) or not len(b):
            continue
        sd = np.sqrt((a.var() + b.var()) / 2)
        d = (a.mean() - b.mean()) / sd if sd > 0 else np.nan
        print(f"    {feat:<16}{a.mean():>16.3f}{b.mean():>14.3f}{d:>+12.2f}")
        res["attribution"].setdefault("features", {})[feat] = {
            "toi_mean": float(a.mean()), "non_toi_mean": float(b.mean()),
            "std_mean_diff": float(d)}

    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
