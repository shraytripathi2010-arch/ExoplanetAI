"""ranking_metrics.py -- MEASUREMENT of the deployed model under retrieval
metrics that match what the tool actually does. Not an experiment: nothing is
trained, tuned, promoted or rejected.

WHY RETRIEVAL METRICS ARE THE RIGHT FRAME
Nobody follows up a whole ranked list. They take the top k, where k is however
many nights of telescope time exist. And the pipeline is built exactly that
way -- `06_download_unknown.py:815` says so in as many words: "The pipeline
never actually applied a binary threshold -- it ranked by probability and took
the top N". `TRIAGE_PROBABILITY_FLOOR = 0.30` is a floor that holds weak
signals out of the shortlist, not a classifier cut, and the 0.5 in
best_model_metadata.json is a reporting convention only.

THE THING THAT MUST BE SAID FIRST
The clean test set is **79.0% positive** (867 confirmed planets, 231 vetted
false positives). The majority class is PLANETS. That has three consequences
which govern how every number below must be read:

  1. A random ranking scores precision@k = 0.790 at EVERY k. So a
     precision@20 of, say, 0.95 is +0.16 over chance, not "excellent".
  2. The PR-AUC no-skill baseline is 0.790, not 0.5.
  3. It is the OPPOSITE of the real candidate queue, where genuine planets are
     rare among unknown stars. Precision@k measured here is therefore an upper
     bound flattered by an inverted class balance, and is NOT a forecast of
     precision@k on the live queue.

Because of (3) this script also reports PREVALENCE-CORRECTED precision@k:
positives are subsampled to simulate rarer positive rates (50%, 25%, 10%, 5%)
while keeping every negative, repeated many times. That is the closest honest
approximation to "what would the top-20 look like if planets were rare", using
only labelled data we actually have.

It also reports the retrieval metrics in the RARE-CLASS direction -- ranking by
1-p to retrieve false positives, where prevalence is 21% -- since that is the
direction in which "average precision beats ROC-AUC on imbalanced data"
actually applies here.
"""
import os
import sys
import json
import glob
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE_CSV = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
# Scored unknown candidates live under results/, not data/candidates/ (which
# holds the DB-backed working set and carries no probability column).
CAND_GLOB = os.path.join(ROOT, "results", "unknown_candidates",
                         "ranked_candidates_in_distribution.csv")
RESULTS = os.path.join(SCRIPT_DIR, "ranking_metrics_results.json")

RANDOM_SEED = 42
N_BOOT = 2000
KS = [10, 20, 50, 100, 200]
TRIAGE_FLOOR = 0.30          # 06_download_unknown.TRIAGE_PROBABILITY_FLOOR
TIER_MEDIUM, TIER_HIGH = 0.90, 0.97   # 08_characterize_candidates.confidence_tier


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def prec_rec_at_k(y, s, k):
    """Precision and recall among the k highest-scored items."""
    k = min(k, len(y))
    idx = np.argsort(-s)[:k]
    hits = y[idx].sum()
    return hits / k, hits / max(y.sum(), 1)


def boot_metric(y, s, fn, n=N_BOOT, seed=RANDOM_SEED):
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


def main():
    res = {}
    m05 = _m05()
    df = pd.read_csv(TRAINING_CSV)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True); y = np.asarray(y)
    tr, te = m05.split_by_host(df)
    tr, te = np.asarray(tr), np.asarray(te)
    cad = pd.read_csv(CADENCE_CSV)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is_2min = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()

    prod = joblib.load(PROD)
    pops = {"full_clean_test": te, "2min_only_test": te & is_2min}

    print("=" * 84)
    print("RETRIEVAL METRICS -- MEASUREMENT OF THE DEPLOYED MODEL (nothing retrained)")
    print("=" * 84)
    print(f"operating points found in code: triage floor {TRIAGE_FLOOR} "
          f"(06_download_unknown.py), tiers {TIER_MEDIUM}/{TIER_HIGH} "
          f"(08_characterize_candidates.py)")
    print("the pipeline RANKS and takes top-N; 0.5 in metadata is a reporting")
    print("convention only, not an applied classification threshold.\n")

    res["operating_points"] = {"triage_floor": TRIAGE_FLOOR,
                               "tier_medium": TIER_MEDIUM, "tier_high": TIER_HIGH,
                               "metadata_threshold_is_reporting_only": True}
    res["populations"] = {}

    for pname, mask in pops.items():
        yy = y[mask]
        p = prod.predict_proba(X[mask])[:, 1]
        sde = pd.to_numeric(X.loc[mask, "SDE"], errors="coerce").fillna(-np.inf).to_numpy()
        prev = yy.mean()
        rec = {"n": int(mask.sum()), "n_pos": int(yy.sum()),
               "n_neg": int((1 - yy).sum()), "prevalence": float(prev)}

        print("=" * 84)
        print(f"{pname}: n={rec['n']} | {rec['n_pos']} planets / {rec['n_neg']} FPs "
              f"| POSITIVE PREVALENCE {prev:.1%}")
        print("=" * 84)

        # ---- headline scalar metrics ----
        roc = roc_auc_score(yy, p)
        ap = average_precision_score(yy, p)
        ap_m, ap_lo, ap_hi = boot_metric(yy, p, average_precision_score)
        roc_m, roc_lo, roc_hi = boot_metric(yy, p, roc_auc_score)
        # rare-class direction: retrieve FALSE POSITIVES (prevalence 21%)
        ap_neg = average_precision_score(1 - yy, 1 - p)
        apn_m, apn_lo, apn_hi = boot_metric(1 - yy, 1 - p, average_precision_score)
        print(f"  ROC-AUC                    {roc:.4f}  CI [{roc_lo:.4f}, {roc_hi:.4f}]"
              f"   (no-skill 0.500)")
        print(f"  PR-AUC / AP  (planets)     {ap:.4f}  CI [{ap_lo:.4f}, {ap_hi:.4f}]"
              f"   (no-skill {prev:.3f} <- NOT 0.5)")
        print(f"  PR-AUC / AP  (false pos)   {ap_neg:.4f}  CI [{apn_lo:.4f}, {apn_hi:.4f}]"
              f"   (no-skill {1-prev:.3f})  <- the rare-class direction")
        rec.update(roc_auc=float(roc), roc_ci=[roc_lo, roc_hi],
                   ap_positive=float(ap), ap_positive_ci=[ap_lo, ap_hi],
                   ap_positive_noskill=float(prev),
                   ap_negative=float(ap_neg), ap_negative_ci=[apn_lo, apn_hi],
                   ap_negative_noskill=float(1 - prev))

        # ---- precision@k / recall@k with baselines ----
        print(f"\n  {'k':>5}  {'P@k':>7}{'R@k':>8}  |  {'P@k SDE':>8}  "
              f"{'P@k random':>11}  {'lift vs random':>15}")
        rec["at_k"] = []
        for k in KS:
            if k > rec["n"]:
                continue
            pk, rk = prec_rec_at_k(yy, p, k)
            pk_sde, _ = prec_rec_at_k(yy, sde, k)
            print(f"  {k:>5}  {pk:>7.3f}{rk:>8.3f}  |  {pk_sde:>8.3f}  "
                  f"{prev:>11.3f}  {pk - prev:>+15.3f}")
            rec["at_k"].append({"k": k, "precision": float(pk), "recall": float(rk),
                                "precision_sde_baseline": float(pk_sde),
                                "precision_random_baseline": float(prev),
                                "lift_over_random": float(pk - prev)})

        # ---- prevalence-corrected precision@k ----
        print(f"\n  PREVALENCE-CORRECTED P@k -- positives subsampled, all {rec['n_neg']} "
              f"negatives kept, 200 repeats")
        print(f"  (this is the honest approximation to a queue where planets are RARE)")
        print(f"  {'target prev':>12}{'n_pos':>7}  " +
              "".join(f"{'P@'+str(k):>9}" for k in KS))
        rec["prevalence_corrected"] = []
        rng = np.random.RandomState(RANDOM_SEED)
        pos_idx = np.flatnonzero(yy == 1)
        neg_idx = np.flatnonzero(yy == 0)
        for target in (0.50, 0.25, 0.10, 0.05):
            n_pos_needed = int(round(target / (1 - target) * len(neg_idx)))
            if n_pos_needed < 5 or n_pos_needed > len(pos_idx):
                continue
            acc = {k: [] for k in KS}
            for _ in range(200):
                sel = np.r_[rng.choice(pos_idx, n_pos_needed, replace=False), neg_idx]
                ys, ps = yy[sel], p[sel]
                for k in KS:
                    if k <= len(ys):
                        acc[k].append(prec_rec_at_k(ys, ps, k)[0])
            row = {"target_prevalence": target, "n_pos": n_pos_needed}
            cells = ""
            for k in KS:
                if acc[k]:
                    mu = float(np.mean(acc[k])); row[f"P@{k}"] = mu
                    cells += f"{mu:>9.3f}"
                else:
                    cells += f"{'--':>9}"
            print(f"  {target:>12.2f}{n_pos_needed:>7}  {cells}")
            rec["prevalence_corrected"].append(row)

        # ---- threshold table ----
        print(f"\n  THRESHOLD TABLE (the tool ranks rather than thresholds; these show")
        print(f"  what a hard cut WOULD give)")
        print(f"  {'thresh':>8}{'n flagged':>11}{'precision':>11}{'recall':>9}{'F1':>8}   note")
        rec["thresholds"] = []
        for t in (0.10, TRIAGE_FLOOR, 0.50, 0.70, TIER_MEDIUM, TIER_HIGH, 0.99):
            pred = (p >= t).astype(int)
            if pred.sum() == 0:
                continue
            pr = precision_score(yy, pred, zero_division=0)
            rc = recall_score(yy, pred, zero_division=0)
            f1 = f1_score(yy, pred, zero_division=0)
            note = {TRIAGE_FLOOR: "<- triage floor (live)",
                    0.50: "reporting convention only",
                    TIER_MEDIUM: "<- Medium tier",
                    TIER_HIGH: "<- High tier"}.get(t, "")
            print(f"  {t:>8.2f}{int(pred.sum()):>11}{pr:>11.3f}{rc:>9.3f}{f1:>8.3f}   {note}")
            rec["thresholds"].append({"threshold": t, "n_flagged": int(pred.sum()),
                                      "precision": float(pr), "recall": float(rc),
                                      "f1": float(f1), "note": note})
        res["populations"][pname] = rec
        print()

    # ---- unknown candidate score distribution vs test positives ----
    print("=" * 84)
    print("SANITY CHECK: do confident UNKNOWNS look like confident CORRECT positives?")
    print("=" * 84)
    files = sorted(glob.glob(CAND_GLOB))
    cand = None
    for f in files:
        try:
            d = pd.read_csv(f)
            if "predicted_probability" in d.columns and len(d) > 20:
                cand = d
                print(f"  using {os.path.basename(f)}  ({len(d)} rows)")
                break
        except Exception:
            continue
    if cand is not None:
        cp = pd.to_numeric(cand["predicted_probability"], errors="coerce").dropna().to_numpy()
        te_p = prod.predict_proba(X[te])[:, 1]
        tp = te_p[y[te] == 1]
        fp = te_p[y[te] == 0]
        qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
        print(f"\n  {'quantile':>10}{'UNKNOWN cands':>16}{'test POSITIVES':>17}"
              f"{'test FALSE POS':>17}")
        for q in qs:
            print(f"  {q:>10.2f}{np.quantile(cp, q):>16.4f}{np.quantile(tp, q):>17.4f}"
                  f"{np.quantile(fp, q):>17.4f}")
        print(f"\n  n                {len(cp):>14}{len(tp):>17}{len(fp):>17}")
        print(f"  mean             {cp.mean():>14.4f}{tp.mean():>17.4f}{fp.mean():>17.4f}")
        print(f"  frac >= {TIER_HIGH}     {(cp>=TIER_HIGH).mean():>14.3f}"
              f"{(tp>=TIER_HIGH).mean():>17.3f}{(fp>=TIER_HIGH).mean():>17.3f}")
        print(f"  frac >= {TRIAGE_FLOOR}     {(cp>=TRIAGE_FLOOR).mean():>14.3f}"
              f"{(tp>=TRIAGE_FLOOR).mean():>17.3f}{(fp>=TRIAGE_FLOOR).mean():>17.3f}")
        res["score_distribution"] = {
            "source_file": os.path.basename(f), "n_unknown": int(len(cp)),
            "unknown_quantiles": {str(q): float(np.quantile(cp, q)) for q in qs},
            "test_positive_quantiles": {str(q): float(np.quantile(tp, q)) for q in qs},
            "test_falsepos_quantiles": {str(q): float(np.quantile(fp, q)) for q in qs},
            "unknown_mean": float(cp.mean()), "test_pos_mean": float(tp.mean()),
            "test_fp_mean": float(fp.mean()),
            "unknown_frac_high_tier": float((cp >= TIER_HIGH).mean()),
            "test_pos_frac_high_tier": float((tp >= TIER_HIGH).mean())}
    else:
        print("  no candidate file with predicted_probability found")
        res["score_distribution"] = None

    with open(RESULTS, "w") as f_:
        json.dump(res, f_, indent=2, default=float)
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
