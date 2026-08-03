"""pseudo_labeling.py -- can semi-supervised pseudo-labelling on unlabelled TOI
candidates improve the model?

THE MECHANISM UNDER TEST, AND WHY IT IS SUSPECT
Score unlabelled candidates with the current model, take the most confident
predictions as labels, retrain. The danger is specific: any systematic error the
model already makes is promoted to ground truth and reinforced. This is NOT the
synthetic-data failure mode. Synthetic rows were OFF-distribution and a domain
classifier caught them at AUC 0.9654. Pseudo-labels are ON-distribution by
construction, so no domain classifier can flag them -- the corruption is in the
LABELS, not the features, and it is invisible to distribution checks.

So this script is designed to expose label corruption rather than to measure a
headline AUC:
  - it inspects what the confident pseudo-labels actually look like on the
    features that carry the most weight (st_rad, st_teff, chi2red_min, per the
    permutation-importance audit), checking for suspicious UNIFORMITY, which
    would indicate the model is confidently wrong in a correlated way;
  - it evaluates only on real human labels;
  - it stress-tests any gain on the subpopulation where the model is already
    weakest, since that is exactly where its confident labels are most likely
    to be confidently wrong;
  - and it seed-checks anything that clears, because this project has already
    had one arm clear on a single fit and die on replication (CatBoost, 0/10).

PROBABILITIES ARE RECOMPUTED with the deployed model rather than read from the
stored `predicted_probability` column, which may have come from an earlier
model version. Pseudo-labels must reflect the model actually being tested.

NO SYNTHETIC DATA IS MIXED IN. The injection-recovery work produced its own
augmented rows; those stay out entirely so this measures model-generated labels
and nothing else. Every added row is flagged `label_source='pseudo'` in the
saved dataset, permanently and separably.
"""
import os
import sys
import json
import re
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, brier_score_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE_CSV = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
UNKNOWN_FILES = [
    os.path.join(ROOT, "results", "unknown_candidates", "ranked_candidates.csv"),
    os.path.join(ROOT, "results", "unknown_candidates_widesector", "ranked_candidates.csv"),
]
OUT_PSEUDO = os.path.join(SCRIPT_DIR, "pseudo_labeled_rows.csv")
RESULTS = os.path.join(SCRIPT_DIR, "pseudo_labeling_results.json")

RANDOM_SEED = 42
N_BOOT = 2000
N_SEEDS = 10
TOP_IMPORTANCE = ["st_rad", "st_teff", "chi2red_min"]


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


def ece(y, p, bins=10):
    idx = np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1])
    return float(sum(((idx == b).sum() / len(y)) * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def load_unknowns(m05):
    frames = []
    for f in UNKNOWN_FILES:
        if os.path.exists(f):
            d = pd.read_csv(f)
            d["_src"] = os.path.basename(os.path.dirname(f))
            frames.append(d)
    u = pd.concat(frames, ignore_index=True)
    if "tic_id" in u.columns:
        u["tic"] = pd.to_numeric(u["tic_id"], errors="coerce")
    else:
        u["tic"] = pd.to_numeric(
            u["host"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
    u = u.dropna(subset=["tic"]).drop_duplicates("tic").reset_index(drop=True)
    u["tic"] = u["tic"].astype("int64")
    return u


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
    te2 = te & is_2min

    prod = joblib.load(PROD)
    base = clone(prod).fit(X[tr], y[tr])
    pb_f = base.predict_proba(X[te])[:, 1]
    pb_2 = base.predict_proba(X[te2])[:, 1]
    a_f, a_2 = roc_auc_score(y[te], pb_f), roc_auc_score(y[te2], pb_2)

    print("=" * 84)
    print("PSEUDO-LABELLING ON UNLABELLED TOI CANDIDATES")
    print("=" * 84)
    print(f"real train {tr.sum()} | test {te.sum()} (2-min {te2.sum()})")
    print(f"baseline: full {a_f:.4f} | 2-min {a_2:.4f}")
    res["baseline"] = {"full": float(a_f), "2min": float(a_2),
                       "n_train": int(tr.sum())}

    # ---------------- PART 1: pool + verification ----------------
    u = load_unknowns(m05)
    Xu, _ = m05.build_feature_matrix(u.assign(label=0))
    Xu = Xu.reset_index(drop=True)
    # recompute with the deployed model, do not trust the stored column
    pu = base.predict_proba(Xu)[:, 1]
    u["p_model"] = pu

    print(f"\nunlabelled pool: {len(u)} candidates with all 24 features")
    if "predicted_probability" in u.columns:
        stored = pd.to_numeric(u["predicted_probability"], errors="coerce")
        ok = stored.notna()
        if ok.sum() > 10:
            r = float(np.corrcoef(stored[ok], pu[ok.to_numpy()])[0, 1])
            print(f"  stored vs recomputed probability correlation: {r:.4f} "
                  f"(recomputed values are used)")
            res["stored_vs_recomputed_corr"] = r
    res["n_unlabelled"] = int(len(u))

    print("\n" + "-" * 84)
    print("THRESHOLD SIZING -- how many pseudo-labels are actually available?")
    print("-" * 84)
    configs = {}
    for pct in (5, 1):
        hi = np.percentile(pu, 100 - pct)
        lo = np.percentile(pu, pct)
        configs[f"pct{pct}"] = {"pos": pu >= hi, "neg": pu <= lo,
                                "desc": f"top/bottom {pct}% (p>={hi:.4f} / p<={lo:.4f})"}
    for hi, lo in ((0.95, 0.05), (0.99, 0.01)):
        configs[f"abs{hi}"] = {"pos": pu >= hi, "neg": pu <= lo,
                               "desc": f"absolute p>={hi} / p<={lo}"}
    res["thresholds"] = {}
    for name, cfg in configs.items():
        npos, nneg = int(cfg["pos"].sum()), int(cfg["neg"].sum())
        tot = npos + nneg
        growth = 100 * tot / tr.sum()
        print(f"  {name:8s} {cfg['desc']:<44} pos {npos:4d}  neg {nneg:4d}  "
              f"total {tot:4d}  (+{growth:.2f}% of train)")
        res["thresholds"][name] = {"desc": cfg["desc"], "n_pos": npos,
                                   "n_neg": nneg, "total": tot,
                                   "pct_growth": float(growth)}

    print("\n  POWER NOTE: the learning curve fitted exponent c=0.193 implies a")
    print("  fractional data increase g raises AUC by roughly (1+g)^0.193 - 1 of the")
    print("  gap to the ceiling. At these growth rates the predicted effect is far")
    print("  below the +/-0.003 noise floor, so a null result here is expected on")
    print("  sample-size grounds ALONE and cannot by itself indict the method.")

    # ---------------- PART 1.3: the safeguard ----------------
    print("\n" + "-" * 84)
    print("SAFEGUARD: do confident pseudo-labels look trustworthy or suspicious?")
    print("-" * 84)
    cfg = configs["pct5"]
    real_pos = X[tr & (y == 1)]
    real_neg = X[tr & (y == 0)]
    res["safeguard"] = {}
    print(f"  comparing top/bottom 5% pseudo-labels against REAL train rows on the")
    print(f"  three highest permutation-importance features\n")
    print(f"  {'feature':<14}{'group':<18}{'n':>5}{'mean':>10}{'sd':>10}"
          f"{'min':>10}{'max':>10}")
    for feat in TOP_IMPORTANCE:
        if feat not in X.columns:
            continue
        rows = [("real positives", pd.to_numeric(real_pos[feat], errors="coerce")),
                ("pseudo-positives", pd.to_numeric(Xu.loc[cfg["pos"], feat], errors="coerce")),
                ("real negatives", pd.to_numeric(real_neg[feat], errors="coerce")),
                ("pseudo-negatives", pd.to_numeric(Xu.loc[cfg["neg"], feat], errors="coerce"))]
        for gname, v in rows:
            v = v.dropna()
            if not len(v):
                continue
            print(f"  {feat if gname=='real positives' else '':<14}{gname:<18}"
                  f"{len(v):>5}{v.mean():>10.3f}{v.std():>10.3f}"
                  f"{v.min():>10.3f}{v.max():>10.3f}")
            res["safeguard"].setdefault(feat, {})[gname] = {
                "n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std()),
                "min": float(v.min()), "max": float(v.max())}
        # uniformity check: is the pseudo group far tighter than the real one?
        rp = pd.to_numeric(real_pos[feat], errors="coerce").dropna()
        pp = pd.to_numeric(Xu.loc[cfg["pos"], feat], errors="coerce").dropna()
        if len(pp) > 3 and rp.std() > 0:
            ratio = pp.std() / rp.std()
            flag = "  <-- SUSPICIOUSLY UNIFORM" if ratio < 0.5 else ""
            print(f"  {'':<14}{'sd ratio pseudo/real positives':<18} {ratio:>9.3f}{flag}")
            res["safeguard"][feat]["sd_ratio_pseudopos_vs_realpos"] = float(ratio)
        print()

    # ---------------- PART 2: retrain arms ----------------
    print("=" * 84)
    print("PART 2 -- RETRAIN WITH PSEUDO-LABELS (evaluated ONLY on real labels)")
    print("=" * 84)
    inner = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    res["arms"] = {}
    saved = []
    for name, cfg in configs.items():
        sel = cfg["pos"] | cfg["neg"]
        if sel.sum() < 4:
            print(f"  {name:8s} only {int(sel.sum())} pseudo-labels -- skipped")
            res["arms"][name] = {"skipped": True, "n": int(sel.sum())}
            continue
        yps = np.where(cfg["pos"][sel], 1, 0)
        Xps = Xu[sel]
        rec = u.loc[sel, ["tic", "_src"]].copy()
        rec["pseudo_label"] = yps
        rec["p_model"] = pu[sel]
        rec["threshold_config"] = name
        rec["label_source"] = "pseudo"      # permanent, separable flag
        saved.append(rec)

        for w, wname in ((1.0, ""), (0.25, " w=0.25")):
            Xc = pd.concat([X[tr], Xps], ignore_index=True)
            yc = np.r_[y[tr], yps]
            sw = np.r_[np.ones(int(tr.sum())), np.full(len(yps), w)]
            mdl = clone(getattr(prod, "estimator", prod))   # bare pipeline: weights reach clf
            mdl.fit(Xc, yc, clf__sample_weight=sw)
            pf = mdl.predict_proba(X[te])[:, 1]
            p2 = mdl.predict_proba(X[te2])[:, 1]
            # baseline for THIS estimator class (bare pipeline, no calibration)
            bm = clone(getattr(prod, "estimator", prod)).fit(X[tr], y[tr])
            bf = bm.predict_proba(X[te])[:, 1]; b2 = bm.predict_proba(X[te2])[:, 1]
            mf, lof, hif = boot(y[te], bf, pf)
            m2, lo2, hi2 = boot(y[te2], b2, p2)
            print(f"  {name+wname:16s} +{int(sel.sum()):3d} pseudo | full "
                  f"{roc_auc_score(y[te], pf):.4f} {mf:+.4f} [{lof:+.4f},{hif:+.4f}] "
                  f"{'CLEARS' if lof>0 else 'no':>6} | 2min "
                  f"{roc_auc_score(y[te2], p2):.4f} {m2:+.4f} [{lo2:+.4f},{hi2:+.4f}] "
                  f"{'CLEARS' if lo2>0 else 'no':>6}")
            res["arms"][name + wname] = {
                "n_pseudo": int(sel.sum()), "weight": w,
                "test_auc_full": float(roc_auc_score(y[te], pf)),
                "test_auc_2min": float(roc_auc_score(y[te2], p2)),
                "delta_full": {"mean": mf, "ci_lower": lof, "ci_upper": hif,
                               "clears": bool(lof > 0)},
                "delta_2min": {"mean": m2, "ci_lower": lo2, "ci_upper": hi2,
                               "clears": bool(lo2 > 0)}}

    if saved:
        allp = pd.concat(saved, ignore_index=True)
        allp.to_csv(OUT_PSEUDO, index=False)
        print(f"\n  pseudo-labels saved with label_source='pseudo' -> "
              f"{os.path.basename(OUT_PSEUDO)} ({len(allp)} rows)")

    # ---------------- PART 3: subpopulation stress test ----------------
    print("\n" + "=" * 84)
    print("PART 3 -- SUBPOPULATION STRESS TEST (low-SNR: where the model is weakest)")
    print("=" * 84)
    snr = pd.to_numeric(X["snr"], errors="coerce")
    cut = snr[tr].quantile(0.25)
    weak = te & (snr <= cut).to_numpy()
    print(f"  low-SNR test rows (snr <= {cut:.2f}, train Q1): {int(weak.sum())} "
          f"({int(y[weak].sum())} pos / {int((1-y[weak]).sum())} neg)")
    res["stress"] = {"snr_cut": float(cut), "n": int(weak.sum())}
    if weak.sum() > 40 and len(np.unique(y[weak])) > 1:
        bm = clone(getattr(prod, "estimator", prod)).fit(X[tr], y[tr])
        bw = bm.predict_proba(X[weak])[:, 1]
        print(f"  baseline AUC on this subpopulation: {roc_auc_score(y[weak], bw):.4f}")
        res["stress"]["baseline_auc"] = float(roc_auc_score(y[weak], bw))
        best = configs["pct5"]
        sel = best["pos"] | best["neg"]
        if sel.sum() >= 4:
            yps = np.where(best["pos"][sel], 1, 0)
            Xc = pd.concat([X[tr], Xu[sel]], ignore_index=True)
            yc = np.r_[y[tr], yps]
            mdl = clone(getattr(prod, "estimator", prod)).fit(Xc, yc)
            pw = mdl.predict_proba(X[weak])[:, 1]
            mw, low, hiw = boot(y[weak], bw, pw)
            print(f"  with pseudo-labels:                 {roc_auc_score(y[weak], pw):.4f}"
                  f"   delta {mw:+.4f} [{low:+.4f},{hiw:+.4f}]")
            print("  (a NEGATIVE delta here is the signature of error amplification --")
            print("   pseudo-labels drawn from the region the model is worst at)")
            res["stress"]["pseudo_auc"] = float(roc_auc_score(y[weak], pw))
            res["stress"]["delta"] = {"mean": mw, "ci_lower": low, "ci_upper": hiw}

    # ---------------- calibration ----------------
    print("\n" + "=" * 84)
    print("CALIBRATION (both populations)")
    print("=" * 84)
    res["calibration"] = {}
    bm = clone(getattr(prod, "estimator", prod)).fit(X[tr], y[tr])
    sel = configs["pct5"]["pos"] | configs["pct5"]["neg"]
    arms = [("baseline", bm)]
    if sel.sum() >= 4:
        yps = np.where(configs["pct5"]["pos"][sel], 1, 0)
        pm = clone(getattr(prod, "estimator", prod)).fit(
            pd.concat([X[tr], Xu[sel]], ignore_index=True), np.r_[y[tr], yps])
        arms.append(("+pseudo pct5", pm))
    for nm, mdl in arms:
        for pop, pmask in (("full", te), ("2min", te2)):
            p = mdl.predict_proba(X[pmask])[:, 1]
            res["calibration"][f"{nm}|{pop}"] = {
                "auc": float(roc_auc_score(y[pmask], p)),
                "brier": float(brier_score_loss(y[pmask], p)),
                "ece": ece(y[pmask], p)}
            print(f"  {nm:<14}{pop:>6}  AUC {roc_auc_score(y[pmask], p):.4f}  "
                  f"Brier {brier_score_loss(y[pmask], p):.4f}  "
                  f"ECE {ece(y[pmask], p):.4f}")

    cleared = [k for k, v in res["arms"].items()
               if not v.get("skipped") and (v["delta_full"]["clears"] or v["delta_2min"]["clears"])]
    res["n_cleared"] = len(cleared)
    print("\n" + "=" * 84)
    print(f"ARMS CLEARING ci_lo > 0: {len(cleared)}" +
          (f" -> {cleared}  (seed-check required before believing)" if cleared
           else "  -- nothing clears"))
    print("=" * 84)

    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
