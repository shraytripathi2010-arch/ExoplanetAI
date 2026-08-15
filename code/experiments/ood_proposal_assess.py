"""ood_proposal_assess.py -- assessing three OOD/novelty proposals against the
OOD detector ALREADY DEPLOYED, and against the just-closed GPC finding.

WHAT IS ALREADY IN PRODUCTION
------------------------------
`06_download_unknown.py` fits an `IsolationForest(n_estimators=200,
contamination=0.02, random_state=42)` on the training set, thresholds it at the
2nd percentile of its own training scores, and uses it in `combine_ood_flags` ->
`split_and_rerank`, where `keep = in_distribution & ~below_triage_floor`. A
flagged candidate is DEMOTED off the shortlist that feeds human review. So
"density-based novelty detection that deprioritises unusual candidates" is not a
gap -- it ships.

THE ONE TEST NOBODY HAS RUN
----------------------------
Does that deployed detector already flag cluster-1 candidates at an elevated
rate? Cluster 1 is the subpopulation whose poor calibration has survived four
independent explanations (SOM diagnostic, RV-label-noise, pool false-positive
risk, ensemble diversity). This proposal is a fifth angle on the same open
question. If the deployed detector already concentrates on cluster 1, the
capability exists; if it is blind to cluster 1, that is a real and specific gap.
Either answer is worth more than any new model here.

THREE VALIDATION CHOICES, made deliberately
--------------------------------------------
1. The output under test is a FLAG, not a classifier feature, so the metric is
   not a resampled AUC delta. For the cluster-1 question it is a flag RATE with
   a Fisher exact test; for ensemble disagreement it is whether the flag
   identifies UNRELIABLE PREDICTIONS.
2. Mahalanobis vs IsolationForest is a REDUNDANCY question, so it is measured as
   rank correlation and flagged-set overlap on the same feature space, not as a
   performance contest.
3. Ensemble disagreement gets the test the GPC finding does NOT already answer:
   the GPC result was about CLASSIFICATION errors. Whether disagreement predicts
   its own model's unreliability is a different claim, so it is measured against
   the only honest baseline -- HGB's own confidence |p - 0.5|. If disagreement
   cannot beat that, it adds nothing that is not already free.
"""
import os
import sys
import json
import importlib.util
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
OOD_MODEL = os.path.join(ROOT, "models", "multivariate_ood_detector.joblib")
OOD_META = os.path.join(ROOT, "models", "multivariate_ood_meta.json")
OUT = os.path.join(HERE, "ood_proposal_assess.json")

POOLS = [("main", "unknown_features.csv", "unknown_candidate_list.csv"),
         ("widesector", "unknown_features_widesector.csv",
          "unknown_candidate_list_widesector.csv")]
SEED = 42
SOM_SEED, GRID, N_SUPER = 42, 6, 8
GAIA = ["gaia_ruwe", "gaia_nss"]
REC = {"n": 108, "planet_pct": 76.85, "auc": 0.8280}


def main():
    from scipy.stats import fisher_exact, spearmanr
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.covariance import MinCovDet, EmpiricalCovariance
    from sklearn.metrics import roc_auc_score

    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols33 = list(m05.FEATURE_COLUMNS)
    cols31 = [c for c in cols33 if c not in GAIA]
    assert len(cols33) == 33

    res = {}

    # ================= PART 0.1: deployed detector spec =================
    print("=" * 78)
    print("PART 0.1 -- DEPLOYED OOD DETECTOR, EXACT SPEC")
    print("=" * 78)
    bundle = joblib.load(OOD_MODEL)
    meta = json.load(open(OOD_META))
    ood_cols = list(bundle["feature_columns"])
    stale = [c for c in cols33 if c not in ood_cols]
    print(f"  features                {len(ood_cols)}  (model now uses {len(cols33)})")
    print(f"  contamination target    {meta['contamination_target']}")
    print(f"  threshold score         {meta['threshold_score']:.6f}")
    print(f"  measured flag rate      {meta['actual_training_flagged_fraction']:.4%} of training rows")
    print(f"  fit on                  {meta['training_rows']} rows "
          f"(training.csv now has {len(pd.read_csv(TRAINING))})")
    print(f"  MISSING {len(stale)} features: {stale}")
    print(f"  effect of a flag        in_distribution=False -> dropped from the "
          f"human-review shortlist by split_and_rerank")
    res["deployed_detector"] = {
        "n_features": len(ood_cols), "features": ood_cols,
        "missing_vs_current": stale,
        "contamination": meta["contamination_target"],
        "threshold": meta["threshold_score"],
        "measured_train_flag_rate": meta["actual_training_flagged_fraction"],
        "fit_rows": meta["training_rows"], "stale": bool(stale)}

    # crash reproduction
    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    try:
        bundle["imputer"].transform(df[cols33].replace([np.inf, -np.inf], np.nan))
        crash = None
    except Exception as e:
        crash = f"{type(e).__name__}: {str(e).splitlines()[0]}"
    print(f"  33-feature call path    {crash or 'OK'}")
    res["deployed_detector"]["crash_on_33"] = crash

    # ================= build pool matrices =================
    print("\n" + "=" * 78)
    print("PART 2 -- PRODUCTION AVAILABILITY ON BOTH POOLS (up front)")
    print("=" * 78)
    pools = {}
    for tag, feat_f, cand_f in POOLS:
        p = pd.read_csv(os.path.join(ROOT, "data", "catalogs", feat_f))
        c = pd.read_csv(os.path.join(ROOT, "data", "catalogs", cand_f))
        p["host"] = p.host.astype(str)
        # The candidate list keys on a BARE INTEGER `tic_id` (231620255) while the
        # feature table keys on `host` ("TIC_231620255"). Joining them naively
        # produces an all-NaN merge that silently median-imputes st_rad/st_teff --
        # caught by asserting coverage below rather than trusted.
        if "host" in c.columns:
            c["host"] = c["host"].astype(str)
        else:
            c["host"] = "TIC_" + c["tic_id"].astype(str)
        keep = [k for k in ("st_rad", "st_teff", "ra", "dec") if k in c.columns]
        c = c.drop_duplicates(subset="host")
        p = p.merge(c[["host"] + keep], on="host", how="left", suffixes=("", "_cand"))
        for k in keep:
            if f"{k}_cand" in p.columns:
                p[k] = pd.to_numeric(p[k], errors="coerce").fillna(
                    pd.to_numeric(p[f"{k}_cand"], errors="coerce"))
        ok = p.status.astype(str).str.startswith("Success") if "status" in p.columns \
            else pd.Series(True, index=p.index)
        p = p[ok].reset_index(drop=True)
        have33 = [c2 for c2 in cols33 if c2 in p.columns]
        cov = {c2: float(pd.to_numeric(p[c2], errors="coerce").notna().mean())
               for c2 in have33}
        print(f"  {tag:<11} n={len(p):<5} has {len(have33)}/33 columns; "
              f"min per-column coverage {min(cov.values()):.2%} "
              f"({min(cov, key=cov.get)})")
        for sc in ("st_rad", "st_teff"):
            assert cov.get(sc, 0) > 0.5, (
                f"{tag}: {sc} coverage {cov.get(sc, 0):.2%} -- the stellar-parameter "
                "merge failed, which would silently median-impute it and invalidate "
                "every flag rate below")
        missing_cols = [c2 for c2 in cols33 if c2 not in p.columns]
        if missing_cols:
            print(f"              columns absent entirely: {missing_cols}")
        pools[tag] = p
    res["pool_availability"] = {
        t: {"n": int(len(p)),
            "cols_present": int(sum(c2 in p.columns for c2 in cols33))}
        for t, p in pools.items()}

    # ================= PART 0.2: cluster-1 flag rate =================
    print("\n" + "=" * 78)
    print("PART 0.2 -- DOES THE DEPLOYED DETECTOR ALREADY FLAG CLUSTER 1?")
    print("=" * 78)
    sspec = importlib.util.spec_from_file_location(
        "somdiag", os.path.join(HERE, "som_cluster_diagnostic.py"))
    sd = importlib.util.module_from_spec(sspec); sys.modules["somdiag"] = sd
    sspec.loader.exec_module(sd)

    Xtr_raw = df[cols31].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan)
    imp = SimpleImputer(strategy="median").fit(Xtr_raw)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             random_state=SOM_SEED).fit(imp.transform(Xtr_raw))
    Ztr = qt.transform(imp.transform(Xtr_raw))
    som = sd.SOM(GRID, Ztr.shape[1]).fit(Ztr)
    bmu_tr, _ = som.bmu(Ztr)
    agg = AgglomerativeClustering(n_clusters=N_SUPER).fit(som.W)
    sup_tr = agg.labels_[bmu_tr]

    # behavioural selection of the cluster matching the recorded profile
    X33, y = m05.build_feature_matrix(df)
    X33 = X33.reset_index(drop=True)[cols33].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    te = m05.frozen_test_mask(df)
    prob = joblib.load(PROD).predict_proba(X33)[:, 1]
    best, bs = None, 1e9
    for c2 in sorted(set(sup_tr[te])):
        m = te & (sup_tr == c2)
        if m.sum() < 40 or len(set(y[m])) < 2:
            continue
        n, pf = int(m.sum()), 100 * y[m].mean()
        a = roc_auc_score(y[m], prob[m])
        s = abs(n - REC["n"]) / REC["n"] + abs(pf - REC["planet_pct"]) / REC["planet_pct"] \
            + abs(a - REC["auc"]) / REC["auc"]
        if s < bs:
            best, bs = c2, s
    m = te & (sup_tr == best)
    n1, pf1, a1 = int(m.sum()), 100 * y[m].mean(), roc_auc_score(y[m], prob[m])
    ok = abs(n1 - REC["n"]) <= 25 and abs(pf1 - REC["planet_pct"]) <= 6
    print(f"  cluster-1 re-derived as supercluster {best}: n={n1}, "
          f"planet={pf1:.2f}%, AUC={a1:.4f}")
    print(f"  behavioural validation vs recorded (n=108, 76.85%, AUC 0.828): "
          f"{'MATCH' if ok else 'NO MATCH -- VOID'}")
    res["cluster1"] = {"supercluster": int(best), "n_test": n1,
                       "planet_pct": float(pf1), "auc": float(a1),
                       "validated": bool(ok)}
    if not ok:
        json.dump(res, open(OUT, "w"), indent=2)
        print("  cluster-1 identification failed; aborting that arm.")
        return

    # fresh 33-feature detector, for the comparison the stale one cannot make
    Ximp33 = SimpleImputer(strategy="median").fit(
        X33.assign(FAP=X33.FAP.fillna(1.0)) if "FAP" in X33 else X33)
    Xtr33 = Ximp33.transform(X33.assign(FAP=X33.FAP.fillna(1.0)) if "FAP" in X33 else X33)
    iso33 = IsolationForest(n_estimators=200, contamination=0.02,
                            random_state=SEED, n_jobs=-1).fit(Xtr33)
    thr33 = float(np.percentile(iso33.score_samples(Xtr33), 2.0))

    # deployed 24-feature detector, on its OWN features (the honest current behaviour)
    imp24, iso24, thr24 = bundle["imputer"], bundle["iso_forest"], meta["threshold_score"]

    rows = []
    for tag, p in pools.items():
        have31 = [c2 for c2 in cols31 if c2 in p.columns]
        if len(have31) < len(cols31):
            print(f"  {tag}: only {len(have31)}/{len(cols31)} SOM features present -- "
                  "cluster assignment skipped")
            continue
        Zp = qt.transform(imp.transform(
            p[cols31].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)))
        bmu_p, _ = som.bmu(Zp)
        sup_p = agg.labels_[bmu_p]
        in_c1 = (sup_p == best)

        P24 = p[ood_cols].replace([np.inf, -np.inf], np.nan).copy()
        if "FAP" in P24: P24["FAP"] = P24["FAP"].fillna(1.0)
        f24 = iso24.score_samples(imp24.transform(P24)) < thr24

        have33p = [c2 for c2 in cols33 if c2 in p.columns]
        f33 = None
        if len(have33p) == len(cols33):
            P33 = p[cols33].replace([np.inf, -np.inf], np.nan).copy()
            if "FAP" in P33: P33["FAP"] = P33["FAP"].fillna(1.0)
            f33 = iso33.score_samples(Ximp33.transform(P33)) < thr33

        for lab, fl in (("deployed 24-feat", f24), ("fresh 33-feat", f33)):
            if fl is None:
                continue
            a, b = int((fl & in_c1).sum()), int((~fl & in_c1).sum())
            c_, d_ = int((fl & ~in_c1).sum()), int((~fl & ~in_c1).sum())
            r_c1 = a / max(a + b, 1); r_rest = c_ / max(c_ + d_, 1)
            orr, pv = fisher_exact([[a, b], [c_, d_]])
            print(f"  {tag:<11} {lab:<17} cluster1 n={a+b:<5} flagged {r_c1:.2%}   "
                  f"rest n={c_+d_:<5} flagged {r_rest:.2%}   OR {orr:.2f}  p {pv:.3g}")
            rows.append({"pool": tag, "detector": lab, "n_c1": a + b,
                         "flag_rate_c1": r_c1, "n_rest": c_ + d_,
                         "flag_rate_rest": r_rest, "odds_ratio": float(orr),
                         "p": float(pv)})
        p["_in_c1"] = in_c1
        p["_f24"] = f24
        if f33 is not None:
            p["_f33"] = f33
    res["cluster1_flag_rates"] = rows

    # ================= PART 0.4: Mahalanobis vs IsolationForest =================
    print("\n" + "=" * 78)
    print("PART 0.4 -- MAHALANOBIS vs ISOLATIONFOREST: different, or same idea?")
    print("=" * 78)
    emp = EmpiricalCovariance().fit(Xtr33)
    rob = MinCovDet(random_state=SEED, support_fraction=0.9).fit(Xtr33)
    iso_tr = iso33.score_samples(Xtr33)
    for lab, cov in (("Mahalanobis (empirical)", emp), ("Mahalanobis (robust MCD)", rob)):
        d2 = cov.mahalanobis(Xtr33)
        rho = float(spearmanr(-d2, iso_tr).statistic)
        k = int(0.02 * len(d2))
        top_m = set(np.argsort(-d2)[:k]); top_i = set(np.argsort(iso_tr)[:k])
        jac = len(top_m & top_i) / len(top_m | top_i)
        print(f"  {lab:<26} rho vs IF score {rho:+.3f}   "
              f"top-2% overlap (Jaccard) {jac:.3f}  [{len(top_m & top_i)}/{k} shared]")
        res.setdefault("mahalanobis", {})[lab] = {
            "spearman_vs_if": rho, "jaccard_top2pct": jac,
            "shared": len(top_m & top_i), "k": k}

    # ================= PART 0.5: ensemble disagreement as an unreliability flag =====
    print("\n" + "=" * 78)
    print("PART 0.5 -- DOES ENSEMBLE DISAGREEMENT PREDICT UNRELIABLE PREDICTIONS?")
    print("=" * 78)
    print("  (the GPC finding was about CLASSIFICATION errors; this asks the")
    print("   different question the proposal actually makes)")
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from catboost import CatBoostClassifier

    tr_mask, _ = m05.split_by_host(df)
    Xtr, ytr = X33[tr_mask], y[tr_mask]
    Xte, yte = X33[te], y[te]
    prod = joblib.load(PROD)
    p_h = clone(prod).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    p_c = CalibratedClassifierCV(
        Pipeline([("i", SimpleImputer(strategy="median")),
                  ("clf", CatBoostClassifier(verbose=0, random_seed=SEED))]),
        cv=5, method="sigmoid").fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    p_g = Pipeline([("i", SimpleImputer(strategy="median")),
                    ("s", StandardScaler()),
                    ("clf", GaussianProcessClassifier(
                        kernel=ConstantKernel(1.0) * RBF(1.0), random_state=SEED))
                    ]).fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    wrong = ((p_h >= 0.5).astype(int) != yte).astype(int)
    signals = {
        "HGB own confidence  |p-0.5|": -np.abs(p_h - 0.5),
        "disagree |hgb-cat|": np.abs(p_h - p_c),
        "disagree |hgb-gpc|": np.abs(p_h - p_g),
        "disagree sd(3 members)": np.std(np.column_stack([p_h, p_c, p_g]), axis=1),
        "deployed OOD score (24f, neg)": None,
    }
    P24t = X33[te][ood_cols].replace([np.inf, -np.inf], np.nan).copy()
    if "FAP" in P24t: P24t["FAP"] = P24t["FAP"].fillna(1.0)
    signals["deployed OOD score (24f, neg)"] = -iso24.score_samples(imp24.transform(P24t))
    print(f"  target = 'HGB is wrong at 0.5' ({int(wrong.sum())}/{len(yte)} stars)")
    print(f"  {'signal':<32}{'AUC for predicting error':>26}")
    res["disagreement"] = {}
    for lab, s in signals.items():
        a = float(roc_auc_score(wrong, s))
        star = "  <- baseline" if lab.startswith("HGB own") else ""
        print(f"  {lab:<32}{a:>26.4f}{star}")
        res["disagreement"][lab] = a

    # Does disagreement add anything ON TOP of the free baseline? Beating 0.5 is
    # not the bar -- beating |p-0.5|, which costs nothing, is.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    conf = np.abs(p_h - 0.5).reshape(-1, 1)
    dis = np.std(np.column_stack([p_h, p_c, p_g]), axis=1).reshape(-1, 1)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    inc = {}
    for lab, M in (("conf alone", conf), ("conf + disagreement", np.hstack([conf, dis]))):
        oof = cross_val_predict(LogisticRegression(max_iter=1000), M, wrong,
                                cv=cv, method="predict_proba")[:, 1]
        inc[lab] = float(roc_auc_score(wrong, oof))
    print(f"\n  INCREMENTAL VALUE (5-fold out-of-fold, target = HGB is wrong):")
    print(f"    conf alone            {inc['conf alone']:.4f}")
    print(f"    conf + disagreement   {inc['conf + disagreement']:.4f}   "
          f"delta {inc['conf + disagreement'] - inc['conf alone']:+.4f}")
    res["disagreement_incremental"] = inc

    # ================= |galactic b| control =================
    print("\n" + "=" * 78)
    print("|GALACTIC LATITUDE| CONTROL on the deployed OOD flag")
    print("=" * 78)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    for tag, p in pools.items():
        if "_f24" not in p.columns or "ra" not in p.columns:
            continue
        ra = pd.to_numeric(p.ra, errors="coerce").to_numpy()
        dec = pd.to_numeric(p.dec, errors="coerce").to_numpy()
        okc = np.isfinite(ra) & np.isfinite(dec)
        if okc.sum() < 50:
            continue
        gb = np.abs(SkyCoord(ra[okc] * u.deg, dec[okc] * u.deg).galactic.b.deg)
        f = p["_f24"].to_numpy()[okc]
        auc_b = float(roc_auc_score(f.astype(int), -gb)) if len(set(f)) > 1 else float("nan")
        rho_b = float(spearmanr(gb, f.astype(int)).statistic)
        print(f"  {tag:<11} AUC of |gal b| for predicting the OOD flag {auc_b:.4f}   "
              f"rho {rho_b:+.3f}")
        res.setdefault("galactic_b", {})[tag] = {"auc": auc_b, "rho": rho_b}

    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
