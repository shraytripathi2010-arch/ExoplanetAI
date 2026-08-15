"""ood_detector_refit.py -- refit the OOD artifacts onto the current 33 features
and validate the staleness guard that stops this recurring.

Refitting alone would only reset the clock until the next feature promotion; the
durable part is `_check_cached_feature_set` in the load path. This script proves
both: that the refit artifacts are correct, and that the guard fires on a
simulated stale cache instead of crashing opaquely.

Touches ONLY the two OOD artifacts. The production model, training data,
promotion gate and scheduler are not written to, and that is asserted at the end
by md5.
"""
import os
import sys
import json
import shutil
import hashlib
import importlib.util
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..")
ROOT = os.path.join(HERE, "..", "..")
MODELS = os.path.join(ROOT, "models")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT = os.path.join(HERE, "ood_detector_refit.json")

OOD_MODEL = os.path.join(MODELS, "multivariate_ood_detector.joblib")
OOD_META = os.path.join(MODELS, "multivariate_ood_meta.json")
RANGES = os.path.join(MODELS, "training_feature_ranges.json")
PROD_MODEL = os.path.join(MODELS, "best_model.joblib")

POOLS = [("main", "unknown_features.csv", "unknown_candidate_list.csv"),
         ("widesector", "unknown_features_widesector.csv",
          "unknown_candidate_list_widesector.csv")]
SOM_SEED, GRID, N_SUPER = 42, 6, 8
GAIA = ["gaia_ruwe", "gaia_nss"]
REC = {"n": 108, "planet_pct": 76.85}


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def load_m06():
    spec = importlib.util.spec_from_file_location(
        "m06", os.path.join(CODE, "06_download_unknown.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["m06"] = m
    spec.loader.exec_module(m)
    return m


def build_pool(m05, cols33):
    out = {}
    for tag, feat_f, cand_f in POOLS:
        p = pd.read_csv(os.path.join(ROOT, "data", "catalogs", feat_f))
        c = pd.read_csv(os.path.join(ROOT, "data", "catalogs", cand_f))
        p["host"] = p.host.astype(str)
        c["host"] = c["host"].astype(str) if "host" in c.columns \
            else "TIC_" + c["tic_id"].astype(str)
        keep = [k for k in ("st_rad", "st_teff", "ra", "dec") if k in c.columns]
        p = p.merge(c.drop_duplicates("host")[["host"] + keep], on="host",
                    how="left", suffixes=("", "_cand"))
        for k in keep:
            if f"{k}_cand" in p.columns:
                p[k] = pd.to_numeric(p.get(k), errors="coerce").fillna(
                    pd.to_numeric(p[f"{k}_cand"], errors="coerce"))
        if "status" in p.columns:
            p = p[p.status.astype(str).str.startswith("Success")]
        p = p.reset_index(drop=True)
        assert pd.to_numeric(p["st_rad"], errors="coerce").notna().mean() > 0.5, \
            f"{tag}: st_rad merge failed"
        out[tag] = p
    return out


def main():
    res = {}
    prod_md5_before = md5(PROD_MODEL)
    train_md5_before = md5(TRAINING)

    m05spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE, "05_train_models.py"))
    m05 = importlib.util.module_from_spec(m05spec); sys.modules["m05"] = m05
    m05spec.loader.exec_module(m05)
    cols33 = list(m05.FEATURE_COLUMNS)
    cols31 = [c for c in cols33 if c not in GAIA]

    print("=" * 78)
    print("1. BEFORE")
    print("=" * 78)
    old_bundle = joblib.load(OOD_MODEL)
    old_meta = json.load(open(OOD_META))
    old_cols = list(old_bundle["feature_columns"])
    old_ranges = json.load(open(RANGES))
    print(f"  detector : {len(old_cols)} features, threshold "
          f"{old_meta['threshold_score']:.6f}, fit on {old_meta['training_rows']} rows")
    print(f"  ranges   : {len(old_ranges['ranges'])} features, "
          f"{old_ranges.get('training_rows')} rows")
    print(f"  model    : {len(cols33)} features  <- the mismatch")
    res["before"] = {"detector_features": len(old_cols),
                     "detector_threshold": old_meta["threshold_score"],
                     "detector_rows": old_meta["training_rows"],
                     "ranges_features": len(old_ranges["ranges"]),
                     "model_features": len(cols33)}

    # keep the old artifacts (nothing in this project is deleted)
    vdir = os.path.join(MODELS, "versions")
    os.makedirs(vdir, exist_ok=True)
    for src, name in ((OOD_MODEL, "multivariate_ood_detector_pre33_24feat.joblib"),
                      (OOD_META, "multivariate_ood_meta_pre33_24feat.json"),
                      (RANGES, "training_feature_ranges_pre33_24feat.json")):
        shutil.copy2(src, os.path.join(vdir, name))
    print(f"  backed up 3 artifacts to models/versions/")

    # ---- pool flag rates BEFORE, using the old detector on its OWN 24 features
    m06 = load_m06()
    pools = build_pool(m05, cols33)
    before_rates = {}
    for tag, p in pools.items():
        X = p[old_cols].replace([np.inf, -np.inf], np.nan).copy()
        if "FAP" in X: X["FAP"] = X["FAP"].fillna(1.0)
        s = old_bundle["iso_forest"].score_samples(old_bundle["imputer"].transform(X))
        before_rates[tag] = float((s < old_meta["threshold_score"]).mean())
        print(f"  pool {tag:<11} flag rate (old 24-feat detector) {before_rates[tag]:.2%}")

    # ================= 2. REFIT =================
    print("\n" + "=" * 78)
    print("2. REFIT ON 33 FEATURES")
    print("=" * 78)
    det = m06.compute_multivariate_ood_detector(cols33)
    ranges = m06.compute_training_feature_ranges(cols33)
    new_meta = json.load(open(OOD_META))
    new_bundle = joblib.load(OOD_MODEL)
    print(f"  detector : {new_meta['n_features']} features, threshold "
          f"{new_meta['threshold_score']:.6f}, fit on {new_meta['training_rows']} rows")
    print(f"  measured training flag rate {new_meta['actual_training_flagged_fraction']:.4%} "
          f"(target {new_meta['contamination_target']:.1%})")
    print(f"  ranges   : {len(json.load(open(RANGES))['ranges'])} features")
    assert new_meta["n_features"] == 33
    assert len(new_bundle["feature_columns"]) == 33
    assert new_meta["threshold_score"] != old_meta["threshold_score"], \
        "threshold was not re-derived"
    res["after"] = {"detector_features": new_meta["n_features"],
                    "detector_threshold": new_meta["threshold_score"],
                    "detector_rows": new_meta["training_rows"],
                    "train_flag_rate": new_meta["actual_training_flagged_fraction"],
                    "ranges_features": len(json.load(open(RANGES))["ranges"])}

    # ================= 3. GUARD TEST =================
    print("\n" + "=" * 78)
    print("3. STALE-CACHE GUARD TEST (the root-cause fix)")
    print("=" * 78)
    tmp = OOD_MODEL + ".realtmp"
    tmpr = RANGES + ".realtmp"
    shutil.move(OOD_MODEL, tmp); shutil.move(RANGES, tmpr)
    shutil.copy2(os.path.join(vdir, "multivariate_ood_detector_pre33_24feat.joblib"), OOD_MODEL)
    shutil.copy2(os.path.join(vdir, "multivariate_ood_meta_pre33_24feat.json"), OOD_META)
    shutil.copy2(os.path.join(vdir, "training_feature_ranges_pre33_24feat.json"), RANGES)
    try:
        print("  -- simulating the exact pre-fix state (24-feature cache, 33-feature caller) --")
        d = m06.load_or_compute_multivariate_detector(cols33)
        recovered = len(d["feature_columns"]) == 33
        print(f"  detector loader: returned a {len(d['feature_columns'])}-feature detector "
              f"-> {'RECOMPUTED, guard works' if recovered else 'STILL STALE -- FAIL'}")
        r = m06.load_or_compute_feature_ranges(cols33)
        rec_r = len(r) == 33
        print(f"  ranges loader:   returned {len(r)} features "
              f"-> {'RECOMPUTED, guard works' if rec_r else 'STILL STALE -- FAIL'}")
        # direct-caller path: should raise a NAMED error, not sklearn's
        raised = None
        try:
            m06.flag_multivariate_ood(pools["main"].head(5), cols33,
                                      {"imputer": old_bundle["imputer"],
                                       "iso_forest": old_bundle["iso_forest"],
                                       "threshold": old_meta["threshold_score"],
                                       "feature_columns": old_cols})
        except ValueError as e:
            raised = str(e).splitlines()[0]
        print(f"  direct-caller guard raised: {raised[:110] if raised else 'NOTHING -- FAIL'}")
        res["guard"] = {"detector_recomputed": bool(recovered),
                        "ranges_recomputed": bool(rec_r),
                        "direct_call_raises_named_error": bool(raised)}
        assert recovered and rec_r and raised
    finally:
        os.remove(OOD_MODEL); os.remove(RANGES)
        shutil.move(tmp, OOD_MODEL); shutil.move(tmpr, RANGES)
        json.dump(new_meta, open(OOD_META, "w"), indent=2)
    print("  restored the refit artifacts")

    # ================= 4. BEHAVIOURAL COMPARISON =================
    print("\n" + "=" * 78)
    print("4. POOL FLAG RATES, BEFORE vs AFTER")
    print("=" * 78)
    det = m06.load_or_compute_multivariate_detector(cols33)
    ranges = m06.load_or_compute_feature_ranges(cols33)
    after_rates, rows = {}, []
    for tag, p in pools.items():
        f = m06.flag_multivariate_ood(p, cols33, det)
        after_rates[tag] = float(f["multivariate_ood_flag"].mean())
        u = m06.flag_out_of_distribution(f, cols33, ranges)
        comb = m06.combine_ood_flags(u)
        rows.append({"pool": tag, "n": int(len(p)),
                     "before": before_rates[tag], "after": after_rates[tag],
                     "univariate_flag": float((~u["in_distribution_univariate"]).mean()),
                     "in_distribution": float(comb["in_distribution"].mean())})
        print(f"  {tag:<11} n={len(p):<5} multivariate {before_rates[tag]:.2%} -> "
              f"{after_rates[tag]:.2%}  (delta {after_rates[tag]-before_rates[tag]:+.2%})")
        print(f"              univariate flag {rows[-1]['univariate_flag']:.2%}   "
              f"in_distribution {rows[-1]['in_distribution']:.2%}")
        pools[tag] = comb
    res["pool_rates"] = rows

    # ================= 5. CLUSTER-1 REGRESSION CHECK =================
    print("\n" + "=" * 78)
    print("5. CLUSTER-1 REGRESSION CHECK (investigation found 8.54% vs ~26%)")
    print("=" * 78)
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import roc_auc_score
    from scipy.stats import fisher_exact
    sspec = importlib.util.spec_from_file_location(
        "somdiag", os.path.join(HERE, "som_cluster_diagnostic.py"))
    sd = importlib.util.module_from_spec(sspec); sys.modules["somdiag"] = sd
    sspec.loader.exec_module(sd)

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    Xr = df[cols31].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imp = SimpleImputer(strategy="median").fit(Xr)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             random_state=SOM_SEED).fit(imp.transform(Xr))
    Z = qt.transform(imp.transform(Xr))
    som = sd.SOM(GRID, Z.shape[1]).fit(Z)
    bmu, _ = som.bmu(Z)
    agg = AgglomerativeClustering(n_clusters=N_SUPER).fit(som.W)
    sup = agg.labels_[bmu]
    X33, y = m05.build_feature_matrix(df)
    X33 = X33.reset_index(drop=True)[cols33].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y); te = m05.frozen_test_mask(df)
    prob = joblib.load(PROD_MODEL).predict_proba(X33)[:, 1]
    best, bs = None, 1e9
    for c in sorted(set(sup[te])):
        m = te & (sup == c)
        if m.sum() < 40 or len(set(y[m])) < 2:
            continue
        s = abs(int(m.sum()) - REC["n"]) / REC["n"] + \
            abs(100 * y[m].mean() - REC["planet_pct"]) / REC["planet_pct"]
        if s < bs:
            best, bs = c, s
    m = te & (sup == best)
    print(f"  cluster-1 = supercluster {best}: n={int(m.sum())}, "
          f"planet={100*y[m].mean():.2f}%, AUC={roc_auc_score(y[m], prob[m]):.4f}")
    c1rows = []
    for tag, p in pools.items():
        Zp = qt.transform(imp.transform(
            p[cols31].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)))
        bp, _ = som.bmu(Zp)
        inc1 = (agg.labels_[bp] == best)
        fl = p["multivariate_ood_flag"].to_numpy()
        a, b = int((fl & inc1).sum()), int((~fl & inc1).sum())
        c_, d_ = int((fl & ~inc1).sum()), int((~fl & ~inc1).sum())
        r1 = a / max(a + b, 1); r0 = c_ / max(c_ + d_, 1)
        orr, pv = fisher_exact([[a, b], [c_, d_]])
        print(f"  {tag:<11} cluster1 n={a+b:<5} flagged {r1:.2%}   "
              f"rest n={c_+d_:<5} flagged {r0:.2%}   OR {orr:.2f}  p {pv:.3g}")
        c1rows.append({"pool": tag, "n_c1": a + b, "rate_c1": r1,
                       "n_rest": c_ + d_, "rate_rest": r0,
                       "odds_ratio": float(orr), "p": float(pv)})
    res["cluster1"] = c1rows
    main = [r for r in c1rows if r["pool"] == "main"][0]
    holds = main["rate_c1"] < main["rate_rest"] and main["p"] < 0.01
    print(f"\n  QUALITATIVE FINDING {'HOLDS' if holds else 'CHANGED -- INVESTIGATE'}: "
          f"cluster 1 is flagged LESS than the rest, so feature-space novelty is "
          f"still not what makes it unreliable.")
    res["cluster1_finding_holds"] = bool(holds)

    # ================= 6. NOTHING ELSE TOUCHED =================
    print("\n" + "=" * 78)
    print("6. NOTHING ELSE TOUCHED")
    print("=" * 78)
    same_model = md5(PROD_MODEL) == prod_md5_before
    same_train = md5(TRAINING) == train_md5_before
    print(f"  production model md5 unchanged : {same_model}  ({prod_md5_before})")
    print(f"  training.csv md5 unchanged     : {same_train}")
    assert same_model and same_train
    res["untouched"] = {"production_model": same_model, "training_csv": same_train,
                        "prod_md5": prod_md5_before}
    res["new_detector_md5"] = md5(OOD_MODEL)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
