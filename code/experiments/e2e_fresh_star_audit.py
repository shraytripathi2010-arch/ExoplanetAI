"""e2e_fresh_star_audit.py -- PART 3 of the system audit.

Takes a GENUINELY FRESH TESS target -- absent from training.csv, both pool
feature tables and both candidate lists -- and pushes it through the REAL
production code path, not a reimplementation:

    m06.try_search / download_one_star   (MAST)
    m06 preprocessing                    (same normalisation 02 applies)
    m06.compute_all_features             (TLS + all 22 TLS-derived features)
    m06.add_crowding_features            (TIC cone search)
    m06.add_variability_features         (RAW light curve, pre-flatten)
    m06.add_gaia_astrometry_features     (VizieR bulk)
    live model  -> probability
    live OOD detector + ranges -> flags
    conformal_calibration.json -> prediction set

Every function above is imported from the production modules unmodified. The
point is to prove a new candidate can complete the journey TODAY, so any silent
gap shows up as a NaN or an exception here rather than in a real run.

Writes NOTHING to production data: no training.csv, no pool CSV, no model.
"""
import os
import sys
import json
import time
import importlib.util
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..")
ROOT = os.path.join(HERE, "..", "..")
OUT = os.path.join(HERE, "e2e_fresh_star_audit.json")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def main():
    m05 = load("m05", os.path.join(CODE, "05_train_models.py"))
    m06 = load("m06", os.path.join(CODE, "06_download_unknown.py"))
    cols = list(m05.FEATURE_COLUMNS)
    res = {"steps": {}}

    # ---------- 1. find a genuinely fresh star ----------
    print("=" * 74); print("STEP 1: find a star absent from EVERY existing table"); print("=" * 74)
    known = set()
    tr = pd.read_csv(os.path.join(ROOT, "data", "training_dataset", "training.csv"))
    known |= set(tr.host.astype(str))
    for f in ("unknown_features.csv", "unknown_features_widesector.csv"):
        p = os.path.join(ROOT, "data", "catalogs", f)
        if os.path.exists(p):
            known |= set(pd.read_csv(p).host.astype(str))
    for f in ("unknown_candidate_list.csv", "unknown_candidate_list_widesector.csv"):
        p = os.path.join(ROOT, "data", "catalogs", f)
        if os.path.exists(p):
            d = pd.read_csv(p)
            key = "host" if "host" in d.columns else "tic_id"
            known |= {str(x) if str(x).startswith("TIC_") else f"TIC_{int(x)}"
                      for x in d[key]}
    print(f"  {len(known)} hosts already known to the project")

    from astroquery.mast import Catalogs, Observations
    # a real, well-observed TESS field; take bright dwarfs and pick the first
    # one the project has never seen
    cat = Catalogs.query_criteria(catalog="TIC", Tmag=[8.0, 9.5],
                                  objType="STAR", dec=[-70, -50], ra=[80, 95])
    cand = None
    for row in cat:
        tid = int(row["ID"])
        h = f"TIC_{tid}"
        if h in known:
            continue
        obs = Observations.query_criteria(target_name=str(tid),
                                          obs_collection="TESS",
                                          dataproduct_type="timeseries")
        if len(obs) == 0:
            continue
        cand = {"tic_id": tid, "host": h,
                "st_rad": float(row["rad"]) if row["rad"] is not None and not np.ma.is_masked(row["rad"]) else np.nan,
                "st_teff": float(row["Teff"]) if row["Teff"] is not None and not np.ma.is_masked(row["Teff"]) else np.nan,
                "st_mass": float(row["mass"]) if row["mass"] is not None and not np.ma.is_masked(row["mass"]) else np.nan,
                "ra": float(row["ra"]), "dec": float(row["dec"]),
                "n_obs": int(len(obs))}
        break
    if cand is None:
        print("  FAILED to find a fresh observed target"); return
    print(f"  SELECTED {cand['host']}  (Tmag-selected, {cand['n_obs']} TESS obs, "
          f"never seen by this project)")
    print(f"    st_rad {cand['st_rad']}  st_teff {cand['st_teff']}  "
          f"ra {cand['ra']:.4f} dec {cand['dec']:.4f}")
    res["star"] = cand

    # ---------- 2. download via production path ----------
    print("\n" + "=" * 74); print("STEP 2: download (production try_search / download_one_star)"); print("=" * 74)
    t0 = time.time()
    search, method = m06.try_search(cand["tic_id"])
    print(f"  try_search -> method={method}, {len(search) if search is not None else 0} products")
    raw_name = cand["host"]
    ok = m06.download_one_star(cand["tic_id"], raw_name)
    raw_path = os.path.join(m06.RAW_FOLDER, raw_name + ".csv")
    exists = os.path.exists(raw_path)
    print(f"  download_one_star -> {ok}; raw file present: {exists} "
          f"({time.time()-t0:.0f}s)")
    res["steps"]["download"] = {"ok": bool(ok), "raw_exists": bool(exists),
                                "method": str(method)}
    if not exists:
        print("  FAIL: no raw light curve; cannot continue")
        json.dump(res, open(OUT, "w"), indent=2, default=str); return
    raw = pd.read_csv(raw_path)
    print(f"  raw cadences: {len(raw)}")

    # ---------- 3. preprocess exactly as production does ----------
    print("\n" + "=" * 74); print("STEP 3: preprocess (production normalisation)"); print("=" * 74)
    os.makedirs(m06.PROCESSED_FOLDER, exist_ok=True)
    proc_path = os.path.join(m06.PROCESSED_FOLDER, cand["host"] + ".csv")
    # The production path is 06.preprocess_candidates(), which wraps
    # 02_preprocess.process_one_file for every raw file lacking a processed
    # counterpart. Calling the real function rather than reimplementing it is
    # the whole point of this audit.
    print("  calling m06.clean_light_curve -- the exact call preprocess_candidates makes")
    cleaned, pstatus = m06.clean_light_curve(raw_path)
    print(f"  status: {pstatus}")
    if cleaned is not None and len(cleaned):
        cleaned.to_csv(proc_path, index=False)
        print(f"  wrote {len(cleaned)} cleaned cadences")
    else:
        print("  clean_light_curve returned nothing")
    print(f"  processed file present: {os.path.exists(proc_path)}")
    res["steps"]["preprocess"] = {"processed_exists": os.path.exists(proc_path)}
    if not os.path.exists(proc_path):
        print("  FAIL: preprocessing produced no output")
        json.dump(res, open(OUT, "w"), indent=2, default=str); return

    # ---------- 4. TLS + all 22 TLS features ----------
    print("\n" + "=" * 74); print("STEP 4: compute_all_features (TLS, production)"); print("=" * 74)
    t0 = time.time()
    feats, status = m06.compute_all_features(proc_path, cand["host"], cand["st_rad"],
                                             cand["st_mass"], cols)
    print(f"  status: {status}   ({time.time()-t0:.0f}s)")
    print(f"  returned {len(feats) if feats else 0} fields")
    res["steps"]["tls"] = {"status": str(status), "n_fields": len(feats or {})}

    row = dict(cand); row.update(feats or {})
    df1 = pd.DataFrame([row])

    # ---------- 5. the three post-TLS feature groups, real wiring ----------
    print("\n" + "=" * 74); print("STEP 5: crowding / variability / Gaia (production wiring)"); print("=" * 74)
    for label, fn, kw in (("crowding", m06.add_crowding_features, {}),
                          ("variability", m06.add_variability_features,
                           {"raw_dir": m06.RAW_FOLDER}),
                          ("gaia", m06.add_gaia_astrometry_features, {})):
        t0 = time.time()
        try:
            df1 = fn(df1, **kw)
            print(f"  {label:<12} OK ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  {label:<12} RAISED {type(e).__name__}: {str(e)[:120]}")
            res["steps"][label] = {"error": f"{type(e).__name__}: {e}"}

    # ---------- 6. all 33 features present? ----------
    print("\n" + "=" * 74); print("STEP 6: all 33 feature columns"); print("=" * 74)
    opt = set(m06.OPTIONAL_FEATURES)
    present, nan_req, nan_opt, absent = [], [], [], []
    for c in cols:
        if c not in df1.columns:
            absent.append(c); continue
        v = pd.to_numeric(df1[c], errors="coerce").iloc[0]
        if pd.isna(v):
            (nan_opt if c in opt else nan_req).append(c)
        else:
            present.append(c)
    print(f"  populated      {len(present)}/33")
    print(f"  NaN (OPTIONAL) {len(nan_opt)}  {nan_opt}")
    print(f"  NaN (required) {len(nan_req)}  {nan_req}")
    print(f"  ABSENT column  {len(absent)}  {absent}")
    res["steps"]["features"] = {"populated": len(present), "nan_optional": nan_opt,
                                "nan_required": nan_req, "absent": absent}
    for c in ("crowd_flux_ratio_max", "crowd_nearest_arcsec", "var_oot_rms",
              "var_excess", "gaia_ruwe", "gaia_nss"):
        v = df1[c].iloc[0] if c in df1.columns else None
        print(f"    {c:<24} {v}")

    # ---------- 7. score / OOD / conformal ----------
    print("\n" + "=" * 74); print("STEP 7: live model, OOD, conformal"); print("=" * 74)
    X = df1.reindex(columns=cols)
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    prod = joblib.load(os.path.join(ROOT, "models", "best_model.joblib"))
    p = float(prod.predict_proba(X)[:, 1][0])
    print(f"  predicted_probability  {p:.4f}")
    det = m06.load_or_compute_multivariate_detector(cols)
    fl = m06.flag_multivariate_ood(X.assign(host=cand["host"]), cols, det)
    ranges = m06.load_or_compute_feature_ranges(cols)
    uni = m06.flag_out_of_distribution(fl, cols, ranges)
    comb = m06.combine_ood_flags(uni)
    print(f"  multivariate_ood_flag  {bool(comb['multivariate_ood_flag'].iloc[0])} "
          f"(score {float(comb['multivariate_ood_score'].iloc[0]):.4f})")
    print(f"  in_distribution        {bool(comb['in_distribution'].iloc[0])}")
    sys.path.insert(0, os.path.join(ROOT, "web"))
    import conformal
    cs = conformal.summary(p)
    print(f"  conformal available    {conformal.available()}")
    print(f"  conformal summary      {json.dumps(cs)[:200]}")
    res["steps"]["scoring"] = {
        "probability": p,
        "multivariate_ood_flag": bool(comb["multivariate_ood_flag"].iloc[0]),
        "in_distribution": bool(comb["in_distribution"].iloc[0]),
        "conformal_available": bool(conformal.available()),
        "conformal": cs}

    json.dump(res, open(OUT, "w"), indent=2, default=str)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
