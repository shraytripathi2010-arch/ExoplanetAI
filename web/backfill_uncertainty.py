"""
backfill_uncertainty.py -- computes the bootstrap uncertainty band for every
candidate already in the database, not just ones found from now on.

RETAINED BUT NOT DISPLAYED (2026-10-11): the app no longer shows the bootstrap band.
Its refit models are systematically different from production, the score fell outside
their 16-84% spread for 117 of 294 candidates, and the band has no coverage property;
the conformal sets are the uncertainty display (RESULTS_SUMMARY.md register section I).
Kept so the decision can be revisited without rebuilding anything.

Same standard as every prior feature in this project: a new evidence type is
applied retroactively to the existing candidate list, so the site never shows
a mix of candidates that have it and candidates that silently don't.

The model's raw feature vectors are NOT stored in the candidates table, so
they are rebuilt from data/catalogs/unknown_features.csv plus the stored
st_rad/st_teff, and must reproduce the stored probability exactly before any
band is written.

Rules (2026-10-11):
  * the ensemble's target_model_md5 must be the deployed model, or nothing is written;
  * a NaN in a feature production imputes is scored through the members' own
    imputer, exactly as production scores it;
  * a NaN in a required feature, or a score not from the deployed model, gets
    NO band, and any old band on that row is cleared;
  * every band is stamped with the model it describes and the ensemble build time.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
import uncertainty

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FEATURES_PATH = os.path.join(ROOT, "data", "catalogs", "unknown_features.csv")
META_PATH = os.path.join(ROOT, "models", "best_model_metadata.json")


def main():
    if not uncertainty.ensemble_available():
        raise SystemExit("No bootstrap ensemble found -- run "
                         "code/09_build_bootstrap_ensemble.py first.")

    # The uncertainty_* columns arrive via MIGRATIONS, which only run inside
    # init_db()/ensure_schema() -- a standalone script that just imports db
    # gets whatever schema is already on disk and then fails on the UPDATE.
    # ensure_schema(), NOT init_db(): init_db also resets any 'running' job to
    # 'failed', which would kill a legitimately in-progress Update if this
    # backfill happened to be run while one was going (the exact bug sync.py
    # already hit once).
    db.ensure_schema()

    with open(META_PATH) as f:
        feature_columns = json.load(f)["feature_columns"]

    # SOURCE CHOICE (a first attempt got this wrong -- worth recording):
    # ranked_candidates.csv is OVERWRITTEN by every pipeline run, so it only
    # ever describes the most recent run's stars. Sourcing from it wrote bands
    # onto 191 unscored rows while missing 95 of the candidates that actually
    # have a displayed probability.
    #
    # unknown_features.csv is APPEND-ONLY across runs (06 concats + dedupes on
    # host), so it covers every candidate the database has ever accumulated.
    # It holds the TLS features but not st_rad/st_teff, which are merged in
    # from the TIC catalog at scoring time -- those come from the candidate's
    # own stored characterization instead.
    feats = pd.read_csv(FEATURES_PATH)
    feats["tic_id"] = feats["host"].str.replace("TIC_", "", regex=False).astype("int64")
    feats = feats.drop_duplicates(subset="tic_id", keep="last").set_index("tic_id")

    # GATE: the ensemble must describe the deployed model. A band from members
    # cloned from a different model is a statement about that other model.
    import hashlib
    with open(os.path.join(ROOT, "models", "best_model.joblib"), "rb") as f:
        deployed_md5 = hashlib.md5(f.read()).hexdigest()
    target = uncertainty.ensemble_target()
    if target["model_md5"] != deployed_md5:
        raise SystemExit(f"REFUSING: the ensemble was built for model {target['model_md5']}, but "
                         f"models/best_model.joblib is {deployed_md5}. Rebuild it with "
                         f"code/09_build_bootstrap_ensemble.py. No band was written.")
    print(f"ensemble: {target['n_members']} members built {target['built_at']} for model {deployed_md5}")

    # The members carry production's own imputer, so a NaN in one of the
    # features production imputes (06 OPTIONAL_FEATURES) is scored exactly as
    # production scores it. A NaN in any OTHER feature means 06 would not score
    # the star at all -- such a star gets no band, and any old band is cleared.
    sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))
    from candidate_pool import _optional_features
    required = [c for c in feature_columns if c not in _optional_features()]

    with db.get_conn() as conn:
        cands = [dict(r) for r in conn.execute(
            "SELECT tic_id, host, predicted_probability, characterization_json, "
            "uncertainty_std FROM candidates")]
    print(f"{len(cands)} candidates total")

    have, rows, clear = [], [], []
    for c in cands:
        ch = json.loads(c["characterization_json"] or "{}")
        why = None
        if c["predicted_probability"] is None:
            why = "no probability"
        elif c["tic_id"] not in feats.index:
            why = "no row in unknown_features.csv"
        else:
            row = feats.loc[c["tic_id"]].to_dict()
            row["st_rad"], row["st_teff"] = ch.get("st_rad"), ch.get("st_teff")
            vec = {k: row.get(k) for k in feature_columns}
            missing = [k for k in required if vec[k] is None or (isinstance(vec[k], float) and np.isnan(vec[k]))]
            if missing:
                why = f"NaN in required feature(s) {missing}"
            elif ch.get("scored_model_md5") != deployed_md5:
                why = f"score is not from the deployed model (scored by {ch.get('scored_model_md5') or 'unknown'})"
        if why:
            clear.append((c, why))
            continue
        have.append(c)
        rows.append(vec)
    print(f"{len(have)} get a band; {len(clear)} get none:")
    for c, why in clear:
        print(f"   {c['host']}: {why}" + (" -- clearing its old band" if c["uncertainty_std"] is not None else ""))

    X = pd.DataFrame(rows, columns=feature_columns).astype(float)
    prod = np.array([c["predicted_probability"] for c in have], dtype=float)

    # CORRECTNESS CHECK: re-score the reconstructed vectors with the
    # production model itself. If they don't reproduce the probabilities
    # already stored in the database, the vectors are wrong and any band
    # computed from them would be attached to the wrong star.
    import joblib
    prod_model = joblib.load(os.path.join(ROOT, "models", "best_model.joblib"))
    recomputed = prod_model.predict_proba(X)[:, 1]
    max_err = float(np.max(np.abs(recomputed - prod))) if len(prod) else 0.0
    print(f"reconstruction check: max |recomputed - stored| = {max_err:.2e}")
    if max_err > 1e-6:
        bad = int((np.abs(recomputed - prod) > 1e-6).sum())
        raise SystemExit(
            f"ABORTING: {bad} reconstructed feature vector(s) do not reproduce the stored "
            f"probability (max error {max_err:.4f}). Writing bands from these would attach "
            f"an uncertainty to the wrong data.")
    print(f"scoring {len(X)} candidates against the bootstrap ensemble...")
    recs = uncertainty.predict_with_uncertainty(X, production_probability=prod)

    n = 0
    outside = []
    for c, rec in zip(have, recs):
        if rec is None:
            continue
        db.save_candidate_uncertainty(c["tic_id"], rec, target["model_md5"], target["built_at"])
        n += 1
        p = c["predicted_probability"]
        if not (rec["uncertainty_p16"] <= p <= rec["uncertainty_p84"]):
            outside.append((c["host"], p, rec["uncertainty_p16"], rec["uncertainty_p84"]))
    for c, _ in clear:
        db.clear_candidate_uncertainty(c["tic_id"])
    print(f"wrote bands for {n} candidates; cleared {len(clear)}")

    # The deployed model is not one of the refit members, so its score need not
    # sit inside their spread. Count it rather than let it pass silently; the
    # page says so for each such star.
    print(f"\n{len(outside)} of {n} scores lie outside the 16-84% spread of the refit models")
    for h, p, lo, hi in outside[:10]:
        print(f"   {h}: score {p:.3f}, refit spread {lo:.3f}-{hi:.3f}")


if __name__ == "__main__":
    main()
