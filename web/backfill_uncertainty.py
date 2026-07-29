"""
backfill_uncertainty.py -- computes the bootstrap uncertainty band for every
candidate already in the database, not just ones found from now on.

Same standard as every prior feature in this project: a new evidence type is
applied retroactively to the existing candidate list, so the site never shows
a mix of candidates that have it and candidates that silently don't.

The model's raw feature vectors are NOT stored in the candidates table (only 7
of the 24 columns survive into characterization_json), so they are re-read from
the pipeline's own ranked-candidates CSV, which is the same table the
production score was computed from. Candidates with no row there cannot be
scored and are left null rather than guessed at.
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

    with db.get_conn() as conn:
        cands = [dict(r) for r in conn.execute(
            "SELECT tic_id, host, predicted_probability, characterization_json FROM candidates")]

    # Only candidates that actually carry a displayed probability get a band.
    # A band on an unscored row would describe nothing the user can see.
    scored = [c for c in cands if c["predicted_probability"] is not None]
    print(f"{len(cands)} candidates total; {len(scored)} carry a displayed probability")

    have, rows = [], []
    for c in scored:
        if c["tic_id"] not in feats.index:
            continue
        ch = json.loads(c["characterization_json"])
        row = feats.loc[c["tic_id"]].to_dict()
        row["st_rad"], row["st_teff"] = ch.get("st_rad"), ch.get("st_teff")
        vec = {k: row.get(k) for k in feature_columns}
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vec.values()):
            continue
        have.append(c)
        rows.append(vec)
    print(f"{len(have)} have a complete, reconstructable feature vector "
          f"({len(scored)-len(have)} skipped rather than guessed)")

    X = pd.DataFrame(rows, columns=feature_columns)
    prod = np.array([c["predicted_probability"] for c in have], dtype=float)

    # CORRECTNESS CHECK: re-score the reconstructed vectors with the
    # production model itself. If they don't reproduce the probabilities
    # already stored in the database, the vectors are wrong and any band
    # computed from them would be attached to the wrong star.
    import joblib
    prod_model = joblib.load(os.path.join(ROOT, "models", "best_model.joblib"))
    recomputed = prod_model.predict_proba(X)[:, 1]
    max_err = float(np.max(np.abs(recomputed - prod))) if len(prod) else 0.0
    print(f"reconstruction check: max |recomputed - stored| = {max_err:.6f}")
    if max_err > 1e-6:
        bad = int((np.abs(recomputed - prod) > 1e-6).sum())
        raise SystemExit(
            f"ABORTING: {bad} reconstructed feature vector(s) do not reproduce the stored "
            f"probability (max error {max_err:.4f}). Writing bands from these would attach "
            f"an uncertainty to the wrong data.")
    print(f"scoring {len(X)} candidates against the bootstrap ensemble...")
    recs = uncertainty.predict_with_uncertainty(X, production_probability=prod)

    n = 0
    disagreements = []
    for c, rec in zip(have, recs):
        if rec is None:
            continue
        db.save_candidate_uncertainty(c["tic_id"], rec)
        n += 1
        if rec.get("disagreement_sigma", 0) > 3:
            disagreements.append((c["host"], c["predicted_probability"],
                                  rec["bootstrap_mean"], rec["disagreement_sigma"]))
    print(f"wrote uncertainty for {n} candidates")

    # A large gap between the deployed model's answer and the ensemble's centre
    # means the band should not be read as an interval around the displayed
    # number. Surface it rather than let it pass silently.
    if disagreements:
        print(f"\n{len(disagreements)} candidate(s) where the production probability sits >3 sigma "
              f"from the bootstrap mean -- the band does NOT bracket the displayed score:")
        for h, p, m, s in sorted(disagreements, key=lambda r: -r[3])[:10]:
            print(f"   {h}: production {p:.3f} vs bootstrap mean {m:.3f}  ({s:.1f} sigma)")
    else:
        print("\nNo candidate has the production probability >3 sigma from the bootstrap mean.")


if __name__ == "__main__":
    main()
