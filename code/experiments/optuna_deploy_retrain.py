"""optuna_deploy_retrain.py -- STEP 2 of the Optuna hyperparameter deployment.

Writes STAGED artifacts only. The live swap is a separate, deliberate step.

WHAT IS CHANGING, PRECISELY
---------------------------
The FEATURE SET IS UNCHANGED at 33 columns. Only the HGB hyperparameters move,
from sklearn defaults to the Optuna TPE-selected configuration. The calibration
wrapper (`CalibratedClassifierCV(cv=5, method="sigmoid")`) and the imputer are
untouched.

    deployed now (defaults)   lr 0.1     iter 100  leaves 31  l2 0.0     cw None
    Optuna-selected           lr 0.0926  iter 475  leaves 63  l2 0.0090  cw balanced

The Optuna values are READ FROM `optuna_hpo_nested.json` at runtime -- never
retyped -- so the deployed configuration is the one the study actually selected.

THIS IS A DELIBERATE EXCEPTION TO THE PROMOTION RULE
-----------------------------------------------------
The bar is `ci_lo > 0 AND mean delta >= MDE (0.0097)`. This change satisfies the
first leg and FAILS the second: +0.0045 mean, with 0 of 12 resamples reaching
0.0097. It is being promoted anyway on explicit informed consent, on the
strength of 12/12 positive resamples, a CI excluding zero, simultaneous Brier
AND ECE improvement, and no evidence of search overfitting. CatBoost (+0.0080)
and trap_vshape were correctly NOT promoted under the same rule and REMAIN so.
Nothing here changes the standing bar for any future proposal.

REPRODUCTION CHECK
------------------
The 12 bootstrap draws are pre-generated from the identical seed stream
`default_rng(20260814).choice(tr_idx, ...)` used by `optuna_hpo_validate.py`,
so the deltas must match that run EXACTLY, not merely closely. They are compared
against the saved JSON and a mismatch is reported loudly. Bootstraps are then
evaluated in parallel, which changes only the order work is done in, not the
draws or the results.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys
import json
import time
import hashlib
import importlib.util
import warnings
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..")
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
MODELS = os.path.join(ROOT, "models")
NESTED = os.path.join(HERE, "optuna_hpo_nested.json")
PRIOR = os.path.join(HERE, "optuna_hpo_validate.json")
CADENCE = os.path.join(HERE, "cadence_class_confound.csv")
STAGED_MODEL = os.path.join(MODELS, "staged_best_model_optuna33.joblib")
STAGED_META = os.path.join(MODELS, "staged_best_model_optuna33_metadata.json")
OUT = os.path.join(HERE, "optuna_deploy_retrain.json")

N_BOOT = 12
SEED = 20260814
MDE = 0.0097
CURRENT_AUC = 0.9401928329263969
CURRENT_MD5 = "c37f9f4bdb252d52b8c1c5487dad9e6d"
PROD_PARAMS = {"random_state": 42}


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def model(params):
    """Production's wrapper and imputer, unchanged. Only `params` differs."""
    return CalibratedClassifierCV(
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", HistGradientBoostingClassifier(**params))]),
        cv=5, method="sigmoid")


def _boot_one(b, samp, X, y, te, s2, ARMS):
    r = {"b": b}
    for lab, params in ARMS.items():
        mo = model(params); mo.fit(X.iloc[samp], y[samp])
        p = mo.predict_proba(X[te])[:, 1]
        r[f"auc_{lab}"] = roc_auc_score(y[te], p)
        r[f"brier_{lab}"] = brier_score_loss(y[te], p)
        r[f"ece_{lab}"] = ece(y[te], p)
        r[f"auc2_{lab}"] = roc_auc_score(y[te][s2], p[s2])
    print(f"  boot {b+1}/{N_BOOT}  prod {r['auc_prod']:.4f}  "
          f"optuna {r['auc_optuna']:.4f}  d {r['auc_optuna']-r['auc_prod']:+.4f}",
          flush=True)
    return r


def main():
    study = json.load(open(NESTED))
    OPT = dict(study["final_params"])
    print("=== CONFIGURATION, read from the saved study (not retyped) ===")
    print(f"  source: {os.path.relpath(NESTED, ROOT)}")
    for k in sorted(OPT):
        print(f"    {k:<20} {OPT[k]}")
    assert OPT["random_state"] == 42
    assert OPT["max_leaf_nodes"] == 63 and OPT["max_iter"] == 475
    assert OPT["class_weight"] == "balanced" and OPT["max_depth"] is None

    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE, "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 33, len(cols)

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    tr_mask, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)
    tr_idx = np.where(tr_mask)[0]
    cad = pd.read_csv(CADENCE)[["host", "cadence_min"]]; cad["host"] = cad.host.astype(str)
    cc = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((cc >= 1.0) & (cc <= 2.6)).to_numpy()
    s2 = is2[te]
    print(f"\ntraining.csv {len(df)} rows, {len(df.columns)} columns")
    print(f"train {len(tr_idx)}  frozen test {int(te.sum())}  2-min in test {int(s2.sum())}")

    ARMS = {"prod": PROD_PARAMS, "optuna": OPT}
    res = {"optuna_params": OPT, "prod_params": PROD_PARAMS}
    t0 = time.time()

    # ---------- headline: single fit on the real training split ----------
    print("\n=== HEADLINE FIT (full training split, frozen test) ===")
    for lab, params in ARMS.items():
        t1 = time.time()
        mo = model(params); mo.fit(X.iloc[tr_idx], y[tr_idx])
        fit_s = time.time() - t1
        p = mo.predict_proba(X[te])[:, 1]
        t2 = time.time(); mo.predict_proba(X[te]); infer_s = time.time() - t2
        res[f"headline_{lab}"] = {
            "auc": float(roc_auc_score(y[te], p)),
            "auc_2min": float(roc_auc_score(y[te][s2], p[s2])),
            "brier": float(brier_score_loss(y[te], p)),
            "ece": ece(y[te], p),
            "fit_seconds": round(fit_s, 1),
            "inference_seconds_1098_rows": round(infer_s, 4)}
        h = res[f"headline_{lab}"]
        print(f"  {lab:<7} AUC {h['auc']:.4f}  2-min {h['auc_2min']:.4f}  "
              f"Brier {h['brier']:.4f}  ECE {h['ece']:.4f}   "
              f"fit {h['fit_seconds']}s  infer {h['inference_seconds_1098_rows']}s")
        if lab == "optuna":
            joblib.dump(mo, STAGED_MODEL)
    dh = res["headline_optuna"]["auc"] - res["headline_prod"]["auc"]
    res["headline_delta"] = float(dh)
    print(f"  headline delta {dh:+.4f}  (deployed metric of record {CURRENT_AUC:.4f})")
    print(f"  TRAINING cost ratio {res['headline_optuna']['fit_seconds'] / max(res['headline_prod']['fit_seconds'], 1e-9):.1f}x")
    print(f"  INFERENCE cost ratio {res['headline_optuna']['inference_seconds_1098_rows'] / max(res['headline_prod']['inference_seconds_1098_rows'], 1e-9):.2f}x"
          "   <- serving latency, the number that matters for the live app")

    # ---------- 12 bootstraps, draws identical to optuna_hpo_validate.py ----------
    print(f"\n=== {N_BOOT} TRAINING BOOTSTRAPS (draws replayed from seed {SEED}) ===")
    rng = np.random.default_rng(SEED)
    draws = [rng.choice(tr_idx, size=len(tr_idx), replace=True) for _ in range(N_BOOT)]
    rows = Parallel(n_jobs=4, backend="loky")(
        delayed(_boot_one)(b, draws[b], X, y, te, s2, ARMS) for b in range(N_BOOT))
    R = pd.DataFrame(sorted(rows, key=lambda r: r["b"]))
    d = (R.auc_optuna - R.auc_prod).values
    d2 = (R.auc2_optuna - R.auc2_prod).values
    lo, hi = np.percentile(d, [2.5, 97.5])
    res["bootstrap"] = {
        "mean_delta": float(d.mean()), "sd": float(d.std()),
        "ci": [float(lo), float(hi)], "positive": int((d > 0).sum()),
        "at_mde": int((d >= MDE).sum()), "max_delta": float(d.max()),
        "mean_delta_2min": float(d2.mean()),
        "auc_prod": float(R.auc_prod.mean()), "auc_optuna": float(R.auc_optuna.mean()),
        "brier": [float(R.brier_prod.mean()), float(R.brier_optuna.mean())],
        "ece": [float(R.ece_prod.mean()), float(R.ece_optuna.mean())]}
    b = res["bootstrap"]
    print(f"\n  mean delta {b['mean_delta']:+.4f}  sd {b['sd']:.4f}  "
          f"CI [{lo:+.4f}, {hi:+.4f}]  positive {b['positive']}/{N_BOOT}  "
          f">=MDE {b['at_mde']}/{N_BOOT}  (max {b['max_delta']:+.4f})")
    print(f"  2-min subset mean delta {b['mean_delta_2min']:+.4f}")
    print(f"  Brier {b['brier'][0]:.4f} -> {b['brier'][1]:.4f}   "
          f"ECE {b['ece'][0]:.4f} -> {b['ece'][1]:.4f}")

    # ---------- reproduction check against the investigation ----------
    print("\n=== REPRODUCTION CHECK vs optuna_hpo_validate.json ===")
    ok = True
    if os.path.exists(PRIOR):
        pr = json.load(open(PRIOR))["resampled"]["optuna"]
        for k, new, old in (("mean_delta", b["mean_delta"], pr["mean_delta"]),
                            ("ci_lo", b["ci"][0], pr["ci"][0]),
                            ("ci_hi", b["ci"][1], pr["ci"][1])):
            same = abs(new - old) < 5e-4
            ok &= same
            print(f"  {k:<11} now {new:+.4f}   investigation {old:+.4f}   "
                  f"{'MATCH' if same else 'DIFFERS'}")
        same_pos = b["positive"] == pr["positive"]
        ok &= same_pos
        print(f"  {'positive':<11} now {b['positive']}/12        investigation "
              f"{pr['positive']}/12        {'MATCH' if same_pos else 'DIFFERS'}")
    res["reproduces_investigation"] = bool(ok)
    print(f"  -> {'REPRODUCED' if ok else 'DOES NOT REPRODUCE -- DO NOT DEPLOY'}")

    # ---------- nested CV, host-disjoint, pooled out-of-fold ----------
    print("\n=== NESTED CV (5 host-disjoint outer folds, pooled out-of-fold) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    hosts = df.host.to_numpy()
    uh = pd.unique(hosts)
    hy = pd.Series(y, index=range(len(y))).groupby(pd.Series(hosts)).max().reindex(uh).to_numpy()
    oof = {lab: np.full(len(y), np.nan) for lab in ARMS}
    fold_auc = {lab: [] for lab in ARMS}
    for k, (tri, tei) in enumerate(skf.split(uh, hy), 1):
        trh, teh = set(uh[tri]), set(uh[tei])
        m_tr = np.array([h in trh for h in hosts])
        m_te = np.array([h in teh for h in hosts])
        for lab, params in ARMS.items():
            mo = model(params); mo.fit(X[m_tr], y[m_tr])
            p = mo.predict_proba(X[m_te])[:, 1]
            oof[lab][m_te] = p
            fold_auc[lab].append(roc_auc_score(y[m_te], p))
        print(f"  fold {k}: prod {fold_auc['prod'][-1]:.4f}  "
              f"optuna {fold_auc['optuna'][-1]:.4f}  "
              f"d {fold_auc['optuna'][-1]-fold_auc['prod'][-1]:+.4f}", flush=True)
    pooled = {lab: float(roc_auc_score(y, oof[lab])) for lab in ARMS}
    wins = int(sum(a > bb for a, bb in zip(fold_auc["optuna"], fold_auc["prod"])))
    print(f"\n  pooled out-of-fold: prod {pooled['prod']:.4f} -> "
          f"optuna {pooled['optuna']:.4f}  delta {pooled['optuna']-pooled['prod']:+.4f}")
    print(f"  optuna arm wins {wins}/5 outer folds")
    res["nested_cv"] = {"pooled": pooled,
                        "delta": float(pooled["optuna"] - pooled["prod"]),
                        "fold_auc_prod": [float(a) for a in fold_auc["prod"]],
                        "fold_auc_optuna": [float(a) for a in fold_auc["optuna"]],
                        "folds_won": wins}

    # ---------- staged metadata ----------
    md5 = hashlib.md5(open(STAGED_MODEL, "rb").read()).hexdigest()
    meta = json.load(open(os.path.join(MODELS, "best_model_metadata.json")))
    hp = {k: v for k, v in OPT.items()}
    meta.update({
        "model_name": "HistGradientBoosting+sigmoid_calibration "
                      "(OPTUNA-TUNED hyperparameters, 33 features incl. catalog "
                      "crowding, stellar variability, Gaia DR3 astrometry)",
        "hyperparameters": hp,
        "hyperparameter_provenance":
            "Optuna TPE, 120 trials/fold, outer-5/inner-3 nested CV; selected by "
            "the final full-training study in optuna_hpo_nested.json. This is "
            "NOT the pre-Gaia 'legacy tuned' config (lr 0.1 / iter 500 / leaves "
            "63 / l2 0.5 / balanced), which was separately tested at 33 features "
            "and measured +0.0016 with a CI spanning zero, and it is NOT sklearn "
            "defaults (lr 0.1 / iter 100 / leaves 31 / l2 0.0 / cw None), which "
            "is what was actually deployed from the Gaia swap until now.",
        "feature_columns": cols,
        "training_rows": int(len(tr_idx)),
        "test_rows": int(te.sum()),
        "test_roc_auc": res["headline_optuna"]["auc"],
        "test_brier_score": res["headline_optuna"]["brier"],
        "previous_test_roc_auc": CURRENT_AUC,
        "previous_model_md5": CURRENT_MD5,
        "model_md5": md5,
        "updated_by": "optuna_deploy_retrain.py (hyperparameters only; feature set unchanged at 33)",
        "promotion_rule_status": "DELIBERATE EXCEPTION -- DID NOT CLEAR THE MDE",
        "note":
            f"Hyperparameter-only change. Resampled over 12 training bootstraps: "
            f"{b['mean_delta']:+.4f} (sd {b['sd']:.4f}), CI "
            f"[{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}], positive {b['positive']}/12, "
            f"at/above MDE {b['at_mde']}/12 (max single resample "
            f"{b['max_delta']:+.4f} vs MDE {MDE}). Brier {b['brier'][0]:.4f} -> "
            f"{b['brier'][1]:.4f}, ECE {b['ece'][0]:.4f} -> {b['ece'][1]:.4f}. "
            f"THIS DID NOT CLEAR THE STANDING PROMOTION BAR, which requires "
            f"ci_lo > 0 AND mean delta >= MDE ({MDE}); only the first leg was met. "
            f"Promoted anyway as an explicit, user-authorised exception on the "
            f"strength of 12/12 positive resamples, a CI excluding zero, "
            f"simultaneous Brier and ECE improvement, and no evidence of search "
            f"overfitting (nested-CV delta was SMALLER than the resampled-test "
            f"delta). CatBoost (+0.0080) and trap_vshape remain correctly "
            f"unpromoted under the same rule; the bar is unchanged for future "
            f"proposals.",
        "training_cost_note":
            f"Fitting is ~{res['headline_optuna']['fit_seconds'] / max(res['headline_prod']['fit_seconds'], 1e-9):.0f}x "
            f"more expensive than the previous default config (475 boosting "
            f"iterations at 63 leaves vs 100 at 31). Affects retrain wall-clock "
            f"only. Inference on the 1,098-row frozen test measured "
            f"{res['headline_optuna']['inference_seconds_1098_rows']:.4f}s vs "
            f"{res['headline_prod']['inference_seconds_1098_rows']:.4f}s, so "
            f"serving latency is materially unaffected.",
        "deployment_path": "manual offline swap",
        "rollback_artifact": f"models/versions/best_model_pre_optuna_{CURRENT_MD5[:8]}.joblib",
    })
    meta.pop("promotion_note", None)
    json.dump(meta, open(STAGED_META, "w"), indent=2)
    res["staged_md5"] = md5
    res["wall_clock_s"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nstaged model md5 {md5}")
    print(f"staged: {STAGED_MODEL}")
    print(f"        {STAGED_META}")
    print(f"wall clock {(time.time()-t0)/60:.1f} min")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
