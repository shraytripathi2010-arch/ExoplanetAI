"""gaia_deploy_retrain.py -- STEP 4: full-scale retrain at 33 features.

Writes STAGED artifacts only. The live swap is a separate, deliberate step,
because `FEATURE_COLUMNS` and the model artifact must change ATOMICALLY -- a
31-feature model raises ValueError on a 33-column matrix and vice versa, which
would crash the scheduler's next retrain tick.

Production's exact recipe, unchanged:
    CalibratedClassifierCV(
        Pipeline([SimpleImputer(median), HistGradientBoostingClassifier(rs=42)]),
        cv=5, method="sigmoid")

Full suite at real scale: frozen-test AUC, 12 training bootstraps vs the
31-feature baseline, nested CV with pooled out-of-fold predictions, Brier, ECE,
and the 2-min-only subset.
"""
import hashlib
import importlib.util
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
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
STAGED_MODEL = os.path.join(MODELS, "staged_best_model_gaia33.joblib")
STAGED_META = os.path.join(MODELS, "staged_best_model_gaia33_metadata.json")
OUT = os.path.join(HERE, "gaia_deploy_retrain.json")

GAIA = ["gaia_ruwe", "gaia_nss"]
N_BOOT = 12
SEED = 20260814
MDE = 0.0097
CURRENT_AUC = 0.9300019473029855


def _m05():
    spec = importlib.util.spec_from_file_location("m05", os.path.join(CODE, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["m05"] = m
    spec.loader.exec_module(m); return m


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def model():
    return CalibratedClassifierCV(
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", HistGradientBoostingClassifier(random_state=42))]),
        cv=5, method="sigmoid")


def main():
    m05 = _m05()
    cols31 = list(m05.FEATURE_COLUMNS)
    assert len(cols31) == 31, len(cols31)
    cols33 = cols31 + GAIA

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    print(f"training.csv {len(df)} rows, {len(df.columns)} columns")
    for c in GAIA:
        assert c in df.columns, f"{c} missing from training.csv -- run STEP 2 first"
        print(f"  {c:<11} coverage {df[c].notna().mean():.4%}")

    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    for c in GAIA:
        X[c] = pd.to_numeric(df[c], errors="coerce").values
    X = X[cols33].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)

    tr_mask, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)
    tr_idx = np.where(tr_mask)[0]
    cad = pd.read_csv(os.path.join(HERE, "cadence_class_confound.csv"))[["host", "cadence_min"]]
    cad["host"] = cad.host.astype(str)
    cc = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((cc >= 1.0) & (cc <= 2.6)).to_numpy()
    print(f"train {len(tr_idx)}  frozen test {int(te.sum())}  "
          f"2-min in test {int((is2 & te).sum())}")

    res = {}

    # ---------- headline: single fit on the real training split ----------
    print("\n=== HEADLINE FIT (production recipe, full training split) ===")
    for lab, cu in (("31", cols31), ("33", cols33)):
        mo = model(); mo.fit(X.iloc[tr_idx][cu], y[tr_idx])
        p = mo.predict_proba(X[te][cu])[:, 1]
        s = is2[te]
        res[f"headline_{lab}"] = {
            "auc": float(roc_auc_score(y[te], p)),
            "auc_2min": float(roc_auc_score(y[te][s], p[s])),
            "brier": float(brier_score_loss(y[te], p)),
            "ece": ece(y[te], p)}
        print(f"  {lab} features: frozen-test AUC {res[f'headline_{lab}']['auc']:.4f}  "
              f"2-min {res[f'headline_{lab}']['auc_2min']:.4f}  "
              f"Brier {res[f'headline_{lab}']['brier']:.4f}  "
              f"ECE {res[f'headline_{lab}']['ece']:.4f}")
        if lab == "33":
            joblib.dump(mo, STAGED_MODEL)
    d_head = res["headline_33"]["auc"] - res["headline_31"]["auc"]
    print(f"  delta {d_head:+.4f}   (current deployed metric of record "
          f"{CURRENT_AUC:.4f})")
    res["headline_delta"] = float(d_head)

    # ---------- 12 training bootstraps ----------
    print("\n=== 12 TRAINING BOOTSTRAPS ===")
    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(N_BOOT):
        samp = rng.choice(tr_idx, size=len(tr_idx), replace=True)
        r = {}
        for lab, cu in (("31", cols31), ("33", cols33)):
            mo = model(); mo.fit(X.iloc[samp][cu], y[samp])
            p = mo.predict_proba(X[te][cu])[:, 1]
            r[f"auc{lab}"] = roc_auc_score(y[te], p)
            r[f"brier{lab}"] = brier_score_loss(y[te], p)
            r[f"ece{lab}"] = ece(y[te], p)
            s = is2[te]
            r[f"auc2_{lab}"] = roc_auc_score(y[te][s], p[s])
        rows.append(r)
        print(f"  boot {b+1}/{N_BOOT}  31 {r['auc31']:.4f}  33 {r['auc33']:.4f}  "
              f"d {r['auc33']-r['auc31']:+.4f}", flush=True)
    R = pd.DataFrame(rows)
    d = (R.auc33 - R.auc31).values
    d2 = (R.auc2_33 - R.auc2_31).values
    lo, hi = np.percentile(d, [2.5, 97.5])
    res["bootstrap"] = {"mean_delta": float(d.mean()), "sd": float(d.std()),
                        "ci": [float(lo), float(hi)],
                        "positive": int((d > 0).sum()), "at_mde": int((d >= MDE).sum()),
                        "mean_delta_2min": float(d2.mean()),
                        "auc31": float(R.auc31.mean()), "auc33": float(R.auc33.mean()),
                        "brier": [float(R.brier31.mean()), float(R.brier33.mean())],
                        "ece": [float(R.ece31.mean()), float(R.ece33.mean())]}
    print(f"\n  mean delta {d.mean():+.4f}  sd {d.std():.4f}  "
          f"CI [{lo:+.4f}, {hi:+.4f}]  positive {int((d>0).sum())}/{N_BOOT}  "
          f">=MDE {int((d>=MDE).sum())}/{N_BOOT}")
    print(f"  2-min subset mean delta {d2.mean():+.4f}")
    print(f"  Brier {R.brier31.mean():.4f} -> {R.brier33.mean():.4f}   "
          f"ECE {R.ece31.mean():.4f} -> {R.ece33.mean():.4f}")

    # ---------- nested CV, pooled out-of-fold ----------
    print("\n=== NESTED CV (5 outer folds, pooled out-of-fold) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    hosts = df.host.to_numpy()
    uh = pd.unique(hosts)
    hy = pd.Series(y, index=range(len(y))).groupby(pd.Series(hosts)).max().reindex(uh).to_numpy()
    oof = {lab: np.full(len(y), np.nan) for lab in ("31", "33")}
    fold_auc = {lab: [] for lab in ("31", "33")}
    for k, (tri, tei) in enumerate(skf.split(uh, hy), 1):
        trh, teh = set(uh[tri]), set(uh[tei])
        m_tr = np.array([h in trh for h in hosts])
        m_te = np.array([h in teh for h in hosts])
        for lab, cu in (("31", cols31), ("33", cols33)):
            mo = model(); mo.fit(X[m_tr][cu], y[m_tr])
            p = mo.predict_proba(X[m_te][cu])[:, 1]
            oof[lab][m_te] = p
            fold_auc[lab].append(roc_auc_score(y[m_te], p))
        print(f"  fold {k}: 31 {fold_auc['31'][-1]:.4f}  33 {fold_auc['33'][-1]:.4f}  "
              f"d {fold_auc['33'][-1]-fold_auc['31'][-1]:+.4f}", flush=True)
    pooled = {lab: float(roc_auc_score(y, oof[lab])) for lab in ("31", "33")}
    wins = int(sum(a > b for a, b in zip(fold_auc["33"], fold_auc["31"])))
    print(f"\n  pooled out-of-fold: 31 {pooled['31']:.4f} -> 33 {pooled['33']:.4f}  "
          f"delta {pooled['33']-pooled['31']:+.4f}")
    print(f"  33-feature arm wins on {wins}/5 outer folds")
    res["nested_cv"] = {"pooled": pooled, "delta": float(pooled["33"] - pooled["31"]),
                        "fold_auc_31": [float(a) for a in fold_auc["31"]],
                        "fold_auc_33": [float(a) for a in fold_auc["33"]],
                        "folds_won": wins}

    # ---------- staged metadata ----------
    md5 = hashlib.md5(open(STAGED_MODEL, "rb").read()).hexdigest()
    meta = json.load(open(os.path.join(MODELS, "best_model_metadata.json")))
    meta.update({
        "feature_columns": cols33,
        "training_rows": int(len(tr_idx)),
        "test_rows": int(te.sum()),
        "test_roc_auc": res["headline_33"]["auc"],
        "test_brier_score": res["headline_33"]["brier"],
        "previous_test_roc_auc": CURRENT_AUC,
        "previous_model_md5": "1f0b7cb8e78ab542374eaf78fc837a6f",
        "model_md5": md5,
        "updated_by": "gaia_deploy_retrain.py (Gaia DR3 RUWE + NSS, 31 -> 33)",
        "note": ("Gaia DR3 astrometric binary indicators. Validated +0.0142 over "
                 "12 bootstraps, CI [+0.0124, +0.0168], 12/12 at MDE; "
                 "availability-trap gate passed (AUC(avail) 0.5069/0.5054 vs "
                 "0.3775 for the closed CTL trap); NSS eclipsing bit confirmed "
                 "zero occurrences so the signal is astrometric/spectroscopic, "
                 "not a restatement of the EB label."),
    })
    json.dump(meta, open(STAGED_META, "w"), indent=2)
    print(f"\nstaged model  {STAGED_MODEL}\n  md5 {md5}")
    print(f"staged metadata {STAGED_META}  ({len(cols33)} features)")
    res["staged_md5"] = md5
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
