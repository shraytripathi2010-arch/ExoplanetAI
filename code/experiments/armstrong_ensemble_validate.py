"""armstrong_ensemble_validate.py -- Armstrong et al. 2022's ensemble members
(RF, Extra Trees, LDA) fresh at the CURRENT 33-feature baseline, plus the
cluster-1-specific question.

PART 0 ROUTING
--------------
  Random Forest   tested ONLY in the original bake-off, at 24 features, before
                  crowding / variability / Gaia RUWE+NSS. STALE -- the CatBoost
                  seed-ensemble precedent showed a family's edge can halve
                  between feature eras, so it gets a fresh test.
  Extra Trees     never mentioned anywhere in RESULTS_SUMMARY.md. UNTESTED.
  LDA             never mentioned anywhere. UNTESTED, and architecturally
                  distinct from everything tried (trees, calibrated trees,
                  dense nets).
  MLP             ALREADY TESTED and strongly negative (mlp_small -0.0644,
                  mlp_med -0.0377, CIs entirely below zero). Deliberately
                  EXCLUDED from the ensemble: including a component whose
                  failure is already understood would bias the result down for
                  a known reason rather than test a new question.

SCALING NOTE
------------
Trees are scale-invariant and get no scaler. LDA is affine-equivariant in
theory, but is given a StandardScaler anyway for numerical conditioning -- so
the comparison cannot be an artefact of one model silently receiving unscaled
heavy-tailed inputs.

CLUSTER-1 CAVEAT, stated up front
---------------------------------
The SOM diagnostic saved only AGGREGATE statistics -- no per-host cluster
labels. So the partition is RE-DERIVED here with the original method (6x6 SOM,
8 agglomerated superclusters, QuantileTransformer, SEED 42) on the ORIGINAL 31
features, because re-deriving on 33 would produce a different partition
entirely. Reproduction is VALIDATED against the recorded cluster-1 profile
(n = 532, 83.46% planet) before any conclusion is drawn from it; if that check
fails, the cluster-1 result is void and is reported as such.

Nothing is promoted.
"""
import os
import sys
import json
import importlib.util
import warnings
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(HERE, "cadence_class_confound.csv")
OUT = os.path.join(HERE, "armstrong_ensemble_validate.json")

N_BOOT = 12
SEED = 20260814
SOM_SEED = 42
GRID, N_SUPER = 6, 8
MDE = 0.0097
GAIA = ["gaia_ruwe", "gaia_nss"]


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def cal(est, scale=False):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", est))
    return CalibratedClassifierCV(Pipeline(steps), cv=5, method="sigmoid")


def members():
    return {
        "hgb":  lambda: cal(HistGradientBoostingClassifier(random_state=42)),
        "rf":   lambda: cal(RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        "et":   lambda: cal(ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        "lda":  lambda: cal(LinearDiscriminantAnalysis(), scale=True),
    }


def rederive_cluster1(df, cols31, m05):
    """Reproduce the SOM partition with the original method, and VALIDATE."""
    spec = importlib.util.spec_from_file_location(
        "somdiag", os.path.join(HERE, "som_cluster_diagnostic.py"))
    sd = importlib.util.module_from_spec(spec); sys.modules["somdiag"] = sd
    spec.loader.exec_module(sd)

    X = df[cols31].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imp = SimpleImputer(strategy="median")
    sc = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             random_state=SOM_SEED)
    Xtr = sc.fit_transform(imp.fit_transform(X))
    som = sd.SOM(GRID, Xtr.shape[1]).fit(Xtr)
    bmu, _ = som.bmu(Xtr)
    sup = AgglomerativeClustering(n_clusters=N_SUPER).fit(som.W).labels_[bmu]

    te = m05.frozen_test_mask(df)
    y = df.label.astype(int).to_numpy()
    print("\n=== RE-DERIVED SOM PARTITION, frozen test composition ===")
    print(f"  {'cluster':<9}{'n':>6}{'planet%':>10}")
    best, bestscore = None, 1e9
    for c in sorted(set(sup[te])):
        m = te & (sup == c)
        n = int(m.sum()); pf = 100.0 * y[m].mean() if n else float("nan")
        print(f"  {c:<9}{n:>6}{pf:>10.2f}")
        s = abs(n - 532) / 532 + abs(pf - 83.46) / 83.46
        if n > 50 and s < bestscore:
            best, bestscore = c, s
    m = te & (sup == best)
    n, pf = int(m.sum()), 100.0 * y[te & (sup == best)].mean()
    ok = abs(n - 532) <= 60 and abs(pf - 83.46) <= 5.0
    print(f"\n  best match to recorded cluster-1 (n=532, 83.46% planet): "
          f"cluster {best}, n={n}, planet%={pf:.2f}")
    print(f"  REPRODUCTION {'VALIDATED' if ok else 'FAILED -- cluster-1 result is VOID'}")
    return (sup == best), ok, {"cluster": int(best), "n": n, "planet_pct": float(pf),
                               "validated": bool(ok)}


def main():
    spec = importlib.util.spec_from_file_location("m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 33
    cols31 = [c for c in cols if c not in GAIA]

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
    print(f"train {len(tr_idx)}  frozen test {int(te.sum())}  2-min {int((is2&te).sum())}")

    c1_mask, c1_ok, c1_info = rederive_cluster1(df, cols31, m05)
    c1_te = c1_mask[te]
    print(f"  cluster-1 stars inside the frozen test: {int(c1_te.sum())}")

    M = members()
    ARMS = list(M.keys()) + ["avg_rf_et_lda", "meta_lr"]
    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(N_BOOT):
        samp = rng.choice(tr_idx, size=len(tr_idx), replace=True)
        Xs, ys = X.iloc[samp], y[samp]
        P = {}
        for name, mk in M.items():
            mo = mk(); mo.fit(Xs, ys)
            P[name] = mo.predict_proba(X[te])[:, 1]
        P["avg_rf_et_lda"] = np.mean([P["rf"], P["et"], P["lda"]], axis=0)
        # meta-learner: LR on out-of-fold member outputs (no leakage)
        oof = np.column_stack([cross_val_predict(M[n_](), Xs, ys, cv=5,
                                                 method="predict_proba")[:, 1]
                               for n_ in ("rf", "et", "lda")])
        lr = LogisticRegression(max_iter=1000).fit(oof, ys)
        P["meta_lr"] = lr.predict_proba(
            np.column_stack([P["rf"], P["et"], P["lda"]]))[:, 1]

        rec = {}
        for a in ARMS:
            p = P[a]
            rec[f"{a}_auc"] = roc_auc_score(y[te], p)
            rec[f"{a}_brier"] = brier_score_loss(y[te], p)
            rec[f"{a}_ece"] = ece(y[te], p)
            rec[f"{a}_auc2"] = roc_auc_score(y[te][is2[te]], p[is2[te]])
            if c1_ok and c1_te.sum() > 50 and len(set(y[te][c1_te])) > 1:
                rec[f"{a}_c1_auc"] = roc_auc_score(y[te][c1_te], p[c1_te])
                rec[f"{a}_c1_ece"] = ece(y[te][c1_te], p[c1_te])
        rows.append(rec)
        print(f"  boot {b+1}/{N_BOOT}  hgb {rec['hgb_auc']:.4f}  rf {rec['rf_auc']:.4f}  "
              f"et {rec['et_auc']:.4f}  lda {rec['lda_auc']:.4f}  "
              f"avg {rec['avg_rf_et_lda_auc']:.4f}", flush=True)

    R = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print("AGGREGATE (frozen test), vs production HGB")
    print(f"{'arm':<16}{'AUC':>9}{'mean d':>10}{'95% CI':>22}{'pos':>7}{'>=MDE':>7}"
          f"{'Brier':>9}{'ECE':>8}")
    out = {"cluster1": c1_info, "mde": MDE, "arms": {}}
    for a in ARMS:
        d = (R[f"{a}_auc"] - R["hgb_auc"]).values
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{a:<16}{R[f'{a}_auc'].mean():>9.4f}{d.mean():>+10.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{f'{(d>0).sum()}/{N_BOOT}':>7}"
              f"{f'{(d>=MDE).sum()}/{N_BOOT}':>7}{R[f'{a}_brier'].mean():>9.4f}"
              f"{R[f'{a}_ece'].mean():>8.4f}")
        out["arms"][a] = {"auc": float(R[f"{a}_auc"].mean()),
                          "mean_delta": float(d.mean()), "ci": [float(lo), float(hi)],
                          "positive": int((d > 0).sum()), "at_mde": int((d >= MDE).sum()),
                          "auc_2min": float(R[f"{a}_auc2"].mean()),
                          "brier": float(R[f"{a}_brier"].mean()),
                          "ece": float(R[f"{a}_ece"].mean()),
                          "clears": bool(lo > 0 and d.mean() >= MDE)}

    if c1_ok and f"hgb_c1_auc" in R.columns:
        print("\n" + "=" * 96)
        print(f"CLUSTER-1 SUBPOPULATION ONLY (n={int(c1_te.sum())} in frozen test) "
              f"-- the targeted question")
        print(f"{'arm':<16}{'c1 AUC':>10}{'d vs hgb':>11}{'95% CI':>22}{'better':>8}"
              f"{'c1 ECE':>10}")
        for a in ARMS:
            d = (R[f"{a}_c1_auc"] - R["hgb_c1_auc"]).values
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"{a:<16}{R[f'{a}_c1_auc'].mean():>10.4f}{d.mean():>+11.4f}"
                  f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{f'{(d>0).sum()}/{N_BOOT}':>8}"
                  f"{R[f'{a}_c1_ece'].mean():>10.4f}")
            out["arms"][a].update({"c1_auc": float(R[f"{a}_c1_auc"].mean()),
                                   "c1_delta": float(d.mean()),
                                   "c1_ci": [float(lo), float(hi)],
                                   "c1_positive": int((d > 0).sum()),
                                   "c1_ece": float(R[f"{a}_c1_ece"].mean())})
    print("\n2-min-only subset:")
    for a in ARMS:
        print(f"  {a:<16} AUC {R[f'{a}_auc2'].mean():.4f}  "
              f"d {(R[f'{a}_auc2']-R['hgb_auc2']).mean():+.4f}")
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
