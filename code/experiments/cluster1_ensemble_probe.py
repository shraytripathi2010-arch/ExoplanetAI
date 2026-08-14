"""cluster1_ensemble_probe.py -- PART 3: does model diversity help on the KNOWN
weak subpopulation, even where it does not move the aggregate?

CORRECTING A MISTAKE IN THE FIRST PASS
--------------------------------------
`armstrong_ensemble_validate.py` validated its re-derived partition against
`som_cluster1_profile.json`'s **n = 532, 83.46% planet** and declared the
reproduction failed. That reference is over the FULL labelled set (5,486 rows).
The weak subpopulation this question is actually about is cluster 1 **on the
frozen test**, recorded in `som_cluster_diagnostic.json` as:

    n = 108,  planet 76.85%,  AUC 0.8280,  ECE 0.0934

The re-derived partition DOES contain that group -- it is cluster 0 there
(n = 109, planet 77.06%). The reproduction succeeded; the check targeted the
wrong numbers.

VALIDATION USED HERE is therefore stricter and behavioural, not just
compositional: the selected cluster must match on size AND planet fraction AND
reproduce production HGB's recorded AUC ~0.828 on it. If HGB does not score
~0.83 on the chosen group, it is not the same subpopulation and the result is
declared void rather than reported.
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
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT = os.path.join(HERE, "cluster1_ensemble_probe.json")

N_BOOT = 12
SEED = 20260814
SOM_SEED, GRID, N_SUPER = 42, 6, 8
GAIA = ["gaia_ruwe", "gaia_nss"]
REC = {"n": 108, "planet_pct": 76.85, "auc": 0.8280, "ece": 0.0934}


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


MK = {"hgb": lambda: cal(HistGradientBoostingClassifier(random_state=42)),
      "rf":  lambda: cal(RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
      "et":  lambda: cal(ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
      "lda": lambda: cal(LinearDiscriminantAnalysis(), scale=True)}


def main():
    import joblib
    spec = importlib.util.spec_from_file_location("m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols = list(m05.FEATURE_COLUMNS)
    cols31 = [c for c in cols if c not in GAIA]

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    te = m05.frozen_test_mask(df)
    tr_idx = np.where(m05.split_by_host(df)[0])[0]

    # ---- re-derive the partition ----
    sspec = importlib.util.spec_from_file_location(
        "somdiag", os.path.join(HERE, "som_cluster_diagnostic.py"))
    sd = importlib.util.module_from_spec(sspec); sys.modules["somdiag"] = sd
    sspec.loader.exec_module(sd)
    Xs = df[cols31].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Xtr = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                              random_state=SOM_SEED).fit_transform(
        SimpleImputer(strategy="median").fit_transform(Xs))
    som = sd.SOM(GRID, Xtr.shape[1]).fit(Xtr)
    bmu, _ = som.bmu(Xtr)
    sup = AgglomerativeClustering(n_clusters=N_SUPER).fit(som.W).labels_[bmu]

    # ---- pick the cluster matching the RECORDED frozen-test profile ----
    prod = joblib.load(os.path.join(ROOT, "models", "best_model.joblib"))
    prob = prod.predict_proba(X)[:, 1]
    print("=== candidate clusters, frozen test, scored by DEPLOYED model ===")
    print(f"  target: n~{REC['n']}, planet~{REC['planet_pct']}%, HGB AUC~{REC['auc']}")
    print(f"  {'cl':<4}{'n':>6}{'planet%':>10}{'HGB AUC':>10}{'HGB ECE':>10}")
    best, bs = None, 1e9
    for c in sorted(set(sup[te])):
        m = te & (sup == c)
        n = int(m.sum())
        if n < 40 or len(set(y[m])) < 2:
            continue
        pf = 100.0 * y[m].mean(); a = roc_auc_score(y[m], prob[m]); e = ece(y[m], prob[m])
        print(f"  {c:<4}{n:>6}{pf:>10.2f}{a:>10.4f}{e:>10.4f}")
        s = abs(n - REC["n"]) / REC["n"] + abs(pf - REC["planet_pct"]) / REC["planet_pct"] \
            + abs(a - REC["auc"]) / REC["auc"]
        if s < bs:
            best, bs = c, s
    m = te & (sup == best)
    n = int(m.sum()); pf = 100.0 * y[m].mean()
    a = roc_auc_score(y[m], prob[m]); e = ece(y[m], prob[m])
    ok = abs(n - REC["n"]) <= 25 and abs(pf - REC["planet_pct"]) <= 6 and abs(a - REC["auc"]) <= 0.06
    print(f"\n  selected cluster {best}: n={n}, planet={pf:.2f}%, HGB AUC={a:.4f}, ECE={e:.4f}")
    print(f"  behavioural validation vs recorded (n=108, 76.85%, AUC 0.828): "
          f"{'MATCH' if ok else 'NO MATCH -- result VOID'}")
    if not ok:
        json.dump({"validated": False}, open(OUT, "w"), indent=2)
        return

    c1 = (sup == best)[te]
    yte = y[te]
    ARMS = ["hgb", "rf", "et", "lda", "avg_rf_et_lda", "meta_lr"]
    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(N_BOOT):
        samp = rng.choice(tr_idx, size=len(tr_idx), replace=True)
        Xb, yb = X.iloc[samp], y[samp]
        P = {}
        for k, mk in MK.items():
            mo = mk(); mo.fit(Xb, yb)
            P[k] = mo.predict_proba(X[te])[:, 1]
        P["avg_rf_et_lda"] = np.mean([P["rf"], P["et"], P["lda"]], axis=0)
        oof = np.column_stack([cross_val_predict(MK[k](), Xb, yb, cv=5,
                                                 method="predict_proba")[:, 1]
                               for k in ("rf", "et", "lda")])
        lr = LogisticRegression(max_iter=1000).fit(oof, yb)
        P["meta_lr"] = lr.predict_proba(np.column_stack([P["rf"], P["et"], P["lda"]]))[:, 1]
        rec = {}
        for k in ARMS:
            rec[f"{k}_auc"] = roc_auc_score(yte[c1], P[k][c1])
            rec[f"{k}_ece"] = ece(yte[c1], P[k][c1])
        rows.append(rec)
        print(f"  boot {b+1}/{N_BOOT}  hgb {rec['hgb_auc']:.4f}  rf {rec['rf_auc']:.4f}  "
              f"et {rec['et_auc']:.4f}  avg {rec['avg_rf_et_lda_auc']:.4f}", flush=True)

    R = pd.DataFrame(rows)
    print("\n" + "=" * 86)
    print(f"CLUSTER-1 ONLY (n={int(c1.sum())}), vs production HGB -- the targeted question")
    print(f"{'arm':<16}{'c1 AUC':>10}{'d vs hgb':>11}{'95% CI':>22}{'better':>9}{'c1 ECE':>10}")
    out = {"validated": True, "cluster": int(best), "n": n, "planet_pct": float(pf),
           "hgb_auc_check": float(a), "arms": {}}
    for k in ARMS:
        d = (R[f"{k}_auc"] - R["hgb_auc"]).values
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{k:<16}{R[f'{k}_auc'].mean():>10.4f}{d.mean():>+11.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{f'{(d>0).sum()}/{N_BOOT}':>9}"
              f"{R[f'{k}_ece'].mean():>10.4f}")
        out["arms"][k] = {"c1_auc": float(R[f"{k}_auc"].mean()),
                          "delta": float(d.mean()), "ci": [float(lo), float(hi)],
                          "better": int((d > 0).sum()), "c1_ece": float(R[f"{k}_ece"].mean())}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
