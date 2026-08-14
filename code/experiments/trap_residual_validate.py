"""trap_residual_validate.py -- 33 -> 34/35 with the trapezoid fit residual.

Run even though `trap_rmse` already fails two of this project's standing bars
(|rho| 0.962 vs `depth_mean` against a 0.80 threshold, and a |gal b| control arm
swinging 0.816 -> 0.396), because it PASSES the class-rate gate and a measured
null is a stronger closure than an argued one.

Arms, including the combined-with-trap_vshape arm that settles whether the
residual and the shape are complements or near-duplicates:

    base            the deployed 33
    +rmse           34
    +bic            34
    +rmse,bic       35
    +rmse,vshape    35   (trap_vshape is NOT deployed; this is the complement test)

Production's exact recipe, frozen split, 12 training bootstraps, full frozen
test and the 2-min-only subset, Brier and ECE. MDE ~0.0097. Nothing promoted.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
TRAP = os.path.join(HERE, "trapezoid_shape_features.csv")
CADENCE = os.path.join(HERE, "cadence_class_confound.csv")
OUT = os.path.join(HERE, "trap_residual_validate.json")

N_BOOT = 12
SEED = 20260814
MDE = 0.0097
K_PARAMS = 5


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
    spec = importlib.util.spec_from_file_location("m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 33, len(cols)

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    t = pd.read_csv(TRAP); t["host"] = t.host.astype(str)
    r = pd.to_numeric(t.trap_rmse, errors="coerce")
    n = pd.to_numeric(t.trap_nbins, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        t["trap_bic"] = n * np.log((n * r ** 2) / n) + K_PARAMS * np.log(n)
    t.loc[~np.isfinite(t.trap_bic), "trap_bic"] = np.nan
    df = df.merge(t[["host", "trap_rmse", "trap_bic", "trap_vshape"]], on="host", how="left")

    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    for c in ("trap_rmse", "trap_bic", "trap_vshape"):
        X[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).values

    tr_mask, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)
    tr_idx = np.where(tr_mask)[0]
    cad = pd.read_csv(CADENCE)[["host", "cadence_min"]]; cad["host"] = cad.host.astype(str)
    cc = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((cc >= 1.0) & (cc <= 2.6)).to_numpy()
    print(f"train {len(tr_idx)}  frozen test {int(te.sum())}  2-min in test {int((is2&te).sum())}")

    ARMS = {"base": cols,
            "+rmse": cols + ["trap_rmse"],
            "+bic": cols + ["trap_bic"],
            "+rmse,bic": cols + ["trap_rmse", "trap_bic"],
            "+rmse,vshape": cols + ["trap_rmse", "trap_vshape"]}

    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(N_BOOT):
        samp = rng.choice(tr_idx, size=len(tr_idx), replace=True)
        rec = {}
        for lab, cu in ARMS.items():
            mo = model(); mo.fit(X.iloc[samp][cu], y[samp])
            p = mo.predict_proba(X[te][cu])[:, 1]
            rec[f"{lab}_auc"] = roc_auc_score(y[te], p)
            rec[f"{lab}_brier"] = brier_score_loss(y[te], p)
            rec[f"{lab}_ece"] = ece(y[te], p)
            s = is2[te]
            rec[f"{lab}_auc2"] = roc_auc_score(y[te][s], p[s])
        rows.append(rec)
        print(f"  boot {b+1}/{N_BOOT}  base {rec['base_auc']:.4f}  "
              f"+rmse {rec['+rmse_auc']:.4f}  d {rec['+rmse_auc']-rec['base_auc']:+.4f}",
              flush=True)

    R = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print(f"{'arm':<15}{'AUC':>9}{'mean d':>10}{'sd':>8}{'95% CI':>22}{'pos':>7}"
          f"{'>=MDE':>7}{'Brier':>9}{'ECE':>8}")
    out = {"n_boot": N_BOOT, "mde": MDE, "arms": {}}
    for lab in ARMS:
        d = (R[f"{lab}_auc"] - R["base_auc"]).values
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{lab:<15}{R[f'{lab}_auc'].mean():>9.4f}{d.mean():>+10.4f}{d.std():>8.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{f'{(d>0).sum()}/{N_BOOT}':>7}"
              f"{f'{(d>=MDE).sum()}/{N_BOOT}':>7}{R[f'{lab}_brier'].mean():>9.4f}"
              f"{R[f'{lab}_ece'].mean():>8.4f}")
        out["arms"][lab] = {"auc": float(R[f"{lab}_auc"].mean()),
                            "mean_delta": float(d.mean()), "ci": [float(lo), float(hi)],
                            "positive": int((d > 0).sum()), "at_mde": int((d >= MDE).sum()),
                            "auc_2min": float(R[f"{lab}_auc2"].mean()),
                            "brier": float(R[f"{lab}_brier"].mean()),
                            "ece": float(R[f"{lab}_ece"].mean()),
                            "clears": bool(lo > 0 and d.mean() >= MDE)}
    print("\n2-min-only subset:")
    for lab in ARMS:
        d2 = (R[f"{lab}_auc2"] - R["base_auc2"]).values
        print(f"  {lab:<15} AUC {R[f'{lab}_auc2'].mean():.4f}  mean d {d2.mean():+.4f}")
    print(f"\nA clearing arm needs ci_lo > 0 AND mean delta >= MDE ({MDE}).")
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
