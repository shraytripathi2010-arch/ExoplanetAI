"""gpc_ensemble_validate.py -- PART 3: the full resampling protocol.

The screen (`gpc_screen.py`) pointed hard at a negative, but single fits are not
a result in this project -- the calibration sweep dissolved seven of them at
once, and that lesson applies to calibration claims as much as to AUC claims,
which matters here because calibration quality is this proposal's stated
purpose. So the negative gets measured properly.

COST CORRECTION, recorded because it changed the plan
------------------------------------------------------
Part 0 extrapolated a fitted t = 8.25e-10 * n^3.30 to 14.8 min for a bare GPC
fit at 4,390 rows and 35.4 min for a calibrated one, projecting ~7 h for this
protocol. MEASURED at full scale: bare 3.9 min, calibrated 15.5 min -- the
power law was ~3.8x too pessimistic because it was fitted on small-n points
where fixed overhead dominates. The prior run's 267.9 s at 4,392 rows was the
accurate anchor and should have outweighed the extrapolation. With the true
cost, the full protocol is affordable and there is no reason to reduce rigour.

MEMBER CHOICE, decided by measurement rather than by convention
---------------------------------------------------------------
The GPC member is BARE, not wrapped in CalibratedClassifierCV, because the
screen measured the wrapper to be actively counterproductive for a GP:

    gpc_bare  AUC 0.9100   ECE 0.0239   3.9 min
    gpc_cal   AUC 0.9073   ECE 0.0295  15.5 min   (rho 0.993 vs bare)

A GP's Laplace posterior is already a calibrated probability -- bare GPC's ECE
(0.0239) matches production HGB's (0.0241). Wrapping it costs accuracy, costs
calibration, and costs 4x the time. Using the wrapper anyway "for comparability"
would handicap the arm being tested.

EFFICIENT STACKING, not a shortcut
-----------------------------------
One StratifiedKFold(5) pass per member per bootstrap produces BOTH the
out-of-fold probabilities the meta-learner needs AND the test predictions
(averaged over the fold models). This is standard stacking practice and is what
`CalibratedClassifierCV` does internally; it is not a corner cut.

Baseline is the live production artifact, cloned. Nothing is promoted.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys
import json
import time
import importlib.util
import warnings
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
CADENCE = os.path.join(HERE, "cadence_class_confound.csv")
OUT = os.path.join(HERE, "gpc_ensemble_validate.json")

N_BOOT = 12
SEED = 20260814
MDE = 0.0097
PROD_AUC = 0.9454155993948381


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def make_gpc():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", GaussianProcessClassifier(
                         kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
                         random_state=42))])


def make_catboost():
    from catboost import CatBoostClassifier
    return CalibratedClassifierCV(
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", CatBoostClassifier(verbose=0, random_seed=42))]),
        cv=5, method="sigmoid")


def oof_and_test(make, Xb, yb, Xte, seed):
    """One cv=5 pass -> (out-of-fold train probs, mean test probs)."""
    oof = np.zeros(len(yb))
    tps = []
    for i_tr, i_va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xb, yb):
        m = make().fit(Xb.iloc[i_tr], yb[i_tr])
        oof[i_va] = m.predict_proba(Xb.iloc[i_va])[:, 1]
        tps.append(m.predict_proba(Xte)[:, 1])
    return oof, np.mean(tps, axis=0)


def one_boot(b, samp, X, y, te, s2, prod_proto):
    t0 = time.time()
    Xb, yb, Xte, yte = X.iloc[samp], y[samp], X[te], y[te]

    # production baseline: its own recipe, refit on this resample
    base = clone(prod_proto).fit(Xb, yb)
    p_hgb = base.predict_proba(Xte)[:, 1]

    oof_h, tp_h = oof_and_test(lambda: clone(prod_proto), Xb, yb, Xte, 100 + b)
    oof_c, tp_c = oof_and_test(make_catboost, Xb, yb, Xte, 200 + b)
    oof_g, tp_g = oof_and_test(make_gpc, Xb, yb, Xte, 300 + b)

    P = {"hgb_prod": p_hgb, "gpc_only": tp_g, "catboost_only": tp_c,
         "avg_hgb_cat_gpc": np.mean([p_hgb, tp_c, tp_g], axis=0),
         "avg_hgb_gpc": np.mean([p_hgb, tp_g], axis=0)}
    lr3 = LogisticRegression(max_iter=1000).fit(
        np.column_stack([oof_h, oof_c, oof_g]), yb)
    P["meta_hgb_cat_gpc"] = lr3.predict_proba(np.column_stack([tp_h, tp_c, tp_g]))[:, 1]
    lr2 = LogisticRegression(max_iter=1000).fit(np.column_stack([oof_h, oof_g]), yb)
    P["meta_hgb_gpc"] = lr2.predict_proba(np.column_stack([tp_h, tp_g]))[:, 1]

    rec = {"b": b, "w3": lr3.coef_[0].tolist(), "w2": lr2.coef_[0].tolist()}
    for k, p in P.items():
        rec[f"{k}|auc"] = roc_auc_score(yte, p)
        rec[f"{k}|brier"] = brier_score_loss(yte, p)
        rec[f"{k}|ece"] = ece(yte, p)
        rec[f"{k}|auc2"] = roc_auc_score(yte[s2], p[s2])
    print(f"  boot {b+1}/{N_BOOT}  hgb {rec['hgb_prod|auc']:.4f}  "
          f"gpc {rec['gpc_only|auc']:.4f}  avg3 {rec['avg_hgb_cat_gpc|auc']:.4f}  "
          f"meta3 {rec['meta_hgb_cat_gpc|auc']:.4f}  [{(time.time()-t0)/60:.1f} min]",
          flush=True)
    return rec


def main():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 33

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    tr_mask, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)
    tr = np.where(tr_mask)[0]
    cad = pd.read_csv(CADENCE)[["host", "cadence_min"]]; cad["host"] = cad.host.astype(str)
    cc = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((cc >= 1.0) & (cc <= 2.6)).to_numpy()
    s2 = is2[te]

    prod = joblib.load(PROD)
    hp = prod.estimator.named_steps["clf"].get_params()
    assert hp["max_iter"] == 475 and hp["class_weight"] == "balanced", \
        "live model is NOT the Optuna config"
    print(f"train {len(tr)}  frozen test {int(te.sum())}  2-min {int(s2.sum())}")
    print(f"baseline = live production artifact (Optuna HGB), AUC of record {PROD_AUC:.4f}\n")

    rng = np.random.default_rng(SEED)
    draws = [rng.choice(tr, size=len(tr), replace=True) for _ in range(N_BOOT)]
    t0 = time.time()
    rows = Parallel(n_jobs=4, backend="loky")(
        delayed(one_boot)(b, draws[b], X, y, te, s2, prod) for b in range(N_BOOT))
    R = pd.DataFrame(sorted(rows, key=lambda r: r["b"]))

    ARMS = ["hgb_prod", "gpc_only", "catboost_only", "avg_hgb_cat_gpc",
            "avg_hgb_gpc", "meta_hgb_cat_gpc", "meta_hgb_gpc"]
    print("\n" + "=" * 104)
    print(f"{'arm':<20}{'AUC':>9}{'mean d':>10}{'sd':>8}{'95% CI':>22}{'pos':>7}"
          f"{'>=MDE':>7}{'Brier':>9}{'ECE':>8}{'2-min d':>10}")
    out = {"n_boot": N_BOOT, "mde": MDE, "prod_auc_of_record": PROD_AUC, "arms": {}}
    for k in ARMS:
        d = (R[f"{k}|auc"] - R["hgb_prod|auc"]).values
        d2 = (R[f"{k}|auc2"] - R["hgb_prod|auc2"]).values
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{k:<20}{R[f'{k}|auc'].mean():>9.4f}{d.mean():>+10.4f}{d.std():>8.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{f'{(d>0).sum()}/{N_BOOT}':>7}"
              f"{f'{(d>=MDE).sum()}/{N_BOOT}':>7}{R[f'{k}|brier'].mean():>9.4f}"
              f"{R[f'{k}|ece'].mean():>8.4f}{d2.mean():>+10.4f}")
        out["arms"][k] = {"auc": float(R[f"{k}|auc"].mean()),
                          "mean_delta": float(d.mean()), "sd": float(d.std()),
                          "ci": [float(lo), float(hi)],
                          "positive": int((d > 0).sum()), "at_mde": int((d >= MDE).sum()),
                          "brier": float(R[f"{k}|brier"].mean()),
                          "ece": float(R[f"{k}|ece"].mean()),
                          "auc_2min": float(R[f"{k}|auc2"].mean()),
                          "delta_2min": float(d2.mean()),
                          "clears": bool(lo > 0 and d.mean() >= MDE)}
    w3 = np.array(R.w3.tolist()).mean(axis=0)
    w2 = np.array(R.w2.tolist()).mean(axis=0)
    print(f"\nmean meta-learner weights (out-of-fold fitted, {N_BOOT} resamples):")
    print(f"  3-member: hgb={w3[0]:+.2f}  catboost={w3[1]:+.2f}  gpc={w3[2]:+.2f}")
    print(f"  2-member: hgb={w2[0]:+.2f}  gpc={w2[1]:+.2f}")
    out["meta_weights"] = {"three": w3.tolist(), "two": w2.tolist()}
    print(f"\nClearing needs ci_lo > 0 AND mean delta >= MDE ({MDE}).")
    print(f"wall clock {(time.time()-t0)/60:.1f} min")
    out["wall_clock_min"] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
