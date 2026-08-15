"""gpc_screen.py -- PART 1 (screening): where does a GPC actually land at 33
features, on the frozen split, calibrated -- and is an ensemble worth 7 hours?

WHY A SCREEN RATHER THAN THE FULL PROTOCOL
-------------------------------------------
Part 0 measured exact GPC at t = 8.25e-10 * n^3.30, i.e. 35.4 min per
`CalibratedClassifierCV(cv=5)` fit at 4,390 rows, so the full 12-bootstrap
protocol is ~7.1 h of GPC time even in its efficient form. That is a real budget
and it was authorised to be spent only if there is a reason to. This screen
supplies the reason or removes it.

NOTHING HERE IS A PROMOTION-GRADE RESULT. Single fits, explicitly. The standing
rule -- 10+ resamples before any AUC/Brier/ECE claim is treated as real -- is
NOT being relaxed; these numbers exist to size a decision, and are labelled as
such everywhere they appear.

THE DECISIVE DIAGNOSTIC IS DIVERSITY, NOT GPC's OWN SCORE
---------------------------------------------------------
An averaging ensemble can only beat its best member if the members make
DIFFERENT errors. Part C already found the meta-learner leaning almost entirely
on HGB (weights HGB 4.13 / GP 2.18 / CNN 2.33) and landing at 0.9018 vs 0.9016.
So the screen reports rank correlation between member probabilities and, more
importantly, the AUC of each member on the stars HGB gets WRONG -- if GPC cannot
rank HGB's own mistakes better than chance, no weighting of the two can help,
and the 7 h is answerable in advance.

THE META-CALIBRATOR SCREEN, AND ITS HONEST LIMITATION
------------------------------------------------------
The requested LR meta-calibrator normally needs out-of-fold member probabilities
on the TRAINING set, which for GPC is another 35 min per resample. Instead this
screens it on random stratified HALVES OF THE FROZEN TEST: fit the LR on one
half's member probabilities, score the other half, repeat over many splits.
That is leakage-free (the base models never saw the test set) and is the same
halves device the conformal work already uses. It is NOT the production-style
protocol, and it is reported as a screening estimate only.

Baseline is the CURRENT production recipe: Optuna-tuned HGB read from the live
artifact, not retyped.
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
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
CADENCE = os.path.join(HERE, "cadence_class_confound.csv")
OUT = os.path.join(HERE, "gpc_screen.json")

SEED = 42
N_META_SPLITS = 200


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def main():
    from scipy.stats import spearmanr
    from catboost import CatBoostClassifier

    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(ROOT, "code", "05_train_models.py"))
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
    tr = np.where(tr_mask)[0]
    cad = pd.read_csv(CADENCE)[["host", "cadence_min"]]; cad["host"] = cad.host.astype(str)
    cc = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((cc >= 1.0) & (cc <= 2.6)).to_numpy()
    Xtr, ytr, Xte, yte, s2 = X.iloc[tr], y[tr], X[te], y[te], is2[te]
    print(f"train {len(tr)}  frozen test {int(te.sum())}  2-min {int(s2.sum())}")

    # ---- production recipe, cloned from the LIVE artifact ----
    prod = joblib.load(PROD)
    hp = prod.estimator.named_steps["clf"].get_params()
    print(f"production recipe (from live artifact): lr={hp['learning_rate']:.4f} "
          f"iter={hp['max_iter']} leaves={hp['max_leaf_nodes']} "
          f"msl={hp['min_samples_leaf']} l2={hp['l2_regularization']:.4f} "
          f"cw={hp['class_weight']}")
    assert hp["max_iter"] == 475 and hp["class_weight"] == "balanced", \
        "live model is NOT the Optuna config -- stale-config guard tripped"

    def gpc(cal=True):
        base = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("clf", GaussianProcessClassifier(
                             kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
                             random_state=SEED))])
        return CalibratedClassifierCV(base, cv=5, method="sigmoid") if cal else base

    def nystroem():
        return CalibratedClassifierCV(
            Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler()),
                      ("approx", Nystroem(kernel="rbf", n_components=1000,
                                          random_state=SEED)),
                      ("clf", LogisticRegression(max_iter=2000))]),
            cv=5, method="sigmoid")

    def catboost():
        return CalibratedClassifierCV(
            Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("clf", CatBoostClassifier(verbose=0, random_seed=SEED))]),
            cv=5, method="sigmoid")

    ARMS = [("hgb_prod", lambda: clone(prod)),
            ("catboost", catboost),
            ("gpc_cal", lambda: gpc(True)),
            ("gpc_bare", lambda: gpc(False)),
            ("nystroem", nystroem)]

    P, res = {}, {"n_train": int(len(tr)), "n_test": int(te.sum()), "members": {}}
    print(f"\n=== SINGLE-FIT MEMBERS (screening only -- NOT promotion-grade) ===")
    print(f"  {'member':<12}{'fit min':>9}{'AUC':>9}{'2-min':>9}{'Brier':>9}{'ECE':>8}")
    for name, mk in ARMS:
        t0 = time.time()
        mo = mk().fit(Xtr, ytr)
        el = (time.time() - t0) / 60
        p = mo.predict_proba(Xte)[:, 1]
        P[name] = p
        a, b, e = roc_auc_score(yte, p), brier_score_loss(yte, p), ece(yte, p)
        a2 = roc_auc_score(yte[s2], p[s2])
        print(f"  {name:<12}{el:>9.1f}{a:>9.4f}{a2:>9.4f}{b:>9.4f}{e:>8.4f}", flush=True)
        res["members"][name] = {"fit_min": round(el, 2), "auc": float(a),
                                "auc_2min": float(a2), "brier": float(b), "ece": float(e)}

    # ---- diversity: the precondition for ensembling ----
    print("\n=== DIVERSITY (an ensemble can only help if members disagree) ===")
    print("  Spearman rho between member probabilities on the frozen test:")
    names = [n for n, _ in ARMS]
    res["rho"] = {}
    for i, a in enumerate(names):
        for b_ in names[i + 1:]:
            r = float(spearmanr(P[a], P[b_]).statistic)
            res["rho"][f"{a}|{b_}"] = r
            flag = "  <- near-duplicate" if abs(r) >= 0.90 else ""
            print(f"    {a:<10} vs {b_:<10} {r:+.3f}{flag}")

    # ---- can any member rank the stars HGB gets WRONG? ----
    ph = P["hgb_prod"]
    wrong = (ph >= 0.5).astype(int) != yte
    print(f"\n  HGB misclassifies {int(wrong.sum())}/{len(yte)} test stars at 0.5.")
    print("  AUC of each member ON THOSE STARS ONLY (0.5 = no rescue possible):")
    res["auc_on_hgb_errors"] = {}
    for n in names:
        yy = yte[wrong]
        if len(set(yy)) > 1:
            a = float(roc_auc_score(yy, P[n][wrong]))
            res["auc_on_hgb_errors"][n] = a
            print(f"    {n:<12}{a:.4f}")

    # ---- single-fit ensembles (free, given the member probabilities) ----
    print("\n=== SINGLE-FIT ENSEMBLES (screening only) ===")
    print(f"  {'ensemble':<28}{'AUC':>9}{'d vs prod':>11}{'Brier':>9}{'ENSE':>8}")
    ENS = {"avg hgb+cat+gpc": ["hgb_prod", "catboost", "gpc_cal"],
           "avg hgb+gpc": ["hgb_prod", "gpc_cal"],
           "avg hgb+cat": ["hgb_prod", "catboost"],
           "avg hgb+cat+nystroem": ["hgb_prod", "catboost", "nystroem"]}
    base_auc = res["members"]["hgb_prod"]["auc"]
    res["ensembles"] = {}
    for lab, mem in ENS.items():
        p = np.mean([P[m] for m in mem], axis=0)
        a, b, e = roc_auc_score(yte, p), brier_score_loss(yte, p), ece(yte, p)
        print(f"  {lab:<28}{a:>9.4f}{a-base_auc:>+11.4f}{b:>9.4f}{e:>8.4f}")
        res["ensembles"][lab] = {"auc": float(a), "delta": float(a - base_auc),
                                 "brier": float(b), "ece": float(e)}

    # ---- LR meta-calibrator, screened on stratified halves of the test ----
    print(f"\n=== LR META-CALIBRATOR, screened on {N_META_SPLITS} stratified "
          "test halves ===")
    print("  (fit on one half's member probabilities, scored on the other --")
    print("   leakage-free, but NOT the production OOF protocol)")
    sss = StratifiedShuffleSplit(n_splits=N_META_SPLITS, test_size=0.5, random_state=SEED)
    res["meta"] = {}
    for lab, mem in (("meta hgb+cat+gpc", ["hgb_prod", "catboost", "gpc_cal"]),
                     ("meta hgb+gpc", ["hgb_prod", "gpc_cal"]),
                     ("meta hgb+cat+gpc+nys",
                      ["hgb_prod", "catboost", "gpc_cal", "nystroem"])):
        M = np.column_stack([P[m] for m in mem])
        das, coefs = [], []
        for i_a, i_b in sss.split(M, yte):
            lr = LogisticRegression(max_iter=1000).fit(M[i_a], yte[i_a])
            pm = lr.predict_proba(M[i_b])[:, 1]
            das.append(roc_auc_score(yte[i_b], pm) - roc_auc_score(yte[i_b], ph[i_b]))
            coefs.append(lr.coef_[0])
        das = np.array(das); C = np.array(coefs).mean(axis=0)
        lo, hi = np.percentile(das, [2.5, 97.5])
        print(f"  {lab:<22} mean d vs HGB {das.mean():+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  better {int((das>0).sum())}/{N_META_SPLITS}")
        print(f"    {'mean LR weights:':<22}" +
              "  ".join(f"{m}={c:+.2f}" for m, c in zip(mem, C)))
        res["meta"][lab] = {"mean_delta": float(das.mean()),
                            "ci": [float(lo), float(hi)],
                            "better": int((das > 0).sum()), "n_splits": N_META_SPLITS,
                            "weights": {m: float(c) for m, c in zip(mem, C)}}

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
