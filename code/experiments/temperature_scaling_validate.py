"""temperature_scaling_validate.py -- PART 2: temperature scaling as the calibrator.

WHY THIS IS NOT A DUPLICATE OF THE CALIBRATION SWEEP
----------------------------------------------------
The sweep covered {sigmoid, isotonic, bag-only} x cv={3,5,10,20} plus dedicated
prefit holdouts at 5/10/20%. Temperature scaling is a DIFFERENT calibrator:

    Platt / sigmoid      p = expit(a*z + b)     TWO free parameters
    temperature scaling  p = expit(z / T)       ONE free parameter, NO bias

The missing bias term is the whole point. Platt can shift the base rate; a
temperature can only sharpen (T<1) or flatten (T>1) an existing decision
function. On a 79% positive training set that is a real constraint, not a
detail -- and it is exactly why temperature scaling is the standard choice in
the deep-learning literature (Guo et al. 2017), where the network's logits are
already roughly centred.

The prediction this makes, stated before running: temperature scaling should be
MORE robust on small calibration slices than either sigmoid or isotonic, because
one parameter is far cheaper to estimate than two (Platt) or a whole step
function (isotonic, which collapsed to 0.8876 at a 220-row slice). If the prefit
arms are where anything shows up, that is the mechanism.

STRUCTURAL FAIRNESS
-------------------
`CalibratedClassifierCV(cv=k)` does two things: fits k base models on (k-1)/k of
the data and AVERAGES their calibrated outputs. To isolate the calibrator, this
harness reproduces that structure exactly -- same `StratifiedKFold(k,
shuffle=False)` sklearn uses for an integer `cv`, so the k base models are
IDENTICAL between the sigmoid baseline and the temperature arm at the same k.
Only the mapping from logit to probability differs.

AUC NOTE, restated because it governs how to read the table: a monotone
transform cannot move AUC, so `temp cv=k` and `sigmoid cv=k` share base models
and can differ in AUC only through the averaging of k DIFFERENT monotone curves
(an average of monotone functions applied to different models is not itself a
monotone transform of any single score). Expect near-zero AUC deltas. Brier and
ECE are where a calibrator is allowed to matter, and ECE is the column the
prefit/holdout sweep established as the one that actually tracks calibration.

Production's exact recipe is the baseline. 12 training bootstraps. Nothing
promoted.
"""
import os
import sys
import json
import time
import importlib.util
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize_scalar
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(HERE, "cadence_class_confound.csv")
OUT = os.path.join(HERE, "temperature_scaling_validate.json")

N_BOOT = 12
SEED = 20260814
MDE = 0.0097


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def base_pipe():
    """Production's base estimator, unwrapped. `decision_function` on this
    Pipeline returns HGB's raw log-odds, which is the logit temperature scaling
    is defined on."""
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", HistGradientBoostingClassifier(random_state=42))])


def fit_temperature(z, y):
    """One scalar T > 0 minimising negative log-likelihood of expit(z / T).
    Optimised over log T so T stays positive without a constrained solver."""
    y = np.asarray(y, dtype=float)

    def nll(log_t):
        p = np.clip(expit(z / np.exp(log_t)), 1e-12, 1 - 1e-12)
        return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    r = minimize_scalar(nll, bounds=(-5.0, 5.0), method="bounded")
    return float(np.exp(r.x))


class TemperatureCalibrated:
    """Mirror of CalibratedClassifierCV with a temperature in place of Platt.

    cv=k        -> k base models, each temperature-fitted on its held-out fold,
                   outputs averaged.  Same folds sklearn uses for integer cv.
    holdout=f   -> ONE base model on (1-f) of the rows, T fitted on the disjoint
                   slice.  No averaging.  Matches the sweep's prefit arms.
    """

    def __init__(self, cv=5, holdout=None, seed=SEED):
        self.cv, self.holdout, self.seed = cv, holdout, seed

    def fit(self, X, y):
        y = np.asarray(y)
        self.pairs_, self.temps_ = [], []
        if self.holdout is not None:
            i_base, i_cal = train_test_split(
                np.arange(len(y)), test_size=self.holdout, stratify=y,
                random_state=self.seed)
            m = clone(base_pipe()).fit(X.iloc[i_base], y[i_base])
            t = fit_temperature(m.decision_function(X.iloc[i_cal]), y[i_cal])
            self.pairs_.append((m, t)); self.temps_.append(t)
            return self
        for i_tr, i_va in StratifiedKFold(self.cv, shuffle=False).split(X, y):
            m = clone(base_pipe()).fit(X.iloc[i_tr], y[i_tr])
            t = fit_temperature(m.decision_function(X.iloc[i_va]), y[i_va])
            self.pairs_.append((m, t)); self.temps_.append(t)
        return self

    def predict_proba_pos(self, X):
        return np.mean([expit(m.decision_function(X) / t)
                        for m, t in self.pairs_], axis=0)


def sigmoid_arm(cv, X, y, Xte):
    mo = CalibratedClassifierCV(base_pipe(), cv=cv, method="sigmoid").fit(X, y)
    return mo.predict_proba(Xte)[:, 1]


def main():
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
    tr_idx = np.where(tr_mask)[0]
    cad = pd.read_csv(CADENCE)[["host", "cadence_min"]]; cad["host"] = cad.host.astype(str)
    cc = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((cc >= 1.0) & (cc <= 2.6)).to_numpy()
    Xte, yte, s2 = X[te], y[te], is2[te]
    print(f"train {len(tr_idx)}  frozen test {int(te.sum())}  2-min {int(s2.sum())}  "
          f"positive rate {y[tr_mask].mean():.4f}")

    # arm name -> (kind, param).  Matched pairs so cross-fit vs holdout differ
    # only in averaging, exactly as the prefit sweep set them up.
    ARMS = [("sigmoid cv=5 [PROD]", "sig", 5),
            ("sigmoid cv=10", "sig", 10),
            ("temp cv=3", "temp_cv", 3),
            ("temp cv=5", "temp_cv", 5),
            ("temp cv=10", "temp_cv", 10),
            ("temp prefit 20%", "temp_ho", 0.20),
            ("temp prefit 10%", "temp_ho", 0.10),
            ("temp prefit 5%", "temp_ho", 0.05)]

    rng = np.random.default_rng(SEED)
    rows, temps = [], {a: [] for a, k, _ in ARMS if k != "sig"}
    t0 = time.time()
    for b in range(N_BOOT):
        samp = rng.choice(tr_idx, size=len(tr_idx), replace=True)
        Xb, yb = X.iloc[samp], y[samp]
        rec = {}
        for name, kind, par in ARMS:
            if kind == "sig":
                p = sigmoid_arm(par, Xb, yb, Xte)
            else:
                tc = TemperatureCalibrated(
                    cv=par if kind == "temp_cv" else None,
                    holdout=par if kind == "temp_ho" else None).fit(Xb, yb)
                p = tc.predict_proba_pos(Xte)
                temps[name].append(float(np.mean(tc.temps_)))
            rec[f"{name}|auc"] = roc_auc_score(yte, p)
            rec[f"{name}|brier"] = brier_score_loss(yte, p)
            rec[f"{name}|ece"] = ece(yte, p)
            rec[f"{name}|auc2"] = roc_auc_score(yte[s2], p[s2])
        rows.append(rec)
        print(f"  boot {b+1}/{N_BOOT}  prod {rec['sigmoid cv=5 [PROD]|auc']:.4f}  "
              f"temp5 {rec['temp cv=5|auc']:.4f}  "
              f"ece prod {rec['sigmoid cv=5 [PROD]|ece']:.4f} "
              f"temp5 {rec['temp cv=5|ece']:.4f}  [{time.time()-t0:.0f}s]", flush=True)

    R = pd.DataFrame(rows)
    BASE = "sigmoid cv=5 [PROD]"
    print("\n" + "=" * 104)
    print(f"{'arm':<22}{'AUC':>9}{'mean d':>10}{'95% CI':>22}{'pos':>7}"
          f"{'Brier':>9}{'d Brier':>10}{'ECE':>8}{'d ECE':>9}{'mean T':>8}")
    out = {"n_boot": N_BOOT, "mde": MDE, "arms": {}}
    for name, kind, par in ARMS:
        d = (R[f"{name}|auc"] - R[f"{BASE}|auc"]).values
        lo, hi = np.percentile(d, [2.5, 97.5])
        db = (R[f"{name}|brier"] - R[f"{BASE}|brier"]).mean()
        de = (R[f"{name}|ece"] - R[f"{BASE}|ece"]).mean()
        mt = np.mean(temps[name]) if name in temps and temps[name] else float("nan")
        print(f"{name:<22}{R[f'{name}|auc'].mean():>9.4f}{d.mean():>+10.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{f'{(d>0).sum()}/{N_BOOT}':>7}"
              f"{R[f'{name}|brier'].mean():>9.4f}{db:>+10.4f}"
              f"{R[f'{name}|ece'].mean():>8.4f}{de:>+9.4f}"
              f"{('' if np.isnan(mt) else f'{mt:.3f}'):>8}")
        out["arms"][name] = {
            "auc": float(R[f"{name}|auc"].mean()), "mean_delta": float(d.mean()),
            "ci": [float(lo), float(hi)], "positive": int((d > 0).sum()),
            "brier": float(R[f"{name}|brier"].mean()), "delta_brier": float(db),
            "ece": float(R[f"{name}|ece"].mean()), "delta_ece": float(de),
            "auc_2min": float(R[f"{name}|auc2"].mean()),
            "mean_T": None if np.isnan(mt) else float(mt),
            "clears": bool(lo > 0 and d.mean() >= MDE)}
    print(f"\nA clearing arm needs ci_lo > 0 AND mean delta >= MDE ({MDE}).")
    print("Lower Brier and lower ECE are better; d Brier / d ECE negative = better "
          "than production.")
    print(f"wall clock {time.time()-t0:.0f}s")
    out["wall_clock_s"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
