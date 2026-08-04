"""baseline_stability_audit.py -- does the +/-0.005 baseline wobble invalidate
the recorded negatives, or does it cancel?

THE FINDING THAT PROMPTED THIS
Dropping a single training row moves the bare-HGB refit baseline by up to
+0.0068 (sd 0.0018), upward in 15/15 draws. That is larger than the +/-0.003
noise floor and comparable to every effect this project has tried to detect.
The obvious worry: 28 recorded negatives were measured against exactly this
kind of refit baseline, so were their confidence intervals too narrow?

THE QUESTION IS NOT WHETHER THE BASELINE MOVES -- IT IS WHETHER THE DELTA DOES

Every one of those experiments is a PAIRED comparison: baseline and challenger
are fit on the SAME training rows. If a dropped row perturbs both arms the same
way, the perturbation cancels in the difference and the recorded CIs are fine.
If the two arms respond differently, the delta inherits the instability and the
CIs were too narrow.

That is an empirical question and it is cheap to answer, so it is answered here
rather than argued. For each leave-one-out perturbation, BOTH arms are refit on
the identical perturbed data and the delta recomputed on the frozen test set.

TWO EXPERIMENT SHAPES ARE TESTED SEPARATELY, because they have different
reasons to cancel or not:

  A. FEATURE ADDITION (24 -> 26 features, same learner) -- the dominant pattern
     in this project. Same algorithm, same hyperparameters, nearly the same
     model; strong reason to expect cancellation.
  B. MODEL SWAP (HGB -> CatBoost) -- different algorithm entirely; much weaker
     reason to expect cancellation.

What matters is sd(delta) compared against the +/-0.003 noise floor and against
the paired-bootstrap CI half-widths those experiments actually reported.
"""
import os
import sys
import json
import importlib.util
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
SECONDARY = os.path.join(SCRIPT_DIR, "weak_secondary_features.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "baseline_stability_audit.json")

N_LOO = 12
SEED = 42
NEW = ["sec_significance", "sec_depth_windowed"]
CAT_PARAMS = dict(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=9.0)


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def imp(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", est)])


def main():
    m05 = _m05()
    df = pd.read_csv(TRAINING)
    sec = pd.read_csv(SECONDARY)
    df["host"] = df["host"].astype(str)
    sec["host"] = sec["host"].astype(str)
    df = df.merge(sec[["host"] + NEW], on="host", how="left")

    X0, y = m05.build_feature_matrix(df)
    X0 = X0.reset_index(drop=True)
    y = np.asarray(y)
    tr, te = m05.split_by_host(df)
    tr, te = np.asarray(tr), np.asarray(te)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce")
    te2 = te & (((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy())

    Xn = X0.copy()
    for f in NEW:
        Xn[f] = pd.to_numeric(df[f], errors="coerce").to_numpy()

    prod = joblib.load(PROD)
    hgb_params = getattr(prod, "estimator", prod).named_steps["clf"].get_params()

    def hgb():
        return imp(HistGradientBoostingClassifier(**hgb_params))

    def cat():
        from catboost import CatBoostClassifier
        return imp(CatBoostClassifier(verbose=0, random_seed=SEED,
                                      auto_class_weights="Balanced",
                                      allow_writing_files=False, **CAT_PARAMS))

    tri = np.where(tr)[0]
    rng = np.random.RandomState(0)
    drops = [None] + list(rng.choice(tri, N_LOO, replace=False))

    print("=" * 88)
    print("BASELINE STABILITY AUDIT -- does the wobble cancel in a PAIRED delta?")
    print("=" * 88)
    print(f"  {N_LOO} leave-one-out perturbations; BOTH arms refit on the same")
    print(f"  perturbed rows each time; test set frozen ({int(te.sum())} / "
          f"{int(te2.sum())} 2-min).")

    out = {"n_loo": N_LOO}
    for shape, mk_a, mk_b, label in (
            ("A. feature addition (24 -> 26, same learner)", hgb, hgb, "feat"),
            ("B. model swap (HGB -> CatBoost)", hgb, cat, "swap")):
        print("\n" + "-" * 88)
        print(shape)
        print("-" * 88)
        print(f"  {'drop':>8}{'baseline':>11}{'challenger':>12}{'delta':>10}"
              f"{'delta 2min':>12}")
        base, chal, dF, d2 = [], [], [], []
        for d in drops:
            m = tr.copy()
            if d is not None:
                m[d] = False
            Xa = X0 if label == "swap" else X0
            Xb = X0 if label == "swap" else Xn
            ma = mk_a().fit(Xa[m], y[m])
            mb = mk_b().fit(Xb[m], y[m])
            aF = roc_auc_score(y[te], ma.predict_proba(Xa[te])[:, 1])
            bF = roc_auc_score(y[te], mb.predict_proba(Xb[te])[:, 1])
            a2 = roc_auc_score(y[te2], ma.predict_proba(Xa[te2])[:, 1])
            b2 = roc_auc_score(y[te2], mb.predict_proba(Xb[te2])[:, 1])
            base.append(aF); chal.append(bF)
            dF.append(bF - aF); d2.append(b2 - a2)
            tag = "none" if d is None else str(d)
            print(f"  {tag:>8}{aF:>11.4f}{bF:>12.4f}{bF-aF:>+10.4f}{b2-a2:>+12.4f}",
                  flush=True)
        base, chal = np.array(base), np.array(chal)
        dF, d2 = np.array(dF), np.array(d2)
        print(f"\n  sd(baseline)   {base.std():.4f}   range {base.max()-base.min():.4f}")
        print(f"  sd(challenger) {chal.std():.4f}   range {chal.max()-chal.min():.4f}")
        print(f"  sd(DELTA)      {dF.std():.4f}   range {dF.max()-dF.min():.4f}"
              f"   <- the number that matters")
        print(f"  sd(DELTA 2min) {d2.std():.4f}   range {d2.max()-d2.min():.4f}")
        cancel = 1.0 - (dF.std() / base.std()) if base.std() else float("nan")
        print(f"\n  cancellation: {100*cancel:.0f}% of the baseline wobble "
              f"disappears in the paired delta")
        print(f"  delta mean {dF.mean():+.4f}  (sign flips: "
              f"{int((np.sign(dF) != np.sign(dF[0])).sum())}/{len(dF)})")
        out[label] = {
            "sd_baseline": float(base.std()), "sd_challenger": float(chal.std()),
            "sd_delta": float(dF.std()), "sd_delta_2min": float(d2.std()),
            "range_delta": float(dF.max() - dF.min()),
            "delta_mean": float(dF.mean()), "cancellation": float(cancel),
            "deltas": [float(v) for v in dF]}

    print("\n" + "=" * 88)
    print("READ AGAINST THE BAR")
    print("=" * 88)
    print("  noise floor quoted throughout this project: +/-0.003")
    for k, nm in (("feat", "feature addition"), ("swap", "model swap")):
        s = out[k]["sd_delta"]
        print(f"  {nm:<20} sd(delta) {s:.4f} -> "
              f"{'INSIDE' if s < 0.003 else 'EXCEEDS'} the noise floor")
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
