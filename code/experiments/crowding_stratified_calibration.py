"""crowding_stratified_calibration.py -- PART 3: per-subpopulation calibration on
the HIGH-CROWDING axis, the second test of a mechanism already measured once.

WHAT IS ALREADY KNOWN, AND WHY THIS IS STILL WORTH MEASURING
------------------------------------------------------------
Arm C of the giant-star investigation fitted a per-group Platt sigmoid for
giants vs dwarfs. It was aimed at a REAL defect (giant ECE 0.0867 vs dwarf
0.0204, 4.2x) and it made calibration WORSE:

    giant ECE   0.0956 -> 0.1094   (+0.0138)
    overall ECE 0.0417 -> 0.0508
    overall AUC delta -0.0020, 2/12 positive, 0/12 clearing

The stated mechanism was variance, not bias: the per-group scaler saw ~900
giant training rows (~570 distinct under bootstrap) where the global one sees
4,386. Splitting the calibration set costs more variance than specialisation
buys. That is the same lesson as `sigmoid cv=3`, on a second axis.

PREDICTION, STATED BEFORE RUNNING
---------------------------------
High crowding is a comparably small slice of the same training set, so the
variance arithmetic is essentially identical and the prediction is that this
fails the same way: no clearing arm, and subpopulation ECE flat-to-worse.

Two things could in principle make crowding behave differently, and both are
checked rather than assumed:

  1. The giant axis carried a SEVERE spatial confound (AUC of |galactic b|
     alone within giants = 0.7092). Crowding is even more obviously a
     galactic-plane quantity, so the confound should be at least as bad here --
     which if anything strengthens the prediction.
  2. Crowding might not have a calibration defect to fix at all. Giants did.
     If high-crowding ECE already matches low-crowding ECE, there is nothing
     for specialisation to buy and the arm is dead on arrival for a second,
     independent reason. That is measured up front, before any arm is fitted.

THE CONTROL THAT MAKES THIS A MECHANISM TEST
--------------------------------------------
A RANDOM group of the same size gets the identical stratified treatment. If the
random control degrades calibration by about as much as the crowding split
does, the damage is the split itself -- pure variance cost -- and has nothing to
do with which subpopulation was chosen. The giant investigation asserted that
mechanism; this measures it.

Production's exact recipe, frozen split, 12 training bootstraps. Nothing
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
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT = os.path.join(HERE, "crowding_stratified_calibration.json")

N_BOOT = 12
SEED = 20260814
MDE = 0.0097
CROWD_THR = 1.0        # neighbour flux >= target flux inside the aperture
CROWD_THR_ALT = 0.5    # sensitivity arm


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def base_pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", HistGradientBoostingClassifier(random_state=42))])


def prod_fit(Xb, yb):
    return CalibratedClassifierCV(base_pipe(), cv=5, method="sigmoid").fit(Xb, yb)


def stratified_calibration(Xb, yb, gb, Xte, gte, seed=SEED):
    """One base model (ranking untouched), per-group Platt sigmoid fitted on
    out-of-fold probabilities. Identical in structure to giant arm C."""
    oof = np.zeros(len(yb))
    for i_tr, i_va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xb, yb):
        mm = clone(base_pipe()).fit(Xb.iloc[i_tr], yb[i_tr])
        oof[i_va] = mm.predict_proba(Xb.iloc[i_va])[:, 1]
    full = clone(base_pipe()).fit(Xb, yb)
    raw = full.predict_proba(Xte)[:, 1]

    out = np.empty(len(raw)); eps = 1e-6
    for flag in (False, True):
        m_tr, m_te = (gb == flag), (gte == flag)
        if m_te.sum() == 0:
            continue
        if m_tr.sum() < 30 or len(np.unique(yb[m_tr])) < 2:
            out[m_te] = raw[m_te]
            continue
        q = np.clip(oof[m_tr], eps, 1 - eps)
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(
            np.log(q / (1 - q)).reshape(-1, 1), yb[m_tr])
        qt = np.clip(raw[m_te], eps, 1 - eps)
        out[m_te] = lr.predict_proba(np.log(qt / (1 - qt)).reshape(-1, 1))[:, 1]
    return out


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

    cf = pd.to_numeric(df.crowd_flux_ratio_max, errors="coerce").to_numpy()
    sr = pd.to_numeric(df.st_rad, errors="coerce").to_numpy()
    crowd = np.nan_to_num(cf, nan=0.0) >= CROWD_THR
    crowd_alt = np.nan_to_num(cf, nan=0.0) >= CROWD_THR_ALT

    print("=== SUBPOPULATION SIZE (the variance arithmetic, up front) ===")
    print(f"  full training set                       {len(tr_idx)}")
    for lab, g in (("high crowding (flux_ratio>=1.0)", crowd),
                   ("high crowding (flux_ratio>=0.5)", crowd_alt),
                   ("giants (st_rad>=1.5), for scale", np.nan_to_num(sr, nan=0.0) >= 1.5)):
        n_tr, n_te = int((tr_mask & g).sum()), int((te & g).sum())
        print(f"  {lab:<38} train {n_tr:>5} ({n_tr/len(tr_idx):.1%})   "
              f"test {n_te:>5}   ~{int(n_tr*0.632)} distinct under bootstrap")

    # ---- is there a calibration defect to fix at all? deployed model, frozen test ----
    import joblib
    prod = joblib.load(os.path.join(ROOT, "models", "best_model.joblib"))
    pp = prod.predict_proba(X)[:, 1]
    print("\n=== IS THERE A DEFECT TO FIX? deployed model on the frozen test ===")
    print(f"  {'group':<28}{'n':>6}{'planet%':>10}{'AUC':>9}{'ECE':>9}")
    defect = {}
    for lab, g in (("high crowding >=1.0", crowd), ("low crowding <1.0", ~crowd),
                   ("giants >=1.5 (reference)", np.nan_to_num(sr, nan=0.0) >= 1.5),
                   ("dwarfs <1.5 (reference)", np.nan_to_num(sr, nan=0.0) < 1.5)):
        m = te & g
        a, e = roc_auc_score(y[m], pp[m]), ece(y[m], pp[m])
        print(f"  {lab:<28}{int(m.sum()):>6}{100*y[m].mean():>10.2f}{a:>9.4f}{e:>9.4f}")
        defect[lab] = {"n": int(m.sum()), "planet_pct": float(100 * y[m].mean()),
                       "auc": float(a), "ece": float(e)}

    # ---- spatial confound, mandatory ----
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra = pd.to_numeric(df.ra, errors="coerce").to_numpy()
    dec = pd.to_numeric(df.dec, errors="coerce").to_numpy()
    okc = np.isfinite(ra) & np.isfinite(dec)
    gb = np.full(len(df), np.nan)
    gb[okc] = np.abs(SkyCoord(ra[okc] * u.deg, dec[okc] * u.deg).galactic.b.deg)
    print("\n=== |GALACTIC LATITUDE| CONTROL, per the standing rule ===")
    for lab, g in (("within high crowding", crowd), ("within low crowding", ~crowd)):
        m = te & g & okc
        print(f"  AUC of |gal b| alone, {lab:<22}{roc_auc_score(y[m], -gb[m]):.4f}")
    rho = pd.Series(cf[okc]).corr(pd.Series(gb[okc]), method="spearman")
    print(f"  corr(crowd_flux_ratio_max, |gal b|)         {rho:+.4f}")

    # ---- arms ----
    ARMS = ["base", "C: crowd-stratified", "C-alt: crowd>=0.5", "CONTROL: random group"]
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    for b in range(N_BOOT):
        samp = rng.choice(tr_idx, size=len(tr_idx), replace=True)
        Xb, yb = X.iloc[samp], y[samp]
        gb_c, gb_a = crowd[samp], crowd_alt[samp]
        # random group matched in size to the crowding group, redrawn each rep
        rr = np.random.default_rng(SEED + b)
        rand_all = rr.random(len(df)) < crowd[tr_mask].mean()
        gb_r, gte_r = rand_all[samp], rand_all[te]

        P = {"base": prod_fit(Xb, yb).predict_proba(X[te])[:, 1],
             "C: crowd-stratified": stratified_calibration(Xb, yb, gb_c, X[te], crowd[te]),
             "C-alt: crowd>=0.5": stratified_calibration(Xb, yb, gb_a, X[te], crowd_alt[te]),
             "CONTROL: random group": stratified_calibration(Xb, yb, gb_r, X[te], gte_r)}
        rec = {}
        for k, p in P.items():
            rec[f"{k}|auc"] = roc_auc_score(y[te], p)
            rec[f"{k}|brier"] = brier_score_loss(y[te], p)
            rec[f"{k}|ece"] = ece(y[te], p)
            m = crowd[te]
            rec[f"{k}|auc_c"] = roc_auc_score(y[te][m], p[m])
            rec[f"{k}|ece_c"] = ece(y[te][m], p[m])
        rows.append(rec)
        print(f"  boot {b+1}/{N_BOOT}  base {rec['base|auc']:.4f}/{rec['base|ece_c']:.4f}  "
              f"strat {rec['C: crowd-stratified|auc']:.4f}/"
              f"{rec['C: crowd-stratified|ece_c']:.4f}  [{time.time()-t0:.0f}s]", flush=True)

    R = pd.DataFrame(rows)
    print("\n" + "=" * 108)
    print(f"{'arm':<24}{'AUC':>9}{'mean d':>10}{'95% CI':>22}{'pos':>7}"
          f"{'ECE':>8}{'crowd AUC':>11}{'d crowd':>10}{'crowd ECE':>11}{'d cECE':>9}")
    out = {"n_boot": N_BOOT, "mde": MDE, "crowd_threshold": CROWD_THR,
           "defect_profile": defect, "arms": {}}
    for k in ARMS:
        d = (R[f"{k}|auc"] - R["base|auc"]).values
        dc = (R[f"{k}|auc_c"] - R["base|auc_c"]).values
        de = (R[f"{k}|ece_c"] - R["base|ece_c"]).mean()
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{k:<24}{R[f'{k}|auc'].mean():>9.4f}{d.mean():>+10.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{f'{(d>0).sum()}/{N_BOOT}':>7}"
              f"{R[f'{k}|ece'].mean():>8.4f}{R[f'{k}|auc_c'].mean():>11.4f}"
              f"{dc.mean():>+10.4f}{R[f'{k}|ece_c'].mean():>11.4f}{de:>+9.4f}")
        out["arms"][k] = {"auc": float(R[f"{k}|auc"].mean()),
                          "mean_delta": float(d.mean()), "ci": [float(lo), float(hi)],
                          "positive": int((d > 0).sum()),
                          "brier": float(R[f"{k}|brier"].mean()),
                          "ece": float(R[f"{k}|ece"].mean()),
                          "auc_crowd": float(R[f"{k}|auc_c"].mean()),
                          "delta_auc_crowd": float(dc.mean()),
                          "ece_crowd": float(R[f"{k}|ece_c"].mean()),
                          "delta_ece_crowd": float(de),
                          "clears": bool(lo > 0 and d.mean() >= MDE)}
    print(f"\nA clearing arm needs ci_lo > 0 AND mean delta >= MDE ({MDE}).")
    print(f"wall clock {time.time()-t0:.0f}s")
    out["wall_clock_s"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
