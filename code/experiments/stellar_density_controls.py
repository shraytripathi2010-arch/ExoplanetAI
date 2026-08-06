"""stellar_density_controls.py -- the controls that decide whether arm B is real.

WHY THIS EXISTS. The first validation pass returned:

    A: +rho_ratio          +0.0030    2/12 clear
    B: +all four           +0.0091   12/12 clear   6/12 at/above MDE
    C: +rho_ratio | sky    +0.0024    2/12 clear
    D: availability ONLY   +0.0024    6/12 clear

Arm B clears on every resample. Two things make that insufficient to believe:

1. The spatial control arm was run on A, not on B. B is the arm that clears,
   and it contains `st_rho`, which carries the LARGEST sky exposure of the four
   (Spearman +0.192 vs |galactic b|). A control on the arm that did not clear
   proves nothing about the arm that did. This is the same methodological gap
   flagged in the time-series write-up, so it is closed here rather than noted.

2. Arm D shows a missingness-only indicator is worth +0.0024 on its own. That
   is not the 108% that killed the multi-sector depth-consistency feature, but
   it is roughly a quarter of B's gain and it is NOT nothing. The four columns
   share an 81.5%-coverage NaN pattern, so B gets that channel four times over.

ARMS. Each against its own reference, identical except the columns under test.

  B_sky        base + abs_gal_b       -> + all four
      Does B's gain survive when sky position is already in the model?

  B_avail4     base                   -> + four INDICATOR columns, no values
      Matches B's column count with pure missingness. If this reproduces B,
      B is bookkeeping. This is the control that killed multi-sector.

  B_restricted base                   -> + all four,
      BOTH fitted and evaluated ONLY on rows where st_rho is present, so
      missingness is constant by construction and cannot carry information.
      This is the decisive test: it isolates the measured VALUES.

The restricted arm reports a delta on a smaller test set, so its confidence
interval is wider and it is not directly comparable to B's magnitude -- it
answers "is there signal in the values at all", not "how large is it".
"""
import os
import sys
import json
import time
import importlib.util
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import fast_auc  # noqa: E402

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
DFEAT = os.path.join(SCRIPT_DIR, "stellar_density_features.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "stellar_density_controls_results.json")

SEED, N_RESAMPLES, N_BOOT, N_WORKERS = 42, 12, 1500, 6
MDE = 0.0097
ALL4 = ["st_logg", "st_rho", "rho_circ", "rho_ratio"]
AVAIL4 = [c + "_avail" for c in ALL4]
SKY = "abs_gal_b"


def paired_boot(y, pa, pb, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        yi = y[i]
        if yi.sum() in (0, len(yi)):
            continue
        d.append(fast_auc(yi, pb[i]) - fast_auc(yi, pa[i]))
    d = np.asarray(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


_G = {}


def _init():
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    m05 = _m05()
    df = pd.read_csv(TRAINING)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    y = np.asarray(y)

    d = pd.read_csv(DFEAT)
    merged = df[["host"]].merge(d[["host"] + ALL4], on="host", how="left")
    for c in ALL4:
        X[c] = pd.to_numeric(merged[c], errors="coerce").to_numpy()
        X[c + "_avail"] = np.isfinite(X[c]).astype(float)

    ra = pd.to_numeric(df["ra"], errors="coerce")
    dec = pd.to_numeric(df["dec"], errors="coerce")
    ok = ra.notna() & dec.notna()
    b = np.full(len(df), np.nan)
    b[ok.to_numpy()] = np.abs(
        SkyCoord(ra[ok].values * u.deg, dec[ok].values * u.deg).galactic.b.deg)
    X[SKY] = b

    tr, _ = m05.split_by_host(df)
    frozen = m05.frozen_test_mask(df)
    have = np.isfinite(X["st_rho"]).to_numpy()
    prod = joblib.load(PROD)
    _G.update(X=X, y=y, tr=np.asarray(tr), te=frozen, have=have,
              base=list(m05.FEATURE_COLUMNS),
              hgb=clone(getattr(prod, "estimator", prod)))


def _fit(cols, Xb, yb, X, te, expect):
    assert len(cols) == expect, f"expected {expect} columns, got {len(cols)}"
    est = CalibratedClassifierCV(clone(_G["hgb"]), cv=5, method="sigmoid")
    est.fit(Xb[cols], yb)
    return est.predict_proba(X.loc[te, cols])[:, 1]


def run_resample(rep):
    if not _G:
        _init()
    X, y, tr, te, have = _G["X"], _G["y"], _G["tr"], _G["te"], _G["have"]
    base = _G["base"]
    rng = np.random.RandomState(1000 + rep)
    row = {"rep": rep, "arms": {}}

    # ---------- full-population arms ----------
    Xtr, ytr = X[tr], y[tr]
    idx = rng.randint(0, len(ytr), len(ytr))
    Xb, yb = Xtr.iloc[idx], ytr[idx]

    # B_sky: reference already contains sky
    ref = base + [SKY]
    rp = _fit(ref, Xb, yb, X, te, len(ref))
    pp = _fit(ref + ALL4, Xb, yb, X, te, len(ref) + 4)
    dl, lo, hi = paired_boot(y[te], rp, pp)
    row["arms"]["B_sky: +all four | sky"] = {
        "n_features": len(ref) + 4, "delta": dl, "ci_lo": lo, "ci_hi": hi,
        "clears": bool(lo > 0), "auc": float(roc_auc_score(y[te], pp))}

    # B_avail4: four indicators, no values
    bp = _fit(base, Xb, yb, X, te, len(base))
    ap = _fit(base + AVAIL4, Xb, yb, X, te, len(base) + 4)
    dl, lo, hi = paired_boot(y[te], bp, ap)
    row["arms"]["B_avail4: indicators ONLY"] = {
        "n_features": len(base) + 4, "delta": dl, "ci_lo": lo, "ci_hi": hi,
        "clears": bool(lo > 0), "auc": float(roc_auc_score(y[te], ap))}
    row["base_auc"] = float(roc_auc_score(y[te], bp))

    # ---------- restricted population: missingness held constant ----------
    trh = tr & have
    teh = te & have
    Xtrh, ytrh = X[trh], y[trh]
    idx2 = rng.randint(0, len(ytrh), len(ytrh))
    Xb2, yb2 = Xtrh.iloc[idx2], ytrh[idx2]
    rp2 = _fit(base, Xb2, yb2, X, teh, len(base))
    pp2 = _fit(base + ALL4, Xb2, yb2, X, teh, len(base) + 4)
    dl, lo, hi = paired_boot(y[teh], rp2, pp2)
    row["arms"]["B_restricted: values only"] = {
        "n_features": len(base) + 4, "delta": dl, "ci_lo": lo, "ci_hi": hi,
        "clears": bool(lo > 0), "n_test": int(teh.sum()),
        "auc": float(roc_auc_score(y[teh], pp2))}
    return row


def main():
    print("=" * 100)
    print("DENSITY CONTROLS -- does arm B's +0.0091 survive sky and missingness?")
    print("=" * 100)
    _init()
    print(f"  base {len(_G['base'])}; restricted population "
          f"{int(_G['have'].sum())}/{len(_G['have'])} stars have st_rho")
    print(f"  {N_RESAMPLES} bootstraps\n")

    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_resample, r): r for r in range(N_RESAMPLES)}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            print(f"  resample {futs[f]} done ({i}/{N_RESAMPLES}, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
    rows.sort(key=lambda r: r["rep"])

    names = list(rows[0]["arms"].keys())
    print("\n" + "=" * 100)
    print(f"  {'arm':<30}{'nfeat':>6}{'mean d':>10}{'sd':>8}{'min':>9}"
          f"{'max':>9}{'pos':>7}{'clr':>6}{'>=MDE':>7}")
    out = {"n_resamples": len(rows), "rows": rows, "summary": {}}
    for n in names:
        d = np.array([r["arms"][n]["delta"] for r in rows])
        c = sum(r["arms"][n]["clears"] for r in rows)
        nf = rows[0]["arms"][n]["n_features"]
        print(f"  {n:<30}{nf:>6}{d.mean():>+10.4f}{d.std():>8.4f}{d.min():>+9.4f}"
              f"{d.max():>+9.4f}{int((d>0).sum()):>4}/{len(rows)}{c:>3}/{len(rows)}"
              f"{int((d>=MDE).sum()):>4}/{len(rows)}")
        out["summary"][n] = {
            "n_features": nf, "delta_mean": float(d.mean()), "delta_sd": float(d.std()),
            "delta_min": float(d.min()), "delta_max": float(d.max()),
            "n_positive": int((d > 0).sum()), "n_clearing": c,
            "n_at_or_above_mde": int((d >= MDE).sum())}
    print("=" * 100)
    json.dump(out, open(RESULTS, "w"), indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
