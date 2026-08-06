"""variability_worstcase_control.py -- every control at once, the most
conservative test available before a deployment decision.

The three controls already run were run SEPARATELY:

    D_sky     sky held constant                +0.0098   12/12 clear
    D_avail   indicators only, no values       +0.0000    0/12
    D_nooot   drop the redundant column        +0.0088   10/12 clear

Passing three separate controls is not the same as passing all of them at once.
Any two could be absorbing different parts of the same confound, and the
combination could be weaker than the weakest individually. This runs the
worst case:

  REFERENCE  26 production
             + abs_gal_b                (sky already available to the model)
             + five availability indicators (missingness already available)
  TEST       the same, plus the five measured VALUES

  and BOTH are fitted and evaluated ONLY on the restricted population where all
  five values are present, so missingness is constant by construction and
  cannot carry information at all.

Whatever survives that is attributable to the measured variability values and
nothing else. The restricted test set is smaller, so the CI is wider -- this
answers "is the effect real", not "how big is it".

A second arm repeats the same worst case WITHOUT `var_oot_rms`, since that
column correlates 0.967 with `chi2red_min`. If the four-column version also
survives, the result does not depend on a near-duplicate of an existing
feature.
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
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import fast_auc  # noqa: E402

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
VFEAT = os.path.join(SCRIPT_DIR, "stellar_variability_features.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "variability_worstcase_results.json")

SEED, N_RESAMPLES, N_BOOT, N_WORKERS = 42, 12, 1500, 6
MDE = 0.0097
VAR5 = ["var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]
VAR4 = ["var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]
AVAIL5 = [c + "_avail" for c in VAR5]
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

    v = pd.read_csv(VFEAT)
    merged = df[["host"]].merge(v[["host"] + VAR5], on="host", how="left")
    for c in VAR5:
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
    te = m05.frozen_test_mask(df)
    have = np.isfinite(X[VAR5]).all(axis=1).to_numpy()
    prod = joblib.load(PROD)
    _G.update(X=X, y=y, tr=np.asarray(tr), te=te, have=have,
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

    # restricted population: all five values present -> missingness constant
    trh, teh = tr & have, te & have
    Xtr, ytr = X[trh], y[trh]
    rng = np.random.RandomState(1000 + rep)
    idx = rng.randint(0, len(ytr), len(ytr))
    Xb, yb = Xtr.iloc[idx], ytr[idx]

    # reference already contains sky AND the availability indicators
    ref = base + [SKY] + AVAIL5
    rp = _fit(ref, Xb, yb, X, teh, len(ref))
    row = {"rep": rep, "n_test": int(teh.sum()),
           "ref_auc": float(roc_auc_score(y[teh], rp)), "arms": {}}

    for label, extra in [("WORST CASE: +all five values", VAR5),
                         ("WORST CASE: +four (no var_oot_rms)", VAR4)]:
        pp = _fit(ref + extra, Xb, yb, X, teh, len(ref) + len(extra))
        d, lo, hi = paired_boot(y[teh], rp, pp)
        row["arms"][label] = {
            "n_features": len(ref) + len(extra), "delta": d, "ci_lo": lo,
            "ci_hi": hi, "clears": bool(lo > 0),
            "auc": float(roc_auc_score(y[teh], pp))}
    return row


def main():
    print("=" * 100)
    print("WORST-CASE CONTROL -- sky + availability + restricted population, all at once")
    print("=" * 100)
    _init()
    print(f"  restricted population: {int(_G['have'].sum())}/{len(_G['have'])} stars "
          f"have all five values")
    print(f"  reference = 26 + abs_gal_b + 5 availability indicators = 32 columns")
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
    print(f"  n_test = {rows[0]['n_test']} (restricted frozen test set)")
    print(f"  {'arm':<38}{'nfeat':>6}{'mean d':>10}{'sd':>8}{'min':>9}"
          f"{'max':>9}{'pos':>7}{'clr':>6}{'>=MDE':>7}")
    out = {"n_resamples": len(rows), "n_test": rows[0]["n_test"],
           "rows": rows, "summary": {}}
    for n in names:
        d = np.array([r["arms"][n]["delta"] for r in rows])
        c = sum(r["arms"][n]["clears"] for r in rows)
        nf = rows[0]["arms"][n]["n_features"]
        print(f"  {n:<38}{nf:>6}{d.mean():>+10.4f}{d.std():>8.4f}{d.min():>+9.4f}"
              f"{d.max():>+9.4f}{int((d>0).sum()):>4}/{len(rows)}{c:>3}/{len(rows)}"
              f"{int((d>=MDE).sum()):>4}/{len(rows)}")
        out["summary"][n] = {
            "n_features": nf, "delta_mean": float(d.mean()),
            "delta_sd": float(d.std()), "delta_min": float(d.min()),
            "delta_max": float(d.max()), "n_positive": int((d > 0).sum()),
            "n_clearing": c, "n_at_or_above_mde": int((d >= MDE).sum())}
    print("=" * 100)
    json.dump(out, open(RESULTS, "w"), indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
