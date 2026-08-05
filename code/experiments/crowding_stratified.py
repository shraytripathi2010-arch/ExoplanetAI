"""crowding_stratified.py -- does crowding help INSIDE a matched sky region?

THE QUESTION PROMOTION ACTUALLY TURNS ON

Crowding beyond galactic position is +0.0120. But that is measured by handing
the model |b| as a feature and reading the increment -- which removes the
latitude axis without removing the fact that positives and negatives were drawn
from different parts of the sky in the first place.

The cleaner test is to delete the difference instead of modelling it: restrict
BOTH classes to a common |galactic b| band, where planets and false positives
occupy the same sky, and ask whether crowding still helps.

  * if the effect survives at roughly its full size, the mechanism is physical
    (crowded field -> blend -> false positive) and should transfer to new
    candidates
  * if it collapses toward zero, the effect was the sky-region difference and
    would not transfer

POWER, STATED UP FRONT. Restricting to a band roughly halves the evaluation
set, so the detection threshold for THIS test rises to about 0.0097 * sqrt(2)
~= 0.014. A +0.012 effect is therefore not expected to clear here even if
entirely real. The point estimate is the informative quantity, not the
clearance flag, and it is reported as such.
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
CROWD = os.path.join(SCRIPT_DIR, "crowding_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "crowding_stratified_results.json")

SEED, N_RESAMPLES, N_BOOT, N_WORKERS = 42, 12, 1500, 6
B_LO, B_HI = 8.0, 40.0        # band holding a workable share of both classes
CROWD_COLS = ["crowd_flux_ratio_max", "crowd_nearest_arcsec"]


def paired_boot(y, pa, pb, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        yi = y[i]
        s = yi.sum()
        if s == 0 or s == len(yi):
            continue
        d.append(fast_auc(yi, pb[i]) - fast_auc(yi, pa[i]))
    d = np.asarray(d)
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


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
    crowd = pd.read_csv(CROWD)
    merged = df[["host"]].merge(crowd[["host"] + CROWD_COLS], on="host", how="left")
    for c in CROWD_COLS:
        X[c] = pd.to_numeric(merged[c], errors="coerce").to_numpy()

    ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy()
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy()
    gb = np.full(len(df), np.nan)
    ok = np.isfinite(ra) & np.isfinite(dec)
    gb[ok] = np.abs(SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg).galactic.b.deg)

    band = np.isfinite(gb) & (gb >= B_LO) & (gb <= B_HI)
    tr, _ = m05.split_by_host(df)
    frozen = m05.frozen_test_mask(df)
    _G.update(X=X, y=y, gb=gb, band=band,
              tr=np.asarray(tr) & band, te=frozen & band,
              base_cols=list(m05.FEATURE_COLUMNS),
              hgb=clone(getattr(joblib.load(PROD), "estimator",
                                joblib.load(PROD))))


def _fit(cols, Xb, yb, X, te):
    est = CalibratedClassifierCV(clone(_G["hgb"]), cv=5, method="sigmoid")
    est.fit(Xb[cols], yb)
    return est.predict_proba(X.loc[te, cols])[:, 1]


def run_resample(rep):
    if not _G:
        _init()
    X, y, tr, te = _G["X"], _G["y"], _G["tr"], _G["te"]
    base = _G["base_cols"]
    Xtr, ytr = X[tr], y[tr]
    rng = np.random.RandomState(1000 + rep)
    idx = rng.randint(0, len(ytr), len(ytr))
    Xb, yb = Xtr.iloc[idx], ytr[idx]
    b = _fit(base, Xb, yb, X, te)
    c = _fit(base + CROWD_COLS, Xb, yb, X, te)
    d, lo, hi = paired_boot(y[te], b, c)
    return {"rep": rep, "baseline_auc": float(roc_auc_score(y[te], b)),
            "crowd_auc": float(roc_auc_score(y[te], c)),
            "delta": d, "clears": bool(lo > 0)}


def main():
    print("=" * 96)
    print(f"CROWDING INSIDE A MATCHED SKY BAND  |b| in [{B_LO}, {B_HI}] deg")
    print("=" * 96)
    _init()
    y, band, tr, te = _G["y"], _G["band"], _G["tr"], _G["te"]
    n_tr, n_te = int(tr.sum()), int(te.sum())
    print(f"  band retains {int(band.sum())}/{len(y)} stars "
          f"({band.mean()*100:.0f}%)   train {n_tr}   test {n_te}")
    print(f"  class balance in band: {int(y[band].sum())} planets / "
          f"{int((1-y[band]).sum())} false positives")
    gbb = _G["gb"][band]
    print(f"  median |b| in band: planets {np.median(gbb[y[band]==1]):.1f} deg, "
          f"false positives {np.median(gbb[y[band]==0]):.1f} deg  "
          f"(was 16.9 vs 12.5 unrestricted)")
    mde_band = 0.0097 * np.sqrt(1098.0 / max(n_te, 1))
    print(f"  detection threshold rescaled for n={n_te}: ~{mde_band:.4f}\n")

    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_resample, r): r for r in range(N_RESAMPLES)}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            print(f"  resample {futs[f]} done ({i}/{N_RESAMPLES}, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
    rows.sort(key=lambda r: r["rep"])
    d = np.array([r["delta"] for r in rows])
    cl = sum(r["clears"] for r in rows)

    print("\n" + "=" * 96)
    print(f"  crowding delta INSIDE the band: {d.mean():+.4f} "
          f"(sd {d.std():.4f}, range {d.min():+.4f}..{d.max():+.4f})")
    print(f"  positive {int((d>0).sum())}/{len(rows)}   clears {cl}/{len(rows)}")
    print(f"\n  unrestricted crowding-beyond-position estimate was +0.0120")
    ratio = d.mean() / 0.0120 if d.mean() > 0 else 0.0
    print(f"  retained inside a matched sky region: {ratio*100:.0f}%")
    if d.mean() > 0.008:
        v = "SURVIVES -- effect is physical, not the sky-region difference"
    elif d.mean() > 0.004:
        v = "PARTIALLY survives -- physical component present but reduced"
    else:
        v = "COLLAPSES -- the effect was largely the sky-region difference"
    print(f"  reading: {v}")
    print("=" * 96)
    json.dump({"band": [B_LO, B_HI], "n_train": n_tr, "n_test": n_te,
               "mde_rescaled": float(mde_band), "delta_mean": float(d.mean()),
               "delta_sd": float(d.std()), "n_positive": int((d > 0).sum()),
               "n_clearing": cl, "fraction_retained": float(ratio),
               "verdict": v, "rows": rows}, open(RESULTS, "w"), indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
