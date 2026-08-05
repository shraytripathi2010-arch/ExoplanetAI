"""crowding_control.py -- is the crowding gain physics, or just sky position?

WHY THIS RUN EXISTS

The crowding arms cleared decisively: +0.0167 (12/12 resamples, all above the
0.0097 detection threshold), which no other feature experiment in this project
has managed. That is exactly when to look hardest for a confound.

The confound is spatial. Measured on the training set:

    |galactic latitude| alone   AUC 0.6287   <-- HIGHER than the feature
    right ascension alone       AUC 0.6032
    crowd_nearest_arcsec        AUC 0.6039

    median |b|:              planets 16.8 deg, false positives 12.5 deg
    median nearest neighbour: planets 10.3",  false positives  7.4"

The two classes come from different places on the sky. Confirmed planets skew
to well-studied, high-latitude, uncrowded fields; TOI false positives are
all-sky TESS detections concentrated toward the galactic plane. Stellar density
rises sharply toward the plane, so "how crowded is this star" partly restates
"which survey found this star" -- a property of how the training set was
assembled, not of the astrophysics.

THE TEST

    +2 crowding      the original arms
    +|b| only        sky position alone, no crowding at all
    +|b| + crowding  both

Reading it:
  * if `+|b| only` reproduces roughly the crowding gain, the crowding features
    are largely a spatial proxy and the result is provenance, not blend physics
  * if `+|b| + crowding` beats `+|b| only` by a clear margin, crowding carries
    information beyond position and the result survives

Sky position is deliberately NOT proposed as a production feature. It encodes
survey coverage, would not generalise to new candidates drawn differently, and
is included here only as a control.
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
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
CROWD = os.path.join(SCRIPT_DIR, "crowding_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "crowding_control_results.json")

SEED = 42
N_RESAMPLES = 12
N_BOOT = 1500
N_WORKERS = 6
DETECTION_THRESHOLD = 0.0097

CROWD_COLS = ["crowd_flux_ratio_max", "crowd_nearest_arcsec"]
POS_COLS = ["sky_abs_galactic_b"]
ARMS = [("+2 crowding", CROWD_COLS),
        ("+|b| only (control)", POS_COLS),
        ("+|b| + crowding", POS_COLS + CROWD_COLS)]


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
    b = np.full(len(df), np.nan)
    ok = np.isfinite(ra) & np.isfinite(dec)
    b[ok] = np.abs(SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg).galactic.b.deg)
    X["sky_abs_galactic_b"] = b

    tr, _ = m05.split_by_host(df)
    frozen = m05.frozen_test_mask(df)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    prod = joblib.load(PROD)
    _G.update(X=X, y=y, tr=np.asarray(tr), te=frozen, te2=frozen & is2,
              base_cols=list(m05.FEATURE_COLUMNS),
              hgb=clone(getattr(prod, "estimator", prod)))


def _fit(cols, Xb, yb, X, te, te2):
    est = CalibratedClassifierCV(clone(_G["hgb"]), cv=5, method="sigmoid")
    est.fit(Xb[cols], yb)
    return (est.predict_proba(X.loc[te, cols])[:, 1],
            est.predict_proba(X.loc[te2, cols])[:, 1])


def run_resample(rep):
    if not _G:
        _init()
    X, y, tr, te, te2 = _G["X"], _G["y"], _G["tr"], _G["te"], _G["te2"]
    base = _G["base_cols"]
    Xtr, ytr = X[tr], y[tr]
    rng = np.random.RandomState(1000 + rep)
    idx = rng.randint(0, len(ytr), len(ytr))
    Xb, yb = Xtr.iloc[idx], ytr[idx]

    bF, b2 = _fit(base, Xb, yb, X, te, te2)
    row = {"rep": rep, "baseline_full": float(roc_auc_score(y[te], bF)), "arms": {}}
    for name, extra in ARMS:
        pF, p2 = _fit(base + extra, Xb, yb, X, te, te2)
        dF, loF, hiF = paired_boot(y[te], bF, pF)
        d2, lo2, hi2 = paired_boot(y[te2], b2, p2)
        row["arms"][name] = {
            "auc_full": float(roc_auc_score(y[te], pF)),
            "brier": float(brier_score_loss(y[te], pF)),
            "delta_full": dF, "clears_full": bool(loF > 0),
            "delta_2min": d2, "clears_2min": bool(lo2 > 0)}
    return row


def main():
    print("=" * 104)
    print("CONTROL -- is the crowding gain physics, or sky position?")
    print("=" * 104)
    for n, e in ARMS:
        print(f"    {n:<22} {e}")
    print(f"\n  {N_RESAMPLES} training bootstraps, baseline refit on the same rows.\n")

    _init()
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_resample, r): r for r in range(N_RESAMPLES)}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            json.dump(rows, open(RESULTS + ".partial", "w"), default=float)
            print(f"  resample {futs[f]} done ({i}/{N_RESAMPLES}, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
    rows.sort(key=lambda r: r["rep"])

    bf = np.array([r["baseline_full"] for r in rows])
    print("\n" + "=" * 104)
    print(f"  baseline mean AUC {bf.mean():.4f} (sd {bf.std():.4f})\n")
    print(f"  {'arm':<24}{'mean d':>10}{'sd':>8}{'min':>9}{'max':>9}"
          f"{'pos':>7}{'clears':>8}{'>=MDE':>7}{'d_2min':>9}")
    out = {"n_resamples": len(rows), "baseline_mean_auc": float(bf.mean()),
           "rows": rows, "summary": {}}
    for name, _ in ARMS:
        dF = np.array([r["arms"][name]["delta_full"] for r in rows])
        d2 = np.array([r["arms"][name]["delta_2min"] for r in rows])
        cF = sum(r["arms"][name]["clears_full"] for r in rows)
        print(f"  {name:<24}{dF.mean():>+10.4f}{dF.std():>8.4f}{dF.min():>+9.4f}"
              f"{dF.max():>+9.4f}{int((dF>0).sum()):>4}/{len(rows)}"
              f"{cF:>5}/{len(rows)}{int((dF>=DETECTION_THRESHOLD).sum()):>4}/{len(rows)}"
              f"{d2.mean():>+9.4f}")
        out["summary"][name] = {
            "delta_full_mean": float(dF.mean()), "delta_full_sd": float(dF.std()),
            "n_positive": int((dF > 0).sum()), "n_clearing": cF,
            "n_at_or_above_mde": int((dF >= DETECTION_THRESHOLD).sum()),
            "delta_2min_mean": float(d2.mean())}

    c_only = out["summary"]["+2 crowding"]["delta_full_mean"]
    b_only = out["summary"]["+|b| only (control)"]["delta_full_mean"]
    both = out["summary"]["+|b| + crowding"]["delta_full_mean"]
    print("\n" + "=" * 104)
    print("READING")
    print("=" * 104)
    print(f"  crowding alone      {c_only:+.4f}")
    print(f"  position alone      {b_only:+.4f}   <- if close to crowding, it is a spatial proxy")
    print(f"  both together       {both:+.4f}")
    print(f"  crowding ON TOP of position: {both - b_only:+.4f}")
    out["crowding_beyond_position"] = float(both - b_only)
    print("=" * 104)
    json.dump(out, open(RESULTS, "w"), indent=2, default=float)
    if os.path.exists(RESULTS + ".partial"):
        os.remove(RESULTS + ".partial")
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
