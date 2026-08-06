"""stellar_variability_controls.py -- the controls that decide whether arm D is real.

WHY THIS EXISTS. The first validation pass returned:

    A: +var_excess      +0.0043    6/12 clear
    B: +var_ls_power    +0.0060    7/12 clear
    C: +clean trio      +0.0086    9/12 clear   3/12 >= MDE
    D: +all five        +0.0101   12/12 clear   8/12 >= MDE
    E: +pair | sky      +0.0071   10/12 clear

Arm D clears on every resample and its mean exceeds the 0.0097 detection
threshold. Three things must be checked before that can be believed, and the
first is a gap in my own arm design.

1. THE SPATIAL CONTROL WAS RUN ON THE WRONG ARM. Arm E holds sky constant for
   `var_excess` and `var_ls_power` -- the two metrics with the SMALLEST sky
   exposure (+0.121 and -0.010 vs |galactic b|). Arm D additionally contains
   `var_oot_rms` (-0.296) and `var_ls_amp` (-0.286), the two largest spatial
   correlations measured anywhere in this project. So the winning arm has never
   been tested against the confound it is most exposed to. `D_sky` fixes that.

2. IS IT JUST chi2red_min RE-DERIVED? `var_oot_rms` correlates 0.967 with
   `chi2red_min`, the model's #3 feature by permutation importance. Arm D minus
   `var_oot_rms` is arm C, which drops from +0.0101 to +0.0086 -- so the
   redundant column carries roughly 15% of the gain, not all of it. `D_nooot`
   re-runs that comparison directly against the same reference for a clean
   attribution rather than inferring it across two separately-referenced arms.

3. MISSINGNESS. Coverage is 99.7% with availability-AUC 0.5057, so this is
   expected to be null. `D_avail` runs it anyway, because this project's
   corrected heuristic requires the indicator-only control rather than an
   argument that it cannot matter.
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
RESULTS = os.path.join(SCRIPT_DIR, "stellar_variability_controls_results.json")

SEED, N_RESAMPLES, N_BOOT, N_WORKERS = 42, 12, 1500, 6
MDE = 0.0097
ALL5 = ["var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]
NOOOT = ["var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]
AVAIL5 = [c + "_avail" for c in ALL5]
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
    merged = df[["host"]].merge(v[["host"] + ALL5], on="host", how="left")
    for c in ALL5:
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
    prod = joblib.load(PROD)
    _G.update(X=X, y=y, tr=np.asarray(tr), te=m05.frozen_test_mask(df),
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
    X, y, tr, te = _G["X"], _G["y"], _G["tr"], _G["te"]
    base = _G["base"]
    Xtr, ytr = X[tr], y[tr]
    rng = np.random.RandomState(1000 + rep)
    idx = rng.randint(0, len(ytr), len(ytr))
    Xb, yb = Xtr.iloc[idx], ytr[idx]
    row = {"rep": rep, "arms": {}}

    bp = _fit(base, Xb, yb, X, te, len(base))
    row["base_auc"] = float(roc_auc_score(y[te], bp))

    # 1. THE arm that clears, with sky held constant
    ref = base + [SKY]
    rp = _fit(ref, Xb, yb, X, te, len(ref))
    pp = _fit(ref + ALL5, Xb, yb, X, te, len(ref) + 5)
    dl, lo, hi = paired_boot(y[te], rp, pp)
    row["arms"]["D_sky: +all five | sky"] = {
        "n_features": len(ref) + 5, "delta": dl, "ci_lo": lo, "ci_hi": hi,
        "clears": bool(lo > 0), "auc": float(roc_auc_score(y[te], pp))}

    # 2. drop the chi2red_min-redundant column, same reference as D
    pp2 = _fit(base + NOOOT, Xb, yb, X, te, len(base) + 4)
    dl, lo, hi = paired_boot(y[te], bp, pp2)
    row["arms"]["D_nooot: drop var_oot_rms"] = {
        "n_features": len(base) + 4, "delta": dl, "ci_lo": lo, "ci_hi": hi,
        "clears": bool(lo > 0), "auc": float(roc_auc_score(y[te], pp2))}

    # 3. indicators only
    pp3 = _fit(base + AVAIL5, Xb, yb, X, te, len(base) + 5)
    dl, lo, hi = paired_boot(y[te], bp, pp3)
    row["arms"]["D_avail: indicators ONLY"] = {
        "n_features": len(base) + 5, "delta": dl, "ci_lo": lo, "ci_hi": hi,
        "clears": bool(lo > 0), "auc": float(roc_auc_score(y[te], pp3))}
    return row


def main():
    print("=" * 100)
    print("VARIABILITY CONTROLS -- does arm D's +0.0101 survive sky, redundancy, missingness?")
    print("=" * 100)
    _init()
    print(f"  base {len(_G['base'])}, {N_RESAMPLES} bootstraps\n")

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
    print(f"  {'arm':<32}{'nfeat':>6}{'mean d':>10}{'sd':>8}{'min':>9}"
          f"{'max':>9}{'pos':>7}{'clr':>6}{'>=MDE':>7}")
    out = {"n_resamples": len(rows), "rows": rows, "summary": {}}
    for n in names:
        d = np.array([r["arms"][n]["delta"] for r in rows])
        c = sum(r["arms"][n]["clears"] for r in rows)
        nf = rows[0]["arms"][n]["n_features"]
        print(f"  {n:<32}{nf:>6}{d.mean():>+10.4f}{d.std():>8.4f}{d.min():>+9.4f}"
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
