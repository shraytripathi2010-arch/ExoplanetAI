"""giant_star_diagnose.py -- re-measure the giant-star blind spot on the
CURRENTLY DEPLOYED 0.9208 model, and ask WHY before proposing any fix.

Diagnostic only: retrains nothing, touches nothing.

The claim under test comes from `audit_calibration_threshold_errors.py`, which
measured a 5.8% error rate for stars that are neither large-radius nor
high-SDE, versus 21.1% for the cell that is BOTH -- a 3.6x ratio. That was
measured on the PREVIOUS 24-feature 0.9031 model. Two things could have moved
it since: the crowding features were deployed, and crowding is physically
linked to blends, which is exactly the failure mode blamed for the giant-star
errors. So it is re-measured rather than assumed.

Error is defined identically to the original audit: prediction at the 0.5
operating point, on the frozen test set.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import fast_auc  # noqa: E402

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "giant_star_diagnose.json")

RAD_CUT, SDE_CUT, OPERATING_POINT = 1.5, 10.0, 0.5


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def auc_or_nan(y, p):
    y = np.asarray(y)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return float("nan")
    return fast_auc(y, np.asarray(p))


def main():
    m05 = _m05()
    df = pd.read_csv(TRAINING)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    te = m05.frozen_test_mask(df)
    prod = joblib.load(PROD)
    p = prod.predict_proba(X[te])[:, 1]
    yte = y[te]
    dte = df[te].reset_index(drop=True)
    pred = (p >= OPERATING_POINT).astype(int)
    err = (pred != yte)
    rad = pd.to_numeric(dte["st_rad"], errors="coerce").to_numpy()
    sde = pd.to_numeric(dte["SDE"], errors="coerce").to_numpy()

    out = {"n_test": int(te.sum()), "overall_auc": float(fast_auc(yte, p)),
           "overall_error_pct": float(err.mean() * 100)}
    print("=" * 92)
    print("GIANT-STAR BLIND SPOT, RE-MEASURED ON THE DEPLOYED 0.9208 MODEL")
    print("=" * 92)
    print(f"  frozen test n={out['n_test']}   overall AUC {out['overall_auc']:.4f}"
          f"   overall error {out['overall_error_pct']:.1f}% at p>={OPERATING_POINT}")
    print(f"  st_rad available for {np.isfinite(rad).mean()*100:.1f}% of test stars")

    # ---- 1. the original 2x2 claim, re-measured -------------------------
    print("\n" + "-" * 92)
    print(f"1. THE ORIGINAL 2x2 CLAIM (st_rad >= {RAD_CUT}, SDE >= {SDE_CUT})")
    print("-" * 92)
    print(f"  {'cell':<34}{'n':>7}{'error %':>10}{'AUC':>9}{'planets %':>11}")
    cells, ok = {}, np.isfinite(rad) & np.isfinite(sde)
    for name, mask in [
            ("neither (small rad, low SDE)", ok & (rad < RAD_CUT) & (sde < SDE_CUT)),
            ("large radius only", ok & (rad >= RAD_CUT) & (sde < SDE_CUT)),
            ("high SDE only", ok & (rad < RAD_CUT) & (sde >= SDE_CUT)),
            ("BOTH (the blind spot)", ok & (rad >= RAD_CUT) & (sde >= SDE_CUT))]:
        n = int(mask.sum())
        e = float(err[mask].mean() * 100) if n else float("nan")
        a = auc_or_nan(yte[mask], p[mask])
        pl = float(yte[mask].mean() * 100) if n else float("nan")
        print(f"  {name:<34}{n:>7}{e:>10.1f}{a:>9.4f}{pl:>11.1f}")
        cells[name] = {"n": n, "error_pct": e, "auc": None if np.isnan(a) else a,
                       "planet_pct": pl}
    base = cells["neither (small rad, low SDE)"]["error_pct"]
    both = cells["BOTH (the blind spot)"]["error_pct"]
    ratio = both / base if base else float("nan")
    print(f"\n  ratio BOTH / neither = {ratio:.2f}x   (previously reported 3.6x "
          f"on the 24-feature 0.9031 model)")
    out["cells"], out["ratio_both_vs_neither"] = cells, ratio

    # ---- 2. st_rad alone, which is what a model-level fix would target --
    print("\n" + "-" * 92)
    print("2. BY st_rad BUCKET ALONE (what a radius-targeted fix would address)")
    print("-" * 92)
    print(f"  {'bucket':<22}{'n':>7}{'% of test':>11}{'error %':>10}{'AUC':>9}"
          f"{'planets %':>11}")
    buckets, edges = {}, [(0, 1.5), (1.5, 3), (3, 10), (10, np.inf)]
    for lo, hi in edges:
        mask = np.isfinite(rad) & (rad >= lo) & (rad < hi)
        n = int(mask.sum())
        if not n:
            continue
        e = float(err[mask].mean() * 100)
        a = auc_or_nan(yte[mask], p[mask])
        lbl = f"{lo} <= st_rad < {hi}" if np.isfinite(hi) else f"st_rad >= {lo}"
        print(f"  {lbl:<22}{n:>7}{n/len(yte)*100:>11.1f}{e:>10.1f}{a:>9.4f}"
              f"{yte[mask].mean()*100:>11.1f}")
        buckets[lbl] = {"n": n, "pct_of_test": n / len(yte) * 100, "error_pct": e,
                        "auc": None if np.isnan(a) else a,
                        "planet_pct": float(yte[mask].mean() * 100)}
    out["buckets"] = buckets

    giant = np.isfinite(rad) & (rad >= RAD_CUT)
    out["giant_n"] = int(giant.sum())
    out["giant_pct"] = float(giant.mean() * 100)
    out["giant_error_pct"] = float(err[giant].mean() * 100)
    out["dwarf_error_pct"] = float(err[np.isfinite(rad) & (rad < RAD_CUT)].mean() * 100)
    out["giant_auc"] = auc_or_nan(yte[giant], p[giant])
    print(f"\n  st_rad >= {RAD_CUT}: {out['giant_n']} stars = "
          f"{out['giant_pct']:.1f}% of the test set, error {out['giant_error_pct']:.1f}% "
          f"vs {out['dwarf_error_pct']:.1f}% for dwarfs "
          f"({out['giant_error_pct']/out['dwarf_error_pct']:.2f}x), "
          f"AUC {out['giant_auc']:.4f}")

    # ---- 3. WHY: how the population itself differs ----------------------
    print("\n" + "-" * 92)
    print("3. MECHANISM -- how giants differ, beyond being harder")
    print("-" * 92)
    print(f"  {'feature':<26}{'dwarf median':>15}{'giant median':>15}{'ratio':>9}")
    mech = {}
    for c in ["depth", "duration", "snr", "SDE", "period", "rp_rs",
              "depth_mean_std", "chi2red_min", "secondary_eclipse_depth",
              "odd_even_mismatch", "crowd_nearest_arcsec", "crowd_flux_ratio_max"]:
        if c not in X.columns:
            continue
        v = pd.to_numeric(X[te][c], errors="coerce").to_numpy()
        d_med = float(np.nanmedian(v[np.isfinite(rad) & (rad < RAD_CUT)]))
        g_med = float(np.nanmedian(v[giant]))
        r = g_med / d_med if d_med not in (0.0,) and np.isfinite(d_med) else float("nan")
        print(f"  {c:<26}{d_med:>15.5g}{g_med:>15.5g}{r:>9.2f}")
        mech[c] = {"dwarf_median": d_med, "giant_median": g_med, "ratio": r}
    out["mechanism"] = mech

    # class balance is itself a mechanism candidate
    print(f"\n  class balance: dwarfs {yte[np.isfinite(rad)&(rad<RAD_CUT)].mean()*100:.1f}% "
          f"planets, giants {yte[giant].mean()*100:.1f}% planets")
    print("  (a different prior alone shifts the error rate at a FIXED 0.5 threshold,")
    print("   independently of whether the ranking within giants is any worse)")

    json.dump(out, open(RESULTS, "w"), indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
