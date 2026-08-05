"""trapezoid_validate.py -- resampled validation of the fitted trapezoid shape
metric against the deployed 0.9208 / 26-feature model.

THE CONFOUND THIS SCRIPT IS BUILT AROUND

The pre-model checks passed the class-rate gate decisively -- false positives
are measurably more V-shaped (median vshape 0.578 vs 0.435, single-feature AUC
0.3595, the strongest separation any candidate feature has shown in this
project) -- and simultaneously turned up a serious problem:

    usable rate    planets 25.2%   false positives 53.4%   gap -28.1pp
    AUC of AVAILABILITY ALONE                              0.3593

Availability is as predictive as the feature itself, to three decimals. A
trapezoid fit is only usable when the event is deep enough to constrain an
ingress, and eclipsing binaries are deep while planets are shallow -- so "this
star has a usable shape fit" is largely a restatement of depth. Confirmed
directly: availability is predicted by `snr` at AUC 0.856, `SDE` at 0.824 and
`duration` at 0.823, all three ALREADY among the 26 production features.

HOW MISSINGNESS ACTUALLY ENTERS THIS RECIPE -- checked, not assumed. The
deployed estimator is a Pipeline of SimpleImputer(strategy="median",
add_indicator=False) -> HistGradientBoostingClassifier. NaN therefore never
reaches the classifier: unusable stars are given the training median. Bare HGB
would read NaN natively as a signal; this pipeline does not, because the
imputer erases it first. So the missingness channel is largely closed by
construction, which makes arm C the only clean measurement of what the
availability pattern is actually worth, and arm D the only clean measurement of
what the shape VALUE is worth once availability is already known.

A raw improvement therefore proves nothing on its own. The arms are built so
the value and its availability can be told apart.

ARMS -- each measured against a reference identical except for the test columns
  A  + vshape                          the naive version: value AND missingness
  B  + vshape, t14_ratio               plus fitted-vs-TLS duration
  C  + availability indicator ONLY     no shape value at all. If this moves the
                                       model as much as A, then A is the
                                       missingness pattern, not the shape.
  D  + vshape, GIVEN availability      the decisive test: reference already has
                                       the indicator, so the delta is what the
                                       measured SHAPE buys on top of knowing
                                       whether a fit exists
  E  + vshape, GIVEN |galactic b|      spatial control. This project's training
                                       set IS spatially confounded (|b| alone
                                       buys +0.0135 and clears 12/12), so any
                                       feature can borrow that. Checked, not
                                       assumed, per the giant-star lesson.

COLUMN-COUNT ASSERTION. Every fit asserts its feature count before fitting.
`build_feature_matrix` selects only `FEATURE_COLUMNS`, so a merged column that
never reaches X produces a perfectly null result for a comparison that never
differed -- the exact bug that first made the closed flux-statistics run look
clean. The run aborts rather than returning a reassuring zero.
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
FEAT = os.path.join(SCRIPT_DIR, "trapezoid_shape_features.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "trapezoid_validate_results.json")

SEED, N_RESAMPLES, N_BOOT, N_WORKERS, N_ECE_BINS = 42, 12, 1500, 6, 15
MDE = 0.0097
MIN_DEPTH_SNR, MAX_VSHAPE_ERR = 3.0, 0.30
SKY = "sky_abs_galactic_b"

AVAIL = "trap_avail"

# (name, reference columns added to base, extra columns under test)
# Each arm is measured against a reference that is IDENTICAL except for the
# columns under test, so the delta attributes cleanly.
ARMS = [
    ("A: +vshape",          [],        ["trap_vshape"]),
    ("B: +vshape,t14r",     [],        ["trap_vshape", "trap_t14_ratio"]),
    ("C: availability ONLY", [],       [AVAIL]),
    ("D: +vshape | avail",  [AVAIL],   ["trap_vshape"]),
    ("E: +vshape | sky",    [SKY],     ["trap_vshape"]),
]


def ece(y, p, bins=N_ECE_BINS):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = np.linspace(0, 1, bins + 1)
    i = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    return float(sum((i == b).mean() * abs(y[i == b].mean() - p[i == b].mean())
                     for b in range(bins) if (i == b).any()))


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
    m05 = _m05()
    df = pd.read_csv(TRAINING)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    y = np.asarray(y)

    tf = pd.read_csv(FEAT)
    mg = df[["host"]].merge(tf, on="host", how="left")
    ok = (mg["trap_vshape"].notna() & (mg["trap_status"] == "ok")
          & (pd.to_numeric(mg["trap_depth_snr"], errors="coerce") >= MIN_DEPTH_SNR)
          & (pd.to_numeric(mg["trap_vshape_err"], errors="coerce") <= MAX_VSHAPE_ERR))
    for c in ("trap_vshape", "trap_t14_ratio"):
        X[c] = pd.to_numeric(mg[c], errors="coerce").where(ok).to_numpy()
    # the missingness pattern, made an explicit column so it can be given to a
    # reference arm and thereby subtracted out
    X[AVAIL] = ok.to_numpy().astype(float)

    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy()
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy()
    good = np.isfinite(ra) & np.isfinite(dec)
    gb = np.full(len(df), np.nan)
    gb[good] = np.abs(SkyCoord(ra[good] * u.deg, dec[good] * u.deg).galactic.b.deg)
    X[SKY] = gb

    tr, _ = m05.split_by_host(df)
    frozen = m05.frozen_test_mask(df)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    prod = joblib.load(PROD)
    _G.update(X=X, y=y, tr=np.asarray(tr), te=frozen, te2=frozen & is2,
              base=list(m05.FEATURE_COLUMNS),
              hgb=clone(getattr(prod, "estimator", prod)))


def _fit(cols, Xb, yb, X, te, te2, expect):
    assert len(cols) == expect, f"expected {expect} columns, got {len(cols)}"
    est = CalibratedClassifierCV(clone(_G["hgb"]), cv=5, method="sigmoid")
    est.fit(Xb[cols], yb)
    return (est.predict_proba(X.loc[te, cols])[:, 1],
            est.predict_proba(X.loc[te2, cols])[:, 1])


def run_resample(rep):
    if not _G:
        _init()
    X, y, tr, te, te2 = _G["X"], _G["y"], _G["tr"], _G["te"], _G["te2"]
    base = _G["base"]
    Xtr, ytr = X[tr], y[tr]
    rng = np.random.RandomState(1000 + rep)
    idx = rng.randint(0, len(ytr), len(ytr))
    Xb, yb = Xtr.iloc[idx], ytr[idx]

    cache = {}

    def fit_cached(cols):
        k = tuple(cols)
        if k not in cache:
            cache[k] = _fit(list(cols), Xb, yb, X, te, te2, len(cols))
        return cache[k]

    bF, b2 = fit_cached(base)
    row = {"rep": rep, "base_auc": float(roc_auc_score(y[te], bF)),
           "base_brier": float(brier_score_loss(y[te], bF)),
           "base_ece": ece(y[te], bF), "arms": {}}
    for name, ref_extra, test_extra in ARMS:
        ref_cols = base + ref_extra
        aF, a2 = fit_cached(ref_cols)
        cols = ref_cols + test_extra
        pF, p2 = _fit(cols, Xb, yb, X, te, te2, len(cols))
        d, lo, hi = paired_boot(y[te], aF, pF)
        d2, lo2, hi2 = paired_boot(y[te2], a2, p2)
        row["arms"][name] = {
            "n_features": len(cols), "n_ref_features": len(ref_cols),
            "auc": float(roc_auc_score(y[te], pF)),
            "ref_auc": float(roc_auc_score(y[te], aF)),
            "delta": d, "ci_lo": lo, "ci_hi": hi, "clears": bool(lo > 0),
            "delta_2min": d2, "clears_2min": bool(lo2 > 0),
            "brier": float(brier_score_loss(y[te], pF)), "ece": ece(y[te], pF)}
    return row


def main():
    print("=" * 108)
    print("FITTED TRAPEZOID SHAPE -- resampled validation vs the deployed 0.9208")
    print("=" * 108)
    _init()
    print(f"  base features: {len(_G['base'])}")
    for n, r_, e in ARMS:
        nb = len(_G["base"])
        print(f"    {n:<22} {nb+len(r_)} -> {nb+len(r_)+len(e)} features   test {e}"
              f"{'   [reference also has ' + ', '.join(r_) + ']' if r_ else ''}")
    print(f"  {N_RESAMPLES} bootstraps, production recipe refit on identical rows\n")

    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_resample, r): r for r in range(N_RESAMPLES)}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            json.dump(rows, open(RESULTS + ".partial", "w"), default=float)
            print(f"  resample {futs[f]} done ({i}/{N_RESAMPLES}, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
    rows.sort(key=lambda r: r["rep"])

    ba = np.array([r["base_auc"] for r in rows])
    bb = np.array([r["base_brier"] for r in rows])
    be = np.array([r["base_ece"] for r in rows])
    print("\n" + "=" * 108)
    print(f"  baseline: AUC {ba.mean():.4f} (sd {ba.std():.4f})  "
          f"Brier {bb.mean():.4f}  ECE {be.mean():.4f}\n")
    print(f"  {'arm':<20}{'nfeat':>7}{'mean d':>9}{'sd':>8}{'min':>9}{'max':>9}"
          f"{'pos':>7}{'clr':>6}{'>=MDE':>7}{'d_2min':>9}{'Brier':>9}{'ECE':>8}")
    out = {"n_resamples": len(rows), "baseline_auc": float(ba.mean()),
           "baseline_brier": float(bb.mean()), "baseline_ece": float(be.mean()),
           "rows": rows, "summary": {}}
    for name, _, _ in ARMS:
        d = np.array([r["arms"][name]["delta"] for r in rows])
        d2 = np.array([r["arms"][name]["delta_2min"] for r in rows])
        c = sum(r["arms"][name]["clears"] for r in rows)
        c2 = sum(r["arms"][name]["clears_2min"] for r in rows)
        br = np.mean([r["arms"][name]["brier"] for r in rows])
        ec = np.mean([r["arms"][name]["ece"] for r in rows])
        nf = rows[0]["arms"][name]["n_features"]
        print(f"  {name:<20}{nf:>7}{d.mean():>+9.4f}{d.std():>8.4f}{d.min():>+9.4f}"
              f"{d.max():>+9.4f}{int((d>0).sum()):>4}/{len(rows)}{c:>3}/{len(rows)}"
              f"{int((d>=MDE).sum()):>4}/{len(rows)}{d2.mean():>+9.4f}{br:>9.4f}{ec:>8.4f}")
        out["summary"][name] = {
            "n_features": nf, "delta_mean": float(d.mean()), "delta_sd": float(d.std()),
            "delta_min": float(d.min()), "delta_max": float(d.max()),
            "n_positive": int((d > 0).sum()), "n_clearing": c,
            "n_at_or_above_mde": int((d >= MDE).sum()),
            "delta_2min_mean": float(d2.mean()), "n_clearing_2min": c2,
            "mean_brier": float(br), "mean_ece": float(ec),
            "brier_vs_base": float(br - bb.mean()), "ece_vs_base": float(ec - be.mean())}
    print("=" * 108)
    json.dump(out, open(RESULTS, "w"), indent=2, default=float)
    if os.path.exists(RESULTS + ".partial"):
        os.remove(RESULTS + ".partial")
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
