"""stellar_variability_validate.py -- resampled validation of the variability
metrics that survived the Part 2 gate.

WHAT THE GATE RETURNED, and why the arms are shaped this way.

The stated prediction was that ALL metrics would score AUC below 0.5 (false
positives more variable). It held for two of five and INVERTED for three:

    var_oot_rms    0.6526   OPPOSITE   |r| 0.967 vs chi2red_min  -> REDUNDANT
    var_excess     0.4067   as predicted   |r| 0.576
    var_ls_amp     0.5927   OPPOSITE   |r| 0.785 vs depth_consistency_std
    var_ls_power   0.4168   as predicted   |r| 0.266
    var_ls_period  0.5889   OPPOSITE   |r| 0.162

That split is not noise, it is the confound named in advance in
`stellar_variability.py`. RAW scatter (`var_oot_rms`) runs the wrong way and is
0.967-correlated with `chi2red_min` -- the model's #3 feature by permutation
importance. It is not measuring stellar activity; it is re-deriving a noise
level the model already has. Scatter measured RELATIVE to each star's own
photometric error (`var_excess`) runs the predicted way. The two metrics
disagreeing is the diagnostic.

`var_oot_rms` is therefore reported and dropped, not modelled alone: at 0.967
it is over the 0.80 redundancy threshold by a wide margin.

ARMS, each against its own reference so deltas attribute cleanly:

  A  base -> + var_excess                  the noise-normalised metric
  B  base -> + var_ls_power                the most orthogonal (|r| 0.266)
  C  base -> + var_excess, var_ls_power, var_ls_period      the clean trio
  D  base -> + all five                    including the redundant ones,
                                           tested rather than assumed
  E  base + abs_gal_b -> + var_excess, var_ls_power
       THE SPATIAL CONTROL ARM. Required here, not optional: `var_oot_rms`
       and `var_ls_amp` carry the largest sky exposure measured anywhere in
       this project (Spearman -0.296 and -0.286 vs |galactic b|). A held-in
       arm, not a correlation coefficient.

Column counts are ASSERTED before each fit.
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
VFEAT = os.path.join(SCRIPT_DIR, "stellar_variability_features.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "stellar_variability_validate_results.json")

SEED, N_RESAMPLES, N_BOOT, N_WORKERS, N_ECE_BINS = 42, 12, 1500, 6, 15
MDE = 0.0097
ALL5 = ["var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]
TRIO = ["var_excess", "var_ls_power", "var_ls_period"]
SKY = "abs_gal_b"

ARMS = [
    ("A: +var_excess",      [],     ["var_excess"]),
    ("B: +var_ls_power",    [],     ["var_ls_power"]),
    ("C: +clean trio",      [],     TRIO),
    ("D: +all five",        [],     ALL5),
    ("E: +pair | sky",      [SKY],  ["var_excess", "var_ls_power"]),
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

    ra = pd.to_numeric(df["ra"], errors="coerce")
    dec = pd.to_numeric(df["dec"], errors="coerce")
    ok = ra.notna() & dec.notna()
    b = np.full(len(df), np.nan)
    b[ok.to_numpy()] = np.abs(
        SkyCoord(ra[ok].values * u.deg, dec[ok].values * u.deg).galactic.b.deg)
    X[SKY] = b

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

    row = {"rep": rep, "arms": {}}
    bF, b2 = _fit(base, Xb, yb, X, te, te2, len(base))
    row["base_auc"] = float(roc_auc_score(y[te], bF))
    row["base_brier"] = float(brier_score_loss(y[te], bF))
    row["base_ece"] = ece(y[te], bF)

    for name, ref_extra, test_extra in ARMS:
        ref_cols = base + ref_extra
        if ref_extra:
            rF, r2 = _fit(ref_cols, Xb, yb, X, te, te2, len(ref_cols))
        else:
            rF, r2 = bF, b2
        cols = ref_cols + test_extra
        pF, p2 = _fit(cols, Xb, yb, X, te, te2, len(ref_cols) + len(test_extra))
        dl, lo, hi = paired_boot(y[te], rF, pF)
        d2, lo2, _ = paired_boot(y[te2], r2, p2)
        row["arms"][name] = {
            "n_features": len(cols), "n_ref": len(ref_cols),
            "auc": float(roc_auc_score(y[te], pF)),
            "ref_auc": float(roc_auc_score(y[te], rF)),
            "delta": dl, "ci_lo": lo, "ci_hi": hi, "clears": bool(lo > 0),
            "delta_2min": d2, "clears_2min": bool(lo2 > 0),
            "brier": float(brier_score_loss(y[te], pF)), "ece": ece(y[te], pF)}
    return row


def main():
    print("=" * 110)
    print("STELLAR VARIABILITY -- resampled validation vs the deployed 0.9208 / 26 features")
    print("=" * 110)
    _init()
    print(f"  base features: {len(_G['base'])}")
    for n, r, t in ARMS:
        print(f"    {n:<20} ref {len(_G['base'])+len(r):>2} -> test "
              f"{len(_G['base'])+len(r)+len(t):>2}   {t}")
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
    print("\n" + "=" * 110)
    print(f"  baseline: AUC {ba.mean():.4f} (sd {ba.std():.4f})  "
          f"Brier {bb.mean():.4f}  ECE {be.mean():.4f}\n")
    print(f"  {'arm':<20}{'nfeat':>6}{'mean d':>9}{'sd':>8}{'min':>9}{'max':>9}"
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
        print(f"  {name:<20}{nf:>6}{d.mean():>+9.4f}{d.std():>8.4f}{d.min():>+9.4f}"
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
    print("=" * 110)
    json.dump(out, open(RESULTS, "w"), indent=2, default=float)
    if os.path.exists(RESULTS + ".partial"):
        os.remove(RESULTS + ".partial")
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
