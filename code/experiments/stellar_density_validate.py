"""stellar_density_validate.py -- resampled validation of the density features.

ARM DESIGN. Each arm is measured against its OWN reference, identical except
for the columns under test, so every delta attributes cleanly. This is the
structure the trapezoid work used for its sky arm, and it is the form this
project has established as required -- a held-in control ARM, not a
correlation coefficient.

  A  base                    -> base + rho_ratio
       the only new column that passed the 0.80 redundancy threshold
  B  base                    -> base + all four
       st_logg/st_rho/rho_circ are individually REDUNDANT (max |r| 0.896,
       0.921, 0.905). Included anyway because a redundancy threshold is a
       guideline about pairwise linear association, and a boosted ensemble can
       in principle use a correlated column. Tested rather than assumed.
  C  base + abs_gal_b        -> base + abs_gal_b + rho_ratio
       THE SPATIAL CONTROL ARM. st_rho carries the largest sky exposure of the
       four (Spearman +0.192 vs |galactic b|). If A's delta survives here, it
       is not a sky-position artifact.
  D  base                    -> base + rho_ratio_available
       indicator only, NO measured values. The multi-sector depth-consistency
       result was killed by exactly this arm reproducing 108% of its gain, and
       this project's corrected heuristic requires it for any feature with
       incomplete coverage. Coverage here is 81.5% with availability-AUC
       0.4984 (planets 81.4% vs FP 81.8%), so this is expected to be null --
       it is run to demonstrate that, not to discover it.

Column counts are ASSERTED before every fit, so the silent-null bug that once
made a flux-statistics run report a perfect zero for a comparison that never
differed cannot recur here.
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
DFEAT = os.path.join(SCRIPT_DIR, "stellar_density_features.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "stellar_density_validate_results.json")

SEED, N_RESAMPLES, N_BOOT, N_WORKERS, N_ECE_BINS = 42, 12, 1500, 6, 15
MDE = 0.0097
ALL4 = ["st_logg", "st_rho", "rho_circ", "rho_ratio"]
SKY = "abs_gal_b"
AVAIL = "rho_ratio_available"

# (name, reference_extra, test_extra)
ARMS = [
    ("A: +rho_ratio",        [],      ["rho_ratio"]),
    ("B: +all four",         [],      ALL4),
    ("C: +rho_ratio | sky",  [SKY],   ["rho_ratio"]),
    ("D: availability ONLY", [],      [AVAIL]),
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

    d = pd.read_csv(DFEAT)
    merged = df[["host"]].merge(d[["host"] + ALL4], on="host", how="left")
    for c in ALL4:
        X[c] = pd.to_numeric(merged[c], errors="coerce").to_numpy()
    X[AVAIL] = np.isfinite(X["rho_ratio"]).astype(float)

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
    print("STELLAR DENSITY -- resampled validation vs the deployed 0.9208 / 26 features")
    print("=" * 110)
    _init()
    print(f"  base features: {len(_G['base'])}")
    for n, r, t in ARMS:
        print(f"    {n:<22} ref {len(_G['base'])+len(r):>2} -> test "
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
    print(f"  {'arm':<22}{'nfeat':>6}{'mean d':>9}{'sd':>8}{'min':>9}{'max':>9}"
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
        print(f"  {name:<22}{nf:>6}{d.mean():>+9.4f}{d.std():>8.4f}{d.min():>+9.4f}"
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
