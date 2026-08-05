"""crowding_final.py -- the promotion-decision artifact for crowding features.

Four arms, full metrics (AUC / Brier / ECE), 12 training bootstraps, production
refit on identical rows. Nothing here is read from a single fit.

    +2 crowding          the deployable candidate
    +|b| only            sky position alone -- the confound, as a control
    +|b| + crowding      position modelled EXPLICITLY alongside crowding
    +eclat + crowding    same idea with |ecliptic latitude|, which drives TESS
                         sector coverage and is a different spatial axis from
                         galactic latitude

The third arm exists to answer a specific question: if the model is given
position honestly as its own feature, does crowding still contribute, or was
crowding just laundering position? The fourth checks whether the galactic-plane
axis is the only spatial confound or whether TESS's own observing geometry adds
a second one.

IMPORTANT -- why the |b| arms are DIAGNOSTIC, not promotion candidates.
Measured |galactic b| distributions:

    training planets          median 16.9 deg
    training false positives  median 12.5 deg
    UNKNOWN candidate pool    median 32.4 deg   (KS vs planets D=0.26, p=3e-17)

The stars actually scored in production live at systematically higher galactic
latitude than either training class. A model that learned "higher |b| -> more
likely planet" from this training set would apply that rule to a pool where
almost everything is high-|b|, inflating scores across the board for a reason
that has nothing to do with transits. Test-set AUC would rise; production
behaviour would degrade. So these arms measure the confound; they are not
proposals.
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
RESULTS = os.path.join(SCRIPT_DIR, "crowding_final_results.json")

SEED, N_RESAMPLES, N_BOOT, N_WORKERS, N_ECE_BINS = 42, 12, 1500, 6, 15
MDE = 0.0097

CROWD_COLS = ["crowd_flux_ratio_max", "crowd_nearest_arcsec"]
ARMS = [("+2 crowding (candidate)", CROWD_COLS),
        ("+|b| only (control)", ["sky_abs_galactic_b"]),
        ("+|b| + crowding", ["sky_abs_galactic_b"] + CROWD_COLS),
        ("+|eclat| + crowding", ["sky_abs_ecliptic_lat"] + CROWD_COLS)]


def ece(y, p, bins=N_ECE_BINS):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


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
    el = np.full(len(df), np.nan)
    ok = np.isfinite(ra) & np.isfinite(dec)
    sc = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg)
    gb[ok] = np.abs(sc.galactic.b.deg)
    el[ok] = np.abs(sc.barycentrictrueecliptic.lat.deg)
    X["sky_abs_galactic_b"] = gb
    X["sky_abs_ecliptic_lat"] = el

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
    row = {"rep": rep, "baseline_full": float(roc_auc_score(y[te], bF)),
           "baseline_brier": float(brier_score_loss(y[te], bF)),
           "baseline_ece": ece(y[te], bF), "arms": {}}
    for name, extra in ARMS:
        pF, p2 = _fit(base + extra, Xb, yb, X, te, te2)
        dF, loF, hiF = paired_boot(y[te], bF, pF)
        d2, lo2, hi2 = paired_boot(y[te2], b2, p2)
        row["arms"][name] = {
            "auc_full": float(roc_auc_score(y[te], pF)),
            "brier": float(brier_score_loss(y[te], pF)), "ece": ece(y[te], pF),
            "delta_full": dF, "clears_full": bool(loF > 0),
            "delta_2min": d2, "clears_2min": bool(lo2 > 0)}
    return row


def main():
    print("=" * 112)
    print("CROWDING -- FINAL ARMS FOR THE PROMOTION DECISION")
    print("=" * 112)
    for n, e in ARMS:
        print(f"    {n:<26} {e}")
    print()
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
    bb = np.array([r["baseline_brier"] for r in rows])
    be = np.array([r["baseline_ece"] for r in rows])
    print("\n" + "=" * 112)
    print(f"  baseline: AUC {bf.mean():.4f} (sd {bf.std():.4f})  "
          f"Brier {bb.mean():.4f}  ECE {be.mean():.4f}\n")
    print(f"  {'arm':<26}{'mean d':>9}{'sd':>8}{'min':>9}{'max':>9}"
          f"{'pos':>7}{'clears':>8}{'>=MDE':>7}{'d_2min':>9}{'Brier':>9}{'ECE':>9}")
    out = {"n_resamples": len(rows), "mde": MDE,
           "baseline_mean_auc": float(bf.mean()),
           "baseline_mean_brier": float(bb.mean()),
           "baseline_mean_ece": float(be.mean()), "rows": rows, "summary": {}}
    for name, _ in ARMS:
        dF = np.array([r["arms"][name]["delta_full"] for r in rows])
        d2 = np.array([r["arms"][name]["delta_2min"] for r in rows])
        cF = sum(r["arms"][name]["clears_full"] for r in rows)
        br = np.array([r["arms"][name]["brier"] for r in rows])
        ec = np.array([r["arms"][name]["ece"] for r in rows])
        print(f"  {name:<26}{dF.mean():>+9.4f}{dF.std():>8.4f}{dF.min():>+9.4f}"
              f"{dF.max():>+9.4f}{int((dF>0).sum()):>4}/{len(rows)}"
              f"{cF:>5}/{len(rows)}{int((dF>=MDE).sum()):>4}/{len(rows)}"
              f"{d2.mean():>+9.4f}{br.mean():>9.4f}{ec.mean():>9.4f}")
        out["summary"][name] = {
            "delta_full_mean": float(dF.mean()), "delta_full_sd": float(dF.std()),
            "delta_full_min": float(dF.min()), "delta_full_max": float(dF.max()),
            "n_positive": int((dF > 0).sum()), "n_clearing": cF,
            "n_at_or_above_mde": int((dF >= MDE).sum()),
            "delta_2min_mean": float(d2.mean()),
            "mean_brier": float(br.mean()), "mean_ece": float(ec.mean()),
            "brier_vs_baseline": float(br.mean() - bb.mean()),
            "ece_vs_baseline": float(ec.mean() - be.mean())}

    s = out["summary"]
    c = s["+2 crowding (candidate)"]["delta_full_mean"]
    b = s["+|b| only (control)"]["delta_full_mean"]
    cb = s["+|b| + crowding"]["delta_full_mean"]
    ce = s["+|eclat| + crowding"]["delta_full_mean"]
    print("\n" + "=" * 112)
    print("DECOMPOSITION")
    print("=" * 112)
    print(f"  crowding alone                        {c:+.4f}")
    print(f"  position alone                        {b:+.4f}")
    print(f"  crowding on top of galactic position  {cb - b:+.4f}   "
          f"({'clears' if cb - b > MDE else 'below'} MDE {MDE})")
    print(f"  ecliptic-latitude variant             {ce:+.4f}")
    out["crowding_beyond_galactic"] = float(cb - b)
    print("=" * 112)
    json.dump(out, open(RESULTS, "w"), indent=2, default=float)
    if os.path.exists(RESULTS + ".partial"):
        os.remove(RESULTS + ".partial")
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
