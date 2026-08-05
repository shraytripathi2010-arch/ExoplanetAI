"""crowding_validate.py -- do catalog crowding features help? Resampled.

ARMS, chosen to separate physics from provenance

    base            production's 24 features, exactly as deployed
    +2 full-cov     adds crowd_flux_ratio_max and crowd_nearest_arcsec, both
                    computed from the cone search and available for ~100% of
                    stars
    +4 all          additionally adds TIC's own crowd_contratio / crowd_numcont

The split is not stylistic. TIC populates `contratio` only for Candidate Target
List stars, and TOI false positives are in the CTL BY CONSTRUCTION (they are
2-min targets), while many confirmed planets are not. Measured on a 200-star
pilot: contratio is present for 78.4% of false positives but only 52.5% of
planets, and the mere PRESENCE of the value has single-feature AUC 0.3705.

That is label provenance, not astrophysics -- and HistGradientBoosting handles
NaN natively, so it will happily learn the missingness pattern if allowed to.
Running +2 and +4 as separate arms measures how much of any apparent gain is
that artifact instead of assuming either way.

PROTOCOL, per the calibration-sweep lesson: nothing here is reported from a
single fit. 12 training-data bootstraps (seeds 1000..1011, matching the earlier
runs), baseline refit on the identical resampled rows each time, and AUC, Brier
and ECE all read from the resampled distribution.

Two outcome categories are reported separately, as the brief asks:
    clears  -- ci_lo > 0 on the paired bootstrap
    positive but unprovable -- delta > 0 consistently, below what a 1,098-star
                               test set can certify (~0.0097)
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
RESULTS = os.path.join(SCRIPT_DIR, "crowding_validate_results.json")

SEED = 42
N_RESAMPLES = 12
N_BOOT = 1500
N_WORKERS = 6
N_ECE_BINS = 15
DETECTION_THRESHOLD = 0.0097     # measured MDE at 1,098 test stars

FULL_COV = ["crowd_flux_ratio_max", "crowd_nearest_arcsec"]
TIC_NATIVE = ["crowd_contratio", "crowd_numcont"]
ARMS = [("+2 full-coverage", FULL_COV),
        ("+4 incl. TIC native", FULL_COV + TIC_NATIVE)]


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
    m05 = _m05()
    df = pd.read_csv(TRAINING)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    y = np.asarray(y)

    crowd = pd.read_csv(CROWD)
    keep = ["host"] + FULL_COV + TIC_NATIVE
    merged = df[["host"]].merge(crowd[keep], on="host", how="left")
    for c in FULL_COV + TIC_NATIVE:
        X[c] = pd.to_numeric(merged[c], errors="coerce").to_numpy()

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
    row = {"rep": rep,
           "baseline_full": float(roc_auc_score(y[te], bF)),
           "baseline_2min": float(roc_auc_score(y[te2], b2)),
           "baseline_brier": float(brier_score_loss(y[te], bF)),
           "baseline_ece": ece(y[te], bF), "arms": {}}

    for name, extra in ARMS:
        pF, p2 = _fit(base + extra, Xb, yb, X, te, te2)
        dF, loF, hiF = paired_boot(y[te], bF, pF)
        d2, lo2, hi2 = paired_boot(y[te2], b2, p2)
        row["arms"][name] = {
            "auc_full": float(roc_auc_score(y[te], pF)),
            "auc_2min": float(roc_auc_score(y[te2], p2)),
            "brier": float(brier_score_loss(y[te], pF)),
            "ece": ece(y[te], pF),
            "delta_full": dF, "ci_full": [loF, hiF], "clears_full": bool(loF > 0),
            "delta_2min": d2, "clears_2min": bool(lo2 > 0)}
    return row


def main():
    print("=" * 108)
    print("CATALOG CROWDING FEATURES -- resampled validation (no single-fit reporting)")
    print("=" * 108)
    print(f"  baseline: production (HGB + sigmoid cv=5) on the 24 base features,")
    print(f"  refit on the SAME resampled rows as each arm. {N_RESAMPLES} bootstraps.")
    for n, e in ARMS:
        print(f"    {n:<22} {e}")
    print()

    _init()
    print(f"  train {int(_G['tr'].sum())} / frozen test {int(_G['te'].sum())} "
          f"(2-min {int(_G['te2'].sum())})\n")

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
    print("\n" + "=" * 108)
    print("RESAMPLED DISTRIBUTION")
    print("=" * 108)
    print(f"  baseline: AUC {bf.mean():.4f} (sd {bf.std():.4f})  "
          f"Brier {bb.mean():.4f}  ECE {be.mean():.4f}\n")
    print(f"  {'arm':<22}{'mean d':>9}{'sd':>8}{'min':>9}{'max':>9}"
          f"{'pos':>7}{'clears':>8}{'>=MDE':>7}{'d_2min':>9}{'Brier':>9}{'ECE':>9}")

    out = {"n_resamples": len(rows), "detection_threshold": DETECTION_THRESHOLD,
           "baseline_mean_auc": float(bf.mean()), "baseline_sd_auc": float(bf.std()),
           "baseline_mean_brier": float(bb.mean()),
           "baseline_mean_ece": float(be.mean()),
           "arms": {n: e for n, e in ARMS}, "rows": rows, "summary": {}}
    for name, _ in ARMS:
        dF = np.array([r["arms"][name]["delta_full"] for r in rows])
        d2 = np.array([r["arms"][name]["delta_2min"] for r in rows])
        cF = sum(r["arms"][name]["clears_full"] for r in rows)
        br = np.array([r["arms"][name]["brier"] for r in rows])
        ec = np.array([r["arms"][name]["ece"] for r in rows])
        mde = int((dF >= DETECTION_THRESHOLD).sum())
        print(f"  {name:<22}{dF.mean():>+9.4f}{dF.std():>8.4f}{dF.min():>+9.4f}"
              f"{dF.max():>+9.4f}{int((dF>0).sum()):>4}/{len(rows)}"
              f"{cF:>5}/{len(rows)}{mde:>4}/{len(rows)}"
              f"{d2.mean():>+9.4f}{br.mean():>9.4f}{ec.mean():>9.4f}")
        out["summary"][name] = {
            "delta_full_mean": float(dF.mean()), "delta_full_sd": float(dF.std()),
            "delta_full_min": float(dF.min()), "delta_full_max": float(dF.max()),
            "n_positive_full": int((dF > 0).sum()), "n_clearing_full": cF,
            "n_at_or_above_mde": mde,
            "delta_2min_mean": float(d2.mean()),
            "n_clearing_2min": sum(r["arms"][name]["clears_2min"] for r in rows),
            "mean_brier": float(br.mean()), "mean_ece": float(ec.mean()),
            "brier_vs_baseline": float(br.mean() - bb.mean()),
            "ece_vs_baseline": float(ec.mean() - be.mean()), "n": len(rows)}

    print("\n" + "=" * 108)
    for name, _ in ARMS:
        s = out["summary"][name]
        if s["n_clearing_full"] >= 0.9 * len(rows):
            v = "CLEARS ci_lo>0 robustly"
        elif s["n_positive_full"] == len(rows):
            v = f"positive on every resample but unprovable (mean {s['delta_full_mean']:+.4f} vs MDE {DETECTION_THRESHOLD})"
        elif s["n_positive_full"] >= 0.75 * len(rows):
            v = "mostly positive, not robust"
        else:
            v = "no effect"
        print(f"  {name:<22} {v}")
    print("=" * 108)

    json.dump(out, open(RESULTS, "w"), indent=2, default=float)
    if os.path.exists(RESULTS + ".partial"):
        os.remove(RESULTS + ".partial")
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
