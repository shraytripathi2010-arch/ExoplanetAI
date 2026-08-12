"""
secondary_swap_validate.py -- PART 3: replace the deployed
`secondary_eclipse_depth` with the correctly-folded, duration-aware version and
validate as a like-for-like 31 -> 31 swap.

WHY A SWAP AND NOT AN ADDITION
-------------------------------
The earlier weak-secondary work ADDED `sec_depth_windowed` alongside the broken
column (24 -> 26 features) and no arm cleared. This is the untested variant:
REPLACE the broken column at the same slot, so the feature count is unchanged
and the comparison is directly against deployed production.

WHAT WAS FOUND FIRST (and is the real headline)
------------------------------------------------
The deployed feature is not merely blunt -- it is computed at the WRONG PHASE.
TLS's `r.folded_phase` places the PRIMARY at phase 0.5, verified directly with a
pure transit carrying no secondary at all:

    phase 0.49-0.51 depth 0.009202     <- the primary is HERE
    phase 0.00-0.01 depth 0.001232

Production's `sec_mask = (phase > 0.45) & (phase < 0.55)` therefore samples the
PRIMARY, over a window ~6x wider than the transit, so the median is dominated by
out-of-transit baseline. Result: single-feature AUC 0.4935 on 5,373 training
stars -- chance. It measures neither the secondary (wrong location) nor the
primary (too diluted).

`weak_secondary.py` builds its own fold, `phase = ((t - t0)/period) % 1.0`,
which puts the primary at 0 and the secondary at 0.5, so ITS window is correct.
Its values are reused here rather than recomputed.

    deployed  secondary_eclipse_depth  AUC 0.4935   |AUC-0.5| 0.0065
    corrected sec_depth_windowed       AUC 0.4350   |AUC-0.5| 0.0650   (10x)

Protocol: production's exact recipe, frozen split, 12 training bootstraps,
reported on the full test set and the 2-min subset. Nothing is promoted here.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..")
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
WEAK = os.path.join(HERE, "weak_secondary_features.csv")
CADENCE = os.path.join(HERE, "cadence_by_host.csv")
OUT = os.path.join(HERE, "secondary_swap_results.json")

N_BOOT = 12
SEED = 20260812
MDE = 0.0097


def _m05():
    spec = importlib.util.spec_from_file_location("m05", os.path.join(CODE, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["m05"] = m
    spec.loader.exec_module(m); return m


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def model():
    return CalibratedClassifierCV(
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", HistGradientBoostingClassifier(random_state=42))]),
        cv=5, method="sigmoid")


def main():
    m05 = _m05()
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 31 and "secondary_eclipse_depth" in cols
    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    w = pd.read_csv(WEAK); w["host"] = w.host.astype(str)
    df = df.merge(w[["host", "sec_depth_windowed"]], on="host", how="left")

    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    tr_mask, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)

    # coverage, reported before anything else
    old_cov = X["secondary_eclipse_depth"].notna().mean()
    new_cov = pd.to_numeric(df["sec_depth_windowed"], errors="coerce").notna().mean()
    print(f"COVERAGE  deployed {old_cov:.4f}   corrected {new_cov:.4f}")

    X_new = X.copy()
    X_new["secondary_eclipse_depth"] = pd.to_numeric(
        df["sec_depth_windowed"], errors="coerce").replace([np.inf, -np.inf], np.nan).values

    # redundancy check on the swapped column against the other 30
    c = X_new.corr(method="spearman")["secondary_eclipse_depth"].drop("secondary_eclipse_depth")
    print(f"REDUNDANCY max |rho| vs other 30: {c.abs().max():.3f} "
          f"({c.abs().idxmax()})   threshold 0.80")

    cad = pd.read_csv(CADENCE) if os.path.exists(CADENCE) else None
    if cad is not None:
        cc = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
        is2 = ((cc >= 1.0) & (cc <= 2.6)).to_numpy() | cc.isna().to_numpy()
    else:
        is2 = np.ones(len(df), bool)

    tr_idx = np.where(tr_mask)[0]
    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(N_BOOT):
        samp = rng.choice(tr_idx, size=len(tr_idx), replace=True)
        r = {}
        for lab, Xu in (("old", X), ("new", X_new)):
            mo = model(); mo.fit(Xu.iloc[samp], y[samp])
            p = mo.predict_proba(Xu[te])[:, 1]
            r[f"{lab}_auc"] = roc_auc_score(y[te], p)
            r[f"{lab}_brier"] = brier_score_loss(y[te], p)
            r[f"{lab}_ece"] = ece(y[te], p)
            sub = is2[te]
            r[f"{lab}_auc2"] = roc_auc_score(y[te][sub], p[sub]) if sub.sum() > 50 else np.nan
        r["delta"] = r["new_auc"] - r["old_auc"]
        r["delta2"] = r["new_auc2"] - r["old_auc2"]
        rows.append(r)
        print(f"  boot {b+1}/{N_BOOT}  old {r['old_auc']:.4f}  new {r['new_auc']:.4f}  "
              f"d {r['delta']:+.4f}", flush=True)

    R = pd.DataFrame(rows)
    d = R.delta.values; d2 = R.delta2.values
    print("\n" + "=" * 70)
    print(f"{'arm':<28}{'mean d':>9}{'sd':>8}{'min':>9}{'max':>9}{'pos':>7}{'>=MDE':>7}")
    print(f"{'swap (full test)':<28}{d.mean():>+9.4f}{d.std():>8.4f}{d.min():>+9.4f}"
          f"{d.max():>+9.4f}{f'{(d>0).sum()}/{N_BOOT}':>7}{f'{(d>=MDE).sum()}/{N_BOOT}':>7}")
    print(f"{'swap (2-min subset)':<28}{np.nanmean(d2):>+9.4f}{np.nanstd(d2):>8.4f}"
          f"{np.nanmin(d2):>+9.4f}{np.nanmax(d2):>+9.4f}"
          f"{f'{(d2>0).sum()}/{N_BOOT}':>7}{f'{(d2>=MDE).sum()}/{N_BOOT}':>7}")
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n  95% CI on delta: [{lo:+.4f}, {hi:+.4f}]   clears ci_lo>0: "
          f"{'YES' if lo > 0 else 'NO'}")
    print(f"  Brier  old {R.old_brier.mean():.4f} -> new {R.new_brier.mean():.4f}")
    print(f"  ECE    old {R.old_ece.mean():.4f} -> new {R.new_ece.mean():.4f}")
    json.dump({"n_boot": N_BOOT, "mean_delta": float(d.mean()),
               "sd": float(d.std()), "ci": [float(lo), float(hi)],
               "positive": int((d > 0).sum()), "at_mde": int((d >= MDE).sum()),
               "mean_delta_2min": float(np.nanmean(d2)),
               "old_auc": float(R.old_auc.mean()), "new_auc": float(R.new_auc.mean()),
               "brier": [float(R.old_brier.mean()), float(R.new_brier.mean())],
               "ece": [float(R.old_ece.mean()), float(R.new_ece.mean())],
               "coverage": [float(old_cov), float(new_cov)],
               "max_redundancy": float(c.abs().max())}, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
