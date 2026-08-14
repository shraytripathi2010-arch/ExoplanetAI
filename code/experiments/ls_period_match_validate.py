"""ls_period_match_validate.py -- 31 -> 32 with `ls_period_match`.

The only one of the three ExoMiner++-inspired proposals to clear the pre-model
checks (see `exominer_triple_assess.py`):

    single-feature AUC          0.5887   (|AUC-0.5| = 0.0887)
    max |rho| vs the 31         0.301 with `period`   (threshold 0.80)
    AUC by ecliptic-lat quartile [0.596, 0.547, 0.607, 0.603]  -- stable
    rho with |galactic b|       -0.004
    availability on both candidate pools   100% of Success rows

The other two failed: `trend_*` is redundant with `var_ls_amp` (|rho| 0.83,
above threshold) AND spatially unstable (quartile AUCs 0.434 - 0.715);
`straylight_frac` is at chance (AUC 0.4964).

Protocol is production's exact recipe on the frozen split, 12 training
bootstraps, reported on the full frozen test set and the 2-min-only subset,
with Brier and ECE. MDE on this test set is ~0.0097. Nothing is promoted here.
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
FEATS = os.path.join(HERE, "exominer_triple_features.csv")
CADENCE = os.path.join(HERE, "cadence_class_confound.csv")
OUT = os.path.join(HERE, "ls_period_match_results.json")

N_BOOT = 12
SEED = 20260813
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
    assert len(cols) == 31
    NEW = "ls_period_match"

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    nf = pd.read_csv(FEATS); nf["host"] = nf.host.astype(str)
    df = df.merge(nf[["host", NEW]], on="host", how="left")

    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    X32 = X.copy()
    X32[NEW] = pd.to_numeric(df[NEW], errors="coerce").replace(
        [np.inf, -np.inf], np.nan).values
    cols32 = cols + [NEW]

    tr_mask, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)
    tr_idx = np.where(tr_mask)[0]
    print(f"train {len(tr_idx)}  frozen test {int(te.sum())}")
    print(f"{NEW} coverage: {X32[NEW].notna().mean():.4f}")

    cad = pd.read_csv(CADENCE)[["host", "cadence_min"]]
    cad["host"] = cad.host.astype(str)
    cc = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((cc >= 1.0) & (cc <= 2.6)).to_numpy()
    print(f"2-min subset within frozen test: {int((is2 & te).sum())} stars")

    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(N_BOOT):
        samp = rng.choice(tr_idx, size=len(tr_idx), replace=True)
        r = {}
        for lab, Xu, cu in (("base", X, cols), ("plus", X32, cols32)):
            mo = model(); mo.fit(Xu.iloc[samp][cu], y[samp])
            p = mo.predict_proba(Xu[te][cu])[:, 1]
            r[f"{lab}_auc"] = roc_auc_score(y[te], p)
            r[f"{lab}_brier"] = brier_score_loss(y[te], p)
            r[f"{lab}_ece"] = ece(y[te], p)
            s = is2[te]
            r[f"{lab}_auc2"] = roc_auc_score(y[te][s], p[s]) if s.sum() > 50 else np.nan
        r["delta"] = r["plus_auc"] - r["base_auc"]
        r["delta2"] = r["plus_auc2"] - r["base_auc2"]
        rows.append(r)
        print(f"  boot {b+1}/{N_BOOT}  base {r['base_auc']:.4f}  plus {r['plus_auc']:.4f}  "
              f"d {r['delta']:+.4f}", flush=True)

    R = pd.DataFrame(rows)
    d, d2 = R.delta.values, R.delta2.values
    lo, hi = np.percentile(d, [2.5, 97.5])
    print("\n" + "=" * 74)
    print(f"{'arm':<26}{'mean d':>9}{'sd':>8}{'min':>9}{'max':>9}{'pos':>7}{'>=MDE':>7}")
    print(f"{'31 -> 32 (full test)':<26}{d.mean():>+9.4f}{d.std():>8.4f}{d.min():>+9.4f}"
          f"{d.max():>+9.4f}{f'{(d>0).sum()}/{N_BOOT}':>7}{f'{(d>=MDE).sum()}/{N_BOOT}':>7}")
    print(f"{'31 -> 32 (2-min subset)':<26}{np.nanmean(d2):>+9.4f}{np.nanstd(d2):>8.4f}"
          f"{np.nanmin(d2):>+9.4f}{np.nanmax(d2):>+9.4f}"
          f"{f'{(d2>0).sum()}/{N_BOOT}':>7}{f'{(d2>=MDE).sum()}/{N_BOOT}':>7}")
    print(f"\n  95% CI on delta: [{lo:+.4f}, {hi:+.4f}]   clears ci_lo > 0: "
          f"{'YES' if lo > 0 else 'NO'}")
    print(f"  AUC    base {R.base_auc.mean():.4f} -> plus {R.plus_auc.mean():.4f}")
    print(f"  Brier  base {R.base_brier.mean():.4f} -> plus {R.plus_brier.mean():.4f}")
    print(f"  ECE    base {R.base_ece.mean():.4f} -> plus {R.plus_ece.mean():.4f}")
    print(f"\n  MDE = {MDE}. A clearing arm needs ci_lo > 0 AND mean delta >= MDE.")

    json.dump({"n_boot": N_BOOT, "seed": SEED, "mean_delta": float(d.mean()),
               "sd": float(d.std()), "ci": [float(lo), float(hi)],
               "positive": int((d > 0).sum()), "at_mde": int((d >= MDE).sum()),
               "mean_delta_2min": float(np.nanmean(d2)),
               "base_auc": float(R.base_auc.mean()), "plus_auc": float(R.plus_auc.mean()),
               "brier": [float(R.base_brier.mean()), float(R.plus_brier.mean())],
               "ece": [float(R.base_ece.mean()), float(R.plus_ece.mean())],
               "clears": bool(lo > 0 and d.mean() >= MDE)}, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
