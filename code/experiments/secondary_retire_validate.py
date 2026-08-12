"""
secondary_retire_validate.py -- PART 2: is RETIRING `secondary_eclipse_depth`
(31 -> 30) better than either keeping the broken version or swapping in the
corrected one?

Three options are now on the table for a column measured at AUC 0.4935 (chance):

  (a) KEEP AS-IS   31 features, buggy formula          <- current production
  (b) FIX AND SWAP 31 features, corrected formula      <- already tested:
                                                          -0.0024, CI entirely
                                                          below zero. Regression.
  (c) RETIRE       30 features, column removed         <- THIS FILE

PAIRING
-------
Uses SEED = 20260812 and N_BOOT = 12, identical to `secondary_swap_validate.py`,
so the bootstrap resamples are the SAME DRAWS. That makes all three arms paired
rather than compared across independent runs. The KEEP arm here must reproduce
the prior run's `old_auc` = 0.9212137 exactly; that is asserted as a built-in
consistency check, and if it fails the pairing claim is void.

Also measures permutation importance of `secondary_eclipse_depth` as deployed,
to answer whether the model relies on it at all or whether retirement is a
clean no-op.

Nothing is promoted. Production untouched.
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
from sklearn.inspection import permutation_importance

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..")
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT = os.path.join(HERE, "secondary_retire_results.json")

N_BOOT = 12
SEED = 20260812          # MUST match secondary_swap_validate.py for pairing
MDE = 0.0097
PRIOR_KEEP_AUC = 0.9212137023555709   # from secondary_swap_results.json


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
    cols30 = [c for c in cols if c != "secondary_eclipse_depth"]

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    tr_mask, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)
    tr_idx = np.where(tr_mask)[0]

    # ---- permutation importance of the column as DEPLOYED ----
    print("permutation importance of secondary_eclipse_depth (as deployed)...", flush=True)
    mo = model(); mo.fit(X.iloc[tr_idx], y[tr_idx])
    pi = permutation_importance(mo, X[te], y[te], n_repeats=10,
                                random_state=SEED, scoring="roc_auc", n_jobs=1)
    order = np.argsort(pi.importances_mean)[::-1]
    k = cols.index("secondary_eclipse_depth")
    rank = int(np.where(order == k)[0][0]) + 1
    print(f"  importance {pi.importances_mean[k]:+.5f} +/- {pi.importances_std[k]:.5f}"
          f"   RANK {rank}/31")
    print("  top 5:", [(cols[i], round(pi.importances_mean[i], 4)) for i in order[:5]])
    print("  bottom 5:", [(cols[i], round(pi.importances_mean[i], 4)) for i in order[-5:]])

    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(N_BOOT):
        samp = rng.choice(tr_idx, size=len(tr_idx), replace=True)
        r = {}
        for lab, use in (("keep", cols), ("drop", cols30)):
            mo = model(); mo.fit(X.iloc[samp][use], y[samp])
            p = mo.predict_proba(X[te][use])[:, 1]
            r[f"{lab}_auc"] = roc_auc_score(y[te], p)
            r[f"{lab}_brier"] = brier_score_loss(y[te], p)
            r[f"{lab}_ece"] = ece(y[te], p)
        r["delta"] = r["drop_auc"] - r["keep_auc"]
        rows.append(r)
        print(f"  boot {b+1}/{N_BOOT}  keep {r['keep_auc']:.4f}  drop {r['drop_auc']:.4f}  "
              f"d {r['delta']:+.4f}", flush=True)

    R = pd.DataFrame(rows)
    keep_mean = R.keep_auc.mean()
    print(f"\nPAIRING CHECK: keep-arm mean AUC {keep_mean:.7f} vs prior run "
          f"{PRIOR_KEEP_AUC:.7f}  diff {abs(keep_mean-PRIOR_KEEP_AUC):.2e}")
    paired = abs(keep_mean - PRIOR_KEEP_AUC) < 1e-9
    print(f"  -> bootstrap draws identical to the swap run: "
          f"{'YES, arms are paired' if paired else 'NO -- pairing claim VOID'}")

    d = R.delta.values
    lo, hi = np.percentile(d, [2.5, 97.5])
    print("\n" + "=" * 72)
    print(f"{'arm':<34}{'mean d':>9}{'sd':>8}{'min':>9}{'max':>9}{'pos':>7}{'>=MDE':>7}")
    print(f"{'(c) RETIRE  30 feat vs keep':<34}{d.mean():>+9.4f}{d.std():>8.4f}"
          f"{d.min():>+9.4f}{d.max():>+9.4f}{f'{(d>0).sum()}/{N_BOOT}':>7}"
          f"{f'{(d>=MDE).sum()}/{N_BOOT}':>7}")
    print(f"  95% CI [{lo:+.4f}, {hi:+.4f}]   clears ci_lo>0: {'YES' if lo > 0 else 'NO'}")
    print(f"  Brier  keep {R.keep_brier.mean():.4f} -> drop {R.drop_brier.mean():.4f}")
    print(f"  ECE    keep {R.keep_ece.mean():.4f} -> drop {R.drop_ece.mean():.4f}")

    json.dump({"n_boot": N_BOOT, "seed": SEED, "paired_with_swap_run": bool(paired),
               "keep_auc": float(keep_mean), "drop_auc": float(R.drop_auc.mean()),
               "mean_delta": float(d.mean()), "sd": float(d.std()),
               "ci": [float(lo), float(hi)],
               "positive": int((d > 0).sum()), "at_mde": int((d >= MDE).sum()),
               "brier": [float(R.keep_brier.mean()), float(R.drop_brier.mean())],
               "ece": [float(R.keep_ece.mean()), float(R.drop_ece.mean())],
               "perm_importance": float(pi.importances_mean[k]),
               "perm_importance_std": float(pi.importances_std[k]),
               "perm_rank_of_31": rank}, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
