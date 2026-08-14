"""mlp_tabular_validate.py -- a SMALL dense network on the EXISTING 33 features.

WHY THIS IS NOT THE CLOSED CNN QUESTION
---------------------------------------
The closed CNN investigation trained a network on RAW PHASE-FOLDED FLUX and had
to learn a useful representation of the light curve from ~5k examples; it
reached 0.68-0.70 against a then-0.90+ tree model, and the gap has since WIDENED
to ~0.25 (production is 0.9402). That bottleneck is representation learning from
raw flux at low data volume.

This is a different, much smaller question: the 33 features are ALREADY
computed, so no representation has to be learned from raw data. It asks only
whether a dense network is a better FUNCTION APPROXIMATOR on 33 tabular inputs
than gradient-boosted trees, at n = 4,390 training rows.

WHAT HAS ALREADY BEEN TESTED (and why this is still distinct)
-------------------------------------------------------------
Stacking on model OUTPUTS has been tested twice: Part C (HGB + GP + CNN) and the
small-lift trio (HGB + RF + LR meta-learner, +0.0039, CI [-0.0022, +0.0106], no
clear). Neither used a DENSE NETWORK as the base classifier on the 33 features,
nor as a meta-learner over [HGB output + the 33 features]. Those two are what
this file tests.

DATA-VOLUME ARITHMETIC, stated before running
----------------------------------------------
    MLP (16, 8)   33*16+16 + 16*8+8 + 8+1  =   689 params  -> 6.4 rows/param
    MLP (64, 32)  33*64+64 + 64*32+32 + 33 = 4,289 params  -> 1.0 rows/param

At ~1 row per parameter the larger net is heavily over-parameterised for 4,390
training rows, so both sizes are run and the train-vs-test gap is reported as an
explicit overfitting diagnostic. Trees do well at this scale because axis-aligned
splits are a strong inductive bias for tabular data; that is the standing
literature result for medium-sized tabular problems, and it is what this test
is checking against rather than assuming.

Production's exact recipe is the baseline. Nothing is promoted.
"""
import os
import sys
import json
import importlib.util
import warnings
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(HERE, "cadence_class_confound.csv")
OUT = os.path.join(HERE, "mlp_tabular_validate.json")

N_BOOT = 12
SEED = 20260814
MDE = 0.0097


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def hgb():
    """Production's exact recipe."""
    return CalibratedClassifierCV(
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", HistGradientBoostingClassifier(random_state=42))]),
        cv=5, method="sigmoid")


def mlp(hidden):
    """Same calibration wrapper so the comparison isolates the model family.
    StandardScaler is required for a net and irrelevant to a tree -- including
    it is the fair setup, not an advantage."""
    return CalibratedClassifierCV(
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("scale", StandardScaler()),
                  ("clf", MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-3,
                                        max_iter=600, early_stopping=True,
                                        n_iter_no_change=20, random_state=42))]),
        cv=5, method="sigmoid")


def main():
    spec = importlib.util.spec_from_file_location("m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 33

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    tr_mask, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)
    tr_idx = np.where(tr_mask)[0]
    cad = pd.read_csv(CADENCE)[["host", "cadence_min"]]; cad["host"] = cad.host.astype(str)
    cc = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((cc >= 1.0) & (cc <= 2.6)).to_numpy()
    print(f"train {len(tr_idx)}  frozen test {int(te.sum())}  2-min {int((is2&te).sum())}")
    for h, n in (((16, 8), 689), ((64, 32), 4289)):
        print(f"  MLP{h}: ~{n} params -> {len(tr_idx)/n:.1f} training rows per parameter")

    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(N_BOOT):
        samp = rng.choice(tr_idx, size=len(tr_idx), replace=True)
        Xs, ys = X.iloc[samp], y[samp]
        rec = {}

        # --- production baseline ---
        mo = hgb(); mo.fit(Xs, ys)
        p_te = mo.predict_proba(X[te])[:, 1]
        rec["base_auc"] = roc_auc_score(y[te], p_te)
        rec["base_train_auc"] = roc_auc_score(ys, mo.predict_proba(Xs)[:, 1])
        rec["base_brier"] = brier_score_loss(y[te], p_te)
        rec["base_ece"] = ece(y[te], p_te)
        rec["base_auc2"] = roc_auc_score(y[te][is2[te]], p_te[is2[te]])

        # --- MLP as the classifier, two sizes ---
        for lab, h in (("mlp_small", (16, 8)), ("mlp_med", (64, 32))):
            mm = mlp(h); mm.fit(Xs, ys)
            p = mm.predict_proba(X[te])[:, 1]
            rec[f"{lab}_auc"] = roc_auc_score(y[te], p)
            rec[f"{lab}_train_auc"] = roc_auc_score(ys, mm.predict_proba(Xs)[:, 1])
            rec[f"{lab}_brier"] = brier_score_loss(y[te], p)
            rec[f"{lab}_ece"] = ece(y[te], p)
            rec[f"{lab}_auc2"] = roc_auc_score(y[te][is2[te]], p[is2[te]])

        # --- MLP meta-learner on [out-of-fold HGB output + the 33 features] ---
        oof = cross_val_predict(hgb(), Xs, ys, cv=5, method="predict_proba")[:, 1]
        Xs_meta = Xs.copy(); Xs_meta["hgb_p"] = oof
        Xte_meta = X[te].copy(); Xte_meta["hgb_p"] = p_te
        ms = mlp((16, 8)); ms.fit(Xs_meta, ys)
        p = ms.predict_proba(Xte_meta)[:, 1]
        rec["stack_mlp_auc"] = roc_auc_score(y[te], p)
        rec["stack_mlp_train_auc"] = roc_auc_score(ys, ms.predict_proba(Xs_meta)[:, 1])
        rec["stack_mlp_brier"] = brier_score_loss(y[te], p)
        rec["stack_mlp_ece"] = ece(y[te], p)
        rec["stack_mlp_auc2"] = roc_auc_score(y[te][is2[te]], p[is2[te]])

        rows.append(rec)
        print(f"  boot {b+1}/{N_BOOT}  base {rec['base_auc']:.4f}  "
              f"small {rec['mlp_small_auc']:.4f}  med {rec['mlp_med_auc']:.4f}  "
              f"stack {rec['stack_mlp_auc']:.4f}", flush=True)

    R = pd.DataFrame(rows)
    ARMS = ["base", "mlp_small", "mlp_med", "stack_mlp"]
    print("\n" + "=" * 100)
    print(f"{'arm':<12}{'test AUC':>10}{'mean d':>10}{'95% CI':>22}{'pos':>7}{'>=MDE':>7}"
          f"{'train AUC':>11}{'train-test':>12}{'Brier':>9}{'ECE':>8}")
    out = {"n_boot": N_BOOT, "mde": MDE, "arms": {}}
    for lab in ARMS:
        d = (R[f"{lab}_auc"] - R["base_auc"]).values
        lo, hi = np.percentile(d, [2.5, 97.5])
        gap = (R[f"{lab}_train_auc"] - R[f"{lab}_auc"]).mean()
        print(f"{lab:<12}{R[f'{lab}_auc'].mean():>10.4f}{d.mean():>+10.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{f'{(d>0).sum()}/{N_BOOT}':>7}"
              f"{f'{(d>=MDE).sum()}/{N_BOOT}':>7}{R[f'{lab}_train_auc'].mean():>11.4f}"
              f"{gap:>12.4f}{R[f'{lab}_brier'].mean():>9.4f}{R[f'{lab}_ece'].mean():>8.4f}")
        out["arms"][lab] = {"auc": float(R[f"{lab}_auc"].mean()),
                            "mean_delta": float(d.mean()), "ci": [float(lo), float(hi)],
                            "positive": int((d > 0).sum()), "at_mde": int((d >= MDE).sum()),
                            "train_auc": float(R[f"{lab}_train_auc"].mean()),
                            "train_test_gap": float(gap),
                            "auc_2min": float(R[f"{lab}_auc2"].mean()),
                            "brier": float(R[f"{lab}_brier"].mean()),
                            "ece": float(R[f"{lab}_ece"].mean()),
                            "clears": bool(lo > 0 and d.mean() >= MDE)}
    print("\n2-min-only subset:")
    for lab in ARMS:
        print(f"  {lab:<12} AUC {R[f'{lab}_auc2'].mean():.4f}  "
              f"mean d {(R[f'{lab}_auc2']-R['base_auc2']).mean():+.4f}")
    print(f"\nClearing needs ci_lo > 0 AND mean delta >= MDE ({MDE}).")
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
