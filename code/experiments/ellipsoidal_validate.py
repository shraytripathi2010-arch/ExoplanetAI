"""ellipsoidal_validate.py -- PART 3: full resampled model test for whichever
ellipsoidal-variation features cleared the pre-model gates.

Production's exact recipe (Optuna-tuned HGB inside CalibratedClassifierCV,
sigmoid, cv=5) on the frozen split. Baseline is the live 33-feature production
configuration. Reported on the full frozen test set AND the 2-min-only subset,
with Brier and ECE. MDE on this test set is ~0.0097; clearing requires
ci_lo > 0 AND mean delta >= MDE. Nothing is promoted here.
"""
import os
import sys
import json
import importlib.util
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
FEATS = os.path.join(HERE, "ellipsoidal_features.csv")
CADENCE = os.path.join(HERE, "cadence_class_confound.csv")
OUT = os.path.join(HERE, "ellipsoidal_validate.json")

N_BOOT = 12
SEED = 20260828
MDE = 0.0097

# production's Optuna-tuned hyperparameters (deployed 2026-08-14)
OPTUNA = dict(learning_rate=0.09258475971800786, max_iter=475,
              max_leaf_nodes=63, max_depth=None, min_samples_leaf=24,
              l2_regularization=0.009012660266897076,
              class_weight="balanced", random_state=42)

# arms are filled in from the pre-model battery's verdict
ARMS = json.loads(os.environ.get("ELL_ARMS", "{}"))


def m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(ROOT, "code", "05_train_models.py"))
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
                  ("clf", HistGradientBoostingClassifier(**OPTUNA))]),
        cv=5, method="sigmoid")


def main():
    M = m05()
    cols = list(M.FEATURE_COLUMNS)
    assert len(cols) == 33

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    F = pd.read_csv(FEATS); F["host"] = F.host.astype(str)
    F = F[F._set == "train"].drop_duplicates("host")
    extra = sorted({c for v in ARMS.values() for c in v})
    df = df.merge(F[["host"] + extra], on="host", how="left")

    X, y = M.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    for c in extra:
        X[c] = pd.to_numeric(df[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan).values

    tr_mask, _ = M.split_by_host(df)
    te = M.frozen_test_mask(df)
    tr_idx = np.where(tr_mask)[0]
    print(f"train {len(tr_idx)}   frozen test {int(te.sum())}")
    print(f"arms: { {k: v for k, v in ARMS.items()} }")

    cad = pd.read_csv(CADENCE)[["host", "cadence_min"]]
    cad["host"] = cad.host.astype(str)
    cc = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                       errors="coerce")
    is2 = ((cc >= 1.0) & (cc <= 2.6)).to_numpy()
    print(f"2-min subset within frozen test: {int((is2 & te).sum())} stars")

    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(N_BOOT):
        samp = rng.choice(tr_idx, size=len(tr_idx), replace=True)
        r = {}
        for lab, add in [("base", [])] + list(ARMS.items()):
            cu = cols + list(add)
            mo = model(); mo.fit(X.iloc[samp][cu], y[samp])
            p = mo.predict_proba(X[te][cu])[:, 1]
            r[f"{lab}_auc"] = roc_auc_score(y[te], p)
            r[f"{lab}_brier"] = brier_score_loss(y[te], p)
            r[f"{lab}_ece"] = ece(y[te], p)
            s = is2[te]
            r[f"{lab}_auc2"] = roc_auc_score(y[te][s], p[s]) if s.sum() > 50 else np.nan
        for lab in ARMS:
            r[f"d_{lab}"] = r[f"{lab}_auc"] - r["base_auc"]
            r[f"d2_{lab}"] = r[f"{lab}_auc2"] - r["base_auc2"]
        rows.append(r)
        print(f"  boot {b+1}/{N_BOOT}  base {r['base_auc']:.4f}  " +
              "  ".join(f"{k} {r[f'd_{k}']:+.4f}" for k in ARMS), flush=True)

    R = pd.DataFrame(rows)
    out = {"n_boot": N_BOOT, "mde": MDE, "arms": ARMS,
           "base_auc_mean": float(R.base_auc.mean()),
           "base_brier_mean": float(R.base_brier.mean()),
           "base_ece_mean": float(R.base_ece.mean()),
           "base_auc2_mean": float(R.base_auc2.mean())}
    print("\n" + "=" * 92)
    print(f"{'arm':<22}{'mean d':>10}{'95% CI':>24}{'pos':>7}{'>=MDE':>8}"
          f"{'d 2-min':>10}{'Brier':>9}{'ECE':>8}")
    print("=" * 92)
    print(f"{'base (33)':<22}{'--':>10}{'--':>24}{'--':>7}{'--':>8}{'--':>10}"
          f"{R.base_brier.mean():>9.4f}{R.base_ece.mean():>8.4f}")
    for lab in ARMS:
        d = R[f"d_{lab}"].to_numpy(); d2 = R[f"d2_{lab}"].to_numpy()
        lo, hi = np.percentile(d, [2.5, 97.5])
        npos = int((d > 0).sum()); nmde = int((d >= MDE).sum())
        out[lab] = {"mean_delta": float(d.mean()), "sd": float(d.std(ddof=1)),
                    "ci_lo": float(lo), "ci_hi": float(hi),
                    "n_positive": npos, "n_ge_mde": nmde,
                    "mean_delta_2min": float(np.nanmean(d2)),
                    "auc": float(R[f"{lab}_auc"].mean()),
                    "brier": float(R[f"{lab}_brier"].mean()),
                    "ece": float(R[f"{lab}_ece"].mean()),
                    "clears": bool(lo > 0 and d.mean() >= MDE)}
        print(f"{lab:<22}{d.mean():>+10.4f}[{lo:>+8.4f},{hi:>+8.4f}]"
              f"{npos:>4}/{N_BOOT}{nmde:>5}/{N_BOOT}{np.nanmean(d2):>+10.4f}"
              f"{R[f'{lab}_brier'].mean():>9.4f}{R[f'{lab}_ece'].mean():>8.4f}")
    print("=" * 92)
    print(f"clearing requires ci_lo > 0 AND mean delta >= MDE ({MDE})")
    for lab in ARMS:
        print(f"  {lab:<22}{'CLEARS' if out[lab]['clears'] else 'DOES NOT CLEAR'}")
    json.dump(out, open(OUT, "w"), indent=2, default=float)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
