"""optuna_hpo_validate.py -- PART 1b: the searched configuration under this
project's actual bar.

`optuna_hpo_nested.py` produces an honest nested-CV estimate of the SEARCH
PROCEDURE and one selected configuration. Neither is a result here. Every other
finding in this project had to survive 12 training bootstraps against the frozen
test with `ci_lo > 0 AND mean delta >= MDE`, and the calibration sweep is the
standing proof of why: seven arms beat production on a single fit and all seven
dissolved.

Three arms, production's full calibrated recipe throughout, only the HGB
hyperparameters differing:

    prod     what `models/best_model.joblib` carries today -- sklearn defaults
    optuna   the TPE-selected configuration from the final full-training study
    legacy   the configuration every model version before the Gaia swap carried

The legacy arm exists because of what the deduplication check turned up: the
Gaia deployment rebuilt the model from `HistGradientBoostingClassifier(
random_state=42)` rather than cloning the incumbent, so the deployed artifact
silently reverted five tuned hyperparameters AND `class_weight='balanced'`. The
+0.0142 Gaia feature delta is unaffected (both arms of that comparison were at
defaults), but "production's hyperparameters" and "the tuned hyperparameters"
are two different things right now, and only one of them has ever been measured
at 33 features. This arm measures the other.
"""
import os
import sys
import json
import time
import importlib.util
import warnings
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
NESTED = os.path.join(HERE, "optuna_hpo_nested.json")
CADENCE = os.path.join(HERE, "cadence_class_confound.csv")
OUT = os.path.join(HERE, "optuna_hpo_validate.json")

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


def calibrated(params):
    return CalibratedClassifierCV(
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", HistGradientBoostingClassifier(**params))]),
        cv=5, method="sigmoid")


def main():
    nested = json.load(open(NESTED))
    ARMS = {"prod": nested["prod_params"],
            "optuna": nested["final_params"],
            "legacy": nested["legacy_params"]}
    print("arms:")
    for k, v in ARMS.items():
        print(f"  {k:<8} {v}")

    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 33, len(cols)

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
    Xte, yte, s2 = X[te], y[te], is2[te]
    print(f"\ntrain {len(tr_idx)}  frozen test {int(te.sum())}  2-min {int(s2.sum())}")

    # ---- clean single fit on the real training split, for the record ----
    print("\nsingle fit on the full training split (NOT a result -- context only):")
    single = {}
    for k, params in ARMS.items():
        mo = calibrated(params).fit(X[tr_mask], y[tr_mask])
        p = mo.predict_proba(Xte)[:, 1]
        single[k] = {"auc": float(roc_auc_score(yte, p)),
                     "brier": float(brier_score_loss(yte, p)), "ece": ece(yte, p)}
        print(f"  {k:<8} AUC {single[k]['auc']:.4f}  Brier {single[k]['brier']:.4f}  "
              f"ECE {single[k]['ece']:.4f}", flush=True)

    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    for b in range(N_BOOT):
        samp = rng.choice(tr_idx, size=len(tr_idx), replace=True)
        Xb, yb = X.iloc[samp], y[samp]
        rec = {}
        for k, params in ARMS.items():
            mo = calibrated(params).fit(Xb, yb)
            p = mo.predict_proba(Xte)[:, 1]
            rec[f"{k}|auc"] = roc_auc_score(yte, p)
            rec[f"{k}|brier"] = brier_score_loss(yte, p)
            rec[f"{k}|ece"] = ece(yte, p)
            rec[f"{k}|auc2"] = roc_auc_score(yte[s2], p[s2])
        rows.append(rec)
        print(f"  boot {b+1}/{N_BOOT}  prod {rec['prod|auc']:.4f}  "
              f"optuna {rec['optuna|auc']:.4f}  legacy {rec['legacy|auc']:.4f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    R = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(f"{'arm':<10}{'AUC':>9}{'mean d':>10}{'sd':>8}{'95% CI':>22}{'pos':>7}"
          f"{'>=MDE':>7}{'Brier':>9}{'ECE':>8}{'2-min d':>10}")
    out = {"n_boot": N_BOOT, "mde": MDE, "arms": ARMS, "single_fit": single,
           "nested_summary": nested["nested_summary"], "resampled": {}}
    for k in ARMS:
        d = (R[f"{k}|auc"] - R["prod|auc"]).values
        d2 = (R[f"{k}|auc2"] - R["prod|auc2"]).values
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{k:<10}{R[f'{k}|auc'].mean():>9.4f}{d.mean():>+10.4f}{d.std():>8.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{f'{(d>0).sum()}/{N_BOOT}':>7}"
              f"{f'{(d>=MDE).sum()}/{N_BOOT}':>7}{R[f'{k}|brier'].mean():>9.4f}"
              f"{R[f'{k}|ece'].mean():>8.4f}{d2.mean():>+10.4f}")
        out["resampled"][k] = {
            "auc": float(R[f"{k}|auc"].mean()), "mean_delta": float(d.mean()),
            "sd": float(d.std()), "ci": [float(lo), float(hi)],
            "positive": int((d > 0).sum()), "at_mde": int((d >= MDE).sum()),
            "brier": float(R[f"{k}|brier"].mean()), "ece": float(R[f"{k}|ece"].mean()),
            "auc_2min": float(R[f"{k}|auc2"].mean()), "delta_2min": float(d2.mean()),
            "clears": bool(lo > 0 and d.mean() >= MDE)}

    ns = nested["nested_summary"]
    print("\nNESTED-CV vs RESAMPLED-TEST -- the search-overfitting check")
    print(f"  {'':<10}{'nested d':>12}{'resampled d':>14}{'agree?':>10}")
    for k, nd in (("optuna", ns["delta_search"]), ("legacy", ns["delta_legacy"])):
        rd = out["resampled"][k]["mean_delta"]
        print(f"  {k:<10}{nd:>+12.4f}{rd:>+14.4f}"
              f"{('yes' if np.sign(nd) == np.sign(rd) else 'NO -- DISAGREE'):>10}")
    print(f"\ninner-CV best {ns['inner_best_mean']:.4f} vs its own outer "
          f"{ns['searched']:.4f} = {ns['inner_best_mean']-ns['searched']:+.4f} "
          "optimism from the search itself")
    print(f"\nA clearing arm needs ci_lo > 0 AND mean delta >= MDE ({MDE}).")
    print(f"wall clock {time.time()-t0:.0f}s")
    out["wall_clock_s"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
