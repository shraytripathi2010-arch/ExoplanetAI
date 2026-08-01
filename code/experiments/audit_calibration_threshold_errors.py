"""audit_calibration_threshold_errors.py -- calibration, threshold selection and
error analysis for the CURRENTLY DEPLOYED model. Diagnostic: retrains nothing,
promotes nothing, changes no live setting.

The deployed model is already HistGradientBoosting + SIGMOID calibration, so
part 1 is not "is it calibrated at all" but "does the sigmoid in place actually
produce honest probabilities, and would isotonic beat it".

METHODOLOGY NOTE that determines whether any of this is trustworthy:

To ask "would recalibration help", a recalibrator must be FIT on data and
EVALUATED on different data. The obvious shortcut -- fit isotonic on the
training split -- is invalid here, because the production model was itself fit
on that split, so its training-set probabilities are optimistically biased and
isotonic would learn to undo a distortion that does not exist on unseen stars.

Instead: K-fold WITHIN the frozen test set. For each fold, fit the recalibrator
on the other K-1 folds' production probabilities and score the held-out fold.
Every star is scored by a recalibrator that never saw it, the frozen
train/test boundary is untouched, and the training split is not used at all.

Adoption bar is the project standard, not a softer one: paired bootstrap on the
Brier-score difference must have its 95% CI entirely on the improvement side.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, brier_score_loss, log_loss,
                             precision_recall_curve, precision_score,
                             recall_score, f1_score)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
MODEL_PATH = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "audit_calibration_threshold_errors.json")

RANDOM_SEED = 42
N_BOOTSTRAP = 2000
N_CALIB_FOLDS = 5
N_RELIABILITY_BINS = 10


def _load_m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def expected_calibration_error(y, p, n_bins=N_RELIABILITY_BINS):
    """Weighted mean |predicted - observed| across probability bins. This is
    the number that matters for 'is 0.96 literally 96%'."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, edges[1:-1], right=False)
    ece, mce, rows = 0.0, 0.0, []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc, w = p[m].mean(), y[m].mean(), m.sum() / len(y)
        gap = abs(conf - acc)
        ece += w * gap
        mce = max(mce, gap)
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                     "mean_predicted": float(conf), "observed_frequency": float(acc),
                     "gap": float(conf - acc)})
    return float(ece), float(mce), rows


def paired_bootstrap_metric(y, p_a, p_b, metric, seed=RANDOM_SEED, n=N_BOOTSTRAP):
    """CI on metric(b) - metric(a). For Brier/log-loss, negative = b better."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(metric(y[i], p_b[i]) - metric(y[i], p_a[i]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def recalibrate_oof(y, p, kind, seed=RANDOM_SEED):
    """Out-of-fold recalibration inside the test set -- see module docstring."""
    out = np.zeros_like(p)
    skf = StratifiedKFold(n_splits=N_CALIB_FOLDS, shuffle=True, random_state=seed)
    for tr, te in skf.split(p.reshape(-1, 1), y):
        if kind == "isotonic":
            m = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr])
            out[te] = m.predict(p[te])
        elif kind == "sigmoid":
            m = LogisticRegression().fit(p[tr].reshape(-1, 1), y[tr])
            out[te] = m.predict_proba(p[te].reshape(-1, 1))[:, 1]
    return np.clip(out, 1e-6, 1 - 1e-6)


def main():
    res = {}
    m05 = _load_m05()
    df = pd.read_csv(TRAINING_CSV)
    X, y = m05.build_feature_matrix(df)
    train_mask, test_mask = m05.split_by_host(df)
    model = joblib.load(MODEL_PATH)

    Xte, yte = X[test_mask], np.asarray(y[test_mask])
    p = model.predict_proba(Xte)[:, 1]
    dfte = df[test_mask].reset_index(drop=True)

    print("=" * 74)
    print("AUDIT OF THE DEPLOYED MODEL -- calibration / thresholds / errors")
    print("=" * 74)
    print(f"model: {MODEL_PATH}")
    print(f"test stars: {len(yte)}  ({int(yte.sum())} positive, {int((1-yte).sum())} negative)")
    print(f"ROC-AUC: {roc_auc_score(yte, p):.4f}")
    res["n_test"] = int(len(yte))
    res["roc_auc"] = float(roc_auc_score(yte, p))

    # ---------------- 1. CALIBRATION ----------------
    print("\n" + "=" * 74)
    print("1. CALIBRATION OF THE CURRENTLY DEPLOYED MODEL")
    print("=" * 74)
    ece, mce, rows = expected_calibration_error(yte, p)
    brier = brier_score_loss(yte, p)
    ll = log_loss(yte, p)
    print(f"Brier {brier:.4f} | log-loss {ll:.4f} | ECE {ece:.4f} | max bin gap {mce:.4f}")
    print("\nreliability table (gap = predicted - observed; + means OVERCONFIDENT):")
    print(f"  {'bin':<12}{'n':>6}{'predicted':>12}{'observed':>11}{'gap':>9}")
    for r in rows:
        print(f"  {r['bin']:<12}{r['n']:>6}{r['mean_predicted']:>12.3f}"
              f"{r['observed_frequency']:>11.3f}{r['gap']:>+9.3f}")
    res["calibration_current"] = {"brier": float(brier), "log_loss": float(ll),
                                  "ece": ece, "max_gap": mce, "bins": rows}

    # High-confidence region specifically -- that is what the UI shows.
    for thr in (0.9, 0.95):
        m = p >= thr
        if m.sum():
            print(f"\n  stars scored >= {thr}: n={m.sum()}, mean predicted "
                  f"{p[m].mean():.3f}, actually positive {yte[m].mean():.3f}")
            res[f"highconf_{thr}"] = {"n": int(m.sum()),
                                      "mean_predicted": float(p[m].mean()),
                                      "observed": float(yte[m].mean())}

    print("\n--- would recalibration help? (out-of-fold within the test set) ---")
    res["recalibration"] = {}
    for kind in ("isotonic", "sigmoid"):
        pc = recalibrate_oof(yte, p, kind)
        b2 = brier_score_loss(yte, pc)
        e2 = expected_calibration_error(yte, pc)[0]
        mean_d, lo, hi = paired_bootstrap_metric(yte, p, pc, brier_score_loss)
        clears = hi < 0  # entire CI on the improvement side
        print(f"  {kind:9s} Brier {b2:.4f} (vs {brier:.4f})  ECE {e2:.4f} (vs {ece:.4f})  "
              f"AUC {roc_auc_score(yte, pc):.4f}")
        print(f"            bootstrap dBrier {mean_d:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]"
              f"  -> {'CLEARS' if clears else 'does not clear'}")
        res["recalibration"][kind] = {"brier": float(b2), "ece": float(e2),
                                      "auc": float(roc_auc_score(yte, pc)),
                                      "mean_dbrier": mean_d, "ci_lower": lo,
                                      "ci_upper": hi, "clears_bar": bool(clears)}

    # ---------------- 2. THRESHOLDS ----------------
    print("\n" + "=" * 74)
    print("2. THRESHOLD SELECTION")
    print("=" * 74)
    print(f"  {'thr':>6}{'precision':>11}{'recall':>9}{'F1':>8}{'F2':>8}"
          f"{'flagged':>9}{'missed':>8}")
    grid = []
    for t in np.arange(0.05, 0.96, 0.05):
        pred = (p >= t).astype(int)
        if pred.sum() == 0:
            continue
        pr, rc = precision_score(yte, pred), recall_score(yte, pred)
        f1 = f1_score(yte, pred)
        f2 = (5 * pr * rc / (4 * pr + rc)) if (pr + rc) else 0.0
        missed = int(((yte == 1) & (pred == 0)).sum())
        grid.append({"threshold": round(float(t), 2), "precision": float(pr),
                     "recall": float(rc), "f1": float(f1), "f2": float(f2),
                     "n_flagged": int(pred.sum()), "n_missed_planets": missed})
        print(f"  {t:>6.2f}{pr:>11.3f}{rc:>9.3f}{f1:>8.3f}{f2:>8.3f}"
              f"{pred.sum():>9}{missed:>8}")
    res["threshold_grid"] = grid
    best_f1 = max(grid, key=lambda r: r["f1"])
    best_f2 = max(grid, key=lambda r: r["f2"])
    print(f"\n  best F1 at {best_f1['threshold']} (F1={best_f1['f1']:.3f})")
    print(f"  best F2 at {best_f2['threshold']} (F2={best_f2['f2']:.3f})  "
          f"[F2 weights recall 2x -- closer to this project's actual use]")
    res["best_f1"], res["best_f2"] = best_f1, best_f2

    # ---------------- 3. ERROR ANALYSIS ----------------
    print("\n" + "=" * 74)
    print("3. ERROR ANALYSIS AT THE 0.5 OPERATING POINT")
    print("=" * 74)
    pred = (p >= 0.5).astype(int)
    fp = (pred == 1) & (yte == 0)
    fn = (pred == 0) & (yte == 1)
    print(f"false positives: {fp.sum()}   false negatives: {fn.sum()}")
    res["n_fp"], res["n_fn"] = int(fp.sum()), int(fn.sum())

    dims = ["period", "depth", "SDE", "snr", "transit_count", "st_rad", "st_teff",
            "duration", "odd_even_mismatch"]
    res["error_by_dimension"] = {}
    for col in dims:
        if col not in dfte.columns:
            continue
        v = pd.to_numeric(dfte[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if v.notna().sum() < 50:
            continue
        try:
            q = pd.qcut(v, 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"], duplicates="drop")
        except ValueError:
            continue
        print(f"\n  {col}:")
        print(f"    {'quartile':<10}{'n':>6}{'range':>24}{'err%':>8}{'FP':>5}{'FN':>5}")
        entry = []
        for lvl in q.cat.categories:
            m = (q == lvl).to_numpy()
            if m.sum() == 0:
                continue
            err = ((pred[m] != yte[m]).mean()) * 100
            rng = f"[{v[m].min():.3g}, {v[m].max():.3g}]"
            print(f"    {str(lvl):<10}{m.sum():>6}{rng:>24}{err:>8.1f}"
                  f"{int((fp & m).sum()):>5}{int((fn & m).sum()):>5}")
            entry.append({"quartile": str(lvl), "n": int(m.sum()), "error_pct": float(err),
                          "fp": int((fp & m).sum()), "fn": int((fn & m).sum()),
                          "range": rng})
        res["error_by_dimension"][col] = entry
        errs = [e["error_pct"] for e in entry]
        if errs and (max(errs) - min(errs)) > 10:
            print(f"    ^ SPREAD {max(errs)-min(errs):.1f} pp across quartiles "
                  f"-- candidate pattern")

    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
