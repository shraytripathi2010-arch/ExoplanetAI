"""pseudo_labeling_replication.py -- replication test done on the right axis.

A FIRST ATTEMPT AT THIS WAS WRONG, AND IS RECORDED RATHER THAN QUIETLY REPLACED

`pseudo_labeling_seedcheck.py` varied `random_state` across 10 seeds and
reported 0/10 clearing. That number was meaningless: every seed returned an
IDENTICAL delta (sd exactly 0.0000). The cause is that sklearn's
HistGradientBoosting has `early_stopping='auto'`, which disables early stopping
at n <= 10,000, and its binning only subsamples above 10,000 rows. With 4,387
training rows the fit is fully DETERMINISTIC and `random_state` changes nothing.
The test varied nothing and could only ever have returned 0/10.

(The earlier CatBoost seed check was valid -- CatBoost is genuinely stochastic
and produced real spread, sd 0.0024. The mistake here was assuming that
transferred to HGB.)

That same broken run also generated its pseudo-labels from the BARE pipeline
rather than the calibrated production model, yielding 208 pseudo-labels instead
of the 43 in the arm that actually cleared. Two different arms.

WHAT REPLICATION MEANS FOR A DETERMINISTIC LEARNER

If the fit is deterministic given the data, then the single-fit +0.0092 is not
a lucky seed -- it is exactly reproducible. Fit variance is not the threat. The
threat is sensitivity to the TRAINING DATA DRAW, so that is what gets varied:
bootstrap-resample the training rows, regenerate the pseudo-labels from a model
fit on that resample, retrain, and compare against a baseline fit on the SAME
resample. Repeated many times, that measures whether the procedure produces a
gain in general or only on this particular training set.

The test set is never resampled -- it stays the frozen real-label set, as in
every other experiment here.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE_CSV = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
UNKNOWN_FILES = [
    os.path.join(ROOT, "results", "unknown_candidates", "ranked_candidates.csv"),
    os.path.join(ROOT, "results", "unknown_candidates_widesector", "ranked_candidates.csv"),
]
RESULTS = os.path.join(SCRIPT_DIR, "pseudo_labeling_replication.json")

N_REPS = 20
HI, LO, W = 0.95, 0.05, 0.25


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def main():
    m05 = _m05()
    df = pd.read_csv(TRAINING_CSV)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True); y = np.asarray(y)
    tr, te = m05.split_by_host(df)
    tr, te = np.asarray(tr), np.asarray(te)
    cad = pd.read_csv(CADENCE_CSV)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is_2min = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    te2 = te & is_2min

    frames = [pd.read_csv(f) for f in UNKNOWN_FILES if os.path.exists(f)]
    u = pd.concat(frames, ignore_index=True)
    u["tic"] = (pd.to_numeric(u["tic_id"], errors="coerce") if "tic_id" in u.columns
                else pd.to_numeric(u["host"].astype(str).str.extract(r"(\d+)", expand=False),
                                   errors="coerce"))
    u = u.dropna(subset=["tic"]).drop_duplicates("tic").reset_index(drop=True)
    Xu, _ = m05.build_feature_matrix(u.assign(label=0))
    Xu = Xu.reset_index(drop=True)

    prod = joblib.load(PROD)
    proto_cal = clone(prod)                       # calibrated: generates labels
    proto_bare = clone(getattr(prod, "estimator", prod))   # bare: accepts sample_weight

    Xtr, ytr = X[tr], y[tr]
    n = len(ytr)

    print("=" * 84)
    print(f"REPLICATION ON THE TRAINING-DATA AXIS -- {N_REPS} bootstrap resamples")
    print("=" * 84)
    print("HGB is deterministic at this data size, so the single-fit +0.0092 is")
    print("exactly reproducible and seed variation is not the threat. What is")
    print("tested here is whether the PROCEDURE gains on a different training draw.\n")
    print(f"  {'rep':>4}{'n_pseudo':>10}{'base full':>11}{'pseudo full':>13}"
          f"{'d_full':>10}{'d_2min':>10}")

    rows = []
    rng = np.random.RandomState(7)
    for rep in range(N_REPS):
        idx = rng.randint(0, n, n)
        Xb, yb = Xtr.iloc[idx], ytr[idx]
        if len(np.unique(yb)) < 2:
            continue
        # baseline on this resample
        b = clone(proto_bare).fit(Xb, yb)
        bf = b.predict_proba(X[te])[:, 1]; b2 = b.predict_proba(X[te2])[:, 1]
        # pseudo-labels from a CALIBRATED model on the same resample (matches Part 2)
        gen = clone(proto_cal).fit(Xb, yb)
        pu = gen.predict_proba(Xu)[:, 1]
        pos, neg = pu >= HI, pu <= LO
        sel = pos | neg
        if sel.sum() < 4:
            continue
        yps = np.where(pos[sel], 1, 0)
        sw = np.r_[np.ones(len(yb)), np.full(int(sel.sum()), W)]
        mp = clone(proto_bare).fit(pd.concat([Xb, Xu[sel]], ignore_index=True),
                                   np.r_[yb, yps], clf__sample_weight=sw)
        pf = mp.predict_proba(X[te])[:, 1]; p2 = mp.predict_proba(X[te2])[:, 1]
        d_f = roc_auc_score(y[te], pf) - roc_auc_score(y[te], bf)
        d_2 = roc_auc_score(y[te2], p2) - roc_auc_score(y[te2], b2)
        rows.append({"rep": rep, "n_pseudo": int(sel.sum()),
                     "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
                     "base_full": float(roc_auc_score(y[te], bf)),
                     "pseudo_full": float(roc_auc_score(y[te], pf)),
                     "delta_full": float(d_f), "delta_2min": float(d_2)})
        print(f"  {rep:>4}{int(sel.sum()):>10}{rows[-1]['base_full']:>11.4f}"
              f"{rows[-1]['pseudo_full']:>13.4f}{d_f:>+10.4f}{d_2:>+10.4f}")

    r = pd.DataFrame(rows)
    print("\n  " + "-" * 78)
    print(f"  n_pseudo   : mean {r.n_pseudo.mean():.1f}  range {r.n_pseudo.min()}-{r.n_pseudo.max()}"
          f"  (pos {r.n_pos.mean():.1f} / neg {r.n_neg.mean():.1f})")
    print(f"  delta_full : mean {r.delta_full.mean():+.4f}  sd {r.delta_full.std():.4f}"
          f"  min {r.delta_full.min():+.4f}  max {r.delta_full.max():+.4f}")
    print(f"  delta_2min : mean {r.delta_2min.mean():+.4f}  sd {r.delta_2min.std():.4f}"
          f"  min {r.delta_2min.min():+.4f}  max {r.delta_2min.max():+.4f}")
    pos_frac = float((r.delta_full > 0).mean())
    print(f"  resamples with a POSITIVE delta on full test : "
          f"{int((r.delta_full > 0).sum())}/{len(r)} ({pos_frac:.0%})")
    print(f"  resamples with a POSITIVE delta on 2-min     : "
          f"{int((r.delta_2min > 0).sum())}/{len(r)}")

    # one-sample test on the resample deltas
    from scipy import stats
    t, p = stats.ttest_1samp(r.delta_full, 0.0)
    print(f"\n  mean delta vs 0 across resamples: t={t:.2f}, p={p:.4f}")

    print("\n" + "=" * 84)
    if pos_frac >= 0.9 and p < 0.05:
        v = ("CONSISTENT -- the procedure gains across training draws. But the "
             "Part 1 safeguard failed and low-SNR performance dropped, so this is "
             "a real effect with a documented harm, not a clean win.")
    elif pos_frac >= 0.6:
        v = ("MIXED -- gains on most training draws but not reliably. Combined "
             "with the safeguard failure, not adoptable.")
    else:
        v = ("NOT CONSISTENT -- the gain does not survive a different training "
             "draw. The single-fit clear was an artefact of this particular "
             "training set.")
    print(v)
    print("=" * 84)

    with open(RESULTS, "w") as f:
        json.dump({"rows": rows, "n_reps": len(r),
                   "delta_full_mean": float(r.delta_full.mean()),
                   "delta_full_sd": float(r.delta_full.std()),
                   "delta_2min_mean": float(r.delta_2min.mean()),
                   "delta_2min_sd": float(r.delta_2min.std()),
                   "frac_positive_full": pos_frac,
                   "ttest_t": float(t), "ttest_p": float(p),
                   "verdict": v}, f, indent=2, default=float)
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
