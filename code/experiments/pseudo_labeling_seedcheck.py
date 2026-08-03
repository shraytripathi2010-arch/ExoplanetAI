"""pseudo_labeling_seedcheck.py -- the decisive replication test.

Five pseudo-labelling arms cleared ci_lo > 0 on a single fit, the strongest
(`abs0.95 w=0.25`) on both test populations at +0.0092 / +0.0081. This project
has already had exactly one arm clear on a single fit and then die on
replication -- CatBoost, 0/10 seeds -- so a single-fit clear is treated as a
hypothesis, not a result.

WHAT MAKES THIS TEST DIFFERENT FROM THE CATBOOST ONE

For CatBoost, only the model fit was stochastic. Here the PSEUDO-LABELS
themselves come from a model, so seed variation propagates twice:

    seed -> baseline fit -> confident predictions -> pseudo-labels -> final fit

Holding the labels fixed while varying only the final fit would understate the
variance badly, because it would treat one particular draw of the labels as
given. So each seed regenerates the whole chain: fit the baseline with that
seed, take ITS confident predictions as pseudo-labels, retrain, and compare
against the same-seed baseline. That is the quantity that matters -- would this
procedure, run fresh, produce a gain?
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
RESULTS = os.path.join(SCRIPT_DIR, "pseudo_labeling_seedcheck.json")

N_SEEDS = 10
N_BOOT = 2000
BASE_SEED = 2000


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def boot(y, pa, pb, n=N_BOOT, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5))


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

    frames = []
    for f in UNKNOWN_FILES:
        if os.path.exists(f):
            frames.append(pd.read_csv(f))
    u = pd.concat(frames, ignore_index=True)
    u["tic"] = (pd.to_numeric(u["tic_id"], errors="coerce") if "tic_id" in u.columns
                else pd.to_numeric(u["host"].astype(str).str.extract(r"(\d+)", expand=False),
                                   errors="coerce"))
    u = u.dropna(subset=["tic"]).drop_duplicates("tic").reset_index(drop=True)
    Xu, _ = m05.build_feature_matrix(u.assign(label=0))
    Xu = Xu.reset_index(drop=True)

    proto = getattr(joblib.load(PROD), "estimator", joblib.load(PROD))

    print("=" * 84)
    print(f"SEED CHECK -- {N_SEEDS} seeds, pseudo-labels REGENERATED per seed")
    print("=" * 84)
    print("arm under test: abs0.95 w=0.25 (p>=0.95 pseudo-positive, p<=0.05")
    print("pseudo-negative, pseudo rows weighted 0.25) -- the strongest single-fit")
    print("result, which cleared on BOTH populations.\n")
    print(f"  {'seed':>5}{'n_pseudo':>10}{'base full':>11}{'pseudo full':>13}"
          f"{'d_full':>9}{'lo':>9}{'':>3}{'d_2min':>9}{'lo':>9}")

    rows = []
    for i in range(N_SEEDS):
        sd = BASE_SEED + i
        # 1. baseline fit at this seed
        bp = clone(proto)
        bp.set_params(clf__random_state=sd)
        bp.fit(X[tr], y[tr])
        bf = bp.predict_proba(X[te])[:, 1]
        b2 = bp.predict_proba(X[te2])[:, 1]

        # 2. THIS model's confident predictions become the pseudo-labels
        pu = bp.predict_proba(Xu)[:, 1]
        pos, neg = pu >= 0.95, pu <= 0.05
        sel = pos | neg
        if sel.sum() < 4:
            print(f"  {sd:>5}{int(sel.sum()):>10}   too few pseudo-labels at this seed")
            continue
        yps = np.where(pos[sel], 1, 0)

        # 3. retrain with them, same seed
        mp = clone(proto)
        mp.set_params(clf__random_state=sd)
        sw = np.r_[np.ones(int(tr.sum())), np.full(int(sel.sum()), 0.25)]
        mp.fit(pd.concat([X[tr], Xu[sel]], ignore_index=True),
               np.r_[y[tr], yps], clf__sample_weight=sw)
        pf = mp.predict_proba(X[te])[:, 1]
        p2 = mp.predict_proba(X[te2])[:, 1]

        mf, lof = boot(y[te], bf, pf)
        m2, lo2 = boot(y[te2], b2, p2)
        rows.append({"seed": sd, "n_pseudo": int(sel.sum()),
                     "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
                     "base_full": float(roc_auc_score(y[te], bf)),
                     "pseudo_full": float(roc_auc_score(y[te], pf)),
                     "delta_full": mf, "ci_lo_full": lof, "clears_full": bool(lof > 0),
                     "delta_2min": m2, "ci_lo_2min": lo2, "clears_2min": bool(lo2 > 0)})
        print(f"  {sd:>5}{int(sel.sum()):>10}{rows[-1]['base_full']:>11.4f}"
              f"{rows[-1]['pseudo_full']:>13.4f}{mf:>+9.4f}{lof:>+9.4f}"
              f"{'  C' if lof>0 else '  .':>3}{m2:>+9.4f}{lo2:>+9.4f}"
              f"{'  C' if lo2>0 else '  .'}")

    r = pd.DataFrame(rows)
    print("\n  " + "-" * 78)
    print(f"  n_pseudo per seed : mean {r.n_pseudo.mean():.1f} "
          f"(range {r.n_pseudo.min()}-{r.n_pseudo.max()})")
    print(f"  delta_full : mean {r.delta_full.mean():+.4f}  sd {r.delta_full.std():.4f}"
          f"  min {r.delta_full.min():+.4f}  max {r.delta_full.max():+.4f}")
    print(f"  delta_2min : mean {r.delta_2min.mean():+.4f}  sd {r.delta_2min.std():.4f}"
          f"  min {r.delta_2min.min():+.4f}  max {r.delta_2min.max():+.4f}")
    print(f"  seeds clearing on full test  : {int(r.clears_full.sum())}/{len(r)}")
    print(f"  seeds clearing on 2-min-only : {int(r.clears_2min.sum())}/{len(r)}")

    frac = r.clears_full.mean()
    print("\n" + "=" * 84)
    if frac >= 0.9:
        v = ("ROBUST -- the gain reproduces on essentially every seed. Given the "
             "safeguard failure this would still need explaining, not adopting.")
    elif frac >= 0.5:
        v = ("PARTIALLY ROBUST -- clears on some seeds, not others. Too close to "
             "the bar for a single fit to settle.")
    else:
        v = ("NOT ROBUST -- the single-fit clear was substantially a seed draw, "
             "exactly as with CatBoost.")
    print(v)
    print("=" * 84)

    with open(RESULTS, "w") as f:
        json.dump({"rows": rows, "n_seeds": len(r),
                   "delta_full_mean": float(r.delta_full.mean()),
                   "delta_full_sd": float(r.delta_full.std()),
                   "delta_2min_mean": float(r.delta_2min.mean()),
                   "delta_2min_sd": float(r.delta_2min.std()),
                   "n_clearing_full": int(r.clears_full.sum()),
                   "n_clearing_2min": int(r.clears_2min.sum()),
                   "verdict": v}, f, indent=2, default=float)
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
