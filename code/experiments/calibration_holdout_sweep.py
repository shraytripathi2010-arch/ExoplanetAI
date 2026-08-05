"""calibration_holdout_sweep.py -- the calibration-wrapper gaps, incl. prefit.

WHAT THIS ADDS THAT THE EARLIER SWEEP DID NOT

`calibration_sweep.py` covered {sigmoid, bag-only} x cv={3,5,10,20} plus
isotonic cv={5,10}, and stage 2 resampled every arm that beat production.
Three things in that sweep were missing:

  1. DEDICATED-HOLDOUT ("prefit") calibration -- entirely untested, and it is
     the whole mechanism hypothesis: base model fit on nearly all training
     data, a disjoint slice used ONLY to fit the calibrator.
  2. ECE -- only Brier was ever computed, and this is a calibration experiment.
  3. isotonic cv=20 -- a hole in the grid.

THE DESIGN POINT: HOLDOUT FRACTIONS ARE MATCHED TO FOLD COUNTS

Cross-fitting confounds two things -- each base model sees (k-1)/k of the data,
AND k models get averaged. Prefit has ONE model and NO averaging. Choosing

    f = 0.20  <->  cv=5   (base sees 80%)
    f = 0.10  <->  cv=10  (base sees 90%)
    f = 0.05  <->  cv=20  (base sees 95%)

makes each prefit arm share its per-fit training size with a cross-fit arm, so
the pair differs ONLY in averaging. That is what turns this from a survey into
a test of "CatBoost needs more data per fit than cross-fitting gives it".

Under that hypothesis prefit should RECOVER CatBoost's bare-model advantage.
If prefit instead lands at or below the matched cross-fit arm, the hypothesis
is refuted and the gain is averaging, not data volume.

CARVING THE HOLDOUT (requirement 4)

Training has exactly one row per star (5,485 rows / 5,485 unique hosts,
verified), so a stratified row split IS a star split -- no group machinery
needed and no star can straddle the base/calibration boundary. The frozen TEST
set is never touched: the holdout is carved out of the 4,386 training rows only.

BASELINE DISCIPLINE (the known trap)

Every delta below is against the production CONFIGURATION refit inside this
same run -- HGB + sigmoid + cv=5 -- never against the stored 0.9031 artifact
number, and never bare-vs-calibrated. Bare arms are reported for mechanism, and
their deltas are also vs that same calibrated production arm, so the table is
internally consistent.

sklearn 1.9 removed cv="prefit" (deprecated 1.6); FrozenEstimator is the
supported path and is what this uses.
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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import brier_score_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import roc_auc_score, fast_auc  # exact drop-ins, ~23x faster in loops

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "calibration_holdout_results.json")
# Raw per-arm test probabilities, checkpointed as each fit lands. The first run
# of this script lost 22 minutes of fitting to a KeyError in the aggregation
# step below; the fits are the expensive part and must not depend on the
# reporting code being correct. Re-running now reuses whatever is cached.
RAW_CACHE = os.path.join(SCRIPT_DIR, "calibration_holdout_raw.json")

SEED = 42
N_BOOT = 2000
N_WORKERS = 6
N_ECE_BINS = 15
CAT_PARAMS = dict(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=9.0)

PRODUCTION = ("HGB", "sigmoid cv=5")   # must match arm_name() exactly

# (kind, method, param)  param = folds for crossfit, holdout fraction for prefit
ARMS = [
    ("bare", None, None),
    ("crossfit", "sigmoid", 3),
    ("crossfit", "sigmoid", 5),
    ("crossfit", "sigmoid", 10),
    ("crossfit", "sigmoid", 20),
    ("crossfit", "isotonic", 5),
    ("crossfit", "isotonic", 10),
    ("crossfit", "isotonic", 20),
    ("prefit", "sigmoid", 0.20),
    ("prefit", "sigmoid", 0.10),
    ("prefit", "sigmoid", 0.05),
    ("prefit", "isotonic", 0.20),
    ("prefit", "isotonic", 0.10),
    ("prefit", "isotonic", 0.05),
]


def arm_name(kind, method, param):
    if kind == "bare":
        return "bare (no wrapper)"
    if kind == "crossfit":
        return f"{method} cv={param}"
    return f"{method} prefit holdout={int(param * 100)}%"


def ece(y, p, bins=N_ECE_BINS):
    """Expected calibration error, equal-width bins, |acc - conf| weighted."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


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
    tr, _ = m05.split_by_host(df)
    tr = np.asarray(tr)
    # frozen manifest test subset -- keeps this comparable to every earlier run
    frozen = m05.frozen_test_mask(df)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()
    prod = joblib.load(PROD)
    hgb = clone(getattr(prod, "estimator", prod))
    from catboost import CatBoostClassifier
    cat = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("clf", CatBoostClassifier(
                        verbose=0, random_seed=SEED,
                        auto_class_weights="Balanced",
                        allow_writing_files=False, **CAT_PARAMS))])
    _G.update(X=X, y=y, tr=tr, te=frozen, te2=frozen & is2,
              models={"HGB": hgb, "CatBoost": cat})


def fit_arm(family, kind, method, param):
    """Returns test probabilities + effective training sizes for one arm."""
    if not _G:
        _init()
    X, y, tr = _G["X"], _G["y"], _G["tr"]
    base = clone(_G["models"][family])
    Xtr, ytr = X[tr], y[tr]
    n = len(ytr)
    t0 = time.time()

    if kind == "bare":
        est = base.fit(Xtr, ytr)
        n_base, n_cal, n_models = n, 0, 1
    elif kind == "crossfit":
        est = CalibratedClassifierCV(base, cv=param, method=method).fit(Xtr, ytr)
        n_base, n_cal, n_models = int(n * (param - 1) / param), int(n / param), param
    else:
        Xb, Xh, yb, yh = train_test_split(
            Xtr, ytr, test_size=param, stratify=ytr, random_state=SEED)
        base.fit(Xb, yb)
        est = CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(Xh, yh)
        n_base, n_cal, n_models = len(yb), len(yh), 1

    te, te2 = _G["te"], _G["te2"]
    return {
        "family": family, "arm": arm_name(kind, method, param), "kind": kind,
        "method": method or "-", "param": param,
        "n_base_per_fit": n_base, "n_calibration": n_cal, "n_models": n_models,
        "fit_s": round(time.time() - t0, 1),
        "p_full": est.predict_proba(X[te])[:, 1].tolist(),
        "p_2min": est.predict_proba(X[te2])[:, 1].tolist(),
    }


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


def main():
    print("=" * 118)
    print("CALIBRATION-WRAPPER SWEEP -- cross-fit vs DEDICATED HOLDOUT (prefit)")
    print("=" * 118)
    print("  holdout fractions matched to fold counts: 20%<->cv=5, 10%<->cv=10,")
    print("  5%<->cv=20, so prefit vs cross-fit differs ONLY in averaging.")
    print(f"  baseline for every delta: {PRODUCTION[0]} {PRODUCTION[1]}, refit here.")
    print(f"  {len(ARMS) * 2} arms, {N_BOOT}-iteration paired bootstrap, "
          f"ECE with {N_ECE_BINS} bins.\n")

    # --dry-run exercises the reporting path on fabricated probabilities so a
    # bug there is found in seconds instead of after 22 minutes of fitting.
    # It writes NOTHING: not the cache, not the results file.
    dry = "--dry-run" in sys.argv
    if dry:
        print("  *** DRY RUN -- fabricated probabilities, no fits, no writes ***\n")

    _init()
    X, y, te, te2 = _G["X"], _G["y"], _G["te"], _G["te2"]
    y_full, y_2min = y[te], y[te2]
    print(f"  train {int(_G['tr'].sum())} / frozen test {int(te.sum())} "
          f"(2-min subset {int(te2.sum())})\n")

    all_jobs = [(f, k, m, p) for f in ("HGB", "CatBoost") for k, m, p in ARMS]

    cached = {}
    if dry:
        rng = np.random.RandomState(0)
        for fam, k, m, p in [(f, *a) for f in ("HGB", "CatBoost") for a in ARMS]:
            noise = rng.normal(0, 0.3, int(te.sum()))
            sc = 1 / (1 + np.exp(-(y[te] * 1.5 + noise)))
            cached[(fam, arm_name(k, m, p))] = {
                "family": fam, "arm": arm_name(k, m, p), "kind": k,
                "method": m or "-", "param": p, "n_base_per_fit": 4000,
                "n_calibration": 386, "n_models": 1, "fit_s": 0.0,
                "p_full": sc.tolist(),
                "p_2min": sc[: int(te2.sum())].tolist()}
    elif os.path.exists(RAW_CACHE):
        for r in json.load(open(RAW_CACHE)):
            cached[(r["family"], r["arm"])] = r
        print(f"  reusing {len(cached)} cached fit(s) from a previous run\n")

    raw = dict(cached)
    jobs = [j for j in all_jobs
            if (j[0], arm_name(j[1], j[2], j[3])) not in cached]

    def _checkpoint():
        if not dry:
            json.dump(list(raw.values()), open(RAW_CACHE, "w"))

    t0 = time.time()
    if jobs:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(fit_arm, *j): j for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                raw[(r["family"], r["arm"])] = r
                _checkpoint()          # survive any failure past this point
                print(f"  [{i}/{len(jobs)}] {r['family']:<9} {r['arm']:<32} "
                      f"{r['fit_s']:>7.1f}s  ({(time.time()-t0)/60:.1f} min)",
                      flush=True)

    missing = [k for k in ((f, arm_name(k_, m, p))
                           for f in ("HGB", "CatBoost") for k_, m, p in ARMS)
               if k not in raw]
    if missing:
        raise SystemExit(f"missing arms after fitting: {missing}")

    base_p = np.asarray(raw[PRODUCTION]["p_full"])
    base_p2 = np.asarray(raw[PRODUCTION]["p_2min"])

    rows = []
    for (family, arm), r in raw.items():
        pf, p2 = np.asarray(r["p_full"]), np.asarray(r["p_2min"])
        d, lo, hi = paired_boot(y_full, base_p, pf)
        d2, lo2, hi2 = paired_boot(y_2min, base_p2, p2)
        rows.append({
            "family": family, "arm": arm, "kind": r["kind"],
            "method": r["method"], "param": r["param"],
            "n_base_per_fit": r["n_base_per_fit"],
            "n_calibration": r["n_calibration"], "n_models": r["n_models"],
            "fit_s": r["fit_s"],
            "auc_full": float(roc_auc_score(y_full, pf)),
            "auc_2min": float(roc_auc_score(y_2min, p2)),
            "brier": float(brier_score_loss(y_full, pf)),
            "ece": ece(y_full, pf),
            "ece_2min": ece(y_2min, p2),
            "delta_full": d, "ci_full": [lo, hi], "clears_full": bool(lo > 0),
            "delta_2min": d2, "ci_2min": [lo2, hi2], "clears_2min": bool(lo2 > 0),
        })
    rows.sort(key=lambda r: (r["family"], -r["auc_full"]))

    hdr = (f"  {'arm':<32}{'n_base':>8}{'mdl':>5}{'AUC':>9}{'AUC2m':>9}"
           f"{'Brier':>9}{'ECE':>8}{'delta':>10}{'ci_lo':>9}{'ci_hi':>9}{'clr':>5}")
    for family in ("HGB", "CatBoost"):
        print("\n" + "=" * 118)
        print(f"{family}")
        print("=" * 118)
        print(hdr)
        for r in [x for x in rows if x["family"] == family]:
            mark = " *" if r["clears_full"] else ""
            print(f"  {r['arm']:<32}{r['n_base_per_fit']:>8}{r['n_models']:>5}"
                  f"{r['auc_full']:>9.4f}{r['auc_2min']:>9.4f}{r['brier']:>9.4f}"
                  f"{r['ece']:>8.4f}{r['delta_full']:>+10.4f}"
                  f"{r['ci_full'][0]:>+9.4f}{r['ci_full'][1]:>+9.4f}"
                  f"{('yes' if r['clears_full'] else 'no'):>5}{mark}")

    print("\n" + "=" * 118)
    print("MECHANISM TEST -- prefit vs cross-fit at MATCHED per-fit training size")
    print("=" * 118)
    print(f"  {'family':<10}{'matched pair':<46}{'prefit':>9}{'crossfit':>10}"
          f"{'prefit-crossfit':>17}")
    mech = []
    for family in ("HGB", "CatBoost"):
        for frac, folds in ((0.20, 5), (0.10, 10), (0.05, 20)):
            a = arm_name("prefit", "sigmoid", frac)
            b = arm_name("crossfit", "sigmoid", folds)
            ra = next(r for r in rows if r["family"] == family and r["arm"] == a)
            rb = next(r for r in rows if r["family"] == family and r["arm"] == b)
            gap = ra["auc_full"] - rb["auc_full"]
            label = f"holdout {int(frac*100)}% vs cv={folds} (~{ra['n_base_per_fit']} rows/fit)"
            print(f"  {family:<10}{label:<46}{ra['auc_full']:>9.4f}"
                  f"{rb['auc_full']:>10.4f}{gap:>+17.4f}")
            mech.append({"family": family, "holdout": frac, "folds": folds,
                         "prefit_auc": ra["auc_full"], "crossfit_auc": rb["auc_full"],
                         "gap": float(gap),
                         "n_base_prefit": ra["n_base_per_fit"],
                         "n_base_crossfit": rb["n_base_per_fit"]})

    clearing = [r for r in rows if r["clears_full"]]
    print("\n" + "=" * 118)
    if clearing:
        print(f"CLEARS ci_lo>0 ON THIS SINGLE FIT ({len(clearing)}) -- "
              f"NOT a finding until resampled:")
        for r in clearing:
            print(f"    {r['family']:<9} {r['arm']:<32} {r['delta_full']:+.4f}  "
                  f"ci [{r['ci_full'][0]:+.4f}, {r['ci_full'][1]:+.4f}]")
    else:
        print("NO ARM CLEARS ci_lo>0 EVEN ON A SINGLE FIT")
    print("=" * 118)

    out = {"baseline": f"{PRODUCTION[0]} {PRODUCTION[1]} (refit in this run)",
           "baseline_auc_full": float(roc_auc_score(y_full, base_p)),
           "baseline_auc_2min": float(roc_auc_score(y_2min, base_p2)),
           "baseline_brier": float(brier_score_loss(y_full, base_p)),
           "baseline_ece": ece(y_full, base_p),
           "n_train": int(_G["tr"].sum()), "n_test": int(te.sum()),
           "n_test_2min": int(te2.sum()),
           "ece_bins": N_ECE_BINS, "n_boot": N_BOOT,
           "rows": rows, "mechanism": mech,
           "single_fit_clearing": [{"family": r["family"], "arm": r["arm"],
                                    "delta": r["delta_full"]} for r in clearing]}
    if dry:
        print("\nDRY RUN complete -- reporting path exercised, nothing written.")
        return
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
