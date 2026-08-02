"""injection_diagnostic.py -- does synthetic data move the classifier?

THE QUESTION THIS ANSWERS

The production model has sat at ~0.90 test ROC-AUC through eleven consecutive
feature/architecture experiments. Two explanations remain:

  (A) DATA-STARVED. The feature set is fine; there simply aren't enough
      examples. If so, Kepler's thousands of additional real labelled stars is
      the high-value next move.
  (B) FEATURE-STARVED. The 24 TLS-derived features have extracted essentially
      all the signal they can express, and no volume of additional examples in
      the same representation will help. If so, Kepler is low-value and the
      architecture (CNN on raw flux) is the remaining lever.

Injection-recovery gives a cheap probe. Synthetic transits injected into real
light curves produce genuinely new training rows with known labels, at a scale
real data cannot reach quickly. If adding them lifts held-out performance on
REAL stars, (A) gains support. If it does nothing -- matching the flat result
from the learning curve, the TOI-FA expansion, hyperparameter tuning and
ensembling -- (B) gains support.

WHAT SYNTHETIC DATA IS AND ISN'T

A synthetic positive is a real star's real noise with a batman transit
multiplied in. It is NOT evidentiary equivalent to a confirmed planet: the
signal is idealised (circular orbit, fixed quadratic limb darkening, no TTVs,
no stellar activity correlated with the transit), and the host is always a
negative-class star. This experiment is a controlled diagnostic, not a claim
that synthetic examples substitute for real detections. Nothing here is
promoted to production regardless of outcome.

THREE THINGS THIS RUN DOES THAT THE ORIGINAL PART B RUN DID NOT

1. FROZEN SPLIT. The original used a positional `train_test_split`, predating
   the split freeze. This uses `m05.split_by_host`, so the test set is the
   same 1,140 rows every other current experiment reports against.

2. DECONTAMINATION. 163 of the original 953 synthetic rows were injected into
   HELD-OUT TEST stars, leaking those stars' noise and stellar parameters into
   training. Those rows are dropped here, and the new batch
   (`augment_train_only.py`) cannot contain any by construction.

3. A DISTRIBUTION-SHIFT CONTROL. "Synthetic data didn't help" has two very
   different causes: the model is not data-starved, or the synthetic rows are
   simply off-distribution and act as label noise. These are distinguishable
   -- train a classifier to tell real rows from synthetic ones. If it can do
   that easily, the rows are off-distribution and a null result says little
   about data-starvation. This control is the difference between a diagnostic
   and a number.

A SCALE CURVE, NOT A SINGLE POINT. Performance is measured at several synthetic
volumes. A single "real+all-synthetic" number cannot distinguish "no effect"
from "helps then saturates" from "monotonically harmful"; the curve can.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     RandomizedSearchCV)
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
AUG_ORIGINAL = os.path.join(SCRIPT_DIR, "augmented_classical_dataset.csv")
AUG_TRAIN_ONLY = os.path.join(SCRIPT_DIR, "augmented_train_only.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "injection_diagnostic_results.json")

RANDOM_SEED = 42
N_BOOTSTRAP = 2000
N_CV = 5
DOWNWEIGHTS = [0.5, 0.25, 0.1]


def _load_m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def hgb():
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        class_weight="balanced", random_state=RANDOM_SEED)


def pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", hgb())])


def boot(y, pa, pb, n=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Paired bootstrap: the same resampled test indices score both models each
    iteration, so the CI is on the DIFFERENCE rather than on two independently
    noisy estimates."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
    d = np.array(d)
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


def ece(y, p, bins=10):
    idx = np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1])
    return float(sum(((idx == b).sum() / len(y)) * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def fit_eval(X_tr, y_tr, X_te, y_te, w=None):
    m = pipe()
    m.fit(X_tr, y_tr, **({"clf__sample_weight": w} if w is not None else {}))
    p = m.predict_proba(X_te)[:, 1]
    return p, roc_auc_score(y_te, p)


def load_synthetic(m05, real_df, test_hosts, res):
    """Assemble the synthetic pool from both batches, dropping any row whose
    source star is in the held-out test split."""
    frames = []
    for path, tag in ((AUG_ORIGINAL, "original"), (AUG_TRAIN_ONLY, "train_only")):
        if not os.path.exists(path):
            print(f"  {tag}: not present, skipped")
            continue
        d = pd.read_csv(path)
        ok = d[d["status"].astype(str) == "Success"].copy()
        src = ok["source_file"].astype(str).str.replace(".csv", "", regex=False)
        contaminated = src.isin(test_hosts)
        ok = ok[~contaminated.to_numpy()].copy()
        ok["batch"] = tag
        frames.append(ok)
        print(f"  {tag}: {len(d)} attempted, {int((d['status']=='Success').sum())} usable, "
              f"{int(contaminated.sum())} dropped as test-split-sourced, "
              f"{len(ok)} kept")
        res["batches"][tag] = {
            "attempted": int(len(d)),
            "usable": int((d["status"] == "Success").sum()),
            "dropped_test_sourced": int(contaminated.sum()),
            "kept": int(len(ok)),
        }
    if not frames:
        raise SystemExit("no synthetic rows available -- run augment_train_only.py first")
    syn = pd.concat(frames, ignore_index=True)
    return syn


def main():
    # --quick skips nested CV and calibration. Those two are ~95% of the
    # runtime and neither changes the DIRECTION of the result -- the held-out
    # test AUC and its bootstrap CI decide that. Quick mode exists so an
    # interim read is available while a multi-hour generation job is still
    # occupying the machine; the full suite is run once, at the end, on the
    # final dataset. It is not a substitute for the full run.
    quick = "--quick" in sys.argv
    res = {"batches": {}, "quick_mode": quick}
    if quick:
        print("[--quick] nested CV and calibration SKIPPED -- interim read only\n")
    m05 = _load_m05()

    real = pd.read_csv(TRAINING_CSV)
    X_real, y_real = m05.build_feature_matrix(real)
    X_real = X_real.reset_index(drop=True)
    y_real = np.asarray(y_real)
    tr, te = m05.split_by_host(real)
    test_hosts = set(real.loc[te, "host"].astype(str))

    X_tr_real, y_tr_real = X_real[tr], y_real[tr]
    X_te, y_te = X_real[te], y_real[te]
    print("=" * 78)
    print("INJECTION-RECOVERY DIAGNOSTIC -- is the ceiling data or features?")
    print("=" * 78)
    print(f"real: {tr.sum()} train / {te.sum()} test "
          f"({int(y_te.sum())} positive / {int((1-y_te).sum())} negative in test)")
    res["n_real_train"] = int(tr.sum())
    res["n_real_test"] = int(te.sum())

    print("\nsynthetic pool:")
    syn = load_synthetic(m05, real, test_hosts, res)
    X_syn, y_syn = m05.build_feature_matrix(syn)
    X_syn = X_syn.reset_index(drop=True)
    y_syn = np.asarray(y_syn)
    n_syn = len(X_syn)
    print(f"  TOTAL usable synthetic: {n_syn} "
          f"({int((y_syn==1).sum())} transit/positive, {int((y_syn==0).sum())} EB/negative)")
    print(f"  as a fraction of the real training set: {100*n_syn/tr.sum():.1f}%")
    res["n_synthetic"] = int(n_syn)
    res["n_synthetic_pos"] = int((y_syn == 1).sum())
    res["n_synthetic_neg"] = int((y_syn == 0).sum())

    # ---------------- DISTRIBUTION SHIFT CONTROL ----------------
    print("\n" + "=" * 78)
    print("CONTROL: are synthetic rows on-distribution with real ones?")
    print("=" * 78)
    X_dom = pd.concat([X_tr_real, X_syn], ignore_index=True)
    y_dom = np.r_[np.zeros(len(X_tr_real)), np.ones(len(X_syn))]
    cvd = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    dom = cross_validate(pipe(), X_dom, y_dom, cv=cvd, scoring="roc_auc")
    dom_auc = float(dom["test_score"].mean())
    print(f"  real-vs-synthetic discriminator AUC: {dom_auc:.4f}")
    print("  (0.5 = indistinguishable; ~1.0 = trivially separable, i.e. the")
    print("   synthetic rows are off-distribution and a null result would say")
    print("   little about data-starvation)")
    res["domain_shift_auc"] = dom_auc

    # per-feature standardized mean difference, to name WHICH features shift
    smd = {}
    for c in X_real.columns:
        a, b = X_tr_real[c].astype(float), X_syn[c].astype(float)
        sd = np.sqrt((a.var() + b.var()) / 2)
        if np.isfinite(sd) and sd > 0:
            smd[c] = float((b.mean() - a.mean()) / sd)
    top = sorted(smd.items(), key=lambda kv: -abs(kv[1]))[:8]
    print("\n  largest standardized mean differences (synthetic - real):")
    for k, v in top:
        print(f"    {k:26s} {v:+.2f}")
    res["standardized_mean_diff"] = smd

    # ---------------- MAIN ARMS ----------------
    print("\n" + "=" * 78)
    print("ARMS (frozen split; synthetic NEVER enters the test set)")
    print("=" * 78)
    X_plus = pd.concat([X_tr_real, X_syn], ignore_index=True)
    y_plus = np.r_[y_tr_real, y_syn]

    p_base, auc_base = fit_eval(X_tr_real, y_tr_real, X_te, y_te)
    print(f"  {'a. real only':34s} test AUC {auc_base:.4f}")
    res["arms"] = {"real_only": {"test_auc": float(auc_base)}}

    p_plus, auc_plus = fit_eval(X_plus, y_plus, X_te, y_te)
    m_, lo, hi = boot(y_te, p_base, p_plus)
    print(f"  {'b. real + synthetic':34s} test AUC {auc_plus:.4f}   "
          f"delta {m_:+.4f} CI [{lo:+.4f}, {hi:+.4f}] -> "
          f"{'CLEARS' if lo > 0 else 'does not clear'}")
    res["arms"]["real_plus_synthetic"] = {
        "test_auc": float(auc_plus), "mean_delta": m_, "ci_lower": lo,
        "ci_upper": hi, "clears": bool(lo > 0)}

    res["arms"]["downweighted"] = {}
    for w in DOWNWEIGHTS:
        sw = np.r_[np.ones(len(X_tr_real)), np.full(len(X_syn), w)]
        p_w, auc_w = fit_eval(X_plus, y_plus, X_te, y_te, w=sw)
        m2, lo2, hi2 = boot(y_te, p_base, p_w)
        print(f"  {f'c. real + synthetic (w={w})':34s} test AUC {auc_w:.4f}   "
              f"delta {m2:+.4f} CI [{lo2:+.4f}, {hi2:+.4f}] -> "
              f"{'CLEARS' if lo2 > 0 else 'does not clear'}")
        res["arms"]["downweighted"][str(w)] = {
            "test_auc": float(auc_w), "mean_delta": m2, "ci_lower": lo2,
            "ci_upper": hi2, "clears": bool(lo2 > 0)}

    # ---------------- SCALE CURVE ----------------
    print("\n" + "=" * 78)
    print("SCALE CURVE -- effect vs how much synthetic data is added")
    print("=" * 78)
    rng = np.random.RandomState(RANDOM_SEED)
    order = rng.permutation(n_syn)
    fracs = [0.125, 0.25, 0.5, 1.0]
    res["scale_curve"] = []
    for f in fracs:
        k = max(2, int(round(f * n_syn)))
        idx = order[:k]
        Xs, ys = X_syn.iloc[idx], y_syn[idx]
        Xc = pd.concat([X_tr_real, Xs], ignore_index=True)
        yc = np.r_[y_tr_real, ys]
        p_c, auc_c = fit_eval(Xc, yc, X_te, y_te)
        m3, lo3, hi3 = boot(y_te, p_base, p_c)
        print(f"  +{k:5d} synthetic ({100*f:5.1f}%)  test AUC {auc_c:.4f}   "
              f"delta {m3:+.4f} CI [{lo3:+.4f}, {hi3:+.4f}]")
        res["scale_curve"].append(
            {"n_synthetic": int(k), "fraction": f, "test_auc": float(auc_c),
             "mean_delta": m3, "ci_lower": lo3, "ci_upper": hi3})

    # ---------------- OVERLAP-CORRECTED ARMS ----------------
    # The domain-shift control above is not just a caveat generator -- it also
    # supplies the correction. If synthetic rows are separable from real ones,
    # the honest question becomes: do the synthetic rows that DO look real
    # help? Two standard covariate-shift corrections, both driven by
    # cross-validated p(synthetic) so no row is scored by a model that saw it:
    #
    #   REJECTION SAMPLING -- keep only synthetic rows in the overlap region
    #       (p(synthetic) below a threshold). Fewer rows, but on-distribution.
    #   IMPORTANCE WEIGHTING -- keep every row, weight it p(real)/p(synthetic),
    #       the textbook density-ratio correction. Rows deep in synthetic-only
    #       territory get driven toward zero weight automatically.
    #
    # If corrected synthetic data STILL fails to help, the null is much harder
    # to blame on distribution mismatch, and the feature-starved reading gets
    # real support. If it starts helping, the original null was an artefact of
    # the injector's realism and says nothing about data-starvation.
    print("\n" + "=" * 78)
    print("OVERLAP-CORRECTED ARMS (covariate-shift corrected)")
    print("=" * 78)
    from sklearn.model_selection import cross_val_predict
    p_syn = cross_val_predict(pipe(), X_dom, y_dom, cv=cvd,
                              method="predict_proba")[:, 1]
    p_syn_only = p_syn[len(X_tr_real):]
    res["overlap"] = {}

    for thr in (0.5, 0.75):
        keep = p_syn_only < thr
        n_keep = int(keep.sum())
        print(f"\n  rejection sampling, p(synthetic) < {thr}: "
              f"{n_keep}/{n_syn} synthetic rows survive ({100*n_keep/n_syn:.1f}%)")
        if n_keep < 30:
            print("    too few rows to train on -- skipped. That scarcity is itself")
            print("    the result: almost no synthetic row looks like a real one.")
            res["overlap"][f"reject_{thr}"] = {"n_kept": n_keep, "skipped": True}
            continue
        Xk = pd.concat([X_tr_real, X_syn[keep]], ignore_index=True)
        yk = np.r_[y_tr_real, y_syn[keep]]
        p_k, auc_k = fit_eval(Xk, yk, X_te, y_te)
        m4, lo4, hi4 = boot(y_te, p_base, p_k)
        print(f"    test AUC {auc_k:.4f}   delta {m4:+.4f} CI [{lo4:+.4f}, {hi4:+.4f}] -> "
              f"{'CLEARS' if lo4 > 0 else 'does not clear'}")
        res["overlap"][f"reject_{thr}"] = {
            "n_kept": n_keep, "test_auc": float(auc_k), "mean_delta": m4,
            "ci_lower": lo4, "ci_upper": hi4, "clears": bool(lo4 > 0)}

    # density-ratio weights, clipped so one extreme row cannot dominate
    ratio = np.clip((1 - p_syn_only) / np.clip(p_syn_only, 1e-6, None), 0, 10)
    print(f"\n  importance weighting: median weight {np.median(ratio):.4f}, "
          f"max {ratio.max():.2f}, effective n "
          f"{(ratio.sum()**2)/np.clip((ratio**2).sum(), 1e-9, None):.1f} of {n_syn}")
    sw = np.r_[np.ones(len(X_tr_real)), ratio]
    p_iw, auc_iw = fit_eval(X_plus, y_plus, X_te, y_te, w=sw)
    m5, lo5, hi5 = boot(y_te, p_base, p_iw)
    print(f"    test AUC {auc_iw:.4f}   delta {m5:+.4f} CI [{lo5:+.4f}, {hi5:+.4f}] -> "
          f"{'CLEARS' if lo5 > 0 else 'does not clear'}")
    res["overlap"]["importance_weighted"] = {
        "median_weight": float(np.median(ratio)),
        "effective_n": float((ratio.sum() ** 2) / max((ratio ** 2).sum(), 1e-9)),
        "test_auc": float(auc_iw), "mean_delta": m5, "ci_lower": lo5,
        "ci_upper": hi5, "clears": bool(lo5 > 0)}

    # ---------------- REAL-DATA LEARNING CURVE ----------------
    # The synthetic arms above are confounded by distribution shift, so they
    # cannot adjudicate data-starved vs feature-starved. This can. Subsample
    # the REAL training set and measure held-out AUC on the same frozen real
    # test set. No synthetic data, no distribution shift, no injector
    # assumptions -- just "does this model get better when given more of
    # exactly the data it already trains on?"
    #
    # A curve still climbing at 100% means more real data (Kepler) should help.
    # A curve flat from ~75% to 100% means the feature set has saturated and
    # additional examples in the same representation will not move it.
    #
    # Subsampling is done by HOST, not by row, so a star never appears at one
    # sample size and vanishes at another for reasons unrelated to volume.
    print("\n" + "=" * 78)
    print("REAL-DATA LEARNING CURVE -- the unconfounded version of the question")
    print("=" * 78)
    train_hosts = real.loc[tr, "host"].astype(str).to_numpy()
    uniq = np.unique(train_hosts)
    rng2 = np.random.RandomState(RANDOM_SEED)
    perm = rng2.permutation(len(uniq))
    res["real_learning_curve"] = []
    N_REPEATS_LC = 5
    for frac in (0.25, 0.5, 0.75, 1.0):
        aucs = []
        for rep in range(1 if frac == 1.0 else N_REPEATS_LC):
            r = np.random.RandomState(RANDOM_SEED + rep)
            pick = set(uniq[r.permutation(len(uniq))[:max(2, int(frac * len(uniq)))]])
            sel = np.array([h in pick for h in train_hosts])
            Xs, ys = X_tr_real[sel], y_tr_real[sel]
            if len(np.unique(ys)) < 2:
                continue
            _, a = fit_eval(Xs, ys, X_te, y_te)
            aucs.append(a)
        if not aucs:
            continue
        n_rows = int(sel.sum())
        print(f"  {100*frac:5.1f}% of train hosts (~{n_rows:5d} rows)  "
              f"test AUC {np.mean(aucs):.4f}"
              + (f" +/- {np.std(aucs):.4f} ({len(aucs)} reps)" if len(aucs) > 1 else ""))
        res["real_learning_curve"].append(
            {"fraction": frac, "n_rows": n_rows, "mean_test_auc": float(np.mean(aucs)),
             "std_test_auc": float(np.std(aucs)), "n_repeats": len(aucs)})

    lc = res["real_learning_curve"]
    if len(lc) >= 2:
        gain = lc[-1]["mean_test_auc"] - lc[-2]["mean_test_auc"]
        print(f"\n  gain from the last 25% of real training data: {gain:+.4f}")
        print("  (a curve still climbing here would argue MORE REAL DATA helps;")
        print("   a flat one argues the feature set, not the data volume, is the wall)")
        res["learning_curve_last_quartile_gain"] = float(gain)

    if quick:
        with open(RESULTS_PATH, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[--quick] stopping before nested CV/calibration. Saved to {RESULTS_PATH}")
        return

    # ---------------- NESTED CV ----------------
    print("\n" + "=" * 78)
    print("NESTED CV (outer 5-fold / inner 3-fold RandomizedSearchCV, 15 iters)")
    print("=" * 78)
    grid = {"clf__max_iter": [200, 300, 400], "clf__max_depth": [3, 4, 5, 6],
            "clf__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "clf__min_samples_leaf": [10, 20, 40]}
    res["nested_cv"] = {}
    for name, Xv, yv in (("real_only", X_tr_real, y_tr_real),
                         ("real_plus_synthetic", X_plus, y_plus)):
        inner = StratifiedKFold(3, shuffle=True, random_state=RANDOM_SEED)
        outer = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
        s = RandomizedSearchCV(pipe(), grid, n_iter=15, scoring="roc_auc",
                               cv=inner, random_state=RANDOM_SEED, n_jobs=-1)
        sc = cross_validate(s, Xv, yv, cv=outer, scoring="roc_auc")
        res["nested_cv"][name] = {"mean": float(sc["test_score"].mean()),
                                  "std": float(sc["test_score"].std())}
        print(f"  {name:22s} {sc['test_score'].mean():.4f} +/- {sc['test_score'].std():.4f}")
    print("  NOTE: real_plus_synthetic's nested CV folds contain synthetic rows,")
    print("  so it is NOT comparable to real_only as an estimate of real-world")
    print("  performance. The held-out real test set above is the honest number.")

    # ---------------- CALIBRATION ----------------
    print("\n" + "=" * 78)
    print("CALIBRATION (sigmoid-wrapped, as production deploys it)")
    print("=" * 78)
    res["calibrated"] = {}
    for name, Xv, yv in (("real_only", X_tr_real, y_tr_real),
                         ("real_plus_synthetic", X_plus, y_plus)):
        cal = CalibratedClassifierCV(pipe(), method="sigmoid", cv=3)
        cal.fit(Xv, yv)
        p = cal.predict_proba(X_te)[:, 1]
        res["calibrated"][name] = {"test_auc": float(roc_auc_score(y_te, p)),
                                   "brier": float(brier_score_loss(y_te, p)),
                                   "ece": ece(y_te, p)}
        print(f"  {name:22s} AUC {roc_auc_score(y_te, p):.4f}  "
              f"Brier {brier_score_loss(y_te, p):.4f}  ECE {ece(y_te, p):.4f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
