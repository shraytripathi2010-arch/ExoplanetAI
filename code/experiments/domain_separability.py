"""domain_separability.py -- "is this new data the same KIND of data?", reusable.

WHY THIS EXISTS AS A MODULE

The same diagnostic has now decided three separate questions in this project,
and each time it was written inline against one specific axis:

  * synthetic-vs-real injected transits  -> discriminator AUC 0.9654, mixing HURT
  * 2-min vs coarse-cadence (FFI) rows   -> discriminator AUC 0.9717, coverage
                                            lever only, not an accuracy lever
  * (next) TESS vs another survey

Written inline a fourth time it would be re-derived, and small differences in
CV setup would quietly make the numbers non-comparable to the two that already
decided something. The reference values above are only meaningful as a
yardstick if the measurement is identical, so the measurement now lives in one
place.

WHAT IT MEASURES, AND HOW TO READ IT

Fit a classifier to predict SOURCE (not label) from the same feature matrix the
real model uses, scored out-of-fold. If it cannot tell the sources apart
(AUC ~0.5) the new data is drawn from the same distribution and can be pooled.
If it separates them easily, pooling means the model can learn "which survey is
this" as a shortcut, and any apparent gain may be that shortcut rather than
skill.

The AUC alone is not the whole answer, so two more outputs come with it:

  * standardized mean differences, to name WHICH features carry the shift
  * per-feature label-AUC against domain-AUC, to see whether the features that
    separate the domains are the same ones that carry the label. A feature that
    predicts the domain but not the label is a nuisance variable; a feature
    that predicts BOTH is the dangerous case, because the shortcut and the
    signal are entangled and cannot be separated by dropping the feature.

NO THRESHOLD IS ENFORCED HERE. This reports; the caller decides. The two
reference numbers above are supplied in the output so a new result can be read
against decisions that were actually made rather than against an abstract bar.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

RANDOM_SEED = 42
MIN_MINORITY = 30

# Kept verbatim from cadence_audit.py so a new measurement is comparable to the
# two that already closed a question. Changing these invalidates that
# comparison -- if they ever change, the reference values below must be
# re-measured too, not just relabelled.
DOMAIN_CLF = dict(max_iter=300, max_depth=4, learning_rate=0.05,
                  class_weight="balanced", random_state=RANDOM_SEED)

REFERENCE = {
    "synthetic_vs_real": {"auc": 0.9654,
                          "outcome": "mixing hurt; augmentation abandoned"},
    "2min_vs_coarse_cadence": {"auc": 0.9717,
                               "outcome": "coverage lever only, not an accuracy lever"},
}


def domain_report(X, domain, y=None, names=("A", "B"), verbose=True):
    """Measure how separable two data sources are under the model's features.

    X       : DataFrame, the SAME feature matrix the real model consumes
              (build it with m05.build_feature_matrix, not a bespoke subset --
              the question is whether the production model can see the seam).
    domain  : 0/1 array-like. 1 marks the NEW source.
    y       : optional true labels, enabling the redundancy check.
    names   : display names for domain 0 and 1.

    Returns a dict; never raises on a degenerate split, returns
    `domain_auc=None` with a `note` instead, because "too few rows to tell" is
    a real answer and should not look like a failure.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    dom = np.asarray(domain).astype(int)
    n1, n0 = int(dom.sum()), int((dom == 0).sum())
    out = {"n_domain_0": n0, "n_domain_1": n1, "names": list(names),
           "reference": REFERENCE}

    if verbose:
        print(f"  {names[0]}: {n0} rows | {names[1]}: {n1} rows")

    if min(n0, n1) < MIN_MINORITY:
        out["domain_auc"] = None
        out["note"] = (f"minority source has {min(n0, n1)} rows "
                       f"(< {MIN_MINORITY}); not enough to measure separability")
        if verbose:
            print(f"  {out['note']}")
        return out

    pipe = Pipeline([("i", SimpleImputer(strategy="median")),
                     ("c", HistGradientBoostingClassifier(**DOMAIN_CLF))])
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    p = cross_val_predict(pipe, X, dom, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(dom, p))
    out["domain_auc"] = auc
    if verbose:
        print(f"  domain classifier ({names[0]} vs {names[1]}) AUC: {auc:.4f}")
        for k, v in REFERENCE.items():
            print(f"    reference: {k} scored {v['auc']:.4f} -> {v['outcome']}")

    smd = {}
    for c in X.columns:
        a = pd.to_numeric(X.loc[dom == 0, c], errors="coerce").astype(float)
        b = pd.to_numeric(X.loc[dom == 1, c], errors="coerce").astype(float)
        sd = np.sqrt((a.var() + b.var()) / 2)
        if np.isfinite(sd) and sd > 0:
            smd[c] = float((b.mean() - a.mean()) / sd)
    out["standardized_mean_diff"] = smd
    if verbose and smd:
        print(f"\n  largest standardized mean differences ({names[1]} - {names[0]}):")
        for k, v in sorted(smd.items(), key=lambda kv: -abs(kv[1]))[:8]:
            print(f"    {k:24s} {v:+.2f}")

    if y is not None:
        y = np.asarray(y)
        pairs = []
        for c in X.columns:
            v = pd.to_numeric(X[c], errors="coerce")
            m = v.notna().to_numpy()
            if m.sum() < 50 or len(np.unique(y[m])) < 2 or len(np.unique(dom[m])) < 2:
                continue
            try:
                pairs.append((c, float(roc_auc_score(y[m], v[m])),
                              float(roc_auc_score(dom[m], v[m]))))
            except ValueError:
                continue
        pairs.sort(key=lambda r: -abs(r[2] - 0.5))
        out["per_feature"] = [{"feature": c, "label_auc": la, "domain_auc": da}
                              for c, la, da in pairs]
        if verbose and pairs:
            print(f"\n  {'feature':<24}{'label-AUC':>11}{'domain-AUC':>12}")
            for c, la, da in pairs[:8]:
                print(f"  {c:<24}{la:>11.3f}{da:>12.3f}")
    return out


def _selfcheck():
    """Reproduce BOTH recorded cadence domain AUCs through this function.

    A refactor that quietly changes the measurement is worse than no refactor,
    because every past number silently stops being comparable. Run this after
    touching anything above.

    There are two recorded numbers and they are NOT the same measurement -- a
    first pass at this check asserted the wrong one against the wrong grouping:

      cadence_audit_results.json    domain_auc                 0.9466  2-min vs non-2-min
      ffi_mixing_results.json       domain_auc_2min_vs_coarse  0.9717  2-min vs COARSE only

    The difference is the 401 FINE (20-second) rows. `non-2-min` lumps them in
    with the 231 COARSE rows even though they are FINER than 2-min, not
    coarser; excluding them isolates the actual coarse-cadence population and
    raises separability. 0.9717 is the number the FFI decision rests on, so
    both are checked here to keep the distinction visible.
    """
    import os
    import sys
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    code = os.path.join(here, "..")
    sys.path.insert(0, code)
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(code, "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m05
    spec.loader.exec_module(m05)

    df = pd.read_csv(os.path.join(code, "..", "data", "training_dataset",
                                  "training.csv"))
    cad = pd.read_csv(os.path.join(here, "cadence_per_star.csv"))
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce")
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy()
    coarse = (c > 2.6).to_numpy()
    ok = c.notna().to_numpy()

    checks = [
        ("2-min vs non-2-min (cadence_audit)", ok, (~is2)[ok], 0.9466338101474532,
         ("2-min", "non-2-min")),
        ("2-min vs COARSE only (ffi_mixing)", is2 | coarse, coarse[is2 | coarse],
         0.9716678622284558, ("2-min", "COARSE")),
    ]
    ok_all = True
    for label, mask, dom, expected, names in checks:
        print("=" * 78)
        print(f"SELF-CHECK: {label} -- expected {expected:.4f}")
        print("=" * 78)
        r = domain_report(X[mask], dom, y=y[mask], names=names)
        got = r["domain_auc"]
        passed = abs(got - expected) < 0.005
        ok_all &= passed
        print(f"\n  got {got:.10f}  expected {expected:.10f}  "
              f"delta {abs(got - expected):.2e}  "
              f"{'PASS' if passed else 'MISMATCH -- investigate'}\n")
    print("ALL CHECKS PASS" if ok_all else "AT LEAST ONE CHECK FAILED")


if __name__ == "__main__":
    _selfcheck()
