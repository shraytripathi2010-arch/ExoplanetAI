"""Exact ROC-AUC that is fast enough to bootstrap with.

WHY THIS EXISTS

`sklearn.metrics.roc_auc_score` costs ~25 ms per call at n=1098 on this
machine. Almost none of that is arithmetic -- it is input validation, which
runs in full on every call. A paired bootstrap does 2 calls x 2000 iterations
per comparison, so validation alone dominates: the 28-arm calibration sweep
projected at 47 minutes of pure bootstrap, and the earlier stage-2 validation
runs took 35-75 minutes largely for this reason. It is also why those runs were
sized at 8 resamples rather than more.

WHAT IT COMPUTES

The same number, via the rank-sum (Mann-Whitney U) identity:

    AUC = (sum of ranks of positives - n1(n1+1)/2) / (n1 * n0)

with **averaged ranks for ties**. Tie handling is not optional here: isotonic
calibration emits long plateaus of identical probabilities, so ordinal ranks
would give wrong answers for exactly the arms most likely to be compared.

Verified to 1e-12 against sklearn over 400 random tie-heavy cases and against
real stored model probabilities. Run this file directly to re-verify:

    python3 code/fast_auc.py

USING IT

`fast_auc(y, p)` is the direct entry point for hot loops.

`roc_auc_score(y, p)` is a **drop-in replacement** for sklearn's: it fast-paths
the binary, 1-D, no-keyword, finite case and **delegates to sklearn for
everything else** (multiclass, `sample_weight`, `max_fpr`, `average`, NaNs,
probability matrices). So swapping the import cannot change behaviour in a case
the fast path does not handle -- it either computes the identical value or
calls the original.
"""
import numpy as np
from sklearn.metrics import roc_auc_score as _sk_roc_auc_score

__all__ = ["fast_auc", "roc_auc_score"]


def fast_auc(y_true, y_score):
    """Exact binary ROC-AUC. Expects 1-D y in {0,1} and finite scores.

    No validation -- call `roc_auc_score` below if the inputs are not already
    known to be clean.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_score, dtype=np.float64)
    n = p.shape[0]

    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    newgrp = np.empty(n, dtype=bool)
    newgrp[0] = True
    np.not_equal(sp[1:], sp[:-1], out=newgrp[1:])
    gid = np.cumsum(newgrp) - 1
    # average rank within each tie group
    avg = (np.bincount(gid, weights=np.arange(1, n + 1, dtype=np.float64))
           / np.bincount(gid))
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg[gid]

    pos = y == 1
    n1 = int(pos.sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("Only one class present in y_true; ROC-AUC undefined.")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def roc_auc_score(y_true, y_score, **kwargs):
    """Drop-in for `sklearn.metrics.roc_auc_score`, exact and faster.

    Falls back to sklearn whenever the fast path does not strictly apply, so
    this is safe to substitute at an import site without auditing call sites.
    """
    if kwargs:
        return _sk_roc_auc_score(y_true, y_score, **kwargs)

    y = np.asarray(y_true)
    p = np.asarray(y_score)
    if y.ndim != 1 or p.ndim != 1 or y.shape[0] != p.shape[0] or y.shape[0] < 2:
        return _sk_roc_auc_score(y_true, y_score)

    # binary 0/1 (or boolean) only -- no sort, unlike np.unique
    if y.dtype == bool:
        y = y.astype(np.int8)
    else:
        if not np.issubdtype(y.dtype, np.number):
            return _sk_roc_auc_score(y_true, y_score)
        if not np.all((y == 0) | (y == 1)):
            return _sk_roc_auc_score(y_true, y_score)
    n1 = int((y == 1).sum())
    if n1 == 0 or n1 == y.shape[0]:
        return _sk_roc_auc_score(y_true, y_score)   # let sklearn raise its error

    p = np.asarray(p, dtype=np.float64)
    if not np.isfinite(p).all():
        return _sk_roc_auc_score(y_true, y_score)

    return fast_auc(y, p)


def _self_check(n_cases=400, seed=0):
    """Verifies the fast path against sklearn, including tie-heavy inputs."""
    rng = np.random.RandomState(seed)
    worst = 0.0
    checked = 0
    for _ in range(n_cases):
        n = rng.randint(10, 1500)
        y = (rng.rand(n) < rng.uniform(0.05, 0.95)).astype(int)
        if y.sum() == 0 or y.sum() == n:
            continue
        # round to 1-2 dp on many cases to force heavy ties, as isotonic does
        p = np.round(rng.rand(n), rng.choice([1, 2, 3, 16]))
        worst = max(worst, abs(fast_auc(y, p) - _sk_roc_auc_score(y, p)))
        checked += 1
    return checked, worst


if __name__ == "__main__":
    import time

    checked, worst = _self_check()
    print(f"exactness: {checked} tie-heavy cases, max |delta| = {worst:.2e}")
    assert worst < 1e-12, "fast_auc disagrees with sklearn"

    # fallback paths must not diverge either
    rng = np.random.RandomState(1)
    y = (rng.rand(500) < 0.5).astype(int)
    p = rng.rand(500)
    w = rng.rand(500)
    # Delegated cases must be bit-identical -- they ARE sklearn's return value.
    assert (roc_auc_score(y, p, sample_weight=w)
            == _sk_roc_auc_score(y, p, sample_weight=w)), "sample_weight fallback"
    ylab = np.where(y == 1, "yes", "no")
    assert roc_auc_score(ylab, p) == _sk_roc_auc_score(ylab, p), "string-label fallback"
    # Boolean labels take the FAST path, so they agree to floating-point noise
    # rather than bitwise -- a different summation order, not a different answer.
    assert abs(roc_auc_score(y.astype(bool), p)
               - _sk_roc_auc_score(y.astype(bool), p)) < 1e-12, "boolean labels"
    print("fallbacks: sample_weight + string labels bitwise identical; "
          "boolean labels fast-pathed and equal to <1e-12")

    n = 1098
    y = (rng.rand(n) < 0.5).astype(int)
    p = rng.rand(n)
    for name, fn in (("sklearn", _sk_roc_auc_score), ("fast_auc", fast_auc)):
        t = time.time()
        for _ in range(300):
            i = rng.randint(0, n, n)
            fn(y[i], p[i])
        print(f"  {name:<9} {(time.time() - t) / 300 * 1000:7.3f} ms/call")
