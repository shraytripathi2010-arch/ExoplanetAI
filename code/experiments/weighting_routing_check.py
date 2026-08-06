"""weighting_routing_check.py -- does sample_weight actually reach the trees?

THE TRAP, documented in this project's own FFI write-up: passing
`sample_weight` to a `CalibratedClassifierCV` wrapping a Pipeline made sklearn
warn that it "does not appear to accept sample_weight [so] sample weights will
only be used for the calibration itself". The down-weighted arm was therefore
SILENTLY IDENTICAL to the unweighted one, and reported a meaningless result
until it was caught.

This project has been bitten by that exact bug once. Before any weighting
experiment is designed, verify empirically -- for the sklearn version actually
installed -- which call form reaches the underlying HistGradientBoosting fit.

Method: fit with wildly asymmetric weights (one class crushed to ~0) and check
whether the PREDICTIONS move. If the trees saw the weights, predictions must
change dramatically. If only the calibrator saw them, the ranking is nearly
untouched (sigmoid calibration is monotonic, so AUC would be identical).
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

SEED = 42
print(f"sklearn {sklearn.__version__}\n")

# A HARD problem on purpose. A first draft used an easy one, hit AUC 1.0000 in
# every arm, and the AUC comparison could not discriminate at all.
X, y = make_classification(n_samples=1500, n_features=20, n_informative=5,
                           n_redundant=2, class_sep=0.6, flip_y=0.15,
                           weights=[0.2, 0.8], random_state=SEED)


def pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", HistGradientBoostingClassifier(
                         random_state=SEED, max_iter=100))])


# extreme weights: crush class 1 so the fit MUST change if it sees them
w = np.where(y == 1, 0.001, 1.0)

print("=" * 78)
print("A. BARE PIPELINE with clf__sample_weight  (the documented correct form)")
print("=" * 78)
a0 = pipe().fit(X, y)
a1 = pipe().fit(X, y, clf__sample_weight=w)
p0, p1 = a0.predict_proba(X)[:, 1], a1.predict_proba(X)[:, 1]
d_bare = float(np.abs(p0 - p1).mean())
print(f"   mean |unweighted - weighted| prediction change : {d_bare:.4f}")
print(f"   AUC {roc_auc_score(y, p0):.4f} -> {roc_auc_score(y, p1):.4f}")
print(f"   weights reached the trees? {'YES' if d_bare > 0.01 else 'NO'}")

print("\n" + "=" * 78)
print("B. CalibratedClassifierCV(pipeline).fit(..., sample_weight=w)")
print("=" * 78)
b0 = CalibratedClassifierCV(pipe(), cv=5, method="sigmoid").fit(X, y)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    b1 = CalibratedClassifierCV(pipe(), cv=5, method="sigmoid").fit(
        X, y, sample_weight=w)
    msgs = [str(m.message)[:150] for m in caught]
q0, q1 = b0.predict_proba(X)[:, 1], b1.predict_proba(X)[:, 1]
d_cal = float(np.abs(q0 - q1).mean())
print(f"   mean |unweighted - weighted| prediction change : {d_cal:.4f}")
print(f"   AUC {roc_auc_score(y, q0):.4f} -> {roc_auc_score(y, q1):.4f}")
for m in msgs:
    print(f"   WARNING: {m}")
print(f"   weights reached the trees? {'YES' if d_cal > 0.01 else 'NO / partial'}")

print("\n" + "=" * 78)
print("C. Does AUC alone reveal the difference?")
print("=" * 78)
print(f"   bare pipeline  AUC delta from weighting : "
      f"{roc_auc_score(y, p1) - roc_auc_score(y, p0):+.4f}")
print(f"   calibrated     AUC delta from weighting : "
      f"{roc_auc_score(y, q1) - roc_auc_score(y, q0):+.4f}")
print("\n   If B's AUC delta is ~0 while A's is large, the calibration wrapper")
print("   swallowed the weights and any 'weighted' arm built that way would be")
print("   a silent no-op -- exactly the bug this project already hit once.")

print("\n" + "=" * 78)
print("D. THE DECISIVE TEST -- did the RANKING change?")
print("=" * 78)
from scipy.stats import spearmanr
print("   Mean |prediction change| CONFLATES two different things: the trees")
print("   changing, and the sigmoid CALIBRATOR being refit on reweighted data.")
print("   Only the first alters ranking. So compare rank correlation:")
rho_bare = spearmanr(p0, p1).statistic
rho_cal = spearmanr(q0, q1).statistic
print(f"   bare pipeline   Spearman(unweighted, weighted) = {rho_bare:.6f}")
print(f"   calibrated      Spearman(unweighted, weighted) = {rho_cal:.6f}")
print("   rho == 1.000000 means the underlying trees are IDENTICAL and only")
print("   the probability mapping moved -- i.e. the weights never reached the fit.")

print("\n" + "=" * 78)
print("DESIGN CONSEQUENCE")
print("=" * 78)
trees_changed_bare = rho_bare < 0.9999
trees_changed_cal = rho_cal < 0.9999
print(f"   weights reached the TREES via bare pipeline : {trees_changed_bare}")
print(f"   weights reached the TREES via CalibratedCV  : {trees_changed_cal}")
if trees_changed_bare and not trees_changed_cal:
    print("   Weighting experiments MUST fit the BARE Pipeline with")
    print("   clf__sample_weight. AUC comparisons stay valid because sigmoid")
    print("   calibration is monotonic and cannot change ranking; Brier/ECE")
    print("   must then be reported from a separately calibrated refit or")
    print("   flagged as not directly comparable to production's.")
elif d_cal > 0.01:
    print("   This sklearn version DOES route sample_weight through the")
    print("   calibration wrapper to the inner estimator. Verify before relying.")
else:
    print("   Neither form moved predictions -- investigate before proceeding.")
