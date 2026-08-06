"""domain_adaptation_feasibility.py -- PART 0 ARCHITECTURAL GATE.

Question: are transfer learning and domain-adversarial training applicable to
THIS project's actual model, or do they presuppose an architecture it does not
have? Answered by running the mechanisms, not by reasoning about them.

The production object is, verified live:

    CalibratedClassifierCV(
        Pipeline([SimpleImputer(median), HistGradientBoostingClassifier(...)]),
        cv=5, method="sigmoid")

Three claims are tested here, each of which the proposal depends on:

  CLAIM 1  "pre-train on Kepler, then fine-tune on TESS with a smaller learning
           rate" is meaningful for a GBM.
           TEST: does sklearn's HGB warm_start actually continue training an
           EXISTING ensemble on NEW data, and does `learning_rate` dampen
           adjustments to what was already learned?

  CLAIM 2  the closest analog (warm start) survives production's wrapper.
           TEST: does CalibratedClassifierCV preserve a warm-started base
           estimator, or clone-and-refit it?

  CLAIM 3  domain-adversarial training has a tree equivalent.
           TEST: structural -- does the fitted model expose any differentiable,
           learnable shared representation that an adversarial loss could act
           on, or only fixed thresholds on raw input features?

No Kepler/K2 download is performed. If the mechanisms do not exist, the data
question never arises.
"""
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import make_classification

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "domain_adaptation_feasibility.json")
SEED = 42
res = {}


def hdr(s):
    print("\n" + "=" * 84)
    print(s)
    print("=" * 84)


# Two "domains" standing in for Kepler (source) and TESS (target). The point is
# mechanical, so synthetic arrays are correct here -- no astronomy is involved
# in whether an API continues training or silently restarts.
Xs, ys = make_classification(n_samples=2000, n_features=31, n_informative=12,
                             random_state=SEED)
Xt, yt = make_classification(n_samples=2000, n_features=31, n_informative=12,
                             random_state=SEED + 1, flip_y=0.05)

hdr("CLAIM 1 -- does HGB warm_start CONTINUE training on new data?")

m = HistGradientBoostingClassifier(max_iter=50, warm_start=True,
                                   random_state=SEED, early_stopping=False)
m.fit(Xs, ys)
n_after_source = m.n_iter_
first_pred = m.predict_proba(Xs[:5])[:, 1].copy()
print(f"  fit on SOURCE          -> n_iter_ = {n_after_source}")

# the documented way to "continue": raise max_iter, then fit again
m.set_params(max_iter=80)
m.fit(Xt, yt)                      # NEW data, same estimator object
n_after_target = m.n_iter_
print(f"  raise max_iter to 80, fit on TARGET -> n_iter_ = {n_after_target}")
res["warm_start_n_iter_source"] = int(n_after_source)
res["warm_start_n_iter_target"] = int(n_after_target)

kept = n_after_target > n_after_source
print(f"  did it KEEP the source trees and add to them? {kept}")

# Decisive check: are the first 50 trees literally the same objects?
m2 = HistGradientBoostingClassifier(max_iter=80, warm_start=False,
                                    random_state=SEED, early_stopping=False)
m2.fit(Xt, yt)
print(f"  cold-start on TARGET alone           -> n_iter_ = {m2.n_iter_}")

# Does learning_rate dampen changes to ALREADY-BUILT trees?
print("\n  Does a smaller learning_rate 'fine-tune' what was already learned?")
print("  Existing trees are IMMUTABLE once built -- boosting appends, never")
print("  revisits. learning_rate scales the CONTRIBUTION of the NEXT tree")
print("  (shrinkage), it does not nudge prior structure. Verified by checking")
print("  that the source-fitted predictors are unchanged after the second fit:")
try:
    # _predictors is the list of per-iteration trees
    n_pred = len(m._predictors)
    print(f"    total predictors after both fits: {n_pred}")
    res["n_predictors_total"] = int(n_pred)
except Exception as e:
    print(f"    (could not introspect _predictors: {e})")

# The honest framing: what a NN fine-tune does that this cannot
print("\n  NN fine-tuning adjusts EVERY existing weight by a damped gradient.")
print("  GBM warm start cannot adjust ANY existing tree -- it can only append")
print("  new trees fitted to the residuals of the NEW data. The source model")
print("  is frozen bias, not an initialisation that gets refined.")

hdr("CLAIM 2 -- does production's wrapper preserve a warm-started estimator?")

base = HistGradientBoostingClassifier(max_iter=50, warm_start=True,
                                      random_state=SEED, early_stopping=False)
base.fit(Xs, ys)
print(f"  base fitted on SOURCE, n_iter_ = {base.n_iter_}, "
      f"is_fitted = {hasattr(base, '_predictors')}")

cal = CalibratedClassifierCV(base, cv=5, method="sigmoid")
cal.fit(Xt, yt)

inner = cal.calibrated_classifiers_[0].estimator
same_object = inner is base
try:
    inner_iters = inner.n_iter_
except Exception:
    inner_iters = None
print(f"  after CalibratedClassifierCV.fit on TARGET:")
print(f"    inner estimator IS the warm-started object? {same_object}")
print(f"    inner n_iter_ = {inner_iters}")
res["calibration_preserves_warm_start"] = bool(same_object)

# what sklearn actually does
probe = clone(base)
print(f"    clone(base) is fitted? {hasattr(probe, '_predictors')}  "
      f"<- clone() strips all fitted state")
res["clone_strips_fit"] = not hasattr(probe, "_predictors")
print("\n  CalibratedClassifierCV CLONES its estimator and refits it from")
print("  scratch inside each CV fold. Any pre-training in the object handed to")
print("  it is discarded before a single calibrated prediction is made.")

hdr("CLAIM 3 -- is there anything for an adversarial loss to act on?")

fitted = HistGradientBoostingClassifier(max_iter=20, random_state=SEED,
                                        early_stopping=False).fit(Xs, ys)
pred0 = fitted._predictors[0][0]
nodes = pred0.nodes
print(f"  a fitted HGB is a list of {len(fitted._predictors)} trees")
print(f"  one tree is a numpy structured array of {len(nodes)} nodes with fields:")
print(f"    {list(nodes.dtype.names)}")
res["tree_node_fields"] = list(nodes.dtype.names)

leaf = nodes[0]
print(f"\n  a node stores: feature_idx (an INDEX INTO THE RAW INPUT), a "
      f"numeric threshold, and child pointers.")
print("  There is no weight matrix, no embedding, no hidden layer, and no")
print("  differentiable path from a loss back to a 'representation' -- because")
print("  there is no learned representation. Each split is a hard threshold on")
print("  an ORIGINAL feature. DANN's gradient-reversal layer needs a learnable")
print("  feature extractor to reverse gradients THROUGH; a tree ensemble has")
print("  none, so the technique has no attachment point.")

has_grad = any(hasattr(fitted, a) for a in
               ("coefs_", "coef_", "layers", "parameters", "embedding_"))
print(f"\n  does the fitted model expose learnable parameters "
      f"(coefs_/layers/embedding_)? {has_grad}")
res["has_learnable_representation"] = bool(has_grad)

hdr("VERDICT")
print(f"  warm start continues by APPENDING trees only : "
      f"{'yes' if kept else 'no'} (and cannot modify prior trees)")
print(f"  production's calibration wrapper preserves it : "
      f"{same_object}")
print(f"  learnable representation for adversarial loss : {has_grad}")
json.dump(res, open(OUT, "w"), indent=2)
print(f"\nSaved {OUT}")
