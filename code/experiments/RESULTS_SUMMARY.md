# Model architecture experiments -- honest results summary

Production model is **unchanged**: `models/best_model.joblib`
(HistGradientBoosting, test ROC-AUC 0.9032) remains deployed. Nothing here
beat it by a real margin, so nothing was merged -- same standard as every
other experiment in this project. This now includes the full-TLS-search
classical-model augmentation path (see Part B below), which was originally
deprioritized as an estimate but was subsequently run in full and also
came back negative, and the pixel-level centroid-displacement feature
(see below), the first genuinely different KIND of information (spatial
position, not light-curve shape/TLS statistics) tested as an actual model
input rather than displayed-only evidence.

## Centroid displacement as a classifier feature -- NEGATIVE RESULT

Difference-image centroid displacement (`shift_pixels`) was already built
and validated as human-facing evidence on candidate pages, but never fed
into the classifier itself. Tested properly for the first time:

- **Real prerequisite gaps found and fixed before this could even run**:
  negative-class training stars were missing ra/dec 98.8% of the time
  (backfilled live from the TIC catalog, 1141/1141 recovered); positive-
  class stars had no TIC ID at all, only names (resolved 3982/4336 via
  coordinate cross-match against the TIC catalog, rejecting name-based
  resolution after it returned up to 58 ambiguous matches for one star).
- Ran the real, unmodified difference-image centroid check across all
  5,497 training rows (parallelized, 8 workers). **Real bug found live**:
  the first attempt hung for ~21 hours with no crash -- the TPF download
  had no network timeout (unlike other download paths in this project)
  and the run spanned the machine sleeping overnight. Fixed with the same
  network-timeout configuration `06_download_unknown.py` already uses,
  plus a per-job backstop timeout, and re-run under `caffeinate`.
- Result: 2,709/5,086 attempted (53.3%) produced a usable value -- the
  rest failed overwhelmingly (96% of failures) via the same sector-
  matching depth-consistency guard already validated earlier this
  session, correctly rejecting real ephemeris mismatches, not a bug.
- Leakage check: NaN-rate-by-class was moderately asymmetric (34.9%
  missing for negative class vs 50.2% for positive -- plausibly real,
  since TOI false positives are often deep EB blends easier for the
  guard to confirm) but not extreme, and critically skewed in the SAFE
  direction (positive class more likely to be MISSING data, not more
  likely to have it -- ruling out the dangerous "data presence predicts
  the label" shortcut). Single-feature AUC of raw shift_pixels: 0.532
  (barely above chance, no red flag); AUC of the missingness indicator
  itself: 0.423.
- Retrained the same HistGradientBoosting model (same hyperparameters,
  same train/test split, missing values median-imputed via the same
  SimpleImputer convention every other feature already uses):

  | | 5-fold CV (train) | Held-out test ROC-AUC |
  |---|---|---|
  | Base (no centroid) | 0.9142 +/- 0.0066 | 0.9030 |
  | + centroid feature | 0.9140 +/- 0.0062 | 0.9051 |

  Paired bootstrap (2,000 resamples) on the test-set difference: mean
  +0.0021, 95% CI **[-0.0020, +0.0059]** -- includes zero, does not clear
  this project's promotion bar. Production model untouched.

This is a genuinely different negative result from the others: unlike
synthetic light curves and the CNN (same underlying information as the
original model, both of which measurably hurt), centroid displacement is
categorically different information (spatial, not photometric) and here
it's merely *noise*, not harmful -- but still not enough signal on its
own, at this coverage rate (49%) and with this simple a feature
(raw displacement magnitude, not a noise-normalized significance version),
to move the model.

## Part B: injection-recovery synthetic data system -- BUILT, VALIDATED

- `injection.py`: batman-based transit injector + EB-like negative injector,
  parameters resampled from real training data distributions, injected into
  real processed negative-class light curves (real noise/systematics).
- `completeness_curve.py`: validated against 100 real injection+recovery
  trials (Hippke & Heller style). Recovery rate rose monotonically with
  depth (200ppm: 5% -> 8000ppm: 95%) and declined mildly with period
  (1d: 55% -> 12d: 40%) -- exactly the expected physical shape, confirming
  the injector is trustworthy. Real wall time: 1394s for 100 trials
  (individual TLS runs varied 36s-several minutes depending on which real
  light curve file was randomly selected; some real files are 5-6x larger
  than the file used for the original time estimate).
- `build_cnn_dataset.py`: generated 4,000 synthetic CNN training examples
  (2,000 positive, 2,000 negative) cheaply, by folding at the KNOWN injected
  period (no TLS search needed) -- confirms the cost asymmetry identified
  during feasibility scoping.
- Classical-model synthetic augmentation (full TLS search per synthetic
  example, the expensive path) -- **run in full, NEGATIVE RESULT.**
  `augment_classical_dataset.py` injected 1,600 synthetic signals (800
  transit/positive, 800 EB/negative) into real negative-class light curves
  and ran the SAME full TLS feature extraction real candidates go through
  (all v2 features required finite, exactly production's own bar). Real
  wall time: 14,928s (~4.15hr) across 8 workers; 953/1600 (59.6%) produced
  usable rows (442 positive, 511 negative) -- the rest failed the same
  finite-feature requirement real candidates sometimes fail too (most
  commonly `transit_shape_ratio`, needing enough resolved in-transit points).
  `retrain_with_augmentation.py` retrained the same HistGradientBoosting
  model (same hyperparameters/features) on real-only vs real+953-synthetic,
  evaluated both on the identical real held-out test set (synthetic never
  entered the test set):

  | | Test ROC-AUC |
  |---|---|
  | Real-only (reproduced) | 0.8985 |
  | Production (recorded) | 0.9032 |
  | Real + 953 synthetic | 0.8854 |

  Paired bootstrap (2,000 resamples): mean diff -0.0131, 95% CI
  [-0.0212, -0.0049] -- **entirely below zero**, i.e. the synthetic data
  measurably HURT test performance, not just "no improvement." Production
  model was left untouched, per this project's standing rule. (Minor note:
  the real-only reproduction, 0.8985, doesn't exactly match production's
  recorded 0.9032 -- a small, expected gap from environment/library-version
  nondeterminism in HistGradientBoosting, not a bug; the bootstrap CI
  compares the two models trained in the same run, so this doesn't affect
  the real-vs-synthetic conclusion.) This matches the CNN's own
  real+synthetic result above (also flat-to-slightly-worse) and this
  project's prior finding that more real data (TOI FA expansion) didn't
  move the classical model either -- the model was already extracting
  essentially all the signal the current feature set can support, and the
  synthetic examples' feature distributions are evidently different enough
  from real ones (despite passing the same completeness-curve validation
  for period *recovery*) to add noise rather than signal for classification.

## Part A: CNN on phase-folded light curves -- NEGATIVE RESULT

| | Test ROC-AUC |
|---|---|
| Classical model (production) | 0.9032 |
| CNN, real data only (5,378 examples) | 0.6964 |
| CNN, real + 4,000 synthetic | 0.6807 |

- Confirmed via learning curve: **no overfitting** (train/val AUC track
  closely throughout training, final 0.727/0.735) -- this is a genuine
  capability gap given the current data, not a fixable training-length or
  regularization issue.
- Synthetic augmentation did not help (if anything, very slightly worse,
  within run-to-run noise for a single seed).
- Confirms the upfront concern from feasibility scoping: ~5,491 real
  examples (1,155 of them negative) is not enough for this CNN architecture
  to outperform the well-engineered classical feature set.

## Part C: genuinely different model families -- NEGATIVE RESULT

| Model | Test ROC-AUC (same subset) |
|---|---|
| Classical HGB alone | 0.9016 |
| Gaussian Process alone | 0.8673 |
| CNN alone | 0.7044 |
| **Stacked ensemble (all 3)** | **0.9018** |

- GP vs classical: bootstrap CI on the difference = [0.017, 0.054], entirely
  above zero -- classical model wins by a real, non-noise margin.
- Stacked ensemble vs classical alone: 0.9018 vs 0.9016 -- a 0.0002
  difference, clear noise. The meta-learner's own fitted weights confirm
  this (coefficients: HGB 4.13, GP 2.18, CNN 2.33 -- HGB dominates).
- Honest caveat: the ensemble script's row-validity filter didn't
  exclude the same ~2% of rows with missing period/T0 that
  `build_cnn_dataset.py` correctly excluded, so a small number of
  degenerately-folded CNN views may be included here. Given the CNN's
  own contribution to the ensemble is negligible regardless, this
  doesn't change the conclusion, but is flagged rather than silently
  glossed over.

## Part D: uncertainty quantification -- BUILT AND DEMONSTRATED, NOT YET WIRED INTO PRODUCTION

- `uncertainty.py`: Monte Carlo propagation of period/depth/duration/
  stellar-radius/stellar-mass uncertainty into planet radius, semi-major
  axis, and equilibrium temperature ranges (16th/50th/84th percentile).
  Tested against a real candidate's parameters -- produced sensible ranges
  (e.g. planet radius 1.24-1.59 R_Earth around a 1.41 median).
- **Real gap found while building this**: `depth_mean_std` and
  `period_uncertainty` are already computed elsewhere in this pipeline
  (same TLS call, for both known and unknown stars) but aren't actually
  populated in the candidate characterization data that reaches
  `derive_physical_params` -- confirmed live (both `None` for a real
  candidate). Same "computed somewhere, dropped before it's used" pattern
  as the st_mass bug found earlier in this project. The module falls back
  to documented, explicit default fractional uncertainties (12% stellar
  radius, 10% stellar mass, 5% depth/duration) when a real value isn't
  available, and reports which fields used a fallback rather than silently
  presenting a generic assumption as a real measurement.
- Classifier probability uncertainty (bootstrap ensemble variance, 30
  refits): demonstrated and working -- correctly gives low variance for
  confident predictions (0.999 +/- 0.001) and high variance for borderline
  ones (0.850 +/- 0.106). Real cost: ~14s/refit, ~7 minutes for 30 refits --
  feasible as a periodic (per model-training-run) computation, not
  something to redo per candidate during a normal Update.
- **Not yet wired into `08_characterize_candidates.py` or the web app** --
  built and validated as standalone modules given the scope of everything
  else in this session; wiring in is straightforward from here (add
  e_rad/e_mass to the TIC catalog query in `06_download_unknown.py`, call
  `propagate_uncertainty()` inside `derive_physical_params`, add the ranges
  as new additive fields).

## Files

All in `code/experiments/`: `injection.py`, `completeness_curve.py`,
`phase_fold_views.py`, `build_cnn_dataset.py`, `train_cnn.py`,
`gp_classifier.py`, `stacked_ensemble.py`, `uncertainty.py`, plus saved
results (`*.json`, `completeness_curve_results.csv`, `cnn_dataset.npz`).
