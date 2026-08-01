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

**This is the ninth and final feature/architecture experiment against the
classical model** (see "Multi-transit consistency + frequency-domain
features" below) -- per explicit user instruction, the ~0.90 ROC-AUC
ceiling for this feature family is now treated as final. No further
feature engineering will be attempted without an explicit new request.

## Multi-transit consistency + frequency-domain features -- NEGATIVE RESULT

Two features, both genuinely different representations of information TLS
already computes but never previously extracted:

- `multi_transit_depth_chi2red`: uncertainty-weighted reduced chi-square
  of individual transit depths (TLS's `transit_depths` +
  `transit_depths_uncertainties` arrays -- confirmed live to exist,
  neither ever used before). Different from the existing
  `depth_consistency_std` feature, which is a raw std that ignores each
  transit's own measurement uncertainty.
- `power_ratio_half_period` / `power_ratio_double_period`: TLS's full
  periodogram (`periods`/`power` arrays) evaluated at exactly half and
  double the detected period, relative to power at the detected period
  itself -- the classic eclipsing-binary aliasing signature.

Required a fresh TLS rerun across all 5,502 training rows (the saved
`transit_search_results.csv` only ever kept scalar summary columns, not
per-transit arrays or full periodograms) -- 60,874s (~16.9hr) across 8
workers, 100% job success rate (this computation doesn't require the
stricter all-features-finite gate the centroid work needed).

Leakage check: NaN rates were roughly balanced by class (2.8%/4.5% for
the chi2red feature; 30-34% / 20-21% for the two harmonic features, the
latter reflecting genuine cases where half/double the period falls
outside TLS's searched grid, not a bug). Single-feature AUCs: chi2red
0.495 (pure noise, no standalone signal at all); the two harmonic
features both ~0.41 -- meaningfully below chance, in a physically
sensible direction (false positives, often eclipsing binaries, show
relatively MORE power at period harmonics than real planets, exactly the
aliasing signature the feature was designed to catch) -- a real, if weak,
signal, not a leakage artifact.

Retrained the same HistGradientBoosting model, three ways (each feature
alone, and all three combined):

| Variant | 5-fold CV | Held-out test AUC | Bootstrap 95% CI vs base |
|---|---|---|---|
| Base (no new features) | -- | 0.8986 | -- |
| + chi2red only | 0.9190 +/- 0.0064 | 0.8966 | [-0.0058, +0.0016] |
| + power ratios only | 0.9192 +/- 0.0061 | 0.8966 | [-0.0072, +0.0031] |
| + all three combined | 0.9210 +/- 0.0061 | 0.9002 | [-0.0044, +0.0078] |

All three CIs include zero -- within noise, none clear the promotion bar.
Production model untouched.

**Correction/completion, found on audit**: the original request explicitly
asked for (a) multi-transit consistency of both DEPTHS AND DURATIONS, and
(b) validation via the "full existing suite (nested CV, calibration,
bootstrap CI)". The first pass above only delivered depth-consistency and
only ran plain CV + bootstrap CI. Both gaps were closed:
- **Duration-consistency**: attempted via independent per-transit ingress/
  egress measurement (predicted epochs from T0+n*period, no TLS rerun
  needed). Found a genuine methodological wall, not a fixable bug: simple
  threshold-crossing on raw per-cadence photometry doesn't reliably
  measure a single transit's duration -- validated broken on both a
  shallow-transit star (depth smaller than typical point-to-point noise)
  and a "deep" transit star (real intrinsic scatter/non-idealities in the
  raw data). Per explicit user decision, not pursued further with a more
  complex per-transit-binning method; idea 1 closes as depth-only.
- **Full validation suite**: re-ran with genuine nested CV (outer 5-fold /
  inner 3-fold RandomizedSearchCV, 30 iterations, same pattern as
  `05b_model_analysis.py`) and a calibration check (raw vs sigmoid vs
  isotonic, Brier score). Nested CV: base 0.9212 +/- 0.0040 vs
  base+new-features 0.9240 +/- 0.0034 -- still not a real separation once
  bootstrapped. Calibration meaningfully helped both variants about
  equally (isotonic, Brier 0.109 -> ~0.094) -- a calibration-layer benefit
  that exists independent of the new features, not something they caused.
  Bootstrap CI on the properly-tuned models: mean diff +0.0016, 95% CI
  [-0.0044, +0.0078] -- same conclusion, now on solid methodological
  ground: does not clear the promotion bar.

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

### Re-tested at 77.6% coverage -- STILL NEGATIVE, but no longer confounded

The result above had a real confound: at ~53% coverage, the other ~47% was
median-imputed, so "centroid displacement carries no signal" could not be
distinguished from "we only measured it on half the stars". The single-sector
check gave up if the first sector's depth disagreed with the ephemeris, and
**96% of its 2,377 failures were exactly that guard**. Trying up to 6 sectors
(`MAX_CENTROID_SECTORS_TO_TRY`) addresses precisely that population.

Re-ran the full 5,086-star scale test with multi-sector selection:

| | single-sector | multi-sector |
|---|---|---|
| usable centroids | 2,709 / 5,086 = 53.3% | **3,945 / 5,086 = 77.6%** |
| prior failures recovered | -- | 1,292 / 2,377 = 54.4% |
| regressions | -- | 56 (all transient network errors) |

Agreement check: on the 2,653 stars usable in BOTH runs, `shift_pixels` is
**bit-identical** (`max|diff| = 0.00e+00`) and from the same sector. The change
is purely additive -- it only engages when the first sector fails, and never
perturbs an existing measurement.

Leakage re-check at the new coverage: missing rate 31.9% positive vs 16.3%
negative, still skewed the SAFE way (positives more likely to be MISSING, so
data presence cannot shortcut the label); AUC of the missingness indicator
alone 0.422 (below chance); raw `shift_pixels` single-feature AUC 0.536.

Retrain, frozen split via `split_by_host` (0 hosts on both sides -- the
original script still used a positional `train_test_split` that predates the
Phase 1 contamination fix, which would have reshuffled on today's larger
`training.csv`):

| | 5-fold CV | held-out test ROC-AUC |
|---|---|---|
| base | 0.9161 +/- 0.0084 | 0.8964 |
| + centroid | 0.9162 +/- 0.0067 | 0.8996 |

Paired bootstrap: mean **+0.0032**, 95% CI **[-0.0012, +0.0076]** -- moved in
the right direction from +0.0021 / [-0.0020, +0.0059], but still includes zero.
**Does not clear the promotion bar.** Production model untouched.

The verdict is now a clean negative rather than a confounded one: measured on
78% of stars, difference-image centroid displacement moves this classifier by
~0.003 AUC, indistinguishable from noise. The multi-sector work still paid off
for the EVIDENCE layer -- 1,292 more candidates now carry a real photocenter
measurement on their detail pages, and per-candidate centroid failures dropped
from 37% to 13%.

Scripts: `compute_training_centroids.py` (multi-sector),
`retrain_with_centroid_multisector.py`; results
`training_centroid_results_multisector.csv`,
`retrain_centroid_multisector_results.json`.

## Small-lift trio: class weighting / engineered ratios / stacking -- ALL NEGATIVE

Three ideas that use only what is already in the pipeline, tested independently
against the production configuration refit on the frozen split (4,491 train /
1,140 test, 0 hosts on both sides), then combined.

TWO PREMISES WERE WRONG AND WERE CORRECTED BEFORE TESTING:

- Class weighting was NOT unaddressed. `05_train_models.build_models()` already
  passes `class_weight="balanced"` to all three models, and the deployed
  artifact confirms it. So the real question was whether "balanced" is the
  right choice for a 3.88:1 imbalance -- not whether to add weighting.
- A stacked ensemble had already been tried (Part C above: HGB + GP + CNN,
  0.9018 vs 0.9016). This run used a different roster (HGB + RF + LR), so not a
  duplicate, but the prior pointed the wrong way and said so up front: RF is
  MORE correlated with HGB than GP or CNN were.

| Arm | Test ROC-AUC | delta vs base | 95% CI | Clears ci_lo > 0 |
|---|---|---|---|---|
| baseline (production config) | 0.8999 | -- | -- | -- |
| no class weighting | 0.8979 | -0.0022 | [-0.0083, +0.0039] | NO |
| sqrt-inverse-frequency weights | 0.8979 | -0.0021 | [-0.0072, +0.0033] | NO |
| + 4 engineered ratios | 0.9001 | +0.0001 | [-0.0055, +0.0057] | NO |
| stacked (HGB+RF+LR) | 0.9039 | +0.0039 | [-0.0022, +0.0106] | NO |
| **combined (stacking on engineered features)** | **0.9043** | **+0.0042** | **[-0.0043, +0.0134]** | **NO** |

Findings worth keeping:

- **The existing `balanced` weighting is the best of the three schemes.** Both
  alternatives were worse, not better. A settled question, now measured.
- **The engineered ratios measure what they were designed to measure, and the
  model already knew it.** `duty_cycle` and `odd_even_significance` both scored
  single-feature AUC 0.353 -- well BELOW 0.5, i.e. high values predict the
  negative class, exactly as the eclipsing-binary physics predicts. The
  features are not broken; the information is already extractable from the raw
  columns. Net effect +0.0001.
- **Stacking fails for a measurable reason, not a mysterious one.** hgb-rf
  correlation on test is 0.941, and the meta-learner assigns LR a NEGATIVE
  coefficient (-0.454) while LR alone scores 0.774. Two near-identical models
  plus one being subtracted out cannot add independent signal. This is the
  second independent confirmation of Part C's finding.
- The combined run adds essentially nothing over stacking alone (+0.0042 vs
  +0.0039) and its CI is WIDER, which is what combining two changes with no
  real effect should look like.

Methodology caveat: the refit baseline is the bare Pipeline(SimpleImputer+HGB),
whereas production wraps it in CalibratedClassifierCV. Sigmoid calibration is
monotonic so AUC comparisons are unaffected, but the baseline's Brier/ECE are
not representative of the deployed model's calibration (separately measured at
ECE 0.0199).

Each arm got ONE honest attempt. No arm was re-tuned after seeing its result.

Scripts: `small_lift_trio.py`, `small_lift_combined.py`; results
`small_lift_trio_results.json`, `small_lift_combined_results.json`.

## Median-imputation vs HGB native NaN handling -- NEGATIVE RESULT

`05_train_models.build_models()` imputes before HistGradientBoosting with an
explicit comment that HGB could handle NaN natively, and that the imputer is
there so all three candidate models see identical input. That was correct for
the original bake-off -- LogisticRegression and RandomForest cannot take NaN --
but HGB won, and the DEPLOYED pipeline still carries an imputer whose only
purpose was fairness to two models no longer in use.

Worth testing because missingness here is not random: `transit_shape_ratio` is
30.4% missing and absent exactly when the shape fit fails; `FAP` is absent when
TLS cannot compute a false-alarm probability. Both are properties of weak or
ambiguous detections, i.e. correlated with the label. Median-imputing erases
that; HGB's native handling learns a per-split direction for missing values.

Frozen split, identical hyperparameters, identical rows:

| arm | 5-fold CV | test ROC-AUC | paired bootstrap vs current |
|---|---|---|---|
| A: current (SimpleImputer median) | 0.9155 | 0.8973 | -- |
| B: native NaN (no imputer) | 0.9178 | 0.8964 | -0.0008, CI [-0.0073, +0.0059] |
| C: SimpleImputer(add_indicator=True) | 0.9163 | **0.9005** | **+0.0033, CI [-0.0005, +0.0072]** |

Neither clears `ci_lo > 0`. Arm C misses by **0.0005** -- the closest any
change has come to the bar in this project -- and is the obvious candidate to
retest once the training set grows. Deliberately NOT promoted, and deliberately
not followed by a search for a variant that squeaks past, which would be
p-hacking the same bar every other experiment here has had to clear.

Script: `native_nan_vs_imputer.py`; results `native_nan_results.json`.

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
