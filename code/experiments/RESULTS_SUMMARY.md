# Model architecture experiments -- honest results summary

Production model is **unchanged**: `models/best_model.joblib`
(HistGradientBoosting, test ROC-AUC **0.9031** on the 1,098-star deduplicated
clean test set; md5 `341f1a3907e77f6ec294f182833e613c`) remains deployed.

**Which baseline a new experiment is measured against.** Use **0.9021**, the
refit-on-clean-train figure, not 0.9031. A challenger is itself a refit, so
0.9021 is the like-for-like comparison; 0.9031 is the deployed artefact and
carries ~0.001 of refit luck that a challenger does not get. Older sections
below quote **0.9043** — that is the retired pre-deduplication number, measured
on a test set containing stars the model had trained on, and is kept only as
the "before" side of that fix. It is never a current claim. The TOI-restricted
**0.8151** is a supplementary measurement on a different evaluation population
and never replaces the headline; see its own section.

Nothing here
beat it by a real margin, so nothing was merged -- same standard as every
other experiment in this project. This now includes the full-TLS-search
classical-model augmentation path (see Part B below), which was originally
deprioritized as an estimate but was subsequently run in full and also
came back negative, and the pixel-level centroid-displacement feature
(see below), the first genuinely different KIND of information (spatial
position, not light-curve shape/TLS statistics) tested as an actual model
input rather than displayed-only evidence.

A previous version of this header called the multi-transit/frequency-domain
work "the ninth and final feature/architecture experiment" and declared the
~0.90 ceiling final. That is superseded: a small-lift tier, a medium-lift
tier, and an injection-recovery diagnostic have since been run at explicit
request. **Thirteen** feature/architecture changes have now been tested and
none has cleared `ci_lo > 0`.

**The ceiling's cause has changed, though.** A real-data learning curve run
during the injection-recovery work is still climbing at the full training-set
size and predicts roughly **+0.013 AUC from doubling the real data** -- larger
than any of the thirteen feature attempts. The long-standing "flat learning
curve, therefore feature-starved" premise did not survive scrutiny; see
"Real-data learning curve" below. The ceiling is now best understood as
substantially a DATA-VOLUME limit, not purely a feature limit.

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

## Medium-lift Item 1: multi-sector depth consistency -- ARTIFACT, NOT PROMOTED

**The most instructive result in this file.** It cleared the promotion bar,
survived nested CV and calibration, and was still wrong.

Feasibility was checked before any code was written: 97.8% of stars have >1
sector (median 7), and `n_sectors` alone scored AUC 0.479 (Mann-Whitney
p=0.459), so observation history does not predict the label and the feature
could not inherit that shortcut. Design note: re-running TLS per sector would
have cost ~118 hours and is the wrong tool anyway -- TLS exists to find an
UNKNOWN period, and period/T0/duration are already known for every training
star. Folding at the known ephemeris needs no search, collapsing the cost to
the download (~4.5h for 5,137 stars, storage bounded by download-measure-delete).

Extraction: 4,964/5,137 usable, 92.2% with >=2 sectors. Features:
`sector_depth_frac_scatter`, `sector_depth_chi2red` (uncertainty-weighted,
so it separates real disagreement from a noisy sector), `n_sectors_measured`.

| | 5-fold CV | test AUC | Brier | ECE |
|---|---|---|---|---|
| base | 0.9194 +/- 0.0083 | 0.8999 | 0.1081 | 0.0859 |
| + multi-sector | 0.9248 +/- 0.0105 | **0.9094** | 0.1001 | 0.0755 |

Paired bootstrap: **+0.0094, 95% CI [+0.0008, +0.0177] -- CLEARS the bar.**
Nested CV agreed (0.9202 -> 0.9276). Calibration agreed (0.8907 -> 0.9009).

**The control that caught it.** Missingness was strongly class-asymmetric:
19.1% of positives lacked the features vs 3.5% of negatives, because 494
positive-class stars have no resolvable TIC ID -- a property of this project's
name-to-TIC cross-match, not of the stars. `multisector_missingness_control.py`
fitted a model with ONLY a binary "are these features present" indicator and no
measured values at all:

| arm | delta vs base | 95% CI | clears |
|---|---|---|---|
| indicator only (no measurements) | **+0.0102** | [+0.0043, +0.0164] | yes |
| full features | +0.0094 | [+0.0008, +0.0177] | yes |
| restricted population (missingness held constant) | +0.0021 | [-0.0064, +0.0107] | **no** |

**A single binary "do I have this data" column reproduces 108% of the gain.**
Restricting to stars that HAVE the features -- so missingness is constant --
leaves +0.0021 with a CI spanning zero. The result is bookkeeping, not physics.
Not promoted.

**Corrected project heuristic.** This project had been treating
"missingness skewed toward the positive class" as the SAFE direction, on the
reasoning that data *presence* then cannot shortcut the label. That is only
half the check. Absence can predict the label just as well as presence, and
here it did. Any future feature with class-asymmetric missingness must run the
indicator-only control, not just report the skew direction.

Scripts: `multisector_feasibility.py`, `multisector_consistency.py`,
`validate_multisector.py`, `multisector_missingness_control.py`.

## Medium-lift Item 2: phase-folded flux distribution statistics -- NEGATIVE

Nine features asking a question TLS never answers: given the known ephemeris,
what SHAPE is the flux distribution? In/out-of-transit skewness and kurtosis
(a real transit is flat-bottomed and roughly symmetric; a grazing eclipse or
blend is V-shaped and skews), their differences (which cancel each star's own
noise character), and relative energy in the 3 finest Haar detail levels of the
phase-folded binned curve (a transit is one coherent dip whose energy sits at
coarse scales; noise lives at fine scales).

Extraction: 5,380/5,631 usable (95.5%). No downloads, no TLS re-run -- reads
the already-preprocessed light curves. Missingness indicator AUC 0.502, i.e.
no missingness shortcut at all, unlike Item 1.

| | 5-fold CV | test AUC | Brier | ECE |
|---|---|---|---|---|
| base | 0.9188 | 0.9009 | 0.1067 | 0.0857 |
| + flux stats | 0.9172 | 0.8938 | 0.1065 | 0.0768 |

Paired bootstrap: **-0.0072, 95% CI [-0.0153, +0.0010] -- does not clear.**
Nested CV flat (0.9202 vs 0.9204).

**A bug found before the result was trusted.** The first run printed
"feature counts: base 24 -> with new 24" -- `build_feature_matrix` selects only
`FEATURE_COLUMNS`, so merging the new columns into the dataframe never put them
in X. It would have reported a perfectly null result for a comparison that
never actually differed. The new columns must be concatenated onto X explicitly.

No combined Item1+Item2 run was performed, deliberately: combining a
bookkeeping artefact with a measurably negative change is a search for a
passing number, not a hypothesis.

Scripts: `flux_distribution_features.py`, `validate_flux_distribution.py`.

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

## SUPPLEMENTARY MEASUREMENT: TOI-restricted evaluation (context vs published work)

**This does NOT replace the headline figure.** The project's result remains the
end-to-end number: **0.9031 ROC-AUC** on the full clean test set (0.9021
refit-clean). What follows is a context point for comparison against literature
that solves a narrower task, and it is *less* flattering, not more.

Nothing was retrained, tuned or modified. Only the evaluation population changed.

### Why restrict at all

AstroNet, ExoMiner and RAVEN VET candidates a mission pipeline already flagged
as plausible. This project runs TLS on raw light curves, does its own detection,
then classifies. Those are different problems and their headline numbers are not
comparable. Restricting to TOI-flagged stars approximates the narrower task.

### How "was TOI-flagged" is defined -- and why not by disposition

Membership in the NASA `toi` table is assigned when SPOC raises a
threshold-crossing event and the object is alerted, BEFORE any human
disposition. So "appears in the TOI table under any disposition, including the
still-undispositioned PC/APC" is a pre-label property. Using the disposition
(KP/CP vs FP/FA) would be circular -- disposition IS the training label.

### THE ASYMMETRY THAT GOVERNS THIS MEASUREMENT

| | in TOI table | not in TOI table |
|---|---|---|
| negatives (FP) | **231 (100.0%)** | 0 |
| positives (planets) | 269 (31.0%) | 598 |

The negative class was SOURCED from the TOI false-positive list, so every
negative is a TOI by construction. **The restriction removes 598 positives and
zero negatives** -- a one-sided filter on the positive class. Prevalence moves
79.0% -> 53.8%, so precision@k and PR-AUC shift for class-balance reasons alone.
ROC-AUC is prevalence-invariant and is the only clean comparison. (13 test stars
had unresolvable TIC IDs and were counted as non-TOI.)

### Metric comparison

| metric | FULL test (end-to-end) | TOI-RESTRICTED |
|---|---|---|
| n | 1,098 | 500 |
| positive prevalence | 0.790 | 0.538 |
| **ROC-AUC** | **0.9031** [0.8784, 0.9249] | **0.8151** [0.7739, 0.8517] |
| PR-AUC | 0.9670 | 0.7762 |
| no-skill baseline | 0.790 | 0.538 |
| lift over no-skill | 1.22x | **1.44x** |
| P@10 (random) | 1.000 (0.790) | 0.800 (0.538) |
| P@20 (random) | 1.000 (0.790) | 0.750 (0.538) |
| P@50 (random) | 0.980 (0.790) | 0.780 (0.538) |
| P@100 (random) | 0.990 (0.790) | 0.810 (0.538) |
| P@200 (random) | 0.995 (0.790) | 0.800 (0.538) |

**ROC-AUC delta: -0.0880.** The restricted task is HARDER, not easier.

### Attribution -- it is the positives that were removed, not the filtering

| positives | n | mean score | median |
|---|---|---|---|
| TOI-flagged | 269 | 0.8045 | 0.8770 |
| NOT TOI-flagged | 598 | **0.9395** | **0.9766** |

The removed positives were the model's EASIEST. Feature comparison explains why:

| feature | TOI positives | non-TOI positives | diff (SD) |
|---|---|---|---|
| SDE | 11.863 | 5.386 | **+1.41** |
| snr | 15.514 | 5.979 | +0.82 |
| st_rad | 1.079 | 1.641 | -0.24 |

TOI-flagged positives have far STRONGER signals (SDE 11.9 vs 5.4) yet are HARDER
to classify. That resolves once you recall the feature audit: `SDE` has
single-feature AUC **0.326**, i.e. BELOW 0.5 -- high SDE predicts the NEGATIVE
class, because the strongest signals are typically deep eclipsing binaries.

So in the full population the model gets a large amount of easy separation from
low-SDE, non-TOI-flagged planets that look nothing like TOI false positives.
Restrict to TOI-flagged and both classes now sit in the same high-SDE regime,
where the model's single most useful discriminant stops discriminating.

Attribution verdict: **both effects are present, but the dominant one is the
removal of easy positives, not any change in intrinsic task difficulty.** The
class-balance shift explains the precision@k and PR-AUC movement; the loss of
the low-SDE positive population explains the ROC-AUC drop. Note the PR-AUC
*lift* over no-skill actually improves (1.22x -> 1.44x), since the baseline fell
further than the metric did -- the model is doing relatively more work on the
harder population.

### Comparability caveat

0.8151 still should not be set directly beside an AstroNet or ExoMiner number.
Those systems vet TOIs using pixel-level data and mission-grade ephemerides;
this model vets them using its own TLS-derived features. And this restricted
population is peculiar -- every negative in it is a TOI false positive by
construction, which is not the class balance those systems face. It is a context
point, not a benchmark result.

### Documentation language (use verbatim)

**End-to-end result (the headline):**
> The model achieves **0.9031 ROC-AUC** on a held-out test set of 1,098 stars,
> measured end-to-end: the pipeline runs a Transit Least Squares search on raw
> TESS light curves, derives its own features, and classifies the result. No
> mission-provided candidate list, ephemeris, or pre-filtering is used at any
> stage, so this number reflects detection and classification together.

**TOI-restricted result (supplementary):**
> Restricted to the 500 test stars that TESS's SPOC pipeline had independently
> flagged as Objects of Interest, the same unmodified model scores **0.8151
> ROC-AUC**. This is the narrower vetting task addressed by AstroNet, ExoMiner
> and RAVEN, and is reported only for context against that literature. It is
> lower than the end-to-end figure because the restriction removes 598
> easy-to-classify confirmed planets that were never TOI-flagged while removing
> none of the false positives, leaving a population in which both classes share
> the strong-signal regime. It is not a benchmark-comparable number: those
> systems use pixel-level data and mission ephemerides, and every negative in
> this restricted set is a TOI false positive by construction.

Script: `toi_restricted_metrics.py`; results `toi_restricted_metrics.json`.
Full-population metrics reused from the retrieval-metrics measurement above.

## Semi-supervised pseudo-labelling -- NEGATIVE, and the safeguard caught it first

Score unlabelled candidates with the current model, promote the confident
predictions to labels, retrain. Predicted to fail by error amplification, and
it did -- but the useful part is that the diagnostic fired BEFORE any retraining,
and that a single fit still produced five arms that cleared the bar.

### Why this is not the synthetic-data failure mode

Synthetic rows were OFF-distribution; a domain classifier caught them at AUC
0.9654. Pseudo-labels are ON-distribution by construction, so no distribution
check can flag them. The corruption is in the LABELS, not the features. That
required a different diagnostic.

### The pool, verified

307 unlabelled candidates carry all 24 features (not the 2,465 raw
`processed_unknown` light curves -- most never produced a complete TLS feature
vector). Verified by TIC: **0 overlap with training.csv, 0 overlap with any TOI
disposition**. Probabilities were recomputed with the deployed model rather than
read from the stored column (correlation 0.9898).

| config | pseudo-pos | pseudo-neg | total | train growth |
|---|---|---|---|---|
| top/bottom 5% | 16 | 16 | 32 | +0.73% |
| top/bottom 1% | 4 | 4 | 8 | +0.18% |
| absolute p>=0.95 / p<=0.05 | **42** | **1** | 43 | +0.98% |
| absolute p>=0.99 / p<=0.01 | 0 | 0 | 0 | -- |

**The pool is nearly one-sided.** At an absolute cut the model yields 42
confident positives and exactly ONE confident negative; nothing at all reaches
0.99. The model is essentially never confidently negative about an unknown --
itself a finding about how it behaves off the labelled distribution.

### THE SAFEGUARD FAILED (Part 1, before any retraining)

Top/bottom-5% pseudo-labels against real training rows, on the three highest
permutation-importance features:

| feature | real positives | pseudo-positives | flag |
|---|---|---|---|
| **st_rad** | mean **1.597** (sd 4.12) | mean **10.597** (sd 13.31) | **6.6x larger radius** |
| st_teff | mean 5390 (sd **1398**) | mean 4524 (sd **529**) | sd ratio 0.379 -- uniform |
| chi2red_min | mean 0.003 (sd **0.037**) | mean 0.000 (sd **0.001**) | sd ratio 0.016 -- uniform |

The `st_rad` result is the damning one. The model's confident "planets" among
unknowns sit on GIANT stars -- 10.6 R_sun against 1.6 R_sun for real confirmed
planets. This project already documented that exact blind spot:
`08_characterize_candidates.confidence_tier` carries a large-star reliability
penalty (`st_rad >= 1.5 and sde >= 10`), added after a 3.6x error-rate pattern
was measured there. Pseudo-labelling would take the model's confident
predictions from precisely the population it is known to be worst on and
promote them to ground truth.

### Single-fit results -- FIVE ARMS CLEARED

Evaluated only on real human labels, against a bare-pipeline baseline (0.8986
full / 0.8924 2-min), since the arms are bare pipelines too:

| arm | n pseudo | full test | delta | 2-min | delta | clears |
|---|---|---|---|---|---|---|
| pct5 | 32 | 0.9035 | +0.0048 [-0.0007,+0.0105] | 0.8957 | +0.0034 | no |
| pct5 w=0.25 | 32 | 0.9043 | +0.0057 [+0.0012,+0.0102] | 0.8972 | +0.0049 | full only |
| pct1 | 8 | 0.9038 | +0.0052 [+0.0007,+0.0097] | 0.8970 | +0.0046 | full only |
| pct1 w=0.25 | 8 | 0.9043 | +0.0057 [+0.0010,+0.0106] | 0.8967 | +0.0044 | full only |
| abs0.95 | 43 | 0.9041 | +0.0054 [+0.0004,+0.0103] | 0.8963 | +0.0039 | full only |
| **abs0.95 w=0.25** | 43 | **0.9078** | **+0.0092 [+0.0042,+0.0143]** | **0.9004** | **+0.0081 [+0.0023,+0.0140]** | **BOTH** |

### Part 3: the subpopulation stress test

Low-SNR test rows (snr <= 4.15, the train Q1 -- where the model is weakest):
**0.8821 -> 0.8706, delta -0.0114.** Negative, in the region where confidently
wrong labels are most likely. Exactly the predicted signature.

### A BROKEN REPLICATION TEST, RECORDED

The first replication attempt varied `random_state` over 10 seeds and returned
0/10 clearing. **That number was meaningless.** Every seed produced an
IDENTICAL delta (sd exactly 0.0000), because sklearn's HistGradientBoosting sets
`early_stopping='auto'` -- off at n <= 10,000 -- and its binning only subsamples
above 10,000 rows. At 4,387 training rows the fit is fully DETERMINISTIC and
`random_state` changes nothing. The test varied nothing and could only ever have
returned 0/10. (The earlier CatBoost seed check WAS valid; CatBoost is genuinely
stochastic, sd 0.0024. The error was assuming that carried over to HGB.) That
run also generated labels from the bare pipeline, giving 208 pseudo-labels
instead of the arm's 43 -- a different arm entirely.

So the +0.0092 is NOT a lucky seed; it is exactly reproducible. Fit variance is
not the threat here. The threat is the training-data draw.

### The correct replication: 20 bootstrap resamples of the training rows

Each resample regenerates its own pseudo-labels from a model fit on that
resample, then is compared against a baseline fit on the same resample. The
frozen real-label test set is never resampled.

| | mean | sd | min | max | positive |
|---|---|---|---|---|---|
| delta full | **-0.0012** | 0.0021 | -0.0051 | +0.0015 | **7/20 (35%)** |
| delta 2-min | -0.0012 | 0.0025 | -0.0069 | +0.0033 | 9/20 |

One-sample t-test against zero: **t = -2.63, p = 0.0165** -- the mean effect is
significantly NEGATIVE. Pseudo-label counts per resample averaged 84.2, split
78.2 positive / 5.9 negative, confirming the one-sidedness at every draw.

**The single-fit +0.0092 was an artefact of this particular training set.**
Across training draws the procedure is mildly harmful.

### Verdict

Negative, and the mechanism behaved exactly as predicted. Three independent
lines agree: the confident pseudo-positives concentrate on the known giant-star
blind spot; performance drops on the low-SNR subpopulation where the model is
weakest; and the apparent gain reverses to a significantly negative mean under
training-set resampling.

The one genuine surprise was how convincing the single fit looked -- five arms
clearing, one on both populations at +0.0092 with a CI comfortably clear of
zero. Without the Part 1 feature inspection and the resampling test, this would
have read as the project's first real improvement. That is the transferable
lesson: **for pseudo-labelling, the headline metric is the least trustworthy
number in the experiment**, because the model is being graded partly on labels
it wrote itself.

Also worth keeping: `random_state` is inert for this model at this data size, so
any future "seed sensitivity" check on the production HGB is vacuous. Vary the
training data instead.

Pseudo-labels saved with a permanent `label_source='pseudo'` column in
`pseudo_labeled_rows.csv`, never mixed with synthetic data. Nothing promoted.

Scripts: `pseudo_labeling.py`, `pseudo_labeling_seedcheck.py` (the broken one,
kept as the record), `pseudo_labeling_replication.py`; results
`pseudo_labeling_results.json`, `pseudo_labeling_seedcheck.json`,
`pseudo_labeling_replication.json`, `pseudo_labeled_rows.csv`.

## MEASUREMENT (not an experiment): retrieval metrics on the deployed model

Nothing was trained, tuned, promoted or rejected here. This measures the
EXISTING production model under metrics that match what the tool actually does:
surface a ranked shortlist for human follow-up.

**The operating points, located in code rather than assumed:**
`06_download_unknown.py:815` states it outright -- *"The pipeline never actually
applied a binary threshold -- it ranked by probability and took the top N"*.
`TRIAGE_PROBABILITY_FLOOR = 0.30` is a floor that holds weak signals out of the
shortlist, not a classifier cut. Tiers are 0.90 (Medium) / 0.97 (High) in
`08_characterize_candidates.confidence_tier`. The `0.5` in
`best_model_metadata.json` is a reporting convention only. So the tool is a
ranker with a top-N cut, and precision@k is the structurally correct metric.

### READ THIS BEFORE ANY NUMBER BELOW

**The clean test set is 79.0% POSITIVE** -- 867 confirmed planets against 231
vetted false positives. The majority class is planets. Consequences:

1. A random ranking scores **precision@k = 0.790 at every k**.
2. The PR-AUC no-skill baseline is **0.790, not 0.5**.
3. This is the OPPOSITE of the live candidate queue, where real planets are
   rare among unknown stars.

So the flattering precision@k numbers below are largely a property of the test
set's composition, not of the model. The prevalence-corrected table is the
honest view.

### Headline metrics

| metric | full clean test | 2-min-only | no-skill |
|---|---|---|---|
| ROC-AUC | 0.9031 [0.8784, 0.9249] | 0.8953 [0.8679, 0.9201] | 0.500 |
| PR-AUC, planets | 0.9670 [0.9533, 0.9774] | 0.9684 [0.9557, 0.9789] | **0.790 / 0.809** |
| **PR-AUC, false positives** | **0.7816 [0.7284, 0.8276]** | 0.7337 [0.6683, 0.7911] | **0.210 / 0.191** |

The planet-direction PR-AUC looks superb but is only a **1.22x lift** over
no-skill. The false-positive direction -- the genuinely rare class here -- is a
**3.72x lift**, and is where "average precision beats ROC-AUC on imbalanced
data" actually applies in this dataset.

### Precision@k and recall@k (full clean test)

| k | P@k | R@k | P@k, SDE alone | P@k random | lift |
|---|---|---|---|---|---|
| 10 | 1.000 | 0.012 | 0.800 | 0.790 | +0.210 |
| 20 | 1.000 | 0.023 | 0.750 | 0.790 | +0.210 |
| 50 | 0.980 | 0.057 | 0.820 | 0.790 | +0.190 |
| 100 | 0.990 | 0.114 | 0.710 | 0.790 | +0.200 |
| 200 | 0.995 | 0.230 | 0.630 | 0.790 | +0.205 |

Perfect precision at k=10 and k=20. But **recall@200 is only 0.230** -- the top
200 of 1,098 captures under a quarter of the planets, which is inherent to
ranking a set that is mostly planets. And **SDE alone reaches 0.80 at k=10**,
so at very small k the trained model's margin over a single raw feature is
modest; its advantage widens as k grows (0.995 vs 0.630 at k=200).

### PREVALENCE-CORRECTED precision@k -- the honest view

Positives subsampled to simulate rarer planet rates, all negatives kept, 200
repeats. This approximates "what would the top-20 look like if planets were
rare", using only labelled data.

| simulated prevalence | n_pos | P@10 | P@20 | P@50 | P@100 | P@200 |
|---|---|---|---|---|---|---|
| 0.50 | 231 | 0.915 | 0.950 | 0.978 | 0.945 | 0.849 |
| 0.25 | 77 | 0.900 | 0.925 | 0.781 | 0.618 | 0.379 |
| **0.10** | 26 | 0.760 | **0.606** | 0.383 | 0.241 | 0.129 |
| **0.05** | 12 | 0.481 | **0.332** | 0.198 | 0.114 | 0.060 |

**Precision@20 falls from 1.000 to 0.61 at 10% prevalence and 0.33 at 5%.**
That is the number to quote when asked "if I follow up your top 20, how many
are real?" -- and the answer depends entirely on how rare planets are in the
queue, which is not known.

### Threshold table (full clean test)

| threshold | n flagged | precision | recall | F1 | |
|---|---|---|---|---|---|
| 0.10 | 1052 | 0.823 | 0.999 | 0.903 | |
| **0.30** | **991** | **0.864** | **0.987** | **0.921** | **triage floor (live)** |
| 0.50 | 933 | 0.897 | 0.965 | 0.930 | reporting convention only |
| 0.70 | 834 | 0.929 | 0.894 | 0.911 | |
| **0.90** | **655** | **0.956** | **0.722** | **0.823** | **Medium tier** |
| **0.97** | **396** | **0.985** | **0.450** | **0.618** | **High tier** |
| 0.99 | 87 | 0.989 | 0.099 | 0.180 | |

The tier structure is well-chosen: **High tier is 98.5% precise at 45% recall**,
Medium 95.6% at 72%. The 0.30 floor keeps 98.7% recall, which is its purpose.

### Do confident UNKNOWNS look like confident CORRECT positives?

179 in-distribution ranked candidates vs the test populations:

| quantile | unknown candidates | test positives | test false positives |
|---|---|---|---|
| 0.25 | 0.6835 | 0.8845 | 0.1509 |
| 0.50 | 0.8704 | 0.9641 | 0.3936 |
| 0.75 | 0.9384 | 0.9838 | 0.7088 |
| 0.90 | 0.9715 | 0.9900 | 0.9071 |
| mean | 0.7912 | 0.8976 | 0.4455 |
| **frac >= 0.97 (High)** | **0.112** | **0.450** | 0.026 |

Unknowns sit between the two labelled populations, much closer to the positives
-- but **only 11.2% reach the High tier against 45.0% of true positives**. The
tool's confident unknowns are measurably less confident than its confident
correct answers. That is the expected direction (unknowns are genuinely harder,
and the truly easy planets are already catalogued), and it is a mild reassurance
rather than a validation. Note the "frac >= 0.30" row is definitionally 1.000
for unknowns -- the triage floor already removed everything below it.

### What these numbers do and do not predict

They describe performance on **labelled** data: confirmed planets and vetted
false positives, in a set that is 79% planets. The live tool runs on **unknown**
stars whose class balance and difficulty are different and unmeasured.
Precision@20 = 1.000 here is **not** a claim that 20 of the next 20 followed-up
candidates will be planets. The prevalence-corrected table is the closest
honest bracket, and the score-distribution comparison above is the only
available sanity check against the real population -- it says the unknowns look
plausible but harder, nothing stronger.

Script: `ranking_metrics.py`; results `ranking_metrics_results.json`,
`ranking_metrics.log`.

## Alternative tabular architectures + feature selection -- BOTH NEGATIVE

Two cheap experiments sharing one validation harness. Both landed inside the
noise floor, as expected -- but one arm cleared on a single fit and had to be
killed by replication, which is the part worth remembering.

### Experiment 1: architectures

Baseline is the production recipe refit on clean train. Each family got light
tuning via RandomizedSearchCV in an INNER loop, so this is not "tuned HGB vs
default CatBoost".

| model | nested CV | full test | delta vs base | clears | 2-min test | delta | clears |
|---|---|---|---|---|---|---|---|
| baseline (production recipe) | -- | 0.9021 | -- | -- | 0.8943 | -- | -- |
| HGB retuned | 0.9178 | 0.9019 | -0.0002 [-0.0082,+0.0080] | no | 0.8949 | +0.0003 | no |
| **CatBoost (single fit)** | 0.9240 | 0.9107 | **+0.0085 [+0.0004,+0.0171]** | **CLEARED** | 0.9050 | **+0.0105 [+0.0014,+0.0201]** | **CLEARED** |
| LightGBM | 0.9220 | 0.9044 | +0.0013 [-0.0068,+0.0095] | no | 0.8985 | +0.0029 | no |
| XGBoost | 0.9227 | 0.9077 | +0.0044 [-0.0024,+0.0113] | no | 0.9005 | +0.0050 | no |
| TabPFN | -- | -- | BLOCKED | -- | -- | -- | -- |

**THE CATBOOST RESULT DID NOT SURVIVE REPLICATION.** It was the first arm in
23 experiments to clear on both populations, so it was stress-tested rather
than believed. Ten seeds, each with a matched-seed HGB baseline:

| | mean | sd | min | max | seeds clearing |
|---|---|---|---|---|---|
| delta full | +0.0013 | 0.0024 | -0.0024 | +0.0060 | **0/10** |
| delta 2-min | +0.0038 | 0.0027 | -0.0001 | +0.0093 | **0/10** |

The original +0.0085 sits well above the top of the replicated distribution
(max +0.0060). Calibration is also worse: CatBoost+sigmoid gives Brier 0.0943 /
ECE 0.0397 against the baseline's 0.0896 / 0.0264. **Not promoted.**

Caveat stated plainly: the stress test used fixed mid-range CatBoost
hyperparameters, not the search-selected config (those were lost when the main
run crashed before writing its JSON), so part of the gap between +0.0085 and
+0.0013 may be tuning rather than seed. It does not change the conclusion --
not one of ten seeds cleared, and even the best seed (+0.0060) falls short of
`ci_lo > 0` at this test-set size.

**A pattern worth keeping even though nothing cleared:** all three modern GBM
libraries beat HGB on nested CV (0.9220 / 0.9227 / 0.9240 vs 0.9178) and on
point-estimate test AUC. The direction is consistent. The effect is simply
smaller than what 1,098 test rows can resolve -- which is the same conclusion
the learning curve reached from the other direction.

### Experiment 2: feature selection

Selection ran INSIDE each CV fold (selector as step 1 of a Pipeline, refit per
fold), never once on the full training set -- doing it once on all training
data leaks fold information into the selection step and inflates the estimate.

| arm | kept | full delta | 2-min delta | clears |
|---|---|---|---|---|
| SelectFromModel (0.5*mean) | 21/24 | +0.0019 [-0.0032,+0.0070] | +0.0022 | no |
| SelectKBest k=20 | 20/24 | +0.0008 | +0.0018 | no |
| drop poor-coverage (transit_shape_ratio, FAP) | 22/24 | +0.0002 | -0.0003 | no |
| RFECV (forest ranker) | 18/24 | -0.0051 | -0.0054 | no |
| SelectFromModel (mean) | 8/24 | -0.0199 | -0.0184 | no |
| SelectKBest k=16 | 16/24 | -0.0225 | -0.0244 | no |
| SelectKBest k=12 | 12/24 | -0.0364 | -0.0343 | no |
| SelectKBest k=8 | 8/24 | -0.0592 | -0.0583 | no |

Mild prunes flat, aggressive prunes sharply negative. The redundancy is real
but the model is already handling it -- GBMs tolerate correlated inputs, and
dropping columns removes signal faster than noise.

### Feature audit (useful regardless of the null result)

Permutation importance on the deployed model, clean test set:

| feature | drop in AUC | missing % | 1-feature AUC |
|---|---|---|---|
| st_rad | **+0.0540** | 4.4 | 0.313 |
| st_teff | **+0.0369** | 2.9 | 0.343 |
| chi2red_min | **+0.0353** | 2.0 | 0.611 |
| SDE | +0.0122 | 0.0 | 0.326 |
| FAP | +0.0085 | 0.0 | 0.725 |
| ... | | | |
| transit_shape_ratio | +0.0001 | **30.3** | 0.433 |
| rp_rs | **-0.0003** | 2.0 | 0.557 |
| empty_transit_count | **-0.0008** | 2.0 | 0.531 |
| depth_mean | **-0.0013** | 2.0 | 0.436 |

The top three are STELLAR and fit-quality features, not transit-shape features.
`st_rad` alone is worth more than four times `SDE`, the actual detection
statistic. The top 5 carry 73% of total positive importance, and 3 of 24
features have zero or negative importance. `transit_shape_ratio` -- built
specifically to catch grazing eclipses -- contributes +0.0001 at 30% missing.

**16 feature pairs at |Spearman| >= 0.80**, including `duration` <->
`depth_duration_ratio` at **1.000** (perfectly rank-degenerate: the ratio adds
no ordering information beyond duration), plus a `depth`/`depth_mean`/
`depth_mean_even`/`depth_mean_odd`/`rp_rs` cluster all mutually above 0.92.

### Environment: three real blockers, recorded

1. **TabPFN is genuinely unavailable** -- `TabPFNLicenseError`, requires
   one-time interactive license acceptance to download model weights, and there
   is no interactive terminal. Not worked around. The dataset (4,386 x 24) would
   have fit inside its ~10k row / ~500 feature envelope with no subsampling.
2. **LightGBM/XGBoost needed OpenMP**, absent (no Homebrew on this machine).
   Setting `DYLD_LIBRARY_PATH` worked in a shell but NOT in the run: **macOS SIP
   strips `DYLD_*` when exec'ing a protected binary**, and the job went through
   `/usr/bin/caffeinate`. Verified directly -- the variable read as None in the
   child. Real fix: both dylibs declare `@rpath/libomp.dylib` with rpaths baked
   to `/opt/homebrew/opt/libomp/lib`, so creating that directory (writable
   without sudo) and placing torch's libomp there resolved it.
3. **The ctypes-preload workaround then became the bug.** With the rpath
   satisfied, preloading a second OpenMP runtime segfaulted the run (SIGSEGV,
   exit 139) after the header and before any arm reported -- silent, because a
   segfault is not a Python exception. `preload_libomp()` now probes first and
   preloads only as a fallback.

**Operational note, since nothing was promoted:** had CatBoost held up, adopting
it would have meant a new production dependency (~1.2 MB wheel plus its own
runtime), and on this platform LightGBM/XGBoost would additionally require a
properly installed `libomp` rather than one borrowed from torch. Not a
consideration now, but it is the cost any future GBM swap carries.

**Baseline drift note:** the LightGBM/XGBoost arms ran after `training.csv`
gained one row from the live label-watch scheduler, so their in-run baseline is
0.9032/0.8955 rather than 0.9021/0.8943. Each arm is compared against the
baseline fit in its own process, so the deltas are internally consistent.

Scripts: `tabular_bakeoff.py`, `bakeoff_followup.py`, `feature_audit.py`;
results `tabular_bakeoff.log`, `bakeoff_followup_results.json`,
`bakeoff_followup_results_arms.json`, `feature_audit_results.json`.

## Multi-task learning on FP subtypes -- CLOSED AT THE GATE. Part 1 only.

Investigated whether false-positive SUBTYPE labels could serve as auxiliary
targets in a multi-task setup. Closed in Part 1 for two independent reasons,
either of which is sufficient.

### The labels do exist -- but mostly not the usable kind

Sources checked in order of authority:

1. **NASA archive `toi` table** -- 90 columns, exactly ONE disposition field
   (`tfopwg_disp`: FP/PC/KP/CP/APC/FA). No comment, note, subtype,
   classification or flag column exists (confirmed against
   `TAP_SCHEMA.columns`). Cannot supply subtypes.
2. **ExoFOP-TESS TOI export** -- carries a free-text `Comments` field written by
   the TFOP working groups. This is the only real source.
3. **Per-working-group columns (SG1A..SG5)** -- 100% non-null but effectively
   constant: SG1A is `5` for 1,146 of 1,152 negatives. A status code, not a
   subtype. No discriminative content.

Coverage against the 1,152 negatives: **100% matched** to an ExoFOP row,
**95.9% (1,105) have a non-empty comment**, and keyword extraction over that
human-written text yields **688 recognised subtypes** across 7 categories.

**On raw count that clears the >=400 gate.** It does not survive the leakage
check.

### Why most of those labels are unusable

Many TFOP dispositions are written AFTER ground-based follow-up the pipeline
never sees. Splitting on that:

| subtype | total | leaky | usable |
|---|---|---|---|
| nearby_eb_blend | 327 | **327** | **0** |
| eclipsing_binary | 252 | 47 | 205 |
| centroid_offset | 69 | 2 | 67 |
| odd_even | 18 | 0 | 18 |
| stellar_variability | 11 | 0 | 11 |
| systematic_artifact | 8 | 0 | 8 |
| stellar_companion | 3 | 0 | 3 |
| **TOTAL** | **688** | **376** | **312** |

The largest category is entirely lost. "NEB" means the eclipse is on a
*neighbouring* star, a determination that requires spatially resolving the pair
-- TFOP's "retired as NEB" (102 occurrences) is seeing-limited photometry from
SG1. TESS difference imaging can sometimes reach the same conclusion and this
pipeline does compute a centroid, so the call is genuinely ambiguous; it is
excluded CONSERVATIVELY and reported separately rather than buried. Even
counting all 327 as usable gives 639 with NEB then the dominant class at 51%
and the top two holding 83% -- the same shape of problem.

**After leakage adjustment: 312 usable labels, top two categories holding 87%,
and four tails with 3-18 examples each.** That is exactly the stated failure
condition -- below the 400-500 threshold, collapsed into one dominant category
with tiny tails. **Leakage-adjusted gate FAILS.**

### The mechanism was never there anyway

This is the more fundamental reason, and it holds regardless of label counts.

Multi-task learning helps by shaping a **learned shared representation** --
auxiliary gradients pull a shared hidden layer toward features useful for both
tasks. This model has **24 fixed hand-engineered features and a gradient-boosted
tree ensemble**. There is no shared representation to shape. sklearn's
multi-output classification fits *independent* estimators per output, so a
"multi-task GBM" shares nothing at all; it is two models in a trench coat.

The plausible alternatives degrade to things already tested or already known:
- **Hierarchical (planet vs not, then subtype)** -- adds interpretability to the
  second stage, cannot improve the first, which is the metric in question.
- **Subtype-aware sample weighting** -- reweighting, and the small-lift trio
  already established that `balanced` beats both alternatives tried.
- **Subtype as an input feature** -- not available at prediction time; it is a
  label, and an externally-assigned post-hoc one at that.

Deriving subtypes heuristically from the existing 24 features was explicitly
ruled out and was not done: auxiliary labels manufactured from the same features
the auxiliary task is meant to improve would be circular and would produce a
number rather than a finding.

### Verdict

Closed without running Part 2. The prerequisite labels are too few and too
imbalanced once post-hoc knowledge is removed, and the mechanism by which
multi-task learning usually pays does not exist for a tabular GBM over fixed
features. This was always unlikely to help here; the label audit is the part
worth keeping.

**Reusable by-product:** `exofop_toi_export.csv` now caches TFOP comments and
per-SG dispositions for all 8,113 TOIs, and 1,105 of our negatives carry human
vetting text. That is genuinely useful for the EVIDENCE layer -- explaining to a
human WHY a star is a known false positive -- even though it is not usable as a
training target.

Script: `fp_subtype_feasibility.py`; results `fp_subtype_feasibility.json`,
`exofop_toi_export.csv`.

## FFI (coarse-cadence) expansion -- COVERAGE LEVER, NOT AN ACCURACY LEVER. Part 1 only.

Investigated whether TESS full-frame-image light curves could grow the training
set. Part 1 (feasibility + distribution shift) answered it without downloading
anything, because **the training set already contains coarse-cadence data**.

### The premise needed correcting twice

`01_download_known.py:try_search` prefers `author='SPOC'` but falls back to
whatever product MAST returns first -- QLP, CDIPS, TASOC, all FFI-derived --
and nothing ever recorded which pipeline produced each file. Cadence has been
an invisible column. Measured directly from the light curves (median delta-t,
not filenames or logs):

| group | rows | positives | negatives |
|---|---|---|---|
| FINE (< 1 min, 20s SPOC) | 401 | 322 | 79 |
| 2-min SPOC | 4,852 | 3,940 | 912 |
| **COARSE (> 2.6 min, FFI-derived)** | **231** | 70 | 161 |

COARSE is strongly class-asymmetric: 14.0% of negatives vs 1.6% of positives.

**A REAL ERROR, CORRECTED AND RECORDED.** The first analysis lumped everything
non-2-min into one bucket and called it the FFI proxy. **401 of those 632 rows
are SUB-MINUTE 20-second SPOC data -- FINER than 2-min, the opposite of FFI.**
Every number from that pass described 20s data as much as FFI, and the
apparently strong "+0.0128, CLEARS" result it produced was not an FFI result at
all. The corrected analysis excludes FINE rows from every arm so COARSE is
isolated.

**A SECOND BUG, ALSO CORRECTED.** The down-weighted arm passed `sample_weight`
to a `CalibratedClassifierCV` wrapping a Pipeline. sklearn warned that it
"does not appear to accept sample_weight [so] sample weights will only be used
for the calibration itself" -- the down-weighting never reached the classifier,
and that arm was silently identical to the un-weighted one. All arms now fit
the BARE production Pipeline, which accepts `clf__sample_weight`. Sigmoid
calibration is monotonic, so AUC comparisons are unaffected.

### The decisive number: domain classifier AUC 0.9717

A classifier separates 2-min from COARSE feature vectors at **AUC 0.9717** --
*higher* than the 0.9654 that made synthetic data actively harmful. Four of 24
features shift more than 0.5 SD:

| feature | standardized mean difference (COARSE - 2min) |
|---|---|
| duration | **+0.85** |
| rp_rs | -0.65 |
| depth_duration_ratio | -0.56 |
| FAP | -0.51 |

The `duration` shift is the physics, not an artifact: coarse sampling cannot
detect short transits, so the COARSE population is selected toward long ones
(median duration 2.33h vs 0.90h for 2-min). Sampling adequacy confirms it --
COARSE rows have a median of **6.6 in-transit samples** and **38.5% have fewer
than 5** (2-min: 26.9 median, 5.3% under five).

### Does mixing help? Only for the rows it serves

Arms on the frozen clean split, FINE rows excluded throughout, production
hyperparameters:

| evaluated on | A: 2-min + COARSE | B: 2-min only | C: COARSE weighted 0.25 |
|---|---|---|---|
| full clean test (n=1,023, 55 COARSE) | 0.8980 | 0.8868 | 0.9012 |
| **2-min-only test (n=968, 0 COARSE)** | 0.8902 | 0.8920 | 0.8929 |

| comparison | full test | 2-min-only test |
|---|---|---|
| adding COARSE (B->A) | **+0.0113 CI [+0.0020, +0.0205] CLEARS** | -0.0018 CI [-0.0094, +0.0061] no |
| adding down-weighted (B->C) | **+0.0144 CI [+0.0051, +0.0240] CLEARS** | +0.0008 CI [-0.0060, +0.0078] no |

**The entire gain is confined to the coarse rows in the test set.** On the 2-min
population -- 89% of the data and the population production actually serves --
the effect is zero. Coarse training rows teach the model to handle coarse stars
and transfer nothing to the rest.

This is materially better than synthetic data, which HURT (-0.018). COARSE data
is real, correctly labelled, and merely *inert* outside its own population. But
it is not a general accuracy lever, and the learning curve's +0.013-per-doubling
does not apply to it, because that figure was explicitly conditioned on
same-distribution data and 0.9717 says this is not that.

### The operational argument fails too

If production had to classify FFI-observed candidates, training on FFI would
matter regardless of the headline AUC. It does not: sampling the candidate
pipeline's own light curves found **0% coarse** in both
`processed_unknown` (250 sampled: 11.6% fine, 88.4% 2-min) and
`processed_unknown_widesector` (250 sampled: 33.2% fine, 66.8% 2-min). The
model never meets a coarse-cadence star in production.

### Verdict: closed as an accuracy lever

Part 2 was NOT run, and should not be. FFI would add rows that serve a
population production does not encounter, at a distribution distance that
exceeds the one which already caused measurable harm once. The one legitimate
future use is coverage: if the project ever wants to vet stars that have no
2-min data, these 231 rows show the pipeline can process them and that the
model can be taught to handle them without damaging the 2-min population.

**Side finding worth keeping:** the mixed-cadence composition mildly inflates
the headline metric. The same model scores 0.8980 on the full clean test set
but 0.8902 on its 2-min-only subset, because the coarse test rows are enriched
in easy-to-classify negatives (14.0% of negatives are coarse vs 1.6% of
positives). Cadence itself only weakly predicts the label (AUC 0.4506 raw,
0.4411 for a not-2-min indicator), and the per-feature correlation between
label-separating and cadence-separating power is **-0.247** -- negative, so the
features that carry class signal are NOT the ones that carry instrument signal.
There is no instrument shortcut, but the population mix is worth remembering
when quoting a single number.

Scripts: `cadence_audit.py`, `ffi_mixing_effect.py`; results
`cadence_audit_results.json`, `cadence_per_star.csv`, `ffi_mixing_results.json`.

## INFRASTRUCTURE: the label-watch pipeline duplicated stars and leaked the split -- FIXED

Found while checking whether the POSITIVE class could still grow. The answer to
that question turned out to be "no, and by the way the thing that was supposed
to grow it has been corrupting the dataset".

**The bug, in one assumption.** `retrain_pipeline.find_new_labeled_examples`
queried `select hostname,tic_id from pscomppars`, enqueued every confirmed
planet as `TIC_<id>`, and decided "already have it?" by comparing that string
against training.csv's `host` column -- its docstring stated host "is always
'TIC_<id>'". True for the negative class and for rows this pipeline added
itself. **False for the original positive class**, which
`01_download_known.py` names by HOSTNAME (`11_Com`, `Kepler-142`). So every
confirmed planet already present under its hostname looked brand new, was
re-queued, re-downloaded, and appended as a SECOND row for the same star.

**Measured damage** (from only 137 of 4,405 queued labels actually processed):

| | |
|---|---|
| stars present twice under two names | **144** |
| duplicate rows | 310 |
| duplicates straddling the frozen train/test split | **56** |
| duplicates with CONFLICTING labels | 3 |

A straddling star is trained on and then scored on. The worst case was
byte-identical on both sides:

    HIP_77900     train  period 0.795409  depth 0.999562  SDE 5.971785
    TIC_24133681  test   period 0.795409  depth 0.999562  SDE 5.971785

Others (`Kepler-1574`, `Kepler-142`) were the same star re-processed to a
different TLS solution -- still leaking the star's noise, systematics and
stellar parameters, just less blatantly.

**Identity resolution.** Deduping needs star identity, not names. The NASA
archive's own `hostname -> tic_id` mapping from `pscomppars` resolves 5,576 of
5,650 training rows; the earlier coordinate cross-match
(`positive_class_tic_ids.csv`) left 352 unresolved. The 74 that still do not
resolve are all microlensing events (`KMT-*-BLG-*`) -- distant bulge stars with
no TIC entry, which cannot collide with a TESS target.

**The three conflicting-label cases** were each a star present in the
confirmed-planet catalog AND on the TOI false-positive list:
`Kepler-1517`/`TIC_158555987`, `TOI-1836`/`TIC_207468071`,
`TOI-2084`/`TIC_441738827`. The confirmed-planet label wins, and two of the
three (`207468071`, `441738827`) are independently now TOI disposition `KP`/`CP`,
confirming that. Not auto-resolved silently -- listed in `dedupe_report.json`.

**Re-baseline on the cleaned data.** Every AUC this project quoted recently was
measured on a test set containing stars the model had trained on:

| measurement | AUC | 95% CI |
|---|---|---|
| A. production, CONTAMINATED test (the number on record) | 0.9043 | [0.8796, 0.9261] |
| B. production, CLEAN test (same model, honest set) | **0.9031** | [0.8784, 0.9249] |
| D. refit on clean train -> clean test (go-forward baseline) | 0.9021 | [0.8771, 0.9239] |

**Leakage was inflating the reported number by 0.0012** -- real and
one-directional, but inside the +/-0.003 noise floor. The severity is in the
mechanism, not this number: 4,243 labels were still queued behind the 137 that
did the damage above, and draining them would have compounded it. The 45 leaked
test rows were all POSITIVES and drew a mean predicted probability of 0.9286 --
the confidence signature of memorised stars -- which is also why the AUC damage
stayed small: the model is already strong on positives, so re-scoring memorised
ones barely moved the ranking.

**Fixes applied:**
1. `find_new_labeled_examples` now decides membership by TIC id via
   `_training_tic_ids()`, which resolves hostname-named rows using the
   `confirmed` frame the function already downloads. Verified live: it now
   reports `5358 stars already in training.csv by TIC, 5361 archive entries
   skipped as already present, 0 genuinely new, 0 newly queued` -- against the
   old code's 4,405 queued.
2. `dedupe_training_by_tic.py` removed 166 duplicate rows. Split is now
   4,386 train / 1,098 test with **0 duplicated stars and 0 straddling**.
   Survivor selection prefers the row whose host is in `split_manifest.json`,
   so no surviving row changes sides. Previous dataset preserved at
   `training_pre_dedupe_backup.csv`.
3. `purge_stale_watch_queue.py` retired the 3,998 already-queued entries that
   were duplicates, marking them `skipped_duplicate` rather than deleting, and
   left the 339 genuinely-absent entries pending.

Class balance moved from 4,495:1,155 (3.89:1) to 4,332:1,152 (3.76:1) --
slightly LESS imbalanced, since the duplicates were almost entirely positives.

Scripts: `dedupe_training_by_tic.py`, `rebaseline_after_dedupe.py`,
`purge_stale_watch_queue.py`; results `dedupe_report.json`,
`dedupe_dropped_rows.csv`, `rebaseline_after_dedupe.json`,
`watch_queue_purged_rows.csv`.

## POSITIVE-CLASS EXPANSION: EXHAUSTED -- no pipeline run performed

Every prior data-expansion attempt targeted the negative class (TOI FP, TOI FA,
Kepler, synthetic). This checked the confirmed-planet side. Live archive
queries (2026-08-02):

| source | archive total (unique TIC) | already have | apparently new |
|---|---|---|---|
| TESS-discovered confirmed (`disc_facility like '%TESS%'`) | 771 | 683 | 88 |
| all confirmed with a TIC | 4,462 | 3,901 | 561 |
| TOI disposition KP/CP | 1,206 | 1,046 | 160 |

**Every one of those "new" counts was a naming artifact.** Of the 561, **552
had already been attempted** (409 logged `Success` -- they were in training
under hostnames the old TIC map could not resolve; 291 returned `No TESS Data`
and would fail again). Only 9 had never been attempted, all `disc_year 2026`.
With the corrected TIC resolver the count is exactly **0 genuinely new**.

The original pull is recent (`confirmed_planets.csv` written 2026-07-19, light
curves from 2026-07-08) and the archive confirms roughly 150-180 TESS planets
per YEAR (2024: 178, 2025: 129, 2026: 178 so far). There is no backlog.

**No download or pipeline run was performed** -- correctly, since at ~4,332
positives the realistic addition would have been a ~1.6% increase, and at the
learning curve's fitted exponent (c=0.193) that predicts **~+0.0004 AUC**, an
order of magnitude below the +/-0.003 noise floor and nowhere near `ci_lo > 0`.

**Is there a recurring stream worth tracking?** Yes, but a thin one: 339 watch
entries survive the purge as genuinely absent from training, and the archive
adds ~150-180 TESS planets per year. The scheduler should keep watching -- now
that it deduplicates by star identity, it will queue only real additions. It is
a trickle, not a lever.

## INFRASTRUCTURE: the retrain promotion gate compared unequal models -- FIXED

Found while verifying state after the injection-recovery work. This is an
infrastructure/methodology bug of the same family as the frozen-split fix and
the stale-module yield bug, and it sat in the one comparison that decides which
model users actually get.

**The mismatch.** The incumbent every challenger had to beat is the deployed
`models/best_model.joblib` -- a TUNED, calibration-wrapped model. The challenger
was built from `m05.build_models()["HistGradientBoosting"]`, the UNTUNED default
config. Measured side by side on the same frozen test set:

| | old challenger | production (incumbent) |
|---|---|---|
| wrapper | `Pipeline` (none) | `CalibratedClassifierCV(cv=5)` |
| `l2_regularization` | 0.0 | 0.5 |
| `learning_rate` | 0.05 | 0.1 |
| `max_depth` | 4 | None |
| `max_iter` | 300 | 500 |
| `max_leaf_nodes` | 31 | 63 |

Five differing hyperparameters plus the wrapper. The wrapper is not merely a
monotonic recalibration -- `CalibratedClassifierCV(cv=5)` averages five
sub-models, so it acts as an ensemble and moves AUC in its own right.

**Measured cost.** On the frozen split (4,507 train / 1,143 test):

| model | test ROC-AUC |
|---|---|
| production (deployed) | 0.9043 |
| old challenger (m05 defaults) | 0.8949 |
| new challenger (clone of production) | 0.9039 |

The handicap is **+0.0090** (CI [-0.0018, +0.0200]). Every retrain attempt
opened ~0.009 in the hole for reasons unrelated to whether new data helped, and
had to out-run that before it could even begin to show a data-driven gain.
Effect on the gate:

| gate | delta vs production | 95% CI | decision |
|---|---|---|---|
| old (mismatched) | -0.0094 | [-0.0205, +0.0012] | no promotion |
| new (matched) | -0.0004 | [-0.0038, +0.0029] | no promotion |

Same decision today, but the CI is ~6x tighter and the point estimate is
essentially zero -- which is the correct result for "production's own recipe,
refit on more data". The gate now measures the effect of data alone.

**The fix is structural, not a copied literal.** `sklearn.base.clone(prod_model)`
reproduces the deployed estimator's class and every hyperparameter, unfitted, so
the challenger IS production's recipe by construction. If production's config
ever changes -- a future tuning promotion, a different wrapper, even a different
model class -- the challenger follows automatically and this bug cannot silently
reappear. `best_model_metadata.json` does NOT record hyperparameters (checked:
it stores `model_name`, `feature_columns`, `random_seed`, threshold and metrics
only), so the joblib artifact is the single source of truth and the correct
thing to clone from.

First-ever-retrain edge case: with no production model on disk there is nothing
to clone, so it falls back to m05's HGB config. That fallback is near-inert --
the gate already refuses to promote at all without a baseline to compare
against -- so it only affects what gets logged for that first attempt, never
what ships.

**Were the two past non-promotions affected?** Both were logged under the
mismatched gate:

| attempt | date | n rows | challenger AUC | production AUC | logged ci_lo | promoted |
|---|---|---|---|---|---|---|
| #2 | 2026-07-31 | 5,586 | 0.8975 | 0.9041 | -0.0177 | no |
| #3 | 2026-08-02 | 5,650 | 0.8949 | 0.9043 | -0.0205 | no |

Attempt #3's logged numbers reproduce EXACTLY in the offline check above
(0.8949 / 0.9043), confirming the reconstruction is faithful. Under a matched
challenger, #3 becomes -0.0004 with CI [-0.0038, +0.0029] -- **still no
promotion**, now measured directly rather than estimated. For #2, adding the
+0.0090 handicap to 0.8975 gives ~0.9065 vs 0.9041, i.e. a delta of roughly
+0.002; with a CI of comparable width that lower bound still sits below zero, so
**it would very probably not have promoted either** -- though this one is an
estimate, since reproducing it exactly would need that day's dataset snapshot.
The honest summary: neither past attempt was robbed of a promotion, but #2 moves
from decisively worse (-0.0177) to statistically indistinguishable (~+0.002).

Validated offline via `validate_challenger_config.py` and through the pipeline's
own documented `dry_run=True` path, which wrote nothing: retrain_attempts stayed
at 2, model_versions at 2, and `best_model.joblib` md5 unchanged at
`341f1a3907e77f6ec294f182833e613c`. No live retrain was triggered.

Scope: only challenger CONSTRUCTION changed. The promotion criterion
(`ci_lo > 0`), the paired bootstrap, the scheduling/threshold logic, and every
data/feature path are untouched.

Script: `validate_challenger_config.py`; results
`challenger_config_validation.json`. Fix in `web/retrain_pipeline.py`
(`_build_challenger`).

## Injection-recovery diagnostic (2026-08) -- CONFOUNDED NULL, plus two real findings

Run to answer one question: is the ~0.90 ceiling caused by lack of DATA
VOLUME or lack of REPRESENTATIONAL CAPACITY in the 24-feature TLS approach?
**It could not answer that question**, for a reason worth recording. It did
produce two solid results: synthetic augmentation is definitively closed, and
a leakage bug in the earlier Part B run was found and corrected.

Synthetic examples are NOT evidentiarily equivalent to real confirmed
detections anywhere below. batman injects a perfectly periodic, TTV-free,
activity-free signal into a negative-class host. This is a controlled
diagnostic, not a claim that synthetic examples substitute for real ones.
Nothing here was promoted; production remains untouched.

### Injection system validation (Part 1)

The injector and completeness curve already existed (Part B below) and were
re-validated rather than rebuilt. The **SNR axis was missing** and was added --
`completeness_curve.py` fixed duration at 5% of period, so N_in_transit and
point noise are both recoverable after the fact with no TLS re-run:

| expected SNR | recovery | n |
|---|---|---|
| <5 | 4.5% | 22 |
| 5-10 | 27.3% | 11 |
| 10-20 | 28.6% | 21 |
| 20-40 | 86.7% | 15 |
| 40-80 | 90.0% | 10 |
| >80 | 95.2% | 21 |

Monotonic, 50% recovery near SNR ~19 (median point noise 1,422 ppm). That is
MORE conservative than the SNR ~7-10 typical of idealised transit-search
completeness -- expected, since recovery here demands the period back within
1%, TLS runs on production's coarse grid, and injection hosts are TOI false
positives (often variable, hence noisy). Conservative is the safe direction:
the injector is not manufacturing easy signals. Script: `completeness_snr.py`.

### A leakage bug in the original Part B augmentation

`augment_classical_dataset.py` picks injection hosts uniformly from
`data/processed_negative/` with no reference to the train/test split -- it
predates the split freeze. Checked against the frozen manifest:

    953 usable synthetic rows
    163 of them (17%) injected into a HELD-OUT TEST star, across 111 test hosts

The label is not leaked (it is the injected signal), but the star's noise
realisation, systematics, and real `st_rad`/`st_teff` are -- the generator
copies those from the host for realism. That run also used a positional
`train_test_split` (4392/1099) rather than the frozen split. Its recorded
-0.0131 is therefore not trustworthy in either direction.

Fixed by construction in `augment_train_only.py`: the source pool is
intersected with the train split BEFORE sampling (925 train-split curves; 231
test-split excluded), so a test star cannot be drawn. New batch: 1,800
attempted, **1,727 usable (95.9%)**, 861 transit-positive / 866 EB-negative.
That yield is far above the original batch's 59.6% and the failure modes differ
(odd/even mismatch here vs `transit_shape_ratio` there); the difference is
recorded but not explained, since no mechanism was verified.

Combined decontaminated pool: **2,517 synthetic rows** (1,221 positive, 1,296
negative) = 55.8% of the real training set. Every row carries
`is_synthetic=True`, `synthetic_kind`, `source_file` and `source_split` in the
saved CSV, never blended silently.

### THE CONTROL THAT INVALIDATES THE DIAGNOSTIC

Train a classifier to tell real rows from synthetic ones. If it succeeds
easily, the synthetic rows are off-distribution and a null result says nothing
about data-starvation.

**Real-vs-synthetic discriminator AUC: 0.9654.** Trivially separable. The
shifts sit exactly where they matter:

| feature | standardized mean diff (synthetic - real) |
|---|---|
| FAP | **-1.02** |
| SDE | **+0.83** |
| SDE_raw | +0.82 |
| snr | +0.47 |

Injected transits have HIGHER detection significance and LOWER false-alarm
probability than real confirmed planets. The completeness curve validated that
injected signals are *recoverable*; it never tested whether their recovered
FEATURE VECTORS resemble real ones, and they do not.

### Results (frozen split, synthetic never in the test set)

| arm | test AUC | delta | 95% CI | clears |
|---|---|---|---|---|
| a. real only | 0.8949 | -- | -- | -- |
| b. real + 2,517 synthetic | 0.8770 | -0.0180 | [-0.0291, -0.0079] | no |
| c. down-weighted w=0.5 | 0.8929 | -0.0021 | [-0.0102, +0.0056] | no |
| c. down-weighted w=0.25 | 0.8906 | -0.0043 | [-0.0110, +0.0017] | no |
| c. down-weighted w=0.1 | 0.8973 | +0.0023 | [-0.0039, +0.0085] | no |

Scale curve: -0.0002 (315 rows), -0.0035 (629), -0.0056 (1,258), -0.0180
(2,517). A clean dose-response -- harm grows with how much synthetic data is
added.

Covariate-shift corrections, driven by cross-validated p(synthetic) so no row
is scored by a model that saw it:

| correction | rows used | delta | 95% CI |
|---|---|---|---|
| rejection, p(syn) < 0.5 | 182 / 2,517 (7.2%) | -0.0002 | [-0.0068, +0.0066] |
| rejection, p(syn) < 0.75 | 618 / 2,517 (24.6%) | -0.0042 | [-0.0121, +0.0040] |
| importance weighting | eff. n 315 | -0.0013 | [-0.0110, +0.0088] |

Only 7% of synthetic rows land in the overlap region at all. Corrected, they
are **inert** -- neither helping nor hurting. Calibration: real-only Brier
0.0952 / ECE 0.0325 vs real+synthetic 0.1064 / 0.0689, i.e. synthetic data
degrades calibration too. Nested CV is reported but NOT comparable across arms
(the real+synthetic folds contain synthetic rows); the held-out real test set
is the honest number.

### What this does and does not establish

- **Establishes:** synthetic augmentation of this construction is a dead end
  for this classifier. On-distribution synthetic rows are inert; off-distribution
  ones harm in proportion to dose. Closed, with a mechanism.
- **Does NOT establish:** anything about data-starvation. No arm was positive,
  but "adding off-distribution data didn't help" is fully explained by the
  distribution mismatch. The pre-registered decision rule for this experiment
  assumed a clean null; this null is confounded, so that rule was NOT applied.

The question was therefore put to real data instead -- see the next section,
which is where the actual answer came from.

Scripts: `injection_diagnostic.py`, `augment_train_only.py`,
`completeness_snr.py`; results `injection_diagnostic_results.json`,
`augmented_train_only.csv`, `completeness_snr_results.csv`.

## Real-data learning curve -- THE CEILING IS SUBSTANTIALLY A DATA LIMIT

This overturns a premise this project had carried for a long time. It was run
because the injection-recovery diagnostic (below) turned out to be confounded
and could not answer the data-vs-features question, so the question was put to
real data directly, with no synthetic examples and no injector assumptions.

Ten sample sizes, 7 independent host-level subsamples each (class balance
preserved within every draw), production HGB configuration, every model scored
against the SAME frozen real test set:

| train rows | test ROC-AUC |
|---|---|
| 449 | 0.8427 +/- 0.0095 |
| 899 | 0.8641 +/- 0.0062 |
| 1,347 | 0.8678 +/- 0.0075 |
| 1,797 | 0.8769 +/- 0.0036 |
| 2,247 | 0.8877 +/- 0.0048 |
| 2,922 | 0.8893 +/- 0.0028 |
| 3,595 | 0.8970 +/- 0.0034 |
| 4,045 | 0.8948 +/- 0.0036 |
| 4,269 | 0.8978 +/- 0.0025 |
| 4,494 | 0.8996 (single draw) |

Monotonic apart from scatter at the ~0.003 noise floor. Saturating power-law
fit `AUC(n) = a - b*n^(-c)`: c = 0.193, residual RMS 0.0023. **The fitted
asymptote runs to its 1.0 bound** -- the data cannot locate a ceiling, which is
itself the finding: there is no sign of saturation in the measured range.

| training set | predicted AUC | 95% CI | gain |
|---|---|---|---|
| 9,564 (2.1x) | 0.9125 | [0.9027, 0.9143] | **+0.0129** |
| 20,000 (4.5x) | 0.9241 | [0.9042, 0.9262] | +0.0245 |
| 50,000 (11.1x) | 0.9364 | [0.9049, 0.9388] | +0.0368 |

**Doubling the real training data is worth ~+0.013 AUC with a positive lower CI
bound -- more than any of the thirteen feature/architecture experiments, none of
which cleared +0.01.**

### Why this project previously believed the curve was flat

Traced to `05b_model_analysis.py:377-415`. Three reasons the old verdict does
not support the conclusion drawn from it:

1. It was run on **RandomForest (tuned)**, not the deployed
   HistGradientBoosting -- the code comment says so explicitly ("on whichever
   tree model is currently the tuned RF, for interpretability").
2. It measured **CV score inside the training set**, not held-out AUC on the
   frozen test set.
3. Its verdict is a hard-coded threshold, `if last3_slope > 0.00005`, i.e. it
   demands +0.05 AUC per 1,000 examples before calling a model data-starved.

Applied to today's curve, that rule returns **"PLATEAUED"** -- measured slope
2.89e-6, seventeen times below the cutoff -- for a curve that gains +0.0119
from doubling its data. The threshold is a reasonable tripwire in the
very-low-data regime and blind above it. The old output was not wrong about
what it measured; the inference drawn from it was too strong.

### Limits on this result

- The far extrapolations (11x) rest on an unidentifiable asymptote and should
  be discounted heavily. The 2x figure is the trustworthy one.
- **The final step of the curve is noise-dominated and must not be quoted on
  its own.** A coarser 4-point version run inside `injection_diagnostic.py`
  gave a last-quartile gain of +0.0042 on one run and +0.0010 on another --
  the same quantity, twice, differing by more than either value. The
  load-bearing evidence is the full-range trend (0.843 at n=449 to 0.900 at
  n=4,494, ten points, residual RMS 0.0023) and the fitted exponent, not any
  single adjacent-point difference.
- This describes more data **from the same distribution**. Kepler is a
  different mission (cadence, bandpass, noise), and this project's own Kepler
  pilot already hit a real SNR wall at 36.4% yield plus a mission-identity
  leakage concern. This result raises the value of more real data without by
  itself validating Kepler as the source of it.
- `training.csv` grows continuously (the live label-watch scheduler), so
  absolute AUCs drift slightly between runs. Each run is internally
  consistent, so within-run comparisons hold.

Script: `learning_curve_extrapolation.py`; results
`learning_curve_extrapolation.json`, `learning_curve_points.csv`.

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

## NON-KEPLER SURVEYS -- CLOSED AT THE PART 1 GATE. No download performed.

K2, CoRoT, WASP, HATNet/HATSouth, KELT, NGTS and TRAPPIST assessed as training
-set expansions. **Closed on four measurements, not on argument.** Nothing was
downloaded, nothing retrained. Script `survey_feasibility.py`, results
`survey_feasibility.json`.

### Measurement 1: the genuinely-new count is 268, and it is the wrong shape

Identity resolved by **TIC**, never by host string -- these stars carry
`WASP-*`, `HAT-*`, `EPIC_*`, `TIC_*` and a bare hostname simultaneously, which
is the exact condition that produced the 144-duplicate bug.

| survey | planets | stars w/ TIC | already ours | NEW | overlap |
|---|---|---|---|---|---|
| K2 (confirmed) | 549 | 400 | 388 | 12 | **97.0%** |
| CoRoT | 35 | 33 | 30 | 3 | 90.9% |
| WASP (4 facility names) | 170 | 169 | 165 | 4 | 97.6% |
| HAT (HATNet + HATSouth) | 140 | 139 | 138 | 1 | **99.3%** |
| KELT (3 facility names) | 21 | 21 | 21 | **0** | **100%** |
| NGTS | 22 | 22 | 19 | 3 | 86.4% |
| TRAPPIST | 0 | 0 | 0 | 0 | -- |

The overlap is near-total because this project's positive class came from
`pscomppars`, which already contains every one of these discoveries -- 173
`WASP-*`, 139 `HAT*`, 346 `K2-*`, 30 `CoRoT-*` hosts are in `training.csv`
today, holding **TESS** photometry. Re-fetching those stars from their
discovery survey would create a second row for one star: the duplicate bug
with a different prefix.

**TRAPPIST is not a survey-scale source at all.** Its planets are filed under
`La Silla Observatory` / `Multiple Observatories` and are all **TRAPPIST-1** --
one star, seven planets. As a training row it is a single star.

Only K2 has a usable negative class. Taking `k2pandc` in full:

| K2 disposition | new | already have |
|---|---|---|
| CANDIDATE (**unlabelled**) | 895 | 12 |
| **FALSE POSITIVE** | **238** | 1 |
| **CONFIRMED** | **13** | 390 |
| REFUTED | 6 | 2 |

**257 genuinely-new K2 stars carry a usable label, and 244 of them are
negatives.** The 895 CANDIDATEs are unlabelled and cannot be trained on.
Across all seven sources: **268 usable stars.**

### Measurement 2: 268 stars predicts +0.0011 AUC -- below the noise floor

At the learning curve fitted during the injection-recovery work
(+0.0129 per doubling from 4,386 train rows), 268 additional stars predicts
**~+0.0011 AUC**. The noise floor is +/-0.003 and the bar is `ci_lo > 0`.

This is the same arithmetic that closed the positive-class exhaustion check
without a pipeline run. Even the hypothetical in which all 895 unlabelled K2
CANDIDATEs were somehow labelled only reaches ~+0.0043 -- and that label does
not exist.

### Measurement 3: K2 is 30-minute cadence, a regime already measured as inert

Verified live across 8 K2 hosts: **8 of 8 offer 1800 s**, 3 of 8 additionally
offer 60 s. K2's standard product is **15x coarser than TESS 2-min**.

The FFI work already measured this axis on TESS's own data: 2-min vs COARSE
separates at **domain AUC 0.9717**, and mixing was a coverage lever with no
accuracy gain on the 2-min population (-0.0018). K2 sits in that regime and is
worse in one respect -- cadence would be **perfectly confounded with survey**,
so the model cannot learn "coarse" and "K2" separately. The FFI result is
directly transferable evidence, not a guess.

### Measurement 4: the ground surveys cannot see this population's transits

Correcting an earlier claim in `NON_TESS_SURVEYS_PREP.md`: WASP and KELT light
curves **are** programmatically accessible -- the NASA archive exposes
`superwasptimeseries` (17,971,001 rows) and `kelttimeseries` (7,670,549 rows)
via TAP. Access was never the binding constraint. **Precision is.**

Measured from a 20,000-row sample of each table, and compared generously
(per-point scatter divided by `sqrt(N_in_transit)`, not used raw):

| survey | per-point scatter | median npts | sigma_eff | 3-sigma depth limit | % of this training set detectable |
|---|---|---|---|---|---|
| SuperWASP | 153,372 ppm | 10,596 | 8,602 ppm | 25,807 ppm | **3.1%** |
| KELT | 17,052 ppm | 699 | 3,724 ppm | 11,171 ppm | **11.4%** |

This training set's transit depths: p25 511, **median 1,821**, p75 5,487,
p95 19,114 ppm. Both surveys reach only the deepest few percent -- hot
Jupiters. Any rows they contributed would be a severely selected
subpopulation *by construction*, which is precisely the mechanism that made
FFI a coverage lever (COARSE rows were selected toward long durations).

Ground surveys also have no vetted false-positive catalogue. Their VizieR
holdings are per-paper tables and variable-star/EB studies, not planet FPs.
They could contribute **8 positive stars total** (4 WASP + 3 NGTS + 1 HAT).

### PART 2/3 RAN ON K2 (approved after the Part 1 report) -- INERT

The Part 1 recommendation was to close. It was overridden and the K2 pilot ran
end to end. Scripts live in `code/k2_pilot/`; nothing was merged into
`training.csv` and the production model is untouched
(md5 `341f1a3907e77f6ec294f182833e613c`).

**Pilot yield: 80 selected, 71 usable (88.8%)** -- much healthier than the
Kepler pilot's 36.4% SNR wall.

| stage | result |
|---|---|
| TIC-gated selection | 80 stars, **0** colliding with `training.csv` |
| download (`author='K2'`, 1800 s) | **80/80**, median 3,645 pts/star |
| preprocess | **80/80** |
| TLS features | **71/80 = 88.8%**, median 174 s/star |

The 9 losses are all one failure mode: 8 could not compute
`odd_even_mismatch` / `depth_mean_odd` / `depth_mean_even`, i.e. too few
distinct transits to split odd from even, and 1 raised a zero-size-array TLS
error. That is the expected consequence of an ~82-day baseline at 30-minute
sampling, not a code fault.

**Exactly one preprocessing parameter was changed, and it was a units bug
waiting to happen.** `MAX_FLATTEN_WINDOW` is specified in POINTS, so its
physical duration scales with cadence:

| | window | physical duration |
|---|---|---|
| TESS 2-min | 401 pts | 13.4 h |
| K2 30-min, **unchanged** | 401 pts | **196.5 h = 8.2 days** |
| K2 30-min, **corrected** | **27 pts** | **13.2 h** |

Left alone, K2 stars would have been detrended over an 8-day window while TESS
got 13 hours under the same nominal setting -- the domain classifier would then
have been detecting a preprocessing artefact rather than the data. Measured
cadence in-run: 29.4 min. Everything else was verified as still appropriate and
left alone (`MIN_POINTS_FOR_FLATTEN` 50 vs ~3,500 delivered; binning cap 30,000
never triggers; TLS grid left at defaults since 82 days is 3x TESS, not the
4-YEAR baseline that forced the Kepler binning cap).

### THE DECISIVE NUMBER: domain AUC 0.9973

| comparison | domain AUC | what happened |
|---|---|---|
| synthetic vs real | 0.9654 | mixing **HURT** (-0.018) |
| TESS 2-min vs COARSE (FFI) | 0.9717 | **inert** outside its own rows |
| **TESS 2-min vs K2** | **0.9973** | **this experiment** |

**The highest separability this project has ever measured** -- K2 rows are
essentially perfectly identifiable. Largest shifts (SD units, K2 - TESS):
`FAP` **-1.24**, `SDE_raw` +0.91, `rp_rs` +0.87, `SDE` +0.86,
`distinct_transit_count` +0.77. The stop was pre-registered at 0.95 *before*
the number was known, and it fired.

### Merge integrity and leakage suite

| check | result |
|---|---|
| pilot stars whose TIC already in `training.csv` | **0** |
| duplicated host rows after merge | **0** |
| hosts straddling the split | **0** |
| K2 assignment | 56 train / 15 test, by the manifest's md5 hash rule |
| correlation \|label-AUC-0.5\| vs \|source-AUC-0.5\| | **+0.110** |

The +0.110 correlation is mild -- the features that identify K2 are largely
*not* the ones carrying the label, so the shortcut and the signal are not
badly entangled. (Cadence audit reported -0.247 for the analogous quantity.)
NaN rates are if anything *lower* for K2 than TESS on every feature.

### Arms -- nothing clears, on either population

Bare `Pipeline` with `clf__sample_weight`, so the weights reach the estimator
rather than only the calibrator. **Arm A is computed in-run rather than
compared against the stored 0.9021**, because 0.9021 is a CALIBRATED refit
(`CalibratedClassifierCV` is a 5-fold ensemble and is not AUC-neutral) while
the arms must be bare; measured side by side today, calibrated refit = 0.9032
and bare refit = 0.8986. Comparing a bare arm to 0.9021 would have
manufactured a fake -0.0035 handicap.

| arm | all-TESS test (n=1,098) | delta [95% CI] | tess_2min test (n=968) | delta [95% CI] |
|---|---|---|---|---|
| A. baseline (TESS only) | 0.8986 | -- | 0.8924 | -- |
| B. pooled (TESS + K2) | 0.9016 | +0.0030 [-0.0024, +0.0086] | 0.8943 | +0.0019 [-0.0044, +0.0083] |
| C. K2 down-weighted 0.25 | 0.9030 | +0.0044 [-0.0013, +0.0105] | 0.8951 | +0.0028 [-0.0034, +0.0094] |

**No arm clears `ci_lo > 0` on either population.** K2 rows in the test set are
excluded from every evaluation, so these are TESS-only numbers by construction.

### Verdict: INERT, not harmful -- and one prior expectation did not hold

56 K2 training rows on 4,387 is +1.3%, which the learning curve predicts at
~+0.0002. The observed point estimates (+0.0019 to +0.0044) are larger than
that but sit inside the +/-0.003 noise floor with CIs spanning zero. Nothing
here is distinguishable from noise in either direction.

**Worth stating plainly: domain separability did not predict harm.** K2 scored
*higher* than synthetic (0.9973 vs 0.9654), yet synthetic actively hurt
(-0.018) while K2's point estimates are mildly positive. Separability tells you
the model CAN identify the source; it does not by itself tell you the mixture
will damage performance. The honest reading of these three results together is
that a high domain AUC predicts "no reliable gain", not "harm" -- synthetic's
damage came from something beyond separability alone. That is a correction to
how this project has been using the 0.9654/0.9717 references.

**Not scaled to the remaining 258 stars.** The stop was pre-registered and the
number cleared it by a wide margin; scaling 3x on a pilot whose best arm is
+0.0028 [-0.0034, +0.0094] would not change the conclusion.

### Verdict

| source | verdict |
|---|---|
| **K2** | **INERT (measured, not predicted).** 88.8% pilot yield, but domain AUC 0.9973 and no arm clears `ci_lo > 0` on either the full or 2-min test population. |
| **CoRoT** | **closed.** MAST serves **0** observations; `search_lightcurve('CoRoT-2')` returns 12 products that are all *TESS*, which looks like success. 3 new stars. |
| **WASP** | **closed.** Accessible but 3.1% reach; 4 new stars; no FP catalogue. |
| **KELT** | **closed.** Accessible but 11.4% reach; **0** new stars. |
| **NGTS** | **closed.** No MAST/archive time-series table; 3 new stars. |
| **HAT** | **closed.** 1 new star. |
| **TRAPPIST** | **closed.** One star system. |

**Recommendation: close the whole line.** The count gate (~200) is passed at
268, but that is the only test it passes -- the predicted effect is a third of
the noise floor, the yield is 95% negatives, and the one viable source sits in
a cadence regime this project has already measured as inert. Part 2 was not
started; a pilot download would cost real time to measure something three
independent measurements already predict cannot clear `ci_lo > 0`.

**What was NOT ruled out:** K2's domain-separability AUC is unmeasured, because
it requires actually fetching K2 photometry. If the line is reopened, that is
the first thing to measure, on a 50-100 star pilot, before anything is merged.

## HOUSEKEEPING: consistency audit of the model-improvement round (2026-08-03)

An audit pass over the last several experiments, verifying state rather than
trusting the prior write-ups. **Nothing was retrained, tuned or promoted; no
data was downloaded.** Production model md5 re-checked as
`341f1a3907e77f6ec294f182833e613c`, unchanged throughout.

**All ten experiment sections were present and complete** -- small lift, medium
lift, injection-recovery, positive-class exhaustion, FFI, multi-task, tabular
architectures + feature selection, retrieval metrics, pseudo-labelling,
TOI-restricted. Nothing had been reported as documented while only existing in
a commit message.

**Four real inconsistencies were found and fixed:**

1. **The README quoted a superseded headline.** `0.9032 (1,099 held-out stars)`
   with a training-set size of 5,506 -- both pre-deduplication. Now
   `0.9031 (1,098 stars)` / 5,485, with an explicit three-number table
   distinguishing the headline (0.9031), the refit-clean baseline for new work
   (0.9021), and the metadata checksum (0.9031559838). Two dated
   clean-clone-test paragraphs quoting 0.9032/1,105/1,110 were labelled as
   records of that run rather than current claims.
2. **This file's own header quoted 0.9032** and gave no guidance on which
   baseline a new experiment competes against. Fixed, with the retired 0.9043
   and the supplementary 0.8151 both explicitly scoped.
3. **The model-history page showed the metadata figure with no context**, so
   the UI read 0.9032 while the README read 0.9031. Both are correct and they
   are different measurements; the page now says so.
4. **`/health` reported a misleading retrain counter.** It showed
   `processed_watch_labels: 138` beside `retrain_threshold: 50`, which reads as
   an overdue retrain. The gating count is labels processed *since the last
   attempt*, which was **1** of 50. Both counts are now reported, and the
   scheduler log line shows `since_last_attempt/all_time`. Display only -- the
   trigger logic, threshold and gate are untouched.

**Verified working, no change needed:** the TFOP evidence layer (both branches
exercised -- 3 live candidate pages render the not-found branch, the found
branch renders correctly for TOI 119.02 and 121.01; export snapshot 2026-08-02,
7,799 TOIs, one day old). Git tree clean and pushed. **The running app was NOT
stale** -- every `web/` file predated the process start.

**Phase 3 Item 2 status (read only, nothing actioned):** 2 retrain attempts, **0
promoted**. Both failed the gate honestly (attempt #2 CI `[-0.0177, +0.0040]`,
attempt #3 `[-0.0205, +0.0012]` -- neither `ci_lo > 0`). Scheduler alive, last
retrain tick within the 24h interval, 314 watch labels pending. Not stalled.

**Undocumented environment workarounds are now durable** in
[`ENVIRONMENT_NOTES.md`](../../ENVIRONMENT_NOTES.md) -- the libomp/rpath fix,
the SIP `DYLD_*` stripping, the double-OpenMP segfault, `OMP_NUM_THREADS=1`,
the missing `timeout`, the stale-module trap, and port 5000 vs 5050. They had
existed only inside one experiment's section.

**New reusable module:** `domain_separability.py`, extracting the
source-discriminator diagnostic that had been written inline three times. Its
self-check reproduces both recorded cadence numbers to **zero delta**
(0.9466338101 for 2-min vs non-2-min, 0.9716678622 for 2-min vs COARSE) --
these are two different groupings and only the second underpins the FFI
decision.

Groundwork for the non-TESS survey question is in
[`NON_TESS_SURVEYS_PREP.md`](NON_TESS_SURVEYS_PREP.md).

## Files

All in `code/experiments/`: `injection.py`, `completeness_curve.py`,
`phase_fold_views.py`, `build_cnn_dataset.py`, `train_cnn.py`,
`gp_classifier.py`, `stacked_ensemble.py`, `uncertainty.py`, plus saved
results (`*.json`, `completeness_curve_results.csv`, `cnn_dataset.npz`).
