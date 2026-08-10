# Model architecture experiments -- honest results summary

**DEPLOYED 2026-08-05: the number of record is now 0.9208.**
`models/best_model.joblib` (HistGradientBoosting + sigmoid calibration, **26
features**, md5 `0c996a41a76cc765895d3013830a536b`) replaced the long-standing
0.9031 model (md5 `341f1a3907e77f6ec294f182833e613c`, kept at
`models/versions/best_model_pre_crowding_341f1a39.joblib`).

**This is the first deployed model change in the project's history** -- the
model-improvement track's first gain that actually shipped, after ~31
experiments that did not. The change is two catalog neighbour-crowding features
(`crowd_flux_ratio_max`, `crowd_nearest_arcsec`); see the crowding section below
for the full validation and the galactic-latitude confound analysis that set the
honest expectation at +0.010 to +0.012 rather than the +0.0167 headline.

Measured at real scale on the complete backfilled dataset, old vs new:

| | old (24 feat) | new (26 feat) |
|---|---|---|
| headline test ROC-AUC | 0.9021 refit / 0.9031 artefact | **0.9208** |
| resampled mean (12 bootstraps) | 0.8961 | **0.9128**, delta +0.0166, 12/12 positive, 12/12 clearing |
| nested CV (5 folds) | 0.9222 | **0.9298**, wins 5/5 |
| Brier | 0.0945 | **0.0879** |
| ECE | 0.0441 | **0.0417** |

Deployed by **manual offline swap, not the live promotion gate**: the gate
compares challenger and production on one feature matrix, and a 24-feature model
raises `ValueError` on a 26-column matrix, so it cannot judge a feature-set
change. After the swap both sides use 26 features and the gate works normally
again.

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

### PROPOSED AND REJECTED: "surrogate center-of-light shift" -- READ THIS BEFORE RE-PROPOSING

**Proposed (2026-08-05):** a "surrogate centroid / center-of-light shift"
feature -- use pixel time-series data to compute the photocenter position
in-transit versus out-of-transit and take the difference, as a
background-binary/blend detector, citing LEO-Vetter's centroid test.

**Verdict: DUPLICATE of the experiment closed above. Not implemented.** No
feature was built, nothing was retrained, production was untouched.

**Why it is the same test.** Both use the same data source (TESS Target Pixel
Files via lightkurve), the same phase split into in-transit vs out-of-transit
cadences, the same operation (a flux-weighted centroid, i.e. "center of
light"), and target the same physical mechanism (a blended source producing
the dimming instead of the target). The only difference is the arithmetic:

    BUILT (and closed):  centroid( median_OOT_image - median_IT_image ),
                         compared to the target's catalog position via WCS
    PROPOSED:            centroid(IT image) - centroid(OOT image)

**Why the difference makes it strictly WEAKER, not different.** The difference
image contains only the flux that changed, so its centroid sits on whichever
star actually dimmed, independent of transit depth. The direct photocenter
shift is diluted by every photon that did not change: for fractional aperture
depth `d` and true source offset `r`, the whole-stamp photocenter moves by
about `d*r`, while the difference image reports `r` regardless.

Simulated on an 11x11 TESS-like stamp with the transit deliberately placed on a
contaminant 2.000 px from the target -- the exact blend case both variants
exist to catch (`centroid_variant_sensitivity.py`, reproducible):

| aperture depth | DIRECT photocenter shift | DIFFERENCE-image offset |
|---|---|---|
| 11.54% | 0.20060 px | **1.99946 px** |
| 4.61% | 0.07442 px | **1.99946 px** |
| 1.15% | 0.01795 px | **1.99946 px** |
| 0.23% | **0.00356 px** | **1.99946 px** |

The difference image recovers the true 2 px offset at every depth. The direct
shift scales with depth and collapses toward zero. **Real transits in this
dataset are ~0.01-1% deep**, so the proposed statistic would be measuring a
~0.004 px displacement against pixel-scale measurement noise -- it is the same
test with the signal divided by the transit depth.

Two supporting points. The proposal cited **LEO-Vetter's centroid test**, which
is itself a difference-image offset test -- i.e. the citation points at the
implementation that already exists here. And **coverage would be bounded
identically** (77.6%), since the direct variant needs the same TPF downloads
and the same ephemeris-validation guard; that guard caused 96% of the failures
and rejecting those sectors is correct, not a bug.

### The one genuinely untested variant, considered and deliberately NOT pursued

The closed experiment used **raw displacement magnitude only**. Confirmed by
inspection: `training_centroid_results.csv` and
`training_centroid_results_multisector.csv` contain a single `shift_pixels`
column and no uncertainty. A **noise-normalised significance** version
(displacement divided by a per-star positional uncertainty) is therefore a real
open thread, and it is not a no-op for a tree model -- significance is not a
monotone transform of displacement, so it is genuinely a different feature.

**It was considered and deliberately deprioritised**, for three reasons that
should be weighed before anyone spends compute on it:

- **Weak ceiling.** Raw `shift_pixels` has single-feature AUC 0.536 and the
  retrain landed at +0.0021, CI [-0.0020, +0.0059]. Re-scaling a feature that
  weak is unlikely to clear the 0.0097 detection threshold.
- **Same coverage limit.** It inherits the identical 77.6% ceiling; ~22% of
  stars would still be missing, with the same class-asymmetric missingness
  (31.9% positive vs 16.3% negative).
- **Requires infrastructure that does not exist.** There is no per-star
  centroid uncertainty anywhere in the pipeline. Producing one means either
  bootstrapping the centroid over cadences or propagating per-pixel flux
  errors through `center_of_mass` -- new code plus a re-run across all 5,486
  training stars and both candidate pools.

If it is ever revisited, it is a fresh experiment subject to the standing
rules: 10+ training bootstraps for AUC *and* Brier/ECE, and -- because a
centroid is a position-derived quantity, exactly the exposure class the
spatial-confound audit flagged -- a `|galactic b|` control arm and a
matched-sky-band test.

Script: `centroid_variant_sensitivity.py` (run it to re-derive the table).

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

## POLICY CHANGE: newly-labelled stars now split 50/50 train/test (2026-08-04)

`05_train_models.POST_FREEZE_TEST_FRACTION = 0.5`, raised from the 0.2 that
matched the original split. **The frozen manifest is untouched** -- all 4,392
train and 1,099 test hosts keep their sides forever. This changes only where
FUTURE stars land.

Assignment remains a deterministic md5 of the host name, so a given star lands
on the same side no matter when or how often it is seen. Verified on 4,000
simulated host names: **0.501** routed to test, and repeat lookups agree. That
reproducibility property is what the whole frozen-split design rests on and was
not traded away for balance.

**A 100% -to-test variant was implemented first and reverted at the user's
direction.** It grows resolution fastest but starves the continuous-retraining
pipeline completely -- challengers get no new data, reproduce the incumbent,
and Phase 3 Item 2's purpose is suspended. 50/50 still gives test growth 2.5x
the old rate while keeping the pipeline fed over what is a multi-year label
stream. That trade is the reason for the number.

### Why: the binding constraint moved from training to test

| side | measured |
|---|---|
| training | +262 stars on 4,387 predicts **~+0.0011** AUC from the fitted learning curve -- below the noise floor, unmeasurable |
| test | the smallest effect a 1,098-star test set can certify at `ci_lo>0` is **~0.0097** |

The best real candidate the project has found -- CatBoost, **+0.0080**, positive
on 8/8 resamples on both populations -- sits **below** that threshold and
therefore cannot be promoted no matter how real it is. Test resolution is worth
more right now than an unmeasurable training gain.

### Comparability is preserved explicitly

`m05.frozen_test_mask(df)` returns the original 1,099 manifest test hosts only.
**Every historical figure, 0.9031 above all, must be reported on that mask.**
Verified after the change:

    production on FROZEN test subset : 0.9031241730   <- identical to before
    production on GROWN test set     : 0.9032307938   <- different population

Reporting the grown-set number as if it were the headline would be exactly the
class of error the frozen split was built to prevent. Report both, label which
is which.

### Integrity, verified

| check | result |
|---|---|
| manifest test hosts still all in test | yes |
| manifest train hosts still all in train | yes |
| hosts on both sides | **0** |
| frozen test subset | 1,098 (unchanged) |
| grown test / train | 1,099 / 4,386 |
| `retrain_pipeline` imports and gate logic intact | yes |

**Side effect worth recording:** the one post-manifest star, `TIC_200385493`,
moved train -> test, returning training to 4,386 rows. That is the star whose
presence caused the +/-0.005 baseline instability documented above, and its
removal restores the refit baselines to the cluster values: **bare 0.8986 ->
0.9036**, calibrated **0.9032 -> 0.9021**. Future refit-baseline experiments
should quote 0.9036 / 0.9021, not the 4,387-row figures.

### THE COST at 50/50

The retrain pipeline keeps working -- half of every new batch still reaches
training -- so Phase 3 Item 2 stays exercised rather than suspended. The price
is that test grows half as fast as it would at 100%.

### This does NOT fix today's resolution problem, and the honest arithmetic is slow

Certifying a +0.0080 effect needs roughly **1,880 test stars**. At 50%
allocation:

| label supply | n_test | MDE | certifies +0.0080? |
|---|---|---|---|
| 262 available now | 1,229 | 0.0093 | no |
| + 1 year (~165) | 1,311 | 0.0090 | no |
| + 3 years | 1,476 | 0.0087 | no |
| + 5 years | 1,641 | 0.0083 | no |
| (100% allocation, 262 now) | 1,360 | 0.0089 | no |

**~8 years at 50%**, and even 100% allocation of everything available today
falls short. Neither setting rescues the CatBoost result on any near horizon.

That is worth stating bluntly: **this policy change is directionally correct
and practically marginal.** It makes future evidence slightly stronger; it does
not unlock the +0.0080 finding. Anyone hoping the reallocation would let
CatBoost be promoted should read the table above instead.

The genuine implication is the one the learning curve, the noise-floor audit
and the CatBoost resampling all reached independently: **at this label supply
the classifier cannot be improved provably**, and effort is better spent on the
evidence layers and candidate pipeline than on the model.

`TEST_GROWTH_TARGET = 1900` records where the fraction should be reduced again.
Changing it is a one-line edit; nothing is destroyed by trying either setting.

## Calibration / ensembling sweep -- NEGATIVE, and a textbook case of why stage 2 exists

Prompted by the observation that `CalibratedClassifierCV(cv=5)` was worth
+0.0046 to HGB -- larger than any of the 28 feature experiments -- arriving as
a side effect of a wrapper nobody treated as a modelling choice. Question: is
production's cv=5 leaving anything on the table?

### The mechanism, which is real and worth keeping regardless

**ROC-AUC is invariant under monotone transforms, so a sigmoid cannot move it.**
It is arithmetically impossible for "calibration" to be the source of +0.0046.
`CalibratedClassifierCV(cv=k)` does two things: fits k models on (k-1)/k of the
data each, then averages their calibrated outputs. Only the averaging can
change AUC. **Production has been running a 5-model bagging ensemble since it
shipped, labelled as a calibration wrapper.** A `bag-only` arm (same folds,
raw probabilities averaged, no sigmoid) was added to separate the two.

### Stage 1 -- single-fit landscape (HGB, full clean test)

| arm | AUC | 2-min | Brier |
|---|---|---|---|
| bare | 0.8986 | 0.8924 | 0.1051 |
| sigmoid cv=3 | 0.8968 | 0.8876 | 0.0935 |
| bag-only cv=3 | 0.9061 | 0.8986 | 0.0935 |
| **sigmoid cv=5 [PRODUCTION]** | **0.9032** | **0.8955** | **0.0893** |
| bag-only cv=5 | 0.9053 | 0.8978 | 0.0952 |
| sigmoid cv=10 | 0.9059 | 0.8987 | 0.0882 |
| bag-only cv=10 | 0.9049 | 0.8976 | 0.0970 |
| sigmoid cv=20 | 0.9061 | 0.8990 | 0.0889 |
| bag-only cv=20 | 0.9056 | 0.8978 | 0.0985 |
| isotonic cv=5 | 0.9025 | 0.8944 | 0.0891 |
| isotonic cv=10 | 0.9051 | 0.8982 | 0.0881 |

Seven arms beat production, the best by +0.0029. **None of it survived.**

### Stage 2 -- 8 bootstrap resamples, every arm refit alongside the baseline

| arm | stage 1 delta | **stage 2 mean delta** | sd | positive | clears |
|---|---|---|---|---|---|
| bag-only cv=3 | +0.0029 | **-0.0011** | 0.0014 | 3/8 | 0/8 |
| bag-only cv=5 | +0.0021 | +0.0006 | 0.0015 | 5/8 | 0/8 |
| sigmoid cv=10 | +0.0027 | **+0.0004** | 0.0014 | 5/8 | 0/8 |
| bag-only cv=10 | +0.0016 | +0.0007 | 0.0017 | 6/8 | 1/8 |
| sigmoid cv=20 | +0.0029 | **+0.0001** | 0.0013 | 5/8 | 0/8 |
| bag-only cv=20 † | +0.0024 | **+0.0001** | 0.0013 | 3/8 | 1/8 |
| isotonic cv=10 † | +0.0019 | **+0.0002** | 0.0014 | 4/8 | 0/8 |

† Run separately afterwards -- the survivor list was frozen while stage 1 was
still going, so these two were missed. Same harness, same baseline, same seeds
(1000..1007), and the run reproduced the baseline to **0.8961 vs 0.8960**
(sd 0.0033 both), so the rows are directly comparable rather than merely
similar. Two adaptations were needed: an `isotonic` branch the original harness
lacked (the mechanical reason they could not just be appended), and evaluation
pinned to `frozen_test_mask` because the post-freeze allocation changed to 50/50
between the runs and `split_by_host` now returns 1,099 rows, not 1,098.

**No arm clears `ci_lo > 0`. None is positive on all 8 resamples on both
populations. Every gain collapsed to within +/-0.0007 of zero, and the
best-looking stage-1 arm (bag-only cv=3, +0.0029) reversed sign to -0.0011.**

**The Brier column dissolved too, which matters more than the AUC here.**
`isotonic cv=10`'s reason to exist was its single-fit Brier of **0.0881, the
best in the entire sweep** -- a genuine reason to prefer it even at flat AUC.
Resampled it averages **0.0943, identical to production's 0.0943**. Stage 1
overstates calibration quality by exactly the mechanism it overstates ranking.
`bag-only cv=20` lands at 0.1040, worse than production, consistent with the
other uncalibrated bagging arms.

This is the baseline-instability finding demonstrated **prospectively**. The
re-audit measured sd(delta) ~0.0024 for paired arms and warned that single-fit
differences of that size are not evidence. These differences were +0.0027 to
+0.0029 -- right in that band -- and stage 2 confirmed they were the training
draw, not the method. Had stage 1 been reported as a result, this project would
have shipped a "+0.0029 improvement" that does not exist.

The production baseline's own spread across resamples, **0.8960 with sd
0.0033**, is the same story from the other side.

### What is nonetheless established

- **Production's cv=5 is not obviously suboptimal.** Nothing tested beats it
  reliably. The setting can be left alone with evidence rather than by default.
- **Brier confirms the sigmoid earns its place.** Across resamples, bag-only
  arms sit at ~0.102 versus production's 0.0943, and degrade monotonically with
  fold count (0.1021 -> 0.1023 -> 0.1033). Bagging improves ranking but never
  fixes probability scale, and the more models averaged uncalibrated, the worse
  the scale drifts. `sigmoid cv=10/20` match production's Brier (0.0943/0.0940)
  without beating its AUC.
- **A cv=3 sigmoid is actively harmful, in BOTH families** -- HGB 0.8968 and
  CatBoost 0.9017, each well below its own bare score. With three folds each
  per-fold sigmoid is fit on a third of the data, and averaging a few
  badly-warped curves destroys ranking. Do not lower the fold count.

### A correction to the previous experiment's stated mechanism

That write-up concluded "calibration lifts HGB +0.0046 and costs CatBoost
-0.0024; the wrapper gives CatBoost nothing." **That was specific to cv=5.**
At cv=20 CatBoost's sigmoid arm reaches 0.9120, *above* its bare 0.9113 --
CatBoost does not dislike calibration, it dislikes FIVE-fold calibration.

The verdict is unaffected: CatBoost vs HGB at the better setting is
0.9120 vs 0.9061 = **+0.0059**, essentially the +0.0057 measured at cv=5, which
did not clear. Both families gain about equally from more folds, so the gap
does not move. Only the explanation was wrong, not the conclusion.

### CatBoost arms, RESAMPLED -- the one result in this project that did not collapse

Run afterwards with the SAME 8 resample seeds as the HGB stage 2, so the two
are directly comparable. Baseline is production refit on each resample.

| arm | mean d_full | sd | t | positive | clears | mean d_2min | positive | clears | Brier |
|---|---|---|---|---|---|---|---|---|---|
| CatBoost bare | +0.0057 | 0.0032 | 5.0 | **8/8** | 2/8 | +0.0073 | **8/8** | 2/8 | 0.0976 |
| CatBoost sigmoid cv=5 (like-for-like) | +0.0074 | 0.0036 | 5.9 | **8/8** | 3/8 | +0.0092 | **8/8** | 5/8 | 0.0975 |
| **CatBoost bag-only cv=10** | **+0.0080** | 0.0032 | 7.0 | **8/8** | 4/8 | **+0.0099** | **8/8** | 5/8 | 0.0952 |
| CatBoost sigmoid cv=20 | +0.0078 | 0.0029 | 7.6 | **8/8** | 3/8 | +0.0096 | **8/8** | 5/8 | 0.0972 |

**Every arm is positive on every resample, on both populations.** Contrast the
HGB wrapper arms measured on the identical seeds: means of +0.0001..+0.0007,
3-6/8 positive, one arm reversing sign. The HGB gains were the training draw;
these are not. Per-resample deltas for the best arm span +0.0039 to +0.0144 --
never near zero.

**It still does not clear the bar.** `ci_lo > 0` holds on only 2-5 of 8
resamples. This is not a contradiction, it is precisely the regime the re-audit
predicted: the measured detection threshold on a 1,098-star test set is
**0.0097**, and the best arm's true effect is about **+0.0080**. An effect real
enough to appear in 8 draws out of 8 but smaller than the smallest thing this
test set can certify.

**Verdict: NOT PROMOTED.** The bar is `ci_lo > 0` and it is not met. That is
the correct outcome under the rule, and the rule should not be bent because the
result is finally interesting.

Three things worth separating:

- **Most of the gain is the model family, not the wrapper.** Like-for-like
  (same sigmoid, same cv=5, only the model swapped) is +0.0074 of the +0.0080
  best. Wrapper choice moves ~0.0006.
- **The 2-min-only population shows the effect MORE strongly** (+0.0099 vs
  +0.0080) and clears more often (5/8 vs 4/8) -- the reverse of the FFI
  artefact pattern, and reassuring, since 2-min is the population the model is
  actually deployed on.
- **CatBoost costs calibration quality.** Every arm's Brier is worse than
  production's 0.0943 (best 0.0952, worst 0.0976). Adopting it trades slightly
  worse probabilities for better ranking, which matters because the UI, the
  confidence tiers and the CTOI export all display probabilities.

**What would change the verdict: a bigger test set, not a better model.** At
+0.0080 with test-set sd 0.0043, certifying this needs roughly 1.5-2x the
current 1,098 held-out stars. That is the same conclusion the learning curve
and the noise-floor audit reached independently -- three separate routes now
point at the label supply as the binding constraint.

### CatBoost arms: single-fit stage-1 numbers, superseded above

Stage 2 covered HGB arms only. CatBoost's best single-fit arms (bag-only cv=10
at 0.9124, +0.0092 over production) are **exactly the kind of number stage 2
just dissolved for HGB** and should be treated with the same suspicion until
resampled. What IS separately established is that CatBoost beats HGB as a bare
learner (11/12 and 12/12 positive across resamples, mean +0.0073); what is not
established is which wrapper suits it best.

**Coverage gap now closed.** Two HGB arms that beat production were left out of
the first stage 2 because the survivor list was frozen before stage 1 finished:
`bag-only cv=20` (0.9056) and `isotonic cv=10` (0.9051, best Brier 0.0881).
They have since been resampled on the identical harness and both dissolved --
see the † rows in the stage-2 table above. The prediction that they would not
change the outcome was correct, but it is now measured rather than argued.
**All seven HGB arms that beat production on a single fit have been resampled;
none survives.**

Scripts: `calibration_sweep.py`, `calibration_sweep_validate.py`,
`calibration_sweep_validate_catboost.py`,
`calibration_sweep_validate_remaining.py`; results
`calibration_sweep_results.json`, `calibration_sweep_validate_results.json`,
`calibration_sweep_catboost_results.json`,
`calibration_sweep_remaining_results.json`.

## Dedicated-holdout (prefit) calibration -- NEGATIVE, and the mechanism hypothesis is REFUTED

The sweep above covered `{sigmoid, bag-only} x cv={3,5,10,20}` plus isotonic at
cv={5,10}. Three things were missing, and this round closes them:

1. **dedicated-holdout ("prefit") calibration** -- base model fit on nearly all
   the training data, a disjoint slice used ONLY to fit the calibrator. Never
   tested, and it is the entire mechanism hypothesis.
2. **ECE** -- only Brier had ever been computed, in a project whose headline
   wrapper is a calibrator.
3. **isotonic cv=20** -- a hole in the grid.

### First, a premise that was already superseded

The motivating claim was that calibration "helps HGB (+0.0046) and hurts
CatBoost (-0.0024)", suggesting a family-level asymmetry. **That is specific to
cv=5.** At cv=20 CatBoost's sigmoid arm (0.9120) sits *above* its own bare
score (0.9113). CatBoost does not dislike calibration; it dislikes FIVE-fold
calibration. An asymmetry that reverses with fold count is a fold-count effect,
which already argued against the data-per-fit story before this ran.

Also restated because it governs every verdict below: the **"+/-0.003 noise
floor" is not the bar.** That is the spread of one AUC estimate. The smallest
delta a 1,098-star test set can certify at `ci_lo > 0` was measured at
**0.0097**.

### The design that makes this a test rather than a survey

Cross-fitting confounds two things: each base model sees `(k-1)/k` of the data,
AND `k` models get averaged. Prefit has ONE model and NO averaging. Holdout
fractions were therefore matched to fold counts so each pair differs *only* in
averaging:

| holdout | equivalent | rows per base fit |
|---|---|---|
| 20% | cv=5 | 3,508 |
| 10% | cv=10 | 3,947 |
| 5% | cv=20 | 4,166 |

Training has one row per star (5,485 rows / 5,485 hosts, verified), so a
stratified row split IS a star split -- no star straddles the base/calibration
boundary, and no group machinery is needed. The frozen TEST set is untouched;
the slice is carved out of the 4,386 training rows only.

### RESULT: prefit loses at every matched size, in both families

| family | matched pair (rows/fit) | prefit | cross-fit | gap |
|---|---|---|---|---|
| HGB | holdout 20% vs cv=5 (3,508) | 0.9015 | 0.9021 | **-0.0006** |
| HGB | holdout 10% vs cv=10 (3,947) | 0.9031 | 0.9048 | **-0.0018** |
| HGB | holdout 5% vs cv=20 (4,166) | 0.8998 | 0.9051 | **-0.0053** |
| CatBoost | holdout 20% vs cv=5 (3,508) | 0.9072 | 0.9086 | **-0.0013** |
| CatBoost | holdout 10% vs cv=10 (3,947) | 0.9103 | 0.9115 | **-0.0013** |
| CatBoost | holdout 5% vs cv=20 (4,166) | 0.9112 | 0.9122 | **-0.0010** |

**Six matched pairs, six negative gaps.** The dedicated holdout never preserves
CatBoost's bare-model advantage better than cross-fitting does -- it is worse at
every size, in both families. **"CatBoost needs more training data per fit than
cross-fitting gives it" is refuted directly.**

The HGB column carries the mechanism plainly: the gap *widens* with fold count
(-0.0006 -> -0.0018 -> -0.0053). Prefit gains nothing from a larger base set,
while cross-fitting gains from averaging more models. Data-per-fit is not the
active ingredient. **Averaging is** -- which is what AUC's invariance under
monotone transforms requires, and this is now measured rather than argued.

### Full sweep, single fit (28 arms, frozen test, baseline = production refit here)

HGB:

| arm | rows/fit | models | AUC | 2-min | Brier | ECE | delta | ci_lo | clears |
|---|---|---|---|---|---|---|---|---|---|
| sigmoid cv=20 | 4166 | 20 | 0.9051 | 0.8979 | 0.0893 | 0.0308 | +0.0030 | -0.0005 | no |
| isotonic cv=20 | 4166 | 20 | 0.9049 | 0.8979 | 0.0892 | 0.0234 | +0.0028 | -0.0008 | no |
| sigmoid cv=10 | 3947 | 10 | 0.9048 | 0.8972 | 0.0885 | 0.0193 | +0.0027 | -0.0003 | no |
| isotonic cv=10 | 3947 | 10 | 0.9044 | 0.8965 | 0.0884 | **0.0181** | +0.0023 | -0.0012 | no |
| bare | 4386 | 1 | 0.9036 | 0.8955 | 0.1046 | **0.0926** | +0.0015 | -0.0036 | no |
| sigmoid prefit 10% | 3947 | 1 | 0.9031 | 0.8986 | 0.0909 | 0.0260 | +0.0011 | -0.0054 | no |
| **sigmoid cv=5 [PRODUCTION]** | 3508 | 5 | **0.9021** | 0.8943 | 0.0896 | 0.0393 | -- | -- | -- |
| sigmoid prefit 20% | 3508 | 1 | 0.9015 | 0.8967 | 0.0919 | 0.0282 | -0.0005 | -0.0080 | no |
| isotonic prefit 10% | 3947 | 1 | 0.9005 | 0.8960 | 0.0897 | 0.0192 | -0.0014 | -0.0101 | no |
| isotonic cv=5 | 3508 | 5 | 0.9004 | 0.8918 | 0.0896 | 0.0245 | -0.0018 | -0.0042 | no |
| sigmoid prefit 5% | 4166 | 1 | 0.8998 | 0.8946 | 0.0929 | 0.0373 | -0.0022 | -0.0088 | no |
| isotonic prefit 20% | 3508 | 1 | 0.8988 | 0.8941 | 0.0944 | 0.0301 | -0.0032 | -0.0114 | no |
| sigmoid cv=3 | 2924 | 3 | 0.8961 | 0.8863 | 0.0931 | 0.0326 | -0.0061 | -0.0115 | no |
| isotonic prefit 5% | 4166 | 1 | 0.8876 | 0.8809 | 0.0980 | 0.0466 | -0.0146 | -0.0236 | no |

CatBoost:

| arm | rows/fit | models | AUC | 2-min | Brier | ECE | delta | ci_lo | clears |
|---|---|---|---|---|---|---|---|---|---|
| isotonic cv=20 | 4166 | 20 | 0.9125 | 0.9076 | 0.0877 | 0.0367 | +0.0104 | +0.0028 | yes |
| sigmoid cv=20 | 4166 | 20 | 0.9122 | 0.9075 | 0.0895 | 0.0410 | +0.0101 | +0.0028 | yes |
| sigmoid cv=10 | 3947 | 10 | 0.9115 | 0.9065 | 0.0896 | 0.0410 | +0.0094 | +0.0020 | yes |
| isotonic cv=10 | 3947 | 10 | 0.9115 | 0.9062 | 0.0882 | 0.0294 | +0.0093 | +0.0024 | yes |
| sigmoid prefit 5% | 4166 | 1 | 0.9112 | 0.9057 | 0.0900 | 0.0389 | +0.0091 | +0.0007 | yes |
| bare | 4386 | 1 | 0.9107 | 0.9050 | 0.0938 | 0.0443 | +0.0085 | +0.0004 | yes |
| sigmoid prefit 10% | 3947 | 1 | 0.9103 | 0.9055 | 0.0914 | 0.0351 | +0.0082 | -0.0007 | no |
| sigmoid cv=5 | 3508 | 5 | 0.9086 | 0.9031 | 0.0902 | 0.0306 | +0.0065 | -0.0005 | no |
| isotonic cv=5 | 3508 | 5 | 0.9086 | 0.9034 | 0.0882 | 0.0264 | +0.0064 | -0.0007 | no |
| sigmoid prefit 20% | 3508 | 1 | 0.9072 | 0.9018 | 0.0923 | 0.0419 | +0.0052 | -0.0034 | no |
| isotonic prefit 20% | 3508 | 1 | 0.9058 | 0.9006 | 0.0898 | 0.0244 | +0.0037 | -0.0060 | no |
| sigmoid cv=3 | 2924 | 3 | 0.9024 | 0.8939 | 0.0923 | 0.0434 | +0.0001 | -0.0083 | no |
| isotonic prefit 10% | 3947 | 1 | 0.8967 | 0.8914 | 0.0932 | 0.0317 | -0.0054 | -0.0162 | no |
| isotonic prefit 5% | 4166 | 1 | 0.8964 | 0.8896 | 0.0905 | 0.0250 | -0.0058 | -0.0159 | no |

### ECE: the column nobody was measuring, and it justifies the wrapper

| arm | AUC | Brier | ECE |
|---|---|---|---|
| HGB bare | 0.9036 | 0.1046 | **0.0926** |
| HGB sigmoid cv=5 (production) | 0.9021 | 0.0896 | **0.0393** |
| HGB isotonic cv=10 | 0.9044 | 0.0884 | **0.0181** |

**Production's wrapper cuts ECE by 2.4x versus bare while leaving AUC
statistically unmoved.** Brier hinted at this (0.1046 -> 0.0896) but understated
it badly, because Brier mixes calibration with sharpness and ECE isolates
calibration. This is the strongest evidence on record that the wrapper earns its
place -- and it was invisible across 30+ experiments that tracked AUC and Brier
only. **Track ECE in any future calibration work.**

Bare CatBoost's ECE is 0.0443 against bare HGB's 0.0926: CatBoost is far better
calibrated out of the box, which is a real point in its favour independent of
ranking.

### Isotonic collapses on small calibration slices

At holdout 5% the calibration slice is 220 rows, and HGB isotonic scores
**0.8876 against sigmoid's 0.8998 (-0.0122)** on the same base model. Isotonic
fits a step function whose plateaus tie large blocks of stars together, and ties
destroy ranking information. The effect is monotone in slice size
(5% -> 10% -> 20% = 220 / 439 / 878 rows). Do not pair isotonic with a small
dedicated holdout.

(This is also why the fast AUC below had to average ranks across ties rather
than use ordinal ranks: for precisely these arms, the naive version is wrong.)

### The six single-fit "winners" were substantially the baseline's own draw

All six clearing arms were CatBoost, so all six were resampled.

**A comparison to avoid:** production scores 0.9021 on the clean single fit and
0.8961 (sd 0.0031) as a mean across resamples, but those are not the same
quantity and the gap is not "bad luck". A bootstrap sample of the training rows
contains only ~63% distinct stars, so *every* arm fit on one scores lower --
the resampled baseline is depressed by construction. What resampling corrects
is not a lucky baseline but training-draw luck in the **pair**, since baseline
and arm are refit together on identical rows and only the delta is read.

Resampled over 12 training-data bootstraps
(seeds 1000..1011; the first 8 match the earlier stage-2 seeds), baseline refit
on the same rows each time:

| arm | single fit | **resampled mean** | sd | range | positive | clears | 2-min clears |
|---|---|---|---|---|---|---|---|
| CatBoost sigmoid cv=10 | +0.0094 | **+0.0083** | 0.0028 | +0.0040..+0.0133 | **12/12** | 6/12 | 7/12 |
| CatBoost sigmoid cv=20 | +0.0101 | **+0.0082** | 0.0028 | +0.0038..+0.0141 | **12/12** | 6/12 | 8/12 |
| CatBoost isotonic cv=20 | +0.0104 | **+0.0077** | 0.0030 | +0.0036..+0.0143 | **12/12** | 4/12 | 4/12 |
| CatBoost bare | +0.0085 | **+0.0073** | 0.0034 | +0.0013..+0.0148 | **12/12** | 5/12 | 5/12 |
| CatBoost isotonic cv=10 | +0.0093 | **+0.0071** | 0.0029 | +0.0029..+0.0135 | **12/12** | 4/12 | 4/12 |
| CatBoost sigmoid prefit 5% | +0.0091 | **+0.0053** | 0.0037 | -0.0001..+0.0111 | 11/12 | 2/12 | 3/12 |

**No arm clears on more than 6 of 12. None is robust.**

**And the calibration cost is not marginal -- it is the one thing here that IS
consistent.** Across the same 12 resamples, production averages Brier 0.0945 /
**ECE 0.0441**, and every CatBoost arm is worse on both:

| arm | Brier | vs prod | **ECE** | **vs prod** |
|---|---|---|---|---|
| CatBoost isotonic cv=10 | 0.0952 | +0.0007 | 0.0541 | **+0.0100** |
| CatBoost isotonic cv=20 | 0.0955 | +0.0010 | 0.0561 | **+0.0120** |
| CatBoost bare | 0.0970 | +0.0026 | 0.0529 | **+0.0088** |
| CatBoost sigmoid cv=20 | 0.0972 | +0.0027 | 0.0577 | **+0.0136** |
| CatBoost sigmoid cv=10 | 0.0972 | +0.0028 | 0.0586 | **+0.0145** |
| CatBoost sigmoid prefit 5% | 0.1003 | +0.0059 | 0.0610 | **+0.0169** |

That is a **20-38% relative worsening of calibration error**, in the same
direction for all six arms on all 12 draws. So the trade is not "a possible
+0.008 in ranking for free": it is an *uncertifiable* ranking gain against a
*consistent* calibration loss, in an application whose UI, confidence tiers,
CTOI export and conformal layer all consume the probability itself. Ranking the
arms by ECE also reverses the AUC order -- `sigmoid cv=10` is the best on AUC
and the second-worst on ECE.

Three readings worth separating:

- **This reproduces the earlier CatBoost measurement.** `sigmoid cv=20` lands at
  +0.0082 here; the earlier independent stage 2 put the best CatBoost arm at
  +0.0080. Different arm set, partly different seeds, same answer. The CatBoost
  advantage is real and stable at **~+0.008** -- and still below the **0.0097**
  this test set can certify. Consistent measurement, not a new result.
- **The prefit arm is the weakest of the six** (+0.0053, and the only one to go
  negative on any resample), despite looking competitive at +0.0091 on the
  single fit. Independent confirmation that averaging, not the holdout, is the
  active ingredient.
- **Isotonic trails sigmoid at matched folds** on resampling (+0.0077 vs +0.0082
  at cv=20; +0.0071 vs +0.0083 at cv=10), consistent with the tie mechanism.

### An unplanned corroboration of the baseline-instability finding

HGB bare scores **0.9036** here but **0.8986** in the earlier sweep -- a
**+0.0050** shift, from `TIC_200385493` moving train -> test in the 50/50
allocation change. Production's own arm moved **-0.0011** (0.9032 -> 0.9021) on
that same single row.

**One row, two arms, opposite directions.** That is a direct demonstration of
why paired comparison *amplifies* baseline instability here instead of
cancelling it -- arrived at independently of the audit that first measured it.

### NESTED CV: the ranking is NOT a test-set artifact

Resampling the training rows tests sensitivity to the training draw. It cannot
test whether the 1,098 held-out stars are simply a panel that happens to suit
CatBoost. Nested CV is the complement: 5 outer folds rotate *which* stars are
evaluated, with each arm's own calibration CV fit strictly inside the outer
training fold. Run on the **training rows only** -- the frozen test set was not
read, because its value is that it has not been optimised against.

| arm | f0 | f1 | f2 | f3 | f4 | mean | sd |
|---|---|---|---|---|---|---|---|
| HGB sigmoid cv=5 (production) | 0.9368 | 0.9173 | 0.9308 | 0.9063 | 0.9199 | 0.9222 | 0.0107 |
| **CatBoost sigmoid cv=20** | 0.9410 | 0.9222 | 0.9332 | 0.9137 | 0.9212 | **0.9263** | 0.0096 |
| CatBoost bare | 0.9399 | 0.9202 | 0.9310 | 0.9119 | 0.9187 | 0.9243 | 0.0099 |
| HGB bare | 0.9353 | 0.9118 | 0.9332 | 0.9093 | 0.9181 | 0.9215 | 0.0108 |

Pooled out-of-fold -- every training star predicted exactly once by a model that
never saw it (n = 4,386 full, 3,884 2-min):

| arm | AUC | Brier | ECE | delta | ci_lo | ci_hi | clears | AUC 2-min | d_2min | ci_lo 2-min | clears |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HGB sigmoid cv=5 (production) | 0.9220 | 0.0838 | **0.0208** | -- | -- | -- | -- | 0.9211 | -- | -- | -- |
| **CatBoost sigmoid cv=20** | 0.9262 | 0.0830 | 0.0335 | **+0.0042** | **+0.0012** | +0.0074 | **YES** | 0.9249 | **+0.0038** | **+0.0003** | **yes** |
| CatBoost bare | 0.9240 | 0.0872 | 0.0408 | +0.0020 | -0.0014 | +0.0056 | no | 0.9225 | +0.0014 | -0.0024 | no |
| HGB bare | 0.9215 | 0.0945 | **0.0749** | -0.0005 | -0.0027 | +0.0017 | no | 0.9203 | -0.0007 | -0.0033 | no |

**CatBoost sigmoid cv=20 beats production on 5 of 5 outer folds and clears
`ci_lo > 0` on the pooled predictions, on both populations.** The ordering is a
property of the models, not of the frozen test set.

**But the 2-min result barely clears (`ci_lo = +0.0003`) and, importantly, the
2-min effect is SMALLER than the full-population one here (+0.0038 vs +0.0042)
-- the opposite of the frozen-test pattern, where 2-min showed the effect more
strongly (+0.0095 vs +0.0082).** The population that helps CatBoost most is not
consistent across evaluations, which is a reason to distrust reading much into
either subgroup split at these effect sizes.

Three cautions on reading this:

- **The absolute AUCs (~0.92) are not comparable to the ~0.90 headline.** This
  evaluates training-distribution stars under 5-fold rotation; it is a *ranking*
  check, never a performance estimate. It does not restate 0.9031.
- **It clears because n is 4x larger, not because the effect is bigger.** The
  effect here is +0.0042, roughly *half* the +0.0082 measured on the frozen
  test. 4,386 pooled points simply certify a smaller delta than 1,098 can. This
  is the same "label supply is the binding constraint" conclusion from a fourth
  independent direction.
- **It confirms the wrapper's calibration value again.** HGB bare's pooled ECE
  is 0.0749 against production's 0.0208 -- a 3.6x difference, matching the
  frozen-test finding. Note CatBoost's ECE (0.0335) is *worse* than
  production's here, so the ranking gain would cost probability quality.

### VERDICT: not promoted, and the reason is the pre-registered rule

Collecting three independent lines of evidence on CatBoost sigmoid cv=20:

| evaluation | delta | verdict |
|---|---|---|
| frozen test, single fit | +0.0101 | clears `ci_lo > 0` |
| frozen test, 12 training bootstraps | +0.0082 | positive 12/12, clears only **6/12** |
| nested CV, pooled out-of-fold | +0.0042 | clears `ci_lo > 0`, 5/5 folds |
| **calibration (ECE), 12 bootstraps** | **+0.0136 worse** | **consistent on 12/12** |

**The effect is almost certainly real.** It is positive in every one of 12
training draws, positive on all 5 rotated star panels, and reproduces an earlier
independent estimate (+0.0080). Nothing else in this project has that record.

**It still does not meet the bar, and the bar is not bent.** The standard is
`ci_lo > 0` on the frozen test, robust across resamples -- and it holds on 6 of
12, against the >=90% requirement. Both the single-fit pass and the nested-CV
pass come from evaluations that are either one lucky pairing or a different
population. **NOT PROMOTED.**

**The decisive asymmetry is that the benefit is uncertain and the cost is
not.** The ranking gain cannot be certified on any evaluation this project
trusts for promotion; the calibration loss (+0.0136 ECE on the like-for-like
arm) appears on 12 of 12 resamples in the same direction. Trading a measured,
consistent degradation in the quantity the product actually displays for an
unmeasurable improvement in one it does not is a bad trade regardless of how
the AUC argument resolves.

### What promoting it would involve (for the record -- not done, not recommended yet)

Should the decision ever be revisited, the cost is not just a model swap:

1. **A new production dependency.** `catboost` is currently experiment-only.
   Adding it to `requirements.txt` puts a ~100 MB native wheel on the clean-clone
   reproduction path, which is presently HGB + sklearn only.
2. **Worse probabilities, in a UI that displays them.** CatBoost's ECE is 0.0335
   pooled vs production's 0.0208, and its frozen-test Brier is worse across
   every arm. The confidence tiers, the candidate pages and the CTOI export all
   surface probabilities, and the conformal layer is calibrated against the
   current model's score distribution -- it would need regenerating
   (`models/conformal_calibration.json` records `model_md5`, so a swap
   invalidates it by design).
3. **Retraining cost.** cv=20 is 20 CatBoost fits, ~780 s single-threaded here
   versus ~200 s for production's cv=5, inside a scheduler-triggered retrain.
4. **It would not pass the live gate anyway.** The promotion gate applies the
   same `ci_lo > 0` test the resampling just failed 6/12, so this would have to
   be a deliberate manual offline swap that bypasses the gate -- which is
   exactly the kind of exception the gate exists to prevent.

The honest recommendation is unchanged: **the classifier is done at ~0.903
pending more labels.** Four independent routes -- the learning curve, the
noise-floor audit, the CatBoost resampling, and now nested CV -- all point at
test-set size as the binding constraint rather than at model choice. Certifying
a +0.008 effect needs roughly 1.5-2x the current 1,098 held-out stars.

Scripts: `calibration_holdout_sweep.py` (28-arm sweep, ECE, mechanism test),
`calibration_holdout_resample.py` (12 bootstraps), `calibration_nested_cv.py`;
results `calibration_holdout_results.json`,
`calibration_holdout_resample_results.json`,
`calibration_nested_cv_results.json`.

Two methodology notes worth carrying forward:

- **`cv="prefit"` was removed in scikit-learn 1.6** (this env runs 1.9). The
  supported path is `CalibratedClassifierCV(FrozenEstimator(base), method=...)`
  fit on the holdout slice.
- **`roc_auc_score` costs ~25 ms/call at n=1098**, almost all input validation,
  which put this sweep's bootstrap at 47 minutes on its own. Replaced by an
  exact rank-sum `fast_auc` with averaged ranks for ties (~18x faster, verified
  to 1e-12 over 400 tie-heavy cases). Ties matter here specifically because
  isotonic emits plateaus of identical probabilities. See ENVIRONMENT_NOTES §9.

## Catalog crowding features -- THE FIRST ROBUST CLEAR, and a dataset-level confound found while checking it

Catalog-based neighbour-star crowding from the TIC (which already cross-matches
Gaia and 2MASS). **Distinct from the closed pixel-centroid work**: that asked
whether the photocentre shifts during transit, from TESS images; this asks how
many catalogued stars sit near the target and how bright they are, using no
pixel data at all. Their measured correlation is **-0.084** (see below), so they
are near-independent probes of the same physical worry.

### Part 1 -- coverage: dense, not sparse

**5,452 / 5,485 stars resolved (99.4%)**; the 33 failures are all "nearest TIC
source > 15 arcsec", i.e. genuinely unresolvable, not errors.

| radius | >=1 neighbour | median | p90 | max |
|---|---|---|---|---|
| 21" (1 TESS px) | 82.4% | 3 | 12 | 131 |
| 42" (2 TESS px) | **96.1%** | 12 | 48 | 499 |
| 63" (3 TESS px) | 99.5% | 26 | 105 | 1033 |

The expectation going in was a rare-but-informative signal. **The opposite is
true**: at the aperture scale that matters essentially every TESS target is
crowded, and the feature is about degree, not presence.

Aperture: 42" (2 px). A TESS pixel is ~21", SPOC apertures run 1-4 px, and
contaminating flux is dominated by the inner 1-2. One 63" query per star serves
all radii, so re-deriving at another radius needs no re-query. 0.34 s/star,
~18 min for the full set.

### Part 2 -- TIC provides contamination natively, and it is unusable here

`contratio` (flux contamination ratio) and `numcont` come back from the same
batch-by-ID call the pipeline already makes for st_rad/st_teff. No separate
Gaia/2MASS query is needed. **But they fail two independent checks:**

| check | result |
|---|---|
| availability, planets vs false positives | 54.2% vs **78.7%** |
| single-feature AUC of *mere availability* | **0.3775** |
| availability on the unknown pool | **37.5%** |

TIC populates `contratio` only for **Candidate Target List** stars, and TOIs are
in the CTL *by construction*. So its missingness encodes **label provenance**,
and HistGradientBoosting handles NaN natively -- it would learn exactly that.
Worse, a feature present for 78.7% of training negatives but 37.5% of real
candidates is a train/serve mismatch on top of the leak. **Reported, not used.**

The two features actually carried forward are computed from the cone search and
are clean: availability gaps of 0.8 / 1.1 points, availability-AUC 0.496 /
0.495, **100% present on unknown candidates**, and max |r| against the existing
24 features of only **0.204** -- no redundancy anywhere near the 0.80 threshold.

### Part 4 -- resampled result: it clears, decisively

12 training bootstraps, production refit on identical rows, nothing from a
single fit. Baseline 0.8961 (sd 0.0031), Brier 0.0945, ECE 0.0441.

| arm | mean delta | sd | range | positive | clears | >= MDE 0.0097 | delta 2-min | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| **+2 full-coverage** | **+0.0167** | 0.0019 | +0.0138..+0.0201 | 12/12 | **12/12** | **12/12** | +0.0160 | **0.0879** | **0.0417** |
| +4 incl. TIC native | +0.0197 | 0.0017 | +0.0161..+0.0223 | 12/12 | 12/12 | 12/12 | +0.0193 | 0.0881 | 0.0411 |

**This is the first robust clear in ~31 experiments.** It is ~1.7x the detection
threshold, positive on every resample, and -- unlike CatBoost -- it *improves*
calibration rather than trading it away (Brier -0.0066, ECE -0.0024).

### THE CONTROL THAT COMPLICATES IT: sky position alone also clears

Crowding tracks stellar density, which tracks galactic latitude, and the two
classes are not drawn from the same sky:

    |galactic latitude| alone   AUC 0.6287   <-- HIGHER than the feature itself
    right ascension alone       AUC 0.6032
    crowd_nearest_arcsec        AUC 0.6039

    median |b|:               planets 16.8 deg, false positives 12.5 deg
    median nearest neighbour: planets 10.3",   false positives  7.4"

Same 12 resamples, adding `|galactic b|` as a control feature:

| arm | mean delta | clears | >= MDE |
|---|---|---|---|
| +2 crowding | +0.0167 | 12/12 | 12/12 |
| **+\|b\| only (control)** | **+0.0135** | **12/12** | **12/12** |
| +\|b\| + crowding | +0.0255 | 12/12 | 12/12 |

**Raw sky position alone buys +0.0135 and clears on every resample.** Sky
coordinates cannot cause a transit signal to be planetary; that gain is purely
an artifact of how the training set was assembled -- confirmed planets from
well-studied high-latitude fields, TOI false positives from all-sky TESS
detections concentrated toward the plane.

**This is a finding about the dataset, not about crowding, and it is arguably
the more important half of this experiment.** It sets a floor of ~+0.0135 on
what *any* position-correlated feature can appear to earn here without meaning
anything, and it means other features correlated with sky position may already
be flattered by the same effect.

**Crowding survives the control, but at a discount:**

    crowding beyond position:  +0.0255 - 0.0135 = +0.0120   (still above MDE)
    position beyond crowding:  +0.0255 - 0.0167 = +0.0088   (below MDE)

So crowding is **not** merely a spatial proxy -- it carries ~+0.0120 of
independently certifiable information, and it subsumes position better than
position subsumes it. But **+0.0120 is the defensible estimate, not +0.0167**,
and even that is measured inside a spatially confounded sample: controlling for
|b| removes the latitude axis, not every route by which survey selection could
enter.

### Relationship to the closed pixel-centroid result

Both proxy the same physical concern (blended sources). On the 2,693 stars with
both measured:

| | value |
|---|---|
| corr(shift_pixels, crowd_nearest_arcsec) | **-0.084** |
| corr(shift_pixels, crowd_flux_ratio_max) | -0.009 |
| single-feature AUC, shift_pixels | 0.5328 |
| single-feature AUC, crowd_nearest_arcsec | 0.6355 |

**They are nearly independent.** The catalog proxy does not replace the pixel
one and does not explain why it failed; a faint contaminant unresolvable in
TESS pixels still appears in Gaia, and a centroid shift can come from a source
the catalog missed. The centroid result stands as recorded (+0.0032, CI crossed
zero, 77.6% coverage). This is a different measurement that happens to work
better, not a re-run of it.

### FINALISED: the confound-adjusted decision (supersedes the verdict below)

Four further runs were done to settle what crowding is actually worth. Full
metrics, 12 bootstraps each, production refit on identical rows.

| arm | mean delta | sd | clears | >=MDE | Brier | ECE |
|---|---|---|---|---|---|---|
| +2 crowding (candidate) | +0.0167 | 0.0019 | 12/12 | 12/12 | 0.0879 | 0.0417 |
| +\|b\| only (control) | +0.0135 | 0.0023 | 12/12 | 12/12 | 0.0879 | 0.0402 |
| +\|b\| + crowding | +0.0255 | 0.0014 | 12/12 | 12/12 | 0.0840 | 0.0375 |
| +\|eclat\| + crowding | **+0.0306** | 0.0023 | 12/12 | 12/12 | 0.0806 | 0.0347 |

**A methodological correction, stated first because it invalidates an earlier
claim in this file.** `|ecliptic latitude|` has a standalone AUC of **0.4920** --
no marginal signal whatsoever -- and on that basis it was previously written
here that TESS observing geometry is not a confound. **That inference was
wrong.** Adding `|eclat|` on top of crowding produces **+0.0306**, the largest
arm in the experiment. A variable with no marginal predictive power can still be
a powerful *conditioning* variable: ecliptic latitude sets how many TESS sectors
a star receives, which sets baseline length, transit count and SNR, which tells
the model how to read every other feature.

**Correlation and marginal AUC therefore cannot rule out a confound.** Any
retrospective clearance in this file that rests on those two diagnostics alone
(including parts of the audit below) is weaker evidence than it appears.

**Neither position variable is a promotion candidate**, despite posting the
largest numbers, because neither transfers to the pool actually scored:

| | planets | false positives | UNKNOWN pool | KS vs planets |
|---|---|---|---|---|
| median \|b\| | 16.9 deg | 12.5 deg | **32.4 deg** | D=0.26, p=3e-17 |
| median \|eclat\| | 59.2 deg | 52.0 deg | **39.7 deg** | **D=0.51, p=1e-67** |

A model taught "higher \|b\| -> more likely planet" would apply it to candidates
drawn from systematically different sky. Test-set AUC rises, production
behaviour degrades. These arms measure the confound; they are not proposals.

### The test that decides it: crowding inside a MATCHED sky region

Modelling `|b|` removes the latitude axis but not the fact that the classes were
drawn from different sky. So instead of modelling the difference, delete it:
restrict both classes to `|b|` in [8, 40] deg.

    band retains 3,510/5,485 stars (64%)  -- train 2,797, test 713
    median |b| INSIDE band: planets 15.9 deg, false positives 19.7 deg
                            (was 16.9 vs 12.5 -- the class difference INVERTS)
    detection threshold rescaled for n=713: ~0.0120

| | value |
|---|---|
| crowding delta inside the band | **+0.0097** (sd 0.0027, range +0.0028..+0.0141) |
| positive | **12/12** |
| clears (against the raised 0.0120 threshold) | 7/12 |
| fraction of the +0.0120 estimate retained | **80%** |

**The effect survives with the sky-region difference inverted.** That is the
strongest available evidence that the mechanism is physical -- a crowded field
really does produce blended false positives -- rather than an artifact of which
survey found the star.

### RECOMMENDATION: GO, at +0.010 to +0.012, not +0.0167

Three independent estimates converge: +0.0167 raw, **+0.0120 beyond modelled
position**, **+0.0097 inside a matched sky region**. The honest expected value is
**~+0.010-0.012**, at or just above the 0.0097 threshold -- roughly a third
smaller than the headline. It is positive on 12/12 resamples in every framing
tested, improves Brier (-0.0066) and ECE (-0.0024), needs no new dependency, is
100% available on unknown candidates from the TIC key alone, and is
near-orthogonal to the existing 24 (max |r| 0.204).

**The decision is yours; nothing has been promoted.** What it would involve:

1. **`06_download_unknown.py`** gains the cone search (~0.34 s/star, threaded,
   resumable) so candidates get the features at scoring time.
2. **`training.csv` must gain `crowd_flux_ratio_max` / `crowd_nearest_arcsec`
   BEFORE `FEATURE_COLUMNS` is edited.** `build_feature_matrix` raises
   `SystemExit` on missing columns, so editing the feature list first would
   crash the next scheduled retrain. Values already exist in
   `crowding_per_star.csv` for all 5,485 stars.
3. **This is NOT automatically a manual offline decision -- correcting an
   earlier assumption.** `web/retrain_pipeline.py` reads `m05.FEATURE_COLUMNS`
   and calls `m05.build_feature_matrix` directly, so once the feature list
   changes the **scheduler picks it up on its next retrain tick** and the
   challenger would be fitted on the new feature set while production still
   uses the old. That silently breaks the gate's own stated invariant ("the
   same model, trained on more data") and would likely auto-promote, since
   +0.012 clears `ci_lo > 0`. **Sequence deliberately: backfill training.csv,
   pause or verify the retrain tick, edit `FEATURE_COLUMNS`, retrain offline,
   inspect, then resume.**
4. **Regenerate `models/conformal_calibration.json`** -- it records `model_md5`
   and is invalidated by design when the model changes.

### Verdict (original, superseded by the finalised decision above)

**NOT PROMOTED** -- per the standing instruction, a robust clear is reported,
not deployed. What promoting it would involve:

- **Pipeline work, not just a model swap.** `06_download_unknown.py` would need
  the cone search wired in (~0.34 s/star, resumable) so candidates get the
  features at scoring time. The features themselves are confirmed available:
  100% of a 40-star unknown sample resolved from the `TIC_<id>` key alone, with
  no ra/dec and no label.
- **A decision about the confound.** The honest expected gain is the +0.0120
  that survives the position control, not the +0.0167 headline -- and the
  spatial confound should probably be investigated on its own terms first,
  since it affects how every other feature in this set should be read.
- **No new dependency and no calibration cost** -- astroquery is already a
  dependency, and both Brier and ECE improve.
- **The frozen split, the gate, the scheduler and the model are untouched.**

Scripts: `crowding_features.py` (fetch), `crowding_checks.py` (coverage,
leakage, production availability), `crowding_validate.py` (12 resamples),
`crowding_control.py` (sky-position control); data `crowding_per_star.csv`;
results `crowding_checks.json`, `crowding_validate_results.json`,
`crowding_control_results.json`.

## RETROSPECTIVE SPATIAL-CONFOUND AUDIT

Prompted by the crowding work, where `|galactic b|` alone produced a fully
clearing +0.0135 that means nothing astrophysically. Any earlier result resting
on features correlated with position, survey provenance or target selection
could be inflated the same way -- or, for a closed negative, could have had a
real effect masked.

### Diagnostic 1 -- how exposed are the 24 production features?

| feature | \|r\| vs \|b\| | \|r\| vs \|eclat\| |
|---|---|---|
| period_uncertainty | 0.145 | **0.213** |
| rp_rs | **0.192** | 0.130 |
| st_teff | 0.143 | 0.055 |
| distinct_transit_count | 0.130 | 0.140 |
| transit_count | 0.105 | 0.113 |
| secondary_eclipse_depth | 0.018 | 0.045 |
| st_rad | 0.016 | 0.039 |
| depth_duration_ratio | 0.008 | 0.010 |

Nothing exceeds 0.25, against crowding which IS a position-derived quantity.
The transit-shape and depth features cannot be confounded this way.

**But this diagnostic is weaker than it looks** -- see the `|eclat|` correction
above. A feature can be uncorrelated with position and still combine with it.
Low correlation is evidence, not proof.

### Diagnostic 2 -- CatBoost, the one case correlations cannot settle

A model-family change could extract more from weakly position-correlated
features than HGB does. Only refitting answers that. Like-for-like
(HGB+sigmoid+cv=5 vs CatBoost+sigmoid+cv=5), 12 bootstraps:

| comparison | mean | sd | positive | clears |
|---|---|---|---|---|
| CatBoost - HGB, neither sees position | +0.0074 | 0.0029 | 12/12 | 5/12 |
| CatBoost - HGB, BOTH given \|b\| | +0.0055 | 0.0036 | 12/12 | 3/12 |

Shift **-0.0019**, which is **smaller than the per-resample sd** and therefore
not distinguishable from zero. **No verdict revision:** CatBoost's advantage is
not meaningfully positional. If anything the position-controlled estimate
(+0.0055) sits further below the 0.0097 threshold, reinforcing the existing
"real but uncertifiable" reading rather than overturning it.

### What was checked, and what changed

| prior result | exposure | outcome |
|---|---|---|
| CatBoost +0.0080 | model family; weak proxies only | **unchanged** (+0.0074 -> +0.0055 under control, within noise) |
| multi-sector consistency | `|eclat|` drives sector coverage | **no revision**, but see caveat |
| weak secondary eclipse | \|r\| 0.018 / 0.045 | not exposed |
| GBM averaging ensemble | same 24 features | not exposed |
| stellar parameters (st_rad/st_teff) | \|r\| <= 0.143 | not exposed |
| pixel centroid (closed negative) | pixel-domain, not catalog | not exposed; corr with crowding -0.084 |
| **catalog crowding** | position-derived by construction | **REVISED: +0.0167 -> +0.010-0.012** |

**Only one verdict moved, and it is the new one.** No previously closed negative
shows signs of a masked real effect, and no previously reported positive besides
crowding needs revision.

The multi-sector entry carries a real caveat: it was cleared on correlation and
marginal AUC, exactly the reasoning the `|eclat|` result just undermined. It was
a negative result, so a hidden confound would have had to *mask* signal rather
than manufacture it, which is the less likely direction -- but it is not proven
clean and is the first place to look if that line is ever reopened.

### METHODOLOGICAL RULE GOING FORWARD (recommended)

**Yes -- a spatial control should be standard.** Specifically:

1. **Any feature derived from a star catalog, sky position, stellar density, or
   anything correlated with target-selection history gets a `|galactic b|`
   control arm**, reported next to the headline. Cost is one extra arm.
2. **Do not clear a feature on low correlation or marginal AUC alone.**
   `|eclat|` had AUC 0.4920 and still contributed +0.0139 in combination.
   Marginal independence does not imply conditional independence.
3. **Where a control shows exposure, run the matched-stratum test** -- restrict
   both classes to a common band and re-measure. That is what distinguished
   crowding's physical component (+0.0097 retained, 80%) from the artifact.
4. **Check the unknown pool's distribution before adopting any feature that
   correlates with position.** Both `|b|` (KS D=0.26) and `|eclat|` (D=0.51)
   differ sharply between training and the candidates actually scored; a
   relationship learned on one does not transfer to the other.

Scripts: `crowding_final.py`, `crowding_stratified.py`,
`catboost_spatial_control.py`; results `crowding_final_results.json`,
`crowding_stratified_results.json`, `catboost_spatial_control_results.json`.

## UNCERTAINTY QUANTIFICATION (not an accuracy change): split conformal prediction

**Nothing about the model changed.** ROC-AUC is still 0.9031, the artifact md5 is
still `341f1a3907e77f6ec294f182833e613c`, the frozen split is untouched. This
wraps the deployed model's existing outputs with a per-prediction set and a
coverage guarantee. It belongs in this file as infrastructure, not as
experiment #29.

### Calibration vs conformal -- related, different, both wanted

`CalibratedClassifierCV` (already in production) makes probabilities correct
**on average**: of all candidates scored 0.7, roughly 70% are planets. It says
nothing about any individual candidate and carries no guarantee -- it is a
fitted sigmoid that can simply be wrong. Conformal prediction adds a
**finite-sample, distribution-free** guarantee about a per-prediction **output
set**: over exchangeable data the true label lies in the set at least
(1-alpha) of the time, for any model, at any sample size, **without assuming
the probabilities were any good**. They are complementary, and conformal is the
only one of the two that can output "I don't know".

### Calibration source: Option B, and why Option A is invalid here

Option A (carve calibration out of the training split) is **not valid as
stated**: `best_model.joblib` was fit on all 4,387 training stars, so
nonconformity scores computed there are optimistically small, the quantile
comes out too tight, and coverage silently fails. It could be rescued by
refitting on a reduced training set -- but then the wrapper calibrates a
*different* model than the deployed one and the guarantee does not transfer.

**Option B used:** the frozen 1,098-star test set, which the model has never
seen. This does **not** alter the frozen split -- no star moves between train
and test; it is a post-hoc analysis partition for a new measurement, and the
headline 0.9031 is still computed on all 1,098. Coverage was validated on
random stratified halves, repeated **300 times per alpha**, rather than trusting
one partition.

### Empirical coverage -- and the failure plain LAC hides

| method | alpha | target | coverage | cov POS | cov NEG | set size | %ambiguous |
|---|---|---|---|---|---|---|---|
| LAC | 0.10 | 0.90 | 0.9043 | 0.9782 | **0.6278** | 1.049 | 4.9 |
| APS | 0.10 | 0.90 | 0.9420 | 0.9885 | 0.7681 | 1.205 | 20.5 |
| **Mondrian** | 0.10 | 0.90 | 0.9060 | **0.9041** | **0.9129** | 1.249 | 24.9 |
| LAC | 0.05 | 0.95 | 0.9525 | 0.9939 | **0.7974** | 1.225 | 22.5 |
| **Mondrian** | 0.05 | 0.95 | 0.9568 | 0.9547 | **0.9649** | 1.474 | 47.4 |
| LAC | 0.01 | 0.99 | 0.9923 | 1.0000 | 0.9633 | 1.621 | 62.1 |
| **Mondrian** | 0.01 | 0.99 | 0.9926 | 0.9928 | 0.9917 | 1.798 | 79.8 |

**Marginal coverage is essentially exact for every method -- and that is the
trap.** Plain LAC hits its 90% target while covering the negative class only
**62.8%** of the time, because the calibration set is 79% planets and the
marginal guarantee is dominated by them. A user looking at one candidate cares
about the class-conditional number, not the average. **Mondrian
(class-conditional) conformal fixes it: 90.4% / 91.3%.** It is what ships.

Results are materially identical on the 2-min-only subset (Mondrian 0.9051 /
0.9563 / 0.9918), so unlike most findings in this file this one does **not**
differ between the two populations.

### Efficiency: the honest cost

The price of per-class validity is set size. At 95%, Mondrian marks **47.4%** of
stars ambiguous versus LAC's 22.5%. That is not a defect -- those stars
genuinely cannot be separated at that confidence -- but it means the tool says
"I don't know" for roughly half of them. **For this app that is the right
trade.** The output feeds human follow-up decisions where a false "probably a
planet" costs telescope time, and the existing evidence layers (TFOP, centroid,
RV) exist precisely to resolve ambiguity. At 90% only 24.9% are ambiguous, which
is why the UI shows all three levels rather than picking one.

### THE LIMIT THAT MATTERS: the guarantee does not transfer to candidates

Conformal coverage requires **exchangeability** between calibration data and the
scored point. Calibration is labelled TESS stars (79% planets, TOI-sourced); the
app scores unknown candidates. Measured with this project's own
`domain_separability` module:

**domain AUC 0.9763** (calibration stars vs the 308 unknown candidates) --
higher than the synthetic-data 0.9654 and the FFI 0.9717 that closed two
earlier lines. Largest shifts: `FAP` -1.34 SD, `st_rad` +0.55,
`distinct_transit_count` +0.40.

**They are not exchangeable, so the finite-sample guarantee is valid for stars
like the test set and is NOT valid as stated for the candidates the app applies
it to.** On those it is a well-calibrated indication. The UI says exactly that,
including the AUC, rather than promising coverage it cannot deliver. Reporting
"95% guaranteed" on a candidate page would have been the most consequential
error available in this task.

### Empty sets

Structurally impossible at the shipped thresholds: `q_neg + q_pos > 1` at every
alpha, so no probability can fall outside both. The branch is implemented and
labelled anyway (thresholds change if the model does), but the honest
distribution-shift signal for this app is the domain AUC above, not an empty
set. Recorded so nobody later mistakes an unreachable branch for a working
alarm.

### Refresh / maintenance

`models/conformal_calibration.json` records the `model_md5` it was generated
from. **It must be regenerated whenever the production model changes**, because
the thresholds are properties of that specific model's score distribution;
stale thresholds would silently mis-cover. Re-run `conformal_prediction.py`.
The natural hook is immediately after a successful promotion in
`retrain_pipeline.maybe_trigger_retrain` -- **not implemented here, since the
promotion gate is out of scope for this task.** Growth of the frozen test set
also shifts thresholds slightly; regenerating on promotion covers the case that
matters, since nothing is promoted without a retrain.

Scripts: `conformal_prediction.py`; module `web/conformal.py`; artifact
`models/conformal_calibration.json`; results
`conformal_prediction_results.json`.

## RE-AUDIT: do the recorded negatives survive the unstable baseline? YES -- but the noise floor was misquoted

The one-training-row finding raised an obvious worry: 28 negatives were measured
against refit baselines carrying +/-0.005 of arbitrary variation. Were their
confidence intervals too narrow, and did any of them hide a real effect?

**The question is not whether the BASELINE moves -- it is whether the DELTA
does.** Every one of those experiments is paired: baseline and challenger are
fit on the SAME rows. If a dropped row perturbs both arms alike, it cancels in
the difference. That is empirical, so it was measured rather than argued
(`baseline_stability_audit.py`): 13 leave-one-out perturbations, BOTH arms
refit on the identical perturbed rows each time, test set frozen.

### Pairing does not protect. It AMPLIFIES.

| | feature addition (24->26) | model swap (HGB->CatBoost) |
|---|---|---|
| sd(baseline) | 0.0016 | 0.0016 |
| sd(challenger) | 0.0016 | 0.0016 |
| **sd(DELTA)** | **0.0024** | **0.0024** |
| cancellation | **-54%** | **-50%** |
| delta mean | -0.0024 | +0.0077 |
| sign flips | **10/13** | **0/13** |

The delta is *more* variable than either arm alone -- the two respond to a
dropped row in partially anti-correlated ways. The intuition that same-learner,
same-data pairing would cancel this is **wrong**, and it was worth checking
rather than assuming.

Note the two shapes differ in what matters. `sd(delta)` is identical (0.0024),
but the feature addition's mean sits at -0.0024 and flips sign in 10 of 13
draws, while CatBoost's +0.0077 is positive in **13 of 13** -- 3.2 sd from
zero. Same instability, different signal.

**A number reported earlier in this file was a favourable draw.** The
weak-secondary bare arm's **+0.0026** is exactly the unperturbed draw; the
perturbed distribution is centred at **-0.0024** and spans +0.0027 to -0.0054.
It did not clear either way, but it should not be read as a positive point
estimate.

### How much did this actually cost? 14%.

The recorded paired-bootstrap CIs capture test-set resampling only, omitting the
training-draw component entirely. Across all 88 CIs recorded in this project's
result files, the median half-width is 0.0085, implying a test-set sd of 0.0043.
Adding the measured training-draw sd in quadrature:

    test-set sd (from recorded CIs)   0.0043
    training-draw sd (omitted)        0.0024
    combined                          0.0049

    minimum detectable effect (ci_lo > 0, 95%)
      as reported   0.0085
      corrected     0.0097     <- 14% larger

The omitted component is the SMALLER of the two, so the correction is modest.
The track was ~14% less sensitive than advertised, not half as sensitive.

### Verdict: no recorded negative is overturned

The direction of the error protects them. **Adding an omitted variance
component widens intervals; an interval that already spanned zero still spans
zero.** No "does not clear" can become "clears" by admitting more uncertainty.
What this undermines is anything that *did* clear -- and only CatBoost-bare ever
did, already retired as inflated in the section below.

The real cost is **statistical power, not validity**. A genuine +0.005 feature
would probably have been recorded as negative. That is a Type II exposure across
the track, and it slightly weakens how much "27 negatives" should be taken as
proof the feature space is exhausted -- though at a corrected MDE of 0.0097, an
effect large enough to matter operationally would still have been caught.

### The "+/-0.003 noise floor" quoted throughout this project is optimistic

That figure describes run-to-run wobble, not the threshold at which
`ci_lo > 0` becomes achievable. **The real 95% detection threshold on this test
set is ~0.0097** -- more than 3x the quoted floor. Every experiment here was
being asked to produce roughly a +0.010 effect to pass, and the ~+0.003 figure
made near-misses look closer to the bar than they were. That is the more
consequential correction of the two.

**Practical guidance going forward:** the binding constraint is the 1,098-star
test set, not the training procedure. Reducing the training-draw component
(averaging leave-one-out refits, or reporting against the calibrated
configuration, which halves it) buys little while test-set uncertainty
dominates at 0.0043. Materially better resolution requires a larger test set --
i.e. more labelled data -- which is the same conclusion the learning curve
reached by a different route.

Scripts: `baseline_stability_audit.py`; results
`baseline_stability_audit.json`.

## Resolving the CatBoost discrepancy -- it was ONE TRAINING ROW, and that is the real finding

Two results appeared to contradict each other:

    bakeoff_followup.py  seed stress    0/10 cleared, mean delta +0.0013
    gbm_ensemble_control.py  resamples   11/12 positive, mean delta +0.0073

The earlier writeup guessed the difference was the randomisation axis (seeds vs
training draws). **That guess was wrong.** The two scripts differed in three
ways at once, so a factorial was run --
{seed-check params, tuned params} x {seeds, resamples} -- against one freshly
measured baseline. Script `catboost_discrepancy.py`.

### The single-row result

`TIC_200385493` is the one star in `training.csv` that post-dates the frozen
manifest; the label-watch scheduler appended it after the bake-off ran.
Removing it reproduces BOTH recorded numbers **exactly**:

| | with the row (today) | without it | seed check recorded |
|---|---|---|---|
| bare HGB | 0.8986 | **0.9036** | **0.9036** |
| calibrated HGB | 0.9032 | **0.9021** | **0.9021** |

And the arithmetic closes: the old run measured +0.0013 against HGB@0.9036;
the identical params on the identical axis today give +0.0063 against
HGB@0.8986. The difference is 0.0050 -- exactly the baseline shift.

### Full factorial

| CatBoost params | axis | mean delta (full) | sd | positive | clearing |
|---|---|---|---|---|---|
| seed-check (d6, l2 3) | seeds | +0.0063 | 0.0018 | 10/10 | **0/10** |
| seed-check (d6, l2 3) | resamples | +0.0059 | 0.0041 | 9/10 | 1/10 |
| tuned (d8, l2 9) | seeds | +0.0122 | 0.0018 | 10/10 | **10/10** |
| tuned (d8, l2 9) | resamples | +0.0076 | 0.0037 | 9/10 | 4/10 |

| cause | effect on the mean delta |
|---|---|
| CatBoost hyperparameters (d6/l2-3 -> d8/l2-9) | **+0.0059** -- flips 0/10 to 10/10 |
| HGB baseline (one training row) | **-0.0050** |
| **axis (seeds vs resamples)** | **-0.0004 -- essentially nothing** |

**The axis never mattered for the mean.** It matters only for the VARIANCE
(sd 0.0018 on seeds vs 0.0037-0.0041 on resamples), and for a specific reason:
the seeds axis holds the HGB baseline fixed at one arbitrary value, while the
resample axis refits it every draw and therefore *propagates the baseline's own
instability* into the delta. That makes the resample axis the more honest test
-- not because training draws are more fundamental, but because it stops
treating one arbitrary baseline fit as ground truth.

### THE INCIDENTAL FINDING, which matters more than the original question

**The bare HGB refit baseline is unstable to a single training row.** Dropping
one random training row and refitting, 15 draws:

    range  +0.0001 to +0.0068     sd 0.0018     UPWARD in 15 of 15

Systematic, one-directional, and larger than the +/-0.003 noise floor. The
4,387-row fit at 0.8986 is an outlier low; 4,386-row fits cluster near 0.903.
The same 8 draws through the production calibration wrapper:

    CALIBRATED  sd 0.0009, range 0.0029, two-sided (some negative)
    BARE        sd 0.0018, range 0.0067, one-directional

`CalibratedClassifierCV(cv=5)` averages five models and roughly halves the
instability, which is why the calibrated comparison is the trustworthy one --
it is simultaneously the production-correct configuration and the numerically
stable one.

**This does NOT affect the promotion gate.** `retrain_pipeline.py:405` scores
the SAVED production artifact (`prod_model.predict_proba`) rather than refitting
it, so the incumbent's predictions are fixed and immune to this. What it does
affect is **every experiment in this track that refits its own baseline**,
including the ones in this file.

### What this does and does not change

It does **not** change the CatBoost verdict. The calibrated single-fit
comparison -- the production configuration, and the stable one -- gives
+0.0057 [-0.0017, +0.0132] full and +0.0074 [-0.0010, +0.0160] 2-min, and does
not clear. It **does** retire the "bare clears" reading as inflated: that
comparison used an outlier-low incumbent.

It also settles that CatBoost genuinely outperforms HGB *as a bare learner*
(+0.0122, 10/10 clearing on fixed-baseline seeds). The reason that does not
reach production is specific and measured: calibration lifts HGB by +0.0046 and
costs CatBoost -0.0024, closing most of the gap.

**Recommended going forward:** report refit-baseline experiments against the
CALIBRATED configuration, or against a baseline averaged over several
leave-one-out refits. A single bare refit carries +/-0.005 of arbitrary
variation, which is larger than anything this track has been trying to detect.

Scripts: `catboost_discrepancy.py`; results
`catboost_discrepancy_results.json`.

## GBM averaging ensemble -- NEGATIVE as an ensemble; the CatBoost swap it exposed does not clear either

Averaging predicted probabilities across HGB + CatBoost + LightGBM + XGBoost,
all four available (libomp rpath fix intact, no ctypes preload -- see
`ENVIRONMENT_NOTES.md`). Robustness measured on **12 training-data bootstrap
resamples**, the axis established during the pseudo-labelling work, not seeds.

### The headline number looked like the first real win in 28 experiments

| population | mean delta vs production HGB | sd | positive | clearing `ci_lo>0` |
|---|---|---|---|---|
| full clean test | **+0.0077** | 0.0028 | **12/12** | 8/12 |
| 2-min-only | **+0.0096** | 0.0032 | **12/12** | 9/12 |

### It decomposes into something that is not an ensemble result at all

| | mean AUC, 12 resamples |
|---|---|
| ensemble (4 averaged) | 0.9034 |
| **CatBoost ALONE** | **0.9032** |
| HGB (production) | 0.8958 |

**Averaging contributes +0.0002.** The entire gap is "CatBoost scores higher
than HGB", which is consistent with the stacking result (families too
correlated to add independent signal -- hgb/rf test correlation 0.941 there).
So the ensemble is negative, and anything positive here is a MODEL SWAP, a
different proposal with different costs.

### A confound in the experiment's own design, measured rather than assumed

CatBoost/LightGBM/XGBoost each got a `RandomizedSearchCV` pass; HGB used the
deployed config untuned. Tuned challengers vs an untuned incumbent explains
+0.0075 without any family effect. Control: give HGB the identical budget over
the bake-off's own HGB grid, same 12 resamples, same RNG seed.

| | mean AUC, 12 resamples |
|---|---|
| HGB production (untuned) | 0.8958 |
| **HGB tuned** (10 candidates, 3-fold) | **0.8959** -- tuning alone **+0.0001** |
| CatBoost | 0.9032 -- family, over TUNED HGB **+0.0073** |

**Tuning accounts for 1% of the gap, model family for 99%.** Two findings: the
production HGB configuration was already essentially optimal (a fresh search
found nothing better, retroactively validating the original tuning), and
CatBoost's advantage is real rather than an artefact of the setup. The control
was expected to deflate the result and did not.

### The decisive test: the production configuration is CALIBRATED

The promotion gate compares production against a challenger cloned from
production's config -- both wrapped in `CalibratedClassifierCV`. Single
full-data fit, which is what the gate would actually see:

| configuration | population | HGB | CatBoost | delta [95% CI] | clears |
|---|---|---|---|---|---|
| bare | full | 0.8986 | 0.9113 | +0.0126 [+0.0040, +0.0215] | YES* |
| bare | 2-min | 0.8924 | 0.9061 | +0.0137 [+0.0039, +0.0238] | YES* |
| **calibrated** | full | 0.9032 | 0.9089 | +0.0057 [-0.0017, +0.0132] | **no** |
| **calibrated** | 2-min | 0.8955 | 0.9029 | +0.0074 [-0.0010, +0.0160] | **no** |

**Bare clears on both populations; calibrated clears on neither, and
calibrated is what production runs.**

**\* The bare "clears" is inflated and should not be read as a near miss.**
The follow-up investigation (see "Resolving the CatBoost discrepancy" below)
found the bare HGB refit is unstable to a *single* training row: dropping any
one row moves it by +0.0001 to +0.0068 (sd 0.0018), **upward in 15 of 15
draws**. The 4,387-row fit at 0.8986 is an outlier low; 4,386-row fits cluster
near 0.903. So the bare arm compared CatBoost against an unusually weak
incumbent, and against the cluster the bare advantage is roughly +0.008, not
+0.0126. The calibrated wrapper is twice as stable (sd 0.0009, and two-sided
rather than systematically upward), which is why the calibrated row is the one
that decides this -- it is both the production-correct comparison and the
numerically trustworthy one.

**Why, mechanically.** `CalibratedClassifierCV(cv=5)` is itself a 5-fold model
ensemble, and it is not AUC-neutral:

    HGB       bare 0.8986 -> calibrated 0.9032   **+0.0046**
    CatBoost  bare 0.9113 -> calibrated 0.9089   **-0.0024**

The wrapper hands HGB most of what CatBoost provides natively, and gives
CatBoost nothing -- a strong learner gains little from 5-fold averaging. The
production stack has, in effect, already bought this gain by another route.
Brier moves the wrong way too: 0.0893 -> 0.0900.

### Verdict: does not clear. Both experiments negative.

Under the resample distribution the effect is *consistently* positive
(11-12/12) but certifiable in a minority of individual draws (4-5/12), and
under the production calibrated configuration a single full fit does not clear
on either population. This is the closest anything has come in 28 experiments
and it is still a "no" -- "probably real but not certifiable at n=1,098" is a
different statement from "indistinguishable from noise", and neither meets the
bar.

**Operationally, had it cleared** -- recorded because it is counter-intuitive:
CatBoost would have been **cheaper**, not dearer. Calibrated artifact
**10.58 MB vs HGB's 21.86 MB**, and inference on 1,098 rows **0.173 s vs
1.298 s, ~7.5x FASTER**. The real cost is a new production dependency
(`catboost`, plus its runtime) and re-validating the whole downstream stack.

### An unresolved discrepancy, recorded rather than smoothed over

The earlier seed check returned **0/10** for CatBoost; this training-draw check
returns **11/12 and 12/12** positive. They vary different things (fit
stochasticity vs training draw) and the earlier one compared against an HGB
baseline scoring ~0.9036 where the same refit gives 0.8986 today, so they are
not directly comparable. Which is the better estimate of the truth is **not
settled here**, and the conclusion above does not depend on it: the calibrated
single-fit test fails the bar either way.

Tuned 3-fold CV AUC on the train split, all on the same folds: **CatBoost
0.9244, XGBoost 0.9242, LightGBM 0.9234, HGB 0.9221**. Note that the CV
ordering (CatBoost best by +0.0023 over HGB) does survive to held-out data --
the disagreement is only about whether it survives *calibration*, which it does
not.

Scripts: `gbm_ensemble.py`, `gbm_ensemble_control.py`; results
`gbm_ensemble_results.json`, `gbm_ensemble_control_results.json`. Run logs are
gitignored under `code/experiments/*.log`, so the CV AUCs above were backfilled
into the JSON rather than left only in the log -- `tuned_cv_auc` was originally
written as zeros by a bad `zip`, the printed values were always correct, and
the script is fixed for future runs.

## Weak secondary eclipse (noise-normalised) -- NEGATIVE, and informative about why

The published weak-secondary test (Kepler DV, ExoMiner) normalises the
phase-0.5 dip by the LOCAL NOISE. This project had the raw depth
(`secondary_eclipse_depth`, AUC 0.550, shipped with the note "weak, likely
needs a better estimator") and had tested the depth RATIO
(`secondary_eclipse_depth / depth_mean`, part of the engineered-ratio arm that
returned +0.0001). Neither is the published statistic. This is:

```
sigma_eff    = 1.4826 * MAD(out-of-eclipse flux) / sqrt(N_in_window)
significance = secondary_depth / sigma_eff
```

Two method changes from the existing feature, both deliberate: normalise by
noise rather than by primary depth (a 200 ppm dip is meaningless on a noisy
star and decisive on a quiet one -- a depth ratio cannot tell those apart), and
size the window at half the primary duration PER STAR instead of a fixed
0.45-0.55 slab, which is 10% of the orbit and roughly 20x too wide for a 0.5%
duty cycle. **No TLS re-run**: the fold was reconstructed from the stored
`period`/`T0` against the detrended light curves already on disk. Coverage
5,365/5,485 = **97.8%** (112 no ephemeris, 8 insufficient window).

### The raw class rates are real and physically correct

| | n | median significance | % > 3 sigma | % > 5 sigma |
|---|---|---|---|---|
| planets | 4,213 | 0.15 | 5.3% | 2.6% |
| false positives | 1,152 | 0.75 | **18.2%** | **9.6%** |

False positives show **3.4x the rate** of significant secondaries. The feature
is not broken and it is not noise.

### It is also genuinely NEW information, by correlation

| feature | NaN% pos | NaN% neg | single-feature AUC | max abs corr vs existing 24 |
|---|---|---|---|---|
| `sec_significance` | 2.8 | 0.0 | **0.381** | 0.245 (`snr`) |
| `sec_depth_windowed` | 2.8 | 0.0 | 0.435 | 0.478 (`secondary_eclipse_depth`) |

Both are far below the 0.80 redundancy cutoff -- this is not a disguised copy
of an existing column. AUC **0.381 is below 0.5**, meaning high significance
predicts the NEGATIVE class, exactly as eclipsing-binary physics requires.

### And it still does not move the model

| arm | full test | delta [95% CI] | 2-min test | delta [95% CI] |
|---|---|---|---|---|
| bare baseline (24 feat) | 0.8986 | -- | 0.8924 | -- |
| bare + weak secondary (26) | 0.9013 | +0.0026 [-0.0037, +0.0088] | 0.8924 | -0.0000 [-0.0074, +0.0070] |
| calibrated baseline (24 feat) | 0.9032 | -- | 0.8955 | -- |
| calibrated + weak secondary (26) | 0.9004 | -0.0029 [-0.0083, +0.0024] | 0.8917 | -0.0038 [-0.0102, +0.0021] |

Nested CV 0.9224 -> 0.9234. Calibration essentially unchanged (Brier
0.0893 -> 0.0893, ECE 0.0231 -> 0.0212). **No arm clears `ci_lo > 0` on either
population.**

**The bare and calibrated arms disagree in SIGN** (+0.0026 vs -0.0029, a
0.0055 swing). That is larger than the effect being measured, and it is the
cleanest demonstration yet of why the baseline-matching rule matters: had only
the bare arm been reported this would have looked like a modest positive, and
had only the calibrated arm been reported it would have looked like a
regression. Neither is real.

**Why a real signal produces no gain.** This is now the third feature to
follow the same pattern -- `odd_even_significance` (AUC 0.353) and
`duty_cycle` did the same. The information is genuinely present and genuinely
discriminative in isolation, but a gradient-boosted tree ensemble on 24
correlated TLS statistics has already extracted it by other routes. Low
pairwise correlation with any SINGLE existing feature does not mean the
information is unavailable as a COMBINATION of them. Novel-by-correlation is
not the same as novel-to-the-model, and this project has now measured that
distinction three times.

Scripts: `weak_secondary.py`, `weak_secondary_validate.py`; results
`weak_secondary_features.csv`, `weak_secondary_results.json`.

### PROPOSED AND REJECTED: "formalize odd-even + add secondary eclipse SNR" -- BOTH ALREADY EXIST

**Proposed (2026-08-05):** formalise the odd-even depth test into a proper
significance statistic `|d_odd - d_even| / sqrt(sigma_odd^2 + sigma_even^2)`
per LEO-Vetter, and add a secondary-eclipse SNR feature. Stated premise: the
project has only a *raw* odd-even depth comparison
(`depth_mean_odd`/`depth_mean_even`), with no significance normalisation.

**Verdict: BOTH halves already exist. Nothing was built or retrained.** Two of
the stated premises were also wrong and are corrected below.

### Half 1 -- secondary eclipse SNR: ALREADY RUN, and negative

Covered in full by "Weak secondary eclipse (noise-normalised)" above. That
experiment computed exactly the proposed statistic --
`significance = secondary_depth / sigma_eff`, `sigma_eff = 1.4826 * MAD(out-of-
eclipse) / sqrt(N_in_window)` -- at **97.8% coverage**, with the class-rate
check the proposal asks for (false positives show 18.2% above 3 sigma versus
5.3% for planets, a 3.4x rate). Single-feature AUC 0.381, max correlation
against existing features 0.245. **No arm cleared**: bare +0.0026
[-0.0037, +0.0088], calibrated -0.0029 [-0.0083, +0.0024] -- the two arms
disagree in sign by more than the effect. The paired GBM-averaging ensemble
from that same round also ran (`gbm_ensemble_results.json`). Do not re-run.

### Half 2 -- odd-even significance: ALREADY A PRODUCTION FEATURE

The premise that the project holds only raw depths is **false**.
`odd_even_mismatch` is one of the 26 deployed features, and TLS computes it as
the significance statistic, not a raw difference:

```
stats.py:392   depth_mean_odd_std = std(flux_intransit_odd) / sqrt(N_odd)   <- standard ERROR
main.py:394    odd_even_difference = abs(depth_mean_odd - depth_mean_even)
main.py:395    odd_even_std_sum    = depth_mean_odd_std + depth_mean_even_std
main.py:396    odd_even_mismatch   = odd_even_difference / odd_even_std_sum
```

Same numerator as LEO-Vetter, same per-group standard errors. **The only
difference is that TLS sums the errors linearly where LEO-Vetter sums them in
quadrature**, and that difference is bounded:

    OE_quadrature / OE_linear = (1 + r) / sqrt(1 + r^2),  r = sigma_odd/sigma_even
    over r in [0.01, 100]:  min 1.0099, max 1.4142
    at equal errors (the typical case): exactly sqrt(2) = 1.4142

So the proposed statistic is the deployed one multiplied by a factor between
1.00 and 1.41 -- and very nearly a *constant* sqrt(2) whenever the odd and even
groups have comparable scatter, which they generally do since they have similar
N. It is not strictly a monotone transform (the factor varies with
sigma_odd/sigma_even), so a tree model *could* in principle see it differently,
but the room for that is a <=41% rescaling of one feature.

A further engineered variant, `odd_even_mismatch / depth_mean_std`, was ALSO
already tested in the small-lift trio: single-feature AUC 0.353, and the
four-ratio arm containing it returned **+0.0001 [-0.0055, +0.0057]**.

### Two premise corrections

- **`depth_mean_odd` vs `depth_mean_even` are NOT correlated at 0.981.**
  Measured on the current training set: **|r| = 0.162**. The pair is not
  redundant, and no 0.981 figure for this pair appears in the feature audit.
- **The existing odd-even treatment is not "raw depth only".** There are three
  separate columns -- the two group depths AND the significance statistic --
  and the significance one is in production.

### Why a refinement has essentially no headroom

Permutation importance in the **deployed 0.9208 model** (frozen test, 5 repeats,
ROC-AUC scoring):

| feature | importance | rank |
|---|---|---|
| st_rad | +0.04454 | 1 |
| chi2red_min | +0.03558 | 2 |
| st_teff | +0.02781 | 3 |
| crowd_flux_ratio_max | +0.01719 | 4 |
| secondary_eclipse_depth | +0.00663 | 6 |
| **odd_even_mismatch** | **+0.00346** | **13 / 26** |
| depth_mean_even | +0.00192 | 16 |
| depth_mean_odd | +0.00100 | 20 |

Single-feature AUCs: `odd_even_mismatch` 0.4650, `depth_mean_odd` 0.4295,
`depth_mean_even` 0.4340, `secondary_eclipse_depth` 0.4935.

The entire odd/even family contributes about **+0.006** of permutation
importance combined, an order of magnitude below `st_rad`. Re-deriving one of
those columns with a denominator that changes it by at most 41% cannot
plausibly produce the ~+0.010 needed to clear the detection threshold.

### The one genuinely new piece, NOT built: fit-based odd-even

Everything above is *box*-based (TLS's in-transit flux means). A **fitted
transit model** per odd/even group -- fitting a limb-darkened transit to each
group separately and comparing fitted depths -- is genuinely untested. `batman`
is already a dependency (used by the injection-recovery work), so the model
itself is available.

**It was not built, and is not recommended, for three measured reasons:**

- **No headroom.** The box-based version it would refine ranks 13/26 at +0.0035.
- **Heavy new infrastructure**, which this project has repeatedly found not to
  pay: per-group transit fits across 5,486 training stars plus both candidate
  pools, with per-star convergence handling, versus a statistic already
  computed for free inside the existing TLS call.
- **The documented pattern.** `odd_even_significance` (AUC 0.353), `duty_cycle`
  (0.353) and `sec_significance` (0.381) all measured real, physically correct,
  low-correlation signal and all returned ~0.000 in the model. Novel-by-
  correlation is not novel-to-the-model, now measured four times.

If it is ever revisited it is a fresh experiment owing the standing rules:
10+ training bootstraps for AUC *and* Brier/ECE against the **0.9208 / 26-feature**
baseline, and a spatial-confound check (a transit-shape feature is a priori not
exposed, but the rule says confirm).

Detection threshold unchanged: the frozen manifest test set is still 1,098
stars (`frozen_test_mask`); the 50/50 post-freeze allocation has added 2 stars
to `split_by_host`'s test side (1,100), which does not move the ~0.0097 MDE.

Script: none required -- this entry is a code and results audit.

## GIANT-STAR BLIND SPOT -- the premise does not survive re-measurement. ALL THREE FIXES NULL.

Investigated whether the model itself can be improved for large-radius stars,
which are currently handled only by a post-hoc confidence-tier penalty.
**Nothing was promoted; production stays at 0.9208 / 26 features.** The
interesting result is the diagnosis, not the fixes.

### What the post-hoc penalty actually does

`confidence_tier()` in `08_characterize_candidates.py` applies `score -= 1` when
`st_rad >= 1.5 AND SDE >= 10`, and appends a human-readable reason.
**`predicted_probability` is never modified** -- it is only read. So the penalty
is not cosmetic (the candidate list sorts by tier first, so a High -> Medium
demotion moves a candidate materially) but it does not touch the probability,
the conformal sets, or any probability threshold.

### The 3.6x figure is stale, and it never described st_rad alone

The original number came from the JOINT cell: 21.1% error for
`st_rad>=1.5 AND SDE>=10` against 5.8% for neither = 3.64x, measured on the
**24-feature 0.9031 model**. Re-measured on the deployed 0.9208 model:

| cell | n | error % | AUC | planets % |
|---|---|---|---|---|
| neither | 629 | 6.7 | 0.8637 | 92.4 |
| large radius only | 180 | **20.6** | 0.8974 | 57.8 |
| high SDE only | 204 | 16.2 | 0.8990 | 67.6 |
| BOTH (the "blind spot") | 49 | 14.3 | 0.9158 | 44.9 |

**Now 2.14x, not 3.6x -- and the cell structure inverted.** The joint cell the
penalty targets is no longer the worst; "large radius only" is.

### The blind spot is mostly BASE RATE and CALIBRATION, not ranking

| population | n | planets % | AUC | err@0.5 | err@best thr | ECE |
|---|---|---|---|---|---|---|
| dwarfs <1.5 | 833 | 86.3 | **0.9017** | 9.0 | 8.9 | **0.0204** |
| giants >=1.5 | 229 | 55.0 | **0.9013** | 19.2 | 16.2 | **0.0867** |
| subgiants 1.5-3 | 176 | 51.1 | **0.8548** | 23.9 | 19.9 | 0.1044 |
| giants proper >=3 | 53 | 67.9 | **0.9935** | 3.8 | 3.8 | 0.0787 |

Three findings that reframe the problem:

1. **The giant-vs-dwarf AUC gap is -0.0004.** The model ranks giants as well as
   it ranks dwarfs. The error-rate gap is largely the class prior: giants are
   55% planets against 86% for dwarfs, so a FIXED 0.5 threshold necessarily
   misclassifies more of them at identical ranking quality. Re-thresholding
   alone takes 19.2% -> 16.2%.
2. **Giant calibration is 4.2x worse** (ECE 0.0867 vs 0.0204) -- a real defect
   AUC cannot see, and the one the diagnosis actually points at.
3. **"Giant" is the wrong label for the deficit.** The genuine ranking gap is
   confined to **subgiants 1.5-3** (AUC 0.8548, -0.0469 vs dwarfs, 16% of the
   test set). Giants *proper* (>=3) score **0.9935** -- the model's BEST
   population, better than dwarfs.

### Mechanism: how giants differ

Median giant/dwarf ratios on the test set: duration 1.56x, snr 1.76x, period
1.41x, but rp_rs 0.51x, depth_mean_std 0.24x, chi2red_min 0.05x,
secondary_eclipse_depth 0.19x, crowd_flux_ratio_max 0.11x. Longer, higher-SNR,
smoother events on quieter hosts -- the eclipsing-binary corner, exactly as the
penalty's comment says. SDE is identical (1.00x), which is why an SDE-based
bonus cannot separate them.

### THREE FIXES, ALL NULL (12 bootstraps, production recipe, vs 0.9208)

Baseline resampled: AUC 0.9128, giant AUC 0.8944, Brier 0.0879, ECE 0.0417,
giant ECE 0.0956.

| arm | d_overall | sd | pos | clears | >=MDE | **d_GIANT** | pos | clears | Brier | ECE | giantECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A: +st_rad interactions | -0.0005 | 0.0010 | 3/12 | 0/12 | 0/12 | **+0.0011** | 9/12 | 1/12 | 0.0878 | 0.0407 | 0.0954 |
| B: giant upweight x3 | -0.0001 | 0.0002 | 6/12 | 0/12 | 0/12 | +0.0001 | 6/12 | 0/12 | 0.0882 | 0.0433 | 0.0971 |
| C: stratified calibration | -0.0020 | 0.0032 | 2/12 | 0/12 | 0/12 | +0.0008 | 8/12 | 0/12 | 0.0901 | 0.0508 | **0.1094** |

**No arm clears on either the overall or the giant-subpopulation delta.**

- **A** was expected to be weak and is: `st_rad` is already the single most
  important feature in the deployed model (permutation importance +0.04454,
  rank 1/26), so a tree can already split on it directly. An explicit
  indicator or interaction adds nothing a tree cannot already represent.
- **B** is a no-op (-0.0001, sd 0.0002). Distinct from the closed
  class-weighting experiment -- that reweighted by LABEL, this by
  SUBPOPULATION -- but the answer is the same.
- **C is the informative failure.** It was designed to fix the calibration
  defect and made calibration **worse**: giant ECE 0.0956 -> 0.1094 (+0.0138),
  overall ECE 0.0417 -> 0.0508. **Splitting the calibration set costs more in
  variance than specialisation gains in bias**: the per-group Platt scaler sees
  ~900 giant training rows (~570 unique under bootstrap) where the global one
  sees 4,386. This is the same lesson as the `sigmoid cv=3` result -- a
  calibrator fitted on a small slice is a bad calibrator -- now measured on a
  second axis (subpopulation rather than fold count).

### SPATIAL CONFOUND: severe, and concentrated exactly where a fix would apply

Mandatory per the standing rule, and it fires hard:

    AUC of |galactic b| ALONE, within giants : 0.7092   (|0.5-a| = 0.209)
    AUC of |galactic b| ALONE, within dwarfs : 0.5482
    corr(st_rad, |galactic b|)               : +0.016
    KS giants vs dwarfs on |b|: D=0.187, p=3e-29

`st_rad` itself is not position-correlated, but the giant SUBPOPULATION is far
more spatially segregated by class than dwarfs are. Physically consistent:
giants are intrinsically bright, detected to greater distance, sampling a
larger volume toward the galactic plane where EB false positives concentrate.
**So even arm A's +0.0011 giant-subpopulation blip cannot be trusted** -- inside
that subpopulation, position alone is a stronger predictor than anything being
tested. Any future giant-targeted work owes a matched-sky-band test up front,
not as a follow-up.

### Production pool composition -- the practically important number

`st_rad` is 100% present on both ranked candidate pools; **no backfill would be
required.** But the pools are far more giant-heavy than the test set:

| pool | n | giants (st_rad>=1.5) |
|---|---|---|
| frozen test set | 1,098 | 20.9% |
| ranked_candidates | 254 | **49.6%** |
| ranked_candidates (widesector) | 54 | **42.6%** |

Roughly **half of what production actually scores** is in this subpopulation,
a 2.4x enrichment over the test set. That raises the stakes of the null result
rather than lowering them -- and raises the extrapolation risk for any future
correction learned on a spatially-segregated training giant population.

### RECOMMENDATION: do not promote. And the existing penalty needs its numbers refreshed.

No arm earns promotion; none clears on either framing. The post-hoc tier
penalty remains the right instrument, because the defect is a
calibration/threshold effect on a subpopulation with a different class prior --
not a ranking failure the feature set can repair.

**CLOSED (2026-08-05): the stale candidate-facing text is corrected.** The
penalty asserted *"21% error vs 6% elsewhere on held-out data"*; on the deployed
model those cells are 14.3% and 6.7%.

But the refresh found the sentence was **wrong, not merely stale**. It claimed
the classifier is "measurably least reliable in this combination". By AUC the
opposite holds: the targeted cell has the **highest** discrimination of the four
(0.9158) and the "safe" cell the **lowest** (0.8637). The error-rate ordering is
inverted relative to AUC purely by class balance -- 92.4% planets in the safe
cell against 44.9% here. The text now states the real mechanism (a near-50/50
population, so the same probability carries weaker odds) and reports the AUC
next to the error rate so the distinction is visible.

**The triggering condition was deliberately NOT broadened**, despite
`st_rad>=1.5` alone now having the higher raw error rate (20.6%):

1. The `-1` exists to temper the `+2` the SDE bonus (`sde>=10 AND
   n_transits>=5`) just awarded -- it is structurally tied to `sde>=10`. Applied
   to giants that never received that bonus it becomes a different, unjustified
   penalty.
2. That cell's higher error rate is the same base-rate artifact (57.8% planets
   vs 92.4%), and its AUC (0.8974) is BETTER than the safe cell's. Re-aiming on
   error rate alone would repeat the exact reasoning error being corrected.
3. Giants are 49.6% of the ranked candidate pool. Broadening moves the flag from
   18/254 candidates to 126/254, and a flag that fires on half of everything
   stops being a flag.

**Staleness is now structurally prevented.** The figures are read at runtime
from `giant_star_diagnose.json` via `_giant_regime_stats()`, so re-running the
diagnostic refreshes what candidate pages say; a hand-verified fallback plus
`GIANT_STATS_LAST_VERIFIED` covers a missing file. Verified live: it returns
14.2857/6.6773 from the JSON versus the 14.3/6.7 fallback, proving the file is
read rather than silently defaulting.

A latent bug was caught building it: this module does not import `json` at top
level, so the helper would have raised `NameError`, been swallowed by its own
`except`, and pinned the text to the fallback forever -- the identical silent
failure the change exists to prevent. Fixed with a local import.

Scope: tier text and its data source only. `predicted_probability`, the
conformal layer, the model, the gate and the scheduler are untouched, and
`score -= 1` and the trigger are unchanged, so **no candidate changes tier**.

Scripts: `giant_star_diagnose.py`, `giant_star_fix.py`; results
`giant_star_diagnose.json`, `giant_star_fix_results.json`.

## Fitted transit shape + odd-even TIMING -- shape is a REFINEMENT, timing is NOVEL AND EMPTY

Two proposals. Neither reached a model fit, for different and specific reasons.

### transit_shape_ratio is structurally broken -- so a fitted version is a genuine refinement, NOT a duplicate

The existing feature is not a working shape metric. It is a ratio of two
FIXED-PHASE windows that never scale to the transit's duration:

```
center_mask = |phase| < 0.005
edge_mask   = 0.005 <= |phase| < 0.015
shape_ratio = (1 - median(flux[edge])) / (1 - median(flux[center]))
```

Whether those windows land on the flat bottom, the ingress, or entirely outside
the transit is decided by the star's duty cycle, not by its shape. Measured
against the actual half-duration in phase units:

| where the windows fall | share of stars | what the feature returns |
|---|---|---|
| transit fits inside the CENTER window | **36.9%** | edge window samples out-of-transit, ratio -> 0 |
| both windows inside the flat bottom | **18.2%** | both depths ~ full depth, ratio -> 1 |
| windows near ingress/egress | 45.0% | actually measures shape |

**For 55% of stars the value is fixed by geometry, not by the transit profile.**
Coverage is 69.7%, single-feature AUC 0.4333, permutation importance +0.0001.

The initial hypothesis -- that it is a disguised duty-cycle proxy -- was WRONG
and was checked rather than assumed: |corr| with duration/period is 0.018, with
duration 0.016, with period 0.001. It is not a proxy for anything. It is mostly
noise.

**Verdict: a properly duration-scaled trapezoid fit is a refinement with real
headroom, not a duplicate.** The physical quantity (ingress/egress versus flat
bottom) has never actually been measured in this pipeline; the existing column
is a broken implementation of the same idea, so its near-zero importance is
evidence about the implementation, not about whether shape information helps.
**Not built here** -- it needs its own fitting infrastructure and its own
validation cycle, and the timing piece was prioritised as the more novel of the
two.

### Odd-even TIMING offset -- genuinely novel, cleanly built, and the signature is ABSENT

Distinct from everything deployed: `odd_even_mismatch` compares odd/even
DEPTHS, the closed weak-secondary work is depth significance, and no existing
feature uses transit phase POSITION. Its nearest neighbour is
`power_ratio_half_period` (already tested, negative), which asks the same
physical question in the frequency domain.

Built by folding each parity group together and measuring its flux-weighted
phase centroid -- deliberately NOT per-transit mid-times, because this project
already hit a wall there ("threshold-crossing on raw per-cadence photometry
doesn't reliably measure a single transit's duration"). Group folding recovers
the SNR individual events lack. No TLS re-run; the fold comes from stored
period/T0.

    offset_frac = |centroid_even - centroid_odd| / (duration / period)

**Coverage 4,315/5,486 = 78.7%.** Failures are almost entirely too-few-epochs
(a parity group needs >=2 distinct epochs), which is the honest limitation of
asking a per-parity question on short baselines.

Every hygiene check passes -- and the feature still fails, which is the point:

| check | result |
|---|---|
| missingness by class | 78.4% planets vs 79.5% false positives, gap -1.1pp |
| AUC of mere availability | 0.4946 (no leakage) |
| max correlation vs the 26 production features | 0.174 (`depth_duration_ratio`) |
| orthogonality vs `odd_even_mismatch` | **0.064** -- genuinely different information |
| spatial confound, corr vs \|galactic b\| | 0.034 / 0.020 (clean) |
| production pools | computable, 16/20 and 15/20 sampled candidates |

**THE CLASS-RATE GATE IS DECISIVE AND NEGATIVE:**

| group | n | median offset | p90 | % > 0.25 duration | % > 0.5 duration |
|---|---|---|---|---|---|
| planets | 3,399 | 0.0305 | 0.1019 | 0.4% | **0.0%** |
| false positives | 916 | 0.0250 | 0.0804 | 0.2% | **0.0%** |

**Not one star in either class shows a timing offset beyond half a transit
duration**, and the tiny median difference runs the WRONG way -- planets show
slightly larger offsets than false positives, the reverse of the eclipsing-binary
prediction. Single-feature AUC 0.5570 for the offset (wrong sign for the
physics) and 0.5062 for the significance version, i.e. noise.

**The phenomenon the feature was built to detect does not occur in this dataset
at measurable levels.** Physically consistent: depth-asymmetric binaries are
already caught by `odd_even_mismatch`, circular ones put the secondary at
exactly phase 0.5 and produce no offset by construction, so the residual
population is eccentric binaries whose eclipse depths happen to match -- and
that population is empirically empty here.

**STOPPED AT THE GATE, no model fit run.** The brief specifies the class-rate
check as a test to run BEFORE modelling, and it says there is nothing to model:
a feature whose two classes are indistinguishable and whose extreme tail is
empty in both cannot produce the ~0.0097 needed to clear. Resampling exists to
stop a positive being believed too early; declaring a negative does not require
it when the raw class rates are this flat. A 12-resample retrain can be run for
completeness on request.

**RECOMMENDATION: do not promote either piece.** The timing feature is closed as
an empty-signature negative. The fitted-shape refinement remains genuinely open
and is the better of the two to revisit, with the caveat that it inherits this
project's four-times-measured pattern that a real, orthogonal signal can still
add nothing to the model.

Scripts: `oddeven_timing.py`; data `oddeven_timing_features.csv`.

## Expanded time-series statistics -- variance ratio + residual autocorrelation. NEGATIVE.

Three statistics proposed. One is a duplicate and was not rebuilt; two were
genuinely new, were built, passed every pre-model check, and still moved
nothing.

### Part 0: what was already tested, read from the code rather than summarised

The closed medium-lift "phase-folded flux distribution statistics" experiment
produced exactly nine columns, confirmed from its own output file
`flux_distribution_features.csv`:

    in_skew, in_kurt, out_skew, out_kurt, skew_diff, kurt_diff,
    wavelet_e1, wavelet_e2, wavelet_e3

| proposed statistic | verdict | reasoning |
|---|---|---|
| skewness/kurtosis of the phase-folded profile | **DUPLICATE** | exactly `in_/out_skew`, `in_/out_kurt` and their differences, on the phase-folded curve, split in/out of transit. Result was -0.0072, CI [-0.0153, +0.0010], nested CV flat. Not rebuilt. |
| in-transit vs out-of-transit VARIANCE ratio | **GENUINELY NEW** | that work differenced the 3rd and 4th moments but never compared the 2nd. Its `n_in`/`n_out` columns are point counts, not variances. |
| autocorrelation of residuals | **GENUINELY NEW** | repo-wide grep for `autocorr\|acf\|lag_1\|variance_ratio` returns zero matches. A different mathematical object: temporal correlation, not a single-point moment. |

**A premise correction.** The request described the medium-lift round as having
"a control that disqualified an apparent win as an artifact" for
skewness/kurtosis. That belongs to a DIFFERENT experiment: Item 1 (multi-sector
depth consistency) is the one headed "ARTIFACT, NOT PROMOTED". Item 2 (flux
statistics) was measurably negative, and its issue was a bookkeeping bug caught
before the result was trusted -- `build_feature_matrix` selects only
`FEATURE_COLUMNS`, so merged columns never entered X and the run would have
reported a perfectly null comparison that never actually differed. Two distinct
failures; the summary in circulation had merged them.

### The two new features, and the direction predicted BEFORE measuring

```
ts_var_ratio = var(in-transit flux) / var(out-of-transit flux)
ts_acf_lag1, ts_acf_1hr = autocorrelation of (flux - own phase-folded binned
                          profile) at 1 cadence and at ~1 hour
```

Predicted, in writing, before any measurement: **all three should score AUC
below 0.5** (false positives above planets). A flat-bottomed transit leaves
in-transit scatter near the out-of-transit noise and near-white residuals;
grazing eclipses, blends and instrumental systematics inflate both.

**All three predictions held.** Coverage 5,309/5,486 = **96.8%**.

| feature | AUC | \|0.5-AUC\| | direction | max \|r\| vs the 26 | vs \|galactic b\| |
|---|---|---|---|---|---|
| `ts_var_ratio` | 0.4195 | 0.0805 | as predicted | 0.370 (`odd_even_mismatch`) | 0.060 |
| `ts_acf_lag1` | 0.4131 | 0.0869 | as predicted | 0.340 (`snr`) | 0.073 |
| `ts_acf_1hr` | 0.4491 | 0.0509 | as predicted | 0.413 (`duration`) | 0.016 |

Class medians: false positives show higher in-transit variance (1.0518 vs
1.0138) and more correlated residuals (+0.0017 vs -0.0036). Missingness clean
(96.4% planets / 98.1% false positives, availability AUC 0.4917). No spatial
exposure -- checked rather than assumed, per the giant-star lesson. Computable
on both candidate pools, 20/20 and 20/20, checked BEFORE modelling.

This is the strongest pre-model profile of any feature tested recently: real
signal, correct sign, orthogonal, clean, high coverage. It earned a model fit.

### And it still moves nothing (12 bootstraps, production recipe, vs 0.9208)

Baseline resampled: AUC 0.9128 (sd 0.0020), Brier 0.0879, ECE 0.0417.

| arm | features | mean delta | sd | range | positive | clears | >=MDE | delta 2-min | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| A: +var_ratio | 27 | **-0.0015** | 0.0013 | -0.0042..+0.0001 | 1/12 | 0/12 | 0/12 | -0.0016 | 0.0884 | 0.0412 |
| B: +residual ACF | 28 | **-0.0003** | 0.0012 | -0.0027..+0.0015 | 6/12 | 0/12 | 0/12 | -0.0017 | 0.0886 | 0.0409 |
| C: all three | 29 | **-0.0010** | 0.0009 | -0.0029..+0.0001 | 2/12 | 0/12 | 0/12 | -0.0022 | 0.0889 | 0.0421 |

**No arm clears, on either population. All three are slightly negative.**

**The bug that nulled the earlier flux-statistics run was guarded against
explicitly.** Every fit asserts its column count (26 -> 27/28/29) and aborts
otherwise, so these zeros are real comparisons rather than the silent
no-op that experiment first produced.

### RECOMMENDATION: do not promote any of the three

This is now the **fifth** independent measurement of the same pattern: a
feature with genuine, physically-correct, low-correlation signal in isolation
that adds nothing to the model. The others were `odd_even_significance` (AUC
0.353), `duty_cycle` (0.353), `sec_significance` (0.381) and the harmonic
power ratios (~0.41). Novel-by-correlation is not novel-to-the-model: a
gradient-boosted ensemble over 26 correlated TLS statistics has already
extracted this information by other routes, and low pairwise correlation with
any SINGLE existing feature does not mean the information is unavailable as a
COMBINATION of them.

The practical implication is worth stating plainly: the remaining headroom in
this feature space is small enough that new derived statistics from the same
light curves are unlikely to clear 0.0097, however well motivated. The crowding
result -- the one deployed win -- came from genuinely EXTERNAL data (a star
catalog), not from another statistic computed on the same photometry.

Scripts: `timeseries_stats.py`, `timeseries_stats_validate.py`; data
`timeseries_stats_features.csv`; results
`timeseries_stats_validate_results.json`.

### One stated limitation, and a re-request confirming closure (2026-08-06)

The spatial confound check here was a **correlation** against |galactic b|
(|r| <= 0.073 for all three), not a held-in control ARM of the kind used for
`trap_vshape` (arm E). That is the weaker of the two forms. It is moot in this
case and only in this case: a spatial control arm exists to disqualify an
apparent WIN as a sky-position artifact, and no arm here produced a win to
disqualify -- all three deltas are negative. Were any of these three ever
revisited and found positive, the control arm would have to be run before the
result could be believed.

This investigation was re-requested on 2026-08-06 with the same scope
(variance ratio + residual ACF, skew/kurt excluded as already closed). No
recomputation was performed: the existing run already satisfies every element
of that scope -- single-sector only, pipeline's own in/out phase
classification, class-rate gate before modelling, pool availability checked up
front, 12 resamples, both populations. Production reconfirmed live and
unchanged at 0.9208 / 26 features / md5 `0c996a41a76cc765895d3013830a536b`,
`transit_shape_ratio` still present.

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

## Duration-scaled TRAPEZOID shape -- the real V-shape metric the broken column never measured

The prior round diagnosed `transit_shape_ratio` as structurally broken and
concluded a properly duration-scaled fit was "a refinement with real headroom,
not a duplicate". This is that fit, built and measured.

### Part 1: what was built, and what was deliberately NOT used

A five-parameter symmetric trapezoid fitted to the phase-folded curve, with
every window scaled to the star's own TLS duration rather than fixed in phase:

```
f(phi) = baseline                          x >= T14/2
       = baseline - depth                  x <= T23/2      x = |phi - phi0|
       = linear ramp between the two       otherwise       T23 = T14 * (1 - w)

trap_vshape = w = (T14 - T23) / T14      the ingress+egress fraction of T14
```

Free parameters: baseline, depth, T14, w, phi0. Parameterising the flat bottom
as `T23 = T14*(1-w)` rather than fitting `T23` directly bounds the metric to
[0,1] and makes the invalid region `T23 > T14` unreachable by construction
instead of relying on the optimiser to stay out of it.

**Physical reading.** `w -> 0` is a flat-bottomed transit: the occulter is small
relative to the star and sits fully inside the disc for most of the event.
`w -> 1` is V-shaped with no flat bottom at all -- the signature of a GRAZING
eclipse (the occulter never fully enters the disc) or an EQUAL-SIZE binary
(ingress occupies a large fraction of the event). For a central transit,
ingress/T14 ~ Rp/Rs, so a hot Jupiter gives w ~ 0.2-0.4 and a stellar companion
w ~ 0.6-1.0. This is LEO-Vetter's discriminator.

**Reused, not rebuilt:** the phase-fold-from-stored-ephemeris pattern of
`timeseries_stats.py` / `oddeven_timing.py` (no TLS re-run, no downloads) and
`scipy.optimize.least_squares` (scipy already a dependency;
`learning_curve_extrapolation.py` uses `curve_fit`).

**`batman` was a live option and was rejected on three specific grounds.** It
is a real dependency here (`injection.py` uses it), so this was a genuine
choice, not an availability constraint. (1) It is a limb-darkened PLANETARY
model parameterised by Rp/Rs, a/Rs, inc and limb-darkening coefficients --
fitting it to a grazing eclipsing binary forces a planet-shaped curve onto
exactly the profiles the feature exists to flag, so it can report how badly a
star fits a planet but not what shape the transit actually has. (2) a/Rs and
inc are near-degenerate at single-sector TESS SNR. (3) ~100x the per-star cost
over 5,486 stars for a worse-posed fit. The trapezoid is the model-agnostic
parameterisation professional vetting pipelines use for this test.

### Part 2: coverage, and the confound the coverage check exposed

**Convergence is high; USABLE coverage is not, and the gap is the story.** A
shallow, noisy transit cannot constrain a shape parameter however cleanly the
optimiser terminates -- the ingress is below the noise. Each fit therefore
carries a Jacobian-derived parameter uncertainty, and the measured
distributions were inspected before any threshold was chosen: `trap_vshape_err`
runs 0.055 / 0.341 / 290823 at q10 / q50 / q90, i.e. for a large tail the shape
parameter is formally unconstrained. Usable requires `depth_snr >= 3` and
`vshape_err <= 0.30`.

| population | rows | scorable | fit converged | USABLE |
|---|---|---|---|---|
| training | 5,486 | 5,486 | **93.6%** | **31.2%** |
| candidate pool | 2,454 | **488** | 99.2% | 47.1% |
| widesector pool | 271 | **69** | 97.1% | 23.2% |

**The pool denominator is not the row count, and using the row count would have
been wrong.** Only 488 of 2,454 pool rows carry a full ephemeris at all; the
rest are stars whose TLS post-processing failed before period/T0/duration were
written, so no phase-folded feature -- deployed or new -- exists for them.
Scoring against 2,454 would have measured TLS's failure rate, not this
feature's coverage.

Training failures: 7.2% "T14 hit bound" (a fit that ran to a duration bound has
not measured a duration), 2.0% no usable ephemeris, 1.4% too few in-transit
bins, 0.4% non-convergence; the rest of the drop from 93.6% to 31.2% is the
two usability filters.

**THE MISSINGNESS CONFOUND -- the most important number in this section:**

| | planets | false positives | gap |
|---|---|---|---|
| usable rate | 25.2% | 53.4% | **-28.1pp** |

**AUC of AVAILABILITY ALONE = 0.3593** -- identical to the feature's own AUC of
0.3595 to three decimals. A trapezoid fit is only usable when the event is deep
enough to constrain an ingress, and eclipsing binaries are deep while planets
are shallow. Confirmed directly: availability is predicted by `snr` at AUC
0.856, `SDE` at 0.824 and `duration` at 0.823 -- all three ALREADY among the 26
production features.

**How missingness actually enters this recipe -- checked, not assumed.** The
deployed estimator is `SimpleImputer(strategy="median", add_indicator=False)`
-> `HistGradientBoostingClassifier`. NaN never reaches the classifier;
unusable stars are given the training median. Bare HGB would read NaN natively
as a signal, but this pipeline erases it first. So the missingness channel is
largely closed by construction -- which is why the validation adds an explicit
availability indicator as its own arm rather than assuming either that the
confound is fatal or that it is absent.

### Correlations -- the new feature is NOT the old one

| | vs `transit_shape_ratio` | max \|r\| over all 26 |
|---|---|---|
| `trap_vshape` | **0.008** | 0.284 (`SDE_raw`) |
| `trap_t14_ratio` | **0.009** | 0.268 (`empty_transit_count`) |

Near-zero correlation with the column it replaces, in both directions. Combined
with the prior finding that the old column is uncorrelated with duration
(|r| 0.018) and has +0.0001 importance, this confirms the old feature is noise
rather than a degraded version of the same measurement.

Spatial exposure is clean and was checked rather than assumed: |r| vs
|galactic b| is **0.092** for `trap_vshape` and 0.023 for `trap_t14_ratio`.

### THE CLASS-RATE GATE -- PASSES, and by the widest margin yet measured here

| feature | planets (median) | false positives (median) | AUC |
|---|---|---|---|
| `trap_vshape` | 0.4346 | **0.5783** | **0.3595** |
| `trap_t14_ratio` | 1.3936 | 1.4997 | 0.4399 |

Tail rates, where the grazing/EB signature should live:

| group | n | >0.5 | >0.7 | >0.85 | >0.95 |
|---|---|---|---|---|---|
| planets | 1,094 | 40.9% | 14.7% | 3.3% | 0.2% |
| false positives | 615 | **60.5%** | **29.8%** | **8.0%** | 0.7% |

**False positives are measurably more V-shaped, in the direction the physics
predicts, at every tail cut.** An AUC deviation of 0.1405 from 0.5 is the
largest single-feature separation any candidate feature has produced in this
project -- against 0.4195 (variance ratio), 0.4131 (residual ACF), 0.5570
(odd-even timing, wrong sign), 0.353 (`odd_even_significance`). The gate is
passed and Part 3 is warranted.

### Part 3: resampled validation -- POSITIVE BUT UNPROVABLE

12 bootstraps, production recipe (`CalibratedClassifierCV(cv=5, sigmoid)` over
the deployed `SimpleImputer(median) -> HGB` pipeline) refit on identical rows,
frozen 4,386/1,100 split. Baseline **AUC 0.9128 (sd 0.0020)**, Brier 0.0879,
ECE 0.0417. Every arm is measured against a reference identical except for the
columns under test.

| arm | nfeat | mean d | sd | min | max | pos | clears | >=MDE | d 2-min |
|---|---|---|---|---|---|---|---|---|---|
| A: +vshape | 27 | **+0.0007** | 0.0010 | −0.0008 | +0.0022 | 9/12 | 0/12 | 0/12 | +0.0010 |
| B: +vshape, t14_ratio | 28 | −0.0001 | 0.0015 | −0.0029 | +0.0022 | 6/12 | 0/12 | 0/12 | +0.0001 |
| C: availability ONLY | 27 | **−0.0004** | 0.0005 | −0.0017 | +0.0006 | 3/12 | 0/12 | 0/12 | −0.0005 |
| D: +vshape GIVEN avail | 28 | **+0.0005** | 0.0007 | −0.0008 | +0.0016 | 8/12 | 0/12 | 0/12 | +0.0010 |
| E: +vshape GIVEN \|b\| | 28 | **+0.0002** | 0.0011 | −0.0018 | +0.0017 | 8/12 | 0/12 | 0/12 | +0.0008 |

**The three controls agree, and they attribute the signal correctly:**

* **C is NEGATIVE (−0.0004).** The availability pattern, handed to the model as
  an explicit indicator, is worth nothing. The confound that looked fatal in
  Part 2 does not survive contact with a model that already has `snr`, `SDE`
  and `duration` -- and median imputation had already closed the channel.
* **D ≈ A (+0.0005 vs +0.0007).** Conditioning on availability barely changes
  the delta, so what little the feature buys comes from the measured SHAPE
  VALUE, not from which stars have one. This is the decisive attribution and
  it is the one the raw Part 2 numbers could not have told us.
* **E stays positive (+0.0002).** It shrinks under the spatial control, as
  most things in this project do, but does not invert.

**And it does not clear.** +0.0007 is roughly **one fourteenth** of the 0.0097
detection threshold, and 0/12 resamples clear on either population. `pos 9/12`
is a consistent lean, not a significant one; with sd 0.0010 that is what a true
effect near +0.0007 looks like, and also what noise around zero looks like at
this n. **Adding `trap_t14_ratio` makes it worse** (arm B, −0.0001), so the
fitted-vs-TLS duration ratio is noise and is not carried forward.

Arm E's better Brier (0.0844) and ECE (0.0361) belong to its `|b|`-augmented
REFERENCE, not to the shape feature, and are not evidence for it.

### RECOMMENDATION: do not promote -- classified POSITIVE BUT UNPROVABLE

Production stays at **0.9208 / 26 features**. Nothing was deployed.

This is a materially different outcome from the recent run of negatives, and
worth recording as such. The feature is real: the physics prediction held, the
class separation is the widest yet measured here (AUC 0.3595), the value is not
a repackaged missingness indicator (arm C), and it survives the spatial control
(arm E). It is simply **too small to prove at n=1,098** -- the effect and the
detection floor are an order of magnitude apart. Believing +0.0007 on 9/12
positive resamples would be exactly the error the ≥10-resample rule exists to
prevent.

The binding constraint is **usable coverage, not the metric**. The shape is
only measurable for 31.2% of training stars, because a trapezoid fit needs an
ingress above the noise. Any future attempt should attack coverage -- multi-
sector stacking to raise per-star SNR is the obvious route, and would test the
same feature on a population where it is actually measurable.

### Should `transit_shape_ratio` be retired? A flag, not a decision.

The evidence now says the deployed column is measuring nothing:

* permutation importance **+0.0001**, coverage 69.7%, single-feature AUC 0.4333
* geometry, not shape, decides its value for **55%** of stars
* |r| **0.018** with duration -- not even a duty-cycle proxy
* |r| **0.008** with the fitted metric, so **removing it would not remove shape
  information**, because it never carried any

Retiring it is a **26 -> 25 feature production change**, which needs its own
retrain, its own resampled validation, and explicit go-ahead. Not done here.

**One stale-artifact correction, checked rather than assumed.** 1,218 rows in
`unknown_features.csv` carry `Required feature(s) not computable:
[... 'transit_shape_ratio']`, which reads like the old column disqualifying
candidates today. It does not: `OPTIONAL_FEATURES = {"transit_shape_ratio",
"FAP"}` means neither can appear in `blocking` under the current code. Those
status strings predate the Phase-3 optional-feature change and are historical.
The retirement case rests on the importance and correlation numbers above, not
on a candidate-loss claim.

Scripts: `trapezoid_shape.py`, `trapezoid_checks.py`, `trapezoid_validate.py`;
data `trapezoid_shape_features.csv`, `trapezoid_shape_pool.csv`,
`trapezoid_shape_widesector.csv`, `trapezoid_checks.json`,
`trapezoid_validate_results.json`.

## Multi-sector stacking to rescue trap_vshape -- CLOSED AT THE PART 1 GATE

The trapezoid write-up above ended by naming multi-sector stacking "the obvious
route" to make the +0.0007 provable. **That recommendation was wrong, and this
is the arithmetic that shows why.** No download was performed.

### A real finding surfaced on the way: the pipeline uses ONE sector per star

`01_download_known.py:201` and `01_download_negative.py:189` both call
`search[0].download()` -- the FIRST matching product only. Confirmed downstream
rather than inferred: processed light curves span 24-27 days with **zero gaps
> 5 days**, i.e. a single TESS sector each. And the data is there to be had --
a live MAST query over 60 stars from the recoverable population (59 resolved)
found **93% have >= 2 sectors, median 3, mean 4.6**.

So this project is training on roughly a **third** of the TESS photometry
available to it. Sector availability was never the bottleneck. That is a
pipeline-wide observation with implications far beyond this one feature, and it
is flagged at the end of this section as its own question.

### What stacking means here, stated precisely

Concatenate per-cadence flux from every available sector, phase-fold the
combined series on the stored period/T0, fit ONE trapezoid to that deeper fold.
Not per-sector fits averaged afterwards -- the point is more in-transit points
under a single fit, so the ingress rises above the noise. Required sectors to
clear the gate follow from `sqrt(N)` noise averaging: `N >= (3/depth_snr)^2`.

### Expected recovery is LARGE -- coverage would genuinely improve

| excluded bucket | n | expected recovered |
|---|---|---|
| `depth_snr < 3` | 1,812 | **822** |
| `vshape_err > 0.30` | 409 | **275** |
| non-converged (structural) | 746 | not modelled |
| **total** | 3,777 | **1,096** |

**Coverage 31.2% -> 51.1%, a 1.64x gain on the covered fraction.** This is a
real, substantial improvement and the reason the investigation was worth
running rather than dismissing.

### A counter-argument I nearly missed, checked rather than assumed

The naive model is that stacking only *adds stars*. It does more: it moves
existing stars into better-measured bands, and separation genuinely deepens
there.

| `vshape_err` | n | planet frac | AUC | \|dev\| |
|---|---|---|---|---|
| 0.00-0.05 | 233 | 0.751 | **0.2931** | **0.2069** |
| 0.05-0.10 | 313 | 0.588 | 0.3696 | 0.1304 |
| 0.10-0.20 | 755 | 0.603 | 0.3791 | 0.1209 |
| 0.20-0.30 | 408 | 0.686 | 0.4051 | 0.0949 |

**And it is not a class-mix artifact** -- planet fraction runs
0.751/0.588/0.603/0.686, non-monotone, while AUC is strictly monotone. The
value gap is real: in the best band planets sit at 0.302 and false positives at
0.586 (gap 0.284), against 0.443 vs 0.534 (gap 0.091) in the worst. Precision
sharpens the separation by **1.47x**. This argues FOR feasibility and is
included because leaving it out would have made the no-go look stronger than it
is.

### THE DECISIVE ARITHMETIC

Assuming the model-level delta scales linearly with the fraction of test stars
carrying a measured value (first-order, and probably **generous** -- newly
recovered stars are the marginal, noisiest ones):

| scenario | coverage x | precision x | delta | vs 0.0097 |
|---|---|---|---|---|
| measured now | 1.00 | 1.00 | +0.00070 | 0.07x |
| expected stacking | 1.64 | 1.00 | +0.00115 | 0.12x |
| expected + precision gain | 1.64 | 1.47 | +0.00169 | 0.17x |
| CEILING: 100% coverage | 3.21 | 1.00 | +0.00225 | 0.23x |
| **CEILING: 100% + best precision** | 3.21 | 1.47 | **+0.00331** | **0.34x** |

**Required multiplier to reach 0.0097: 13.9x. Maximum physically available:
4.7x.**

Even the impossible ceiling -- every star in the training set carrying a
trapezoid fit at best-band precision -- lands at **+0.0033, roughly one third
of the detection threshold**. The gap is not closable by more photometry.

### VERDICT: NO-GO. Parts 2 and 3 not run.

**The premise of the task does not hold.** Coverage is not what makes +0.0007
unprovable; the effect is simply small relative to what n=1,098 can resolve.
Correcting my own earlier framing: I wrote that "the binding constraint is
usable coverage, not the metric". That was wrong. Coverage caps the effect at
~4.7x its current size, and 13.9x is needed.

Not run, and the cost is the reason it matters: re-downloading, re-preprocessing
and re-fitting several thousand stars is hours of network and compute, and the
measured payoff would be an effect still 3x below the bar with 0/12 resamples
clearing. Production stays at **0.9208 / 26 features**; nothing was deployed
and nothing was downloaded.

### The question worth asking instead

The single-sector finding is the real result here and it is **not** about
`trap_vshape`. Every SNR-limited quantity in the pipeline -- TLS's SDE, depth
significance, `odd_even_mismatch`, the secondary-eclipse depth, the whole v2
feature set -- is computed from one sector when a median of three exist. The
useful experiment is not "does stacking rescue one feature at +0.0007", it is
"does stacking raise the model as a whole". That is a different, larger, and
better-motivated investigation, and it should be scoped on its own rather than
smuggled in under this one. **Not started; awaiting direction.**

Scripts: `multisector_feasibility.py`; data `multisector_availability.json`
(live MAST sample), `multisector_feasibility.json`.

## PIPELINE-WIDE multi-sector stacking -- STAGE 0 COST ESTIMATE. Awaiting decision.

Follow-on from the finding above. Nothing downloaded, nothing reprocessed,
production untouched at **0.9208 / 26 features** (md5 `0c996a41…`,
`transit_shape_ratio` still present -- the retirement task has not run).

### A correction to my own sector numbers, and a prior measurement I should have found first

The trap_vshape section reported "median 3, mean 4.6" sectors. That sampled only
the *recoverable excluded* subset -- a biased slice. Re-sampled properly,
stratified 60 planets / 40 false positives across the whole training set
(96/100 resolved): **median 4, mean 6.1, 86% with >= 2 sectors.**

And this project had already measured it at full scale. The medium-lift
multi-sector depth-consistency work records **97.8% of stars have >1 sector,
median 7**, over 5,137 stars, along with two cost figures that bear directly
here: download of all sectors **~4.5 h for 5,137 stars**, and per-sector TLS
**~118 h -- explicitly rejected as the wrong tool.** That section also already
answered a Stage 2 question: `n_sectors` alone scores **AUC 0.479**
(Mann-Whitney p=0.459), so observation history does not predict the label.

### THE COST DRIVER IS NOT SECTOR COUNT -- IT IS TIME SPAN

TLS cost is set by the period grid, which scales with the first-to-last
baseline, not the number of points. Two of this project's own measurements pin
the scaling almost exactly linear:

| dataset | baseline | points/star | TLS s/star |
|---|---|---|---|
| TESS, single sector | 27.4 d | 16,395 | **58.2** (measured, `elapsed_s`, n=488) |
| K2 pilot | 82 d | 3,645 | **174** (measured, n=71) |
| ratio | 3.04x | 0.22x | **2.99x** |

3.04x baseline gives 2.99x time on 4.5x FEWER points. **Cost tracks baseline;
point count is second-order.**

Sectors for one star are not contiguous -- they are scattered across the primary
and extended missions. Measured spans from the same 96-star sample:

| percentile | span | vs 27.4 d |
|---|---|---|
| p25 | 219 d | 8.0x |
| **p50** | **808 d (2.2 yr)** | **29.5x** |
| p75 | 1,891 d (5.2 yr) | 69x |
| p90 | 2,603 d (7.1 yr) | 95x |

**The median star's stacked baseline is 2.2 years and the p90 is 7.1 years --
longer than the 4-year Kepler baseline that already forced this project to add
a binning cap.**

### COST

| item | estimate | basis |
|---|---|---|
| download, all sectors, 8,211 stars | **~7 h** | project's measured 4.5 h / 5,137 stars |
| storage if retained | **~200 GB** | ~5 extra sectors x 8,211 x ~5 MB |
| **disk free right now** | **12 GiB (98% full)** | `df` |
| TLS re-search @ median 29.5x | **~27 days** | 8,211 x 1,717 s / 6 workers |
| TLS re-search @ p25 8x (optimistic) | **~7 days** | 8,211 x 466 s / 6 workers |

**Download is cheap. The TLS re-search is a multi-week job, and storage is a
hard blocker unless download-measure-delete is used** (the pattern this project
already applied for exactly this reason).

### WHAT STACKING CAN AND CANNOT REACH -- the deflating part

From the deployed model's own permutation importance, the top 5 features carry
73% of total positive importance:

| feature | importance | affected by stacking? |
|---|---|---|
| `st_rad` | **+0.0540** | **NO** -- stellar catalog |
| `st_teff` | **+0.0369** | **NO** -- stellar catalog |
| `chi2red_min` | +0.0353 | yes, but needs full TLS |
| `SDE` | +0.0122 | yes, but needs full TLS |
| `FAP` | +0.0085 | yes, but needs full TLS |

**The two single largest contributors (+0.0909 combined, more than the other
three together) are catalog stellar parameters that no amount of photometry
changes.** Crowding is likewise catalog-derived and unaffected.

That splits the work into two very unequal halves:

* **Cheap path (~10 h total):** fold at the ALREADY-KNOWN period/T0 -- no
  search needed, the trick this project used to avoid the 118 h -- and
  recompute the fold-derived features: `odd_even_mismatch`,
  `secondary_eclipse_depth`, `depth_consistency_std`, the `depth_mean_*`
  family, `trap_vshape`. **But these are largely the low-importance features**;
  `depth_mean` is measured at **-0.0013**.
* **Expensive path (7-27 days):** `chi2red_min`, `SDE`, `FAP`, `period`,
  `duration`, `transit_count` all come from the TLS *search* and cannot be had
  without re-running it at 2.2-year baselines.

The importance sits on the expensive side of that line.

### THREE PIPELINE ASSUMPTIONS THAT WOULD BREAK -- found by reading the code, not assumed

1. **Cadence mixing re-runs the K2 units bug.** `MAX_FLATTEN_WINDOW = 401` is
   specified in POINTS (`02_preprocess.py:33`, `06_download_unknown.py:213`).
   One star's sectors span 20-s, 2-min and FFI cadences across mission phases,
   so concatenating them and applying a fixed 401-point window gives a
   different physical window per chunk -- exactly the failure that made K2
   stars detrend over 8.2 days while TESS got 13.4 hours.
2. **Detrending must happen per sector, before stacking.** With a median
   808-day span built from ~27-day sectors, a Savitzky-Golay pass over the
   concatenated array smooths straight across month-to-year gaps. The existing
   code sorts by time and flattens the whole array; `02_preprocess.py:185`
   already flags stitched multi-sector input as a corruption risk for exactly
   this reason.
3. **The binning cap may actively harm the shape features.**
   `MAX_POINTS_BEFORE_BINNING = 30000` -> `TARGET_POINTS_AFTER_BINNING = 15000`.
   Six stacked sectors give ~100k points, binned back to ~15k -- about what one
   sector already delivers (16,395). Total SNR should be preserved (each bin
   averages more raw points) but the bins get coarser in TIME, and `trap_vshape`
   measures ingress *duration*. This needs verifying, not assuming.

### RECOMMENDATION: NO-GO on Stage 2 as specified. A narrowed Stage 1 is defensible.

**Do not authorise the full re-download and reprocess.** It is a multi-week
compute job, it needs ~200 GB against 12 GiB free, and the model's two largest
features are untouchable by it.

If you want to proceed, the pilot I would actually run is narrower than the
brief's Stage 1 and aimed at the one question that decides everything else:

> **Does `chi2red_min` / `SDE` / `FAP` improve materially at a 2.2-year
> baseline?** That is where the importance is. If those three do not move, the
> expensive path is dead and only the low-importance cheap path remains.

Scoped at ~120 stars that is roughly a **10-12 h overnight run** (TLS dominates
at ~28 min/star median, 6 workers), and it would also settle the three break
risks above on real data. I have not started it.

A cheaper alternative worth considering first: run the **cheap path only** on
the full set (~10 h, no TLS, fold at known ephemeris) and measure whether the
fold-derived features move the model at all. If they do not -- and their
importances suggest they will not -- that closes the question at a tenth of the
cost, without ever paying for the TLS re-search.

**Awaiting your decision. Nothing downloaded.**

Scripts: `multisector_feasibility.py`; data `multisector_span_sample.json`
(96-star stratified live MAST sample with per-star sector IDs).

## CHEAP-PATH multi-sector stacking -- CLOSED AT THE STAGE A GATE. There is no cheap path.

Ran the ~10 h cheap path proposed at the end of Stage 0: fold at the STORED
ephemeris across all sectors, no TLS re-search, recompute only fold-derived
features. **Stopped at Stage A. Stage B and C not run** -- not because the gain
was small, but because the method is structurally invalid, for a reason that
also explains why the expensive path cannot be avoided.

Production untouched: **0.9208 / 26 features**, md5 `0c996a41…`,
`transit_shape_ratio` still present.

### Disk: 12 GiB free respected, nowhere near the 200 GB Stage 2 estimate

Download-measure-delete, one sector at a time. **Peak on-disk footprint 12.5 MB
per star**, scratch cache verified empty mid-run and after. Free space was
12 GiB before, during and after (`df` checked at each point). 25 stars, ~10 min.

### Break-risk checks -- two confirmed real, one benign, and a FOURTH found

| risk | outcome |
|---|---|
| 1. `MAX_FLATTEN_WINDOW` in POINTS | **REAL, hit 10/25 stars (40%)** |
| 2. detrending across gaps | avoided by construction (per-sector flatten) |
| 3. disk | safe, 12.5 MB peak |
| **4. mixed time systems** | **REAL, not predicted, silently corrupting** |

**Risk 1 confirmed live.** 10 of 25 stars mix 120-s and 20-s cadence across
their own sectors. A fixed 401-point window would have detrended the 120-s
sectors over 13.4 h and the 20-s sectors over 2.2 h -- the K2 units bug again,
inside a single star. Deriving the window from each sector's measured cadence
gives 403 and 2413 points respectively, both 13.4 h, and the per-sector values
are recorded so the fix is auditable.

**Risk 4, found only by running real data.** MAST returns products for ONE star
in TWO time systems: most in BTJD (~1930-2664) but some in full BJD
(~2458929). `TIC_373729723` has both. Concatenating put its sectors **2.45
million days apart** (span 2,457,735 d = 6,700 yr), which folds to meaningless
phase with no error raised. Fixed by normalising any array with median time
> 2.4e6; span then reads a sane 759 d median, 2,819 d max. Stage A was re-run
from scratch after the fix and all numbers below are post-fix.

### Stacking DOES work mechanically -- and that makes the result unambiguous

| quantity | single -> multi (median ratio) | expected |
|---|---|---|
| `n_in_transit` | **x6.14** | more sectors, more points |
| `depth_se` (precision) | **x0.46** | sqrt(6.14) = 2.48x better -> 0.40 |

Precision improves almost exactly as sqrt(N) predicts. The plumbing is correct.

### THE FINDING: the stored ephemeris cannot phase-fold a multi-year baseline

| feature | median ratio multi/single |
|---|---|
| **`depth_mean`** | **0.28** |
| `depth_mean_odd` | 0.33 |
| `depth_mean_even` | 0.44 |
| `secondary_eclipse_depth` | 0.24 |
| `odd_even_mismatch` | 0.88 |
| `trap_vshape` | 0.99 |

The measured transit **depth collapses to 28% of its single-sector value**.
That is not noise averaging down -- it is the transit being smeared out of
phase.

Cause, tested rather than asserted. The stored period comes from a TLS fit to
ONE 27-day sector. Folding a 759-day median span at that period accumulates a
timing error of `span x sigma_P / P`, expressed below in transit durations:

| accumulated drift | n | median depth ratio |
|---|---|---|
| < 10 durations | 6 | **0.77** |
| 10 - 50 | 3 | 0.15 |
| 50 - 150 | 7 | 0.13 |
| > 150 | 5 | **0.02** |

A clean monotone dose-response, and **95% of stars drift by more than one full
transit duration** (median 68 durations). Correlations all run the right way:
depth ratio vs log drift −0.321, vs log span −0.478, vs n_sectors −0.270.

**The quantitative statement:** holding drift under one transit duration over
the stacked span needs a median `sigma_P` of **1.75e-04 d**. The stored value
is **9.88e-03 d** -- **~56x too imprecise.**

### WHY THIS CLOSES THE WHOLE QUESTION, not just this variant

The cheap path was defined as "avoid the expensive TLS re-search by reusing the
known ephemeris". **That is circular.** A period good enough to fold N sectors
coherently can only be obtained by searching over those N sectors -- which is
the expensive path. The cheap path does not sit alongside the expensive one as
a budget alternative; it *depends on the expensive one's output* and cannot be
run first.

So the options collapse to one:

* **Cheap path:** structurally invalid. Produces worse features than the
  single-sector baseline it was meant to improve (depth at 28%, secondary at
  24%). Not a small gain -- a regression.
* **Expensive path:** 7-27 days of TLS re-search, ~200 GB against 12 GiB free,
  and even then it can only touch `chi2red_min`/`SDE`/`FAP` while the model's
  two largest features (`st_rad` +0.0540, `st_teff` +0.0369) are catalog-derived
  and untouchable.

**RECOMMENDATION: do not promote, do not proceed.** Multi-sector stacking is
closed for this pipeline at the cheap tier on correctness grounds and at the
expensive tier on the Stage 0 cost/benefit grounds. Nothing was deployed.

**One thing worth keeping.** The two bugs found here are real and would have
silently corrupted any future multi-sector work in this repo: the
points-vs-time flatten window under mixed cadence, and the BTJD/BJD mixing
within a single star's products. Both fixes live in
`multisector_cheap_path.py` with the diagnostics that caught them.

Scripts: `multisector_cheap_path.py`; data `multisector_cheap_stageA.csv`
(per-star single vs stacked values, sector cadences, windows, spans).

## Multi-sector transit CONSISTENCY re-proposal -- DUPLICATE **and** BLOCKED. Nothing built.

Proposed as untested headroom: "check if transits repeat consistently across
sectors -- if a candidate transits in multiple sectors that boosts confidence,
or the reverse if inconsistent." Closed at Part 0 on two independent grounds,
either of which alone would be sufficient. No feature was built, no model fit.

### Ground 1: it is the same measurement, not a different framing

`multisector_consistency.py` already implements exactly this. Read from the
code, not from a summary:

| proposal | what the closed experiment already did |
|---|---|
| "do transits repeat consistently across sectors" | `_measure_depth()` folds **each sector separately** at the stored ephemeris and measures its depth |
| a consistency score | `sector_depth_frac_scatter` = std(depth)/mean(depth) across sectors |
| inconsistency weighted by noise | `sector_depth_chi2red` = uncertainty-weighted reduced chi-square of per-sector depths about their mean |
| "transits in multiple sectors" | `n_sectors_measured` |

Coverage then: 4,964/5,137 usable, 92.2% with >=2 sectors. There is no residual
variant here -- depth, duration and count consistency were all covered.

**A premise correction.** The re-proposal describes the prior test as finding
"no gain". It did not. It found **+0.0094, 95% CI [+0.0008, +0.0177] -- it
CLEARED the promotion bar**, and agreed under nested CV (0.9202 -> 0.9276) and
after calibration (0.8907 -> 0.9009). It was then **disqualified by control**:
missingness was class-asymmetric (19.1% of positives lacked the features vs
3.5% of negatives, from 494 positives with no resolvable TIC ID), and an
indicator-only arm carrying no measured values at all returned **+0.0102 --
108% of the gain**. Holding missingness constant left +0.0021, CI spanning
zero. The failure mode was bookkeeping, not absent signal. That distinction
matters: "no gain" invites a retest, "cleared then disqualified as artifact"
does not.

### Ground 2: the ephemeris wall applies here too, and harder

The proposal folds per sector rather than stacking sectors, so it might have
escaped the wall that closed stacking. It does not. `_measure_depth()` uses the
**stored** `t0` and `period` with a **fixed** phase window `|phase| <= half`.
Accumulated timing error `span x sigma_P / P` therefore walks each successive
sector's transit out of that window, and the measured depth falls with sector
epoch. Apparent "inconsistency" is manufactured by drift.

Measured on the 4,664 stars of the closed run's own output that have >=2
sectors and a usable `period_uncertainty` (`duration` is in DAYS in
`training.csv` -- a units slip caught before it reached these numbers):

| quantity | value |
|---|---|
| median accumulated drift | **124.7 transit durations** |
| fraction drifting > 1 full duration | **99.5%** |
| median `sector_depth_frac_scatter` | **2.20** |

The stacking work found 95% drifting past one duration at a 759-d median span;
this is **99.5%** at a 1,151-d median span. Strictly worse, because consistency
needs the full sector baseline by construction.

The scatter magnitude settles it independently of any drift model. A median
`frac_scatter` of 2.20 means the per-sector depths vary by **220% of their own
mean**. A genuinely repeating transit repeats to within a few percent. Depths
scattering twice their mean are not astrophysical measurements of anything.

**One honest limit on the causal chain.** The drift dose-response here is
FLAT (median `frac_scatter` 1.95 / 2.15 / 2.25 / 2.14 across the 1-10, 10-100,
100-1000, >1000 duration bins; Spearman vs log drift only +0.042) -- unlike the
clean monotone dose-response the stacking work found. That is not evidence
against the diagnosis: it is **saturation**. Once drift exceeds a duration the
transit sits at effectively random phase in the fold, and drifting further
changes nothing. The only bin below the cliff, `drift < 1`, has the lowest
scatter (1.40 vs 2.20), which points the right way, but at n=23 it is a weak
confirmation and is reported as such rather than dressed up.

### The reliably-measurable population is 0.45%

Applying an explicit precision bar -- drift below one transit duration, the
threshold the stacking work established:

| bar | n | share of 5,137 | class split |
|---|---|---|---|
| drift < 1.0 duration | **23** | **0.45%** | 20 planets / 3 FPs |
| drift < 0.5 duration | 4 | 0.08% | 3 planets / 1 FP |

23 stars, 3 of them false positives. On the frozen 1,098-star test set that is
roughly **5 stars**. No AUC, class-rate gate, spatial control arm or resampled
delta can be estimated from that, at any effort level. Part 1 was not run
because there is no population to run it on.

### Loose end closed: `chi2red`'s apparent separation is mostly SNR

`sector_depth_chi2red` scores AUC 0.3120 (FP median 15.89 vs planet 2.32),
which looks like strong separation. Its largest correlation against the 26
production features is **|r| = 0.729 vs `snr`** (then SDE 0.606,
`depth_duration_ratio` 0.601). Below the 0.80 redundancy threshold, but it is
substantially restating signal-to-noise -- deeper, higher-SNR eclipses give
tighter per-sector errors and hence larger reduced chi-square. Not an
independent consistency channel.

### RECOMMENDATION: do not build. No promotion, no Part 1.

Blocked twice over. As a duplicate it has already been measured, cleared, and
disqualified by a control that reproduced 108% of its gain. As a measurement it
is corrupted for 99.5% of the stars it would apply to, leaving 23 usable stars.
Measuring "consistency" across sectors requires an ephemeris precise enough to
fold coherently across the full multi-year baseline, and obtaining one requires
searching that baseline -- **the same circularity that closed the cheap path**.
Production stays at 0.9208 / 26 features / md5 `0c996a41`.

Scripts: `multisector_consistency.py`, `multisector_missingness_control.py`
(both pre-existing, unmodified); data `multisector_consistency.csv`.

## Stellar parameters round 2: DENSITY (do not promote) and VARIABILITY (**promotion candidate**)

Two mechanistically different additions, run as two investigations. One dies at
its controls; the other is the strongest result since crowding and is the first
thing in ~40 experiments recommended for promotion. **Nothing was promoted --
production remains 0.9208 / 26 features / md5 `0c996a41`, verified before and
after.**

### Part 0: the TIC data was already arriving and being thrown away

Verified live rather than assumed. `06_download_unknown.py:468` calls
`Catalogs.query_criteria(catalog="Tic", ID=chunk)`, which returns a **125-column**
row per star. The next line keeps eight:

    r[["ID", "ra", "dec", "rad", "e_rad", "mass", "e_mass", "Teff"]]

Among the 117 discarded columns are **`logg`, `e_logg`, `rho`, `e_rho`, `lum`**.
This is the `contratio`/`numcont` pattern exactly -- data already in the response,
unused -- except that this time the discarded fields turn out to be clean. No new
fetch mechanism was needed, only a wider column selection.

**Does duration/period already encode density?** Partly, and the distinction
matters. For a central circular transit `rho_circ = 3P / (G pi^2 T14^3)`, so the
transit-IMPLIED density is largely reconstructable from `period` and `duration`,
both production features. The discriminator is the RATIO to the star's catalogued
density -- and that needs `rho_star`, which requires stellar MASS. `st_mass` is
**not** among the 26 features. So the numerator is available to the model and the
denominator is not.

### Part 1: density passes its gate, then fails its controls

CTL trap checked first, against the specific failure crowding hit:

| field | planets | FP | availability-AUC |
|---|---|---|---|
| `rho` / `logg` / `rho_ratio` | 81.4% | 81.8% | **0.4984** |
| `contratio` (reference) | 50.3% | 78.8% | **0.3574** |
| `priority` (reference) | 50.2% | 78.7% | 0.3575 |

Clean -- and the same test reproduces the CTL signature on `contratio` in the
same run, so it is demonstrably sensitive rather than merely silent.

| feature | AUC | coverage | max \|r\| vs the 26 | verdict |
|---|---|---|---|---|
| `rho_circ` | 0.7087 | 98.0% | 0.905 (`duration`) | REDUNDANT |
| `st_rho` | 0.6929 | 81.5% | 0.921 (`st_rad`) | REDUNDANT |
| `st_logg` | 0.6775 | 81.5% | 0.896 (`st_rad`) | REDUNDANT |
| `rho_ratio` | 0.6015 | 81.5% | 0.754 | passes |

Those are the widest single-feature separations measured in this project. But
three of four exceed the 0.80 redundancy threshold, and `rho` is largely a
function of `st_rad` because mass and radius track each other on the main
sequence.

**Resampled, 12 bootstraps, vs resampled baseline 0.9128:**

| arm | nfeat | mean d | clears | >=MDE | d 2-min | Brier | ECE |
|---|---|---|---|---|---|---|---|
| A: +rho_ratio | 27 | +0.0030 | 2/12 | 0/12 | +0.0025 | 0.0868 | 0.0414 |
| **B: +all four** | 30 | **+0.0091** | **12/12** | 6/12 | +0.0094 | 0.0848 | 0.0389 |
| C: +rho_ratio \| sky | 28 | +0.0024 | 2/12 | 0/12 | +0.0023 | 0.0832 | 0.0364 |
| D: availability ONLY | 27 | +0.0024 | 6/12 | 0/12 | +0.0026 | 0.0873 | 0.0394 |

**A methodological error caught and fixed rather than shipped.** The sky control
was run on arm A, which does not clear -- proving nothing about arm B, which
does, and which contains the most spatially exposed column. The controls were
re-run on the arm that actually won:

| control | nfeat | mean d | clears |
|---|---|---|---|
| B_sky: all four, sky held constant | 31 | +0.0075 | 11/12 |
| B_avail4: four indicators, NO values | 30 | +0.0025 | 6/12 |
| **B_restricted: values only, missingness constant** | 30 | **+0.0036** | **4/12** |

**+0.0091 decomposes into roughly +0.0036 measured values, +0.0025 missingness,
+0.0016 sky -- and the values-only arm does not clear.** Same shape as the
multi-sector depth-consistency artifact, though far milder than that 108%.
Compounding it: pool availability is **~65%** against 87.1% in training, a real
train/serve gap.

**RECOMMENDATION for density: do not promote.** Positive but not attributable to
the physics it was proposed for.

### Part 2: variability -- the prediction half-inverted, and that was the diagnostic

Computed from RAW light curves, not `data/processed/`. `02_preprocess.py`
savgol-flattens with a window capped at 401 points (~13.4 h), which is a
high-pass filter: it removes exactly the multi-day rotational signal this
feature is about. The pipeline's own `validate_schema`/`choose_flux_columns` are
imported rather than reimplemented (that file documents EIGHT real schemas), and
every cleaning step is kept except the flatten. Transits are masked at 2x
duration so the signal cannot inflate its own vetting statistic. Single sector
per star by construction.

Predicted in writing before measuring: false positives more variable, AUC below
0.5 for all five. **It held for two of five and inverted for three:**

| feature | AUC | direction | max \|r\| | vs \|b\| |
|---|---|---|---|---|
| `var_oot_rms` | 0.6526 | OPPOSITE | **0.967** (`chi2red_min`) | -0.296 |
| `var_excess` | 0.4067 | as predicted | 0.576 | +0.121 |
| `var_ls_amp` | 0.5927 | OPPOSITE | 0.785 | -0.286 |
| `var_ls_power` | 0.4168 | as predicted | 0.266 | -0.010 |
| `var_ls_period` | 0.5889 | OPPOSITE | 0.162 | -0.000 |

The split is the confound named in advance. RAW scatter runs backwards and
correlates **0.967 with `chi2red_min`**, the model's #3 feature -- it is not
measuring activity, it is re-deriving a noise level already present. Normalising
by each star's own photometric error (`var_excess`) restores the predicted
direction. Coverage **99.7%**, availability-AUC 0.5057.

**Resampled, 12 bootstraps, vs 0.9128:**

| arm | nfeat | mean d | clears | >=MDE | d 2-min | Brier | ECE |
|---|---|---|---|---|---|---|---|
| A: +var_excess | 27 | +0.0043 | 6/12 | 0/12 | +0.0045 | 0.0862 | 0.0381 |
| B: +var_ls_power | 27 | +0.0060 | 7/12 | 1/12 | +0.0067 | 0.0848 | 0.0388 |
| C: +clean trio | 29 | +0.0086 | 9/12 | 3/12 | +0.0089 | 0.0835 | 0.0369 |
| **D: +all five** | 31 | **+0.0101** | **12/12** | **8/12** | +0.0096 | 0.0832 | 0.0365 |
| E: +pair \| sky | 29 | +0.0071 | 10/12 | 0/12 | +0.0078 | 0.0808 | 0.0355 |

Same error as the density arm, same fix: arm E holds sky constant for the two
LEAST exposed metrics while arm D contains the two MOST exposed. Re-run on D:

| control | nfeat | mean d | clears | >=MDE |
|---|---|---|---|---|
| **D_sky: all five, sky held constant** | 32 | **+0.0098** | **12/12** | 7/12 |
| D_nooot: drop the `chi2red_min`-redundant column | 30 | +0.0088 | 10/12 | 3/12 |
| **D_avail: five indicators, NO values** | 31 | **+0.0000** | **0/12** | 0/12 |

**Every control passes.** Sky costs 0.0003 despite the arm carrying the largest
spatial exposure in the project. Missingness contributes **exactly zero**.
Dropping the redundant column costs 0.0013, so this is not `chi2red_min`
relabelled. Brier improves 0.0879 -> 0.0832 and ECE 0.0417 -> 0.0365. Pool
availability checked on real candidates: **30/30 and 30/30 = 100%** on both
pools, against density's ~65%.

### RECOMMENDATION: variability is a promotion candidate. NOT promoted here.

The honest caveat: the mean delta sits AT the 0.0097 detection threshold, not
comfortably above it -- 8/12 resamples reach MDE, and the sky-controlled version
7/12. This is a real effect at roughly the smallest size this test set can
resolve. Before any promotion it still needs nested CV, which the deployed
validation suite requires and which was not run here.

Scripts: `stellar_density_fetch.py`, `stellar_density_checks.py`,
`stellar_density_validate.py`, `stellar_density_controls.py`,
`stellar_variability.py`, `stellar_variability_checks.py`,
`stellar_variability_validate.py`, `stellar_variability_controls.py`.

### An unrelated machine problem found while running this

A **runaway process from an earlier Claude session** (task `bkbpqnywm`, 8 days
old, orphaned to launchd) was crash-looping: a heredoc-run script using
`multiprocessing` spawn, whose children each tried to re-import `__main__` from
`code/<stdin>` and died instantly, forever. It had burned **36 CPU-hours** and
written a **28 GB** log while only 12 GiB was free, and it was holding this
round's workers to ~19% CPU. Killed; disk available went 15 GiB -> 43 GiB and
the resample rate roughly quadrupled. Not the project's scheduler -- checked
before touching it. The general lesson for this repo: `python3 - <<EOF` and
`multiprocessing` must not be combined, because spawn cannot re-import `<stdin>`.

## DEPLOYED: stellar variability, 26 -> 31 features, 0.9208 -> 0.9300

**The second deployed model change in the project's history**, and the first
since crowding. Production is now **0.9300 / 31 features / md5
`1f0b7cb8e78ab542374eaf78fc837a6f`** (rollback:
`models/versions/best_model_pre_variability_0c996a41.joblib`).

### The gate that was missing, and what it showed

The prior round left one thing unrun: nested CV, which the deployed validation
suite requires. Nested CV and the resampled test-set numbers answer DIFFERENT
questions -- resampling perturbs the training draw against one fixed 1,098-star
panel; nested CV rotates WHICH stars are held out -- so both are reported.

| test | old | new | delta | verdict |
|---|---|---|---|---|
| nested CV, pooled out-of-fold (4,386 rows) | 0.9295 | **0.9389** | **+0.0094**, CI [+0.0064, +0.0129] | clears |
| nested CV, mean of 5 outer folds | 0.9298 | 0.9393 | — | **new arm wins 5/5 folds** |
| fold-to-fold sd | 0.0097 | **0.0067** | — | more stable, not just higher |

Being precise rather than rounding toward the expected answer: the nested-CV
mean is **+0.0094, marginally BELOW the 0.0097 MDE**. That MDE was derived for
the 1,098-star frozen test set; nested CV pools 4,386 out-of-fold predictions,
so its interval is tighter and the operative criterion is that the CI excludes
zero. It agrees with the held-out +0.0101/+0.0098 rather than contradicting it.

### The combined worst-case control

Three controls had been run SEPARATELY. Passing three separately is not passing
all at once, so the most conservative available test was run before deploying:
reference = 26 + `abs_gal_b` + five availability indicators, both arms fitted
and scored ONLY on the population where all five values exist, so sky,
missingness and population are all constant by construction.

| arm | nfeat | mean d | positive | clears |
|---|---|---|---|---|
| worst case, +all five VALUES | 37 | **+0.0087** | 12/12 | 9/12 |
| worst case, +four (no `var_oot_rms`) | 36 | +0.0075 | 12/12 | 8/12 |

For contrast the density feature's equivalent control gave +0.0036, 4/12 --
which is why density was not promoted and this was.

### Final validation at real scale, on the fully backfilled dataset

| measure | old (26) | new (31) |
|---|---|---|
| frozen test AUC, single fit | 0.9208 | **0.9300** (+0.0091, CI [+0.0019, +0.0164]) |
| 12 training bootstraps | — | **+0.0100** (sd 0.0014), positive 12/12, **clears 12/12**, >=MDE 8/12 |
| 2-min-only subset | — | +0.0097, clears 10/12 |
| Brier (resampled) | 0.0879 | **0.0832** |
| ECE (resampled) | 0.0417 | **0.0365** |
| nested CV pooled OOF | 0.9295 | **0.9389** |

The 26-feature arm reproduced **0.9208 exactly**, confirming the recipe matches
what is deployed.

### The raw-light-curve dependency, and how it is protected

These five are the ONLY model inputs not computed from `data/processed*`.
`02_preprocess.py` savgol-flattens over ~13.4 h, a high-pass filter that
removes the multi-day rotation signal being measured -- a processed file would
yield the detrending residual, silently, with no error. Protections:

* `variability_features.py` takes the raw path as an EXPLICIT ARGUMENT. An
  earlier draft kept it in module state; under `spawn` the workers reverted to
  the default directory and the pool broke (`BrokenProcessPool`). Per-call
  paths make that unreachable.
* Both forward paths were wired and are documented at the call site:
  `06_download_unknown.add_variability_features` (reads `RAW_FOLDER`, which is
  run-tag aware so widesector runs follow automatically) and
  `retrain_pipeline._variability_for_raw` (reads `raw_path`, NOT
  `processed_path`).
* The requirement is restated in `FEATURE_COLUMNS`, in the module docstring,
  and in `best_model_metadata.json` under `raw_lightcurve_dependency`.

### Backfill integrity

| target | rows | verification |
|---|---|---|
| `training.csv` | 5,486 | 44 pre-existing columns byte-identical, host order identical |
| `unknown_features.csv` | 2,454 | 28 pre-existing columns byte-identical; 100% on all 488 scorable rows |
| `unknown_features_widesector.csv` | 271 | 28 pre-existing columns byte-identical; 100% on all 69 scorable rows |

Backups written to `*.pre_variability.bak`. Raw light curves found for
2,454/2,454 and 271/271 -- nothing imputed for missing input.

### Live consumption, proven -- and reported honestly

Deployed model reports `n_features_in_ = 31`. Partial dependence over the
**488 real scorable candidates**, sweeping each feature p10 -> p90:

| feature | direction | marginal AUC predicts | agrees? |
|---|---|---|---|
| `var_oot_rms` | UP 0.668 -> 0.766 | UP (0.6526) | yes |
| `var_ls_amp` | UP 0.694 -> 0.796 | UP (0.5927) | yes |
| `var_ls_power` | DOWN 0.752 -> 0.663 | DOWN (0.4168) | yes |
| `var_excess` | UP 0.653 -> 0.766 | DOWN (0.4067) | **no** |
| `var_ls_period` | 0.725 -> 0.713, ~flat | UP (0.5889) | **no** (flat) |

Consumption is unambiguous -- predictions move materially and monotonically.
**Three of five match their marginal direction, two do not**, and that is
stated rather than smoothed over. A boosted ensemble's conditional effect at
fixed values of 30 other features need not match a univariate marginal, and the
candidate pool is a different population from training (the documented
non-exchangeability). This is evidence of real consumption, not independent
confirmation of the physical hypothesis.

### Deployment mechanics

Manual offline swap, same structural reason as crowding: the live promotion
gate compares challenger and production on ONE matrix built from the current
`FEATURE_COLUMNS`, and the deployed model raises `ValueError` across a
feature-set change. Scheduler was removed with `launchctl bootout` first so the
race was impossible rather than unlikely; that also terminated a retrain tick
that had been **hung for 6.5 hours** on an archive query (it had modified
nothing). Post-deploy the scheduler was bootstrapped back, `/health` returns
`status: ok` with a live thread, and **a real retrain tick completed
successfully against the 31-feature configuration** (19:39:52, 139 watch
labels) -- live proof the auto-retrain path does not crash.

`conformal_calibration.json` was regenerated proactively and records the new
md5 (this was missed and had to be fixed reactively during crowding).

### A pre-existing bug found while verifying, and fixed

The 32-member bootstrap ensemble (`models/bootstrap_ensemble/`) was still at
**24 features**, dated 2026-07-29. It was never rebuilt for the crowding
promotion despite its own manifest saying "rebuild whenever the production
model is replaced". Its `predict_proba` raised `ValueError` listing
`crowd_flux_ratio_max`/`crowd_nearest_arcsec` as unseen -- proving it had been
broken since 2026-08-05, before this change. Only caller is the batch script
`backfill_uncertainty.py`, not the live request path. Rebuilt at 31 features
(32 members, 4.2 min).

Scripts: `variability_nested_cv.py`, `variability_worstcase_control.py`,
`variability_features.py` (production), `variability_backfill.py`,
`promote_variability_retrain.py`, `deploy_variability_model.py`.

## RAVEN-STYLE SYNTHETIC FALSE POSITIVES -- CLOSED AT THE PART 0 FEASIBILITY GATE

**This closes synthetic data for this project.** Second and final attempt.
Nothing was built; the decision rests on measurement, not argument. Production
untouched at 0.9300 / 31 features / md5 `1f0b7cb8`.

### The proposal's central premise is factually wrong

The brief proposes "use real stellar light curves as injection hosts" as the
single most important available realism upgrade, on the belief that the first
attempt "may not have fully used" them. Read from `injection.py` rather than
assumed, the first attempt ALREADY did all of this:

* line 2: *"synthetic transit injection into REAL processed light curves"*
* *"so the injected signal sits in genuine TESS noise/systematics, not
  synthetic noise"* -- real activity, real instrumental structure, real noise
  correlation, already inherited
* period/depth/duration drawn by **empirical resampling from real positives**,
  explicitly *"not an assumed parametric distribution"*
* `inject_eclipsing_binary()` already produced a **grazing V-shaped EB with a
  secondary eclipse** -- one of the very scenarios the brief proposes adding

So the largest proposed upgrade is already spent, and it produced 0.9654.

### What actually makes RAVEN work, and what this project lacks

From the RAVEN paper (arXiv:2509.17645) and its MNRAS follow-up: five
scenarios (Planet, EB, BEB, HEB, HTP) are simulated with **PASTIS**, plus a
**non-simulated** set of real TESS candidates driven by stellar variability and
systematics.

| RAVEN ingredient | this project |
|---|---|
| real TESS light curves as hosts | **already has it** |
| non-simulated FP set of real candidates | **already has it** -- the negative class IS real TOI false positives |
| **PASTIS**: Robin/Besancon galactic population synthesis + MESA isochrones, generating physically self-consistent blended/background stars with their own mass, age, radius, temperature and hence correct dilution | **does not have it.** `import pastis` fails; only `batman` is installed |

That last row is the whole difference, and it is precisely what the blend
scenarios need. BEB/HEB/HTP are *defined* by a second star whose luminosity
ratio sets the dilution. Without a galactic population model those dilution
factors would be invented -- exactly the "convenient assumption over real data"
this project has repeatedly found produces off-distribution features.

### The decisive measurement: the gap is not where a better injector could fix it

Domain-separability run on the 1,800 existing synthetic rows vs 5,486 real
training rows, splitting the production features into how the signal was FOUND
versus what it LOOKS like:

| feature group | n | domain AUC |
|---|---|---|
| all shared features | 24 | **0.9692** (reproduces the recorded 0.9654) |
| **DETECTION statistics only** (SDE, SDE_raw, FAP, snr, chi2red_min, transit counts) | 8 | **0.9382** |
| **SHAPE / everything else** | 16 | **0.9500** |

**A hypothesis stated in the script BEFORE running it was refuted.** The
prediction was that separability would concentrate in the detection statistics
-- a selection-function mismatch that better simulation cannot fix. It does
not. **Both halves are independently near-trivially separable.** Deleting the
entire detection group still leaves 0.9500; deleting all shape features still
leaves 0.9382. There is no subset to fix.

Per-feature standardized mean differences confirm it spans both kinds:

| feature | SMD (synth - real) | group | single-feature domain AUC |
|---|---|---|---|
| SDE | +0.78 | DETECTION | 0.7444 |
| SDE_raw | +0.76 | DETECTION | 0.7620 |
| FAP | -0.48 | DETECTION | 0.5400 |
| snr | +0.47 | DETECTION | 0.7793 |
| **st_teff** | **+0.44** | shape | **0.7284** |
| depth_duration_ratio | -0.37 | shape | 0.6079 |

`st_teff` is the quiet killer. It is a **stellar** property, not something any
injector controls: it reflects which hosts were injected into. Synthetic
positives are made by injecting into negative-class light curves, so they
inherit the negative class's stellar population, which differs from the
confirmed-planet population. Any host choice imports that host population's
parameters. A perfect transit simulator does not touch this.

### 7 of 31 production features cannot be given honest synthetic values at all

Found while running the comparison, and structural rather than incidental:

    crowd_flux_ratio_max, crowd_nearest_arcsec,
    var_oot_rms, var_excess, var_ls_amp, var_ls_power, var_ls_period

* **Crowding** is a TIC-catalog property of a real star at real coordinates.
  An injected row inherits the *host's* real neighbours, which describe the
  host, not the simulated scenario. For a simulated BEB this is actively
  contradictory: the scenario's defining feature is a blended companion that
  the crowding columns know nothing about.
* **Variability** is computed from the RAW pre-flatten light curve. Injection
  happens into ALREADY-PROCESSED curves, so a synthetic row has no raw
  counterpart and the five columns cannot be computed at all.

So 23% of the deployed feature space -- including **both deployed wins** --
would arrive either NaN or filled with values that contradict the injected
physics. Both options are themselves a distribution shift, on top of the 0.969
already measured.

### The pattern this fits

Both deployed wins came from information the photometry does not contain:
crowding from an external star catalog, variability from a *different
representation* of the data (raw rather than flattened). Synthetic injection
adds neither -- it resamples the information already present. That is the
mechanism-level reason it has now failed twice, and it is not addressed by
adding scenarios.

### RECOMMENDATION: do not build. Do not attempt a third time without PASTIS.

Every scenario in the brief was assessed:

| scenario | buildable here? |
|---|---|
| grazing EB | yes -- **already built** (`inject_eclipsing_binary`), part of the 0.9654 |
| background/diluted EB | only with invented dilution factors; needs a galactic population model |
| hierarchical triple / HEB / HTP | no -- needs PASTIS-equivalent multi-body + population synthesis |
| non-simulated FP (variability/systematics) | **already have it** -- that is the real negative class |

The measured position: the gap is 0.969 overall, 0.950 on shape alone, 0.938 on
detection alone, spans stellar parameters no injector controls, and 7 of 31
features are uncomputable in principle. The bar set for proceeding was
"meaningfully below ~0.90". Nothing available here plausibly reaches it.

**This is the definitive closure of synthetic-data approaches for this
project.** A third attempt should not be made unless PASTIS (or an equivalent
galactic population synthesis + multi-body simulator) becomes available AND the
crowding/variability computability problem is solved -- and even then the
`st_teff` host-population mismatch would remain.

Script: `synthetic_gap_attribution.py`; results
`synthetic_gap_attribution.json`.

### MDE re-verified at the current test-set composition

Asked to confirm rather than assume the ~0.0097 threshold after the
backfills and the 31-feature deployment. Frozen test set is unchanged at
**n=1,098 (867 positive / 231 negative, prevalence 0.790)**, deployed model
scoring 0.9300 on it.

A first estimate here perturbed one model's predictions and reported 0.0022.
**That was measuring the wrong quantity** and is discarded: perturbation leaves
the two prediction vectors almost perfectly correlated, so it understates the
variance of a genuine refit-vs-refit comparison. The correct figure is the one
measured directly during the variability deployment on this same test set --
the 26-vs-31 headline comparison gave CI [+0.0019, +0.0164] around +0.0091,
a **half-width of ~0.0072**. **~0.0097 has NOT materially shifted** and remains
the working threshold.

## TRANSFER LEARNING / DOMAIN-ADVERSARIAL TRAINING -- CLOSED AT THE ARCHITECTURAL GATE

Proposed to make Kepler/K2 usable after direct mixing failed (Kepler-family
domain AUC ~0.97; K2 **0.9973**, the highest ever measured here). Closed
without any data work, because both techniques presuppose an architecture this
project does not have. **Nothing built. Production untouched at 0.9300 / 31
features / md5 `1f0b7cb8`.** Mechanisms were RUN, not reasoned about --
`domain_adaptation_feasibility.py`.

### First: the proposal's own fallback is the investigation just closed

The brief offers "inject FPs into real TESS light curves" as a fallback. That
is exactly the synthetic-injection path closed in the section above, at its
Part 0 gate, permanently: already done with real hosts and empirically
resampled parameters, domain AUC 0.9654 originally and 0.9692 on
re-measurement, separable at 0.9500 on shape alone and 0.9382 on detection
alone, with 7 of 31 features uncomputable in principle. Same conclusion, not a
new path. Not reopened.

### Production architecture, verified live

    CalibratedClassifierCV(
        Pipeline([SimpleImputer(median), HistGradientBoostingClassifier]),
        cv=5, method="sigmoid")

Tree-based. CatBoost, LightGBM and XGBoost were all tested earlier and closed;
none is a neural network either.

### Claim 1 -- "pre-train on Kepler, fine-tune with a smaller learning rate"

Warm start is **mechanically real**: fitting on source gave `n_iter_ = 50`,
then raising `max_iter` to 80 and fitting on target gave `n_iter_ = 80`, i.e.
the 50 source trees were kept and 30 target trees appended.

But it is **not fine-tuning, and the difference is not cosmetic**:

* Boosting **appends; it never revisits.** The 50 source trees are immutable
  once built. Nothing adjusts them.
* `learning_rate` is **shrinkage on the contribution of the NEXT tree**. There
  is no learning rate on tree-structure decisions and no gradient-based
  adjustment of what was already learned, so "fine-tune with a smaller
  learning rate" has no referent.
* Neural fine-tuning nudges EVERY existing weight by a damped gradient; the
  pre-trained state is an *initialisation that gets refined*. GBM warm start
  leaves the source model as **frozen bias** that the new trees must fit
  residuals around.

Under domain shift that asymmetry runs the wrong way. At domain AUC 0.9973 the
frozen Kepler/K2 trees encode mission-specific thresholds, and the TESS trees
would spend capacity *undoing* them rather than inheriting anything useful.

### Claim 2 -- and the wrapper discards it anyway

Handing a warm-started, already-fitted estimator to production's recipe:

    inner estimator IS the warm-started object?  False
    clone(base) is fitted?                       False   <- clone() strips fit

`CalibratedClassifierCV` **clones its estimator and refits from scratch inside
each CV fold**. Any pre-training is discarded before a single calibrated
prediction exists. So even the weak analog cannot reach production without
abandoning the calibration wrapper -- and calibration is what delivers the ECE
this project tracks (0.0417 -> 0.0365 at the last deployment).

### Claim 3 -- domain-adversarial training has no attachment point

A fitted HGB is a list of trees; one tree is a numpy structured array whose
node fields are exactly:

    value, count, feature_idx, num_threshold, missing_go_to_left,
    left, right, gain, depth, is_leaf, bin_threshold, is_categorical, bitset_idx

`feature_idx` is an **index into the RAW input**. There is no weight matrix, no
embedding, no hidden layer, and no differentiable path from a loss back to a
"representation" -- because **there is no learned representation**. Probing for
learnable parameters (`coefs_`/`layers`/`embedding_`) returns `False`.

DANN works by inserting a gradient-reversal layer between a learnable feature
extractor and a domain classifier, so the extractor is pushed toward
domain-invariant features. A tree ensemble has no extractor to reverse
gradients through. The literature states the general form of this constraint
directly: adversarial approaches "assume the use of differentiable learning
models, hence cannot be applied to Gradient Boosted Decision Trees."

**One distinction worth recording, because it invites misreading.** A paper
titled *"Adversarial Training of Gradient-Boosted Decision Trees"*
(CIKM 2019) does exist. It concerns **adversarial ROBUSTNESS** -- resistance to
evasion attacks on inputs -- which is a different problem from domain-adversarial
ADAPTATION. It does not provide DANN-style representation alignment, and citing
it as evidence that "adversarial training works for GBDTs" would be wrong.

### What tree-compatible domain adaptation actually looks like -- and it was already run

The honest answer is not "no domain adaptation exists for trees." It is that
the tree-compatible family is **instance reweighting**, not representation
alignment: importance weighting by estimated density ratio, rejection sampling
on p(source), TrAdaBoost-style example reweighting. These operate on the data,
not on an internal representation, so they work with any learner.

**This project has already run that family**, on the synthetic case:

| correction | rows used | delta | 95% CI |
|---|---|---|---|
| rejection, p(syn) < 0.5 | 182 / 2,517 (7.2%) | -0.0002 | [-0.0068, +0.0066] |
| importance weighting | eff. n 315 | -0.0013 | [-0.0110, +0.0088] |

It neither helped nor cleared. And note what reweighting does at domain AUC
0.9973: the overlap region is nearly empty by construction, so the effective
sample size collapses toward zero. Reweighting cannot manufacture overlap that
does not exist -- it can only downweight toward the few points that already
overlap, which for K2 is close to none.

### Would a neural architecture rescue it? Two unsolved problems, not one

Applying either technique as intended requires a learnable representation,
i.e. switching to a neural model. This project's own CNN attempt:

| model | test AUC |
|---|---|
| CNN, real data only (5,378 examples) | **0.6964** |
| CNN, real + 4,000 synthetic | 0.6807 |
| CNN alone within the stacked ensemble | 0.7044 |
| **deployed tree model** | **0.9300** |

That is ~0.23 AUC short, on a dataset the CNN write-up already judged roughly
an order of magnitude too small for the architecture. Building a neural model
*in order to* enable domain adaptation means simultaneously solving (a) making
a competitive neural model on insufficient data, which already failed, and
(b) the domain adaptation itself, which has never been demonstrated to rescue a
0.997-separable pair. Starting a 0.696 architecture to fix a 0.930 model is the
wrong direction regardless of what is bolted onto it.

### What "success" would even mean -- and why it is unreachable here

The primary gate would be a REDUCTION in domain AUC. But domain AUC is a
property of the DATA REPRESENTATION. For a tree model the representation is the
31 raw production features, fixed. Nothing in a tree pipeline can change them,
so nothing can move the number -- the quantity the gate measures is not
addressable by the technique. For a neural model the representation is learned
and CAN be pushed toward invariance, which is precisely why DANN is a neural
technique and why the gate is only meaningful there.

### RECOMMENDATION: close. Do not build.

| technique | applicable to this architecture? |
|---|---|
| transfer learning (pre-train / fine-tune) | **no** -- warm start appends immutable trees, has no learning rate on prior structure, and is discarded by the calibration wrapper (verified `False`) |
| domain-adversarial training | **no** -- no learnable representation exists to regularize; literature states it requires differentiable models |
| instance reweighting (the real tree-compatible analog) | yes, but **already run**, non-clearing, and collapses to ~zero effective sample size at 0.9973 |

Revisit only if BOTH change: a neural architecture that is independently
competitive with 0.9300 on this data volume, AND evidence that adversarial
alignment rescues a domain pair separable at ~0.99. Neither is close.

Script: `domain_adaptation_feasibility.py`; results
`domain_adaptation_feasibility.json`.

## PER-EXAMPLE WEIGHTING: continuous SNR/width, and focal-style -- BOTH NEGATIVE

Two genuinely new weighting mechanisms, both tested to completion. Neither
promoted; production stays at 0.9300 / 31 features / md5 `1f0b7cb8`. The focal
arm is the **largest single negative** measured in this family.

### Part 0: what was already closed, and why these are still new

| prior experiment | mechanism | result |
|---|---|---|
| small-lift class weighting | per-CLASS constant: `balanced` vs none vs sqrt-inverse-frequency | `balanced` best; alternatives **-0.0022** and **-0.0021**, neither clearing |
| giant-star arm B | per-SUBPOPULATION constant: giants x3 | **-0.0001** (sd 0.0002), 0/12 overall AND 0/12 on the giant subpopulation |

Both are **static, pre-assigned** weights. The two angles here differ:

* **SNR/width weighting** is still static but **continuous in a feature value**
  rather than a per-group constant, and points the OPPOSITE way from arm B --
  the giant write-up records giants at **snr 1.76x** dwarfs, so "upweight
  giants" was upweighting the HIGH-SNR corner, while this upweights LOW SNR.
  Partial overlap in family, opposite in direction and functional form.
* **Focal-style weighting is dynamic**: the weight depends on the model's own
  current error, not on any observable fixed in advance. A repo-wide grep for
  "focal" returns **nothing** -- untouched.

### The routing trap, re-verified for sklearn 1.9.0 before designing anything

This project was bitten once already: `sample_weight` passed to a
`CalibratedClassifierCV` never reached the trees, and that arm was silently
identical to the unweighted one. Re-tested live:

| call form | Spearman(unweighted, weighted) | reached the trees? |
|---|---|---|
| bare `Pipeline`, `clf__sample_weight` | **0.873** | yes |
| `CalibratedClassifierCV(...).fit(sample_weight=)` | **0.997** | no -- sklearn warns *"sample weights will only be used for the calibration itself"* |

**A first version of this check got the verdict wrong** and is recorded because
the error is instructive: it judged by mean |prediction change|, which was
*larger* for the calibrated form (0.72 vs 0.06) and looked like success. That
change is the sigmoid CALIBRATOR being refit on reweighted data -- it moves
probabilities but **cannot move ranking**. Only the rank correlation separates
"the trees changed" from "the probability mapping changed", and AUC delta was
exactly +0.0000 either way. Every arm below therefore fits the BARE pipeline,
including the baseline, so all arms are like-for-like.

Consequence, stated rather than buried: these arms are **uncalibrated**, so
their Brier/ECE are comparable to each other but **not** to production's
0.0832 / 0.0365. `class_weight="balanced"` is retained as deployed; sklearn
multiplies sample_weight on top of it.

### The hypothesis, stated before measuring, and it cut both ways

Upweighting low-SNR rows concentrates capacity where classification is hard and
where marginal candidate decisions actually get made. The opposite risk was
named as equally plausible up front: low-SNR rows have the **noisiest feature
vectors**, so upweighting them may amplify label-independent noise and blur
splits the high-SNR rows currently define cleanly.

### Results: 12 bootstraps, bare pipeline, frozen test set

Baseline (bare, unweighted) AUC 0.9223 (sd 0.0035); low-SNR subpopulation AUC
0.9145 (n=271, `snr <= 4.15`, bottom test quartile).

| arm | mean d | sd | pos | clears | >=MDE | d 2-min | **d LOW-SNR** | clears low |
|---|---|---|---|---|---|---|---|---|
| snr_mild `(snr/med)^-0.25` | +0.0003 | 0.0022 | 6/12 | 1/12 | 0/12 | +0.0003 | +0.0001 | 0/12 |
| snr_mod `^-0.5` | -0.0003 | 0.0024 | 5/12 | 0/12 | 0/12 | -0.0002 | -0.0017 | 0/12 |
| snr_agg `^-1.0` | -0.0011 | 0.0028 | 3/12 | 0/12 | 0/12 | -0.0014 | +0.0017 | 0/12 |
| width_mod `(dur/med)^-0.5` | -0.0002 | 0.0027 | 5/12 | 0/12 | 0/12 | -0.0001 | -0.0020 | 0/12 |
| **focal** `(1-p_true)^2` | **-0.0116** | 0.0072 | **0/12** | 0/12 | 0/12 | -0.0104 | **-0.0166** | 0/12 |

**The static arms are a flat null with a mild dose-response toward harm:**
+0.0003 -> -0.0003 -> -0.0011 as inverse-SNR strength rises. The
noise-amplification risk named in advance is the one that materialised, mildly.
Decisively, **no arm helps the subpopulation it targets** -- 0/12 clearing on
the low-SNR delta for every single arm, including the aggressive one.

### The focal arm fails hardest exactly where it aims

-0.0116 overall, **0/12 resamples even positive**, and **-0.0166 on the
low-SNR subpopulation** -- it hurts the hard cases it upweights more than it
hurts the aggregate. That is mechanistically interpretable rather than
mysterious: `(1-p_true)^gamma` upweights whatever the model currently gets
wrong, and in this dataset the most-wrong rows are disproportionately
genuinely ambiguous or mislabelled, not merely hard. The modulating factor
cannot distinguish "hard" from "wrong label", which is the documented failure
mode of focal loss under label noise. Its ECE is actually the best of the group
(0.0692) -- it is confidently uninformative, not miscalibrated.

### True focal loss is not implementable in HGB, and the alternative is a closed door

Verified directly: `HistGradientBoostingClassifier`'s `loss` parameter accepts
only `{'log_loss'}` -- no callable objective. True focal loss reweights INSIDE
the objective every boosting round; what was tested is a one-step out-of-fold
approximation, and is labelled as such rather than as focal loss.

Implementing the real thing requires a custom objective, i.e. XGBoost or
LightGBM -- a **model-family change**. That is not worth doing here:

* CatBoost, LightGBM and XGBoost were each tested individually in the earlier
  bake-off and **none cleared alone**; the averaging ensemble did not either.
* The approximation of the technique, in the model that is deployed, came back
  at **-0.0116 with 0/12 positive** -- not marginal, and pointing away.
* So it would mean adopting a model family that already failed, in order to
  implement a technique whose approximation just failed decisively.

This is the same shape as the domain-adaptation closure: the technique requires
first solving a larger problem this project has already tested and closed. The
honest caveat is recorded too -- a one-step approximation is not proof that
per-round focal loss would fail. But the direction, the magnitude, the fact
that it damages its own target population most, and the architecture cost make
it a poor use of the next effort.

### RECOMMENDATION

| approach | verdict |
|---|---|
| continuous inverse-SNR weighting (3 strengths) | **do not promote** -- flat null, mild dose-response toward harm, 0/12 on its own target subpopulation |
| inverse-width weighting | **do not promote** -- same |
| focal-style dynamic reweighting | **do not promote** -- -0.0116, 0/12 positive, worst on the cases it targets |
| true focal loss via XGBoost/LightGBM | **not recommended** -- requires a model family already tested and closed |

This is now the **fourth** null in the reweighting family: per-class, per-
subpopulation, continuous-per-example, and dynamic-per-example. All four
reweight the SAME rows carrying the SAME information; none adds any. Consistent
with the pattern behind both deployed wins, which added new information
(external catalog; a different representation of the data) rather than
redistributing attention over what was already there.

Scripts: `weighting_routing_check.py`, `weighting_experiments.py`; results
`weighting_experiments_results.json`.

## GBM AVERAGING ENSEMBLE -- the result that was run but never written up

Recovered from `gbm_ensemble_results.json` (written 2026-08-04 00:12) and
`gbm_ensemble.log`. It was run, it produced real numbers, and it never reached
this file. Recorded here so it stops living only as a stray JSON. **Nothing was
re-run.**

### What was actually tested

HGB (production config) + CatBoost + LightGBM + XGBoost, **simple probability
averaging** (not weighted, not stacked). The three non-HGB models were tuned
once on the train split via `RandomizedSearchCV` (n_iter=10, cv=3):

| model | tuning CV AUC | selected |
|---|---|---|
| CatBoost | 0.9244 | lr 0.05, depth 8, l2_leaf_reg 9.0, 500 iters |
| LightGBM | 0.9234 | lr 0.03, 31 leaves, 500 trees, min_child 40 |
| XGBoost | 0.9242 | lr 0.03, max_depth 7, reg_lambda 5.0, 500 trees |

Then **12 training-data bootstrap resamples**, all four models refit per
resample, evaluated on the frozen test set and the 2-min subset.

### The numbers

| population | mean delta | sd | min | max | positive | clears |
|---|---|---|---|---|---|---|
| full clean test | **+0.0077** | 0.0026 | +0.0018 | +0.0112 | **12/12** | **8/12** |
| 2-min-only test | **+0.0096** | 0.0032 | +0.0048 | +0.0140 | **12/12** | **9/12** |

HGB baseline 0.8958 (sd 0.0038) -> ensemble 0.9034 (sd 0.0037). The file's own
`clears_robustly` flag is **False**.

Per-model mean AUC on the full test set across the 12 resamples:

| model | mean AUC |
|---|---|
| HGB (production) | 0.8958 |
| **CatBoost** | **0.9032** |
| LightGBM | 0.8974 |
| XGBoost | 0.8962 |

### The decisive internal reading: this is not an ensembling result

**CatBoost alone scored 0.9032. The four-model ensemble scored 0.9034.**
Averaging LightGBM and XGBoost -- both within ~0.002 of HGB -- on top of
CatBoost contributed about **+0.0002**. So the ensemble's +0.0077 is very
nearly the CatBoost-minus-HGB gap (+0.0074) with the averaging adding nothing
meaningful.

That matters because **CatBoost vs HGB is already an investigated, closed
question.** A single-fit CatBoost cleared at +0.0085 [+0.0004, +0.0171] -- the
first arm in 23 experiments to clear on both populations -- and was therefore
stress-tested rather than believed. Across ten seeds with matched-seed HGB
baselines it collapsed to **+0.0013 (sd 0.0024), 0/10 seeds clearing**, with
worse calibration (Brier 0.0943 vs 0.0896, ECE 0.0397 vs 0.0264). Not promoted.

So the ensemble result is a re-expression of a finding that already failed
replication, not independent evidence.

### It did not clear its own bar even in its own era

+0.0077 on the full test set sits **below the ~0.0097 MDE** that a 1,098-star
test set can certify, and it cleared only 8/12 resamples. By the promotion rule
in force then and now (`ci_lo > 0` robustly), this was already a non-promotion
at the time it was run.

### BASELINE COMPARABILITY: NOT COMPARABLE to current production

This is the load-bearing caveat and the old numbers must not be read as current:

| | at time of run (2026-08-04) | now |
|---|---|---|
| production features | **24** | **31** |
| production headline AUC | 0.9031 | **0.9300** |
| HGB resampled baseline in this run | **0.8958** | ~0.9223 bare / 0.9300 deployed |
| frozen split | 4,387 train / 1,098 test, "1 star not in manifest" | 4,386 / 1,100, "2 post-manifest stars" |

Confirmed from git history of `05_train_models.py`: crowding (24 -> 26,
0.9031 -> 0.9208) landed 2026-08-05 and variability (26 -> 31, 0.9208 ->
0.9300) landed 2026-08-06 -- **both after** this run. The split itself has also
drifted slightly as new labelled stars arrived.

### RECOMMENDATION: on the record as a non-promotion; a fresh run is optional and low-prior

Strictly, +0.0077 with 8/12 clearing is a **near-miss positive**, not a
decisive negative, so it is exactly the kind of result that could move in
either direction once seven more features are present. Taken literally, saying
anything about 0.9300 would require re-running the whole thing on the current
31-feature baseline (re-tune three GBMs, ~15 min, plus 12 resamples x 4 models
-- roughly an hour or more).

But the prior on that run is poor, for a reason independent of baseline drift:
the gain is CatBoost's, and CatBoost's advantage already failed replication at
0/10 seeds. Crowding and variability also added genuinely new information,
which if anything narrows whatever edge a different GBM family had on the
older, thinner feature set.

**Not re-run, and not recommended without a deliberate decision to spend the
compute.** Recorded as: tested, positive, never cleared robustly, explained by
an already-closed CatBoost finding, and measured against a baseline two
deployments out of date.

Scripts: `gbm_ensemble.py`, `gbm_ensemble_control.py`; data
`gbm_ensemble_results.json`, `gbm_ensemble_control_results.json`,
`gbm_ensemble.log`.

## SOM / UNSUPERVISED CLUSTER DIAGNOSTIC -- exploratory pass, ONE REAL LEAD

**Diagnostic only. Nothing built, nothing trained, nothing promoted.** Read-only
over `training.csv`, both candidate pools and the deployed model. Production
untouched at 0.9300 / 31 features / md5 `1f0b7cb8`.

### What RAVEN actually does with SOMs (and why this pass deliberately differs)

Confirmed from the paper: RAVEN's SOM is "trained to distinguish between Planets
and FPs based on **transit shape**", and its output is a **metric fed into** the
downstream XGBoost/GP classifiers, following Armstrong et al. (2017). That is a
feature generator, i.e. a training-pipeline change -- explicitly out of scope
here. This pass clusters primarily on the **full 31 production features**,
because the question is "what is the classifier missing" and the classifier
operates in that space; a shape-only view is reported as a secondary check.

SOM implemented in ~40 lines of numpy (minisom/somoclu/sklearn_som all absent),
6x6 grid, node weights agglomerated to 8 superclusters.

### A method failure caught and fixed before it became a "finding"

The first run used `RobustScaler` and produced a **degenerate partition**:
5,426 of 5,486 rows in one cluster, the rest 1-26 row outlier nodes. It also
made the validity check *appear to pass* -- "the giant-enriched cluster IS the
worst-calibrated: True" -- which was vacuous, since only one cluster had enough
rows to compute anything. These features are heavy-tailed, so a linear scaler
leaves extremes that drag individual nodes out while everything else collapses.
Switched to a rank-based `QuantileTransformer` and added an explicit degeneracy
guard. After the fix the largest cluster holds **18.7%**.

### PART 1 -- VALIDITY CHECK: PARTIAL FAILURE, and that caps confidence

The naive check would be near-tautological: `st_rad` is one of the clustered
features, so "a cluster with big stars" proves nothing. The real test is whether
the known giant signature is recovered -- **elevated ECE at ordinary AUC**
(giants 0.9013/0.0867 vs dwarfs 0.9017/0.0204).

| | cluster | giant% | AUC | ECE |
|---|---|---|---|---|
| most giant-enriched | 0 | 35.3 | 0.9460 | 0.0625 |
| worst-calibrated | 1 | 23.1 | 0.8280 | 0.0934 |

**They are not the same cluster.** Giants spread across the partition (37.5,
19.7, 5.9, 30.4, 21.1, 34.6, 0, 6.3 %) rather than isolating. So the method
**did not recover the one structural feature already known to exist.** By this
task's own standard -- "if the method can't recover a known structural feature,
don't trust it to find unknown ones" -- everything below is a **lead requiring
independent confirmation, not a finding.**

### PART 2 -- cluster structure (frozen test subset, model out-of-sample)

| cluster | n(test) | planet% | giant% | AUC | ECE | err@0.5 | n(full) |
|---|---|---|---|---|---|---|---|
| 0 | 218 | 80.3 | 35.3 | 0.9460 | 0.0625 | 7.8 | 961 |
| **1** | **108** | **76.9** | **23.1** | **0.8280** | **0.0934** | **18.5** | **532** |
| 2 | 202 | 92.6 | 4.0 | 0.9191 | 0.0314 | 4.5 | 1024 |
| 3 | 138 | 63.8 | 25.4 | 0.8857 | 0.0836 | 18.8 | 727 |
| 4 | 140 | 80.7 | 19.3 | 0.9400 | 0.0641 | 9.3 | 754 |
| 5 | 165 | 61.8 | 32.1 | 0.9264 | 0.0917 | 15.2 | 839 |
| 6 | 16 | 100.0 | 0.0 | -- | -- | -- | 112 |
| 7 | 111 | 92.8 | 3.6 | 0.9660 | 0.0397 | 3.6 | 537 |

AUC ranges **0.8280 to 0.9660** -- real, wide structure. No negative-dominated
cluster (<25% planet) exists at all, so Part 2.2(b)'s "overlooked planet type
hiding among negatives" returns **nothing**.

### THE LEAD: cluster 1, and it is probably a LABEL problem, not a model problem

Cluster 1 is worst on both ranking (0.8280 vs median 0.9264) and calibration
(0.0934), and it is **not** explained by giants (23.1%, mid-range). Its feature
profile against the rest of the population is coherent:

| feature | cluster 1 | rest | ratio |
|---|---|---|---|
| **FAP** | 0.0192 | 8.0e-05 | **240x** |
| **transit_count** | 4 | 12 | **0.33x** |
| distinct_transit_count | 4 | 10 | 0.40x |
| **period** | 6.38 d | 2.45 d | 2.61x |
| period_uncertainty | 0.0406 | 0.0092 | 4.42x |
| empty_transit_count | 0 | 1 | -- |

Long period, few transits, weak detection significance -- internally consistent
(longer period -> fewer transits in a 27-day sector -> weaker detection).

**But the host names point somewhere more specific.** Cluster 1 contains 444
confirmed positives, and the sample includes `51_Peg`, `11_UMi`, `24_Sex`,
`4_UMa`, `BD+20_2457`, `BD+03_2562` -- largely **radial-velocity discoveries
whose hosts are not known to transit**. 51 Peg is the canonical non-transiting
RV hot Jupiter.

That suggests cluster 1 is substantially a group of rows labelled positive
because *the host has a known planet*, while the TESS photometry contains **no
detectable transit at all** -- which is exactly why FAP is 240x higher and
transit counts a third of normal. If so, the model scores worst there because it
is being asked to learn a mapping that the data cannot support, not because of a
modelling deficiency.

**This is stated as a hypothesis, not a result.** It was not verified: no
external check was run on whether these specific hosts transit. That
verification is the obvious next step and is cheap.

### PART 2.3 -- candidate pools: a real coverage gap, and it converges

| pool | n scorable | median quantization error | beyond training p95 |
|---|---|---|---|
| training reference | 5,486 | 3.93 | 5% by construction |
| pool A | 254 | 5.42 | **25.2%** (5x expected) |
| pool B (widesector) | 54 | 7.39 | **63.0%** (12.6x expected) |

Occupancy shows where they land:

| cluster | training% | pool A% | pool B% |
|---|---|---|---|
| **1** | 9.7 | **33.1** | **25.9** |
| **4** | 13.7 | **40.9** | 16.7 |
| 0 | 17.5 | 1.6 | 9.3 |
| 7 | 9.8 | 1.6 | 7.4 |

**The convergence is the interesting part.** Cluster 1 is simultaneously the
worst-performing region on labelled data (AUC 0.8280) and **3.4x
over-represented** among real pool-A candidates (33.1% vs 9.7%). The region
where the model is weakest is disproportionately where real candidates actually
live. Pool B's 63% beyond-p95 figure rests on only 54 candidates and should be
treated as indicative.

### Secondary: shape-only SOM (RAVEN-faithful)

Same qualitative picture, no additional structure: AUC 0.7719-0.9642 across
clusters, worst cluster again a small weak-detection group. Shape-space says
nothing the full space did not.

### HONEST OVERALL READ

**Partially informative, with the validity check failed.** The method found real
structure (AUC 0.83-0.97 across clusters) and a real distributional gap between
training and the pools, but it did **not** recover the known giant/calibration
structure, so its ability to find genuinely new structure is unproven.

**The single most promising follow-up, scoped precisely:** take the 444
confirmed positives in cluster 1 -- characterised by FAP ~240x population
median, transit_count ~4 vs ~12, period ~6.4 d vs ~2.4 d -- and check what
fraction are **non-transiting RV discoveries** rather than transiting planets.
If a meaningful fraction are, they are label noise for a transit classifier, and
the questions that follow are whether excluding or down-weighting them improves
the model, and whether the pool over-representation of that same region means
production is being asked to score many candidates from a region its training
labels do not really describe. Nothing was acted on here.

Script: `som_cluster_diagnostic.py`; results `som_cluster_diagnostic.json`,
`som_cluster1_profile.json`.

## CLUSTER 1 / NON-TRANSITING RV HYPOTHESIS -- PARTIALLY CONFIRMED, THEN REFUTED WHERE IT MATTERED

Verification of the lead raised by the SOM diagnostic. **Read-only: nothing
modified -- not `training.csv`, not the frozen split, not the model.**
Production untouched at 0.9300 / 31 features / md5 `1f0b7cb8`.

Threshold declared BEFORE looking, per the task: "meaningful" = **>30% of
cluster 1's positives non-transiting**.

### Reproducing the partition

The diagnostic never persisted per-star cluster labels, only the count. Every
step is seeded, so the pipeline was re-run and **verified to reproduce cluster 1
exactly: n=532, 444 positives** -- the numbers already on record. The script
aborts otherwise, so nothing here rests on a different partition.

### PART 1 -- discovery method (NASA Exoplanet Archive `pscomppars`, 6,336 planets)

**444/444 cluster-1 positives matched the archive (100%).**

| method set | hosts |
|---|---|
| Transit | 284 |
| **Radial Velocity** | **119** |
| Imaging | 17 |
| Radial Velocity \| Transit | 12 |
| Transit \| TTV | 5 |
| Microlensing | 2 |
| other | 5 |

**68.2% of cluster 1 was discovered by transit.** The cluster is not
predominantly RV -- the recognisable RV names that prompted the hypothesis were
a visible minority, not the bulk.

### PART 2 -- does the host actually transit? (`tran_flag`)

| | n | % of matched |
|---|---|---|
| transits (tran_flag=1 on >=1 planet) | 305 | 68.7% |
| **does NOT transit (all tran_flag=0)** | **139** | **31.3%** |

**31.3% vs a declared threshold of 30% -- a marginal pass, not a decisive one,
and it is reported as marginal.**

Geometry check, done properly. A first version of this printed a FIXED caption
("far from 90 deg => cannot transit") beside every row, which was wrong for
HD 100777 at 90.0 deg and HD 111232 at 87.9 deg. Recomputed as |i - 90|:

| |i - 90| | n | reading |
|---|---|---|
| > 15 deg | 24 | geometrically cannot transit for realistic a/R* |
| 5 - 15 deg | 3 | very unlikely |
| < 5 deg | 8 | possible; tran_flag=0 means searched-and-absent, not geometry |

Median offset 40 deg. **But inclination is available for only 35 of the 139**, so
this supports the geometric reading for a subset, not for all.

### PART 3 -- THE PART THAT REFUTES THE HYPOTHESIS'S POINT

The hypothesis was not merely "some cluster-1 positives don't transit". It was
that these rows are **label noise teaching the model an unlearnable pattern**,
and that this explains cluster 1's poor AUC (0.8280). That claim fails:

| | value |
|---|---|
| non-transiting positives in the frozen test set | **25** |
| their median predicted probability | **0.9622** |
| all test positives, median predicted probability | 0.9715 |
| fraction scored below 0.5 | **4.0%** |
| test AUC, all 1,098 rows | **0.9300** |
| test AUC excluding those 25 rows (1,073) | **0.9301** |
| difference | **+0.0001** |

**The model scores these rows like ordinary positives, not like noise.** If they
were unlearnable label noise the model would score them low and they would drag
the metric down; instead they sit within 0.01 of the normal positive median, and
removing them moves test AUC by one ten-thousandth.

So: the compositional claim is marginally true, and the causal claim is false.
**Cluster 1's 0.8280 AUC is NOT explained by non-transiting contamination, and
its cause remains unexplained.**

### Test-set integrity: no issue, stated precisely

25 rows, 2.28% of the frozen test set, worth **+0.0001** AUC if excluded. Every
AUC this project has reported is unaffected at the reported precision. If
anything the direction is mildly pessimistic rather than inflated. No action
needed and none taken.

### Scale, for the record

| quantity | value |
|---|---|
| non-transiting rows | 139 |
| fraction of the positive class (4,334) | **3.21%** |
| fraction of all training rows (5,486) | 2.53% |
| of those, in the frozen test set | 25 |

### What a "fix" would involve -- and why it is NOT recommended

A future task could flag `tran_flag=0` hosts and exclude or down-weight them.
The evidence says do not bother:

* the measured benefit is **+0.0001** on the test set -- three orders of
  magnitude below the ~0.0097 detection threshold;
* it would delete **139 positives (3.21% of the class)** for that;
* the tension the task asked to state honestly does not even arise here. The
  "bad labels vs less data" trade-off matters when the labels are actually
  hurting. These are not: the model already fits them, and they cost the metric
  nothing. Removing them is pure data loss.

### The one thing genuinely worth flagging, as a hypothesis

The model confidently assigns **0.96 median probability to signals that, by
definition, are not transits** -- a non-transiting planet cannot produce a
transit, so whatever TLS locked onto in those light curves is something else
(stellar variability, a systematic, or a spurious period). It has evidently
learned that this class of feature vector means "planet", generalising from the
~114 such rows in the training split to the 25 held out.

That is benign for labelled data but is exactly the behaviour that would
manufacture false positives on unknown candidates -- and cluster 1 is **3.4x
over-represented in the real candidate pool** (33.1% of pool A vs 9.7% of
training). Whether that actually produces bad candidates is untested and is
stated as a hypothesis, not a finding. It would be a separate investigation:
take pool candidates falling in cluster 1, and check whether their existing
vetting evidence (centroid, crowding, VSX/SIMBAD variability) flags them at a
higher rate than the pool baseline.

Scripts: `cluster1_rv_verification.py`; data `cluster1_rv_verification.json`,
`cluster1_rv_verification.csv`, `cluster1_test_impact.json`,
`nasa_pscomppars_cache.csv`.

## CLUSTER-1 POOL EVIDENCE CROSS-CHECK -- HYPOTHESIS NOT SUPPORTED. Thread closed.

Third and final part of the cluster-1 thread. **Read-only: nothing modified.**
Production untouched at 0.9300 / 31 features / md5 `1f0b7cb8`.

The hypothesis had **two** parts and both had to hold: cluster-1 pool candidates
should be (a) scored at least as confidently as the rest of the pool, and
(b) trip independent false-positive evidence MORE often. High confidence alone
is not a problem; high confidence *combined with* independent red flags is.

**(a) holds. (b) fails, and the two genuinely external checks point the other
way.**

### Part 1 -- cluster-1 candidates in the current pools

Partition reproduced exactly (cluster 1: n=532, 444 positives; 9.7% of
training), then pool rows mapped to their best-matching unit -- the same fitted
partition, not a re-fit.

| pool | scorable | in cluster 1 | over-representation |
|---|---|---|---|
| A | 254 | **84 (33.1%)** | **3.41x** |
| B (widesector) | 54 | 14 (25.9%) | 2.67x |

The earlier 33.1% figure **reproduces exactly** against current pool state.

**Confidence is genuinely elevated in pool A** -- prediction (a) confirmed:

| | median prob | mean | >= 0.9 |
|---|---|---|---|
| cluster 1 | **0.9214** | 0.8315 | **57.1%** |
| rest of pool | 0.7905 | 0.7054 | 32.4% |

Mann-Whitney **p = 1.5e-05**. In pool B there is no difference at all
(0.8162 vs 0.8023, p = 0.98).

### Part 2 -- independent evidence: the prediction fails

| check | cluster 1 | rest | odds ratio | p |
|---|---|---|---|---|
| **VSX variable-star HIT** | 20/75 (**26.7%**) | 48/126 (38.1%) | **0.59** | 0.12 |
| **Gaia blend risk HIGH** | 16/75 (**21.3%**) | 41/126 (32.5%) | **0.56** | 0.11 |
| ExoFOP/TFOP | 201/201 NO_HIT | -- | -- | -- |

Both external checks run **opposite** to the prediction -- cluster-1 candidates
trip them *less* often, not more. Neither reaches significance, but neither
supports the hypothesis either. ExoFOP returns nothing, confirming rather than
assuming that these candidates remain TOI-free by construction.

In-pipeline signatures are genuinely mixed:

| signature | cluster 1 | rest | p | reads as |
|---|---|---|---|---|
| odd_even_mismatch | 1.765 | 0.582 | <1e-4 | **concerning** (EB-like) |
| var_excess | 2.046 | 1.172 | <1e-4 | **concerning** (variable host) |
| var_ls_power | 0.195 | 0.084 | 0.0008 | **concerning** |
| secondary_eclipse_depth | 8.7e-06 | 1.5e-04 | <1e-4 | *reassuring* (less EB-like) |
| crowd_flux_ratio_max | 0.0034 | 0.0258 | 0.005 | *reassuring* (less contaminated) |
| SDE / FAP | 6.71 / 0.0155 | 7.45 / 0.0043 | <1e-4 | **circular** -- these define cluster 1 |

The SDE/FAP rows are not evidence: low SDE and high FAP are part of what puts a
star in cluster 1 in the first place, so finding them there is tautological.
They are listed only to be explicit about which comparisons carry no weight.

### VERDICT: NOT SUPPORTED

The model is measurably more confident in this region (0.9214 vs 0.7905,
p=1.5e-05), but that confidence is **not** accompanied by elevated independent
false-positive evidence. On the two checks that are genuinely external to the
feature space -- VSX and Gaia blending -- cluster-1 candidates look *cleaner*
than the pool baseline, not dirtier.

**Two honest caveats against over-reading the null:**

1. **Under-powered.** 75 vs 126 candidates with evidence; p = 0.12 and 0.11.
   These are not tight nulls, they are non-significant differences that happen
   to point the reassuring way.
2. **The VSX direction may be partly mechanical, which cuts against the
   reassuring reading too.** Cluster 1 is long-period by construction
   (6.38 d vs 2.45 d) and VSX is richest in short-period variables, so a lower
   cross-match rate for long-period signals could be a detection bias rather
   than genuine cleanliness. This weakens the "cluster 1 looks clean"
   interpretation as much as it weakens the hypothesis.

So the correct statement is not "cluster-1 candidates are fine". It is: **the
specific predicted signature -- high confidence plus elevated independent red
flags -- was looked for and is not there.**

### No mitigation is proposed

A confidence-tier caveat for cluster-1-like candidates, analogous to the
existing giant-star penalty, would have been the natural mitigation had the
hypothesis held. It did not, so building one would be penalising a
subpopulation on evidence that does not exist. Not recommended, not scoped.

### THE CLUSTER-1 THREAD IS CLOSED, WITH ITS CENTRAL FINDING STILL UNEXPLAINED

Three investigations, each ruling out one explanation:

| # | investigation | outcome |
|---|---|---|
| 1 | SOM diagnostic | found cluster 1: worst AUC 0.8280, worst ECE 0.0934, 3.4x over-represented in the pool. Validity check only PARTIALLY passed (giants did not isolate), so it was flagged as a lead, not a finding |
| 2 | RV-discovery verification | 31.3% non-transiting -- a marginal pass of the declared 30% bar -- but the label-noise mechanism was **refuted**: those rows score 0.9622 median and removing them moves test AUC by +0.0001 |
| 3 | pool evidence cross-check (this) | confidence elevation **confirmed** (p=1.5e-05); elevated independent FP evidence **not found**; the two external checks point the other way |

**Cluster 1's 0.8280 AUC remains genuinely unexplained.** Two plausible
mechanisms were proposed, tested and eliminated. That is a real if unsatisfying
result, and it is recorded as open rather than dressed up. Anyone resuming this
should note the diagnostic's own validity check never cleanly passed, so a
prior worth holding is that cluster 1 may be partly an artifact of an
unsupervised partition that was never shown able to recover known structure.

Scripts: `som_cluster_diagnostic.py`, `cluster1_rv_verification.py`,
`cluster1_pool_evidence.py`; data `cluster1_pool_evidence.json`.

## CATBOOST SEED-ENSEMBLE on the 31-feature baseline -- POSITIVE BUT UNPROVABLE. Not promoted.

Fresh run against the CURRENT baseline. Both prior CatBoost results were on the
24-feature/0.9031 model, two deployments stale. **Production untouched at
0.9300 / 31 features / md5 `1f0b7cb8`.**

The original stress test's open caveat is now closed: it used "fixed mid-range
CatBoost hyperparameters" because the search-selected config was lost in a
crash. Those parameters were **recovered from `gbm_ensemble_results.json`**
(lr 0.05, depth 8, l2_leaf_reg 9.0, 500 iterations) and used throughout here.

### Part 1 -- single-fit CatBoost, both arms on production's exact recipe

| model | AUC | 2-min | Brier | ECE |
|---|---|---|---|---|
| HGB (production recipe) | **0.9300** | 0.9210 | 0.0768 | 0.0316 |
| CatBoost | **0.9342** | 0.9269 | 0.0771 | **0.0385** |

Paired bootstrap **+0.0042, 95% CI [-0.0013, +0.0100] -- does NOT clear**
(2-min +0.0059, CI [-0.0006, +0.0121], also not clearing). The HGB arm
reproduced 0.9300 exactly, confirming the recipe matches deployment.

**The edge has roughly halved.** On 24 features a single CatBoost fit gave
+0.0085 and DID clear. With crowding and variability present it is +0.0042 and
does not -- consistent with those two features having absorbed part of what
CatBoost was exploiting, which is what was predicted when the stale ensemble
result was retrieved.

### Level 1 -- seed stability (40 members, same training split)

| N | ensemble AUC | 2-min | delta | 95% CI | clears |
|---|---|---|---|---|---|
| 10 | 0.9288 | 0.9203 | -0.0013 | [-0.0076, +0.0051] | no |
| 20 | 0.9286 | 0.9199 | -0.0015 | [-0.0078, +0.0049] | no |
| 40 | 0.9288 | 0.9202 | -0.0012 | [-0.0075, +0.0050] | no |

Individual member AUC: **mean 0.9279, sd 0.0012**, min 0.9251, max 0.9309.

**Two findings that matter more than the table.**

1. **There is almost no seed variance left to average away.** Member sd is
   0.0012, and averaging 40 members moves the mean member only 0.9279 ->
   0.9288. `CalibratedClassifierCV`'s internal fold-averaging already absorbs
   most of the seed noise that destroyed the original result -- so the premise
   that seed-ensembling would stabilise a hidden advantage is largely
   pre-empted by the wrapper production already uses.
2. **The seed-42 single fit (0.9342) sits ABOVE the maximum of all 40 members
   (0.9309).** That is the original lucky-seed pattern reappearing.

### Level 2 -- data-draw robustness, the actual bar (12 bootstraps, N=20)

| arm | mean d | sd | min | max | positive | clears | >=MDE | d 2-min | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| **CatBoost seed-ensemble** | **+0.0042** | 0.0018 | +0.0012 | +0.0082 | **12/12** | **2/12** | **0/12** | +0.0049 | 0.0854 | 0.0459 |
| production baseline | -- | -- | -- | -- | -- | -- | -- | -- | **0.0832** | **0.0365** |

Baseline 0.9228 -> ensemble 0.9270.

**Seed-ensembling genuinely fixed the instability.** The original result was
0/10 seeds clearing at +0.0013 (sd 0.0024); this is **12/12 bootstraps
positive** at +0.0042 (sd 0.0018). The effect is now consistently measurable
rather than seed-dependent.

**And it is consistently too small to promote.** +0.0042 against a ~0.0097
MDE: 2/12 clearing, **0/12 reaching MDE**. Calibration is also worse on both
metrics -- Brier 0.0832 -> 0.0854, **ECE 0.0365 -> 0.0459 (+26%)** -- which
reproduces the original CatBoost calibration finding exactly.

### An honest wrinkle: the advantage appears under bootstrapping, not on full data

Level 1 (full training split) put the ensemble at **-0.0013 vs baseline**;
Level 2 (bootstrap samples) puts it at **+0.0042**. Bootstrap draws contain
~63% unique rows, so both arms degrade -- baseline 0.9300 -> 0.9228 -- but
CatBoost degrades less. The advantage is therefore partly a **robustness to
reduced effective sample size**, and production trains on the full set, which
is precisely the condition where the ensemble measured slightly *worse*.

**A design limitation, stated plainly.** Ensemble members used cv=3 (a cost
concession with precedent in `09_build_bootstrap_ensemble`) while the single
fit used cv=5. cv changes AUC, not just calibration, because
`CalibratedClassifierCV` averages its fold models. A single cv=5 fit (0.9342)
beat the 40-member cv=3 ensemble (0.9288), which suggests **fold-averaging
matters more for CatBoost than seed-averaging** and that the ensemble as tested
is not CatBoost's best configuration. A cv=5 40-member version was not run
(~1 hour more); it is unlikely to change the verdict, since the cv=5 single fit
itself landed at +0.0042 with a CI crossing zero -- the same place.

### Operational cost, measured

| | production | N=20 seed-ensemble |
|---|---|---|
| artifact size | 18.3 MB | **127.0 MB (7.0x)** |
| inference, 1,098 rows | 142 ms | **261 ms (1.8x)** |
| fits per retrain | 5 | **60 (20 members x 3 folds)** |
| wall-clock, this validation | -- | ~2 h for 12 bootstraps |

### RECOMMENDATION: positive-but-unprovable. DO NOT PROMOTE.

The effect is real and now reliably measurable -- 12/12 bootstraps positive at
+0.0042, a genuine improvement on the 0/10 that closed the original
investigation. But it sits at **less than half the ~0.0097 the 1,098-star test
set can certify**, reaching MDE on 0/12 resamples, while costing 7x storage,
1.8x inference, 12x the retrain fits, and **degrading calibration by 26% on
ECE** -- a metric this project explicitly tracks and improved at the last
deployment.

Promoting would trade a measurable-but-uncertifiable ranking gain for a
certain calibration loss and a large operational cost. This lands in the same
place as the learning-curve and noise-floor audits: **at this label supply the
classifier cannot be improved provably**, and CatBoost is the clearest example
of a real effect that the test set simply cannot resolve.

Scripts: `catboost_singlefit_31.py`, `catboost_seed_ensemble.py`; results
`catboost_singlefit_31_results.json`, `catboost_seed_ensemble_results.json`,
`catboost_ensemble_cost.json`.

## Files

All in `code/experiments/`: `injection.py`, `completeness_curve.py`,
`phase_fold_views.py`, `build_cnn_dataset.py`, `train_cnn.py`,
`gp_classifier.py`, `stacked_ensemble.py`, `uncertainty.py`, plus saved
results (`*.json`, `completeness_curve_results.csv`, `cnn_dataset.npz`).

## CROSS-MISSION (Kepler/K2 + TESS) TRAINING -- CLOSED AT PART 1. Nothing built.

Prompted by a claim that **ExoMiner++** and a paper called **"PlanetNet-MMG"**
reach ~0.973 AUC by fusing Kepler and TESS data. Closed at the deduplication
gate: the proposal reduces to the **K2 pooling experiment already run**, whose
pooled arm is *literally* ExoMiner++'s design. **Production untouched at
0.9300 / 31 features / md5 `1f0b7cb8e78ab542374eaf78fc837a6f` (verified live).**

### Part 0 -- citation verification: PlanetNet-MMG IS REAL. Its number is NOT verifiable.

The brief flagged this as possibly hallucinated. **It is not.** It is a real,
registered journal article:

| field | value |
|---|---|
| title | *PlanetNet-MMG: A robust multi-modal graph-based deep learning model for exoplanet candidate classification* |
| DOI | `10.1016/j.eswa.2026.132396` |
| journal | Expert Systems with Applications, vol. **324**, art. 132396 |
| published | 2026-04-18 (issue dated 2026-08) |
| authors | Dubey, Behera, Rout, Umer, Jain, Andreu-Perez |

Confirmed independently in **Crossref** and **OpenAlex**. Why general web
search missed it: it is ~4 months old, **not on arXiv** (the only arXiv
"PlanetNet" is an unrelated Saturn-mapping paper), and Elsevier-paywalled.

**Two process notes worth recording, because both were nearly errors.**

1. The web-search *summariser* twice asserted things its own returned links did
   not support -- first glossing the acronym ("multi-modal graph") as if
   reporting a finding, then emitting a specific ScienceDirect URL absent from
   the result list. Both later proved *correct*, but they were unsupported at
   the time. Crossref/OpenAlex, not the summariser, are what settled this.
2. My first arXiv API calls returned empty and I nearly logged that as evidence
   of absence. They were **HTTP 301s I wasn't following**. A false negative,
   not a null result.

**The ~0.973 figure remains UNVERIFIED and is not used as evidence anywhere
below.** OpenAlex reports `is_oa: false`, `oa_status: closed`, no abstract, no
repository fulltext; ScienceDirect returns 403. Whether the paper even *fuses
Kepler with TESS* could not be established -- "multi-modal" in the title may
refer to modalities (flux / centroid / stellar params), which is what
ExoMiner++ and ExoNet mean by it, not to missions. **Nothing about this
investigation's conclusion rests on it.**

### Part 0 -- ExoMiner++: the premise was wrong. It does NOT transfer-learn.

The brief stated ExoMiner++ "pretrains a deep neural network on Kepler, then
transfer-learns to TESS." **Direct check of arXiv:2502.09790 says otherwise.**
In the authors' own words:

> "a simpler approach of combining Kepler and TESS data to create a larger
> training set proved more effective"

> "we incorporated Kepler data into the training set of all cross validation
> iterations, ensuring that the validation and testing were performed only on
> TESS data."

That is **pooled multi-source training, not pretrain-then-fine-tune** -- and
the phrase "proved more effective" implies the more complex alternative was
tried and lost. So the one verifiable citation here is, if anything, **evidence
against** the transfer-learning framing it was offered to support.

This cuts both ways and the second half matters more:

* It **weakens** the "architecturally inapplicable" objection. Pooling rows and
  evaluating only on target is learner-agnostic; it needs no differentiable
  representation and works fine for trees. The domain-adaptation closure does
  **not** dispose of it.
* It **strengthens** the duplication objection, fatally. Because pooling is
  exactly what this project already ran.

### Part 1 -- the proposal maps onto the K2 investigation, not the transfer-learning one

ExoMiner++'s design placed against the K2 experiment's arms:

| | ExoMiner++ | this project's K2 run |
|---|---|---|
| non-TESS rows into training | yes | **Arm B, yes** |
| eval on target mission only | yes | **yes** -- "K2 rows in the test set are excluded from every evaluation" |
| reweighting variant | -- | **Arm C**, K2 down-weighted 0.25 |

**Same design.** Measured result, frozen 1,098-star test set:

| arm | test AUC | delta [95% CI] |
|---|---|---|
| A. baseline (TESS only) | 0.8986 | -- |
| B. pooled (TESS + K2) | 0.9016 | +0.0030 [-0.0024, +0.0086] |
| C. K2 down-weighted 0.25 | 0.9030 | +0.0044 [-0.0013, +0.0105] |

Neither clears `ci_lo > 0`. Domain AUC **0.9973**, still the highest ever
measured here.

**Verdict on the two sub-questions the brief asked:**

* *"Incorporate Kepler/K2 data and labels into training"* -> **already tested.
  Inert.** This is the K2 pooling experiment, re-proposed.
* *"A transfer-learning step"* -> **already closed**, architecturally, and
  additionally **not what ExoMiner++ does**. Two independent reasons.

### Would better feature standardisation change it? No -- and the reason isn't separability.

This is the one genuinely new element, so it gets a real answer rather than a
dismissal. The mechanism is superficially plausible: 4 of K2's top-5
separability drivers are detection statistics that scale with baseline and
cadence (`FAP` -1.24 SD, `SDE_raw` +0.91, `SDE` +0.86, `distinct_transit_count`
+0.77), exactly what per-mission standardisation would target. Three reasons it
still fails:

1. **There is no harm to remove.** K2 pooling was *inert, not harmful* -- and
   the K2 write-up already corrected this project's own overreading of domain
   AUC: separability predicts "no reliable gain", not "damage." Lowering domain
   AUC removes a penalty that was never charged.
2. **Separability isn't concentrated anywhere removable.** Measured on the
   synthetic case: deleting *all* detection features still leaves 0.9500;
   deleting *all* shape features still leaves 0.9382. There is no subset to fix.
3. **The binding constraint is row count, and it is arithmetic.** 56 K2 rows on
   4,387 is +1.3%. No amount of feature alignment changes that.

### The decisive number: even a FULL Kepler pull cannot clear this test set

Using this project's own fitted learning curve (`A=1.0, B=0.5129, c=0.1930`;
refit here and reproducing the logged extrapolation exactly -- n=9,564 ->
0.91252 vs 0.91252 logged, required_n(0.91)=8,256 vs 8,256 logged):

| added rows | total n | predicted AUC | gain | vs MDE 0.0097 |
|---|---|---|---|---|
| 56 (the K2 pilot) | 4,550 | 0.8990 | +0.0002 | below |
| 1,000 | 5,494 | 0.9026 | +0.0038 | below |
| **1,700** (full Kepler pull, ~5.8 days compute) | 6,194 | 0.9049 | **+0.0061** | **below** |
| 3,000 | 7,494 | 0.9083 | +0.0095 | below |
| 4,494 (doubling) | 8,988 | 0.9115 | +0.0127 | clears |

The Kepler pilot's **36.4% yield wall** (confirmed not a fixable bug) caps a
realistic full pull at **~1,700 usable rows**, predicting **+0.0061 -- below
this test set's ~0.0097 detection threshold.** Roughly a *doubling* is needed
to clear.

**And that row is an over-estimate twice over**: the learning curve describes
*same-distribution* data, which Kepler at domain AUC ~0.97 is not, and its
asymptote is unidentifiable. The honest reading: a full Kepler pull is
**unmeasurable on this test set even under assumptions known to be too
generous.** That is a stronger and more quantitative statement than the
original Kepler closure, which rested on cost and yield rather than on a
ceiling proof.

### Why ExoMiner++'s result doesn't transfer, in one line

ExoMiner++ pools **tens of thousands** of Kepler TCEs into a deep multi-branch
CNN. This project can reach ~1,700 rows into a tree model whose test set cannot
resolve +0.0061. The mechanism is real; the **scale** that makes it pay is not
available here.

### Verdict

**CLOSED at Part 1. Part 2 not entered** -- its precondition (a technique that
is neither re-mixing nor neural) is not met. Nothing built, nothing downloaded,
no training data / split / model / pipeline touched.

| proposal element | maps onto | status |
|---|---|---|
| mix Kepler/K2 into training, eval on TESS | K2 pooling, Arms B/C | closed -- inert, no arm clears |
| + better feature standardisation | -- | assessed above; cannot help an inert result |
| transfer-learning step | domain-adaptation closure | closed -- and not ExoMiner++'s method |
| PlanetNet-MMG's ~0.973 | -- | real paper, **unverifiable number**, not used |

**Do not propose cross-mission training a third time without new information.**
The specific new information that would justify reopening: a way past the
**36.4% Kepler yield wall** that delivers **>3,000 usable rows** (below that the
predicted gain is under the MDE regardless of how the data is standardised or
weighted), or a larger test set that lowers the ~0.0097 MDE.

## "SYNTHETIC INJECTION" IS THREE DIFFERENT PROPOSALS. Three separate verdicts.

Assessed as three mechanistically distinct things, because they are routinely
bundled and should not be. **The closure of (1) does not close (3)** -- one
modifies training data, the other never touches it.

| # | proposal | verdict |
|---|---|---|
| 1 | inject synthetic transits/FPs to EXPAND TRAINING DATA | **CLOSED (third time). No compute spent.** |
| 2 | VAE anomaly-detection framework | **NOT CLOSED, but do not build.** Needs explicit go-ahead. |
| 3 | injection-recovery as a SENSITIVITY VALIDATION tool | **RUN. Results below.** Useful, ongoing. |

Production verified live and untouched throughout: **0.9300 / 31 features /
md5 `1f0b7cb8e78ab542374eaf78fc837a6f`**.

### Proposal 1 -- training-data augmentation: same closure, third proposal

Not materially different from what closed twice (v1 domain AUC 0.9654, harmful
-0.0180 at full dose with a clean dose-response; v2/RAVEN closed at the
feasibility gate). The brief's new element is "enrich rare-planet examples at
LOW SNR". That does not touch any of the three mechanisms that killed it, and
it makes one of them worse:

1. **Separability is not an SNR-regime property.** Splitting the features by
   how the signal was FOUND vs what it LOOKS like gave 0.9382 (detection only)
   and 0.9500 (shape only) -- **independently** near-total. Retargeting the
   injected SNR moves where rows sit in detection space; it creates no overlap
   in shape space. There was no repairable subset before and re-aiming does not
   create one.
2. **`st_teff` is invariant to it.** SMD +0.44, single-feature domain AUC
   0.7284 -- a *host-population* property. Injecting at different SNR does not
   change WHICH stars are injected into. No injector setting touches this.
3. **The 7 uncomputable features are invariant to it.** Crowding is a catalog
   property of the real host; the 5 variability features are computed from the
   RAW pre-flatten curve and injection happens post-flatten, so they cannot be
   computed at all. Still 23% of the feature space, still including both
   deployed wins, regardless of SNR target.

**And the new framing is actively self-defeating, which is the one genuinely
new argument here.** Measured recovery vs expected SNR: **4.5% below SNR 5**,
27.3% at SNR 5-10. Aiming at low SNR means (a) very few usable rows survive,
and (b) far worse, **the ones that survive are exactly those where noise
fluctuated favourably.** That is a biased draw whose detection statistics are
systematically inflated relative to a real low-SNR planet -- it *deepens* the
very SDE/SDE_raw/snr shifts (SMD +0.78 / +0.76 / +0.47) that drive the
separability. Part 3 below re-measures this wall on the deployed pipeline and
confirms the regime is where recovery collapses.

**No compute spent re-measuring domain AUC**, per the brief: the mechanism is
unchanged, so the measurement would be too. Documented as proposed-and-rejected.
**Do not propose a fourth time** absent a genuinely different generator (PASTIS
or equivalent forward-modelling of the full pixel-level scenario), which is the
same condition the v2 closure set.

### Proposal 2 -- VAE anomaly detection: NOT closed, but do not build yet

Deliberately not auto-closed just because it is neural. Two findings decide it.

**Finding 1: this capability already exists in production, in feature space.**
`06_download_unknown.py` deploys an **IsolationForest** multivariate OOD
detector (`multivariate_ood_flag`, 200 estimators, contamination 0.02, with the
flag threshold *calibrated against real training data rather than assumed*),
plus a separate univariate in-distribution check. "Flag candidates outside the
model's competence" is therefore **not a missing capability** -- a VAE would be
a different *representation* (raw light curve rather than the 31 features) of
something already deployed. That reframes the question from "should we add
anomaly detection" to the much narrower "are there light-curve-space anomalies
that feature-space OOD misses?", which is a real but far smaller question.

**Finding 2: the data-scale objection is weaker than the CNN result suggests --
but a different objection is fatal.**

Honest assessment of the scale question the brief asked, both directions:

* *In favour:* a VAE genuinely is less label-hungry. It needs no labels at all,
  so it can train on the unlabelled candidate pool as well as the 5,486 labelled
  stars. And reconstruction supplies thousands of regression targets per light
  curve rather than one bit, which really does lower sample complexity relative
  to binary classification. **The CNN's 0.6964 does not automatically transfer.**
* *Against, and decisive:* **reconstruction error is dominated by whatever
  carries the most variance, and in a TESS light curve that is instrumental
  systematics and stellar variability -- not a 84-5,000 ppm transit.** This
  project has direct evidence for exactly that: the single biggest recent win
  was the 5 stellar-variability features (+0.0092 to production), promoted
  *because* out-of-transit scatter and Lomb-Scargle structure vary strongly
  star to star. A reconstruction-based detector would rediscover that
  variability and call it anomaly. The transit is a rounding error in the loss.

**Finding 3: the validation problem is unsolved, and the brief is right that it
must be settled first.** There is no ground-truth set of "interesting
anomalies" here. The only tractable proxy -- "does it flag known planets?" --
is the classification task in disguise, which the deployed model already does
at 0.9300.

**VERDICT: real idea, wrong order of magnitude of effort for the expected
payoff. Do not build. Requires explicit go-ahead.** Not closed, because the
label-efficiency argument above is genuine and the CNN result does not settle it.

**If it is ever approved, this is the bounded pilot -- criteria pre-registered
BEFORE building, since this is not a classification-accuracy question:**

* *Scope, ~1-2 days.* Data prep is nearly free: `phase_fold_views.py` and
  `cnn_dataset.npz` (6.8 MB, local+global folded views) already exist from the
  CNN work. Train a small 1D-conv VAE on those views; no new dependency.
* *Success criterion 1 (does it see transits at all).* Reconstruction error
  must separate injected-transit curves from no-transit curves at **AUC > 0.75**.
  Directly checkable with the Part 3 tooling below. Below that it is not
  detecting transits and nothing else matters.
* *Success criterion 2 (is it additive).* Its flags must disagree with the
  deployed `multivariate_ood_flag` on a material fraction of the candidate
  pool. Full agreement means it is a slower reimplementation of a shipped model.
* *Success criterion 3 (is it meaningful).* Among pool candidates, VAE-flagged
  anomalies must be enriched in independently-vetted-bad candidates relative to
  the pool baseline -- the same evidence cross-check design used for cluster 1.
* *KILL CRITERION, pre-registered.* If reconstruction error correlates with
  `var_oot_rms` at **|r| > 0.6**, it is a stellar-variability detector, not an
  anomaly detector. Stop there and report; do not tune around it.

### Proposal 3 -- injection-recovery as a SENSITIVITY TOOL: RUN. This one is useful.

540 trials, 0 errors, 6.8 core-hours, median 28 s/trial. New script
`injection_recovery_sensitivity.py`; raw
`injection_recovery_sensitivity_results.csv`, summary `..._summary.json`.
**No training data, split, model or production code touched.**

Runs the FULL deployed chain -- inject -> production TLS invocation -> 31-feature
vector -> deployed model -> score -- and reports detection and classification as
SEPARATE stages. The existing `completeness_curve.py` stops at detection.

**Why the synthetic-feature objection does not apply here.** The host is a REAL
star, so the 31 features split cleanly: 22 TLS-derived features are recomputed
from the injected curve, and the 9 host-derived ones (`st_rad`, `st_teff`,
crowding x2, variability x5) are that real star's REAL measured values, read
from `training.csv`. Nothing is invented. The "7 of 31 cannot be given honest
synthetic values" blocker is specific to fabricating a training row; it does not
bind when characterising sensitivity on a real host. Hosts are drawn from the
**216 frozen TEST-split negative-class stars** with complete host features, so
the classifier never trained on them.

**Two deliberate design fixes over the old script**, both of which make these
numbers MORE conservative and more honest:

* *Duration from physics, not 5% of period.* `completeness_curve.py` fixes
  duration at `0.05 * period`, which at P = 16 d implies a **19-hour** transit --
  physically absurd, and it makes long periods look far easier than they are.
  Here `a/R*` comes from Kepler's third law using the host's real M*/R*.
* *A zero-depth control arm.* Without it the classification numbers are
  uninterpretable. It turned out to be the most informative part of the run.

#### Stage 1 -- TLS detection completeness by depth (n=60/depth, 95% binomial CI)

| depth (ppm) | median implied Rp | detected | 95% CI |
|---|---|---|---|
| **84 (Earth / Sun-like)** | **1.36 R_e** | **0/60 = 0.000** | **[0.000, 0.060]** |
| 150 | 1.96 R_e | 3/60 = 0.050 | [0.010, 0.139] |
| 250 | 2.60 R_e | 6/60 = 0.100 | [0.038, 0.205] |
| 400 | 3.28 R_e | 8/60 = 0.133 | [0.059, 0.246] |
| 700 | 4.36 R_e | 23/60 = 0.383 | [0.261, 0.518] |
| 1200 | 4.53 R_e | 28/60 = 0.467 | [0.337, 0.600] |
| 2500 | 8.51 R_e | 44/60 = 0.733 | [0.603, 0.839] |
| 5000 | 11.56 R_e | 40/60 = 0.667 | [0.533, 0.783] |

**An Earth-size transit is not detected at all -- 0 of 60, upper CI 6%.** The
50% detection point sits between 1,200 and 2,500 ppm, i.e. **a Neptune-to-
Jupiter-size planet**. (The 2,500 > 5,000 inversion is noise; the CIs overlap
heavily at n=60. Read the marginals, not individual cells, which are n=10.)

Worse than a clean miss: at 84 ppm TLS still returns a **median SDE of 9.55**
on a *spurious* signal (median recovered period 3.03 d, unrelated to the
injected one), and the deployed model then scores **70% of those above the
triage floor**. The pipeline does not report "nothing found" -- it confidently
reports something else.

#### Stage 1b -- the long-period wall is STRUCTURAL, not a sensitivity limit

TLS's default `period_max` is ~half the baseline (it requires >=2 transits).
Host baselines are single-sector TESS, median 24.9 d. Measured per trial:

| injected P (d) | median transits | **injected P inside TLS's searched grid** | detected | of which EXACT period | alias-only |
|---|---|---|---|---|---|
| 1 | 25.0 | 1.000 | 0.425 | 0.225 | 0.200 |
| 3 | 8.4 | 1.000 | 0.388 | 0.375 | 0.013 |
| 6 | 4.3 | 1.000 | 0.388 | 0.325 | 0.062 |
| 10 | 2.6 | 0.988 | 0.362 | 0.362 | 0.000 |
| **14** | 1.8 | **0.062** | 0.212 | 0.062 | 0.150 |
| **20** | 1.3 | **0.000** | 0.125 | **0.000** | 0.125 |

Median searched `period_max` = **12.5 d**. Above it the signal is
**unsearchable, not merely faint** -- no depth fixes this. The apparent
detections at P = 14/20 are almost entirely **aliases** (14 half-period, 8
third-period, 5 exact); at P = 20 the exact-period rate is **0.000**. So beyond
~12.5 d the pipeline does not just miss the planet, it reports the **wrong
period**. Aliasing is not confined to long periods either: at P = 1 d, only
0.225 of 0.425 "detections" are the true period.

#### Stage 2 -- and the control arm changes how to read all of it

| population | n | median score | >= 0.30 triage floor |
|---|---|---|---|
| **zero-depth CONTROL (nothing injected)** | 60 | **0.503** | **70%** |
| detected injections, <300 ppm | 9 | 0.490 | 67% |
| detected injections, 300-1,500 ppm | 59 | 0.224 | **36%** |
| detected injections, >=1,500 ppm | 84 | 0.482 | 79% |

**Real negative-class stars with NO transit injected pass the triage floor 70%
of the time.** Detected injections in the 300-1,500 ppm range score *below*
that. So the Stage-2 percentages are **not** evidence the classifier recognises
injected transits -- at the triage floor it barely separates them from nothing.

**This is consistent with the deployed operating point, not a new defect.**
Checked directly against the frozen test set: at threshold 0.30, **56.3% of
real label-0 rows already score above it** (median label-0 score 0.353) in
exchange for **99.3% recall**. The control's 70% [~57-81% at n=60] is the same
number within sampling error. It is the deliberate, documented cost of an
F2-optimal recall-weighted floor.

What this run adds is a **measured** figure for a caveat `06_download_unknown.py`
had only stated qualitatively ("precision figures are an upper bound and do NOT
transfer to deployment; recall transfers, precision does not"). It now has a
number, on held-out stars.

The 300-1,500 ppm dip below control is interpretable rather than paradoxical: a
marginal injection gives TLS weak, inconsistent statistics that the model
correctly reads as unconvincing, whereas a control host is a **TOI false
positive** whose strongest real signal is often genuinely sharp -- so TLS finds
that instead and it scores well.

#### End-to-end, and what it means

| depth | detected AND >= 0.30 |
|---|---|
| 84 ppm | **0.000** |
| 150-400 ppm | 0.050 - 0.067 |
| 700-1,200 ppm | 0.133 - 0.150 |
| 2,500-5,000 ppm | 0.500 - 0.600 |

**End-to-end sensitivity is governed almost entirely by TLS detection**, since
the classifier's floor passes most things that reach it. Improving the
classifier does not move this curve; improving detection, or the baseline, does.

#### Honest limits of this measurement

* **These are BEST-CASE numbers.** An injected batman transit has no TTVs, no
  spot crossings, no correlated residual from the host's own activity. The
  domain-separability results (0.95-0.97 on shape AND detection features
  independently) are direct evidence injected and real transits stay
  distinguishable. **Recovering an injected Earth is not the same as recovering
  a real one** -- the real system does no better than this, plausibly worse.
* **Hosts are TOI false positives, not random field stars.** They are the
  adversarially hard confusers, often variable. "70% of controls pass triage"
  means 70% *of known TOI FPs*, not of the sky.
* n = 10 per grid cell. Only the marginals (n = 60/80) carry weight.
* Single-sector baselines only. Multi-sector hosts would push `period_max` out
  proportionally -- which is exactly what the existing multi-sector
  strengthening action does, and this quantifies why it matters.

#### Verdict on Proposal 3

**Keep it. This is a genuinely useful, repeatable characterisation tool** and
the only one of the three proposals that should be run again -- after any
detection-stage change, or on multi-sector data to confirm the `period_max`
wall moves as predicted. It answers "what can this system actually find?", which
none of the classifier-accuracy metrics address.

Three concrete things it established that were not previously measured here:
**(1)** Earth-size is a hard zero, 0/60, not merely hard; **(2)** there is a
**structural ~12.5-day period ceiling** from TLS's default grid, above which
detections are aliases at the wrong period; **(3)** the triage floor's
false-positive cost now has a measured value on held-out real negatives.
