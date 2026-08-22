# Model architecture experiments -- honest results summary

> **CURRENT PRODUCTION (2026-08-15): AUC 0.9454, 33 features, md5
> `fe3fa82f36cc978396c68be07d6057f9`.** Everything in this header block below is
> the 2026-08-05 state and is retained as history, NOT as current fact. Four
> deployments have happened since: crowding (0.9208), variability (0.9300), Gaia
> DR3 astrometry (0.9402), and the Optuna hyperparameter change (0.9454).
>
> **The most recent one did NOT clear the standing promotion bar and was
> deployed as an explicit, user-authorised exception.** See the section
> `>>> DEPLOYED 2026-08-15: A DELIBERATE EXCEPTION TO THE PROMOTION RULE <<<`
> at the end of this document before citing it. The bar itself is unchanged.

**DEPLOYED 2026-08-05 (HISTORICAL): the number of record was then 0.9208.**
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

## PERIOD CEILING vs BASELINE -- PREDICTION CONFIRMED. The ceiling is real and it moves.

Direct follow-up to the injection-recovery sensitivity run above, which found a
**structural ~12.5 d period ceiling** on single-sector TESS. Question: does
TLS's `period_max` bound extend proportionally on a longer contiguous baseline?
**Yes, exactly as predicted.** 700 trials, **0 errors**, 29.4 core-hours.
Scripts `injection_recovery_widesector.py` (a thin reuse of
`injection_recovery_sensitivity.py` -- same injector, same production TLS call,
same 22-TLS/9-host feature split, same deployed model, same control arm; only
the host pool and period grid differ), pool `widesector_host_pool.csv`, raw
`injection_recovery_widesector_results.csv`.

**Production untouched: 0.9300 / 31 features / md5 `1f0b7cb8e78ab542374eaf78fc837a6f`,
verified before and after.**

### THIS IS NOT THE CLOSED MULTI-SECTOR STACKING INVESTIGATION

Recorded explicitly so a future reader does not conflate them:

| | closed multi-sector stacking | **this run** |
|---|---|---|
| operation | **FOLD** a transit at a STORED ephemeris across sector gaps | **CONCATENATE** sectors into one time array; TLS does its own blind period search |
| failure mode | phase error accumulates as `span x sigma_P / P` in transit durations | n/a -- **no stored ephemeris is used at all** |
| quantity tested | can a known signal be coherently stacked? | what is `period_max`, a function of `max(t) - min(t)` alone? |

The closed result was a data-quality problem about ephemeris precision. This is
a property of TLS's period grid. They are independent, and this run does not
reopen that closure.

### The sample: 76.3 d contiguous TESS, measured not assumed

`data/processed_unknown_widesector/` -- 3 consecutive sectors already
concatenated by `06_download_unknown.download_one_star`'s multi-sector branch
(`pd.concat` + sort-by-time). Three filters, each measured:

1. **Continuous**: 70-85 d span AND max internal gap < 5 d. Excludes stars
   whose span crosses a year-long gap between non-consecutive sectors, which
   would hand TLS an enormous period grid over almost no data. Result: span
   **76.0 d median, max gap 2.17 d, 6 gaps > 0.5 d** (TESS downlinks + sector
   breaks), **86% duty cycle**.
2. **Flux-clean**: median flux within 1% of 1.0, robust sigma in (0, 0.05),
   <1% of points beyond 10 sigma. **~36% of this pool fails this** (raw std up
   to ~2,000) where the single-sector negative pool does not -- a pre-existing
   data-quality tail in the wide-sector pool, recorded here as an incidental
   finding. Excluded so baseline length is the ONLY difference from the prior run.
3. All 9 host features finite.

**133 hosts** survive all three (vs 216 single-sector).

### The analytical prediction, stated before the grid ran

TLS's default `period_max` = `(max(t) - min(t)) / 2` (it requires >= 2 transits):

| | baseline | predicted `period_max` | **measured** |
|---|---|---|---|
| single sector | 25.3 d | 12.7 d | **12.66 d** |
| wide sector | 76.3 d | **38.2 d** | **38.15 d** |

Confirmed at the grid level on one real curve *before* launching: a 76.2 d host
searched 0.532 -> **38.105 d** vs 76.2/2 = 38.1. The grid moves. Whether
DETECTION follows is a separate question -- a period can sit inside the grid and
still be undetectable for want of transits or SNR -- which is what the 700
trials test.

### THE ANSWER: side by side, same grid, exact-period detection

| P (d) | \|  single-sector: in-range / detected / **EXACT** | \|  wide-sector: in-range / detected / **EXACT** |
|---|---|---|
| 1 | 1.000 / 0.425 / **0.225** | 1.000 / 0.475 / **0.338** |
| 3 | 1.000 / 0.388 / **0.375** | 1.000 / 0.475 / **0.463** |
| 6 | 1.000 / 0.388 / **0.325** | 1.000 / 0.312 / **0.287** |
| 10 | 0.988 / 0.362 / **0.362** | 1.000 / 0.412 / **0.325** |
| **14** | 0.062 / 0.212 / **0.062** | 1.000 / 0.338 / **0.263** |
| **20** | 0.000 / 0.125 / **0.000** | 1.000 / 0.200 / **0.200** |
| **30** | *unsearchable* | 1.000 / 0.225 / **0.212** |
| **40** | *unsearchable* | **0.000** / 0.100 / **0.000** |

With 95% binomial CIs at the decisive points (n=80 each):

| P | single-sector EXACT | wide-sector EXACT |
|---|---|---|
| 14 | 5/80 = 0.062 [0.021, 0.140] | **21/80 = 0.263 [0.170, 0.373]** |
| **20** | **0/80 = 0.000 [0.000, 0.045]** | **16/80 = 0.200 [0.119, 0.304]** |
| **30** | not searchable at all | **17/80 = 0.212 [0.129, 0.318]** |
| **40** | not searchable at all | **0/80 = 0.000 [0.000, 0.045]** |

**P = 20 d goes from a hard zero to 0.200, with non-overlapping CIs.** P = 14
roughly quadruples, also non-overlapping. P = 30 -- entirely unsearchable
before -- lands at 0.212, statistically indistinguishable from P = 20. And
**P = 40 is the new hard zero**, `in_range` 0.000, with detections that are
100% aliases: exactly the signature P = 20 showed on single-sector data. The
alias tell reproduces at the new boundary.

**The ceiling is real, structural, and moves proportionally with baseline. Both
the old and the new ceiling behave identically at their respective boundaries.**

### Two things that did NOT change much, reported plainly

**Depth.** 84 ppm went 0/60 -> **2/80 (0.025)**. Directionally consistent with
more transits to stack, but this is not a meaningful rescue, and per the brief
it was included for consistency rather than as the question. Note the implied
planet is not even the same size: these hosts are larger (**median st_rad 1.90
vs 1.48**), so 84 ppm here means **1.81 R_e** vs 1.36 R_e before. The
Earth-size floor stands.

**The classifier stage still contributes nothing at the triage floor -- and here
it is starker.** The zero-depth control on this pool scores **95.0% above 0.30**
(median score 0.796), against 70.0% (median 0.503) single-sector. So the
Stage-2 "classified | detected" table, which is ~1.00 nearly everywhere, is
**not** discrimination -- the model passes essentially everything in this
population. Two causes, both real: a different host population (unknown-pool
stars, larger radii) and a 3x longer baseline giving more transits, higher SDE,
and therefore higher scores across the board. **End-to-end sensitivity remains
governed almost entirely by TLS detection**, which is the same conclusion the
single-sector run reached, arrived at from a different direction.

### Limits of this comparison

* **The period axis is clean; the depth and classification axes are not.** Host
  population differs (unknown-pool vs TOI false positives; st_rad 1.90 vs 1.48).
  `period_max` is a pure function of `max(t) - min(t)`, so the period result is
  unaffected; absolute depth rates and all Stage-2 numbers are population-shifted
  and should not be read as a like-for-like depth comparison.
* Still best-case: injected batman transits have no TTVs, no spot crossings.
* n = 10 per cell; only the marginals (n = 80) carry weight.
* At P = 30-40 there are only **2.5 and 1.9 transits** in the baseline -- right
  at TLS's 2-transit floor, so detection there is inherently marginal even when
  searchable.

### Does this change the operational case for longer baselines? YES -- and it is a COVERAGE argument, not an accuracy one

**The detection-ceiling argument stands on its own, independent of AUC.** With
single-sector data the pipeline is *structurally incapable* of finding anything
beyond ~12.5 d. That is not a threshold to tune or a model to improve: the
search never looks there. No classifier improvement can recover a signal that
was never searched for, which is why this argument is orthogonal to the closed
question of whether longer baselines move AUC.

Concretely, 12.5 d -> 38.2 d is the difference between searching only very
short-period planets and covering the M-dwarf habitable-zone range. Tripling
the baseline triples the ceiling, exactly and predictably.

**Stated honestly, the gain is coverage, not sensitivity.** At periods already
searchable (P <= 10) the improvement is modest and within noise (0.425 -> 0.475
at P = 1). The real prize is the newly-searchable 12.5-38 d band, where
single-sector detection was **identically zero**. And it does not touch the
depth floor: Earth-size remains ~0.025.

**Recommendation: treat multi-sector coverage as a detection-reach lever, and
prioritise it on that basis alone.** The existing per-candidate multi-sector
strengthening action is the right mechanism and this quantifies what it buys.
The natural next extension, if ever wanted, is 6-13 consecutive sectors
(~160-350 d, `period_max` ~80-175 d) -- but note the 2-transit floor means
usable detection will fall off well before the nominal ceiling, as the P = 30-40
rows here already show.

### Follow-up: is the baseline comparison confounded by CADENCE? NO -- hypothesis refuted

A concern was raised (by me) that the single-sector vs 3-sector comparison above
was not a pure baseline comparison, because production's `bin_lightcurve`
targets a FIXED 15,000 points, so effective cadence degrades with baseline:

    1 sector  ~15,700 pts  bin factor  1  ->   2 min   (60 samples in a 2 h transit)
    3 sectors ~46,700 pts  bin factor  4  ->   8 min   (15 samples)
    6 sectors ~96,000 pts  bin factor  7  ->  14 min   ( 8.6 samples)
    13 sectors ~208,000 pts bin factor 14 ->  28 min   ( 4.0 samples)

**Tested directly, and the concern does not hold.** Paired design on
single-sector hosts (which sit below production's 30,000-point threshold and so
already get native 2 min): re-run the identical seeds with bin factor 4 forced,
giving 8 min -- exactly what production does to a 3-sector curve. Seeds fix the
host draw, t0 and impact parameter, so every trial is an exact paired twin and
**only cadence differs**. 160 trials, 153 matched pairs, 5.6 min.

| depth (ppm) | 2-min | 8-min | delta |
|---|---|---|---|
| 700 | 0.395 | 0.421 | +0.026 |
| 1200 | 0.425 | 0.425 | 0.000 |
| 2500 | 0.658 | 0.684 | +0.026 |
| 5000 | 0.757 | 0.784 | +0.027 |
| **pooled (n=153)** | **0.556** | **0.575** | **+0.020** |

Only **5 discordant pairs of 153** (8-min-only wins 4, 2-min-only wins 1),
**McNemar exact p = 0.375**. Median SDE 10.98 -> 11.25.

**Coarser cadence does not hurt exact-period detection over 2 -> 8 min; if
anything it marginally helps** (binning lowers per-point noise, and TLS fits a
duration grid rather than relying on dense in-transit sampling). So the
single-sector vs wide-sector result above **stands as reported** -- it is not a
disguised cadence effect.

**Limit, stated so it is not over-extended:** only 2 -> 8 min was tested. At 13
sectors production would impose ~28 min, where a 2 h transit is 4 samples; this
result does not license extrapolating there. (Moot in practice -- see the
availability finding below.)

**One method note worth recording.** The obvious version of this test -- take
76 d curves and DISABLE binning to recover 2-min cadence -- was attempted and
**abandoned as unaffordable**: at native cadence TLS's period grid over 76 d is
enormous, and the run went 10 trials in 23 min then 10 more in 193 min,
projecting past 7 hours. Killed. Running the comparison in the cheap direction
(ADD binning to a short curve rather than REMOVE it from a long one) answers
the identical question at single-sector cost. Script
`injection_recovery_cadence_arm.py`; the abandoned one is
`injection_recovery_binning_arm.py`, kept for the record.

### Availability finding: 13 consecutive sectors essentially does not exist

Probed MAST live for SPOC 120 s targets near the south ecliptic pole (the best
case for continuous coverage), and counted the **longest CONSECUTIVE run** per
target -- total sector count is misleading, since TESS revisits the CVZ each
year and a 60-sector star may have only ~13 in a row separated by year-long
gaps (a gap that would hand TLS a huge period grid over sparse data, the same
pathology filtered out of the K2 pool earlier).

Of **827** targets with 2-min SPOC data:

| longest consecutive run | targets |
|---|---|
| >= 6 | **333** |
| >= 8 | **135** |
| >= 10 | 3 |
| >= 12 | 3 |
| >= 13 | **1** |

**So a 13-sector (~350 d) sample is not obtainable; 6-8 sectors (~160-215 d,
predicted `period_max` ~80-107 d) is the realistic ceiling.**

*(An earlier version of this probe reported "max 4 consecutive" for every
target. That was a bug: the sector regex `s(\d{4})` matched the YEAR in
`tess2018206045859-...` rather than the sector in `-s0001-`. Corrected to
`-s(\d{4})-`. The 4-sector figure was an artifact and is void.)*

## 8-SECTOR EXTENSION: the ceiling law holds a THIRD time -- and one earlier claim of mine was WRONG

Third point on the period-ceiling curve, on a real 216-day contiguous TESS
baseline. 320 trials, **0 errors**, 27.3 core-hours. Scripts
`eightsector_build_pool.py` (select/download/preprocess/features) and
`eightsector_run.py` (grid, a thin reuse of `injection_recovery_sensitivity.py`).
**Production untouched: 0.9300 / 31 features / md5
`1f0b7cb8e78ab542374eaf78fc837a6f`, verified before and after.**

### The sample, and why 8 and not 13

45 CVZ stars downloaded at 8 consecutive sectors each (7.9 min, 1.9 GB, 45/45
success). Filters -- continuity (gap < 10 d, duty > 0.7), flux-clean, all 9 host
features -- leave **39 hosts**:

| | 1 sector | 3 sectors | **8 sectors** |
|---|---|---|---|
| baseline | 25.3 d | 76.3 d | **216.5 d** |
| max gap | -- | 2.2 d | 6.2 d |
| duty cycle | -- | 0.86 | **0.83** |
| st_rad median | 1.48 | 1.90 | **1.36** |
| effective cadence (production binning) | 2 min | 8 min | **18 min** |

`st_rad` 1.36 makes this a **closer population match to the single-sector pool
(1.48) than the wide-sector pool was (1.90)** -- the depth-axis confound flagged
in the previous section is smaller here.

13 sectors was ruled out on measurement, not preference: only 1 of 827 CVZ
targets has a run that long (see the availability table above).

Sectors are concatenated time-ordered and TLS runs its own blind period search.
**No folding at a stored ephemeris** -- the closed stacking investigation is
still not reopened.

### THE LAW: predicted = measured, three times, to two decimals

    period_max = (max(t) - min(t)) / 2

| baseline | predicted | **measured** |
|---|---|---|
| 25.3 d | 12.66 | **12.66** |
| 76.3 d | 38.16 | **38.15** |
| **216.5 d** | **108.23** | **108.22** |

And each baseline's hard zero sits exactly at its own ceiling: exact-period
detection is **0.000** at P=20 (1 sector), P=40 (3 sectors), P=120 (8 sectors) --
and `in_tls_range` is 0.000 in every one of those cells. Three independent
confirmations of the same structural bound.

### Exact-period detection, all three runs

| P (d) | 1 sector | 3 sectors | 8 sectors |
|---|---|---|---|
| 3 | 0.375 | 0.463 | **0.700** |
| 10 | 0.362 | 0.325 | **0.650** |
| 20 | **0.000** | 0.200 | **0.625** |
| 40 | -- | **0.000** | 0.550 |
| 60 | -- | -- | 0.550 |
| 90 | -- | -- | **0.475** |
| 120 | -- | -- | **0.000** |

### CORRECTION: "the gain is coverage, not sensitivity" was WRONG

The previous section concluded, from the 3-sector data, that longer baselines
buy **coverage** of newly-searchable periods but little sensitivity where the
search already reached. **At 8 sectors that is clearly false.** At periods
searchable by ALL THREE baselines:

| P | 1 sector | 8 sectors | Fisher exact |
|---|---|---|---|
| 3 d | 30/80 = 0.375 | 28/40 = **0.700** | **p = 0.0010** |
| 10 d | 29/80 = 0.362 | 26/40 = **0.650** | **p = 0.0036** |

Detection nearly **doubles** at already-searchable periods, significantly. The
earlier claim was drawn from the 3-sector run, where the same gain was small
and within noise -- it did not generalise, and stating it as a general
conclusion was an over-reach. **Longer baselines buy BOTH coverage and
sensitivity.**

### The falloff shape -- gradual, NOT a collapse at the 2-transit floor

Within the 8-sector run, depth >= 1200 ppm, in-range only:

| injected P | median transits | exact detection |
|---|---|---|
| 3 d | 72.2 | 0.900 |
| 10 d | 21.7 | 0.867 |
| 20 d | 10.8 | 0.833 |
| 40 d | 5.4 | 0.733 |
| 60 d | 3.6 | 0.700 |
| **90 d** | **2.4** | **0.633** |

Clean and monotonic, but **gentle**: a 30x reduction in transit count costs only
~30% relative detection, and at **2.4 transits -- essentially TLS's 2-transit
floor -- detection is still 0.633.** The earlier expectation that "usable
detection will fall off well before the nominal ceiling" is **only weakly
supported**: the ceiling itself is the hard wall; approach to it is a soft slope.

### Transit count is NOT the governing variable -- baseline is

Pooling all three runs by transit count flattens out entirely (0.556-0.733,
overlapping CIs), because at MATCHED transit count the longer baseline still
wins:

| transits | 1 sector | 3 sectors | 8 sectors |
|---|---|---|---|
| 1.5-3 | 0.727 | 0.467 | 0.633 |
| 3-5 | 0.567 | 0.400 | 0.700 |
| 5-9 | 0.733 | 0.633 | 0.733 |
| 9-15 | -- | 0.633 | 0.833 |
| 15-30 | 0.400 | 0.767 | **0.867** |
| 30-200 | -- | 0.467 | **0.900** |

This explains why the earlier attempt to read a falloff law out of two baselines
produced a flat, unreadable picture: **transit count was the wrong x-axis.**
Baseline carries information transit count does not -- total photons collected,
more out-of-transit data for the noise estimate, better detrending leverage.

### Unchanged: the Earth-size floor, and the useless classification stage

* **84 ppm: 3/70 = 0.043** even at 216 days (vs 0/60 and 2/80). Still a floor.
  Longer baselines do not rescue Earth-size.
* **Control arm 97.5%** of zero-depth trials pass the 0.30 triage floor (70% /
  95% / 97.5% across the three runs). The Stage-2 table is ~1.00 nearly
  everywhere and is **not** discrimination. End-to-end sensitivity remains
  governed almost entirely by TLS detection, for the third time.

### Caveat carried forward

Production's binning gives this sample ~18 min effective cadence. The paired
cadence arm tested only 2 -> 8 min (no harm, McNemar p = 0.375); 18 min is
beyond what was measured. Since detection IMPROVED at 8 sectors anyway, any
residual cadence penalty is evidently smaller than the baseline gain -- but it
is not separately measured.

### Operational conclusion, revised

**Multi-sector coverage is a stronger lever than the previous section
concluded.** It buys (a) search reach that scales exactly and predictably as
baseline/2, and (b) a genuine, significant sensitivity gain at periods already
searchable -- roughly a doubling of exact-period detection from 1 to 8 sectors.
Both are independent of the closed question of whether longer baselines move
classifier AUC; no classifier change can recover a period the search never
covered, and none of this touches the model.

Storage cost for the sample: 1.9 GB raw + 472 MB processed for 45 stars at 8
sectors.

## VAE ANOMALY DETECTION -- PILOT RUN. KILL CRITERION FIRED. Closed.

Proposal 2 from the three-way synthetic-injection split was deferred pending
explicit go-ahead. Go-ahead given; **Phase 1 built and run against the criteria
pre-registered in this file BEFORE any code was written**, so the verdict could
not be moved after seeing the numbers. Script `vae_anomaly_pilot.py`, results
`vae_anomaly_pilot_results.json` / `..._scores.csv`.
**Production untouched: 0.9300 / 31 features / md5 `1f0b7cb8e78ab542374eaf78fc837a6f`.**

### Setup

Small two-branch 1D-conv VAE (latent 8) on the global (201) + local (61)
phase-folded views already built for the CNN work. Trained **unsupervised on
924 real TRAIN-split label-0 curves** -- TOI false positives, i.e. things that
looked transit-like and were not. It never saw a label-1 curve or a test-split
host. Early-stopped on a held-out 138-curve validation slice.

**The VAE trained fine.** Validation reconstruction fell 2.036 -> 0.0399 and
plateaued. This is not a training failure -- it converged and learned to
reconstruct well. It simply learned the wrong thing.

### Results against the pre-registered criteria

| criterion | threshold | measured | verdict |
|---|---|---|---|
| **SC1** real transit vs no-transit (TEST split, n=852/231) | AUC > 0.75 | **0.6806** | **FAIL** |
| **SC1** injected transit vs no-transit (n=4000/231) | AUC > 0.75 | **0.6265** | **FAIL** |
| **KILL** \|corr(recon error, `var_oot_rms`)\| (n=5,357) | > 0.60 | **Pearson +0.830, Spearman +0.829** (p ~ 0) | **FIRED** |

**SC2 and SC3 were not run**, by design: they require phase-folded views for the
unknown candidate pool, which do not exist and cost real work to build. Not
building them once the kill fired is the entire purpose of having a kill
criterion.

### The mechanism predicted at proposal time is exactly what happened

When this was deferred, the stated reason was:

> reconstruction error is dominated by whatever carries the most variance, and
> in a TESS light curve that is instrumental systematics and stellar
> variability -- not a 84-5,000 ppm transit ... A reconstruction-based detector
> would rediscover that variability and call it anomaly.

Measured correlation between reconstruction error and `var_oot_rms`: **+0.83**.
The VAE is a stellar-variability meter. It reproduces, unsupervised and at
much greater cost, information the deployed model already has as an explicit
feature -- `var_oot_rms` is one of the five variability features promoted in
August 2026 and part of the current 31.

Note the ordering of the two failures: SC1 alone (AUC 0.68) might have invited
tuning -- a bigger latent, more epochs, a different beta. The kill criterion is
what makes that pointless. At r = 0.83 the model is not an
undertrained transit detector; it is a well-trained detector **of the wrong
quantity**. Tuning a variability meter produces a better variability meter.

### Limitation that would have applied even on a pass

The views are folded **at the known period**, produced by `phase_fold_views.py`
for the CNN. That presupposes TLS already found the signal, so this VAE sits
DOWNSTREAM of detection -- it is not the independent "flag light curves that
look like nothing in training" detector the original proposal imagined. It is
the cheap version the pre-registered scope specified, precisely because the
folded views already existed. A genuine version would run on unfolded data and
be a substantially larger build. **This is recorded because it would have
capped how far a PASSING result could have been read, and it does not rescue
the failing one** -- folding at the known period makes the transit MORE visible,
so this was the generous case and it still failed.

### VERDICT: CLOSED

Not deferred, not "needs more tuning" -- **closed on a pre-registered kill
criterion that fired at 0.83 against a 0.60 threshold.** Do not re-propose
reconstruction-based anomaly detection on photometry for this project without
a mechanism that decouples reconstruction error from stellar variability. The
capability the proposal was reaching for -- flagging candidates outside the
model's competence -- **already ships**, as the IsolationForest
`multivariate_ood_flag` in `06_download_unknown.py`.

Total cost: one script, ~4 minutes of training. The bounded-pilot-with-kill
design worked exactly as intended -- it spent minutes to close a question that
had been estimated at 1-2 days.

## BLS AS A COMPLEMENTARY DETECTOR + 2-SECTOR STACKING -- both assessed, both NO

Two mechanistically distinct proposals, two separate verdicts. Detection-stage
assessment only. **Production untouched: 0.9300 / 31 features / md5
`1f0b7cb8e78ab542374eaf78fc837a6f`, verified before and after. Nothing built
into the pipeline.**

### PART 0 -- "stack the 2 highest-SNR sectors": which operation is it?

"Stacking to increase per-transit SNR" only means anything if the data are
**phase-combined**, so this is interpretation (a) -- folding at an ephemeris --
not (b) concatenation. (Concatenation does not stack signal; it extends the
search baseline, which is the already-validated wide-sector work below.)

So the drift wall applies. Computed on this project's own **5,290 stars** with
usable `period_uncertainty`. Max span keeping accumulated drift under one
transit duration is `P x duration / sigma_P`:

| percentile | max allowable span |
|---|---|
| 25th | 4.4 d |
| **50th** | **9.4 d** |
| 75th | 16.6 d |
| 90th | 31.7 d |

**The median star can tolerate 9.4 days -- shorter than a single 27-day sector.**

Against real sector geometry (measured from this project's own 8-sector sample:
one sector spans 27.1 d, two adjacent span 54.1 d):

| 2-sector configuration | span | stars avoiding the drift wall |
|---|---|---|
| adjacent sectors | ~27 d | **12.0%** |
| two sectors, one gap between | ~54 d | **5.3%** |
| four apart | ~110 d | 2.0% |
| one year apart | ~365 d | 0.2% |
| *(closed stacking run's median)* | 759 d | 0.0% |
| *(consistency re-proposal's median)* | 1,151 d | 0.0% |

**Reducing to two sectors does NOT escape the wall -- it moves 0.0% to 5.3%.**
The brief's own framing was right: the failure is driven by SPAN, not sector
count, and two adjacent sectors still span 54 d against a 9.4 d median budget.
5.3% of 5,290 is ~280 stars, squarely inside the 200-696 range that closed
prior multi-sector attempts on scale grounds regardless.

**PART 2 NOT RUN** -- Part 0 found no untested non-circular version.

**Why the wide-sector work escaped this entirely, restated for future readers:**
it never folds at a stored period. Sectors are concatenated and **TLS runs its
own blind search**, so `sigma_P` never enters the calculation. Same raw
material, two different operations, opposite outcomes. That distinction is the
whole reason one is closed and the other confirmed a scaling law three times.

### PART 1 -- BLS vs TLS: 300 paired trials, both detectors on the identical curve

`astropy.timeseries.BoxLeastSquares` (astropy 8.0.1, **no new dependency**).
Every trial injects one signal into one real light curve and runs BOTH
detectors on the same array, so any difference is the detector alone. BLS gets
the **same period range TLS actually searched** on that curve and the same
alias-aware recovery test. Both shapes the proposal names:
`inject_transit` (U-shaped, limb-darkened) and `inject_eclipsing_binary`
(grazing, V-shaped, with a half-depth secondary). Periods down to 0.5 d.
289/300 ok; script `bls_vs_tls_detector.py`.

#### Runtime: BLS is genuinely much cheaper

    TLS median 38.5 s     BLS median 6.97 s     -> 6x faster

#### THE OPERATIVE RESULT: on detection recall, BLS adds NOTHING

"Does the star get flagged at all" (recovery at any alias -- an aliased
detection still surfaces the star for review, and the period is refined later):

| shape | both | TLS-only | BLS-only | neither | McNemar | TLS -> union |
|---|---|---|---|---|---|---|
| EB (V-shaped) | 88 | 2 | 3 | 52 | p = 1.000 | 0.621 -> 0.641 |
| transit (U-shaped) | 95 | 3 | 2 | 44 | p = 1.000 | 0.681 -> 0.694 |
| **POOLED** | **183** | **5** | **5** | **96** | **p = 1.000** | **0.651 -> 0.668 (+0.017)** |

**Perfectly symmetric: 5 TLS-only against 5 BLS-only.** There is no
complementary population. The +0.017 union gain is noise, and the hypothesis
that BLS's box model wins on non-canonical shapes is **not supported on
detection**.

#### There IS a real BLS advantage -- but it is period ACCURACY on EBs, not recall

On EXACT-period recovery, BLS wins: BLS-only 16 vs TLS-only 5 pooled
(McNemar **p = 0.0266**), concentrated in the V-shaped EB arm (12 vs 3,
p = 0.0352). The mechanism is specific and visible in the alias breakdown at
EB, P = 10 d (n = 29):

| | exact | any-alias | aliases when detected |
|---|---|---|---|
| TLS | 0.138 | 0.552 | **half 12**, exact 4 |
| BLS | **0.379** | 0.517 | **exact 11**, half 4 |

**TLS finds the signal just as often -- it reports the wrong period.** A
V-shaped eclipse plus a secondary at phase 0.5 looks to TLS's physical transit
template like two identical transits at half the period, so it locks onto the
half-period alias. BLS's plain box is less prone to it.

Real, significant, and correctly diagnosed -- but note what it buys: **EBs are
the NEGATIVE class.** Better period assignment on eclipsing binaries does not
find planets.

**One unmeasured downstream hypothesis, flagged as such.** At half-period the
secondary eclipse folds onto phase 0 alongside the primary, which is exactly
where `secondary_eclipse_depth` and `odd_even_mismatch` are measured -- the two
features specifically designed to catch EBs. If TLS is mis-periodding EBs at
this rate, those features may be degraded on precisely the rows they exist to
flag. **This was not tested here** and would be its own investigation.

#### Integration cost kills the "cheap add-on" framing anyway

A BLS-only detection cannot be scored by the deployed model. Of the **22
TLS-derived features** in the production 31, BLS natively supplies **3**
(`period`, `duration`, `depth`). The other **19** -- `SDE`, `SDE_raw`, `FAP`,
`period_uncertainty`, `depth_mean*`, `odd_even_mismatch`, `rp_rs`, `snr`,
transit counts, `chi2red_min`, `depth_consistency_std`,
`secondary_eclipse_depth`, `transit_shape_ratio`, `depth_duration_ratio` -- are
TLS outputs with no BLS equivalent. So any BLS-only candidate would still need
a full TLS run to be scored, and BLS's 6x speed advantage buys nothing at the
pipeline level.

### RECOMMENDATION

**Do not add BLS as a complementary detector.** On the question actually asked
-- does BLS catch a population TLS misses -- the answer is measured and
symmetric: 5 vs 5, McNemar p = 1.000, union gain +0.017. TLS's higher general
sensitivity does dominate, as its design premise predicts, and the box-model
advantage on non-canonical shapes does not appear in detection recall.

**Do not pursue 2-sector stacking.** It is the closed folding operation at
smaller scale, and the smaller scale does not rescue it: 5.3% of stars at a
54-day span, ~280 stars, below the scale bar that closed the earlier attempts.

**Worth keeping from this run** (neither is a pipeline change): the measured
TLS half-period alias behaviour on V-shaped EBs, and the derived hypothesis
that it may be degrading the very features meant to reject EBs.

### Known defect in this run

11 of 300 trials (6 transit, 5 EB) failed with
`ValueError: The maximum transit duration must be shorter than the minimum
period` -- my BLS duration grid reaches 0.20 d while the floor on
`minimum_period` is also 0.2 d. A bug in the harness, not in either detector.
Losses are spread across both shapes so the paired comparison is not biased,
but the affected cells are the very shortest periods and the fix would be to
scale the duration grid to `minimum_period`.

### Follow-up: does the half-period alias degrade the EB-catching features? NO -- it AMPLIFIES one

The BLS assessment above raised, but did not test, the worry that TLS's
half-period alias on eclipsing binaries compromises
`secondary_eclipse_depth` and `odd_even_mismatch` -- the two features that
exist to reject EBs. **Tested. The worry was wrong, in an informative way.**
90 EB trials, 90/90 ok, 32 min. Script `half_period_feature_test.py`.
**Production untouched, md5 `1f0b7cb8e78ab542374eaf78fc837a6f`.**

**Design.** Inject a known EB, then run TLS **twice on the identical array**:
ARM A the free production search, ARM B with the period grid clamped to
+/-0.5% of the TRUE injected period, forcing the correct fold. Same code path,
same star, same signal -- so any feature difference is the fold period alone.
Splitting ARM A by which alias it landed on isolates the half-period effect.

**The prediction was written into the script before running**, so the reading
could not be fitted afterwards:

> At the half period the primary and secondary both fold onto phase 0 and
> alternate; phase 0.5 is empty. So `secondary_eclipse_depth` -> DESTROYED,
> `odd_even_mismatch` -> AMPLIFIED.

#### Result: half right

| feature | group | free fold | true fold | ratio | MWU p |
|---|---|---|---|---|---|
| `secondary_eclipse_depth` | free=EXACT (n=32) | 0.000276 | 0.000254 | 1.09 | 0.677 |
| | **free=HALF (n=17)** | **0.000284** | **0.000180** | **1.58** | **0.235 (ns)** |
| `odd_even_mismatch` | free=EXACT (n=31) | 0.484 | 0.507 | 0.95 | 0.364 |
| | **free=HALF (n=17)** | **7.253** | **1.549** | **4.68** | **0.044** |

* **`odd_even_mismatch` AMPLIFIED 4.7x -- prediction CONFIRMED** (7.25 sigma vs
  1.55 sigma, p = 0.044).
* **`secondary_eclipse_depth` NOT destroyed -- prediction REFUTED.** It is
  slightly higher at the half fold, and the difference is not significant.

**Where the amplification actually comes from** (checked rather than assumed):
the odd/even depth difference grows only 1.29x (1,113 ppm vs 860 ppm), but
`odd_even_mismatch` is that difference **normalised by its uncertainty**, and
at half period there are twice as many folded eclipses, so the denominator
shrinks. Most of the 4.7x is the uncertainty term, not the signal term.

#### The practical conclusion inverts the worry

**The half-period alias does not make EBs harder to reject -- it makes them
easier.** It drives a 4.7x increase in exactly the feature designed to catch
them, while leaving the other unharmed. The concern raised in the BLS
assessment is retracted; no action follows from it, and BLS's better period
accuracy on EBs buys even less than it appeared to.

#### An unrelated weakness the test surfaced instead

`secondary_eclipse_depth` recovers only **20-30% of the injected secondary
depth even at the CORRECT fold**:

| injected depth | expected secondary | measured at true fold | recovered |
|---|---|---|---|
| 2,500 ppm | 900 ppm | 201 ppm | **0.22** |
| 5,000 ppm | 1,800 ppm | 494 ppm | **0.27** |

(expected = 0.36 x primary, since the injector sets secondary `rp` = 0.6 x
primary `rp`.)

Mechanism: the feature takes the **median** over phase 0.45-0.55 -- **10% of
the phase** -- while the eclipse itself spans roughly 2%. The median is
therefore dominated by out-of-eclipse flux and dilutes the secondary by
roughly the duty-cycle ratio. This is a property of the implementation, not of
the fold, and it is why the alias made no difference: the feature is blunt at
every period.

**Recorded as an observation, not a proposed change.** A duration-aware window
(or a depth measured over the eclipse rather than a fixed phase band) would
plausibly sharpen it, but that is a production feature in the deployed 31 and
any change needs its own full validation cycle. Not attempted here.

#### Limits

n = 17 in the HALF group and p = 0.044 is marginal -- directional, not
bulletproof. And 41 of 90 trials landed on neither the exact period nor the
half alias: grazing EBs at impact parameter 0.9 are genuinely hard to recover,
which is itself consistent with the low EB detection rates in the BLS run.

## SECONDARY_ECLIPSE_DEPTH REBUILD -- found a PHASE-CONVENTION BUG, but fixing it does not help

Task: rebuild `secondary_eclipse_depth` with a duration-aware window, since the
half-period diagnostic measured it recovering only 20-30% of injected secondary
depth. **The premise was right that the window is wrong, but incomplete: the
window is also in the WRONG PLACE.** Fixing it is a measurable small
REGRESSION. **Nothing promoted. Production untouched at 0.9300 / 31 features /
md5 `1f0b7cb8e78ab542374eaf78fc837a6f`, verified before and after.**

### First: the proposed formula already exists

`weak_secondary.py` (2026-08-03) already implements exactly the proposed
duration-aware window as `sec_depth_windowed`:

    dur_phase = duration / period
    half      = max(0.5 * dur_phase, 0.002)
    window    = |phase - 0.5| < half

with values already computed for 5,365 stars, and already reported: AUC 0.435,
max |corr| 0.478, coverage 97.8%, no clearing arm when ADDED at 24 -> 26. The
brief's framing -- that the earlier investigation tested "a different statistic,
noise-normalised significance, not depth accuracy" -- is half right: it tested
**both**. Only the REPLACEMENT variant (31 -> 31) was genuinely untested.

### THE REAL FINDING: production measures the secondary at the PRIMARY's phase

**TLS's `r.folded_phase` places the PRIMARY at phase 0.5, not 0.** Verified
directly with a pure transit carrying no secondary at all (8,000 ppm injected):

| phase window | measured depth |
|---|---|
| **0.49 - 0.51** | **0.009202** <- the primary is HERE |
| 0.00 - 0.01 | 0.001232 |
| 0.99 - 1.00 | 0.000347 |

Production (`06_download_unknown.compute_all_features`) does:

    sec_mask     = (phase > 0.45) & (phase < 0.55)     -> samples the PRIMARY
    primary_mask = (phase < 0.02) | (phase > 0.98)     -> samples ANTI-transit

So `secondary_eclipse_depth` is a median over a slab centred on the **primary
transit** and ~6x wider than it, leaving the median dominated by out-of-transit
baseline. It measures neither the secondary (wrong phase) nor the primary (too
diluted).

**Confirmed on real data**, 5,485 training stars:

| feature | fold used | n | single-feature AUC | \|AUC-0.5\| |
|---|---|---|---|---|
| `secondary_eclipse_depth` (DEPLOYED) | TLS fold, window on primary | 5,373 | **0.4935** | **0.0065** |
| `sec_depth_windowed` (correct) | own fold, duration-aware | 5,365 | 0.4350 | 0.0650 |
| `sec_significance` (correct) | own fold, noise-normalised | 5,365 | 0.3811 | 0.1189 |

The deployed column sits at chance and correlates with nothing (max |rho| 0.160
against `depth_mean`) -- the signature of a near-noise measurement.

**Why `weak_secondary.py` escaped this:** it builds its own fold,
`phase = ((t - t0)/period) % 1.0`, which puts the primary at 0 and the secondary
at 0.5, so its window is correct. That resolves a discrepancy that has been
sitting unexplained in this log: why the "weak secondary" work got physically
sensible class rates (FPs at 3.4x the significant-secondary rate) while the
deployed feature stayed weak. Different fold conventions.

**A Part 1 run of mine was invalidated by the same bug and is reported anyway**,
because it is corroboration. `secondary_window_accuracy.py` measured
depth-recovery against noiseless ground truth, but both its windows assumed the
secondary sat at phase 0.5 in TLS's fold -- so it measured PRIMARY recovery. Its
"new window" recovered **2.387x** the true secondary depth; the primary/secondary
depth ratio in the injected model is **2.44**. The bug predicts that number.

### Class-rate gate: PASSES

|AUC - 0.5| is **0.0650 corrected vs 0.0065 deployed -- 10x better class
separation.** So Part 3 was authorised and run.

### Part 3: like-for-like swap at the same slot (31 -> 31) -- SMALL SIGNIFICANT REGRESSION

Production recipe, frozen split, 12 training bootstraps, corrected column
replacing the broken one.

| | coverage | max \|rho\| vs other 30 |
|---|---|---|
| deployed | 0.9796 | -- |
| corrected | 0.9779 | 0.151 (`transit_shape_ratio`) |

| arm | mean d | sd | min | max | pos | >=MDE |
|---|---|---|---|---|---|---|
| swap (full test) | **-0.0024** | 0.0014 | -0.0048 | +0.0001 | **1/12** | 0/12 |
| swap (2-min subset) | -0.0024 | 0.0014 | -0.0048 | +0.0001 | 1/12 | 0/12 |

**95% CI [-0.0046, -0.0002] -- entirely below zero.** Brier 0.0840 -> 0.0853
(worse), ECE 0.0353 -> 0.0356 (flat).

So **replacing a near-noise feature with a genuinely informative one makes the
model measurably, if slightly, worse.** Counterintuitive, and consistent with a
pattern this project has now measured four times: novel-by-correlation is not
novel-to-the-model. The corrected secondary information is already reachable by
the tree ensemble through `odd_even_mismatch`, `depth_mean_odd/even` and
friends, so the swap perturbs a well-fit model without adding anything.

### RECOMMENDATION: report the bug, do NOT promote the swap

* **The bug is real** and worth recording for correctness and interpretability:
  two deployed features are computed on the wrong phase regions.
* **Fixing it does not help the model** -- the swap is a small but statistically
  clean regression (CI entirely below zero, 1/12 positive).
* **Do not promote.** Production keeps its current formula.

### Carried forward, not tested here

* **`transit_shape_ratio` is affected by the same bug.** Its `primary_mask`
  targets the anti-transit region, so it is not measuring transit shape. That is
  consistent with its weak AUC (0.4333, coverage 3,825/5,485) and its standing
  as a retirement candidate, but it was NOT tested directly and no claim is made
  about what fixing it would do.
* **The untested third option is RETIREMENT, not repair.** If the deployed
  column is near-noise (AUC 0.4935) and the corrected version is worse, the
  open question is whether dropping it entirely (31 -> 30) is better than
  either. Same 12-bootstrap harness, ~15 min. Not run -- it is a different
  question from the one asked, and expanding scope unilaterally is what the
  brief asked me not to do.

## PHASE-BUG FOLLOW-UPS: `transit_shape_ratio` has the SAME bug; and RETIRING the secondary column is ALSO worse

Two cheap diagnostics following the phase-convention bug investigation.
**Production untouched: 0.9300 / 31 features / md5
`1f0b7cb8e78ab542374eaf78fc837a6f`. Nothing promoted, nothing changed.**

### PART 1 -- `transit_shape_ratio` DOES have the phase-convention bug. Confirmed empirically.

Same verification device as before: inject a clean, deep (12,000 ppm),
low-impact **U-shaped** transit -- a case where edge/center is well-defined --
and read what the deployed windows actually land on in TLS's real fold.

| region | n | median depth |
|---|---|---|
| production's `primary_mask` = `(phase<0.02) \| (phase>0.98)` | 781 | **-0.000274** |
| where the transit ACTUALLY is, `\|phase-0.5\|<0.02` | 744 | **0.010267** |

The deployed mask samples flux **slightly ABOVE the baseline** -- it is pure
out-of-transit noise, 37x shallower than the real transit and of the wrong sign.

Computing the edge/center statistic at both locations:

| | center depth | edge depth | **ratio** |
|---|---|---|---|
| **AS DEPLOYED** (windows at phase 0) | 0.000521 | **-0.000557** | **-1.0684** |
| at the CORRECT location (phase 0.5) | 0.013178 | 0.011052 | **0.8387** |

A negative ratio is physically impossible for a real transit. The correct
computation gives 0.84 -- edge slightly shallower than center, exactly what a
U-shaped transit should produce.

**Corroborated at scale on the deployed column itself** (5,486 training stars):

    coverage 3,826/5,486 = 69.7%     AUC 0.4333
    **48.5% of values are NEGATIVE**  (1,855 of 3,826)
    percentiles [5,25,50,75,95] = [-3.855, -0.636, 0.035, 0.761, 4.520]

Half the deployed values are negative and the spread runs -3.9 to +4.5 around a
median of 0.035. That is a noise-over-noise ratio, not a shape measurement.

#### Same root cause, or distinct? BOTH -- and they compound

* **Phase-convention bug (DOMINANT, newly confirmed here).** The windows are
  centred on phase 0 while TLS puts the transit at 0.5, so the statistic is
  computed on baseline. This alone accounts for the negative values and is
  decisive.
* **Window-geometry issue (previously diagnosed, still real, secondary).** The
  center (`<0.005`) and edge (`0.005-0.015`) windows are fixed in phase and
  never scale to duration. In this test the injected transit spans +/-0.0178 in
  phase, so both windows happen to sit inside it; for a shorter-duration transit
  they would straddle the ingress. **This only starts to matter once the phase
  is fixed** -- currently it is masked by the larger error.

So the earlier diagnosis was correct but incomplete, in the same way the
`secondary_eclipse_depth` brief was: the window is the wrong WIDTH *and* in the
wrong PLACE, and the placement error dominates.

#### No fix proposed or tested, per the brief

Reporting the correct-phase single-feature AUC on real data would require
recomputing the feature for all 5,486 training stars, i.e. a full TLS re-run
(~38 s/star, ~8 h at 7 workers). That is new infrastructure, not the existing
harness, so per the brief's Part 1.5 and the `trap_vshape` precedent this stops
at diagnosis. **What is established:** the deployed values are noise ratios,
and the same statistic computed correctly is physically sensible.

### PART 2 -- three-way comparison for `secondary_eclipse_depth`: KEEP wins

12 training bootstraps, production recipe, frozen split. Uses **SEED 20260812,
identical to the swap run**, so all three arms are PAIRED on the same draws --
verified, not assumed: the keep arm reproduced the prior run's mean AUC to
**0.00e+00** (0.9212137 both times).

| arm | features | mean d vs keep | sd | 95% CI | pos | >=MDE |
|---|---|---|---|---|---|---|
| **(a) KEEP AS-IS** (buggy formula) | 31 | **0.0000** (reference) | -- | -- | -- | -- |
| (b) FIX AND SWAP (corrected formula) | 31 | -0.0024 | 0.0014 | [-0.0046, -0.0002] | 1/12 | 0/12 |
| **(c) RETIRE** (column removed) | 30 | **-0.0025** | 0.0011 | **[-0.0044, -0.0010]** | **0/12** | 0/12 |

Calibration moves the same way: Brier 0.0840 -> 0.0859 and ECE 0.0353 -> 0.0382
on retirement (both worse), against 0.0853 / 0.0356 on the swap.

**Both alternatives are small, statistically clean regressions of nearly
identical size.** Neither CI touches zero.

(Absolute AUCs here are ~0.921 rather than production's 0.9300 because these are
bootstrap-resampled training sets with reduced effective n -- consistent with
every resampled run in this log.)

#### The surprise: the model relies on this column heavily

Permutation importance of `secondary_eclipse_depth` as deployed, frozen test,
10 repeats:

    importance +0.00680 +/- 0.00136     RANK 9 of 31

**Top third of the feature set** -- above `FAP`, and far above `depth_mean`
(+0.0003), `rp_rs` (+0.0007) and `distinct_transit_count` (+0.0009).

That directly contradicts its single-feature AUC of 0.4935 (chance), and it
explains why BOTH alternatives hurt by the same ~0.0025: the model is genuinely
using this column, so replacing its values or deleting it destroys something
real.

**What it is probably encoding.** It is a median over a 10%-wide phase slab
centred on the primary. That value is dominated by the local out-of-transit
baseline, modulated by how much of the slab the transit fills -- so it behaves
as a **composite duty-cycle and local-noise proxy**, not as a secondary-eclipse
measurement at all. Useless alone (AUC 0.4935), useful in combination (rank
9/31). This is the mirror image of the pattern recorded four times in this log:
there, features informative alone added nothing to the model; here, a feature
uninformative alone is load-bearing within it.

### RECOMMENDATION: KEEP AS-IS. Production stays at 31 features, current formula.

| option | verdict |
|---|---|
| keep the buggy formula | **BEST measured.** Reference. |
| fix the phase convention | -0.0024, CI below zero. Do not. |
| retire the column | -0.0025, CI below zero, 0/12 positive. Do not. |

The honest summary is uncomfortable but well-measured: **a feature that is
demonstrably computing the wrong quantity is nonetheless the best of the three
available options**, because the accidental quantity it computes is useful to
the model. The bug is documented; the fix is not worth applying.

`transit_shape_ratio` carries the same bug and is left alone for the same
reason pending its own investigation -- with the added caution that it too may
be load-bearing despite being wrong, and its permutation importance was NOT
measured here.

## CETRA (arXiv:2503.20875) FEASIBILITY -- ASSESSED FRESH. NOT PURSUED. Definitive entry.

**Status check first: no prior CETRA record existed.** Zero mentions in this
file, no scripts, no git history. Assessed here for the first time, so this
entry is the record. **Production untouched: 0.9300 / 31 features / md5
`1f0b7cb8e78ab542374eaf78fc837a6f`.** Detection-stage assessment only.

### PART 0 -- HARD GATE: no NVIDIA GPU on this machine

    nvidia-smi          NOT PRESENT
    nvcc                absent
    torch.cuda          False        (torch 2.13.0)
    platform            Apple M1, arm64 -- Metal only, MPS True

CETRA requires an NVIDIA GPU with the CUDA toolkit and `nvcc` on PATH (stated
by both the paper and the repo). **Part 2 is hardware-blocked here.**

**Cloud GPU is cheap and is NOT the barrier.** Current on-demand rates: T4-class
~$0.30-0.60/hr, Vast.ai spot ~$0.35/hr, RTX 4090 ~$0.34-0.69/hr. A few hours of
evaluation is **$1-3**; a full day **$8-15**. The barrier is whether the
expected payoff justifies the setup, not the money.

### Input/output compatibility: drop-in on INPUT, NOT on OUTPUT

CETRA takes three 1D arrays -- `times`, `fluxes`, `flux_errors` -- which matches
this project's processed CSVs exactly. **No input redesign needed.**

Output is the problem. CETRA returns period, duration, depth, depth variance,
t0, SNR and likelihood ratios: roughly **6 of the 22 TLS-derived production
features**. The other **16 have no CETRA equivalent** -- including **`SDE`,
`SDE_raw` and `FAP`**. `FAP` ranks **5th of 31** in permutation importance on
the frozen test set. Replacing TLS would forfeit a top-5 feature and require
redesigning feature extraction.

### PART 1 -- do CETRA's claims address THIS project's measured bottlenecks?

#### The period ceiling: CETRA INHERITS IT. It does not solve it.

The decisive question was whether `period_max` is a property of the DATA or the
ALGORITHM. The paper answers directly:

> "the periodic search is only a function of light curve length, since this
> increases the highest checked period (**assuming a requirement of 2 or more
> visible transits**)"

**Same >=2-transit requirement, therefore the same `period_max` = baseline/2.**
GPU speed does not touch it -- it is an information constraint, not a compute
one. This project measured that ceiling at 12.66 / 38.15 / 108.22 d across three
baselines, matching baseline/2 to two decimals. **CETRA would reproduce those
same numbers.**

**But there IS a genuinely new capability, and it sidesteps the ceiling:**

> "A preliminary search in linear space also enables a search in these results
> for **single transits (a.k.a. monotransits)**, which are early indicators of
> the presence of long period planets"
> ... "cetra can do this in **under a second**"

TLS has no mono-transit mode. This project therefore has **no capability at all**
to flag a long-period planet from a single event. CETRA would not determine the
period, but it would surface the candidate -- which is exactly the gap the
period-ceiling work identified. **This is the most interesting finding here.**

#### The Earth-size floor: the headline ratio is real, the absolute gain is tiny

The "20% more low-SNR transits" figure is measured on an **Earth-analogue
subset**: 1,165 of 20,000 synthetic curves, **75-125 ppm depth** (which brackets
this project's 84 ppm test point), 2-3 observed transits, 173-day baseline,
600 s cadence. Absolute recoveries from the paper's Table 1:

| noise | CETRA | TLS | ratio | **absolute gain** |
|---|---|---|---|---|
| 34 ppm/hr | 97.2% | 95.5% | 1.02x | +1.7 pp |
| 160 ppm/hr | **8.1%** | **3.5%** | 2.3x | **+4.6 pp** |
| 800 ppm/hr | 0.7% | 0.3% | 2.3x | **+0.4 pp (8 vs 3 of 1,165)** |

**The improvement is largest where TLS already partly works and collapses where
TLS is near zero.** At the hardest noise level the impressive-sounding ratio is
five extra detections out of 1,165. A better algorithm helps in the marginal
regime, not the impossible one.

**Mapping to this project's own numbers.** Measured Earth-size (84 ppm) recovery
here: **0/60 at 25 d baseline, 3/70 = 4.3% at 216 d.** That 4.3% sits right
alongside CETRA's TLS figure of 3.5% at 160 ppm/hr -- good agreement, and it
suggests this project's data sits in that noise regime. Applying CETRA's 2.3x
there projects roughly **4.3% -> ~10%**. Real, but still far below anything
useful for Earth-size detection, and **0% -> ~0% on single-sector data**.

**Caveat on the comparison:** CETRA's tests are SYNTHETIC light curves with a
173-day baseline; this project's floor was measured on REAL TESS photometry of
TOI false positives (often variable) at 25-216 days. The regimes overlap but are
not identical, and the projection above is an extrapolation, not a measurement.

#### Replacement or complement? COMPLEMENT.

Given 16 of 22 TLS features have no CETRA equivalent and `FAP` is top-5 by
permutation importance, CETRA is not a realistic drop-in replacement. Its
defensible framing is a **fast first-pass or mono-transit companion**, with TLS
retained for feature extraction.

### PART 2 -- NOT RUN. Hardware-blocked, and not justified by Part 1.

### PART 3 -- BROADER SCAN: three scoped leads, none acted on

**LEAD A (largest, cheapest, needs no CETRA): most of this project's data is
still searched at a 12.3-day ceiling.**

| pool | n | median baseline | implied `period_max` | single-sector |
|---|---|---|---|---|
| training negatives | 1,253 | 25.4 d | **12.7 d** | 99.9% |
| candidate pool (single-sector) | 2,465 | 24.6 d | **12.3 d** | 96.9% |
| candidate pool (wide-sector) | 271 | 76.0 d | 38.0 d | 0.7% |

**96.9% of the candidate pool is being searched with a ~12.3-day period
ceiling**, while the multi-sector concatenation that lifts it is already built,
already validated at three baselines, and needs no new dependency. The
8-sector work also measured that longer baselines nearly **double** detection at
periods already searchable (P=3: 0.375 -> 0.700, Fisher p=0.0010). This is a
larger and far cheaper win than anything CETRA offers. Scoped, not started.

**LEAD B (~5 minutes): `transit_shape_ratio` permutation importance is still
unmeasured.** The retirement harness already exists. `secondary_eclipse_depth`
turned out to rank 9/31 despite being demonstrably broken, so the same check is
the prerequisite before anyone considers fixing or retiring
`transit_shape_ratio`. Cheap and decisive.

**LEAD C: mono-transit detection is absent from this pipeline entirely.** CETRA
supplies it, but the capability is not intrinsically GPU-bound -- a CPU
implementation is possible, just slower. Worth separating the *capability* from
the *tool* if long-period reach is ever prioritised.

### RECOMMENDATION: do not pursue CETRA now.

| question | answer |
|---|---|
| Does it solve the period ceiling? | **No** -- explicitly inherits the >=2-transit constraint |
| Does it solve the Earth-size floor? | **No** -- ~4.3% -> ~10% projected; +0.4 pp where TLS is near zero |
| Is it a drop-in replacement? | **No** -- 16 of 22 features missing, incl. top-5 `FAP` |
| Is cost the barrier? | **No** -- $1-3 for a few GPU hours |
| Is there anything genuinely new? | **Yes, one thing: mono-transit search** |

**Revisit only if** (a) mono-transit capability becomes a priority, or (b) an
NVIDIA GPU becomes routinely available AND Lead A has already been exhausted.
Lead A addresses this project's biggest measured detection limitation directly,
with validated in-house tooling and no new dependency -- it should come first.

## MULTI-SECTOR ROLLOUT -- STOPPED AT PART 0. Reprocessing training data is CLASS-CORRELATED.

Proposal: roll the validated multi-sector concatenation out across the full
training set and candidate pools, closing the ~12.3-day period ceiling that the
CETRA assessment showed still applies to 96.9% of the pool. **Stopped at the
Part 0 gate and reported for decision, per the brief's own instruction to halt
if training data could be revised in ways that invalidate prior comparisons.
Nothing reprocessed, nothing wired. Production untouched: 0.9300 / 31 features /
md5 `1f0b7cb8e78ab542374eaf78fc837a6f`.**

### This is RETROACTIVE REVISION, not clean addition

**5,471 of 5,486 training stars (99.73%) are currently single-sector** (median
baseline: negatives 25.3 d, positives 26.4 d); only 15 already exceed 40 d. Any
star reprocessed on a wider baseline can return a different period, depth,
duration, SDE and FAP -- so this revises the feature values underpinning the
frozen split and every prior experiment's baseline, rather than adding new rows.

### THE DISQUALIFYING FINDING: reprocessability is correlated with the label

TESS sector count is a direct function of |ecliptic latitude|, and that differs
by class in this training set (n=5,484 with coordinates):

| | positives | negatives |
|---|---|---|
| \|ecl lat\| > 54 deg | 55.24% | 45.49% |
| \|ecl lat\| > 78 deg (near-CVZ) | 2.56% | **7.55%** |

KS D = 0.1687, **p = 4.2e-23**.

Measured directly against MAST on a random sample of 120 training stars, asking
how many have a usable (>=2 **consecutive**) sector run:

| class | reprocessable | 95% CI |
|---|---|---|
| **positives** | **66/91 = 72.5%** | [0.622, 0.814] |
| **negatives** | **12/29 = 41.4%** | [0.235, 0.611] |

**Fisher exact p = 0.0034, odds ratio 3.74, confidence intervals do not
overlap.** Positives are ~1.75x more likely to be reprocessable than negatives.

Reprocessing "every star that can be reprocessed" would therefore apply an
SDE-boosting treatment to **72.5% of positives and only 41.4% of negatives**.
Measured shift from multi-sector processing on data already in hand:
**SDE +0.62 SD**, `transit_count` -0.41 SD. That implies roughly a **0.19 SD
artificial class signal in SDE arising from PROCESSING alone**, before any
astrophysics.

**This is precisely the confound `FEATURE_COLUMNS` deliberately excludes.** Sky
position is kept out of the model with the recorded reasoning that the unknown
pool "sits at a systematically different latitude than either training class, so
a learned position->label rule would not transfer." Partial reprocessing would
reintroduce it not as a feature -- where a control arm could catch it -- but as a
distortion of the feature VALUES, where no control arm can see it. It would
inflate measured AUC while making deployment worse, since candidates are
processed uniformly rather than class-correlated.

### Real addressable population (sampled, n=120, scaled to 5,486)

| criterion | fraction | ~stars |
|---|---|---|
| >= 2 sectors total | 79.2% | ~4,343 |
| **>= 2 CONSECUTIVE (usable)** | **65.0%** | **~3,565** |
| >= 3 consecutive | 25.8% | ~1,417 |

(Total sector count overstates it: non-consecutive sectors separated by
year-long gaps give TLS a huge period grid over sparse data -- the pathology
already filtered out of the K2 and 8-sector pools.)

### The "reprocess a uniform subset instead" escape does not work either

Restricting to the 65% that ARE reprocessable and treating them uniformly would
remove the within-set inhomogeneity, but: it drops ~34% of training rows (5,486
-> ~3,619), which the fitted learning curve prices at roughly **-0.008 AUC**;
and because retention is itself class-correlated, the surviving set is *more*
imbalanced than the original (~86.8% positive vs 79.0%). It costs data and
worsens balance to fix a confound.

### Part 3 (pools only) is NOT risk-free either

Training (single-sector) versus the existing widesector pool, on detection
features alone: **domain separability AUC 0.9248**. Reprocessing pools while
training stays single-sector means scoring candidates whose SDE/snr sit off the
distribution the model was fit on. The deployed IsolationForest OOD flag would
likely fire more often on exactly the long-period candidates this is meant to
surface -- which may be correct behaviour, but it is a real effect that needs
measuring before wiring, not after.

### VERDICT AND RECOMMENDATION

| option | verdict |
|---|---|
| **Reprocess training data (Parts 1-2)** | **DO NOT PROCEED.** Class-correlated at p=0.0034, OR 3.74. Would inject ~0.19 SD of artificial class signal and invalidate every prior experiment's baseline. |
| Uniform-subset reprocessing | Not attractive: -34% rows (~-0.008 AUC) and worse class balance. |
| **Reprocess candidate pools only (Part 3)** | **Viable, and the only sensible version** -- but measure the 0.9248 train/serve separability effect on OOD flagging first. Not wired. |
| Hold entirely | Defensible; the ceiling is real but so is the confound. |

**Recommended: pools-only, and only after an OOD-impact measurement.** The
detection-reach argument remains valid and unchanged -- 96.9% of the candidate
pool is still searched at ~12.3 days, and the 8-sector work showed longer
baselines nearly double detection even at already-searchable periods. But the
training-side half of this proposal cannot be done without corrupting the
comparison base, and no amount of validation downstream repairs a
class-correlated processing artifact upstream.

**Awaiting explicit direction. Nothing has been reprocessed or wired.**

### CLOSEOUT: OOD impact of pools-only multi-sector reprocessing -- SAFE. Rollout recommended.

Last open question from the multi-sector rollout. Measurement only; **nothing
modified. Production untouched: 0.9300 / 31 features / md5
`1f0b7cb8e78ab542374eaf78fc837a6f`.**

The deployed IsolationForest detector was run as-is against both pools. Note it
operates on **24 features** (built 2026-07-11, predating the crowding and
variability promotions), threshold -0.5341, contamination target 0.02.

#### 1. Flag rates

| population | n | flagged | rate |
|---|---|---|---|
| training (single-sector, its own calibration set) | 5,486 | 109 | **2.0%** |
| candidate pool, single-sector | 488 | 92 | **18.9%** |
| candidate pool, **MULTI-SECTOR** | 69 | 20 | **29.0%** |

The training figure reproduces the stored calibration baseline (2.00%) exactly --
a sanity check that the detector was loaded and applied correctly.

Multi-sector vs single-sector pool: **29.0% vs 18.9%, Fisher p = 0.0549,
OR 1.76.** Elevated, marginally significant, n=69.

**Context that matters more than the delta:** the pool is ALREADY at 18.9%
against training's 2.0% -- a ~9x elevation that has nothing to do with
multi-sector processing. Reprocessing adds ~10 points on top of a large
pre-existing gap.

#### 2. Selective, NOT systematic

**71% of multi-sector candidates pass the OOD check.** Nothing resembling the
near-100% systematic over-firing that would have made the detector useless on
this population.

And the flag is not tracking the processing artifact. Spearman rho of OOD score
against each feature, across both pools (n=557):

| processing-SHIFTED features | rho | | other features | rho |
|---|---|---|---|---|
| `SDE` | -0.242 | | **`depth`** | **+0.801** |
| `SDE_raw` | -0.223 | | **`rp_rs`** | **-0.801** |
| `snr` | +0.065 | | `period` | +0.285 |
| `transit_count` | -0.330 | | `duration` | +0.128 |
| `distinct_transit_count` | -0.312 | | `odd_even_mismatch` | +0.154 |

**The OOD score is dominated by `depth`/`rp_rs` (|rho| = 0.80) -- genuine
astrophysical unusualness -- while the features that multi-sector processing
actually shifts correlate at only 0.06-0.33.** (`depth` and `rp_rs` mirror each
other because `rp_rs` = sqrt(depth).) The detector is flagging unusual
candidates, not re-detecting "this was processed differently."

#### 3. Practical impact: the flag SUPPRESSES, it is not a caveat

Checked in code rather than assumed. `split_and_rerank`:

    keep = ranked_df["in_distribution"] & ~ranked_df["below_triage_floor"]

with `in_distribution = in_distribution_univariate & ~multivariate_ood_flag`.
Flagged candidates are routed to a separate `out_dist` table and **held out of
the in-distribution shortlist, which the code comments identify as "the thing
that feeds characterization and human review."** So an OOD flag is a real
demotion, not an informational note. That raises the stakes on question 4 --
which makes the next result the important one.

#### 4. THE CRUX: the anticipated tension does not apply -- it INVERTS

The brief flagged the risk that OOD handling would suppress exactly the novel
long-period candidates the rollout exists to find. Measured directly, splitting
the multi-sector pool at the 12.5 d single-sector ceiling:

| multi-sector candidates | n | OOD flagged |
|---|---|---|
| P <= 12.5 d (single-sector could also find these) | 33 | **42.4%** |
| **P > 12.5 d (NEWLY reachable)** | 36 | **16.7%** |

**Fisher p = 0.0323, odds ratio 0.27** -- the newly-reachable long-period
candidates are about **4x LESS** likely to be flagged. OOD score medians
-0.4159 vs -0.5182, MWU p = 0.0006.

And the yield is the point of the exercise:

| pool | candidates with P > 12.5 d |
|---|---|
| single-sector | 11 / 488 = **2.3%** |
| multi-sector | 36 / 69 = **52.2%** |

A **23x enrichment** in long-period candidates, and they are the ones the OOD
detector likes best.

#### RECOMMENDATION: proceed with the pools-only rollout as-is. No OOD fix needed.

| question | answer |
|---|---|
| Does OOD over-fire systematically on multi-sector? | **No** -- 29.0% vs 18.9%, 71% pass |
| Is it re-detecting the processing shift? | **No** -- driven by depth/rp_rs (0.80), not SDE/transit_count (0.06-0.33) |
| Does a flag actually suppress? | **Yes** -- held out of the review shortlist |
| Does it suppress the NOVEL candidates? | **No -- the opposite.** 16.7% vs 42.4%, OR 0.27 |

None of the proposed mitigations (recalibrating the detector, separate baselines
per processing method, informational-only flagging) is warranted by this
measurement. Adding them would be solving a problem that does not exist here.

**Honest caveats.** n = 69 scoreable multi-sector candidates is small, and the
headline 29.0% vs 18.9% is only marginally significant (p = 0.0549); the crux
result (p = 0.0323, and p = 0.0006 on scores) is firmer than the headline. The
detector also runs on the 24-feature set, so it does not see crowding or
variability -- rebuilding it on the current 31 is a separate, unexamined
question. And this measures the EXISTING widesector pool, not a fresh
full-pool reprocess.

#### Incidental finding: a latent bug is silently dropping candidates

**237 pool rows** carry status `"Post-processing error: 'bool' object has no
attribute 'sum'"` -- **222 of 2,454 single-sector (9.0%) and 15 of 271
widesector (5.5%)**. This is the same defect fixed in
`injection_recovery_sensitivity.py`: when TLS fits no transit it returns scalars
rather than arrays, and `sec_mask.sum()` then fails on a bool. These stars are
dropped from feature extraction entirely. Scoped, not fixed -- it is a
production code change in `06_download_unknown.compute_all_features` and needs
its own go-ahead.

### FINAL STATE OF THE MULTI-SECTOR ROLLOUT INVESTIGATION

| scope | verdict |
|---|---|
| **Training data** | **PERMANENTLY EXCLUDED.** Eligibility is class-correlated (72.5% vs 41.4%, Fisher p=0.0034, OR 3.74), injecting ~0.19 SD of artificial class signal. |
| **Candidate pools** | **SAFE TO ROLL OUT.** OOD impact measured and benign; novel long-period candidates are favoured, not suppressed. 23x enrichment above the old ceiling. |
| **OOD detector** | **No change required.** |
| **Awaiting** | explicit go-ahead to wire pools-only reprocessing into `06_download_unknown.py`. |

## MULTI-SECTOR CONCATENATION WIRED INTO PRODUCTION (candidate pools only) -- DEPLOYED

Closes the multi-sector rollout investigation. Concatenation is now part of
`06_download_unknown.py`'s standard candidate path. **Training data untouched
and permanently excluded. Production model untouched: 0.9300 / 31 features /
md5 `1f0b7cb8e78ab542374eaf78fc837a6f`; `training.csv` md5
`e58ec25aa89476cb8f45cba665b54079`.**

### What was added

| piece | behaviour |
|---|---|
| `longest_consecutive_sectors()` | longest run of CONSECUTIVE sectors. `{1,2,3,28,29}` -> `[1,2,3]`, not 5 -- concatenating across a year gap would hand TLS a huge grid over sparse data |
| `multi_sector_quality()` | the established flux filter (median within 1% of 1, robust sigma in (0,0.05), <1% beyond 10 sigma) PLUS continuity (max gap <= 10 d, duty >= 0.70) |
| `annotate_consecutive_sectors()` | annotates the ALREADY-SELECTED default pool with `sectors_observed`; does not change which candidates are chosen or their order |
| `audit_processing_mode()` (STAGE E2) | quality-gates each concatenated curve, falls back where it fails, writes `processing_mode{tag}.csv` |
| provenance columns | `processing_mode`, `n_sectors_used`, `baseline_days`, `quality_gate` merged onto every feature row |

**Single-sector is unchanged BY CONSTRUCTION, not merely by test.** A star whose
longest run is one sector gets `sectors_observed = NaN`, which routes it down
`download_one_star`'s original branch untouched. Verified anyway: 8 single-sector
files **byte-identical (md5) before and after** the new stage, all labelled
`single_sector`, gate not applied.

**Spawn safety:** `audit_processing_mode` runs in the parent process only, and
the existing `_tls_worker` already receives explicit paths as arguments rather
than reading module-level state -- the pattern that caused the variability
deployment's `BrokenProcessPool`. No new multiprocessing was introduced.

### Validation 1 -- reproduces the existing widesector pool exactly

Production gate against all 271 existing widesector curves: **165 pass (60.9%)**.
That reconciles precisely with the earlier ad-hoc figure rather than differing
from it:

    271 total
    -  98 fail the FLUX criteria (97 robust sigma, 1 outlier fraction)
    = 173   <- exactly the 173/271 = 63.8% measured in the wide-sector investigation
    -   8 fail the added CONTINUITY criteria (7 max-gap, 1 duty cycle)
    = 165

So the production gate is the established filter **plus** continuity, and the
numbers decompose exactly. Passing population: baseline 76.2 d median, max gap
2.0 d, duty 0.877.

### Validation 2 -- fresh, never-before-downloaded data, end to end

10 CVZ stars, none previously in any pool, run through
annotate -> download -> preprocess -> audit:

    downloaded 10/10, preprocessed 10/10, 0 failures
    multi-sector (passed gate) : 8
    fell back to single-sector : 2

| outcome | n | baseline | max gap | duty |
|---|---|---|---|---|
| `multi_sector` | 8 | **209.7 - 218.1 d** | 4.1 - 5.9 d | 0.80 - 0.87 |
| `single_sector_fallback` | 2 | 22.6 / 24.4 d after trim | **30.1 / 28.3 d** | 0.65 / 0.66 |

**The two fallbacks fired for exactly the right reason:** a ~28-30 day hole,
i.e. a MISSING SECTOR inside the nominal run. That is the precise pathology the
continuity guard exists to catch, and it caught it on data it had never seen.

Reach on the 8 that passed: median baseline **216.7 d -> period_max 108.3 d**,
against the single-sector ceiling of 12.3 d -- **8.8x**. (These are 8-sector CVZ
targets, chosen to exercise the path; a general pool will have a lower
multi-sector fraction and shorter runs. This is a capability demonstration, not
a projected pool-wide yield.)

### A real bug caught in testing, before deployment

The first fallback implementation located sector boundaries by splitting on time
gaps > 5 days. **Consecutive TESS sectors are separated by only ~1-2 day
downlink gaps** (measured max 3.88 d on this pool), so no boundary was ever
found, `start` stayed 0, and the whole curve was written back -- while still
being labelled `single_sector_fallback`. It would have shipped a provenance
column asserting a fallback that never happened. Replaced with a trim to the
last `SECTOR_LENGTH_DAYS` (27.5). Retested on 25 random curves: **12/12
fallbacks now trim correctly**, 47,548 -> 16,460 rows median, 27.5 d baseline.

### Known limitation, measured and recorded rather than hidden

**Trim-fallback is not identical to true single-sector processing.** The curve
was normalised GLOBALLY across all its sectors during preprocessing, so a slice
inherits that normalisation. On the 25-star sample, **all 12 trimmed curves
still failed the gate** (10 robust sigma, 2 median flux). That is consistent
with EITHER intrinsically bad photometry OR the slice inheriting a distorted
global normalisation, and the two cannot be separated without re-downloading and
re-preprocessing that sector alone. The `quality_gate` column records the
post-trim verdict per star, so the condition is visible downstream instead of
silent. A true re-download fallback is the correct fix and was deliberately not
attempted.

**Related, scoped, NOT done:** ~39% of concatenated curves fail on flux. The
likely root cause is per-sector flux-level offsets at concatenation, and the
principled fix is per-sector normalisation BEFORE concat. It was not applied
because it would change results and break the "reproduce the existing pool"
requirement. Worth its own task.

### Process note

The fresh-data test appended 10 rows to
`data/catalogs/unknown_download_log.csv` -- a real production catalog -- because
the test overrode `RAW_FOLDER`/`PROCESSED_FOLDER` but not the log path. Caught
in the pre-commit check and reverted with `git checkout`; the file is back to
its committed state. Recorded because a sandbox that overrides only some paths
is a trap this pipeline will present again.

### FINAL STATE OF THE MULTI-SECTOR ROLLOUT

| scope | state |
|---|---|
| **Training data** | **PERMANENTLY EXCLUDED.** Eligibility class-correlated (72.5% vs 41.4%, Fisher p=0.0034, OR 3.74); would inject ~0.19 SD of artificial class signal. Documented in code next to the constants so it cannot be re-proposed accidentally. |
| **Candidate pools** | **DEPLOYED.** Concatenation + quality gate + fallback + provenance columns in the standard path. |
| **OOD detector** | Unchanged -- measured safe (novel long-period candidates flagged 16.7% vs 42.4%, OR 0.27). |
| **Open** | true re-download fallback; per-sector normalisation before concat. |

## TLS NO-FIT POST-PROCESSING BUG -- FIXED. 0 candidates recovered, and that is the correct answer.

The `Post-processing error: 'bool' object has no attribute 'sum'` affecting 237
pool rows (222 of 2,454 single-sector, 15 of 271 wide-sector). **Production
model untouched: 0.9300 / md5 `1f0b7cb8e78ab542374eaf78fc837a6f`;
`training.csv` md5 `e58ec25aa89476cb8f45cba665b54079`.**

### Root cause, traced to source

When TLS finds no transit it does not return empty arrays -- it returns
**scalars where arrays are expected**. Measured directly on `TIC_85136394`:

    r.SDE = 0 (int)        r.period = nan        r.T0 = 0 (int)
    r.folded_phase = nan   r.folded_y = nan      r.transit_depths = nan

So `(phase > 0.45) & (phase < 0.55)` becomes the plain Python bool `False`, and
the next line's `.sum()` raises `AttributeError`.

### But TLS was RIGHT to find nothing -- the photometry is broken upstream

The processed curves of all 237 have median flux exactly 1.0 but **robust sigma
1.18 to 8.33** (a healthy curve is ~0.0002-0.006), only 2-5% of points within
+/-10% of unity, and **negative flux** -- physically impossible for normalised
photometry. Tracing upstream on `TIC_85136394`: the raw **PDCSAP flux itself**
has median 4.28 with **4,366 of 15,669 points <= 0**. Dividing by a savgol trend
passing near zero produces the sign flips and the enormous spread.

**Notably, `sap_flux` for the same star is clean** (median 77.5, min 40.3, no
non-positive values). A SAP fallback could plausibly recover this whole
population -- but that changes the photometry source, a real confound. Scoped,
NOT done.

### The fix: explicit no-fit detection, not a broad try/except

`tls_result_is_degenerate()` checks whether `folded_phase` is a real array and
`period` is finite and positive, and returns
`"No transit fit (TLS returned a degenerate no-fit result)"`. A bare `except`
around post-processing is what produced the original symptom: it surfaced a
Python error message as though it were a data condition, for 237 stars, hiding a
photometry problem behind what looked like a code defect. Same failure pattern
as the calibration-staleness bug and the scheduler's silent `except: pass`.

### A REGRESSION I INTRODUCED AND BACKED OUT

The first version of the fix also added a pre-TLS check rejecting anything with
robust sigma > 0.05, to skip the wasted search. **Measured against real data it
rejected 15 of 80 previously-successful candidates, 6 of which had genuine TLS
detections** (SDE 5.7-9.7, snr 2.0-27.0). Those are high-amplitude VARIABLE
STARS, not corrupted ones -- `TIC_281595262` has robust sigma 0.077, min 0.66,
max 1.36, 80% of points within 10% of the median, and TLS found SDE 7.14. The
genuinely broken curves sit an order of magnitude further out, and no single
statistical cut separates the two.

`validate_flux_for_search()` is now **structural only** -- non-positive median,
fewer than 50 finite points, or zero scatter. TLS makes the statistical
judgement. Cost: the pathological stars each burn a full search before being
reported. That is the correct trade -- **a false rejection loses a real
candidate permanently; a wasted search costs seconds.**

### Results

| population | n | valid features | statuses |
|---|---|---|---|
| **previously affected** | 237 | **0** | 237 x `No transit fit (degenerate no-fit result)` |
| **regression control** | 80 | **79** | 65 Success + 14 Success-with-imputed-optional |

**Old `AttributeError`: 0 occurrences in either group.**

**Breakdown for the 237: all genuinely unfittable, none incorrectly dropped.**
TLS now runs to completion on every one (median 11.8 s) and returns a degenerate
no-fit. **Recovery is 0, and that is the correct outcome** -- these stars were
always correctly excluded; only the stated REASON was wrong, and it looked like
a code defect rather than the photometry problem it is.

The single regression-control failure (`TIC_376669357`, `period_uncertainty` not
computable) is a **test-harness artifact, verified not a code regression**: that
star is absent from `unknown_candidate_list.csv`, so the harness passed default
stellar params (R*=1.0) instead of its real ones, changing TLS's period grid.

### SEPARATE PRE-EXISTING CRITICAL BUG FOUND -- NOT FIXED HERE, NEEDS A DECISION

The regression arm surfaced something larger. `best_model_metadata.json` lists
**31** feature columns, and `main()` passes all 31 through
`extract_features` -> `_tls_worker` -> `compute_all_features`. But that function
structurally **cannot produce 7 of them** -- `crowd_flux_ratio_max`,
`crowd_nearest_arcsec`, `var_oot_rms`, `var_excess`, `var_ls_amp`,
`var_ls_power`, `var_ls_period` -- which are added by separate later stages.
They are not in `OPTIONAL_FEATURES`, so they are always "blocking".

**Measured: with the real 31-column metadata, 64 of 80 previously-successful
stars fail on exactly those 7 columns.** With the 24 producible columns, 79 of
80 succeed.

**Implication: a fresh candidate run today would produce ZERO scored
candidates.** The existing `unknown_features.csv` was generated while the
metadata still had 24 columns, before the crowding (2026-08-05) and variability
(2026-08-06) promotions -- neither of which updated this gate.

Not fixed here: it is outside this task's scope, and the right fix (exclude
non-TLS columns from the gate, or move the check after the crowding/variability
stages) is a design decision, not a patch. **Flagged for explicit direction.**

**UPDATE 2026-08-13: FIXED.** See "THE 31-vs-24 FEATURE-COLUMN MISMATCH" below
-- `NON_TLS_FEATURE_COLUMNS`, validated end-to-end, plus the damage assessment
(candidate pools intact; 9 training labels lost and flagged, not recovered).

---

## THE 31-vs-24 FEATURE-COLUMN MISMATCH -- FIXED. The tool could not score a single new candidate for a week.

**Date: 2026-08-13. Production model UNCHANGED (0.9300 AUC, 31 features, md5
`1f0b7cb8e78ab542374eaf78fc837a6f`). `training.csv` UNCHANGED (md5
`e58ec25aa89476cb8f45cba665b54079`). Promotion gate and scheduler untouched.**

Found as a side-effect of the TLS no-fit investigation and flagged there for
direction. This is the fix.

### The bug

`compute_all_features()` ends with a completeness gate: every column in
`required_columns` must be present and finite, or it returns `None` and the star
is dropped. That gate hard-coded a single exception:

```python
if c not in ("st_rad", "st_teff")   # these come from the catalog, not TLS
```

`main()` passes the model's full feature list -- read from
`best_model_metadata.json` -- straight through `extract_features` ->
`_tls_worker` -> `compute_all_features`. That list was 24 columns when the
exception was written. Then:

* **2026-08-05, crowding promotion:** `crowd_flux_ratio_max`,
  `crowd_nearest_arcsec` added to `FEATURE_COLUMNS` (24 -> 26).
* **2026-08-06, variability promotion:** `var_oot_rms`, `var_excess`,
  `var_ls_amp`, `var_ls_power`, `var_ls_period` added (26 -> 31).

Neither promotion extended the exception. Both stages run **after** TLS -- they
are called per batch inside `extract_features`, on the frame `_tls_worker`
returns. So from 2026-08-06 the gate demanded **7 columns that cannot exist at
that point in the pipeline**, marked them blocking (none are in
`OPTIONAL_FEATURES`), and returned `None` for every star.

There was no crash, no traceback, no silent wrong answer. Every star got an
honest, precise, *entirely useless* status:

```
Required feature(s) not computable: ['crowd_flux_ratio_max', 'crowd_nearest_arcsec',
 'var_oot_rms', 'var_excess', 'var_ls_amp', 'var_ls_power', 'var_ls_period']
```

**A fresh candidate run produced ZERO scored candidates, and had for seven days.**

### Blast radius -- four consumers, one root cause

| caller | what it does | passes |
|---|---|---|
| `code/06_download_unknown.py` `main()` | the candidate pipeline | metadata's 31 |
| `web/job_runner.py:987` | per-candidate multi-sector strengthening | metadata's 31 |
| `web/retrain_pipeline.py:281` | **label watcher -> training.csv** | `m05.FEATURE_COLUMNS` |
| `code/k2_pilot/`, `code/kepler_pilot/` | closed pilots | `m05.FEATURE_COLUMNS` |

All four add crowding/variability *after* the call, so all four were broken and
all four are fixed by the single source-level change.

### Damage assessment -- measured, not assumed

**Candidate pools: INTACT.** `unknown_features.csv` (2,454 rows, 488 Success)
and `unknown_features_widesector.csv` (271 rows, 69 Success) both carry all 29
TLS-side columns and are dated 2026-08-06 15:18/15:19 -- written *before* the
variability promotion landed that evening. Nothing was corrupted; the pipeline
simply could not add to them. (`st_rad`/`st_teff` are correctly absent from the
feature file: `score_candidates` merges them in from the candidate list.)

**Training data: 9 real labels lost.** The label watcher calls
`compute_all_features` on every newly-confirmed planet / false positive before
appending to `training.csv`. Every one since the promotion failed with the error
above and sits as `failed` in `label_watch_queue`:

| host | label | failed at |
|---|---|---|
| TIC_21113347 | 1 | 2026-08-06 19:38 |
| TIC_125520907 | 1 | 2026-08-07 19:40 |
| TIC_230741378 | 1 | 2026-08-10 19:57 |
| TIC_65910228 | 1 | 2026-08-10 19:58 |
| TIC_345143460 | 1 | 2026-08-10 20:01 |
| TIC_156514476 | 1 | 2026-08-10 20:01 |
| TIC_302070274 | 1 | 2026-08-11 20:12 |
| TIC_341630071 | 1 | 2026-08-11 20:19 |
| TIC_30499203 | 0 | 2026-08-12 20:24 |

The last *successful* append was **TIC_453789494 on 2026-08-05 11:43** -- the
day before. The timeline is unambiguous: training-set growth stopped dead on the
promotion date. **These 9 are recoverable by re-running the watcher, but that
writes to `training.csv`, so it is NOT done here and needs explicit direction.**

### The fix

A named constant, not another inline tuple:

```python
NON_TLS_FEATURE_COLUMNS = {
    "st_rad", "st_teff",
    "crowd_flux_ratio_max", "crowd_nearest_arcsec",
    "var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period",
}
```

with a comment block recording this outage and stating the rule directly: **any
future feature added to `FEATURE_COLUMNS` that is not computed by
`compute_all_features` MUST be added to this set.** The original two-element
tuple was correct when written and became wrong silently, twice, because nothing
tied it to `FEATURE_COLUMNS`. Naming it is what makes the next promotion trip
over it.

**Nothing is weakened.** `score_candidates()` already re-validates the
*complete* 31-column row -- `blocking_cols` NaN exclusion plus a hard
`missing_required` schema check that raises `SystemExit`. A genuinely missing
crowding or variability value still drops the star; it now happens one stage
later, at the only point where those columns actually exist. This is the same
principle as the no-fit fix: name the real condition instead of collapsing it
into a generic failure.

### Validation

**(1) The exact configuration that failed, 317 real stars, real 31-column metadata:**

| population | n | produced features | still blocked on a non-TLS column |
|---|---|---|---|
| regression control (previously Success) | 80 | **79** (65 Success + 14 imputed-optional) | **0** |
| previously unfittable (from the no-fit fix) | 237 | 0 -- all `No transit fit` | **0** |

Before the fix, 64 of those same 80 failed on exactly the 7 columns. The single
remaining failure is `TIC_376669357`, already established as a harness artifact
(absent from the candidate list, so default stellar params changed TLS's grid).
The 237 correctly stay excluded for the right reason.

**(2) Fresh end-to-end pass, 40 real stars, full `extract_features` ->
`score_candidates`** (sandboxed outputs; production catalogs untouched):

* 29/40 produced a complete feature row; **0 blocked on a non-TLS column**
* all 7 previously-impossible columns non-null on **29/29** successful rows
* 12 stars correctly excluded at scoring for missing `st_rad`/`st_teff` in TIC
* **17 candidates scored**, top probability 0.994
* 9 flagged `imputed_features` (`FAP` / `transit_shape_ratio`) -- the optional-
  feature path still works and is still reported per star

**(3) Bootstrap uncertainty ensemble against those freshly-scored rows:** 32
members, 31 features, manifest feature list **identical** to the metadata list;
produced a band for **17/17**. Median band width (p84-p16) 0.108; max
`disagreement_sigma` 2.94.

### Two harness defects of my own, recorded because both produced convincing wrong answers

1. **`ModuleNotFoundError: No module named 'm06'`.** Loading the pipeline via
   `importlib.util.spec_from_file_location` works in the parent only;
   `extract_features`'s spawn-based pool re-imports by name in each child, which
   fails, and all 40 stars came back as worker errors. The first run therefore
   printed **"ZERO scored -- the outage is NOT fixed"** with the fix working
   correctly. Fixed with a real importable shim that sets `__file__` to the true
   source so every path constant resolves as in production. Production is
   unaffected: it runs the file as `__main__`, which children re-import via the
   `if __name__ == "__main__"` guard.
2. `extract_features` prints `"Feature extraction: 10/40 stars succeeded"` using
   an exact `== "Success"` test, while `score_candidates` correctly uses
   `startswith("Success")`. The real count was 29. **Cosmetic in the log only --
   no star is lost by it** -- but it understates success by the entire
   imputed-optional population and is worth correcting the next time that
   function is touched.

### Status

**FIXED and deployed to the candidate path.** Production model, `training.csv`,
promotion gate and scheduler untouched. The 9 lost training labels are flagged,
not recovered.

---

## LABEL RECOVERY AFTER THE GATE BUG -- 8 of 9 recovered. The 9th is a genuine TLS failure.

**Date: 2026-08-13, immediately after af0df0ac. Production model UNCHANGED
(0.9300, 31 features, md5 `1f0b7cb8e78ab542374eaf78fc837a6f`). No retrain
triggered. Promotion gate untouched.**

`training.csv`: **5,486 -> 5,494 rows**, md5
`e58ec25aa89476cb8f45cba665b54079` -> `10452580b9cfbb70ef0efc3520e82d07`.
Backup `training_BACKUP_pre_label_recovery_20260813_042540.csv` (gitignored,
same convention as the crowding backup).

### STEP 1 -- the 9, re-validated live before anything was written

| host | TIC | label | archive name | TOI disp | queued | lost at |
|---|---|---|---|---|---|---|
| TIC_21113347 | 21113347 | 1 | HATS-58 A | KP | 07-19 | 08-06 19:38 |
| TIC_125520907 | 125520907 | 1 | TOI-6019 | CP | 07-19 | 08-07 19:40 |
| TIC_230741378 | 230741378 | 1 | SPECULOOS-3 | -- | 07-19 | 08-10 19:57 |
| TIC_65910228 | 65910228 | 1 | NGTS-38 | -- | 07-19 | 08-10 19:58 |
| TIC_345143460 | 345143460 | 1 | TOI-1533 | CP,PC | 07-19 | 08-10 20:01 |
| TIC_156514476 | 156514476 | 1 | TOI-6884 | CP | 07-19 | 08-10 20:01 |
| TIC_302070274 | 302070274 | 1 | BD+48 740 | -- | 07-19 | 08-11 20:12 |
| TIC_341630071 | 341630071 | 1 | TOI-2147 | CP | 07-19 | 08-11 20:19 |
| TIC_30499203 | 30499203 | 0 | -- | FP | 07-19 | 08-12 20:24 |

Exactly 8 positive / 1 negative, as reported. **All 9 labels re-checked against
the LIVE archive today** (4,464 confirmed-planet TICs, 1,252 FP TICs):
**9/9 still valid**, no disposition changed since queuing.

**Duplicate check, by TIC id via `retrain_pipeline._training_tic_ids` -- the
resolver written for exactly the hostname-vs-TIC trap that caused the original
144-star duplication.** 5,358 existing training TICs resolved (direct `TIC_`
prefix plus the archive's hostname->TIC map). **0 of the 9 present under any
identifier.** All genuinely new.

**A 10th thing the queue revealed:** `TIC_230741378` did not fail on the 7
gate-blocked columns alone -- its error listed **`'snr'` as well**, a genuine
TLS output. Flagged before the run as "may still legitimately fail". It did.

### STEP 2 -- the recovery, scoped so it could not overreach

The run did **not** reimplement the append. It called
`retrain_pipeline.process_and_append_new_examples()` **unchanged**, with
`db.get_pending_watch_labels` patched to return exactly these 9 queue rows --
so the recovered rows travel byte-for-byte the same
download -> preprocess -> `compute_all_features` -> crowding -> variability ->
reindex -> append path as every other row in the file, and the 94 unrelated
`pending` labels were structurally unreachable. Asserted before running:
target set == the expected 9; `training.csv` md5 == the pre-recovery baseline.

**Scheduler race safety:** the Flask app and its in-process scheduler were live,
and the retrain tick (24 h cadence, last fired 2026-08-12 20:26 UTC) calls this
same function. `launchctl bootout` for the duration, `bootstrap` after --
verified stopped before the first write and running again after the last.

### STEP 3 -- integrity, before and after

| check | result |
|---|---|
| row count | 5,486 -> 5,494, **+8** |
| column list | unchanged, 49 columns |
| pre-existing rows | **identical to backup**, all 49 columns, NaN-aware |
| file bytes | backup is a **byte-exact prefix** -- a pure append, nothing rewritten |
| new rows | all 8 are targets; 0 unexpected |
| duplicate host strings | 0 |
| duplicate TIC ids | 0 |
| hostname-named row colliding with a recovered TIC | 0 |

**The 7 formerly-impossible columns are populated on the new rows** -- which is
the entire point:

| column | populated |
|---|---|
| crowd_flux_ratio_max | 8/8 |
| crowd_nearest_arcsec | 7/8 |
| var_oot_rms / var_excess / var_ls_amp / var_ls_power / var_ls_period | 8/8 each |

### The "all 31 non-null" check FAILED as written, and that check was wrong

Stated literally, 0/8 rows have all 31 features non-null. **That standard is
unmeetable and always was** -- 46.9% of the 5,486 pre-existing rows have a NaN
`FAP`. Every NaN in the new rows falls in a column that already carries NaNs in
the existing set at a comparable or higher rate:

| column | NaN rate, 5,486 existing | NaN, 8 new | verdict |
|---|---|---|---|
| FAP | 46.9% | 4 | in `OPTIONAL_FEATURES` -- NaN by design |
| transit_shape_ratio | 30.3% | 7 | in `OPTIONAL_FEATURES` -- NaN by design |
| st_rad | 4.5% (244 rows) | 2 | TIC catalog has no value for those stars |
| st_teff | 2.9% | 2 | same |
| crowd_nearest_arcsec | 1.1% (62 rows) | 1 | `crowd_flux_ratio_max` = 0.0 for that star: **no catalogued neighbour at all**, so "distance to nearest" is undefined. Correct encoding, not a gap |

**No new kind of gap was introduced.** The right standard -- no NaN in a column
that is not already legitimately NaN-bearing -- is met. Recording the failing
check rather than quietly re-scoping it, because a check that cannot pass is
itself the defect.

### STEP 3.5 -- the split, and a correction to the task's premise

The task assumed these 9 "should only ever be candidates for the training
side." **That is not the deployed policy.** `POST_FREEZE_TEST_FRACTION` was
deliberately set to **50%** on 2026-08-04: post-manifest stars are assigned by
stable md5 hash of the host name, and **5 of the 9 hash to test, 4 to train**.
Forcing all 9 to train would have biased the split, so the policy was followed,
not overridden. Realised assignment of the 8 recovered: **4 train, 4 test**.

What actually matters is guaranteed:

* **none of the 9 is in the frozen manifest** (4,392 train / 1,099 test hosts)
* **frozen test set: 1,098 -> 1,098 stars, membership identical**
* **no manifest star changed side**
* full split 4,386/1,100 -> 4,390/1,104

### STEP 4 -- the scheduler counts them

All 8 now `status='processed'` with a fresh `processed_at`.
`count_processed_watch_labels_since('2026-08-02 11:20:52 UTC')` returns
**10 / threshold 50** -- the 8 recovered plus the 2 that succeeded before the
bug. They are visible to the retrain trigger and count toward it. **Threshold
not crossed; no retrain fired, and none was triggered by hand.**

### The one that stayed out: TIC_230741378 (SPECULOOS-3)

Reproduced directly. TLS runs to completion on 18,319 points over a 26.6 d
baseline, with real stellar params (R\* = 0.126 R_sun, Teff = 2822 K -- an
ultracool dwarf), and dies inside its own statistics:

```
transitleastsquares/stats.py:458: RuntimeWarning: divide by zero
  snr_pink_per_transit[i] = (1 - mean_flux) / pinknoise
```

`pinknoise` is 0, so `snr` comes back non-finite. `snr` is a genuine TLS output
and a genuine blocking feature -- **the same class of correct exclusion as the
237 no-fit stars, not gate-bug residue.** It remains `failed` in the queue with
an accurate reason. Appending it would mean fabricating an `snr`.

### Minor defect found, NOT fixed (out of scope)

`db.mark_watch_label_processed` does not clear `error_message`. All 8 recovered
rows now read `status='processed'` while still carrying the old gate-bug failure
text -- which is actively misleading to anyone auditing the queue later, and
briefly misled this report. One-line fix in `web/db.py`, flagged for direction
rather than taken.

### Status

**RECOVERY COMPLETE: 8 of 9.** The 9th is correctly excluded on a real TLS
failure and is not recoverable without fabricating a feature. Production model,
promotion gate and scheduler configuration untouched; no retrain triggered.
**The incident opened by the crowding/variability promotions on 2026-08-06 is
now closed end to end: bug found, root cause named, fix deployed and validated,
data recovered.**

---

## FOUR-THREAD CLOSEOUT AUDIT -- all four confirmed closed; 5 loose ends resolved; 1 new lead with a real number

**Date: 2026-08-13. Production UNCHANGED: 0.9300 / 31 features / md5
`1f0b7cb8e78ab542374eaf78fc837a6f`, verified before and after.
`training.csv` 5,494 rows / md5 `10452580b9cfbb70ef0efc3520e82d07` (post label
recovery). Nothing built, nothing promoted, no closed experiment re-run.**

### PART 1 -- documentation status of the four threads: COMPLETE, no gaps

| thread | entry | key numbers present |
|---|---|---|
| Cross-mission (Kepler/K2) | "CROSS-MISSION ... CLOSED AT PART 1" | 36.4% yield wall; K2 domain AUC 0.9973; full-pull projection **+0.0061 vs 0.0097 MDE**; PlanetNet-MMG real-but-unverifiable |
| Synthetic injection | "THREE DIFFERENT PROPOSALS" + RAVEN entry + VAE entry | v1 domain AUC 0.9654; RAVEN 0.95-0.97 on detection AND shape independently; 7/31 features uncomputable; VAE kill criterion |
| CETRA | "CETRA (arXiv:2503.20875) FEASIBILITY" | no NVIDIA GPU; `period_max` inherited from baseline per CETRA's own paper; 16/22 TLS features have no equivalent; FAP ranks 5/31 |
| BLS / stacking | "BLS AS A COMPLEMENTARY DETECTOR + 2-SECTOR STACKING" | drift wall 9.4 d median budget; 2-sector escape **5.3%**; concatenation deployed separately (1088e3ee) |

**No documentation gap found. Nothing rewritten.** Every one of the five loose
ends below was already flagged in the file -- this entry resolves them, it does
not discover them.

### PART 2 -- the five loose ends

#### 1. The 9 lost training labels -- ALREADY DONE

Completed and verified earlier the same day; see "LABEL RECOVERY AFTER THE GATE
BUG". **8 of 9 recovered**, byte-exact append, frozen test set unchanged at
1,098. The 9th (`TIC_230741378`, SPECULOOS-3) is excluded on a genuine TLS
failure. Nothing outstanding.

#### 2. SAP flux fallback -- MEASURED. Recommend a scoped pilot. NOT built.

The claim on record rested on one star. Measured across **all 237**
unfittable stars, comparing PDCSAP against SAP in the raw files (which already
carry `sap_flux` -- **no re-download needed**):

| case | n | share |
|---|---|---|
| both PDCSAP and SAP unusable | 185 | 78.1% |
| **PDCSAP bad, SAP CLEAN** | **52** | **21.9%** |

The 52 are not marginal. Their PDCSAP carries **27-44% negative flux values** --
physically impossible for photometry, and robust sigma of 1.0-6.5 -- while their
SAP is clean at robust sigma **0.022-0.048**, inside the gate. For these stars
the question is not "which photometry is better"; PDCSAP is simply broken.

**Recommendation: a scoped pilot is worth it, and this is a real decision to
make, not something to build unilaterally.** Scope: run the existing
`compute_all_features` on SAP-derived processed curves for these 52, and report
how many produce valid features and what their SDE distribution looks like
against the pool. **The confound must be carried, not waved away:** SAP is not
background-subtracted or systematics-corrected, so any feature it produces comes
from a different photometry pipeline than every other row in the project -- which
is precisely the kind of domain shift that closed the synthetic and cross-mission
threads. A SAP-derived candidate would need its own provenance column and its own
OOD check before it could be ranked beside a PDCSAP one. Estimated effort: half a
day. **Ceiling on the payoff: 52 stars, ~2% of the 2,465-star pool.**

#### 3. `transit_shape_ratio` permutation importance -- MEASURED. Stays closed.

Never measured before; measured now (`shape_ratio_importance.py`), same protocol
as the secondary-eclipse run (frozen test mask, SEED 20260812, n_repeats 10,
roc_auc) with **both columns ranked in ONE run** so they are comparable.
Baseline frozen-test AUC 0.9294.

| feature | rank | importance | NaN rate |
|---|---|---|---|
| `secondary_eclipse_depth` | **8/31** | +0.00721 +/- 0.00132 | 2% |
| `transit_shape_ratio` | **20/31** | **+0.00193 +/- 0.00056** | 30% |

**The answer is neither of the two options the question offered.**
`transit_shape_ratio` is **not dead weight** -- shuffling it costs a consistently
positive amount, 3.4 sd from zero. But it is **not load-bearing in the sense
`secondary_eclipse_depth` is** either: it costs **3.7x less**, and at **20% of
the 0.0097 MDE** its contribution is far below what this test set can resolve.

So it lands in exactly the trap the secondary-eclipse work already mapped:
**measurably non-zero, too small for any fix or retirement to be provable.**
Fixing the phase convention or retiring the column would both move the model by
~0.002 AUC in an unknown direction, unmeasurable on 1,098 test stars. **Closed
on the same grounds the secondary-eclipse swap and retirement were closed --
and now for a measured reason rather than an unmeasured suspicion.**

(Top of the ranking, for the record: `st_rad` +0.0367, `st_teff` +0.0241,
`var_oot_rms` +0.0162, `crowd_flux_ratio_max` +0.0139, `FAP` +0.0107. The two
2026-08 promotions hold ranks 3 and 4.)

#### 4. Per-sector normalisation before concatenation -- HYPOTHESIS REFUTED. Closed.

The recorded hypothesis: ~39% of concatenated curves fail the flux gate because
of per-sector flux-level offsets at the boundary, fixable by normalising each
sector before concatenating.

Measured on the **271 real processed multi-sector curves** (median 7 segments),
which reproduce the recorded failure rate -- **35.8% fail**, against the ~39% on
record, so this is the right population and stage:

| quantity | value | gate |
|---|---|---|
| per-sector offset spread | median **0.00014** (p90 0.042) | median must be within 0.01 of 1 |
| robust sigma, before | median 0.00488 | must be < 0.05 |
| robust sigma, after per-sector normalisation | median **0.00488** | -- |
| **curves rescued** | **0** | -- |
| curves newly broken | 0 | -- |

Among the 97 failing curves: median robust sigma **0.7934** -- sixteen times the
0.05 ceiling -- while median `|median - 1|` is **0.00000**. **The median level is
already perfect; the failures are pure scatter.** Preprocessing normalises each
file before concatenation, so the boundary offsets the hypothesis blamed are
~0.014% and were never the problem.

**These are intrinsically variable stars, not mis-assembled curves** -- the same
population the pre-TLS sigma cut was backed out for after it rejected 6 genuine
detections. Per-sector normalisation would recover **0 of 97**. Removed from the
open list.

**A methodology note on my own first pass:** I initially ran this on RAW PDCSAP
and got "2 of 120 rescued", which pointed the same direction for the wrong
reason -- raw flux is un-normalised, so its sigma (~0.49) is not comparable to a
gate that operates on processed curves. Re-run on the correct stage before
reporting. The refutation stands on the processed measurement, not the raw one.

#### 5. CETRA's mono-transit capability -- CONFIRMED CPU-FEASIBLE. Separable from CETRA.

The CETRA entry's LEAD C already argued the capability "is not intrinsically
GPU-bound". Confirmed concretely: **`MonoTools`** (H. Osborn) is a real,
`pip install MonoTools`, **CPU** package built exactly for this -- detection,
vetting and modelling of monotransits and unknown-period planets, on a
PyMC v5 / pytensor + celerite backend. No CUDA anywhere in that stack. Its
`MonoSearch` component also vets detected monotransits against variability,
asteroids and background EBs, which overlaps this project's existing
centroid/crowding checks.

**So the capability does NOT stay closed with CETRA.** CETRA remains correctly
not pursued (GPU-blocked, inherits the period ceiling, forfeits 16 of 22
features). Mono-transit search is a **separate, CPU-reachable capability this
pipeline lacks entirely**, and it is the one thing in this whole four-thread
sweep that is genuinely new rather than a variation on something already closed.

**Not scoped further here, deliberately.** It is a detection-stage addition, not
a model change: a new search mode producing a new kind of candidate with no
period, which the 31-feature model **cannot score** (`period`,
`transit_count`, `odd_even_mismatch` and most of the rest are undefined for a
single event). It would need its own candidate class, its own vetting path and
its own UI treatment. That is a project, not a follow-up, and it should be
proposed and approved as one.

### PART 3 -- honest final verdict

**Three of the four threads are comprehensively closed and should not be
reopened.** Cross-mission, synthetic injection, and BLS/stacking each now have a
*quantitative* closure -- a measured wall (36.4% yield, 0.9973 domain AUC,
+0.0061 vs MDE, 5.3% drift escape), not a judgement call. Two of the five loose
ends (per-sector normalisation, `transit_shape_ratio`) are now **closed with
numbers that did not exist this morning**, and both closed negative.

**Two things are genuinely live, and only two:**

1. **SAP fallback -- 52 stars, 21.9% of the unfittable population.** Bounded,
   cheap, real, with a real confound. Worth a pilot; worth *deciding* rather
   than defaulting.
2. **Mono-transit search -- CPU-feasible via MonoTools.** The only capability
   gap in this sweep that is not a re-run of a closed idea. Also the largest
   piece of work, and the one most likely to be worth it, because it addresses
   the period ceiling *by not needing a period at all* -- the one constraint
   that survived every other attempt in this file.

Everything else across these four topic areas is done. **No further modelling
angle is being recommended, because there is not one worth recommending** -- the
learning curve says the ceiling is a data limit, the MDE says a ~0.002 feature
change cannot be proven on 1,098 test stars, and every detector-side alternative
inherits the same baseline-derived period ceiling. Manufacturing a fifth
proposal here would be padding.


---

## GP DETRENDING vs SAVITZKY-GOLAY -- NEGATIVE. The better-powered metric moves the WRONG way.

**Date: 2026-08-13. Production UNCHANGED: 0.9300 / 31 features / md5
`1f0b7cb8e78ab542374eaf78fc837a6f`. `02_preprocess.py` detrending UNCHANGED.
Nothing promoted, nothing wired in.** Detection-stage assessment only.

### PART 0 -- current detrending, confirmed exactly

`02_preprocess.process_one_file`, step 5:
`savgol_filter(window_length=min(401, n-1) forced odd, polyorder=2, mode="interp")`,
applied AFTER a 5-sigma MAD clip, then divide and renormalise to median 1.

**`MAX_FLATTEN_WINDOW = 401` is in POINTS, so the PROTECTED TIMESCALE IS
CADENCE-DEPENDENT.** The implementation is internally correct (odd, `< n`), but
its physical width varies ~90x across the real training set:

| cadence | share of 400 sampled training curves | 401-pt window |
|---|---|---|
| 20-sec | 14.5% | **2.2 h** |
| 2-min | 82.0% | 13.4 h |
| 10-min | 2.8% | 2.8 d |
| 30-min | 0.8% | 8.4 d |

The "~13.4 h" figure quoted throughout this file is the **2-min case only**.
Recorded because it is a live property of the deployed pipeline, not a bug
found here, and because a *cadence-aware* window is a far cheaper change than
any of the three methods assessed -- see the closing recommendation.

### PART 0 -- feasibility of the three proposed methods

| rank | method | verdict |
|---|---|---|
| 1 | **GP (celerite2)** | **FEASIBLE.** `celerite2` 0.3.3 installs as a pure wheel, no CUDA. O(N) scaling: ~1-3 s/star fit+predict against TLS's 60-70 s. **Detrending would be ~3-5% of existing per-star cost** -- affordable at full scale. Piloted below. |
| 2 | **CoFiAM** | **REAL BUT ARCHITECTURALLY MISMATCHED.** Verified, not assumed: Kipping et al.'s HEK method, 1-30 cosine models per epoch chosen by Durbin-Watson autocorrelation. But its protection guarantee is defined **relative to a KNOWN transit duration** -- it is a post-detection characterisation filter. Detrending here runs BEFORE TLS, when no duration exists. No package found; would be a from-paper implementation of a method that does not fit the slot. |
| 3 | **PLD** | **BLOCKED ON DATA THIS PROJECT DOES NOT KEEP.** `lightkurve.TessPLDCorrector` ships and works, so the method is available -- but PLD needs target pixel files and **this pipeline retains no pixel data**: `web/job_runner.py:1232` and `:1267` delete each TPF immediately after the centroid check, by design. Re-acquiring for ~8,000 stars is ~29 h of download and ~64 GB. PLD also corrects **pointing systematics**, not the stellar variability savgol removes -- a different problem from the one posed. |

### PART 1 -- the pilot: 60 paired trials, 180 TLS searches

`detrend_gp_pilot.py`. The existing sensitivity harness could not be reused
directly: it injects into ALREADY-FLATTENED curves, so it cannot compare
detrenders. This one injects into the **RAW** light curve, then detrends the
same injected series three ways. **One star, one injection, one noise draw ->
three arms**, so a difference cannot be a different star or a different draw.

Hosts: 60 draws from the 217 real negative TEST-split stars with a raw file.
Grid: depths [84, 250, 700, 2500] ppm x periods [2, 6, 10] d x 5 repeats.
Durations from the host's real M*/R*. Recovery, aliases and tolerance follow
the existing harness exactly. **60/60 trials usable in all three arms.**

Arms: `savgol` (production exactly), `gp_protect` (SHOTerm + jitter, undamped
period floored at 0.5 d ~ savgol's 13.4 h), `gp_tight` (floor 0.1 d).

#### Recovery rate -- nominally better, NOT significant

| arm | recovered | rate |
|---|---|---|
| savgol | 6/60 | 10.0% |
| gp_protect | 8/60 | 13.3% |
| gp_tight | 8/60 | 13.3% |

Paired McNemar exact, vs savgol: **savgol-only 1, GP-only 3, p = 0.6250** for
both GP arms. Two extra detections on four discordant pairs. **There is no
detectable difference in recovery, and this design has almost no power to find
a small one** -- the union of all three arms recovers only 9/60.

By depth, the arm ordering is not even consistent: at **84 ppm -- the
Earth-size depth this whole question is about -- savgol got 1/15 and both GP
arms got 0/15.** Nominally worse, on one detection. At 250 ppm the sign flips.
This is noise at n=15 per cell, and it should not be read either way.

#### SDE -- the well-powered measurement, and it goes the WRONG way

Every trial yields an SDE whether or not the period was recovered, so this uses
all 60 pairs instead of the 4 discordant ones:

| comparison | savgol | GP | median delta | Wilcoxon p |
|---|---|---|---|---|
| vs `gp_protect` | 11.18 | 8.85 | **-0.355** | **0.0088** |
| vs `gp_tight` | 11.18 | 8.35 | **-1.138** | **0.0003** |

**GP detrending significantly REDUCES TLS's detection statistic**, and the more
aggressive arm reduces it ~3x more. The `gp_tight` floor genuinely bound in
**28 of 60** trials (fitted rho below 0.5 d), so the aggressiveness axis is
real -- and it produces a clean damage gradient in SDE while leaving recovery
verdicts identical in **60/60** trials.

**That is the answer to "does tighter detrending remove real signal?" -- yes,
measurably.** The GP absorbs transit power into its own model of the trend. The
+3.3 pp recovery difference is 2 events on 4 discordant pairs (p = 0.63); the
SDE loss is n = 60, paired, p < 0.01. **When the underpowered metric favours a
change and the well-powered one opposes it, the well-powered one wins.**

Residual scatter barely moves: GP/savgol ratio **0.9879** -- a 1.2% tighter
curve, bought with a significant SDE loss. That is the whole trade in one line.

### PART 1.3 -- the variability features: architecturally safe, and the counterfactual measured

**Architecturally unambiguous.** Every `var_*` consumer reads a RAW, pre-flatten
file -- `06.add_variability_features` -> `RAW_FOLDER`;
`retrain_pipeline._variability_for_raw` -> `raw_path`;
`eightsector_build_pool` -> `raw_dir=RAW_DIR`. Detrending lives in
`02_preprocess.process_one_file`, which READS raw and WRITES `data/processed/`,
never back. **A detrender swap changes the TLS-input copy only, by construction
rather than by convention.**

Confirming a no-op would prove nothing, so `detrend_variability_isolation.py`
measures the **counterfactual**: what each detrender would destroy if it were
ever wrongly applied upstream of the variability path (30 stars, ratio of the
detrended value to the raw value production actually uses):

| feature | savgol | gp_protect |
|---|---|---|
| var_oot_rms | 0.97 | 0.95 |
| var_excess | 0.97 | 0.94 |
| var_ls_amp | 0.51 | **0.24** |
| var_ls_power | 0.36 | **0.087** |
| var_ls_period | 0.14 | **0.11** |

The three Lomb-Scargle rotation features are **destroyed** by either detrender,
and **the GP is strictly worse than savgol** -- it retains 8.7% of
`var_ls_power` against savgol's 36%. Median measured rotation period collapses
from **1.96 d to 0.22 d**. Unsurprising: modelling rotation is exactly what a GP
is for.

**So adopting a GP would make the raw/processed separation MORE safety-critical,
not less.** The architecture that must be refused if ever proposed: detrending
the raw file in place, or handing a detrended curve to `variability_for_raw`.
Both are the mistake `add_variability_features`' own docstring already warns
about, and a GP would make it roughly 4x more damaging.

### PARTS 2 AND 3 -- NOT ENTERED, correctly

Part 2 was gated on Part 1 showing real promise. It does not: no significant
recovery gain, and a significant SDE loss. Re-deriving TLS features for 5,494
training stars under a detrender that measurably lowers SDE would be a
retroactive revision of the entire feature table in exchange for a measured
negative. No modelling was run, no training data touched.

### Verdict

**CLOSED. GP detrending is feasible and cheap, and it does not help.** CoFiAM
is architecturally mismatched to a pre-detection pipeline. PLD is blocked on
pixel data this project deliberately does not retain and addresses a different
noise source anyway.

**Do not re-propose "better detrending" as an SNR lever without new
information.** What would justify reopening: a detrending method that provably
preserves in-transit points (e.g. an iterative fit with transit masking AFTER a
first-pass TLS detection -- a fundamentally different, two-pass architecture),
or a test with enough trials to resolve a few-percentage-point recovery
difference, which needs hundreds of injections rather than 60.

**The one cheap idea this surfaced, NOT built and NOT recommended without its
own test:** the savgol window is capped in POINTS, so 14.5% of training stars
(20-sec cadence) are detrended with a 2.2 h protected timescale rather than
13.4 h -- six times more aggressive than the design intent, on stars where
transit durations are often 2-4 h. **A cadence-aware window would be a
~5-line change** and is far better motivated than any of the three methods
assessed here. It is a separate task with its own validation, not a change to
make on the strength of this entry.

### Process notes

* My time estimate was wrong again: 13 min projected from a single-trial smoke
  test (94 s), 55 min actual under 7 parallel workers.
* `report()` imported `statsmodels`, which is not installed here -- it would
  have crashed after the 55-minute run. Caught mid-run and replaced with the
  exact binomial form of McNemar's test via scipy, which is the same test.
* `celerite2` 0.3.3 was pip-installed into the environment for this pilot. It
  is not imported by any production code path.

---

## CADENCE-AWARE DETRENDING WINDOW -- CLOSED. Confound gate FAILED, and the fix is a wash anyway.

**Date: 2026-08-13. Production UNCHANGED: 0.9300 / 31 features / md5
`1f0b7cb8e78ab542374eaf78fc837a6f`. `02_preprocess.py` UNCHANGED (md5
`fd4ead26896059037fe59b2d53141249`). `training.csv` UNCHANGED (5,494 rows, md5
`10452580b9cfbb70ef0efc3520e82d07`). Nothing promoted, nothing reprocessed.**

The premise was correct and well motivated: `MAX_FLATTEN_WINDOW = 401` is a cap
in POINTS, so the physical protected timescale varies ~90x across the training
set, and the "~13.4 h design intent" holds only for the 82% at 2-min cadence.
**Two independent findings then killed it, either of which is sufficient.**

### PART 0 -- THE GATE FAILED, far harder than the multi-sector precedent

Cadence measured directly from all 5,494 training stars' processed curves
(`cadence_class_confound.py`), never read from a cached table.

| cadence | negatives | positives | % of neg | % of pos | positive rate |
|---|---|---|---|---|---|
| 20-sec | 79 | 324 | 6.85% | 7.46% | 80.4% |
| 2-min | 913 | 3,945 | 79.18% | 90.88% | 81.2% |
| 10-min | 35 | 30 | 3.04% | 0.69% | 46.2% |
| **30-min** | **126** | **42** | **10.93%** | **0.97%** | **25.0%** |

Overall class balance is 79.0% positive. **The 30-min bucket is 25.0% positive
-- an 11.3x enrichment of negatives.**

**"Affected by the fix" (i.e. non-2-min): 9.12% of positives vs 20.82% of
negatives.**

    Fisher exact   OR = 0.382,  p < 1e-6
    chi-square     chi2 = 352.45, dof 3, p = 4.4e-76
    affected-rate difference  -11.69 pp,  z = -11.03

**The training-side multi-sector rollout was permanently excluded on OR 3.74,
p = 0.0034. This is the same disqualifying pattern, seventy orders of magnitude
further past the line.**

The mechanism is a selection effect that cannot be sampled away: confirmed
planets are preferentially on TESS's 2-min target list, while TOI false
positives include many stars observed only in coarse-cadence FFI data. So
"reprocess only the non-2-min stars" would apply a systematic feature change to
a population 2.7x enriched in negatives -- manufacturing class separation
exactly as the multi-sector rollout would have.

**PARTS 2 AND 3 NOT ENTERED. No training star was reprocessed. `training.csv`
was never opened for writing.**

### PART 1 -- the fix still had a live question: deploy to the CANDIDATE path?

The candidate path carries no class label, so it cannot be confounded this way
-- the same reasoning that let multi-sector concatenation ship to pools while
being permanently barred from training. So the fix was built and tested for
that use only.

#### The regression gate: the 82% baseline case is provably untouched

`TARGET_PROTECTED_HOURS` is DERIVED as `401 * 2.0 / 60.0`, never written as a
bare `13.4`. **This constant has caused real unit bugs in this project before**,
and the derivation is what makes the no-op provable rather than asserted:

* `cadence_aware_window()` returns **exactly 401** for **all 4,858** real
  2-min-cadence training stars (measured cadences 1.9998945 - 2.0002020 min)
* identical to production's `choose_savgol_window()` at n = 60, 101, 402,
  1,000, 20,000

#### What the fix does to the populations it targets

54 paired trials on a sample enriched for non-2-min cadence, injecting into RAW
curves so the window is the only variable (same harness as the GP pilot):

| bucket | n | old pts | new pts | old protected | new protected | transit spans |
|---|---|---|---|---|---|---|
| 20-sec | 18 | 401 | 2,405 | 2.23 h | 13.36 h | 452 pts |
| 10-min | 18 | 401 | 160 | 44.6 h | 13.28 h | 58.9 pts |
| **30-min** | 18 | 401 | **27** | 200.5 h | 13.50 h | **8.2 pts** |

#### Recovery: it helps exactly where predicted, hurts exactly where predicted, and cancels

| bucket | n | old | new | old-only | new-only | McNemar exact p |
|---|---|---|---|---|---|---|
| 20-sec | 18 | 1 | **4** | 0 | 3 | 0.2500 |
| 10-min | 18 | 6 | 6 | 1 | 1 | 1.0000 |
| 30-min | 18 | **9** | 6 | 3 | 0 | 0.2500 |
| **POOLED** | **54** | **16** | **16** | **4** | **4** | **1.0000** |

**A perfect wash: 16 vs 16, four discordant pairs each way, p = 1.0000.**

SDE, paired over all 54 trials, is null everywhere -- pooled median delta
**+0.24, Wilcoxon p = 0.4208**; no bucket reaches p < 0.22. Unlike the GP pilot,
where the well-powered metric moved significantly against the change, here it
simply does not move.

#### The real finding: physical width is the WRONG invariant at coarse cadence

The `transit spans` column explains the whole table. Holding the *physical*
window fixed at 13.4 h means holding the *point count* fixed only if cadence is
fixed. At 30-min cadence the new 27-point window sits against a transit that
spans **8.2 points** -- a polyorder-2 savgol fit over 27 points will partially
absorb an 8-point dip, so the "fix" detrends the transit away. At 20-sec the
2,405-point window sits against a 452-point transit, comfortably protected.

**So the correct invariant is the window-to-transit-duration ratio in POINTS,
not physical hours.** The current 401-point cap is accidentally right for coarse
cadence for exactly this reason, and wrong for fine cadence. That is a genuinely
non-obvious result and it is the useful output of this investigation.

#### Variability isolation -- architecturally safe, re-verified not assumed

Architecture is unchanged from the GP pilot and re-confirmed: every `var_*`
consumer reads a RAW file (`add_variability_features` -> `RAW_FOLDER`;
`_variability_for_raw` -> `raw_path`), detrending writes only
`data/processed/`. A window change is invisible to that path **by
construction**.

Quantitative counterfactual on 24 non-2-min stars (ratio to the raw value
production actually uses):

| feature | old (401 pts) | new (cadence-aware) |
|---|---|---|
| var_oot_rms | 0.989 | 0.928 |
| var_excess | 0.989 | 0.928 |
| var_ls_amp | 0.050 | 0.452 |
| var_ls_power | 0.005 | 0.412 |
| var_ls_period | 0.277 | 0.088 |

**Neither window is safe for the variability path** -- which is the point:
production reads RAW, and that separation, not the choice of window, is what
protects the deployed feature.

### Verdict

**DO NOT PROMOTE, on two independent grounds.**

1. **Training data: barred.** Cadence is class-correlated at p = 4e-76. This is
   the multi-sector exclusion again, and it is permanent for the same reason.
2. **Candidate path: no reason to.** The fix is a measured wash (16 vs 16,
   p = 1.0000), because its 20-sec gain and its 30-min loss cancel.

**Scoped follow-up, NOT built and NOT recommended without its own test:** a
*conditional* window -- `max(401, cadence_aware)`, still capped at `n-1` --
would take the 20-sec lengthening while leaving coarse cadence on the existing
401-point cap, capturing the only arm that trended positive and avoiding the one
that trended negative. **It is untested**, and both per-bucket signals rest on
three discordant pairs (p = 0.25), so this is a hypothesis generated by this
pilot, not a result of it. It would also apply to ~7.5% of candidates, so
demonstrating it would need far more than 18 trials per bucket.

**Do not re-propose a cadence-aware window for TRAINING data.** The confound is
structural, not a sampling artifact.

### Process note

Estimated ~30-45 min for the pilot; actual wall time **8.8 min**. Wrong in the
useful direction this time, but wrong again.

---

## THREE ExoMiner++-INSPIRED FEATURES -- one is a duplicate-in-part, one is blocked, none clears

**Date: 2026-08-13. Production UNCHANGED: 0.9300 / 31 features / md5
`1f0b7cb8e78ab542374eaf78fc837a6f`. `training.csv` UNCHANGED (5,494 rows, md5
`10452580b9cfbb70ef0efc3520e82d07`). Nothing promoted.** MDE on this test set
~0.0097; frozen test 1,098 stars.

### PART 0 -- Lomb-Scargle duplicate check: PARTIAL duplicate, with a genuinely new angle

`var_ls_amp`, `var_ls_power`, `var_ls_period` **are already deployed and are
Lomb-Scargle features**. Exact definition read from `variability_features.py`:
`astropy.timeseries.LombScargle` on RAW flux that has been quality-filtered,
5-sigma clipped, **out-of-transit masked** (`|phase| > duration/period`, so the
transit cannot inflate its own statistic) and 10-minute binned; peak by
`argmax` of `autopower` over 0.2-13 d; amplitude from the fitted model.

So the three deployed columns answer **"is this star variable, at what period,
how strongly"**. They do **not** answer the proposal's question, which is
whether the dominant periodicity sits **at or near a harmonic of the
candidate's OWN transit period**. That is a *relation between two numbers*, not
a new measurement, and it is computed nowhere in this project.

**Verdict: not a duplicate of the measurement, but it is a derived quantity --
zero new data, zero new compute.** `ls_period_match` is defined as the minimum
over harmonics n in {1, 2, 1/2, 3, 1/3} of `|log(P_ls / (n * P_transit))|`, in
log space so 2:1 and 1:2 are symmetric -- the same alias family the
secondary-eclipse work already characterised. Hypothesis: a transit period
sitting on the star's own rotation period, or a harmonic of it, suggests
stellar activity rather than a planet.

### PART 3 -- systematics flags: FEASIBILITY SPLITS THE PROPOSAL IN TWO

**Momentum dumps are NOT recoverable from retained data.** Every download in
this project calls a bare `.download()`, and lightkurve's
`TessQualityFlags.DEFAULT_BITMASK = 17087` **includes bit 32, "Desaturation
event"** -- the momentum-dump flag. Those cadences are removed *at download
time*, before any CSV is written. Measured across 60 raw files / 1,063,181
cadences: **bit 32 appears 0 times**, as the bitmask guarantees.

**Scattered light IS retained** -- bits 2048/4096 are not in the default mask:

| bit | flag | cadences | % | files (of 60) |
|---|---|---|---|---|
| 4096 | Straylight2 | 86,240 | **8.11%** | 47 |
| 32768 | Insufficient Targets for Error Correction | 10,749 | 1.01% | 18 |
| 1024 | Cosmic ray in collateral data | 2,000 | 0.19% | 2 |
| 2048 | Straylight | 462 | 0.04% | 2 |

So only the straylight half was buildable, and it was built. Recovering
momentum dumps would need a **full re-download of all 5,494 training stars plus
both candidate pools with `quality_bitmask=0`** -- same class of blocker as PLD,
though cheaper (light curves, not pixel files). **Not attempted: the straylight
half, which IS available, turned out to carry no signal at all (below), so
there is no evidence the momentum-dump half would either.**

### The pre-model battery, all three proposals

| feature | NaN pos | NaN neg | single-feature AUC | \|AUC-0.5\| | max \|rho\| vs the 31 |
|---|---|---|---|---|---|
| `ls_period_match` | 2.6% | 1.2% | **0.5887** | 0.0887 | **0.301** (`period`) |
| `trend_slope_ppm_day` | 0.2% | 0.1% | 0.6087 | 0.1087 | **0.826** (`var_ls_amp`) |
| `trend_amp_frac` | 0.2% | 0.1% | 0.6110 | 0.1110 | **0.829** (`var_ls_amp`) |
| `straylight_frac` | 0.2% | 0.1% | **0.4964** | **0.0036** | 0.430 |
| `flagged_frac` | 0.2% | 0.1% | 0.5326 | 0.0326 | 0.471 |

**|ecliptic latitude| differs between classes: KS D = 0.169, p = 4.2e-23**
(medians 59.2 vs 52.0 deg). The documented spatial confound is live, which is
why the control arm below is a stratified AUC and not a correlation.

| feature | rho \|gal b\| | rho \|ecl lat\| | AUC by ecliptic-latitude quartile |
|---|---|---|---|
| `ls_period_match` | **-0.004** | +0.062 | **[0.596, 0.547, 0.607, 0.603]** stable |
| `trend_slope_ppm_day` | -0.248 | +0.149 | [0.524, **0.434**, 0.709, 0.705] unstable |
| `trend_amp_frac` | -0.254 | +0.161 | [0.522, **0.435**, 0.715, 0.710] unstable |
| `straylight_frac` | +0.127 | -0.111 | [0.497, 0.429, 0.475, 0.588] |

#### PROPOSAL 2 (trend/slope) -- REJECTED before modelling, on two counts

First, the mechanism question was answered before computing anything: savgol
divides by a ~13.4 h trend, so **anything slower -- including a full-baseline
linear slope -- is already divided out**. A trend feature on PROCESSED data is a
null re-derivation. It was therefore computed on RAW, the same reasoning that
put the variability features on raw.

On raw it is **redundant**: |rho| **0.826 / 0.829 with `var_ls_amp`**, above the
0.80 threshold. A slow trend on a raw curve largely *is* the low-frequency
variability amplitude already deployed. And it is **spatially unstable** --
quartile AUCs swing from 0.434 (below chance) to 0.715, with rho -0.25 against
|galactic b|. Nominally the strongest single-feature AUC of the three, and the
control arm shows why that number cannot be trusted.

#### PROPOSAL 3 (straylight) -- REJECTED. No signal.

`straylight_frac` is at **chance: AUC 0.4964, |AUC-0.5| = 0.0036**, despite
being abundant (8.11% of cadences, 47/60 files). `flagged_frac` reaches only
0.5326 and is likewise spatially unstable. **The prior concern -- that a
straylight feature would be a proxy for ecliptic position -- did not even get
the chance to bite: there is nothing there to be confounded.**

#### PROPOSAL 1 (`ls_period_match`) -- cleared EVERY pre-model check, then did not clear the model

* not redundant: max |rho| **0.301**, far under 0.80
* spatially clean: rho with |galactic b| **-0.004**, quartile AUCs stable
* **production availability: 100% of Success rows on BOTH pools** (488/488 main,
  69/69 widesector) -- checked before modelling, per the standing rule
* one caveat, reported: availability is mildly class-correlated (97.35% of
  positives vs 98.79% of negatives, OR 0.452, p = 0.003). The magnitude is
  **1.4 pp**, against the **31 pp** that disqualified multi-sector, and the NaN
  pattern is **inherited from `var_ls_period`, an already-deployed column** --
  so this adds no new missingness structure to production.

**Result, 12 training bootstraps, production's exact recipe, frozen split:**

| arm | mean delta | sd | min | max | positive | >= MDE |
|---|---|---|---|---|---|---|
| 31 -> 32 (full frozen test) | **-0.0006** | 0.0018 | -0.0040 | +0.0021 | 5/12 | **0/12** |
| 31 -> 32 (2-min subset, n=968) | -0.0006 | 0.0019 | -0.0051 | +0.0023 | 5/12 | 0/12 |

    95% CI on delta  [-0.0035, +0.0019]   ci_lo > 0: NO
    AUC    0.9207 -> 0.9201
    Brier  0.0851 -> 0.0852
    ECE    0.0404 -> 0.0394

**Does not clear.** Mean delta is negative, the CI straddles zero, and no
bootstrap reaches the MDE. ECE improves by 0.0010, which is not a promotion
criterion and is well inside noise.

### Verdict

| proposal | outcome |
|---|---|
| 1. LS peak / period matching | **NOT a duplicate** -- the deployed LS columns measure variability, this measures alias proximity. Genuinely novel, cleanly non-redundant, spatially robust. **Does not clear: -0.0006, CI [-0.0035, +0.0019], 0/12 at MDE. DO NOT PROMOTE.** |
| 2. Overall trend / slope | **REJECTED pre-model.** Null on processed data by construction; on raw it is redundant with `var_ls_amp` (\|rho\| 0.83 > 0.80) and spatially unstable (quartile AUC 0.434-0.715). |
| 3. Systematics flags | **SPLIT. Momentum dumps: FEASIBILITY-BLOCKED** -- destroyed at download by `DEFAULT_BITMASK` bit 32; would need a full re-download at `quality_bitmask=0`. **Scattered light: available and tested, at chance (AUC 0.4964).** |

**Recommendation: promote none of them. Production stays at 0.9300 / 31
features.**

The one result worth carrying forward is negative but informative:
`ls_period_match` is a *well-motivated, non-redundant, spatially clean* feature
that still returns nothing. That is the fourth consecutive vetting-style feature
(after weak-secondary, trapezoid shape, and odd-even timing) to pass every
design check and fail the model -- consistent with the learning-curve finding
that this ceiling is a **data limit, not a feature limit**.

**Do not re-propose a re-download for momentum-dump flags** without new
information: the straylight half of the same proposal is available, abundant,
and carries no class signal, which is direct evidence against the family.

### Process note

`raw_features()` crashed on the first pass -- one archive file has a
non-standard schema with no `time` column, the same condition
`02_preprocess.validate_schema` already rejects. Guarded rather than allowed to
kill a 5,494-star pass; 5,483 stars processed, 10 with no raw file, 1
non-standard.

---

## GAIA RUWE + NSS -- **THE STRONGEST CLEAR IN THIS PROJECT'S HISTORY.** Recommended, NOT promoted.

**Date: 2026-08-14. Production UNCHANGED and untouched: 0.9300 / 31 features /
md5 `1f0b7cb8e78ab542374eaf78fc837a6f`. `training.csv` UNCHANGED (5,494 rows,
md5 `10452580b9cfbb70ef0efc3520e82d07`). Nothing promoted. This entry is a
RECOMMENDATION awaiting explicit go-ahead.**

Motivated by Armstrong et al. 2022, who reject flagged astrometric binaries
outright for KOI validation.

### PART 0.1 -- the "already-pulled-and-unused" pattern does NOT repeat a third time

Checked directly. It was real twice (crowding's `contratio`/`numcont`, stellar
density's `rho`/`logg`); this time it is **false**. The TIC batch query returns
**125 columns, none of them RUWE or NSS**:

    TIC version 20190415  -- TIC v8, built on Gaia DR2
    gaia columns: GAIA, GAIAmag, e_GAIAmag, gaiabp, gaiarp, gaiaqflag
    'ruwe' present: False       'nss' present: False

RUWE and `non_single_star` are Gaia **DR3** products and TIC v8 predates DR3, so
a genuinely new query is required.

**NSS is CHEAPER than the brief assumed.** `non_single_star` is a bitfield in the
MAIN DR3 source table (1 astrometric, 2 spectroscopic, 4 eclipsing) -- only the
detailed orbital solutions live in `nss_two_body_orbit` etc. **One query returns
both fields.**

### An infrastructure finding: the obvious route is unusable, measured not assumed

| approach | measured |
|---|---|
| `Gaia.launch_job` per star, 8 workers | **16 stars did not finish in 10 minutes** |
| Gaia TAP async `tap_upload` cross-match | hung past 10 minutes |
| **VizieR `I/355/gaiadr3`, bulk table upload** | **200 stars in 33 s = 165 ms/star** |

VizieR accepts a whole coordinate table per call -- same data, ~200x throughput,
~15 min for all 5,494 training stars. Recorded because the naive route would
have made this look infeasible.

### PART 0.2 / 0.4 -- THE AVAILABILITY-TRAP GATE: **BOTH PASS**

The check that closed the CTL and density cases: does *mere availability*
predict the label?

| field | avail (pos) | avail (neg) | Fisher OR | p | **AUC(availability)** | \|AUC-0.5\| |
|---|---|---|---|---|---|---|
| `gaia_ruwe` | 97.90% | 96.53% | 1.678 | 0.0089 | **0.5069** | **0.0069** |
| `gaia_nss` | 99.52% | 98.44% | 3.262 | 0.00047 | **0.5054** | **0.0054** |
| *(closed CTL trap, for scale)* | 78.7% | -- | -- | -- | *0.3775* | *0.1225* |

**Roughly 18x smaller than the trap that disqualified `contratio`.** The p-values
are significant only because n = 5,494; the effect is ~1.4 pp.

### PART 0.5 -- TRAIN vs SERVE on BOTH real pools: **PASSES**

This is the comparison that actually caught the crowding trap.

| field | train-neg | train-pos | main pool | widesector pool |
|---|---|---|---|---|
| `gaia_ruwe` | 96.53% | 97.90% | **95.20%** | **98.15%** |
| `gaia_nss` | 98.44% | 99.52% | **95.95%** | **98.89%** |
| *(closed crowding trap)* | *78.7%* | -- | *37.5%* | -- |

**No train/serve mismatch.** Two data-quality issues were fixed before this
table was trusted: one VizieR chunk died on a transient `ConnectionError`
(200 rows left NaN -- would have faked a gap) and was retried; and the
**widesector candidate list carries no ra/dec at all**, so coordinates were
resolved from TIC by id first (271/271).

### Redundancy: essentially ORTHOGONAL to everything deployed

| | max \|rho\| vs the 31 | vs `crowd_flux_ratio_max` | vs `crowd_nearest_arcsec` |
|---|---|---|---|
| `gaia_ruwe` | 0.147 (`st_teff`) | **0.027** | **0.001** |
| `gaia_nss` | 0.132 (`st_rad`) | **0.083** | **0.008** |

`gaia_ruwe` vs `gaia_nss`: **0.155**. The hypothesis that photometric blend and
astrometric wobble are different physical channels is **confirmed
quantitatively**, not assumed -- the deployed crowding pair and these two are
nearly independent.

### |GALACTIC LATITUDE| CONTROL ARM: clean

| field | rho \|gal b\| | AUC by \|gal b\| quartile |
|---|---|---|
| `gaia_ruwe` | +0.069 | [0.445, 0.377, 0.375, 0.389] |
| `gaia_nss` | +0.008 | [0.463, 0.416, 0.427, 0.411] |

Consistently informative in the same direction in every quartile -- not a
spatial confound, unlike the `trend_*` features closed the day before.

### Value-level signal, and the circularity check that could have sunk NSS

    RUWE > 1.4    8.02% of positives  vs  23.59% of negatives   (2.9x)
    NSS  > 0      0.83% of positives  vs  12.92% of negatives   (15.6x)

RUWE > 1.4 is the standard Gaia "likely non-single" cut (Lindegren,
GAIA-C3-TN-LU-LL-124), reported as a diagnostic; the model receives the
continuous value.

**THE CIRCULARITY CONCERN, TESTED:** the negative class is TOI false positives,
many of which are eclipsing binaries -- and NSS bit 4 *is* "eclipsing". If the
signal came from that bit, the feature would largely be Gaia restating the
label. Decomposed:

| NSS bit | n | % of positives | % of negatives |
|---|---|---|---|
| 1 astrometric | 90 | 0.62% | 5.55% |
| 2 spectroscopic | 116 | 0.28% | 9.16% |
| **4 eclipsing** | **0** | -- | -- |

**The eclipsing bit appears ZERO times.** The entire signal is astrometric and
spectroscopic orbital motion -- exactly Armstrong et al.'s mechanism, and
independent of how the label was assigned. The concern is fully resolved.

### PART 3 -- resampled model comparison, 12 bootstraps, production's exact recipe

Frozen split, frozen test 1,098 stars, MDE 0.0097.

| arm | AUC | mean delta | sd | 95% CI | positive | >= MDE | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| base (31) | 0.9198 | -- | -- | -- | -- | -- | 0.0852 | 0.0388 |
| +ruwe (32) | 0.9278 | +0.0081 | 0.0017 | [+0.0059, +0.0114] | 12/12 | 3/12 | 0.0800 | 0.0347 |
| +nss (32) | 0.9282 | +0.0085 | 0.0013 | [+0.0066, +0.0110] | 12/12 | 1/12 | 0.0798 | 0.0370 |
| **+both (33)** | **0.9339** | **+0.0142** | 0.0015 | **[+0.0124, +0.0168]** | **12/12** | **12/12** | **0.0763** | 0.0355 |

2-min-only subset (968 test stars): base 0.9122, +ruwe +0.0089, +nss +0.0103,
**+both +0.0163**.

**`+both` CLEARS on both criteria: ci_lo > 0 AND mean delta >= MDE, with 12/12
bootstraps at the MDE.** Brier improves 0.0852 -> 0.0763 and ECE 0.0388 ->
0.0355, so this is not an AUC-only artefact.

Individually **neither clears**: both have ci_lo > 0, but mean deltas of 0.0081
and 0.0085 sit *below* the 0.0097 MDE. **They are complementary, not
substitutes** -- rho 0.155, and their separate gains (0.0081 + 0.0085 = 0.0166)
are close to the joint 0.0142. Two different physical channels.

*(The 0.9198 base is the bootstrap-refit baseline, not the deployed model's
recorded 0.9300 -- the same offset every prior bootstrap comparison in this file
shows. The delta is the number that matters.)*

### NSS as a HARD EXCLUSION rule -- architectural answer, NOT built

Armstrong et al. reject flagged binaries outright. **This project's architecture
is not suited to that as a trainable feature, and the distinction matters:**

* A hard exclusion is a **vetting-layer** rule, not a model input. It belongs
  where `08_characterize_candidates.py` already puts VSX variable-star and Gaia
  blend evidence -- as a `confidence_tier` doubting line and a UI badge, with
  the candidate still visible and the reason stated.
* Encoding it as a model feature is strictly more informative: the model learns
  *how much* to down-weight, rather than being handed a threshold. The measured
  result supports that -- the continuous RUWE contributes on its own, and NSS is
  only 3.37% prevalent, so a hard cut would discard 185 training rows to encode
  what one column already carries.
* **Recommendation: soft feature for the model, and a doubting-evidence line in
  the vetting layer.** Not a filter that silently removes candidates.

### RECOMMENDATION -- promote the PAIR, pending explicit go-ahead

| field | verdict |
|---|---|
| `gaia_ruwe` | **Promote as part of the pair.** Alone: +0.0081, ci_lo > 0 but below MDE. |
| `gaia_nss` | **Promote as part of the pair.** Alone: +0.0085, ci_lo > 0 but below MDE. |
| **both together** | **PROMOTE: +0.0142, CI [+0.0124, +0.0168], 12/12 at MDE.** Larger than crowding (+0.010-0.012) and variability (+0.0092-0.0101), the two features currently deployed. |

**NOT DONE, and requiring explicit approval:** backfilling the two columns into
`training.csv` (5,494 rows), extending `FEATURE_COLUMNS` 31 -> 33, wiring the
VizieR query into `06_download_unknown.py` alongside `add_crowding_features` and
`add_variability_features`, retraining, and running the promotion gate. Each is
a production change and none was taken unilaterally.

Two caveats to carry into that decision, both small and both stated rather than
buried: availability is mildly class-correlated (1.4 pp, p = 0.0089 -- 18x under
the CTL trap but not zero), and the ~4.8% of main-pool candidates with no Gaia
match would score with two imputed columns, so they need the existing
`imputed_features` treatment.

---

## DEPLOYED: Gaia DR3 RUWE + NSS, 31 -> 33 features, 0.9300 -> 0.9402

**Date: 2026-08-14. THIS IS THE NEW NUMBER OF RECORD.**

    model      models/best_model.joblib   md5 c37f9f4bdb252d52b8c1c5487dad9e6d
    features   33   (was 31, md5 1f0b7cb8e78ab542374eaf78fc837a6f)
    frozen-test AUC  0.9294 -> 0.9402   (headline fit, same 1,098 stars)
    training.csv     5,494 rows x 51 cols, md5 3bf4a34317acbfcaf42972ee875ac0be
    rollback   models/versions/best_model_pre_gaia_1f0b7cb8.joblib

### Imputation policy -- IDENTICAL on both sides, by construction

Unmatched stars stay **NaN**. Both columns are in `OPTIONAL_FEATURES`, so:
`score_candidates` does not drop the star, the NaN is filled by production's own
`SimpleImputer(median)` **inside the fitted pipeline** (so training and serving
use the same imputer fitted on the same data), and the star is flagged in
`imputed_features`. Filling them anywhere else would put a second, different
imputer in the serving path only -- a train/serve mismatch by construction.

Coverage: training 97.62% / 99.29%; main pool Success rows 95.90% / 96.72%;
widesector Success rows 98.55% / 100%.

### The gate that lost 9 labels last time -- checked BEFORE deploying

`gaia_ruwe` and `gaia_nss` were added to **`NON_TLS_FEATURE_COLUMNS` (now 11)**
in the same edit that promoted them. Verified live after the change:

    {'gaia_ruwe','gaia_nss'} <= NON_TLS_FEATURE_COLUMNS   True
    OPTIONAL_FEATURES  ['FAP','gaia_nss','gaia_ruwe','transit_shape_ratio']

The crowding (Aug 5) and variability (Aug 6) promotions each skipped this and
silently blocked every star for a week. It cannot recur for these two.

### Backfill integrity

**A first attempt FAILED its own check and was discarded.** Round-tripping
through `pandas.read_csv -> to_csv` re-serialised every float and changed 8
cells of `chi2red_min` in the 16th significant digit
(`5.11146182632035e-09` -> `5.1114618263203505e-09`). Physically meaningless,
but it breaks the byte-identical standard, and "negligible" is not a judgement
to make silently on production training data. Restored from backup and redone as
a **textual column append**, so:

    every original line is a byte-exact PREFIX of its new line   True
    row count 5,494 -> 5,494, host order identical               True
    pre-existing data identical                                  True
    both new columns round-trip exactly                          True

**Candidate pools: a real gap was found and closed.** Joining the validated
fetch by host (never positionally) exposed that `unknown_features.csv` holds
**2,454 rows** while `unknown_candidate_list.csv` holds only **2,000** -- the
feature table accumulated across runs. Coverage came out at 55.5%. The 1,081
un-queried hosts were then fetched **with the newly wired production function**
(987 matched), taking the main pool to 95.72% / 96.58%. This is the variability
deployment's pool-gap lesson applied rather than rediscovered.

### Full-scale retrain reproduces the validation exactly

| metric | 31 features | 33 features | delta |
|---|---|---|---|
| frozen-test AUC (headline fit) | 0.9294 | **0.9402** | +0.0108 |
| 2-min subset | 0.9216 | 0.9343 | +0.0127 |
| bootstrap mean (12 resamples) | 0.9198 | 0.9340 | **+0.0142** |
| bootstrap 95% CI | -- | -- | **[+0.0124, +0.0168]** |
| positive / at MDE | -- | -- | **12/12 and 12/12** |
| nested CV pooled out-of-fold | 0.9383 | **0.9471** | +0.0088, wins 5/5 folds |
| Brier (bootstrap mean) | 0.0852 | 0.0763 | better |
| ECE (bootstrap mean) | 0.0388 | 0.0355 | better |

**+0.0142 / CI [+0.0124, +0.0168] is identical to the pre-deployment
validation**, which is the consistency check that mattered.

**One metric moved the wrong way and is recorded rather than buried:** on the
single headline fit, ECE went **0.0210 -> 0.0298**. The bootstrap-mean ECE
improves (0.0388 -> 0.0355) and Brier improves in both framings, so calibration
is not broadly degraded -- but the headline-fit ECE is worse and conformal
calibration was regenerated against the new model, which is where that is
handled.

### Live consumption proof, on a real candidate (TIC_466307646)

    gaia_ruwe  0.9 -> 1.0 -> 1.2 -> 1.4 -> 2.0 -> 3.0 -> 5.0
    p(planet)  0.9699 0.9653 0.9523 0.9413 0.8973 0.8903 0.8761   monotonic DOWN

    gaia_nss   0 -> 1 -> 2
    p(planet)  0.9678 -> 0.9085 -> 0.5127

    clean single star (ruwe 0.95, nss 0)  p = 0.9647
    flagged binary    (ruwe 3.0,  nss 1)  p = 0.7244    delta -0.2402

Physically correct in both channels: more astrometric wobble, or a published
non-single-star solution, lowers the planet probability.

### Downstream artifacts, both done PROACTIVELY

* **conformal_calibration.json** regenerated; `model_md5`
  `c37f9f4bdb252d52b8c1c5487dad9e6d`, n_calibration 1,104.
* **bootstrap ensemble** rebuilt: 32 members, **33 features**, and verified
  live against a real candidate (band [0.9714, 0.9861]). After crowding this
  broke silently and was only found by a downstream crash.

### TWO PRE-EXISTING PROBLEMS FOUND, NEITHER CAUSED BY THIS DEPLOYMENT

**1. The web app has been down since 2026-08-14 01:05 UTC**, ~11 hours before
this work started (last scheduler tick 920; a 62-minute gap before it is
consistent with the Mac sleeping). `launchctl` reports **exit 78 (EX_CONFIG) on
3 attempts with ZERO output** -- python never starts. The app runs perfectly
when launched from an authorized terminal, so this is a launchd/TCC-class issue
on the machine, not a code fault:

    manual start -> /health {"status":"ok","scheduler_thread_alive":true}, port 5050

The app is currently **running via a manual `nohup` start**, so the service is
live on the new model -- but it is **NOT under launchd supervision**, and will
not survive a reboot until the agent is fixed. Requires the user: grant the
LaunchAgent access to `~/Downloads` (System Settings -> Privacy & Security ->
Files and Folders / Full Disk Access), then:

```
launchctl bootout gui/$(id -u)/com.exoplanetai.app
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.exoplanetai.app.plist
```

**2. `conformal_prediction.py`'s exchangeability diagnostic has been broken
since 2026-08-06.** It reads `results/unknown_candidates/ranked_candidates.csv`,
dated **2026-08-05**, which predates the variability promotion and so lacks
`var_*`. It has failed on every run since, independent of this change. The
conformal artifact production actually consumes is regenerated and current.

### Verified after deployment

* `FEATURE_COLUMNS` (33) == metadata `feature_columns` (33) == model
  `n_features_in_` (33) -- swapped atomically with the artifact, per the
  ValueError-on-mismatch lesson.
* A real `maybe_trigger_retrain` tick runs clean against the 33-feature config
  (`10/50 new examples -- not yet`, no crash).
* `retrain_pipeline._gaia_for_host` returns values matching an independent fetch
  exactly (TIC_231620255 -> 0.992, TIC_315398983 -> 0.935). An earlier NaN was
  traced to a genuine no-Gaia-source-within-3-arcsec case, not a defect --
  confirmed by re-querying three known-good stars and getting their exact
  archived values back.

### FOLLOW-UP (2026-08-14): launchd root-caused to macOS TCC; real tick verified

**The launchd failure is NOT fixable in code.** Diagnosed with two throwaway
probe LaunchAgents rather than by assumption (both removed afterwards):

    probe 1, WorkingDirectory = .../ExoplanetAI/web
      shell-init: error retrieving current directory:
      getcwd: cannot access parent directories: Operation not permitted

    probe 2, absolute paths, no WorkingDirectory
      head /Users/.../ExoplanetAI/web/app.py  -> Operation not permitted   ABS_READ_DENIED
      ls   /Users/.../models/best_model_metadata.json -> OK                ABS_LIST_OK

**Directory entries are listable but file CONTENTS are unreadable** -- the
signature of macOS TCC protection on `~/Downloads`. Any process launchd spawns
is denied, so `python3 -u app.py` cannot read `app.py` and exits EX_CONFIG (78)
before it can log anything. That is why the log has no traceback.

This is a user-consent security boundary and was deliberately NOT worked around.
`tccutil` can only reset grants, not create them, and editing TCC.db directly is
SIP-protected and would be circumventing a security control.

**Fix requires the user, one of:**

1. **Grant Full Disk Access** to the interpreter the agent runs --
   `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3` --
   in System Settings -> Privacy & Security -> Full Disk Access, then reload:

       launchctl bootout gui/$(id -u)/com.exoplanetai.app
       launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.exoplanetai.app.plist

2. **Structural alternative: move the project out of `~/Downloads`.**
   `~/Downloads`, `~/Desktop` and `~/Documents` are TCC-protected; a path like
   `~/ExoplanetAI` is not, and launchd would work with no permission grant at
   all. This is the more durable fix but touches absolute paths in the plist,
   the DB, logs and caches, so it is offered rather than taken.

Until then the app runs via a manual `nohup` start (live on port 5050,
`/health` ok, scheduler thread alive) but is **not supervised and will not
survive a reboot**.

**A REAL RETRAIN TICK WAS VERIFIED against the 33-feature config**, forced with
`maybe_trigger_retrain(threshold=1, dry_run=True)` -- the pipeline's own dry-run
mode runs the identical retrain/compare/promote logic while writing nothing:

    10 new examples -- triggering retrain
    Challenger built as clone of production CalibratedClassifierCV
    New model test ROC-AUC 0.9386 vs production 0.9386
    Paired bootstrap (new - production): mean +0.0000, 95% CI [+0.0000, +0.0000]
    NOT promoted: CI includes zero
    production model md5 unchanged: True

Correct on every count: retraining on unchanged data reproduces the same model,
so the delta is exactly zero and the gate declines. **No ValueError and no
feature-count crash -- the 33-feature configuration survives a full tick**,
which was the real risk the atomic swap existed to avoid. (0.9386 here vs the
0.9402 headline is the retrain path evaluating on the GROWN test set of 1,104
rather than the frozen 1,098 mask.)

### RESOLVED (2026-08-14): launchd supervision restored. Two causes, not one.

**The open item from the RUWE/NSS deployment report is CLOSED.** The app runs
under real launchd supervision on the 33-feature model. It took two fixes,
because the first one uncovered a second, independent blocker.

**Cause 1 -- TCC (fixed by the user).** Full Disk Access granted to
`/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`, then
`bootout` + `bootstrap`. Verified independently here with a throwaway probe
LaunchAgent running that exact interpreter: reading `web/app.py` returned
**READ_OK**, where the same read had been `Operation not permitted` before.

**Cause 2 -- stale `com.apple.macl` attributes on the EXISTING log files
(found and fixed here).** After cause 1 was fixed the job still failed,
`runs = 22, last exit code = 78: EX_CONFIG`, with **zero output**. Bisected with
three probe agents:

| probe | WorkingDirectory | log path | result |
|---|---|---|---|
| A | none | /tmp | exit 0 |
| B | project | /tmp | exit 0, correct cwd |
| C | project | **project logs/ (NEW files)** | exit 0, correct cwd |

Probe C matched the real job's configuration and passed -- so TCC was genuinely
resolved and neither `chdir` nor the log directory was at fault. The difference
was that probes created **new** log files while the real job reopened the
**pre-existing** `launchd_stdout.log` / `launchd_stderr.log`, both carrying
`com.apple.macl` extended attributes written before the FDA grant. launchd
itself opens those files before exec, could not, and returned EX_CONFIG(78)
before python ever started -- which is why nothing was ever logged.

Fixed by rotating them aside (renamed to `*.pre_fdafix`, not deleted) so launchd
creates fresh ones. **Immediately after: `state = running`, `runs = 1`,
`last exit code = (never exited)`.**

**A third, self-inflicted confound was also cleared:** the temporary `nohup`
instance from the deployment was still holding port 5050 with PPID 1. PPID 1 is
indistinguishable at a glance from launchd supervision, and a healthy `/health`
does not say which process answered -- so the earlier "it's fixed" reading was
against that orphan, not a supervised job. Killed before the real verification.

**Supervision proven, not assumed:** `launchctl` reported `pid = 10444`, `ps`
confirmed that same pid with PPID 1, and a deliberate `kill -9` produced
`old pid 10444 -> new pid 10477`, `runs = 2`, `state = running` -- KeepAlive
demonstrably restarts it.

**Serving the right model:** `/health` `status ok`, `scheduler_thread_alive
true`, fresh tick. `/models` returns HTTP 200 and reports **0.9402**. On disk:
33 features, `test_roc_auc` 0.9402, md5
`c37f9f4bdb252d52b8c1c5487dad9e6d`, `gaia_ruwe`/`gaia_nss` present, conformal
`model_md5` matching, bootstrap ensemble at 33 features.

**Nothing degraded by the ~11.75 h outage (01:05 -> 12:51 UTC).** The retrain
tick fires on a 24 h cadence and last ran **2026-08-13 20:29:47 UTC**, so the
next was not due until 2026-08-14 20:29 UTC -- **outside the outage window, so
no tick was missed**. The Update scheduler is disabled (`enabled = 0`), so no
candidate runs were skipped. No run is in a non-terminal state (3 completed /
6 failed, all dated 2026-07-29). `processed_watch_labels` 147, unchanged.
**Zero label-queue failures during or after the outage window, and zero
failures mentioning gaia** -- confirming the two new columns block nothing.

---

## THIRD CENTROID PROPOSAL -- DUPLICATE OF BOTH PRIOR ONES. Nothing built. READ BEFORE PROPOSING A FOURTH.

**Date: 2026-08-14. Production UNCHANGED: 0.9402 / 33 features / md5
`c37f9f4bdb252d52b8c1c5487dad9e6d`. Nothing implemented, nothing retrained.**
Frozen test still 1,098 stars, so the **MDE is unchanged at ~0.0097**.

**Proposed:** "refine centroid shifts from flux-weighted moments", with the
closing suggestion of "even a simple statistic of in-transit vs out-of-transit
centroid", plus using **DAVE**.

### PART 0 -- the two clauses map onto the two ALREADY-CLOSED methods, by code

**Clause 1, "centroid shifts from flux-weighted moments" = METHOD (1), built
and closed.** `scipy.ndimage.center_of_mass` **is** the flux-weighted first
moment, Sum(I*x)/Sum(I). The deployed `_centroid_from_tpf`
(`web/job_runner.py:1083`) already does exactly this:

```python
in_transit_img     = np.nanmedian(flux_cube[in_transit], axis=0)
out_of_transit_img = np.nanmedian(flux_cube[out_of_transit], axis=0)
diff_img           = out_of_transit_img - in_transit_img
diff_img_clipped   = np.nan_to_num(np.clip(diff_img, 0, None), nan=0.0)
row_com, col_com   = center_of_mass(diff_img_clipped)      # <- flux-weighted moment
col_exp, row_exp   = tpf.wcs.world_to_pixel(target_coord)
shift_pixels       = np.hypot(row_com - row_exp, col_com - col_exp)
```

There is no "refinement" available here that is not already the operation:
difference image -> flux-weighted moment -> WCS comparison. **Closed at +0.0021,
CI [-0.0020, +0.0059], and again at 77.6% coverage: +0.0032, CI [-0.0012,
+0.0076].**

**Clause 2, "in-transit vs out-of-transit centroid" = METHOD (2), the rejected
surrogate.** That is `centroid(IT) - centroid(OOT)`, verbatim the 2026-08-05
proposal already documented as a duplicate and proven **strictly weaker**: the
whole-stamp photocenter moves by ~`d*r` while the difference image reports `r`
regardless, so at this dataset's 0.01-1% depths it measures **~0.004 px against
pixel-scale noise instead of the difference image's stable 1.999 px**.

**Neither clause reaches the one genuinely untested variant** already recorded
in this file -- a **noise-normalised significance** (displacement / per-star
positional uncertainty). The proposal describes flux-weighted moments (which is
the existing computation) and an IT-vs-OOT difference (which is the rejected
one). Significance is not mentioned.

### PART 0.5 -- DAVE: real, and its centroid test IS method (1)

Verified live, not assumed. DAVE (Discovery And Vetting of Exoplanets, Kostov et
al.) is real and in active use for TESS vetting. Its centroid module
**"produces a difference image by subtracting the overall in-transit image from
the out-of-transit image, then calculates the photocenter of the light
distribution by fitting the TESS pixel response function"**, and reports the
statistical significance of the offset.

So DAVE's centroid test is **the same difference-image test already built
here**. Its two refinements over this project's version are:

| DAVE refinement | status here |
|---|---|
| **PRF fitting** instead of `center_of_mass` | **BLOCKED.** `oktopus` NOT INSTALLED, `pyke` NOT INSTALLED, `dave` NOT INSTALLED, and `lightkurve.prf.tpfmodel` raises `ModuleNotFoundError` -- the PRF machinery does not exist in this environment. |
| **statistical significance** of the offset | = the already-documented untested thread, already deprioritised with reasons. |

**And DAVE has the same blocking data dependency as PLD.** It needs the pixel
data, and this project **deletes every TPF immediately after the centroid
check** by design -- `web/job_runner.py:1232` and `:1267`, both
`os.remove(tpf.path)`. Same feasibility class as PLD; closed on the same
precedent.

### A NEW measurement, because the baseline moved -- and it REFUTED my own hypothesis

The prior centroid results were measured against a **0.9030** baseline. Production
is now **0.9402** with `gaia_ruwe`/`gaia_nss` deployed, which target the same
physical question (is this a blend / unresolved binary?). I expected the Gaia
pair to have absorbed the centroid's role, which would have made the closure
easier. **It did not:**

| deployed feature | \|rho\| vs `shift_pixels` |
|---|---|
| `gaia_ruwe` | **0.012** |
| `gaia_nss` | **0.057** |
| `crowd_flux_ratio_max` | 0.229 |
| `crowd_nearest_arcsec` | 0.096 |

max \|rho\| against all 33 features: **0.312** (`chi2red_min`). **Centroid
displacement is essentially INDEPENDENT of the Gaia astrometric pair** --
astrometric wobble and a transit-time photocentre shift really are different
channels. Recording this because it is the opposite of what I expected.

Independence is not the same as usefulness, though. Measured at the current
state (5,494 training rows, coverage 71.8%):

| feature | single-feature AUC | \|AUC-0.5\| |
|---|---|---|
| `gaia_ruwe` | 0.4169 | **0.0831** |
| `gaia_nss` | 0.4383 | **0.0617** |
| `shift_pixels` | 0.5359 | **0.0359** |
| `crowd_flux_ratio_max` | 0.5277 | 0.0277 |

`shift_pixels` carries **less than half** the single-feature signal of either
Gaia column -- and those two, which are far stronger, together moved the model
by +0.0142. Centroid moved it by +0.0032 with a CI crossing zero **against a
lower baseline with more headroom**. The case is weaker now, not stronger.

Its missingness is also still class-asymmetric (31.4% of positives vs 16.4% of
negatives) at 71.8% coverage -- down from 77.6% only because `training.csv` has
grown past the 5,086 stars the centroid run covered.

### Verdict

**CLOSED AS A DUPLICATE. Do not build. Third time.**

| element of the proposal | reduces to | already-recorded outcome |
|---|---|---|
| "centroid shifts from flux-weighted moments" | method (1) | +0.0021 then +0.0032, CI crosses zero both times |
| "in-transit vs out-of-transit centroid" | method (2) | duplicate, strictly weaker, 0.004 px vs 1.999 px |
| DAVE centroid module | method (1) + PRF + significance | same test; PRF blocked (no oktopus/pyke/tpfmodel, no retained TPFs) |

**The ONLY untested thread in this space remains the noise-normalised
significance variant**, and it is *still* not recommended: it inherits a
71.8% coverage ceiling with class-asymmetric missingness, rescales a feature
whose single-feature |AUC-0.5| is 0.0359, needs per-star centroid uncertainty
infrastructure that does not exist (bootstrapping over cadences or propagating
per-pixel errors through `center_of_mass`), plus a full re-run across all 5,494
training stars and both pools -- and it would now have to clear the same
~0.0097 MDE from a **0.9402** baseline rather than 0.9030.

**A fourth centroid proposal should not be entertained unless it (a) is the
significance variant, (b) arrives with the per-star uncertainty already
computed, and (c) explains why re-scaling a 0.0359-strength feature should clear
0.0097 when the raw version twice failed to.**

---

## FOUR SHAPE/RESIDUAL SUB-PROPOSALS -- 1 duplicate, 1 blocked-by-construction, 1 tested NEGATIVE, 1 closed architecture

**Date: 2026-08-14. Production UNCHANGED: 0.9402 / 33 features / md5
`c37f9f4bdb252d52b8c1c5487dad9e6d`. Nothing promoted.** Frozen test 1,098, so
**MDE unchanged at ~0.0097**.

### PART 0 -- routing, by code and by prior result

| # | sub-proposal | verdict |
|---|---|---|
| 1 | skewness of the binned phase-folded transit | **DUPLICATE** of the closed Medium-lift Item 2 |
| 2 | ingress/egress duration RATIO | **NOT derivable** -- the existing fit is symmetric BY CONSTRUCTION |
| 3 | trapezoid fit residual (chi2/BIC) | **Zero-new-compute variant. Tested below. NEGATIVE.** |
| 4 | local-flux CNN / deep embedding | **Reopens a closed architecture question. Not built.** |

**(1) Skewness -- duplicate.** "Medium-lift Item 2: phase-folded flux
distribution statistics" already computed **in/out-of-transit skewness AND
kurtosis on the binned phase-folded curve, plus their differences** (which
cancel each star's own noise character -- i.e. already a normalisation, which is
the only thing the new wording adds). 5,380/5,631 usable, missingness AUC 0.502.
Result: **-0.0072, 95% CI [-0.0153, +0.0010]**, nested CV flat. Re-running it
under the word "normalized" would re-derive the same number.

**(2) Ingress/egress ratio -- blocked by construction, and this is a code fact,
not a judgement.** `trapezoid_shape.py:142` is:

```python
x = np.abs(phi - phi0)
```

The trapezoid is **symmetric about `phi0`**, so ingress and egress are the SAME
fitted quantity and their ratio is **identically 1 for every star**. It is
therefore *not* a cheap variant of the existing fit -- extracting it needs a new
asymmetric 6-parameter model, refitted across 5,494 training stars and both
pools. And the motivation is weak independently of cost: a real transit is
time-symmetric, so ingress/egress asymmetry measures detrending and systematics
artefacts rather than the blend physics the proposal targets. It would also
inherit `trap_vshape`'s **31.2% usable training coverage**.

**(4) CNN -- closed, and the gap has WIDENED.** A small 1D CNN over phase-folded
points is the same data-volume-limited approach already built and closed at
**0.68-0.70**. That was measured against a then-0.90+ tree model; production is
now **0.9402**, so the gap has grown from ~0.20 to **~0.25 AUC** while nothing
about the available data volume has changed -- the learning-curve work put the
requirement at roughly 10x more labelled data. **No CNN infrastructure was
built and no compute was spent.** Documentation only, as specified.

### PART 1-3 -- the one genuinely testable sub-proposal: the fit residual

**It was already computed and already saved, and had never been fed to the
model.** `trapezoid_shape.py:259` writes `trap_rmse` (the weighted RMS residual)
for training AND both pools, and the trapezoid validation's arms were **A:
+vshape, B: +vshape,t14r, C: availability only, D: +vshape|avail, E:
+vshape|sky** -- **none included the residual.** Zero new fitting required.

Also derived `trap_bic = n*ln(RSS/n) + k*ln(n)` with `RSS = n*rmse^2`, `k = 5`.
This is **not** a monotone transform of rmse across stars, because `trap_nbins`
varies per star -- so it is a genuinely different ordering for a tree model.

**Availability, checked up front (and it is excellent, unlike `trap_vshape`):**

| population | rows | trap_rmse | trap_bic |
|---|---|---|---|
| training | 5,494 | **93.5%** | 93.5% |
| main pool (Success) | 488 | **99.2%** | 99.2% |
| widesector pool (Success) | 69 | **97.1%** | 97.1% |

**Class-rate gate: PASSES.** avail 93.62% positive vs 93.06% negative, Fisher
OR 1.094, **p = 0.502**, AUC(availability) **0.5028**.

**Single-feature signal: the STRONGEST in this project's history -- and it is a
mirage.**

| feature | AUC | \|AUC-0.5\| | max \|rho\| vs the 33 | rho vs `trap_vshape` |
|---|---|---|---|---|
| `trap_rmse` | **0.6507** | **0.1507** | **0.962 (`depth_mean`)** | 0.062 |
| `trap_bic` | 0.6356 | 0.1356 | 0.932 (`depth_mean`) | 0.045 |

For scale: `gaia_ruwe`, the strongest feature promoted this project, has
|AUC-0.5| = 0.0831 -- barely half of `trap_rmse`'s. **But `trap_rmse` is 0.962
correlated with `depth_mean`, which is already deployed.** A weighted RMS
residual scales with the amplitude of the signal being fitted, so it is
substantially transit depth restated, not an independent goodness-of-fit
measure. The 0.80 redundancy threshold is exceeded outright.

**|Galactic latitude| control arm: FAILS.**

| feature | rho \|gal b\| | AUC by \|gal b\| quartile |
|---|---|---|
| `trap_rmse` | -0.236 | **[0.735, 0.816, 0.633, 0.396]** |
| `trap_bic` | -0.214 | [0.720, 0.787, 0.619, 0.386] |

Swinging from 0.816 to **below chance at 0.396** is the same spatial-confound
signature that closed the `trend_*` features.

**Modelled anyway** -- it passed the class-rate gate, and a measured null closes
harder than an argued one. 12 bootstraps, production's exact recipe, frozen
split:

| arm | AUC | mean delta | 95% CI | positive | >= MDE | Brier | ECE |
|---|---|---|---|---|---|---|---|
| base (33) | 0.9339 | -- | -- | -- | -- | 0.0763 | 0.0355 |
| +rmse | 0.9328 | **-0.0011** | [-0.0037, +0.0019] | 3/12 | **0/12** | 0.0762 | 0.0353 |
| +bic | 0.9336 | -0.0004 | [-0.0027, +0.0018] | 4/12 | 0/12 | 0.0758 | 0.0341 |
| +rmse,bic | 0.9335 | -0.0004 | [-0.0025, +0.0029] | 5/12 | 0/12 | 0.0758 | 0.0362 |
| **+rmse,vshape** | 0.9315 | **-0.0025** | [-0.0045, +0.0005] | 2/12 | 0/12 | 0.0782 | 0.0382 |

2-min-only subset: -0.0011 / -0.0002 / -0.0002 / -0.0030. **Every arm is
negative. Nothing clears. The combined-with-`trap_vshape` arm is the worst**,
settling that question too: residual and shape are not complements here.

**This is the cleanest demonstration in this file that single-feature AUC is not
a promotion criterion.** `trap_rmse` has the widest single-feature separation
ever measured here (0.6507) and moves the model by **-0.0011**, because the
information is already in `depth_mean` at rho 0.962.

### Verdict

**Promote nothing. Build nothing further.**

| sub-proposal | recommendation |
|---|---|
| skewness | **duplicate** -- closed at -0.0072, do not re-run |
| ingress/egress ratio | **not pursued** -- identically 1 in the symmetric fit; needs a new asymmetric fit for a time-symmetric quantity |
| fit residual (rmse / BIC) | **DON'T PROMOTE** -- tested, -0.0011, 0/12 at MDE, redundant at rho 0.962 and spatially unstable |
| local-flux CNN | **NOT BUILT** -- gap widened from ~0.20 to ~0.25 AUC |

**Before proposing a fifth shape feature:** the deployed shape information is
`odd_even_mismatch`, and `transit_shape_ratio` (phase-convention-broken, rank
20/31, retirement unprovable). `trap_vshape` remains positive-but-unprovable at
+0.0007 with 31.2% coverage. Skewness/kurtosis, Haar energy, trapezoid residual
and BIC are all now measured and negative. A new shape proposal needs to say
which of these it is NOT, and why it would not simply restate `depth_mean`.

---

## TWO STELLAR-ROTATION SUB-PROPOSALS -- 1 exact duplicate, 1 NOT subsumed (my first read was wrong) and tested NEGATIVE

**Date: 2026-08-14. Production UNCHANGED: 0.9402 / 33 features / md5
`c37f9f4bdb252d52b8c1c5487dad9e6d`. Nothing promoted.** MDE ~0.0097, frozen
test 1,098.

### (1) "Amplitude of the strongest sinusoid in the out-of-transit light curve" -- EXACT DUPLICATE of deployed `var_ls_amp`

`variability_features.py:165-177`, on out-of-transit-masked, 5-sigma-clipped,
10-minute-binned RAW flux:

```python
ls  = LombScargle(tb, fb)
fr, pw = ls.autopower(minimum_frequency=1/min(MAX_P, span/2),
                      maximum_frequency=1/MIN_P, normalization="standard")
i = int(np.argmax(pw))                                   # the STRONGEST peak
out["var_ls_amp"] = float(np.std(ls.model(tb, fr[i]) - np.mean(fb)) * np.sqrt(2))
```

astropy's `LombScargle.model()` at default `nterms=1` **is the best-fit pure
sinusoid** at that frequency, and for a sinusoid of amplitude A the standard
deviation is `A/sqrt(2)` -- so multiplying by `sqrt(2)` recovers **A exactly**.

Same periodogram, same out-of-transit masking, same strongest-peak selection,
same sinusoid amplitude. **`var_ls_amp` IS "the amplitude of the strongest
sinusoid in the out-of-transit light curve", and it has been in production
since 2026-08-06.** Nothing to build.

### (2) "rotation period / candidate period" -- NOT a strict subset. My first reading was WRONG, and the measurement is what corrected it.

The obvious call was "the harmonic-aware `ls_period_match` already failed, so a
simpler non-harmonic ratio is a strict subset that cannot do better." **That is
backwards.** `ls_period_match = min over n in {1,2,1/2,3,1/3} of
|log(P_ls/(n*P_tr))|` is a **deterministic, MANY-TO-ONE function of the raw
ratio** `r = P_ls/P_tr` -- it collapses which harmonic matched, and the sign and
size of the offset. **The raw ratio strictly CONTAINS `ls_period_match`.**

Checked rather than argued, including the specific question of whether the
harmonic search was adding noise:

| statistic | single-feature AUC | \|AUC-0.5\| |
|---|---|---|
| **raw ratio `P_ls/P_tr`** | **0.6028** | **0.1028** |
| min over n (`ls_period_match`, already tested) | 0.5887 | 0.0887 |
| n=1 only, `\|log r\|` | 0.5596 | 0.0596 |

Which harmonic wins the minimisation: **n=1 only 22.3%** of the time, n=3
34.6%, n=1/3 15.0%, n=2 16.2%, n=1/2 12.0%. So the harmonic search was **not**
noise -- collapsing to n=1 alone is measurably worse (0.5596) -- but taking the
`min` discards information, and `|rho|` between the raw ratio and
`ls_period_match` is only **0.329**.

**So the exception the brief allowed was actually met**: a specific, measured
reason the simpler version could differ. It proceeded to a full test.

#### Full battery

Availability, up front: **training 97.65%, main pool 100.00% of Success rows
(488), widesector 100.00% (69)**.

| check | result |
|---|---|
| single-feature AUC | 0.6028 |
| \|rho\| vs `var_ls_period` | 0.626 |
| \|rho\| vs `var_ls_amp` | 0.179 |
| \|rho\| vs `ls_period_match` | 0.329 |
| **max \|rho\| vs the 33** | **0.667 (`period`)** |
| class-rate gate | avail 97.35% pos vs 98.79% neg, OR 0.452, p 0.0030, AUC(avail) **0.4928** |
| \|gal b\| control arm | rho **-0.048**, quartile AUCs **[0.615, 0.670, 0.589, 0.533]** -- stable, no spatial confound |

#### Result: NEGATIVE

12 bootstraps, production's exact recipe, frozen split:

| arm | AUC | mean delta | 95% CI | positive | >= MDE |
|---|---|---|---|---|---|
| base (33) | 0.9339 | -- | -- | -- | -- |
| +ratio | 0.9335 | **-0.0004** | [-0.0017, +0.0010] | 5/12 | **0/12** |
| +ratio,match | 0.9331 | -0.0009 | [-0.0025, +0.0014] | 3/12 | 0/12 |

2-min-only subset: -0.0006 and -0.0011. Brier 0.0763 -> 0.0762, ECE flat.

**Why the 0.6028 AUC does not translate: `|rho| = 0.667 with `period` itself.**
`P_ls` is bounded to the 0.2-13 d Lomb-Scargle search window, so the ratio is
dominated by `1/P_transit` -- it is substantially the deployed `period` column
restated. **This is the second time in two days that the highest single-feature
AUC in a batch turned out to be a deployed feature in disguise** (`trap_rmse`,
AUC 0.6507, `|rho|` 0.962 with `depth_mean`, delta -0.0011).

### Verdict

| sub-proposal | verdict |
|---|---|
| strongest-sinusoid amplitude | **EXACT DUPLICATE of deployed `var_ls_amp`.** Nothing built. |
| rotation/candidate period ratio | **NOT subsumed -- genuinely tested. DON'T PROMOTE:** -0.0004, CI [-0.0017,+0.0010], 0/12 at MDE. |

**Production stays at 0.9402 / 33 features.**

**For any future proposal in the rotation/periodicity space**, the space is now
mapped end to end: `var_ls_period`, `var_ls_amp`, `var_ls_power` are DEPLOYED;
`ls_period_match` (harmonic-aware distance) was tested and failed at -0.0006;
the RAW ratio was tested and failed at -0.0004; the n=1-only distance is
measurably the weakest of the three formulations (AUC 0.5596). A new proposal
here must explain what it measures that is not the dominant OOT periodicity,
its amplitude, its power, or its relationship to the transit period -- and why
it would not simply restate `period`.

**A methodological note worth keeping.** I expected to close sub-proposal (2)
as a strict subset and was wrong: the many-to-one direction runs the opposite
way, and the raw ratio had the higher single-feature AUC. The deduplication
check has to compare the actual functional relationship, not the apparent
sophistication of the two formulas.

---

## THIRD DEEP-LEARNING PROPOSAL -- both literature claims FAIL verification; the small meta-learner was TESTED and is far worse

**Date: 2026-08-14. Production UNCHANGED: 0.9402 / 33 features / md5
`c37f9f4bdb252d52b8c1c5487dad9e6d`. Nothing promoted, no CNN built.** MDE
~0.0097, frozen test 1,098.

### PART 0 -- the two cited numbers, re-verified. NEITHER can be used as evidence.

**ExoNet (Ansdell et al. 2018, arXiv:1810.13434) -- the "~0.955 AUC" DOES NOT
EXIST.** Fetched the abstract and the ar5iv fulltext:

> **The paper does not report any AUC value at all.**
> Its metrics are **97.5% accuracy** and **98.0% average precision**, plus
> precision-recall curves, versus AstroNet's baseline (+1.7% accuracy,
> +2.5% average precision).

This is the **accuracy-vs-AUC confusion this project has already been bitten by
twice** in citation checking. The cited 0.955 matches nothing in the source.

Three further facts make it non-comparable even if the number had been real:

| | ExoNet | this project |
|---|---|---|
| population | **Kepler DR24 TCE table, `av_training_set` column** -- signals a mission pipeline ALREADY flagged | raw stars, no mission candidate list |
| task | vetting a pre-filtered candidate list | end-to-end detection **plus** classification |
| labelled volume | **3,600 planets + 12,137 FPs = 15,737**, split 80/10/10 | 5,494 rows |
| inputs | global + local views + **centroid time series** + DR25 stellar parameters | 33 tabular features |

So ExoNet is ~2.9x this project's labelled volume, on the **narrower vetting-only
task**, with centroid inputs -- the exact distinction this project's own
TOI-restricted-evaluation measurement already showed changes the picture. Its
number is not a target this project's setup can be measured against.

**PlanetNet-MMG -- STILL UNVERIFIABLE, second attempt.** Re-checked via OpenAlex
on DOI `10.1016/j.eswa.2026.132396`: real (Expert Systems with Applications,
vol. 324, art. 132396, 2026), but **closed access, NO abstract, NO
abstract_inverted_index, no full-text URL** ($3,220 APC listed). The ~0.973
figure cannot be sourced, exactly as the cross-mission investigation found.
**It does not inform this recommendation, and should not be cited again as
evidence without a fulltext.**

### PART 1 -- the SMALL proposal is genuinely distinct, and it was TESTED

It is a fair point that a dense net on the **already-computed** 33 features is
not the closed CNN question: no representation has to be learned from raw flux,
so the data-volume argument that closed the CNN does not transfer directly. And
stacking had only ever been tested on model OUTPUTS (Part C: HGB+GP+CNN; the
small-lift trio: HGB+RF+LR, +0.0039, CI [-0.0022, +0.0106]) -- never with a
dense network as the base classifier or as a meta-learner over the raw features.

So it was built and run: 12 bootstraps, production's exact recipe as baseline,
same calibration wrapper on every arm so the comparison isolates model family.

Parameter arithmetic, stated before running:

    MLP (16, 8)     689 params  ->  6.4 training rows per parameter
    MLP (64, 32)  4,289 params  ->  1.0 training rows per parameter

| arm | test AUC | mean delta | 95% CI | positive | train AUC | train-test gap |
|---|---|---|---|---|---|---|
| base (HGB, 33 feat) | **0.9339** | -- | -- | -- | 1.0000 | 0.0661 |
| mlp_small (16,8) | 0.8696 | **-0.0644** | [-0.0833, -0.0460] | **0/12** | 0.9225 | 0.0529 |
| mlp_med (64,32) | 0.8963 | **-0.0377** | [-0.0461, -0.0302] | **0/12** | 0.9873 | **0.0910** |
| stack_mlp (meta on HGB output + 33) | 0.9015 | **-0.0324** | [-0.0402, -0.0206] | **0/12** | 0.9760 | 0.0745 |

2-min subset: -0.0677 / -0.0338 / -0.0317. **Every CI lies entirely BELOW zero.**
These are not null results -- the networks are decisively worse, by 3-7x the MDE.

**Both failure modes are visible at once, which is the signature of a data-volume
wall rather than a tuning problem.** The larger net has the biggest train-test
gap (0.0910 at 1.0 rows/parameter) -- overfitting. The smaller net has a smaller
gap (0.0529) but the worst test AUC -- underfitting. There is no size between
them that escapes: the capacity needed to match the tree cannot be supported by
4,390 rows.

**The stacked arm is the most damning result and it was the strongest version of
the idea.** `stack_mlp` receives **HGB's own out-of-fold prediction as an input
feature**. It could match the baseline by passing that number through. It scores
**0.0324 WORSE**. A dense net at this data volume cannot even preserve
information it is handed, let alone add to it.

### PART 2 -- the CNN + late-fusion hybrid: NOT a genuine escape. NOT built.

Late fusion changes **how** a raw-flux embedding is consumed, not **whether it
can be learned**. The closed CNN's failure was the embedding itself: 0.68-0.70
learning a light-curve representation from ~5k examples. Attaching a fusion head
adds zero training data to that branch, so the bottleneck is untouched.

**And this investigation adds a NEW argument against it that did not exist
before.** The fusion head in such a hybrid is a dense network over
[CNN embedding + tabular features]. Measured above: a dense network over
[HGB output + the 33 tabular features] -- a strictly EASIER problem, with a
known-good signal handed to it -- **loses 0.0324 AUC**. So even granting a
perfect CNN embedding for free, the fusion layer itself is a liability at
n = 4,390. The hybrid would have to overcome both a branch that failed and a
combiner now measured to destroy information.

**Not built, per instruction, and no separate go-ahead is recommended.**

### Verdict

| sub-proposal | verdict |
|---|---|
| ExoNet ~0.955 AUC as justification | **claim does not exist** in the source (97.5% accuracy / 98.0% AP, no AUC), and the task is vetting-only on 2.9x the data |
| PlanetNet-MMG ~0.973 AUC | **still unverifiable** (closed access, no abstract). Cannot be used. |
| small dense net on the 33 features | **TESTED. -0.0644 / -0.0377, CIs entirely below zero. Don't promote.** |
| dense meta-learner on HGB output + 33 | **TESTED. -0.0324, CI entirely below zero. Don't promote.** |
| 1D CNN + late fusion hybrid | **NOT built.** Fusion does not address the embedding bottleneck, and the fusion head is now independently measured to lose 0.0324. |

**Production stays at 0.9402 / 33 features.**

**Before a fourth neural proposal:** the closed evidence is now (a) a full CNN on
raw flux at 0.68-0.70, gap since widened to ~0.25; (b) stacking on model outputs,
twice, no clear; (c) a dense net as the classifier on the existing features,
-0.0377 to -0.0644 with CIs below zero; (d) a dense meta-learner given the
tree's own output, -0.0324. A new proposal must explain which of these four it
is not, and must not rest on ExoNet's or PlanetNet-MMG's headline figures --
the first does not report AUC and the second cannot be read.

---

## TRANSFER LEARNING, THIRD PROPOSAL -- FULLY COVERED by two existing closures. No new work. And the cited paper says the OPPOSITE.

**Date: 2026-08-14. Production UNCHANGED: 0.9402 / 33 features / md5
`c37f9f4bdb252d52b8c1c5487dad9e6d`. Nothing built, nothing run beyond
re-verification.** This entry is a confirmation of prior closures, not a new
investigation.

### PART 0.1 -- warm-start is still discarded by the calibration wrapper. Re-verified empirically, not cited from memory.

```
warm-started base fitted: n_iter_ = 8
clone(base) is fitted?  False              <-- fitted state STRIPPED by clone()
calibrated sub-estimators: 5
any sub-estimator IS the warm-started base object?  False
sub-estimator n_iter_ values: [8, 8, 8, 8, 8]
base object n_iter_ after calibration: 8   (unchanged -> base never continued)

through production's exact Pipeline wrapper:
any sub-pipeline IS the fitted pipeline object?     False
inner HGB objects distinct from the original?       True
```

`CalibratedClassifierCV(cv=5)` clones the estimator and **refits from scratch on
every fold**. A warm-started base contributes nothing. The original finding
stands unchanged.

### PART 0.2 -- ExoMiner++ does NOT demonstrate transfer learning. It explicitly REJECTED it. Third correction of this premise.

The abstract confirms the mechanism is pooled training:

> "we leverage multi-source training by **combining** high-quality labeled data
> from the Kepler space telescope with TESS data"

And the fulltext contains the direct head-to-head, which is **stronger evidence
than the prior closure recorded** -- ExoMiner++ tried *precisely what this brief
proposes* and abandoned it:

> "**Initially, we experimented with various transfer learning approaches**
> (Ng 2016), such as **training on Kepler data and fine-tuning certain layers of
> the model using TESS data**. However, as TESS data grew in size and label
> quality, **a simpler approach of combining Kepler and TESS data to create a
> larger training set proved more effective**."

So the single paper cited to justify reopening transfer learning is **direct
evidence against it**: a team with vastly more data, a neural architecture where
layer-wise fine-tuning is natural and cheap, and a strong incentive to make it
work, tested pretrain-on-Kepler/fine-tune-on-TESS and **found pooling beat it**.

**This premise has now been corrected three times in this conversation.** It is
recorded here in quotable form so a fourth proposal can be checked in seconds.

### PART 0.3 -- Roman and PLATO have no data. Neither can inform any near-term action.

Checked live against the mission pages, not assumed:

| mission | status today (2026-08-14) | science data |
|---|---|---|
| **Roman** (NASA) | **not launched**; scheduled 2026-08-30, Falcon Heavy from KSC LC-39A; in integrated operations | **none** |
| **PLATO** (ESA) | **not launched**; planned March 2027, Ariane 6; assembly complete Oct 2025, in final testing | **none** |

Roman launches in ~16 days and PLATO in ~19 months. Neither has produced a
single light curve. **They cannot be a basis for action now regardless of
anything else**, and a future proposal citing them should be dated accordingly.

### PART 0.4 -- does anything survive? No.

| proposal element | already closed by |
|---|---|
| CNN pretrain-on-Kepler, fine-tune-on-TESS | the CNN closure (0.68-0.70, four-deep evidence base, data-volume wall at every size) **and** ExoMiner++'s own rejection of exactly this |
| tree warm-start "under a different scheme" | the domain-adaptation closure, re-verified above |
| domain-adversarial adaptation | structurally inapplicable -- a fitted tree has no learnable representation to adapt |
| "ExoMiner++ demonstrates transfer learning" | **factually wrong**; the paper says the opposite, verbatim |
| Roman / PLATO leverage | no data exists |

### PART 1 -- the "different scheme" question, answered without an experiment

Mechanically, yes: dropping `CalibratedClassifierCV`, or using `cv="prefit"`,
would preserve warm-started state. **That is not the binding constraint, and
running the experiment would not address the two that are.**

**(a) It trades away something measured and load-bearing.** The sigmoid/cv=5
calibration was validated across an extensive sweep and carries the deployed
model's Brier 0.0763 and ECE 0.0355. Any warm-start scheme built on removing it
must show the warm-start gain **exceeds the calibration loss**, not merely that
warm-start "works" -- and no warm-start gain of any size has ever been measured
here.

**(b) The data problem is SCHEME-INDEPENDENT, which is the decisive point.**
Warm-starting on *what*? Kepler is domain-separable from TESS at ~0.97, K2 at
**0.9973**. Warm-started trees import Kepler's domain-specific splits as
**"frozen bias the new trees must fit around"** -- the original finding's exact
language. **No deployment mechanism repairs a domain mismatch that lives in the
pretrained structure itself.** Changing the wrapper changes whether the bias
survives; it cannot change whether the bias is wrong.

And the quantitative ceiling still binds independently: a **full** Kepler pull
projects **+0.0061 AUC, below the 0.0097 MDE**, under same-distribution
assumptions Kepler demonstrably does not meet. A warm-start scheme cannot beat
the ceiling of simply *having all that data*, which is itself under the
detection threshold.

**Verdict: no worthwhile experiment exists here.** The domain-separability
result makes every warm-start scheme moot regardless of implementation.

### Recommendation

**FULL CLOSURE. No new work. Production stays at 0.9402 / 33 features.**

This is stated plainly rather than partially: there is no salvageable fragment,
no scoped pilot worth defining, and no "revisit when X" condition other than the
ones already on record -- >3,000 usable cross-mission rows past the 36.4% Kepler
yield wall, or a larger test set that lowers the MDE. Roman/PLATO data would be
new information, but the earliest possible date for any is well after Roman's
2026-08-30 launch plus commissioning.

**Pattern flag for future proposals.** Three distinct proposals in this
conversation have now cited ExoMiner++ as evidence for transfer learning. The
paper states the opposite in one sentence, quoted above. Check that quote first.

---

## ARMSTRONG-STYLE ENSEMBLE (RF / EXTRA TREES / LDA) + THE CLUSTER-1 QUESTION -- both NEGATIVE

**Date: 2026-08-14. Production UNCHANGED: 0.9402 / 33 features / md5
`c37f9f4bdb252d52b8c1c5487dad9e6d`. Nothing promoted.** MDE ~0.0097.

### PART 0 -- what was genuinely untested

| member | status |
|---|---|
| Random Forest | tested ONLY in the original bake-off at **24 features**, pre-crowding/variability/Gaia. **STALE** -- retested fresh, per the CatBoost precedent where a family's edge halved between feature eras. |
| Extra Trees | **never mentioned anywhere** in this file. Untested. |
| LDA | **never mentioned anywhere.** Untested, and architecturally distinct from trees / calibrated trees / dense nets. |
| MLP | **already tested and strongly negative** (-0.0644 / -0.0377, CIs entirely below zero). **Deliberately excluded** from the ensemble -- including a component whose failure is already understood would bias the result down for a known reason rather than test a new question. |

### PART 1-2 -- fresh single models and ensembles at the 33-feature baseline

Production's exact calibration recipe on every arm. Trees get no scaler (they are
scale-invariant); **LDA gets a StandardScaler** so the comparison cannot be an
artefact of one model silently receiving unscaled heavy-tailed inputs. 12
bootstraps, frozen split.

| arm | AUC | mean delta | 95% CI | positive | >= MDE | Brier | ECE |
|---|---|---|---|---|---|---|---|
| **hgb (production)** | **0.9339** | -- | -- | -- | -- | 0.0763 | 0.0355 |
| rf | 0.9215 | **-0.0124** | [-0.0169, -0.0084] | 0/12 | 0/12 | 0.0837 | 0.0426 |
| et | 0.9126 | **-0.0213** | [-0.0256, -0.0160] | 0/12 | 0/12 | 0.0845 | 0.0325 |
| lda | 0.8331 | **-0.1008** | [-0.1044, -0.0982] | 0/12 | 0/12 | 0.1161 | 0.0218 |
| avg_rf_et_lda | 0.9052 | **-0.0288** | [-0.0338, -0.0238] | 0/12 | 0/12 | 0.0861 | 0.0292 |
| meta_lr | 0.9195 | **-0.0144** | [-0.0195, -0.0085] | 0/12 | 0/12 | 0.0878 | 0.0558 |

2-min subset: -0.0144 / -0.0218 / -0.1092 / -0.0319 / -0.0170. **Every arm is
negative with the CI entirely below zero. 0/12 positive everywhere.**

**Random Forest's staleness is now resolved and the answer is the same as in the
24-feature era: HGB still wins**, by 0.0124. The CatBoost pattern (edge shrinks
between feature eras) did not reverse anything here.

**The ensemble is worse than its own best member.** `avg_rf_et_lda` (0.9052)
sits below RF alone (0.9215) because averaging drags the two tree models toward
LDA, which is 0.10 AUC behind. The LR meta-learner recovers some of that
(0.9195, it can down-weight LDA) but still lands below plain RF and well below
HGB. This is the same mechanism the GBM-averaging entry recorded: **averaging
helps only when members are comparably strong, and here they are not.**

### PART 3 -- the cluster-1 question, answered on its own terms

**A methodological correction, recorded because it nearly produced a wrong
answer.** The first pass validated its re-derived SOM partition against
`som_cluster1_profile.json`'s **n = 532, 83.46% planet** and declared the
reproduction FAILED. That reference is over the **full labelled set** (5,486
rows). The weak subpopulation this question is about is cluster 1 **on the
frozen test**, recorded in `som_cluster_diagnostic.json` as **n = 108,
planet 76.85%, AUC 0.8280, ECE 0.0934**. Against the correct target the
reproduction **succeeded**:

    re-derived cluster 0:  n = 109,  planet 77.06%,  HGB ECE 0.0895     MATCH

Validation was made **behavioural, not just compositional** -- the selected
cluster had to reproduce production's performance on it, not merely its size and
class mix.

**An incidental finding worth keeping: the Gaia deployment already improved this
subpopulation.** Production scores **AUC 0.8790** on cluster 1 today versus the
**0.8280** recorded in the 31-feature era. The weak spot is still the weakest
cluster, but it is materially less weak than when it was first identified, and
nothing in this investigation was aimed at it.

**Does model diversity help there? No.** 12 bootstraps, cluster-1 only (n=109):

| arm | cluster-1 AUC | delta vs HGB | 95% CI | better than HGB | cluster-1 ECE |
|---|---|---|---|---|---|
| **hgb** | **0.8660** | -- | -- | -- | 0.1175 |
| rf | 0.8439 | -0.0221 | [-0.0525, -0.0015] | 0/12 | 0.1356 |
| et | 0.8427 | -0.0233 | [-0.0515, +0.0081] | 1/12 | 0.1064 |
| lda | 0.7290 | -0.1370 | [-0.1648, -0.1041] | 0/12 | 0.0896 |
| avg_rf_et_lda | 0.8137 | -0.0523 | [-0.0809, -0.0206] | 0/12 | 0.0855 |
| meta_lr | 0.8468 | -0.0192 | [-0.0527, +0.0160] | 2/12 | 0.1526 |

**Nothing beats HGB on cluster 1.** The best alternative (`meta_lr`) is 0.0192
behind and wins on only 2 of 12 resamples.

**One honest nuance: several arms have BETTER cluster-1 calibration while being
worse at ranking.** `avg_rf_et_lda` reaches ECE 0.0855 and LDA 0.0896 against
HGB's 0.1175. That is the averaging effect pulling probabilities toward the base
rate -- it improves ECE precisely by being less confident, while ranking worse.
It is not a fix for the calibration problem: a model that discriminates less well
is not a better model for this subpopulation, and the aggregate cost is -0.0288
to -0.1008 AUC.

**So cluster-1's weakness is NOT an architecture-diversity problem.** Four model
families spanning bagged trees, extremely randomised trees, a linear
discriminant, boosted trees and (from the prior investigation) dense networks
all do the same or worse there. That is consistent with the information
available for those stars being the limit, rather than HGB specifically being
badly suited to them -- though it is not proof, and the SOM partition itself
remains an unvalidated clustering, as the original investigation noted.

### Verdict

| question | recommendation |
|---|---|
| Armstrong-style RF/ET/LDA ensemble, aggregate | **DON'T PROMOTE.** Every arm negative, every CI below zero, 0/12 positive. |
| RF at the current baseline (staleness resolved) | **DON'T PROMOTE.** -0.0124; HGB still wins as it did at 24 features. |
| Extra Trees, LDA (first tests) | **DON'T PROMOTE.** -0.0213 and -0.1008. |
| ensembling to fix cluster-1 | **NO.** Nothing beats HGB there; apparent ECE gains come from being less confident, not more accurate. |

**Production stays at 0.9402 / 33 features.** Armstrong et al.'s composition
works in their setting; it does not transfer here, and the cluster-1 calibration
problem remains unexplained after a third distinct line of attack -- now with
model architecture ruled out alongside RV-discovery label noise and elevated
candidate-pool false-positive risk.

## Temperature scaling -- NEGATIVE. The missing bias term is the whole story.

The calibration sweep and the prefit round together covered
`{sigmoid, isotonic, bag-only} x cv={3,5,10,20}` plus dedicated holdouts at
5/10/20%. **Temperature scaling was never tested**, and it is a genuinely
different calibrator rather than another point on that grid:

    Platt / sigmoid      p = expit(a*z + b)     TWO parameters
    temperature scaling  p = expit(z / T)       ONE parameter, NO bias

### The design, which makes this an isolation rather than a survey

`CalibratedClassifierCV(cv=k)` fits k base models on (k-1)/k of the data and
averages their calibrated outputs. The harness reproduces that structure exactly
and uses the same `StratifiedKFold(k, shuffle=False)` sklearn uses for an integer
`cv`, so **the k base models are identical between `sigmoid cv=k` and
`temp cv=k`** and only the logit-to-probability map differs. HGB's
`decision_function` supplies the raw log-odds temperature scaling is defined on.

### RESULT: no arm clears, and the cross-fit arms are worse where it counts

12 training bootstraps, frozen test, production recipe.

| arm | AUC | mean delta | 95% CI | positive | Brier | d Brier | ECE | d ECE | mean T |
|---|---|---|---|---|---|---|---|---|---|
| **sigmoid cv=5 [PRODUCTION]** | **0.9339** | -- | -- | -- | **0.0763** | -- | **0.0355** | -- | -- |
| sigmoid cv=10 | 0.9341 | +0.0001 | [-0.0021, +0.0029] | 6/12 | 0.0762 | -0.0001 | 0.0361 | +0.0006 | -- |
| temp cv=3 | 0.9332 | -0.0007 | [-0.0026, +0.0016] | 5/12 | 0.0770 | +0.0007 | 0.0398 | +0.0043 | 1.264 |
| temp cv=5 | 0.9339 | +0.0000 | [-0.0002, +0.0001] | 6/12 | 0.0772 | +0.0009 | 0.0411 | +0.0056 | 1.125 |
| temp cv=10 | 0.9340 | +0.0001 | [-0.0021, +0.0028] | 6/12 | 0.0775 | +0.0012 | 0.0440 | +0.0085 | 1.054 |
| temp prefit 20% | 0.9287 | -0.0052 | [-0.0098, -0.0017] | 0/12 | 0.0797 | +0.0034 | 0.0454 | +0.0099 | 1.131 |
| temp prefit 10% | 0.9289 | -0.0051 | [-0.0092, -0.0002] | 1/12 | 0.0794 | +0.0031 | 0.0439 | +0.0085 | 1.096 |
| temp prefit 5% | 0.9293 | -0.0047 | [-0.0088, -0.0019] | 0/12 | 0.0799 | +0.0036 | 0.0478 | +0.0123 | 1.013 |

**`temp cv=5` is AUC-identical to production to four decimals, CI
[-0.0002, +0.0001].** That is the tightest interval measured anywhere in this
project, and it is a harness validity check passing rather than a finding: the
two arms share base models and differ only by a monotone map, so AUC *cannot*
move except through the averaging of k different curves.

**Where a calibrator is actually judged, temperature loses: ECE +0.0056 and
Brier +0.0009 at matched folds.** The mechanism is the missing bias term. Every
fitted T is **above 1** (1.01-1.26), so HGB is overconfident and needs
flattening -- which a temperature can do. What it cannot do is SHIFT. On a 79%
positive training set the calibration curve needs an offset as well as a slope,
and Platt has both parameters where temperature has one. Trading the bias term
for variance-robustness is a bad trade at this sample size.

**The prediction stated before running was half right, and the right half does
not rescue it.** Temperature was predicted to survive small calibration slices
better than isotonic, because one parameter is cheaper to estimate than a step
function. It does: at a 220-row slice `temp prefit 5%` is **-0.0047** where
`isotonic prefit 5%` was **-0.0146**, a 3x smaller collapse on the same slice.
But every prefit arm still loses to production by -0.0047..-0.0052, so being the
most robust small-slice calibrator only means losing by less. The prefit family
stays dead for the reason already established: the expensive part is giving up
the 5-model averaging, not the calibrator.

**Verdict: DO NOT PROMOTE.** Nothing clears; the cross-fit arms are AUC-neutral
and calibration-negative, the prefit arms are negative on everything.

Script: `temperature_scaling_validate.py`; results
`temperature_scaling_validate.json`.

## Subpopulation-specific calibration on HIGH CROWDING -- NEGATIVE, and a random control closes the whole family

Arm C of the giant-star investigation fitted a per-group Platt sigmoid for
giants vs dwarfs and made calibration worse (giant ECE 0.0956 -> 0.1094, overall
-0.0020, 0/12 clearing). The stated mechanism was variance, not bias: a per-group
scaler sees ~900 giant rows where the global one sees 4,386. **That mechanism was
argued from arithmetic and never measured against a null.** This round tests the
same idea on the crowding axis and adds the control.

### Prediction, stated before running

High crowding is a comparably small slice of the same training set, so the
variance arithmetic is nearly identical and the arm should fail the same way.

| subpopulation | train rows | share | ~distinct under bootstrap | test rows |
|---|---|---|---|---|
| giants (`st_rad>=1.5`) | 1,002 | 22.8% | ~633 | 229 |
| high crowding (`crowd_flux_ratio_max>=1.0`) | 920 | 21.0% | ~581 | 216 |
| high crowding (`crowd_flux_ratio_max>=0.5`) | 1,238 | 28.2% | ~782 | 290 |

### There is barely a defect to fix -- checked BEFORE fitting any arm

Deployed model, frozen test:

| group | n | planet % | AUC | ECE |
|---|---|---|---|---|
| high crowding >=1.0 | 216 | 85.65 | 0.9365 | **0.0421** |
| low crowding <1.0 | 882 | 77.32 | 0.9382 | **0.0322** |
| giants >=1.5 (reference) | 229 | 55.02 | 0.9219 | **0.0942** |
| dwarfs <1.5 (reference) | 869 | 85.27 | 0.9340 | **0.0254** |

Giants were a **3.7x** ECE gap -- a real defect specialisation could in principle
repair. High crowding is **1.3x**, with an AUC gap of -0.0017. The arm was dead
on arrival for a second, independent reason: there is nothing for a per-group
calibrator to buy.

### Spatial control: correlated, but NOT segregated by class

    corr(crowd_flux_ratio_max, |galactic b|)   -0.4947
    AUC of |gal b| alone, within high crowding  0.3787   (|0.5-a| = 0.121)
    AUC of |gal b| alone, within low crowding   0.3511   (|0.5-a| = 0.149)

Crowding is strongly correlated with galactic latitude, exactly as physics
requires. But unlike giants -- where position alone reached 0.7092 *inside* the
target group against 0.5482 outside it -- the spatial signal here is comparable
in both groups. Crowding is spatially correlated without being spatially
segregated by class, which is a cleaner situation than the giant axis. It simply
does not help.

### RESULT: all arms negative, and the RANDOM CONTROL matches them exactly

12 bootstraps, production recipe, one base model with per-group Platt from OOF.

| arm | AUC | mean delta | 95% CI | positive | ECE | crowd AUC | d crowd | crowd ECE | d crowd ECE |
|---|---|---|---|---|---|---|---|---|---|
| **base** | **0.9339** | -- | -- | -- | **0.0355** | **0.9430** | -- | **0.0487** | -- |
| C: crowd-stratified >=1.0 | 0.9303 | -0.0036 | [-0.0096, +0.0025] | 1/12 | 0.0416 | 0.9392 | -0.0039 | 0.0483 | -0.0003 |
| C-alt: crowd >=0.5 | 0.9304 | -0.0036 | [-0.0085, +0.0022] | 1/12 | 0.0407 | 0.9392 | -0.0039 | 0.0490 | +0.0004 |
| **CONTROL: random group** | 0.9303 | -0.0036 | [-0.0087, +0.0021] | 1/12 | 0.0409 | 0.9395 | -0.0035 | 0.0478 | -0.0009 |

**The control IS the finding.** A randomly chosen group of matched size, given
the identical stratified treatment, costs **-0.0036** -- indistinguishable from
the crowding split's -0.0036 to four decimals, at both thresholds. The damage
has nothing to do with which subpopulation was selected. **Splitting the
calibration set is itself the cost**, and the giant investigation's asserted
mechanism is now measured directly against a null.

One honest difference from the giant case: crowd ECE did NOT degrade (-0.0003)
the way giant ECE did (+0.0138). That is consistent rather than contradictory --
giants had a real defect a bad specialised calibrator could make worse, while
crowding has essentially none and the global calibrator was already doing fine
there. Crowding loses ranking without the compensating story, which is a cleaner
null.

**Verdict: DO NOT PROMOTE.** The prediction was stated before the test and the
test confirmed it.

**Subpopulation-specific calibration is now closed as a FAMILY, not per-axis.**
It has failed on an axis with a large calibration defect (giants), on an axis
with essentially none (crowding), and against a random control demonstrating
that the split alone explains the entire loss. Future proposals of this shape
should be routed here rather than re-tested, unless they come with a
substantially larger subpopulation than ~20% of the training set.

Script: `crowding_stratified_calibration.py`; results
`crowding_stratified_calibration.json`. Cross-references: the giant-star
investigation (arm C) and the calibration/ensembling sweep (`sigmoid cv=3`, the
first measurement of the same variance mechanism on the fold-count axis).

## Optuna Bayesian hyperparameter search -- the strongest sub-MDE result on record. NOT PROMOTED (SUPERSEDED -- see below)

> **STATUS UPDATE 2026-08-15: this configuration WAS subsequently deployed**, as
> an explicit user-authorised EXCEPTION to the promotion rule, not because it
> ever cleared it. The verdict table at the end of this section says "DON'T
> PROMOTE" and was correct under the standing rule at the time it was written.
> See `>>> DEPLOYED 2026-08-15: A DELIBERATE EXCEPTION TO THE PROMOTION RULE <<<`
> at the end of this document.

### Two things the deduplication check established first

**1. Every prior search here was `RandomizedSearchCV`, never Bayesian.**
`05b_model_analysis.py` (n_iter=30), `tabular_bakeoff.py` (12), `gbm_ensemble.py`,
`validate_multisector.py`, `retrain_with_new_features_full_suite.py` (15-30).
Nothing conditions on previous trials. The HGB grid was 5 discrete dimensions --
`{max_iter, max_depth, learning_rate, l2_regularization, max_leaf_nodes}`, 2,000
combinations sampled 30 times = **1.5% coverage** -- and never varied
`min_samples_leaf` or `class_weight` at all. "We did some random tuning" was
accurate; "we did a thorough Bayesian search" was not.

Nested CV, however, is NOT new: `05b_model_analysis.py:150` already ran outer-5
/ inner-3 with the search inside the inner loop. What changed is that the guard
matters far more at 120 TPE trials than at 30 random draws.

**2. PRODUCTION IS NOT RUNNING THE TUNED HYPERPARAMETERS. It has not since the
Gaia swap.**

| | pre-crowding | pre-variability | pre-Gaia | **deployed now** |
|---|---|---|---|---|
| `learning_rate` | 0.1 | 0.1 | 0.1 | 0.1 |
| `max_iter` | 500 | 500 | 500 | **100** |
| `max_leaf_nodes` | 63 | 63 | 63 | **31** |
| `l2_regularization` | 0.5 | 0.5 | 0.5 | **0.0** |
| `class_weight` | balanced | balanced | balanced | **None** |

`gaia_deploy_retrain.py` builds from `HistGradientBoostingClassifier(
random_state=42)` under a docstring reading *"Production's exact recipe,
unchanged"*. It was not: four tuned hyperparameters plus `class_weight` reverted
to sklearn defaults in that swap.

What this does NOT invalidate: the **+0.0142 Gaia feature delta** (both arms of
that comparison were at defaults, so the feature effect is clean), the deployed
**0.9402** (a real measurement of the artifact that exists), and the **retrain
gate** (`clone(prod_model)`, so it defends whatever is deployed -- self-
consistent). What it does mean is that `best_model_metadata.json`'s
`model_name` still says `"(tuned, ...)"`, which is now false, and that the tuned
configuration had never been measured at 33 features. It is measured below as
the `legacy` arm.

### Design and cost

Outer 5-fold / inner 3-fold, **120 TPE trials per outer fold**, median pruning
across inner folds. The inner objective scores the BARE pipeline (AUC is
invariant under each fold's sigmoid, and this is what the historical
`RandomizedSearchCV(pipe, ...)` calls did); the OUTER evaluation uses
production's full calibrated recipe, so the reported number is for the config as
it would actually deploy. Space: `learning_rate` log-uniform 0.01-0.3,
`max_iter` 50-500, `max_leaf_nodes` log 8-128, `max_depth` {None,2,3,4,6,8,12},
`min_samples_leaf` log 5-100, `l2_regularization` log 1e-4..10,
`class_weight` {None, balanced}.

**Wall clock, 8-core M-series, OMP_NUM_THREADS=2:** nested stage **36.3 min**
(5 outer folds in parallel), final full-training study **14.4 min**, total
**50.7 min** for the search. The 12-bootstrap validation cost a further **91
min**, because the selected config runs 475 iterations at 63 leaves against
production's 100 at 31 -- roughly **15x the fit cost**. That is a real
deployment consideration, not just a lab number.

### Nested CV

| config | nested-CV AUC | sd | delta vs prod | folds positive |
|---|---|---|---|---|
| production (defaults) | 0.9465 | 0.0073 | -- | -- |
| **Optuna-searched** | **0.9487** | 0.0091 | **+0.0022** | **4/5** |
| legacy tuned | 0.9464 | 0.0086 | -0.0001 | 1/5 |

### The per-fold winners are the most informative output of the whole search

| fold | lr | max_iter | leaves | depth | min_leaf | l2 | class_weight |
|---|---|---|---|---|---|---|---|
| 1 | 0.083 | 375 | 70 | None | 9 | 0.0023 | balanced |
| 2 | 0.164 | 250 | 48 | 12 | 12 | 0.0028 | None |
| 3 | 0.096 | 475 | 48 | 8 | 26 | 0.1169 | None |
| 4 | 0.073 | 375 | 42 | None | 5 | 0.0190 | None |
| 5 | 0.097 | 275 | 43 | 12 | 7 | 0.0012 | None |

**Five folds, five substantially different "best" configurations.**
`min_samples_leaf` spans 5-26, `l2` spans 0.0012-0.1169, `max_depth` flips
between None and 8/12, and `class_weight` disagrees with the final selection on
4 of 5. That is what a flat objective surface looks like: the search resolves
noise, not structure.

The final full-training study nonetheless landed close to the LEGACY config on
six of seven dimensions -- `lr` 0.0926 vs 0.1, `max_iter` 475 vs 500,
`max_leaf_nodes` **63 vs 63**, `max_depth` None vs None, `min_samples_leaf` 24 vs
20, `class_weight` balanced vs balanced -- differing mainly in `l2` (0.0090 vs
0.5). A 120-trial TPE search over a continuous space independently rediscovered
the region a 30-draw random search found years earlier.

### RESULT: 12 training bootstraps, frozen test

Harness validity check first: the single fit reproduces the deployed model at
**0.9402**, matching `best_model_metadata.json` to four decimals.

| arm | AUC | mean delta | sd | 95% CI | positive | >=MDE | Brier | ECE | 2-min delta |
|---|---|---|---|---|---|---|---|---|---|
| **prod [PRODUCTION]** | **0.9339** | -- | -- | -- | -- | -- | **0.0763** | **0.0355** | -- |
| **optuna** | **0.9385** | **+0.0045** | 0.0015 | **[+0.0024, +0.0070]** | **12/12** | **0/12** | **0.0739** | **0.0338** | +0.0037 |
| legacy tuned | 0.9355 | +0.0016 | 0.0023 | [-0.0025, +0.0052] | 9/12 | 0/12 | 0.0746 | 0.0344 | +0.0002 |

**The Optuna arm is positive on all 12 resamples, its CI excludes zero, and it
improves Brier AND ECE simultaneously. It still does not clear.** The bar is
`ci_lo > 0 AND mean delta >= MDE`, and at +0.0045 against an MDE of 0.0097 the
second leg fails -- no single resample even reached 0.0097 (0/12), the largest
being +0.0074.

**Why passing `ci_lo > 0` is not a contradiction of the MDE, which matters for
reading every table in this document.** These are two different intervals. The
CI here is over TRAINING bootstraps against a FIXED test set: it measures whether
the effect survives the training draw. The MDE was measured against TEST-SET
sampling error on 1,098 stars. Passing the first says the effect is stable;
it says nothing about the second. Both legs exist precisely because either alone
is insufficient.

**This is the strongest sub-MDE result this project has produced** -- stronger
than the CatBoost arms (+0.0080, 12/12 positive but clearing on only 4-6 of 12),
because it is 12/12 positive AND has a CI excluding zero AND improves
calibration rather than costing it. It is the same verdict for the same reason:
*"That is the correct outcome under the rule, and the rule should not be bent
because the result is finally interesting."*

### Search overfitting: no evidence of it

| arm | nested-CV delta | resampled-test delta | agree? |
|---|---|---|---|
| optuna | +0.0022 | +0.0045 | yes |
| legacy | -0.0001 | +0.0016 | sign flip on ~zero |

The resampled delta is LARGER than the nested-CV estimate, which is the opposite
of what search overfitting produces. The `legacy` sign flip is between -0.0001
and +0.0016 -- two numbers that are both zero to within their spread, so the
harness's automatic "DISAGREE" flag is over-reading noise there, not detecting
anything.

**A label in `optuna_hpo_validate.py` to distrust:** it prints inner-CV best
(0.9435) minus outer (0.9487) as "optimism from the search itself" and the sign
comes out backwards. That comparison is confounded -- inner fits see 2/3 of the
outer-train rows BARE, while the outer evaluation refits on all of it inside the
5-model calibrated wrapper, so more data plus bagging swamps the selection bias.
The valid measure of search overfitting is the nested-vs-resampled table above.

### Verdict

| sub-proposal | recommendation |
|---|---|
| Optuna Bayesian search | **DON'T PROMOTE -- but POSITIVE AND REAL.** +0.0045, 12/12 positive, CI [+0.0024, +0.0070], better Brier and ECE. Fails the MDE leg (0/12 >= 0.0097). |
| restoring the legacy tuned config | **DON'T PROMOTE.** +0.0016, CI spans zero, 9/12. The hyperparameters lost in the Gaia swap were worth approximately nothing. |

**Production stays at 0.9402 / 33 features / sklearn-default HGB.** What would
change the verdict is the same thing three other investigations converged on: a
larger test set, not a better configuration. At +0.0045 with this test set's
noise, certifying it needs roughly 4x the current 1,098 held-out stars.

**Open item, deliberately NOT actioned:** `best_model_metadata.json`'s
`model_name` field still describes the deployed model as `"(tuned, ...)"`. It is
a production artifact, so it is flagged rather than edited.

Scripts: `optuna_hpo_nested.py`, `optuna_hpo_validate.py`; results
`optuna_hpo_nested.json`, `optuna_hpo_validate.json`. Requires `optuna` (4.9.0
here), installed for this experiment and deliberately NOT added to
`requirements.txt` -- same treatment as CatBoost, LightGBM and XGBoost, which are
also experiment-only and absent from it. Nothing in the production path imports
it.

# >>> DEPLOYED 2026-08-15: A DELIBERATE EXCEPTION TO THE PROMOTION RULE <<<

## READ THIS BEFORE CITING THE OPTUNA RESULT

**The Optuna hyperparameter configuration was promoted to production on
2026-08-15 even though IT DID NOT CLEAR THE STANDING EVIDENTIARY BAR.** This is
the only promotion in this project's history that did not clear, and it must
never be summarised later as "a result that cleared the bar", because it did
not.

### What the bar is, and exactly how this failed it

    clearing requires:  ci_lo > 0  AND  mean delta >= MDE (0.0097)

| leg | required | measured | verdict |
|---|---|---|---|
| `ci_lo > 0` | > 0 | **+0.0024** | **PASSED** |
| `mean delta >= MDE` | >= 0.0097 | **+0.0045** | **FAILED** |

No single one of the 12 resamples reached the MDE either (**0/12**, largest
+0.0075 against a 0.0097 threshold).

### Why it was promoted anyway

Promoted on the user's explicit, informed instruction, with the shortfall stated
in advance. The supporting evidence, none of which is sufficient on its own:

- **12/12 resamples positive**, deltas +0.0024..+0.0075, never touching zero
- **95% CI [+0.0024, +0.0070] excludes zero**
- **Brier AND ECE improve simultaneously** (0.0763 -> 0.0739, 0.0355 -> 0.0338),
  so it is not buying ranking at the cost of probability quality
- **no evidence of search overfitting**: the nested-CV delta (+0.0022) is
  SMALLER than the resampled-test delta (+0.0045), the opposite of what an
  overfit search produces
- **host-disjoint nested CV agrees**: pooled out-of-fold 0.9471 -> 0.9503
  (+0.0032), winning **5/5** outer folds

### THE RULE HAS NOT CHANGED

**CatBoost (+0.0080, 12/12 positive, clearing on only 4-6 of 12) and
`trap_vshape` remain correctly NOT promoted, and every future near-miss stays
unpromoted by default.** The bar was not lowered, retired, or reinterpreted --
it was overridden once, deliberately, in writing, for this specific change.
Anything of this shape in future requires the same explicit sign-off; it does
not inherit this precedent automatically. If a later reader finds a sub-MDE
result cited as justification for skipping the bar, that reader should treat it
as an error.

**A caveat that belongs next to the decision:** the per-fold instability in the
search (five outer folds produced five substantially different "best" configs)
indicates a FLAT objective surface. The gain is real-but-small, not a robust
global optimum. A different seed would likely select a different configuration
of comparable performance.

## Deployment record

| | before | after |
|---|---|---|
| md5 | `c37f9f4bdb252d52b8c1c5487dad9e6d` | **`fe3fa82f36cc978396c68be07d6057f9`** |
| frozen-test AUC | 0.9402 | **0.9454** |
| features | 33 | 33 (**unchanged**) |
| `learning_rate` | 0.1 | 0.09258475971800786 |
| `max_iter` | 100 | 475 |
| `max_leaf_nodes` | 31 | 63 |
| `max_depth` | None | None |
| `min_samples_leaf` | 20 | 24 |
| `l2_regularization` | 0.0 | 0.009012660266897076 |
| `class_weight` | None | **balanced** |

Hyperparameters were read from `optuna_hpo_nested.json` at deploy time and
asserted against the live artifact after the swap -- never retyped.

**WHICH OF THE THREE CONFIGURATIONS IS DEPLOYED**, stated explicitly because
this project has already had one silent drift on exactly this point:

| config | status |
|---|---|
| sklearn defaults (lr 0.1 / 100 / 31 / l2 0.0 / cw None) | what the Gaia swap silently left deployed 2026-08-14 to 2026-08-15. **NO LONGER DEPLOYED.** |
| pre-Gaia "legacy tuned" (lr 0.1 / 500 / 63 / l2 0.5 / balanced) | tested at 33 features, +0.0016, CI spanning zero. **NEVER RE-DEPLOYED.** |
| **Optuna-tuned (lr 0.0926 / 475 / 63 / l2 0.0090 / balanced)** | **DEPLOYED NOW.** |

The metadata inaccuracy from the same investigation is fixed in the same swap:
`model_name` previously read `"(tuned, ...)"` while defaults were running. It
now reads `"(OPTUNA-TUNED hyperparameters, 33 features ...)"` and carries an
explicit `hyperparameters` block plus `hyperparameter_provenance`.

### Validation at deploy scale reproduced the investigation EXACTLY

The 12 bootstrap draws were replayed from the identical seed stream, so this is
an exact reproduction rather than a close one:

| quantity | investigation | deploy-scale rerun | |
|---|---|---|---|
| mean delta | +0.0045 | +0.0045 | MATCH |
| ci_lo | +0.0024 | +0.0024 | MATCH |
| ci_hi | +0.0070 | +0.0070 | MATCH |
| positive | 12/12 | 12/12 | MATCH |

Headline single fit: prod **0.9402** (reproducing the deployed metric of record
to four decimals, a harness validity check) -> optuna **0.9454**, delta +0.0052.
2-min-only subset delta +0.0037.

### Deployment mechanics, and why the gate could not do this itself

Manual offline swap, as with crowding/variability/Gaia -- but for a **different
reason**, worth recording because the usual one does not apply here. Feature
additions needed a manual swap because the gate raises `ValueError` across a
column-count change. **This change is feature-set-neutral, so it does NOT break
the gate at all.** It still cannot be done by the gate, because the gate's
challenger is `clone(prod_model)` -- production's OWN configuration refit on more
data. A gate built to answer "is the same model better with more data?" is
structurally incapable of proposing a DIFFERENT configuration. Hyperparameter
changes must come from outside it by construction.

Swap safety: staged artifact verified to load, predict, reproduce its recorded
AUC and carry the study's exact hyperparameters; outgoing model verified against
its expected md5; rollback copied and verified byte-identical BEFORE any write;
model and metadata replaced atomically via `os.replace`; live md5 and
hyperparameters re-asserted after. Rollback:
`models/versions/best_model_pre_optuna_c37f9f4b.joblib`.

### Downstream artifacts regenerated

| artifact | status |
|---|---|
| `models/conformal_calibration.json` | regenerated, `model_md5` = `fe3fa82f...` |
| `models/bootstrap_ensemble/` (32 members) | rebuilt 02:31 UTC, members carry the Optuna config, manifest size matches the new artifact |

### Operational cost, measured rather than estimated

- **Training ~9.7x**: 3.5 s -> 34.1 s for one full-training fit on a clean
  8-core machine. (An earlier ~15x figure was measured under CPU contention and
  was too high.)
- **A retrain tick costs ~30 s more, NOT 10x.** The gate's 2,000 bootstraps
  resample TEST PREDICTIONS, not refit models, so only the single challenger fit
  gets more expensive. A forced dry-run tick completed in **174 s**.
- **Bootstrap ensemble rebuild: 4.4 min** for 32 members -- the most expensive
  downstream job, and still cheap.
- **Inference 4.7x relatively, negligible absolutely**: 0.0659 s -> 0.307 s to
  score 1,098 rows = **0.28 ms/star**, so the full 254-candidate ranked pool
  costs ~0.07 s versus ~0.015 s. Serving latency is not a practical concern at
  this scale, but it is NOT unchanged, and the ratio would matter if the scored
  pool grew by orders of magnitude.

### Scheduler verification

`launchctl bootout` before touching artifacts (next retrain tick was ~18.8 h
away, so there was margin, but precedent was followed), `bootstrap` after.
Verified live: pid 18237 supervised, `/health` ok, thread alive, real loop tick
`{"retrain": "not_due", "tick": 1}`, and the web UI serving `OPTUNA-TUNED` /
0.9454.

**Future auto-retrains correctly inherit the new baseline** -- verified directly
rather than assumed: `_build_challenger(prod)` returns
`clone of production CalibratedClassifierCV` carrying lr 0.0926 / 475 / 63 /
msl 24 / l2 0.0090 / balanced, with **zero mismatches** against the saved study.
A forced dry-run retrain then ran the whole path end to end: challenger 0.9435
vs production 0.9435, delta +0.0000, CI [+0.0000, +0.0000], **not promoted**,
production md5 unchanged. Delta of exactly zero is the correct result when the
challenger IS production's recipe on the same data.

### A pre-existing bug found during this deployment, NOT caused by it

`conformal_prediction.py`'s final exchangeability section has been **crashing
since 2026-08-06** and nobody noticed, because it runs AFTER the deployment
artifact is written -- so the artifact succeeds and the script exits non-zero at
the very end. It feeds `results/unknown_candidates*/ranked_candidates.csv`
(37 and 34 columns) to `build_feature_matrix`, which now requires the 33
FEATURE_COLUMNS including the five `var_*` (added Aug 6) and two `gaia_*` (added
Aug 14) that those exports never gained. Evidence it is stale:
`conformal_prediction_results.json` is dated **Aug 4** and has not been rewritten
by any run since, through TWO prior deployments.

**`conformal_calibration.json` -- the artifact production actually uses -- is
current and correct.** Only the domain-shift diagnostic is dead. Flagged as
separate work rather than fixed inside a production deployment.

Scripts: `optuna_deploy_retrain.py` (staged retrain + validation),
`deploy_optuna_model.py` (checksum-verified atomic swap, `--dry-run` capable);
results `optuna_deploy_retrain.json`.

## GPC as an ensemble member + GPC-inclusive meta-calibrator -- NEGATIVE on every arm

### PREMISE CORRECTION: a real GPC was already built here

The proposal was framed as "the first time a real GP classifier has actually
been built in this project". **That is not accurate.** `gp_classifier.py` and
`gp_results.json` are on disk: sklearn `GaussianProcessClassifier` (Laplace),
kernel `ConstantKernel(1.0) * RBF(length_scale=1.0)`, median-impute +
StandardScaler, **test AUC 0.8673, fit time 267.9 s on 4,392 rows**, against a
then-baseline of 0.9032. Part C also already STACKED it (HGB + GP + CNN ->
0.9018 vs 0.9016 alone), and the bootstrap CI on GP-vs-classical was
**[0.017, 0.054], entirely above zero** -- the tree model won by a real margin,
not by noise.

Four things about that run were genuinely stale, which is why a re-test was
legitimate rather than a duplicate:

1. **24 features**, not the current 33 (no crowding, variability or Gaia)
2. a **random `train_test_split`, NOT the frozen host split** -- it predates
   that fix, so it may carry host leakage, which would have FLATTERED the GP
3. **never calibrated** -- no Brier or ECE on record, and calibration quality is
   this proposal's entire stated purpose
4. **single fit, never resampled**, against a baseline now 0.9454

So: partial overlap with real novelty. The re-test below is the new part.

### Part 0: measured cost, and a correction to my own extrapolation

Exact GPC scaling at 33 features, fitted in log space rather than assumed:

| n | fit s | test AUC | Brier | ECE |
|---|---|---|---|---|
| 400 | 0.3 | 0.7717 | 0.1674 | 0.0357 |
| 800 | 3.6 | 0.8691 | 0.1057 | 0.0396 |
| 1600 | 28.9 | 0.8820 | 0.0986 | 0.0258 |
| 2400 | 124.3 | 0.8977 | 0.0937 | 0.0243 |

    t = 8.25e-10 * n^3.30   (theory says n^3; hyperparameter optimisation adds the rest)

**That extrapolation was ~3.8x too pessimistic and the error is recorded
deliberately.** It predicted 14.8 min for a bare fit at 4,390 rows and 35.4 min
for a calibrated one, projecting ~7 h for the protocol. Measured at full scale:
**bare 3.9 min, calibrated 15.5 min.** The power law was fitted on small-n
points where fixed overhead dominates; the prior run's 267.9 s at 4,392 rows was
the accurate anchor all along and should have outweighed a fitted curve. The
lesson generalises: extrapolating a cost curve from cheap points over-predicts.

Actual wall clock for the 12-bootstrap protocol: **178 min** (4 workers; per-
bootstrap time rose from ~15 min serial to ~55 min under contention, which is
the OTHER direction the estimate can be wrong in).

**Kernel choice: no basis for Matern** (n=1600) -- RBF 0.8865, Matern nu=1.5
0.8869, Matern nu=2.5 0.8871, a 0.0006 spread, and RBF has the best ECE (0.0293
vs 0.037-0.038). RBF retained, matching the prior run.

**Approximation fallback, measured not assumed:** `Nystroem(1000)+LR` reaches
**0.8794 in 1.1 s**, roughly 200x cheaper than exact GPC for ~0.03 less AUC.
`RBFSampler(1000)+LR` collapses to **0.6470** -- data-dependent Nystroem works
here, data-independent random Fourier features do not. Nystroem is NOT a
Gaussian Process (no posterior, no predictive variance), so it is a fair stand-in
for "another kernel family" and not for "GP-style uncertainty".

### Part 1: fresh single-model GPC at 33 features

| model | AUC | 2-min | Brier | ECE | fit min |
|---|---|---|---|---|---|
| **hgb_prod (production)** | **0.9454** | 0.9396 | 0.0695 | 0.0241 | 1.9 |
| catboost | 0.9402 | 0.9328 | 0.0730 | 0.0270 | 0.4 |
| gpc_bare | 0.9100 | 0.9086 | 0.0895 | **0.0239** | 3.9 |
| gpc_cal | 0.9073 | 0.9044 | 0.0904 | 0.0295 | 15.5 |
| nystroem | 0.8749 | 0.8651 | 0.1008 | 0.0221 | 0.1 |

**The nine added features are worth ~+0.04 to the GP** (0.8673 -> 0.9100), so
the old number really was stale. It is still **-0.035 below production**.

**A genuine GP property confirmed, and it changes nothing.** Bare GPC's native
Laplace probabilities are already as well calibrated as production: **ECE 0.0239
vs 0.0241**. Wrapping GPC in `CalibratedClassifierCV` makes it WORSE on both
axes (AUC 0.9100 -> 0.9073, ECE 0.0239 -> 0.0295) at 4x the cost, with rho 0.993
between the two. The GP-calibration premise is real; there is simply nothing
left to win, because production is already there. **The validated arms therefore
use BARE GPC** -- using the wrapper "for comparability" would have handicapped
the arm under test.

### The diagnostic that decided this in advance

An averaging ensemble can only beat its best member if members err differently.

    Spearman rho, member probabilities on the frozen test
      hgb vs catboost   +0.950   <- near-duplicate
      hgb vs gpc        +0.814   <- genuinely more diverse
      hgb vs nystroem   +0.781

**GPC is meaningfully more diverse than CatBoost.** But diversity is necessary,
not sufficient -- and the sufficiency test fails outright:

    AUC of each member ON THE 109 STARS HGB GETS WRONG   (0.5 = independent info)
      hgb_prod   0.0000   (inverted by construction)
      catboost   0.0690
      gpc_cal    0.2173
      gpc_bare   0.2234
      nystroem   0.1891

**Every member is far BELOW 0.5**, meaning they do not merely fail to rescue
HGB's mistakes -- they rank those same stars in the same inverted direction. The
models are confidently wrong together. No weighting of them can recover what
none of them has.

### Part 3: full protocol, 12 bootstraps, vs production 0.9454

| arm | AUC | mean delta | sd | 95% CI | positive | >=MDE | Brier | ECE | 2-min delta |
|---|---|---|---|---|---|---|---|---|---|
| **hgb_prod [PRODUCTION]** | **0.9385** | -- | -- | -- | -- | -- | **0.0739** | 0.0338 | -- |
| gpc_only | 0.8949 | **-0.0436** | 0.0039 | [-0.0491, -0.0368] | 0/12 | 0/12 | 0.0981 | 0.0478 | -0.0426 |
| catboost_only | 0.9331 | -0.0053 | 0.0028 | [-0.0096, +0.0000] | 1/12 | 0/12 | 0.0794 | 0.0386 | -0.0060 |
| avg_hgb_cat_gpc | 0.9333 | -0.0052 | 0.0022 | [-0.0092, -0.0019] | 0/12 | 0/12 | 0.0741 | **0.0296** | -0.0052 |
| avg_hgb_gpc | 0.9311 | -0.0074 | 0.0020 | [-0.0114, -0.0048] | 0/12 | 0/12 | 0.0762 | 0.0376 | -0.0076 |
| meta_hgb_cat_gpc | 0.9368 | -0.0016 | 0.0018 | [-0.0039, +0.0020] | 1/12 | 0/12 | 0.0788 | 0.0497 | -0.0015 |
| meta_hgb_gpc | 0.9366 | -0.0019 | 0.0017 | [-0.0043, +0.0012] | 1/12 | 0/12 | 0.0771 | 0.0455 | -0.0018 |

**Nothing clears. Nothing is even positive.** Both averaging arms have CIs
entirely below zero. The two meta-calibrators are the least-bad arms and still
sit at -0.0016 / -0.0019, positive on 1 of 12 resamples.

**The meta-learner rediscovers Part C's answer.** Mean out-of-fold weights:

    3-member   hgb +5.29   catboost +1.93   gpc +2.24
    2-member   hgb +6.61   gpc +2.79

Part C recorded HGB 4.13 / GP 2.18 / CNN 2.33. Two independent fits, four years
of feature work apart, both conclude the meta-learner should mostly just use
HGB. Notably GPC out-weighs CatBoost (2.24 vs 1.93) despite scoring far worse
alone -- the diversity is real and the meta-learner does try to use it. It still
cannot make it pay.

**The same ECE illusion as the Armstrong ensemble round, and it is not a win.**
`avg_hgb_cat_gpc` posts the best ECE in the table (0.0296 vs production's
0.0338) while ranking 0.0052 worse. Averaging heterogeneous models pulls
probabilities toward the base rate, which improves ECE by being less confident.
A model that discriminates worse is not better calibrated in any useful sense --
the identical pattern was recorded for `avg_rf_et_lda` and should not be
mistaken for a result a second time.

### Was an LR meta-calibrator a different proposition from the dense net?

**Yes, and it deserved its own test rather than inheriting that verdict.** The
dense-net meta-learner lost **0.0324** given HGB's own output as an input,
because it had the capacity to overfit it. Logistic regression on 2-3 inputs has
3-4 parameters and structurally cannot. The measurement confirms the distinction
is real: LR loses **0.0016**, not 0.0324 -- a 20x smaller loss.

But it still loses. Capacity WAS the variable that differed, and removing the
overfitting reveals there was no signal underneath it to find. Both the
test-halves screen (200 splits, -0.0002, CI [-0.0060, +0.0046], 96/200) and the
proper out-of-fold protocol (12 bootstraps, -0.0016, 1/12) agree.

### Verdict

| sub-proposal | recommendation |
|---|---|
| GPC as a standalone model | **DON'T PROMOTE.** -0.0436, CI far below zero, 0/12. |
| HGB+CatBoost+GPC averaging | **DON'T PROMOTE.** -0.0052, CI entirely below zero, 0/12. |
| LR meta-calibrator with GPC | **DON'T PROMOTE.** -0.0016, 1/12 positive. The best of the alternatives and still negative. |
| Nystroem as a cheap GP stand-in | **DON'T PROMOTE.** 0.8749 alone; adding it costs -0.0104. |

**Production stays at 0.9454 / 33 features / Optuna-tuned HGB.** Nothing here is
a promote-with-sign-off candidate either: the Optuna exception was granted to an
arm that was **positive on 12/12 resamples with a CI excluding zero**. Every arm
here is NEGATIVE with CIs at or below zero, so the question of a deliberate
exception does not arise. **The Optuna precedent must not be read as lowering
the bar** -- it applies only to results that are positive, robust and merely
sub-threshold, and nothing in this investigation is any of those.

Scripts: `gpc_feasibility.py`, `gpc_screen.py`, `gpc_ensemble_validate.py`;
results `gpc_feasibility.json`, `gpc_screen.json`, `gpc_ensemble_validate.json`.
Cross-references: Part C (HGB+GP+CNN stacking), the small-lift stacking trio,
the dense-net meta-learner, the CatBoost seed-instability findings, and the
Armstrong RF/ET/LDA ensemble round (same ECE-from-underconfidence pattern).

## OOD/novelty proposals -- ALL THREE CLOSED. Plus a live crash bug in the deployed detector.

Three pieces (autoencoder reconstruction error, isolation-forest/density detection,
Mahalanobis distance / ensemble disagreement), assessed against the OOD detector
**already in production** and against the just-closed GPC finding.

### Part 0.1 -- the deployed detector, exact spec, AND IT IS BROKEN

`06_download_unknown.py` fits `IsolationForest(n_estimators=200,
contamination=0.02, random_state=42)`, thresholds at the 2nd percentile of its
own training scores, and a flag sets `in_distribution=False`, which
`split_and_rerank` uses to DROP the candidate from the human-review shortlist
(`keep = in_distribution & ~below_triage_floor`).

| property | value |
|---|---|
| features | **24** (model now uses 33) |
| missing | `crowd_flux_ratio_max`, `crowd_nearest_arcsec`, the five `var_*`, `gaia_ruwe`, `gaia_nss` |
| contamination target | 0.02 |
| threshold score | -0.534061 |
| measured train flag rate | 2.0033% |
| fit on | 5,491 rows, 2026-07-11 (training.csv now has 5,494) |

**It is not merely stale -- the call path RAISES.** `feature_columns` comes from
`best_model_metadata.json` (line 2171), which is now 33 columns, while
`load_or_compute_multivariate_detector` returns the CACHED 24-feature bundle. The
next line calls `detector["imputer"].transform(X[33 cols])`. Reproduced:

    ValueError: The feature names should match those that were passed during fit.

**This is the same failure mode as the 31-vs-24 FEATURE_COLUMNS gate bug** -- a
cached artifact versus a grown feature list -- and it is live in
`06_download_unknown.py`, which `job_runner.py:222` runs as a subprocess for the
Update job.

**Currently DORMANT, not firing:** `scheduler_config.enabled = 0` and `/health`
reports `"update": "disabled"`, so no automatic run reaches it. It would fire on
a manual Update press or on re-enabling the Update scheduler. Consistent with
the stale outputs: `ranked_candidates.csv` is 2026-08-05 and
`ranked_candidates_in_distribution.csv` is 2026-08-01, both predating the
variability (Aug 6) and Gaia (Aug 14) deployments.

### Part 0.2 -- THE DECISIVE FINDING: the detector already sees cluster 1, and calls it NORMAL

The proposal's motivation was that "cluster 1 suggests an unusual regime". That
had never been checked. Using the behaviourally-validated SOM partition
(re-derived cluster matched the recorded profile: n=109, planet 77.06% vs the
recorded n=108 / 76.85%):

| pool | detector | cluster-1 flag rate | rest-of-pool flag rate | odds ratio | p (Fisher) |
|---|---|---|---|---|---|
| main (n=488) | deployed 24-feat | **8.54%** (n=164) | **25.93%** (n=324) | **0.27** | **2.45e-06** |
| main | fresh 33-feat | **8.54%** | 25.62% | 0.27 | 3.84e-06 |
| widesector (n=69) | deployed 24-feat | 21.43% (n=14) | 36.36% (n=55) | 0.48 | 0.356 |
| widesector | fresh 33-feat | 14.29% | 36.36% | 0.29 | 0.198 |

**Cluster-1 candidates are flagged at roughly ONE THIRD the rate of everything
else, at p ~ 2e-6.** The detector is not blind to cluster 1; it actively judges
cluster-1 candidates to be MORE in-distribution than the pool average.

**This refutes the proposal's premise by measurement.** Cluster 1's problem is
not feature-space novelty -- in density terms it is more typical than average. A
better OOD detector would flag it LESS, not more. Upgrading to 33 features does
not change this (8.54% either way), so the staleness is not what is hiding a
cluster-1 signal.

**This narrows the four-times-unexplained cluster-1 question**, which is the most
valuable output here. Ruled out so far: RV-discovery label noise, elevated
candidate-pool false-positive risk, model-architecture diversity, and now
**feature-space novelty**. Cluster 1 is a region where the model is badly
calibrated while the data look perfectly ordinary.

### Part 0.4 -- Mahalanobis vs IsolationForest: same idea, no stated mechanism

Measured on the identical 33-feature training space:

| method | Spearman rho vs IF score | top-2% flagged overlap (Jaccard) |
|---|---|---|
| Mahalanobis (empirical covariance) | **+0.839** | 0.346 (56/109 shared) |
| Mahalanobis (robust MCD) | **+0.857** | 0.166 (31/109 shared) |

Both exceed this project's 0.80 redundancy threshold as SCORES. The honest
nuance: the top-2% flagged SETS overlap only weakly, so the two methods disagree
about which points are most extreme -- that disagreement is real. But there is no
ground truth of "should have been flagged", so nothing here shows either
ordering is better, and the proposal supplies no mechanism for why Mahalanobis
would win. If anything the assumption runs the wrong way: Mahalanobis presumes a
single ellipsoidal (Gaussian-ish) bulk, while these 33 features are heavy-tailed
and multi-modal; IsolationForest assumes no distributional shape at all.
**Redundant re-implementation. Don't build.**

### Part 0.5 -- ensemble disagreement: the GPC closure does NOT transfer, but it still fails

Thinking this through rather than assuming, as asked: the GPC finding was about
CLASSIFICATION errors (members are confidently wrong on the SAME stars).
"Disagreement signals unreliability" is a different claim and deserved its own
test. The right target is not planet-vs-not; it is **"is this prediction
wrong?"**, and the right baseline is not 0.5 -- it is HGB's OWN confidence,
which costs nothing.

Frozen test, target = HGB wrong at 0.5 (109/1,098 stars):

| signal | AUC for predicting HGB's error |
|---|---|
| **HGB own confidence \|p-0.5\| (free baseline)** | **0.8706** |
| disagreement sd(HGB, CatBoost, GPC) | 0.7624 |
| disagreement \|HGB - CatBoost\| | 0.7383 |
| disagreement \|HGB - GPC\| | 0.7105 |
| deployed OOD score | 0.4533 |

**Disagreement is genuinely informative** -- 0.76 is far above chance, so the GPC
closure does NOT transfer automatically and it was right not to assume it. It is
simply **dominated by a free baseline**. The incremental test settles it
(5-fold out-of-fold logistic regression):

    conf alone            0.8686
    conf + disagreement   0.8717     delta +0.0031

**+0.0031 AUC**, bought at the price of fitting a GPC (~4 min) and a CatBoost on
every scoring run. Not worth building.

**A separate result worth keeping: OOD-ness and model unreliability are close to
orthogonal.** The deployed OOD score predicts HGB's errors at **AUC 0.4533** --
no better than chance, marginally inverted. The OOD flag is doing a different
job (is this candidate weird?) from reliability estimation (is this prediction
wrong?), and should not be reinterpreted as the latter.

### Part 0.6 -- autoencoder: closed, and precisely why

The VAE pilot is already closed with a **pre-registered kill criterion that
fired**: reconstruction error correlated **+0.830** with `var_oot_rms`, and
transit-vs-no-transit AUC was 0.6806 against a 0.75 threshold. The mechanism was
predicted before building: reconstruction error tracks whatever carries the most
variance, which in a TESS light curve is stellar variability, not an 84-5,000
ppm transit.

Checking the framing precisely, as asked -- an "autoencoder for OOD flagging"
differs from the closed VAE only in what it is fed:

- **on light curves** -> identical to the closed pilot. Closed.
- **on the 33 tabular features** -> a reconstruction-based density model on
  exactly the space IsolationForest already occupies, i.e. it collapses into the
  Part 0.4 redundancy question, with the added handicap that an autoencoder on
  4,390 rows x 33 columns is a far heavier way to estimate density than an
  isolation forest.

Either way, closed. No third framing survives.

### Part 2 -- availability and spatial control

| pool | n (Success) | 33/33 columns present | min per-column coverage |
|---|---|---|---|
| main | 488 | yes | 52.05% (`st_rad`) |
| widesector | 69 | yes | 78.26% (`st_rad`) |

**|galactic b| control on the deployed OOD flag: clean.** AUC of |gal b| for
predicting the flag is **0.5443**, rho **-0.061** -- no meaningful spatial
exposure, unlike the crowding and giant-star axes.

**A bug in this analysis, caught and fixed rather than shipped:** the first run
joined the candidate list on a bare integer `tic_id` against the feature table's
`host` ("TIC_231620255"), producing an all-NaN merge that silently
median-imputed `st_rad`/`st_teff` -- the exact "computed somewhere, dropped
before use" pattern this project has hit before. Fixed, an assertion added so
coverage below 50% now aborts, and every number above is post-fix. The cluster-1
result was unchanged by the fix (8.54% vs 25.93% after, 8.54% vs 24.07% before),
which is why it is trustworthy rather than merely unrefuted.

### Verdicts

| piece | verdict |
|---|---|
| autoencoder reconstruction error | **CLOSED.** Duplicate of the killed VAE pilot on light curves; collapses into IsolationForest redundancy on features. |
| isolation forest / density detection | **ALREADY DEPLOYED.** Not a gap. |
| Mahalanobis distance | **CLOSED as redundant.** rho +0.84/+0.86 vs the deployed score, no mechanism offered, and its distributional assumption suits this data less well. |
| ensemble disagreement | **CLOSED, on its own merits.** Real signal (0.76) but dominated by a free baseline; +0.0031 incremental. |
| cluster-1 motivation | **REFUTED.** The deployed detector already flags cluster 1 at 1/3 the pool rate; cluster 1 is more in-distribution, not less. |

**Nothing to build. Production stays at 0.9454 / 33 features.**

### The one real item: SCOPED, NOT BUILT, needs explicit go-ahead

The 24-feature detector is stale and its call path crashes. That is a genuine
maintenance gap, entirely independent of this proposal, and it is NOT fixed here
because it touches the production candidate pipeline.

Scope if approved: refit `IsolationForest` on the current 33 features and 5,494
rows, re-derive the 2nd-percentile threshold, overwrite
`multivariate_ood_detector.joblib` + `multivariate_ood_meta.json`, and add a
guard so a cached bundle whose `feature_columns` disagree with the caller's is
recomputed rather than returned (the missing piece that turned staleness into a
crash). Cost: seconds to fit. Expected behavioural impact, already measured
above: **essentially none on the cluster-1 question** (8.54% either way) and a
pool-wide flag rate of 25.62% vs 25.93% -- so this is a correctness and
crash-safety fix, not a performance change.

Script: `ood_proposal_assess.py`; results `ood_proposal_assess.json`.
Cross-references: the multi-sector pools-only OOD-impact measurement, the VAE
anomaly-detection kill, the GPC/ensemble-diversity investigation, and the
four-part cluster-1 thread.

## OOD detector refit + staleness guard -- CLOSED. Two stale artifacts, one crash, one silent degradation.

Fixes the crash found by the OOD/novelty investigation above. **Correctness and
crash-safety only -- no model, training-data or threshold-policy change.**
Production untouched at 0.9454 / 33 features / md5
`fe3fa82f36cc978396c68be07d6057f9` (asserted by md5 before and after).

### What was broken -- and it was two things, not one

Both OOD checks cache an artifact derived from `FEATURE_COLUMNS`, and both rotted
when the feature set grew 24 -> 26 -> 31 -> 33. Only one of them announced it.

| artifact | built | features | failure mode |
|---|---|---|---|
| `multivariate_ood_detector.joblib` | 2026-07-11, 5,491 rows | 24 | **CRASH** -- `imputer.transform(X[33])` raises `ValueError: The feature names should match those that were passed during fit` |
| `training_feature_ranges.json` | 2026-07-11, 5,491 rows | 24 | **SILENT** -- `flag_out_of_distribution` does `if feat not in ranges: continue`, so the 9 newer features were simply never range-checked |

**The second one is the more instructive failure.** The investigation found the
crash; the silent one only surfaced when the same call site was examined for the
same bug class. `in_distribution = in_distribution_univariate &
~multivariate_ood_flag`, so fixing only the crashing half would have left the
deployed flag quietly degraded.

**Third instance of one bug class in this project** -- a cached artifact versus a
grown feature list, after `NON_TLS_FEATURE_COLUMNS` (which blocked all scoring
for a week) and the pre-Gaia hyperparameter drift.

### The durable fix, which is the point

Refitting alone resets the clock until the next feature promotion. The guard
lives in the LOAD PATH, so it fires automatically for every future one:

```python
def _check_cached_feature_set(cached_features, current_features, artifact_name):
    """Returns None if the cache covers the current feature set, else a reason."""
```

Wired into three places:

1. `load_or_compute_multivariate_detector` -- mismatch prints
   `STALE OOD DETECTOR -- REFITTING` and recomputes instead of returning an
   incompatible bundle.
2. `load_or_compute_feature_ranges` -- same, and it matters more here because
   the old behaviour degraded silently.
3. `flag_multivariate_ood` -- defence in depth for any caller that builds a
   detector dict itself; raises a message naming the fix rather than sklearn's
   opaque feature-name error.

Both `compute_*` functions now return/record `feature_columns`, and
`multivariate_ood_meta.json` gained `n_features` + `feature_columns` so
staleness is visible from the JSON without loading the joblib.

### Refit result

| | before | after |
|---|---|---|
| features | 24 | **33** |
| training rows | 5,491 | **5,494** |
| threshold (2nd pct, re-derived) | -0.534061 | **-0.508247** |
| measured training flag rate | 2.0033% | **2.0022%** (target 2.0%) |
| ranges features | 24 | **33** |

Old artifacts preserved at `models/versions/*_pre33_24feat.*`.

### Behavioural change: minimal, as predicted, on BOTH checks

| pool | n | multivariate before | after | delta | univariate before | after | delta |
|---|---|---|---|---|---|---|---|
| main | 488 | 20.08% | 19.88% | **-0.20%** | 9.63% | 9.84% | **+0.20%** |
| widesector | 69 | 33.33% | 31.88% | **-1.45%** | 68.12% | 68.12% | **+0.00%** |

The three new-feature univariate violations in the main pool are single stars
(`var_excess` 1, `var_ls_power` 1, `gaia_nss` 1).

**One number that looks alarming and is not caused by this fix:** the widesector
pool's 68.12% univariate flag rate. Measured against the OLD 24-feature ranges it
is also 68.12% -- delta exactly 0.00%. It is a pre-existing property of the
wide-sector-window pool, which by construction contains more extreme candidates,
and is unrelated to this change. Flagged so it is not later misread as damage
from the refit.

### Cluster-1 regression check: the qualitative finding holds

Re-measured against the REFIT detector, with the SOM partition re-derived and
behaviourally validated (n=109, planet 77.06%, AUC 0.8871):

| pool | cluster-1 flagged | rest flagged | odds ratio | p |
|---|---|---|---|---|
| main | **8.54%** (n=164) | **25.62%** (n=324) | 0.27 | **3.84e-06** |
| widesector | 14.29% (n=14) | 36.36% (n=55) | 0.29 | 0.198 |

Identical to the investigation's pre-refit numbers. **Cluster 1 is still flagged
at a third the rate of the rest**, so the conclusion that feature-space novelty
is not what makes cluster 1 unreliable survives the refit -- it was never a
coverage artifact of the stale detector.

### Validation

- Refit detector loads and scores **both pools** with no errors (488 + 69 rows).
- **Guard tested by simulation**, not by inspection: the 24-feature bundle and
  ranges were temporarily restored to reproduce the exact pre-fix state. Both
  loaders detected the mismatch and recomputed; the direct-call path raised the
  named error. Refit artifacts restored afterwards.
- Only the OOD flagging step is touched. Classification is unaffected --
  `score_candidates` runs before OOD flagging and neither loader is on its path.
- **Production model md5 and training.csv md5 asserted unchanged** across the
  whole run. Promotion gate and scheduler untouched; the scheduler was not even
  stopped, because nothing it reads was modified.

Scripts: `ood_detector_refit.py`; results `ood_detector_refit.json`. Closes the
scoped item raised by the OOD/novelty-detection investigation above.

## VESPA as an independent FPP cross-check -- CLOSED AT THE PART 0 FEASIBILITY GATE

Nothing was built. **Production untouched at 0.9454 / 33 features / md5
`fe3fa82f36cc978396c68be07d6057f9`.** The decision rests on a measured install
attempt and on VESPA's own source, not on argument.

### Two premises in the brief could NOT be verified from this file

Recorded because this project's standard is that cited prior findings are
checked, not assumed:

1. The brief attributes to "the RAVEN comparison" a claim that VESPA is *"the
   dominant non-ML validation method historically... most currently-known
   validated planets used vespa specifically"*. **RESULTS_SUMMARY.md contains
   zero occurrences of "vespa".** The only mentions anywhere in the repo are two
   passing docstring references (`stellar_density_checks.py:24`,
   `stellar_density_fetch.py:27`) noting that stellar density is "the core of how
   vespa ... reason[s]". The claim may well be true of the field -- it is simply
   not a finding this project ever recorded, and is not treated as one here.
2. The cited RV 2.7% / imaging 0.7% coverage figures are likewise **not in this
   file**. They appear to come from the Phase 2/3 accessibility investigations
   whose numbers were never written up here. Rather than cite them, the imaging
   dependency below is established **directly from VESPA's own source**, which is
   stronger evidence anyway.

### Part 0.1 -- VESPA installs, and then does not run

`vespa` 0.6 is real and on PyPI (versions 0.0 through 0.6). Tested in an
**isolated venv** specifically so a dependency cascade could not damage the
project environment -- which turned out to matter.

A plain `pip install vespa` fails immediately:

    NameError: name 'cythonize' is not defined     (in its own setup.py)

With `Cython` pre-installed and `--no-build-isolation`, **it does build and
install**, pulling `isochrones`, `simpledist`, `plotutils`, `batman`, `emcee`,
`corner`. So "uninstallable" would be the wrong verdict. The wall is at import:

| step | failure |
|---|---|
| `import vespa` | `ModuleNotFoundError: numba` -- installable |
| then | `AttributeError: np.recfromtxt was removed in the NumPy 2.0 release` -> needs `numpy<2` |
| `numpy<2` | `astropy 8.0.1 requires numpy>=2.0` -> needs `astropy<7` |
| `astropy 6.1.7` + `numpy 1.26` | `AttributeError: module 'numpy' has no attribute 'float'` -- raised from **`vespa/_transitutils.pyx` line 8**, a COMPILED Cython extension, so not source-patchable without a rebuild. `np.float` was removed in numpy 1.24 -> needs `numpy<1.24` |
| `numpy 1.23.5` | `matplotlib 3.11 requires numpy>=1.25`; `scipy 1.17 requires >=1.26.4`; `pandas 3.0.5 requires >=1.26` |
| full legacy stack attempt (`numpy 1.23.5`, `scipy 1.9.3`, `pandas 1.5.3`, `matplotlib 3.6.3`, `astropy 5.2.2`, `tables 3.8.0`) | `tables` build fails: `ModuleNotFoundError: pkg_resources` (removed from modern setuptools); pandas C extension left broken |

**VESPA requires an entire ~2022-era scientific stack that is mutually
incompatible with this project's** (numpy 2.4.6, scipy 1.17.1, pandas 2.3.3).
It is not a version pin or two; it is a second environment, and the blocking
`np.float` call sits inside a compiled extension.

### Part 0.2 -- per-candidate cost COULD NOT BE MEASURED, and that is the honest answer

The brief asked for measured per-candidate wall-clock. **No number is reported,
because none could be obtained** -- `FPPCalculation` never imported, so no
candidate was ever scored. Reporting an estimate here would be inventing the
one figure the brief specifically asked to measure. Two further requirements,
both confirmed from the installed source, would gate cost even with a working
import:

* **MultiNest.** `isochrones/starmodel.py:27` logs *"PyMultiNest not imported.
  MultiNest fits will not work."* `pymultinest` itself pip-installs, but it is a
  wrapper: the actual sampler is a **FORTRAN library** (`libmultinest`) needing a
  separate cmake + gfortran + LAPACK build. The stellar-model fit underpinning
  every FPP scenario runs through it.
* **MIST isochrone grids**, downloaded from
  `waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/...` (multi-GB, per
  `isochrones/mist/models.py:119-123`).

### Part 0.4 -- BLOCKED BY AN ALREADY-CLOSED CONSTRAINT, verified from VESPA's source

This is the finding that would matter even if every packaging problem above were
solved. From `vespa/fpp.py`:

    line 111:  maxrad = 10        # exclusion radius [arcsec]
    line 306:  maxrad = float(config['constraints']['maxrad']); fpp.set_maxrad(maxrad)
    line 334:  cc = ContrastCurveFromFile(ccfile, band, name=name)

**`maxrad` and `ContrastCurveFromFile` are first-class inputs, and both are
high-resolution-imaging products.** `maxrad` is the radius inside which a
blended eclipsing binary could still be hiding; a contrast curve is what
converts an imaging non-detection into that constraint. They are the dominant
lever on the BEB/HEB scenario priors -- which is precisely what makes VESPA a
*validation* tool rather than a classifier.

Without them VESPA silently falls back to `maxrad = 10 arcsec`, i.e. "assume a
blend could hide anywhere within 10 arcsec". For TESS at ~21 arcsec/pixel that
assumption is close to worthless, and an FPP computed under it is not the
quantity the literature validates planets with. **This is the same
follow-up-observation-tier input this project already closed as unavailable** --
the identical constraint, reached independently, and it is structural rather
than a packaging inconvenience.

### Parts 1 and 2 -- not run, and why

**Part 1 (selective veto pilot) could not run**: no working import, so no FPP for
any candidate. Had it run, the cost analysis would still have forced the
selective top-N framing the brief anticipated -- but that is now moot.

**Part 2 (FPP as a trainable feature) is explicitly skipped**, as the brief
instructed if Part 0 failed. Two independent reasons, either sufficient:
computing FPP for a meaningful fraction of 5,494 training stars is impossible
when it cannot be computed for one; and the prior is poor regardless. **Eight
prior derived-astrophysical-statistic features** -- `trap_vshape`, secondary
eclipse depth and significance, odd-even significance, stellar density,
depth/duration ratio, transit shape ratio, trapezoid residual -- have all shown
the same pattern: real, physically-correct, low-correlation signal that the tree
model does not convert into performance. `trap_vshape` produced the largest
single-feature separation ever measured here (AUC 0.3595, |AUC-0.5| = 0.1405)
and still moved the model by nothing. A VESPA FPP would be the ninth instance of
a now well-established rule, not a surprising failure.

### Recommendation: DO NOT INTEGRATE. Not worth it, and blocked besides.

| framing | verdict |
|---|---|
| VESPA as a selective advisory/veto tool on top-N candidates | **NOT FEASIBLE.** Requires a second, mutually-incompatible Python environment plus a FORTRAN sampler plus multi-GB grids -- and would still need imaging contrast curves this project does not have. |
| VESPA FPP as a trainable feature | **SKIPPED** per the brief; infeasible, and the 8-for-8 derived-feature prior argues against it independently. |
| the underlying *idea* -- an independent physical FP cross-check | **Not closed by this.** What is closed is VESPA specifically, on packaging and on inputs. |

**What would change this:** not effort, but data. VESPA becomes meaningful for a
candidate only when that candidate has a high-resolution imaging contrast curve,
which in practice means it has already entered the TFOP follow-up process. At
that point the community runs VESPA-class validation itself, and the existing
ExoFOP/TFOP evidence layer already surfaces that status. **The tool is most
useful exactly where this project's own contribution has already ended.**

No UI work was done and none is proposed; any evidence-layer change would need
separate sign-off regardless, and there is nothing here to surface.

Environment note: the install attempt lives in a throwaway venv under the
session scratchpad, deliberately isolated. **The project environment was
verified intact afterwards** (numpy 2.4.6, scipy 1.17.1, pandas 2.3.3) -- the
isolation held, which given the cascade above was not a formality.

Cross-references: the RAVEN-style synthetic-FP closure (also closed at a Part 0
feasibility gate, on the missing PASTIS population-synthesis dependency -- the
same shape of finding), the CETRA and PLD closures, and the derived-feature
pattern above.

# SYSTEM INTEGRITY AUDIT -- 2026-08-15

Full-stack verification after four deployed model changes, ~20 investigated
proposals and three infrastructure fixes. Not an experiment: the question was
whether everything is currently consistent and whether a genuinely new candidate
can complete the pipeline TODAY. **It can -- verified end to end on a fresh
star.** Five defects were found; two were safe to fix and were fixed, three
require sign-off and were reported, not touched.

## 1. Production state -- ACCURATE, after two metadata corrections

| property | value |
|---|---|
| md5 (artifact == metadata) | `fe3fa82f36cc978396c68be07d6057f9` |
| frozen-test AUC | 0.9454 |
| features | 33 |
| recipe | `CalibratedClassifierCV(Pipeline([SimpleImputer(median), HGB]), cv=5, sigmoid)` |
| hyperparameters | lr 0.09258, iter 475, leaves 63, depth None, msl 24, l2 0.009013, cw balanced, seed 42 |

Matches the Optuna deployment exactly, read from the live artifact.

**Feature constants are mutually consistent (1.3, clean):** `FEATURE_COLUMNS`
(33) is byte-identical in content AND ORDER to `best_model_metadata.json`; all 11
`NON_TLS_FEATURE_COLUMNS` and all 4 `OPTIONAL_FEATURES` are present in it; no
orphans in either direction. 22 TLS-derived / 11 non-TLS.

**TWO STALE METADATA FIELDS FOUND AND FIXED** (both write-only -- `class_balance`
is written by `05_train_models` and never read for decisions, `promoted_at` is
read nowhere, so correcting them changes no behaviour):

| field | was | now |
|---|---|---|
| `promoted_at` | `2026-08-06 19:30:16 UTC` -- the **VARIABILITY** deployment | `2026-08-15 02:17:14 UTC` (live artifact mtime) |
| `class_balance` | `{4336, 1155}` (sum 5,491) | `{4341, 1153}` (sum 5,494) |

`promoted_at` survived THREE deployments untouched -- each deploy script updated
metrics but never this field. Same drift class as the `"(tuned, ...)"` label.
Both corrections carry a `_note` recording the old value and why.

**Noted, not changed:** `training_rows` (4,390) + `test_rows` (1,098) = 5,488,
not 5,494, because `test_rows` records the FROZEN mask while `split_by_host`
yields 1,104. Consistent with every prior version's convention and the AUC is
measured on the frozen mask, so this is definitional, not an error.

## 2. Downstream artifacts -- ALL CURRENT

| artifact | status |
|---|---|
| `conformal_calibration.json` | `model_md5` = `fe3fa82f...` **CURRENT** |
| `bootstrap_ensemble/` | 32 members, 33 features, built 02:31 UTC, size-matches the artifact, **members carry the Optuna config** |
| `multivariate_ood_detector.joblib` | 33 features, 5,494 rows |
| `training_feature_ranges.json` | 33 features, 5,494 rows |

Systematic search (`joblib.load`/`joblib.dump` and cached JSON across `code/`
and `web/`, excluding experiments) found **no additional feature-dependent
cached artifact** beyond those four. Three staged leftovers exist
(`best_model_variability_staged`, `staged_best_model_gaia33`,
`staged_best_model_optuna33`) with **zero references in production code** --
clutter, not a correctness risk.

**STILL BROKEN: the conformal exchangeability diagnostic.** The spawned fix never
landed on main -- `conformal_prediction.py`'s last commit is still `1112d41e`,
the section still feeds `ranked_candidates.csv` (37/34 columns) to
`build_feature_matrix` (33 required), and `conformal_prediction_results.json` is
**still dated 2026-08-04**. The deployment artifact it writes BEFORE that section
is current and correct, so only the domain-shift diagnostic is dead.

## 3. END-TO-END LIVE TEST -- FULL PASS on a genuinely fresh star

`TIC_149121385`, checked against all 8,735 hosts known to the project (training
+ both feature tables + both candidate lists) and absent from every one.

| step | result |
|---|---|
| 1 fresh-star selection | PASS -- never seen |
| 2 download (`try_search`/`download_one_star`) | PASS -- `tic_id_spoc`, 74 products, **18,274 cadences**, 3 s |
| 3 preprocess (`clean_light_curve`) | PASS -- Success, **18,271 cleaned cadences** |
| 4 TLS (`compute_all_features`) | PASS -- Success, 23 fields, **54 s** |
| 5a crowding (`add_crowding_features`) | PASS -- resolved 1/1, `crowd_flux_ratio_max` 0.000606, `crowd_nearest_arcsec` 9.016 |
| 5b variability (`add_variability_features`) | PASS -- 1/1 from RAW, `var_oot_rms` 0.000520, `var_excess` 1.361 |
| 5c Gaia (`add_gaia_astrometry_features`) | PASS -- matched 1/1 in DR3, `gaia_ruwe` 1.063, `gaia_nss` 0 |
| 6 all 33 features | **32/33 populated, 0 required NaN, 0 absent**; the single NaN is `FAP`, which is in `OPTIONAL_FEATURES` |
| 7 model / OOD / conformal | probability **0.9935**; `multivariate_ood_flag` False (score -0.3721); `in_distribution` True; conformal available, returned a real `{Planet}` set at 90% |

**All three post-TLS feature groups fired correctly for a star that had never
been through them.** Every function called was imported unmodified from the
production modules.

The harness needed three iterations to find the right production entry points
(`clean_light_curve`, not `process_one_file`) -- **those were defects in the
audit script, not the pipeline**, and are recorded as such so the pass is not
overstated.

**Test residue removed**: the raw and processed light curves this test created
were deleted; the star was never added to any pool table or to training.csv.

### 3.5 UI -- renders cleanly, but 9 model features are INVISIBLE to reviewers

`/candidates/447400458` returns **HTTP 200, 23.8 KB, zero** Traceback / Internal
Server Error / `None<` / `nan<` markers. Conformal (14 refs), in-distribution,
ExoFOP (26), TFOP (5), centroid (8) all render.

**But `grep` across `web/templates/` finds ZERO references to any of:**
`crowd_flux_ratio_max`, `crowd_nearest_arcsec`, the five `var_*`, `gaia_ruwe`,
`gaia_nss`.

The model uses all nine; the human reviewer cannot see any of them. Three of the
four deployed model changes are invisible in the evidence layer. Not a
correctness bug -- scoring is unaffected -- but it is a real reviewer-facing gap.
**A UI change needs separate sign-off, so it is reported, not built.**

## 4. Scheduler and operations -- HEALTHY

- **launchd supervision verified by PID/PPID, not by `/health`**: pid 18237,
  **PPID = 1**, owns 127.0.0.1:5050, up 4h55m, `launchctl` status 0. Genuinely
  supervised -- this is the exact check that previously exposed an orphaned
  `nohup` answering `/health`.
- **Label-append path is COMPLETE**: `retrain_pipeline` calls
  `_crowding_for_host` (322), `_variability_for_raw` (328) and `_gaia_for_host`
  (335) after `compute_all_features` -- all 33 features for any newly labelled
  star.
- **Promotion gate re-verified NOW**: `_build_challenger` returns
  `clone of production CalibratedClassifierCV` with **zero mismatches** against
  the deployed hyperparameters.
- **Process sweep CLEAN**: no zombies, no stray `nohup`, no leftover pool
  workers from any investigation in this conversation.

## 5. Data integrity -- CLEAN, with one real gap

**Frozen split re-verified fresh:** 5,494 rows, **5,494 unique hosts (0
duplicates)**, train 4,390 / test 1,104, **0 straddling hosts**, frozen-test
1,098 a proper subset of split-test, train+test == all rows.

**Disk: 62 GiB free** (16% used). No repeat of the fork-bomb incident.

**GAP FOUND -- 15 training rows have computable but uncomputed variability.**
17 rows carry NaN across all five `var_*`. Diagnosed rather than assumed:

* all 17 have a **complete ephemeris** (period, T0, duration)
* **15 of 17 have a raw light curve on disk** -- so `var_*` is computable for
  them and simply was never computed
* only 2 (`TIC_200385493`, `TIC_453789494`) genuinely lack raw data

They sit in a contiguous block (indices 5469-5485), i.e. rows added after the
variability backfill ran. Impact is small -- 0.27% of training rows, and the
median imputer absorbs them -- but it is a genuine silent gap.

**NOT FIXED: this writes to `training.csv`**, which every prior modification in
this project has treated with explicit ceremony (backup, byte-exact prefix
verification, integrity re-check). It also changes the data the model trains on.
Reported for sign-off rather than actioned unilaterally.

Gaia/crowding gaps by contrast are **not** backfill failures: `gaia_nss` is 99.29%
and `gaia_ruwe` 97.62%, with the difference being genuine DR3 non-matches and
RUWE legitimately undefined for some sources.

## 6. CONSOLIDATED OPEN ITEMS -- complete list, current status

| # | item | status |
|---|---|---|
| 1 | **Conformal exchangeability diagnostic** feeds 37/34-column exports to a 33-feature builder; crashes after writing the artifact; results JSON stale since Aug 4 | **STILL OPEN** -- spawned fix never landed |
| 2 | **15 training rows missing computable `var_*`** | **OPEN, newly found here.** Needs sign-off (writes training.csv) |
| 3 | **9 model features invisible in the UI** (crowding, variability, Gaia astrometry) | **OPEN, newly found here.** Needs sign-off (UI change) |
| 4 | `db.mark_watch_label_processed` doesn't clear `error_message`, so recovered rows read `processed` while carrying stale failure text | **STILL OPEN** -- one-line fix, flagged during label recovery, never actioned |
| 5 | Project lives under `~/Downloads`, whose TCC protection caused the launchd outage | **OPEN by choice** -- structural move offered and not taken; currently working |
| 6 | SAP-flux fallback pilot | **SCOPED, NOT BUILT** -- measured, recommended as a pilot, needs go-ahead |
| 7 | True re-download fallback for corrupt cached light curves | **OPEN** -- deliberately not built |
| 8 | Three staged model leftovers in `models/` | **COSMETIC** -- zero production references |
| 9 | `promoted_at` / `class_balance` metadata drift | **CLOSED HERE** |
| 10 | OOD detector + ranges stale at 24 features | **CLOSED** (2026-08-15 refit + load-path guard) |
| 11 | `NON_TLS_FEATURE_COLUMNS` gate | **CLOSED** |
| 12 | VESPA integration | **CLOSED** -- infeasible on packaging and on imaging inputs |
| 13 | Optuna promotion as a deliberate MDE exception | **CLOSED**, documented as an exception |

Items 1-7 are genuinely outstanding. **None of them breaks the live pipeline**,
which Part 3 proved end to end.

## Verdict

**The system is consistent and the live pipeline works.** Production state,
every downstream artifact, the scheduler, the promotion gate and the frozen
split all verify clean. A brand-new candidate completed the entire journey today
with 0 required-feature gaps. Two metadata defects were corrected; three real
gaps (conformal diagnostic, 15 variability rows, UI evidence layer) are reported
with sign-off requested rather than patched unilaterally.

Scripts: `e2e_fresh_star_audit.py`; results `e2e_fresh_star_audit.json`.

## Conformal exchangeability diagnostic -- FIXED AND LANDED. The work existed; it was never committed.

Closes open item #1 from the 2026-08-15 system audit. **Production untouched:
model md5 `fe3fa82f...`, training.csv md5 `3bf4a343...`, and
`models/conformal_calibration.json` verified BYTE-IDENTICAL across the run.**

### What actually happened -- recovered, not re-implemented

The fix was **not lost and not missing**. It was found intact as **uncommitted
working-tree changes** in `.claude/worktrees/quizzical-aryabhata-fa87b2`, on
branch `claude/quizzical-aryabhata-fa87b2`.

The branch head was `9997a2ce` -- an ordinary earlier main commit. **Nothing was
ever committed on that branch.** `git diff main..branch` showed the branch
*behind* main, and the real work sat only in the worktree's dirty state: three
modified files, of which `conformal_prediction.py` (+182/-12) was the fix.

The base blob of `conformal_prediction.py` was **identical** in main and in the
worktree's HEAD, so copying the worktree's working copy applied the fix and
nothing else -- verified by the resulting diff being exactly +182/-12.

**Only that one file was taken.** The worktree's `RESULTS_SUMMARY.md` is 1,023
lines BEHIND main (it predates the Optuna deployment, the OOD refit, the GPC and
VESPA closures and the audit); merging that branch as-is would silently revert
all of it. Its stale `conformal_prediction_results.json` was also discarded, since
the run regenerates it.

### A premise worth correcting

The task brief said to read the fix description from RESULTS_SUMMARY.md.
**RESULTS_SUMMARY.md contains zero occurrences of `load_unknown_candidates`,
"paired, not pooled" or "unsourceable"** -- that description was never written
here. It exists only in the recovered source's own docstrings, which is where it
was read from. The brief's description was accurate; its stated location was not.

### What the recovered fix does

`load_unknown_candidates()` joins the 7 columns missing from the stale ranked
exports (five `var_*`, two `gaia_*`) out of each pool's OWN feature table:

* **paired, not pooled** -- `CANDIDATE_POOLS` pairs each export with the feature
  table it was scored from, because a host in both pools has different TLS
  features in each (different sector baseline)
* **raises rather than truncates** -- `FileNotFoundError` if the table is absent,
  `KeyError` naming unsourceable columns, `ValueError` on duplicate hosts, and
  `validate="many_to_one"` on the merge
* **a `_matched` sentinel**, not a non-null test, so a host that legitimately
  matches a row with NaN `gaia_*` (no Gaia source within 3 arcsec) counts as
  joined rather than as a data problem
* **`except (Exception, SystemExit)`** -- deliberately not a bare
  `except Exception`, because `build_feature_matrix` signals schema mismatch
  with `SystemExit`, which derives from `BaseException`; catching only
  `Exception` would reproduce the exact silent failure being fixed
* **results written BEFORE and AFTER** the optional section, so the file can
  never again be older than the artifact beside it, and a skip is recorded as
  `skipped: true` rather than leaving a stale verdict looking current

It also correctly declines to "fix" the export writer:
`score_candidates` builds the ranked frame FROM `unknown_features*.csv` and hard
-fails on a missing feature column, so a run today would export all 33 columns.
The CSVs are merely stale (2026-08-05); refreshing them needs a full MAST+TLS
run that would overwrite the live candidate tables. Making an offline diagnostic
depend on that is the wrong dependency.

### Clean run against current production

Exit code **0**.

| pool | ranked rows | present | joined | matched |
|---|---|---|---|---|
| `unknown_candidates` | 254 | 26/33 | 7 | **254/254** |
| `unknown_candidates_widesector` | 54 | 26/33 | 7 | **54/54** |

`var_*` 100% non-null in both; `gaia_*` 96.1% (main) and 100% (widesector).

**The diagnostic produced a real verdict that has been invisible since
2026-08-06:**

    domain classifier (calibration test set vs unknown candidates) AUC = 0.9798
    VERDICT: NOT exchangeable

Largest standardized mean differences: `FAP` -1.34, `var_ls_power` +0.60,
`st_rad` +0.55. So the finite-sample coverage guarantee is **valid for stars like
the frozen test set and NOT valid as stated for unknown candidates**, where it is
a well-calibrated heuristic. For scale, `synthetic_vs_real` scored 0.9654 and was
abandoned on that basis; this is higher.

Validation: `conformal_prediction_results.json` regenerated at 2026-08-15 03:53
with `exchangeability.skipped = false`, `domain_auc 0.9798`, `n_unknown 308` and
per-pool provenance; `conformal_calibration.json` carries the production md5 and
is **byte-identical** to its pre-run state, confirming the calibration itself is
deterministic and untouched. Exactly two files changed:
`conformal_prediction.py` and its results JSON.

### PROCESS LESSON: work in a worktree is not landed until it is merged

This is the second instance of the same failure mode in this project, and the
more dangerous one because it is invisible:

1. The **feature-set staleness** class (`NON_TLS_FEATURE_COLUMNS`, the 24-feature
   OOD detector, the pre-Gaia hyperparameter drift) -- a cached artifact drifting
   from a grown definition.
2. **This one: work that was done, validated and then never committed.** No
   error, no failing test, no stale artifact -- just a file on main that was
   never touched. It survived undetected until a full-system audit diffed the
   file against its own history.

What made it detectable was checking the ARTIFACT's date
(`conformal_prediction_results.json`, still 2026-08-04) rather than trusting that
a reported fix had landed. **A fix is real when `git log` on main shows it, not
when a task reports success.** Cheapest general guard: for any delegated or
worktree-isolated fix, verify with `git log --oneline -- <path>` on main before
recording it as closed.

**REMAINING HAZARD, flagged and NOT actioned:** the worktree and branch
`claude/quizzical-aryabhata-fa87b2` still exist, and its `RESULTS_SUMMARY.md` is
**1,023 lines behind main**. Merging that branch as-is would revert the Optuna
deployment record, the OOD refit, the GPC/VESPA closures and the audit. It is now
fully superseded -- the only work it held is landed here. Deleting the branch and
worktree is the right cleanup but is destructive, so it needs explicit go-ahead:

```bash
git worktree remove .claude/worktrees/quizzical-aryabhata-fa87b2
git branch -D claude/quizzical-aryabhata-fa87b2
```

Script: `conformal_prediction.py`; results `conformal_prediction_results.json`.

## var_* gap backfill -- 3 rows filled, 14 correctly NaN. The audit's diagnosis was WRONG IN BOTH DIRECTIONS.

Closes open item #2 from the 2026-08-15 system audit. **Production model
untouched (`fe3fa82f...`); frozen-test AUC bit-identical at 0.9454155994 before
and after.**

### The audit said 15 computable / 2 impossible. The truth is 3 computable / 14 not -- and they are DIFFERENT rows.

Re-checking rather than trusting the audit is what surfaced this, and it went
wrong twice in opposite directions:

**Error 1 -- the audit missed a raw directory.** It globbed `data/*lightcurve*`,
which does not match `data/retrain_pipeline/raw`. That directory is named as a
legitimate raw source by `best_model_metadata.json`'s own
`raw_lightcurve_dependency` field. So the 2 stars the audit called impossible
(`TIC_200385493`, `TIC_453789494`) **do** have raw light curves -- 15,687 and
104,053 cadences -- and are the two that computed cleanly.

**Error 2 -- file existence is not usability.** The audit checked only that a
file existed. Asking the production function for its `var_status` gives the real
answer:

    15  non-standard schema
     2  ok

The 15 in `data/known_lightcurves_negative` are **QLP products**
(`kspsap_flux`, ~1,200 cadences), not SPOC. `02_preprocess.validate_schema`
requires `{time, flux, flux_err, pdcsap_flux, pdcsap_flux_err, quality}`; these
have no `pdcsap_flux` at all, so `variability_for_raw` correctly returns
`non-standard schema`. **They are not a backfill gap. They are correctly NaN,
and the original variability backfill was right to skip them.**

### A third row WAS recoverable, from a filename defect

`Teegarden's_Star` exists on disk twice:

| file | rows | schema |
|---|---|---|
| `Teegarden's_Star.csv` (matches the host) | 16,276 | **missing `time`** -> rejected |
| `Teegardens_Star.csv` | 16,276 | valid |

Proven to be the same observation, not assumed: `flux`, `flux_err`,
`pdcsap_flux`, `quality` and `cadenceno` are **numerically identical across all
16,276 rows**. The apostrophe is the obvious cause of the truncated write. An
explicit, documented `RAW_ALIASES` entry reaches the intact file, and only after
the primary fails `validate_schema` -- so the alias can never mask a good file.

### The three rows filled

| host | source | `var_oot_rms` | `var_excess` | `var_ls_amp` | `var_ls_power` | `var_ls_period` |
|---|---|---|---|---|---|---|
| `Teegarden's_Star` | `known_lightcurves` (alias) | 0.001769 | 1.0592 | 0.000328 | 0.0522 | 1.82079 |
| `TIC_200385493` | `retrain_pipeline/raw` | 0.000321 | 1.3853 | 0.000139 | 0.1700 | 5.32098 |
| `TIC_453789494` | `retrain_pipeline/raw` | 0.003868 | 1.0638 | 0.000522 | 0.0556 | 0.91835 |

Computed with `web/retrain_pipeline._variability_for_raw` **unmodified** -- the
same function the label-append path uses -- not a reimplementation.

### Write discipline and verification

Backup `training_BACKUP_pre_var_gap_20260815_044223.csv`, md5-verified identical
to the pre-write file. **Textual column update, never `read_csv -> to_csv`** --
the discipline that exists because a prior crowding backfill silently altered 8
cells of `chi2red_min` in the 16th significant digit.

| check | result |
|---|---|
| lines changed | **exactly 3** (5470, 5485, 5486) |
| every pre-existing byte preserved | **PASS** -- no non-`var_*` column altered, no pre-existing value overwritten |
| row count / column count | 5,494 / 51 unchanged |
| non-var columns differing | **NONE** |
| host order identical | yes |
| duplicate hosts | 0 |
| straddling hosts | 0 |
| **frozen-test AUC** | **0.9454155994 -> 0.9454155994** (bit-identical) |
| rows still all-var-NaN | 17 -> **14**, as intended |

md5 `3bf4a343...` -> `16d77ded...`.

**One nuance reported rather than glossed:** the brief assumed all affected rows
were training-side. Two are not:

| host | train | split-test | FROZEN test |
|---|---|---|---|
| `Teegarden's_Star` | yes | no | no |
| `TIC_200385493` | no | yes | **no** |
| `TIC_453789494` | no | yes | **no** |

Both are post-manifest stars, allocated to `split_by_host`'s test partition by
the 50/50 post-freeze rule but excluded from the frozen mask (1,104 vs 1,098).
**The frozen test set is untouched and the AUC of record is unchanged.** This is
not leakage: `var_*` is computed from the star's own light curve with no label
and no cross-partition information, exactly as production computes it at serve
time. The only consequence is that a future retrain-gate evaluation sees two
test rows with real values instead of imputed ones.

### Not retrained, and no retrain recommended

Three rows out of 5,494 (0.05%), only one of them training-side. That is far
below anything this project treats as material -- the MDE is 0.0097 and the
learning curve predicts +0.013 AUC for *doubling* the dataset. **No retrain is
warranted and none was run.** Production stays at 0.9454 / 33 features.

### Follow-on items, NOT actioned

1. **`Teegarden's_Star.csv` is corrupt on disk** (missing `time`). The alias
   fixes training data, but the production path still resolves `host + '.csv'`
   and would fail for this star. Correct fix: re-download or rename, so the
   normal path works.
2. **14 QLP-schema negatives can never have `var_*`** under the current
   SPOC-only schema check. Options are re-downloading them as SPOC where
   available, or teaching `choose_flux_columns` about `kspsap_flux`. Both change
   feature values for existing rows and need their own validation cycle.

Script: `variability_gap_backfill.py` (`--apply` to write, dry run by default);
results `variability_gap_backfill.json`.

## Evidence layer: crowding / variability / Gaia now visible -- CLOSED

Closes open item #3 from the 2026-08-15 system audit. **Presentation only.**
Production model untouched (`fe3fa82f...`), training data untouched, promotion
gate and scheduler untouched.

### The gap

The audit found **zero** templates referencing `crowd_*`, `var_*`, `gaia_ruwe`
or `gaia_nss`. Nine of the model's thirty-three inputs -- three of the four
deployed model improvements -- drove the score at the top of every candidate
page while being invisible to the person reading it. A reviewer could not audit
the reasoning behind the number they were being shown.

### What was added

`web/model_features.py`, deliberately mirroring `exofop_vetting.py`: one cached
index over the per-pool feature tables, `lookup(tic_id, transit_period)`, never
raises, no model call and no network on render. Values are **read, never
recomputed** -- the same numbers the classifier was given.

A new `Contamination & blend checks` card sits ABOVE the TFOP panel, and the
attribution is the point of the ordering: this is **model input** derived by
this project, the panel below it is **external expert opinion**. Each group gets
a plain-language verdict plus the raw values in monospace for anyone who wants
them.

**The most useful thing on the panel is not a raw feature.** It is the derived
rotation/transit period ratio: if the star's strongest brightness cycle sits at
1x, 2x or 1/2x the transit period, that is the starspot-masquerading-as-transit
signature, and the panel says so in those words. 30 of 296 live candidates trip
it.

### The NaN-versus-zero distinction, which the brief had slightly wrong

The task described `crowd_nearest_arcsec == 0.0` as the genuine-zero case.
Measured, it is not: that column has **2 NaN and zero zeros**. The genuine-zero
lives in `crowd_flux_ratio_max` (17 candidates at exactly 0.0). The real
structure is three-way, and all three render distinctly:

| case | n | rendered as |
|---|---|---|
| `ratio > 0` | 279 | "Some neighbour flux in the aperture" / "Neighbour outshines the target" |
| `ratio 0.0`, distance present | 15 | "**Neighbour present, but contributes no flux** — a real measurement of zero, not an absent one" |
| `ratio 0.0`, distance NaN | 2 | "**No catalogued neighbour** — there is no distance to report because there is no neighbour; a clean result, not missing data" |

Gaia NaN (10 candidates) is a fourth, different absence and says so: *"No Gaia
source within 3 arcsec, so the astrometric checks could not run. Both values are
optional model inputs and the classifier imputes them, so the score is still
valid."*

### Live verification

Five real candidates, each chosen for a specific edge case, all **HTTP 200, zero
Traceback / Internal Server Error / jinja2 markers**:

| TIC | case | crowding | variability | Gaia |
|---|---|---|---|---|
| 447400458 | typical | caution | pass | pass |
| 314017939 | ratio 0.0 **with** a neighbour + **NaN Gaia** | pass | caution | **unknown** |
| 101949434 | **no neighbour at all** | pass | caution | pass |
| 345087856 | Gaia **NSS binary flag** | neutral | neutral | **caution** |
| 149594966 | RUWE > 1.4, NSS 0 | neutral | pass | caution |

Rendered text, verbatim from the live page:

    Nearby-star contamination — Neighbour present, but contributes no flux
      The nearest catalogued star sits 44.3 arcsec away and contributes no
      measurable flux to the aperture. A real measurement of zero, not an absent one.
      [crowd_flux_ratio_max 0.0000 · crowd_nearest_arcsec 44.33"]

    Nearby-star contamination — No catalogued neighbour
      No TIC neighbour was found inside the search radius at all ...
      [crowd_flux_ratio_max 0.0000 · crowd_nearest_arcsec none within search radius]

    Gaia DR3 companion check — Gaia flags this as a non-single star
      Gaia's own analysis classifies this source as an astrometric binary ...
      [gaia_ruwe 3.279 (threshold 1.4) · gaia_nss 1 (astrometric)]

**Legacy/coverage check:** all **296 of 296** live candidates resolve in the
feature tables, so every page renders the populated panel. The `available=False`
branch is a safety net for a future or legacy star and was verified separately
against a non-existent TIC.

Status distribution across all 296: crowding 17 pass / 239 neutral / 40 caution;
variability 196 / 70 / 30; Gaia 229 pass / 10 unknown / 57 caution.

### Two copy defects caught by reading the rendered output

Neither would have shown up in a pass/fail test:

* *"which is **the same period of** the 1.40 d transit signal"* -- the phrasing
  worked for "twice"/"half" and not for the 1:1 case. Now "**the same as**".
* *"as **a astrometric** binary"* -- article computed from the flag name.

Also corrected during build: the caution icon. The first draft mapped caution to
`status-icon--fail` (a red "x"), which does not exist in the design system as a
caution and overstates the finding -- **none of these checks can rule a
candidate out on its own**. Now `status-icon--caution` ("!"), with neutral and
unknown on `status-icon--skip`.

### Scope confirmation

`git diff --stat`: `web/app.py` **+10** (one import, one `lookup` call, one
template variable) and `web/templates/candidate_detail.html` **+97**, plus the
new `web/model_features.py`. **Zero deletions, zero modifications to any
scoring, ranking, tier or filter path.** The panel closes with the point stated
plainly to the reader: these values are *inputs* to the score above, not an
additional penalty applied after it.

**The evidence layer now surfaces all four deployed model improvements** --
crowding (2026-08-05), variability (2026-08-06), Gaia astrometry (2026-08-14)
and the Optuna hyperparameters (2026-08-15, visible via the score itself) --
alongside the pre-existing TFOP, centroid, multi-sector, RV and conformal
layers.

### Recurrence worth recording

Restarting the app for this change hit **launchd exit 78 (EX_CONFIG)** again --
`com.apple.macl` extended attributes had re-accumulated on the log files, the
same root cause as the earlier TCC outage. Rotating the logs aside and
re-bootstrapping fixed it (pid 18665, PPID 1). **This is the second occurrence**,
which strengthens the standing open item about the project living under
`~/Downloads`: the fix is reliable but it is a recurring manual step, not a
solved problem.

Files: `web/model_features.py` (new), `web/app.py`, `web/templates/candidate_detail.html`.

# >>> PROJECT RELOCATED 2026-08-22 <<<

## CANONICAL PATH IS NOW `/Users/anujtripathi/Developer/ExoplanetAI`

**The project no longer lives under `~/Downloads`.** Every future task, script,
plist and doc reference must use:

    /Users/anujtripathi/Developer/ExoplanetAI

The old path `/Users/anujtripathi/Downloads/ExoplanetAI` **no longer exists**.

### Why: three identical launchd failures, not two

`com.apple.macl` extended attributes accumulated on the launchd log files under
`~/Downloads`, and launchd then failed to open them, exiting **78 (EX_CONFIG)**.
Occurrences: the original TCC/Full-Disk-Access incident, the recurrence during
the UI evidence-layer deployment, and a **third, found during this migration's
own pre-flight** -- the service was already dead on arrival (last tick
2026-08-21 00:39 UTC), so this was not a healthy service being taken offline.

macOS TCC protects exactly three user folders: **Desktop, Documents,
Downloads**. `~/Developer` is outside that set.

### The move

`mv` on the **same filesystem** (device 16777230 both sides), so a rename, not a
copy -- it completed in **0.003 s**. This mattered: the tree is **96 GB**
(71 GB `data/`, 24 GB `.git`) against **65 GiB free**, so a copy-based move
would have failed outright. A fresh `git clone` was never an option either: most
of the bulk is gitignored light curves.

Verified after: source directory gone, 96 GB at the destination, all 17
top-level entries byte-identical, `git fsck` clean, 111 commits intact.

**Integrity was a pure location change.** Every checksum matched the
pre-migration snapshot:

| artifact | status |
|---|---|
| `best_model.joblib` | `fe3fa82f...` MATCH |
| `best_model_metadata.json` | MATCH |
| `conformal_calibration.json` | MATCH |
| `multivariate_ood_detector.joblib` | MATCH |
| `bootstrap_ensemble` manifest + 32 members | MATCH |
| `training.csv` | md5 AND **sha256** `28b70225...` MATCH |

### Paths updated -- found by repo-wide grep, not by guessing

`grep -rl "/Downloads/ExoplanetAI"` returned 8 files. Four were updated, four
were deliberately left alone:

| file | action |
|---|---|
| `~/Library/LaunchAgents/com.exoplanetai.app.plist` | **updated** (3 refs: WorkingDirectory, StandardOutPath, StandardErrorPath); `plutil -lint` OK |
| `web/com.exoplanetai.app.plist` | **updated** (3 refs); verified identical to the live copy |
| `.claude/settings.local.json` | **updated** (2 permission entries) |
| `models/multivariate_ood_meta.json`, `models/training_feature_ranges.json` | **updated** (informational `source` field only -- verified by grep that no code ever reads it as a path) |
| `models/versions/*_pre33_24feat.json` (2) | **NOT touched** -- preserved historical artifacts; rewriting them would falsify where they were actually built |
| `code/k2_pilot/download_run.log`, `tls_run.log` | **NOT touched** -- historical run logs, same reasoning |

No virtualenv exists (system Python 3.11), so there were no baked-in venv paths.
The stale `claude/quizzical-aryabhata-fa87b2` worktree and branch did not survive
the move and **that is the desired outcome** -- its only work was already landed
in `be4794ce`, and it was recorded as a merge hazard (1,023 lines behind main).
`git worktree list` now shows one clean entry; `git worktree prune --dry-run` is
empty.

### THE XATTR EXPERIMENT -- the reason this migration exists

The moved log files carried their `com.apple.macl` attributes with them (xattrs
travel with the file), so they were rotated aside to force launchd to create
**fresh** files at the new path. That makes the comparison clean: same process,
same filenames, only the location differs.

| file | location | xattrs |
|---|---|---|
| `launchd_stderr.log` (fresh) | `~/Developer` | **NONE** |
| `launchd_stdout.log` (fresh) | `~/Developer` | **NONE** |
| `launchd_stderr_preMigration_*.log` | was `~/Downloads` | `com.apple.macl` **and** `com.apple.provenance` |

Re-checked after real activity: still **no `com.apple.macl` anywhere**.
`scheduler.log` carries `com.apple.provenance` alone, which was never the
failure signature -- the Downloads files carried both.

**What this rules out vs. what it does not.** It confirms the immediate
mechanism is location-specific: the OS is not applying `com.apple.macl` at the
new path. It does **not** yet confirm the long-term fix. That requires observing
at least one more natural restart cycle -- a reboot, a sleep/wake, or several
days of uptime -- without the exit-78 signature returning. **Zero EX_CONFIG and
zero errors in the new log so far.**

### Verified working from the new location

* launchd-supervised: **PID 32129, PPID 1**, `launchctl` status **0** (not 78),
  cwd `/Users/anujtripathi/Developer/ExoplanetAI/web`, port 5050 owned by the
  same PID -- no orphan.
* Frozen-test AUC recomputed from the new path: **0.9454155994**, bit-identical
  to the metadata of record.
* Full end-to-end on a genuinely fresh star, **TIC_179583882**: download ->
  preprocess -> TLS Success -> all 33 features (**0 required NaN, 0 absent**;
  `FAP` and `transit_shape_ratio` optional-NaN) -> scored **p = 0.9598** -> OOD
  flag False, in-distribution True, conformal set returned.
* Candidate pages render HTTP 200 with zero errors, all evidence layers present
  including the new contamination/blend panel.
* Frozen split re-verified: 5,496 rows, 5,496 unique hosts, **0 duplicates, 0
  straddling**, frozen test still 1,098.
* Git fully functional: remote intact, `fetch` reachable, history complete.

### Old location

`/Users/anujtripathi/Downloads/ExoplanetAI` **is gone** -- the `mv` consumed it.
Nothing to delete; no cleanup decision remains. `~/Downloads` still contains the
user's unrelated personal files, untouched.

## Retrain-tick timeout -- FIXED. Plus two corrections to my own diagnosis.

Pre-existing bug, **surfaced by the migration restart, not caused by it**. The
migration only mattered because restarting made a due tick fire immediately.
Production model `fe3fa82f...` and the promotion gate untouched; this changes
only the scheduler's calling code.

### The bug

`_scheduler_loop` called `retrain_pipeline.scheduler_tick()` **bare**, with no
bound, at `job_runner.py:396`. The same file already had `_call_with_timeout`
and already used it for the reverify path (30s) and `fetch_fresh_exclusion_data`
(60s) -- its own docstring describes the identical failure mode ("called plain
`pd.read_csv(url)`, which has no timeout of its own"). The retrain tick simply
never got the guard.

### CORRECTION 1: it was NOT hung. I called that wrong.

I reported the tick as "genuinely hung, not slow", from an ESTABLISHED-but-idle
MAST socket with empty send AND receive queues, one port held 18+ minutes, and
~0% CPU. That evidence is consistent with a hang -- and it was still the wrong
conclusion.

**Proof it was working:** the tick had appended a complete row for
`TIC_73448352` -- label 1, all 33 features populated, **zero required-NaN and
zero optional-NaN**. The file parsed clean afterwards at 5,497 rows, 51 columns,
0 duplicate hosts and **0 rows with a wrong field count**. No truncation, no
partial write, despite my restarting it mid-flight.

The idle socket was the gap *between* per-star operations, not a dead
connection. `process_and_append_new_examples` runs a full download -> preprocess
-> TLS -> crowding/variability/Gaia pipeline per star, and most of that is not
network time.

### CORRECTION 2: my first timeout value (600s) was too short and would have caused harm

Sized against the wrong model of the problem. The real numbers:

* `PER_TICK_MAX_NEW = 25` -- one tick processes up to 25 stars
* measured per-star cost (`e2e_fresh_star_audit`): TLS alone **54s**, ~60-90s
  end to end
* so a full batch is legitimately **25-37 minutes**
* there are currently **51 pending watch labels**, i.e. real backlog -- full
  batches are the expected case, not the exception

600s would have truncated a legitimate batch at roughly star 8 of 25. Because
the work is resumable via `label_watch_queue` that loses no data, but it would
have silently throttled throughput to ~10 stars per 24h cycle instead of 25 --
a self-inflicted regression dressed as a safety fix.

**`RETRAIN_TICK_TIMEOUT = 3600`** (1 hour): ~2x headroom over the worst
legitimate batch, while still converting an indefinite stall into a logged,
recoverable event.

### The fix, following the existing convention exactly

```python
result = _call_with_timeout(retrain_pipeline.scheduler_tick,
                            timeout=RETRAIN_TICK_TIMEOUT,
                            default=_TICK_TIMED_OUT)
db.set_last_retrain_tick_at(db.now_iso())
if result is _TICK_TIMED_OUT:
    retrain_status = "timeout"
    log.error("RETRAIN    tick TIMED OUT after %ss and was abandoned; ...")
    raise _TickTimeout()
```

Three details that are deliberate:

* **A sentinel object, not `None`.** `_call_with_timeout` signals a timeout by
  *returning* `default`, not by raising. `scheduler_tick()` returns `None` on
  success, so a `None` default would make a successful tick indistinguishable
  from a timed-out one.
* **The timestamp is written on BOTH paths.** If a timed-out tick left it unset,
  the tick would stay due and re-fire every 60s, each retry abandoning another
  thread against the same stalled endpoint -- turning one stall into a thread
  leak. (`_call_with_timeout` uses `shutdown(wait=False)`, and the abandoned
  threads are not daemon threads; a test process would not even exit while one
  was alive.) The cost is at most one skipped 24h cycle, logged at ERROR.
* **A private `_TickTimeout`** so the timeout skips the post-tick counter read
  without being logged as an unexpected exception by the generic handler.

### Tested, not just reviewed

| test | result |
|---|---|
| forward references resolve (constants defined *below* their use site) | PASS |
| timeout FIRES on a function that sleeps 3600s, bound 2s | returned in **2.01s**, sentinel returned |
| normal path unchanged -- fast call returning `None` | 0.0003s, correctly **not** treated as a timeout |
| loop RECOVERS -- 3 consecutive timed-out cycles | 3/3 completed, exception never escaped |

Then live: service restarted (**PID 34017, PPID 1, launchctl status 0**), health
ok, `RETRAIN_TICK_TIMEOUT = 3600` confirmed in the running module, and the next
due tick fired and began real per-star work against MAST.

### STILL OPEN and now MORE important: `/health` "stalled" semantics

`/health` returns **503 `stalled`** whenever `seconds_since_last_tick` exceeds
its threshold. But the scheduler loop legitimately blocks inside a 25-37 minute
batch, so **a perfectly healthy tick now reports `stalled` for most of its
duration** -- observed climbing past 217s on the very next tick.

Previously I called this low-priority because it flagged a genuine problem. With
the correct understanding it is worse: it will fire on every full batch, i.e.
routinely. It conflates "the loop has not ticked recently" with "the loop is
broken". The fix is to report an in-progress tick distinctly (a
`retrain_in_progress` state, or refresh the heartbeat from inside the batch).
**Not fixed here** -- it is a behaviour change to a health endpoint and wants its
own decision.

### Data integrity

`training.csv` is now 5,497 rows: the 2 rows from before plus `TIC_73448352`
appended by the interrupted tick. All three are complete and well-formed; 0
duplicate hosts. **They remain UNCOMMITTED local state** by standing
instruction -- committing them is a separate decision. Model, metadata,
conformal, ensemble, OOD detector and promotion gate all untouched.
