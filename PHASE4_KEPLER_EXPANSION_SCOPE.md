# Phase 4 scope: multi-mission (Kepler) data fusion

**Status: not built, not scheduled. This is a reference document for a future
decision, written after Phases 1-3 of the multi-data-source effort
(pixel-level centroid checks at scale, RV archive cross-check, speckle/AO
imaging investigation) were completed.** Do not start this work from this
document alone -- re-confirm the "worth it if" conditions below are still
true first.

## Why this exists

Kepler expansion was raised early in this project as a way past the
classifier's ~0.90 ROC-AUC ceiling, and deliberately not pursued: a cheaper
test first (adding more TESS-only negative examples) showed no improvement,
and full Kepler integration carries real cross-mission leakage risk and a
multi-day cost. This document reassesses that call now that Phases 1-3 are
done, and lays out what a safe attempt would actually require if it's ever
picked up.

## Updated honest opinion: lower priority now, not higher

Kepler expansion and Phases 1-3 solve **different problems**, and that
difference is exactly why priority shifted:

- **Phases 1-3 add independent vetting evidence for candidates the
  classifier already found** -- centroid position, RV consistency, (and
  imaging, once relevant) -- none of it touches the classifier itself.
  Their value doesn't depend on the classifier improving at all.
- **Kepler expansion would target the classifier's own ceiling** -- more
  and more diverse training examples, on the theory that the plateau is a
  data-volume/diversity limit rather than a feature-representation limit.
  That theory is still unverified (the earlier test only tried *more of
  the same kind* of data, not *differently-flavored* data), so it remains
  a legitimate, real lever -- just not a cheap or safe one.

The practical effect of Phases 1-3 is that a candidate's trustworthiness to
a human reviewer no longer rests on the classifier's probability alone --
a Low-tier candidate with a clean centroid result and consistent RV data is
now demonstrably more useful than a Low-tier score in isolation was before
any of this existed. That reduces the urgency of chasing marginal AUC
gains, without eliminating the underlying question. Recommendation:
**treat this as a "someday, if the conditions below are met" item, not a
next-in-line priority.**

## What would need to be true before starting

All of these, not some:

1. **The plateau is reconfirmed current**, not stale. Re-run the "more
   TESS-only negatives" comparison (or equivalent) against the *current*
   feature set and model before assuming Kepler is the next lever --
   features and training data have both changed since the original
   6-experiment plateau finding.
2. **There's a real downstream use for a better classifier score.** With
   Phases 1-3 in place, ask concretely: would a higher classifier AUC
   actually change which candidates get a human's attention, given that
   tier placement is already informed by independent evidence? If the
   answer is "not much," this isn't worth it regardless of feasibility.
3. **A genuine multi-day, uninterrupted block of time is available**, not
   squeezed in piecemeal -- the leakage-safety work below is exactly the
   kind of thing that produces a subtly wrong model if rushed or split
   across sessions with context loss in between.
4. **Willingness to fully validate before trusting it**, including the
   possibility the answer is "no improvement, don't merge" after the full
   multi-day investment -- same honest-reporting standard already applied
   to every other experiment in this project.

## What "doing it safely" actually requires

This project has already built real leakage-detection machinery (the
dedicated leakage-check passes run against the v3 feature set and the
17-recovered-star merge). Reuse and extend that, rather than building new
verification from scratch. Specific risks unique to a cross-mission merge:

1. **Target overlap between missions.** Some stars were observed by both
   Kepler/K2 and TESS. Any such star must not appear in both train and
   validation/test splits, and not inconsistently across positive/negative
   classes. Requires a real cross-match (by TIC/KIC/EPIC identifiers and by
   coordinates, since not every star has all three IDs) before any split is
   drawn, not after.
2. **Instrumental-signature leakage.** Kepler's long-cadence data, quarter
   rolls, safe-mode gaps, and detrending artifacts look nothing like TESS's
   sector structure. A classifier can learn to separate "Kepler-shaped
   data" from "TESS-shaped data" instead of "transit-shaped signal" from
   "not" -- this would show up as excellent cross-validation performance
   that doesn't generalize, exactly the failure mode the leakage checks
   elsewhere in this project were built to catch. Needs its own dedicated
   check: verify the trained model's decision boundary isn't trivially
   predictable from mission-identity alone (e.g., a simple classifier
   trained only on "which mission is this from" style summary stats
   shouldn't be able to reconstruct the real model's predictions).
3. **Feature-scale mismatch.** Kepler's ~4-year continuous baseline yields
   far more transits per star than a TESS sector (or even a few stitched
   sectors) ever will. Raw distinct-transit-count and similar features need
   harmonization (e.g., per-baseline-length normalization) or the model
   could use transit count as a proxy for "which mission," not planet
   likelihood -- a specific case of risk #2 worth calling out on its own
   since it's the most likely one to slip through unnoticed.
4. **Label harmonization.** Kepler KOI dispositions (CONFIRMED/CANDIDATE/
   FALSE POSITIVE, with their own vetting flags) don't map 1:1 onto this
   project's existing TOI-based labeling scheme. Needs an explicit,
   documented mapping decided before training, not inferred implicitly by
   whoever merges the datasets.

## Realistic time estimate

Broken down, not just "multi-day":

| Step | Estimate |
|---|---|
| Kepler light curve acquisition (reusing `06_download_unknown.py`'s lightkurve-based tooling, adapted for Kepler's mission/cadence conventions) + validating the download actually works against Kepler's data format quirks | 1-2 days |
| Preprocessing/TLS adaptation for Kepler's longer baselines and quarter-gap structure (`02_preprocess.py`/`03_transit_search.py` or their 06-embedded equivalents) | 1 day |
| Cross-mission target de-duplication, label harmonization, and leakage-safe split construction | 1 day |
| Retraining + the full honest-validation pass this project always does before trusting a new model (bootstrap CIs, leakage re-check, error analysis) | 1 day |
| **Total** | **~4-6 focused days** |

This assumes no major surprises in Kepler's data format or labeling: treat
it as a floor, not a ceiling.

## What's explicitly NOT in scope even if this is picked up later

- Automating this as a recurring pipeline stage -- this would be a one-time
  (or rare, deliberate) retraining event, not something that runs on every
  Update.
- Touching anything in Phases 1-3 -- those are independent evidence layers
  and stay exactly as built regardless of what happens here.
