# Non-TESS surveys — groundwork, measured before any data was fetched

Prepared 2026-08-03. **Nothing was downloaded.** Every number below comes from
a catalogue or metadata query; light-curve searches stopped at the search-result
table and never called `.download()`. No model, training set, scheduler,
promotion gate or frozen split was touched.

This exists so the actual experiment does not spend its first hour rediscovering
access syntax, and — more importantly — so two findings that change the shape of
the experiment are on the table *before* it is designed.

---

## The two things to read first

### 1. The K2 positive class is mostly already in the training set

K2 has 2,375 CONFIRMED entries, which reads like a large positive-class
expansion. It is not. Resolving `k2pandc` to TIC ids and intersecting with
`training.csv` **by TIC** (the same identity test that fixed the duplicate bug):

| | stars |
|---|---|
| unique K2 stars with a TIC | 1,557 |
| **already in `training.csv`** | **405 (26.0%)** — of which 390 CONFIRMED |
| genuinely new | 1,152 |

And the genuinely-new 1,152 break down very differently from the headline:

| K2 disposition | genuinely new stars |
|---|---|
| CANDIDATE (unlabelled) | 895 |
| **FALSE POSITIVE** | **238** |
| **CONFIRMED** | **13** |
| REFUTED | 6 |

**So the usable labelled yield is roughly 257 stars, and 244 of them are
negatives.** The confirmed-planet overlap is near-total because this project's
positive class was drawn from `pscomppars`, which already includes K2
discoveries — `training.csv` today holds 346 `K2-*` and 26 `EPIC_*` hosts.

Two consequences worth deciding on before writing code:

- This is a **negative-class** expansion, not a positive-class one. The last
  negative-class expansion attempted here (TOI FA dispositions, +98 stars) came
  back as noise. That is not a reason to skip this one — it is four times
  larger — but it should set the prior.
- Those 405 overlapping stars are in `training.csv` as **TESS** photometry.
  Fetching K2 photometry for them creates a second row for the same star with
  different photometry. That is the *exact* shape of the bug that put 56 stars
  on both sides of the frozen split. Exclude by TIC before downloading, not
  after.

### 2. K2's standard cadence is 30 minutes, and this project has already
### measured what that does

Verified live across 8 K2 hosts:

| exposure time | hosts offering it |
|---|---|
| **1800 s (30 min)** | **8 of 8** |
| 60 s (short cadence) | 3 of 8 |

30-minute sampling is **15× coarser than TESS 2-minute**. The FFI investigation
already measured this axis on TESS's own data and found a domain classifier
separates 2-min from coarse-cadence feature vectors at **AUC 0.9717** — higher
than the 0.9654 that made synthetic data actively harmful — and concluded
coarse-cadence rows are a coverage lever, not an accuracy lever.

K2 at 30 min sits squarely in that regime, and it is worse than the FFI case in
one respect: cadence would be **perfectly confounded with survey**, so a model
cannot learn "K2" and "coarse" separately. This does not decide the experiment,
but it means the domain-separability check must run *before* any retraining arm,
and a high AUC should be treated as a stopping condition rather than a caveat.

---

## Part 2, item by item

### 1. K2 via lightkurve — CONFIRMED WORKING

`lightkurve 2.6.0`. Both identifier styles resolve:

```python
lk.search_lightcurve("EPIC 201367065", mission="K2")   # 4 products
lk.search_lightcurve("K2-3", mission="K2")             # 4 products
```

Returned metadata: `mission='K2 Campaign 01'`, `exptime=1800.0`,
`author=['EVEREST', 'K2', 'K2SFF', 'K2VARCAT']`.

**Author choice is a real decision, not a default.** `K2` is the raw mission
product; `EVEREST`, `K2SFF` and `K2SC` are systematics-corrected HLSPs, each
applying its own detrending. This project's `02_preprocess.py` does its own
detrending, so stacking a corrected HLSP under it means detrending twice.
Prefer `author="K2"` for consistency with the TESS SPOC path, and record the
choice — it is a domain-shift lever in its own right.

### 2. CoRoT — NOT AVAILABLE, and it fails in a way that looks like success

```python
lk.search_lightcurve("CoRoT-2")   # returns 12 products
```

Those 12 products are **TESS Sectors 54 and 81**. Lightkurve resolved the star
name and returned TESS observations *of* it. There is no CoRoT data in the
result. Confirmed directly against MAST:

| query | result |
|---|---|
| `Observations.query_criteria(obs_collection="CoRoT")` | **0 observations** |
| `Observations.query_criteria(obs_collection="COROT")` | **0 observations** |

CoRoT light curves live on VizieR (`B/corot`, the N2-4.4 observation log) and
the CoRoT/IAS archive, in a different format, reachable by neither
`lightkurve` nor this project's existing download path. With only 35 confirmed
CoRoT planets total, building a second ingestion path for it is not worth it.
**Recommend closing CoRoT.**

### 3. Ground-based surveys — catalogues yes, light curves mostly no

Confirmed planets by discovery facility (`pscomppars`), with TIC coverage:

| facility | planets | with a TIC |
|---|---|---|
| Kepler | 2,784 | 2,784 (100%) |
| TESS | 917 | 916 |
| K2 | 549 | 548 |
| SuperWASP | 122 | 122 (100%) |
| HATSouth | 73 | 73 (100%) |
| HATNet | 67 | 67 (100%) |
| CoRoT | 35 | 35 (100%) |
| KMTNet | 139 | **0** |
| OGLE | 111 | 8 |

VizieR is reachable via `astroquery.vizier` and `Vizier.find_catalogs()` works,
but what it returns for these surveys is **published per-paper tables** —
"Follow-up photometry of HATS-1", "Transit photometry of NGTS-14Ab",
"RVs and light curves for HATS-60–HATS-69" — not a uniform survey-wide light
curve archive. SuperWASP is the best case (19 catalogues) and even there the
holdings are variable-star and eclipsing-binary studies rather than a
queryable photometric archive keyed by target.

**Assessment:** WASP/HATNet/HATSouth/KELT/NGTS together contribute ~262 confirmed
planets and **no negative class at all**. Each would need its own ingestion
path, per-survey format handling, and its own detrending assumptions. The yield
does not justify it. **Recommend closing the ground-based surveys**, or
scoping them explicitly as a label source rather than a photometry source.

### 4. Identity resolution — REUSABLE, needs one specific extension

The utility is `_training_tic_ids(confirmed)` in
[`web/retrain_pipeline.py:72`](../../web/retrain_pipeline.py). It resolves
`training.csv` hosts to TIC ids two ways: regex `^TIC_(\d+)` for
pipeline-named rows, and a `hostname -> tic_id` map built from the
`pscomppars` frame for hostname-named rows.

**It works today because the NASA archive supplies the mapping for free, and
that property holds for K2 as well:** `k2pandc` carries `tic_id` on
**4,060 of 4,064 rows** and `epic_hostname` on 4,062. So EPIC → TIC needs no
cross-match, no coordinate matching, no new service.

**What genuinely needs extending, and why it is not optional.** The function
takes one frame and reads `confirmed["hostname"]`. Host strings in
`training.csv` are already heterogeneous:

| pattern | rows |
|---|---|
| `Kepler-*` | 1,778 |
| bare hostname (`11_Com`, `16_Cyg_B`) | 1,760 |
| `TIC_<digits>` | 1,207 |
| `K2-*` | 346 |
| `WASP-*` | 173 |
| `HAT*` | 139 |
| `CoRoT-*` | 30 |
| `EPIC_<digits>` | 26 |
| `KMT-*` | 26 |

A new K2 row would arrive as `EPIC_201367065`. The current code would extract
no TIC from that string (the regex is `TIC_` only) and find no entry in a
`pscomppars`-only name map, so **the star would look brand new even when it is
already present** — reproducing the original bug with a different prefix.

Minimum extension, all mechanical:

1. Accept a list of mapping frames, not one, and add `k2pandc`
   (`epic_hostname`/`hostname` → `tic_id`).
2. Add `^EPIC[_ ](\d+)` to the identifier regexes, mapping EPIC → TIC via that
   frame rather than treating the EPIC number as an id in its own right.
3. Keep the existing behaviour for unresolvable stars — microlensing hosts
   (KMTNet 0/139, OGLE 8/111) have no TIC, cannot collide with a TESS target,
   and are correctly just absent from the set.

Write it as a shared function rather than a second copy. Two independent
implementations of "is this star already here" is how the first bug survived.

### 5. Domain-separability diagnostic — EXTRACTED AND VALIDATED, ready to use

Previously this was written inline three times against three different axes.
It is now [`domain_separability.py`](domain_separability.py), taking
`(X, domain, y)` and returning the discriminator AUC, standardized mean
differences, and the per-feature label-AUC vs domain-AUC redundancy table.

It is validated by a self-check that reproduces **both** previously recorded
numbers to **zero delta**:

| grouping | recorded | reproduced |
|---|---|---|
| 2-min vs non-2-min (`cadence_audit_results.json`) | 0.9466338101 | 0.9466338101 |
| 2-min vs COARSE only (`ffi_mixing_results.json`) | 0.9716678622 | 0.9716678622 |

Those are two different measurements and the distinction matters: `non-2-min`
lumps in 401 FINE 20-second rows, which are *finer* than 2-min, not coarser.
**0.9717 is the one the FFI decision rests on.** Run `python3
domain_separability.py` after touching that module.

Usable as-is for a new source — pass `domain=1` for the new survey's rows.
No changes needed.

### 6. Environment readiness — nothing to install

| need | status |
|---|---|
| `lightkurve` 2.6.0 | K2 search confirmed working |
| `astroquery` 0.4.11 | VizieR + MAST + NASA archive all reachable |
| K2 file format | same FITS light curves lightkurve already returns; `02_preprocess.py` needs no new reader |
| new dependencies | **none identified** |

Two operational notes rather than dependencies:

- `astroquery`'s `query_criteria(..., group=...)` emits invalid ADQL against
  the NASA archive (`ORA-00924: missing BY keyword`). Use the TAP endpoint
  directly with raw SQL, as `retrain_pipeline._read_csv_url` already does.
- An unfiltered `Observations.query_criteria(dataproduct_type="timeseries")`
  across all collections **exceeded the 600 s timeout**. Query per target or
  per collection.

---

## Open questions for the user

1. **Does a ~257-star, 95%-negative K2 expansion at 30-minute cadence still
   interest you**, given the FFI result on coarse cadence and the null result
   from the last negative-class expansion? The prompt as framed assumes a
   larger and more positive-skewed prize than the catalogue actually holds.
2. **Close CoRoT and the ground-based surveys?** Both are recommended closed
   above — CoRoT for zero MAST availability, the ground surveys for ~262
   positives, no negatives, and a bespoke ingestion path each.
3. **Which K2 author product** — raw `K2`, or a corrected HLSP? Recommend raw
   `K2` so this project's own detrending is not stacked on someone else's.
