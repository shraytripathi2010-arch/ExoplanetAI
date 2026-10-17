# Follow-up of the vetting leads against all TESS data

**Dates:** 2026-10-12 (TIC 173877131, TIC 152358293) and 2026-10-17 (TIC 231053750).

**Verdict: all three refuted.** They are recorded as manual rejections in the web app, each pointing to this file. No candidate from the 296-star pool survived vetting plus follow-up (register section M).

The numbers below are derived by this project from public TESS light curves and from the stellar parameters already stored with each candidate. No ExoFOP content or observer data is reproduced; ExoFOP was used only to check TOI/CTOI status.

## Method, in brief

**Data and cleaning:**
- Every SPOC and HLSP light-curve product for the star, from MAST.
- Each product cleaned with a verbatim copy of the pipeline's own cleaning (`06_download_unknown.clean_light_curve`): PDCSAP flux, `quality == 0`, a 5-MAD clip, then a Savitzky–Golay flatten over 13.4 h.
- The copy reproduces the pipeline's stored processed light curves for the original sectors exactly in point count and timestamps (|Δflux| ≤ 5e-7).

**Dip measurement:** binned medians, as on the candidate page, in bins of duration/5 with the baseline at least one duration away. The error is the measured bin-to-bin scatter / √n, so red noise is included.

**Ephemeris propagation:** the stored TLS period and T0 uncertainties were carried forward to every other sector. Wherever the predicted timing uncertainty exceeded a quarter of the transit duration (in practice, every sector), a box search near the stored period replaced a blind fold. Each search was checked three ways:
- against an **inverted-flux null** (the same search for bumps);
- by **injecting the original dip** at random phases, to test "would it have been seen?";
- with a single-event scan or a joint multi-sector search, where the signal strength allowed.

**Products not used:**
- 10- and 30-minute FFI products have ≤ 3 points per transit and are not decisive.
- TASOC is excluded (zero-centred flux), and TARS/T16 could not be read.

## TIC 173877131: refuted

| | |
|---|---|
| **Stored signal** | P = 6.4712 ± 0.0436 d, duration 1.02 h, depth 180 ± 18 ppm (10.1σ), visibility ratio 4.52, sector 12 only (3 transit epochs) |
| **Star** (stored) | Teff 8,061 K, R★ 3.67 R☉, M★ 1.95 M☉ |
| **Sectors checked** | 12, 38, 103 (2-minute SPOC; 103 also 20-second), plus FFI products. Sector 65 is on the detector with no light curve. |
| **Predicted timing uncertainty** | 115 h in sector 38, 409 h in sector 103: no blind fold possible |

**Evidence against:**
- **Sector 12 itself:** a single-event scan finds one clean event (5.6σ, 283 ppm) and one in the sector's last hours (4.9σ, 247 ppm). The stored T0 epoch shows nothing (0.5σ, 25 ppm against 180 expected).
- **Sector 38, the most sensitive data:**
  - event noise is 37 ppm, so a 180 ppm transit would be 4.9σ per event;
  - **no event reaches 4σ;**
  - the injected original dip is recovered in **100%** of trials (median 11σ);
  - the best box near the period is 74 ppm (3.8σ, against a 2.9σ bump null), inconsistent with 180 ppm.
- **Sector 103:** no consistent dip in the 2-minute data. The 20-second box solution rests on two events, one of the three predicted falling in a data gap.
- **All sectors combined:** the dip dilutes to 46 ppm, as expected if only sector 12 contains it.
- **Geometry:** the 1.02-h transit is 0.10× the longest central transit on this star (10.1 h), so it would need a grazing orbit (b ≈ 1).

**ExoFOP:** not a TOI or CTOI as of 2026-10-12 18:50 UTC (target page and full TOI/CTOI lists).

## TIC 152358293: refuted as a planet on this star

| | |
|---|---|
| **Stored signal** | P = 0.83348 ± 0.00215 d, duration 0.75 h, depth 513 ± 44 ppm, visibility ratio 5.17, sector 104 only (27 transits) |
| **Star** (stored) | Teff 5,987 K, R★ 0.92 R☉, M★ 1.1 M☉ |
| **Sectors checked** | 104, 105, 106 (2-minute SPOC), 97 (QLP, 200 s), plus 3, 4, 30 (FFI, not decisive). Sector 107 is on the detector with no light curve. |
| **Predicted timing uncertainty** | 2.5 h (sector 105), 4.2 h (106), 14 h (97): no blind fold possible even in the adjacent sectors |

**Evidence against:**
- **The dip is real within sector 104 and steady across both orbits:** 497 ± 128 ppm (13 transits) and 442 ± 123 ppm (14 transits).
  - The original 11.5σ overstates it: the error at that phase was below the 10th percentile of errors at other phases.
  - At a typical error it is about 6σ, still above every noise-only phase (maximum 4.7σ).
- **It is absent from the next sectors.** Joint box search over sectors 104–106 near the period:

  | | S104 | S105 | S106 |
  |---|---|---|---|
  | at the best joint ephemeris | 7.5σ | 0.4σ | 0.7σ |
  | injected original dip recovered at ≥ 3σ | 97% | 82% | 89% |

  - No coherent joint solution: 2.7σ, against a 2.4σ bump null.
  - Sector 105 starts two days after sector 104 ends.
- **Sector 97 (2025):** no consistent dip; the injected dip is recovered 5/5 (median 6.8σ).
- **Interpretation:** a dip steady within one sector and absent from the next is the signature of a nearby eclipsing binary in that sector's aperture, or a sector-specific systematic, not a planet on the target.

**ExoFOP:** not a TOI or CTOI as of 2026-10-12 18:50 UTC.

## TIC 231053750: refuted

| | |
|---|---|
| **Stored signal** | P = 2.2806 ± 0.0083 d, duration 1.85 h, depth 295 ± 67 ppm (4.4σ), visibility ratio 1.98, sector 2 |
| **Star** (stored) | Teff 5,010 K, R★ 10.2 R☉, mass not catalogued (pipeline default 1 M☉). A K giant. |
| **Sectors checked** | 2, 3, 29, 30, 69, 96, 104, 105 (2-minute SPOC; 96, 104, 105 also 20-second), plus FFI products |
| **Predicted timing uncertainty** | 3.6 h (sector 3) to 250 h (sector 105) |

**Evidence against:**
- **Not significant in its own sector:** 9.1% of random phases in sector 2 are at least as deep (noise-only 99th percentile 8.1σ, maximum 9.4σ). The folded "dip" is a trough in a ±2.5-h wave, consistent with the correlated variability of a giant star.
- **No coherent signal across 8 sectors:**
  - per-sector box solutions scatter from 2.256 to 2.305 d, 9 of 27 at the edge of the search window;
  - 20-second and 2-minute products of the same sector disagree;
  - a joint search over the eight 2-minute sectors gives 124 ppm (3.8σ), against a 3.6σ bump null;
  - its per-sector pattern (quadrature sum 10.4σ) ranks above only 15% of time-scrambled nulls (median 12.0σ), and the inverted-flux null gives 9.9σ.
- **Geometry:**
  - **the orbit:** at P = 2.28 d the semi-major axis is about 7.3 R☉ (1 M☉), 9.2 R☉ (2 M☉) or 10.5 R☉ (3 M☉);
  - **the star:** R★ is ≈ 9.9 R☉ from its catalogue brightness, distance and Teff, matching the stored 10.2 R☉;
  - so the orbit would lie inside, or at the surface of, the star.

**ExoFOP:** not a TOI or CTOI as of 2026-10-17 18:29 UTC.
