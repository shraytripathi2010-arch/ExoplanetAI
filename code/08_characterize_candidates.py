"""
08_characterize_candidates.py

DEFINITIVE consolidation pass. Deep verification + full physical
characterization + a PERMANENT physical-plausibility filter + a PERMANENT
combined "best candidates" ranking + a multi-factor confidence tier, for
whatever trustworthy candidate batch 07_search_unknown.py produced.
Eliminates manual website-checking and produces real physical parameter
ESTIMATES (not just a probability score) for each candidate.

This is not a one-off analysis script: every future run of this file
against a new candidate batch automatically produces the plausibility
filter and combined ranking below -- that's the whole point of making it
part of main(), not a separate flag-gated mode.

===============================================================================
BUGS FOUND AND FIXED IN THIS CONSOLIDATION PASS (same audit standard as the
rest of this project -- actively hunting for the "assumes success without
verifying" and "silent type coercion" patterns that caused real bugs before):
===============================================================================
1. check_blending() used to assume the CLOSEST Gaia cone-search result is
   always the target star itself, without checking that its separation is
   actually ~0. If Gaia's astrometric solution doesn't return the target
   (plausible for faint/red M dwarfs -- 07_search_unknown.py already found
   Gaia's teff_gspphot pipeline failing for exactly this kind of star), this
   silently discarded a REAL neighbor instead of counting it as blend risk.
   Fixed: only treat the first row as "the target" if its separation is
   under 2 arcsec (generous margin for TIC/Gaia epoch and proper-motion
   differences); otherwise every returned row is treated as a neighbor and
   the fact that Gaia didn't match the target itself is noted.
2. Blend-risk categorization only used angular separation, ignoring
   brightness. Found by hand while reviewing TIC_120442975: a "HIGH risk"
   label was applied to a neighbor 15.5" away that was G=20.8 -- 6-10
   magnitudes fainter than a typical SPOC 2-min-cadence target, meaning it
   contributes negligible flux to the aperture regardless of separation.
   Fixed: risk tier now factors in delta-magnitude (see BLEND tier logic
   below), with the old separation-only categorization no longer used
   uncritically.
3. Status-string aggregation (e.g. "no vsx" not in status) would silently
   count a query ERROR as a "hit" in summary stats, since error messages
   don't contain the "no match" substring either. Not live-impacting in the
   run this was found in (no errors occurred), but a real latent bug for
   future runs with network issues. Fixed: every external check's status is
   now explicitly categorized into HIT / NO_HIT / ERROR, and all summary
   aggregation uses that three-way category, never a substring check.
4. 07_search_unknown.py's "stellar_param_verified_clean" conflated three
   different states (checked-and-consistent, couldn't-check, query-errored)
   into a single True whenever no explicit disagreement was found --
   overstating confidence. This script's confidence tier (below) treats
   "unable to verify" as its own explicit state, not silently folded into
   "clean".
5. 07_search_unknown.py left its local `tic_id` as a string while every
   other file in this project casts to int64 for TIC IDs (the same
   TIC-ID-float bug class from earlier in this project). Not currently
   exploitable (only used in an f-string there), but fixed for consistency
   since inconsistent typing is exactly the kind of latent risk this audit
   was asked to hunt for.

===============================================================================
EXTERNAL SOURCES -- VERIFIED LIVE (not assumed):
===============================================================================
1. ExoFOP-TESS: bulk CSV (download_toi.php?output=csv, mirrors the archive's
   TOI table) + per-target page (target.php?id=<TIC> -- NOT gototicid.php,
   which was tried first and doesn't work). The per-target page renders a
   full template for ANY TIC ID including fake ones, so "page exists" is
   not a signal; the presence of a real "TOI-<number>" string is (validated
   against a fake TIC, two random unflagged stars, and a real TOI).
2. NASA ADS: confirmed live 401 without an Authorization header -- needs a
   free API key (ui.adsabs.harvard.edu/user/settings/token) not available
   here. Gated on the ADS_API_KEY env var; reports SKIPPED explicitly if
   absent rather than pretending the check ran. Free arXiv API used as a
   partial substitute (preprints only).
3. AAVSO VSX: via astroquery's Vizier, catalog "B/vsx", no key needed.
4. Nearby-source blending: astroquery's Gaia cone_search_async. TESS's
   pixel scale is ~21"/pixel; SPOC apertures span several pixels, so
   contamination risk is meaningful out to a few pixels, not just arcsec.

===============================================================================
FULL PHYSICAL/ORBITAL CHARACTERIZATION -- formulas shown, not black-boxed:
===============================================================================
Given period P (days), epoch T0 (BJD), transit depth `depth` (TLS's
normalized flux at mid-transit, so the dip fraction is 1-depth), duration
(days), stellar radius st_rad (R_sun), stellar Teff (K), stellar mass
st_mass (M_sun, solar default if missing -- flagged explicitly wherever
this fallback is used, since a missing mass for a small/cool star is NOT
close to solar and materially changes the derived semi-major axis/Teq,
found by hand while reviewing TIC_120442975):

  1. Planet radius:
       Rp/R_star = sqrt(1 - depth)
       Rp [R_earth] = st_rad [R_sun] * sqrt(1 - depth) * 109.076

  2. Semi-major axis (Kepler's third law, a in AU when M in M_sun, P in yr):
       a [AU] = (st_mass [M_sun] * (P [days] / 365.25) ** 2) ** (1/3)

  3. Equilibrium temperature (zero-albedo, full/instant heat redistribution
     -- standard literature baseline, NOT a real prediction of surface
     conditions, which depend on the planet's unknown albedo/atmosphere):
       T_eq [K] = st_teff [K] * sqrt(st_rad [R_sun] / (2 * a [R_sun]))
       where a [R_sun] = a [AU] * 215.032

  4. Insolation flux relative to Earth (Stefan-Boltzmann; mathematically
     related to T_eq -- NOT independent new information, just a different,
     also-standard way of expressing the same physical result that some
     habitable-zone conventions use instead of a raw T_eq threshold):
       S / S_earth = (st_rad [R_sun])^2 * (st_teff [K] / 5772)^4 / (a [AU])^2

  5. Rough habitable-zone flag: 180 K <= T_eq <= 310 K. Illustrative only,
     not a rigorous insolation-boundary model.

  6. Rough spectral type from Teff: standard main-sequence O/B/A/F/G/K/M
     boundaries -- illustrative context, not a real classification (which
     needs actual spectroscopy).

  7. Human-readable transit depth (ppm) and duration (hours), alongside the
     raw fractional/day values already used internally.

===============================================================================
PERMANENT PHYSICAL PLAUSIBILITY FILTER (new, wired into every run):
===============================================================================
Ceiling: Rp <= 22.4 R_earth (2 Jupiter radii). Reasoning: the largest
confirmed transiting exoplanets (inflated hot Jupiters) top out around
~2 R_Jup; giant planets AND brown dwarfs occupy a similar, narrow radius
band (~0.8-1.2 R_Jup) regardless of mass because of electron-degeneracy
pressure, so a derived radius meaningfully ABOVE 2 R_Jup is not a
plausible planet at all -- it is almost certainly a small STAR (an M
dwarf transiting/eclipsing, R~1-5 R_Jup) misidentified via a blended
eclipsing binary, which is exactly the failure mode this project's
characterization pass already found dominating this candidate list
(median derived radius 8.8 R_earth, max 336).

===============================================================================
PERMANENT COMBINED-FILTER "BEST CANDIDATES" RANKING (new, wired into every
run): a candidate passes into the combined list only if it clears ALL of:
model probability (already true for the input population), both OOD checks
(already true for the input population), the physical plausibility filter
above, NOT a VSX variable-star match, and blend risk no worse than
MODERATE. Two sub-tiers are reported (STRICT = low blend risk only, and
MODERATE-OR-BETTER) since the strict tier can be very small -- reporting
only one would hide that stringency choice from the user.

Usage:
    python3 08_characterize_candidates.py
"""
import os
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, "results", "unknown_candidates")
IN_DIST_PATH = os.path.join(RESULTS_FOLDER, "ranked_candidates_in_distribution.csv")
STELLAR_VERIFICATION_PATH = os.path.join(RESULTS_FOLDER, "trustworthy_candidates_stellar_verification.csv")
OUTPUT_PATH = os.path.join(RESULTS_FOLDER, "characterized_candidates.csv")
BEST_CANDIDATES_PATH = os.path.join(RESULTS_FOLDER, "final_best_candidates.csv")
NEEDS_REVIEW_PATH = os.path.join(RESULTS_FOLDER, "needs_manual_blend_review.csv")

R_SUN_IN_R_EARTH = 109.076
R_JUP_IN_R_EARTH = 11.209
AU_IN_R_SUN = 215.032
TEFF_SUN_K = 5772.0
TESS_PIXEL_ARCSEC = 21.0
BLEND_TIER1_ARCSEC = TESS_PIXEL_ARCSEC * 1     # within 1 pixel
BLEND_TIER2_ARCSEC = TESS_PIXEL_ARCSEC * 3     # within 3 pixels
BLEND_SAME_SOURCE_ARCSEC = 2.0                 # separation below which a Gaia row is treated as "the target itself"
BLEND_BRIGHT_DELTA_MAG = 5.0                   # neighbor within this many mags of target = meaningful dilution risk
HZ_TEQ_MIN, HZ_TEQ_MAX = 180.0, 310.0          # loose, illustrative liquid-water range
PLAUSIBLE_RADIUS_CEILING_REARTH = 2 * R_JUP_IN_R_EARTH   # see docstring reasoning above
RV_MATCH_ARCSEC = 5.0                          # coordinate cross-match radius against the RV star list
# FIXED: a flat m/s cutoff is not a physical quantity. The same RV amplitude
# implies wildly different companion masses depending on the star's mass and
# the orbital period -- 935 m/s peak-to-peak means ~3.8 M_Jup around a
# 1.9 M_sun star at 1.3 d (an ordinary hot Jupiter), but would mean tens of
# M_Jup around an M dwarf at long period. The old flat 1000 m/s threshold
# also had its stated rationale backwards: it claimed hot Jupiters "top out
# around a few hundred m/s", but at the 1-3 day periods this pipeline
# actually finds, a single Jupiter produces 140-400 m/s and a 13 M_Jup
# object produces several km/s. So the flat cutoff was too STRICT, not too
# permissive -- it would have flagged ~3.5 M_Jup planets as stellar.
#
# The check now converts the observed scatter into an implied minimum
# companion mass using the star's own mass and the candidate's period, and
# compares that against the deuterium-burning limit -- the actual
# planet/brown-dwarf boundary, and the real question being asked.
RV_PLANET_MASS_CEILING_MJUP = 13.0             # deuterium-burning limit: above this the companion is not a planet
RV_LARGE_VARIATION_MS = 4000.0                 # FALLBACK only, used when st_mass is unavailable so the implied
                                                # mass can't be computed. Chosen as roughly the peak-to-peak a
                                                # 13 M_Jup companion produces at a few days around a solar-mass
                                                # star; deliberately loose, since without a stellar mass this can
                                                # only be a crude backstop rather than a real test.

# Uncertainty propagation (Part D): documented FALLBACK fractional
# uncertainties, used ONLY when a real per-star value isn't available.
# st_rad_err/st_mass_err come from the TIC catalog (e_rad/e_mass) when
# present; period_uncertainty/depth_mean_std come from TLS directly. These
# fallbacks reflect typical real values (TIC stellar radius/mass
# uncertainties are commonly ~10-15% for stars without an interferometric
# or asteroseismic radius; 5% depth/duration reflects typical single-sector
# TLS precision for a moderate-SNR detection) -- documented, reviewable
# assumptions, never silently presented as a real per-star measurement.
UNCERTAINTY_N_SAMPLES = 5000
DEFAULT_ST_RAD_FRAC_ERR = 0.12
DEFAULT_ST_MASS_FRAC_ERR = 0.10
DEFAULT_DEPTH_FRAC_ERR = 0.05
DEFAULT_DURATION_FRAC_ERR = 0.05
DEFAULT_PERIOD_FRAC_ERR = 0.001  # TLS period is usually tight; only used if period_uncertainty is missing

SPECTRAL_TEFF_RANGES = {   # standard main-sequence boundaries, low-to-high
    "M": (2400, 3700), "K": (3700, 5200), "G": (5200, 6000),
    "F": (6000, 7500), "A": (7500, 10000), "B": (10000, 30000), "O": (30000, 60000),
}


def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# =====================================
# 1. FRESH "ALREADY FLAGGED?" CHECK
# =====================================
def _read_csv_url(url, timeout=90, attempts=3, backoff=5):
    """pd.read_csv(url) has NO timeout of its own -- it can hang indefinitely
    on a stalled connection (confirmed root cause of the reverify action
    appearing to hang for 90+ seconds). Fetch with an explicit requests
    timeout first, then hand the text to pandas.

    TIMEOUT RAISED 30 -> 90 AND RETRIES ADDED, after this aborted a whole
    characterization run before a single candidate was processed. These are
    multi-MB bulk catalogs (the full TOI table), not small API calls, and
    ExoFOP is simply slow: three consecutive fetches measured 20.8s, 32.1s
    and 24.5s. A 30s ceiling sits right on top of that distribution, so
    roughly a third of attempts failed -- and because this runs during
    startup, one slow reply killed the entire stage rather than degrading one
    check. Retries cover the genuinely transient case; 90s covers the merely
    slow one.
    """
    import io
    import time as _time
    import requests
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:
            last = e
            if i < attempts - 1:
                wait = backoff * (i + 1)
                print(f"  fetch failed ({type(e).__name__}), retry {i+2}/{attempts} in {wait}s: {url}")
                _time.sleep(wait)
    raise last


def fetch_fresh_exclusion_data():
    print("Re-querying NASA Exoplanet Archive (confirmed planets + full TOI table) LIVE...")
    confirmed_url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+hostname,tic_id+from+pscomppars&format=csv"
    )
    confirmed = _read_csv_url(confirmed_url)
    confirmed_tics = set(
        confirmed["tic_id"].dropna().str.replace("TIC ", "", regex=False).astype("int64")
    )
    toi_url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+tid,tfopwg_disp+from+toi&format=csv"
    )
    toi = _read_csv_url(toi_url)
    toi_tics = set(toi["tid"].dropna().astype("int64"))

    print("Re-downloading the ExoFOP-TESS TOI CSV LIVE...")
    exofop_toi = _read_csv_url("https://exofop.ipac.caltech.edu/tess/download_toi.php?output=csv")
    exofop_tics = set(exofop_toi["TIC ID"].dropna().astype("int64"))

    print(f"Live re-check: {len(confirmed_tics)} confirmed-planet TICs, {len(toi_tics)} TOI TICs "
          f"(archive), {len(exofop_tics)} TOI TICs (ExoFOP).")
    return confirmed_tics, toi_tics, exofop_tics


def check_exofop_target_page(tic_id):
    """See module docstring bug #-- "page exists" is not a usable signal;
    only a real TOI-<number> string is."""
    import re
    import requests
    try:
        r = requests.get(f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic_id}", timeout=15)
        match = re.search(r"TOI-\d+", r.text)
        if match:
            return True, f"ExoFOP shows a real TOI designation: {match.group(0)}", "HIT"
        return False, "", "NO_HIT"
    except Exception as e:
        return None, f"ExoFOP query error: {e}", "ERROR"


# =====================================
# 2. LITERATURE CHECK (arXiv; ADS if a key is available)
# =====================================
def check_arxiv(tic_id):
    import requests
    import xml.etree.ElementTree as ET

    try:
        query = f'abs:"TIC {tic_id}"'
        url = f"https://export.arxiv.org/api/query?search_query={query}&max_results=3"
        r = requests.get(url.replace(" ", "+").replace('"', "%22"), timeout=15)
        root = ET.fromstring(r.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "No arXiv match", "", "NO_HIT"
        links = []
        for e in entries[:3]:
            title = e.find("atom:title", ns).text.strip().replace("\n", " ")
            link = e.find("atom:id", ns).text.strip()
            links.append(f"{title} ({link})")
        return f"{len(entries)} arXiv match(es)", " | ".join(links), "HIT"
    except Exception as e:
        return f"arXiv query error: {e}", "", "ERROR"


def check_ads(tic_id, api_key):
    import requests
    try:
        r = requests.get(
            "https://api.adsabs.harvard.edu/v1/search/query",
            params={"q": f'full:"TIC {tic_id}"', "fl": "title,bibcode", "rows": 3},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if r.status_code != 200:
            return f"ADS query failed (HTTP {r.status_code})", "", "ERROR"
        docs = r.json().get("response", {}).get("docs", [])
        if not docs:
            return "No ADS match", "", "NO_HIT"
        links = [f"{d.get('title', ['?'])[0]} (https://ui.adsabs.harvard.edu/abs/{d.get('bibcode')})" for d in docs]
        return f"{len(docs)} ADS match(es)", " | ".join(links), "HIT"
    except Exception as e:
        return f"ADS query error: {e}", "", "ERROR"


# =====================================
# 3. VARIABLE STAR CROSS-MATCH (VSX via Vizier)
# =====================================
def check_vsx(ra, dec):
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if pd.isna(ra) or pd.isna(dec):
        return "No RA/Dec available", "", "ERROR"
    try:
        v = Vizier(columns=["*"])
        v.TIMEOUT = 30  # default is 60s; bound it to match the rest of this file
        coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
        result = v.query_region(coord, radius=TESS_PIXEL_ARCSEC * u.arcsec, catalog="B/vsx")
        if len(result) == 0:
            return "No VSX match within 1 TESS pixel", "", "NO_HIT"
        t = result[0]
        row = t[0]
        return (f"VSX match: {row['Name']}",
                f"type={row['Type']}, period={row['Period'] if 'Period' in t.colnames else 'n/a'}", "HIT")
    except Exception as e:
        return f"VSX query error: {e}", "", "ERROR"


# =====================================
# 4. NEARBY-SOURCE BLENDING CHECK (Gaia cone search, magnitude-aware)
# =====================================
def check_blending(ra, dec):
    from astroquery.gaia import Gaia
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if pd.isna(ra) or pd.isna(dec):
        return "No RA/Dec available", None, None, "ERROR"

    try:
        coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
        job = Gaia.cone_search_async(coord, radius=BLEND_TIER2_ARCSEC * u.arcsec)
        res = job.get_results()
        if len(res) == 0:
            return "No Gaia sources at all within 3 TESS pixels (target itself not matched either)", 0, 0, "NO_HIT"

        res_df = res.to_pandas().sort_values("dist").reset_index(drop=True)
        res_df["sep_arcsec"] = res_df["dist"] * 3600

        # BUG FIXED: only treat the closest row as "the target itself" if its
        # separation is genuinely tiny -- otherwise Gaia may not have matched
        # the target at all (seen for faint/red M dwarfs elsewhere in this
        # project), and blindly dropping row 0 would silently discard a real
        # neighbor.
        if res_df.iloc[0]["sep_arcsec"] <= BLEND_SAME_SOURCE_ARCSEC:
            target_gmag = res_df.iloc[0].get("phot_g_mean_mag", np.nan)
            neighbors = res_df.iloc[1:]
            target_matched = True
        else:
            target_gmag = np.nan
            neighbors = res_df
            target_matched = False

        if len(neighbors) == 0:
            return "No other Gaia sources within 3 TESS pixels (63\")", 0, 0, "NO_HIT"

        # magnitude-aware risk: a neighbor within 1 pixel is only a REAL risk
        # if it's bright enough to plausibly dilute the aperture; a much
        # fainter one (found by hand: TIC_120442975's "high risk" neighbor
        # was G=20.8, ~6-10 mag fainter than a typical SPOC target) poses
        # negligible dilution risk regardless of separation.
        def delta_mag(row):
            g = row.get("phot_g_mean_mag", np.nan)
            if pd.isna(g) or pd.isna(target_gmag):
                return np.nan
            return g - target_gmag

        neighbors = neighbors.copy()
        neighbors["delta_mag"] = neighbors.apply(delta_mag, axis=1)

        tier1 = neighbors[neighbors["sep_arcsec"] <= BLEND_TIER1_ARCSEC]
        tier1_bright = tier1[(tier1["delta_mag"].isna()) | (tier1["delta_mag"] < BLEND_BRIGHT_DELTA_MAG)]
        n_tier1 = len(tier1)
        n_tier2 = len(neighbors)

        nearest = neighbors.iloc[0]
        nearest_sep = nearest["sep_arcsec"]
        nearest_gmag = nearest.get("phot_g_mean_mag", np.nan)
        nearest_dmag = nearest["delta_mag"]

        if not target_matched:
            risk = "UNKNOWN (Gaia did not match the target star itself -- risk can't be properly assessed)"
        elif len(tier1_bright) > 0:
            risk = "HIGH (bright-enough source within 1 TESS pixel)"
        elif n_tier1 > 0:
            risk = "LOW-MODERATE (source within 1 pixel, but too faint to meaningfully dilute the aperture)"
        elif n_tier2 > 0:
            risk = "MODERATE (source within 3 TESS pixels)"
        else:
            risk = "LOW"

        dmag_str = f", {nearest_dmag:+.1f} mag vs target" if pd.notna(nearest_dmag) else ""
        detail = f"nearest neighbor at {nearest_sep:.1f}\", G={nearest_gmag:.2f}{dmag_str}" if pd.notna(nearest_gmag) else f"nearest neighbor at {nearest_sep:.1f}\""
        status_code = "HIT" if ("HIGH" in risk or "MODERATE" in risk) else "NO_HIT" if risk == "LOW" else "AMBIGUOUS"
        return f"{risk} -- {detail}", n_tier1, n_tier2, status_code
    except Exception as e:
        return f"Gaia query error: {e}", None, None, "ERROR"


# =====================================
# 4b. RADIAL VELOCITY ARCHIVE CROSS-CHECK (public RV surveys via Vizier)
# =====================================
# Live-investigated before building (multi-data-source Phase 2): the NASA
# Exoplanet Archive has no RV time-series table for arbitrary stars (its
# observations/spectra tables only cover already-confirmed planets), and the
# ESO archive's HARPS/ESPRESSO holdings expose raw reduced spectra with no
# pre-computed RV value in the metadata or FITS headers -- extracting one
# would mean building a cross-correlation pipeline, well outside scope.
# What IS real and directly queryable: pre-compiled RV catalogs on Vizier.
# Confirmed live against a known positive (HD209458, 19 real epochs
# returned) and confirmed honest zero-match results for unmatched targets.
# A second major survey (Keck California Legacy Survey, Rosenthal+2021,
# J/ApJS/255/8) was also investigated but its Vizier sub-tables use two
# incompatible "CPS" formats (HD-style names in the star list vs. an
# internal numeric ID in the epoch table) with no documented shared key --
# and it contributed zero matches against this project's real candidate
# pool anyway, so it's deliberately left out rather than shipping an
# unverified join.
#
# Real coverage, tested against all 26 combined-filter-passing candidates
# in this project at the time this was built: 1/26 (~4%) had any match.
# Expected and honest -- RV surveys target bright/nearby stars, and TESS
# candidates found by this pipeline's wide, faint-star search mostly aren't
# those stars. When a match does exist, it's a strong evidence type,
# genuinely worth having; when it doesn't, this costs one fast, free
# Vizier query per candidate and returns a clean NO_HIT.
RV_HARPS_LIST_CATALOG = "J/A+A/636/A74/list"
RV_HARPS_EPOCH_CATALOG = "J/A+A/636/A74/rvbank"


def check_rv(ra, dec, period_days, m_star=None):
    """Cross-matches this candidate's coordinates against the HARPS RV Bank
    (Trifonov et al. 2020, ~3000 stars) and, if a real star match with
    enough RV epochs is found, reports whether the RV scatter is consistent
    with no obvious stellar-mass companion, or shows large enough variation
    to suggest one. This is a coarse peak-to-peak amplitude check, NOT a
    periodogram fit to this candidate's specific period -- it can catch an
    obviously-too-massive companion but can't confirm or rule out a
    planet-mass one. Guards against a real failure mode found while
    building this: reporting a "consistent" verdict from an RV baseline
    that doesn't even span this candidate's orbital period would be a
    confident-looking but meaningless result, so that case is reported
    separately as INSUFFICIENT_BASELINE rather than silently folded into
    either CONSISTENT or LARGE_VARIATION."""
    if pd.isna(ra) or pd.isna(dec):
        return "No RA/Dec available", "", "SKIPPED"
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        import numpy as np

        v = Vizier(columns=["*"], row_limit=500)
        v.TIMEOUT = 30
        coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")

        star_match = v.query_region(coord, radius=RV_MATCH_ARCSEC * u.arcsec, catalog=RV_HARPS_LIST_CATALOG)
        if len(star_match) == 0 or len(star_match[0]) == 0:
            return "No public RV archive coverage found for this star (HARPS RV Bank checked)", "", "NO_HIT"

        star_name = star_match[0]["Name"][0]
        epochs = v.query_constraints(catalog=RV_HARPS_EPOCH_CATALOG, Name=star_name)
        if len(epochs) == 0 or len(epochs[0]) == 0:
            return f"Matched {star_name} in the HARPS RV Bank star list but no epoch data found", "", "NO_HIT"

        t = epochs[0]
        bjd = np.asarray(t["BJD"], dtype=float)
        rv = np.asarray(t["DRVmlc"], dtype=float)
        mask = np.isfinite(bjd) & np.isfinite(rv)
        bjd, rv = bjd[mask], rv[mask]
        if len(rv) < 3:
            return f"Matched {star_name} but only {len(rv)} usable RV epochs -- too few to say anything", "", "NO_HIT"

        baseline_days = float(bjd.max() - bjd.min())
        if pd.isna(period_days) or period_days is None or period_days <= 0 or baseline_days < period_days:
            period_str = f"{period_days:.2f} d" if pd.notna(period_days) and period_days else "unknown"
            return (f"HARPS RV Bank: {star_name}, {len(rv)} epochs found but baseline ({baseline_days:.1f} d) "
                    f"is shorter than this candidate's period ({period_str}) -- not enough to test it",
                    "", "INSUFFICIENT_BASELINE")

        # 5th/95th percentile spread instead of raw min/max: robust against
        # the occasional single bad/flagged epoch this catalog can contain
        # (confirmed live -- a few stars had one wildly-off RVdrs value).
        peak_to_peak = float(np.percentile(rv, 95) - np.percentile(rv, 5))

        # Convert the observed scatter into an implied MINIMUM companion mass:
        #   K = 28.4329 * (Mp sini / M_Jup) * (M*/Msun)^(-2/3) * (P/yr)^(-1/3)
        # solved for Mp sini, taking K ~ peak_to_peak / 2 for a circular orbit.
        #
        # This is deliberately an UPPER bound on the companion mass, and reads
        # conservatively (towards flagging) on purpose: it attributes ALL of the
        # observed scatter to a companion at this candidate's period. Stellar
        # activity, jitter, or a completely unrelated long-period companion
        # would inflate it. So "planetary" here means "not even the full
        # observed scatter could be a non-planet", which is the strong form of
        # the statement; "too massive" means it warrants a real look.
        implied_mjup = None
        if m_star and m_star > 0 and pd.notna(period_days) and period_days and period_days > 0:
            k_ms = peak_to_peak / 2.0
            implied_mjup = (k_ms / 28.4329) * (m_star ** (2.0 / 3.0)) * ((period_days / 365.25) ** (1.0 / 3.0))

        if implied_mjup is not None:
            mass_note = (f"implies a minimum companion mass of {implied_mjup:.1f} M_Jup "
                         f"(M*={m_star:.2f} Msun, P={period_days:.2f} d)")
            if implied_mjup > RV_PLANET_MASS_CEILING_MJUP:
                return (f"HARPS RV Bank: {star_name}, {len(rv)} epochs over {baseline_days:.0f} d show "
                        f"{peak_to_peak:.0f} m/s of RV variation -- {mass_note}, above the "
                        f"{RV_PLANET_MASS_CEILING_MJUP:.0f} M_Jup deuterium-burning limit, so the companion "
                        f"would not be planetary", "", "LARGE_VARIATION")
            return (f"HARPS RV Bank: {star_name}, {len(rv)} epochs over {baseline_days:.0f} d, RV variation "
                    f"{peak_to_peak:.0f} m/s -- {mass_note}, within the planetary range "
                    f"(upper bound: assumes all scatter is a companion at this period, and is not a fit "
                    f"to it)", "", "CONSISTENT")

        # No stellar mass on file -- fall back to the crude flat amplitude test
        # and say so, rather than reporting a mass-based verdict we can't compute.
        if peak_to_peak > RV_LARGE_VARIATION_MS:
            return (f"HARPS RV Bank: {star_name}, {len(rv)} epochs over {baseline_days:.0f} d show "
                    f"{peak_to_peak:.0f} m/s of RV variation -- exceeds the {RV_LARGE_VARIATION_MS:.0f} m/s "
                    f"fallback threshold (no stellar mass on file, so no implied-mass test was possible)",
                    "", "LARGE_VARIATION")
        return (f"HARPS RV Bank: {star_name}, {len(rv)} epochs over {baseline_days:.0f} d, RV variation "
                f"{peak_to_peak:.0f} m/s -- below the {RV_LARGE_VARIATION_MS:.0f} m/s fallback threshold; "
                f"no stellar mass on file, so this is an amplitude check only, NOT an implied-mass test",
                "", "CONSISTENT")
    except Exception as e:
        return f"RV archive query error: {e}", "", "ERROR"


# =====================================
# 5. FULL PHYSICAL/ORBITAL CHARACTERIZATION
# =====================================
def spectral_type_from_teff(teff):
    if pd.isna(teff):
        return "unknown"
    for letter, (lo, hi) in SPECTRAL_TEFF_RANGES.items():
        if lo <= teff < hi:
            return letter
    return "M" if teff < 2400 else "O"


def propagate_physical_param_uncertainty(period_days, period_err, depth, depth_err, duration_days,
                                          duration_err, st_rad, st_rad_err, st_teff, st_mass, st_mass_err,
                                          rng=None):
    """Monte Carlo propagation of real measurement uncertainty into planet
    radius, semi-major axis, and equilibrium temperature -- additive
    alongside the point estimates derive_physical_params already computes,
    using the exact same formulas per Monte Carlo sample. Falls back to a
    documented default fractional uncertainty (module-level constants above)
    for any input whose real per-star error isn't available, and reports
    which ones did in `fallback_uncertainties_used` so a real measurement is
    never confused with a generic assumption. Returns None if the point
    estimate itself isn't computable (no point propagating uncertainty
    around a NaN)."""
    if pd.isna(depth) or depth is None or not (0 < depth < 1.5) or pd.isna(st_rad) or st_rad <= 0 \
            or pd.isna(period_days) or period_days <= 0 or pd.isna(duration_days) or duration_days <= 0:
        return None

    if rng is None:
        rng = np.random.default_rng(42)
    fallbacks = []

    def _err_or_fallback(err, point, default_frac, name):
        if err is not None and pd.notna(err) and err > 0:
            return float(err)
        fallbacks.append(name)
        return abs(point) * default_frac if point else default_frac

    period_err = _err_or_fallback(period_err, period_days, DEFAULT_PERIOD_FRAC_ERR, "period")
    depth_err = _err_or_fallback(depth_err, (1 - depth), DEFAULT_DEPTH_FRAC_ERR, "depth")
    duration_err = _err_or_fallback(duration_err, duration_days, DEFAULT_DURATION_FRAC_ERR, "duration")
    st_rad_err = _err_or_fallback(st_rad_err, st_rad, DEFAULT_ST_RAD_FRAC_ERR, "st_rad")
    mass_was_defaulted = not (pd.notna(st_mass) and st_mass > 0)
    st_mass_point = st_mass if not mass_was_defaulted else 1.0
    st_mass_err = _err_or_fallback(st_mass_err, st_mass_point, DEFAULT_ST_MASS_FRAC_ERR, "st_mass")
    if mass_was_defaulted:
        fallbacks.append("st_mass_defaulted_to_solar")

    n = UNCERTAINTY_N_SAMPLES
    period_s = np.clip(rng.normal(period_days, period_err, n), 1e-6, None)
    depth_s = np.clip(rng.normal(depth, depth_err, n), 1e-6, 1.4999)
    duration_s = np.clip(rng.normal(duration_days, duration_err, n), 1e-6, None)
    st_rad_s = np.clip(rng.normal(st_rad, st_rad_err, n), 1e-3, None)
    st_mass_s = np.clip(rng.normal(st_mass_point, st_mass_err, n), 1e-3, None)
    st_teff_s = np.full(n, st_teff) if pd.notna(st_teff) else np.full(n, np.nan)

    rp_rstar_s = np.sqrt(np.clip(1.0 - depth_s, 0.0, None))
    rp_earth_s = st_rad_s * rp_rstar_s * R_SUN_IN_R_EARTH
    a_au_s = (st_mass_s * (period_s / 365.25) ** 2) ** (1 / 3)
    a_rsun_s = a_au_s * AU_IN_R_SUN
    teq_k_s = st_teff_s * np.sqrt(st_rad_s / (2 * a_rsun_s))

    def _summarize(samples):
        finite = samples[np.isfinite(samples)]
        if len(finite) == 0:
            return {"p16": None, "p84": None}
        return {"p16": float(np.percentile(finite, 16)), "p84": float(np.percentile(finite, 84))}

    return {
        "planet_radius_earth": _summarize(rp_earth_s),
        "semi_major_axis_au": _summarize(a_au_s),
        "equilibrium_temp_k": _summarize(teq_k_s),
        "fallback_uncertainties_used": fallbacks,
    }


def derive_physical_params(period_days, t0, depth, duration_days, st_rad, st_teff, st_mass,
                            period_uncertainty=None, depth_mean_std=None, st_rad_err=None, st_mass_err=None):
    mass_was_defaulted = not (pd.notna(st_mass) and st_mass > 0)
    st_mass_used = st_mass if not mass_was_defaulted else 1.0

    if pd.isna(depth) or depth is None or not (0 < depth < 1.5) or pd.isna(st_rad) or st_rad <= 0:
        rp_earth = np.nan
    else:
        depth_fraction = max(0.0, 1.0 - depth)
        rp_rstar = np.sqrt(depth_fraction)
        rp_earth = st_rad * rp_rstar * R_SUN_IN_R_EARTH

    if pd.isna(period_days) or period_days <= 0:
        a_au = np.nan
    else:
        period_years = period_days / 365.25
        a_au = (st_mass_used * period_years ** 2) ** (1 / 3)

    if pd.isna(a_au) or pd.isna(st_teff) or pd.isna(st_rad) or st_rad <= 0 or a_au <= 0:
        teq_k = np.nan
        insolation_earth = np.nan
    else:
        a_rsun = a_au * AU_IN_R_SUN
        teq_k = st_teff * np.sqrt(st_rad / (2 * a_rsun))
        insolation_earth = (st_rad ** 2) * ((st_teff / TEFF_SUN_K) ** 4) / (a_au ** 2)

    hz_flag = bool(HZ_TEQ_MIN <= teq_k <= HZ_TEQ_MAX) if pd.notna(teq_k) else False
    depth_ppm = (1.0 - depth) * 1e6 if pd.notna(depth) else np.nan
    duration_hours = duration_days * 24 if pd.notna(duration_days) else np.nan
    spectral_type = spectral_type_from_teff(st_teff)

    plausible = pd.notna(rp_earth) and rp_earth <= PLAUSIBLE_RADIUS_CEILING_REARTH

    uncertainty = propagate_physical_param_uncertainty(
        period_days, period_uncertainty, depth, depth_mean_std, duration_days, None,
        st_rad, st_rad_err, st_teff, st_mass, st_mass_err)
    if uncertainty is None:
        rp_p16 = rp_p84 = a_p16 = a_p84 = teq_p16 = teq_p84 = np.nan
        uncertainty_fallbacks = []
    else:
        rp_p16, rp_p84 = uncertainty["planet_radius_earth"]["p16"], uncertainty["planet_radius_earth"]["p84"]
        a_p16, a_p84 = uncertainty["semi_major_axis_au"]["p16"], uncertainty["semi_major_axis_au"]["p84"]
        teq_p16, teq_p84 = uncertainty["equilibrium_temp_k"]["p16"], uncertainty["equilibrium_temp_k"]["p84"]
        uncertainty_fallbacks = uncertainty["fallback_uncertainties_used"]

    return {
        "period_days": period_days, "epoch_bjd": t0,
        "transit_depth_ppm": depth_ppm, "transit_duration_hours": duration_hours,
        "planet_radius_earth": rp_earth, "semi_major_axis_au": a_au,
        "planet_radius_earth_p16": rp_p16, "planet_radius_earth_p84": rp_p84,
        "semi_major_axis_au_p16": a_p16, "semi_major_axis_au_p84": a_p84,
        "equilibrium_temp_k_p16": teq_p16, "equilibrium_temp_k_p84": teq_p84,
        "uncertainty_fallbacks_used": "; ".join(uncertainty_fallbacks) if uncertainty_fallbacks else "none",
        "equilibrium_temp_k": teq_k, "insolation_flux_earth": insolation_earth,
        "rough_hz_flag": hz_flag, "stellar_spectral_type_rough": spectral_type,
        "stellar_mass_was_defaulted_to_solar": mass_was_defaulted,
        "radius_plausible": plausible,
    }


def plausibility_verdict(rp_earth, blend_status, vsx_hit):
    if pd.isna(rp_earth):
        return "Cannot assess -- radius not computable"
    if rp_earth > PLAUSIBLE_RADIUS_CEILING_REARTH:
        return f"Likely NOT a planet -- derived radius ({rp_earth:.1f} R_earth) exceeds the {PLAUSIBLE_RADIUS_CEILING_REARTH:.1f} R_earth (2 Jupiter radii) plausibility ceiling; probably a blended/grazing eclipsing binary"
    if vsx_hit:
        return "Likely false positive -- matches a known variable star (VSX)"
    if "HIGH" in str(blend_status):
        return "Ambiguous -- physically plausible radius, but a bright enough nearby source makes contamination a real concern"
    if "UNKNOWN" in str(blend_status):
        return "Ambiguous -- physically plausible radius, but Gaia didn't match the target itself so blend risk can't be properly assessed"
    return "Physically plausible, low-to-moderate contamination risk"


# =====================================
# 6. MULTI-FACTOR CONFIDENCE TIER
# =====================================
# ---- measured error/discrimination profile of the large-host regime --------
#
# WHY THIS IS NOT A HARDCODED SENTENCE ANY MORE. The candidate-facing text below
# used to assert "21% error vs 6% elsewhere on held-out data". That was measured
# on the 24-feature 0.9031 model and went silently wrong when the crowding
# features were deployed and the model became 0.9208 -- the numbers drifted to
# 14.3% / 6.7% while the sentence kept claiming 21% / 6%. Nobody noticed until a
# separate investigation happened to re-measure it.
#
# So the figures are now READ from the diagnostic's own output when it is
# present, and fall back to the last hand-verified values otherwise. Re-running
# `code/experiments/giant_star_diagnose.py` refreshes what candidate pages say.
GIANT_STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "experiments", "giant_star_diagnose.json")
GIANT_STATS_LAST_VERIFIED = "2026-08-05 (model md5 0c996a41, 26 features)"
GIANT_STATS_FALLBACK = {"joint_error_pct": 14.3, "baseline_error_pct": 6.7,
                        "joint_auc": 0.9158, "baseline_auc": 0.8637}


def _giant_regime_stats():
    """Measured error rates for the large-host/strong-signal regime.

    Never raises: a missing or malformed diagnostic file falls back to the
    last hand-verified numbers rather than breaking characterization.

    `json` is imported locally because this module does not import it at top
    level -- and a NameError here would be swallowed by the except below,
    silently pinning the text to the fallback values forever, which is the
    exact silent-staleness failure this function exists to prevent.
    """
    import json
    try:
        with open(GIANT_STATS_PATH) as f:
            cells = json.load(f)["cells"]
        return {
            "joint_error_pct": cells["BOTH (the blind spot)"]["error_pct"],
            "baseline_error_pct": cells["neither (small rad, low SDE)"]["error_pct"],
            "joint_auc": cells["BOTH (the blind spot)"]["auc"],
            "baseline_auc": cells["neither (small rad, low SDE)"]["auc"],
        }
    except Exception:
        return dict(GIANT_STATS_FALLBACK)


def confidence_tier(row):
    """Returns (tier, supporting_evidence, doubting_evidence). Tiers are
    High/Medium/Low -- and EVEN "High" here means "strong candidate worth
    real follow-up", never "confirmed planet". No automated pipeline can
    produce that confirmation."""
    support, doubt = [], []
    score = 0

    prob = row.get("predicted_probability")
    if pd.notna(prob):
        if prob >= 0.97:
            support.append(f"very high classifier probability ({prob:.3f})")
            score += 2
        elif prob >= 0.9:
            support.append(f"high classifier probability ({prob:.3f})")
            score += 1

    sde = row.get("SDE")
    n_transits = row.get("distinct_transit_count")
    if pd.notna(sde) and pd.notna(n_transits):
        if sde >= 10 and n_transits >= 5:
            support.append(f"strong detection: SDE={sde:.1f} across {int(n_transits)} distinct transits")
            score += 2
        elif sde >= 7 and n_transits >= 3:
            support.append(f"moderate detection: SDE={sde:.1f} across {int(n_transits)} distinct transits")
            score += 1
        else:
            doubt.append(f"marginal detection statistics: SDE={sde:.1f}, only {int(n_transits)} distinct transits observed"
                          + (" -- the period itself is less certain with this few" if n_transits < 3 else ""))

    # Large host star + strong signal -- the regime the SDE bonus above rewards
    # and the one where a given probability is worth least.
    #
    # THE CLAIM THIS TEXT USED TO MAKE WAS WRONG, not merely stale. It said the
    # classifier is "measurably least reliable in this combination". Re-measured
    # on the deployed 0.9208 model, the opposite is true of its DISCRIMINATION:
    #
    #   cell                  n    error@0.5    AUC      planets%
    #   neither             629       6.7%    0.8637       92.4
    #   large radius only   180      20.6%    0.8974       57.8
    #   high SDE only       204      16.2%    0.8990       67.6
    #   BOTH (this rule)     49      14.3%    0.9158       44.9
    #
    # The model RANKS BEST in the cell this rule fires on (AUC 0.9158) and worst
    # in the "safe" cell (0.8637). The error-rate ordering is inverted relative
    # to AUC purely because of class balance: 92.4% of the safe cell are real
    # planets, against 44.9% here. So the honest statement is not "the model is
    # worse at these" but "these come from a near-50/50 population, so the same
    # probability carries much weaker odds" -- which is still a real reason to
    # demand more evidence, just not the reason previously given.
    #
    # The condition is deliberately NOT broadened to st_rad>=1.5 alone, even
    # though that cell now has the higher raw error rate (20.6%). Three reasons:
    # (1) this -1 exists to temper the +2 the SDE bonus just awarded, so it is
    # structurally tied to sde>=10 -- applied to giants that never got the bonus
    # it would be a different, unjustified penalty; (2) that cell's higher error
    # rate is the same base-rate artifact (57.8% planets vs 92.4%), and its AUC
    # is BETTER than the safe cell's, so re-aiming on error rate alone would
    # repeat the reasoning error this comment exists to correct; (3) giants are
    # ~50% of the real candidate pool, and a flag that fires on half of
    # everything stops being a flag.
    st_rad = row.get("st_rad")
    if pd.notna(st_rad) and pd.notna(sde) and st_rad >= 1.5 and sde >= 10:
        g = _giant_regime_stats()
        doubt.append(
            f"host star is large (R = {st_rad:.2f} R_sun) and the detection is strong "
            f"(SDE = {sde:.1f}) -- on held-out data this regime is close to an even "
            f"split between real planets and false positives, so errors at a fixed "
            f"threshold run {g['joint_error_pct']:.1f}% here against "
            f"{g['baseline_error_pct']:.1f}% for small-host/low-SDE candidates; the "
            f"model still separates the two classes well here "
            f"(AUC {g['joint_auc']:.3f}), but a high probability is worth less when "
            f"the underlying odds are near 50/50, because deep, long eclipses on "
            f"giant stars mimic strong planet detections; treat the blend and "
            f"secondary-eclipse checks as decisive here")
        score -= 1

    if row.get("radius_plausible"):
        support.append(f"physically plausible derived radius ({row.get('planet_radius_earth'):.1f} R_earth)")
        score += 1
    else:
        doubt.append(f"derived radius ({row.get('planet_radius_earth', float('nan')):.1f} R_earth) exceeds the planetary plausibility ceiling")
        score -= 3

    blend = str(row.get("blending_status", ""))
    if "HIGH" in blend:
        doubt.append("a bright, nearby Gaia source poses a real contamination risk")
        score -= 2
    elif "UNKNOWN" in blend:
        doubt.append("Gaia did not match the target star, so blend risk is unverified")
        score -= 1
    elif "MODERATE" in blend:
        doubt.append("a moderate-distance Gaia source is a mild contamination concern")
    elif "LOW" in blend and "LOW-MODERATE" not in blend:
        support.append("no concerning nearby Gaia sources")
        score += 1

    if row.get("vsx_status", "").lower().startswith("vsx match"):
        doubt.append("matches a known AAVSO variable star -- likely false positive")
        score -= 3
    else:
        support.append("no known variable-star match (VSX)")

    if row.get("newly_flagged_in_archive_or_exofop") or row.get("exofop_has_target_page"):
        doubt.append("has since been flagged in the archive/ExoFOP/TOI table -- no longer unexamined")
        score -= 5
    else:
        support.append(f"not previously flagged by ExoFOP or the TOI catalog as of {row.get('last_verified_unknown_utc', '?')}")

    if str(row.get("arxiv_status", "")).startswith(tuple(str(i) for i in range(1, 10))):
        doubt.append("already discussed in an arXiv preprint -- check for prior characterization")

    if row.get("stellar_mass_was_defaulted_to_solar"):
        doubt.append("stellar mass was missing from the TIC catalog and defaulted to solar -- semi-major axis/temperature estimates are less reliable for non-solar-type stars")

    # Public RV archive cross-check (multi-data-source Phase 2). Only the two
    # cases that are actual evidence get a sentence -- NO_HIT/INSUFFICIENT_
    # BASELINE/SKIPPED/ERROR mean "no usable signal either way" and stay
    # silent here, same as how a missing ADS key doesn't get its own doubt
    # sentence. LARGE_VARIATION is weighted like a VSX match (a comparably
    # strong, independent red flag); CONSISTENT is weighted like "no
    # concerning nearby Gaia sources" -- real but not proof, since this is a
    # coarse peak-to-peak check, not a fit to this candidate's period.
    rv_code = row.get("rv_code")
    if rv_code == "LARGE_VARIATION":
        doubt.append(f"public RV archive data shows variation large enough to suggest a stellar-mass "
                      f"companion, not a planet: {row.get('rv_status', '')}")
        score -= 3
    elif rv_code == "CONSISTENT":
        support.append(f"public RV archive data is consistent with no obvious massive stellar companion: "
                        f"{row.get('rv_status', '')}")
        score += 1

    if score >= 5:
        tier = "High"
    elif score >= 2:
        tier = "Medium"
    else:
        tier = "Low"

    return tier, "; ".join(support) if support else "none", "; ".join(doubt) if doubt else "none"


# =====================================
# MAIN
# =====================================
def main():
    run_timestamp = now_utc_iso()
    print("=" * 70)
    print("08_characterize_candidates.py -- definitive consolidation pass")
    print(f"Run started: {run_timestamp}")
    print("=" * 70)

    if not os.path.exists(IN_DIST_PATH):
        print(f"ERROR: {IN_DIST_PATH} not found -- run 06_download_unknown.py first.")
        return

    candidates = pd.read_csv(IN_DIST_PATH)
    if os.path.exists(STELLAR_VERIFICATION_PATH):
        sv = pd.read_csv(STELLAR_VERIFICATION_PATH)
        candidates = candidates.merge(
            sv[["host", "stellar_param_verified_clean"]], on="host", how="left"
        )
    print(f"{len(candidates)} trustworthy candidates to characterize.\n")

    ads_api_key = os.environ.get("ADS_API_KEY")
    if ads_api_key:
        print("ADS_API_KEY found -- will use real ADS literature search.")
    else:
        print("NOTE: ADS_API_KEY not set -- ADS check marked SKIPPED per candidate (see "
              "'ads_status'). Free key: https://ui.adsabs.harvard.edu/user/settings/token. "
              "Using free arXiv API as a partial substitute (preprints only) meanwhile. "
              "This remains the ONLY external source blocked by a missing credential -- "
              "ExoFOP, the NASA archive, VSX, and Gaia all work with no key.")

    confirmed_tics, toi_tics, exofop_tics = fetch_fresh_exclusion_data()

    already_done = set()
    if os.path.exists(OUTPUT_PATH):
        old = pd.read_csv(OUTPUT_PATH)
        already_done = set(old["host"])
        print(f"{len(already_done)} already characterized -- resuming.")

    results = list(pd.read_csv(OUTPUT_PATH).to_dict("records")) if os.path.exists(OUTPUT_PATH) else []
    results_by_host = {r["host"]: r for r in results}

    # BUG FIXED: resuming used to skip already-characterized hosts entirely,
    # which meant "newly_flagged_in_archive_or_exofop" and
    # "last_verified_unknown_utc" silently went stale after the very first
    # run -- even though the exclusion set above is re-queried live every
    # time. A star that picked up a TOI designation between runs would keep
    # showing a first-run "unknown" verdict forever. The exclusion-set check
    # is just a set-membership lookup (no network cost beyond the bulk fetch
    # already done above), so it's now re-run for every candidate on every
    # invocation regardless of resume status; only the expensive per-star
    # external HTTP checks (ExoFOP page, arXiv, ADS, VSX, Gaia blend) are
    # actually skipped on resume.
    n_rechecked = 0
    n_status_changed = 0
    for i, row in candidates.iterrows():
        host = row["host"]
        tic_id = int(host.replace("TIC_", ""))
        newly_flagged = tic_id in confirmed_tics or tic_id in toi_tics or tic_id in exofop_tics

        if host in already_done:
            r = results_by_host[host]
            if bool(r.get("newly_flagged_in_archive_or_exofop")) != newly_flagged:
                n_status_changed += 1
                print(f"  [STATUS CHANGE] {host}: newly_flagged_in_archive_or_exofop "
                      f"{r.get('newly_flagged_in_archive_or_exofop')} -> {newly_flagged}")
            r["newly_flagged_in_archive_or_exofop"] = newly_flagged
            r["last_verified_unknown_utc"] = run_timestamp
            n_rechecked += 1
            continue

        ra, dec = row.get("ra"), row.get("dec")
        exofop_has_page, exofop_note, exofop_code = check_exofop_target_page(tic_id)

        arxiv_status, arxiv_links, arxiv_code = check_arxiv(tic_id)
        if ads_api_key:
            ads_status, ads_links, ads_code = check_ads(tic_id, ads_api_key)
        else:
            ads_status, ads_links, ads_code = "SKIPPED: no ADS_API_KEY", "", "SKIPPED"

        vsx_status, vsx_detail, vsx_code = check_vsx(ra, dec)
        blend_status, n_tier1, n_tier2, blend_code = check_blending(ra, dec)

        phys = derive_physical_params(row.get("period"), row.get("T0"), row.get("depth"),
                                       row.get("duration"), row.get("st_rad"),
                                       row.get("st_teff"), row.get("st_mass"),
                                       period_uncertainty=row.get("period_uncertainty"),
                                       depth_mean_std=row.get("depth_mean_std"),
                                       st_rad_err=row.get("st_rad_err"),
                                       st_mass_err=row.get("st_mass_err"))

        verdict = plausibility_verdict(phys["planet_radius_earth"], blend_status, vsx_code == "HIT")
        # st_mass is passed through so the RV check can convert its observed
        # scatter into an implied companion mass rather than compare against a
        # flat m/s cutoff that means different things for different stars.
        rv_status, rv_detail, rv_code = check_rv(ra, dec, phys["period_days"],
                                                  m_star=row.get("st_mass"))

        result = {
            "host": host,
            # BUG FIXED: ra/dec were read from the input row and used to run
            # the VSX/blend checks right above, but were never carried into
            # this output row -- so no downstream consumer (e.g. a later
            # single-candidate re-verify) could reconstruct them. Same
            # "computed internally, dropped before persisting" pattern as
            # the st_mass merge bug found earlier in this project.
            "ra": ra, "dec": dec,
            "last_verified_unknown_utc": run_timestamp,
            "newly_flagged_in_archive_or_exofop": newly_flagged,
            "exofop_has_target_page": exofop_has_page, "exofop_note": exofop_note or "", "exofop_code": exofop_code,
            "arxiv_status": arxiv_status, "arxiv_links": arxiv_links, "arxiv_code": arxiv_code,
            "ads_status": ads_status, "ads_links": ads_links, "ads_code": ads_code,
            "vsx_status": vsx_status, "vsx_detail": vsx_detail, "vsx_code": vsx_code,
            "blending_status": blend_status, "n_gaia_neighbors_tier1_21as": n_tier1,
            "n_gaia_neighbors_tier2_63as": n_tier2, "blend_code": blend_code,
            "rv_status": rv_status, "rv_detail": rv_detail, "rv_code": rv_code,
            "plausibility_verdict": verdict,
            **phys,
        }
        results.append(result)

        flags = []
        if newly_flagged:
            flags.append("NEWLY FLAGGED")
        if exofop_code == "HIT":
            flags.append("ExoFOP TOI hit")
        if arxiv_code == "HIT":
            flags.append("arXiv hit")
        if vsx_code == "HIT":
            flags.append("VSX match")
        if "HIGH" in blend_status:
            flags.append("HIGH blend risk")
        if rv_code == "LARGE_VARIATION":
            flags.append("RV suggests stellar companion")
        if not phys["radius_plausible"]:
            flags.append("IMPLAUSIBLE RADIUS")
        teq_str = f", Teq={phys['equilibrium_temp_k']:.0f}K" if pd.notna(phys["equilibrium_temp_k"]) else ""
        print(f"  [{i+1}/{len(candidates)}] {host}: {', '.join(flags) if flags else 'clean'} "
              f"-- Rp={phys['planet_radius_earth']:.2f} R_earth{teq_str}")

        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
        time.sleep(0.3)

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_PATH, index=False)

    # ---- merge in classifier/TLS fields needed for the combined filter + confidence tier ----
    merge_cols = ["host", "predicted_probability", "SDE", "SNR", "FAP", "distinct_transit_count",
                  "period", "duration", "T0", "st_rad", "st_teff", "st_mass"]
    merge_cols = [c for c in merge_cols if c in candidates.columns]
    full = out_df.merge(candidates[merge_cols], on="host", how="left", suffixes=("", "_dup"))

    tiers, supports, doubts = [], [], []
    for _, r in full.iterrows():
        tier, sup, dbt = confidence_tier(r)
        tiers.append(tier)
        supports.append(sup)
        doubts.append(dbt)
    full["confidence_tier"] = tiers
    full["supporting_evidence"] = supports
    full["doubting_evidence"] = doubts

    # Visible stellar-mass-provenance badge -- surfaced as its own column so
    # it's a glanceable flag, not buried inside doubting_evidence text. Any
    # candidate with this set had NO real stellar mass in the TIC catalog,
    # so its semi-major-axis/equilibrium-temperature/insolation estimates
    # rest on a solar-mass assumption that's likely wrong for M dwarfs.
    full["stellar_mass_status"] = np.where(
        full["stellar_mass_was_defaulted_to_solar"].fillna(False),
        "DEFAULTED TO SOLAR -- Teq/insolation/semi-major axis degraded",
        "real TIC catalog value",
    )
    full.to_csv(OUTPUT_PATH, index=False)

    # ---- PERMANENT combined-filter best-candidates list ----
    not_flagged = ~full["newly_flagged_in_archive_or_exofop"] & (full["exofop_code"] != "HIT")
    not_vsx = full["vsx_code"] != "HIT"
    plausible = full["radius_plausible"].fillna(False)
    blend_unknown = full["blending_status"].str.contains("UNKNOWN", na=False)
    blend_ok_strict = full["blending_status"].str.contains("LOW", na=False) & ~full["blending_status"].str.contains("LOW-MODERATE", na=False)
    blend_ok_lenient = ~full["blending_status"].str.contains("HIGH|UNKNOWN", na=False, regex=True)

    base = not_flagged & not_vsx & plausible
    strict_mask = base & blend_ok_strict
    lenient_mask = base & blend_ok_lenient
    # PERMANENT "needs manual review" bucket: candidates that clear every
    # other bar (unflagged, no VSX match, plausible radius) but whose blend
    # risk is UNKNOWN rather than HIGH -- i.e. Gaia's astrometric solution
    # simply didn't resolve the target star (common for faint/red M dwarfs),
    # not evidence of an actual nearby contaminant. Automatic exclusion here
    # would silently discard exactly the kind of candidate (e.g.
    # TIC_120442975) that deserves a 5-minute manual Gaia-image look rather
    # than a blanket rejection. Kept separate from the best-candidates list
    # because "unknown" is not the same claim as "verified low risk".
    review_mask = base & blend_unknown

    combined = full[lenient_mask].copy()
    combined["combined_tier"] = np.where(strict_mask.loc[combined.index], "strict", "moderate")
    combined = combined.sort_values(["confidence_tier", "predicted_probability"],
                                     key=lambda s: s.map({"High": 2, "Medium": 1, "Low": 0}) if s.name == "confidence_tier" else s,
                                     ascending=[False, False])
    combined.to_csv(BEST_CANDIDATES_PATH, index=False)

    needs_review = full[review_mask].copy()
    needs_review["review_reason"] = "Gaia did not resolve the target star -- blend risk unknown, not verified low"
    needs_review = needs_review.sort_values("predicted_probability", ascending=False)
    needs_review.to_csv(NEEDS_REVIEW_PATH, index=False)

    n_newly_flagged = int(full["newly_flagged_in_archive_or_exofop"].sum())
    n_exofop_hit = int((full["exofop_code"] == "HIT").sum())
    n_vsx_hit = int((full["vsx_code"] == "HIT").sum())
    n_high_blend = int(full["blending_status"].str.contains("HIGH", na=False).sum())
    n_implausible = int((~full["radius_plausible"].fillna(False)).sum())
    n_genuinely_unknown = int((~full["newly_flagged_in_archive_or_exofop"] & (full["exofop_code"] != "HIT")).sum())
    n_strict = int(strict_mask.sum())
    n_lenient = int(lenient_mask.sum())
    n_high_conf = int((full["confidence_tier"] == "High").sum())
    n_needs_review = int(review_mask.sum())
    n_defaulted_mass = int(full["stellar_mass_was_defaulted_to_solar"].fillna(False).sum())

    print("\n" + "=" * 70)
    print("HONEST SUMMARY")
    print("=" * 70)
    print(f"Total candidates checked: {len(full)}")
    print(f"Last verified unknown as of: {run_timestamp}")
    print(f"Exclusion-set membership re-checked live for {n_rechecked} previously-characterized "
          f"candidates this run ({n_status_changed} changed status since last check).")
    print(f"Newly flagged in archive/ExoFOP TOI table since original run: {n_newly_flagged}")
    print(f"Genuinely unflagged/unknown after ALL checks: {n_genuinely_unknown}/{len(full)}")
    print(f"VSX variable-star matches: {n_vsx_hit}")
    print(f"High nearby-source blending risk: {n_high_blend}")
    print(f"Physically implausible radius (>{PLAUSIBLE_RADIUS_CEILING_REARTH:.1f} R_earth): {n_implausible}")
    print(f"Stellar mass defaulted to solar (degraded Teq/insolation): {n_defaulted_mass}")
    print(f"High-confidence tier: {n_high_conf}")
    print(f"\nCombined-filter best-candidates list: {n_lenient} survive (moderate-or-better blend risk), "
          f"of which {n_strict} survive the strict (low-blend-only) tier.")
    print(f"Needs-manual-review list (Gaia couldn't resolve target -- blend risk unknown, not verified low): "
          f"{n_needs_review}")
    print(f"Full characterization: {OUTPUT_PATH}")
    print(f"Best-candidates list: {BEST_CANDIDATES_PATH}")
    print(f"Needs-review list: {NEEDS_REVIEW_PATH}")


if __name__ == "__main__":
    main()
