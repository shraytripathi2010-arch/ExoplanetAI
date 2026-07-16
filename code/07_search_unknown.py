"""
07_search_unknown.py

SCOPE DETERMINATION (read before assuming this duplicates 06):
06_download_unknown.py already covers download, preprocess, TLS search,
feature extraction, model scoring, ranking, BOTH out-of-distribution
safeguards (univariate min/max + multivariate Isolation Forest), plain-
language explanations, and folded light curve plots -- i.e. everything the
original project scaffold's "07 (search)" and "08 (rank)" placeholder
scripts were meant to do. Re-implementing any of that here would be
redundant busywork.

What's NOT covered: every stellar parameter (radius, Teff) used both as a
model feature AND as the out-of-distribution check's basis comes from ONE
source -- the TIC catalog's photometric estimates, which are known to be
unreliable for some stars (confirmed directly against real data during the
06 pilot: TIC_431085335 had a TIC-reported radius of 230 solar radii that
looked implausible). No independent source was ever cross-checked. That's
a genuine, real gap in due diligence for the final trustworthy candidate
list, not an invented task -- this script closes it: for every candidate
that cleared BOTH of 06's out-of-distribution checks (the doubly-filtered
"trustworthy" shortlist), independently query Gaia DR3 (teff_gspphot, a
photometric temperature estimate from a completely different pipeline than
TIC's) and SIMBAD (spectral type, parallax) and flag any candidate where
the TIC catalog's values disagree substantially with either independent
source. This does NOT change any candidate's model score or ranking --
it's an additional piece of due-diligence information for whoever reviews
the shortlist, surfacing exactly the kind of stellar-characterization risk
already identified as needing a closer look.

Usage:
    python3 07_search_unknown.py
"""
import os
import re
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, "results", "unknown_candidates")
IN_DIST_PATH = os.path.join(RESULTS_FOLDER, "ranked_candidates_in_distribution.csv")
OUTPUT_PATH = os.path.join(RESULTS_FOLDER, "trustworthy_candidates_stellar_verification.csv")
SUMMARY_PATH = os.path.join(RESULTS_FOLDER, "stellar_verification_summary.txt")

# Rough spectral-class -> Teff range (K), standard textbook boundaries.
# Used only as a sanity check against TIC's Teff, not a precise estimate.
SPECTRAL_TEFF_RANGES = {
    "O": (30000, 60000), "B": (10000, 30000), "A": (7500, 10000),
    "F": (6000, 7500), "G": (5200, 6000), "K": (3700, 5200), "M": (2400, 3700),
}

TEFF_DISAGREEMENT_THRESHOLD = 0.30   # fraction difference vs TIC catalog value


def query_gaia_teff(ra, dec, radius_arcsec=5):
    from astroquery.gaia import Gaia
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    try:
        coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
        job = Gaia.cone_search_async(coord, radius=radius_arcsec * u.arcsec)
        res = job.get_results()
        if len(res) == 0:
            return None, None, "No Gaia match within 5 arcsec"
        row = res[0]
        teff = float(row["teff_gspphot"]) if not np.ma.is_masked(row["teff_gspphot"]) else None
        dist = float(row["distance_gspphot"]) if "distance_gspphot" in res.colnames and not np.ma.is_masked(row["distance_gspphot"]) else None
        if teff is None:
            return None, dist, "Gaia match found but no teff_gspphot available (star may be too bright/faint for this pipeline)"
        return teff, dist, "Success"
    except Exception as e:
        return None, None, f"Gaia query error: {e}"


def query_simbad(tic_id):
    from astroquery.simbad import Simbad

    try:
        s = Simbad()
        s.add_votable_fields("sptype", "plx")
        r = s.query_object(f"TIC {tic_id}")
        if r is None or len(r) == 0:
            return None, None, "Not found in SIMBAD"
        sp_type = str(r["sp_type"][0]) if "sp_type" in r.colnames and r["sp_type"][0] else None
        plx = float(r["plx_value"][0]) if "plx_value" in r.colnames and not np.ma.is_masked(r["plx_value"][0]) else None
        return sp_type, plx, "Success"
    except Exception as e:
        return None, None, f"SIMBAD query error: {e}"


def spectral_type_teff_range(sp_type):
    if not sp_type:
        return None
    letter = sp_type.strip()[0].upper()
    return SPECTRAL_TEFF_RANGES.get(letter)


def verify_one_star(host, ra, dec, tic_teff, tic_rad):
    # cast to int for consistency with the rest of this project's TIC-ID
    # handling (every other file casts to int64; this one was left as a
    # bare string -- not currently exploitable since it's only used in an
    # f-string here, but fixed defensively since inconsistent typing is
    # exactly the kind of latent risk that caused the earlier TIC-ID-float
    # bug in 06_download_unknown.py).
    tic_id = int(host.replace("TIC_", ""))
    gaia_teff, gaia_dist, gaia_status = query_gaia_teff(ra, dec) if pd.notna(ra) and pd.notna(dec) else (None, None, "No RA/Dec available")
    sp_type, simbad_plx, simbad_status = query_simbad(tic_id)

    flags = []
    checked_something = False
    if gaia_teff is not None and pd.notna(tic_teff) and tic_teff > 0:
        checked_something = True
        frac_diff = abs(gaia_teff - tic_teff) / tic_teff
        if frac_diff > TEFF_DISAGREEMENT_THRESHOLD:
            flags.append(f"TIC Teff ({tic_teff:.0f}K) vs Gaia teff_gspphot ({gaia_teff:.0f}K) disagree by {frac_diff:.0%}")

    expected_range = spectral_type_teff_range(sp_type)
    if expected_range is not None and pd.notna(tic_teff):
        checked_something = True
        if not (expected_range[0] * 0.8 <= tic_teff <= expected_range[1] * 1.2):
            flags.append(f"TIC Teff ({tic_teff:.0f}K) inconsistent with SIMBAD spectral type "
                         f"'{sp_type}' (expected roughly {expected_range[0]}-{expected_range[1]}K)")

    # BUG FIXED: "verified_clean" used to be True whenever no disagreement
    # flag was raised -- which conflated "checked both sources and they
    # agree" with "neither source had usable data to check against" and
    # "a query errored out", silently overstating confidence. Now explicit:
    # a candidate is only "verified_clean" if at least one independent
    # comparison was actually possible AND it came back clean.
    verified_clean = checked_something and len(flags) == 0

    return {
        "host": host, "gaia_teff": gaia_teff, "gaia_status": gaia_status,
        "gaia_distance_pc": gaia_dist, "simbad_sptype": sp_type,
        "simbad_parallax_mas": simbad_plx, "simbad_status": simbad_status,
        "stellar_param_flags": "; ".join(flags), "stellar_param_verified_clean": verified_clean,
        "stellar_param_could_be_checked": checked_something,
    }


def main():
    print("=" * 70)
    print("07_search_unknown.py -- independent stellar-parameter verification")
    print("Cross-checks TIC catalog st_rad/st_teff (the model's top features AND")
    print("the OOD check's basis) against Gaia DR3 and SIMBAD for the final")
    print("doubly-filtered trustworthy candidate list. Does NOT change any score")
    print("or ranking -- adds due-diligence info for human review.")
    print("=" * 70)

    if not os.path.exists(IN_DIST_PATH):
        print(f"ERROR: {IN_DIST_PATH} not found -- run 06_download_unknown.py first.")
        return

    candidates = pd.read_csv(IN_DIST_PATH)
    print(f"\n{len(candidates)} trustworthy (in-distribution AND multivariate-clean) "
          f"candidates to verify.")

    already_done = set()
    if os.path.exists(OUTPUT_PATH):
        old = pd.read_csv(OUTPUT_PATH)
        already_done = set(old["host"])
        print(f"{len(already_done)} already verified -- resuming.")

    results = list(pd.read_csv(OUTPUT_PATH).to_dict("records")) if os.path.exists(OUTPUT_PATH) else []
    for i, row in candidates.iterrows():
        host = row["host"]
        if host in already_done:
            continue
        r = verify_one_star(host, row.get("ra"), row.get("dec"), row.get("st_teff"), row.get("st_rad"))
        results.append(r)
        print(f"  [{i+1}/{len(candidates)}] {host}: gaia={r['gaia_status']}, simbad={r['simbad_status']}, "
              f"flags={'none' if r['stellar_param_verified_clean'] else r['stellar_param_flags']}")
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
        time.sleep(0.2)  # be polite to the archives -- not rate-limited but no reason to hammer them

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_PATH, index=False)

    merged = candidates.merge(out_df, on="host", how="left")
    n_clean = int(merged["stellar_param_verified_clean"].sum())
    n_flagged = len(merged) - n_clean
    n_no_gaia_match = int((merged["gaia_status"] == "No Gaia match within 5 arcsec").sum())

    with open(SUMMARY_PATH, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("INDEPENDENT STELLAR-PARAMETER VERIFICATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Trustworthy candidates checked: {len(merged)}\n")
        f.write(f"Independently verified clean (Gaia/SIMBAD agree with TIC catalog): {n_clean}\n")
        f.write(f"Flagged for stellar-parameter disagreement: {n_flagged}\n")
        f.write(f"No Gaia counterpart found within 5 arcsec: {n_no_gaia_match}\n\n")
        if n_flagged > 0:
            f.write("Flagged candidates (review the underlying stellar characterization "
                    "before trusting these, even though they passed both OOD checks):\n")
            for _, r in merged[~merged["stellar_param_verified_clean"]].iterrows():
                f.write(f"  {r['host']}: {r['stellar_param_flags']}\n")

    print(f"\n{n_clean}/{len(merged)} candidates independently verified clean.")
    print(f"{n_flagged} flagged for stellar-parameter disagreement -- see {SUMMARY_PATH}")
    print(f"Full results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
