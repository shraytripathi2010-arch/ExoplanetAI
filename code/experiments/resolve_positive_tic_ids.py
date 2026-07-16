"""
resolve_positive_tic_ids.py -- resolves a real TIC ID for each positive-
class training star (4,336 of them have none -- only a name), using the
star's own real ra/dec (already in training.csv from the confirmed-planet
catalog) cross-matched against the TIC catalog by coordinate, not name.

Name-based resolution was tested and rejected: querying by name (e.g.
"16 Cyg B") returned up to 58 candidate matches for a single star, with no
principled way to pick the right one. Coordinate-based cross-match against
the star's own precise position is far more reliable -- tested on a
40-star random sample: 18/40 had exactly one TIC catalog entry within
10 arcsec, and of the 21 with multiple entries, nearly all had a nearest
match separated from the 2nd-nearest by a wide margin (e.g. 0.03" vs
5.72"), meaning "nearest match" was still clearly correct -- only 1-2
genuinely borderline cases (nearest and 2nd-nearest close together) should
be excluded rather than guessed.

Decision rule (documented, not silent): a match is CONFIDENT if the
nearest TIC entry is within RESOLVE_RADIUS_ARCSEC AND either it's the only
match, or it's at least SEPARATION_RATIO times closer than the 2nd-nearest
entry. Anything else is marked unresolved and excluded, not guessed.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from tqdm import tqdm

TRAINING_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                             "data", "training_dataset", "training.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "positive_class_tic_ids.csv")

RESOLVE_RADIUS_ARCSEC = 10
SEPARATION_RATIO = 3.0  # nearest must be at least this many times closer than 2nd-nearest


def resolve_one(ra, dec):
    from astroquery.mast import Catalogs
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    r = Catalogs.query_region(coord, radius=RESOLVE_RADIUS_ARCSEC * u.arcsec, catalog="Tic")
    if len(r) == 0:
        return None, "no_match", None
    rdf = r.to_pandas().sort_values("dstArcSec").reset_index(drop=True)
    nearest_sep = rdf.iloc[0]["dstArcSec"]
    if len(rdf) == 1:
        return int(rdf.iloc[0]["ID"]), "resolved_unique", nearest_sep
    second_sep = rdf.iloc[1]["dstArcSec"]
    if second_sep >= SEPARATION_RATIO * max(nearest_sep, 0.01):
        return int(rdf.iloc[0]["ID"]), "resolved_clear_nearest", nearest_sep
    return None, "ambiguous", nearest_sep


def main():
    df = pd.read_csv(TRAINING_CSV)
    pos = df[df["label"] == 1].dropna(subset=["ra", "dec"])
    print(f"{len(pos)} positive-class rows with ra/dec to resolve...")

    results = []
    for _, row in tqdm(pos.iterrows(), total=len(pos), desc="Resolving TIC IDs"):
        tic_id, status, sep = resolve_one(row["ra"], row["dec"])
        results.append({"host": row["host"], "tic_id": tic_id, "resolution_status": status,
                         "nearest_sep_arcsec": sep})

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_CSV, index=False)
    print(out["resolution_status"].value_counts())
    n_resolved = out["tic_id"].notna().sum()
    print(f"\n{n_resolved}/{len(out)} positive-class stars resolved to a confident TIC ID.")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
