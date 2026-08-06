"""stellar_density_fetch.py -- pull the TIC fields the pipeline already
receives and throws away.

PART 0 FINDING, verified live rather than assumed: the existing batch query in
06_download_unknown.py is

    Catalogs.query_criteria(catalog="Tic", ID=chunk)

which returns a 125-COLUMN row per star. The next line selects eight of them:

    r[["ID", "ra", "dec", "rad", "e_rad", "mass", "e_mass", "Teff"]]

Among the 117 discarded columns are `logg`, `e_logg`, `rho`, `e_rho` and `lum`.
This is exactly the pattern the crowding investigation hit with
contratio/numcont: the data was already arriving and was simply unused. No new
fetch mechanism is needed, only a wider column selection.

THE FEATURE THAT MATTERS IS NOT rho ITSELF -- IT IS THE DENSITY RATIO

For a central transit, a/R* = P / (pi * T14), and Kepler's third law gives
rho_star = 3*pi/(G P^2) * (a/R*)^3. Substituting:

    rho_circ = 3 P / (G pi^2 T14^3)

That is the stellar density IMPLIED by the observed period and duration if the
orbit is circular and central. Comparing it to the star's CATALOGUED density is
a standard false-positive discriminator -- it is the core of how vespa and
similar validation tools reason. A grazing eclipsing binary, or a blend diluted
by a third star, produces a duration inconsistent with the host's true density,
so the ratio departs from 1.

Why this is not already in the model, checked against the deployed feature
list: the 26 production features include `period` and `duration` (the
NUMERATOR ingredients) but the denominator needs rho_star, which requires MASS
and radius. `st_mass` is NOT a production feature -- only `st_rad` and
`st_teff` are. So rho_star is not derivable from what the model already sees.

WHAT THIS CANNOT DO, stated up front. The giant-star investigation found the
giant/dwarf AUC gap is near zero -- the model already RANKS giants correctly
and the deficit was calibration/base-rate, not information. A density feature
is therefore not a fix for the giant-star issue, and is not proposed as one.

THE CTL TRAP IS CHECKED, NOT ASSUMED. contratio/numcont turned out to be
populated only for Candidate Target List stars, so availability itself scored
AUC 0.3775 and only 37.5% of unknown candidates had it. `priority`, `contratio`
and `numcont` are fetched here alongside rho/logg purely so the same test can
be run on these fields.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
POS_IDS = os.path.join(SCRIPT_DIR, "positive_class_tic_ids.csv")
OUT = os.path.join(SCRIPT_DIR, "stellar_density_raw.csv")

WANT = ["ID", "rad", "e_rad", "mass", "e_mass", "Teff", "logg", "e_logg",
        "rho", "e_rho", "lum", "lumclass", "priority", "contratio", "numcont"]
CHUNK = 400


def host_to_tic():
    """Map every training host to a TIC id.

    Negatives are natively TIC_-keyed. Positives are name-keyed and were
    resolved by coordinate cross-match in the crowding work; that mapping is
    reused rather than re-resolved.
    """
    tr = pd.read_csv(TRAINING)
    m = {}
    h = tr["host"].astype(str)
    for x in h[h.str.startswith("TIC_")]:
        try:
            m[x] = int(x.split("_")[1])
        except Exception:
            pass
    pos = pd.read_csv(POS_IDS)
    for _, r in pos.iterrows():
        t = r["tic_id"]
        if pd.notna(t) and str(r["host"]) not in m:
            m[str(r["host"])] = int(t)
    return tr, m


def main():
    from astroquery.mast import Catalogs

    tr, m = host_to_tic()
    print(f"training rows {len(tr)}; hosts with a TIC id {len(m)}", flush=True)

    ids = sorted(set(m.values()))
    print(f"unique TIC ids to query: {len(ids)}", flush=True)

    rows = []
    for i in range(0, len(ids), CHUNK):
        chunk = [str(x) for x in ids[i:i + CHUNK]]
        try:
            r = Catalogs.query_criteria(catalog="Tic", ID=chunk).to_pandas()
            have = [c for c in WANT if c in r.columns]
            rows.append(r[have])
        except Exception as e:
            print(f"  chunk {i} failed: {type(e).__name__}: {e}", flush=True)
        print(f"  {min(i + CHUNK, len(ids))}/{len(ids)}", flush=True)

    cat = pd.concat(rows, ignore_index=True)
    cat["ID"] = cat["ID"].astype("int64")
    cat = cat.drop_duplicates(subset="ID")

    inv = pd.DataFrame({"host": list(m.keys()), "ID": list(m.values())})
    out = inv.merge(cat, on="ID", how="left").rename(columns={"ID": "tic_id"})
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  rows={len(out)}", flush=True)
    for c in ["rho", "logg", "mass", "lum", "contratio", "priority"]:
        if c in out.columns:
            print(f"  {c:10s} non-null {out[c].notna().sum():5d} "
                  f"({out[c].notna().mean() * 100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
