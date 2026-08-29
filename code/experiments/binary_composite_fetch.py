"""binary_composite_fetch.py -- fetch the three genuinely-new inputs.

WHAT IS AND IS NOT ALREADY AVAILABLE, checked in source:

  gaia_ruwe / gaia_nss     DEPLOYED. Query was
                           Vizier(columns=["Source","RUWE","NSS","Gmag","+_r"]).
  astrometric excess noise NOT fetched -- `epsi`/`sepsi` were never requested.
                           Genuinely new; needs a Gaia re-query.
  TIC Tmag                 FETCHED AND DISCARDED. `fetch_stellar_params` calls
                           Catalogs.query_criteria(catalog="Tic", ...) -- which
                           returns the full ~125-column TIC row -- then subsets
                           to ["ID","ra","dec","rad","e_rad","mass","e_mass",
                           "Teff"]. Same discard pattern as logg/rho. Cheap to
                           recover by widening the subset.
  SB9                      Genuinely new external catalogue (VizieR B/sb9).

Fetches all three so the TIC-vs-Gaia photometric offset and the SB9 gate can be
measured. Reads only; writes one CSV.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CAT = os.path.join(ROOT, "data", "catalogs")
OUT = os.path.join(HERE, "binary_composite_raw.csv")

GAIA = "I/355/gaiadr3"
SB9 = "B/sb9/main"
MATCH = 5.0
CHUNK = 500


def bulk(ra, dec, catalog, cols, prefix, tag):
    from astropy.table import Table
    import astropy.units as u
    from astroquery.vizier import Vizier
    n = len(ra)
    out = pd.DataFrame({c: np.full(n, np.nan) for c in prefix.values()})
    out[f"{tag}_hit"] = False
    v = Vizier(columns=list(cols) + ["+_r"], row_limit=-1)
    t0 = time.time()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        sub = np.arange(s, e)
        ok = np.isfinite(ra[sub]) & np.isfinite(dec[sub])
        if not ok.any():
            continue
        idx = sub[ok]
        t = Table({"_RAJ2000": ra[idx], "_DEJ2000": dec[idx]})
        t["_RAJ2000"].unit = u.deg; t["_DEJ2000"].unit = u.deg
        try:
            res = v.query_region(t, radius=MATCH * u.arcsec, catalog=catalog)
        except Exception as ex:
            print(f"    {tag} chunk {s}-{e}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
            continue
        if not len(res):
            continue
        r = res[0].to_pandas()
        if "_q" not in r.columns:
            continue
        r = r.sort_values("_r").drop_duplicates("_q")
        pos = idx[(r["_q"].astype(int) - 1).to_numpy()]
        for src, dst in prefix.items():
            if src in r.columns:
                out.loc[pos, dst] = pd.to_numeric(r[src], errors="coerce").to_numpy()
        out.loc[pos, f"{tag}_hit"] = True
        print(f"    [{e}/{n}] {tag} {(time.time()-t0)/60:.1f} min", flush=True)
    return out


def tic_mags(tic_ids):
    """Recover Tmag -- fetched and discarded by fetch_stellar_params."""
    from astroquery.mast import Catalogs
    rows = []
    ids = [int(x) for x in tic_ids if pd.notna(x)]
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        try:
            r = Catalogs.query_criteria(catalog="Tic", ID=chunk)
            keep = [c for c in ("ID", "Tmag", "GAIAmag", "Vmag") if c in r.colnames]
            rows.append(r[keep].to_pandas())
            print(f"    TIC [{i+len(chunk)}/{len(ids)}]", flush=True)
        except Exception as ex:
            print(f"    TIC chunk {i}: {type(ex).__name__}", flush=True)
    if not rows:
        return pd.DataFrame(columns=["tic_id", "tic_tmag", "tic_gaiamag"])
    d = pd.concat(rows, ignore_index=True).drop_duplicates("ID")
    d = d.rename(columns={"ID": "tic_id", "Tmag": "tic_tmag", "GAIAmag": "tic_gaiamag",
                          "Vmag": "tic_vmag"})
    d["tic_id"] = d["tic_id"].astype("int64")
    return d


def main():
    tr = pd.read_csv(TRAINING); tr["host"] = tr.host.astype(str)
    ra = pd.to_numeric(tr.ra, errors="coerce").to_numpy()
    dec = pd.to_numeric(tr.dec, errors="coerce").to_numpy()
    print(f"training {len(tr)}")

    print("\n--- Gaia astrometric excess noise (epsi/sepsi) ---")
    g = bulk(ra, dec, GAIA, ["epsi", "sepsi", "Gmag"],
             {"epsi": "gaia_epsi", "sepsi": "gaia_sepsi", "Gmag": "gaia_gmag2"},
             "gaia")
    print("\n--- SB9 spectroscopic binary catalogue ---")
    s = bulk(ra, dec, SB9, ["Sp1"], {"Sp1": "sb9_sp1"}, "sb9")

    out = pd.concat([tr[["host", "label", "ra", "dec"]].reset_index(drop=True),
                     g.reset_index(drop=True), s.reset_index(drop=True)], axis=1)

    # TIC magnitudes, via the resolved TIC ids from the DV availability work
    av = os.path.join(HERE, "dv_star_availability.csv")
    if os.path.exists(av):
        a = pd.read_csv(av)
        a = a[a._set == "train"][["host", "tic"]].drop_duplicates("host")
        a["host"] = a.host.astype(str)
        out = out.merge(a, on="host", how="left")
        print(f"\n--- TIC magnitudes for {int(out.tic.notna().sum())} resolved ids ---")
        m = tic_mags(out.tic.dropna().tolist())
        out = out.merge(m, left_on="tic", right_on="tic_id", how="left")
    out.to_csv(OUT, index=False)
    print(f"\nsaved {OUT}  {len(out)} rows")
    for c in ("gaia_epsi", "gaia_sepsi", "tic_tmag", "tic_gaiamag"):
        if c in out.columns:
            print(f"  {c:<16}{pd.to_numeric(out[c], errors='coerce').notna().mean():.2%}")
    print(f"  sb9_hit         {float(out.sb9_hit.mean()):.2%}  ({int(out.sb9_hit.sum())} matches)")


if __name__ == "__main__":
    main()
