"""gaia_kinematics_fetch.py -- PART 0/1: fetch proper motions, parallax and
radial velocity for Galactic-kinematics features.

PART 0 FINDING: the Gaia RUWE/NSS deployment queried VizieR's Gaia DR3 mirror
(`I/355/gaiadr3`) but requested only `Source, RUWE, NSS, Gmag, +_r`. **pmRA,
pmDE, Plx and RV were never fetched** -- so unlike the TIC/logg case there is no
discarded column to recover, and a fresh query IS needed. What the deployment
DID retain is `gaia_source` for **99.29%** of training rows, so this reuses the
identical proven bulk-query method rather than rebuilding anything.

A PREDICTION MADE BEFORE THIS RAN, from the already-cached `gaia_gmag`:
negatives are ~1.8 mag BRIGHTER than positives (median G 11.047 vs 12.891;
AUC(gmag) = 0.6432). Gaia DR3 radial velocities are magnitude-limited, so **RV
availability should favour NEGATIVES by roughly 27-29 pp**. This script tests
that rather than assuming it.

Two feature families with DIFFERENT availability profiles, deliberately kept
apart because they may earn different verdicts:

  FULL 3D (U,V,W)      needs pmRA + pmDE + Plx + RV   -- RV is the bottleneck
  TANGENTIAL ONLY      needs pmRA + pmDE + Plx        -- no RV, ~99% astrometry

Reads only; writes one CSV. Touches nothing in production.
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
OUT = os.path.join(HERE, "gaia_kinematics_raw.csv")

CATALOG = "I/355/gaiadr3"
MATCH_ARCSEC = 5.0
CHUNK = 500


def fetch(ra, dec, tag):
    from astropy.table import Table
    import astropy.units as u
    from astroquery.vizier import Vizier
    n = len(ra)
    out = pd.DataFrame({"idx": np.arange(n), "pmra": np.nan, "pmdec": np.nan,
                        "plx": np.nan, "rv": np.nan, "gmag": np.nan,
                        "source": np.nan, "sep_arcsec": np.nan})
    v = Vizier(columns=["Source", "pmRA", "pmDE", "Plx", "RV", "Gmag", "+_r"],
               row_limit=-1)
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
            res = v.query_region(t, radius=MATCH_ARCSEC * u.arcsec, catalog=CATALOG)
        except Exception as ex:
            print(f"    chunk {s}-{e} failed ({type(ex).__name__}); left NaN", flush=True)
            continue
        if not len(res):
            continue
        r = res[0].to_pandas()
        if "_q" not in r.columns:
            continue
        r = r.sort_values("_r").drop_duplicates("_q")
        pos = idx[(r["_q"].astype(int) - 1).to_numpy()]
        for col, key in (("pmra", "pmRA"), ("pmdec", "pmDE"), ("plx", "Plx"),
                         ("rv", "RV"), ("gmag", "Gmag"), ("source", "Source")):
            out.loc[pos, col] = pd.to_numeric(r.get(key), errors="coerce").to_numpy()
        out.loc[pos, "sep_arcsec"] = pd.to_numeric(r.get("_r"), errors="coerce").to_numpy() * 60.0
        el = time.time() - t0
        print(f"    [{e}/{n}] {tag} {el/60:.1f} min, eta "
              f"{el/max(e,1)*(n-e)/60:.1f} min", flush=True)
    return out.drop(columns=["idx"])


def pool_frames():
    out = {}
    for tag, ff, cf in (("main", "unknown_features.csv", "unknown_candidate_list.csv"),
                        ("widesector", "unknown_features_widesector.csv",
                         "unknown_candidate_list_widesector.csv")):
        p = pd.read_csv(os.path.join(CAT, ff)); p["host"] = p.host.astype(str)
        if "status" in p.columns:
            p = p[p.status.astype(str).str.startswith("Success")]
        c = pd.read_csv(os.path.join(CAT, cf))
        c["host"] = c["host"].astype(str) if "host" in c.columns \
            else "TIC_" + c["tic_id"].astype(str)
        keep = [k for k in ("ra", "dec") if k in c.columns]
        if keep:
            p = p.merge(c.drop_duplicates("host")[["host"] + keep], on="host",
                        how="left", suffixes=("", "_c"))
            for k in keep:
                if k + "_c" in p.columns:
                    p[k] = pd.to_numeric(p.get(k), errors="coerce").fillna(
                        pd.to_numeric(p[k + "_c"], errors="coerce"))
        out[tag] = p.reset_index(drop=True)
    return out


def main():
    frames = []
    tr = pd.read_csv(TRAINING); tr["host"] = tr.host.astype(str)
    print(f"training {len(tr)}")
    g = fetch(pd.to_numeric(tr.ra, errors="coerce").to_numpy(),
              pd.to_numeric(tr.dec, errors="coerce").to_numpy(), "train")
    g["host"] = tr.host.values; g["label"] = tr.label.values; g["_set"] = "train"
    g["ra"] = pd.to_numeric(tr.ra, errors="coerce").values
    g["dec"] = pd.to_numeric(tr.dec, errors="coerce").values
    frames.append(g)

    for tag, p in pool_frames().items():
        if "ra" not in p.columns:
            print(f"  {tag}: no coordinates, skipped"); continue
        print(f"{tag} {len(p)}")
        q = fetch(pd.to_numeric(p.ra, errors="coerce").to_numpy(),
                  pd.to_numeric(p.dec, errors="coerce").to_numpy(), tag)
        q["host"] = p.host.values; q["label"] = np.nan; q["_set"] = tag
        q["ra"] = pd.to_numeric(p.ra, errors="coerce").values
        q["dec"] = pd.to_numeric(p.dec, errors="coerce").values
        frames.append(q)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"\nsaved {OUT}  {len(out)} rows")
    for s, d in out.groupby("_set"):
        print(f"  {s:<12} pm {d.pmra.notna().mean():.2%}  plx {d.plx.notna().mean():.2%}  "
              f"RV {d.rv.notna().mean():.2%}")


if __name__ == "__main__":
    main()
