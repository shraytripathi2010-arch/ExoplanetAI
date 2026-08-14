"""gaia_astrometry_features.py -- Gaia DR3 RUWE + non-single-star flag per star,
for the training set AND both real candidate pools.

WHY A NEW QUERY IS NEEDED (the "already-pulled-and-unused" pattern does NOT
repeat a third time)
---------------------------------------------------------------------------
Twice before, a proposed catalog feature turned out to be already present in an
existing response and merely unused: crowding's `contratio`/`numcont`, and
stellar density's `rho`/`logg`. Checked directly here, and this time it is NOT
the case. The TIC batch query returns **125 columns and none is RUWE or NSS**:

    TIC version 20190415  (TIC v8, built on Gaia DR2)
    gaia-related columns: GAIA, GAIAmag, e_GAIAmag, gaiabp, gaiarp, gaiaqflag
    'ruwe' present: False        'nss' present: False

RUWE and NSS are Gaia **DR3** products and TIC v8 predates DR3, so a genuine new
query is required. This is the third time the pattern has been checked and the
first time it has come back negative.

ONE QUERY, BOTH FIELDS
----------------------
`non_single_star` is a bitfield in the MAIN Gaia DR3 source table (1 =
astrometric, 2 = spectroscopic, 4 = eclipsing), so the NSS *flag* does NOT need
the separate `nss_two_body_orbit` / `nss_acceleration_astro` tables -- only the
detailed orbital solutions live there, and this feature does not use them.

WHY VIZIER AND NOT THE GAIA TAP SERVICE
---------------------------------------
Measured, not assumed. `Gaia.launch_job` per star did not complete **16 stars in
10 minutes** with 8 workers, and an async `tap_upload` cross-match hung past the
same limit. VizieR's Gaia DR3 mirror (`I/355/gaiadr3`) takes a whole coordinate
TABLE in ONE call: **200 stars in 33 s = 165 ms/star**, ~15 min for the full
training set. Same data, ~200x the throughput.

MATCH RADIUS
------------
3 arcsec, nearest match wins. TESS pixels are ~21 arcsec, but this is *target
identification* against Gaia's own astrometry -- "which Gaia source IS this
star" -- not an aperture question, so the radius is tight. `_r` (separation) is
kept so a loose match can be audited.

READ-ONLY.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
OUT_TRAIN = os.path.join(HERE, "gaia_astrometry_training.csv")
OUT_POOLS = os.path.join(HERE, "gaia_astrometry_pools.csv")

CATALOG = "I/355/gaiadr3"
MATCH_ARCSEC = 3.0
CHUNK = 200
RUWE_THRESHOLD = 1.4    # standard Gaia "likely non-single" cut; see writeup


def fetch_block(ra, dec, tag=""):
    """Bulk cone match for a block of coordinates. Returns one row per input
    index (NaN where Gaia has nothing within MATCH_ARCSEC).

    Never raises on a chunk: a VizieR hiccup leaves that chunk NaN rather than
    losing the whole pass, same contract as crowding_for.
    """
    from astropy.table import Table
    import astropy.units as u
    from astroquery.vizier import Vizier

    n = len(ra)
    out = pd.DataFrame({"idx": np.arange(n), "gaia_ruwe": np.nan,
                        "gaia_nss": np.nan, "gaia_gmag": np.nan,
                        "gaia_sep_arcsec": np.nan, "gaia_source": np.nan})
    v = Vizier(columns=["Source", "RUWE", "NSS", "Gmag", "+_r"], row_limit=-1)
    t0 = time.time()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        sub = np.arange(s, e)
        ok = np.isfinite(ra[sub]) & np.isfinite(dec[sub])
        if not ok.any():
            continue
        idx = sub[ok]
        t = Table({"_RAJ2000": ra[idx], "_DEJ2000": dec[idx]})
        t["_RAJ2000"].unit = u.deg
        t["_DEJ2000"].unit = u.deg
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
        # VizieR's _q is 1-based into the uploaded table
        r = r.sort_values("_r").drop_duplicates("_q")
        pos = idx[(r["_q"].astype(int) - 1).to_numpy()]
        out.loc[pos, "gaia_ruwe"] = pd.to_numeric(r.get("RUWE"), errors="coerce").to_numpy()
        out.loc[pos, "gaia_nss"] = pd.to_numeric(r.get("NSS"), errors="coerce").to_numpy()
        out.loc[pos, "gaia_gmag"] = pd.to_numeric(r.get("Gmag"), errors="coerce").to_numpy()
        out.loc[pos, "gaia_sep_arcsec"] = pd.to_numeric(r.get("_r"), errors="coerce").to_numpy() * 60.0
        out.loc[pos, "gaia_source"] = pd.to_numeric(r.get("Source"), errors="coerce").to_numpy()
        el = time.time() - t0
        print(f"    [{e}/{n}] {tag} {el/60:.1f} min, eta "
              f"{el/max(e,1)*(n-e)/60:.1f} min", flush=True)
    out["gaia_nss_flag"] = np.where(out.gaia_nss.notna(), (out.gaia_nss > 0).astype(float), np.nan)
    out["gaia_ruwe_high"] = np.where(out.gaia_ruwe.notna(),
                                     (out.gaia_ruwe > RUWE_THRESHOLD).astype(float), np.nan)
    return out.drop(columns=["idx"])


def main():
    # ---- training ----
    df = pd.read_csv(os.path.join(ROOT, "data", "training_dataset", "training.csv"),
                     usecols=["host", "label", "ra", "dec"])
    df["host"] = df.host.astype(str)
    print(f"TRAINING: {len(df)} rows, {int(df.ra.notna().sum())} with coordinates")
    g = fetch_block(pd.to_numeric(df.ra, errors="coerce").to_numpy(),
                    pd.to_numeric(df.dec, errors="coerce").to_numpy(), "train")
    tr = pd.concat([df.reset_index(drop=True), g], axis=1)
    tr.to_csv(OUT_TRAIN, index=False)
    print(f"saved {OUT_TRAIN}   RUWE available {tr.gaia_ruwe.notna().mean():.1%}")

    # ---- both candidate pools ----
    frames = []
    for tag, f in (("main", "unknown_candidate_list.csv"),
                   ("widesector", "unknown_candidate_list_widesector.csv")):
        p = os.path.join(ROOT, "data", "catalogs", f)
        if not os.path.exists(p):
            print(f"  {tag}: {f} not found"); continue
        c = pd.read_csv(p)
        if not {"ra", "dec"} <= set(c.columns):
            print(f"  {tag}: no ra/dec columns -- {list(c.columns)[:8]}"); continue
        c["host"] = "TIC_" + c.tic_id.astype("int64").astype(str)
        print(f"POOL {tag}: {len(c)} rows, {int(c.ra.notna().sum())} with coordinates")
        gg = fetch_block(pd.to_numeric(c.ra, errors="coerce").to_numpy(),
                         pd.to_numeric(c.dec, errors="coerce").to_numpy(), tag)
        frames.append(pd.concat([c[["host"]].reset_index(drop=True).assign(pool=tag),
                                 gg], axis=1))
    if frames:
        pl = pd.concat(frames, ignore_index=True)
        pl.to_csv(OUT_POOLS, index=False)
        print(f"saved {OUT_POOLS}")
        print(pl.groupby("pool").agg(n=("host", "size"),
                                     ruwe_avail=("gaia_ruwe", lambda s: s.notna().mean()),
                                     nss_avail=("gaia_nss", lambda s: s.notna().mean())
                                     ).round(4).to_string())


if __name__ == "__main__":
    main()
