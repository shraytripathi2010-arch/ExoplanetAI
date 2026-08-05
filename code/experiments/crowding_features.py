"""crowding_features.py -- catalog-based neighbour-star crowding, from TIC.

WHAT THIS IS, AND WHAT IT IS NOT

This project already tested and closed a PIXEL-level centroid feature: does the
photocentre shift during transit (77.6% coverage, +0.0032, CI crossed zero).
That measures the TESS image directly.

This is a different thing with the same physical motivation. It asks a CATALOG
question -- how many cataloged stars sit near the target, and how bright are
they relative to it -- using no TESS pixel data at all. A blend can be invisible
to one and obvious to the other: a faint contaminant too close to resolve in
TESS pixels still appears in Gaia, and conversely a centroid shift can come from
a source the catalog missed. Whether the two are correlated in practice is
reported at the end rather than assumed.

DATA SOURCE, AND WHY NO SEPARATE GAIA/2MASS QUERY IS NEEDED

The TESS Input Catalog already cross-references Gaia and 2MASS, and already
carries the two quantities this experiment most wants:

    contratio  -- TIC's own flux contamination ratio, the authoritative
                  version of "how much of the flux in this aperture isn't the
                  target". Computed by the TIC team from the full neighbour
                  list; not something worth rebuilding from scratch.
    numcont    -- number of contaminating sources TIC counted.

Both come back from the same batch-by-ID query `06_download_unknown.py` already
runs for st_rad/st_teff, and both are non-null on inspection. Gaia and 2MASS
photometry (GAIAmag, Jmag, Kmag) are in the same rows.

A cone search adds what TIC does not precompute: the brightness of the single
brightest neighbour, and the distance to the nearest one.

APERTURE RADIUS

A TESS pixel is ~21 arcsec. SPOC optimal apertures are typically 1-4 pixels, and
contaminating flux is dominated by sources within roughly 1-2 pixels. So:

    neighbour features are computed within 42 arcsec (2 TESS pixels)
    counts are also recorded at 21 arcsec (1 pixel) for the sparsity report

One 63-arcsec (3-pixel) query per star serves all of these, so the wider radius
costs nothing extra and leaves headroom to re-derive at other radii without
re-querying.

WORKS AT PREDICTION TIME

Two keying modes, because the two pools are keyed differently:
  - training rows carry ra/dec (host is a star name for positives)
  - unknown candidates are keyed `TIC_<id>`, so the id resolves directly
Either way this needs only a catalog lookup -- no light curve, no pixel data,
nothing that exists solely for already-labelled stars.
"""
import os
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT = os.path.join(SCRIPT_DIR, "crowding_per_star.csv")

QUERY_RADIUS_ARCSEC = 63.0     # 3 TESS pixels -- one query serves all radii
APERTURE_ARCSEC = 42.0         # 2 TESS pixels -- the "in aperture" radius
INNER_ARCSEC = 21.0            # 1 TESS pixel -- for the sparsity report
TARGET_MATCH_ARCSEC = 15.0     # nearest source beyond this = target not resolved
N_WORKERS = 4                  # polite to MAST; 0.37 s/query serially
MAX_RETRY = 3

FEATURES = ["crowd_contratio", "crowd_numcont",
            "crowd_flux_ratio_max", "crowd_nearest_arcsec"]


def _tic_id_from_host(host):
    m = re.fullmatch(r"TIC[_ ]?(\d+)", str(host).strip())
    return m.group(1) if m else None


def crowding_for(host, ra=None, dec=None):
    """Catalog crowding features for one star. Returns a dict, never raises.

    `ra`/`dec` optional: if absent and the host is TIC-keyed, the coordinates
    are resolved from the catalog, which is the path production uses.
    """
    from astroquery.mast import Catalogs
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    out = {"host": host, "crowd_ok": 0, "crowd_note": ""}
    for f in FEATURES:
        out[f] = np.nan

    try:
        if ra is None or dec is None or not np.isfinite(ra) or not np.isfinite(dec):
            tid = _tic_id_from_host(host)
            if tid is None:
                out["crowd_note"] = "no ra/dec and host is not TIC-keyed"
                return out
            t = Catalogs.query_criteria(catalog="Tic", ID=[tid]).to_pandas()
            if t.empty:
                out["crowd_note"] = "TIC id not found"
                return out
            ra, dec = float(t.iloc[0]["ra"]), float(t.iloc[0]["dec"])

        coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
        r = Catalogs.query_region(
            coord, radius=QUERY_RADIUS_ARCSEC * u.arcsec, catalog="Tic").to_pandas()
        if r.empty:
            out["crowd_note"] = "no TIC sources within query radius"
            return out

        r = r.sort_values("dstArcSec").reset_index(drop=True)
        tgt = r.iloc[0]
        out["crowd_target_sep_arcsec"] = float(tgt["dstArcSec"])
        out["crowd_target_tmag"] = float(tgt["Tmag"]) if pd.notna(tgt["Tmag"]) else np.nan
        if float(tgt["dstArcSec"]) > TARGET_MATCH_ARCSEC:
            out["crowd_note"] = f"nearest source {tgt['dstArcSec']:.1f}\" away; target unresolved"
            return out

        # --- TIC's own contamination numbers, for the target row ---
        out["crowd_contratio"] = (float(tgt["contratio"])
                                  if pd.notna(tgt.get("contratio")) else np.nan)
        out["crowd_numcont"] = (float(tgt["numcont"])
                                if pd.notna(tgt.get("numcont")) else np.nan)

        nb = r.iloc[1:]                                   # everything but the target
        out["crowd_n_nb_21"] = int((nb["dstArcSec"] <= INNER_ARCSEC).sum())
        out["crowd_n_nb_42"] = int((nb["dstArcSec"] <= APERTURE_ARCSEC).sum())
        out["crowd_n_nb_63"] = int(len(nb))

        # --- nearest neighbour, any distance inside the query radius ---
        out["crowd_nearest_arcsec"] = (float(nb["dstArcSec"].min())
                                       if len(nb) else np.nan)

        # --- brightest neighbour inside the aperture, as a flux ratio ---
        # 10^(-0.4 dTmag): >1 means the neighbour outshines the target.
        ap = nb[(nb["dstArcSec"] <= APERTURE_ARCSEC) & nb["Tmag"].notna()]
        if len(ap) and pd.notna(tgt["Tmag"]):
            dmag = ap["Tmag"].astype(float) - float(tgt["Tmag"])
            out["crowd_flux_ratio_max"] = float(np.max(10.0 ** (-0.4 * dmag)))
        else:
            # genuinely no catalogued neighbour in the aperture -> zero contaminating
            # flux, which is a real measurement, not missing data.
            out["crowd_flux_ratio_max"] = 0.0

        out["crowd_ok"] = 1
        return out
    except Exception as e:
        out["crowd_note"] = f"{type(e).__name__}: {e}"[:200]
        return out


def _with_retry(args):
    host, ra, dec = args
    for attempt in range(MAX_RETRY):
        res = crowding_for(host, ra, dec)
        if res["crowd_ok"] or "not found" in res["crowd_note"] or "unresolved" in res["crowd_note"]:
            return res
        time.sleep(1.5 * (attempt + 1))
    return res


def main():
    df = pd.read_csv(TRAINING)
    todo = df[["host", "ra", "dec"]].copy()

    done = {}
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = {h: True for h in prev["host"]}
        print(f"resuming: {len(done)} stars already fetched")
    todo = todo[~todo["host"].isin(done)]

    print(f"fetching crowding for {len(todo)} stars "
          f"(query radius {QUERY_RADIUS_ARCSEC}\", aperture {APERTURE_ARCSEC}\")")
    if todo.empty:
        print("nothing to do")
        return

    rows, t0 = [], time.time()
    args = [(r.host, r.ra, r.dec) for r in todo.itertuples()]
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_with_retry, a): a for a in args}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 250 == 0 or i == len(args):
                el = time.time() - t0
                ok = sum(r["crowd_ok"] for r in rows)
                print(f"  {i}/{len(args)}  ok={ok}  {el/60:.1f} min "
                      f"(eta {el/i*(len(args)-i)/60:.1f} min)", flush=True)
                out = pd.DataFrame(rows)
                if done:
                    out = pd.concat([pd.read_csv(OUT), out], ignore_index=True)
                out.drop_duplicates("host", keep="last").to_csv(OUT, index=False)

    out = pd.DataFrame(rows)
    if done:
        out = pd.concat([pd.read_csv(OUT), out], ignore_index=True)
    out = out.drop_duplicates("host", keep="last")
    out.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}  ({len(out)} stars, {int(out['crowd_ok'].sum())} ok)")


if __name__ == "__main__":
    main()
