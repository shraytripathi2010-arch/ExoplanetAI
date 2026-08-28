"""dv_centroid_availability.py -- FULL-SCALE availability of SPOC DV difference-
image centroid results, and the provenance check that decides whether they are
usable at all.

WHY NOT 6,091 MAST QUERIES
--------------------------
The 40-star pilot cone-searched MAST per star at ~37 s each. At full scale that
is ~62 hours. MAST instead publishes a per-sector BULK DOWNLOAD SCRIPT listing
every DV product by filename, with the TIC id embedded:

    tess2020213081515-s0028-s0028-0000000261136679-00364_dvr.xml

Fetching ~105 of those gives the COMPLETE, authoritative inventory of which TIC
ids have a machine-readable DV report -- in minutes, not days, and without
sampling error.

THE CHECK THAT MATTERS
----------------------
DV reports exist only for SPOC-PIPELINE-DETECTED TCEs. This training set's
provenance is almost perfectly class-split: 100.00% of negatives are TIC-named
TOI false positives/alarms (SPOC TCEs by construction), against 1.79% of
positives, which are named confirmed planets from mixed catalogues. If DV
availability tracks that split, then "has a DV report" is a label proxy and the
feature is dead on arrival -- the same failure mode as the TIC-native-field CTL
trap and the multi-sector missingness indicator (+0.0102, 108% of the gain).

That is the single most important number this script produces.
"""
import os
import re
import sys
import json
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CAT = os.path.join(ROOT, "data", "catalogs")
OUT_INV = os.path.join(HERE, "dv_inventory_tics.csv")
OUT = os.path.join(HERE, "dv_centroid_availability.json")
OUT_MAP = os.path.join(HERE, "dv_star_availability.csv")

SECTORS = range(1, 106)
URL = ("https://archive.stsci.edu/missions/tess/download_scripts/sector/"
       "tesscurl_sector_{}_dv.sh")
POOLS = [("main", "unknown_features.csv"),
         ("widesector", "unknown_features_widesector.csv")]


def fetch_sector(s):
    try:
        r = urllib.request.urlopen(URL.format(s), timeout=120)
        body = r.read().decode("utf-8", "replace")
    except Exception:
        return s, set(), 0
    # only the machine-readable reports -- the PDFs carry no extractable numbers
    tics = set(int(m) for m in re.findall(
        r"-(\d{16})-\d+_dvr\.xml", body))
    return s, tics, body.count("_dvr.xml")


def build_inventory():
    if os.path.exists(OUT_INV):
        d = pd.read_csv(OUT_INV)
        print(f"  inventory cached: {len(d)} TICs")
        return set(d.tic.astype(int)), None
    allt, per = set(), {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for s, tics, n in ex.map(fetch_sector, SECTORS):
            per[s] = len(tics)
            allt |= tics
            if len(tics):
                print(f"  sector {s:>3}: {len(tics):>6} TICs with dvr.xml", flush=True)
    pd.DataFrame({"tic": sorted(allt)}).to_csv(OUT_INV, index=False)
    print(f"  TOTAL distinct TICs with a DV report: {len(allt):,}")
    return allt, per


def resolve_tics(df, label):
    """host -> TIC id. TIC_-prefixed hosts are exact; the rest are cone-matched
    against the TIC in one bulk XMatch upload."""
    h = df.host.astype(str)
    tic = pd.Series(np.nan, index=df.index, dtype="float64")
    direct = h.str.match(r"^TIC[_ ]?\d+$")
    tic[direct] = h[direct].str.extract(r"(\d+)")[0].astype(float)
    if "ra" in df.columns and "dec" in df.columns:
        need = (~direct) & pd.to_numeric(df.ra, errors="coerce").notna() \
                         & pd.to_numeric(df.dec, errors="coerce").notna()
    else:
        # pool feature tables carry no coordinates -- but every pool host is
        # already TIC-named, so nothing needs cross-matching
        need = pd.Series(False, index=df.index)
    print(f"  [{label}] direct TIC ids: {int(direct.sum())}; "
          f"to cross-match: {int(need.sum())}")
    if need.sum():
        from astroquery.xmatch import XMatch
        from astropy.table import Table
        import astropy.units as u
        sub = pd.DataFrame({"__id": df.index[need].astype(int),
                            "ra": pd.to_numeric(df.ra[need], errors="coerce"),
                            "dec": pd.to_numeric(df.dec[need], errors="coerce")})
        got = 0
        for i in range(0, len(sub), 3000):
            chunk = sub.iloc[i:i + 3000]
            try:
                res = XMatch.query(cat1=Table.from_pandas(chunk),
                                   cat2="vizier:IV/39/tic82",
                                   max_distance=5 * u.arcsec,
                                   colRA1="ra", colDec1="dec")
                r = res.to_pandas().sort_values("angDist").drop_duplicates("__id")
                m = dict(zip(r["__id"].astype(int), r["TIC"].astype(float)))
                for k, v in m.items():
                    tic.at[k] = v
                got += len(m)
                print(f"    xmatch {i}-{i+len(chunk)}: {len(m)} matched", flush=True)
            except Exception as e:
                print(f"    xmatch chunk {i} FAILED: {type(e).__name__}: {e}", flush=True)
    print(f"  [{label}] TIC resolved: {int(tic.notna().sum())}/{len(df)} "
          f"({tic.notna().mean():.2%})")
    return tic


def main():
    print("=" * 90)
    print("1. BULK DV INVENTORY FROM MAST (complete, not sampled)")
    print("=" * 90)
    inv, _ = build_inventory()

    print("\n" + "=" * 90)
    print("2. RESOLVING THIS PROJECT'S STARS TO TIC IDS")
    print("=" * 90)
    tr = pd.read_csv(TRAINING); tr["host"] = tr.host.astype(str)
    tr["tic"] = resolve_tics(tr, "training")
    tr["has_dv"] = tr.tic.apply(lambda t: (not pd.isna(t)) and int(t) in inv)

    pools = {}
    for tag, ff in POOLS:
        p = pd.read_csv(os.path.join(CAT, ff)); p["host"] = p.host.astype(str)
        if "status" in p.columns:
            p = p[p.status.astype(str).str.startswith("Success")].reset_index(drop=True)
        p["tic"] = resolve_tics(p, tag)
        p["has_dv"] = p.tic.apply(lambda t: (not pd.isna(t)) and int(t) in inv)
        pools[tag] = p

    print("\n" + "=" * 90)
    print("3. FULL-SCALE AVAILABILITY  (the primary deliverable)")
    print("=" * 90)
    y = tr.label.to_numpy()
    res = {"n_inventory_tics": len(inv), "n_train": len(tr)}
    ap = float(tr.has_dv[y == 1].mean()); an = float(tr.has_dv[y == 0].mean())
    print(f"  training overall      {tr.has_dv.mean():>8.2%}  ({int(tr.has_dv.sum())}/{len(tr)})")
    print(f"  training POSITIVES    {ap:>8.2%}  ({int(tr.has_dv[y==1].sum())}/{int((y==1).sum())})")
    print(f"  training NEGATIVES    {an:>8.2%}  ({int(tr.has_dv[y==0].sum())}/{int((y==0).sum())})")
    print(f"  >>> DIFFERENCE        {an-ap:>+8.2%}")
    for tag in pools:
        print(f"  pool {tag:<16}{pools[tag].has_dv.mean():>8.2%}  "
              f"({int(pools[tag].has_dv.sum())}/{len(pools[tag])})")
    res["train_overall"] = float(tr.has_dv.mean())
    res["train_pos"] = ap; res["train_neg"] = an; res["diff"] = an - ap
    for tag in pools:
        res[f"pool_{tag}"] = float(pools[tag].has_dv.mean())
        res[f"pool_{tag}_n"] = int(len(pools[tag]))

    print("\n" + "=" * 90)
    print("4. THE PROVENANCE / CTL-TRAP CHECK")
    print("=" * 90)
    from scipy.stats import fisher_exact
    from sklearn.metrics import roc_auc_score
    a = tr.has_dv.to_numpy()
    tab = [[int((a & (y == 1)).sum()), int((~a & (y == 1)).sum())],
           [int((a & (y == 0)).sum()), int((~a & (y == 0)).sum())]]
    orr, p = fisher_exact(tab)
    auc_avail = float(roc_auc_score(y, a.astype(float)))
    print(f"  contingency [[pos_dv, pos_nodv],[neg_dv, neg_nodv]] = {tab}")
    print(f"  odds ratio {orr:.4g}   Fisher p {p:.4g}")
    print(f"  >>> AUC(availability alone) = {auc_avail:.4f}   "
          f"|AUC-0.5| = {abs(auc_avail-0.5):.4f}")
    print(f"  reference: the CTL trap disqualified at 31 pp; the multi-sector")
    print(f"             missingness indicator scored +0.0102 = 108% of its gain")
    res["gate"] = {"contingency": tab, "odds_ratio": float(orr),
                   "fisher_p": float(p), "auc_availability": auc_avail}

    # identifier provenance, for the record
    direct = tr.host.str.match(r"^TIC[_ ]?\d+$")
    print(f"\n  identifier provenance (the structural cause to rule in or out):")
    print(f"    TIC-named  positives {float(direct[y==1].mean()):>7.2%}   "
          f"negatives {float(direct[y==0].mean()):>7.2%}")
    res["tic_named"] = {"pos": float(direct[y == 1].mean()),
                        "neg": float(direct[y == 0].mean())}

    keep = tr[["host", "label", "tic", "has_dv"]].copy()
    keep["_set"] = "train"
    for tag in pools:
        q = pools[tag][["host", "tic", "has_dv"]].copy()
        q["label"] = np.nan; q["_set"] = tag
        keep = pd.concat([keep, q], ignore_index=True)
    keep.to_csv(OUT_MAP, index=False)
    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nsaved {OUT}\nsaved {OUT_MAP}")


if __name__ == "__main__":
    main()
