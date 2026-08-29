"""spectro_chem_availability.py -- PART 1 availability gate for spectroscopic
stellar-chemistry cross-match features.

GENUINELY NEW TERRITORY: no prior investigation in this project has cross-matched
against a spectroscopic survey, and `FEATURE_COLUMNS` contains no metallicity or
abundance column of any kind.

But this is also the EXACT shape of this project's most-repeated failure. The
availability gate runs BEFORE any chemistry value is looked at, because that
ordering is what caught:

    TIC-native CTL fields        31 pp class split
    SPOC DV centroids            43.83 pp, AUC(availability) 0.2808
    multi-sector consistency     indicator-only arm = 108% of the apparent gain
    sector/epoch proxy           +0.0063 from pure bookkeeping

And it is the same gate `gaia_ruwe`/`gaia_nss` PASSED cleanly before being
deployed -- so the gate is not a formality that always fails.

The specific hazard here is concrete and predictable: APOGEE, GALAH, LAMOST and
Gaia-ESO are magnitude-limited and target-selected. This project's training
positives are named confirmed planets (bright, heavily followed up); its
negatives are TIC-named TOI false positives (fainter, selected only by TESS).
If spectroscopic coverage tracks that split, "has a spectrum" is a label proxy.

Reads only. Writes one JSON + one CSV. Touches nothing in production.
"""
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CAT = os.path.join(ROOT, "data", "catalogs")
OUT = os.path.join(HERE, "spectro_chem_availability.json")
OUT_CSV = os.path.join(HERE, "spectro_chem_crossmatch.csv")

RADIUS_ARCSEC = 5.0
SURVEYS = {
    "APOGEE_DR17":  "vizier:III/286/catalog",
    "GALAH_DR3":    "vizier:J/MNRAS/506/150/catalog",
    "LAMOST_DR7":   "vizier:V/156/lrsdr7",
    "GaiaESO_DR5":  "vizier:J/A+A/666/A121/tableb5",
}


def pools():
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
        if "ra" not in p.columns or pd.to_numeric(
                p.get("ra", pd.Series(dtype=float)), errors="coerce").notna().sum() == 0:
            p = _coords_from_tic(p, tag)
        out[tag] = p.reset_index(drop=True)
    return out


def _coords_from_tic(p, tag):
    """The widesector candidate list carries no coordinates; every host is
    TIC-named, so resolve them from the TIC directly."""
    import re
    tics = [int(m.group(1)) for m in
            (re.match(r"TIC[_ ]?(\d+)", h) for h in p.host.astype(str)) if m]
    if not tics:
        print(f"    [{tag}] no TIC ids to resolve coordinates from")
        return p
    try:
        from astroquery.vizier import Vizier
        v = Vizier(columns=["TIC", "RAJ2000", "DEJ2000"], row_limit=-1)
        t = v.query_constraints(catalog="IV/39/tic82",
                                TIC="=,".join([""] + [str(x) for x in tics])[1:])
        if t:
            d = t[0].to_pandas()
            m = dict(zip(d["TIC"].astype(int), zip(d["RAJ2000"], d["DEJ2000"])))
            p = p.copy()
            p["ra"] = [m.get(int(re.match(r"TIC[_ ]?(\d+)", h).group(1)), (np.nan, np.nan))[0]
                       if re.match(r"TIC[_ ]?(\d+)", h) else np.nan for h in p.host.astype(str)]
            p["dec"] = [m.get(int(re.match(r"TIC[_ ]?(\d+)", h).group(1)), (np.nan, np.nan))[1]
                        if re.match(r"TIC[_ ]?(\d+)", h) else np.nan for h in p.host.astype(str)]
            print(f"    [{tag}] resolved coordinates for "
                  f"{int(pd.to_numeric(p.ra, errors='coerce').notna().sum())}/{len(p)} from TIC")
    except Exception as e:
        print(f"    [{tag}] TIC coordinate resolution failed: {type(e).__name__}: {str(e)[:60]}")
    return p


def xmatch(df, survey_id, label):
    """Bulk cone cross-match via CDS XMatch. Returns a boolean Series."""
    from astroquery.xmatch import XMatch
    from astropy.table import Table
    import astropy.units as u
    if "ra" not in df.columns or "dec" not in df.columns:
        print(f"    [{label}] no ra/dec columns on this table")
        return pd.Series(False, index=df.index), 0
    ra = pd.to_numeric(df["ra"], errors="coerce")
    dec = pd.to_numeric(df["dec"], errors="coerce")
    ok = ra.notna() & dec.notna()
    hit = pd.Series(False, index=df.index)
    if not ok.any():
        print(f"    [{label}] no coordinates available")
        return hit, 0
    sub = pd.DataFrame({"__id": df.index[ok].astype(int),
                        "ra": ra[ok], "dec": dec[ok]})
    matched = set()
    for i in range(0, len(sub), 3000):
        chunk = sub.iloc[i:i + 3000]
        try:
            res = XMatch.query(cat1=Table.from_pandas(chunk), cat2=survey_id,
                               max_distance=RADIUS_ARCSEC * u.arcsec,
                               colRA1="ra", colDec1="dec")
            matched |= set(res.to_pandas()["__id"].astype(int).tolist())
        except Exception as e:
            print(f"    [{label}] chunk {i} FAILED: {type(e).__name__}: {str(e)[:70]}")
    hit.loc[list(matched)] = True
    return hit, int(ok.sum())


def main():
    res = {"radius_arcsec": RADIUS_ARCSEC}
    tr = pd.read_csv(TRAINING); tr["host"] = tr.host.astype(str)
    y = tr.label.to_numpy()
    P = pools()
    print(f"training {len(tr)} ({int((y==1).sum())} pos / {int((y==0).sum())} neg); "
          f"main {len(P['main'])}; widesector {len(P['widesector'])}\n")

    keep = tr[["host", "label"]].copy()
    for name, cid in SURVEYS.items():
        print("=" * 88); print(f"{name}   ({cid})"); print("=" * 88)
        h_tr, n_coord = xmatch(tr, cid, f"{name}/train")
        keep[name] = h_tr.values
        ap = float(h_tr[y == 1].mean()); an = float(h_tr[y == 0].mean())
        tab = [[int((h_tr & (y == 1)).sum()), int((~h_tr & (y == 1)).sum())],
               [int((h_tr & (y == 0)).sum()), int((~h_tr & (y == 0)).sum())]]
        try:
            orr, pv = fisher_exact(tab)
        except Exception:
            orr, pv = np.nan, np.nan
        auc = float(roc_auc_score(y, h_tr.values.astype(float)))
        d = {"train_overall": float(h_tr.mean()), "train_pos": ap, "train_neg": an,
             "diff_pp": (an - ap) * 100, "odds_ratio": float(orr),
             "fisher_p": float(pv), "auc_availability": auc,
             "n_with_coords": n_coord, "contingency": tab}
        print(f"  training overall   {h_tr.mean():>8.2%}  ({int(h_tr.sum())}/{len(tr)})")
        print(f"  training POSITIVES {ap:>8.2%}")
        print(f"  training NEGATIVES {an:>8.2%}")
        print(f"  >>> DIFFERENCE     {(an-ap)*100:>+7.2f} pp     "
              f"odds ratio {orr:.4g}   Fisher p {pv:.4g}")
        print(f"  >>> AUC(availability alone) = {auc:.4f}   |AUC-0.5| = {abs(auc-0.5):.4f}")
        for tag in ("main", "widesector"):
            h, _ = xmatch(P[tag], cid, f"{name}/{tag}")
            d[f"pool_{tag}"] = float(h.mean())
            d[f"pool_{tag}_n"] = int(len(P[tag]))
            print(f"  pool {tag:<12}{h.mean():>8.2%}  ({int(h.sum())}/{len(P[tag])})")
        res[name] = d
        print()

    # any-survey union
    cols = [c for c in SURVEYS if c in keep.columns]
    anyhit = keep[cols].any(axis=1)
    ap = float(anyhit[y == 1].mean()); an = float(anyhit[y == 0].mean())
    auc = float(roc_auc_score(y, anyhit.values.astype(float)))
    print("=" * 88); print("UNION OF ALL SURVEYS"); print("=" * 88)
    print(f"  training overall   {anyhit.mean():>8.2%}")
    print(f"  positives {ap:.2%}   negatives {an:.2%}   diff {(an-ap)*100:+.2f} pp")
    print(f"  >>> AUC(availability alone) = {auc:.4f}   |AUC-0.5| = {abs(auc-0.5):.4f}")
    res["UNION"] = {"train_overall": float(anyhit.mean()), "train_pos": ap,
                    "train_neg": an, "diff_pp": (an - ap) * 100,
                    "auc_availability": auc}
    keep["any_survey"] = anyhit
    keep.to_csv(OUT_CSV, index=False)
    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nsaved {OUT}\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
