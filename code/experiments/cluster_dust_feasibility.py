"""cluster_dust_feasibility.py -- PART 0/1 for two mechanistically distinct proposals.

  (1) STELLAR CLUSTER / MOVING-GROUP MEMBERSHIP  -- a youth/age proxy
  (2) 3D DUST EXTINCTION (Bayestar19)            -- an environmental/distance proxy

Reuses the bulk CDS XMatch method proven by the Gaia RUWE/NSS deployment and the
spectroscopic-chemistry gate (`spectro_chem_availability.py`), unmodified in
substance, so coverage numbers are directly comparable to those precedents.

A NOTE ON WHAT "AVAILABILITY" MEANS, because it differs between the two:

  * For DUST, availability is ordinary missingness: a star outside the map's
    sky footprint, or without a distance, has no value. Same gate as APOGEE.

  * For CLUSTER MEMBERSHIP it is NOT. Membership is a FLAG that is defined for
    every star with coordinates -- a non-member is a measured `False`, not a
    missing value. So the gate is not "what fraction have data" but "what
    fraction are members", which is an EFFECTIVE-n question, plus whether that
    flag is class-correlated. Both are reported.

Reads only. Writes one JSON and one CSV to code/experiments/. Touches nothing
in production.
"""
import os
import re
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CAT = os.path.join(ROOT, "data", "catalogs")
RES = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(HERE, "cluster_dust_feasibility.json")
OUT_CSV = os.path.join(HERE, "cluster_dust_crossmatch.csv")

RADIUS_ARCSEC = 5.0
BAYESTAR_DEC_MIN = -30.0          # Pan-STARRS 1 footprint, verified from dustmaps source

CLUSTER_CATALOGS = [
    ("HuntReffert2023", "vizier:J/A+A/673/A114/members"),   # Gaia DR3, ~7200 clusters, most complete
    ("CantatGaudin2020", "vizier:J/A+A/633/A99/members"),   # Gaia DR2 members
    ("CantatGaudin2018", "vizier:J/A+A/618/A93/members"),   # Gaia DR2, original
]


def auc(y, x):
    from scipy.stats import rankdata
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30 or len(np.unique(y[m])) < 2:
        return float("nan")
    r = rankdata(x[m]); y1 = y[m] == 1
    n1, n0 = y1.sum(), (~y1).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def tic_coords(tic_ids, tag):
    """Resolve ra/dec for a list of TIC ids via VizieR's TIC mirror."""
    out = {}
    try:
        from astroquery.vizier import Vizier
        v = Vizier(columns=["TIC", "RAJ2000", "DEJ2000"], row_limit=-1)
        ids = [int(t) for t in tic_ids if pd.notna(t)]
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            q = v.query_constraints(catalog="IV/39/tic82",
                                    TIC="=,".join([""] + [str(c) for c in chunk])[1:])
            if q:
                t = q[0].to_pandas()
                for _, r in t.iterrows():
                    out[int(r["TIC"])] = (float(r["RAJ2000"]), float(r["DEJ2000"]))
        print(f"    [{tag}] TIC-resolved {len(out)}/{len(ids)}", flush=True)
    except Exception as e:
        print(f"    [{tag}] TIC resolution failed: {type(e).__name__}: {str(e)[:70]}", flush=True)
    return out


def pools():
    """Every candidate population this feature would have to serve, with coords."""
    P = {}
    p = pd.read_csv(os.path.join(RES, "unknown_candidates", "ranked_candidates.csv"))
    P["main_scored"] = p[["host", "ra", "dec"]].copy()

    cl = pd.read_csv(os.path.join(CAT, "unknown_candidate_list.csv"))
    if {"ra", "dec"}.issubset(cl.columns):
        cl = cl.copy()
        cl["host"] = ["TIC_%d" % int(t) for t in cl.tic_id]
        P["main_target_list"] = cl[["host", "ra", "dec"]]

    w = pd.read_csv(os.path.join(CAT, "unknown_features_widesector.csv"))
    if "status" in w.columns:
        w = w[w.status.astype(str).str.startswith("Success")]
    tics = [int(re.match(r"TIC[_ ]?(\d+)", str(h)).group(1))
            for h in w.host if re.match(r"TIC[_ ]?(\d+)", str(h))]
    m = tic_coords(tics, "widesector")
    P["widesector"] = pd.DataFrame({
        "host": ["TIC_%d" % t for t in tics],
        "ra": [m.get(t, (np.nan, np.nan))[0] for t in tics],
        "dec": [m.get(t, (np.nan, np.nan))[1] for t in tics]})
    return P


def xmatch(df, cat_id, label):
    """Bulk cone cross-match via CDS XMatch. Returns (bool Series, n_with_coords)."""
    from astroquery.xmatch import XMatch
    from astropy.table import Table
    import astropy.units as u
    ra = pd.to_numeric(df["ra"], errors="coerce")
    dec = pd.to_numeric(df["dec"], errors="coerce")
    ok = ra.notna() & dec.notna()
    hit = pd.Series(False, index=df.index)
    if not ok.any():
        print(f"    [{label}] no coordinates", flush=True)
        return hit, 0
    sub = pd.DataFrame({"__id": df.index[ok].astype(int), "ra": ra[ok], "dec": dec[ok]})
    matched, failed = set(), 0
    for i in range(0, len(sub), 3000):
        chunk = sub.iloc[i:i + 3000]
        for attempt in range(3):
            try:
                res = XMatch.query(cat1=Table.from_pandas(chunk), cat2=cat_id,
                                   max_distance=RADIUS_ARCSEC * u.arcsec,
                                   colRA1="ra", colDec1="dec")
                matched |= set(res.to_pandas()["__id"].astype(int).tolist())
                break
            except Exception as e:
                if attempt == 2:
                    failed += 1
                    print(f"    [{label}] chunk {i} FAILED after 3 tries: "
                          f"{type(e).__name__}: {str(e)[:70]}", flush=True)
                else:
                    time.sleep(5)
    if failed:
        print(f"    [{label}] WARNING: {failed} chunk(s) failed -- rate is a LOWER BOUND", flush=True)
    hit.loc[list(matched)] = True
    return hit, int(ok.sum())


def main():
    from scipy.stats import fisher_exact
    res = {"radius_arcsec": RADIUS_ARCSEC, "bayestar_dec_min": BAYESTAR_DEC_MIN}

    tr = pd.read_csv(TRAINING); tr["host"] = tr.host.astype(str)
    y = tr.label.to_numpy()
    n_coord = int(pd.to_numeric(tr.ra, errors="coerce").notna().sum())
    print(f"training {len(tr)} rows, {n_coord} with coordinates "
          f"({int((y==1).sum())} pos / {int((y==0).sum())} neg)\n", flush=True)

    P = pools()
    for k, v in P.items():
        nc = int(pd.to_numeric(v["dec"], errors="coerce").notna().sum())
        print(f"  pool {k}: {len(v)} rows, {nc} with coordinates", flush=True)

    # ---------------- (1) CLUSTER MEMBERSHIP ----------------
    print("\n=== CLUSTER / MOVING-GROUP MEMBERSHIP ===", flush=True)
    res["cluster"] = {}
    out = tr[["host", "label"]].copy()
    for name, cat_id in CLUSTER_CATALOGS:
        print(f"  {name} ({cat_id})", flush=True)
        hit, nok = xmatch(tr, cat_id, name)
        out[name] = hit.values
        pos = float(hit[y == 1].mean() * 100)
        neg = float(hit[y == 0].mean() * 100)
        tab = [[int((hit & (y == 1)).sum()), int((~hit & (y == 1)).sum())],
               [int((hit & (y == 0)).sum()), int((~hit & (y == 0)).sum())]]
        try:
            orr, pv = fisher_exact(tab)
        except Exception:
            orr, pv = float("nan"), float("nan")
        entry = {"train_pct": float(hit.mean() * 100), "train_pos_pct": pos,
                 "train_neg_pct": neg, "diff_pp": pos - neg,
                 "n_members_train": int(hit.sum()),
                 "n_members_pos": tab[0][0], "n_members_neg": tab[1][0],
                 "odds_ratio": float(orr), "fisher_p": float(pv),
                 "auc_membership": auc(y, hit.astype(float).values)}
        for k, v in P.items():
            h2, n2 = xmatch(v, cat_id, f"{name}/{k}")
            entry[f"pool_{k}_pct"] = float(h2.mean() * 100)
            entry[f"pool_{k}_n"] = int(h2.sum())
            entry[f"pool_{k}_total"] = int(len(v))
        res["cluster"][name] = entry
        print(f"    train {entry['train_pct']:.2f}%  (pos {pos:.2f}% / neg {neg:.2f}%, "
              f"{entry['diff_pp']:+.2f} pp)  AUC(flag) {entry['auc_membership']:.4f}", flush=True)
        for k in P:
            print(f"    pool {k}: {entry[f'pool_{k}_pct']:.2f}% "
                  f"({entry[f'pool_{k}_n']}/{entry[f'pool_{k}_total']})", flush=True)

    anyc = out[[n for n, _ in CLUSTER_CATALOGS]].any(axis=1)
    out["any_cluster"] = anyc
    res["cluster"]["UNION"] = {
        "train_pct": float(anyc.mean() * 100),
        "train_pos_pct": float(anyc[y == 1].mean() * 100),
        "train_neg_pct": float(anyc[y == 0].mean() * 100),
        "n_members_train": int(anyc.sum()),
        "auc_membership": auc(y, anyc.astype(float).values)}

    # ---------------- (2) 3D DUST, Bayestar19 footprint ----------------
    print("\n=== 3D DUST EXTINCTION (Bayestar19) ===", flush=True)
    dec = pd.to_numeric(tr.dec, errors="coerce")
    inmap = (dec > BAYESTAR_DEC_MIN)
    havec = dec.notna()
    d = {"train_pct_in_footprint": float((inmap & havec).sum() / len(tr) * 100),
         "train_pos_pct": float(inmap[(y == 1) & havec].mean() * 100),
         "train_neg_pct": float(inmap[(y == 0) & havec].mean() * 100),
         "auc_footprint_availability": auc(y[havec.values], inmap[havec].astype(float).values),
         "median_dec_pos": float(dec[(y == 1)].median()),
         "median_dec_neg": float(dec[(y == 0)].median())}
    tab = [[int((inmap & (y == 1) & havec).sum()), int((~inmap & (y == 1) & havec).sum())],
           [int((inmap & (y == 0) & havec).sum()), int((~inmap & (y == 0) & havec).sum())]]
    orr, pv = fisher_exact(tab)
    d["odds_ratio"] = float(orr); d["fisher_p"] = float(pv)
    d["diff_pp"] = d["train_pos_pct"] - d["train_neg_pct"]
    for k, v in P.items():
        dd = pd.to_numeric(v["dec"], errors="coerce")
        d[f"pool_{k}_pct"] = float((dd > BAYESTAR_DEC_MIN).sum() / max(dd.notna().sum(), 1) * 100)
        d[f"pool_{k}_n"] = int((dd > BAYESTAR_DEC_MIN).sum())
        d[f"pool_{k}_total"] = int(dd.notna().sum())
        d[f"pool_{k}_median_dec"] = float(dd.median()) if dd.notna().any() else float("nan")
    res["dust_bayestar19"] = d
    print(f"  training in footprint: {d['train_pct_in_footprint']:.2f}%  "
          f"(pos {d['train_pos_pct']:.2f}% / neg {d['train_neg_pct']:.2f}%, "
          f"{d['diff_pp']:+.2f} pp)  AUC(avail) {d['auc_footprint_availability']:.4f}", flush=True)
    for k in P:
        print(f"  pool {k}: {d[f'pool_{k}_pct']:.2f}% "
              f"({d[f'pool_{k}_n']}/{d[f'pool_{k}_total']}), median dec "
              f"{d[f'pool_{k}_median_dec']:+.1f}", flush=True)

    out.to_csv(OUT_CSV, index=False)
    json.dump(res, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\nsaved {OUT_JSON}\nsaved {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
