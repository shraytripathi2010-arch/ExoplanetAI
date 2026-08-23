"""coverage_gap_audit.py -- PART 2: does the training set actually cover the
regimes the candidate pools contain?

Not a modelling task. This characterises the CURRENT training set against the
CURRENT pools on five axes, using data already on disk, and asks one question
each time: is there a regime that real candidates occupy but training examples
do not?

The framing matters. "Training has few low-SNR rows" is not a gap by itself --
it is only a gap if the POOLS are full of them, because the pools are what the
deployed model actually scores. So every axis is reported as a paired
distribution, training vs pool, with the pool-heavy tail called out.

Reads only. Writes one JSON of results and touches nothing else.
"""
import os
import sys
import json
import importlib.util
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAIN = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CAT = os.path.join(ROOT, "data", "catalogs")
OUT = os.path.join(HERE, "coverage_gap_audit.json")

POOLS = [("main", "unknown_features.csv", "unknown_candidate_list.csv"),
         ("widesector", "unknown_features_widesector.csv",
          "unknown_candidate_list_widesector.csv")]


def qs(v, name):
    v = pd.to_numeric(v, errors="coerce").dropna()
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)),
            "p05": float(v.quantile(.05)), "p25": float(v.quantile(.25)),
            "median": float(v.median()), "p75": float(v.quantile(.75)),
            "p95": float(v.quantile(.95)), "min": float(v.min()), "max": float(v.max())}


def line(label, d):
    if not d.get("n"):
        return f"  {label:<22} (none)"
    return (f"  {label:<22}n={d['n']:<6}p05 {d['p05']:>9.4g}  p25 {d['p25']:>9.4g}  "
            f"med {d['median']:>9.4g}  p75 {d['p75']:>9.4g}  p95 {d['p95']:>9.4g}")


def main():
    res = {}
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)

    tr = pd.read_csv(TRAIN); tr["host"] = tr.host.astype(str)
    pos, neg = tr[tr.label == 1], tr[tr.label == 0]
    print(f"training {len(tr)} rows -- {len(pos)} positive / {len(neg)} negative")

    pool = []
    for tag, ff, cf in POOLS:
        p = pd.read_csv(os.path.join(CAT, ff))
        c = pd.read_csv(os.path.join(CAT, cf))
        p["host"] = p.host.astype(str)
        c["host"] = c["host"].astype(str) if "host" in c.columns \
            else "TIC_" + c["tic_id"].astype(str)
        keep = [k for k in ("st_rad", "st_teff") if k in c.columns]
        p = p.merge(c.drop_duplicates("host")[["host"] + keep], on="host",
                    how="left", suffixes=("", "_c"))
        for k in keep:
            if f"{k}_c" in p.columns:
                p[k] = pd.to_numeric(p.get(k), errors="coerce").fillna(
                    pd.to_numeric(p[f"{k}_c"], errors="coerce"))
        if "status" in p.columns:
            p = p[p.status.astype(str).str.startswith("Success")]
        p["_pool"] = tag
        pool.append(p)
    pool = pd.concat(pool, ignore_index=True)
    print(f"pools    {len(pool)} scored candidates\n")
    res["n"] = {"train": len(tr), "pos": len(pos), "neg": len(neg), "pool": len(pool)}

    # ---------------- 1. SIGNAL-TO-NOISE ----------------
    print("=" * 78); print("1. SIGNAL-TO-NOISE COVERAGE  (SDE and snr)"); print("=" * 78)
    for col in ("SDE", "snr"):
        print(f"  [{col}]")
        d = {"pos": qs(pos[col], col), "neg": qs(neg[col], col),
             "pool": qs(pool[col], col)}
        print(line("training positive", d["pos"]))
        print(line("training negative", d["neg"]))
        print(line("candidate pool", d["pool"]))
        res.setdefault("snr", {})[col] = d
        # the actionable number: what fraction of the POOL sits below the 5th
        # percentile of training, i.e. in a regime training barely covers
        tr_p05 = np.nanpercentile(pd.to_numeric(tr[col], errors="coerce").dropna(), 5)
        pv = pd.to_numeric(pool[col], errors="coerce").dropna()
        frac = float((pv < tr_p05).mean())
        print(f"    pool fraction BELOW training's 5th pct ({tr_p05:.3g}): {frac:.2%}")
        res["snr"][col]["pool_below_train_p05"] = frac
        print()

    # ---------------- 2. DEPTH / PLANET RADIUS ----------------
    print("=" * 78); print("2. TRANSIT DEPTH / PLANET-RADIUS COVERAGE"); print("=" * 78)
    for col in ("depth_mean", "rp_rs"):
        print(f"  [{col}]")
        d = {"pos": qs(pos[col], col), "neg": qs(neg[col], col), "pool": qs(pool[col], col)}
        print(line("training positive", d["pos"]))
        print(line("training negative", d["neg"]))
        print(line("candidate pool", d["pool"]))
        res.setdefault("depth", {})[col] = d
        trv = pd.to_numeric(tr[col], errors="coerce").dropna()
        pv = pd.to_numeric(pool[col], errors="coerce").dropna()
        lo, hi = np.nanpercentile(trv, 5), np.nanpercentile(trv, 95)
        below, above = float((pv < lo).mean()), float((pv > hi).mean())
        print(f"    pool below training p05 ({lo:.4g}): {below:.2%}   "
              f"above training p95 ({hi:.4g}): {above:.2%}")
        res["depth"][col].update(pool_below_p05=below, pool_above_p95=above)
        print()
    # ultra-short period
    trp = pd.to_numeric(tr.period, errors="coerce")
    plp = pd.to_numeric(pool.period, errors="coerce")
    for thr in (0.5, 1.0):
        a, b = float((trp < thr).mean()), float((plp < thr).mean())
        print(f"  period < {thr} d:  training {a:.2%} ({int((trp<thr).sum())})   "
              f"pool {b:.2%} ({int((plp<thr).sum())})")
        res.setdefault("ultra_short", {})[f"lt_{thr}d"] = {"train": a, "pool": b,
            "train_n": int((trp < thr).sum()), "pool_n": int((plp < thr).sum())}

    # ---------------- 3. HOST VARIABILITY ----------------
    print("\n" + "=" * 78); print("3. HOST-VARIABILITY COVERAGE  (deployed var_* features)"); print("=" * 78)
    for col in ("var_ls_power", "var_excess", "var_oot_rms"):
        d = {"pos": qs(pos[col], col), "neg": qs(neg[col], col), "pool": qs(pool[col], col)}
        print(f"  [{col}]")
        print(line("training positive", d["pos"]))
        print(line("training negative", d["neg"]))
        print(line("candidate pool", d["pool"]))
        trv = pd.to_numeric(tr[col], errors="coerce").dropna()
        pv = pd.to_numeric(pool[col], errors="coerce").dropna()
        hi = np.nanpercentile(trv, 95)
        above = float((pv > hi).mean())
        print(f"    pool ABOVE training p95 ({hi:.4g}): {above:.2%}")
        res.setdefault("variability", {})[col] = dict(d, pool_above_p95=above)
        print()

    # ---------------- 4. MULTI-PLANET SYSTEMS ----------------
    print("=" * 78); print("4. MULTI-PLANET SYSTEM REPRESENTATION"); print("=" * 78)
    cp = pd.read_csv(os.path.join(CAT, "confirmed_planets.csv"))
    per_host = cp.groupby("hostname")["pl_name"].nunique()
    multi_hosts = set(per_host[per_host >= 2].index)
    print(f"  confirmed-planet catalog: {len(per_host)} hosts, "
          f"{len(multi_hosts)} with >=2 planets ({len(multi_hosts)/len(per_host):.1%})")
    # training positives are keyed by TIC; match via ra/dec against the catalog
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    cp2 = cp.dropna(subset=["ra", "dec"]).copy()
    cp2["_multi"] = cp2.hostname.isin(multi_hosts)
    cpc = SkyCoord(cp2.ra.values * u.deg, cp2.dec.values * u.deg)
    pr = pos.dropna(subset=["ra", "dec"])
    prc = SkyCoord(pd.to_numeric(pr.ra).values * u.deg, pd.to_numeric(pr.dec).values * u.deg)
    idx, d2d, _ = prc.match_to_catalog_sky(cpc)
    ok = d2d.arcsec < 5.0
    matched = int(ok.sum())
    ismulti = cp2._multi.values[idx][ok]
    print(f"  training positives matched to the catalog within 5\": "
          f"{matched}/{len(pr)} ({matched/len(pr):.1%})")
    print(f"  of those, hosts with >=2 known planets: {int(ismulti.sum())} "
          f"({ismulti.mean():.1%})")
    print(f"  catalog baseline for comparison:        {len(multi_hosts)/len(per_host):.1%}")
    res["multiplanet"] = {"catalog_hosts": int(len(per_host)),
                          "catalog_multi": int(len(multi_hosts)),
                          "catalog_multi_frac": float(len(multi_hosts) / len(per_host)),
                          "train_pos_matched": matched,
                          "train_pos_multi_frac": float(ismulti.mean()) if matched else None}

    # ---------------- 5. NEGATIVE-CLASS COMPOSITION ----------------
    print("\n" + "=" * 78); print("5. NEGATIVE-CLASS COMPOSITION  (Gaia-flagged binaries)"); print("=" * 78)
    def gaia_profile(df, label):
        ru = pd.to_numeric(df.gaia_ruwe, errors="coerce")
        ns = pd.to_numeric(df.gaia_nss, errors="coerce")
        have = ru.notna() | ns.notna()
        hi = (ru > 1.4)
        nss = (ns > 0)
        either = hi.fillna(False) | nss.fillna(False)
        d = {"n": int(len(df)), "gaia_coverage": float(have.mean()),
             "ruwe_gt_1.4": float(hi.sum() / max(ru.notna().sum(), 1)),
             "nss_positive": float(nss.sum() / max(ns.notna().sum(), 1)),
             "either_flag": float(either.sum() / max(have.sum(), 1)),
             "n_either": int(either.sum())}
        print(f"  {label:<22} n={d['n']:<6} gaia cov {d['gaia_coverage']:.1%}  "
              f"RUWE>1.4 {d['ruwe_gt_1.4']:.1%}  NSS>0 {d['nss_positive']:.1%}  "
              f"either {d['either_flag']:.1%} ({d['n_either']})")
        return d
    res["negclass"] = {"train_pos": gaia_profile(pos, "training positive"),
                       "train_neg": gaia_profile(neg, "training NEGATIVE"),
                       "pool": gaia_profile(pool, "candidate pool")}

    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
