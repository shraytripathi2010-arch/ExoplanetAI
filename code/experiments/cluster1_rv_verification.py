"""cluster1_rv_verification.py -- does cluster 1 really hold non-transiting
planets? READ-ONLY verification of the SOM diagnostic's hypothesis.

Nothing is modified: not training.csv, not the frozen split, not the model.

THE HYPOTHESIS UNDER TEST. The SOM diagnostic found cluster 1 to be the
worst-performing region (AUC 0.8280 vs median 0.9264, ECE 0.0934), with
FAP ~240x the population median, transit_count 4 vs 12 and period 6.4 d vs
2.4 d, and 3.4x over-represented in the real candidate pool. Its positives
included 51 Peg, 11 UMi, 24 Sex, 4 UMa, BD+20 2457 -- names associated with
RADIAL-VELOCITY discoveries. The hypothesis: these rows are labelled positive
because the HOST has a known planet, while the TESS photometry contains no
transit to find, making them label noise for a transit classifier.

THRESHOLD DECLARED BEFORE LOOKING, per the task: "meaningful" = **>30% of
cluster 1's positives confirmed non-transiting**. Below that the hypothesis is
not supported as a material data-quality issue, however suggestive the names.

REPRODUCING THE ASSIGNMENT. The diagnostic did NOT persist per-star cluster
labels -- only the count (444). Every step is seeded (SEED=42 for the
QuantileTransformer, the SOM init and its training permutations;
AgglomerativeClustering is deterministic), so re-running the identical pipeline
reproduces the identical partition. This is verified rather than assumed: the
run aborts unless it recovers cluster 1 with n=532 and 444 positives, the
numbers already on record.

AUTHORITATIVE SOURCE. NASA Exoplanet Archive `pscomppars` (one row per planet,
default parameter set), fields:
    discoverymethod  -- 'Radial Velocity', 'Transit', 'Imaging', ...
    tran_flag        -- 1 if the planet is observed to transit, 0 otherwise
`tran_flag` is the decisive field: it separates "RV-discovered but does
transit" from "RV-discovered and does NOT transit", which discovery method
alone cannot.
"""
import os
import sys
import json
import importlib.util
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, SCRIPT_DIR)

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT = os.path.join(SCRIPT_DIR, "cluster1_rv_verification.json")
CSV = os.path.join(SCRIPT_DIR, "cluster1_rv_verification.csv")
CACHE = os.path.join(SCRIPT_DIR, "nasa_pscomppars_cache.csv")

SEED = 42
EXPECT_N, EXPECT_POS = 532, 444
MEANINGFUL = 0.30


def norm(name):
    """training.csv hosts use underscores for spaces: '51_Peg' -> '51 peg'."""
    s = str(name).replace("_", " ").strip().lower()
    return " ".join(s.split())


def fetch_archive():
    if os.path.exists(CACHE):
        print(f"  using cached archive table: {os.path.relpath(CACHE, ROOT)}")
        return pd.read_csv(CACHE)
    print("  querying NASA Exoplanet Archive (pscomppars)...", flush=True)
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    t = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="hostname,pl_name,discoverymethod,tran_flag,rv_flag,"
               "pl_orbper,pl_orbincl,pl_imppar,sy_dist")
    df = t.to_pandas()
    df.to_csv(CACHE, index=False)
    print(f"  fetched {len(df)} planet rows; cached")
    return df


def main():
    print("=" * 96)
    print("CLUSTER 1 VERIFICATION -- are these non-transiting RV discoveries?")
    print("=" * 96)
    print(f"  threshold declared up front: hypothesis supported only if "
          f">{MEANINGFUL*100:.0f}% of cluster 1 positives are non-transiting\n")

    import som_cluster_diagnostic as S
    m05 = S._m05()
    cols = list(m05.FEATURE_COLUMNS)
    df = pd.read_csv(TRAINING)
    y = df["label"].astype(int).to_numpy()
    te = m05.frozen_test_mask(df)

    # --- reproduce the exact partition ---
    imp = SimpleImputer(strategy="median")
    sc = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             random_state=SEED)
    X = sc.fit_transform(imp.fit_transform(S.prep(df, cols)))
    som = S.SOM(6, X.shape[1]).fit(X)
    bmu, _ = som.bmu(X)
    sup = AgglomerativeClustering(n_clusters=8).fit(som.W).labels_[bmu]

    m1 = sup == 1
    n1, pos1 = int(m1.sum()), int((m1 & (y == 1)).sum())
    print(f"  reproduced cluster 1: n={n1}, positives={pos1} "
          f"(recorded: {EXPECT_N}, {EXPECT_POS})")
    assert (n1, pos1) == (EXPECT_N, EXPECT_POS), (
        f"partition did not reproduce ({n1},{pos1}) vs ({EXPECT_N},{EXPECT_POS}) "
        "-- do not proceed on a different partition")
    print("  partition reproduced exactly; proceeding\n")

    idx = np.where(m1 & (y == 1))[0]
    hosts = df.iloc[idx]["host"].astype(str).tolist()
    in_test = te[idx]

    arch = fetch_archive()
    arch["_h"] = arch["hostname"].map(norm)
    # a host may have several planets; treat the HOST as transiting if ANY of
    # its planets transits, which is the conservative direction for the
    # hypothesis (it makes "non-transiting" harder to claim).
    g = arch.groupby("_h").agg(
        any_transit=("tran_flag", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).max())),
        n_planets=("pl_name", "size"),
        methods=("discoverymethod", lambda s: "|".join(sorted(set(s.dropna())))),
        min_incl=("pl_orbincl", lambda s: pd.to_numeric(s, errors="coerce").min()),
        min_imppar=("pl_imppar", lambda s: pd.to_numeric(s, errors="coerce").min()),
    ).reset_index()
    lut = g.set_index("_h")

    rows = []
    for h, t in zip(hosts, in_test):
        k = norm(h)
        if k in lut.index:
            r = lut.loc[k]
            rows.append({"host": h, "in_frozen_test": bool(t), "matched": True,
                         "methods": r["methods"], "n_planets": int(r["n_planets"]),
                         "any_transit": int(r["any_transit"]),
                         "min_incl": float(r["min_incl"]) if pd.notna(r["min_incl"]) else None,
                         "min_imppar": float(r["min_imppar"]) if pd.notna(r["min_imppar"]) else None})
        else:
            rows.append({"host": h, "in_frozen_test": bool(t), "matched": False,
                         "methods": None, "n_planets": 0, "any_transit": None,
                         "min_incl": None, "min_imppar": None})
    R = pd.DataFrame(rows)
    R.to_csv(CSV, index=False)

    n = len(R)
    matched = R[R["matched"]]
    print("=" * 96)
    print("PART 1 -- discovery method")
    print("=" * 96)
    print(f"  cluster 1 positives: {n}")
    print(f"  matched in NASA archive: {len(matched)} ({len(matched)/n*100:.1f}%)"
          f"   unmatched: {n-len(matched)}")
    if len(matched):
        transit_only = matched["methods"].str.contains("Transit", na=False)
        print(f"\n  hosts whose method set INCLUDES Transit : {int(transit_only.sum())} "
              f"({transit_only.mean()*100:.1f}% of matched)")
        print(f"  hosts with NO transit discovery         : {int((~transit_only).sum())} "
              f"({(~transit_only).mean()*100:.1f}% of matched)")
        print("\n  method-set breakdown (top 10):")
        for meth, c in matched["methods"].value_counts().head(10).items():
            print(f"    {str(meth)[:52]:<54}{c:>5}")

    print("\n" + "=" * 96)
    print("PART 2 -- does the host actually transit? (tran_flag)")
    print("=" * 96)
    if len(matched):
        nt = matched[matched["any_transit"] == 0]
        tr = matched[matched["any_transit"] == 1]
        print(f"  transits (tran_flag=1 on >=1 planet) : {len(tr)} "
              f"({len(tr)/len(matched)*100:.1f}% of matched)")
        print(f"  does NOT transit (all tran_flag=0)   : {len(nt)} "
              f"({len(nt)/len(matched)*100:.1f}% of matched)")
        frac_of_cluster = len(nt) / n
        print(f"\n  as a fraction of ALL {n} cluster-1 positives: "
              f"{frac_of_cluster*100:.1f}%")
        print(f"  declared threshold: >{MEANINGFUL*100:.0f}%  -> "
              f"{'SUPPORTED' if frac_of_cluster > MEANINGFUL else 'NOT SUPPORTED'}")

        # geometry sanity check on a handful
        print("\n  geometry sanity check (non-transiting subset, first 8 with data):")
        sub = nt[nt["min_incl"].notna()].head(8)
        if len(sub):
            for _, r in sub.iterrows():
                print(f"    {r['host'][:26]:<28} min inclination {r['min_incl']:.1f} deg"
                      f"   (90 deg = edge-on; far from 90 => geometrically cannot transit)")
        else:
            print("    no inclination values available in the archive for this subset")

        print("\n" + "=" * 96)
        print("PART 3 -- scale, and TEST-SET exposure")
        print("=" * 96)
        n_pos_total = int((y == 1).sum())
        print(f"  non-transiting rows: {len(nt)}")
        print(f"  as fraction of the FULL positive class ({n_pos_total}): "
              f"{len(nt)/n_pos_total*100:.2f}%")
        print(f"  as fraction of all {len(df)} training rows: "
              f"{len(nt)/len(df)*100:.2f}%")
        nt_test = int(nt["in_frozen_test"].sum())
        print(f"\n  of those, in the FROZEN TEST SET: {nt_test} "
              f"({nt_test/max(len(nt),1)*100:.1f}% of the non-transiting subset)")
        print(f"  frozen test set size: {int(te.sum())}; so they are "
              f"{nt_test/int(te.sum())*100:.2f}% of the test set")

        res = {
            "cluster1_positives": n, "matched": int(len(matched)),
            "unmatched": int(n - len(matched)),
            "transits": int(len(tr)), "non_transiting": int(len(nt)),
            "non_transiting_frac_of_cluster1": float(frac_of_cluster),
            "threshold": MEANINGFUL,
            "supported": bool(frac_of_cluster > MEANINGFUL),
            "n_positive_class_total": n_pos_total,
            "non_transiting_frac_of_positive_class": float(len(nt) / n_pos_total),
            "non_transiting_in_frozen_test": nt_test,
            "frozen_test_size": int(te.sum()),
            "non_transiting_pct_of_test": float(nt_test / int(te.sum()) * 100),
            "method_breakdown": {str(k): int(v) for k, v in
                                 matched["methods"].value_counts().head(20).items()},
            "non_transiting_hosts_sample": nt["host"].tolist()[:80],
        }
    else:
        res = {"cluster1_positives": n, "matched": 0,
               "error": "no hosts matched the archive"}

    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nSaved {OUT}\nSaved {CSV}")


if __name__ == "__main__":
    main()
