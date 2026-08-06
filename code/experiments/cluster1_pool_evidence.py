"""cluster1_pool_evidence.py -- do cluster-1 pool candidates carry elevated
false-positive evidence despite high model confidence? READ-ONLY.

Nothing is modified: not training.csv, not the pools, not the model, not the
frozen split, not production code.

THE HYPOTHESIS, from the RV-discovery verification. The model assigns ~0.96
median probability to training positives that, by definition, cannot transit --
it has learned that this class of feature vector means "planet". Benign on
labelled data (those rows really do host planets). The worry is that the same
behaviour on UNKNOWN candidates would manufacture false positives, and cluster 1
is over-represented in the real pool.

PREDICTION, stated before measuring: cluster-1 pool candidates should (a) be
scored at least as confidently as the rest of the pool, and (b) trip independent
false-positive evidence MORE often. Both must hold. High confidence alone is not
a problem; high confidence combined with independent red flags is.

THE INDEPENDENT EVIDENCE IS ALREADY COMPUTED -- verified before designing around
it, rather than assumed:
  vsx_code        AAVSO Variable Star Index cross-match. ALREADY WIRED; no new
                  external query needed. Pool A baseline: 102 HIT / 194 NO_HIT.
  blending_status Gaia neighbour analysis, parsed here into HIGH / LOW-MODERATE
                  / UNKNOWN / clean.
  crowd_*         TIC-derived contamination proxies.
  exofop_code     ALL 296 rows are NO_HIT -- the candidates are TOI-free by
                  construction, so this layer returns nothing. Confirmed, not
                  assumed.
  odd_even_mismatch, secondary_eclipse_depth, var_* -- EB/variability
                  signatures. None cleared the model-improvement bar on their
                  own, but they retain diagnostic value as a cross-check.

CLUSTER ASSIGNMENT uses the SAME fitted partition as the SOM diagnostic: the SOM
and the agglomeration are fitted on the LABELLED population only, then pool rows
are mapped to their best-matching unit. Re-fitting on labelled+unlabelled would
be a different partition and would not answer the question asked.
"""
import os
import sys
import json
import importlib.util
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, SCRIPT_DIR)

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT = os.path.join(SCRIPT_DIR, "cluster1_pool_evidence.json")
SEED = 42
POOLS = [
    ("pool A", "unknown_features.csv", "unknown_candidates"),
    ("pool B (widesector)", "unknown_features_widesector.csv",
     "unknown_candidates_widesector"),
]


def blend_risk(s):
    if not isinstance(s, str):
        return "missing"
    t = s.upper()
    if t.startswith("HIGH"):
        return "HIGH"
    if t.startswith("LOW-MODERATE") or t.startswith("MODERATE"):
        return "LOW-MODERATE"
    if t.startswith("UNKNOWN"):
        return "UNKNOWN"
    return "clean"


def rate_test(name, a_hit, a_n, b_hit, b_n):
    """Fisher exact on a 2x2: cluster1 vs rest."""
    if min(a_n, b_n) == 0:
        return None
    tab = [[a_hit, a_n - a_hit], [b_hit, b_n - b_hit]]
    odds, p = fisher_exact(tab)
    ra, rb = a_hit / a_n, b_hit / b_n
    print(f"    {name:<28} cluster1 {a_hit:>3}/{a_n:<4} ({ra*100:5.1f}%)   "
          f"rest {b_hit:>3}/{b_n:<4} ({rb*100:5.1f}%)   OR {odds:5.2f}  p={p:.4f}"
          f"{'  *' if p < 0.05 else ''}")
    return {"metric": name, "c1_hit": int(a_hit), "c1_n": int(a_n),
            "rest_hit": int(b_hit), "rest_n": int(b_n), "c1_rate": float(ra),
            "rest_rate": float(rb), "odds_ratio": float(odds), "p": float(p)}


def main():
    print("=" * 100)
    print("CLUSTER-1 POOL CANDIDATES vs INDEPENDENT VETTING EVIDENCE -- read-only")
    print("=" * 100)
    import som_cluster_diagnostic as S
    m05 = S._m05()
    cols = list(m05.FEATURE_COLUMNS)
    df = pd.read_csv(TRAINING)
    y = df["label"].astype(int).to_numpy()

    imp = SimpleImputer(strategy="median")
    sc = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             random_state=SEED)
    Xtr = sc.fit_transform(imp.fit_transform(S.prep(df, cols)))
    som = S.SOM(6, Xtr.shape[1]).fit(Xtr)
    bmu, _ = som.bmu(Xtr)
    node2super = AgglomerativeClustering(n_clusters=8).fit(som.W).labels_
    sup_tr = node2super[bmu]
    n1, pos1 = int((sup_tr == 1).sum()), int(((sup_tr == 1) & (y == 1)).sum())
    assert (n1, pos1) == (532, 444), f"partition drift: {(n1,pos1)}"
    print(f"  partition reproduced exactly (cluster 1: n=532, 444 positives)")
    print(f"  training share of cluster 1: {(sup_tr==1).mean()*100:.1f}%\n")

    prod = joblib.load(PROD := os.path.join(ROOT, "models", "best_model.joblib"))
    res = {"training_cluster1_share_pct": float((sup_tr == 1).mean() * 100),
           "pools": {}}

    for pname, feat_file, tag in POOLS:
        print("=" * 100)
        print(f"{pname}")
        print("=" * 100)
        fp = os.path.join(ROOT, "data", "catalogs", feat_file)
        rp = os.path.join(ROOT, "results", tag, "ranked_candidates.csv")
        cp = os.path.join(ROOT, "results", tag, "characterized_candidates.csv")
        if not (os.path.exists(fp) and os.path.exists(rp)):
            print("  required files missing; skipped\n")
            continue
        f = pd.read_csv(fp)
        r = pd.read_csv(rp)
        d = f.merge(r[[c for c in ["host", "st_rad", "st_teff"] if c in r.columns]],
                    on="host", how="inner")
        P = pd.to_numeric(d.get("period"), errors="coerce")
        T = pd.to_numeric(d.get("duration"), errors="coerce")
        d = d[(P > 0) & (T > 0)].reset_index(drop=True)
        miss = [c for c in cols if c not in d.columns]
        if miss or len(d) < 10:
            print(f"  unusable (n={len(d)}, missing={miss})\n")
            continue

        Xp = sc.transform(imp.transform(S.prep(d, cols)))
        pb, _ = som.bmu(Xp)
        d["cluster"] = node2super[pb]
        d["prob"] = prod.predict_proba(
            S.prep(d, cols).replace([np.inf, -np.inf], np.nan))[:, 1]
        c1 = d["cluster"] == 1
        print(f"  {len(d)} scorable candidates with full features")
        print(f"  in cluster 1: {int(c1.sum())} ({c1.mean()*100:.1f}%)   "
              f"vs {(sup_tr==1).mean()*100:.1f}% of training "
              f"-> {c1.mean()/((sup_tr==1).mean()):.2f}x over-representation")

        pe = {"n": int(len(d)), "n_cluster1": int(c1.sum()),
              "pct_cluster1": float(c1.mean() * 100),
              "over_representation": float(c1.mean() / (sup_tr == 1).mean())}

        # --- PART 1.3 probability distribution ---
        if c1.sum() and (~c1).sum():
            a, b = d.loc[c1, "prob"], d.loc[~c1, "prob"]
            u = mannwhitneyu(a, b)
            print(f"\n  predicted probability:")
            print(f"    cluster 1 : median {a.median():.4f}  mean {a.mean():.4f}  "
                  f">=0.9: {(a>=0.9).mean()*100:.1f}%")
            print(f"    rest      : median {b.median():.4f}  mean {b.mean():.4f}  "
                  f">=0.9: {(b>=0.9).mean()*100:.1f}%")
            print(f"    Mann-Whitney p={u.pvalue:.4g}")
            pe["prob"] = {"c1_median": float(a.median()), "rest_median": float(b.median()),
                          "c1_mean": float(a.mean()), "rest_mean": float(b.mean()),
                          "c1_ge90_pct": float((a >= 0.9).mean() * 100),
                          "rest_ge90_pct": float((b >= 0.9).mean() * 100),
                          "mw_p": float(u.pvalue)}

        # --- PART 2 evidence ---
        tests = []
        if os.path.exists(cp):
            ch = pd.read_csv(cp)
            keep = [c for c in ["host", "vsx_code", "blending_status", "exofop_code",
                                "rv_code", "confidence_tier"] if c in ch.columns]
            m = d.merge(ch[keep], on="host", how="left")
            mc1 = m["cluster"] == 1
            print(f"\n  merged evidence for {int(m['vsx_code'].notna().sum())}"
                  f"/{len(m)} candidates")
            print("\n  INDEPENDENT EVIDENCE, cluster 1 vs rest of pool:")
            if "vsx_code" in m:
                v = m["vsx_code"].fillna("")
                t = rate_test("VSX variable-star HIT",
                              int(((v == "HIT") & mc1).sum()), int((v != "").astype(bool)[mc1].sum()),
                              int(((v == "HIT") & ~mc1).sum()), int((v != "").astype(bool)[~mc1].sum()))
                if t: tests.append(t)
            if "blending_status" in m:
                br = m["blending_status"].map(blend_risk)
                t = rate_test("Gaia blend risk = HIGH",
                              int(((br == "HIGH") & mc1).sum()), int((br != "missing")[mc1].sum()),
                              int(((br == "HIGH") & ~mc1).sum()), int((br != "missing")[~mc1].sum()))
                if t: tests.append(t)
            if "exofop_code" in m:
                nz = m["exofop_code"].dropna()
                print(f"    ExoFOP/TFOP: {nz.value_counts().to_dict()} "
                      f"-- candidates are TOI-free by construction, nothing to compare")
            pe["evidence_tests"] = tests
        else:
            print(f"\n  no characterized_candidates.csv for this pool -- "
                  f"evidence layer unavailable")
            pe["evidence_tests"] = None

        # --- PART 2.3 in-pipeline FP-like signatures ---
        print("\n  in-pipeline FP-like signatures (median, cluster 1 vs rest):")
        sig = {}
        for c in ["odd_even_mismatch", "secondary_eclipse_depth", "crowd_flux_ratio_max",
                  "crowd_nearest_arcsec", "var_excess", "var_ls_power", "SDE", "FAP"]:
            if c not in d.columns:
                continue
            a = pd.to_numeric(d.loc[c1, c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            b = pd.to_numeric(d.loc[~c1, c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            p = mannwhitneyu(a, b).pvalue
            print(f"    {c:<24} c1 {a.median():>11.5g}   rest {b.median():>11.5g}   "
                  f"p={p:.4f}{'  *' if p < 0.05 else ''}")
            sig[c] = {"c1_median": float(a.median()), "rest_median": float(b.median()),
                      "p": float(p)}
        pe["signatures"] = sig
        res["pools"][pname] = pe
        print()

    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
