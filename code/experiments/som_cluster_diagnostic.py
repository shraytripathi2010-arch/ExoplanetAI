"""som_cluster_diagnostic.py -- READ-ONLY unsupervised structure diagnostic.

Nothing here trains, modifies or promotes anything in production. It reads
training.csv, the candidate pools and the deployed model, and writes one JSON.

WHAT RAVEN ACTUALLY DOES WITH SOMs, confirmed from the paper rather than
assumed: the SOM is "trained to distinguish between Planets and FPs based on
TRANSIT SHAPE" and its output is a METRIC FED INTO the downstream XGBoost/GP
classifiers -- i.e. a feature generator following Armstrong et al. (2017), not
a standalone diagnostic. This pass deliberately does NOT copy that: building a
SOM feature would be a training-pipeline change, which this task excludes.

FEATURE SPACE -- the choice matters and is made explicitly:

  PRIMARY: all 31 production features. The question asked is "what is the
  classifier missing", and the classifier operates in exactly this space. A
  group that looks exotic in shape-space but is cleanly separated in the full
  space is not a blind spot. Clustering anywhere else would answer a different
  question.

  SECONDARY: the shape-only subset, to stay faithful to what RAVEN clusters on
  and to see whether shape-space says anything the full space does not.

VALIDITY CHECK, and why the obvious version would be near-tautological.
`st_rad` is one of the 31 features, so "a cluster enriched in large-radius
stars" would emerge almost by construction and would prove nothing. The
giant-star investigation's actual finding was sharper: giants and dwarfs have
NEARLY IDENTICAL RANKING (AUC 0.9013 vs 0.9017) but very different CALIBRATION
(ECE 0.0867 vs 0.0204). So the real test is whether an unsupervised partition
recovers a group with that specific signature -- elevated ECE at ordinary AUC.
That is a genuine test of whether the method finds structure that matters.

SOM: implemented in ~40 lines of numpy rather than adding a dependency
(minisom/somoclu/sklearn_som are all absent). Standard online Kohonen training
with Gaussian neighbourhood and linear decay. Node weights are then
agglomerated into a small number of superclusters so per-group sample sizes are
large enough to say anything.
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
POOLS = [
    ("pool A", os.path.join(ROOT, "data", "catalogs", "unknown_features.csv"),
     os.path.join(ROOT, "results", "unknown_candidates", "ranked_candidates.csv")),
    ("pool B (widesector)",
     os.path.join(ROOT, "data", "catalogs", "unknown_features_widesector.csv"),
     os.path.join(ROOT, "results", "unknown_candidates_widesector",
                  "ranked_candidates.csv")),
]
OUT = os.path.join(SCRIPT_DIR, "som_cluster_diagnostic.json")

SEED = 42
GRID = 6            # 6x6 = 36 nodes
N_SUPER = 8         # agglomerated superclusters
N_EPOCH = 40
SHAPE_FEATURES = ["duration", "depth", "depth_mean", "depth_mean_std",
                  "depth_mean_even", "depth_mean_odd", "odd_even_mismatch",
                  "rp_rs", "secondary_eclipse_depth", "transit_shape_ratio",
                  "depth_duration_ratio", "depth_consistency_std"]


def ece(y, p, bins=15):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = np.linspace(0, 1, bins + 1)
    i = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    return float(sum((i == b).mean() * abs(y[i == b].mean() - p[i == b].mean())
                     for b in range(bins) if (i == b).any()))


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


class SOM:
    """Minimal Kohonen SOM. Rectangular grid, Gaussian neighbourhood."""

    def __init__(self, g, dim, seed=SEED):
        rng = np.random.RandomState(seed)
        self.g = g
        self.W = rng.normal(0, 0.5, size=(g * g, dim))
        gx, gy = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        self.pos = np.c_[gx.ravel(), gy.ravel()].astype(float)

    def bmu(self, X):
        d = ((X[:, None, :] - self.W[None, :, :]) ** 2).sum(-1)
        return d.argmin(1), np.sqrt(d.min(1))

    def fit(self, X, epochs=N_EPOCH, seed=SEED):
        rng = np.random.RandomState(seed)
        n = len(X)
        s0, lr0 = self.g / 2.0, 0.5
        for ep in range(epochs):
            frac = ep / max(epochs - 1, 1)
            sigma = max(s0 * (1 - frac), 0.6)
            lr = lr0 * (1 - frac) + 0.01
            for i in rng.permutation(n):
                x = X[i]
                b = ((x - self.W) ** 2).sum(1).argmin()
                d2 = ((self.pos - self.pos[b]) ** 2).sum(1)
                h = np.exp(-d2 / (2 * sigma ** 2))[:, None]
                self.W += lr * h * (x - self.W)
        return self


def prep(df, cols):
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


def analyse(name, labels, y, prob, giant, res, n_min=40):
    """Per-cluster composition, ranking and calibration."""
    print(f"\n  {'cluster':<9}{'n':>6}{'planet%':>9}{'giant%':>8}"
          f"{'AUC':>8}{'ECE':>8}{'err@0.5':>9}")
    rows = []
    for c in sorted(set(labels)):
        m = labels == c
        n = int(m.sum())
        pf = float(y[m].mean() * 100)
        gf = float(giant[m].mean() * 100)
        if n >= n_min and 0 < y[m].sum() < n:
            a = float(roc_auc_score(y[m], prob[m]))
            e = ece(y[m], prob[m])
            err = float(((prob[m] >= 0.5).astype(int) != y[m]).mean() * 100)
        else:
            a = e = err = float("nan")
        rows.append({"cluster": int(c), "n": n, "planet_pct": pf,
                     "giant_pct": gf, "auc": a, "ece": e, "err_pct": err})
        print(f"  {c:<9}{n:>6}{pf:>9.1f}{gf:>8.1f}"
              f"{a:>8.4f}{e:>8.4f}{err:>9.1f}" if np.isfinite(a) else
              f"  {c:<9}{n:>6}{pf:>9.1f}{gf:>8.1f}{'--':>8}{'--':>8}{'--':>9}")
    res[name] = rows
    return rows


def main():
    print("=" * 92)
    print("SOM / UNSUPERVISED CLUSTER DIAGNOSTIC -- read-only")
    print("=" * 92)
    m05 = _m05()
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 31
    df = pd.read_csv(TRAINING)
    y = df["label"].astype(int).to_numpy()
    te = m05.frozen_test_mask(df)
    giant = (pd.to_numeric(df["st_rad"], errors="coerce") >= 1.5).fillna(False).to_numpy()

    prod = joblib.load(PROD)
    # Use production's OWN matrix builder, not raw column selection: training.csv
    # carries inf in a few columns (period_uncertainty, FAP) and the imputer
    # rejects non-finite input. build_feature_matrix is the path production
    # actually uses, so scoring through it matches deployment exactly.
    Xprod, _ = m05.build_feature_matrix(df)
    Xprod = Xprod.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    prob = prod.predict_proba(Xprod)[:, 1]
    print(f"  {len(df)} labelled rows | frozen test {int(te.sum())} "
          f"| giants (st_rad>=1.5) {int(giant.sum())}")
    print(f"  deployed model scored read-only; per-cluster AUC/ECE reported on the")
    print(f"  FROZEN TEST SUBSET only, where the model is genuinely out-of-sample.")

    res = {"n_labelled": int(len(df)), "n_test": int(te.sum()),
           "n_giants": int(giant.sum())}

    imp = SimpleImputer(strategy="median")
    # QuantileTransformer, NOT RobustScaler. A first pass used RobustScaler and
    # the partition came out DEGENERATE -- 5,426 of 5,486 rows in one cluster
    # and the rest 1-26 row outlier nodes. These features are heavy-tailed
    # (period_uncertainty, FAP, depth ratios span orders of magnitude), so a
    # linear scaler leaves extremes that drag individual SOM nodes out to cover
    # them while everything else collapses together. A rank-based transform
    # bounds every feature and makes the map well-conditioned.
    sc = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             random_state=SEED)
    Xtr = sc.fit_transform(imp.fit_transform(prep(df, cols)))

    som = SOM(GRID, Xtr.shape[1]).fit(Xtr)
    bmu, qerr = som.bmu(Xtr)
    agg = AgglomerativeClustering(n_clusters=N_SUPER).fit(som.W)
    node2super = agg.labels_
    sup = node2super[bmu]
    res["train_quantization_error"] = {"median": float(np.median(qerr)),
                                       "p95": float(np.percentile(qerr, 95))}

    print("\n" + "=" * 92)
    print(f"PART 2.1 -- structure on the labelled population "
          f"(6x6 SOM -> {N_SUPER} superclusters), FROZEN TEST subset")
    print("=" * 92)
    analyse("labelled_clusters", sup[te], y[te], prob[te], giant[te], res)

    sizes = np.array([int((sup == c).sum()) for c in sorted(set(sup))])
    frac_max = sizes.max() / sizes.sum()
    res["partition_max_cluster_fraction"] = float(frac_max)
    if frac_max > 0.80:
        print(f"\n  *** DEGENERATE PARTITION: largest cluster holds "
              f"{frac_max*100:.1f}% of rows. Cluster-level statistics below are "
              f"not interpretable as structure. ***")
    else:
        print(f"\n  partition balance OK: largest cluster holds {frac_max*100:.1f}%")

    print("\n  full labelled population (composition only, all 5,486 rows):")
    comp = []
    for c in sorted(set(sup)):
        m = sup == c
        comp.append({"cluster": int(c), "n": int(m.sum()),
                     "planet_pct": float(y[m].mean() * 100),
                     "giant_pct": float(giant[m].mean() * 100)})
        print(f"    cluster {c}: n={int(m.sum()):5d}  planet {y[m].mean()*100:5.1f}%"
              f"  giant {giant[m].mean()*100:5.1f}%")
    res["labelled_composition_full"] = comp

    # ---------------- validity check ----------------
    print("\n" + "=" * 92)
    print("PART 1 -- VALIDITY CHECK: is the KNOWN giant signature recovered?")
    print("=" * 92)
    print("  Known finding: giants vs dwarfs have near-identical RANKING")
    print("  (AUC 0.9013 vs 0.9017) but very different CALIBRATION")
    print("  (ECE 0.0867 vs 0.0204). Recovering 'a cluster with big stars' is")
    print("  near-tautological since st_rad is a clustered feature; the real")
    print("  test is whether a cluster shows ELEVATED ECE AT ORDINARY AUC.")
    rows = res["labelled_clusters"]
    ok = [r for r in rows if np.isfinite(r["ece"])]
    if ok:
        gi = max(ok, key=lambda r: r["giant_pct"])
        hi = max(ok, key=lambda r: r["ece"])
        print(f"\n  most giant-enriched cluster : {gi['cluster']} "
              f"(giant {gi['giant_pct']:.1f}%, AUC {gi['auc']:.4f}, ECE {gi['ece']:.4f})")
        print(f"  worst-calibrated cluster    : {hi['cluster']} "
              f"(giant {hi['giant_pct']:.1f}%, AUC {hi['auc']:.4f}, ECE {hi['ece']:.4f})")
        same = gi["cluster"] == hi["cluster"]
        med_ece = float(np.median([r["ece"] for r in ok]))
        print(f"  median cluster ECE          : {med_ece:.4f}")
        print(f"  giant-enriched cluster IS the worst-calibrated? {same}")
        res["validity"] = {"giant_cluster": gi, "worst_ece_cluster": hi,
                           "same_cluster": bool(same), "median_ece": med_ece}

    # ---------------- anomaly hunting ----------------
    print("\n" + "=" * 92)
    print("PART 2.2 -- anomalous clusters")
    print("=" * 92)
    if ok:
        aucs = np.array([r["auc"] for r in ok])
        worst = min(ok, key=lambda r: r["auc"])
        print(f"  cluster AUC range {aucs.min():.4f} - {aucs.max():.4f} "
              f"(median {np.median(aucs):.4f})")
        print(f"  worst-ranking cluster: {worst['cluster']} AUC {worst['auc']:.4f} "
              f"(n={worst['n']}, planet {worst['planet_pct']:.1f}%)")
    # negative-dominated clusters holding a few positives
    print("\n  (b) negative-dominated clusters (<25% planet) and their positives:")
    leads = []
    for c in sorted(set(sup)):
        m = sup == c
        pf = y[m].mean()
        if pf < 0.25 and m.sum() >= 30:
            pos = np.where(m & (y == 1))[0]
            print(f"    cluster {c}: n={int(m.sum())}, planet {pf*100:.1f}%, "
                  f"{len(pos)} confirmed positives")
            if 0 < len(pos) <= 60:
                hosts = df.iloc[pos]["host"].astype(str).tolist()
                leads.append({"cluster": int(c), "n_cluster": int(m.sum()),
                              "planet_pct": float(pf * 100),
                              "n_positives": int(len(pos)),
                              "hosts": hosts[:60]})
    res["negative_dominated_with_positives"] = leads
    if not leads:
        print("    none")

    # ---------------- pools ----------------
    print("\n" + "=" * 92)
    print("PART 2.3 -- candidate pools: any region with no training support?")
    print("=" * 92)
    res["pools"] = {}
    for pname, feat_path, rank_path in POOLS:
        if not (os.path.exists(feat_path) and os.path.exists(rank_path)):
            print(f"  {pname}: files missing, skipped")
            continue
        f = pd.read_csv(feat_path)
        r = pd.read_csv(rank_path)
        keep = [c for c in ["host", "st_rad", "st_teff"] if c in r.columns]
        merged = f.merge(r[keep], on="host", how="inner")
        P = pd.to_numeric(merged.get("period"), errors="coerce")
        T = pd.to_numeric(merged.get("duration"), errors="coerce")
        merged = merged[(P > 0) & (T > 0)]
        missing = [c for c in cols if c not in merged.columns]
        if missing or len(merged) < 20:
            print(f"  {pname}: unusable (n={len(merged)}, missing={missing})")
            continue
        Xp = sc.transform(imp.transform(prep(merged, cols)))
        pb, pq = som.bmu(Xp)
        psup = node2super[pb]
        thr = float(np.percentile(qerr, 95))
        far = float((pq > thr).mean() * 100)
        print(f"\n  {pname}: {len(merged)} scorable candidates with full features")
        print(f"    quantization error  median {np.median(pq):.3f} "
              f"vs training {np.median(qerr):.3f}")
        print(f"    {far:.1f}% sit beyond the training p95 distance "
              f"(5% expected if identically distributed)")
        occ = []
        for c in sorted(set(sup)):
            tr_share = float((sup == c).mean() * 100)
            po_share = float((psup == c).mean() * 100)
            occ.append({"cluster": int(c), "train_pct": tr_share,
                        "pool_pct": po_share})
        print(f"    {'cluster':<9}{'train%':>9}{'pool%':>8}   flag")
        for o in occ:
            flag = ""
            if o["pool_pct"] >= 10 and o["train_pct"] < 2:
                flag = "  <-- POOL-HEAVY, THIN TRAINING SUPPORT"
            elif o["pool_pct"] >= 2 * max(o["train_pct"], 0.5):
                flag = "  <- over-represented in pool"
            print(f"    {o['cluster']:<9}{o['train_pct']:>9.1f}{o['pool_pct']:>8.1f}{flag}")
        res["pools"][pname] = {"n": int(len(merged)), "pct_beyond_train_p95": far,
                               "median_qerr_pool": float(np.median(pq)),
                               "median_qerr_train": float(np.median(qerr)),
                               "occupancy": occ}

    # ---------------- RAVEN-faithful shape-only view ----------------
    print("\n" + "=" * 92)
    print("SECONDARY -- shape-only SOM (what RAVEN actually clusters on)")
    print("=" * 92)
    sh = [c for c in SHAPE_FEATURES if c in df.columns]
    Xs = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             random_state=SEED).fit_transform(
        SimpleImputer(strategy="median").fit_transform(prep(df, sh)))
    som_s = SOM(GRID, Xs.shape[1], seed=SEED + 1).fit(Xs)
    bs, _ = som_s.bmu(Xs)
    sup_s = AgglomerativeClustering(n_clusters=N_SUPER).fit(som_s.W).labels_[bs]
    print(f"  clustered on {len(sh)} shape features: {sh}")
    analyse("shape_clusters", sup_s[te], y[te], prob[te], giant[te], res)

    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
