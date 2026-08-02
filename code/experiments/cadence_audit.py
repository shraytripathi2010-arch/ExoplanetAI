"""cadence_audit.py -- what cadence is the training set ACTUALLY made of?

The FFI investigation was framed as "can we ADD lower-cadence data". A sampled
check said the premise needs correcting: the training set is already mixed, and
the mix looks class-asymmetric. That turns the central question from "would
adding FFI hurt?" into "is a cadence confound ALREADY in the data?".

Why it can happen: `01_download_known.py:try_search` prefers author='SPOC'
(2-min) but falls back to whatever product MAST returns first when no SPOC
product exists -- QLP, CDIPS, TASOC and friends, which are FFI-derived. The
negative class (`05_...`/TOI false positives) went through its own downloader.
Nothing recorded which pipeline produced each file, so cadence has been an
invisible column this whole time.

Cadence is measured directly from the data (median delta-t of the processed
light curve), not inferred from a filename or a log, so it reflects what the
model actually trained on.

THREE QUESTIONS
  1. What is the real cadence distribution, per class?
  2. Does cadence PREDICT the label? A single-"feature" AUC well away from 0.5
     means the model can, in principle, use instrument identity as a class
     shortcut -- the same failure mode the Kepler cross-mission check was
     designed to catch, but already inside the current dataset.
  3. Are the extracted FEATURES separable by cadence? This is the same domain
     classifier that exposed the synthetic-data failure at AUC 0.97. Run here
     on data already on disk -- no downloads required.
"""
import os
import sys
import json
import glob
import importlib.util
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
PROC_DIRS = [os.path.join(ROOT, "data", "processed"),
             os.path.join(ROOT, "data", "processed_negative")]
OUT_CSV = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
RESULTS = os.path.join(SCRIPT_DIR, "cadence_audit_results.json")

RANDOM_SEED = 42


def _load_m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def _find(host):
    for d in PROC_DIRS:
        p = os.path.join(d, str(host) + ".csv")
        if os.path.exists(p):
            return p
    return None


def measure(host):
    p = _find(host)
    if p is None:
        return {"host": host, "cadence_min": np.nan, "n_points": 0,
                "baseline_days": np.nan}
    try:
        d = pd.read_csv(p, usecols=["time"])
        t = pd.to_numeric(d["time"], errors="coerce").dropna().to_numpy()
        t = np.sort(t)
        if len(t) < 50:
            return {"host": host, "cadence_min": np.nan, "n_points": len(t),
                    "baseline_days": np.nan}
        dt = np.median(np.diff(t))
        return {"host": host, "cadence_min": float(dt * 24 * 60),
                "n_points": int(len(t)),
                "baseline_days": float(t[-1] - t[0])}
    except Exception:
        return {"host": host, "cadence_min": np.nan, "n_points": 0,
                "baseline_days": np.nan}


def bucket(c):
    if not np.isfinite(c):
        return "unknown"
    if c < 1.0:
        return "sub-minute"
    if c <= 2.6:
        return "2-min (SPOC)"
    if c <= 4.5:
        return "200s FFI"
    if c <= 12.0:
        return "10-min FFI"
    if c <= 35.0:
        return "30-min FFI"
    return "coarser"


def main():
    res = {}
    m05 = _load_m05()
    df = pd.read_csv(TRAINING_CSV)
    hosts = df["host"].astype(str).tolist()
    print(f"measuring cadence for {len(hosts)} training stars...")

    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(measure, hosts, chunksize=50))
    cad = pd.DataFrame(rows)
    cad.to_csv(OUT_CSV, index=False)

    df = df.merge(cad, on="host", how="left")
    df["cadence_bucket"] = df["cadence_min"].map(bucket)

    print("\n" + "=" * 78)
    print("1. CADENCE DISTRIBUTION BY CLASS")
    print("=" * 78)
    ct = pd.crosstab(df["cadence_bucket"], df["label"])
    ct.columns = [f"label={c}" for c in ct.columns]
    ct["total"] = ct.sum(axis=1)
    pct = pd.crosstab(df["cadence_bucket"], df["label"], normalize="columns") * 100
    out = ct.join(pct.add_prefix("pct_label="))
    print(out.round(1).to_string())
    res["crosstab"] = json.loads(out.to_json())

    non2 = ~df["cadence_bucket"].isin(["2-min (SPOC)", "unknown"])
    p_non2_pos = 100 * non2[df.label == 1].mean()
    p_non2_neg = 100 * non2[df.label == 0].mean()
    print(f"\n  NON-2-min share:  positives {p_non2_pos:.1f}%   "
          f"negatives {p_non2_neg:.1f}%")
    res["pct_non_2min_positive"] = float(p_non2_pos)
    res["pct_non_2min_negative"] = float(p_non2_neg)

    print("\n" + "=" * 78)
    print("2. DOES CADENCE PREDICT THE LABEL?")
    print("=" * 78)
    y = df["label"].to_numpy()
    ok = df["cadence_min"].notna().to_numpy()
    auc_cad = roc_auc_score(y[ok], df.loc[ok, "cadence_min"].to_numpy())
    auc_non2 = roc_auc_score(y, non2.astype(int).to_numpy())
    print(f"  raw cadence_min as a single feature : AUC {auc_cad:.4f}")
    print(f"  binary 'is not 2-min' indicator     : AUC {auc_non2:.4f}")
    print("  (0.5 = cadence carries no class information; far from 0.5 means")
    print("   instrument identity is itself a class signal -- a shortcut)")
    res["auc_cadence_raw"] = float(auc_cad)
    res["auc_not_2min_indicator"] = float(auc_non2)

    print("\n" + "=" * 78)
    print("3. ARE THE FEATURES SEPARABLE BY CADENCE? (domain classifier)")
    print("=" * 78)
    X, _ = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    dom = non2.astype(int).to_numpy()
    print(f"  2-min rows {int((dom==0).sum())} | non-2-min rows {int((dom==1).sum())}")
    if dom.sum() < 30:
        print("  too few non-2-min rows for a domain classifier")
        res["domain_auc"] = None
    else:
        pipe = Pipeline([("i", SimpleImputer(strategy="median")),
                         ("c", HistGradientBoostingClassifier(
                             max_iter=300, max_depth=4, learning_rate=0.05,
                             class_weight="balanced", random_state=RANDOM_SEED))])
        cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
        p = cross_val_predict(pipe, X, dom, cv=cv, method="predict_proba")[:, 1]
        dauc = roc_auc_score(dom, p)
        print(f"  domain classifier (2-min vs non-2-min) AUC: {dauc:.4f}")
        print("  reference: synthetic-vs-real scored 0.9654 and mixing HURT")
        res["domain_auc"] = float(dauc)

        smd = {}
        for c in X.columns:
            a = X.loc[dom == 0, c].astype(float)
            b = X.loc[dom == 1, c].astype(float)
            sd = np.sqrt((a.var() + b.var()) / 2)
            if np.isfinite(sd) and sd > 0:
                smd[c] = float((b.mean() - a.mean()) / sd)
        top = sorted(smd.items(), key=lambda kv: -abs(kv[1]))[:8]
        print("\n  largest standardized mean differences (non-2min - 2min):")
        for k, v in top:
            print(f"    {k:24s} {v:+.2f}")
        res["standardized_mean_diff"] = smd

    print("\n" + "=" * 78)
    print("4. IS THE CADENCE SIGNAL REDUNDANT WITH THE LABEL SIGNAL?")
    print("=" * 78)
    print("  per-feature: AUC for predicting the LABEL vs AUC for predicting CADENCE")
    print(f"  {'feature':<24}{'label-AUC':>11}{'cadence-AUC':>13}")
    pairs = []
    for c in X.columns:
        v = pd.to_numeric(X[c], errors="coerce")
        m = v.notna().to_numpy()
        if m.sum() < 50:
            continue
        try:
            la = roc_auc_score(y[m], v[m])
            da = roc_auc_score(dom[m], v[m])
        except Exception:
            continue
        pairs.append((c, la, da))
    pairs.sort(key=lambda r: -abs(r[2] - 0.5))
    for c, la, da in pairs[:8]:
        print(f"  {c:<24}{la:>11.3f}{da:>13.3f}")
    if pairs:
        arr = np.array([[abs(l - 0.5), abs(d - 0.5)] for _, l, d in pairs])
        r = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
        print(f"\n  correlation |label-AUC-0.5| vs |cadence-AUC-0.5| across features: {r:+.3f}")
        print("  (strongly positive would mean the features that separate classes are")
        print("   the same ones that separate cadences -- i.e. an instrument shortcut)")
        res["label_vs_cadence_auc_correlation"] = r
        res["per_feature"] = {c: {"label_auc": l, "cadence_auc": d} for c, l, d in pairs}

    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {RESULTS}\n{OUT_CSV}")


if __name__ == "__main__":
    main()
