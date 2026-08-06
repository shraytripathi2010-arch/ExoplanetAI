"""stellar_variability_checks.py -- pre-model gate for the variability metrics.

Same order as the density gate and for the same reason: availability, then the
class-rate gate, then redundancy, then a spatial screen. A feature that dies at
the gate costs minutes instead of a resampled fit.

PREDICTION, recorded before the numbers are read: false positives should show
MORE variability than planets on all four metrics (AUC below 0.5). Spots,
pulsation and the ellipsoidal/reflection modulation of an eclipsing binary all
push variability up, and stellar activity is a documented false-positive
contributor in transit vetting.

THE CONFOUND THIS GATE MUST SEPARATE. This project's positive class is
dominated by bright, well-studied confirmed hosts and its negative class by TOI
false positives. A pure brightness/noise difference between the classes would
produce the SAME direction for a reason that has nothing to do with stellar
activity. `var_excess` -- scatter divided by the star's own photometric error --
is the metric built to be robust to that, so a large gap between `var_oot_rms`
and `var_excess` is itself diagnostic: it would mean the raw signal is mostly
noise level, not activity.

Redundancy is checked with particular attention to `depth_mean_std` and
`snr`, the deployed features most likely to already encode scatter.
"""
import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
VFEAT = os.path.join(SCRIPT_DIR, "stellar_variability_features.csv")
META = os.path.join(ROOT, "models", "best_model_metadata.json")
OUT = os.path.join(SCRIPT_DIR, "stellar_variability_checks.json")

NEW = ["var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]


def main():
    tr = pd.read_csv(TRAINING)
    v = pd.read_csv(VFEAT)
    df = tr.merge(v, on="host", how="left")
    y = df["label"].astype(int)
    res = {}

    print("=" * 70)
    print("PART 2 GATE -- out-of-transit variability / activity")
    print("=" * 70)

    if "var_status" in df.columns:
        print("\n[0] EXTRACTION STATUS")
        vc = df["var_status"].value_counts()
        res["status"] = {str(k): int(x) for k, x in vc.items()}
        for k, x in vc.head(8).items():
            print(f"    {str(k):40s} {x}")

    print("\n[1] AVAILABILITY BY CLASS (the indicator-channel check)")
    res["availability"] = {}
    for f in NEW:
        if f not in df.columns:
            continue
        av = df[f].notna().astype(int)
        a = roc_auc_score(y, av) if av.nunique() > 1 else float("nan")
        res["availability"][f] = {"planet_pct": av[y == 1].mean() * 100,
                                  "fp_pct": av[y == 0].mean() * 100, "avail_auc": a}
        print(f"    {f:15s} planets {av[y==1].mean()*100:5.1f}%  "
              f"FP {av[y==0].mean()*100:5.1f}%  availability-AUC {a:.4f}")

    print("\n[2] CLASS-RATE GATE (prediction: AUC < 0.5 for all)")
    res["gate"] = {}
    for f in NEW:
        val = pd.to_numeric(df[f], errors="coerce")
        k = val.notna()
        if k.sum() < 50 or y[k].nunique() < 2:
            continue
        auc = roc_auc_score(y[k], val[k])
        u = mannwhitneyu(val[k & (y == 1)], val[k & (y == 0)])
        res["gate"][f] = {"coverage": float(k.mean()), "n": int(k.sum()), "auc": auc,
                          "median_planet": float(val[k & (y == 1)].median()),
                          "median_fp": float(val[k & (y == 0)].median()),
                          "mannwhitney_p": float(u.pvalue)}
        d = "as predicted" if auc < 0.5 else "OPPOSITE to prediction"
        print(f"    {f:15s} cov {k.mean()*100:5.1f}%  AUC {auc:.4f}  "
              f"med planet {val[k&(y==1)].median():.5g} vs FP {val[k&(y==0)].median():.5g}"
              f"  p={u.pvalue:.1e}  [{d}]")

    print("\n[3] REDUNDANCY vs the 26 production features (|r| >= 0.80)")
    prod = json.load(open(META))["feature_columns"]
    have = [c for c in prod if c in df.columns]
    res["redundancy"] = {}
    for f in NEW:
        sub = df[[f] + have].apply(pd.to_numeric, errors="coerce")
        c = sub.corr(method="spearman")[f].drop(f).abs().sort_values(ascending=False)
        res["redundancy"][f] = {k: float(x) for k, x in c.head(5).items()}
        s = ", ".join(f"{k} {x:.3f}" for k, x in c.head(3).items())
        print(f"    {f:15s} max |r| {c.iloc[0]:.3f}   [{s}]"
              + ("   <-- REDUNDANT" if c.iloc[0] >= 0.80 else ""))

    print("\n[4] SPATIAL screen vs |galactic b| (correlation; ARM later if it survives)")
    res["spatial_corr"] = {}
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra = pd.to_numeric(df["ra"], errors="coerce")
    dec = pd.to_numeric(df["dec"], errors="coerce")
    ok = ra.notna() & dec.notna()
    b = pd.Series(np.nan, index=df.index)
    b[ok] = np.abs(SkyCoord(ra[ok].values * u.deg, dec[ok].values * u.deg).galactic.b.deg)
    for f in NEW:
        rr = spearmanr(b, pd.to_numeric(df[f], errors="coerce"), nan_policy="omit")[0]
        res["spatial_corr"][f] = float(rr)
        print(f"    {f:15s} r vs |b| = {rr:+.3f}")

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
