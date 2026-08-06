"""stellar_density_checks.py -- Part 0/1 gate for the density features.

Order matters and is deliberate: the CTL-availability trap and the class-rate
gate come BEFORE any model fit, so a dead feature costs minutes not hours.

FEATURES

  st_logg      TIC surface gravity, taken directly from the catalog row the
               pipeline already downloads and discards.

  st_rho       TIC stellar density, solar units, same provenance.

  rho_circ     Density IMPLIED by the observed transit, from Kepler's third
               law assuming a circular central transit:
                   a/R* = P / (pi * T14)
                   rho_circ = 3 P / (G pi^2 T14^3)
               Computed in cgs then divided by rho_sun = 1.408 g/cm^3 so it is
               directly comparable to TIC's solar-unit rho.

  rho_ratio    log10(rho_circ / st_rho). THE discriminator. Near 0 for a real
               central transit on the catalogued star. A grazing eclipsing
               binary has T14 too short for its period, inflating rho_circ; a
               blend diluted by a third star distorts it the other way. This is
               the quantity vespa-style validation tools reason about.

WHY rho_ratio IS NOT ALREADY IN THE MODEL, checked against the deployed list:
`period` and `duration` are production features, so the NUMERATOR is available.
The denominator needs rho_star, which requires stellar MASS -- and `st_mass` is
NOT one of the 26 features (only `st_rad` and `st_teff` are). So the model
cannot form this ratio from what it sees. That bounds the claim: the numerator's
information is present, the denominator's is not.
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
RAW = os.path.join(SCRIPT_DIR, "stellar_density_raw.csv")
META = os.path.join(ROOT, "models", "best_model_metadata.json")
OUT = os.path.join(SCRIPT_DIR, "stellar_density_checks.json")
FEATS = os.path.join(SCRIPT_DIR, "stellar_density_features.csv")

G_CGS = 6.67430e-8
RHO_SUN = 1.408          # g/cm^3
DAY_S = 86400.0
NEW = ["st_logg", "st_rho", "rho_circ", "rho_ratio"]


def build():
    tr = pd.read_csv(TRAINING)
    cat = pd.read_csv(RAW)
    df = tr.merge(cat, on="host", how="left")

    df["st_logg"] = pd.to_numeric(df.get("logg"), errors="coerce")
    df["st_rho"] = pd.to_numeric(df.get("rho"), errors="coerce")

    P = pd.to_numeric(df["period"], errors="coerce") * DAY_S
    T = pd.to_numeric(df["duration"], errors="coerce") * DAY_S   # DAYS in training.csv
    bad = ~(np.isfinite(P) & np.isfinite(T) & (P > 0) & (T > 0))
    rho_c = 3.0 * P / (G_CGS * np.pi ** 2 * T ** 3) / RHO_SUN
    rho_c[bad] = np.nan
    df["rho_circ"] = rho_c

    r = df["rho_circ"] / df["st_rho"]
    r = r.where(np.isfinite(r) & (r > 0))
    df["rho_ratio"] = np.log10(r)
    return df


def main():
    df = build()
    y = df["label"].astype(int)
    res = {}

    print("=" * 66)
    print("PART 0/1 GATE -- stellar density / log g")
    print("=" * 66)

    # ---- CTL-availability trap, the specific one crowding hit -------------
    print("\n[1] CTL-AVAILABILITY TRAP (the contratio/numcont failure mode)")
    print("    reference: contratio scored availability-AUC 0.3775, 37.5% on pool")
    res["availability"] = {}
    for f in NEW + ["contratio", "priority"]:
        if f not in df.columns:
            continue
        av = df[f].notna().astype(int)
        p1 = av[y == 1].mean() * 100
        p0 = av[y == 0].mean() * 100
        a = roc_auc_score(y, av) if av.nunique() > 1 else float("nan")
        res["availability"][f] = {"planet_pct": p1, "fp_pct": p0, "avail_auc": a}
        flag = "  <-- CTL-like" if abs(a - 0.5) > 0.05 else ""
        print(f"    {f:10s} planets {p1:5.1f}%  FP {p0:5.1f}%  availability-AUC {a:.4f}{flag}")

    # ---- class-rate gate --------------------------------------------------
    print("\n[2] CLASS-RATE GATE (before any model fit)")
    res["gate"] = {}
    for f in NEW:
        v = pd.to_numeric(df[f], errors="coerce")
        k = v.notna()
        if k.sum() < 50 or y[k].nunique() < 2:
            continue
        auc = roc_auc_score(y[k], v[k])
        u = mannwhitneyu(v[k & (y == 1)], v[k & (y == 0)])
        res["gate"][f] = {
            "coverage": float(k.mean()), "n": int(k.sum()), "auc": auc,
            "median_planet": float(v[k & (y == 1)].median()),
            "median_fp": float(v[k & (y == 0)].median()),
            "mannwhitney_p": float(u.pvalue)}
        print(f"    {f:10s} cov {k.mean()*100:5.1f}%  AUC {auc:.4f}  "
              f"med planet {v[k&(y==1)].median():8.3f} vs FP {v[k&(y==0)].median():8.3f}  "
              f"p={u.pvalue:.2e}")

    # ---- redundancy against the deployed 26 -------------------------------
    print("\n[3] REDUNDANCY vs the 26 production features (threshold |r| >= 0.80)")
    prod = json.load(open(META))["feature_columns"]
    have = [c for c in prod if c in df.columns]
    res["redundancy"] = {}
    for f in NEW:
        sub = df[[f] + have].apply(pd.to_numeric, errors="coerce")
        c = sub.corr(method="spearman")[f].drop(f).abs().sort_values(ascending=False)
        res["redundancy"][f] = {k: float(v) for k, v in c.head(5).items()}
        top = c.head(3)
        s = ", ".join(f"{k} {v:.3f}" for k, v in top.items())
        print(f"    {f:10s} max |r| {c.iloc[0]:.3f}   [{s}]"
              + ("   <-- REDUNDANT" if c.iloc[0] >= 0.80 else ""))

    # ---- spatial exposure, correlation only; the ARM runs later -----------
    print("\n[4] SPATIAL screen vs |galactic b| (correlation; control ARM later)")
    res["spatial_corr"] = {}
    if "abs_gal_b" in df.columns or {"ra", "dec"}.issubset(df.columns):
        if "abs_gal_b" not in df.columns:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            ra = pd.to_numeric(df["ra"], errors="coerce")
            dec = pd.to_numeric(df["dec"], errors="coerce")
            ok = ra.notna() & dec.notna()
            b = pd.Series(np.nan, index=df.index)
            sc = SkyCoord(ra[ok].values * u.deg, dec[ok].values * u.deg).galactic
            b[ok] = np.abs(sc.b.deg)
            df["abs_gal_b"] = b
        for f in NEW:
            rr = spearmanr(df["abs_gal_b"], pd.to_numeric(df[f], errors="coerce"),
                           nan_policy="omit")[0]
            res["spatial_corr"][f] = float(rr)
            print(f"    {f:10s} r vs |b| = {rr:+.3f}")
    else:
        print("    no ra/dec available in training.csv")

    df[["host", "label"] + NEW].to_csv(FEATS, index=False)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}\nwrote {FEATS}")


if __name__ == "__main__":
    main()
