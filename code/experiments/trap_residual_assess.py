"""trap_residual_assess.py -- the ONE genuinely untested, zero-new-compute
sub-proposal from the four shape/residual ideas: the trapezoid fit's RESIDUAL.

PART 0 ROUTING (why only this one is here)
------------------------------------------
1. skewness of the binned phase-folded transit -> DUPLICATE of the closed
   "Medium-lift Item 2: phase-folded flux distribution statistics", which
   already computed in/out-of-transit skewness AND kurtosis AND their
   differences on the phase-folded binned curve (-0.0072, CI [-0.0153,+0.0010]).
2. ingress/egress duration RATIO -> NOT derivable from the existing fit.
   `trapezoid_shape.py:142` is `x = np.abs(phi - phi0)`: the trapezoid is
   SYMMETRIC BY CONSTRUCTION, so ingress == egress and the ratio is identically
   1. Would need a new asymmetric 6-parameter fit -- new infrastructure, not a
   cheap variant.
3. fit residual (chi2/BIC) -> THIS FILE. `trap_rmse` is already computed
   (`trapezoid_shape.py:259`) and already saved for training AND both pools,
   and arms A-E of the trapezoid validation never included it.
4. local-flux CNN -> reopens the closed CNN question. Not built.

WHAT IS TESTED
--------------
  trap_rmse  the fit's weighted RMS residual, as saved
  trap_bic   BIC = n*ln(RSS/n) + k*ln(n) with RSS = n*rmse^2, k = 5 free
             parameters. NOT a monotone transform of rmse across stars,
             because n (trap_nbins) varies per star -- so it is a genuinely
             different ordering for a tree model, not a rescaling.

Hypothesis: a real transit is well described by a trapezoid, so a LARGE
residual marks a profile the trapezoid cannot represent -- a distorted,
variable, or blended eclipse. This is a goodness-of-fit statistic, orthogonal
in principle to `trap_vshape`, which reports the fitted SHAPE regardless of how
well that shape fits.

Read-only. No production artifact is touched.
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
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
TRAP = os.path.join(HERE, "trapezoid_shape_features.csv")
POOL = os.path.join(HERE, "trapezoid_shape_pool.csv")
WIDE = os.path.join(HERE, "trapezoid_shape_widesector.csv")
OUT = os.path.join(HERE, "trap_residual_assess.json")

K_PARAMS = 5   # baseline, depth, T14, w, phi0


def add_bic(df):
    r = pd.to_numeric(df.trap_rmse, errors="coerce")
    n = pd.to_numeric(df.trap_nbins, errors="coerce")
    rss = n * r ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        df["trap_bic"] = n * np.log(rss / n) + K_PARAMS * np.log(n)
    df.loc[~np.isfinite(df.trap_bic), "trap_bic"] = np.nan
    return df


def main():
    from sklearn.metrics import roc_auc_score
    from scipy.stats import fisher_exact
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    spec = importlib.util.spec_from_file_location("m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols = list(m05.FEATURE_COLUMNS)
    print(f"production baseline: {len(cols)} features")

    tr = pd.read_csv(TRAINING); tr["host"] = tr.host.astype(str)
    t = add_bic(pd.read_csv(TRAP)); t["host"] = t.host.astype(str)
    d = tr.merge(t[["host", "trap_rmse", "trap_bic", "trap_vshape", "trap_nbins",
                    "trap_status"]], on="host", how="left")
    y = d.label.to_numpy()
    NEW = ["trap_rmse", "trap_bic"]

    print("\n=== PRODUCTION AVAILABILITY, up front (training + BOTH pools) ===")
    print(f"{'population':<22}{'rows':>8}{'trap_rmse':>12}{'trap_bic':>11}")
    print(f"{'training':<22}{len(d):>8}{d.trap_rmse.notna().mean():>11.1%}"
          f"{d.trap_bic.notna().mean():>11.1%}")
    for tag, path, feat in (("main pool", POOL, "unknown_features.csv"),
                            ("widesector pool", WIDE, "unknown_features_widesector.csv")):
        p = add_bic(pd.read_csv(path))
        f = pd.read_csv(os.path.join(ROOT, "data", "catalogs", feat))
        succ = f[f.status.astype(str).str.startswith("Success")].host.astype(str)
        p["host"] = p.host.astype(str)
        ps = p[p.host.isin(set(succ))]
        print(f"{tag+' (Success)':<22}{len(ps):>8}{ps.trap_rmse.notna().mean():>11.1%}"
              f"{ps.trap_bic.notna().mean():>11.1%}")

    print("\n=== LEAKAGE / SIGNAL ===")
    print(f"{'feature':<12}{'NaN pos':>9}{'NaN neg':>9}{'AUC':>9}{'|AUC-.5|':>10}"
          f"{'max|rho| vs 33':>16}{'rho vs trap_vshape':>20}")
    res = {}
    for c in NEW:
        v = pd.to_numeric(d[c], errors="coerce")
        ok = v.notna().to_numpy()
        auc = roc_auc_score(y[ok], v[ok])
        r33 = d.loc[ok, cols].apply(pd.to_numeric, errors="coerce").corrwith(
            v[ok], method="spearman").abs()
        vs = pd.to_numeric(d.trap_vshape, errors="coerce")
        mm = ok & vs.notna().to_numpy()
        rvs = abs(pd.Series(v[mm].values).corr(pd.Series(vs[mm].values), method="spearman"))
        res[c] = {"coverage": float(ok.mean()), "auc": float(auc),
                  "max_rho33": float(r33.max()), "max_rho33_with": str(r33.idxmax()),
                  "rho_trap_vshape": float(rvs),
                  "nan_pos": float(v[y == 1].isna().mean()),
                  "nan_neg": float(v[y == 0].isna().mean())}
        print(f"{c:<12}{v[y==1].isna().mean():>8.1%}{v[y==0].isna().mean():>9.1%}"
              f"{auc:>9.4f}{abs(auc-0.5):>10.4f}{r33.max():>10.3f} ({r33.idxmax()[:10]})"
              f"{rvs:>20.3f}")

    print("\n=== CLASS-RATE GATE (is availability itself class-correlated?) ===")
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    for c in NEW:
        v = pd.to_numeric(d[c], errors="coerce")
        ap = int((v.notna() & (y == 1)).sum()); an = int((v.notna() & (y == 0)).sum())
        orr, p = fisher_exact([[ap, npos - ap], [an, nneg - an]])
        auc_av = roc_auc_score(y, v.notna().astype(float))
        res[c].update({"avail_or": float(orr), "avail_p": float(p),
                       "auc_availability": float(auc_av)})
        print(f"  {c:<12} avail pos {ap/npos:6.2%}  neg {an/nneg:6.2%}  OR {orr:6.3f}  "
              f"p {p:.3g}  AUC(avail) {auc_av:.4f}")

    print("\n=== |GALACTIC LATITUDE| CONTROL ARM ===")
    ra = pd.to_numeric(d.ra, errors="coerce").to_numpy()
    dec = pd.to_numeric(d.dec, errors="coerce").to_numpy()
    okc = np.isfinite(ra) & np.isfinite(dec)
    gb = np.full(len(d), np.nan)
    gb[okc] = np.abs(SkyCoord(ra[okc] * u.deg, dec[okc] * u.deg).galactic.b.deg)
    for c in NEW:
        v = pd.to_numeric(d[c], errors="coerce")
        mm = v.notna().to_numpy() & okc
        rho = pd.Series(v[mm].values).corr(pd.Series(gb[mm]), method="spearman")
        q = pd.qcut(pd.Series(gb[mm]), 4, labels=False, duplicates="drop")
        aucs = []
        for k in sorted(pd.Series(q).dropna().unique()):
            s = (q == k).to_numpy()
            if len(set(y[mm][s])) > 1 and s.sum() > 50:
                aucs.append(round(roc_auc_score(y[mm][s], v[mm].values[s]), 3))
        res[c]["rho_gal_b"] = float(rho)
        res[c]["stratified_auc"] = aucs
        print(f"  {c:<12} rho |gal b| {rho:+.3f}   AUC by quartile {aucs}")

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
