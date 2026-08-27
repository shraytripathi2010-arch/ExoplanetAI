"""ls_secondary_peak_assess.py -- pre-model battery for the periodogram
secondary-peak features.

Order is fixed by the standing protocol and nothing is skipped:
  1. coverage on training and BOTH candidate pools, up front
  2. class-rate gate on availability (a feature that is merely MISSING more
     often for one class is a label proxy, not a feature)
  3. single-feature AUC
  4. correlation against `var_ls_power` / `var_ls_period` specifically, then
     against all 33 production features, redundancy threshold 0.80
  5. |galactic latitude| control arm -- correlation AND per-quartile AUC
"""
import os
import sys
import json
import importlib.util
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
FEATS = os.path.join(HERE, "ls_secondary_peak_features.csv")
OUT = os.path.join(HERE, "ls_secondary_peak_assess.json")

NEW = ["fls_p1", "fls_p2", "fls_ratio", "fls_p2_nh", "fls_ratio_nh",
       "fls_period1", "fls_period2",
       "ols_p2", "ols_ratio", "ols_p2_nh", "ols_ratio_nh"]
RHO_MAX = 0.80


def m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["m05"] = m
    spec.loader.exec_module(m); return m


def auc(y, v):
    v = pd.to_numeric(v, errors="coerce")
    ok = v.notna().to_numpy() & np.isfinite(np.asarray(y, float))
    if ok.sum() < 50 or len(np.unique(np.asarray(y)[ok])) < 2:
        return np.nan
    return float(roc_auc_score(np.asarray(y)[ok], v[ok]))


def main():
    res = {}
    M = m05()
    cols33 = list(M.FEATURE_COLUMNS)
    assert len(cols33) == 33, len(cols33)

    F = pd.read_csv(FEATS); F["host"] = F.host.astype(str)
    tr = pd.read_csv(TRAINING); tr["host"] = tr.host.astype(str)
    Ft = F[F._set == "train"].drop_duplicates("host")
    df = tr.merge(Ft[["host"] + NEW + ["fls_status"]], on="host", how="left")
    assert len(df) == len(tr)
    y = df.label.to_numpy()
    print(f"training {len(df)} rows  ({int((y==1).sum())} pos / {int((y==0).sum())} neg)")

    # ---------------- 1. COVERAGE, UP FRONT ----------------
    print("\n" + "=" * 86)
    print("1. COVERAGE  (fraction non-NaN)   training / main pool / widesector pool")
    print("=" * 86)
    print("  training fls_status: " +
          str(df.fls_status.value_counts().head(6).to_dict()))
    cov = {}
    hdr = f"{'feature':<16}{'train':>9}{'tr pos':>9}{'tr neg':>9}{'main':>9}{'wide':>9}"
    print(hdr)
    pools = {}
    for tag in ("main", "widesector"):
        P = F[F._set == tag].drop_duplicates("host")
        pools[tag] = P
        print(f"  [{tag} pool: {len(P)} Success rows, status "
              f"{P.fls_status.value_counts().head(3).to_dict()}]")
    for c in NEW:
        v = pd.to_numeric(df[c], errors="coerce")
        row = {"train": float(v.notna().mean()),
               "train_pos": float(v[y == 1].notna().mean()),
               "train_neg": float(v[y == 0].notna().mean())}
        for tag in ("main", "widesector"):
            row[tag] = float(pd.to_numeric(pools[tag][c], errors="coerce").notna().mean())
        cov[c] = row
        print(f"{c:<16}{row['train']:>8.2%}{row['train_pos']:>9.2%}"
              f"{row['train_neg']:>9.2%}{row['main']:>9.2%}{row['widesector']:>9.2%}")
    res["coverage"] = cov

    # ---------------- 2. CLASS-RATE GATE ----------------
    print("\n" + "=" * 86)
    print("2. CLASS-RATE GATE  (is availability itself a label proxy?)")
    print("=" * 86)
    print(f"{'feature':<16}{'avail pos':>11}{'avail neg':>11}{'odds ratio':>12}"
          f"{'fisher p':>11}{'AUC(avail)':>12}{'verdict':>10}")
    gate = {}
    for c in NEW:
        a = pd.to_numeric(df[c], errors="coerce").notna().to_numpy()
        ap, an = float(a[y == 1].mean()), float(a[y == 0].mean())
        tab = [[int((a & (y == 1)).sum()), int((~a & (y == 1)).sum())],
               [int((a & (y == 0)).sum()), int((~a & (y == 0)).sum())]]
        try:
            orr, p = fisher_exact(tab)
        except Exception:
            orr, p = np.nan, np.nan
        aa = auc(y, pd.Series(a.astype(float)))
        ok = abs(aa - 0.5) < 0.05 if np.isfinite(aa) else False
        gate[c] = {"avail_pos": ap, "avail_neg": an, "odds_ratio": float(orr),
                   "fisher_p": float(p), "auc_avail": aa, "pass": bool(ok)}
        print(f"{c:<16}{ap:>10.2%}{an:>11.2%}{orr:>12.4g}{p:>11.4g}{aa:>12.4f}"
              f"{'PASS' if ok else 'FAIL':>10}")
    res["class_rate_gate"] = gate

    # ---------------- 3. SINGLE-FEATURE AUC ----------------
    print("\n" + "=" * 86)
    print("3. SINGLE-FEATURE AUC  (deployed periodogram features for reference)")
    print("=" * 86)
    sf = {}
    for c in ["var_ls_power", "var_ls_period", "var_ls_amp"] + NEW:
        a = auc(y, df[c])
        sf[c] = a
        tagd = "  <- DEPLOYED" if c.startswith("var_") else ""
        print(f"  {c:<18}AUC {a:>7.4f}   |AUC-0.5| {abs(a-0.5):>7.4f}{tagd}")
    res["single_feature_auc"] = sf

    # ---------------- 4. CORRELATION ----------------
    print("\n" + "=" * 86)
    print("4. CORRELATION  (Spearman |rho|; redundancy threshold 0.80)")
    print("=" * 86)
    X = df[cols33].apply(pd.to_numeric, errors="coerce")
    corr = {}
    print(f"{'feature':<16}{'vs var_ls_power':>17}{'vs var_ls_period':>18}"
          f"{'max |rho| vs 33':>17}{'  which':<22}{'':>3}")
    for c in NEW:
        v = pd.to_numeric(df[c], errors="coerce")
        rr = {}
        for k in cols33:
            r = v.corr(X[k], method="spearman")
            if np.isfinite(r):
                rr[k] = float(abs(r))
        mk = max(rr, key=rr.get) if rr else None
        d = {"vs_var_ls_power": rr.get("var_ls_power"),
             "vs_var_ls_period": rr.get("var_ls_period"),
             "max_rho": rr.get(mk) if mk else None, "max_with": mk,
             "redundant": bool(mk and rr[mk] > RHO_MAX)}
        # also against the other new features
        d["vs_new"] = {o: float(abs(v.corr(pd.to_numeric(df[o], errors="coerce"),
                                           method="spearman")))
                       for o in NEW if o != c}
        corr[c] = d
        flag = "  REDUNDANT" if d["redundant"] else ""
        print(f"{c:<16}{d['vs_var_ls_power']:>17.3f}{d['vs_var_ls_period']:>18.3f}"
              f"{d['max_rho']:>17.3f}  {mk:<22}{flag}")
    res["correlation"] = corr

    # ---------------- 5. |GALACTIC LATITUDE| CONTROL ARM ----------------
    print("\n" + "=" * 86)
    print("5. |GALACTIC LATITUDE| CONTROL ARM")
    print("=" * 86)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra = pd.to_numeric(df.ra, errors="coerce")
    dec = pd.to_numeric(df.dec, errors="coerce")
    ok = ra.notna() & dec.notna()
    b = pd.Series(np.nan, index=df.index)
    sc = SkyCoord(ra[ok].values * u.deg, dec[ok].values * u.deg, frame="icrs")
    b[ok] = np.abs(sc.galactic.b.deg)
    print(f"  |b| available for {int(ok.sum())}/{len(df)} rows")
    qs = pd.qcut(b, 4, labels=False, duplicates="drop")
    gb = {}
    print(f"{'feature':<16}{'rho vs |b|':>12}   AUC by |b| quartile")
    for c in NEW:
        v = pd.to_numeric(df[c], errors="coerce")
        r = float(v.corr(b, method="spearman"))
        qa = []
        for q in range(4):
            m = (qs == q).to_numpy()
            qa.append(auc(y[m], v[m]))
        spread = float(np.nanmax(qa) - np.nanmin(qa))
        gb[c] = {"rho_gal_b": r, "quartile_auc": qa, "spread": spread}
        print(f"{c:<16}{r:>12.3f}   [" + ", ".join(f"{x:.3f}" for x in qa) +
              f"]  spread {spread:.3f}")
    res["gal_b"] = gb

    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
