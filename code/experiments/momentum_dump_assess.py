"""momentum_dump_assess.py -- pre-model battery for the momentum-dump
proximity features.

Standing protocol order, nothing skipped: coverage on training and BOTH pools
up front, class-rate gate, single-feature AUC, correlation (vs the five deployed
var_* features, vs `ls_period_match` specifically, and vs all 33), then the
|galactic latitude| control arm as a per-quartile AUC, not just a coefficient.

Plus the two internal-validity checks this feature has to pass to be believed
at all: the half-period signal must beat the full-period and the wrong-period
control, and the effect must be stronger in short-period systems, where tidal
distortion is physically possible.
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
FEATS = os.path.join(HERE, "momentum_dump_features.csv")
TRIPLE = os.path.join(HERE, "exominer_triple_features.csv")
OUT = os.path.join(HERE, "momentum_dump_assess.json")

NEW = ["md_min_dt", "md_median_dt", "md_frac_near_6h", "md_frac_near_24h",
       "md_n_dumps", "md_dump_interval", "md_period_ratio"]
# md_n_dumps and md_dump_interval are functions of SECTOR alone -- included
# deliberately as the built-in confound probe, not as candidate features.
SECTOR_PROXIES = ["md_n_dumps", "md_dump_interval"]
VAR5 = ["var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]
RHO_MAX = 0.80


def m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["m05"] = m
    spec.loader.exec_module(m); return m


def auc(y, v):
    v = pd.to_numeric(pd.Series(np.asarray(v, dtype=float)), errors="coerce")
    v = v.replace([np.inf, -np.inf], np.nan)
    ok = v.notna().to_numpy()
    yy = np.asarray(y)
    if ok.sum() < 50 or len(np.unique(yy[ok])) < 2:
        return np.nan
    return float(roc_auc_score(yy[ok], v[ok]))


def main():
    res = {}
    M = m05()
    cols33 = list(M.FEATURE_COLUMNS)
    assert len(cols33) == 33

    F = pd.read_csv(FEATS); F["host"] = F.host.astype(str)
    tr = pd.read_csv(TRAINING); tr["host"] = tr.host.astype(str)
    Ft = F[F._set == "train"].drop_duplicates("host")
    df = tr.merge(Ft[["host"] + NEW + ["md_status"]],
                  on="host", how="left")
    assert len(df) == len(tr)
    # ls_period_match, for the correlation the brief asks for by name
    tp = pd.read_csv(TRIPLE); tp["host"] = tp.host.astype(str)
    df = df.merge(tp[["host", "ls_period_match"]].drop_duplicates("host"),
                  on="host", how="left")
    y = df.label.to_numpy()
    P = pd.to_numeric(df.period, errors="coerce")
    print(f"training {len(df)} rows ({int((y==1).sum())} pos / {int((y==0).sum())} neg)")

    # ---------------- 1. COVERAGE ----------------
    print("\n" + "=" * 88)
    print("1. COVERAGE, UP FRONT")
    print("=" * 88)
    print("  training md_status: " + str(df.md_status.value_counts().head(6).to_dict()))
    pools = {}
    for tag in ("main", "widesector"):
        P_ = F[F._set == tag].drop_duplicates("host")
        pools[tag] = P_
        print(f"  [{tag}: {len(P_)} Success rows, "
              f"{P_.md_status.value_counts().head(3).to_dict()}]")
    cov = {}
    print(f"{'feature':<18}{'train':>9}{'tr pos':>9}{'tr neg':>9}{'main':>9}{'wide':>9}")
    for c in NEW:
        v = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        row = {"train": float(v.notna().mean()),
               "train_pos": float(v[y == 1].notna().mean()),
               "train_neg": float(v[y == 0].notna().mean())}
        for tag in ("main", "widesector"):
            row[tag] = float(pd.to_numeric(pools[tag][c], errors="coerce")
                             .replace([np.inf, -np.inf], np.nan).notna().mean())
        cov[c] = row
        print(f"{c:<18}{row['train']:>8.2%}{row['train_pos']:>9.2%}"
              f"{row['train_neg']:>9.2%}{row['main']:>9.2%}{row['widesector']:>9.2%}")
    res["coverage"] = cov

    # ---------------- 2. INTERNAL VALIDITY / CONFOUND PROBE ----------------
    print("\n" + "=" * 88)
    print("2. HYPOTHESIS DIRECTION AND THE SECTOR-PROXY CONFOUND PROBE")
    print("=" * 88)
    iv = {}
    print("  hypothesis: transits soon AFTER a dump are more likely instrumental")
    print("              => negatives should have SMALLER md_*_dt and LARGER frac_near")
    for c in ["md_min_dt", "md_median_dt", "md_frac_near_6h", "md_frac_near_24h"]:
        v = pd.to_numeric(df[c], errors="coerce")
        mp, mn = float(v[y == 1].median()), float(v[y == 0].median())
        iv[c] = {"median_pos": mp, "median_neg": mn}
        exp = "neg < pos" if "dt" in c and "frac" not in c else "neg > pos"
        got = ("neg < pos" if mn < mp else "neg > pos")
        print(f"    {c:<20} pos {mp:>9.4f}  neg {mn:>9.4f}   expect {exp:<9} got {got:<9}"
              f" {'MATCH' if exp == got else 'OPPOSITE'}")
    print("\n  confound probe -- md_n_dumps / md_dump_interval are functions of SECTOR ALONE:")
    for c in SECTOR_PROXIES:
        a = auc(y, df[c].to_numpy())
        iv[f"auc_{c}"] = a
        print(f"    AUC({c:<18}) = {a:.4f}   |AUC-0.5| = {abs(a-0.5):.4f}")
    print("    ^ if these beat the proximity features, the signal is observation")
    print("      epoch, not systematics.")
    res["internal_validity"] = iv

    # ---------------- 3. CLASS-RATE GATE ----------------
    print("\n" + "=" * 88)
    print("3. CLASS-RATE GATE")
    print("=" * 88)
    print(f"{'feature':<18}{'avail pos':>11}{'avail neg':>11}{'odds':>9}"
          f"{'fisher p':>11}{'AUC(avail)':>12}{'verdict':>9}")
    gate = {}
    for c in NEW:
        a = pd.to_numeric(df[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan).notna().to_numpy()
        ap, an = float(a[y == 1].mean()), float(a[y == 0].mean())
        tab = [[int((a & (y == 1)).sum()), int((~a & (y == 1)).sum())],
               [int((a & (y == 0)).sum()), int((~a & (y == 0)).sum())]]
        try:
            orr, p = fisher_exact(tab)
        except Exception:
            orr, p = np.nan, np.nan
        aa = auc(y, a.astype(float))
        ok = abs(aa - 0.5) < 0.05 if np.isfinite(aa) else False
        gate[c] = {"avail_pos": ap, "avail_neg": an, "odds_ratio": float(orr),
                   "fisher_p": float(p), "auc_avail": aa, "pass": bool(ok)}
        print(f"{c:<18}{ap:>10.2%}{an:>11.2%}{orr:>9.3g}{p:>11.3g}{aa:>12.4f}"
              f"{'PASS' if ok else 'FAIL':>9}")
    res["class_rate_gate"] = gate

    # ---------------- 4. SINGLE-FEATURE AUC ----------------
    print("\n" + "=" * 88)
    print("4. SINGLE-FEATURE AUC")
    print("=" * 88)
    sf = {}
    for c in VAR5 + ["ls_period_match"] + NEW:
        a = auc(y, df[c].to_numpy())
        sf[c] = a
        tag = "  <- DEPLOYED" if c in VAR5 else \
              ("  <- closed, not deployed" if c == "ls_period_match" else "")
        print(f"  {c:<20}AUC {a:>7.4f}   |AUC-0.5| {abs(a-0.5):>7.4f}{tag}")
    res["single_feature_auc"] = sf

    # ---------------- 5. CORRELATION ----------------
    print("\n" + "=" * 88)
    print("5. CORRELATION (Spearman |rho|; redundancy threshold 0.80)")
    print("=" * 88)
    X = df[cols33].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan)
    lpm = pd.to_numeric(df.ls_period_match, errors="coerce")
    corr = {}
    print(f"{'feature':<18}" + "".join(f"{c.replace('var_',''):>11}" for c in VAR5)
          + f"{'ls_p_match':>12}{'max|rho|/33':>13}  which")
    for c in NEW:
        v = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rr = {}
        for k in cols33:
            r = v.corr(X[k], method="spearman")
            if np.isfinite(r):
                rr[k] = float(abs(r))
        mk = max(rr, key=rr.get) if rr else None
        d = {"vs_var5": {k: rr.get(k) for k in VAR5},
             "vs_ls_period_match": float(abs(v.corr(lpm, method="spearman"))),
             "max_rho": rr.get(mk) if mk else None, "max_with": mk,
             "redundant": bool(mk and rr[mk] > RHO_MAX)}
        d["vs_new"] = {o: float(abs(v.corr(
            pd.to_numeric(df[o], errors="coerce").replace([np.inf, -np.inf], np.nan),
            method="spearman"))) for o in NEW if o != c}
        corr[c] = d
        print(f"{c:<18}" + "".join(f"{rr.get(k, float('nan')):>11.3f}" for k in VAR5)
              + f"{d['vs_ls_period_match']:>12.3f}{d['max_rho']:>13.3f}  {mk}"
              + ("   REDUNDANT" if d["redundant"] else ""))
    res["correlation"] = corr

    # ---------------- 6. |GALACTIC LATITUDE| CONTROL ARM ----------------
    print("\n" + "=" * 88)
    print("6. |GALACTIC LATITUDE| CONTROL ARM  (quartile AUC, not just rho)")
    print("=" * 88)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra = pd.to_numeric(df.ra, errors="coerce"); dec = pd.to_numeric(df.dec, errors="coerce")
    ok = ra.notna() & dec.notna()
    b = pd.Series(np.nan, index=df.index)
    b[ok] = np.abs(SkyCoord(ra[ok].values * u.deg, dec[ok].values * u.deg,
                            frame="icrs").galactic.b.deg)
    qs = pd.qcut(b, 4, labels=False, duplicates="drop")
    print(f"  |b| for {int(ok.sum())}/{len(df)} rows")
    print(f"{'feature':<18}{'rho vs |b|':>12}   AUC by |b| quartile{'':>18}{'spread':>8}")
    gbd = {}
    for c in NEW:
        v = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        r = float(v.corr(b, method="spearman"))
        qa = [auc(y[(qs == q).to_numpy()], v.to_numpy()[(qs == q).to_numpy()])
              for q in range(4)]
        spread = float(np.nanmax(qa) - np.nanmin(qa))
        gbd[c] = {"rho_gal_b": r, "quartile_auc": qa, "spread": spread}
        print(f"{c:<18}{r:>12.3f}   [" + ", ".join(f"{x:.3f}" for x in qa)
              + f"]{spread:>10.3f}")
    res["gal_b"] = gbd

    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
