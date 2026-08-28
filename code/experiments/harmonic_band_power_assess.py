"""harmonic_band_power_assess.py -- pre-model battery for the harmonic-frequency
and band-integrated periodogram power features.

Order is the standing protocol, nothing skipped: coverage on training and BOTH
pools up front, class-rate gate, single-feature AUC, correlation, then BOTH
control arms via the reusable `control_arms` module -- the |b| spatial arm and
the sector/epoch temporal arm that became a standing requirement after the
momentum-dump investigation.

The correlation block checks the whole PERIODOGRAM FAMILY against itself, not
just against the unrelated 33: var_ls_power / var_ls_amp / var_ls_period, the
secondary-peak features, the ellipsoidal harmonic amplitudes, and
ls_period_match. A new periodogram statistic that is 0.9 correlated with an old
one is not new, however different its formula looks.
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
sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
FEATS = os.path.join(HERE, "harmonic_band_power_features.csv")
SECPEAK = os.path.join(HERE, "ls_secondary_peak_features.csv")
ELLIP = os.path.join(HERE, "ellipsoidal_features.csv")
TRIPLE = os.path.join(HERE, "exominer_triple_features.csv")
OUT = os.path.join(HERE, "harmonic_band_power_assess.json")

BASE = ["hp_p05x", "hp_p1x", "hp_p2x", "hp_p3x", "hp_max", "hp_sum", "hp_frac",
        "bp_band", "bp_band_frac", "bp_band_ratio"]
NEW = [f"{p}{c}" for p in ("f_", "o_") for c in BASE]
# the periodogram family this must be checked against, not just the 33
FAMILY = ["var_ls_power", "var_ls_amp", "var_ls_period",
          "fls_p1", "fls_p2", "fls_ratio", "ols_p2", "ols_ratio",
          "ell_a1", "ell_a2", "ls_period_match"]
RHO_MAX = 0.80


def m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["m05"] = m
    spec.loader.exec_module(m); return m


def auc(y, v):
    v = pd.to_numeric(pd.Series(np.asarray(v, dtype=float)), errors="coerce")
    v = v.replace([np.inf, -np.inf], np.nan)
    ok = v.notna().to_numpy(); yy = np.asarray(y)
    if ok.sum() < 50 or len(np.unique(yy[ok])) < 2:
        return np.nan
    return float(roc_auc_score(yy[ok], v[ok]))


def _merge(df, path, cols, tag):
    if not os.path.exists(path):
        print(f"  [{tag}: file missing, skipped]")
        return df
    x = pd.read_csv(path); x["host"] = x.host.astype(str)
    if "_set" in x.columns:
        x = x[x._set == "train"]
    have = [c for c in cols if c in x.columns]
    return df.merge(x[["host"] + have].drop_duplicates("host"), on="host", how="left")


def main():
    res = {}
    M = m05()
    cols33 = list(M.FEATURE_COLUMNS); assert len(cols33) == 33

    F = pd.read_csv(FEATS); F["host"] = F.host.astype(str)
    tr = pd.read_csv(TRAINING); tr["host"] = tr.host.astype(str)
    Ft = F[F._set == "train"].drop_duplicates("host")
    df = tr.merge(Ft[["host"] + NEW + ["hb_status"]], on="host", how="left")
    assert len(df) == len(tr)
    df = _merge(df, SECPEAK, ["fls_p1", "fls_p2", "fls_ratio", "ols_p2", "ols_ratio"], "secpeak")
    df = _merge(df, ELLIP, ["ell_a1", "ell_a2"], "ellipsoidal")
    df = _merge(df, TRIPLE, ["ls_period_match"], "ls_period_match")
    y = df.label.to_numpy()
    print(f"training {len(df)} rows ({int((y==1).sum())} pos / {int((y==0).sum())} neg)")

    # ---------------- 1. COVERAGE ----------------
    print("\n" + "=" * 92); print("1. COVERAGE, UP FRONT"); print("=" * 92)
    print("  training hb_status: " + str(df.hb_status.value_counts().head(6).to_dict()))
    pools = {}
    for tag in ("main", "widesector"):
        P_ = F[F._set == tag].drop_duplicates("host"); pools[tag] = P_
        print(f"  [{tag}: {len(P_)} Success rows, "
              f"{P_.hb_status.value_counts().head(3).to_dict()}]")
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

    # ---------------- 2. CLASS-RATE GATE ----------------
    print("\n" + "=" * 92); print("2. CLASS-RATE GATE"); print("=" * 92)
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

    # ---------------- 3. SINGLE-FEATURE AUC ----------------
    print("\n" + "=" * 92); print("3. SINGLE-FEATURE AUC"); print("=" * 92)
    sf = {}
    for c in [x for x in FAMILY if x in df.columns] + NEW:
        a = auc(y, df[c].to_numpy()); sf[c] = a
        t = "  <- DEPLOYED" if c.startswith("var_") else (
            "  <- prior periodogram/ellipsoidal work" if c in FAMILY else "")
        print(f"  {c:<18}AUC {a:>7.4f}   |AUC-0.5| {abs(a-0.5):>7.4f}{t}")
    res["single_feature_auc"] = sf

    # ---------------- 4. CORRELATION: FAMILY FIRST ----------------
    print("\n" + "=" * 92)
    print("4. CORRELATION vs THE PERIODOGRAM FAMILY, then vs all 33 (|rho|, bar 0.80)")
    print("=" * 92)
    fam = [c for c in FAMILY if c in df.columns]
    X = df[cols33].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    XF = df[fam].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    corr = {}
    print(f"{'feature':<18}{'max|rho| FAMILY':>17}{'  which':<20}"
          f"{'max|rho| /33':>14}{'  which':<20}")
    for c in NEW:
        v = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rf = {k: float(abs(v.corr(XF[k], method="spearman"))) for k in fam
              if np.isfinite(v.corr(XF[k], method="spearman"))}
        r3 = {k: float(abs(v.corr(X[k], method="spearman"))) for k in cols33
              if np.isfinite(v.corr(X[k], method="spearman"))}
        kf = max(rf, key=rf.get) if rf else None
        k3 = max(r3, key=r3.get) if r3 else None
        d = {"family": rf, "max_family": rf.get(kf), "max_family_with": kf,
             "max_33": r3.get(k3), "max_33_with": k3,
             "redundant": bool((kf and rf[kf] > RHO_MAX) or (k3 and r3[k3] > RHO_MAX))}
        corr[c] = d
        flag = "  REDUNDANT" if d["redundant"] else ""
        print(f"{c:<18}{d['max_family']:>17.3f}  {kf:<20}{d['max_33']:>14.3f}  {k3:<18}{flag}")
    res["correlation"] = corr

    # ---------------- 5. BOTH CONTROL ARMS ----------------
    print()
    import control_arms
    res["controls"] = control_arms.both_controls(df, y, NEW, verbose=True)

    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
