"""rotation_ratio_validate.py -- the RAW rotation/transit period ratio.

WHY THIS IS NOT A DUPLICATE, against my own initial expectation
---------------------------------------------------------------
`ls_period_match` (built and closed at -0.0006, 0/12 clearing) is
`min over n in {1,2,1/2,3,1/3} of |log(P_ls / (n*P_tr))|`. That is a
DETERMINISTIC FUNCTION of the raw ratio r = P_ls/P_tr, and the map is
MANY-TO-ONE -- distance-to-nearest-harmonic discards which harmonic, and the
sign and magnitude of the offset. So the raw ratio strictly CONTAINS
`ls_period_match`, not the other way round.

Measured, rather than argued:

    raw ratio  P_ls/P_tr                 AUC 0.6028   |AUC-0.5| 0.1028
    min over n (ls_period_match, TESTED) AUC 0.5887   |AUC-0.5| 0.0887
    n=1 ONLY  |log r|                    AUC 0.5596   |AUC-0.5| 0.0596
    |rho| raw ratio vs ls_period_match   0.329

The raw ratio carries MORE single-feature signal than the version that was
tested, and is only 0.329 correlated with it. The harmonic search was also NOT
noise -- dropping to n=1 only makes things worse (0.5596) -- but `min` over n
throws away information the tree could use. n=1 wins the minimisation only
22.3% of the time; n=3 wins 34.6%.

So this is a genuinely non-subsumed variant with a specific, measured reason to
expect it could differ, which is exactly the exception the brief allowed.

CAUTION CARRIED FORWARD: `trap_rmse` had the highest single-feature AUC ever
measured here (0.6507) and moved the model -0.0011, because it was 0.962
correlated with a deployed feature. A high single-feature AUC is not evidence
of anything on its own.

Production untouched. Nothing promoted.
"""
import os
import sys
import json
import importlib.util
import warnings
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
EXO = os.path.join(HERE, "exominer_triple_features.csv")
CADENCE = os.path.join(HERE, "cadence_class_confound.csv")
OUT = os.path.join(HERE, "rotation_ratio_validate.json")

N_BOOT = 12
SEED = 20260814
MDE = 0.0097
NEW = "rot_period_ratio"


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def model():
    return CalibratedClassifierCV(
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", HistGradientBoostingClassifier(random_state=42))]),
        cv=5, method="sigmoid")


def ratio_from(pls, ptr):
    pls = pd.to_numeric(pls, errors="coerce")
    ptr = pd.to_numeric(ptr, errors="coerce")
    r = pls / ptr
    return r.where(np.isfinite(r) & (pls > 0) & (ptr > 0))


def main():
    from scipy.stats import fisher_exact
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    spec = importlib.util.spec_from_file_location("m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 33

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    ex = pd.read_csv(EXO); ex["host"] = ex.host.astype(str)
    df = df.merge(ex[["host", "ls_period_match"]], on="host", how="left")
    df[NEW] = ratio_from(df.var_ls_period, df.period)
    y = df.label.to_numpy()

    # ---------- availability, up front, training + BOTH pools ----------
    print("=== PRODUCTION AVAILABILITY (up front) ===")
    print(f"  training                 {df[NEW].notna().mean():.2%}")
    for tag, f in (("main pool", "unknown_features.csv"),
                   ("widesector pool", "unknown_features_widesector.csv")):
        p = pd.read_csv(os.path.join(ROOT, "data", "catalogs", f))
        s = p[p.status.astype(str).str.startswith("Success")]
        r = ratio_from(s.var_ls_period, s.period)
        print(f"  {tag+' (Success)':<24} {r.notna().mean():.2%}  (n={len(s)})")

    # ---------- redundancy ----------
    print("\n=== REDUNDANCY (threshold 0.80) ===")
    v = df[NEW]
    ok = v.notna().to_numpy()
    for c in ("var_ls_period", "var_ls_amp", "var_ls_power", "ls_period_match"):
        o = pd.to_numeric(df[c], errors="coerce")
        m = ok & o.notna().to_numpy()
        print(f"  vs {c:<18}{abs(pd.Series(v[m].values).corr(pd.Series(o[m].values), method='spearman')):.3f}")
    r33 = df.loc[ok, cols].apply(pd.to_numeric, errors="coerce").corrwith(v[ok], method="spearman").abs()
    print(f"  max |rho| vs the 33: {r33.max():.3f} ({r33.idxmax()})")

    auc = roc_auc_score(y[ok], v[ok])
    print(f"\n  single-feature AUC {auc:.4f}  |AUC-0.5| {abs(auc-0.5):.4f}")

    # ---------- class-rate gate ----------
    print("\n=== CLASS-RATE GATE ===")
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    ap, an = int((ok & (y == 1)).sum()), int((ok & (y == 0)).sum())
    orr, pv = fisher_exact([[ap, npos - ap], [an, nneg - an]])
    auc_av = roc_auc_score(y, ok.astype(float))
    print(f"  avail pos {ap/npos:.2%}  neg {an/nneg:.2%}  OR {orr:.3f}  p {pv:.4g}  "
          f"AUC(avail) {auc_av:.4f}")

    # ---------- |galactic b| control arm ----------
    print("\n=== |GALACTIC LATITUDE| CONTROL ARM ===")
    ra = pd.to_numeric(df.ra, errors="coerce").to_numpy()
    dec = pd.to_numeric(df.dec, errors="coerce").to_numpy()
    okc = np.isfinite(ra) & np.isfinite(dec)
    gb = np.full(len(df), np.nan)
    gb[okc] = np.abs(SkyCoord(ra[okc] * u.deg, dec[okc] * u.deg).galactic.b.deg)
    mm = ok & okc
    rho = pd.Series(v[mm].values).corr(pd.Series(gb[mm]), method="spearman")
    q = pd.qcut(pd.Series(gb[mm]), 4, labels=False, duplicates="drop")
    aucs = []
    for k in sorted(pd.Series(q).dropna().unique()):
        s = (q == k).to_numpy()
        if len(set(y[mm][s])) > 1 and s.sum() > 50:
            aucs.append(round(roc_auc_score(y[mm][s], v[mm].values[s]), 3))
    print(f"  rho |gal b| {rho:+.3f}   AUC by quartile {aucs}")

    # ---------- model ----------
    X, yy = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    yy = np.asarray(yy)
    X[NEW] = v.values
    X["ls_period_match"] = pd.to_numeric(df.ls_period_match, errors="coerce").values

    tr_mask, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)
    tr_idx = np.where(tr_mask)[0]
    cad = pd.read_csv(CADENCE)[["host", "cadence_min"]]; cad["host"] = cad.host.astype(str)
    cc = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((cc >= 1.0) & (cc <= 2.6)).to_numpy()
    print(f"\ntrain {len(tr_idx)}  frozen test {int(te.sum())}  2-min {int((is2&te).sum())}")

    ARMS = {"base": cols, "+ratio": cols + [NEW],
            "+ratio,match": cols + [NEW, "ls_period_match"]}
    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(N_BOOT):
        samp = rng.choice(tr_idx, size=len(tr_idx), replace=True)
        rec = {}
        for lab, cu in ARMS.items():
            mo = model(); mo.fit(X.iloc[samp][cu], yy[samp])
            p = mo.predict_proba(X[te][cu])[:, 1]
            rec[f"{lab}_auc"] = roc_auc_score(yy[te], p)
            rec[f"{lab}_brier"] = brier_score_loss(yy[te], p)
            rec[f"{lab}_ece"] = ece(yy[te], p)
            s = is2[te]
            rec[f"{lab}_auc2"] = roc_auc_score(yy[te][s], p[s])
        rows.append(rec)
        print(f"  boot {b+1}/{N_BOOT}  base {rec['base_auc']:.4f}  "
              f"+ratio {rec['+ratio_auc']:.4f}  d {rec['+ratio_auc']-rec['base_auc']:+.4f}",
              flush=True)

    R = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print(f"{'arm':<15}{'AUC':>9}{'mean d':>10}{'sd':>8}{'95% CI':>22}{'pos':>7}"
          f"{'>=MDE':>7}{'Brier':>9}{'ECE':>8}")
    out = {"single_feature_auc": float(auc), "coverage": float(ok.mean()),
           "max_rho33": float(r33.max()), "rho_gal_b": float(rho),
           "stratified_auc": aucs, "arms": {}}
    for lab in ARMS:
        d = (R[f"{lab}_auc"] - R["base_auc"]).values
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{lab:<15}{R[f'{lab}_auc'].mean():>9.4f}{d.mean():>+10.4f}{d.std():>8.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{f'{(d>0).sum()}/{N_BOOT}':>7}"
              f"{f'{(d>=MDE).sum()}/{N_BOOT}':>7}{R[f'{lab}_brier'].mean():>9.4f}"
              f"{R[f'{lab}_ece'].mean():>8.4f}")
        out["arms"][lab] = {"auc": float(R[f"{lab}_auc"].mean()),
                            "mean_delta": float(d.mean()), "ci": [float(lo), float(hi)],
                            "positive": int((d > 0).sum()), "at_mde": int((d >= MDE).sum()),
                            "auc_2min": float(R[f"{lab}_auc2"].mean()),
                            "clears": bool(lo > 0 and d.mean() >= MDE)}
    print("\n2-min-only subset:")
    for lab in ARMS:
        d2 = (R[f"{lab}_auc2"] - R["base_auc2"]).values
        print(f"  {lab:<15} AUC {R[f'{lab}_auc2'].mean():.4f}  mean d {d2.mean():+.4f}")
    print(f"\nClearing needs ci_lo > 0 AND mean delta >= MDE ({MDE}).")
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
