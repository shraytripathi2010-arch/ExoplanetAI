"""gaia_astrometry_validate.py -- redundancy, spatial control, and the
31 -> 32/33 resampled comparison for Gaia RUWE and NSS.

Runs only because both fields PASSED the availability-trap gate
(`gaia_astrometry_gate.py`): AUC(availability) 0.5069 for RUWE and 0.5054 for
NSS, against 0.3775 for the CLOSED CTL trap -- roughly 18x smaller.

Physical hypotheses, stated before the numbers:
  RUWE  -- the renormalized unit weight error of Gaia's single-star astrometric
           fit. A genuine single star fits well (RUWE ~ 1). An unresolved
           companion drags the photocentre and inflates the residual. The
           commonly used "likely non-single" cut is RUWE > 1.4 (Lindegren's
           Gaia technical note GAIA-C3-TN-LU-LL-124; adopted widely, e.g. in
           the Gaia DR2/DR3 binary literature). Threshold used as a REPORTED
           diagnostic, not invented here; the model gets the continuous value.
  NSS   -- `non_single_star` is nonzero when Gaia published a non-single-star
           solution (1 astrometric, 2 spectroscopic, 4 eclipsing). Per Armstrong
           et al. 2022, flagged astrometric binaries were rejected outright for
           KOI validation.

ARMS
----
  base       the deployed 31
  +ruwe      32
  +nss       32
  +both      33
Each is also compared with the deployed crowding pair present (it always is --
it is part of the 31), so the "interaction with crowding" question is answered
by the correlation table plus the fact that every arm contains it.
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
CODE = os.path.join(HERE, "..")
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
GAIA = os.path.join(HERE, "gaia_astrometry_training.csv")
CADENCE = os.path.join(HERE, "cadence_class_confound.csv")
OUT = os.path.join(HERE, "gaia_astrometry_validate.json")

N_BOOT = 12
SEED = 20260814
MDE = 0.0097
CROWD = ["crowd_flux_ratio_max", "crowd_nearest_arcsec"]


def _m05():
    spec = importlib.util.spec_from_file_location("m05", os.path.join(CODE, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["m05"] = m
    spec.loader.exec_module(m); return m


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


def main():
    m05 = _m05()
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 31

    df = pd.read_csv(TRAINING); df["host"] = df.host.astype(str)
    g = pd.read_csv(GAIA); g["host"] = g.host.astype(str)
    df = df.merge(g[["host", "gaia_ruwe", "gaia_nss", "gaia_nss_flag"]], on="host", how="left")

    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)[cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(y)
    for c in ("gaia_ruwe", "gaia_nss"):
        X[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).values

    # ---------- redundancy, especially against the deployed crowding pair ----------
    print("=== REDUNDANCY (Spearman) ===")
    for c in ("gaia_ruwe", "gaia_nss"):
        v = X[c]
        r = X[cols].corrwith(v, method="spearman").abs().sort_values(ascending=False)
        print(f"  {c:<11} max |rho| vs the 31: {r.iloc[0]:.3f} ({r.index[0]})   "
              f"threshold 0.80")
        for cc in CROWD:
            print(f"      vs deployed {cc:<22} {abs(X[cc].corr(v, method='spearman')):.3f}")
    print(f"  gaia_ruwe vs gaia_nss: "
          f"{abs(X.gaia_ruwe.corr(X.gaia_nss, method='spearman')):.3f}")

    # ---------- spatial control arm (stratified, not just correlation) ----------
    print("\n=== |GALACTIC LATITUDE| CONTROL ARM ===")
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra = pd.to_numeric(df.ra, errors="coerce").to_numpy()
    dec = pd.to_numeric(df.dec, errors="coerce").to_numpy()
    okc = np.isfinite(ra) & np.isfinite(dec)
    gb = np.full(len(df), np.nan)
    gb[okc] = np.abs(SkyCoord(ra[okc] * u.deg, dec[okc] * u.deg).galactic.b.deg)
    for c in ("gaia_ruwe", "gaia_nss"):
        v = X[c]
        mm = v.notna().to_numpy() & okc
        rho = pd.Series(v[mm].values).corr(pd.Series(gb[mm]), method="spearman")
        q = pd.qcut(pd.Series(gb[mm]), 4, labels=False, duplicates="drop")
        aucs = []
        for k in sorted(pd.Series(q).dropna().unique()):
            s = (q == k).to_numpy()
            if len(set(y[mm][s])) > 1 and s.sum() > 50:
                aucs.append(round(roc_auc_score(y[mm][s], v[mm].values[s]), 3))
        print(f"  {c:<11} rho |gal b| {rho:+.3f}   AUC by |gal b| quartile {aucs}")

    # ---------- resampled model comparison ----------
    tr_mask, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)
    tr_idx = np.where(tr_mask)[0]
    cad = pd.read_csv(CADENCE)[["host", "cadence_min"]]; cad["host"] = cad.host.astype(str)
    cc = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((cc >= 1.0) & (cc <= 2.6)).to_numpy()
    print(f"\ntrain {len(tr_idx)}  frozen test {int(te.sum())}  "
          f"2-min subset in test {int((is2 & te).sum())}")

    ARMS = {"base": cols, "+ruwe": cols + ["gaia_ruwe"],
            "+nss": cols + ["gaia_nss"], "+both": cols + ["gaia_ruwe", "gaia_nss"]}
    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(N_BOOT):
        samp = rng.choice(tr_idx, size=len(tr_idx), replace=True)
        r = {}
        for lab, cu in ARMS.items():
            mo = model(); mo.fit(X.iloc[samp][cu], y[samp])
            p = mo.predict_proba(X[te][cu])[:, 1]
            r[f"{lab}_auc"] = roc_auc_score(y[te], p)
            r[f"{lab}_brier"] = brier_score_loss(y[te], p)
            r[f"{lab}_ece"] = ece(y[te], p)
            s = is2[te]
            r[f"{lab}_auc2"] = roc_auc_score(y[te][s], p[s]) if s.sum() > 50 else np.nan
        rows.append(r)
        print(f"  boot {b+1}/{N_BOOT}  base {r['base_auc']:.4f}  "
              f"+ruwe {r['+ruwe_auc']:.4f}  +nss {r['+nss_auc']:.4f}  "
              f"+both {r['+both_auc']:.4f}", flush=True)

    R = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print(f"{'arm':<10}{'AUC':>9}{'mean d':>10}{'sd':>8}{'95% CI':>22}{'pos':>7}"
          f"{'>=MDE':>7}{'Brier':>9}{'ECE':>8}")
    out = {"n_boot": N_BOOT, "seed": SEED, "mde": MDE, "arms": {}}
    for lab in ARMS:
        d = (R[f"{lab}_auc"] - R["base_auc"]).values
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{lab:<10}{R[f'{lab}_auc'].mean():>9.4f}{d.mean():>+10.4f}{d.std():>8.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{f'{(d>0).sum()}/{N_BOOT}':>7}"
              f"{f'{(d>=MDE).sum()}/{N_BOOT}':>7}{R[f'{lab}_brier'].mean():>9.4f}"
              f"{R[f'{lab}_ece'].mean():>8.4f}")
        out["arms"][lab] = {"auc": float(R[f"{lab}_auc"].mean()),
                            "mean_delta": float(d.mean()), "sd": float(d.std()),
                            "ci": [float(lo), float(hi)],
                            "positive": int((d > 0).sum()),
                            "at_mde": int((d >= MDE).sum()),
                            "auc_2min": float(np.nanmean(R[f"{lab}_auc2"])),
                            "brier": float(R[f"{lab}_brier"].mean()),
                            "ece": float(R[f"{lab}_ece"].mean()),
                            "clears": bool(lo > 0 and d.mean() >= MDE)}
    print("\n2-min-only subset:")
    for lab in ARMS:
        d2 = (R[f"{lab}_auc2"] - R["base_auc2"]).values
        print(f"  {lab:<10} AUC {np.nanmean(R[f'{lab}_auc2']):.4f}  "
              f"mean d {np.nanmean(d2):+.4f}")
    print(f"\nA clearing arm needs ci_lo > 0 AND mean delta >= MDE ({MDE}).")
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
