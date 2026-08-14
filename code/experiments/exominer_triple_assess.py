"""exominer_triple_assess.py -- three ExoMiner++-inspired proposals, assessed
separately. Read-only: no training data, model, split or pipeline is modified.

PROPOSAL 1 -- LS PERIOD MATCHING (survives the duplicate check; see Part 0)
    `var_ls_amp/power/period` are ALREADY DEPLOYED and are Lomb-Scargle
    features. They answer "is this star variable, at what period, how strongly",
    computed on OUT-OF-TRANSIT-MASKED raw flux.
    They do NOT answer the proposal's actual question: is the star's dominant
    periodicity AT (or a harmonic of) the candidate's own transit period?
    That is a RELATION between two numbers, not a new measurement, and it is
    computed nowhere. Hypothesis: a transit period sitting on the star's own
    rotation period -- or a 2:1 / 1:2 harmonic of it, the same aliasing family
    already characterised in the secondary-eclipse work -- suggests stellar
    activity masquerading as a transit.
    COST: zero new data. It is a function of two columns already in
    training.csv and already produced for both candidate pools.

PROPOSAL 2 -- OVERALL TREND / SLOPE
    Production savgol divides by a ~13.4 h trend, which removes ANY structure
    slower than that -- including a full-baseline linear slope. A trend feature
    computed on PROCESSED data is therefore a null re-derivation of something
    already divided out. It must come from RAW, the same reasoning that put the
    variability features on raw. Hypothesis: a strong uncorrected slope marks
    instrumental drift, a long-period EB, or a poorly-corrected systematic
    rather than a clean planet host.

PROPOSAL 3 -- TESS SYSTEMATICS FLAGS
    Split by feasibility (see the writeup): momentum dumps are UNAVAILABLE --
    lightkurve's DEFAULT_BITMASK (17087) includes bit 32 "Desaturation event",
    and every download in this project calls a bare `.download()`, so those
    cadences never reach a CSV. Scattered light DOES survive (bits 2048/4096
    are not in the default mask). So only the straylight half is buildable
    without a full re-download, and it is what is computed here.
    Hypothesis: heavy straylight means more excluded cadences and a sparser
    curve, giving TLS more room to fit a spurious signal.
    *** PRIOR CONCERN, tested explicitly below: straylight is Earth/Moon light
    and tracks ECLIPTIC position, and this project has already measured a large
    class difference in |ecliptic latitude| (D=0.51, p=1e-67). A straylight
    feature is a prime candidate to be a spatial-confound proxy. ***
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT_CSV = os.path.join(HERE, "exominer_triple_features.csv")
OUT_JSON = os.path.join(HERE, "exominer_triple_summary.json")

RAW_DIRS = [os.path.join(ROOT, "data", "known_lightcurves"),
            os.path.join(ROOT, "data", "known_lightcurves_negative")]
STRAYLIGHT_BITS = 2048 | 4096
HARMONICS = [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]


# ----------------------------------------------------------------- proposal 1
def ls_period_match(p_ls, p_tr):
    """Log-distance from the star's dominant OOT periodicity to the nearest
    harmonic of the candidate's transit period. 0 = exactly on a harmonic.

    Log-space so that 2:1 and 1:2 are treated symmetrically -- the same
    symmetry the half-period alias work needed.
    """
    if not (np.isfinite(p_ls) and np.isfinite(p_tr)) or p_ls <= 0 or p_tr <= 0:
        return np.nan, np.nan
    d = [abs(np.log(p_ls / (n * p_tr))) for n in HARMONICS]
    i = int(np.argmin(d))
    return float(d[i]), float(HARMONICS[i])


# --------------------------------------------------------- proposals 2 and 3
def raw_features(host):
    """One pass over the RAW file for the slope and straylight features."""
    out = {"trend_slope_ppm_day": np.nan, "trend_amp_frac": np.nan,
           "straylight_frac": np.nan, "flagged_frac": np.nan,
           "raw_status": "no raw file"}
    path = None
    for d in RAW_DIRS:
        p = os.path.join(d, host + ".csv")
        if os.path.exists(p):
            path = p
            break
    if path is None:
        return out
    try:
        df = pd.read_csv(path, usecols=lambda c: c in
                         ("time", "pdcsap_flux", "sap_flux", "flux", "quality"),
                         low_memory=False)
    except Exception as e:
        out["raw_status"] = f"read error: {type(e).__name__}"
        return out

    # Non-standard schema is a real, already-known condition in this archive --
    # 02_preprocess.validate_schema rejects the same files. Guard rather than
    # let one bad file kill the whole 5,494-star pass.
    if "quality" in df.columns:
        q = pd.to_numeric(df["quality"], errors="coerce").fillna(0).astype("int64").to_numpy()
        if len(q):
            out["straylight_frac"] = float((q & STRAYLIGHT_BITS).astype(bool).mean())
            out["flagged_frac"] = float((q != 0).mean())
    else:
        q = None

    col = next((c for c in ("pdcsap_flux", "sap_flux", "flux") if c in df.columns), None)
    if col is None or "time" not in df.columns:
        out["raw_status"] = "non-standard schema (no time/flux column)"
        return out
    if q is None:
        q = np.zeros(len(df), dtype="int64")
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy()
    f = pd.to_numeric(df[col], errors="coerce").to_numpy()
    m = np.isfinite(t) & np.isfinite(f) & (q == 0)
    t, f = t[m], f[m]
    if len(t) < 200:
        out["raw_status"] = f"only {len(t)} points"
        return out
    o = np.argsort(t); t, f = t[o], f[o]
    med = np.median(f)
    if not np.isfinite(med) or med == 0:
        out["raw_status"] = "degenerate median"
        return out
    fn = f / med
    # robust slope: Theil-Sen is O(n^2); use a median-of-halves estimator,
    # which is robust to transits/flares and O(n)
    half = len(t) // 2
    t1, t2 = np.median(t[:half]), np.median(t[half:])
    f1, f2 = np.median(fn[:half]), np.median(fn[half:])
    if t2 > t1:
        slope = (f2 - f1) / (t2 - t1)
        out["trend_slope_ppm_day"] = float(abs(slope) * 1e6)
        out["trend_amp_frac"] = float(abs(slope) * (t.max() - t.min()))
    out["raw_status"] = "ok"
    return out


def main():
    df = pd.read_csv(TRAINING)
    df["host"] = df.host.astype(str)
    print(f"training.csv {len(df)} rows "
          f"({int((df.label==1).sum())} pos / {int((df.label==0).sum())} neg)")

    # ---- proposal 1: pure function of two deployed columns ----
    m = [ls_period_match(a, b) for a, b in
         zip(pd.to_numeric(df.var_ls_period, errors="coerce"),
             pd.to_numeric(df.period, errors="coerce"))]
    df["ls_period_match"] = [x[0] for x in m]
    df["ls_matched_harmonic"] = [x[1] for x in m]

    # ---- proposals 2 and 3: one raw pass ----
    print("reading raw light curves for slope + straylight...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(raw_features, df.host.tolist()))
    for k in ("trend_slope_ppm_day", "trend_amp_frac", "straylight_frac",
              "flagged_frac", "raw_status"):
        df[k] = [r[k] for r in res]
    print(f"  raw_status: {df.raw_status.value_counts().to_dict()}")

    NEW = ["ls_period_match", "trend_slope_ppm_day", "trend_amp_frac",
           "straylight_frac", "flagged_frac"]
    df[["host", "label"] + NEW + ["ls_matched_harmonic", "raw_status"]].to_csv(
        OUT_CSV, index=False)

    # ---- leakage battery ----
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols31 = list(m05.FEATURE_COLUMNS)
    y = df.label.to_numpy()

    from sklearn.metrics import roc_auc_score
    from scipy.stats import ks_2samp
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    print("\n" + "=" * 78)
    print(f"{'feature':<24}{'NaN pos':>9}{'NaN neg':>9}{'AUC':>8}{'|AUC-.5|':>10}"
          f"{'max|rho| vs 31':>16}")
    summary = {}
    for c in NEW:
        v = pd.to_numeric(df[c], errors="coerce")
        nan_p = float(v[y == 1].isna().mean()); nan_n = float(v[y == 0].isna().mean())
        ok = v.notna().to_numpy()
        auc = roc_auc_score(y[ok], v[ok]) if ok.sum() > 50 and len(set(y[ok])) > 1 else np.nan
        sub = df.loc[ok, cols31].apply(pd.to_numeric, errors="coerce")
        rho = sub.corrwith(v[ok], method="spearman").abs()
        summary[c] = {"nan_pos": nan_p, "nan_neg": nan_n, "auc": float(auc),
                      "max_rho": float(rho.max()), "max_rho_with": str(rho.idxmax())}
        print(f"{c:<24}{nan_p:>8.1%}{nan_n:>9.1%}{auc:>8.4f}{abs(auc-0.5):>10.4f}"
              f"{rho.max():>10.3f} ({rho.idxmax()[:14]})")
    print("=" * 78)
    print("redundancy threshold 0.80; a |AUC-0.5| near 0 means no single-feature signal")

    # ---- class-rate gate: is availability itself class-correlated? ----
    print("\n=== CLASS-RATE GATE (is the feature's availability class-correlated?) ===")
    from scipy.stats import fisher_exact
    for c in NEW:
        v = pd.to_numeric(df[c], errors="coerce")
        ap = int(v[y == 1].notna().sum()); an = int(v[y == 0].notna().sum())
        np_, nn_ = int((y == 1).sum()), int((y == 0).sum())
        orr, p = fisher_exact([[ap, np_ - ap], [an, nn_ - an]])
        flag = "  <-- CLASS-CORRELATED" if p < 0.05 else ""
        print(f"  {c:<24} avail pos {ap/np_:6.2%}  neg {an/nn_:6.2%}  "
              f"OR {orr:6.3f}  p {p:.4g}{flag}")

    # ---- spatial control: |galactic latitude| AND |ecliptic latitude| ----
    print("\n=== SPATIAL CONTROL (proper: does the feature survive conditioning?) ===")
    ra = pd.to_numeric(df.ra, errors="coerce").to_numpy()
    dec = pd.to_numeric(df.dec, errors="coerce").to_numpy()
    okc = np.isfinite(ra) & np.isfinite(dec)
    gb = np.full(len(df), np.nan); ecl = np.full(len(df), np.nan)
    sc = SkyCoord(ra[okc] * u.deg, dec[okc] * u.deg)
    gb[okc] = np.abs(sc.galactic.b.deg)
    ecl[okc] = np.abs(sc.barycentrictrueecliptic.lat.deg)
    df["abs_gal_b"] = gb; df["abs_ecl_lat"] = ecl
    print(f"  coordinates available for {int(okc.sum())}/{len(df)} stars")
    d = ks_2samp(ecl[(y == 1) & okc], ecl[(y == 0) & okc])
    print(f"  |ecliptic lat| pos vs neg: KS D={d.statistic:.3f}, p={d.pvalue:.3g} "
          f"(medians {np.nanmedian(ecl[(y==1)&okc]):.1f} vs "
          f"{np.nanmedian(ecl[(y==0)&okc]):.1f} deg)")
    for c in NEW:
        v = pd.to_numeric(df[c], errors="coerce")
        mm = v.notna().to_numpy() & okc
        if mm.sum() < 100:
            continue
        rg = pd.Series(v[mm].values).corr(pd.Series(gb[mm]), method="spearman")
        re_ = pd.Series(v[mm].values).corr(pd.Series(ecl[mm]), method="spearman")
        # stratified AUC: within ecliptic-latitude quartiles, does the feature
        # still separate? this is the control arm, not just a correlation
        qs = pd.qcut(pd.Series(ecl[mm]), 4, labels=False, duplicates="drop")
        aucs = []
        for k in sorted(pd.Series(qs).dropna().unique()):
            s = (qs == k).to_numpy()
            yy = y[mm][s]; vv = v[mm].values[s]
            if len(set(yy)) > 1 and s.sum() > 50:
                aucs.append(roc_auc_score(yy, vv))
        summary[c].update({"rho_gal_b": float(rg), "rho_ecl_lat": float(re_),
                           "stratified_aucs": [float(a) for a in aucs]})
        print(f"  {c:<24} rho|gal_b| {rg:+.3f}  rho|ecl_lat| {re_:+.3f}  "
              f"AUC by ecl-lat quartile {[round(a,3) for a in aucs]}")

    json.dump(summary, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()
