"""trapezoid_checks.py -- Part 2 pre-model checks for the fitted trapezoid
shape metric, run in order, with the CLASS-RATE GATE last and decisive.

The order matters. Every check here is cheap; the model fit is not. The
just-closed odd-even TIMING investigation established the discipline: if the
two classes are not distinguishable in the raw feature, a 12-resample retrain
cannot manufacture the ~0.0097 needed to clear, and running one anyway spends
hours to reach a conclusion the class rates already gave for free.

CHECKS
  1. coverage, training + BOTH candidate pools, with failure breakdown
  2. NaN-rate by class, and the AUC of mere availability (leakage via
     missingness -- the trap `contratio` fell into)
  3. correlation against all 26 production features, `transit_shape_ratio`
     called out specifically: the old column is geometrically meaningless for
     55% of stars, so a HIGH correlation would mean the new fit is inheriting
     the same geometry rather than measuring shape, and a LOW one confirms they
     measure different things
  4. spatial exposure vs |galactic b| -- checked, not assumed, per the lesson
     that this project's training set IS spatially confounded
  5. THE CLASS-RATE GATE: does the shape metric actually differ between
     planets and false positives
"""
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import fast_auc  # noqa: E402

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
FEAT = os.path.join(SCRIPT_DIR, "trapezoid_shape_features.csv")
POOL = os.path.join(SCRIPT_DIR, "trapezoid_shape_pool.csv")
WIDE = os.path.join(SCRIPT_DIR, "trapezoid_shape_widesector.csv")
CAND_POOL = os.path.join(ROOT, "data", "catalogs", "unknown_features.csv")
CAND_WIDE = os.path.join(ROOT, "data", "catalogs", "unknown_features_widesector.csv")
OUT = os.path.join(SCRIPT_DIR, "trapezoid_checks.json")

# A converged fit is not automatically a USABLE one. A shallow, noisy transit
# has an ingress below the noise floor, so w is unidentifiable however cleanly
# least_squares terminates. These thresholds are applied after looking at the
# measured distributions, and both are reported with and without.
MIN_DEPTH_SNR = 3.0
MAX_VSHAPE_ERR = 0.30

FEATURES = ["trap_vshape", "trap_t14_ratio"]


def _m05():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def usable(df):
    """Boolean mask of fits whose shape parameter is actually measured."""
    ok = df["trap_vshape"].notna() & (df["trap_status"] == "ok")
    snr = pd.to_numeric(df["trap_depth_snr"], errors="coerce")
    err = pd.to_numeric(df["trap_vshape_err"], errors="coerce")
    return ok & (snr >= MIN_DEPTH_SNR) & (err <= MAX_VSHAPE_ERR)


def describe(name, v, y):
    p, f = v[y == 1], v[y == 0]
    p, f = p[np.isfinite(p)], f[np.isfinite(f)]
    a = fast_auc(np.r_[np.ones(len(p)), np.zeros(len(f))], np.r_[p, f])
    return {"feature": name, "n_planet": len(p), "n_fp": len(f),
            "median_planet": float(np.median(p)), "median_fp": float(np.median(f)),
            "mean_planet": float(p.mean()), "mean_fp": float(f.mean()),
            "p25_planet": float(np.percentile(p, 25)), "p75_planet": float(np.percentile(p, 75)),
            "p25_fp": float(np.percentile(f, 25)), "p75_fp": float(np.percentile(f, 75)),
            "auc": float(a)}


def main():
    res = {}
    df = pd.read_csv(TRAINING)
    tf = pd.read_csv(FEAT)
    m = df[["host", "label"]].merge(tf, on="host", how="left")
    y = m["label"].to_numpy(float)

    print("=" * 100)
    print("TRAPEZOID SHAPE -- Part 2 pre-model checks")
    print("=" * 100)

    # ---- 1. coverage -----------------------------------------------------
    print("\n[1] COVERAGE")
    conv = m["trap_vshape"].notna()
    use = usable(m)
    print(f"  training  fit converged   {conv.sum():>5}/{len(m)} = {conv.mean()*100:5.1f}%")
    print(f"  training  USABLE          {use.sum():>5}/{len(m)} = {use.mean()*100:5.1f}%"
          f"   (depth_snr >= {MIN_DEPTH_SNR}, vshape_err <= {MAX_VSHAPE_ERR})")
    print("\n  failure / downgrade breakdown:")
    reasons = m.loc[~use].copy()
    lab = reasons["trap_status"].fillna("no row").astype(str)
    snr = pd.to_numeric(reasons["trap_depth_snr"], errors="coerce")
    err = pd.to_numeric(reasons["trap_vshape_err"], errors="coerce")
    lab = np.where((lab == "ok") & (snr < MIN_DEPTH_SNR), "converged but depth SNR < 3", lab)
    lab = np.where((lab == "ok") & (err > MAX_VSHAPE_ERR), "converged but shape unconstrained", lab)
    vc = pd.Series(lab).value_counts()
    for r, n in vc.items():
        print(f"    {n:>5}  ({n/len(m)*100:4.1f}%)  {str(r)[:60]}")
    res["coverage_training"] = {"converged": float(conv.mean()), "usable": float(use.mean()),
                                "n": int(len(m)), "reasons": {str(k): int(v) for k, v in vc.items()}}

    # POOL DENOMINATOR. The raw candidate feature files carry one row per star
    # ATTEMPTED, including stars whose TLS post-processing failed before
    # period/T0/duration were ever written. Those rows have no ephemeris for
    # ANY phase-folded feature -- deployed or new -- so scoring the trapezoid
    # against them measures TLS's failure rate, not this feature's coverage.
    # The honest denominator is the rows that actually reached scoring.
    for nm, path, src in (("pool (unknown)", POOL, CAND_POOL),
                          ("pool (widesector)", WIDE, CAND_WIDE)):
        if not (os.path.exists(path) and os.path.exists(src)):
            print(f"  {nm}: NOT COMPUTED"); continue
        pf = pd.read_csv(path)
        sf = pd.read_csv(src)
        p_ = pd.to_numeric(sf["period"], errors="coerce")
        t_ = pd.to_numeric(sf["T0"], errors="coerce")
        d_ = pd.to_numeric(sf["duration"], errors="coerce")
        scorable = set(sf.loc[np.isfinite(p_) & np.isfinite(t_) & np.isfinite(d_)
                              & (p_ > 0) & (d_ > 0) & (d_ < p_), "host"])
        sub = pf[pf["host"].isin(scorable)]
        pu = usable(sub)
        print(f"  {nm:<20} rows {len(pf):>5}   scorable (has ephemeris) {len(sub):>4}"
              f"   converged {sub['trap_vshape'].notna().mean()*100:5.1f}%"
              f"   USABLE {pu.mean()*100:5.1f}%")
        res[f"coverage_{nm}"] = {"n_rows": int(len(pf)), "n_scorable": int(len(sub)),
                                 "converged": float(sub["trap_vshape"].notna().mean()),
                                 "usable": float(pu.mean())}

    # ---- 2. missingness by class + availability AUC ----------------------
    print("\n[2] MISSINGNESS BY CLASS")
    up, uf = use[y == 1].mean(), use[y == 0].mean()
    av = fast_auc(y, use.to_numpy(float))
    print(f"  usable rate: planets {up*100:.1f}%   false positives {uf*100:.1f}%"
          f"   gap {(up-uf)*100:+.1f}pp")
    print(f"  AUC of AVAILABILITY alone: {av:.4f}   (0.5 = no leakage via missingness)")
    res["missingness"] = {"usable_planet": float(up), "usable_fp": float(uf),
                          "availability_auc": float(av)}

    # ---- 3. correlation vs the 26 production features --------------------
    print("\n[3] CORRELATION vs THE 26 PRODUCTION FEATURES")
    m05 = _m05()
    X, _ = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    cors = {}
    for feat in FEATURES:
        v = pd.to_numeric(m[feat], errors="coerce").where(use)
        c = {col: float(abs(pd.Series(v).corr(pd.to_numeric(X[col], errors="coerce"))))
             for col in X.columns}
        c = {k: (0.0 if not np.isfinite(vv) else vv) for k, vv in c.items()}
        cors[feat] = c
        top = sorted(c.items(), key=lambda kv: -kv[1])[:5]
        print(f"\n  {feat}:")
        print(f"    vs transit_shape_ratio  |r| = {c['transit_shape_ratio']:.3f}   <-- the broken one")
        print(f"    max |r| over all 26     |r| = {max(c.values()):.3f}  ({max(c, key=c.get)})")
        print("    top 5: " + ", ".join(f"{k} {vv:.3f}" for k, vv in top))
    res["correlations"] = cors

    # ---- 4. spatial exposure ---------------------------------------------
    print("\n[4] SPATIAL EXPOSURE vs |galactic b|")
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy()
        dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy()
        ok = np.isfinite(ra) & np.isfinite(dec)
        gb = np.full(len(df), np.nan)
        gb[ok] = np.abs(SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg).galactic.b.deg)
        sp = {}
        for feat in FEATURES:
            v = pd.to_numeric(m[feat], errors="coerce").where(use)
            r = float(abs(pd.Series(v).corr(pd.Series(gb))))
            sp[feat] = r
            print(f"  {feat:<18} |r| vs |b| = {r:.3f}")
        res["spatial"] = sp
    except Exception as e:
        print(f"  could not compute: {e}")

    # ---- 5. THE CLASS-RATE GATE ------------------------------------------
    print("\n" + "=" * 100)
    print("[5] CLASS-RATE GATE -- does the shape metric separate the classes?")
    print("=" * 100)
    gate = []
    for feat in FEATURES:
        v = pd.to_numeric(m[feat], errors="coerce").where(use).to_numpy(float)
        d = describe(feat, v, y)
        gate.append(d)
        print(f"\n  {feat}")
        print(f"    {'group':<18}{'n':>7}{'median':>10}{'mean':>10}{'p25':>10}{'p75':>10}")
        print(f"    {'planets':<18}{d['n_planet']:>7}{d['median_planet']:>10.4f}"
              f"{d['mean_planet']:>10.4f}{d['p25_planet']:>10.4f}{d['p75_planet']:>10.4f}")
        print(f"    {'false positives':<18}{d['n_fp']:>7}{d['median_fp']:>10.4f}"
              f"{d['mean_fp']:>10.4f}{d['p25_fp']:>10.4f}{d['p75_fp']:>10.4f}")
        print(f"    single-feature AUC = {d['auc']:.4f}")

    # tail rates: the V-shaped extreme is where grazing/EB signal should live
    print("\n  TAIL RATES for trap_vshape (V-shaped extreme = grazing / EB signature)")
    v = pd.to_numeric(m["trap_vshape"], errors="coerce").where(use).to_numpy(float)
    print(f"    {'group':<18}{'n':>7}" + "".join(f"{'>'+str(t):>10}" for t in (0.5, 0.7, 0.85, 0.95)))
    tails = {}
    for nm, cls in (("planets", 1.0), ("false positives", 0.0)):
        s = v[(y == cls) & np.isfinite(v)]
        row = [float((s > t).mean()) for t in (0.5, 0.7, 0.85, 0.95)]
        tails[nm] = row
        print(f"    {nm:<18}{len(s):>7}" + "".join(f"{r*100:>9.1f}%" for r in row))
    res["gate"] = gate
    res["tails"] = tails

    best = max(abs(g["auc"] - 0.5) for g in gate)
    print("\n" + "=" * 100)
    print(f"  strongest class separation: AUC deviation {best:+.4f} from 0.5")
    print("=" * 100)
    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
