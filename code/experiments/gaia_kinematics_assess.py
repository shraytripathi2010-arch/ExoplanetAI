"""gaia_kinematics_assess.py -- PARTS 2 and 3.

PART 3 IS THE POINT. Kinematics are computed FROM sky position and proper
motion, so a naive test can rediscover "galactic latitude predicts the label"
wearing a physical costume. The standard correlation-based spatial arm is not
strong enough evidence here. This reuses the CROWDING investigation's
matched-sky-band method exactly: restrict both classes to |b| in [8, 40] deg,
where the class difference in |b| INVERTS (planets 15.9 vs FP 19.7 inside the
band, against 16.9 vs 12.5 outside), and ask whether the separation survives.

FEATURES, in two families kept apart because their availability differs:

  TANGENTIAL family (no RV; 97.20% training)
    kin_vtan      4.74 * mu / plx                      km/s
    kin_absz      |distance * sin(b)|                  pc -- height above the plane
    kin_dist      1000/plx                             pc

  FULL-3D family (needs RV; 77.29% training, and +7.85 pp class-split)
    kin_U kin_V kin_W    Galactic space velocities, astropy Galactocentric
    kin_vtot             sqrt(U^2+V^2+W^2)
    kin_thickthin        Bensby et al. 2003 thick/thin probability RATIO

Bensby is used rather than a heavier dynamical package: it is a closed-form
Gaussian-ellipsoid ratio, needs no new dependency, and is the standard
reference for exactly this thin/thick assignment.
"""
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
RAW = os.path.join(HERE, "gaia_kinematics_raw.csv")
OUT = os.path.join(HERE, "gaia_kinematics_assess.json")
OUT_CSV = os.path.join(HERE, "gaia_kinematics_features.csv")

TANGENTIAL = ["kin_vtan", "kin_absz", "kin_dist"]
FULL3D = ["kin_U", "kin_V", "kin_W", "kin_vtot", "kin_thickthin"]
BAND = (8.0, 40.0)          # the crowding investigation's matched band

# Bensby et al. 2003 velocity ellipsoids
POP = {"thin":  dict(sU=35., sV=20., sW=16., Vasym=-15., X=0.94),
       "thick": dict(sU=67., sV=38., sW=35., Vasym=-46., X=0.06),
       "halo":  dict(sU=160., sV=90., sW=90., Vasym=-220., X=0.0015)}


def f_pop(U, V, W, p):
    k = 1.0 / ((2 * np.pi) ** 1.5 * p["sU"] * p["sV"] * p["sW"])
    return k * np.exp(-U**2 / (2 * p["sU"]**2)
                      - (V - p["Vasym"])**2 / (2 * p["sV"]**2)
                      - W**2 / (2 * p["sW"]**2))


def build(d):
    import astropy.units as u
    from astropy.coordinates import SkyCoord, Galactic
    out = pd.DataFrame(index=d.index)
    ra = pd.to_numeric(d.ra, errors="coerce")
    dec = pd.to_numeric(d.dec, errors="coerce")
    pmra = pd.to_numeric(d.pmra, errors="coerce")
    pmdec = pd.to_numeric(d.pmdec, errors="coerce")
    plx = pd.to_numeric(d.plx, errors="coerce")
    rv = pd.to_numeric(d.rv, errors="coerce")
    plx = plx.where(plx > 0.1)                      # reject unusable parallaxes
    dist_pc = 1000.0 / plx
    mu = np.hypot(pmra, pmdec)
    out["kin_dist"] = dist_pc
    out["kin_vtan"] = 4.74047 * mu * dist_pc / 1000.0

    ok = ra.notna() & dec.notna()
    b = pd.Series(np.nan, index=d.index)
    if ok.any():
        b[ok] = SkyCoord(ra[ok].values * u.deg, dec[ok].values * u.deg,
                         frame="icrs").galactic.b.deg
    out["kin_absz"] = np.abs(dist_pc * np.sin(np.radians(b)))
    out["_absb"] = np.abs(b)

    full = ok & pmra.notna() & pmdec.notna() & plx.notna() & rv.notna()
    for c in ("kin_U", "kin_V", "kin_W", "kin_vtot", "kin_thickthin"):
        out[c] = np.nan
    if full.any():
        i = np.where(full)[0]
        sc = SkyCoord(ra=ra[full].values * u.deg, dec=dec[full].values * u.deg,
                      distance=(dist_pc[full].values * u.pc),
                      pm_ra_cosdec=pmra[full].values * u.mas / u.yr,
                      pm_dec=pmdec[full].values * u.mas / u.yr,
                      radial_velocity=rv[full].values * u.km / u.s, frame="icrs")
        g = sc.transform_to(Galactic())
        v = g.velocity.d_xyz.to(u.km / u.s).value
        U, V, W = v[0], v[1], v[2]
        out.iloc[i, out.columns.get_loc("kin_U")] = U
        out.iloc[i, out.columns.get_loc("kin_V")] = V
        out.iloc[i, out.columns.get_loc("kin_W")] = W
        out.iloc[i, out.columns.get_loc("kin_vtot")] = np.sqrt(U**2 + V**2 + W**2)
        fd = f_pop(U, V, W, POP["thin"]) * POP["thin"]["X"]
        ft = f_pop(U, V, W, POP["thick"]) * POP["thick"]["X"]
        out.iloc[i, out.columns.get_loc("kin_thickthin")] = np.log10(
            np.clip(ft, 1e-300, None) / np.clip(fd, 1e-300, None))
    return out


def auc(y, v):
    v = pd.to_numeric(pd.Series(np.asarray(v, float)), errors="coerce")
    v = v.replace([np.inf, -np.inf], np.nan)
    ok = v.notna().to_numpy()
    yy = np.asarray(y)
    if ok.sum() < 50 or len(np.unique(yy[ok])) < 2:
        return np.nan
    return float(roc_auc_score(yy[ok], v[ok]))


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)
    cols33 = list(m05.FEATURE_COLUMNS)

    raw = pd.read_csv(RAW)
    tr_raw = raw[raw._set == "train"].reset_index(drop=True)
    K = build(tr_raw)
    tr = pd.read_csv(TRAINING)
    assert len(tr) == len(K)
    df = pd.concat([tr.reset_index(drop=True), K], axis=1)
    df.to_csv(OUT_CSV, index=False)
    y = df.label.to_numpy()
    ALL = TANGENTIAL + FULL3D
    res = {}

    print("=" * 90); print("PART 2 -- SINGLE-FEATURE AUC and COVERAGE"); print("=" * 90)
    print(f"{'feature':<18}{'coverage':>10}{'AUC':>9}{'|AUC-0.5|':>11}")
    for c in ALL:
        v = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        a = auc(y, v)
        res.setdefault("single", {})[c] = {"cov": float(v.notna().mean()), "auc": a}
        print(f"{c:<18}{v.notna().mean():>9.2%}{a:>9.4f}{abs(a-0.5):>11.4f}")

    print("\n" + "=" * 90)
    print("PART 2 -- CORRELATION vs the 33, and vs |b| ITSELF (the key one here)")
    print("=" * 90)
    X = df[cols33].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    absb = df["_absb"]
    print(f"{'feature':<18}{'|rho| vs |b|':>14}{'max |rho| /33':>15}  which")
    for c in ALL:
        v = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rb = float(abs(v.corr(absb, method="spearman")))
        rr = {k: float(abs(v.corr(X[k], method="spearman"))) for k in cols33
              if np.isfinite(v.corr(X[k], method="spearman"))}
        mk = max(rr, key=rr.get) if rr else None
        res.setdefault("corr", {})[c] = {"vs_absb": rb, "max33": rr.get(mk), "with": mk}
        flag = "   REDUNDANT" if (mk and rr[mk] > 0.80) else ""
        print(f"{c:<18}{rb:>14.3f}{rr.get(mk, float('nan')):>15.3f}  {mk}{flag}")

    print("\n" + "=" * 90)
    print("PART 3 -- MATCHED-SKY-BAND TEST  (the deciding check)")
    print("=" * 90)
    inband = ((absb >= BAND[0]) & (absb <= BAND[1])).to_numpy()
    print(f"  band |b| in [{BAND[0]}, {BAND[1]}] deg retains "
          f"{int(inband.sum())}/{len(df)} ({inband.mean():.1%})")
    mp_all = float(absb[y == 1].median()); mn_all = float(absb[y == 0].median())
    mp_in = float(absb[inband & (y == 1)].median()); mn_in = float(absb[inband & (y == 0)].median())
    print(f"  median |b| OUTSIDE control: planets {mp_all:.1f} deg, FPs {mn_all:.1f} deg")
    print(f"  median |b| INSIDE  band   : planets {mp_in:.1f} deg, FPs {mn_in:.1f} deg"
          f"   -> class difference {'INVERTS' if (mp_all-mn_all)*(mp_in-mn_in) < 0 else 'does NOT invert'}")
    res["band"] = {"n_in": int(inband.sum()), "median_b_pos_all": mp_all,
                   "median_b_neg_all": mn_all, "median_b_pos_in": mp_in,
                   "median_b_neg_in": mn_in}
    print(f"\n{'feature':<18}{'AUC full':>10}{'AUC in-band':>13}{'retained':>11}   verdict")
    for c in ALL:
        v = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        a_all = auc(y, v); a_in = auc(y[inband], v.to_numpy()[inband])
        s_all = abs(a_all - 0.5); s_in = abs(a_in - 0.5)
        keep = s_in / s_all if s_all > 0 else np.nan
        sign_flip = (a_all - 0.5) * (a_in - 0.5) < 0
        verdict = "SIGN FLIPS -- positional" if sign_flip else (
            "survives" if keep > 0.5 else "mostly positional")
        res.setdefault("band_test", {})[c] = {"auc_all": a_all, "auc_in": a_in,
                                              "retained": float(keep),
                                              "sign_flip": bool(sign_flip)}
        print(f"{c:<18}{a_all:>10.4f}{a_in:>13.4f}{keep:>10.0%}   {verdict}")

    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
