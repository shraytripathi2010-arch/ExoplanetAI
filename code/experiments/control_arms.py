"""control_arms.py -- the standing confound control arms, as reusable functions.

Two arms are mandatory for any newly-computed feature in this project:

  |GALACTIC LATITUDE| -- the SPATIAL control. Long-standing. Catches features
  that are really crowding/extinction proxies. A correlation coefficient is not
  enough; the per-quartile AUC is what caught `trend_slope` (0.434-0.715) and
  `fls_p2_nh` (monotone decay 0.259 -> 0.490).

  SECTOR / OBSERVATION EPOCH -- the TEMPORAL control. Added 2026-08-27 as a
  standing requirement after the momentum-dump investigation, where a control
  arm built from sector-only columns returned +0.0063 with 12/12 bootstraps
  positive while every real feature sat at zero. Traced to a 6.6x swing in the
  negative-class rate across sector eras (0.079 in S70-84 vs 0.522 in S27-39)
  and AUC(sector alone) = 0.5803 -- TFOP disposition history, not astrophysics.

  The |b| arm CANNOT catch the temporal one: they are close to independent.

Sector is recovered from the light curve's own time range against the per-sector
table built by `momentum_dump_schedule.py`, so no new download is needed.

Usage:
    from control_arms import gal_b_control, sector_epoch_control, both_controls
    res = both_controls(df, y, ["my_feature_a", "my_feature_b"])
"""
import os
import json
import glob
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
SCHED = os.path.join(HERE, "momentum_dump_schedule.json")
RAW_DIRS = [os.path.join(ROOT, "data", "known_lightcurves"),
            os.path.join(ROOT, "data", "known_lightcurves_negative"),
            os.path.join(ROOT, "data", "retrain_pipeline", "raw"),
            os.path.join(ROOT, "data", "unknown_lightcurves"),
            os.path.join(ROOT, "data", "unknown_lightcurves_widesector")]
BTJD_MAX = 100000.0
_SEC_CACHE = os.path.join(HERE, "sector_map.csv")

# a feature whose quartile AUCs span more than this is flagged unstable; the
# closed `trend_*` features sat at 0.281 and were rejected on it
SPREAD_FLAG = 0.20


def _auc(y, v):
    v = pd.to_numeric(pd.Series(np.asarray(v, dtype=float)), errors="coerce")
    v = v.replace([np.inf, -np.inf], np.nan)
    ok = v.notna().to_numpy()
    yy = np.asarray(y)
    if ok.sum() < 50 or len(np.unique(yy[ok])) < 2:
        return np.nan
    return float(roc_auc_score(yy[ok], v[ok]))


def _quartile_arm(y, v, strat, nq=4):
    """AUC of `v` within each quartile of `strat`, plus the spread."""
    v = pd.to_numeric(pd.Series(np.asarray(v, dtype=float)), errors="coerce")
    q = pd.qcut(pd.Series(strat), nq, labels=False, duplicates="drop")
    aucs = []
    for k in range(nq):
        m = (q == k).to_numpy()
        aucs.append(_auc(np.asarray(y)[m], v.to_numpy()[m]) if m.sum() else np.nan)
    spread = float(np.nanmax(aucs) - np.nanmin(aucs)) if np.isfinite(aucs).any() else np.nan
    return aucs, spread


# ------------------------------------------------------------------ |b| arm
def galactic_b(df):
    """|galactic latitude| in degrees, from the ra/dec columns. NaN where absent."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra = pd.to_numeric(df["ra"], errors="coerce")
    dec = pd.to_numeric(df["dec"], errors="coerce")
    ok = ra.notna() & dec.notna()
    b = pd.Series(np.nan, index=df.index)
    if ok.any():
        b[ok] = np.abs(SkyCoord(ra[ok].values * u.deg, dec[ok].values * u.deg,
                                frame="icrs").galactic.b.deg)
    return b


def gal_b_control(df, y, features):
    b = galactic_b(df)
    out = {"n_with_b": int(b.notna().sum())}
    for c in features:
        v = df[c]
        aucs, spread = _quartile_arm(y, v, b)
        out[c] = {"rho": float(pd.to_numeric(v, errors="coerce").corr(b, method="spearman")),
                  "quartile_auc": aucs, "spread": spread,
                  "unstable": bool(np.isfinite(spread) and spread > SPREAD_FLAG)}
    return out


# --------------------------------------------------------------- sector arm
def _scan_one(p):
    try:
        t = pd.read_csv(p, usecols=lambda c: c == "time", low_memory=False)["time"]
        t = pd.to_numeric(t, errors="coerce").dropna()
        if len(t) < 50:
            return None
        return (os.path.splitext(os.path.basename(p))[0], float(t.min()), float(t.max()))
    except Exception:
        return None


def build_sector_map(force=False):
    """host -> TESS sector, from each light curve's time range against the
    per-sector table. Cached to sector_map.csv; ~3 min to rebuild."""
    if os.path.exists(_SEC_CACHE) and not force:
        m = pd.read_csv(_SEC_CACHE)
        m["host"] = m.host.astype(str)
        return m
    sched = json.load(open(SCHED))["sectors"]
    spans = [(v["sector"], v["t0"], v["t1"]) for v in sched.values()]
    files = []
    for d in RAW_DIRS:
        files += sorted(glob.glob(os.path.join(d, "*.csv")))
    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = [r for r in ex.map(_scan_one, files) if r]
    out = []
    for host, t0, t1 in rows:
        if t0 > BTJD_MAX:
            out.append((host, np.nan)); continue
        best, bov = np.nan, 0.0
        for sec, s0, s1 in spans:
            ov = min(t1, s1) - max(t0, s0)
            if ov > bov:
                best, bov = sec, ov
        out.append((host, best if bov > 1.0 else np.nan))
    m = pd.DataFrame(out, columns=["host", "sector"]).drop_duplicates("host")
    m["host"] = m.host.astype(str)
    m.to_csv(_SEC_CACHE, index=False)
    return m


def sector_epoch_control(df, y, features, host_col="host"):
    """The TEMPORAL control arm. Reports, for each feature, its correlation with
    sector and its AUC within each sector-era quartile -- and, separately, how
    much signal SECTOR ITSELF carries, which is the number that matters."""
    m = build_sector_map()
    sec = pd.Series(df[host_col].astype(str)).map(
        dict(zip(m.host, pd.to_numeric(m.sector, errors="coerce")))).astype(float)
    y = np.asarray(y)
    ok = sec.notna().to_numpy()
    out = {"n_with_sector": int(ok.sum()),
           "auc_sector_alone": _auc(y[ok], sec[ok].to_numpy()),
           "negative_rate_by_era": {}}
    for lo, hi in ((1, 13), (14, 26), (27, 39), (40, 55), (56, 69), (70, 84), (85, 120)):
        k = ((sec >= lo) & (sec <= hi)).fillna(False).to_numpy()
        if k.sum() > 30:
            out["negative_rate_by_era"][f"{lo}-{hi}"] = {
                "n": int(k.sum()), "neg_rate": float(1 - y[k].mean())}
    for c in features:
        v = df[c]
        aucs, spread = _quartile_arm(y, v, sec)
        out[c] = {"rho_sector": float(pd.to_numeric(v, errors="coerce")
                                     .corr(sec, method="spearman")),
                  "quartile_auc": aucs, "spread": spread,
                  "unstable": bool(np.isfinite(spread) and spread > SPREAD_FLAG)}
    return out


def both_controls(df, y, features, host_col="host", verbose=True):
    res = {"gal_b": gal_b_control(df, y, features),
           "sector_epoch": sector_epoch_control(df, y, features, host_col)}
    if verbose:
        print("=" * 88)
        print("CONTROL ARMS  (spatial + temporal; both are standing requirements)")
        print("=" * 88)
        g, s = res["gal_b"], res["sector_epoch"]
        print(f"  |b| available {g['n_with_b']};  sector resolved {s['n_with_sector']}")
        print(f"  AUC(SECTOR ALONE) = {s['auc_sector_alone']:.4f}   "
              f"<- the confound's own strength")
        print("  negative-class rate by sector era: " +
              "  ".join(f"{k}:{v['neg_rate']:.3f}"
                        for k, v in s["negative_rate_by_era"].items()))
        print(f"\n{'feature':<20}{'rho |b|':>9}{'|b| spread':>12}"
              f"{'rho sector':>12}{'sec spread':>12}   flags")
        for c in features:
            fl = []
            if g[c]["unstable"]: fl.append("SPATIAL")
            if s[c]["unstable"]: fl.append("TEMPORAL")
            print(f"{c:<20}{g[c]['rho']:>9.3f}{g[c]['spread']:>12.3f}"
                  f"{s[c]['rho_sector']:>12.3f}{s[c]['spread']:>12.3f}   "
                  f"{','.join(fl) if fl else 'ok'}")
    return res
