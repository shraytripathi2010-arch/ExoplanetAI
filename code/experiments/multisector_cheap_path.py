"""multisector_cheap_path.py -- fold at the STORED ephemeris across all
available sectors and recompute only the fold-derived features.

THE CHEAP PATH, AND WHY IT EXISTS. Stage 0 costed a full multi-sector rebuild
at 7-27 days, dominated by re-running TLS at 2.2-year median baselines. But
TLS exists to find an UNKNOWN period, and period/T0/duration are already stored
for every training star. Folding at the known ephemeris needs no search, which
collapses the cost to the download -- the same trick the medium-lift
multi-sector work used to avoid ~118 h.

IN SCOPE (fold-derived, no search needed):
    depth_mean, depth_mean_odd, depth_mean_even, odd_even_mismatch,
    secondary_eclipse_depth, trap_vshape
OUT OF SCOPE (require the TLS search, explicitly not touched):
    SDE, SDE_raw, FAP, chi2red_min, period, duration, transit_count

THE EXPECTATION, STATED BEFORE MEASURING. The features this path can reach
carry low importance in the deployed model -- `depth_mean` is measured at
-0.0013, i.e. actively negative. The two dominant features (st_rad +0.0540,
st_teff +0.0369) are catalog-derived and untouchable by any amount of
photometry. So the expected outcome is that stacking sharpens these features'
PRECISION without moving their discriminating power, closing the question.
Recorded here so the result cannot be reinterpreted after the fact.

THE THREE BREAK-RISKS FROM STAGE 0, EACH ADDRESSED BY CONSTRUCTION:

1. `MAX_FLATTEN_WINDOW = 401` is specified in POINTS, so its physical duration
   scales with cadence -- the K2 units bug. One star's sectors mix 20-s, 120-s
   and FFI cadences. FIX: the window is derived per sector from that sector's
   OWN measured cadence to hold a fixed PHYSICAL duration (13.4 h, matching
   what 401 points means at TESS 2-min). Every sector's cadence and resulting
   window is recorded so the fix is auditable rather than asserted.

2. Detrending across month-to-year gaps. A Savitzky-Golay pass over a
   concatenated 808-day series smooths straight across the gaps. FIX: each
   sector is flattened and normalised INDEPENDENTLY, and only then
   concatenated. No smoothing is ever applied to the stacked series.

3. Disk. Stage 2 was costed at ~200 GB against 12 GiB free. FIX:
   download-measure-delete -- each sector is downloaded to a scratch dir,
   reduced to arrays, and the file deleted before the next one. Peak on-disk
   footprint is one sector. Measured and reported, not assumed.
"""
import os
import sys
import json
import time
import shutil
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
sys.path.insert(0, SCRIPT_DIR)
from trapezoid_shape import trapezoid  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CACHE = os.path.join(os.environ.get("TMPDIR", "/tmp"), "ms_cheap_cache")

FLATTEN_HOURS = 13.4          # what 401 points means at TESS 2-min cadence
SIGMA_CLIP = 5.0
MIN_PTS_SECTOR = 200
N_BINS = 200
BINS_PER_DURATION = 12
WINDOW_DURATIONS = 2.5


def flatten_one_sector(t, f, cadence_d):
    """Per-sector detrend. Window fixed in PHYSICAL time, derived from this
    sector's own cadence -- this is break-risk 1's fix."""
    npts = int(round((FLATTEN_HOURS / 24.0) / cadence_d))
    npts = max(11, min(npts | 1, (len(f) // 2) | 1))   # odd, and < half the series
    if len(f) < max(50, npts):
        return None, npts
    try:
        trend = savgol_filter(f, npts, 2)
    except Exception:
        return None, npts
    good = np.isfinite(trend) & (trend != 0)
    if good.sum() < 50:
        return None, npts
    out = np.full_like(f, np.nan)
    out[good] = f[good] / trend[good]
    return out, npts


def fold_features(t, f, period, t0, duration):
    """Every fold-derived feature, from one (possibly stacked) series."""
    r = {k: np.nan for k in ("depth_mean", "depth_mean_odd", "depth_mean_even",
                             "odd_even_mismatch", "secondary_eclipse_depth",
                             "trap_vshape", "depth_se", "n_in_transit")}
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if len(t) < 100:
        return r
    cyc = (t - t0) / period
    ep = np.round(cyc).astype(int)
    ph = cyc - ep
    dph = duration / period
    if not (0 < dph < 0.4):
        return r
    intr = np.abs(ph) < dph / 2
    oot = (np.abs(ph) > dph) & (np.abs(np.abs(ph) - 0.5) > dph)
    if intr.sum() < 10 or oot.sum() < 30:
        return r
    base = float(np.median(f[oot]))
    r["n_in_transit"] = int(intr.sum())
    r["depth_mean"] = base - float(np.median(f[intr]))
    # precision of the depth: this is what stacking should sharpen
    r["depth_se"] = float(np.std(f[intr], ddof=1) / np.sqrt(intr.sum()))

    for nm, sel in (("odd", intr & (ep % 2 != 0)), ("even", intr & (ep % 2 == 0))):
        if sel.sum() >= 5:
            r[f"depth_mean_{nm}"] = base - float(np.median(f[sel]))
    o, e = intr & (ep % 2 != 0), intr & (ep % 2 == 0)
    if o.sum() >= 5 and e.sum() >= 5:
        so = np.std(f[o], ddof=1) / np.sqrt(o.sum())
        se = np.std(f[e], ddof=1) / np.sqrt(e.sum())
        den = np.sqrt(so ** 2 + se ** 2)
        if den > 0:
            r["odd_even_mismatch"] = abs(r["depth_mean_odd"] - r["depth_mean_even"]) / den

    sec = np.abs(np.abs(ph) - 0.5) < dph / 2
    if sec.sum() >= 5:
        r["secondary_eclipse_depth"] = base - float(np.median(f[sec]))

    # trapezoid, same parameterisation as the deployed trap_vshape work
    win = min(WINDOW_DURATIONS * dph, 0.5)
    s = np.abs(ph) <= win
    if s.sum() >= 30:
        bw = dph / BINS_PER_DURATION
        nb = max(int(round(2 * win / bw)), 25)
        edges = np.linspace(-win, win, nb + 1)
        idx = np.clip(np.digitize(ph[s], edges) - 1, 0, nb - 1)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        bx, by, bn = [], [], []
        for b in range(nb):
            q = idx == b
            if q.sum() >= 3:
                bx.append(ctr[b]); by.append(np.median(f[s][q])); bn.append(q.sum())
        if len(bx) >= 25:
            bx, by, w = np.asarray(bx), np.asarray(by), np.sqrt(np.asarray(bn, float))
            oo = np.abs(bx) > dph
            b0 = float(np.median(by[oo])) if oo.sum() >= 3 else float(np.median(by))
            it = np.abs(bx) < dph / 2
            d0 = float(np.clip(b0 - np.min(by[it]) if it.any() else 1e-4, 1e-6, 0.5))
            p0 = [b0, d0, dph, 0.4, 0.0]
            lo = [b0 - .05, 1e-7, .3 * dph, 0.0, -.5 * dph]
            hi = [b0 + .05, 0.5, 2.5 * dph, 1.0, .5 * dph]
            p0 = [float(np.clip(v, l + 1e-12, h - 1e-12)) for v, l, h in zip(p0, lo, hi)]
            try:
                res = least_squares(lambda p: (trapezoid(bx, *p) - by) * w,
                                    p0, bounds=(lo, hi), max_nfev=2000)
                if res.status > 0:
                    r["trap_vshape"] = float(res.x[3])
            except Exception:
                pass
    return r


def load_sectors(host, max_sectors=None):
    """Download every sector, flatten each INDEPENDENTLY, delete as we go.
    Returns (list of (t,f) per sector, diagnostics)."""
    from lightkurve import search_lightcurve
    s = str(host)
    tgt = ("TIC " + s.split("_", 1)[1]) if s.startswith("TIC_") else s.replace("_", " ")
    sr = None
    for kw in ({"mission": "TESS", "author": "SPOC"}, {"mission": "TESS"}):
        try:
            q = search_lightcurve(tgt, **kw)
            if q is not None and len(q) > 0:
                sr = q
                break
        except Exception:
            pass
    if sr is None:
        return [], {"status": "no products", "cadences": [], "windows": []}

    out, cads, wins, peak = [], [], [], 0
    n = len(sr) if max_sectors is None else min(len(sr), max_sectors)
    for i in range(n):
        os.makedirs(CACHE, exist_ok=True)
        try:
            lc = sr[i].download(download_dir=CACHE)
            if lc is None:
                continue
            t = np.asarray(lc.time.value, float)
            f = np.asarray(lc.flux.value, float)
            ok = np.isfinite(t) & np.isfinite(f)
            t, f = t[ok], f[ok]
            # BREAK-RISK 4, found by running real data rather than predicted.
            # MAST returns products for ONE star in TWO time systems: most in
            # BTJD (BJD-2457000, values ~1900-3000) but some in full BJD
            # (~2458900). TIC_373729723 has both. Concatenating them places
            # sectors 2.45 MILLION days apart, so folding at a 2-day period
            # produces meaningless phases -- silently, with no error.
            if len(t) and np.median(t) > 2.4e6:
                t = t - 2457000.0
            if len(t) < MIN_PTS_SECTOR:
                continue
            o = np.argsort(t); t, f = t[o], f[o]
            cad = float(np.median(np.diff(t)))
            if not (cad > 0):
                continue
            fl, win = flatten_one_sector(t, f, cad)
            cads.append(round(cad * 86400, 1)); wins.append(win)
            if fl is None:
                continue
            g = np.isfinite(fl)
            t, fl = t[g], fl[g]
            med = np.median(fl)
            sd = 1.4826 * np.median(np.abs(fl - med))
            if sd > 0:
                k = np.abs(fl - med) < SIGMA_CLIP * sd
                t, fl = t[k], fl[k]
            if len(t) >= MIN_PTS_SECTOR:
                out.append((t, fl))
        except Exception:
            pass
        finally:
            # break-risk 3: peak footprint is ONE sector
            if os.path.isdir(CACHE):
                sz = sum(os.path.getsize(os.path.join(dp, fn))
                         for dp, _, fns in os.walk(CACHE) for fn in fns)
                peak = max(peak, sz)
                shutil.rmtree(CACHE, ignore_errors=True)
    return out, {"status": "ok" if out else "no usable sectors",
                 "cadences": cads, "windows": wins, "peak_bytes": peak}


def run(hosts, df, label="stage A"):
    rows = []
    t_start = time.time()
    for i, h in enumerate(hosts, 1):
        r = df[df["host"] == h].iloc[0]
        per, t0, dur = float(r["period"]), float(r["T0"]), float(r["duration"])
        secs, diag = load_sectors(h)
        if not secs:
            rows.append({"host": h, "label": r["label"], "n_sectors": 0,
                         "status": diag["status"]})
            continue
        # SINGLE = first sector only; MULTI = all stacked. Same estimator both
        # sides, so the only thing that differs is how much photometry it sees.
        st, sf = secs[0]
        mt = np.concatenate([a for a, _ in secs])
        mf = np.concatenate([b for _, b in secs])
        o = np.argsort(mt); mt, mf = mt[o], mf[o]
        one = fold_features(st, sf, per, t0, dur)
        many = fold_features(mt, mf, per, t0, dur)
        row = {"host": h, "label": r["label"], "n_sectors": len(secs),
               "status": diag["status"], "cadences": diag["cadences"],
               "windows": diag["windows"], "peak_bytes": diag.get("peak_bytes", 0),
               "span_days": float(mt.max() - mt.min())}
        for k, v in one.items():
            row["single_" + k] = v
        for k, v in many.items():
            row["multi_" + k] = v
        rows.append(row)
        if i % 5 == 0 or i == len(hosts):
            print(f"  {label}: {i}/{len(hosts)}  ({(time.time()-t_start)/60:.1f} min)",
                  flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "multisector_cheap_stageA.csv"))
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    df = pd.read_csv(TRAINING)
    for c in ("period", "T0", "duration"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    good = df.dropna(subset=["period", "T0", "duration"])
    rng = np.random.RandomState(a.seed)
    hosts = []
    for lab, k in ((1.0, a.n // 2), (0.0, a.n - a.n // 2)):
        h = good.loc[good["label"] == lab, "host"].tolist()
        rng.shuffle(h)
        hosts += h[:k]
    print(f"STAGE A: {len(hosts)} stars ({a.n//2} planet / {a.n - a.n//2} FP), "
          f"fold at stored ephemeris, no TLS\n")
    out = run(hosts, good)
    out.to_csv(a.out, index=False)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
