"""ls_secondary_peak_features.py -- PART 2 of the periodogram secondary-peak
assessment.

The deployed `var_ls_power` / `var_ls_period` take `argmax` of ONE periodogram:
astropy Lomb-Scargle over 0.2-13 d on RAW, OUT-OF-TRANSIT-MASKED,
5-sigma-clipped, 10-minute-binned flux. Only the single strongest peak survives.

Two axes are genuinely untested, and this script separates them rather than
confounding them:

  AXIS A -- the INPUT.  Run the identical periodogram on FULL flux, transit
            points INCLUDED.  Hypothesis: the transit itself becomes a spurious
            "periodicity". Lomb-Scargle fits a SINUSOID, and a box of duty cycle
            ~1-5% is a poor sinusoid, so the leaked power should be small and
            should largely restate `period` / `depth` / `snr`.

  AXIS B -- the PEAK RANK.  Take the SECOND-highest peak, not just the first.
            Hypothesis: a blended background eclipsing binary contributes its
            OWN periodicity, so a strong second peak (low primary/secondary
            power ratio) marks a blend -> should be ENRICHED IN NEGATIVES.
            The competing hypothesis is that peak 2 is just the noise floor,
            in which case p2 restates the periodogram's normalisation and
            p1/p2 restates `var_ls_power`.

Both axes are measured on the SAME cleaned series so the comparison against the
deployed features is like for like. Everything before the periodogram is
`variability_features.variability_for_raw`, step for step.

Peak extraction: local maxima of the power array, with the primary's
neighbourhood excluded by +/- PEAK_SEP_WIDTHS independent peak widths (1/span
in frequency) so a sample on the flank of the primary cannot be returned as the
"secondary". A harmonic-excluded variant additionally drops frequencies near
n*f1 and f1/n for n in {2,3}, because a harmonic of the primary is the SAME
physical signal, not an independent second one.

Reads raw light curves only. Writes one CSV. Touches nothing in production.
"""
import os
import sys
import warnings
import importlib.util
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CAT = os.path.join(ROOT, "data", "catalogs")
OUT = os.path.join(HERE, "ls_secondary_peak_features.csv")

TRAIN_RAW_DIRS = [os.path.join(ROOT, "data", "known_lightcurves"),
                  os.path.join(ROOT, "data", "known_lightcurves_negative"),
                  os.path.join(ROOT, "data", "retrain_pipeline", "raw")]
POOLS = [("main", "unknown_features.csv",
          os.path.join(ROOT, "data", "unknown_lightcurves")),
         ("widesector", "unknown_features_widesector.csv",
          os.path.join(ROOT, "data", "unknown_lightcurves_widesector"))]

# identical to variability_features.py
SIGMA = 5
MIN_POINTS = 200
MIN_P, MAX_P = 0.2, 13.0
BIN_MINUTES = 10.0

PEAK_SEP_WIDTHS = 3.0      # independent peak widths (1/span) to exclude
HARMONICS = (2.0, 3.0)     # also excluded in the *_nh variant

COLUMNS = ["fls_p1", "fls_p2", "fls_ratio", "fls_p2_nh", "fls_ratio_nh",
           "fls_period1", "fls_period2",
           "ols_p1", "ols_p2", "ols_ratio", "ols_p2_nh", "ols_ratio_nh",
           "ols_period2"]

_VF = None


def _vf():
    global _VF
    if _VF is None:
        spec = importlib.util.spec_from_file_location(
            "varfeat", os.path.join(HERE, "variability_features.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _VF = m
    return _VF


def _peaks(fr, pw, span):
    """(p1, f1, p2, f2, p2_nh, f2_nh) from one power spectrum."""
    nan6 = (np.nan,) * 6
    if len(pw) < 5:
        return nan6
    i1 = int(np.argmax(pw))
    p1, f1 = float(pw[i1]), float(fr[i1])

    # local maxima only
    loc = np.zeros(len(pw), bool)
    loc[1:-1] = (pw[1:-1] > pw[:-2]) & (pw[1:-1] > pw[2:])
    # exclude the primary's own neighbourhood
    df = PEAK_SEP_WIDTHS / span if span > 0 else 0.0
    loc &= np.abs(fr - f1) > df
    if not loc.any():
        return p1, f1, np.nan, np.nan, np.nan, np.nan
    idx = np.where(loc)[0]
    j = idx[int(np.argmax(pw[idx]))]
    p2, f2 = float(pw[j]), float(fr[j])

    # harmonic-excluded secondary: an integer multiple or divisor of f1 is the
    # SAME signal reappearing, not an independent one
    nh = loc.copy()
    for n in HARMONICS:
        for fh in (n * f1, f1 / n):
            nh &= np.abs(fr - fh) > df
    if nh.any():
        k = np.where(nh)[0]
        m = k[int(np.argmax(pw[k]))]
        p2n, f2n = float(pw[m]), float(fr[m])
    else:
        p2n, f2n = np.nan, np.nan
    return p1, f1, p2, f2, p2n, f2n


def _periodogram(tb, fb):
    span = tb.max() - tb.min() if len(tb) > 2 else 0.0
    if len(tb) <= 50 or span <= 2 * MIN_P:
        return None, None, span
    from astropy.timeseries import LombScargle
    fr, pw = LombScargle(tb, fb).autopower(
        minimum_frequency=1.0 / min(MAX_P, span / 2.0),
        maximum_frequency=1.0 / MIN_P,
        normalization="standard", samples_per_peak=5)
    return fr, pw, span


def secondary_for_raw(raw_path, period=np.nan, t0=np.nan, duration=np.nan):
    out = {c: np.nan for c in COLUMNS}
    out["fls_status"] = "ok"
    if not raw_path or not os.path.exists(raw_path):
        out["fls_status"] = "no raw light curve"
        return out
    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        out["fls_status"] = f"read error: {type(e).__name__}"
        return out
    try:
        PRE = _vf()._pre()
        if PRE.validate_schema(df) is not None:
            out["fls_status"] = "non-standard schema"
            return out
        flux, ferr, _ = PRE.choose_flux_columns(df)
        if flux is None:
            out["fls_status"] = "no usable flux"
            return out
        t = df["time"].to_numpy(); q = df["quality"].to_numpy()
        ok = ~np.isnan(t) & ~np.isnan(flux) & ~np.isnan(ferr)
        t, flux, ferr, q = t[ok], flux[ok], ferr[ok], q[ok]
        g = q == 0
        t, flux, ferr = t[g], flux[g], ferr[g]
        if len(t) < MIN_POINTS:
            out["fls_status"] = f"only {len(t)} points"
            return out
        o = np.argsort(t, kind="stable")
        t, flux, ferr = t[o], flux[o], ferr[o]
        from astropy.stats import sigma_clip
        m = sigma_clip(flux, sigma=SIGMA, stdfunc="mad_std", maxiters=5,
                       masked=True).mask
        t, flux = t[~m], flux[~m]
        med = np.median(flux)
        if not np.isfinite(med) or med == 0:
            out["fls_status"] = "degenerate median flux"
            return out
        f = flux / med

        bin_ = _vf()._acf_bin

        # ---- AXIS A: FULL flux, transit points INCLUDED ----
        tb, fb = bin_(t, f, BIN_MINUTES)
        fr, pw, span = _periodogram(tb, fb)
        if fr is not None:
            p1, f1, p2, f2, p2n, _ = _peaks(fr, pw, span)
            out["fls_p1"] = p1
            out["fls_p2"] = p2
            out["fls_p2_nh"] = p2n
            out["fls_period1"] = 1.0 / f1 if f1 and np.isfinite(f1) else np.nan
            out["fls_period2"] = 1.0 / f2 if f2 and np.isfinite(f2) else np.nan
            out["fls_ratio"] = p1 / p2 if p2 and p2 > 0 else np.nan
            out["fls_ratio_nh"] = p1 / p2n if p2n and p2n > 0 else np.nan
        else:
            out["fls_status"] = "full: too few binned points"

        # ---- AXIS B alone: OOT-masked, i.e. the DEPLOYED input, second peak ----
        if all(np.isfinite([period, t0, duration])) and period > 0 and duration > 0:
            ph = ((t - t0 + 0.5 * period) % period) / period - 0.5
            oot = np.abs(ph) > (duration / period)
        else:
            oot = np.ones(len(t), bool)
        if oot.sum() >= MIN_POINTS:
            tb2, fb2 = bin_(t[oot], f[oot], BIN_MINUTES)
            fr2, pw2, span2 = _periodogram(tb2, fb2)
            if fr2 is not None:
                q1, g1, q2, g2, q2n, _ = _peaks(fr2, pw2, span2)
                out["ols_p1"] = q1
                out["ols_p2"] = q2
                out["ols_p2_nh"] = q2n
                out["ols_period2"] = 1.0 / g2 if g2 and np.isfinite(g2) else np.nan
                out["ols_ratio"] = q1 / q2 if q2 and q2 > 0 else np.nan
                out["ols_ratio_nh"] = q1 / q2n if q2n and q2n > 0 else np.nan
    except Exception as e:
        out["fls_status"] = f"error: {type(e).__name__}: {e}"
    return out


def worker(job):
    host, path, per, t0, dur = job
    r = secondary_for_raw(path, per, t0, dur)
    r["host"] = host
    return r


def _find(host, dirs):
    for d in dirs:
        p = os.path.join(d, str(host) + ".csv")
        if os.path.exists(p):
            return p
    return None


def _jobs(df, dirs):
    out = []
    for _, r in df.iterrows():
        out.append((str(r["host"]), _find(r["host"], dirs),
                    pd.to_numeric(r.get("period"), errors="coerce"),
                    pd.to_numeric(r.get("T0", np.nan), errors="coerce"),
                    pd.to_numeric(r.get("duration"), errors="coerce")))
    return out


def main():
    frames = []
    tr = pd.read_csv(TRAINING); tr["host"] = tr.host.astype(str)
    tr["_set"] = "train"
    jobs = [("train", j) for j in _jobs(tr, TRAIN_RAW_DIRS)]
    for tag, ff, d in POOLS:
        p = pd.read_csv(os.path.join(CAT, ff)); p["host"] = p.host.astype(str)
        if "status" in p.columns:
            p = p[p.status.astype(str).str.startswith("Success")]
        jobs += [(tag, j) for j in _jobs(p, [d])]
    print(f"{len(jobs)} light curves", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for n, (setname, r) in enumerate(
                zip([s for s, _ in jobs], ex.map(worker, [j for _, j in jobs],
                                                 chunksize=8)), 1):
            r["_set"] = setname
            rows.append(r)
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"saved {OUT}  {len(out)} rows")
    print(out.groupby("_set").fls_status.value_counts())


if __name__ == "__main__":
    main()
