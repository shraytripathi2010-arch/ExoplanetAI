"""detrend_gp_pilot.py -- does GP detrending beat production's Savitzky-Golay
for TRANSIT DETECTION?

WHY THE EXISTING INJECTION HARNESS COULD NOT BE REUSED AS-IS
------------------------------------------------------------
`injection_recovery_sensitivity.py` injects into ALREADY-FLATTENED curves from
`data/processed_negative/`. Detrending has already happened there, so it cannot
compare detrenders. This harness injects into the **RAW** light curve and then
detrends the SAME injected series three different ways, so the detrender is the
only thing that varies. Everything downstream (binning, TLS bounds, recovery
tolerance, alias convention) is production's / the existing harness's.

PAIRED BY CONSTRUCTION
----------------------
One star, one injection, one noise realisation -> three detrended copies. A
difference between arms cannot be a different star or a different draw.

THE THREE ARMS
--------------
  savgol      production exactly: 5-sigma MAD clip, savgol(window=min(401,n-1),
              polyorder=2, mode="interp"), divide, renormalise to median 1.
  gp_protect  celerite2 SHOTerm + jitter, undamped period rho floored at
              GP_RHO_FLOOR_PROTECT days. The floor is the GP's analogue of
              savgol's protected timescale: it stops the GP chasing anything
              faster than that, which is what would eat a transit.
  gp_tight    identical but floored at GP_RHO_FLOOR_TIGHT -- deliberately more
              aggressive. This arm exists to answer the question directly:
              does TIGHTER detrending recover more transits, the same, or does
              it start deleting real signal?

WHAT THE INJECTION IS AND IS NOT
--------------------------------
Injected transits are cleaner than real ones (no TTVs, no spot crossings) --
the same caveat the existing sensitivity work records. That biases ALL THREE
ARMS EQUALLY, so the arm-to-arm comparison stands even though the absolute
recovery rates are best-case.

ARCHITECTURAL NOTE -- the variability features are NOT in this path.
`add_variability_features` reads RAW_FOLDER (the pre-flatten downloads);
`compute_all_features` reads PROCESSED_FOLDER. This script writes NOTHING to
either. Nothing in production changes. See `detrend_variability_isolation.py`
for the empirical confirmation of that separation.
"""
import os
import sys
import json
import time
import warnings
import importlib.util
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..")
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)

RAW_NEG = os.path.join(ROOT, "data", "known_lightcurves_negative")
OUT_CSV = os.path.join(HERE, "detrend_gp_pilot_results.csv")
OUT_JSON = os.path.join(HERE, "detrend_gp_pilot_summary.json")

GRID_DEPTHS_PPM = [84, 250, 700, 2500]     # 84 = Earth-size, the documented floor
GRID_PERIODS = [2.0, 6.0, 10.0]            # all under the ~12.5 d single-sector ceiling
N_REPEATS = 5
PERIOD_MATCH_TOLERANCE = 0.01

GP_RHO_FLOOR_PROTECT = 0.5     # days ~ 12 h, comparable to savgol's 13.4 h at 2-min
GP_RHO_FLOOR_TIGHT = 0.10      # days ~ 2.4 h, deliberately aggressive
GP_MAX_POINTS = 20000

_G = {}


def _load(name, fname, d=CODE):
    spec = importlib.util.spec_from_file_location(name, os.path.join(d, fname))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    if _G:
        return _G
    import injection as inj
    _G["m02"] = _load("m02", "02_preprocess.py")
    _G["m06"] = _load("m06", "06_download_unknown.py")
    _G["inj"] = inj
    return _G


# ---------------------------------------------------------------- shared preamble
def raw_preamble(path, m02):
    """Production steps 1-4 on the RAW file: NaN drop, quality filter, sort.
    The 5-sigma clip is deliberately NOT here -- it runs AFTER injection, which
    is what happens to a real transit in production too."""
    df = pd.read_csv(path, low_memory=False)
    flux, err, src = m02.choose_flux_columns(df)
    if flux is None:
        return None
    t = df["time"].to_numpy()
    q = df["quality"].to_numpy() if "quality" in df.columns else np.zeros(len(t))
    ok = ~np.isnan(t) & ~np.isnan(flux) & ~np.isnan(err)
    t, flux, err, q = t[ok], flux[ok], err[ok], q[ok]
    g = q == 0
    t, flux, err = t[g], flux[g], err[g]
    o = np.argsort(t, kind="stable")
    return t[o], flux[o], err[o], src


def clip(t, f, e, m02):
    from astropy.stats import sigma_clip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cr = sigma_clip(f, sigma=m02.SIGMA_CLIP_THRESHOLD, stdfunc="mad_std",
                        maxiters=5, masked=True)
    k = ~cr.mask
    return t[k], f[k], e[k]


# ---------------------------------------------------------------- the three arms
def detrend_savgol(t, f, e, m02):
    """Production, exactly."""
    from scipy.signal import savgol_filter
    w = m02.choose_savgol_window(len(f))
    if w is None:
        return None
    trend = savgol_filter(f, window_length=w, polyorder=m02.SAVGOL_POLYORDER, mode="interp")
    ff, ee = f / trend, e / trend
    if not np.all(np.isfinite(ff)):
        return None
    m = np.median(ff)
    return t, ff / m, ee / m, {"window_pts": int(w),
                               "window_hours": float(w * np.median(np.diff(t)) * 24)}


def detrend_gp(t, f, e, rho_floor):
    """celerite2 SHOTerm + jitter, ML hyperparameters, trend = GP mean prediction.

    `rho_floor` is the protected timescale: the optimiser cannot drive the
    undamped period below it, so the GP physically cannot model structure
    faster than that -- the direct analogue of savgol's window width. Without
    such a floor a GP fitted to ALL the data (which is the only option before
    detection, since transit locations are unknown) will happily absorb the
    transit itself.
    """
    import celerite2
    from celerite2 import terms
    from scipy.optimize import minimize

    n = len(f)
    step = max(1, n // GP_MAX_POINTS)
    ts, fs, es = t[::step], f[::step], e[::step]
    mu = np.median(fs)
    y = fs / mu - 1.0
    yerr = es / mu
    sd = float(np.std(y)) or 1e-4

    def build(p):
        log_sig, log_rho, log_jit = p
        rho = rho_floor + np.exp(log_rho)          # floor enforced by construction
        k = terms.SHOTerm(sigma=np.exp(log_sig), rho=rho, Q=1.0 / np.sqrt(2))
        gp = celerite2.GaussianProcess(k, mean=0.0)
        gp.compute(ts, diag=yerr ** 2 + np.exp(2 * log_jit), quiet=True)
        return gp

    def nll(p):
        try:
            return -build(p).log_likelihood(y)
        except Exception:
            return 1e10

    p0 = [np.log(sd), np.log(1.0), np.log(sd / 10 + 1e-8)]
    try:
        r = minimize(nll, p0, method="L-BFGS-B",
                     bounds=[(np.log(sd) - 6, np.log(sd) + 4),
                             (np.log(1e-3), np.log(30.0)),
                             (np.log(1e-7), np.log(sd + 1e-6))])
        gp = build(r.x)
        trend_s = gp.predict(y, t=t, return_var=False)    # predict at FULL cadence
        rho_fit = rho_floor + np.exp(r.x[1])
    except Exception as ex:
        return None
    trend = (1.0 + trend_s) * mu
    if not np.all(np.isfinite(trend)) or np.any(trend <= 0):
        return None
    ff, ee = f / trend, e / trend
    if not np.all(np.isfinite(ff)):
        return None
    m = np.median(ff)
    return t, ff / m, ee / m, {"rho_days": float(rho_fit),
                               "rho_floor": float(rho_floor),
                               "n_fit": int(len(ts))}


# ---------------------------------------------------------------- TLS
def run_tls(t, f, e, r_star, m_star, m06):
    from transitleastsquares import transitleastsquares
    tb, fb, eb = m06.bin_lightcurve(t, f, e)
    model = transitleastsquares(tb, fb, eb)
    r = model.power(use_threads=1, oversampling_factor=1, duration_grid_step=1.1,
                    show_progress_bar=False, R_star=r_star,
                    R_star_min=min(0.13, r_star * 0.5),
                    R_star_max=max(3.5, r_star * 1.5),
                    M_star=m_star, M_star_min=max(0.1, m_star * 0.5),
                    M_star_max=max(1.0, m_star * 1.5))
    if m06.tls_result_is_degenerate(r):
        return {"period": np.nan, "SDE": np.nan, "snr": np.nan, "depth": np.nan}
    return {"period": float(r.period), "SDE": float(r.SDE),
            "snr": float(r.snr) if np.isfinite(r.snr) else np.nan,
            "depth": float(r.depth)}


def recovered(inj_p, rec_p, tol=PERIOD_MATCH_TOLERANCE):
    if rec_p is None or not np.isfinite(rec_p) or rec_p <= 0:
        return False, None
    for n, nm in [(1, "exact"), (2, "double"), (0.5, "half"), (3, "triple"), (1 / 3, "third")]:
        if abs(rec_p - inj_p * n) / (inj_p * n) < tol:
            return True, nm
    return False, None


def run_one(args):
    host, r_star, m_star, period, depth, rep, seed = args
    G = _init() if not _G else _G
    m02, m06, inj = G["m02"], G["m06"], G["inj"]
    base = {"host": host, "period": period, "depth_ppm": depth, "repeat": rep}
    try:
        pre = raw_preamble(os.path.join(RAW_NEG, host + ".csv"), m02)
        if pre is None:
            return {**base, "status": "no usable flux column"}
        t, f, e, src = pre
        if len(t) < 500:
            return {**base, "status": f"only {len(t)} raw points"}
        base["flux_source"] = src
        base["n_raw"] = len(t)
        base["baseline_d"] = float(t.max() - t.min())

        import injection_recovery_sensitivity as IRS
        dur = IRS.transit_duration_days(period, r_star, m_star, 0.3)
        base["duration_d"] = dur

        rng = np.random.default_rng(seed)
        f_inj, params = inj.inject_transit(t, f, period, depth, dur, rng)
        base["t0"] = float(params.get("t0", np.nan))

        tc, fc, ec = clip(t, f_inj, e, m02)
        base["n_after_clip"] = len(tc)

        arms = {
            "savgol": detrend_savgol(tc, fc, ec, m02),
            "gp_protect": detrend_gp(tc, fc, ec, GP_RHO_FLOOR_PROTECT),
            "gp_tight": detrend_gp(tc, fc, ec, GP_RHO_FLOOR_TIGHT),
        }
        for name, out in arms.items():
            if out is None:
                base[f"{name}_status"] = "detrend failed"
                continue
            ta, fa, ea, meta = out
            t_det = time.time()
            res = run_tls(ta, fa, ea, r_star, m_star, m06)
            ok, alias = recovered(period, res["period"])
            base[f"{name}_status"] = "ok"
            base[f"{name}_recovered"] = bool(ok)
            base[f"{name}_alias"] = alias
            base[f"{name}_SDE"] = res["SDE"]
            base[f"{name}_snr"] = res["snr"]
            base[f"{name}_period"] = res["period"]
            base[f"{name}_scatter"] = float(1.4826 * np.median(np.abs(fa - np.median(fa))))
            base[f"{name}_tls_s"] = round(time.time() - t_det, 1)
            for k, v in meta.items():
                base[f"{name}_{k}"] = v
        base["status"] = "ok"
    except Exception as ex:
        base["status"] = f"error: {type(ex).__name__}: {ex}"
    return base


def build_jobs():
    G = _init()
    split = json.load(open(os.path.join(ROOT, "data", "training_dataset", "split_manifest.json")))
    test_hosts = set(map(str, split["test_hosts"]))
    df = pd.read_csv(os.path.join(ROOT, "data", "training_dataset", "training.csv"))
    df["host"] = df.host.astype(str)
    neg = df[(df.label == 0) & df.host.isin(test_hosts)]
    neg = neg[neg.st_rad.notna() & (neg.st_rad > 0)]
    avail = {f[:-4] for f in os.listdir(RAW_NEG) if f.endswith(".csv")}
    neg = neg[neg.host.isin(avail)].reset_index(drop=True)
    print(f"host pool: {len(neg)} real negative TEST-split stars with a RAW file")
    rng = np.random.default_rng(20260813)
    jobs, seed = [], 771000
    for p in GRID_PERIODS:
        for d in GRID_DEPTHS_PPM:
            for r in range(N_REPEATS):
                row = neg.iloc[int(rng.integers(0, len(neg)))]
                ms = float(row.st_mass) if np.isfinite(row.get("st_mass", np.nan)) and row.get("st_mass", 0) > 0 else 1.0
                jobs.append((str(row.host), float(row.st_rad), ms, p, d, r, seed))
                seed += 1
    return jobs


def main():
    jobs = build_jobs()
    print(f"{len(jobs)} paired trials x 3 detrending arms = {3*len(jobs)} TLS searches",
          flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=7, initializer=_init) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            res.append(fu.result())
            if i % 5 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  [{i}/{len(jobs)}] {el/60:.1f} min, eta {el/i*(len(jobs)-i)/60:.1f} min",
                      flush=True)
    pd.DataFrame(res).to_csv(OUT_CSV, index=False)
    print(f"\nwall {(time.time()-t0)/60:.1f} min -> {OUT_CSV}")
    report()


def report():
    # McNemar's EXACT test is a two-sided binomial test on the discordant
    # pairs, so scipy alone is enough -- statsmodels is not installed in this
    # environment and importing it here would crash the report after the run.
    from scipy.stats import wilcoxon, binomtest

    def mcnemar_exact(b, c):
        return binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else float("nan")

    d = pd.read_csv(OUT_CSV)
    ok = d[d.status == "ok"].copy()
    arms = ["savgol", "gp_protect", "gp_tight"]
    ok = ok[[f"{a}_status" for a in arms]].eq("ok").all(axis=1).pipe(lambda m: ok[m])
    print(f"\ntrials with all three arms usable: {len(ok)}/{len(d)}")
    if not len(ok):
        return

    print("\n=== GP hyperparameters actually fitted ===")
    for a in ("gp_protect", "gp_tight"):
        print(f"  {a:<11} rho median {ok[f'{a}_rho_days'].median():.3f} d "
              f"(floor {ok[f'{a}_rho_floor'].iloc[0]:.2f})")
    print(f"  savgol window median {ok.savgol_window_hours.median():.2f} h "
          f"({ok.savgol_window_pts.median():.0f} pts)")

    print("\n=== RECOVERY RATE (the headline) ===")
    print(f"{'arm':<12}{'recovered':>12}{'rate':>9}")
    for a in arms:
        n = int(ok[f"{a}_recovered"].sum())
        print(f"{a:<12}{n:>12}{n/len(ok):>9.1%}")

    print("\n=== paired McNemar vs savgol (does the arm change WHICH are found?) ===")
    for a in ("gp_protect", "gp_tight"):
        b = int((ok.savgol_recovered & ~ok[f"{a}_recovered"]).sum())   # savgol only
        c = int((~ok.savgol_recovered & ok[f"{a}_recovered"]).sum())   # arm only
        p = mcnemar_exact(b, c)
        print(f"  {a:<11} savgol-only {b}, {a}-only {c}, McNemar exact p = {p:.4f}")

    print("\n=== by injected depth: recovery rate per arm ===")
    g = ok.groupby("depth_ppm")[[f"{a}_recovered" for a in arms]].mean()
    g.columns = arms
    print((g * 100).round(1).to_string())
    print("\n=== by injected period ===")
    g2 = ok.groupby("period")[[f"{a}_recovered" for a in arms]].mean()
    g2.columns = arms
    print((g2 * 100).round(1).to_string())

    print("\n=== SDE on RECOVERED trials (paired, savgol vs each arm) ===")
    for a in ("gp_protect", "gp_tight"):
        m = ok.savgol_recovered & ok[f"{a}_recovered"]
        if m.sum() > 5:
            x, y = ok.savgol_SDE[m], ok[f"{a}_SDE"][m]
            try:
                p = wilcoxon(x, y).pvalue
            except Exception:
                p = float("nan")
            print(f"  {a:<11} n={int(m.sum())}  savgol {x.median():.2f} vs "
                  f"{y.median():.2f}  delta {(y-x).median():+.2f}  Wilcoxon p={p:.4f}")

    print("\n=== residual scatter after detrending (lower = tighter) ===")
    for a in arms:
        print(f"  {a:<11} median {ok[f'{a}_scatter'].median():.6f}")
    print("\n=== cost per star (TLS only; detrending is negligible beside it) ===")
    for a in arms:
        print(f"  {a:<11} median TLS {ok[f'{a}_tls_s'].median():.1f} s")

    out = {"n_trials": int(len(ok)),
           "recovery": {a: float(ok[f"{a}_recovered"].mean()) for a in arms},
           "scatter": {a: float(ok[f"{a}_scatter"].median()) for a in arms},
           "sde_median": {a: float(ok[f"{a}_SDE"].median()) for a in arms}}
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved {OUT_JSON}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
