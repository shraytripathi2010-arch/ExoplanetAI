"""
secondary_window_accuracy.py -- PART 1: does a duration-aware window actually
recover the secondary eclipse depth more accurately than the deployed fixed
0.45-0.55 window?

CONTEXT: THIS FORMULA ALREADY EXISTS
-------------------------------------
`weak_secondary.py` (2026-08-03) already implements the duration-aware window
as `sec_depth_windowed`, with exactly the formula proposed here:

    dur_phase = duration / period
    half      = max(0.5 * dur_phase, 0.002)
    window    = |phase - 0.5| < half

and already reported single-feature AUC 0.435, max |corr| 0.478 (against
`secondary_eclipse_depth`), coverage 97.8%, and no clearing arm when ADDED
alongside the old feature (24 -> 26 features).

What was NEVER measured, for either formula, is **depth-recovery ACCURACY
against a known ground truth**. That is what this file does, and it is the
prerequisite for deciding whether a replacement is worth validating.

METHOD -- ground truth by construction
---------------------------------------
For each trial:
  1. Inject an EB (`inject_eclipsing_binary`: grazing primary + half-depth
     secondary at phase 0.5) into a REAL light curve.
  2. Compute the TRUE secondary depth from the NOISELESS injected model --
     not from the nominal 0.36 x primary ratio, which ignores limb darkening
     and grazing geometry. This is the ground truth both formulas are scored
     against.
  3. Fold at the TRUE period (TLS with its grid clamped to +/-0.5% of it, the
     same forced-fold device used in the half-period diagnostic).
  4. From that SINGLE fold, compute BOTH estimators on the identical arrays:
       OLD  median over 0.45 < phase < 0.55          (deployed)
       NEW  median over |phase - 0.5| < half         (duration-aware)
     so the only difference is the window, not the fold, star, or noise draw.

Reporting recovered/injected ratio per formula per injected depth answers the
question directly: 1.0 is perfect, the deployed formula measured ~0.2-0.3 in
the half-period diagnostic.

Read-only. No training data, model, or pipeline changes.
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import injection_recovery_sensitivity as base

OUT_CSV = os.path.join(HERE, "secondary_window_accuracy_results.csv")
OUT_JSON = os.path.join(HERE, "secondary_window_accuracy_summary.json")

GRID_PERIODS = [2.0, 5.0, 10.0]
GRID_DEPTHS_PPM = [1000, 2500, 5000, 10000]
N_REPEATS = 12

_G = {}


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    _G.update(base._init())
    return _G


def old_window(phase, flux):
    """The DEPLOYED formula: fixed slab, 10% of the orbit."""
    m = (phase > 0.45) & (phase < 0.55)
    return float(1.0 - np.median(flux[m])) if m.sum() > 5 else np.nan


def new_window(phase, flux, duration, period):
    """Duration-aware, matching weak_secondary.py exactly."""
    dur_phase = (duration / period) if (np.isfinite(duration) and duration > 0
                                        and duration < period) else 0.01
    half = max(0.5 * dur_phase, 0.002)
    m = np.abs(phase - 0.5) < half
    return (float(1.0 - np.median(flux[m])) if m.sum() > 5 else np.nan,
            float(half), int(m.sum()))


def true_secondary_depth(t, period, depth_ppm, duration, r_star, seed):
    """Ground truth: build the SAME EB model on a NOISELESS unit-flux array and
    measure its actual secondary depth. Captures limb darkening and grazing
    geometry, which the nominal 0.36 x primary ratio does not."""
    import injection as inj
    rng = np.random.default_rng(seed)
    ones = np.ones_like(t)
    model, params = inj.inject_eclipsing_binary(t, ones, period, depth_ppm, duration, rng)
    ph = ((t - params["t0"]) / period) % 1.0
    sec = np.abs(ph - 0.5) < 0.02
    return (float(1.0 - np.min(model[sec])) if sec.sum() > 3 else np.nan,
            params["t0"], rng)


def run_one(args):
    period, depth_ppm, rep, seed = args
    G = _G if _G else _init()
    rng = np.random.default_rng(seed)
    pool, inj, m06 = G["pool"], G["inj"], G["m06"]
    row = pool.iloc[int(rng.integers(0, len(pool)))]
    host = str(row["host"])
    r_star = float(row["st_rad"]) if np.isfinite(row["st_rad"]) and row["st_rad"] > 0 else 1.0
    t, f, e = base._load_curve(G, host)

    out = {"host": host, "injected_period": period, "injected_depth_ppm": depth_ppm,
           "repeat": rep}
    try:
        dur = base.transit_duration_days(period, r_star, 1.0, 0.9)
        out["injected_duration_days"] = dur
        out["duty_cycle_pct"] = 100.0 * dur / period

        # ---- ground truth from the noiseless model (same seed -> same t0) ----
        truth, _, _ = true_secondary_depth(t, period, depth_ppm, dur, r_star, seed)
        out["true_secondary_depth"] = truth

        # ---- inject into the REAL curve with that same seed ----
        rng2 = np.random.default_rng(seed)
        flux, params = inj.inject_eclipsing_binary(t, f, period, depth_ppm, dur, rng2)

        # ---- forced fold at the TRUE period ----
        from transitleastsquares import transitleastsquares
        tb, fb, eb = m06.bin_lightcurve(t, flux, e)
        model = transitleastsquares(tb, fb, eb)
        r = model.power(use_threads=1, oversampling_factor=1, duration_grid_step=1.1,
                        show_progress_bar=False,
                        period_min=period * 0.995, period_max=period * 1.005,
                        R_star=r_star, R_star_min=min(0.13, r_star * 0.5),
                        R_star_max=max(3.5, r_star * 1.5),
                        M_star=1.0, M_star_min=0.1, M_star_max=1.0)
        phase = np.atleast_1d(np.asarray(r.folded_phase, dtype=float))
        fold = np.atleast_1d(np.asarray(r.folded_y, dtype=float))
        out["fold_period"] = float(r.period)
        # TLS's own fitted duration, which is what production would have
        fit_dur = float(r.duration)
        out["fitted_duration_days"] = fit_dur

        out["old_depth"] = old_window(phase, fold)
        # Production-relevant: window sized from TLS's FITTED duration, which is
        # all production has.
        nd, half, nin = new_window(phase, fold, fit_dur, float(r.period))
        out["new_depth"] = nd; out["new_half_phase"] = half; out["new_n_in_window"] = nin
        # Diagnostic only: window sized from the TRUE injected duration. Not
        # available in production -- included to separate "the duration-aware
        # window is the wrong idea" from "TLS's fitted duration is a poor width
        # estimate for grazing eclipses". If this arm is accurate and the fitted
        # arm is not, the window concept is fine and the duration input is the
        # problem.
        nd2, half2, nin2 = new_window(phase, fold, dur, period)
        out["new_truedur_depth"] = nd2; out["new_truedur_half_phase"] = half2
        out["new_truedur_n_in_window"] = nin2
        out["fitted_over_true_duration"] = fit_dur / dur if dur > 0 else np.nan
        out["old_n_in_window"] = int(((phase > 0.45) & (phase < 0.55)).sum())
        out["status"] = "ok"
    except Exception as ex:
        out.update({"status": f"error: {type(ex).__name__}: {ex}"})
    return out


def main():
    jobs, seed = [], 20260812
    for p in GRID_PERIODS:
        for d in GRID_DEPTHS_PPM:
            for r in range(N_REPEATS):
                jobs.append((p, d, r, seed)); seed += 1
    print(f"{len(jobs)} trials (EB injections, forced fold at true period)", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=7, initializer=_init) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            res.append(fu.result())
            if i % 15 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  [{i}/{len(jobs)}] {el/60:.1f} min, eta "
                      f"{el/i*(len(jobs)-i)/60:.1f} min", flush=True)
    pd.DataFrame(res).to_csv(OUT_CSV, index=False)
    print(f"\nwall {(time.time()-t0)/60:.1f} min -> {OUT_CSV}")
    report()


def report():
    from scipy.stats import wilcoxon
    d = pd.read_csv(OUT_CSV)
    ok = d[d.status == "ok"].copy()
    ok = ok[np.isfinite(ok.true_secondary_depth) & (ok.true_secondary_depth > 0)]
    print(f"\ntrials usable {len(ok)}/{len(d)}")
    ok["old_ratio"] = ok.old_depth / ok.true_secondary_depth
    ok["new_ratio"] = ok.new_depth / ok.true_secondary_depth

    print(f"\nduty cycle (eclipse as % of phase): median {ok.duty_cycle_pct.median():.2f}%"
          f"   -> old window is 10.00% of phase, new is "
          f"{200*ok.new_half_phase.median():.2f}%")
    print(f"points in window: old median {ok.old_n_in_window.median():.0f}, "
          f"new median {ok.new_n_in_window.median():.0f}")

    print("\n=== DEPTH-RECOVERY ACCURACY: recovered / injected (1.0 = perfect) ===")
    ok["new_truedur_ratio"] = ok.new_truedur_depth / ok.true_secondary_depth
    print(f"\nTLS fitted duration / TRUE duration: median "
          f"{ok.fitted_over_true_duration.median():.3f} "
          f"(1.0 = correct; <1 = TLS under-estimates eclipse width)")
    g = ok.groupby("injected_depth_ppm").agg(
        n=("old_ratio", "size"),
        true_ppm=("true_secondary_depth", lambda s: 1e6 * s.median()),
        OLD_ratio=("old_ratio", "median"), NEW_ratio=("new_ratio", "median"),
        NEW_truedur=("new_truedur_ratio", "median"))
    print(g.round(3).to_string())
    print("\n=== by period ===")
    g2 = ok.groupby("injected_period").agg(
        n=("old_ratio", "size"), duty_pct=("duty_cycle_pct", "median"),
        OLD_ratio=("old_ratio", "median"), NEW_ratio=("new_ratio", "median"))
    print(g2.round(3).to_string())

    m = np.isfinite(ok.old_ratio) & np.isfinite(ok.new_ratio)
    try:
        p = wilcoxon(ok.old_ratio[m], ok.new_ratio[m]).pvalue
    except Exception:
        p = float("nan")
    print(f"\n=== POOLED (n={int(m.sum())}) ===")
    print(f"  OLD median ratio {ok.old_ratio[m].median():.3f}   "
          f"|1 - ratio| {abs(1-ok.old_ratio[m].median()):.3f}")
    print(f"  NEW median ratio {ok.new_ratio[m].median():.3f}   "
          f"|1 - ratio| {abs(1-ok.new_ratio[m].median()):.3f}")
    print(f"  paired Wilcoxon p = {p:.3e}")
    better = int((abs(1 - ok.new_ratio[m]) < abs(1 - ok.old_ratio[m])).sum())
    print(f"  NEW closer to truth on {better}/{int(m.sum())} paired trials "
          f"({100*better/max(m.sum(),1):.1f}%)")

    json.dump({"n": int(m.sum()),
               "old_median_ratio": float(ok.old_ratio[m].median()),
               "new_median_ratio": float(ok.new_ratio[m].median()),
               "wilcoxon_p": float(p), "new_closer_frac": float(better/max(m.sum(),1)),
               "by_depth": {int(k): {"old": float(v.OLD_ratio), "new": float(v.NEW_ratio)}
                            for k, v in g.iterrows()}},
              open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved {OUT_JSON}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
