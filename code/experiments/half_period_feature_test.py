"""
half_period_feature_test.py -- does TLS's half-period alias on eclipsing
binaries degrade the two features designed to CATCH eclipsing binaries?

Follow-up to the BLS-vs-TLS assessment, which measured that on injected
V-shaped EBs at P = 10 d, TLS reports the HALF-period alias 12 of 16 times
(a V-shaped primary plus a secondary at phase 0.5 looks to TLS's physical
transit template like two identical transits at P/2). That raised, but did not
test, the worry that `secondary_eclipse_depth` and `odd_even_mismatch` -- both
computed on the fold at TLS's REPORTED period -- are compromised on exactly the
rows they exist to flag.

THE PREDICTION IS NOT "BOTH DEGRADE". Reasoned out before running:

  At the TRUE period P: primary at phase 0, secondary at phase 0.5.
      secondary_eclipse_depth  -> measures the secondary. LARGE.
      odd_even_mismatch        -> consecutive primaries, equal depth. SMALL.

  At the HALF period P/2: primary and secondary BOTH fold onto phase 0 and
  alternate; phase 0.5 is empty.
      secondary_eclipse_depth  -> nothing at phase 0.5. DESTROYED (-> ~0).
      odd_even_mismatch        -> alternating deep/shallow eclipses. AMPLIFIED.

So the expectation is that one EB signature is LOST and a different one is
GAINED, not that the row becomes uniformly harder to reject. Stated here before
the numbers exist so the reading cannot be fitted after the fact.

DESIGN -- controlled, with ground truth
---------------------------------------
Inject a known EB (`injection.inject_eclipsing_binary`: grazing high-impact
primary + half-depth secondary at phase 0.5) into a real light curve, then run
TLS TWICE on the identical injected array:

  ARM A  free search   -- production's own invocation. Finds whatever it finds
                          (exact, half, or something else).
  ARM B  forced fold   -- same call, but the period grid is clamped to
                          [0.995, 1.005] x the TRUE injected period, forcing
                          the correct fold. This is the ground-truth reference.

Both arms produce the full 22-feature TLS vector by the same code path, so any
difference is attributable to the fold period alone. Splitting Arm A by which
alias it landed on then isolates the effect of the half-period alias.

Read-only: no training data, model, or pipeline changes.
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

OUT_CSV = os.path.join(HERE, "half_period_feature_test_results.csv")
OUT_JSON = os.path.join(HERE, "half_period_feature_test_summary.json")

GRID_PERIODS = [2.0, 5.0, 10.0]     # where the half-alias was observed
GRID_DEPTHS_PPM = [2500, 5000]      # deep enough that the EB is clearly found
N_REPEATS = 15

WATCH = ["secondary_eclipse_depth", "odd_even_mismatch",
         "depth_mean_odd", "depth_mean_even", "depth_mean", "transit_shape_ratio",
         "SDE", "snr"]

_G = {}


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    _G.update(base._init())
    return _G


def _tls_forced(t, f, e, r_star, m_star, m06, true_period):
    """Identical to base._tls_features but with the period grid clamped around
    the TRUE period, forcing the correct fold. Ground-truth reference arm."""
    from transitleastsquares import transitleastsquares
    tb, fb, eb = m06.bin_lightcurve(t, f, e)
    model = transitleastsquares(tb, fb, eb)
    r = model.power(
        use_threads=1, oversampling_factor=1, duration_grid_step=1.1,
        show_progress_bar=False,
        period_min=true_period * 0.995, period_max=true_period * 1.005,
        R_star=r_star, R_star_min=min(0.13, r_star * 0.5), R_star_max=max(3.5, r_star * 1.5),
        M_star=m_star, M_star_min=min(0.1, m_star * 0.5), M_star_max=max(1.0, m_star * 1.5),
    )
    return _extract(r)


def _extract(r):
    phase = np.atleast_1d(np.asarray(r.folded_phase, dtype=float))
    flux = np.atleast_1d(np.asarray(r.folded_y, dtype=float))
    if phase.size < 10 or phase.size != flux.size:
        phase = np.array([np.nan]); flux = np.array([np.nan])
    sec = (phase > 0.45) & (phase < 0.55)
    secondary = float(1.0 - np.median(flux[sec])) if sec.sum() > 5 else np.nan
    pm = (phase < 0.02) | (phase > 0.98)
    shape = np.nan
    if pm.sum() > 5:
        itp = np.where(phase > 0.5, phase - 1, phase)[pm]
        cm = np.abs(itp) < 0.005
        em = (np.abs(itp) >= 0.005) & (np.abs(itp) < 0.015)
        cd = float(1.0 - np.median(flux[pm][cm])) if cm.sum() > 2 else np.nan
        ed = float(1.0 - np.median(flux[pm][em])) if em.sum() > 2 else np.nan
        shape = ed / cd if (cd and np.isfinite(cd) and cd > 0) else np.nan
    return {
        "period": float(r.period), "SDE": float(r.SDE), "snr": float(r.snr),
        "depth_mean": float(r.depth_mean[0]),
        "depth_mean_odd": float(r.depth_mean_odd[0]),
        "depth_mean_even": float(r.depth_mean_even[0]),
        "odd_even_mismatch": float(r.odd_even_mismatch),
        "secondary_eclipse_depth": secondary, "transit_shape_ratio": shape,
    }


def run_one(args):
    period, depth_ppm, rep, seed = args
    G = _G if _G else _init()
    rng = np.random.default_rng(seed)
    pool, inj, m06 = G["pool"], G["inj"], G["m06"]
    row = pool.iloc[int(rng.integers(0, len(pool)))]
    host = str(row["host"])
    r_star = float(row["st_rad"]) if np.isfinite(row["st_rad"]) and row["st_rad"] > 0 else 1.0
    m_star = 1.0
    t, f, e = base._load_curve(G, host)

    out = {"host": host, "injected_period": period,
           "injected_depth_ppm": depth_ppm, "repeat": rep}
    try:
        dur = base.transit_duration_days(period, r_star, m_star, 0.9)
        flux, _ = inj.inject_eclipsing_binary(t, f, period, depth_ppm, dur, rng)

        # ARM A: free production search
        a = base._tls_features(t, flux, e, r_star, m_star, m06)
        a.pop("_tls_period_min", None); a.pop("_tls_period_max", None)
        rec, alias = base._period_recovered(period, a["period"])
        out["free_detected"] = bool(rec); out["free_alias"] = alias
        for k in WATCH:
            out[f"free_{k}"] = a.get(k, np.nan)

        # ARM B: forced fold at the TRUE period
        b = _tls_forced(t, flux, e, r_star, m_star, m06, period)
        for k in WATCH:
            out[f"true_{k}"] = b.get(k, np.nan)
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
    print(f"{len(jobs)} EB trials x 2 TLS runs each (free + forced-at-true)", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=7, initializer=_init) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            res.append(fu.result())
            if i % 10 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  [{i}/{len(jobs)}] {el/60:.1f} min, eta "
                      f"{el/i*(len(jobs)-i)/60:.1f} min", flush=True)
    pd.DataFrame(res).to_csv(OUT_CSV, index=False)
    print(f"\nwall {(time.time()-t0)/60:.1f} min -> {OUT_CSV}")
    report()


def report():
    from scipy.stats import mannwhitneyu
    d = pd.read_csv(OUT_CSV)
    ok = d[d.status == "ok"].copy()
    print(f"\ntrials ok {len(ok)}/{len(d)}")
    ok["grp"] = np.where(ok.free_alias.eq("exact"), "free=EXACT",
                 np.where(ok.free_alias.eq("half"), "free=HALF", "free=other/none"))
    print("\n=== what the free search landed on ===")
    print(ok.grp.value_counts().to_string())
    print("\n  by injected period:")
    print(pd.crosstab(ok.injected_period, ok.grp).to_string())

    print("\n=== THE TEST: features at the FREE fold vs the TRUE-period fold ===")
    print("    (true_* is the same star, same signal, forced to the correct period)")
    for feat in ["secondary_eclipse_depth", "odd_even_mismatch"]:
        print(f"\n  --- {feat} ---")
        for g, sub in ok.groupby("grp"):
            fv = sub[f"free_{feat}"].astype(float)
            tv = sub[f"true_{feat}"].astype(float)
            m = np.isfinite(fv) & np.isfinite(tv)
            if m.sum() < 3:
                print(f"    {g:16s} n={m.sum():3d}  (too few)"); continue
            try:
                p = mannwhitneyu(fv[m], tv[m]).pvalue
            except Exception:
                p = float("nan")
            ratio = np.median(fv[m]) / np.median(tv[m]) if np.median(tv[m]) else np.nan
            print(f"    {g:16s} n={m.sum():3d}   free {np.median(fv[m]):+.6f}   "
                  f"true {np.median(tv[m]):+.6f}   ratio {ratio:6.2f}   MWU p={p:.4f}")

    print("\n=== interpretation aid: odd vs even depth at each fold (HALF group) ===")
    h = ok[ok.grp == "free=HALF"]
    if len(h):
        for pre in ("free", "true"):
            o = h[f"{pre}_depth_mean_odd"].astype(float).median()
            e = h[f"{pre}_depth_mean_even"].astype(float).median()
            print(f"    {pre:5s}  depth_odd {o:.6f}   depth_even {e:.6f}   "
                  f"|odd-even|/mean {abs(o-e)/((o+e)/2 if (o+e) else np.nan):.3f}")

    out = {"n_ok": int(len(ok)),
           "groups": ok.grp.value_counts().to_dict()}
    for feat in ["secondary_eclipse_depth", "odd_even_mismatch"]:
        out[feat] = {}
        for g, sub in ok.groupby("grp"):
            fv = sub[f"free_{feat}"].astype(float); tv = sub[f"true_{feat}"].astype(float)
            m = np.isfinite(fv) & np.isfinite(tv)
            if m.sum() >= 3:
                out[feat][g] = {"n": int(m.sum()),
                                "free_median": float(np.median(fv[m])),
                                "true_median": float(np.median(tv[m]))}
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved {OUT_JSON}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
