"""
bls_vs_tls_detector.py -- is BLS a COMPLEMENTARY detector to TLS, or does TLS
dominate?

Paired design: every trial injects ONE signal into ONE real light curve and
runs BOTH detectors on the identical array. So any difference is the detector,
not the star, the noise realisation, or the injected parameters.

WHY THIS COULD GO EITHER WAY
----------------------------
TLS matches a physically-shaped limb-darkened transit template, which is why it
beats BLS on canonical planet transits -- that is its whole design premise. BLS
matches a plain BOX. The standard argument for keeping BLS around is that a box
can be the better match for signals that are NOT canonical transits:
  * V-shaped grazing / eclipsing-binary eclipses (no flat bottom to match)
  * very short periods, where a transit occupies a large phase fraction
That is the population this tests. `injection.py` already provides both
shapes -- `inject_transit` (U-shaped, limb-darkened, planet-like) and
`inject_eclipsing_binary` (grazing, high impact parameter, V-shaped, with a
half-depth secondary) -- so no new injector is needed.

FAIRNESS
--------
BLS is given the SAME period search range TLS actually used on that same light
curve (TLS's own grid min/max, recorded per trial), and the same alias-aware
recovery test (exact / half / double / third / triple). Neither detector gets a
range the other did not have.

Nothing here touches training data, the model, or production. Detection stage
only.
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

OUT_CSV = os.path.join(HERE, "bls_vs_tls_results.csv")
OUT_JSON = os.path.join(HERE, "bls_vs_tls_summary.json")

# Short periods emphasised: BLS's claimed edge is short-period / large-phase-
# fraction signals. 0.5 d is below anything the earlier grids probed.
GRID_PERIODS = [0.5, 1.0, 2.0, 5.0, 10.0]
# Deep enough that detection is not purely SNR-limited -- otherwise both
# detectors just fail together and shape differences are invisible.
GRID_DEPTHS_PPM = [700, 2500, 5000]
KINDS = ["transit", "eb"]          # U-shaped planet vs V-shaped grazing EB
N_REPEATS = 10

_G = {}


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    G = base._init()          # single-sector test-split negative hosts
    _G.update(G)
    return _G


def _bls(t, f, e, pmin, pmax):
    """astropy BoxLeastSquares over the SAME period range TLS used."""
    from astropy.timeseries import BoxLeastSquares
    durations = np.array([0.02, 0.04, 0.08, 0.12, 0.20])   # days
    m = BoxLeastSquares(t, f, dy=e)
    pg = m.autoperiod(durations, minimum_period=max(pmin, 0.2),
                      maximum_period=pmax, minimum_n_transit=2,
                      frequency_factor=1.0)
    r = m.power(pg, durations)
    i = int(np.nanargmax(r.power))
    return float(r.period[i]), float(r.power[i]), float(r.depth[i]), len(pg)


def run_one(args):
    kind, period, depth_ppm, rep, seed = args
    G = _G if _G else _init()
    rng = np.random.default_rng(seed)
    pool, inj, m06 = G["pool"], G["inj"], G["m06"]

    row = pool.iloc[int(rng.integers(0, len(pool)))]
    host = str(row["host"])
    r_star = float(row["st_rad"]) if np.isfinite(row["st_rad"]) and row["st_rad"] > 0 else 1.0
    m_star = 1.0
    t_arr, f_arr, e_arr = base._load_curve(G, host)

    out = {"kind": kind, "host": host, "injected_period": period,
           "injected_depth_ppm": depth_ppm, "repeat": rep}
    try:
        b = float(rng.uniform(0.0, 0.6))
        dur = base.transit_duration_days(period, r_star, m_star, b)
        out["injected_duration_hours"] = dur * 24.0
        if kind == "transit":
            flux, _ = inj.inject_transit(t_arr, f_arr, period, depth_ppm, dur, rng,
                                         impact_param=b)
        else:
            flux, _ = inj.inject_eclipsing_binary(t_arr, f_arr, period, depth_ppm, dur, rng)

        # ---- TLS (production invocation) ----
        t0 = time.monotonic()
        feats = base._tls_features(t_arr, flux, e_arr, r_star, m_star, m06)
        out["tls_s"] = time.monotonic() - t0
        pmin, pmax = feats.pop("_tls_period_min"), feats.pop("_tls_period_max")
        out["tls_period"] = feats["period"]; out["tls_sde"] = feats["SDE"]
        rec, alias = base._period_recovered(period, feats["period"])
        out["tls_detected"] = bool(rec); out["tls_alias"] = alias

        # ---- BLS over the SAME range ----
        tb, fb, eb = m06.bin_lightcurve(t_arr, flux, e_arr)
        t1 = time.monotonic()
        bp, bpow, bdep, ngrid = _bls(tb, fb, eb, pmin, pmax)
        out["bls_s"] = time.monotonic() - t1
        out["bls_period"] = bp; out["bls_power"] = bpow
        out["bls_depth"] = bdep; out["bls_ngrid"] = ngrid
        rec2, alias2 = base._period_recovered(period, bp)
        out["bls_detected"] = bool(rec2); out["bls_alias"] = alias2
        out["period_min"], out["period_max"] = pmin, pmax
        out["status"] = "ok"
    except Exception as e:
        out.update({"status": f"error: {type(e).__name__}: {e}",
                    "tls_detected": False, "bls_detected": False})
    return out


def main():
    jobs, seed = [], 20260812
    for k in KINDS:
        for p in GRID_PERIODS:
            for d in GRID_DEPTHS_PPM:
                for r in range(N_REPEATS):
                    jobs.append((k, p, d, r, seed)); seed += 1
    print(f"{len(jobs)} paired trials "
          f"({len(KINDS)} shapes x {len(GRID_PERIODS)} periods x "
          f"{len(GRID_DEPTHS_PPM)} depths x {N_REPEATS})", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=7, initializer=_init) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            res.append(f.result())
            if i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  [{i}/{len(jobs)}] {el/60:.1f} min, eta "
                      f"{el/i*(len(jobs)-i)/60:.1f} min", flush=True)
    pd.DataFrame(res).to_csv(OUT_CSV, index=False)
    print(f"\nwall {(time.time()-t0)/60:.1f} min -> {OUT_CSV}")
    report()


def report():
    from scipy.stats import binomtest
    d = pd.read_csv(OUT_CSV); ok = d[d.status == "ok"].copy()
    print(f"\ntrials ok {len(ok)}/{len(d)}")
    ok["tls_exact"] = ok.tls_alias.eq("exact")
    ok["bls_exact"] = ok.bls_alias.eq("exact")

    print("\n=== RUNTIME ===")
    print(f"  TLS median {ok.tls_s.median():7.1f} s     BLS median {ok.bls_s.median():7.2f} s"
          f"     speedup {ok.tls_s.median()/max(ok.bls_s.median(),1e-9):.0f}x")
    print(f"  BLS period-grid points: median {ok.bls_ngrid.median():.0f}")

    for lab, col_t, col_b in [("ANY-ALIAS detection", "tls_detected", "bls_detected"),
                              ("EXACT-period detection", "tls_exact", "bls_exact")]:
        print(f"\n=== {lab}: TLS vs BLS, by shape ===")
        t = ok.groupby("kind").agg(n=(col_t, "size"), TLS=(col_t, "mean"), BLS=(col_b, "mean"))
        t["delta"] = t.BLS - t.TLS
        print(t.round(3).to_string())
        print(f"\n=== {lab}: by shape x period ===")
        t2 = ok.groupby(["kind", "injected_period"]).agg(
            n=(col_t, "size"), TLS=(col_t, "mean"), BLS=(col_b, "mean"))
        t2["delta"] = t2.BLS - t2.TLS
        print(t2.round(3).to_string())

    print("\n=== THE DECISIVE QUESTION: is there a BLS-ONLY population? ===")
    for kind, g in ok.groupby("kind"):
        both = int((g.tls_exact & g.bls_exact).sum())
        tls_only = int((g.tls_exact & ~g.bls_exact).sum())
        bls_only = int((~g.tls_exact & g.bls_exact).sum())
        neither = int((~g.tls_exact & ~g.bls_exact).sum())
        disc = tls_only + bls_only
        p = binomtest(bls_only, disc, 0.5).pvalue if disc else float("nan")
        print(f"  {kind:8s} n={len(g):3d}  both {both:3d}  TLS-only {tls_only:3d}  "
              f"**BLS-only {bls_only:3d}**  neither {neither:3d}   McNemar p={p:.4f}")
    both = int((ok.tls_exact & ok.bls_exact).sum())
    tls_only = int((ok.tls_exact & ~ok.bls_exact).sum())
    bls_only = int((~ok.tls_exact & ok.bls_exact).sum())
    disc = tls_only + bls_only
    p = binomtest(bls_only, disc, 0.5).pvalue if disc else float("nan")
    print(f"  {'POOLED':8s} n={len(ok):3d}  both {both:3d}  TLS-only {tls_only:3d}  "
          f"**BLS-only {bls_only:3d}**  neither {neither:3d}   McNemar p={p:.4f}")
    union = float((ok.tls_exact | ok.bls_exact).mean())
    print(f"\n  TLS alone {ok.tls_exact.mean():.3f}   union(TLS,BLS) {union:.3f}   "
          f"gain from adding BLS: {union - ok.tls_exact.mean():+.3f}")

    json.dump({
        "n_ok": int(len(ok)),
        "tls_median_s": float(ok.tls_s.median()), "bls_median_s": float(ok.bls_s.median()),
        "tls_exact": float(ok.tls_exact.mean()), "bls_exact": float(ok.bls_exact.mean()),
        "union_exact": union, "bls_only": bls_only, "tls_only": tls_only,
        "by_kind": {k: {"TLS": float(g.tls_exact.mean()), "BLS": float(g.bls_exact.mean()),
                        "bls_only": int((~g.tls_exact & g.bls_exact).sum())}
                    for k, g in ok.groupby("kind")},
    }, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved {OUT_JSON}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
