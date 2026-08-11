"""
injection_recovery_cadence_arm.py -- isolate CADENCE from BASELINE, cheaply.

Same question as the abandoned `injection_recovery_binning_arm.py`, run in the
opposite (affordable) direction.

THE PROBLEM
-----------
Production's `bin_lightcurve` targets a fixed 15,000 points, so effective
cadence degrades with baseline: 2 min at 1 sector, 8 min at 3, ~14 at 6, ~28 at
13. Baseline and cadence are therefore confounded by construction, and the
single-sector vs 3-sector comparison already on record is contaminated by it.

WHY NOT THE OBVIOUS TEST
------------------------
The direct test -- take 3-sector (76 d) curves and DISABLE binning to recover
2-min cadence -- was tried and ABANDONED. At native cadence TLS's period grid
over a 76 d baseline is enormous: the first 10 trials took 23 min, the next 10
took 193 min, projecting past 7 hours for 60 trials. Measured, then killed.

THIS RUNS IT THE CHEAP WAY
--------------------------
Single-sector hosts (~25 d, ~15,700 pts) sit BELOW production's 30,000-point
binning threshold, so production already gives them native 2-min cadence. So
instead of removing binning from an expensive long curve, ADD binning to a
cheap short one:

    arm A = 2 min  (production on single-sector; reused from
                    injection_recovery_sensitivity_results.csv)
    arm B = 8 min  (bin factor 4, matching what production DOES to a 3-sector
                    curve) -- run here

Same hosts, same baseline, same injected signals. Only cadence differs, and it
differs by exactly the amount production's binning imposes when you go from 1
to 3 sectors. Cost is the single-sector cost (~28 s/trial), not the 76-day one.

Pairing is exact: `run_one`'s seed fixes the host draw, t0 and impact
parameter, so re-running the same seeds gives the paired twin of an already
measured trial.

INTERPRETATION
--------------
If arm B is materially worse than arm A, then a good part of what the earlier
single-vs-wide comparison attributed to "longer baseline" was really "coarser
cadence pulling the other way" -- and the deployed pipeline is partly cancelling
its own gains from longer baselines via a tunable constant
(`TARGET_POINTS_AFTER_BINNING`), not via physics.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import injection_recovery_sensitivity as base

base.BIN_FACTOR_OVERRIDE = 4          # 2 min -> 8 min, what production does at 3 sectors
base.OUT_CSV = os.path.join(HERE, "injection_recovery_cadence_arm_results.csv")

REF_CSV = os.path.join(HERE, "injection_recovery_sensitivity_results.csv")
PERIODS = [1.0, 3.0, 6.0, 10.0]       # all fully in-range at 25 d, so cadence is the only variable
DEPTHS = [700, 1200, 2500, 5000]


def _matched_jobs():
    """Mirror base.main()'s seed assignment for the single-sector run exactly:
    periods outer, depths inner, repeats innermost, seed incremented every
    tuple whether or not we select it."""
    jobs, seed = [], 20260810
    for p in [1.0, 3.0, 6.0, 10.0, 14.0, 20.0]:              # original GRID_PERIODS
        for d in [84, 150, 250, 400, 700, 1200, 2500, 5000]:  # original GRID_DEPTHS_PPM
            for r in range(10):                               # original N_REPEATS
                if p in PERIODS and d in DEPTHS:
                    jobs.append((p, d, r, seed))
                seed += 1
    return jobs


def main():
    jobs = _matched_jobs()
    print(f"{len(jobs)} paired trials at 8-min cadence (bin factor 4)", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=7, initializer=base._init) as ex:
        futs = {ex.submit(base.run_one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 20 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  [{i}/{len(jobs)}] {el/60:.1f} min, eta {el/i*(len(jobs)-i)/60:.1f} min", flush=True)
    pd.DataFrame(results).to_csv(base.OUT_CSV, index=False)
    print(f"\nwall {(time.time()-t0)/60:.1f} min", flush=True)
    compare()


def compare():
    b = pd.read_csv(base.OUT_CSV)
    a = pd.read_csv(REF_CSV)
    a = a[(a.status == "ok") & (~a.is_control)].copy()
    b = b[b.status == "ok"].copy()
    a["exact"] = a.alias.eq("exact"); b["exact"] = b.alias.eq("exact")
    key = ["injected_period", "injected_depth_ppm", "repeat"]
    m = a.merge(b, on=key, suffixes=("_2min", "_8min"))
    m = m[m.host_2min == m.host_8min]
    print(f"\nmatched pairs (same host, same signal, only cadence differs): {len(m)}")
    if not len(m):
        print("  PAIRING FAILED"); return
    print("\n=== EXACT-period detection: 2-min (production at 1 sector) vs 8-min (production at 3 sectors) ===")
    t = m.groupby("injected_depth_ppm").agg(
        n=("exact_2min", "size"), cad_2min=("exact_2min", "mean"), cad_8min=("exact_8min", "mean"))
    t["delta"] = t.cad_8min - t.cad_2min
    print(t.round(3).to_string())
    t2 = m.groupby("injected_period").agg(
        n=("exact_2min", "size"), cad_2min=("exact_2min", "mean"), cad_8min=("exact_8min", "mean"))
    t2["delta"] = t2.cad_8min - t2.cad_2min
    print(); print(t2.round(3).to_string())
    n = len(m)
    d2, d8 = m.exact_2min.mean(), m.exact_8min.mean()
    disc = m[m.exact_2min != m.exact_8min]
    b_only = int(disc.exact_8min.sum()); a_only = int(disc.exact_2min.sum())
    print(f"\n=== pooled (n={n}) ===")
    print(f"  2-min : {d2:.3f}")
    print(f"  8-min : {d8:.3f}")
    print(f"  delta : {d8-d2:+.3f}")
    print(f"  discordant pairs {len(disc)}  (8-min-only wins {b_only}, 2-min-only wins {a_only})")
    if len(disc):
        from scipy.stats import binomtest
        p = binomtest(b_only, len(disc), 0.5).pvalue
        print(f"  McNemar exact p = {p:.4f}")
    print(f"\n  median SDE  2-min {m.recovered_sde_2min.median():.2f}   8-min {m.recovered_sde_8min.median():.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare()
    else:
        main()
