"""
injection_recovery_binning_arm.py -- isolate the CADENCE effect from the
BASELINE effect, on data already downloaded.

WHY THIS EXISTS
---------------
Production's `bin_lightcurve` targets a FIXED 15,000 points, so the effective
cadence it feeds TLS degrades linearly with baseline:

    1 sector  (~15,700 pts)  bin factor 1   ->   2 min   (60 samples in a 2 h transit)
    3 sectors (~46,700 pts)  bin factor 4   ->   8 min   (15 samples)
    6 sectors (~96,000 pts)  bin factor 7   ->  14 min   ( 8.6 samples)
    13 sectors(~208,000 pts) bin factor 14  ->  28 min   ( 4.0 samples)

So "longer baseline" and "coarser cadence" are confounded in this pipeline by
construction. The single-sector vs 3-sector comparison already reported was
therefore NOT a pure baseline comparison -- it also went 2 min -> 8 min. (The
`period_max` result there is unaffected: that is `(max(t)-min(t))/2`, which
binning does not touch. The depth axis and the falloff analysis ARE affected.)

WHAT THIS RUNS
--------------
A perfectly PAIRED comparison on the SAME 3-sector hosts:

    arm A = production binning (8 min)   <- already measured, reused from
                                            injection_recovery_widesector_results.csv
    arm B = no binning       (2 min)     <- run here

Pairing is exact because `run_one`'s seed determines the host draw, t0, and
impact parameter. Re-running the identical seeds with BIN_FACTOR_OVERRIDE=1
gives the same star, same injected signal, same everything -- only the cadence
handed to TLS differs. So the difference is attributable to cadence alone,
with no host-population confound of the kind that muddied the earlier
single-vs-wide comparison.

Cheap by design: no new downloads, no new stars. It exists to decide whether a
much more expensive 6-13 sector experiment should be built around a binning
arm at all, BEFORE spending the download and compute on it.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import injection_recovery_sensitivity as base
import injection_recovery_widesector as ws   # installs the wide-sector pool + lc_dir

base.BIN_FACTOR_OVERRIDE = 1                 # <-- the only thing that changes
base.OUT_CSV = os.path.join(HERE, "injection_recovery_binning_arm_results.csv")
base.OUT_JSON = os.path.join(HERE, "injection_recovery_binning_arm_summary.json")

# Subset of the wide-sector grid. Deep injections only: the question is cadence
# and search reach, not the depth floor (already characterised), and shallow
# trials would just add SNR-limited noise.
PERIODS = [3.0, 10.0, 20.0]
DEPTHS = [1200, 5000]
N_REPEATS = 10


def _matched_jobs():
    """Rebuild the EXACT (period, depth, repeat, seed) tuples the wide-sector
    run used for these grid points, so every trial here is the paired twin of
    one already measured. Seed assignment must mirror base.main()'s loop order
    exactly -- periods outer, depths inner, repeats innermost."""
    jobs, seed = [], 20260810
    for p in ws.base.GRID_PERIODS:
        for d in ws.base.GRID_DEPTHS_PPM:
            for r in range(ws.base.N_REPEATS):
                if p in PERIODS and d in DEPTHS:
                    jobs.append((p, d, r, seed))
                seed += 1
    return jobs


def main():
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = _matched_jobs()
    print(f"{len(jobs)} paired trials at NATIVE 2-min cadence "
          f"(periods {PERIODS}, depths {DEPTHS}, {N_REPEATS} repeats)", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=7, initializer=base._init) as ex:
        futs = {ex.submit(base.run_one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 10 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  [{i}/{len(jobs)}] {el/60:.1f} min, eta {el/i*(len(jobs)-i)/60:.1f} min", flush=True)
    pd.DataFrame(results).to_csv(base.OUT_CSV, index=False)
    print(f"\nwall {(time.time()-t0)/60:.1f} min -> {base.OUT_CSV}")
    compare()


def compare():
    """Paired arm-A vs arm-B comparison, joined on (period, depth, repeat)."""
    b = pd.read_csv(base.OUT_CSV)
    a = pd.read_csv(os.path.join(HERE, "injection_recovery_widesector_results.csv"))
    key = ["injected_period", "injected_depth_ppm", "repeat"]
    a = a[a.status == "ok"]; b = b[b.status == "ok"]
    a["exact"] = a.alias.eq("exact"); b["exact"] = b.alias.eq("exact")
    m = a.merge(b, on=key, suffixes=("_prod", "_native"))
    m = m[m.host_prod == m.host_native]          # pairing sanity check
    print(f"\nmatched pairs (same host, same signal): {len(m)}")
    if not len(m):
        print("  PAIRING FAILED -- seeds did not reproduce the same host draws.")
        return
    print("\n=== EXACT-period detection: production 8-min binning vs native 2-min ===")
    t = m.groupby(["injected_period", "injected_depth_ppm"]).agg(
        n=("exact_prod", "size"),
        prod_8min=("exact_prod", "mean"),
        native_2min=("exact_native", "mean"))
    t["delta"] = t.native_2min - t.prod_8min
    print(t.round(3).to_string())
    print("\n=== pooled ===")
    print(f"  production 8-min : {m.exact_prod.mean():.3f}")
    print(f"  native 2-min     : {m.exact_native.mean():.3f}")
    print(f"  delta            : {m.exact_native.mean() - m.exact_prod.mean():+.3f}")
    disagree = m[m.exact_prod != m.exact_native]
    print(f"  pairs disagreeing: {len(disagree)}/{len(m)}  "
          f"(native-only wins {int((disagree.exact_native).sum())}, "
          f"prod-only wins {int((disagree.exact_prod).sum())})")
    print(f"\n  median SDE  production {m.recovered_sde_prod.median():.2f}  "
          f"native {m.recovered_sde_native.median():.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare()
    else:
        main()
