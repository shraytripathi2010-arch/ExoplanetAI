"""
eightsector_run.py -- Stage E: injection-recovery on a real 8-consecutive-sector
TESS sample (~216 d), the third point on the period-ceiling curve.

Reuses `injection_recovery_sensitivity.py` wholesale (injector, production TLS
call, 22-TLS/9-host feature split, deployed model, alias-aware recovery test,
zero-depth control arm). Only the host pool and grid change.

MEASURED SAMPLE (39 hosts, after continuity + flux-clean + 9-feature filters)
    baseline  216.5 d median   (single sector 25.3, 3 sectors 76.3)
    max gap   <= 6.2 d, duty cycle 0.83  -> genuinely contiguous
    st_rad    1.36 median      (single sector 1.48, wide sector 1.90)
              -- a closer population match than the wide-sector run had

PREDICTION, STATED BEFORE RUNNING
    period_max = baseline / 2 = 108.2 d
    against 12.66 measured at 1 sector and 38.15 at 3 sectors.

GRID
Periods span the whole falloff: 3 d (72 transits) down to 120 d (1.8 transits,
and beyond the predicted 108 d ceiling, as the upper bookend -- the same role
P=40 played in the 3-sector run). Depths are trimmed to 4: the Earth-size floor
is already characterised and shallow trials only add SNR-limited noise to a
question about search REACH. 84 ppm is kept for continuity with both prior runs.

CADENCE NOTE
At ~127,000 points production's fixed 15,000-point binning gives bin factor 9,
i.e. ~18 min effective cadence. The paired cadence arm showed no harm from
2 -> 8 min (delta +0.020, McNemar p = 0.375); 18 min is an extrapolation beyond
what was tested, so it is recorded as a caveat, not assumed away.
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import injection_recovery_sensitivity as base

PROJECT_ROOT = os.path.join(HERE, "..", "..")
LC_DIR = os.path.join(PROJECT_ROOT, "data", "processed_8sector")
POOL_CSV = os.path.join(HERE, "eightsector_final_pool.csv")

base.OUT_CSV = os.path.join(HERE, "eightsector_results.csv")
base.OUT_JSON = os.path.join(HERE, "eightsector_summary.json")
base.GRID_DEPTHS_PPM = [84, 1200, 2500, 5000]
base.GRID_PERIODS = [3.0, 10.0, 20.0, 40.0, 60.0, 90.0, 120.0]
base.N_REPEATS = 10
base.N_CONTROL = 40

_ORIG_INIT = base._init


def _init_eightsector():
    G = _ORIG_INIT()
    pool = pd.read_csv(POOL_CSV)
    assert len(pool) > 20, f"pool too small: {len(pool)}"
    for c in base.HOST_FEATURES:
        assert c in pool.columns, f"missing host feature {c}"
    G["pool"] = pool.reset_index(drop=True)
    G["lc_dir"] = LC_DIR
    return G


base._init = _init_eightsector

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        base.summarize(pd.read_csv(base.OUT_CSV))
    else:
        base.main()
