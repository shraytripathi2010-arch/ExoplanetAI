"""
injection_recovery_widesector.py -- does the ~12.5 d period_max detection
ceiling move as predicted on a LONGER CONTIGUOUS BASELINE?

Follow-up to `injection_recovery_sensitivity.py`, which found on single-sector
TESS (~25 d baseline) that TLS's default `period_max` (~half the baseline)
imposed a **structural ~12.5 d ceiling**: at P = 20 d the exact-period
detection rate was 0.000 and every apparent detection was an alias.

This reuses that script wholesale -- the same injector (`injection.py`), the
same production TLS invocation, the same 22-TLS + 9-host feature split, the
same deployed model, the same zero-depth control arm, the same alias-aware
recovery test. Only the HOST POOL and the period grid change.

WHAT THIS IS NOT
----------------
This is **not** the closed multi-sector stacking/folding investigation. That
one failed because coherently FOLDING a transit at a stored, imprecise
ephemeris across sector gaps accumulates phase error (`span * sigma_P / P` in
transit durations) -- a data-quality problem about a stored ephemeris.

Nothing here folds anything at a stored ephemeris. The sectors are simply
concatenated into one time-ordered array, and **TLS runs its own blind period
search on it**, exactly as it does on a single sector. The only quantity under
test is TLS's own `period_max` search bound, which is a function of
`max(t) - min(t)` alone.

THE HOST POOL, AND THE THREE FILTERS APPLIED TO IT
--------------------------------------------------
`data/processed_unknown_widesector/` -- TESS stars from the wide-sector-window
pilot, 3 consecutive sectors already concatenated by
`06_download_unknown.download_one_star`'s multi-sector branch (pd.concat +
sort-by-time; no folding). Filters, all measured rather than assumed:

  1. CONTINUOUS: baseline 70-85 d and max internal gap < 5 d. This excludes
     stars whose "baseline" spans a year-long gap between non-consecutive
     sectors, which would give TLS a huge period grid over almost no data.
     Selected sample: ~76 d span, ~2.2 d max gap, 6 gaps > 0.5 d (TESS
     downlinks + sector breaks), **86% duty cycle**.
  2. FLUX-CLEAN: median flux within 1% of 1.0, robust sigma in (0, 0.05), and
     <1% of points beyond 10 sigma. Roughly a third of this pool has severe
     flux outliers (raw std up to ~2000) that the single-sector negative-class
     pool does not; excluded so the ONLY difference from the prior run is
     baseline length.
  3. All 9 host features finite (st_rad/st_teff from
     `unknown_candidate_list_widesector.csv`, crowding + variability from
     `unknown_features_widesector.csv`).

ANALYTICAL PREDICTION, STATED BEFORE RUNNING
--------------------------------------------
TLS's default `period_max` = (max(t) - min(t)) / 2 (it requires >= 2 transits).

    single sector : baseline 24.9 d -> period_max 12.5 d   [MEASURED 12.5 d]
    wide sector   : baseline 76.0 d -> period_max 38.0 d   [PREDICTED]

Verified on one real curve before launching the grid: a 76.2 d host gave a
searched grid of 0.532 -> **38.105 d**, vs 76.2/2 = 38.1. So the prediction is
already confirmed at the grid level; this run tests whether DETECTION actually
follows, which is a separate question (a period can be inside the grid and
still be undetectable for want of transits or SNR).

Period grid therefore spans the old ceiling (14, 20), the newly-searchable
region (30), and just beyond the new ceiling (40) as an upper bookend.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import injection_recovery_sensitivity as base

PROJECT_ROOT = os.path.join(HERE, "..", "..")
LC_DIR = os.path.join(PROJECT_ROOT, "data", "processed_unknown_widesector")
POOL_CSV = os.path.join(HERE, "widesector_host_pool.csv")

base.OUT_CSV = os.path.join(HERE, "injection_recovery_widesector_results.csv")
base.OUT_JSON = os.path.join(HERE, "injection_recovery_widesector_summary.json")

# Same depths as the single-sector run, for a like-for-like comparison.
base.GRID_DEPTHS_PPM = [84, 150, 250, 400, 700, 1200, 2500, 5000]
# Same periods PLUS 30 (previously unsearchable, now predicted searchable) and
# 40 (just beyond the new predicted ceiling of 38.0 d).
base.GRID_PERIODS = [1.0, 3.0, 6.0, 10.0, 14.0, 20.0, 30.0, 40.0]
base.N_REPEATS = 10
base.N_CONTROL = 60


# Captured BEFORE the patch below, or _init_widesector would call itself.
_ORIG_INIT = base._init


def _init_widesector():
    """Same worker setup as the base script, then swap in the longer-baseline
    host pool and light-curve directory. Everything else -- feature columns,
    the deployed model, the injector -- is inherited unchanged."""
    G = _ORIG_INIT()
    pool = pd.read_csv(POOL_CSV)
    assert len(pool) > 50, f"host pool too small: {len(pool)}"
    for c in base.HOST_FEATURES:
        assert c in pool.columns, f"missing host feature {c}"
    G["pool"] = pool.reset_index(drop=True)
    G["lc_dir"] = LC_DIR
    return G


base._init = _init_widesector

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        base.summarize(pd.read_csv(base.OUT_CSV))
    else:
        base.main()
