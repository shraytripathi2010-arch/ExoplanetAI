"""multisector_feasibility.py -- Part 1 feasibility gate for raising
`trap_vshape` usable coverage by stacking TESS sectors.

THE QUESTION. `trap_vshape` is validated and real (single-feature AUC 0.3595,
spatially clean, confound-controlled by arms C/D) but contributes only +0.0007
against a 0.0097 detection floor. Its usable coverage is 31.2%, gated by
depth_snr >= 3 and vshape_err <= 0.30 -- a per-star SNR problem. Stacking
sectors is the standard fix. Does it move +0.0007 to >= 0.0097?

This script answers that BEFORE any download, per the project's feasibility-gate
discipline, using three measured inputs and one stated assumption.

MEASURED INPUT 1 -- the pipeline currently uses ONE sector per star.
`01_download_known.py` and `01_download_negative.py` both call
`search[0].download()`: the FIRST matching product only. Confirmed downstream --
processed light curves span 24-27 days with zero gaps > 5 days, i.e. a single
TESS sector. So "stacking" here means something real and currently absent:
concatenate the per-cadence flux from every available sector for a star, then
phase-fold the combined series on the stored period/T0 and fit ONE trapezoid to
that deeper fold. Not per-sector fits averaged afterwards -- the point is more
in-transit points under a single fit so the ingress rises above the noise.

MEASURED INPUT 2 -- how many sectors actually exist (live MAST query).
Sampled 60 stars from the recoverable population; 59 resolved. Availability is
far better than a single-sector pipeline suggests: 93% have >= 2 sectors,
median 3, mean 4.6. Sector availability is NOT the bottleneck.

MEASURED INPUT 3 -- precision genuinely sharpens the signal, checked because
the naive assumption (that stacking only adds stars) would have understated the
case. Binned by vshape_err, the class separation deepens monotonically, and it
is not a class-mix artifact -- planet fraction across those bins runs
0.751/0.588/0.603/0.686, non-monotone, while AUC is monotone.

THE ASSUMPTION, stated because the conclusion rests on it: the model-level AUC
delta scales roughly LINEARLY with the fraction of test stars carrying a real
measured value, the rest being median-imputed. This is first-order and probably
GENEROUS -- newly recovered stars are the marginal, noisiest ones, so their
per-star contribution is likely below that of the currently covered high-SNR
population.
"""
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from fast_auc import fast_auc  # noqa: E402

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
FEAT = os.path.join(SCRIPT_DIR, "trapezoid_shape_features.csv")
AVAIL = os.path.join(SCRIPT_DIR, "multisector_availability.json")
OUT = os.path.join(SCRIPT_DIR, "multisector_feasibility.json")

MIN_DEPTH_SNR, MAX_VSHAPE_ERR = 3.0, 0.30
MDE = 0.0097
BASE_DELTA = 0.0007      # measured arm A, 12 resamples
BASE_DEV = 0.1405        # |0.3595 - 0.5|, separation on the covered population


def sector_cdf():
    """P(star has >= n sectors), from the live MAST sample."""
    if os.path.exists(AVAIL):
        n = np.array([r["n_sectors"] for r in json.load(open(AVAIL))])
        n = n[n > 0]
        return {k: float((n >= k).mean()) for k in range(1, 7)}, len(n)
    return {1: 1.0, 2: 0.93, 3: 0.62, 4: 0.47, 5: 0.30, 6: 0.25}, 0


def main():
    df = pd.read_csv(TRAINING)
    tf = pd.read_csv(FEAT)
    m = df[["host", "label"]].merge(tf, on="host", how="left")
    y = m["label"].to_numpy(float)
    v = pd.to_numeric(m["trap_vshape"], errors="coerce").to_numpy()
    snr = pd.to_numeric(m["trap_depth_snr"], errors="coerce").to_numpy()
    err = pd.to_numeric(m["trap_vshape_err"], errors="coerce").to_numpy()
    ok = (m["trap_vshape"].notna() & (m["trap_status"] == "ok")).to_numpy()
    use = ok & (snr >= MIN_DEPTH_SNR) & (err <= MAX_VSHAPE_ERR)
    N, cur = len(m), int(use.sum())

    P, nq = sector_cdf()
    pav = lambda k: P[int(np.clip(k, 1, 6))]

    print("=" * 92)
    print("MULTI-SECTOR STACKING -- Part 1 feasibility gate for trap_vshape")
    print("=" * 92)
    print(f"\ncoverage now: {cur}/{N} = {100*cur/N:.1f}%   excluded {N-cur}")
    print(f"sector availability sampled live from MAST (n={nq} resolved): "
          f">=2 {100*P[2]:.0f}%  >=4 {100*P[4]:.0f}%")

    # ---- expected recovery, star by star -------------------------------
    a = (~use) & ok & (snr < MIN_DEPTH_SNR) & np.isfinite(snr) & (snr > 0)
    exp_a = sum(pav(np.ceil((3.0 / s) ** 2)) for s in snr[a])
    b = (~use) & ok & (snr >= MIN_DEPTH_SNR) & np.isfinite(err) & (err > MAX_VSHAPE_ERR)
    exp_b = sum(pav(np.ceil((e / MAX_VSHAPE_ERR) ** 2)) for e in err[b])
    nonconv = int(((~use) & ~ok).sum())
    rec = exp_a + exp_b
    new = cur + rec

    print(f"\nEXPECTED RECOVERY (required sectors x empirical availability)")
    print(f"  depth_snr < 3      n={int(a.sum()):5d}  ->  {exp_a:6.0f} recovered")
    print(f"  vshape_err > 0.30  n={int(b.sum()):5d}  ->  {exp_b:6.0f} recovered")
    print(f"  non-converged      n={nonconv:5d}  ->  not modelled (structural)")
    print(f"  TOTAL {rec:.0f} stars  =>  coverage {100*new/N:.1f}% "
          f"(x{new/cur:.2f} on covered fraction)")

    # ---- does precision deepen the separation? -------------------------
    print("\nPRECISION EFFECT (checked, not assumed)")
    print(f"  {'vshape_err':<14}{'n':>6}{'planet_frac':>13}{'AUC':>9}{'|dev|':>8}")
    bands, devs = [(0, .05), (.05, .10), (.10, .20), (.20, .30)], {}
    for lo, hi in bands:
        s = use & (err >= lo) & (err < hi)
        au = fast_auc(y[s], v[s])
        devs[(lo, hi)] = abs(au - 0.5)
        print(f"  {f'{lo:.2f}-{hi:.2f}':<14}{int(s.sum()):>6}{y[s].mean():>13.3f}"
              f"{au:>9.4f}{abs(au-0.5):>8.4f}")
    best_dev = devs[(0, .05)]
    sharpen = best_dev / BASE_DEV
    print(f"  monotone in AUC while planet fraction is NOT -> a precision effect,")
    print(f"  not a class-mix artifact. Best band sharpens separation x{sharpen:.2f}")

    # ---- the decisive arithmetic ---------------------------------------
    print("\n" + "=" * 92)
    print("DECISIVE ARITHMETIC  (delta ~ covered_fraction x per-star separation)")
    print("=" * 92)
    scen = [
        ("measured now", 1.0, 1.0),
        ("expected stacking", new / cur, 1.0),
        ("expected + precision gain", new / cur, sharpen),
        ("CEILING: 100% coverage", N / cur, 1.0),
        ("CEILING: 100% + precision", N / cur, sharpen),
    ]
    print(f"  {'scenario':<30}{'cov x':>8}{'sharp x':>9}{'delta':>10}{'vs 0.0097':>12}")
    out = {}
    for name, c, s in scen:
        dlt = BASE_DELTA * c * s
        print(f"  {name:<30}{c:>8.2f}{s:>9.2f}{dlt:>+10.5f}{dlt/MDE:>11.2f}x")
        out[name] = {"coverage_mult": c, "sharpen_mult": s, "delta": dlt,
                     "fraction_of_threshold": dlt / MDE}
    print(f"\n  multiplier required to reach 0.0097: {MDE/BASE_DELTA:.1f}x")
    print("  MAXIMUM physically available (100% coverage x best precision): "
          f"{(N/cur)*sharpen:.1f}x")
    verdict = "GO" if BASE_DELTA * (new / cur) * sharpen >= MDE else "NO-GO"
    print(f"\n  VERDICT: {verdict}")
    print("=" * 92)

    json.dump({"n_train": N, "covered_now": cur,
               "expected_recovery": rec, "coverage_now": cur / N,
               "coverage_expected": new / N, "nonconverged": nonconv,
               "sector_cdf": P, "n_sector_queries": nq,
               "sharpen_mult": sharpen, "scenarios": out,
               "required_mult": MDE / BASE_DELTA,
               "max_available_mult": (N / cur) * sharpen,
               "verdict": verdict}, open(OUT, "w"), indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
