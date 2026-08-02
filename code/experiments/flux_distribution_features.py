"""flux_distribution_features.py -- ITEM 2: higher-order statistics of the
phase-folded flux distribution.

TLS reports where and how deep a transit is, but nothing about the SHAPE of the
flux distribution it leaves behind. These features ask a different question of
data already on disk: given the known ephemeris, how are in-transit fluxes
distributed relative to out-of-transit ones?

FEATURES AND WHY EACH ONE

  in_skew / in_kurt    Skewness and kurtosis of in-transit flux.
      A real transit is a flat-bottomed box blurred by limb darkening and
      noise: in-transit flux should be roughly symmetric about the depth. A
      grazing eclipse or a blend produces a V-shape, so flux spends unequal
      time at different levels and the distribution skews. Kurtosis separates
      a flat bottom (platykurtic) from a sharp cusp (leptokurtic) -- shape
      information TLS's single depth number cannot express.

  out_skew / out_kurt  Same for out-of-transit flux.
      This is the control. Out-of-transit flux should be near-Gaussian noise
      around 1.0. Departures indicate stellar variability or residual
      systematics -- properties of the STAR, not the transit, and known
      contaminants for false positives.

  skew_diff            in_skew - out_skew
  kurt_diff            in_kurt - out_kurt
      The differences are the point. Absolute moments are dominated by each
      star's own noise character; differencing cancels that and isolates what
      the transit itself does to the distribution. This is the same reasoning
      that made secondary/primary depth RATIO more meaningful than either
      depth alone.

  wavelet_e1..e3       Relative energy in the 3 finest Haar detail levels of
      the phase-folded, binned flux.
      A transit is a single coherent dip: its energy concentrates at coarse
      scales. Noise and short-timescale systematics live at fine scales. The
      ratio of fine-scale to total energy is therefore a compact
      "is this one clean event or a mess" measure. Three levels, not a bank of
      dozens -- the small-lift round already showed that adding many weakly
      motivated columns just adds chances for one to pass by luck.

Uses the already-preprocessed light curves in data/processed[_negative]/ and
the ephemeris already in training.csv. No downloads, no re-preprocessing, no
TLS re-run. Resumable: re-running skips hosts already in the output.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait as cwait

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
PROC_DIRS = [os.path.join(ROOT, "data", "processed"),
             os.path.join(ROOT, "data", "processed_negative")]
OUT_CSV = os.path.join(SCRIPT_DIR, "flux_distribution_features.csv")

N_WORKERS = 8
CHECKPOINT_EVERY = 200
STALL_GAP_S = 900
N_PHASE_BINS = 128


class _Stalled(Exception):
    pass


def _find(host):
    for d in PROC_DIRS:
        p = os.path.join(d, host + ".csv")
        if os.path.exists(p):
            return p
    return None


def _moments(x):
    """Skewness and excess kurtosis. Written out rather than pulled from scipy
    so a degenerate slice returns NaN instead of a warning-and-garbage."""
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return np.nan, np.nan
    m, s = x.mean(), x.std()
    if s <= 0:
        return np.nan, np.nan
    z = (x - m) / s
    return float((z ** 3).mean()), float((z ** 4).mean() - 3.0)


def _haar_details(x, levels=3):
    """Orthonormal Haar detail coefficients, finest level first.

    Implemented directly rather than pulling in PyWavelets: for Haar the
    transform is just normalised pairwise sums and differences, this project
    deliberately keeps its core install lean (torch was moved out to
    requirements-experiments.txt in the reproducibility pass), and adding a
    dependency for three numbers is a poor trade. Equivalent to
    pywt.wavedec(x, 'haar', level=3)[1:] for a power-of-two length input.
    """
    x = np.asarray(x, dtype=float)
    details = []
    cur = x
    for _ in range(levels):
        n = len(cur) // 2
        if n < 1:
            break
        a, b = cur[:2 * n:2], cur[1:2 * n:2]
        details.append((a - b) / np.sqrt(2.0))   # detail
        cur = (a + b) / np.sqrt(2.0)             # approximation, next level
    return details


def compute_one(args):
    host, period, t0, duration = args
    out = {"host": host}
    try:
        path = _find(host)
        if path is None:
            return {**out, "status": "no processed file"}
        if not all(np.isfinite([period, t0, duration])) or period <= 0 or duration <= 0:
            return {**out, "status": "no usable ephemeris"}

        d = pd.read_csv(path)
        t, f = d["time"].to_numpy(float), d["flux"].to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(f)
        t, f = t[ok], f[ok]
        if len(t) < 200:
            return {**out, "status": "too few points"}

        phase = ((t - t0 + 0.5 * period) % period) / period - 0.5   # -0.5..0.5
        half = (duration / period) / 2.0
        if not np.isfinite(half) or half <= 0 or half > 0.4:
            return {**out, "status": "implausible duration/period"}

        intr = np.abs(phase) <= half
        # Guard band: points just outside the transit are contaminated by
        # ingress/egress, so the out-of-transit control skips them rather than
        # quietly blending transit signal into its own baseline.
        outr = np.abs(phase) >= (3 * half)
        if intr.sum() < 20 or outr.sum() < 50:
            return {**out, "status": "too few in/out points"}

        i_sk, i_ku = _moments(f[intr])
        o_sk, o_ku = _moments(f[outr])
        out.update({"in_skew": i_sk, "in_kurt": i_ku,
                    "out_skew": o_sk, "out_kurt": o_ku,
                    "skew_diff": (i_sk - o_sk) if np.isfinite([i_sk, o_sk]).all() else np.nan,
                    "kurt_diff": (i_ku - o_ku) if np.isfinite([i_ku, o_ku]).all() else np.nan,
                    "n_in": int(intr.sum()), "n_out": int(outr.sum())})

        # ---- wavelet energies on the phase-folded, binned curve ----
        try:
            order = np.argsort(phase)
            ph, fl = phase[order], f[order]
            idx = np.clip(((ph + 0.5) * N_PHASE_BINS).astype(int), 0, N_PHASE_BINS - 1)
            binned = np.array([fl[idx == b].mean() if (idx == b).any() else np.nan
                               for b in range(N_PHASE_BINS)])
            # interpolate empty bins so the transform sees a continuous curve
            nans = ~np.isfinite(binned)
            if nans.all():
                raise ValueError("all bins empty")
            if nans.any():
                binned[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans),
                                         binned[~nans])
            details = _haar_details(binned - binned.mean(), levels=3)
            tot = sum(float((c ** 2).sum()) for c in details) + 1e-12
            for i, c in enumerate(details, 1):
                out[f"wavelet_e{i}"] = float((c ** 2).sum()) / tot
        except Exception as e:
            out["wavelet_note"] = f"{type(e).__name__}"

        out["status"] = "ok"
        return out
    except Exception as e:
        return {**out, "status": f"error: {type(e).__name__}: {e}"}


def main():
    df = pd.read_csv(TRAINING_CSV)
    done = set()
    rows = []
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        rows = prev.to_dict("records")
        done = set(prev["host"])
        print(f"resuming: {len(done)} hosts already computed")

    jobs = [(r["host"], r.get("period"), r.get("T0"), r.get("duration"))
            for _, r in df.iterrows() if r["host"] not in done]
    print(f"{len(jobs)} hosts to process with {N_WORKERS} workers")
    if not jobs:
        print("nothing to do")
        return

    ex = ProcessPoolExecutor(max_workers=N_WORKERS)
    futures = {ex.submit(compute_one, j): j for j in jobs}
    waiting = set(futures)
    i, t0 = 0, time.time()
    try:
        while waiting:
            got, waiting = cwait(waiting, timeout=STALL_GAP_S, return_when=FIRST_COMPLETED)
            if not got:
                raise _Stalled()
            for fut in got:
                i += 1
                try:
                    rows.append(fut.result())
                except Exception as e:
                    rows.append({"host": futures[fut][0], "status": f"crash: {e}"})
                if i % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                    print(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    except _Stalled:
        print(f"STALLED: no completion in {STALL_GAP_S}s -- saving partial results")
    finally:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        procs = list((getattr(ex, "_processes", None) or {}).values())
        ex.shutdown(wait=False, cancel_futures=True)
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass

    res = pd.DataFrame(rows)
    print(f"\ndone: {len(res)} rows")
    print(res["status"].value_counts().head(8).to_string())
    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
