"""cadence_window_pilot.py -- PART 1: does a cadence-aware savgol window help,
on the population it is supposed to help?

THE FIX
-------
Production caps the window at 401 POINTS regardless of cadence, so the physical
protected timescale is whatever 401 points happens to span:

    20-sec   401 pts =   2.2 h     6x MORE aggressive than intended
    2-min    401 pts =  13.4 h     the design intent
    10-min   401 pts =  22.3 h
    30-min   401 pts = 200.5 h    15x LESS aggressive than intended

The fix holds the PHYSICAL width fixed instead, at the 2-min design intent
(401 x 2 min = 13.3667 h), and derives the point count per star.

*** MAX_FLATTEN_WINDOW HAS A HISTORY OF UNIT CONFUSION IN THIS PROJECT. ***
That is why `TARGET_PROTECTED_HOURS` is DERIVED from `401 * 2.0 / 60.0` rather
than written as a bare `13.4`: the derivation is the proof that the new default
reproduces the old behaviour for 2-min stars, and it cannot silently drift from
it. `test_reproduces_401_at_2min()` asserts that against real measured cadences.

WHAT THIS IS NOT
----------------
Not a training-data change. PART 0 (`cadence_class_confound.py`) FAILED its
gate -- cadence is strongly class-correlated (9.12% of positives vs 20.82% of
negatives are non-2-min; Fisher OR 0.382, p < 1e-6; chi2 p = 4e-76), which is
the same disqualifying pattern that permanently excluded training-side
multi-sector reprocessing, only far stronger. So this pilot asks ONLY whether
the fix is worth deploying to the CANDIDATE path, which carries no class label
and therefore cannot be confounded this way -- exactly the resolution
multi-sector concatenation received.

Nothing here writes to training.csv, the model, or 02_preprocess.py.
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)

OUT_CSV = os.path.join(HERE, "cadence_window_pilot_results.csv")
OUT_JSON = os.path.join(HERE, "cadence_window_pilot_summary.json")
CONFOUND = os.path.join(HERE, "cadence_class_confound.csv")

# Derived, never hard-coded -- see the docstring.
BASELINE_WINDOW_PTS = 401
BASELINE_CADENCE_MIN = 2.0
TARGET_PROTECTED_HOURS = BASELINE_WINDOW_PTS * BASELINE_CADENCE_MIN / 60.0  # 13.3667

GRID_DEPTHS_PPM = [250, 700, 2500]
GRID_PERIODS = [2.0, 6.0]
N_REPEATS = 3
TARGET_BUCKETS = ["20-sec", "10-min", "30-min"]

_G = {}


def cadence_aware_window(n_points, cadence_min, polyorder=2,
                         target_hours=TARGET_PROTECTED_HOURS):
    """Point count spanning `target_hours` at this star's real cadence.

    Same guarantees as production's `choose_savgol_window`: odd, strictly less
    than n_points, None if degenerate. The ONLY change is what the cap is
    derived from.
    """
    if not np.isfinite(cadence_min) or cadence_min <= 0:
        return None
    target_pts = int(round(target_hours * 60.0 / cadence_min))
    window = min(target_pts, n_points - 1)
    if window % 2 == 0:
        window -= 1
    if window < polyorder + 2 or window < 5:
        return None
    return window


def test_reproduces_401_at_2min(cadences=None):
    """REGRESSION GATE: the 82% of stars at 2-min cadence must be untouched.

    Checked against REAL measured cadences (which are 2.0000232... not exactly
    2.0), not against an idealised 2.0, because that is what the pipeline
    actually sees.
    """
    if cadences is None:
        cadences = [2.0]
    bad = []
    for c in cadences:
        w = cadence_aware_window(1_000_000, c)
        if w != BASELINE_WINDOW_PTS:
            bad.append((c, w))
    return len(bad) == 0, bad


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    if _G:
        return _G
    import detrend_gp_pilot as P
    import injection as inj
    _G["P"] = P
    _G["m02"] = P._load("m02", "02_preprocess.py")
    _G["m06"] = P._load("m06", "06_download_unknown.py")
    _G["inj"] = inj
    return _G


def detrend_with_window(t, f, e, window, polyorder):
    from scipy.signal import savgol_filter
    if window is None:
        return None
    trend = savgol_filter(f, window_length=window, polyorder=polyorder, mode="interp")
    ff, ee = f / trend, e / trend
    if not np.all(np.isfinite(ff)):
        return None
    m = np.median(ff)
    return t, ff / m, ee / m


def run_one(args):
    host, raw_dir, bucket, cad, r_star, m_star, period, depth, rep, seed = args
    G = _init() if not _G else _G
    P, m02, m06, inj = G["P"], G["m02"], G["m06"], G["inj"]
    base = {"host": host, "bucket": bucket, "cadence_min": cad,
            "period": period, "depth_ppm": depth, "repeat": rep}
    try:
        pre = P.raw_preamble(os.path.join(raw_dir, host + ".csv"), m02)
        if pre is None:
            return {**base, "status": "no usable flux column"}
        t, f, e, _src = pre
        if len(t) < 500:
            return {**base, "status": f"only {len(t)} raw points"}

        import injection_recovery_sensitivity as IRS
        dur = IRS.transit_duration_days(period, r_star, m_star, 0.3)
        base["duration_h"] = dur * 24.0
        rng = np.random.default_rng(seed)
        f_inj, _params = inj.inject_transit(t, f, period, depth, dur, rng)
        tc, fc, ec = P.clip(t, f_inj, e, m02)
        base["n_points"] = len(tc)

        w_old = m02.choose_savgol_window(len(fc))
        w_new = cadence_aware_window(len(fc), cad)
        base["window_old_pts"] = w_old
        base["window_new_pts"] = w_new
        base["protected_old_h"] = (w_old * cad / 60.0) if w_old else np.nan
        base["protected_new_h"] = (w_new * cad / 60.0) if w_new else np.nan
        # how many CADENCE POINTS the injected transit actually spans -- the
        # thing that decides whether a window can eat it
        base["transit_pts"] = dur * 1440.0 / cad

        for name, w in (("old", w_old), ("new", w_new)):
            out = detrend_with_window(tc, fc, ec, w, m02.SAVGOL_POLYORDER)
            if out is None:
                base[f"{name}_status"] = "detrend failed"
                continue
            ta, fa, ea = out
            res = P.run_tls(ta, fa, ea, r_star, m_star, m06)
            ok, alias = P.recovered(period, res["period"])
            base[f"{name}_status"] = "ok"
            base[f"{name}_recovered"] = bool(ok)
            base[f"{name}_SDE"] = res["SDE"]
            base[f"{name}_snr"] = res["snr"]
            base[f"{name}_period"] = res["period"]
            base[f"{name}_scatter"] = float(1.4826 * np.median(np.abs(fa - np.median(fa))))
        base["status"] = "ok"
    except Exception as ex:
        base["status"] = f"error: {type(ex).__name__}: {ex}"
    return base


def build_jobs():
    c = pd.read_csv(CONFOUND)
    c = c[c.cadence_min.notna()].copy()
    c["bucket"] = pd.cut(c.cadence_min, [0, 1, 2.6, 11, 31, 1e9],
                         labels=["20-sec", "2-min", "10-min", "30-min", ">30min"])
    tr = pd.read_csv(os.path.join(ROOT, "data", "training_dataset", "training.csv"))
    tr["host"] = tr.host.astype(str)
    c = c.merge(tr[["host", "st_rad", "st_mass"]], on="host", how="left")

    dirs = {"neg": os.path.join(ROOT, "data", "known_lightcurves_negative"),
            "pos": os.path.join(ROOT, "data", "known_lightcurves")}
    avail = {}
    for k, d in dirs.items():
        for fn in os.listdir(d):
            if fn.endswith(".csv"):
                avail.setdefault(fn[:-4], d)
    c["raw_dir"] = c.host.map(avail)
    c = c[c.raw_dir.notna() & c.st_rad.notna() & (c.st_rad > 0)]

    rng = np.random.default_rng(20260813)
    jobs, seed = [], 881000
    for b in TARGET_BUCKETS:
        pool = c[c.bucket == b].reset_index(drop=True)
        print(f"  {b:<8} pool {len(pool)} stars")
        for p in GRID_PERIODS:
            for d in GRID_DEPTHS_PPM:
                for r in range(N_REPEATS):
                    row = pool.iloc[int(rng.integers(0, len(pool)))]
                    ms = float(row.st_mass) if np.isfinite(row.st_mass) and row.st_mass > 0 else 1.0
                    jobs.append((str(row.host), row.raw_dir, b, float(row.cadence_min),
                                 float(row.st_rad), ms, p, d, r, seed))
                    seed += 1
    return jobs


def regression_gate():
    """The 2-min no-op check, done EXACTLY rather than statistically."""
    c = pd.read_csv(CONFOUND)
    two = c[(c.cadence_min > 1.0) & (c.cadence_min <= 2.6)].cadence_min.dropna()
    print(f"=== REGRESSION GATE: {len(two)} real 2-min-cadence training stars ===")
    print(f"  measured cadence range {two.min():.7f} .. {two.max():.7f} min")
    ok, bad = test_reproduces_401_at_2min(list(two))
    print(f"  cadence_aware_window() returns exactly {BASELINE_WINDOW_PTS} for all of them: "
          f"{'YES' if ok else 'NO'}")
    if not ok:
        print(f"  *** {len(bad)} MISMATCHES, e.g. {bad[:5]} ***")
    # and the point-count-limited case must still match production exactly
    mism = []
    for n in (60, 101, 402, 1000, 20000):
        for cad in (2.0, 2.0000232433812926):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "m02x", os.path.join(ROOT, "code", "02_preprocess.py"))
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            if m.choose_savgol_window(n) != cadence_aware_window(n, cad):
                mism.append((n, cad, m.choose_savgol_window(n), cadence_aware_window(n, cad)))
    print(f"  identical to production choose_savgol_window() across n in "
          f"(60,101,402,1000,20000) at 2-min: {'YES' if not mism else 'NO ' + str(mism)}")
    return ok and not mism


def main():
    print("=" * 72)
    gate_ok = regression_gate()
    print("=" * 72)
    if not gate_ok:
        raise SystemExit("REGRESSION GATE FAILED -- the fix changes 2-min behaviour. Stop.")

    print("\nbuilding targeted sample (enriched for NON-2-min cadence):")
    jobs = build_jobs()
    print(f"\n{len(jobs)} paired trials x 2 window arms = {2*len(jobs)} TLS searches",
          flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=7, initializer=_init) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            res.append(fu.result())
            if i % 5 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  [{i}/{len(jobs)}] {el/60:.1f} min, eta "
                      f"{el/i*(len(jobs)-i)/60:.1f} min", flush=True)
    pd.DataFrame(res).to_csv(OUT_CSV, index=False)
    print(f"\nwall {(time.time()-t0)/60:.1f} min -> {OUT_CSV}")
    report()


def report():
    from scipy.stats import wilcoxon, binomtest
    d = pd.read_csv(OUT_CSV)
    ok = d[(d.status == "ok") & (d.old_status == "ok") & (d.new_status == "ok")].copy()
    print(f"\ntrials usable in both arms: {len(ok)}/{len(d)}")
    if not len(ok):
        return

    print("\n=== WHAT THE FIX ACTUALLY CHANGES, per cadence bucket ===")
    g = ok.groupby("bucket").agg(
        n=("host", "size"),
        cadence=("cadence_min", "median"),
        old_pts=("window_old_pts", "median"), new_pts=("window_new_pts", "median"),
        old_h=("protected_old_h", "median"), new_h=("protected_new_h", "median"),
        transit_pts=("transit_pts", "median"))
    print(g.round(2).to_string())

    print("\n=== RECOVERY, paired, per bucket ===")
    print(f"{'bucket':<10}{'n':>5}{'old':>8}{'new':>8}{'old-only':>10}{'new-only':>10}{'p':>9}")
    rows = []
    for b, s in ok.groupby("bucket"):
        no, nn = int(s.old_recovered.sum()), int(s.new_recovered.sum())
        bb = int((s.old_recovered & ~s.new_recovered).sum())
        cc = int((~s.old_recovered & s.new_recovered).sum())
        p = binomtest(bb, bb + cc, 0.5).pvalue if (bb + cc) else float("nan")
        print(f"{b:<10}{len(s):>5}{no:>8}{nn:>8}{bb:>10}{cc:>10}{p:>9.4f}")
        rows.append({"bucket": b, "n": len(s), "old": no, "new": nn, "p": p})
    bo, bn = int(ok.old_recovered.sum()), int(ok.new_recovered.sum())
    b_, c_ = (int((ok.old_recovered & ~ok.new_recovered).sum()),
              int((~ok.old_recovered & ok.new_recovered).sum()))
    pall = binomtest(b_, b_ + c_, 0.5).pvalue if (b_ + c_) else float("nan")
    print(f"{'POOLED':<10}{len(ok):>5}{bo:>8}{bn:>8}{b_:>10}{c_:>10}{pall:>9.4f}")

    print("\n=== SDE, paired over ALL trials (the better-powered metric) ===")
    print(f"{'bucket':<10}{'n':>5}{'old':>9}{'new':>9}{'delta':>9}{'Wilcoxon p':>12}")
    sde = {}
    for b, s in list(ok.groupby("bucket")) + [("POOLED", ok)]:
        m = s[["old_SDE", "new_SDE"]].notna().all(axis=1)
        if m.sum() < 5:
            continue
        x, y = s.old_SDE[m], s.new_SDE[m]
        try:
            p = wilcoxon(x, y).pvalue
        except Exception:
            p = float("nan")
        print(f"{b:<10}{int(m.sum()):>5}{x.median():>9.2f}{y.median():>9.2f}"
              f"{(y-x).median():>+9.2f}{p:>12.4f}")
        sde[str(b)] = {"old": float(x.median()), "new": float(y.median()),
                       "delta": float((y - x).median()), "p": float(p)}

    print("\n=== residual scatter (lower = tighter) ===")
    for b, s in ok.groupby("bucket"):
        print(f"  {b:<10} old {s.old_scatter.median():.6f}  new {s.new_scatter.median():.6f}")

    json.dump({"n": int(len(ok)),
               "recovery": {"old": bo, "new": bn, "p": float(pall)},
               "by_bucket": rows, "sde": sde}, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved {OUT_JSON}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    elif len(sys.argv) > 1 and sys.argv[1] == "--gate":
        regression_gate()
    else:
        main()
