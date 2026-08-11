"""
injection_recovery_sensitivity.py -- END-TO-END sensitivity characterization of
the DEPLOYED pipeline (Proposal 3).

This is a VALIDATION / CHARACTERIZATION tool, not a training experiment. It
touches no training data, no split, no model artifact. It answers one question:

    "If a real transit of a given depth and period existed in this pipeline's
     input data, would TLS detect it AND would the 31-feature classifier
     score it as planet-like?"

HOW THIS DIFFERS FROM completeness_curve.py (the existing script)
----------------------------------------------------------------
completeness_curve.py measures DETECTION ONLY -- it stops at "did TLS recover
the period". This runs the full deployed chain: injection -> production TLS
invocation -> 31-feature vector -> deployed model -> score. It reports
detection and classification as SEPARATE stages, because a signal can be
detected and then thrown away by the classifier, and those are different
failure modes with different fixes.

Two further differences, both deliberate:

  1. DURATION IS DERIVED FROM PHYSICS, not fixed at 5% of the period.
     completeness_curve.py uses FIXED_DURATION_FRACTION = 0.05, which at
     P = 16 d implies a 19-hour transit -- physically absurd and it makes long
     periods look artificially easy to detect (more in-transit points than a
     real long-period planet would ever give). Here a/R* comes from Kepler's
     third law using the host's REAL M*/R*, and the duration follows.

  2. THERE IS A ZERO-DEPTH CONTROL ARM. Without it a "70% classified planet-
     like" number is uninterpretable, because it could just mean the model
     scores everything high. The control injects nothing and scores the real
     light curve as-is, giving the false-positive rate these real negative
     hosts produce on their own.

WHY THE SYNTHETIC-FEATURE OBJECTION DOES NOT APPLY HERE
-------------------------------------------------------
The two closed training-augmentation investigations died partly because 7 of
31 production features cannot be given honest synthetic values for a
FABRICATED star. That objection is specific to training. Here the host is a
REAL TESS star, so the 31 features split cleanly:

  * 22 TLS-derived features are recomputed from the injected curve, by the
    same TLS call production uses.
  * 9 host-derived features (st_rad, st_teff, crowd_flux_ratio_max,
    crowd_nearest_arcsec, var_oot_rms, var_excess, var_ls_amp, var_ls_power,
    var_ls_period) are REAL measured properties of that real star, read from
    training.csv. Nothing is invented.

WHAT THIS CAN AND CANNOT TELL YOU
---------------------------------
CAN: the deployed system's detection/classification sensitivity as a function
of transit depth and period, in genuine TESS noise on genuine stars.

CANNOT: that a real Earth-size planet would be recovered at this rate. An
injected batman transit is cleaner than a real one -- no TTVs, no starspot
crossings, no correlated residual from the real planet's own host activity.
The domain-separability results (0.95-0.97 on shape AND detection features
independently) are direct evidence that injected and real transits remain
distinguishable. So these are BEST-CASE sensitivity numbers: the real system
does no better than this, and plausibly somewhat worse.

Hosts are drawn from FROZEN TEST-SPLIT negatives only, so the classifier never
saw them in training.
"""
import os
import sys
import json
import time
import importlib.util
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(HERE, "..")
PROJECT_ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(PROJECT_ROOT, "data", "training_dataset", "training.csv")
SPLIT = os.path.join(PROJECT_ROOT, "data", "training_dataset", "split_manifest.json")
MODEL = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
OUT_CSV = os.path.join(HERE, "injection_recovery_sensitivity_results.csv")
OUT_JSON = os.path.join(HERE, "injection_recovery_sensitivity_summary.json")

sys.path.insert(0, HERE)

# Depth grid, ppm. Concentrated where the prior detection-only run showed the
# falloff (200 ppm -> 5%, 500 -> 40%, 1000 -> 50%). 84 ppm is an Earth across a
# 1 Rsun star, the regime the brief specifically asks about.
GRID_DEPTHS_PPM = [84, 150, 250, 400, 700, 1200, 2500, 5000]
# Period grid, days. Host baselines are single-sector TESS (median 24.9 d), so
# P = 20 d yields ~1 transit and is included precisely to show the wall.
GRID_PERIODS = [1.0, 3.0, 6.0, 10.0, 14.0, 20.0]
N_REPEATS = 10
N_CONTROL = 60          # zero-depth control trials
TRIAGE_FLOOR = 0.30     # production triage floor (06_download_unknown.py)

HOST_FEATURES = ["st_rad", "st_teff", "crowd_flux_ratio_max", "crowd_nearest_arcsec",
                 "var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]

PERIOD_MATCH_TOLERANCE = 0.01

# None = use production's bin_lightcurve. 1 = no binning (native cadence).
BIN_FACTOR_OVERRIDE = None


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CODE_DIR, fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _period_recovered(injected, recovered, tol=PERIOD_MATCH_TOLERANCE):
    """Aliases counted as recovery but reported separately -- same convention
    as completeness_curve.py and Hippke & Heller's own TLS validation."""
    if recovered is None or not np.isfinite(recovered) or recovered <= 0:
        return False, None
    for n, name in [(1, "exact"), (2, "double"), (0.5, "half"), (3, "triple"), (1 / 3, "third")]:
        target = injected * n
        if abs(recovered - target) / target < tol:
            return True, name
    return False, None


def transit_duration_days(period_days, r_star, m_star, impact_param):
    """Central-transit duration from Kepler's third law using the host's REAL
    stellar parameters. a/R* = 4.2084 * P^(2/3) * M*^(1/3) / R*  (solar units,
    P in days). Duration = P/pi * sqrt(1-b^2) / (a/R*)."""
    a_rs = 4.2084 * (period_days ** (2.0 / 3.0)) * (m_star ** (1.0 / 3.0)) / r_star
    a_rs = float(np.clip(a_rs, 2.0, 1000.0))
    b2 = min(impact_param ** 2, 0.95)
    return float(period_days / np.pi * np.sqrt(1.0 - b2) / a_rs)


_G = {}


def _init():
    """Per-worker setup: load host table and the deployed model once."""
    os.environ["OMP_NUM_THREADS"] = "1"
    import joblib
    import injection as inj

    m05 = _load("m05", "05_train_models.py")
    m06 = _load("m06", "06_download_unknown.py")
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 31, f"expected 31 features, got {len(cols)}"

    split = json.load(open(SPLIT))
    test_hosts = set(map(str, split["test_hosts"]))
    df = pd.read_csv(TRAINING)
    neg = df[df["label"] == 0].copy()
    neg["host"] = neg["host"].astype(str)
    avail = {f[:-4] for f in inj.list_real_negative_lightcurves()}
    pool = neg[neg["host"].isin(test_hosts) & neg["host"].isin(avail)]
    pool = pool[pool[HOST_FEATURES].notna().all(axis=1)].reset_index(drop=True)

    _G["cols"] = cols
    _G["m06"] = m06
    _G["inj"] = inj
    _G["pool"] = pool
    _G["lc_dir"] = inj.PROCESSED_NEGATIVE_DIR
    _G["model"] = joblib.load(MODEL)
    return _G


def _load_curve(G, host):
    """Reads a processed light curve from whichever directory this run's pool
    draws from. Kept separate from injection.load_real_lightcurve so the
    longer-baseline variant can point at a different pool without copying
    any of the injection/TLS/scoring logic."""
    d = pd.read_csv(os.path.join(G["lc_dir"], host + ".csv"))
    return d["time"].to_numpy(), d["flux"].to_numpy(), d["flux_err"].to_numpy()


def _tls_features(t_arr, f_arr, e_arr, r_star, m_star, m06):
    """Runs TLS exactly as production does (same bounds, same binning, same
    post-processing) and returns the 22 TLS-derived features."""
    from transitleastsquares import transitleastsquares

    # BIN_FACTOR_OVERRIDE=1 skips production's binning entirely, keeping native
    # 2-min cadence. Production's bin_lightcurve targets a FIXED 15,000 points,
    # so effective cadence degrades linearly with baseline (2 min at 1 sector,
    # 8 min at 3, ~28 min at 13). That couples cadence to baseline and would
    # otherwise confound any baseline comparison. Default None = production.
    if BIN_FACTOR_OVERRIDE is None:
        t_b, f_b, e_b = m06.bin_lightcurve(t_arr, f_arr, e_arr)
    elif BIN_FACTOR_OVERRIDE == 1:
        t_b, f_b, e_b = t_arr, f_arr, e_arr
    else:
        k = int(BIN_FACTOR_OVERRIDE)
        nb = len(t_arr) // k
        t_b = t_arr[:nb * k].reshape(nb, k).mean(axis=1)
        f_b = f_arr[:nb * k].reshape(nb, k).mean(axis=1)
        e_b = np.sqrt((e_arr[:nb * k].reshape(nb, k) ** 2).sum(axis=1)) / k
    r_star = r_star if (r_star and np.isfinite(r_star) and r_star > 0) else 1.0
    m_star = m_star if (m_star and np.isfinite(m_star) and m_star > 0) else 1.0
    model = transitleastsquares(t_b, f_b, e_b)
    r = model.power(
        use_threads=1, oversampling_factor=1, duration_grid_step=1.1, show_progress_bar=False,
        R_star=r_star, R_star_min=min(0.13, r_star * 0.5), R_star_max=max(3.5, r_star * 1.5),
        M_star=m_star, M_star_min=min(0.1, m_star * 0.5), M_star_max=max(1.0, m_star * 1.5),
    )

    # TLS can return scalars here when it fits no transit at all ("No transit
    # were fit"), which happens on badly-normalised input. Coerce to arrays so
    # a no-fit produces NaN features rather than an AttributeError.
    phase = np.atleast_1d(np.asarray(r.folded_phase, dtype=float))
    flux = np.atleast_1d(np.asarray(r.folded_y, dtype=float))
    if phase.size < 10 or phase.size != flux.size:
        phase = np.array([np.nan]); flux = np.array([np.nan])
    sec_mask = (phase > 0.45) & (phase < 0.55)
    secondary_depth = float(1.0 - np.median(flux[sec_mask])) if sec_mask.sum() > 5 else np.nan

    primary_mask = (phase < 0.02) | (phase > 0.98)
    shape_ratio = np.nan
    if primary_mask.sum() > 5:
        itp = np.where(phase > 0.5, phase - 1, phase)[primary_mask]
        cm = np.abs(itp) < 0.005
        em = (np.abs(itp) >= 0.005) & (np.abs(itp) < 0.015)
        cd = float(1.0 - np.median(flux[primary_mask][cm])) if cm.sum() > 2 else np.nan
        ed = float(1.0 - np.median(flux[primary_mask][em])) if em.sum() > 2 else np.nan
        shape_ratio = ed / cd if (cd and np.isfinite(cd) and cd > 0) else np.nan

    depths = np.asarray(r.transit_depths, dtype=float)
    depths = depths[~np.isnan(depths)]

    feats = {
        "SDE": float(r.SDE), "SDE_raw": float(r.SDE_raw), "FAP": float(r.FAP),
        "period": float(r.period), "period_uncertainty": float(r.period_uncertainty),
        "duration": float(r.duration), "depth": float(r.depth),
        "depth_mean": float(r.depth_mean[0]), "depth_mean_std": float(r.depth_mean[1]),
        "depth_mean_even": float(r.depth_mean_even[0]), "depth_mean_odd": float(r.depth_mean_odd[0]),
        "odd_even_mismatch": float(r.odd_even_mismatch), "rp_rs": float(r.rp_rs), "snr": float(r.snr),
        "transit_count": float(r.transit_count), "distinct_transit_count": float(r.distinct_transit_count),
        "empty_transit_count": float(r.empty_transit_count),
        "chi2red_min": float(r.chi2red_min),
        "depth_consistency_std": float(np.std(depths)) if len(depths) > 1 else np.nan,
        "secondary_eclipse_depth": secondary_depth, "transit_shape_ratio": shape_ratio,
    }
    feats["depth_duration_ratio"] = feats["depth"] / feats["duration"] if feats["duration"] else np.nan

    # TLS's own searched period grid. Recorded because its DEFAULT period_max is
    # ~half the baseline (it requires >=2 transits), which for single-sector
    # TESS (~25 d) means the pipeline is structurally blind above ~12.5 d no
    # matter how deep the transit is. Measured per trial rather than assumed.
    try:
        grid = np.asarray(r.periods, dtype=float)
        feats["_tls_period_min"] = float(grid.min())
        feats["_tls_period_max"] = float(grid.max())
    except Exception:
        feats["_tls_period_min"] = feats["_tls_period_max"] = np.nan
    return feats


def run_one(args):
    period, depth_ppm, rep, seed = args
    G = _G if _G else _init()
    rng = np.random.default_rng(seed)
    pool, inj, m06 = G["pool"], G["inj"], G["m06"]

    row = pool.iloc[int(rng.integers(0, len(pool)))]
    host = str(row["host"])
    r_star = float(row["st_rad"]) if np.isfinite(row["st_rad"]) and row["st_rad"] > 0 else 1.0
    m_star = float(row.get("m_star_used", row.get("st_mass", np.nan)))
    if not (np.isfinite(m_star) and m_star > 0):
        m_star = 1.0

    t_arr, f_arr, e_arr = _load_curve(G, host)
    baseline = float(t_arr.max() - t_arr.min())

    out = {
        "host": host, "injected_period": period, "injected_depth_ppm": depth_ppm,
        "repeat": rep, "baseline_days": baseline, "r_star": r_star, "m_star": m_star,
        "is_control": depth_ppm == 0,
    }

    t0 = time.monotonic()
    try:
        if depth_ppm == 0:
            inj_flux = f_arr                      # control: nothing injected
            out["n_transits_in_baseline"] = np.nan
            out["injected_duration_hours"] = np.nan
            out["rp_earth_implied"] = 0.0
        else:
            b = float(rng.uniform(0.0, 0.6))
            dur = transit_duration_days(period, r_star, m_star, b)
            out["injected_duration_hours"] = dur * 24.0
            out["n_transits_in_baseline"] = baseline / period
            out["rp_earth_implied"] = float(np.sqrt(depth_ppm / 1e6) * r_star * 109.076)
            inj_flux, _ = inj.inject_transit(
                t_arr, f_arr, period, depth_ppm, dur, rng, impact_param=b)

        feats = _tls_features(t_arr, inj_flux, e_arr, r_star, m_star, m06)
        out["tls_period_min"] = feats.pop("_tls_period_min")
        out["tls_period_max"] = feats.pop("_tls_period_max")
        # Was the injected period even inside the grid TLS searched? Separates
        # "too shallow to find" from "structurally unsearchable".
        out["period_in_search_range"] = bool(
            np.isfinite(out["tls_period_max"]) and period <= out["tls_period_max"])

        rec, alias = _period_recovered(period, feats["period"]) if depth_ppm else (False, None)
        out["recovered_period"] = feats["period"]
        out["recovered_sde"] = feats["SDE"]
        out["recovered_snr"] = feats["snr"]
        out["detected"] = bool(rec)
        out["alias"] = alias

        # 22 TLS features from the injected curve + 9 REAL host features
        vec = dict(feats)
        for c in HOST_FEATURES:
            vec[c] = float(row[c])
        X = pd.DataFrame([[vec.get(c, np.nan) for c in G["cols"]]], columns=G["cols"])
        X = X.replace([np.inf, -np.inf], np.nan)
        out["score"] = float(G["model"].predict_proba(X)[0, 1])
        out["classified_planet"] = bool(out["score"] >= TRIAGE_FLOOR)
        out["status"] = "ok"
    except Exception as e:
        out.update({"status": f"error: {type(e).__name__}: {e}", "detected": False,
                    "score": np.nan, "classified_planet": False})
    out["elapsed_s"] = time.monotonic() - t0
    return out


def main():
    jobs, seed = [], 20260810
    for p in GRID_PERIODS:
        for d in GRID_DEPTHS_PPM:
            for r in range(N_REPEATS):
                jobs.append((p, d, r, seed)); seed += 1
    for r in range(N_CONTROL):
        jobs.append((1.0, 0, r, seed)); seed += 1   # period ignored for controls

    print(f"{len(jobs)} trials ({len(GRID_PERIODS)}x{len(GRID_DEPTHS_PPM)}x{N_REPEATS} "
          f"+ {N_CONTROL} zero-depth controls), 7 workers", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=7, initializer=_init) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  [{i}/{len(jobs)}] {el/60:.1f} min elapsed, "
                      f"eta {el/i*(len(jobs)-i)/60:.1f} min", flush=True)
            if i % 100 == 0:
                pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwall time {(time.time()-t0)/60:.1f} min -> {OUT_CSV}")
    summarize(df)


def summarize(df):
    ok = df[df["status"] == "ok"]
    inj_df = ok[~ok["is_control"]]
    ctl = ok[ok["is_control"]]
    err = (df["status"] != "ok").sum()

    print(f"\ntrials ok {len(ok)}/{len(df)}  (errors {err})")

    print("\n=== ZERO-DEPTH CONTROL (real negatives, nothing injected) ===")
    if len(ctl):
        print(f"  n={len(ctl)}  scored >= {TRIAGE_FLOOR}: {ctl['classified_planet'].mean()*100:.1f}%"
              f"   median score {ctl['score'].median():.4f}")

    print("\n=== STAGE 1: TLS DETECTION rate by depth x period ===")
    print(pd.crosstab(inj_df["injected_depth_ppm"], inj_df["injected_period"],
                      values=inj_df["detected"], aggfunc="mean").round(2).to_string())

    print("\n=== STAGE 2: CLASSIFIED planet-like (score >= 0.30) | GIVEN DETECTED ===")
    det = inj_df[inj_df["detected"]]
    if len(det):
        print(pd.crosstab(det["injected_depth_ppm"], det["injected_period"],
                          values=det["classified_planet"], aggfunc="mean").round(2).to_string())

    print("\n=== END-TO-END: detected AND classified ===")
    inj_df = inj_df.copy()
    inj_df["e2e"] = inj_df["detected"] & inj_df["classified_planet"]
    print(pd.crosstab(inj_df["injected_depth_ppm"], inj_df["injected_period"],
                      values=inj_df["e2e"], aggfunc="mean").round(2).to_string())

    print("\n=== MARGINALS by depth ===")
    g = inj_df.groupby("injected_depth_ppm").agg(
        n=("detected", "size"), detected=("detected", "mean"),
        end_to_end=("e2e", "mean"), median_sde=("recovered_sde", "median"),
        median_rp_earth=("rp_earth_implied", "median"))
    print(g.round(3).to_string())

    print("\n=== MARGINALS by period ===")
    g2 = inj_df.groupby("injected_period").agg(
        n=("detected", "size"), detected=("detected", "mean"),
        end_to_end=("e2e", "mean"), median_transits=("n_transits_in_baseline", "median"),
        in_tls_range=("period_in_search_range", "mean"),
        median_tls_period_max=("tls_period_max", "median"))
    print(g2.round(3).to_string())
    print("\n  in_tls_range = fraction of trials where the injected period was inside")
    print("  the grid TLS actually searched. Below 1.0 the signal is unsearchable,")
    print("  not merely too shallow -- a structural limit, not a sensitivity one.")

    print("\n=== alias breakdown among detected ===")
    print(det["alias"].value_counts().to_string() if len(det) else "(none)")

    json.dump({
        "n_trials": int(len(df)), "n_ok": int(len(ok)), "n_errors": int(err),
        "triage_floor": TRIAGE_FLOOR,
        "control_n": int(len(ctl)),
        "control_false_positive_rate": float(ctl["classified_planet"].mean()) if len(ctl) else None,
        "control_median_score": float(ctl["score"].median()) if len(ctl) else None,
        "detection_by_depth": inj_df.groupby("injected_depth_ppm")["detected"].mean().round(4).to_dict(),
        "detection_by_period": inj_df.groupby("injected_period")["detected"].mean().round(4).to_dict(),
        "e2e_by_depth": inj_df.groupby("injected_depth_ppm")["e2e"].mean().round(4).to_dict(),
        "e2e_by_period": inj_df.groupby("injected_period")["e2e"].mean().round(4).to_dict(),
        "median_rp_earth_by_depth": inj_df.groupby("injected_depth_ppm")["rp_earth_implied"].median().round(3).to_dict(),
    }, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved {OUT_JSON}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        summarize(pd.read_csv(OUT_CSV))
    else:
        main()
