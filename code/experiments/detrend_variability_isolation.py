"""detrend_variability_isolation.py -- would a detrending change damage the
deployed variability features?

Two questions, answered separately.

ARCHITECTURAL (code review, below in this docstring)
---------------------------------------------------
Every consumer of the five `var_*` columns reads a RAW, pre-flatten file, and
no consumer reads `data/processed*/`:

  06_download_unknown.add_variability_features(df, raw_dir=None)
      -> raw_dirs = [raw_dir] if raw_dir else [RAW_FOLDER]
      -> RAW_FOLDER = data/unknown_lightcurves{tag}          RAW
  web/retrain_pipeline._variability_for_raw(raw_path, ...)
      -> called with `raw_path`, the download, NOT `processed_path`   RAW
  experiments/eightsector_build_pool.py
      -> add_variability_features(df, raw_dir=RAW_DIR)                RAW

Detrending lives in `02_preprocess.process_one_file`, which READS a raw file
and WRITES `data/processed/`. It never writes back to the raw folder. So a
detrender swapped inside `process_one_file` changes the TLS-input copy ONLY,
and the variability path is untouched **by construction, not by convention**.

The one architecture that WOULD break it -- and the thing to refuse if anyone
proposes it -- is detrending the raw file in place, or feeding a detrended
curve to `variability_for_raw`. Both are the same mistake `add_variability_
features`' docstring already warns about ("Passing a processed file here would
yield the detrending residual and no error").

EMPIRICAL (what this script measures)
-------------------------------------
Confirming a no-op would be worthless -- the pilot writes nothing, so of course
the features do not move. The informative measurement is the counterfactual:
**how much of the variability signal each detrender destroys** if it were ever
(wrongly) applied upstream of the variability computation. That quantifies the
cost of the mistake instead of asserting it.

For each star the same five features are computed three ways:
    raw          what production actually uses
    savgol       raw put through production's current detrender first
    gp_protect   raw put through the GP detrender first
"""
import os
import sys
import warnings
import importlib.util
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..")
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)

RAW_NEG = os.path.join(ROOT, "data", "known_lightcurves_negative")
OUT = os.path.join(HERE, "detrend_variability_isolation.csv")
N_STARS = 30
VAR_COLS = ["var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]


def _load(name, fname, d=CODE):
    spec = importlib.util.spec_from_file_location(name, os.path.join(d, fname))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def main():
    import variability_features as VF
    import detrend_gp_pilot as P
    m02 = _load("m02", "02_preprocess.py")

    hosts = sorted(f[:-4] for f in os.listdir(RAW_NEG) if f.endswith(".csv"))
    rng = np.random.default_rng(20260813)
    hosts = [hosts[i] for i in rng.choice(len(hosts), size=min(N_STARS, len(hosts)),
                                          replace=False)]

    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "var_iso")
    os.makedirs(tmp, exist_ok=True)
    rows = []
    for i, h in enumerate(hosts, 1):
        src = os.path.join(RAW_NEG, h + ".csv")
        base = VF.variability_for_raw(src)
        if base.get("var_status") != "ok":
            continue
        rec = {"host": h}
        for c in VAR_COLS:
            rec[f"raw_{c}"] = base.get(c)

        pre = P.raw_preamble(src, m02)
        if pre is None:
            continue
        t, f, e, _ = pre
        tc, fc, ec = P.clip(t, f, e, m02)
        arms = {"savgol": P.detrend_savgol(tc, fc, ec, m02),
                "gp_protect": P.detrend_gp(tc, fc, ec, P.GP_RHO_FLOOR_PROTECT)}
        for name, out in arms.items():
            if out is None:
                continue
            ta, fa, ea, _meta = out
            # write a file shaped like a RAW download so variability_for_raw
            # reads it the same way -- this is the MISTAKE being quantified
            p = os.path.join(tmp, f"{h}__{name}.csv")
            pd.DataFrame({"time": ta, "flux": fa, "flux_err": ea,
                          "pdcsap_flux": fa, "pdcsap_flux_err": ea,
                          "quality": np.zeros(len(ta), dtype=int)}).to_csv(p, index=False)
            r = VF.variability_for_raw(p)
            for c in VAR_COLS:
                rec[f"{name}_{c}"] = r.get(c) if r.get("var_status") == "ok" else np.nan
            os.remove(p)
        rows.append(rec)
        if i % 10 == 0:
            print(f"  [{i}/{len(hosts)}]", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    print(f"\n{len(d)} stars measured -> {OUT}")

    print("\n=== HOW MUCH VARIABILITY SIGNAL EACH DETRENDER DESTROYS ===")
    print("(ratio of the detrended value to the RAW value production uses;")
    print(" 1.00 = preserved, ->0 = destroyed)")
    print(f"\n{'feature':<16}{'savgol':>22}{'gp_protect':>22}")
    for c in VAR_COLS:
        cells = []
        for a in ("savgol", "gp_protect"):
            if f"{a}_{c}" not in d.columns:
                cells.append("n/a"); continue
            r = pd.to_numeric(d[f"{a}_{c}"], errors="coerce") / pd.to_numeric(
                d[f"raw_{c}"], errors="coerce").replace(0, np.nan)
            r = r.replace([np.inf, -np.inf], np.nan).dropna()
            cells.append(f"{r.median():.4f}  (n={len(r)})" if len(r) else "n/a")
        print(f"{c:<16}{cells[0]:>22}{cells[1]:>22}")

    print("\n=== absolute medians ===")
    print(f"{'feature':<16}{'raw':>14}{'savgol':>14}{'gp_protect':>14}")
    for c in VAR_COLS:
        v = [pd.to_numeric(d.get(f"{p}_{c}"), errors="coerce").median()
             if f"{p}_{c}" in d.columns else np.nan
             for p in ("raw", "savgol", "gp_protect")]
        print(f"{c:<16}{v[0]:>14.6g}{v[1]:>14.6g}{v[2]:>14.6g}")

    print("\nCONCLUSION: these are the values production would get if a detrended")
    print("curve were ever fed to the variability path. Production reads RAW, so")
    print("none of this happens -- but the size of the damage is now on record.")


if __name__ == "__main__":
    main()
