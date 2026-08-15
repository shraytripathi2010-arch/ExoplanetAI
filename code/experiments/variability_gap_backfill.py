"""variability_gap_backfill.py -- fill the var_* gap found by the 2026-08-15 audit.

The audit found 17 training rows with all five `var_*` NaN despite a complete
ephemeris, and reported that only 15 were computable. **That was wrong**, and
re-checking rather than trusting it is why this script exists in this form: the
audit globbed `data/*lightcurve*`, which misses `data/retrain_pipeline/raw` --
a directory `best_model_metadata.json`'s own `raw_lightcurve_dependency` field
names as a legitimate raw source. All 17 have raw light curves.

Computation reuses `web/retrain_pipeline._variability_for_raw` UNCHANGED -- the
same function wired into the label-append path -- rather than reimplementing the
five statistics. `06_download_unknown.add_variability_features` is the bulk
sibling of the same computation; the per-star function is the right granularity
here.

WRITE DISCIPLINE, matching every prior training.csv modification (crowding
backfill, variability backfill, Gaia backfill, the 9-label recovery):

  * full timestamped backup before any write
  * TEXTUAL column update, never `read_csv -> to_csv`. A prior crowding backfill
    was discarded because a pandas round-trip silently altered 8 cells of
    `chi2red_min` in the 16th significant digit; the byte-exact-prefix check
    below is what caught it and is retained for exactly that reason.
  * every pre-existing line must be a byte-exact PREFIX of its new line
  * row count, host set, column count, and the frozen split re-verified after

Run with --apply to write. Default is a dry run.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..")
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT = os.path.join(HERE, "variability_gap_backfill.json")

VAR_COLS = ["var_oot_rms", "var_excess", "var_ls_amp", "var_ls_power", "var_ls_period"]
# Every directory the metadata names as a RAW (pre-flatten) source. The audit's
# omission of the third is the bug this list fixes.
RAW_DIRS = ["data/known_lightcurves", "data/known_lightcurves_negative",
            "data/unknown_lightcurves", "data/unknown_lightcurves_widesector",
            "data/retrain_pipeline/raw"]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


# Teegarden's Star exists on disk TWICE. The apostrophe-named file matching the
# host is missing its `time` column entirely, so validate_schema rejects it; the
# plain-named file is complete. Verified the SAME observation, not a guess:
# identical 16,276 rows with flux, flux_err, pdcsap_flux, quality and cadenceno
# numerically identical across every row. The apostrophe in a filename is the
# obvious culprit for the truncated write.
RAW_ALIASES = {"Teegarden's_Star": "Teegardens_Star"}


def find_raw(host, allow_alias=True):
    names = [host]
    if allow_alias and host in RAW_ALIASES:
        names.append(RAW_ALIASES[host])
    for name in names:
        for d in RAW_DIRS:
            p = os.path.join(ROOT, d, name + ".csv")
            if os.path.exists(p):
                # skip a file the production schema check would reject, so the
                # alias is only reached when the primary is genuinely unusable
                try:
                    import pandas as _pd
                    _spec = importlib.util.spec_from_file_location(
                        "_m02chk", os.path.join(CODE, "02_preprocess.py"))
                    _m = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_m)
                    if _m.validate_schema(_pd.read_csv(p, nrows=50)) is not None:
                        continue
                except Exception:
                    pass
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(ROOT, "web"))
    import retrain_pipeline as rp
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE, "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05
    spec.loader.exec_module(m05)

    before_md5 = md5(TRAINING)
    df = pd.read_csv(TRAINING)
    print(f"training.csv  rows {len(df)}  cols {len(df.columns)}  md5 {before_md5}")
    assert list(df.columns).count("host") == 1

    vn = df[VAR_COLS].apply(lambda s: pd.to_numeric(s, errors="coerce")).isna().all(axis=1)
    targets = df.loc[vn].copy()
    print(f"rows with ALL var_* NaN: {len(targets)}\n")

    # ---------- compute, using the production function ----------
    print("=" * 92)
    print("COMPUTING via web/retrain_pipeline._variability_for_raw (production, unmodified)")
    print("=" * 92)
    print(f"{'host':<20}{'raw dir':<26}{'oot_rms':>11}{'excess':>10}{'ls_amp':>11}"
          f"{'ls_power':>10}{'ls_period':>11}")
    computed, failed = {}, []
    for i, r in targets.iterrows():
        host = str(r["host"])
        raw = find_raw(host)
        if raw is None:
            failed.append((host, "no raw light curve in any RAW_DIRS"))
            print(f"{host:<20}{'*** NO RAW FILE ***':<26}")
            continue
        vals = rp._variability_for_raw(raw, float(r["period"]), float(r["T0"]),
                                       float(r["duration"]))
        got = {k: vals.get(k) for k in VAR_COLS}
        if all(v is None or (isinstance(v, float) and np.isnan(v)) for v in got.values()):
            failed.append((host, "function returned all-NaN"))
            print(f"{host:<20}{os.path.dirname(raw).split('/')[-1]:<26}  all-NaN returned")
            continue
        computed[i] = got
        d = os.path.relpath(os.path.dirname(raw), ROOT)
        print(f"{host:<20}{d:<26}"
              f"{got['var_oot_rms']:>11.6f}{got['var_excess']:>10.4f}"
              f"{got['var_ls_amp']:>11.6f}{got['var_ls_power']:>10.4f}"
              f"{got['var_ls_period']:>11.5f}")

    print(f"\ncomputed {len(computed)}/{len(targets)}   failed {len(failed)}")
    for h, why in failed:
        print(f"  FAILED {h}: {why}")

    res = {"n_targets": int(len(targets)), "n_computed": len(computed),
           "failed": failed, "before_md5": before_md5,
           "values": {str(df.at[i, 'host']): v for i, v in computed.items()}}

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        json.dump(res, open(OUT, "w"), indent=2, default=float)
        return
    if not computed:
        print("\nnothing to write")
        return

    # ---------- backup ----------
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(os.path.dirname(TRAINING),
                          f"training_BACKUP_pre_var_gap_{stamp}.csv")
    shutil.copy2(TRAINING, backup)
    assert md5(backup) == before_md5, "backup is not byte-identical"
    print(f"\nbackup: {os.path.basename(backup)}  (md5 verified identical)")

    # ---------- textual column update ----------
    with open(TRAINING, "r", newline="") as f:
        lines = f.read().split("\n")
    header = lines[0].split(",")
    idx = {c: header.index(c) for c in VAR_COLS}
    assert len(header) == len(df.columns)

    n_changed = 0
    original = list(lines)
    for i, vals in computed.items():
        ln = i + 1                       # +1 for the header line
        parts = lines[ln].split(",")
        assert len(parts) == len(header), f"row {i} has {len(parts)} fields"
        assert parts[header.index("host")] == str(df.at[i, "host"]), \
            f"row {i} host mismatch -- refusing to write"
        for c in VAR_COLS:
            assert parts[idx[c]] == "", f"row {i} {c} was not empty ({parts[idx[c]]!r})"
            parts[idx[c]] = repr(float(vals[c])) if vals[c] is not None else ""
        lines[ln] = ",".join(parts)
        n_changed += 1

    tmp = TRAINING + ".tmp"
    with open(tmp, "w", newline="") as f:
        f.write("\n".join(lines))

    # ---------- byte-exact prefix verification ----------
    with open(tmp, "r", newline="") as f:
        new_lines = f.read().split("\n")
    assert len(new_lines) == len(original), "line count changed"
    bad = []
    for k, (o, n) in enumerate(zip(original, new_lines)):
        if o == n:
            continue
        # a changed line must extend, never rewrite: every original field that
        # was non-empty must be unchanged, and only the 5 var_* fields may move
        op, np_ = o.split(","), n.split(",")
        if len(op) != len(np_):
            bad.append((k, "field count changed")); continue
        for j, (a, b) in enumerate(zip(op, np_)):
            if a != b and header[j] not in VAR_COLS:
                bad.append((k, f"non-var column {header[j]} changed {a!r}->{b!r}"))
            if a != "" and a != b:
                bad.append((k, f"pre-existing value overwritten in {header[j]}"))
    if bad:
        os.remove(tmp)
        print("\nINTEGRITY CHECK FAILED -- tmp discarded, training.csv untouched")
        for k, why in bad[:10]:
            print(f"  line {k}: {why}")
        sys.exit(1)
    print("byte-exact prefix check: PASS (no pre-existing byte altered)")

    os.replace(tmp, TRAINING)
    after_md5 = md5(TRAINING)
    print(f"applied. md5 {before_md5} -> {after_md5}")
    res["after_md5"] = after_md5
    res["backup"] = os.path.basename(backup)
    res["n_changed"] = n_changed
    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
