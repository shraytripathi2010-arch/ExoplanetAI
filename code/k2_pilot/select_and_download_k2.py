"""select_and_download_k2.py -- K2 pilot step 1: pick the sample, fetch raw.

SAMPLE SELECTION IS TIC-GATED, NOT NAME-GATED
Every candidate star is resolved to a TIC id via the archive's own
k2pandc.tic_id, and dropped if that TIC already appears in training.csv.
Selecting by EPIC string would silently re-admit the 405 K2 stars that are
already in the training set under TESS photometry -- the same failure that
produced 144 duplicated stars and 56 split-straddlers.

WHY author='K2' AND NOT EVEREST/K2SFF
Checked live on EPIC 201367065. Only the mission product carries the column
schema 02_preprocess expects:

    K2       time, flux, flux_err, quality, sap_flux, sap_flux_err,
             pdcsap_flux, pdcsap_flux_err          <- matches REQUIRED_COLUMNS
    K2SFF    time, flux, flux_err                  <- no quality, no pdcsap
    EVEREST  time, flux, flux_err, quality         <- no pdcsap

The pipeline's choose_flux_columns() prefers pdcsap_flux, i.e. the mission's
own instrumentally-corrected flux. author='K2' is therefore the choice that
keeps K2 structurally identical to the TESS SPOC path, rather than swapping in
a third party's detrending underneath this project's own.

This writes ONLY into code/k2_pilot/. It does not touch data/processed/,
data/training_dataset/, or anything the production model reads.
"""
import io
import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(PILOT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

RAW_DIR = os.path.join(PILOT_DIR, "raw")
SAMPLE_PATH = os.path.join(PILOT_DIR, "k2_pilot_sample.csv")
DOWNLOAD_LOG = os.path.join(PILOT_DIR, "download_log.csv")
TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

os.makedirs(RAW_DIR, exist_ok=True)

N_PILOT = 80          # 13 CONFIRMED + 6 REFUTED is all that exists; rest are FP
LABEL_MAP = {"CONFIRMED": 1, "FALSE POSITIVE": 0, "REFUTED": 0}


def tap(sql):
    r = requests.get(TAP, params={"query": sql, "format": "csv"}, timeout=600)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def _tic(s):
    return pd.to_numeric(s.astype(str).str.replace("TIC ", "", regex=False),
                         errors="coerce")


def training_tics():
    """TIC set of training.csv -- same resolution path as
    retrain_pipeline._training_tic_ids, plus the EPIC->TIC map from k2pandc."""
    hosts = pd.read_csv(TRAINING_CSV)["host"].astype(str)
    known = set(pd.to_numeric(hosts.str.extract(r"^TIC_(\d+)", expand=False),
                              errors="coerce").dropna().astype("int64"))
    ps = tap("select hostname,tic_id from pscomppars")
    ps["_t"] = _tic(ps["tic_id"])
    k2 = tap("select epic_hostname,hostname,tic_id from k2pandc")
    k2["_t"] = _tic(k2["tic_id"])
    m = {}
    for frame, cols in ((ps, ["hostname"]), (k2, ["epic_hostname", "hostname"])):
        for c in cols:
            for h, t in zip(frame[c].astype(str), frame["_t"]):
                if pd.isna(t) or h in ("nan", ""):
                    continue
                m.setdefault(h, int(t))
                m.setdefault(h.replace(" ", "_"), int(t))
    for h in hosts:
        t = m.get(h)
        if t is not None:
            known.add(t)
    return known


def build_sample():
    if os.path.exists(SAMPLE_PATH):
        s = pd.read_csv(SAMPLE_PATH)
        print(f"Reusing existing sample: {len(s)} stars")
        return s

    known = training_tics()
    print(f"training.csv resolves to {len(known)} distinct TIC ids")

    k2 = tap("select epic_hostname, tic_id, disposition, st_rad, st_mass, "
             "st_teff, default_flag from k2pandc where disposition != 'CANDIDATE'")
    k2["_tic"] = _tic(k2["tic_id"])
    k2 = k2.dropna(subset=["_tic"])
    k2["_tic"] = k2["_tic"].astype("int64")
    # one row per STAR, preferring the default parameter set
    k2 = k2.sort_values("default_flag", ascending=False).drop_duplicates("_tic")
    print(f"k2pandc labelled stars with a TIC: {len(k2)}")

    new = k2[~k2["_tic"].isin(known)].copy()
    print(f"  genuinely new (TIC not in training.csv): {len(new)}")
    print(new["disposition"].value_counts().to_string())

    # Take every CONFIRMED and REFUTED that exists (they are scarce), then top
    # up with FALSE POSITIVEs to the pilot size. Sampling is seeded so the
    # pilot is reproducible.
    rng = np.random.RandomState(42)
    parts = [new[new["disposition"] == d] for d in ("CONFIRMED", "REFUTED")]
    fp = new[new["disposition"] == "FALSE POSITIVE"]
    n_fp = max(0, N_PILOT - sum(len(p) for p in parts))
    if len(fp) > n_fp:
        fp = fp.iloc[rng.permutation(len(fp))[:n_fp]]
    sample = pd.concat(parts + [fp], ignore_index=True)
    sample["label"] = sample["disposition"].map(LABEL_MAP)
    sample["host"] = "EPIC_" + sample["_tic"].astype(str).radd("")  # placeholder
    # host names are keyed on EPIC, which is what lightkurve resolves
    sample["host"] = sample["epic_hostname"].astype(str).str.replace(" ", "_")
    sample = sample[["host", "epic_hostname", "_tic", "disposition", "label",
                     "st_rad", "st_mass", "st_teff"]]
    sample = sample.rename(columns={"_tic": "tic_id"})
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"\nPilot sample: {len(sample)} stars "
          f"({int((sample.label == 1).sum())} pos / "
          f"{int((sample.label == 0).sum())} neg) -> {SAMPLE_PATH}")
    return sample


def download_one(host, epic):
    """Fetch every K2 campaign for one star and concatenate, matching
    02_preprocess's REQUIRED_COLUMNS schema exactly."""
    import lightkurve as lk
    out = os.path.join(RAW_DIR, host + ".csv")
    if os.path.exists(out):
        try:
            if len(pd.read_csv(out)) > 0:
                return len(pd.read_csv(out)), "Already downloaded (resumed)"
        except Exception:
            pass
    try:
        sr = lk.search_lightcurve(str(epic), mission="K2", author="K2")
    except Exception as e:
        return 0, f"Search error: {type(e).__name__}: {e}"
    if len(sr) == 0:
        return 0, "No K2 mission-product light curve found"
    # long cadence only -- 60s short-cadence exists for a minority and mixing
    # two cadences inside one star would reintroduce the confound this whole
    # experiment is about.
    sr = sr[np.isclose(np.asarray(sr.table["exptime"], dtype=float), 1800.0, rtol=0.2)]
    if len(sr) == 0:
        return 0, "No 1800s (long-cadence) product"
    try:
        lcc = sr.download_all()
    except Exception as e:
        return 0, f"Download error: {type(e).__name__}: {e}"
    if lcc is None or len(lcc) == 0:
        return 0, "Download returned no data"

    frames = []
    for lc in lcc:
        try:
            def col(name):
                v = getattr(lc, name, None)
                if v is None:
                    return np.full(len(lc.time.value), np.nan)
                return v.value if hasattr(v, "value") else np.asarray(v)
            frames.append(pd.DataFrame({
                "time": lc.time.value,
                "flux": col("flux"), "flux_err": col("flux_err"),
                "quality": col("quality"),
                "pdcsap_flux": col("pdcsap_flux"),
                "pdcsap_flux_err": col("pdcsap_flux_err"),
            }))
        except Exception as e:
            print(f"  WARNING {host}: one campaign failed to convert: {e}",
                  file=sys.stderr)
    if not frames:
        return 0, "No campaigns converted successfully"
    df = pd.concat(frames, ignore_index=True).sort_values("time")
    df = df.reset_index(drop=True)
    df.to_csv(out, index=False)
    return len(df), "Success"


def main():
    sample = build_sample()
    print("\nDownloading (author='K2', 1800s long cadence)...")
    rows = []
    t0 = time.time()
    for i, r in sample.iterrows():
        n, status = download_one(r["host"], r["epic_hostname"])
        rows.append({"host": r["host"], "tic_id": r["tic_id"],
                     "label": r["label"], "disposition": r["disposition"],
                     "n_points": n, "status": status})
        if (i + 1) % 10 == 0 or status not in ("Success", "Already downloaded (resumed)"):
            el = time.time() - t0
            print(f"  [{i+1}/{len(sample)}] {r['host']:<20} {n:>6} pts  "
                  f"{status}  ({el:.0f}s)", flush=True)
    log = pd.DataFrame(rows)
    log.to_csv(DOWNLOAD_LOG, index=False)
    print("\nDownload status breakdown:")
    print(log["status"].value_counts().to_string())
    ok = log[log["status"].str.startswith(("Success", "Already"))]
    print(f"\n{len(ok)}/{len(log)} usable  "
          f"({int((ok.label == 1).sum())} pos / {int((ok.label == 0).sum())} neg)")
    if len(ok):
        print(f"median points per star: {ok['n_points'].median():.0f}")


if __name__ == "__main__":
    main()
