"""
download_kepler_pilot.py -- Kepler cross-mission expansion, PILOT ONLY
(explicitly scoped this way per user go-ahead: small pilot, not a full
build, given three consecutive "more training data" experiments this
project has already run came back negative or worse).

Samples a real, reproducible pilot batch from the live NASA Exoplanet
Archive KOI cumulative table: N_PER_CLASS CONFIRMED planets (label=1) AND
N_PER_CLASS FALSE POSITIVE dispositions (label=0) -- both classes
deliberately included (not just false positives into the negative class),
per the explicit requirement that mixing Kepler data into only one class
is a likely source of mission-fingerprint leakage by construction.

For each star: downloads and stitches ALL available long-cadence quarters
via lightkurve (a single Kepler quarter is far too short a baseline --
~473 points/90 days at this measured cadence, vs TESS's ~18,600 points/27
days), writes out a CSV matching the EXACT schema 02_preprocess.py already
expects (time, flux, flux_err, quality, pdcsap_flux, pdcsap_flux_err) --
confirmed live that Kepler's quality-flag convention (0 = good) and
pdcsap_flux availability match TESS's, so process_one_file() is reused
UNCHANGED, not reimplemented.

Resumable/checkpointed, matching 06_download_unknown.py's established
pattern -- this environment has shown crashes unrelated to the computation
itself before.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PILOT_DIR, "raw")
KOI_SAMPLE_PATH = os.path.join(PILOT_DIR, "koi_pilot_sample.csv")
DOWNLOAD_LOG_PATH = os.path.join(PILOT_DIR, "download_log.csv")

N_PER_CLASS = 75  # within the user-approved 50-100 pilot range
RANDOM_SEED = 42


def fetch_koi_sample():
    """Live query against the real KOI cumulative table -- not resampled from
    any earlier estimate in this project's history. Cached to disk so re-runs
    don't re-query the archive."""
    if os.path.exists(KOI_SAMPLE_PATH):
        print(f"Reusing existing pilot sample at {KOI_SAMPLE_PATH}")
        return pd.read_csv(KOI_SAMPLE_PATH)

    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    tbl = NasaExoplanetArchive.query_criteria(
        table="cumulative",
        select="kepid,kepoi_name,koi_disposition,koi_period,koi_prad,koi_srad,koi_steff,koi_smass",
    )
    df = tbl.to_pandas()
    print(f"Live KOI cumulative table: {len(df)} rows, "
          f"{(df['koi_disposition']=='CONFIRMED').sum()} CONFIRMED, "
          f"{(df['koi_disposition']=='FALSE POSITIVE').sum()} FALSE POSITIVE")

    rng = np.random.default_rng(RANDOM_SEED)
    confirmed = df[df["koi_disposition"] == "CONFIRMED"].drop_duplicates("kepid")
    false_pos = df[df["koi_disposition"] == "FALSE POSITIVE"].drop_duplicates("kepid")

    conf_sample = confirmed.sample(n=min(N_PER_CLASS, len(confirmed)), random_state=RANDOM_SEED)
    fp_sample = false_pos.sample(n=min(N_PER_CLASS, len(false_pos)), random_state=RANDOM_SEED)

    sample = pd.concat([conf_sample, fp_sample], ignore_index=True)
    sample["label"] = (sample["koi_disposition"] == "CONFIRMED").astype(int)
    sample.to_csv(KOI_SAMPLE_PATH, index=False)
    print(f"Sampled {len(conf_sample)} confirmed + {len(fp_sample)} false-positive stars, "
          f"saved to {KOI_SAMPLE_PATH}")
    return sample


def download_one_star(kepid, label):
    """Downloads + stitches every available long-cadence quarter for one
    star. Returns (n_points, status)."""
    import lightkurve as lk

    out_path = os.path.join(RAW_DIR, f"KIC_{kepid}.csv")
    if os.path.exists(out_path):
        try:
            existing = pd.read_csv(out_path)
            if len(existing) > 0:
                return len(existing), "Already downloaded (resumed)"
        except Exception:
            pass

    try:
        sr = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", cadence="long", author="Kepler")
    except Exception as e:
        return 0, f"Search error: {e}"
    if len(sr) == 0:
        return 0, "No long-cadence light curve products found"

    try:
        lcc = sr.download_all()
    except Exception as e:
        return 0, f"Download error: {e}"
    if lcc is None or len(lcc) == 0:
        return 0, "Download returned no data"

    frames = []
    for lc in lcc:
        try:
            frames.append(pd.DataFrame({
                "time": lc.time.value,
                "flux": lc.flux.value if hasattr(lc.flux, "value") else np.asarray(lc.flux),
                "flux_err": lc.flux_err.value if hasattr(lc.flux_err, "value") else np.asarray(lc.flux_err),
                "quality": np.asarray(lc.quality.value if hasattr(lc.quality, "value") else lc.quality),
                "pdcsap_flux": lc.pdcsap_flux.value if hasattr(lc.pdcsap_flux, "value") else np.asarray(lc.pdcsap_flux),
                "pdcsap_flux_err": lc.pdcsap_flux_err.value if hasattr(lc.pdcsap_flux_err, "value") else np.asarray(lc.pdcsap_flux_err),
            }))
        except Exception as e:
            print(f"  WARNING: one quarter for KIC {kepid} failed to convert: {e}", file=sys.stderr)
            continue

    if not frames:
        return 0, "No quarters converted successfully"

    combined = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    combined.to_csv(out_path, index=False)
    return len(combined), "Success"


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    sample = fetch_koi_sample()

    log_rows = []
    if os.path.exists(DOWNLOAD_LOG_PATH):
        log_rows = pd.read_csv(DOWNLOAD_LOG_PATH).to_dict("records")
    already_logged = {r["kepid"] for r in log_rows}

    t0 = time.time()
    n_todo = len(sample[~sample["kepid"].isin(already_logged)])
    print(f"{len(already_logged)} already logged, {n_todo} to download...")

    for i, row in sample.iterrows():
        kepid = int(row["kepid"])
        if kepid in already_logged:
            continue
        n_points, status = download_one_star(kepid, row["label"])
        log_rows.append({"kepid": kepid, "label": int(row["label"]),
                          "koi_disposition": row["koi_disposition"],
                          "n_points": n_points, "status": status})
        pd.DataFrame(log_rows).to_csv(DOWNLOAD_LOG_PATH, index=False)
        print(f"  [{len(log_rows)}/{len(sample)}] KIC {kepid} (label={row['label']}): "
              f"{status} ({n_points} points, {time.time()-t0:.0f}s elapsed)", flush=True)

    df = pd.DataFrame(log_rows)
    n_ok = (df["status"].isin(["Success", "Already downloaded (resumed)"])).sum()
    print(f"\n{n_ok}/{len(df)} stars downloaded successfully.")
    print(df.groupby("label")["status"].apply(lambda s: (s.isin(["Success", "Already downloaded (resumed)"])).sum()))


if __name__ == "__main__":
    main()
