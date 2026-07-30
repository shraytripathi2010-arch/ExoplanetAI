# ExoplanetAI

An independent, reproducible pipeline that searches TESS light curves for
previously-unflagged transiting-planet candidates, scores them with a trained
classifier, and cross-checks each one against public archives — plus a local
web app for reviewing the results.

**Nothing here confirms a planet.** The pipeline produces *candidates worth
human follow-up*. Only spectroscopic/telescope follow-up by qualified
researchers can confirm a planet, and every output is worded to keep that
distinction. The strongest claim any candidate carries is "high confidence" in
the sense of "worth a real look."

Built entirely from public data (MAST, NASA Exoplanet Archive, ExoFOP, Gaia via
Vizier) and open-source tools. No institutional infrastructure, no proprietary
data, no paid API required.

---

## Requirements

- **Python 3.11+** (developed on 3.11)
- ~2 GB free disk for a small run; the full 5,500-star dataset needs ~70 GB of
  raw light curves, which is why they are not in the repo
- An internet connection — every stage queries a public archive live

```bash
git clone https://github.com/shraytripathi2010-arch/ExoplanetAI.git
cd ExoplanetAI
python3 -m venv .venv && source .venv/bin/activate   # recommended
python3 -m pip install -r requirements.txt
```

`transitleastsquares` compiles native code on install; if it fails, ensure you
have a C compiler (`xcode-select --install` on macOS, `build-essential` on
Debian/Ubuntu).

---

## Quickest thing that proves it works

The repo ships the trained model and the candidate results, so you can inspect
everything without downloading a single light curve:

```bash
python3 web/app.py
```

Then open <http://127.0.0.1:5050>. Set `PORT` to use a different port.

To verify the *pipeline mechanics* end to end on a handful of real stars
(downloads a few light curves from MAST, runs the transit search, extracts
features, scores them):

```bash
cd code
python3 06_download_unknown.py --sample-size 10 --top-n 5
```

That is the single command that exercises download → preprocess → transit
search → feature extraction → classifier scoring. Expect a few minutes,
mostly network and the transit search.

---

## Environment variables

Both are optional; nothing requires a credential to run.

| Variable | Default | What it does |
|---|---|---|
| `ADS_API_KEY` | unset | Enables the NASA ADS literature check in `08_characterize_candidates.py`. Without it the pipeline silently falls back to arXiv and says so in the UI. Free key: <https://ui.adsabs.harvard.edu/user/settings/token> |
| `PORT` | `5050` | Port for the web app |
| `EXOPLANETAI_DOWNLOAD_TIMEOUT` | `45` | Per-star download ceiling in seconds. **Raise this on your first run** — see below. |

### If your first run reports "Downloaded: 0, Failed: N"

Expected on a cold start, and not a bug in your setup. The 45-second per-star
default assumes a warm `lightkurve` cache; a first run has an empty one, and
each light curve is a few MB fetched live from MAST. In a clean-clone test all
10 stars in the quick-start above timed out at 45s — while one of them actually
finished downloading moments after being given up on.

```bash
EXOPLANETAI_DOWNLOAD_TIMEOUT=300 python3 06_download_unknown.py --sample-size 10 --top-n 5
```

Subsequent runs reuse the cache and are much faster, so the default is fine
once you are past the first pull.

---

## Full pipeline order

Each stage writes to `data/` or `results/` and is resumable — re-running skips
work already on disk. Run from inside `code/`.

**Training the classifier from scratch** (only needed if you want to rebuild
the model; a trained one is already in `models/`):

| Stage | Script | What it does |
|---|---|---|
| 1 | `01_download_known.py` | Confirmed-planet host light curves (positive class) |
| 1b | `01_download_negative.py` | TOI false-positive light curves (negative class) |
| 2 | `02_preprocess.py`, `02_preprocess_negative.py` | Detrend, normalize, sigma-clip |
| 3 | `03_transit_search.py`, `03_transit_search_negative.py` | Transit least-squares search |
| 4 | `04_build_training_dataset.py` | Assemble the labelled feature table |
| 5 | `05_train_models.py` | Train + evaluate; writes `models/best_model.joblib` |
| 5b | `05b_model_analysis.py` | Nested CV, calibration, thresholds, bootstrap CIs |

**Searching for new candidates** (uses the already-trained model):

| Stage | Script | What it does |
|---|---|---|
| 6 | `06_download_unknown.py` | Download → preprocess → TLS → features → score, for unexamined stars |
| 7 | `07_search_unknown.py` | Gaia/SIMBAD stellar verification |
| 8 | `08_characterize_candidates.py` | Physical parameters + archive/ExoFOP/arXiv/VSX/blend/RV checks |

**Optional extras:**

| Script | What it does |
|---|---|
| `09_build_bootstrap_ensemble.py` | Builds the 32-member bootstrap ensemble used for per-candidate uncertainty bands. ~15 min, writes ~140 MB to `models/bootstrap_ensemble/` (gitignored — regenerate rather than download) |
| `09b_validate_uncertainty.py` | Sanity-checks those uncertainty bands |
| `web/backfill_uncertainty.py` | Applies uncertainty bands to candidates already in the database |

---

## The train/test split is frozen on purpose

`data/training_dataset/split_manifest.json` records the exact star IDs of the
4,392/1,099 train/test split the deployed model was trained and evaluated on.
**Do not delete or regenerate it.** `05_train_models.split_by_host()` reads it,
and every script that evaluates a model (`05b`, `06`, `web/retrain_pipeline.py`)
goes through that function.

It exists because the split used to be positional — `train_test_split` on row
indices with a fixed seed — which silently reshuffled whenever rows were
appended to `training.csv`. That let 89 stars the model had been *trained* on
drift into the *test* set, inflating measured ROC-AUC from the true 0.9032 to
0.9113. Keying membership to stable star IDs removes that failure mode. Stars
added after the manifest was frozen are assigned by a deterministic hash of the
host name, so they too stay put.

---

## What the numbers are

| | |
|---|---|
| Model | HistGradientBoosting + sigmoid calibration |
| Test ROC-AUC | **0.9032** (1,099 held-out stars) |
| Precision / Recall | 0.8972 / 0.9654 at the 0.5 operating point |
| Training set | 5,506 stars (4,351 confirmed-planet hosts, 1,155 TOI false positives) |

Reproduce that number:

```bash
cd code && python3 05b_model_analysis.py
```

**Scope this figure honestly.** It describes performance on stars that clear
the pipeline's own data requirements — SPOC or FFI photometry available, and
enough transits in the observing window to compute the feature set. A blind
temporal-holdout test on 10 confirmed planets discovered *after* the training
data was assembled scored 8 of 9 correctly (89%), with 1 of 10 unable to be
processed at all. See `code/experiments/RESULTS_SUMMARY.md` for the full record
of what worked and what didn't, including the negative results.

---

## Web app

```bash
python3 web/app.py
```

- Dashboard: run the pipeline, watch live progress, configure scheduled runs
- Candidate list and per-candidate detail pages with every evidence check
- Per-candidate actions: re-verify, check ExoFOP, additional sectors, centroid
- Model history and retrain-attempt log

The SQLite databases (`web/*.db`) are gitignored — they are live application
state, not source. The app creates them on first run.

Scheduled runs only fire while `app.py` is running. For a schedule that
survives the machine sleeping, see `web/README_SCHEDULING.md`.

---

## Repository layout

```
code/            pipeline stages 01-09, numbered in execution order
code/experiments/  every experiment tried, including the ones that failed
code/kepler_pilot/ Kepler cross-mission expansion pilot (closed; see the scope doc)
data/            catalogs and the training dataset (light curves are gitignored)
models/          the deployed classifier + metadata + feature ranges
results/         candidate outputs, figures, tables
web/             Flask app, SQLite schema, background job runner
```

## Not included in the repo

Regenerable or machine-local, deliberately excluded (see `.gitignore`):

- Raw and processed light curves (~70 GB) — re-download via stages 01/02/06
- `models/bootstrap_ensemble/` (~140 MB) — rebuild via `09_build_bootstrap_ensemble.py`
- `web/*.db`, `web/job_logs/`, `web/static/plots/` — live app state
- Experiment `.log` files — the scripts and result JSON/CSVs are tracked

## License / data attribution

Uses public data from MAST (TESS), the NASA Exoplanet Archive, ExoFOP-TESS, and
Gaia/VSX via Vizier. Please cite those sources and the `transitleastsquares`
and `lightkurve` papers if you build on this.

---

## Reproducibility: what has actually been verified

This section exists because "independently reproducible" is a claim, and it was
tested rather than assumed. The test: clone the repo into an empty directory,
create a fresh virtualenv, install only from `requirements.txt`, and follow only
this README — using no knowledge of the original development machine.

**Verified working in a clean clone:**

- `pip install -r requirements.txt` into a fresh venv; all 13 core imports load
- `models/best_model.joblib` loads and reproduces its published metrics exactly:
  **ROC-AUC 0.9032**, precision 0.8979, recall 0.9657 on the frozen 1,105-star
  test split
- The web app starts with `python3 web/app.py`; dashboard, candidate list and
  model-history pages all serve
- Stages A–H of `06_download_unknown.py` all execute, including classifier
  scoring and ranked-output generation

**Gaps this test found, all now fixed:**

| Gap | Kind | Fix |
|---|---|---|
| No README at all | missing docs | this file |
| No dependency list | missing setup | `requirements.txt` (+ `requirements-experiments.txt`) |
| `split_manifest.json` untracked but a hard dependency of every training/scoring path | **portability bug** — a clone raised `FileNotFoundError` | now tracked |
| `--sample-size` silently ignored when a cached candidate list existed; the quick-start started a 2,000-star, ~25-hour download | **real bug** | resume path now truncates to the requested size |
| `torch` (~2 GB) in core requirements for packages the pipeline never imports | packaging | moved to `requirements-experiments.txt`; core install ~8 min |
| 45s download timeout fails every star on a cold cache | configuration, not portable | `EXOPLANETAI_DOWNLOAD_TIMEOUT` env var + documented above |

**Known limitation, not fixed:** the quick-start does not exercise
preprocessing or the transit search in a fresh clone, because the repo ships a
2,454-row `data/catalogs/unknown_features.csv` and the resume logic correctly
skips stars that already have features. To force a genuine from-scratch run of
those stages, move that file aside first:

```bash
mv data/catalogs/unknown_features.csv /tmp/
EXOPLANETAI_DOWNLOAD_TIMEOUT=300 python3 06_download_unknown.py --sample-size 10 --top-n 5
```

**Not yet verified:** cloning from GitHub specifically. The tests above cloned
from a local copy of this repository, which reproduces exactly what a `git
clone` delivers (tracked files only, no local state) but does not exercise the
GitHub transport itself.
