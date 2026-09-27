# Data and model licensing

The source code in this repository is MIT-licensed (see `LICENSE`). **That
license does not cover the data files or the trained models.** Everything
below -- the CSV and JSON tables, the result tables, and the models in
`models/` -- is derived from public astronomical archives and carries **each
source's own terms**. Nothing here grants rights the sources did not.

All quotations were taken from each source's own page on 2026-09-27; the link
beside each quote is where it came from.

## The one term that restricts reuse: Gaia is non-commercial

Gaia data are licensed **CC BY-NC 3.0 IGO**. Two Gaia-derived values,
`gaia_ruwe` and `gaia_nss`, are **input features of the deployed model**, so
**the trained models in `models/` that use them incorporate Gaia-derived
features and inherit the non-commercial term** (details in the Gaia section).
Every other source below is free to use with attribution.

| Source | What it supplies here | Terms (summary) |
|---|---|---|
| TESS / Kepler / K2 light curves and the TIC, via MAST | light-curve features, stellar parameters and crowding for candidates, SPOC DV centroids | public domain for most MAST data; acknowledge mission and STScI |
| NASA Exoplanet Archive | positive-class hosts, TOI dispositions (labels), host stellar parameters | acknowledge; cite the archive paper |
| ExoFOP-TESS | TOI table export, aggregate imaging statistics, "already flagged?" lookups | acknowledge; observer-uploaded data belong to the uploader |
| Gaia DR3 (ESA/Gaia/DPAC) | RUWE, NSS flag (model features), astrometry, Teff/distance checks | **CC BY-NC 3.0 IGO**; credit "ESA/Gaia/DPAC" |
| CDS: VizieR, SIMBAD, X-Match | access to Gaia, VSX, SB9, cluster, spectroscopic-survey, HARPS-RV and TIC catalogues; SIMBAD spectral types/parallaxes | free for scientific use; cite each original catalogue |
| IRSA (NASA/IPAC) | SFD98 / Schlafly & Finkbeiner 2011 E(B-V) | acknowledge IRSA; cite the data set |

---

## TESS, Kepler/K2 and the TESS Input Catalog (MAST, STScI)

**What is derived here.** Transit-search features (TLS), preprocessing
statistics, stellar-variability features and multi-sector processing records
computed from TESS light curves; crowding features and candidate stellar
parameters from the TESS Input Catalog (TIC, including TIC 8.2 via VizieR
`V/39/tic82`); SPOC Data Validation centroid offsets; the K2 and Kepler pilot
features.

**Files.** `data/catalogs/transit_search_results*.csv`,
`transit_search_log*.csv`, `new_tls_features.csv`,
`v3_stellar_ttv_features.csv`, `preprocess*`, `unknown_features*.csv`,
`unknown_candidate_list*.csv`, `unknown_download_log*.csv`,
`processing_mode.csv`; the light-curve, variability and crowding columns of
`data/training_dataset/training.csv`; `results/` (ranked, characterized and
verification tables, folded light-curve plots); in `code/experiments/`,
`dv_*.csv`, `crowding_per_star.csv`, `sector_map.csv` and the per-experiment
feature tables; `code/k2_pilot/*.csv` and `code/kepler_pilot/*.csv`.

**Terms** ([MAST data use](https://archive.stsci.edu/publishing/data-use)):
> "Most data hosted at MAST are in the public domain (see: open data), and
> therefore do not have restrictions on use."
>
> "There are, however, expectations (or in some cases, requirements) that
> researchers acknowledge the originating mission and (usually) STScI in all
> scholarly publications."

**Acknowledgements**
([MAST mission acknowledgements](https://archive.stsci.edu/publishing/mission-acknowledgements)):
> TESS: "This paper includes data collected with the TESS mission, obtained
> from the MAST data archive at the Space Telescope Science Institute (STScI).
> Funding for US Institutions for the TESS mission is provided by the NASA
> Explorer Program. STScI is operated by the Association of Universities for
> Research in Astronomy, Inc., under NASA contract NAS5–26555."
>
> Kepler/K2: "This paper includes data collected by the Kepler mission and
> obtained from the MAST data archive at the Space Telescope Science Institute
> (STScI)."

## NASA Exoplanet Archive

**What is derived here.** The confirmed-planet host list and host stellar
parameters (Planetary Systems Composite Parameters table), which define the
positive class; the TOI table's TFOPWG dispositions (false positives and
false alarms), which define the negative class; discovery metadata and
host-to-TIC resolution used in experiments.

**Files.** `data/catalogs/confirmed_planets.csv`, `toi_false_positives.csv`,
`toi_false_alarms.csv`; the `label`, `pl_names`, `ra`/`dec` and host
stellar-parameter columns of `data/training_dataset/training.csv`; in
`code/experiments/`, `archive_*.csv`, `positive_class_tic_ids.csv` and
`cluster1_rv_verification.csv`.

**Terms and acknowledgement**
([NASA Exoplanet Archive: acknowledging the archive](https://exoplanetarchive.ipac.caltech.edu/docs/acknowledge.html)).
The page states no usage restrictions. It asks for:
> "This research has made use of the NASA Exoplanet Archive, which is operated
> by the California Institute of Technology, under contract with the National
> Aeronautics and Space Administration under the Exoplanet Exploration
> Program."
>
> "Please cite Christiansen et al. (2025), the NASA Exoplanet Archive's
> published paper accepted by the Planetary Science Journal"

## ExoFOP-TESS

**What is derived here.** `code/experiments/exofop_toi_export.csv` is
ExoFOP's TOI table as downloaded (it feeds `web/exofop_vetting.py`);
`code/experiments/exofop_imaging_gate.json` holds **aggregate** statistics of
TFOP high-resolution imaging, with no per-star values; the
`exofop_*` columns of `results/unknown_candidates/characterized_candidates.csv`
record whether each candidate already has an ExoFOP entry.

**Deliberately not included:** `code/experiments/exofop_imaging_star.csv`, the
per-star imaging summary, because it is built from observer-uploaded data.
The README explains how to regenerate it locally.

**Terms** ([ExoFOP Data Use and Professional Conduct Policy](https://exofop.ipac.caltech.edu/tess/pcp.php)):
> "In all instances, any data associated with a given data provider should be
> considered their property and should not be claimed as anyone else's."
>
> "Before any data that have been uploaded to the ExoFOP are utilized, the
> user should contact the data owner (i.e. the user who uploaded the data)
> regarding the quality and use of the data."
>
> "Use of the data in a publication should include, as negotiated bilaterally
> between the data user and the data owner, explicit acknowledgment and/or
> co-authorship of the data owner on the publication."
>
> "Data in ExoFOP uploaded as part of the TESS Follow-Up Observation Program
> Working Group may be subject to a 12 month proprietary period."

**Acknowledgement** ([ExoFOP-TESS](https://exofop.ipac.caltech.edu/tess/)):
> "This research has made use of the Exoplanet Follow-up Observation Program
> (ExoFOP; DOI: 10.26134/ExoFOP5) website, which is operated by the California
> Institute of Technology, under contract with the National Aeronautics and
> Space Administration under the Exoplanet Exploration Program."

## Gaia DR3 (ESA/Gaia/DPAC) -- non-commercial

**What is derived here.**
- `gaia_ruwe` (renormalised unit weight error) and `gaia_nss` (non-single-star
  flag), from `I/355/gaiadr3` via VizieR. **These two are input features of the
  model.**
- In experiments: astrometric excess noise (`gaia_epsi`, `gaia_sepsi`), G
  magnitude, proper motions and parallax (kinematics), and Gaia Teff and
  distance used to check candidate stellar parameters (via the ESA Gaia
  archive).

**Files carrying Gaia-derived values.**
- `data/training_dataset/training.csv` and
  `data/training_dataset/training_BACKUP_pre_var_gap_20260815_044223.csv`
- `data/catalogs/unknown_features.csv`, `data/catalogs/unknown_features_widesector.csv`
- `results/unknown_candidates/ranked_candidates.csv`,
  `ranked_candidates_in_distribution.csv`,
  `ranked_candidates_out_of_distribution.csv`,
  `trustworthy_candidates_stellar_verification.csv`
- `code/experiments/gaia_astrometry_training.csv`, `gaia_astrometry_pools.csv`,
  `gaia_kinematics_raw.csv`, `gaia_kinematics_features.csv`,
  `binary_composite_raw.csv`, `binary_composite_features.csv`

(`GaiaESO_DR5` in `code/experiments/spectro_chem_crossmatch.csv` is the
ground-based Gaia-ESO spectroscopic survey, not ESA Gaia; see CDS below.)

**Trained models.** The following models use `gaia_ruwe` and `gaia_nss` as
features, so **they incorporate Gaia-derived data and inherit the
non-commercial term**: `models/best_model.joblib` (the deployed model),
`models/multivariate_ood_detector.joblib`,
`models/staged_best_model_optuna33.joblib` and
`models/versions/best_model_pre_optuna_c37f9f4b.joblib`. The JSON metadata
beside them (for example `models/training_feature_ranges.json`) summarises the
same features. The older 24-, 26- and 31-feature models in `models/versions/`
predate the Gaia features and do not use them.

**License** ([Gaia data license, ESA](https://www.cosmos.esa.int/web/gaia-users/license)):
> "Gaia data are distributed under the CC BY-NC 3.0 IGO license."
>
> "For details and guidelines concerning commercial use of the Gaia data,
> please see the Terms and Conditions for the use of data in the ESA space
> science archives."

**Credit line and acknowledgement**
([Gaia DR3 credit and citation instructions](https://gea.esac.esa.int/archive/documentation/GDR3/Miscellaneous/sec_credit_and_citation_instructions/)):
> "The Gaia data are open and free to use, provided credit is given to
> 'ESA/Gaia/DPAC'."
>
> "This work has made use of data from the European Space Agency (ESA) mission
> Gaia (https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing
> and Analysis Consortium (DPAC,
> https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC
> has been provided by national institutions, in particular the institutions
> participating in the Gaia Multilateral Agreement."

Cite Gaia Collaboration et al. (2016b), "The Gaia mission", and Gaia
Collaboration et al. (2023j), "Gaia DR3: Summary of the contents and survey
properties".

## CDS Strasbourg: VizieR, SIMBAD and X-Match

**Catalogues accessed through VizieR, and where their values land.**

| Catalogue (VizieR ID) | Used for | Files |
|---|---|---|
| Gaia DR3 (`I/355/gaiadr3`) | RUWE, NSS | see the Gaia section |
| TIC 8.2 (`V/39/tic82`) | coordinates for cross-matching | intermediate only |
| AAVSO International Variable Star Index (`B/vsx`) | known-variable check (`vsx_*` columns) | `results/unknown_candidates/characterized_candidates.csv`, `final_best_candidates.csv`, `needs_manual_blend_review.csv` |
| SB9 spectroscopic binaries (`B/sb9`) | `sb9_*` columns | `code/experiments/binary_composite_raw.csv`, `binary_composite_features.csv` |
| HARPS RV bank (`J/A+A/636/A74`) | RV-coverage check (`rv_*` columns) | `results/unknown_candidates/characterized_candidates.csv` |
| Hunt & Reffert 2023 (`J/A+A/673/A114`), Cantat-Gaudin & Anders 2020 (`J/A+A/633/A99`), Cantat-Gaudin et al. 2018 (`J/A+A/618/A93`) | cluster-membership flags | `code/experiments/cluster_dust_crossmatch.csv` |
| APOGEE DR17 (`III/286`), GALAH DR3 (`J/MNRAS/506/150`), LAMOST DR7 (`V/156`), Gaia-ESO DR5 (`J/A+A/666/A121`) | availability flags only (no abundances stored) | `code/experiments/spectro_chem_crossmatch.csv` |

SIMBAD supplies the spectral types and parallaxes in
`results/unknown_candidates/trustworthy_candidates_stellar_verification.csv`
(`simbad_*` columns). The CDS X-Match service was used for the bulk cluster
cross-match.

**Terms** ([VizieR rules of usage](https://cds.unistra.fr/vizier-org/licences_vizier.html)):
> "The data retrieved with VizieR are free of usage in a scientific context;
> however, as it is the usage in scientific publication, the original authors
> and publication references including the publisher have to be explicitely
> cited"
>
> "The commercial usage of the data is subject to rules depending of the
> origin"

So each catalogue above must be cited through its own reference, listed on its
VizieR page, and commercial use depends on each catalogue's origin (for Gaia,
see above). The VizieR page for `B/vsx` did not render a citation when
retrieved; the AAVSO's own
[data usage guidelines](https://www.aavso.org/data-usage-guidelines) could not
be retrieved automatically (HTTP 403) and should be consulted directly for VSX.

**Acknowledgements** ([CDS acknowledgement](https://cds.unistra.fr/help/acknowledgement/)):
> VizieR: "This research has made use of the VizieR catalogue access tool, CDS,
> Strasbourg Astronomical Observatory, France (DOI: 10.26093/cds/vizier)."
>
> SIMBAD: "This research has made use of the SIMBAD database, CDS, Strasbourg
> Astronomical Observatory, France"
>
> X-Match: "This research has made use of the CDS cross-match service,
> Strasbourg Astronomical Observatory, France."

## IRSA (NASA/IPAC Infrared Science Archive)

**What is derived here.** Galactic dust reddening E(B-V) from Schlegel,
Finkbeiner & Davis (1998) and Schlafly & Finkbeiner (2011), used in the
dust-extinction experiment.

**Files.** `code/experiments/cluster_dust_ebv.csv` (`ebv_sfd`, `ebv_sandf`).

**Acknowledgement** ([IRSA acknowledgement](https://irsa.ipac.caltech.edu/ack.html)):
> "This research has made use of the NASA/IPAC Infrared Science Archive, which
> is funded by the National Aeronautics and Space Administration and operated
> by the California Institute of Technology."
>
> "Each data set has its own Digital Object Identifier (DOI) for the specific
> data set and nearly always has a canonical paper to cite. Please include
> both of those things in your paper."

## Literature lookups (arXiv, NASA ADS)

`results/unknown_candidates/characterized_candidates.csv` records only whether
a candidate's TIC ID appears in arXiv or NASA ADS, with links to the matches.
No text or data from those services is stored.
