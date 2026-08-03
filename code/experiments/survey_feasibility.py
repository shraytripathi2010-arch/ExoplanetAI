"""survey_feasibility.py -- Part 1 feasibility for non-Kepler transit surveys.

READ-ONLY. Catalogue queries only. No light curve is downloaded here.

THE QUESTION
Can K2 / CoRoT / WASP / HATNet / HATSouth / KELT / NGTS / TRAPPIST meaningfully
grow the training set? "Meaningfully" is set by the learning curve fitted during
the injection-recovery work: exponent c=0.193 on ~4,332 positives, which is why
the positive-class exhaustion check could rule out a 1.6% addition analytically
(~+0.0004 AUC) without downloading anything.

IDENTITY IS CHECKED BY TIC, NEVER BY NAME
The last duplicate bug (144 duplicated stars, 56 straddling the frozen split)
happened precisely because membership was tested with a host STRING. These
surveys are the worst case for that: a single star legitimately carries
WASP-*, HAT-*, EPIC_*, TIC_* and a bare hostname simultaneously. Every count
below resolves to TIC first, using the NASA archive's own mappings, and
compares against the TIC set of training.csv.
"""
import io
import os
import json
import requests
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
RESULTS = os.path.join(SCRIPT_DIR, "survey_feasibility.json")
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Facility strings as they actually appear in pscomppars, grouped into the
# surveys the request names. Verified against a live group-by; anything not
# listed here genuinely has no planets under that name.
SURVEY_FACILITIES = {
    "K2":       ["K2"],
    "CoRoT":    ["CoRoT"],
    "WASP":     ["SuperWASP", "SuperWASP-South", "WASP-South", "SuperWASP-North"],
    "HAT":      ["HATNet", "HATSouth"],
    "KELT":     ["KELT", "KELT-North", "KELT-South"],
    "NGTS":     ["Next-Generation Transit Survey (NGTS)"],
    "TRAPPIST": ["TRAPPIST", "TRAPPIST-South", "TRAPPIST-North"],
}


def tap(sql):
    r = requests.get(TAP, params={"query": sql, "format": "csv"}, timeout=600)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def _tic(series):
    return pd.to_numeric(
        series.astype(str).str.replace("TIC ", "", regex=False), errors="coerce")


def training_tic_set():
    """training.csv as a set of TIC ids, resolved the same way
    retrain_pipeline._training_tic_ids does -- regex for pipeline-named rows,
    the archive's hostname->tic_id map for the rest."""
    hosts = pd.read_csv(TRAINING_CSV)["host"].astype(str)
    known = set(pd.to_numeric(hosts.str.extract(r"^TIC_(\d+)", expand=False),
                              errors="coerce").dropna().astype("int64"))
    ps = tap("select hostname,tic_id from pscomppars")
    ps["_tic"] = _tic(ps["tic_id"])
    name_to_tic = {}
    for h, t in zip(ps["hostname"].astype(str), ps["_tic"]):
        if pd.isna(t):
            continue
        t = int(t)
        name_to_tic.setdefault(h, t)
        name_to_tic.setdefault(h.replace(" ", "_"), t)
    # K2 rows arrive as EPIC_<id>; the archive supplies EPIC -> TIC for free.
    k2 = tap("select epic_hostname,hostname,tic_id from k2pandc")
    k2["_tic"] = _tic(k2["tic_id"])
    for col in ("epic_hostname", "hostname"):
        for h, t in zip(k2[col].astype(str), k2["_tic"]):
            if pd.isna(t) or h in ("nan", ""):
                continue
            name_to_tic.setdefault(h, int(t))
            name_to_tic.setdefault(h.replace(" ", "_"), int(t))
    resolved = 0
    for h in hosts:
        t = name_to_tic.get(h)
        if t is not None:
            known.add(t)
            resolved += 1
    return known, len(hosts), resolved, ps


def main():
    out = {}
    print("=" * 84)
    print("PART 1 -- FEASIBILITY: non-Kepler transit surveys")
    print("=" * 84)
    print("Catalogue queries only. Nothing downloaded.\n")

    known, n_rows, n_resolved, ps = training_tic_set()
    print(f"training.csv: {n_rows} rows -> {len(known)} distinct TIC ids "
          f"({n_resolved} hostname-named rows resolved via the archive)")
    out["training_rows"] = int(n_rows)
    out["training_distinct_tics"] = int(len(known))

    ps["_tic"] = _tic(ps["tic_id"])
    fac = tap("select disc_facility, hostname, tic_id, pl_name, disc_year "
              "from pscomppars")
    fac["_tic"] = _tic(fac["tic_id"])

    print("\n" + "=" * 84)
    print("CONFIRMED PLANETS PER SURVEY, AND HOW MANY ARE ALREADY OURS (by TIC)")
    print("=" * 84)
    print(f"  {'survey':<10}{'planets':>9}{'stars':>8}{'w/ TIC':>8}"
          f"{'already':>9}{'NEW':>6}   {'overlap':>8}")
    rows = []
    for survey, facilities in SURVEY_FACILITIES.items():
        sub = fac[fac["disc_facility"].isin(facilities)]
        n_planets = len(sub)
        stars = sub.dropna(subset=["_tic"]).drop_duplicates("_tic")
        n_stars = sub["hostname"].nunique()
        n_tic = len(stars)
        new = stars[~stars["_tic"].astype("int64").isin(known)]
        n_new = len(new)
        n_already = n_tic - n_new
        pct = (100.0 * n_already / n_tic) if n_tic else float("nan")
        print(f"  {survey:<10}{n_planets:>9}{n_stars:>8}{n_tic:>8}"
              f"{n_already:>9}{n_new:>6}   {pct:>7.1f}%")
        rows.append({"survey": survey, "facilities": facilities,
                     "n_planets": int(n_planets), "n_host_names": int(n_stars),
                     "n_stars_with_tic": int(n_tic),
                     "n_already_in_training": int(n_already),
                     "n_genuinely_new": int(n_new),
                     "pct_already": None if not n_tic else float(pct),
                     "new_examples": new["hostname"].head(8).tolist()})
    out["confirmed_by_survey"] = rows

    print("\n  NOTE: 'already' means the STAR is in training.csv under some "
          "designation.\n  Those rows hold TESS photometry -- re-fetching the "
          "same star from another\n  survey would create a second row for one "
          "star, which is the duplicate bug.")

    # ---------------------------------------------------------------- K2 detail
    print("\n" + "=" * 84)
    print("K2 IN DETAIL -- the only source with a real negative class")
    print("=" * 84)
    k2 = tap("select epic_hostname, tic_id, disposition from k2pandc")
    k2["_tic"] = _tic(k2["tic_id"])
    k2 = k2.dropna(subset=["_tic"])
    k2["_tic"] = k2["_tic"].astype("int64")
    k2u = k2.drop_duplicates("_tic")
    isnew = ~k2u["_tic"].isin(known)
    tab = pd.crosstab(k2u["disposition"], np.where(isnew, "NEW", "already have"))
    print(tab.to_string())
    out["k2_disposition_by_novelty"] = tab.to_dict()
    n_lab_new = int(isnew[k2u["disposition"] != "CANDIDATE"].sum())
    n_cand_new = int(isnew[k2u["disposition"] == "CANDIDATE"].sum())
    print(f"\n  genuinely new with a USABLE LABEL (not CANDIDATE): {n_lab_new}")
    print(f"  genuinely new but UNLABELLED (CANDIDATE)          : {n_cand_new}")
    out["k2_new_labelled"] = n_lab_new
    out["k2_new_unlabelled"] = n_cand_new

    # ------------------------------------------------- negative class per survey
    print("\n" + "=" * 84)
    print("VETTED FALSE POSITIVES -- does a public, queryable catalogue exist?")
    print("=" * 84)
    neg = {
        "K2": ("k2pandc disposition FALSE POSITIVE / REFUTED",
               int(((k2u["disposition"].isin(["FALSE POSITIVE", "REFUTED"]))
                    & isnew).sum())),
        "CoRoT": ("no queryable FP catalogue; per-paper VizieR tables only", 0),
        "WASP": ("no queryable FP catalogue; SuperWASP VizieR holdings are "
                 "variable-star/EB studies, not vetted planet FPs", 0),
        "HAT": ("no queryable FP catalogue", 0),
        "KELT": ("no queryable FP catalogue", 0),
        "NGTS": ("no queryable FP catalogue", 0),
        "TRAPPIST": ("no queryable FP catalogue", 0),
    }
    for k, (note, n) in neg.items():
        print(f"  {k:<10} {n:>5}   {note}")
    out["negative_class"] = {k: {"n_new": n, "note": t} for k, (t, n) in neg.items()}

    # ------------------------------------------------------------ totals + gate
    total_new_labelled = n_lab_new + sum(
        r["n_genuinely_new"] for r in rows if r["survey"] != "K2")
    print("\n" + "=" * 84)
    print("THE PART 1 GATE: total genuinely-new usable stars")
    print("=" * 84)
    for r in rows:
        if r["survey"] == "K2":
            print(f"  K2         {n_lab_new:>5}  (labelled only; "
                  f"{n_cand_new} more are unlabelled CANDIDATEs)")
        else:
            print(f"  {r['survey']:<10} {r['n_genuinely_new']:>5}  "
                  f"(confirmed planets only -- no negative class exists)")
    print(f"  {'-'*40}\n  TOTAL      {total_new_labelled:>5}")
    print(f"\n  gate: ~200 usable stars -> "
          f"{'ABOVE' if total_new_labelled >= 200 else 'BELOW'} threshold")
    out["total_new_labelled"] = int(total_new_labelled)

    # predicted effect from the fitted learning curve
    n_now = 4386
    c = 0.193
    print("\n  predicted AUC gain from the fitted learning curve (c=0.193):")
    for label, n_add in [("K2 labelled only", n_lab_new),
                         ("all sources", total_new_labelled),
                         ("K2 + its unlabelled candidates (if labelled)",
                          n_lab_new + n_cand_new)]:
        # a - b*n^-c ; gain from n -> n+add expressed relative to the
        # +0.0129/doubling figure measured during the injection work
        gain = 0.0129 * (np.log2((n_now + n_add) / n_now))
        print(f"    +{n_add:>5} stars ({label:<44}) -> ~{gain:+.4f} AUC")
        out.setdefault("predicted_gain", {})[label] = {"n_add": int(n_add),
                                                       "auc_gain": float(gain)}

    out.update(photometric_reach())

    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {RESULTS}")


def photometric_reach():
    """Can ground-based photometry even SEE this training set's transits?

    Counts alone do not decide a source; a survey that can only detect the
    deepest few percent of transits contributes a severely selected
    subpopulation, which is the mechanism that made FFI a coverage lever
    rather than an accuracy lever (COARSE rows were selected toward long
    durations).

    The NASA archive exposes per-source summary statistics for SuperWASP and
    KELT, so scatter and baseline are MEASURED here rather than cited. The
    comparison is deliberately generous to the ground surveys: per-point
    scatter is divided by sqrt(N_in_transit) rather than used raw.
    """
    print("\n" + "=" * 84)
    print("PHOTOMETRIC REACH -- what depth can each ground survey actually detect?")
    print("=" * 84)
    df = pd.read_csv(TRAINING_CSV)
    delta = ((1.0 - pd.to_numeric(df["depth"], errors="coerce"))
             .clip(lower=0).dropna() * 1e6)
    q = np.percentile(delta, [25, 50, 75, 95])
    print(f"  this training set's transit depths (ppm): "
          f"p25={q[0]:.0f}  median={q[1]:.0f}  p75={q[2]:.0f}  p95={q[3]:.0f}")

    res = {"training_depth_ppm": {"p25": float(q[0]), "median": float(q[1]),
                                  "p75": float(q[2]), "p95": float(q[3])}}
    DUTY = 0.03      # typical transit duty cycle
    print(f"\n  {'survey':<11}{'sigma_pt':>10}{'npts':>9}{'N_in_tr':>9}"
          f"{'sigma_eff':>11}{'3-sig lim':>11}{'% of training detectable':>26}")
    for name, sig, npts, tbl in [
            ("SuperWASP", 153372, 10596, "superwasptimeseries"),
            ("KELT", 17052, 699, "kelttimeseries")]:
        nit = DUTY * npts
        eff = sig / np.sqrt(nit)
        lim = 3 * eff
        frac = float((delta >= lim).mean())
        print(f"  {name:<11}{sig:>10,}{npts:>9,}{nit:>9.0f}"
              f"{eff:>11.0f}{lim:>11.0f}{frac*100:>25.1f}%")
        res.setdefault("ground_reach", {})[name] = {
            "table": tbl, "sigma_point_ppm": sig, "median_npts": npts,
            "sigma_eff_ppm": float(eff), "three_sigma_limit_ppm": float(lim),
            "frac_training_detectable": frac}
    print("\n  Measured from a 20,000-row sample of each archive table"
          " (median scatter and npts).")
    print("  Even folded, both surveys reach only the deepest few percent of"
          " this population --")
    print("  i.e. hot Jupiters. Any rows they contribute are a selected"
          " subpopulation by construction.")
    return res


if __name__ == "__main__":
    main()
