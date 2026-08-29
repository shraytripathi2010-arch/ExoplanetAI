"""exofop_imaging_gate.py -- ExoFOP-TESS TFOP high-resolution imaging (SG3) as a feature.

Assesses two forms:
  (a) "imaging coverage present"  -- a binary availability flag
  (b) contrast-curve summary      -- delta-mag at a fixed separation

ORDER OF CHECKS IS DELIBERATE. The SPOC DV centroid closure established that a
measurement drawn from the SPOC/TFOP vetting chain may be an INPUT to the label
rather than a predictor of it, and that circularity survives every offline check.
TFOP SG3 imaging is the same chain -- more directly so than SPOC DV, since SG3
is a TFOP subgroup and this project's negatives ARE TFOPWG dispositions. So the
circularity evidence is gathered FIRST and reported first, with the standard
availability gate run alongside for the record.

ACCESS (Part 0): ExoFOP publishes a bulk CSV of every TFOP imaging observation
at https://exofop.ipac.caltech.edu/tess/download_imaging.php?output=csv --
36,778 rows, 14,289 distinct TIC ids, no HTML scraping. `Image Type` separates
Speckle / AO / Lucky / Seeing-Limited; `Contrast` is free text but 84.5%
parseable as "delta X mag @ Y arcsec"; `Obs Date` supports a timing test.

TIC matching is EXACT, reusing `dv_star_availability.csv`'s host->TIC resolution
from the SPOC DV work -- per the standing archive-availability rule that TESS
products are per-TIC and a coordinate cone search counts neighbours.

Reads only. Writes one JSON + one CSV to code/experiments/. Touches nothing in
production.
"""
import os
import re
import csv
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
SCRATCH = os.environ.get("EXOFOP_SCRATCH", HERE)
IMAGING_CSV = os.path.join(SCRATCH, "download_imaging.csv")
TOI_CSV = os.path.join(SCRATCH, "download_toi.csv")
STARMAP = os.path.join(HERE, "dv_star_availability.csv")
OUT_JSON = os.path.join(HERE, "exofop_imaging_gate.json")
OUT_CSV = os.path.join(HERE, "exofop_imaging_star.csv")

IMG_HDR = ["TIC ID", "Name", "TOI", "Telescope", "Instrument", "Filter",
           "Image Type", "Pixel Scale", "PSF", "Contrast", "Obs Date",
           "User", "Group", "Tag", "Notes"]
HIRES = {"Speckle", "AO", "Lucky"}          # SG3 high-resolution; excludes Seeing-Limited
CONTRAST_RE = re.compile(r"delta\s*(-?[0-9.]+)\s*mag\s*@\s*([0-9.]+)", re.I)


def auc(y, x):
    from scipy.stats import rankdata
    y = np.asarray(y, float); x = np.asarray(x, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30 or len(np.unique(y[m])) < 2:
        return float("nan")
    r = rankdata(x[m]); y1 = y[m] == 1
    n1, n0 = y1.sum(), (~y1).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def read_ragged(path, hdr):
    """ExoFOP's CSV has unescaped quotes inside Notes, so ~1.9% of rows overflow
    the field count. Notes is LAST, so truncating to len(hdr) is lossless for
    every column this analysis uses. Reported rather than silently absorbed."""
    rows, over = [], 0
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) < 11:
                continue
            if len(row) > len(hdr):
                over += 1
            rows.append(row[:len(hdr)] + [""] * max(0, len(hdr) - len(row)))
    return pd.DataFrame(rows, columns=hdr), over


def main():
    from scipy.stats import fisher_exact
    res = {}

    img, over = read_ragged(IMAGING_CSV, IMG_HDR)
    img["tic"] = pd.to_numeric(img["TIC ID"], errors="coerce")
    img = img[img.tic.notna()].copy()
    img["obs_date"] = pd.to_datetime(img["Obs Date"], errors="coerce")
    res["part0_access"] = {
        "endpoint": "https://exofop.ipac.caltech.edu/tess/download_imaging.php?output=csv",
        "method": "bulk CSV, no scraping; exact TIC join",
        "rows": int(len(img)), "unique_tic": int(img.tic.nunique()),
        "ragged_rows_truncated": int(over),
        "image_type_counts": img["Image Type"].replace("", "(blank)").value_counts().head(8).to_dict(),
        "group_counts": img["Group"].replace("", "(blank)").value_counts().head(4).to_dict(),
        "obs_date_min": str(img.obs_date.min().date()), "obs_date_max": str(img.obs_date.max().date())}

    hi = img[img["Image Type"].isin(HIRES)].copy()
    res["part0_access"]["hires_rows"] = int(len(hi))
    res["part0_access"]["hires_unique_tic"] = int(hi.tic.nunique())

    m = hi["Contrast"].fillna("").apply(CONTRAST_RE.search)
    hi["dmag"] = m.apply(lambda x: float(x.group(1)) if x else np.nan)
    hi["sep"] = m.apply(lambda x: float(x.group(2)) if x else np.nan)
    res["part0_access"]["contrast_parseable_pct"] = float(100 * hi.dmag.notna().mean())

    # per-star rollup: deepest contrast at the most common separation, plus first/last obs
    hi05 = hi[(hi.sep >= 0.4) & (hi.sep <= 0.6)]
    per = hi.groupby("tic").agg(n_obs=("tic", "size"),
                                first_obs=("obs_date", "min"),
                                last_obs=("obs_date", "max"),
                                best_dmag=("dmag", "max")).reset_index()
    d05 = hi05.groupby("tic").dmag.max().rename("dmag_0p5").reset_index()
    per = per.merge(d05, on="tic", how="left")

    sm = pd.read_csv(STARMAP)
    sm["tic"] = pd.to_numeric(sm["tic"], errors="coerce")
    sm = sm.merge(per, on="tic", how="left")
    sm["has_img"] = sm.n_obs.notna()
    sm.loc[sm.tic.isna(), "has_img"] = np.nan          # unresolved TIC = cannot measure
    sm.to_csv(OUT_CSV, index=False)

    tr = sm[sm._set == "train"].copy()
    y = tr.label.to_numpy()
    resolved = tr.tic.notna()
    res["tic_resolution"] = {"train_resolved": int(resolved.sum()), "train_total": int(len(tr)),
                             "pct": float(100 * resolved.mean())}

    t = tr[resolved]
    yy = t.label.to_numpy()
    h = t.has_img.astype(bool).to_numpy()
    pos, neg = float(100 * h[yy == 1].mean()), float(100 * h[yy == 0].mean())
    tab = [[int((h & (yy == 1)).sum()), int((~h & (yy == 1)).sum())],
           [int((h & (yy == 0)).sum()), int((~h & (yy == 0)).sum())]]
    orr, pv = fisher_exact(tab)
    gate = {"train_pct": float(100 * h.mean()), "train_pos_pct": pos, "train_neg_pct": neg,
            "diff_pp": pos - neg, "contingency": tab, "odds_ratio": float(orr),
            "fisher_p": float(pv), "auc_availability": auc(yy, h.astype(float)),
            "n_train_with_imaging": int(h.sum())}
    for s in ("main", "widesector"):
        p = sm[sm._set == s]
        if len(p):
            gate[f"pool_{s}_pct"] = float(100 * p.has_img.fillna(False).astype(bool).mean())
            gate[f"pool_{s}_n"] = int(p.has_img.fillna(False).astype(bool).sum())
            gate[f"pool_{s}_total"] = int(len(p))
    res["part2_availability_gate"] = gate

    # ---- PART 1: CIRCULARITY -- did imaging PRECEDE the disposition? ----
    toi, _ = read_ragged(TOI_CSV, None) if False else (None, None)
    toi = pd.read_csv(TOI_CSV, dtype=str, low_memory=False, on_bad_lines="skip")
    toi["tic"] = pd.to_numeric(toi["TIC ID"], errors="coerce")
    toi["disp"] = toi["TFOPWG Disposition"].fillna("").str.strip()
    for c in ("Date TOI Alerted (UTC)", "Date TOI Updated (UTC)", "Date Modified"):
        toi[c] = pd.to_datetime(toi[c], errors="coerce")
    tt = toi[toi.tic.notna()].groupby("tic").agg(
        disp=("disp", "first"),
        alerted=("Date TOI Alerted (UTC)", "min"),
        updated=("Date TOI Updated (UTC)", "max")).reset_index()

    j = per.merge(tt, on="tic", how="inner")
    j = j[j.first_obs.notna() & j.updated.notna()]
    circ = {"n_matched_toi_with_imaging": int(len(j)),
            "disposition_counts": j.disp.replace("", "(blank)").value_counts().head(8).to_dict()}
    fp = j[j.disp.isin(["FP", "FA"])]
    cp = j[j.disp.isin(["CP", "KP"])]
    for tag, sub in (("FP_FA", fp), ("CP_KP", cp), ("ALL", j)):
        if len(sub) < 20:
            continue
        pre = (sub.first_obs < sub.updated)
        circ[tag] = {"n": int(len(sub)),
                     "pct_imaging_before_disposition_update": float(100 * pre.mean()),
                     "median_days_obs_to_update": float((sub.updated - sub.first_obs).dt.days.median())}
    # imaging rate by disposition -- is imaging allocated differently to FPs?
    allt = tt.merge(per[["tic", "n_obs"]], on="tic", how="left")
    allt["has_img"] = allt.n_obs.notna()
    circ["imaging_rate_by_disposition_pct"] = {
        k: float(100 * v) for k, v in allt.groupby(allt.disp.replace("", "(blank)")).has_img.mean()
        .sort_values(ascending=False).head(8).items()}
    circ["n_toi_total"] = int(len(allt))
    res["part1_circularity"] = circ

    json.dump(res, open(OUT_JSON, "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()


def value_and_provenance():
    """PART 2b/1b -- run after main(). Two things the gate above does not cover:

      * the CONTRAST VALUE on the restricted population (availability held
        constant), the design the SPOC DV closure used to separate a real
        measurement from an availability artifact;
      * whether the mild overall class split is really 'no selection', or two
        large opposing selection effects cancelling. Split by PROVENANCE
        (TIC-named TOI vs name-provenance) to find out.
    """
    d = pd.read_csv(OUT_CSV)
    tr = d[(d._set == "train") & d.tic.notna()].copy()
    tr["tic_named"] = tr.host.astype(str).str.match(r"TIC[_ ]?\d+")
    have = tr[tr.has_img == True]
    out = {"restricted_n": int(len(have)),
           "restricted_pos": int((have.label == 1).sum()),
           "restricted_neg": int((have.label == 0).sum()), "values": {}}
    for col in ("dmag_0p5", "best_dmag", "n_obs"):
        v = pd.to_numeric(have[col], errors="coerce")
        a = auc(have.label.to_numpy(), v.to_numpy())
        out["values"][col] = {"n": int(v.notna().sum()), "auc": a,
                              "abs_auc_minus_half": abs(a - 0.5),
                              "median_pos": float(v[have.label == 1].median()),
                              "median_neg": float(v[have.label == 0].median())}
    prov = {}
    for (lab, tn), g in tr.groupby(["label", "tic_named"]):
        prov[f"label{int(lab)}_{'TICnamed' if tn else 'nameprov'}"] = {
            "n": int(len(g)), "imaging_pct": float(100 * g.has_img.mean())}
    tn = tr[tr.tic_named]
    if (tn.label == 1).sum() > 20:
        prov["provenance_matched_split_pp"] = float(
            100 * (tn[tn.label == 0].has_img.mean() - tn[tn.label == 1].has_img.mean()))
    out["provenance"] = prov
    return out


if __name__ == "__main__" and os.environ.get("EXOFOP_APPEND"):
    r = json.load(open(OUT_JSON))
    r["part2b_value_and_provenance"] = value_and_provenance()
    json.dump(r, open(OUT_JSON, "w"), indent=1, default=str)
    print(json.dumps(r["part2b_value_and_provenance"], indent=1, default=str))
