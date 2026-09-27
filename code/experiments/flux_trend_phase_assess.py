"""flux_trend_phase_assess.py -- PART 1 demonstration + PART 2/3 pre-model battery.

Runs, in order:
  0. the variance-invariance demonstration (folded vs unfolded MAD)
  1. coverage on training and BOTH production pools, class-rate gate
  2. single-feature AUC, incl. the wrong-period null control
  3. correlation against the 33 deployed features, the ellipsoidal columns,
     ls_period_match and the CLOSED trend_slope_ppm_day
  4. |galactic latitude| control arm      (standing requirement)
  5. sector / observation-epoch control arm  (NEW standing requirement, from the
     momentum-dump entry -- implemented here for the first time)
"""
import os, sys, json, importlib.util
import numpy as np, pandas as pd

warn = np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
OUT = os.path.join(HERE, "flux_trend_phase_assess.json")

FT = os.path.join(HERE, "flux_trend_phase_features.csv")
ELL = os.path.join(HERE, "ellipsoidal_features.csv")
TRIPLE = os.path.join(HERE, "exominer_triple_features.csv")
SCHED = os.path.join(HERE, "momentum_dump_schedule.json")

NEW = ["ft_slope_phase", "ft_slope_ctrl", "ft_slope_over_ctrl",
       "ft_c1_signed", "ft_s1_signed", "ft_slope_snr"]


def auc(y, x):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30 or len(np.unique(y[m])) < 2:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(x[m]); y1 = y[m] == 1
    n1, n0 = y1.sum(), (~y1).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    res = {}
    spec = importlib.util.spec_from_file_location("m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05; spec.loader.exec_module(m05)
    FEATS = list(m05.FEATURE_COLUMNS)
    res["production_features"] = len(FEATS)

    ft = pd.read_csv(FT); ft["host"] = ft.host.astype(str)
    tr = pd.read_csv(os.path.join(ROOT, "data", "training_dataset", "training.csv"))
    tr["host"] = tr.host.astype(str)

    # ---- 0. variance invariance ------------------------------------------
    v = ft[np.isfinite(ft.ft_var_unfolded) & np.isfinite(ft.ft_var_folded)]
    d = (v.ft_var_folded - v.ft_var_unfolded).abs()
    res["variance_invariance"] = {
        "n": int(len(v)), "n_bit_identical": int((d == 0).sum()),
        "frac_bit_identical": float((d == 0).mean()),
        "max_abs_diff": float(d.max()), "median_var": float(v.ft_var_unfolded.median())}

    # ---- 1. coverage / class-rate gate -----------------------------------
    cov = {}
    for s in ("train", "main", "widesector"):
        sub = ft[ft._set == s]
        cov[s] = {"n": int(len(sub)),
                  "pct_ok": float(100.0 * np.isfinite(sub.ft_slope_phase).mean())}
    m = tr.merge(ft[ft._set == "train"], on="host", how="left")
    have = np.isfinite(m.ft_slope_phase).astype(int)
    y = m.label.to_numpy()
    from scipy.stats import fisher_exact
    tab = [[int(((have == 1) & (y == 1)).sum()), int(((have == 0) & (y == 1)).sum())],
           [int(((have == 1) & (y == 0)).sum()), int(((have == 0) & (y == 0)).sum())]]
    orr, p = fisher_exact(tab)
    cov["train_pos_pct"] = float(100.0 * have[y == 1].mean())
    cov["train_neg_pct"] = float(100.0 * have[y == 0].mean())
    cov["availability_auc"] = auc(y, have.astype(float))
    cov["availability_or"] = float(orr); cov["availability_fisher_p"] = float(p)
    cov["status_counts"] = ft[ft._set == "train"].ft_status.value_counts().head(8).to_dict()
    res["coverage"] = cov

    # ---- 2. single-feature AUC -------------------------------------------
    res["single_feature_auc"] = {c: auc(y, m[c].to_numpy()) for c in NEW}
    res["single_feature_auc_abs"] = {c: (abs(v - 0.5) if np.isfinite(v) else None)
                                     for c, v in res["single_feature_auc"].items()}
    # direction check on the |slope| too -- sign may not be the informative part
    for c in ("ft_slope_phase", "ft_c1_signed", "ft_s1_signed"):
        res["single_feature_auc"]["abs_" + c] = auc(y, np.abs(m[c].to_numpy()))

    # ---- 3. correlations --------------------------------------------------
    ell = pd.read_csv(ELL); ell["host"] = ell.host.astype(str)
    ell = ell[ell._set == "train"][["host", "ell_a1", "ell_a2", "ell_c2_signed",
                                    "ell_pp_full", "ell_pp_half", "ell_pp_ratio",
                                    "ell_a2_frac", "ell_a2_over_ctrl"]]
    tri = pd.read_csv(TRIPLE); tri["host"] = tri.host.astype(str)
    tri = tri[["host", "ls_period_match", "trend_slope_ppm_day", "trend_amp_frac"]]
    m = m.merge(ell, on="host", how="left").merge(tri, on="host", how="left")

    ref = FEATS + ["ell_a1", "ell_a2", "ell_c2_signed", "ell_pp_full", "ell_pp_half",
                   "ell_pp_ratio", "ell_a2_frac", "ell_a2_over_ctrl",
                   "ls_period_match", "trend_slope_ppm_day", "trend_amp_frac"]
    corr = {}
    for c in NEW:
        row = {}
        for r in ref:
            if r not in m.columns:
                continue
            a, b = m[c].to_numpy(float), pd.to_numeric(m[r], errors="coerce").to_numpy(float)
            k = np.isfinite(a) & np.isfinite(b)
            if k.sum() < 100:
                continue
            from scipy.stats import spearmanr
            rho = spearmanr(a[k], b[k]).statistic
            if np.isfinite(rho):
                row[r] = float(rho)
        top = sorted(row.items(), key=lambda kv: -abs(kv[1]))[:6]
        corr[c] = {"max_abs_rho": float(abs(top[0][1])) if top else None,
                   "max_against": top[0][0] if top else None,
                   "top6": [(k2, round(v2, 3)) for k2, v2 in top],
                   "vs_var_oot_rms": round(row.get("var_oot_rms", np.nan), 3),
                   "vs_var_ls_amp": round(row.get("var_ls_amp", np.nan), 3),
                   "vs_ell_a1": round(row.get("ell_a1", np.nan), 3),
                   "vs_ell_pp_full": round(row.get("ell_pp_full", np.nan), 3),
                   "vs_ell_c2_signed": round(row.get("ell_c2_signed", np.nan), 3),
                   "vs_ls_period_match": round(row.get("ls_period_match", np.nan), 3),
                   "vs_trend_slope_ppm_day": round(row.get("trend_slope_ppm_day", np.nan), 3)}
    res["correlations"] = corr

    # analytic identity check: slope should track (6/pi)*s1 on real data
    a, b = m.ft_slope_phase.to_numpy(float), m.ft_s1_signed.to_numpy(float)
    k = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    from scipy.stats import spearmanr
    res["odd_projection_identity"] = {
        "spearman_slope_vs_s1": float(spearmanr(a[k], b[k]).statistic),
        "median_ratio_slope_over_s1": float(np.median(a[k] / b[k])),
        "predicted_6_over_pi": float(6 / np.pi)}

    # ---- 4. |galactic latitude| control arm -------------------------------
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    g = SkyCoord(ra=m.ra.to_numpy() * u.deg, dec=m.dec.to_numpy() * u.deg).galactic
    m["absb"] = np.abs(g.b.deg)
    gal = {}
    for c in NEW:
        x = m[c].to_numpy(float); bb = m.absb.to_numpy(float)
        k = np.isfinite(x) & np.isfinite(bb)
        rho = float(spearmanr(x[k], bb[k]).statistic)
        q = pd.qcut(m.loc[k, "absb"], 4, labels=False, duplicates="drop")
        aucs = [auc(y[k][q == i], x[k][q == i]) for i in range(4)]
        aucs = [a2 for a2 in aucs if np.isfinite(a2)]
        gal[c] = {"rho_absb": rho, "quartile_auc": [round(a2, 3) for a2 in aucs],
                  "spread": float(max(aucs) - min(aucs)) if aucs else None}
    res["galactic_latitude_control"] = gal

    # ---- 5. sector / observation-epoch control arm (NEW) ------------------
    sch = json.load(open(SCHED))
    binw, t0 = float(sch["bin_width"]), float(sch["t_origin"])
    bin2sec = {int(v["bin"]): int(v["sector"]) for v in sch["sectors"].values()}
    b_idx = np.floor((m.ft_sector_first_t.to_numpy(float) - t0) / binw)
    m["sector"] = [bin2sec.get(int(bi), np.nan) if np.isfinite(bi) else np.nan for bi in b_idx]
    ep = {"n_with_sector": int(np.isfinite(m.sector).sum()),
          "auc_sector_alone": auc(y, m.sector.to_numpy(float))}
    era_edges = [0, 14, 27, 40, 56, 70, 85, 200]
    rows = []
    for lo, hi in zip(era_edges[:-1], era_edges[1:]):
        k = np.isfinite(m.sector) & (m.sector >= lo) & (m.sector < hi)
        if k.sum() > 30:
            rows.append({"era": f"{lo}-{hi-1}", "n": int(k.sum()),
                         "neg_rate": round(float((y[k] == 0).mean()), 3)})
    ep["negative_rate_by_era"] = rows
    for c in NEW:
        x = m[c].to_numpy(float); s = m.sector.to_numpy(float)
        k = np.isfinite(x) & np.isfinite(s)
        rho = float(spearmanr(x[k], s[k]).statistic)
        q = pd.qcut(pd.Series(s[k]), 4, labels=False, duplicates="drop")
        aucs = [auc(y[k][q == i], x[k][q == i]) for i in range(4)]
        aucs = [a2 for a2 in aucs if np.isfinite(a2)]
        ep[c] = {"rho_sector": rho, "quartile_auc": [round(a2, 3) for a2 in aucs],
                 "spread": float(max(aucs) - min(aucs)) if aucs else None}
    res["sector_epoch_control"] = ep

    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    m[["host", "label", "sector", "absb"] + NEW].to_csv(
        os.path.join(HERE, "flux_trend_phase_train.csv"), index=False)


if __name__ == "__main__":
    main()
