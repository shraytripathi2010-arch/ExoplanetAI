"""cluster_dust_matched_band.py -- PART 1/2.

Two things the Part 0 gate left open, both cheap and both decision-relevant:

  (A) CLUSTER MEMBERSHIP -- the flag is defined for every star, so it has no
      missingness gate. What it has instead is an EFFECTIVE-n problem, and a
      question the brief asked to be answered rather than assumed: is cluster
      membership a smooth function of |b| (i.e. sky position again), or does it
      carry structure position does not? Measured here.

  (B) 2D DUST -- Bayestar19 (3D) is closed at Part 0 on footprint. But the
      brief asked whether a 2D (l,b)-only proxy is a restatement of sky
      position. That claim is TESTED here rather than asserted: real SFD98 and
      Schlafly & Finkbeiner 2011 E(B-V) from IRSA (all-sky, so it clears the
      availability gate Bayestar19 failed), then the crowding investigation's
      MATCHED-SKY-BAND test.

MATCHED-BAND METHOD, copied from the crowding deployment rather than reinvented:
restrict BOTH classes to |b| in [8, 40] deg, the band in which this dataset's
class asymmetry by galactic latitude INVERTS (median |b| 16.9 pos / 12.5 neg
overall, becoming 15.9 / 19.7 inside the band), and ask whether the feature
retains separation there.

Reads only; writes one JSON + one CSV to code/experiments/.
"""
import os
import json
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
XM = os.path.join(HERE, "cluster_dust_crossmatch.csv")
OUT_JSON = os.path.join(HERE, "cluster_dust_matched_band.json")
OUT_CSV = os.path.join(HERE, "cluster_dust_ebv.csv")

BAND_LO, BAND_HI = 8.0, 40.0      # the crowding investigation's exact band
SEED = 20260829
N_POS_SAMPLE = 1179               # matched to the full negative count
WORKERS = 6


def auc(y, x):
    from scipy.stats import rankdata
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30 or len(np.unique(y[m])) < 2:
        return float("nan")
    r = rankdata(x[m]); y1 = y[m] == 1
    n1, n0 = y1.sum(), (~y1).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def galb(ra, dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    g = SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg).galactic
    return np.abs(g.b.deg), g.l.deg


_IRSA = None


def _irsa():
    global _IRSA
    if _IRSA is None:
        try:
            from astroquery.ipac.irsa.irsa_dust import IrsaDust
        except Exception:
            from astroquery.irsa_dust import IrsaDust
        _IRSA = IrsaDust
    return _IRSA


def ebv_one(args):
    ra, dec = args
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return (np.nan, np.nan)
    for _ in range(3):
        try:
            t = _irsa().get_query_table(SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                        section="ebv")
            return (float(t["ext SandF mean"][0]), float(t["ext SFD mean"][0]))
        except Exception:
            pass
    return (np.nan, np.nan)


def fetch_ebv(ra, dec, tag):
    print(f"  [{tag}] querying IRSA for {len(ra)} positions ({WORKERS} threads)...", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        out = list(ex.map(ebv_one, list(zip(ra, dec))))
    a = np.array(out, dtype=float)
    print(f"  [{tag}] got {int(np.isfinite(a[:,0]).sum())}/{len(ra)}", flush=True)
    return a[:, 0], a[:, 1]


def main():
    from scipy.stats import spearmanr, fisher_exact
    res = {"band": [BAND_LO, BAND_HI], "seed": SEED}
    rng = np.random.default_rng(SEED)

    tr = pd.read_csv(TRAINING); tr["host"] = tr.host.astype(str)
    xm = pd.read_csv(XM); xm["host"] = xm.host.astype(str)
    tr = tr.merge(xm.drop(columns=[c for c in ("label",) if c in xm.columns]),
                  on="host", how="left")
    ok = pd.to_numeric(tr.ra, errors="coerce").notna() & pd.to_numeric(tr.dec, errors="coerce").notna()
    tr = tr[ok].reset_index(drop=True)
    tr["absb"], tr["gl"] = galb(tr.ra.values, tr.dec.values)
    y = tr.label.to_numpy()
    print(f"training with coords: {len(tr)} ({int((y==1).sum())} pos / {int((y==0).sum())} neg)\n", flush=True)

    inband = (tr.absb >= BAND_LO) & (tr.absb <= BAND_HI)
    res["band_retention"] = {
        "n_in_band": int(inband.sum()), "pct": float(100 * inband.mean()),
        "median_absb_pos_all": float(tr.absb[y == 1].median()),
        "median_absb_neg_all": float(tr.absb[y == 0].median()),
        "median_absb_pos_band": float(tr.absb[inband & (y == 1)].median()),
        "median_absb_neg_band": float(tr.absb[inband & (y == 0)].median()),
        "n_pos_band": int((inband & (y == 1)).sum()),
        "n_neg_band": int((inband & (y == 0)).sum())}
    print("MATCHED BAND |b| in [%g, %g]: %d/%d stars (%.1f%%)" % (
        BAND_LO, BAND_HI, inband.sum(), len(tr), 100 * inband.mean()))
    print("  median |b|  ALL : pos %.1f  neg %.1f" % (
        res["band_retention"]["median_absb_pos_all"], res["band_retention"]["median_absb_neg_all"]))
    print("  median |b| BAND : pos %.1f  neg %.1f   <- inversion check\n" % (
        res["band_retention"]["median_absb_pos_band"], res["band_retention"]["median_absb_neg_band"]), flush=True)

    # ---------------- (A) CLUSTER MEMBERSHIP vs sky position ----------------
    print("=== (A) CLUSTER MEMBERSHIP ===", flush=True)
    cl = {}
    for col in ["HuntReffert2023", "any_cluster"]:
        if col not in tr.columns:
            continue
        f = tr[col].fillna(False).astype(bool).to_numpy()
        rho = float(spearmanr(f.astype(float), tr.absb.to_numpy()).statistic)
        # is membership a smooth function of |b|? member rate by |b| quartile
        q = pd.qcut(tr.absb, 4, labels=False, duplicates="drop")
        by_q = [float(100 * f[q == i].mean()) for i in range(4)]
        e = {"n_members": int(f.sum()), "rate_pct": float(100 * f.mean()),
             "rho_vs_absb": rho, "member_rate_by_absb_quartile_pct": by_q,
             "auc_all": auc(y, f.astype(float)),
             "n_members_in_band": int((f & inband.to_numpy()).sum()),
             "n_members_pos_band": int((f & inband.to_numpy() & (y == 1)).sum()),
             "n_members_neg_band": int((f & inband.to_numpy() & (y == 0)).sum())}
        ib = inband.to_numpy()
        e["auc_in_band"] = auc(y[ib], f[ib].astype(float))
        cl[col] = e
        print(f"  {col}: {e['n_members']} members ({e['rate_pct']:.2f}%), "
              f"rho vs |b| {rho:+.3f}, rate by |b| quartile {['%.2f'%v for v in by_q]}")
        print(f"    AUC(flag) all {e['auc_all']:.4f} | in band {e['auc_in_band']:.4f} "
              f"({e['n_members_in_band']} members in band: "
              f"{e['n_members_pos_band']} pos / {e['n_members_neg_band']} neg)", flush=True)
    res["cluster"] = cl

    # ---------------- (B) 2D DUST: real E(B-V) from IRSA -------------------
    print("\n=== (B) 2D DUST -- real SFD98 / SandF2011 E(B-V), all-sky ===", flush=True)
    neg_idx = np.where(y == 0)[0]
    pos_idx = np.where(y == 1)[0]
    pos_pick = rng.choice(pos_idx, size=min(N_POS_SAMPLE, len(pos_idx)), replace=False)
    samp = np.sort(np.concatenate([neg_idx, pos_pick]))
    s = tr.iloc[samp].reset_index(drop=True)
    print(f"  balanced sample: {len(s)} stars "
          f"({int((s.label==1).sum())} pos / {int((s.label==0).sum())} neg)", flush=True)
    sandf, sfd = fetch_ebv(s.ra.values, s.dec.values, "training")
    s["ebv_sandf"] = sandf
    s["ebv_sfd"] = sfd
    ys = s.label.to_numpy()

    d = {"n": int(len(s)), "n_finite": int(np.isfinite(sandf).sum()),
         "coverage_pct": float(100 * np.isfinite(sandf).mean())}
    for col in ["ebv_sandf", "ebv_sfd"]:
        v = s[col].to_numpy()
        m = np.isfinite(v)
        rho_b = float(spearmanr(v[m], s.absb.to_numpy()[m]).statistic)
        d[col] = {
            "rho_vs_absb": rho_b,
            "auc_all": auc(ys, v),
            "median_pos": float(np.nanmedian(v[ys == 1])),
            "median_neg": float(np.nanmedian(v[ys == 0]))}
        ib = ((s.absb >= BAND_LO) & (s.absb <= BAND_HI)).to_numpy()
        d[col]["auc_in_band"] = auc(ys[ib], v[ib])
        d[col]["n_in_band"] = int(ib.sum())
        d[col]["median_pos_band"] = float(np.nanmedian(v[ib & (ys == 1)]))
        d[col]["median_neg_band"] = float(np.nanmedian(v[ib & (ys == 0)]))
        print(f"  {col}: coverage {d['coverage_pct']:.2f}%  rho vs |b| {rho_b:+.4f}")
        print(f"    AUC  ALL {d[col]['auc_all']:.4f} (median pos {d[col]['median_pos']:.4f} / "
              f"neg {d[col]['median_neg']:.4f})")
        print(f"    AUC BAND {d[col]['auc_in_band']:.4f}  n={d[col]['n_in_band']} "
              f"(median pos {d[col]['median_pos_band']:.4f} / neg {d[col]['median_neg_band']:.4f})", flush=True)
    # reference: what |b| ITSELF scores, the thing dust would have to beat
    d["reference_auc_absb"] = auc(ys, s.absb.to_numpy())
    ib = ((s.absb >= BAND_LO) & (s.absb <= BAND_HI)).to_numpy()
    d["reference_auc_absb_in_band"] = auc(ys[ib], s.absb.to_numpy()[ib])
    print(f"  REFERENCE |b| itself: AUC all {d['reference_auc_absb']:.4f}, "
          f"in band {d['reference_auc_absb_in_band']:.4f}", flush=True)
    res["dust_2d"] = d

    s[["host", "label", "ra", "dec", "absb", "gl", "ebv_sandf", "ebv_sfd"]].to_csv(OUT_CSV, index=False)
    json.dump(res, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\nsaved {OUT_JSON}\nsaved {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
