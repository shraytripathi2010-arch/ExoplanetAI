"""binary_composite_assess.py -- battery for the P(binary) composite and the
two genuinely-new binary-detection inputs.

SB9 is NOT here: 6 matches in 5,534 rows (0.11%). Closed on scale.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, "..", "..")
RAW = os.path.join(HERE, "binary_composite_raw.csv")
OUT = os.path.join(HERE, "binary_composite_assess.json")
OUT_CSV = os.path.join(HERE, "binary_composite_features.csv")
NEW = ["bin_pcomposite", "bin_epsi", "bin_sepsi", "bin_tic_gaia_dmag", "bin_tmag_gmag"]
BAND = (8.0, 40.0)


def auc(y, v):
    v = pd.to_numeric(pd.Series(np.asarray(v, float)), errors="coerce").replace([np.inf,-np.inf], np.nan)
    ok = v.notna().to_numpy(); yy = np.asarray(y)
    if ok.sum() < 50 or len(np.unique(yy[ok])) < 2: return np.nan
    return float(roc_auc_score(yy[ok], v[ok]))


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m05", os.path.join(ROOT,"code","05_train_models.py"))
    m05 = importlib.util.module_from_spec(spec); sys.modules["m05"] = m05; spec.loader.exec_module(m05)
    cols33 = list(m05.FEATURE_COLUMNS)
    tr = pd.read_csv(os.path.join(ROOT,"data","training_dataset","training.csv"))
    raw = pd.read_csv(RAW)
    assert len(tr) == len(raw)
    df = pd.concat([tr.reset_index(drop=True),
                    raw[["gaia_epsi","gaia_sepsi","gaia_gmag2","tic_tmag","tic_gaiamag","sb9_hit"]].reset_index(drop=True)], axis=1)
    y = df.label.to_numpy()

    # composite: OUT-OF-FOLD so it never sees its own row's label
    X = df[["gaia_ruwe","gaia_nss"]].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    pipe = Pipeline([("i",SimpleImputer(strategy="median")),("lr",LogisticRegression(max_iter=2000))])
    df["bin_pcomposite"] = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5,shuffle=True,random_state=0),
                                             method="predict_proba")[:,1]
    df["bin_epsi"] = pd.to_numeric(df.gaia_epsi, errors="coerce")
    df["bin_sepsi"] = pd.to_numeric(df.gaia_sepsi, errors="coerce")
    df["bin_tic_gaia_dmag"] = pd.to_numeric(df.tic_gaiamag,errors="coerce") - pd.to_numeric(df.gaia_gmag2,errors="coerce")
    df["bin_tmag_gmag"] = pd.to_numeric(df.tic_tmag,errors="coerce") - pd.to_numeric(df.gaia_gmag2,errors="coerce")
    df.to_csv(OUT_CSV, index=False)
    res = {}

    print("="*92); print("COVERAGE + CLASS-RATE GATE"); print("="*92)
    print(f"{'feature':<22}{'cov':>8}{'avail pos':>11}{'avail neg':>11}{'fisher p':>11}{'AUC(avail)':>12}")
    for c in NEW:
        a = pd.to_numeric(df[c],errors="coerce").replace([np.inf,-np.inf],np.nan).notna().to_numpy()
        ap,an = float(a[y==1].mean()), float(a[y==0].mean())
        tab=[[int((a&(y==1)).sum()),int((~a&(y==1)).sum())],[int((a&(y==0)).sum()),int((~a&(y==0)).sum())]]
        try: orr,p = fisher_exact(tab)
        except Exception: orr,p = np.nan,np.nan
        aa = auc(y, a.astype(float))
        res.setdefault("gate",{})[c] = {"cov":float(a.mean()),"pos":ap,"neg":an,"p":float(p),"auc_avail":aa}
        print(f"{c:<22}{a.mean():>7.2%}{ap:>11.2%}{an:>11.2%}{p:>11.3g}{aa:>12.4f}")

    print("\n"+"="*92); print("SINGLE-FEATURE AUC and CORRELATION vs the DEPLOYED Gaia pair"); print("="*92)
    r = pd.to_numeric(df.gaia_ruwe,errors="coerce"); n = pd.to_numeric(df.gaia_nss,errors="coerce")
    X33 = df[cols33].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    print(f"{'feature':<22}{'AUC':>9}{'|AUC-.5|':>10}{'rho ruwe':>10}{'rho nss':>9}{'max/33':>9}  which")
    for c in NEW:
        v = pd.to_numeric(df[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
        a = auc(y,v)
        rr = {k: float(abs(v.corr(X33[k],method="spearman"))) for k in cols33 if np.isfinite(v.corr(X33[k],method="spearman"))}
        mk = max(rr,key=rr.get) if rr else None
        res.setdefault("single",{})[c] = {"auc":a,"rho_ruwe":float(abs(v.corr(r,method="spearman"))),
                                          "rho_nss":float(abs(v.corr(n,method="spearman"))),
                                          "max33":rr.get(mk),"with":mk}
        flag = "  REDUNDANT" if (mk and rr[mk]>0.80) else ""
        print(f"{c:<22}{a:>9.4f}{abs(a-0.5):>10.4f}{abs(v.corr(r,method='spearman')):>10.3f}"
              f"{abs(v.corr(n,method='spearman')):>9.3f}{rr.get(mk,float('nan')):>9.3f}  {mk}{flag}")

    print("\n"+"="*92); print("MATCHED-SKY-BAND TEST (astrometry-derived -> required)"); print("="*92)
    from astropy.coordinates import SkyCoord; import astropy.units as u
    ra=pd.to_numeric(df.ra,errors="coerce"); dec=pd.to_numeric(df.dec,errors="coerce")
    ok=ra.notna()&dec.notna(); b=pd.Series(np.nan,index=df.index)
    b[ok]=np.abs(SkyCoord(ra[ok].values*u.deg,dec[ok].values*u.deg,frame="icrs").galactic.b.deg)
    inb=((b>=BAND[0])&(b<=BAND[1])).to_numpy()
    print(f"  band retains {int(inb.sum())}/{len(df)}; median |b| in-band planets "
          f"{float(b[inb&(y==1)].median()):.1f} vs FPs {float(b[inb&(y==0)].median()):.1f} (inverted)")
    print(f"{'feature':<22}{'AUC full':>10}{'AUC in-band':>13}{'retained':>11}   verdict")
    for c in NEW:
        v = pd.to_numeric(df[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
        a_all,a_in = auc(y,v), auc(y[inb], v.to_numpy()[inb])
        s_all,s_in = abs(a_all-0.5), abs(a_in-0.5)
        keep = s_in/s_all if s_all>0 else np.nan
        flip = (a_all-0.5)*(a_in-0.5) < 0
        res.setdefault("band",{})[c]={"auc_all":a_all,"auc_in":a_in,"retained":float(keep),"flip":bool(flip)}
        print(f"{c:<22}{a_all:>10.4f}{a_in:>13.4f}{keep:>10.0%}   "
              f"{'SIGN FLIPS' if flip else ('survives' if keep>0.5 else 'mostly positional')}")

    print()
    import control_arms
    res["controls"] = control_arms.both_controls(df, y, NEW, verbose=True)
    json.dump(res, open(OUT,"w"), indent=2, default=float)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
