"""Restricted-population value test: holding availability CONSTANT, is there any
signal in the DV centroid significance itself? This is the analysis that
disambiguated the multi-sector feature (+0.0094 raw -> +0.0021 restricted)."""
import os, re, urllib.request, warnings, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")
SCR = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/anujtripathi/Developer/ExoplanetAI"
URL = ("https://archive.stsci.edu/missions/tess/download_scripts/sector/"
       "tesscurl_sector_{}_dv.sh")

def fetch(s):
    try:
        b = urllib.request.urlopen(URL.format(s), timeout=120).read().decode("utf-8","replace")
    except Exception:
        return {}
    out = {}
    for fn in re.findall(r"(tess\d+-s\d+-s\d+-(\d{16})-\d+_dvr\.xml)", b):
        out.setdefault(int(fn[1]), fn[0])
    return out

print("building TIC -> dvr.xml filename map...", flush=True)
fmap = {}
with ThreadPoolExecutor(max_workers=8) as ex:
    for d in ex.map(fetch, range(1, 106)):
        for k, v in d.items():
            fmap.setdefault(k, v)
print(f"  {len(fmap):,} TICs mapped")

av = pd.read_csv(os.path.join(ROOT, "code/experiments/dv_star_availability.csv"))
tr = av[(av._set == "train") & av.has_dv].dropna(subset=["label"])
rng = np.random.default_rng(11)
pos = tr[tr.label == 1].sample(min(300, int((tr.label == 1).sum())), random_state=11)
neg = tr[tr.label == 0].sample(min(300, int((tr.label == 0).sum())), random_state=11)
samp = pd.concat([pos, neg])
print(f"stratified sample: {len(pos)} pos / {len(neg)} neg from {len(tr)} DV-available")

def grab(row):
    tic = int(row.tic)
    fn = fmap.get(tic)
    if not fn:
        return None
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS/product/" + fn
    try:
        s = urllib.request.urlopen(url, timeout=120).read().decode("utf-8","replace")
    except Exception:
        return None
    out = {"host": row.host, "label": row.label, "tic": tic}
    for tag in ("meanSkyOffset","meanRaOffset","meanDecOffset"):
        vals=[]
        for m in re.finditer(r'<dv:'+tag+r'([^>]*)/>', s):
            a=dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            try:
                v=float(a.get("value")); u=float(a.get("uncertainty"))
                if u>0: vals.append((v,u,v/u))
            except Exception: pass
        if vals:
            out[tag+"_val"]=float(np.median([x[0] for x in vals]))
            out[tag+"_unc"]=float(np.median([x[1] for x in vals]))
            out[tag+"_sig"]=float(np.median([x[2] for x in vals]))
    return out

rows=[]
with ThreadPoolExecutor(max_workers=10) as ex:
    for i,r in enumerate(ex.map(grab, [r for _,r in samp.iterrows()]),1):
        if r: rows.append(r)
        if i%100==0: print(f"  {i}/{len(samp)}  ({len(rows)} parsed)", flush=True)
d=pd.DataFrame(rows)
d.to_csv(os.path.join(SCR,"dv_values.csv"), index=False)
print(f"\nparsed {len(d)} stars ({int((d.label==1).sum())} pos / {int((d.label==0).sum())} neg)")
from sklearn.metrics import roc_auc_score
print("\nRESTRICTED-POPULATION single-feature AUC (availability held constant):")
y=d.label.to_numpy()
for c in [c for c in d.columns if c.endswith(("_val","_unc","_sig"))]:
    v=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    m=v.notna().to_numpy()
    if m.sum()>50 and len(np.unique(y[m]))>1:
        a=roc_auc_score(y[m],v[m])
        print(f"  {c:<24} n={int(m.sum()):<5} AUC {a:.4f}   |AUC-0.5| {abs(a-0.5):.4f}")
print("\nmedians:")
for c in [c for c in d.columns if c.endswith("_sig")]:
    v=pd.to_numeric(d[c],errors="coerce")
    print(f"  {c:<24} pos {v[y==1].median():>9.3f}   neg {v[y==0].median():>9.3f}")
