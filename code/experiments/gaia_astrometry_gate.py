"""gaia_astrometry_gate.py -- PART 0 availability-trap gate for RUWE and NSS.

THE TRAP THIS EXISTS TO CATCH
-----------------------------
During the crowding deployment, TIC's own `contratio`/`numcont` turned out to be
populated ONLY for Candidate Target List stars. That made *mere availability*
predictive of the label (AUC 0.3775) and produced a severe train/serve mismatch
(37.5% available on real candidates vs 78.7% on training negatives). The same
shape recurred for stellar density's `rho`/`logg`.

Gaia completeness is also not uniform across TESS targets, so RUWE and NSS get
the identical check BEFORE any modelling:

  1. availability rate by class
  2. single-feature AUC of AVAILABILITY ITSELF (not the value)
  3. the same availability rate on BOTH real candidate pools -- the train/serve
     comparison that is what actually caught the crowding trap

A feature fails this gate if availability alone carries class signal, regardless
of how good the values look.
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TRAIN = os.path.join(HERE, "gaia_astrometry_training.csv")
POOLS = os.path.join(HERE, "gaia_astrometry_pools.csv")
OUT = os.path.join(HERE, "gaia_astrometry_gate.json")

FIELDS = ["gaia_ruwe", "gaia_nss"]
CROWDING_TRAP_AUC = 0.3775          # the closed CTL case, for scale
RUWE_THRESHOLD = 1.4


def main():
    from sklearn.metrics import roc_auc_score
    from scipy.stats import fisher_exact, mannwhitneyu

    tr = pd.read_csv(TRAIN)
    y = tr.label.to_numpy()
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    print(f"training {len(tr)} rows: {npos} positive / {nneg} negative")
    print(f"Gaia matched (any field): {tr.gaia_source.notna().mean():.2%}\n")

    res = {}
    print("=" * 82)
    print("GATE 1 -- IS AVAILABILITY ITSELF PREDICTIVE? (the CTL/density trap)")
    print("=" * 82)
    print(f"{'field':<14}{'avail pos':>11}{'avail neg':>11}{'OR':>8}{'p':>12}"
          f"{'AUC(avail)':>12}{'|AUC-.5|':>10}")
    for f in FIELDS:
        a = tr[f].notna().to_numpy()
        ap, an = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
        orr, p = fisher_exact([[ap, npos - ap], [an, nneg - an]])
        auc_avail = roc_auc_score(y, a.astype(float))
        res[f] = {"avail_pos": ap / npos, "avail_neg": an / nneg,
                  "fisher_or": float(orr), "fisher_p": float(p),
                  "auc_availability": float(auc_avail)}
        print(f"{f:<14}{ap/npos:>10.2%}{an/nneg:>11.2%}{orr:>8.3f}{p:>12.4g}"
              f"{auc_avail:>12.4f}{abs(auc_avail-0.5):>10.4f}")
    print(f"\n  for scale: the CLOSED CTL trap had AUC(availability) = {CROWDING_TRAP_AUC}"
          f"  (|AUC-0.5| = {abs(CROWDING_TRAP_AUC-0.5):.4f})")

    print("\n" + "=" * 82)
    print("GATE 2 -- TRAIN vs SERVE availability (what actually caught the crowding trap)")
    print("=" * 82)
    if os.path.exists(POOLS):
        pl = pd.read_csv(POOLS)
        for f in FIELDS:
            row = f"{f:<14}"
            row += f"train-neg {tr.loc[y == 0, f].notna().mean():>7.2%}   "
            row += f"train-pos {tr.loc[y == 1, f].notna().mean():>7.2%}   "
            for pool, g in pl.groupby("pool"):
                row += f"{pool} {g[f].notna().mean():>7.2%}   "
            print(row)
            res[f]["pools"] = {str(k): float(g[f].notna().mean())
                               for k, g in pl.groupby("pool")}
        print("\n  the CLOSED crowding trap was 37.5% on candidates vs 78.7% on "
              "training negatives")
    else:
        print("  pools file not ready yet -- rerun once the pool fetch finishes")

    print("\n" + "=" * 82)
    print("VALUE-LEVEL SIGNAL (only meaningful if the gates above pass)")
    print("=" * 82)
    print(f"{'field':<20}{'AUC':>9}{'|AUC-.5|':>10}{'median pos':>13}{'median neg':>13}"
          f"{'MWU p':>12}")
    for f in FIELDS + ["gaia_ruwe_high", "gaia_nss_flag"]:
        v = pd.to_numeric(tr[f], errors="coerce")
        ok = v.notna().to_numpy()
        if ok.sum() < 100:
            continue
        auc = roc_auc_score(y[ok], v[ok])
        try:
            mp = mannwhitneyu(v[ok & (y == 1)], v[ok & (y == 0)]).pvalue
        except Exception:
            mp = float("nan")
        res.setdefault(f, {})["auc_value"] = float(auc)
        res[f]["median_pos"] = float(v[ok & (y == 1)].median())
        res[f]["median_neg"] = float(v[ok & (y == 0)].median())
        print(f"{f:<20}{auc:>9.4f}{abs(auc-0.5):>10.4f}"
              f"{v[ok & (y==1)].median():>13.4f}{v[ok & (y==0)].median():>13.4f}{mp:>12.3g}")

    r = pd.to_numeric(tr.gaia_ruwe, errors="coerce")
    n = pd.to_numeric(tr.gaia_nss, errors="coerce")
    print(f"\n  RUWE > {RUWE_THRESHOLD}: {(r > RUWE_THRESHOLD).sum()} stars "
          f"({(r > RUWE_THRESHOLD).mean():.2%}) -- "
          f"{(r[y==1] > RUWE_THRESHOLD).mean():.2%} of positives, "
          f"{(r[y==0] > RUWE_THRESHOLD).mean():.2%} of negatives")
    print(f"  NSS > 0: {int((n > 0).sum())} stars ({(n > 0).mean():.2%}) -- "
          f"{(n[y==1] > 0).mean():.2%} of positives, {(n[y==0] > 0).mean():.2%} of negatives")

    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
