"""k2_domain_check.py -- the stop/go gate for the K2 pilot.

THIS RUNS BEFORE ANYTHING IS MERGED OR SCALED.

The question is not "did the pilot work" but "is K2 the same KIND of data".
This project has measured that twice and both answers were bad:

    synthetic injected transits   domain AUC 0.9654  -> mixing HURT (-0.018)
    TESS FFI coarse cadence       domain AUC 0.9717  -> inert outside its
                                                        own rows (-0.0018 on 2-min)

If K2 lands in that range, the pre-registered decision is to STOP rather than
scale, because the two prior results say what happens next and it is not a
gain. The threshold is declared here before the number is known.

Uses domain_separability.domain_report, the same function that reproduces both
reference numbers to zero delta, so this AUC is directly comparable to them
rather than merely similar-looking.
"""
import os
import sys
import json
import importlib.util
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(PILOT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
EXP_DIR = os.path.join(CODE_DIR, "experiments")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, EXP_DIR)

FEATURES = os.path.join(PILOT_DIR, "k2_pilot_features.csv")
SAMPLE = os.path.join(PILOT_DIR, "k2_pilot_sample.csv")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(EXP_DIR, "cadence_per_star.csv")
RESULTS = os.path.join(PILOT_DIR, "k2_domain_check.json")

# Pre-registered, stated before the measurement.
STOP_BAND_LO = 0.95


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def main():
    from domain_separability import domain_report, REFERENCE

    m05 = _m05()
    k2 = pd.read_csv(FEATURES)
    k2 = k2[k2["status"] == "Success"].reset_index(drop=True)
    # st_teff is a model feature but was not carried through the TLS worker
    # (which only needs r_star/m_star for the period grid). Pull it, and the
    # authoritative stellar params, from the sample file the archive supplied.
    smp = pd.read_csv(SAMPLE)[["host", "st_rad", "st_mass", "st_teff"]]
    k2 = k2.drop(columns=[c for c in ("st_rad", "st_mass", "st_teff")
                          if c in k2.columns]).merge(smp, on="host", how="left")
    if len(k2) < 30:
        print(f"Only {len(k2)} successful K2 rows -- below the 30-row minimum "
              f"for a domain classifier. Cannot measure.")
        return

    tess = pd.read_csv(TRAINING)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(tess.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy()
    tess2 = tess[is2].reset_index(drop=True)

    print("=" * 84)
    print("K2 vs TESS 2-min -- DOMAIN SEPARABILITY")
    print("=" * 84)
    print(f"  TESS 2-min rows : {len(tess2)}")
    print(f"  K2 pilot rows   : {len(k2)}  "
          f"({int((k2.label==1).sum())} pos / {int((k2.label==0).sum())} neg)")
    print(f"\n  pre-registered STOP band: domain AUC >= {STOP_BAND_LO}")
    for k, v in REFERENCE.items():
        print(f"    {k:<26} {v['auc']:.4f}  ->  {v['outcome']}")
    print()

    # Both sides through the SAME feature builder.
    Xt, yt = m05.build_feature_matrix(tess2)
    Xk, yk = m05.build_feature_matrix(k2)
    Xt, Xk = Xt.reset_index(drop=True), Xk.reset_index(drop=True)
    Xk = Xk[Xt.columns]
    X = pd.concat([Xt, Xk], ignore_index=True)
    dom = np.r_[np.zeros(len(Xt), int), np.ones(len(Xk), int)]
    y = np.r_[np.asarray(yt), np.asarray(yk)]

    rep = domain_report(X, dom, y=y, names=("TESS 2-min", "K2"))
    auc = rep["domain_auc"]

    print("\n" + "=" * 84)
    if auc is None:
        verdict = "NOT MEASURABLE -- too few rows on one side"
    elif auc >= STOP_BAND_LO:
        verdict = (f"STOP. domain AUC {auc:.4f} is at or above the "
                   f"pre-registered {STOP_BAND_LO} band, i.e. in the same "
                   f"regime as synthetic (0.9654, harmful) and FFI (0.9717, "
                   f"inert). Do not scale.")
    elif auc >= 0.90:
        verdict = (f"CAUTION. domain AUC {auc:.4f} -- below the stop band but "
                   f"far above indistinguishable. Any pooled gain must be "
                   f"checked on the TESS-only subset before it is believed.")
    else:
        verdict = (f"PROCEED. domain AUC {auc:.4f} -- K2 is materially less "
                   f"separable than synthetic or FFI were.")
    print(verdict)
    print("=" * 84)

    rep["verdict"] = verdict
    rep["stop_band"] = STOP_BAND_LO
    rep["n_k2"] = int(len(k2))
    rep["n_tess_2min"] = int(len(tess2))
    with open(RESULTS, "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
