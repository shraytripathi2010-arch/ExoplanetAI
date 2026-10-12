"""conformal_diagnostics.py -- the two measured limits of the conformal sets,
written to models/conformal_diagnostics.json for the app to quote.

conformal_calibration.json holds the THRESHOLDS (conformal_prediction.py). This
holds what is known about where they stop being guaranteed. web/conformal.py
reads it instead of hard-coding numbers, and the page calls it stale if
model_md5 differs from models/best_model.joblib.

  domain_shift          How separable the calibration stars are from the
                        candidate pools under the production model's own
                        features: the same check as conformal_prediction.py's
                        exchangeability section (domain_separability.domain_report;
                        calibration = the frozen test split, candidates =
                        candidate_pool.load_scored_pool for each pool).
  fresh_label_coverage  Coverage of the DEPLOYED thresholds on labelled rows
                        added to training.csv after the calibration snapshot.
                        The production model was never trained on them, and they
                        are not in the calibration set.

The calibration set is the test split of the training.csv that production was
fit on (--snapshot-rev, read with `git show`). Today's training.csv assigns the
rows added later by a host hash, so its test split is not the calibration set.

Usage: python3 code/experiments/conformal_diagnostics.py [--snapshot-rev 3329ad18]
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MODEL = os.path.join(ROOT, "models", "best_model.joblib")
THRESHOLDS = os.path.join(ROOT, "models", "conformal_calibration.json")
OUT = os.path.join(ROOT, "models", "conformal_diagnostics.json")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
SNAPSHOT_PATH = "data/training_dataset/training.csv"


def _m05():
    spec = importlib.util.spec_from_file_location("m05", os.path.join(ROOT, "code", "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--snapshot-rev", default="3329ad18",
                    help="commit holding the training.csv the production model was fit on")
    a = ap.parse_args()

    with open(MODEL, "rb") as f:
        blob = f.read()
    md5 = hashlib.md5(blob).hexdigest()
    model = joblib.load(io.BytesIO(blob))
    art = json.load(open(THRESHOLDS))
    if art.get("model_md5") != md5:
        raise SystemExit(f"conformal_calibration.json is for {art.get('model_md5')}, production is {md5} -- "
                         f"rebuild the thresholds (conformal_prediction.py) first.")
    m05 = _m05()

    raw = subprocess.run(["git", "-C", ROOT, "show", f"{a.snapshot_rev}:{SNAPSHOT_PATH}"],
                         capture_output=True, text=True, check=True).stdout
    snap = pd.read_csv(io.StringIO(raw))
    X, y = m05.build_feature_matrix(snap)
    tr, te = m05.split_by_host(snap)
    Xte, yte = X[te].reset_index(drop=True), y[te].to_numpy()
    if (len(yte), int(yte.sum()), int((yte == 0).sum())) != \
            (art["n_calibration"], art["n_calibration_pos"], art["n_calibration_neg"]):
        raise SystemExit(f"the snapshot's test split ({len(yte)} rows) is not the calibration set recorded "
                         f"in conformal_calibration.json ({art['n_calibration']}); wrong --snapshot-rev")
    feature_columns = list(X.columns)

    # ---- domain shift, as conformal_prediction.py measures it
    sys.path.insert(0, HERE)
    from conformal_prediction import load_unknown_candidates
    from domain_separability import domain_report
    u, prov = load_unknown_candidates(m05, verbose=False)
    Xu, _ = m05.build_feature_matrix(u.assign(label=0))
    Xu = Xu.reset_index(drop=True)[feature_columns]
    Xc = pd.concat([Xte, Xu], ignore_index=True)
    dom = np.r_[np.zeros(len(Xte), int), np.ones(len(Xu), int)]
    rep = domain_report(Xc, dom, names=("calibration (frozen test split)", "candidate pools"), verbose=False)

    # ---- coverage of the deployed thresholds on rows added after the snapshot
    cur = pd.read_csv(TRAINING)
    new = cur[~cur["host"].astype(str).isin(set(snap["host"].astype(str)))]
    Xn, yn = m05.build_feature_matrix(new)
    yn = yn.to_numpy()
    pn = model.predict_proba(Xn[feature_columns])[:, 1]
    by_alpha = {}
    for alpha, t in art["thresholds"].items():
        cov = np.where(yn == 1, (1 - pn) <= t["q_pos"], pn <= t["q_neg"])
        by_alpha[alpha] = {"target": round(1 - float(alpha), 4),
                           "planet": float(cov[yn == 1].mean()) if (yn == 1).any() else None,
                           "not_planet": float(cov[yn == 0].mean()) if (yn == 0).any() else None}

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "generated_by": "code/experiments/conformal_diagnostics.py",
        "model_md5": md5,
        "n_features": len(feature_columns),
        "calibration": {"snapshot_rev": a.snapshot_rev,
                        "training_csv_md5": hashlib.md5(raw.encode()).hexdigest(),
                        "n": int(len(yte)), "n_planet": int(yte.sum()), "n_not_planet": int((yte == 0).sum())},
        "guarantee_scope": "frozen test split of the calibration snapshot",
        "domain_shift": {"domain_auc": rep.get("domain_auc"), "n_calibration": int(len(Xte)),
                         "n_candidates": int(len(Xu)),
                         "candidate_sources": [p["sources"] for p in prov],
                         "method": "domain_separability.domain_report (same as conformal_prediction.py)"},
        "fresh_label_coverage": {"source": f"training.csv rows absent from {a.snapshot_rev}'s training.csv",
                                 "n_planet": int((yn == 1).sum()), "n_not_planet": int((yn == 0).sum()),
                                 "by_alpha": by_alpha},
    }
    with open(OUT + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(json.dumps({k: out[k] for k in ("model_md5", "n_features", "calibration")}, indent=1))
    print(f"domain AUC {out['domain_shift']['domain_auc']:.4f} "
          f"({out['domain_shift']['n_calibration']} calibration vs {out['domain_shift']['n_candidates']} candidates)")
    for al, v in by_alpha.items():
        print(f"fresh-label coverage at {1 - float(al):.0%}: planet {v['planet']:.3f} (n={out['fresh_label_coverage']['n_planet']}), "
              f"not planet {v['not_planet']:.3f} (n={out['fresh_label_coverage']['n_not_planet']})")
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
