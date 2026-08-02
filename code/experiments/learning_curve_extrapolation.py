"""learning_curve_extrapolation.py -- how much real data would the ceiling
actually need?

WHY THIS EXISTS

The injection-recovery diagnostic could not answer "data-starved or
feature-starved" because its synthetic data is off-distribution (real-vs-
synthetic discriminator AUC 0.968), which confounds the null. The
unconfounded version of the same question needs no synthetic data at all:
subsample the REAL training set and watch what held-out performance does.

A first four-point pass came back monotonic and still climbing at 100%
(0.8622 -> 0.8852 -> 0.8954 -> 0.8996), with per-doubling gains of +0.0230
then +0.0144. That is not the flat curve this project's premise assumed, so it
is worth measuring carefully rather than eyeballing four points -- especially
because the top point had no subsampling replicates and its +0.0042 last-step
gain sat inside the neighbouring point's own scatter.

METHOD

Ten sample sizes, several independent host-level subsamples each, all scored
against the SAME frozen real test set. Subsampling is by HOST rather than by
row so a star is never split across sample sizes, and class balance is
preserved within each draw so a small sample cannot look bad merely for having
drifted to a different prior.

The curve is then fitted with the standard saturating power law

    AUC(n) = a - b * n^(-c),      a = asymptotic ceiling

which is the usual functional form for learning curves and, unlike a straight
line on log-n, has an explicit asymptote -- the number actually being asked
for. Uncertainty comes from refitting over bootstrap resamples of the measured
points, so the reported ceiling carries an interval rather than a single
falsely-precise value.

WHAT THIS CAN AND CANNOT SAY

It can say how this model improves with more data DRAWN FROM THE SAME
DISTRIBUTION. It cannot say that Kepler supplies that: Kepler is a different
mission with different cadence, bandpass and noise, and this project's own
Kepler pilot already hit a real SNR wall at 36.4% yield plus a mission-identity
leakage concern. So the extrapolation is best read as an UPPER BOUND on what
additional same-distribution TESS data would buy, and any Kepler plan has to
clear the transfer problem separately before this curve applies to it.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from scipy.optimize import curve_fit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "learning_curve_extrapolation.json")
CURVE_CSV = os.path.join(SCRIPT_DIR, "learning_curve_points.csv")

RANDOM_SEED = 42
FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 0.90, 0.95, 1.00]
N_REPEATS = 7
# Sizes to extrapolate to. 9,564 is roughly the number of dispositioned KOIs
# (confirmed + false positive) available from the Kepler archive; 20,000 is a
# deliberately optimistic "everything, both missions" figure included to show
# where the curve goes even under generous assumptions.
EXTRAPOLATE_TO = [9564, 20000, 50000]


def _load_m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", HistGradientBoostingClassifier(
                         max_iter=300, max_depth=4, learning_rate=0.05,
                         class_weight="balanced", random_state=RANDOM_SEED))])


def power_law(n, a, b, c):
    return a - b * np.power(n, -c)


def main():
    res = {}
    m05 = _load_m05()
    real = pd.read_csv(TRAINING_CSV)
    X, y = m05.build_feature_matrix(real)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    tr, te = m05.split_by_host(real)
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    hosts = real.loc[tr, "host"].astype(str).to_numpy()
    # host -> label, for class-balanced host sampling
    host_label = {}
    for h, lab in zip(hosts, y_tr):
        host_label.setdefault(h, lab)
    pos_hosts = np.array([h for h, l in host_label.items() if l == 1])
    neg_hosts = np.array([h for h, l in host_label.items() if l == 0])
    print("=" * 78)
    print("REAL-DATA LEARNING CURVE + EXTRAPOLATION")
    print("=" * 78)
    print(f"train {tr.sum()} rows / {len(host_label)} hosts "
          f"({len(pos_hosts)} positive-host, {len(neg_hosts)} negative-host)")
    print(f"test  {te.sum()} rows (frozen, never subsampled)\n")

    rows = []
    for frac in FRACTIONS:
        reps = 1 if frac >= 1.0 else N_REPEATS
        aucs, ns = [], []
        for rep in range(reps):
            r = np.random.RandomState(RANDOM_SEED + 1000 * rep + int(frac * 100))
            npos = max(2, int(round(frac * len(pos_hosts))))
            nneg = max(2, int(round(frac * len(neg_hosts))))
            pick = set(r.choice(pos_hosts, npos, replace=False)) | \
                   set(r.choice(neg_hosts, nneg, replace=False))
            sel = np.array([h in pick for h in hosts])
            ys = y_tr[sel]
            if len(np.unique(ys)) < 2:
                continue
            m = pipe().fit(X_tr[sel], ys)
            aucs.append(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
            ns.append(int(sel.sum()))
        if not aucs:
            continue
        n_mean = float(np.mean(ns))
        rows.append({"fraction": frac, "n_rows": n_mean,
                     "mean_auc": float(np.mean(aucs)),
                     "std_auc": float(np.std(aucs)), "n_reps": len(aucs)})
        print(f"  {100*frac:5.1f}%  n={n_mean:7.0f}  AUC {np.mean(aucs):.4f}"
              + (f" +/- {np.std(aucs):.4f}" if len(aucs) > 1 else "  (single draw)"))

    curve = pd.DataFrame(rows)
    curve.to_csv(CURVE_CSV, index=False)
    res["points"] = rows

    # ---------------- FIT ----------------
    n_arr = curve["n_rows"].to_numpy(float)
    a_arr = curve["mean_auc"].to_numpy(float)
    print("\n" + "-" * 78)
    print("saturating power-law fit:  AUC(n) = a - b * n^(-c)")
    print("-" * 78)
    try:
        popt, _ = curve_fit(power_law, n_arr, a_arr,
                            p0=[0.95, 1.0, 0.3],
                            bounds=([0.5, 0.0, 0.01], [1.0, 1e6, 3.0]),
                            maxfev=200000)
    except Exception as e:
        print(f"fit failed: {e}")
        return
    a, b, c = popt
    resid = a_arr - power_law(n_arr, *popt)
    print(f"  ceiling a = {a:.4f}   b = {b:.4g}   c = {c:.3f}   "
          f"residual RMS = {np.sqrt((resid**2).mean()):.5f}")

    # bootstrap the fit over resampled curve points
    rng = np.random.RandomState(RANDOM_SEED)
    boots = []
    for _ in range(2000):
        idx = rng.randint(0, len(n_arr), len(n_arr))
        if len(np.unique(n_arr[idx])) < 4:
            continue
        try:
            p, _ = curve_fit(power_law, n_arr[idx], a_arr[idx], p0=popt,
                             bounds=([0.5, 0.0, 0.01], [1.0, 1e6, 3.0]),
                             maxfev=20000)
            boots.append(p)
        except Exception:
            continue
    boots = np.array(boots)
    if len(boots) > 100:
        lo_a, hi_a = np.percentile(boots[:, 0], [2.5, 97.5])
        print(f"  asymptotic ceiling 95% CI: [{lo_a:.4f}, {hi_a:.4f}]  "
              f"({len(boots)} successful refits)")
        res["ceiling"] = {"estimate": float(a), "ci_lower": float(lo_a),
                          "ci_upper": float(hi_a)}
    else:
        res["ceiling"] = {"estimate": float(a)}

    n_now = float(n_arr[-1])
    auc_now = float(a_arr[-1])
    print(f"\n  current: n={n_now:.0f}, measured AUC {auc_now:.4f}, "
          f"fitted {power_law(n_now, *popt):.4f}")
    print("\n  predicted AUC if the training set grew to:")
    res["extrapolation"] = []
    for n_t in EXTRAPOLATE_TO:
        pred = float(power_law(n_t, *popt))
        if len(boots) > 100:
            pb = power_law(n_t, boots[:, 0], boots[:, 1], boots[:, 2])
            plo, phi = np.percentile(pb, [2.5, 97.5])
            print(f"    n={n_t:6d} ({n_t/n_now:4.1f}x)  AUC {pred:.4f}  "
                  f"CI [{plo:.4f}, {phi:.4f}]   gain {pred-auc_now:+.4f}")
            res["extrapolation"].append(
                {"n": n_t, "multiple": n_t / n_now, "predicted_auc": pred,
                 "ci_lower": float(plo), "ci_upper": float(phi),
                 "gain_vs_now": pred - auc_now})
        else:
            print(f"    n={n_t:6d}  AUC {pred:.4f}   gain {pred-auc_now:+.4f}")
            res["extrapolation"].append(
                {"n": n_t, "predicted_auc": pred, "gain_vs_now": pred - auc_now})

    # How much data to reach a target?
    print("\n  data required to reach a target AUC (same distribution):")
    res["required_n"] = {}
    for target in (0.91, 0.92, 0.93, 0.95):
        if target >= a:
            print(f"    {target:.2f}: UNREACHABLE -- above the fitted ceiling "
                  f"{a:.4f}")
            res["required_n"][str(target)] = None
            continue
        n_req = float(np.power((a - target) / b, -1.0 / c))
        print(f"    {target:.2f}: n ~ {n_req:,.0f} ({n_req/n_now:.1f}x current)")
        res["required_n"][str(target)] = n_req

    print("\n" + "=" * 78)
    print("CAVEAT: this describes more data FROM THE SAME DISTRIBUTION. Kepler is")
    print("a different mission (cadence, bandpass, noise), and this project's")
    print("Kepler pilot already hit a real SNR wall at 36.4% yield plus a")
    print("mission-identity leakage concern. Treat the numbers above as an upper")
    print("bound on additional TESS-like data, not as a Kepler forecast.")
    print("=" * 78)

    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}\n{CURVE_CSV}")


if __name__ == "__main__":
    main()
