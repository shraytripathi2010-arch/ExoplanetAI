"""weighting_experiments.py -- SNR/width per-example weighting and a
focal-loss-style dynamic reweighting pilot.

PART 0 ESTABLISHED WHAT IS ALREADY CLOSED (read from the write-ups):
  * per-CLASS weighting: three schemes compared; production's `balanced` was
    the best. No weighting -0.0022, sqrt-inverse-frequency -0.0021. Settled.
  * per-SUBPOPULATION weighting: giant upweight x3 gave -0.0001 (sd 0.0002),
    0/12 clearing overall AND 0/12 on the giant subpopulation itself.

Both are STATIC, pre-assigned weights. The two angles here differ:
  * SNR/width weighting is still static, but CONTINUOUS in a feature value
    rather than a per-class or per-group constant, and points the OPPOSITE
    way from the giant arm (giants are the HIGH-snr corner -- median snr 1.76x
    dwarfs -- so upweighting LOW snr is a different population).
  * Focal-style weighting is DYNAMIC: the weight depends on the model's own
    current error, not on any observable fixed in advance. Nothing in this
    project has done that.

ROUTING, verified in `weighting_routing_check.py` for sklearn 1.9.0: passing
`sample_weight` to CalibratedClassifierCV does NOT reach the trees -- sklearn
warns "sample weights will only be used for the calibration itself", and the
ranking is preserved (Spearman 0.997) while the bare pipeline genuinely changes
(0.873). Every arm here therefore fits the BARE production Pipeline with
`clf__sample_weight`, including the baseline, so all arms are like-for-like.

CONSEQUENCE FOR BRIER/ECE: these arms are UNCALIBRATED (production wraps the
pipeline in CalibratedClassifierCV(cv=5, sigmoid)). Their Brier/ECE are
internally comparable across arms but NOT comparable to production's 0.0832 /
0.0365. Stated rather than quietly reported side by side.

`class_weight="balanced"` is kept on the estimator, exactly as deployed;
sklearn multiplies sample_weight on top of it rather than replacing it.

THE HYPOTHESIS, stated before measuring, and it genuinely cuts both ways:
upweighting low-SNR examples concentrates capacity where classification is
hard and where a marginal candidate decision actually gets made. The opposite
risk is equally real -- low-SNR rows have the noisiest FEATURE VECTORS, so
upweighting them may just amplify label-independent noise and blur splits that
the high-SNR rows currently define cleanly. Prior on this family is poor: two
static-reweighting experiments have already returned null.
"""
import os
import sys
import json
import time
import importlib.util
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)
from fast_auc import fast_auc  # noqa: E402

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "weighting_experiments_results.json")

SEED, N_RESAMPLES, N_BOOT, N_WORKERS, N_ECE_BINS = 42, 12, 1500, 6, 15
MDE = 0.0097
FOCAL_GAMMA = 2.0


def ece(y, p, bins=N_ECE_BINS):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = np.linspace(0, 1, bins + 1)
    i = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    return float(sum((i == b).mean() * abs(y[i == b].mean() - p[i == b].mean())
                     for b in range(bins) if (i == b).any()))


def paired_boot(y, pa, pb, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        yi = y[i]
        if yi.sum() in (0, len(yi)):
            continue
        d.append(fast_auc(yi, pb[i]) - fast_auc(yi, pa[i]))
    d = np.asarray(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


_G = {}


def _init():
    m05 = _m05()
    df = pd.read_csv(TRAINING)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    cols = list(m05.FEATURE_COLUMNS)
    assert len(cols) == 31, f"expected 31 features, got {len(cols)}"

    tr, _ = m05.split_by_host(df)
    te = m05.frozen_test_mask(df)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"], errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()

    snr = pd.to_numeric(df["snr"], errors="coerce").to_numpy()
    dur = pd.to_numeric(df["duration"], errors="coerce").to_numpy()
    # low-SNR subpopulation on the TEST set: the bottom quartile, i.e. exactly
    # the rows the inverse-SNR weighting is meant to help.
    thr = np.nanpercentile(snr[te], 25)
    low = te & (snr <= thr)

    prod = joblib.load(PROD)
    # the BARE pipeline -- accepts clf__sample_weight; the calibration wrapper
    # does not pass weights through to the trees (verified).
    bare = clone(getattr(prod, "estimator", prod))

    _G.update(X=X[cols], y=y, cols=cols, tr=np.asarray(tr), te=te,
              te2=te & is2, low=low, snr=snr, dur=dur, bare=bare,
              low_thr=float(thr))


def _norm(w):
    """Mean-1 so effective sample size is comparable across arms."""
    w = np.clip(np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0), 1e-3, 1e3)
    return w / w.mean()


def static_weights(kind, idx):
    """Static, pre-assigned weights as a continuous function of a feature."""
    snr = _G["snr"][idx]
    dur = _G["dur"][idx]
    if kind == "base":
        return None
    if kind.startswith("snr"):
        p = {"snr_mild": 0.25, "snr_mod": 0.5, "snr_agg": 1.0}[kind]
        med = np.nanmedian(snr)
        return _norm((snr / med) ** (-p))
    if kind == "width_mod":
        med = np.nanmedian(dur)
        return _norm((dur / med) ** (-0.5))
    raise ValueError(kind)


def focal_weights(Xb, yb):
    """Focal-loss-style DYNAMIC weights, approximated the only way HGB allows.

    True focal loss reweights INSIDE the objective every boosting round.
    HistGradientBoostingClassifier accepts `loss` only from {'log_loss'} --
    no callable objective -- so that is not implementable here (verified).

    The practical approximation: fit once, obtain OUT-OF-FOLD probabilities so
    no row is scored by a model that saw it, then weight each row by
    (1 - p_true)^gamma -- the focal modulating factor -- and refit. This is one
    reweighting step rather than per-round, and is labelled as an approximation
    rather than as focal loss.
    """
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(yb))
    for a, b in skf.split(Xb, yb):
        m = clone(_G["bare"])
        m.fit(Xb.iloc[a], yb[a])
        oof[b] = m.predict_proba(Xb.iloc[b])[:, 1]
    p_true = np.where(yb == 1, oof, 1.0 - oof)
    return _norm((1.0 - p_true) ** FOCAL_GAMMA)


ARMS = ["base", "snr_mild", "snr_mod", "snr_agg", "width_mod", "focal"]


def run_resample(rep):
    if not _G:
        _init()
    X, y, tr, te, te2, low = (_G["X"], _G["y"], _G["tr"], _G["te"],
                              _G["te2"], _G["low"])
    tr_idx = np.where(tr)[0]
    rng = np.random.RandomState(1000 + rep)
    pick = rng.randint(0, len(tr_idx), len(tr_idx))
    boot = tr_idx[pick]
    Xb, yb = X.iloc[boot], y[boot]

    out = {"rep": rep, "arms": {}}
    preds = {}
    for arm in ARMS:
        w = focal_weights(Xb, yb) if arm == "focal" else static_weights(arm, boot)
        m = clone(_G["bare"])
        if w is None:
            m.fit(Xb, yb)
        else:
            m.fit(Xb, yb, clf__sample_weight=w)
        preds[arm] = {
            "full": m.predict_proba(X.loc[te])[:, 1],
            "2min": m.predict_proba(X.loc[te2])[:, 1],
            "low": m.predict_proba(X.loc[low])[:, 1]}
        out["arms"][arm] = {
            "auc": float(roc_auc_score(y[te], preds[arm]["full"])),
            "auc_low": float(roc_auc_score(y[low], preds[arm]["low"])),
            "brier": float(brier_score_loss(y[te], preds[arm]["full"])),
            "ece": ece(y[te], preds[arm]["full"]),
            "w_mean": None if w is None else float(np.mean(w)),
            "w_max": None if w is None else float(np.max(w))}

    for arm in ARMS:
        if arm == "base":
            continue
        d, lo, hi = paired_boot(y[te], preds["base"]["full"], preds[arm]["full"])
        d2, lo2, _ = paired_boot(y[te2], preds["base"]["2min"], preds[arm]["2min"])
        dl, lol, _ = paired_boot(y[low], preds["base"]["low"], preds[arm]["low"])
        out["arms"][arm].update(
            delta=d, ci_lo=lo, ci_hi=hi, clears=bool(lo > 0),
            delta_2min=d2, clears_2min=bool(lo2 > 0),
            delta_low=dl, clears_low=bool(lol > 0))
    return out


def main():
    print("=" * 112)
    print("PER-EXAMPLE WEIGHTING: continuous SNR/width, and focal-style dynamic reweighting")
    print("=" * 112)
    _init()
    print(f"  BARE pipeline + clf__sample_weight (calibration wrapper swallows weights -- verified)")
    print(f"  train {int(_G['tr'].sum())}, frozen test {int(_G['te'].sum())}, "
          f"2-min {int(_G['te2'].sum())}, LOW-SNR subpop {int(_G['low'].sum())} "
          f"(snr <= {_G['low_thr']:.2f}, bottom quartile)")
    print(f"  {N_RESAMPLES} bootstraps, arms: {ARMS}\n")

    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_resample, r): r for r in range(N_RESAMPLES)}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            print(f"  resample {futs[f]} done ({i}/{N_RESAMPLES}, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
    rows.sort(key=lambda r: r["rep"])

    ba = np.array([r["arms"]["base"]["auc"] for r in rows])
    bl = np.array([r["arms"]["base"]["auc_low"] for r in rows])
    print("\n" + "=" * 112)
    print(f"  baseline (bare, unweighted): AUC {ba.mean():.4f} (sd {ba.std():.4f})  "
          f"low-SNR AUC {bl.mean():.4f}")
    print(f"  NOTE: uncalibrated -- Brier/ECE here are NOT comparable to "
          f"production's 0.0832 / 0.0365\n")
    print(f"  {'arm':<12}{'mean d':>9}{'sd':>8}{'min':>9}{'max':>9}{'pos':>7}"
          f"{'clr':>6}{'>=MDE':>7}{'d_2min':>9}{'d_LOWSNR':>10}{'clr_low':>9}"
          f"{'Brier':>9}{'ECE':>8}")
    out = {"n_resamples": len(rows), "baseline_auc": float(ba.mean()),
           "baseline_auc_low": float(bl.mean()),
           "low_snr_threshold": _G["low_thr"], "rows": rows, "summary": {}}
    for arm in ARMS:
        if arm == "base":
            continue
        d = np.array([r["arms"][arm]["delta"] for r in rows])
        d2 = np.array([r["arms"][arm]["delta_2min"] for r in rows])
        dl = np.array([r["arms"][arm]["delta_low"] for r in rows])
        c = sum(r["arms"][arm]["clears"] for r in rows)
        cl = sum(r["arms"][arm]["clears_low"] for r in rows)
        br = np.mean([r["arms"][arm]["brier"] for r in rows])
        ec = np.mean([r["arms"][arm]["ece"] for r in rows])
        print(f"  {arm:<12}{d.mean():>+9.4f}{d.std():>8.4f}{d.min():>+9.4f}"
              f"{d.max():>+9.4f}{int((d>0).sum()):>4}/{len(d)}{c:>3}/{len(d)}"
              f"{int((d>=MDE).sum()):>4}/{len(d)}{d2.mean():>+9.4f}"
              f"{dl.mean():>+10.4f}{cl:>6}/{len(d)}{br:>9.4f}{ec:>8.4f}")
        out["summary"][arm] = {
            "delta_mean": float(d.mean()), "delta_sd": float(d.std()),
            "delta_min": float(d.min()), "delta_max": float(d.max()),
            "n_positive": int((d > 0).sum()), "n_clearing": c,
            "n_at_or_above_mde": int((d >= MDE).sum()),
            "delta_2min": float(d2.mean()),
            "delta_low_snr": float(dl.mean()), "n_clearing_low": cl,
            "mean_brier": float(br), "mean_ece": float(ec)}
    print("=" * 112)
    json.dump(out, open(RESULTS, "w"), indent=2, default=float)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
