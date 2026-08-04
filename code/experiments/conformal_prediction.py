"""conformal_prediction.py -- split conformal prediction over the DEPLOYED model.

This changes no accuracy. It wraps the existing production model's outputs with
a finite-sample coverage guarantee, and then checks empirically whether that
guarantee actually holds.

WHERE THE CALIBRATION SET COMES FROM, AND WHY THE OTHER OPTION IS INVALID

Split conformal needs calibration data the model has never seen. The two
candidates were:

  A. carve a calibration subset out of the TRAINING split
     INVALID as stated. `models/best_model.joblib` was fit on all 4,387
     training stars, so nonconformity scores computed there are optimistically
     small -- the model fits its own training data better than new data. The
     quantile would come out too tight and coverage would silently fail, which
     is exactly the failure mode this task asks to guard against. It could be
     rescued by refitting on a reduced training set, but then the conformal
     wrapper calibrates a DIFFERENT model than the one deployed, and the
     guarantee does not transfer to production.

  B. partition the frozen TEST split into calibration and evaluation halves
     CORRECT here. The production model has never seen any of these 1,098
     stars, so scores are honest.

**Option B does not alter the frozen split.** Nothing moves between train and
test; this is a post-hoc analysis partition of the test set for a new
measurement. The headline 0.9031 is still computed on all 1,098 stars and is
untouched. The only cost is that coverage is *validated* on ~549 stars rather
than 1,098, which is why validation is repeated over many random partitions
rather than trusting one.

METHOD: LAC (least ambiguous set-valued classifier), with APS reported alongside

  score      s_i = 1 - p_hat(true class of i)
  threshold  q_hat = the ceil((n+1)(1-alpha))/n empirical quantile of s
  set        C(x) = { y : 1 - p_hat(y) <= q_hat } = { y : p_hat(y) >= 1 - q_hat }

LAC is chosen as primary because for a binary problem it produces the SMALLEST
average set size among methods with valid marginal coverage. APS is computed
too, because it trades set size for better class-conditional behaviour, and
this project's test set is 79% positive -- a regime where marginal coverage can
be satisfied while one class is badly under-covered. Class-conditional coverage
is therefore reported separately and is the check that matters.

FOUR-VALUED OUTPUT. For binary classification the set can be:
  {1}    confident planet
  {0}    confident false positive
  {0,1}  ambiguous -- the honest "not sure" the raw probability cannot express
  {}     empty -- both classes scored below threshold; a distribution-shift
         signal, not a bug
"""
import os
import sys
import json
import importlib.util
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
CADENCE = os.path.join(SCRIPT_DIR, "cadence_per_star.csv")
PROD = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS = os.path.join(SCRIPT_DIR, "conformal_prediction_results.json")
ARTIFACT = os.path.join(ROOT, "models", "conformal_calibration.json")

ALPHAS = [0.10, 0.05, 0.01]
N_SPLITS = 300
SEED = 42


def _m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------- LAC
def lac_threshold(p_cal, y_cal, alpha):
    """Conformal quantile with the finite-sample (n+1) correction.

    The ceil((n+1)(1-alpha))/n level and the 'higher' interpolation are what
    make the guarantee exact rather than asymptotic; using np.quantile's default
    linear interpolation would slightly under-cover.
    """
    s = 1.0 - np.where(y_cal == 1, p_cal, 1.0 - p_cal)
    n = len(s)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        return 1.0                      # n too small to certify this alpha
    return float(np.quantile(s, level, method="higher"))


def lac_sets(p, q):
    """Returns (contains_0, contains_1) boolean arrays."""
    return (1.0 - (1.0 - p)) <= q, (1.0 - p) <= q


# --------------------------------------------------------------------- APS
def aps_threshold(p_cal, y_cal, alpha, rng):
    """Adaptive prediction sets: cumulative probability mass, most likely first,
    with the standard randomised tie-break so coverage is exact rather than
    conservative."""
    P = np.c_[1.0 - p_cal, p_cal]
    order = np.argsort(-P, axis=1)
    sortedP = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(sortedP, axis=1)
    rank = np.argmax(order == y_cal[:, None], axis=1)
    u = rng.uniform(size=len(p_cal))
    s = cum[np.arange(len(p_cal)), rank] - u * sortedP[np.arange(len(p_cal)), rank]
    n = len(s)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        return 1.0
    return float(np.quantile(s, level, method="higher"))


def aps_sets(p, q, rng):
    P = np.c_[1.0 - p, p]
    order = np.argsort(-P, axis=1)
    sortedP = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(sortedP, axis=1)
    u = rng.uniform(size=(len(p), 1))
    keep = (cum - u * sortedP) <= q
    keep[:, 0] = True                    # always keep the top class
    inc = np.zeros_like(keep)
    np.put_along_axis(inc, order, keep, axis=1)
    return inc[:, 0], inc[:, 1]


# ------------------------------------------------- Mondrian (class-conditional)
def mondrian_thresholds(p_cal, y_cal, alpha):
    """Separate threshold per class, so the guarantee is P(y in C | Y=y) >= 1-a
    FOR EACH CLASS rather than only on average.

    This is the fix for the imbalance plain LAC exhibits here: the test set is
    79% positive, so a marginal guarantee is dominated by positives and the
    negative class can be badly under-covered while the headline number looks
    correct. Conditioning on the calibration example's TRUE label is legitimate
    -- no test label is used -- because the threshold for class y is applied to
    the question "could this point be class y?".
    """
    out = {}
    for cls in (0, 1):
        m = y_cal == cls
        s = 1.0 - (p_cal[m] if cls == 1 else 1.0 - p_cal[m])
        n = int(m.sum())
        if n < 5:
            out[cls] = 1.0
            continue
        level = np.ceil((n + 1) * (1 - alpha)) / n
        out[cls] = 1.0 if level > 1 else float(np.quantile(s, level, method="higher"))
    return out


def mondrian_sets(p, q):
    return p <= q[0], (1.0 - p) <= q[1]


def summarise(c0, c1, y):
    size = c0.astype(int) + c1.astype(int)
    covered = np.where(y == 1, c1, c0)
    return {
        "coverage": float(covered.mean()),
        "coverage_pos": float(covered[y == 1].mean()) if (y == 1).any() else np.nan,
        "coverage_neg": float(covered[y == 0].mean()) if (y == 0).any() else np.nan,
        "mean_set_size": float(size.mean()),
        "pct_singleton": float((size == 1).mean() * 100),
        "pct_ambiguous": float((size == 2).mean() * 100),
        "pct_empty": float((size == 0).mean() * 100),
    }


def main():
    m05 = _m05()
    df = pd.read_csv(TRAINING)
    X, y = m05.build_feature_matrix(df)
    X = X.reset_index(drop=True)
    y = np.asarray(y).astype(int)
    tr, te = m05.split_by_host(df)
    tr, te = np.asarray(tr), np.asarray(te)
    cad = pd.read_csv(CADENCE)
    c = pd.to_numeric(df.merge(cad, on="host", how="left")["cadence_min"],
                      errors="coerce")
    is2 = ((c >= 1.0) & (c <= 2.6)).to_numpy() | c.isna().to_numpy()

    prod = joblib.load(PROD)
    p_all = prod.predict_proba(X)[:, 1]

    p_te, y_te = p_all[te], y[te]
    is2_te = is2[te]

    print("=" * 92)
    print("SPLIT CONFORMAL PREDICTION over the deployed model -- NO retraining")
    print("=" * 92)
    print(f"  frozen test set: {len(y_te)} stars "
          f"({int((y_te==1).sum())} planets / {int((y_te==0).sum())} FPs, "
          f"prevalence {y_te.mean():.3f})")
    print(f"  production ROC-AUC on this set: {roc_auc_score(y_te, p_te):.4f} "
          f"(unchanged; conformal does not alter it)")
    print(f"  2-min-only subset: {int(is2_te.sum())} stars")
    print(f"\n  calibration source: OPTION B -- random stratified halves of the")
    print(f"  frozen TEST set. Frozen split composition untouched.")
    print(f"  validation: {N_SPLITS} independent partitions per alpha.")

    out = {"n_test": int(len(y_te)), "prevalence": float(y_te.mean()),
            "n_splits": N_SPLITS, "auc_unchanged": float(roc_auc_score(y_te, p_te))}

    rng = np.random.RandomState(SEED)
    pos_idx = np.where(y_te == 1)[0]
    neg_idx = np.where(y_te == 0)[0]

    for pop_name, mask in (("FULL test", np.ones(len(y_te), bool)),
                           ("2-min-only", is2_te)):
        print("\n" + "=" * 92)
        print(f"EMPIRICAL COVERAGE -- {pop_name} (n={int(mask.sum())})")
        print("=" * 92)
        print(f"  {'method':<10}{'alpha':>6}{'target':>8}{'coverage':>10}{'sd':>8}"
              f"{'cov POS':>9}{'cov NEG':>9}{'set size':>10}{'%ambig':>8}{'%empty':>8}")
        for a in ALPHAS:
            stats = {"LAC": [], "APS": [], "Mondrian": []}
            for rep in range(N_SPLITS):
                rp = rng.permutation(pos_idx)
                rn = rng.permutation(neg_idx)
                cal = np.r_[rp[: len(rp) // 2], rn[: len(rn) // 2]]
                ev = np.r_[rp[len(rp) // 2:], rn[len(rn) // 2:]]
                ev = ev[mask[ev]]
                if len(ev) < 30:
                    continue
                q = lac_threshold(p_te[cal], y_te[cal], a)
                stats["LAC"].append(summarise(*lac_sets(p_te[ev], q), y_te[ev]))
                qa = aps_threshold(p_te[cal], y_te[cal], a, rng)
                stats["APS"].append(summarise(*aps_sets(p_te[ev], qa, rng), y_te[ev]))
                qm = mondrian_thresholds(p_te[cal], y_te[cal], a)
                stats["Mondrian"].append(
                    summarise(*mondrian_sets(p_te[ev], qm), y_te[ev]))
            for nm in ("LAC", "APS", "Mondrian"):
                D = pd.DataFrame(stats[nm])
                print(f"  {nm:<10}{a:>6.2f}{1-a:>8.2f}{D.coverage.mean():>10.4f}"
                      f"{D.coverage.std():>8.4f}{D.coverage_pos.mean():>9.4f}"
                      f"{D.coverage_neg.mean():>9.4f}{D.mean_set_size.mean():>10.3f}"
                      f"{D.pct_ambiguous.mean():>8.1f}{D.pct_empty.mean():>8.1f}")
                out.setdefault("coverage", {}).setdefault(pop_name, {}) \
                   .setdefault(str(a), {})[nm] = {
                       "target": 1 - a, "coverage_sd": float(D.coverage.std()),
                       **{k: float(D[k].mean()) for k in D.columns}}
            print()

    # ------------------------------------------------- deployment artifact
    print("\n" + "=" * 92)
    print("DEPLOYMENT ARTIFACT -- thresholds fitted on ALL 1,098 test stars")
    print("=" * 92)
    print("  Validation above used halves so coverage could be checked on unseen")
    print("  data. For deployment the whole calibration set is used: more data,")
    print("  tighter valid threshold. Same estimator, same guarantee.")
    import hashlib
    md5 = hashlib.md5(open(PROD, "rb").read()).hexdigest()
    art = {"generated_from": "frozen test split (1,098 stars), production model",
           "model_md5": md5,
           "regenerate_when": "the production model changes, or the frozen test "
                              "set gains stars -- see RESULTS_SUMMARY",
           "n_calibration": int(len(y_te)),
           "n_calibration_pos": int((y_te == 1).sum()),
           "n_calibration_neg": int((y_te == 0).sum()),
           "method": "Mondrian (class-conditional) LAC",
           "prevalence": float(y_te.mean()), "thresholds": {}}
    print(f"\n  {'alpha':>6}{'q_neg':>9}{'q_pos':>9}   include 0 if p <= q_neg;"
          f"  include 1 if (1-p) <= q_pos")
    for a in ALPHAS:
        qm = mondrian_thresholds(p_te, y_te, a)
        art["thresholds"][str(a)] = {"q_neg": qm[0], "q_pos": qm[1],
                                     "lac_q": lac_threshold(p_te, y_te, a)}
        print(f"  {a:>6.2f}{qm[0]:>9.4f}{qm[1]:>9.4f}")
    with open(ARTIFACT, "w") as f:
        json.dump(art, f, indent=2)
    print(f"\n  saved {ARTIFACT}")
    out["artifact"] = art

    # -------------------------------------------------- exchangeability check
    print("\n" + "=" * 92)
    print("DOES THE GUARANTEE TRANSFER TO UNKNOWN CANDIDATES? -- the decisive caveat")
    print("=" * 92)
    print("  Conformal coverage requires EXCHANGEABILITY between the calibration")
    print("  set and the points it is applied to. Calibration here is labelled")
    print("  TESS stars (79% planets, TOI-sourced). The app applies it to UNKNOWN")
    print("  candidates. If those are drawn from a different distribution, the")
    print("  guarantee is void -- so it is measured, not assumed.")
    cand_files = [
        os.path.join(ROOT, "results", "unknown_candidates", "ranked_candidates.csv"),
        os.path.join(ROOT, "results", "unknown_candidates_widesector",
                     "ranked_candidates.csv")]
    frames = [pd.read_csv(f) for f in cand_files if os.path.exists(f)]
    if frames:
        u = pd.concat(frames, ignore_index=True)
        Xu, _ = m05.build_feature_matrix(u.assign(label=0))
        Xu = Xu.reset_index(drop=True)[X.columns]
        sys.path.insert(0, SCRIPT_DIR)
        from domain_separability import domain_report
        Xc = pd.concat([X[te].reset_index(drop=True), Xu], ignore_index=True)
        dom = np.r_[np.zeros(int(te.sum()), int), np.ones(len(Xu), int)]
        rep = domain_report(Xc, dom, names=("calibration (test set)",
                                            "unknown candidates"), verbose=True)
        out["exchangeability"] = {"domain_auc": rep["domain_auc"],
                                  "n_unknown": int(len(Xu))}
        auc = rep["domain_auc"]
        print()
        if auc is not None and auc > 0.90:
            print(f"  VERDICT: domain AUC {auc:.4f} -- calibration stars and unknown")
            print("  candidates are NOT exchangeable. The finite-sample guarantee is")
            print("  VALID for stars like the test set and is NOT VALID as stated for")
            print("  unknown candidates. On those it is a well-calibrated heuristic.")
            print("  The UI must say so rather than promise coverage it cannot give.")
        else:
            print(f"  VERDICT: domain AUC {auc:.4f} -- close enough to exchangeable")
            print("  that the guarantee transfers approximately.")
        out["exchangeability"]["transfers"] = bool(auc is not None and auc <= 0.90)

    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  saved {RESULTS}")


if __name__ == "__main__":
    main()
