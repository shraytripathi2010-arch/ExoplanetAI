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

# (ranked candidate export, the feature table that pool was scored FROM).
# Paired, not pooled: the wide-sector run has its own feature table and the two
# must not be crossed -- a host present in both pools has different TLS features
# in each (different sector baseline), so joining a row against the wrong table
# would silently attach another run's measurements.
CANDIDATE_POOLS = [
    (os.path.join(ROOT, "results", "unknown_candidates", "ranked_candidates.csv"),
     os.path.join(ROOT, "data", "catalogs", "unknown_features.csv")),
    (os.path.join(ROOT, "results", "unknown_candidates_widesector",
                  "ranked_candidates.csv"),
     os.path.join(ROOT, "data", "catalogs", "unknown_features_widesector.csv")),
]


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


def save_results(out):
    """Called twice on purpose -- see the call site before the optional
    exchangeability section."""
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  saved {RESULTS}")


# --------------------------------------------- unknown candidates, re-joined
def load_unknown_candidates(m05, verbose=True):
    """The unknown-candidate pools as a frame carrying all FEATURE_COLUMNS.

    WHY THIS IS A JOIN AND NOT `pd.read_csv`.

    It used to be a bare read, and that is what broke this section for eight
    days (2026-08-06 -> 2026-08-14, found during the Optuna deployment). The
    ranked exports on disk were written 2026-08-05 12:37 and carry 37 and 34
    columns; FEATURE_COLUMNS has since gained the five `var_*` (promoted
    2026-08-06) and the two `gaia_*` (promoted 2026-08-14), so
    `build_feature_matrix` raised SystemExit on all seven. Because that happens
    AFTER models/conformal_calibration.json is written, the deployment artifact
    kept succeeding and only the exit code complained -- which nobody read.

    *** THE EXPORT WRITER IS NOT THE BUG, SO IT IS NOT WHAT IS FIXED HERE. ***
    Checked before choosing: `06_download_unknown.score_candidates` builds the
    ranked frame FROM `unknown_features*.csv` and hard-fails on a missing
    feature column (`missing_required` -> SystemExit, 06_download_unknown.py
    ~1686), then writes that whole frame. A run today would export all 33
    columns unprompted. The CSVs are simply STALE -- the pipeline has not run
    since 2026-08-05 -- and the only way to refresh them is a fresh MAST
    download + TLS search that would also overwrite the live candidate tables.
    Making an offline diagnostic depend on that is the wrong dependency; the
    columns it needs already exist on disk, keyed identically.

    So the missing columns are joined back from the same per-pool feature table
    the exports were built from. This also matches the newest precedent in the
    repo: `cluster1_pool_evidence.py` reads `unknown_features*.csv` as the
    feature source and merges the stellar params in from ranked_candidates.csv
    -- the same join, in the other direction.

    The join is `host`-keyed and exact -- 254/254 and 54/54 matched, feature
    tables unique on `host`, `var_*` 100% present, `gaia_*` 96.1%/100%. Gaia
    NaNs are left as NaN on purpose: they are OPTIONAL_FEATURES, imputed inside
    the fitted pipeline at serve time, and `domain_report` imputes with a
    median inside its own CV pipeline, so this is what production sees too.

    Returns (frame, provenance list). Raises if a column cannot be sourced --
    a domain-shift number measured on a silently-truncated feature set would be
    worse than no number at all.
    """
    need = list(m05.FEATURE_COLUMNS)
    frames, prov = [], []

    for ranked_path, feat_path in CANDIDATE_POOLS:
        if not os.path.exists(ranked_path):
            continue
        pool = os.path.basename(os.path.dirname(ranked_path))
        r = pd.read_csv(ranked_path)
        missing = [c for c in need if c not in r.columns]
        info = {"pool": pool, "n_rows": int(len(r)),
                "export": os.path.relpath(ranked_path, ROOT),
                "missing_from_export": missing, "joined": []}
        if verbose:
            print(f"\n  {pool}: {len(r)} ranked rows, "
                  f"{len(need) - len(missing)}/{len(need)} feature columns present")

        if missing:
            if not os.path.exists(feat_path):
                raise FileNotFoundError(
                    f"{pool}: the export lacks {missing} and its feature table "
                    f"{feat_path} does not exist to join them from.")
            f = pd.read_csv(feat_path)
            unsourceable = [c for c in missing if c not in f.columns]
            if unsourceable:
                raise KeyError(
                    f"{pool}: {unsourceable} are in neither {os.path.basename(ranked_path)} "
                    f"nor {os.path.basename(feat_path)}. If these were just promoted into "
                    f"FEATURE_COLUMNS, the candidate pool has not been re-extracted yet.")
            if f["host"].duplicated().any():
                raise ValueError(
                    f"{pool}: {os.path.basename(feat_path)} has duplicate 'host' values; "
                    f"a many-to-many join would silently multiply candidate rows.")
            # many_to_one: strict on the side that supplies the values, tolerant
            # of the export, which is the object under study rather than the key.
            # `_matched` rather than "any joined value is non-null": a host CAN
            # legitimately match a row whose gaia_* are both NaN (no Gaia source
            # within 3 arcsec), and counting that as a failed join would
            # under-report the join and over-report a data problem.
            r = r.merge(f[["host"] + missing].assign(_matched=True), on="host",
                        how="left", validate="many_to_one")
            matched = int(r.pop("_matched").eq(True).sum())
            info["joined"] = missing
            info["source"] = os.path.relpath(feat_path, ROOT)
            info["hosts_matched"] = matched
            info["coverage"] = {c: float(r[c].notna().mean()) for c in missing}
            if verbose:
                print(f"    joined {len(missing)} column(s) from "
                      f"{os.path.basename(feat_path)} on host -- "
                      f"{matched}/{len(r)} rows matched")
                for c in missing:
                    print(f"      {c:<16} {r[c].notna().mean() * 100:5.1f}% non-null")

        frames.append(r)
        prov.append(info)

    if not frames:
        raise FileNotFoundError(
            "no ranked_candidates.csv found in either candidate pool -- "
            "run 06_download_unknown.py first.")

    u = pd.concat(frames, ignore_index=True)
    still = [c for c in need if c not in u.columns]
    if still:
        raise KeyError(f"after joining, still missing: {still}")
    return u, prov


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

    # ------------------------------------------- results written BEFORE the
    #                                             optional section below
    # Everything above this line is what this script exists to produce, and it
    # is finished. The exchangeability check that follows is a diagnostic over
    # files this script does not own and cannot regenerate, so it is the part
    # most likely to break -- and it did, silently, from 2026-08-06 to
    # 2026-08-14: it raised after the artifact was already on disk, so the run
    # "worked", while conformal_prediction_results.json sat unchanged from
    # Aug 4 through two model deployments and was read as current.
    #
    # Writing here makes that impossible: the results file is always at least
    # as fresh as the artifact beside it. It is written a second time at the
    # end so a successful diagnostic still lands.
    save_results(out)

    # -------------------------------------------------- exchangeability check
    print("\n" + "=" * 92)
    print("DOES THE GUARANTEE TRANSFER TO UNKNOWN CANDIDATES? -- the decisive caveat")
    print("=" * 92)
    print("  Conformal coverage requires EXCHANGEABILITY between the calibration")
    print("  set and the points it is applied to. Calibration here is labelled")
    print("  TESS stars (79% planets, TOI-sourced). The app applies it to UNKNOWN")
    print("  candidates. If those are drawn from a different distribution, the")
    print("  guarantee is void -- so it is measured, not assumed.")
    # `except SystemExit` is deliberate and not redundant with `Exception`:
    # build_feature_matrix signals a schema mismatch with SystemExit, which
    # derives from BaseException, so a bare `except Exception` here would
    # reproduce exactly the silent failure this wrapper exists to end.
    try:
        u, prov = load_unknown_candidates(m05)
        Xu, _ = m05.build_feature_matrix(u.assign(label=0))
        Xu = Xu.reset_index(drop=True)[X.columns]
        sys.path.insert(0, SCRIPT_DIR)
        from domain_separability import domain_report
        Xc = pd.concat([X[te].reset_index(drop=True), Xu], ignore_index=True)
        dom = np.r_[np.zeros(int(te.sum()), int), np.ones(len(Xu), int)]
        print()
        rep = domain_report(Xc, dom, names=("calibration (test set)",
                                            "unknown candidates"), verbose=True)
        out["exchangeability"] = {"domain_auc": rep["domain_auc"],
                                  "n_unknown": int(len(Xu)),
                                  "candidate_sources": prov}
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
        skipped = None
    except (Exception, SystemExit) as e:
        skipped = f"{type(e).__name__}: {e}"
        out["exchangeability"] = {"skipped": True, "error": skipped,
                                  "domain_auc": None, "transfers": None}
        print("\n" + "!" * 92)
        print("DIAGNOSTIC SKIPPED -- the exchangeability check did not run")
        print("!" * 92)
        print(f"  {skipped.splitlines()[0]}")
        for line in skipped.splitlines()[1:]:
            print(f"  {line}")
        print("\n  This is the OPTIONAL section. The conformal thresholds, the coverage")
        print("  validation and the deployment artifact above are complete and were")
        print("  saved before this ran -- nothing there is affected.")
        print("  What IS affected: no current measurement of whether the guarantee")
        print("  transfers to unknown candidates. The last recorded verdict in")
        print("  conformal_prediction_results.json is now marked skipped, not stale.")

    save_results(out)

    if skipped:
        # Non-zero AFTER both files are written, not instead of writing them.
        # The old failure exited non-zero too -- the difference is that the
        # results file is now complete and self-describing either way, so the
        # exit code is a second signal rather than the only one.
        raise SystemExit(
            "conformal_prediction.py: thresholds + artifact OK, "
            "exchangeability diagnostic SKIPPED (see above).")


if __name__ == "__main__":
    main()
