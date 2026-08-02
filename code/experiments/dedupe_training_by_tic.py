"""dedupe_training_by_tic.py -- find and remove stars that appear in
training.csv more than once under different names, and repair the frozen
split's resulting train/test leakage.

THE BUG

The scheduler's label-watch pipeline (`retrain_pipeline.find_new_labeled_
examples`) queries `select hostname,tic_id from pscomppars` and enqueues every
confirmed planet under a `TIC_<id>` host name. The original positive class,
built by `01_download_known.py`, names stars by HOSTNAME (`11_Com`,
`Kepler-142`). The queue dedupes on the host STRING, so it cannot tell that
`TIC_24133681` and `HIP_77900` are the same star, and re-adds it.

Measured before this fix, from only 137 processed watch labels:

    133 stars present twice under two different names
     52 of those straddle the frozen train/test split

A straddling star is trained on and then evaluated on. One case
(HIP_77900 / TIC_24133681) is byte-identical in period, depth and SDE -- the
same measurement on both sides. Others are the same star re-processed to a
different TLS solution, which still leaks the star's noise, systematics and
stellar parameters.

4,243 watch labels are still pending. At the observed ~93% duplicate rate,
draining that queue would add thousands more duplicate rows and progressively
corrupt the split, so the dedupe in the pipeline is fixed too (separately, in
retrain_pipeline.py) -- this script repairs the data already written.

IDENTITY RESOLUTION

Star identity is the TIC id, resolved in priority order:
  1. hosts already named `TIC_<id>` -- parse directly
  2. the NASA archive's own hostname -> tic_id mapping from pscomppars, which
     is authoritative and resolves far more than the earlier coordinate
     cross-match did (74 unresolved vs 352)
  3. `positive_class_tic_ids.csv`, the coordinate cross-match, as fallback

The 74 that remain unresolved are all microlensing events (KMT-*-BLG-*) --
distant bulge stars with no TIC entry at all. They cannot collide with a TESS
target, so leaving them keyed by name is safe.

WHICH COPY IS KEPT

The row whose host name appears in `split_manifest.json` -- i.e. the original,
pre-freeze star. That keeps every surviving row on the side of the split it was
frozen onto, so removing duplicates cannot itself reshuffle the split. The
post-freeze `TIC_`-named copy is dropped. Where neither or both are in the
manifest, the first-seen row wins, deterministically by original file order.

This script WRITES a cleaned dataset to a new path and never overwrites
training.csv in place; promoting it is a separate, explicit step.
"""
import os
import re
import sys
import json
import importlib.util
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..")
ROOT = os.path.join(CODE_DIR, "..")
sys.path.insert(0, CODE_DIR)

TRAINING_CSV = os.path.join(ROOT, "data", "training_dataset", "training.csv")
MANIFEST = os.path.join(ROOT, "data", "training_dataset", "split_manifest.json")
ARCHIVE_CSV = os.path.join(SCRIPT_DIR, "archive_all_confirmed_tic.csv")
OLD_TIC_MAP = os.path.join(SCRIPT_DIR, "positive_class_tic_ids.csv")
CLEAN_CSV = os.path.join(ROOT, "data", "training_dataset", "training_deduped.csv")
REPORT = os.path.join(SCRIPT_DIR, "dedupe_report.json")


def _pid(v):
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else None


def load_m05():
    spec = importlib.util.spec_from_file_location(
        "m05", os.path.join(CODE_DIR, "05_train_models.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m05"] = m
    spec.loader.exec_module(m)
    return m


def build_resolver():
    amap = {}
    if os.path.exists(ARCHIVE_CSV):
        a = pd.read_csv(ARCHIVE_CSV)
        for h, t in zip(a["hostname"].astype(str), a["tic_id"].map(_pid)):
            if t is None:
                continue
            amap.setdefault(h.replace(" ", "_"), t)
            amap.setdefault(h, t)
    omap = {}
    if os.path.exists(OLD_TIC_MAP):
        o = pd.read_csv(OLD_TIC_MAP).dropna(subset=["tic_id"])
        omap = o.set_index("host")["tic_id"].astype("int64").to_dict()

    def resolve(h):
        h = str(h)
        m = re.match(r"^TIC_(\d+)", h)
        if m:
            return int(m.group(1))
        return amap.get(h) or omap.get(h)

    return resolve


def manifest_hosts():
    if not os.path.exists(MANIFEST):
        return set()
    with open(MANIFEST) as f:
        d = json.load(f)
    hosts = set()
    for k, v in d.items():
        if isinstance(v, list):
            hosts |= {str(x) for x in v}
        elif isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, list):
                    hosts |= {str(x) for x in vv}
    return hosts


def main():
    res = {}
    m05 = load_m05()
    df = pd.read_csv(TRAINING_CSV)
    df["_row"] = np.arange(len(df))
    resolve = build_resolver()
    df["tic"] = df["host"].map(resolve)

    tr_mask, te_mask = m05.split_by_host(df)
    df["side"] = np.where(np.asarray(te_mask), "test", "train")
    mh = manifest_hosts()
    df["in_manifest"] = df["host"].astype(str).isin(mh)

    print("=" * 78)
    print("DUPLICATE AUDIT")
    print("=" * 78)
    print(f"rows {len(df)} | resolved to TIC {df.tic.notna().sum()} | "
          f"unresolved {df.tic.isna().sum()} (microlensing, no TIC -- safe)")
    print(f"manifest hosts: {len(mh)}")

    known = df[df.tic.notna()].copy()
    g = known.groupby("tic")
    sizes = g.size()
    dup_tics = sizes[sizes > 1].index
    dups = known[known.tic.isin(dup_tics)]
    straddle = [t for t, sub in dups.groupby("tic") if sub["side"].nunique() > 1]
    cross_label = [t for t, sub in dups.groupby("tic") if sub["label"].nunique() > 1]

    print(f"\n  duplicated stars (same TIC, >1 row): {len(dup_tics)}")
    print(f"  duplicate rows total:                {len(dups)}")
    print(f"  *** straddling train AND test:       {len(straddle)}  <- LEAKAGE")
    print(f"  duplicates with CONFLICTING labels:  {len(cross_label)}")
    res.update(n_rows=int(len(df)), n_resolved=int(df.tic.notna().sum()),
               n_unresolved=int(df.tic.isna().sum()),
               n_duplicate_stars=int(len(dup_tics)),
               n_duplicate_rows=int(len(dups)),
               n_straddling=int(len(straddle)),
               n_conflicting_labels=int(len(cross_label)))
    if cross_label:
        res["conflicting_label_tics"] = [int(t) for t in cross_label]
        print("    (conflicting-label TICs are NOT auto-resolved -- listed in the report)")

    if len(dup_tics) == 0:
        print("\nno duplicates -- nothing to do")
        return

    # ---- choose the survivor: manifest membership first, then file order ----
    keep_idx = []
    drop_rows = []
    for t, sub in known.groupby("tic"):
        if len(sub) == 1:
            keep_idx.append(sub["_row"].iloc[0])
            continue
        pref = sub.sort_values(["in_manifest", "_row"], ascending=[False, True])
        keep = pref.iloc[0]
        keep_idx.append(keep["_row"])
        for _, r in pref.iloc[1:].iterrows():
            drop_rows.append({"tic": int(t), "kept_host": str(keep["host"]),
                              "kept_side": keep["side"],
                              "dropped_host": str(r["host"]),
                              "dropped_side": r["side"],
                              "dropped_label": int(r["label"]),
                              "straddled": bool(keep["side"] != r["side"])})
    keep_idx += df[df.tic.isna()]["_row"].tolist()
    clean = df[df["_row"].isin(set(keep_idx))].drop(
        columns=["_row", "tic", "side", "in_manifest"])

    print(f"\n  rows kept:    {len(clean)}")
    print(f"  rows dropped: {len(df) - len(clean)}")
    dropped = pd.DataFrame(drop_rows)
    if len(dropped):
        print(f"  dropped that were on the opposite side of a kept row "
              f"(the leaking ones): {int(dropped.straddled.sum())}")
        print("\n  sample of dropped duplicates:")
        print(dropped.head(6).to_string(index=False))
        dropped.to_csv(os.path.join(SCRIPT_DIR, "dedupe_dropped_rows.csv"), index=False)

    print("\n  class balance:")
    for name, d in (("before", df), ("after", clean)):
        p, n = int((d.label == 1).sum()), int((d.label == 0).sum())
        print(f"    {name:6s} positives {p} / negatives {n}  (ratio {p/max(n,1):.2f}:1)")
    res["before"] = {"positives": int((df.label == 1).sum()),
                     "negatives": int((df.label == 0).sum())}
    res["after"] = {"positives": int((clean.label == 1).sum()),
                    "negatives": int((clean.label == 0).sum())}
    res["n_dropped"] = int(len(df) - len(clean))

    # ---- verify the fix: re-run the split on the cleaned frame ----
    tr2, te2 = m05.split_by_host(clean)
    c2 = clean.copy()
    c2["tic"] = c2["host"].map(resolve)
    c2["side"] = np.where(np.asarray(te2), "test", "train")
    k2 = c2[c2.tic.notna()]
    g2 = k2.groupby("tic")
    remain_dup = int((g2.size() > 1).sum())
    remain_straddle = sum(1 for _, s in k2.groupby("tic") if s["side"].nunique() > 1)
    print("\n" + "=" * 78)
    print("VERIFY AFTER DEDUPE")
    print("=" * 78)
    print(f"  duplicated stars remaining: {remain_dup}")
    print(f"  straddling train/test:      {remain_straddle}")
    print(f"  split: {int(np.asarray(tr2).sum())} train / {int(np.asarray(te2).sum())} test")
    res["after_dedupe_duplicates"] = remain_dup
    res["after_dedupe_straddling"] = remain_straddle
    res["after_split"] = {"train": int(np.asarray(tr2).sum()),
                          "test": int(np.asarray(te2).sum())}

    clean.to_csv(CLEAN_CSV, index=False)
    with open(REPORT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nCleaned dataset -> {CLEAN_CSV}")
    print(f"Report          -> {REPORT}")
    print("training.csv itself was NOT modified.")


if __name__ == "__main__":
    main()
