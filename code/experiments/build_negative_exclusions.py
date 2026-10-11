"""build_negative_exclusions.py -- writes negative_exclusions.csv, the list of
data/processed_negative/ files that 03_transit_search_negative.py skips.

The rows come from 03's own class guard (classify_refusals), not from a hand
list, so the manifest cannot drift from the checks it records:

  fa_only         not on toi_false_positives.csv but on toi_false_alarms.csv.
                  FA was never adopted (register D3).
  confirmed_host  FP-listed, but the star hosts a confirmed planet
                  (confirmed_hosts.py).

A refused file that fits neither reason is foreign to the negative class. This
script will NOT list it: it exits non-zero and writes nothing, so the manifest
can never become a way to wave an unknown file through.

Run from the repository: python3 code/experiments/build_negative_exclusions.py
"""
import importlib
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(HERE, "..")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, HERE)
import confirmed_hosts

m03 = importlib.import_module("03_transit_search_negative")
FA_PATH = os.path.join(m03.CATALOG_FOLDER, "toi_false_alarms.csv")
FP_PATH = os.path.join(m03.CATALOG_FOLDER, "toi_false_positives.csv")
ARCHIVE = os.path.join(confirmed_hosts.ROOT, confirmed_hosts.ARCHIVE_CSV)


def _tois(path, disp):
    t = pd.read_csv(path)
    out = {}
    for tid, toi in zip(t["tid"].astype("int64"), t["toi"]):
        out.setdefault(int(tid), []).append(f"TOI-{toi} ({disp})")
    return out


def main():
    files = sorted(f for f in os.listdir(m03.INPUT_FOLDER) if f.endswith(".csv"))
    unlisted, hosts = m03.classify_refusals(files)
    resolve = confirmed_hosts.tic_resolver()
    fa, fp = _tois(FA_PATH, "FA"), _tois(FP_PATH, "FP")
    confirmed, _ = confirmed_hosts.confirmed_host_tics()
    a = pd.read_csv(ARCHIVE)
    host_name = {}
    for h, t in zip(a["hostname"].astype(str), a["tic_id"].map(confirmed_hosts._pid)):
        if t is not None:
            host_name.setdefault(t, h)

    def also_confirmed(tic):
        if tic in confirmed:
            return f"; also a confirmed host ({host_name.get(tic, 'confirmed-host set')})"
        return ""

    rows, foreign = [], []
    for f in unlisted:
        tic = resolve(os.path.splitext(f)[0])
        if tic not in fa:
            foreign.append(f)
            continue
        rows.append({"filename": f, "tic": tic, "reason": "fa_only",
                     "source": "toi_false_alarms.csv " + ", ".join(fa[tic]) + also_confirmed(tic)})
    for f in hosts:
        tic = resolve(os.path.splitext(f)[0])
        rows.append({"filename": f, "tic": tic, "reason": "confirmed_host",
                     "source": "toi_false_positives.csv " + ", ".join(fp.get(tic, ["?"]))
                               + f"; confirmed host {host_name.get(tic, '(confirmed-host set)')}"})
    if foreign:
        sys.exit(f"ERROR: {len(foreign)} refused file(s) are neither FA-only nor confirmed hosts, "
                 f"so they do not belong in the manifest -- nothing written: {foreign[:10]}")

    out = pd.DataFrame(rows, columns=["filename", "tic", "reason", "source"]).sort_values("filename")
    out.to_csv(m03.EXCLUSIONS_PATH, index=False)
    print(f"wrote {m03.EXCLUSIONS_PATH}: {len(out)} rows "
          f"({out['reason'].value_counts().to_dict()}) from {len(files)} files in {m03.INPUT_FOLDER}")


if __name__ == "__main__":
    main()
