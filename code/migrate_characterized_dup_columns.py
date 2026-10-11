"""migrate_characterized_dup_columns.py -- one-time collapse of the stacked
_dup columns in 08's outputs to canonical columns with provenance.

Until 2026-10-11, 08 joined each run's scores with merge(suffixes=("", "_dup"))
onto a frame that already carried them. Every run therefore added a layer, and
the canonical column kept each star's FIRST-ever value. In the committed
outputs (ce695533), every one of the 10 scored fields has 13 layers. The raw
header holds `f`, `f_dup`, `f_dup.1` ... `f_dup.10` and a second `f_dup`, which
pandas reads as `f_dup.11`.

RULE (per star): take the NEWEST layer that holds a probability, and take all
10 fields from that same layer, so a row never mixes runs. Record which model
produced it in scored_model_md5 (see LAYER_MODELS). The previous distinct
probability goes to <field>_prev. An old-model score is never presented as
current: the app labels anything not stamped with the production md5.

LAYER_MODELS was derived by matching each layer, value for value, to the
committed ranked_candidates_in_distribution.csv of its era. It describes THIS
13-layer history only. A file with any other layer count is refused rather
than guessed at.

Usage (never in place; the output must not exist):
    python3 code/migrate_characterized_dup_columns.py --input IN.csv --output OUT.csv
"""
import argparse
import csv
import os
import re
import sys
from datetime import datetime, timezone

SCORED_FIELDS = ["predicted_probability", "SDE", "SNR", "FAP", "distinct_transit_count",
                 "period", "duration", "T0", "st_rad", "st_teff", "st_mass"]
M341 = "341f1a3907e77f6ec294f182833e613c"      # 24 features, production until 2026-08-05
MFE3 = "fe3fa82f36cc978396c68be07d6057f9"      # 33 features, production since 2026-08-14
# (model md5, how the layer was attributed), by layer ordinal in header order.
LAYER_MODELS = (
    [("unknown", "original column: first characterization, scored before the first commit")]
    + [("unknown", f"layer {n}: 08 re-run before 2026-07-16; identical to the original column")
       for n in ["_dup"] + [f"_dup.{i}" for i in range(1, 8)]]
    + [(M341, "layer _dup.8 = ranked_candidates_in_distribution.csv as committed 2026-07-16"),
       (M341, "layer _dup.9 = the same ranked file (unchanged through 2026-07-29)"),
       (M341, "layer _dup.10 = ranked file as committed 2026-08-01, before the 2026-08-05 deploy"),
       (MFE3, "layer _dup.11 = ranked file of the 2026-08-29 run 32 (ce695533)")]
)
EXPECTED_LAYERS = len(LAYER_MODELS)          # 13
NEW_COLS = ["scored_model_md5", "scored_model_md5_prev", "scored_at", "scored_source"]


def layer_positions(header, field):
    """Header positions of `field` and its _dup layers, in creation order."""
    pat = re.compile(re.escape(field) + r"(_dup(\.\d+)?)?$")
    return [i for i, c in enumerate(header) if pat.fullmatch(c)]


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    if os.path.abspath(a.input) == os.path.abspath(a.output):
        sys.exit("ERROR: --output must differ from --input; this script never migrates in place.")
    if os.path.exists(a.output):
        sys.exit(f"ERROR: {a.output} exists; refusing to overwrite it.")

    with open(a.input, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    if any(c in header for c in NEW_COLS):
        sys.exit(f"ERROR: {a.input} already has provenance columns -- it is not a pre-fix file.")

    pos = {}
    for fld in SCORED_FIELDS:
        p = layer_positions(header, fld)
        if not p:
            continue
        if len(p) != EXPECTED_LAYERS:
            sys.exit(f"ERROR: {fld} has {len(p)} layer(s); LAYER_MODELS describes the "
                     f"{EXPECTED_LAYERS}-layer history of the committed outputs only. "
                     f"Refusing to attribute an unknown history.")
        if header[p[0]] != fld:
            sys.exit(f"ERROR: first {fld} layer is named {header[p[0]]!r}, not the canonical name.")
        pos[fld] = p
    if "predicted_probability" not in pos:
        sys.exit("ERROR: no predicted_probability layers found -- nothing to migrate.")
    layer_cols = {i for p in pos.values() for i in p[1:]}
    keep = [i for i in range(len(header)) if i not in layer_cols]
    fields = list(pos)
    out_header = [header[i] for i in keep] + [f"{f}_prev" for f in fields] + NEW_COLS
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    out_rows, chosen = [], {}
    pp = pos["predicted_probability"]
    for r in body:
        r = r + [""] * (len(header) - len(r))
        filled = [k for k, i in enumerate(pp) if r[i] != ""]
        vals = {f: r[pos[f][0]] for f in fields}
        prev = {f: "" for f in fields}
        md5, md5_prev, source = "", "", ""
        if filled:
            L = max(filled)
            vals = {f: r[pos[f][L]] for f in fields}
            older = [k for k in filled if k < L and _num(r[pp[k]]) != _num(r[pp[L]])]
            if older:
                P = max(older)
                prev = {f: (r[pos[f][P]] if r[pos[f][P]] != vals[f] else "") for f in fields}
                md5_prev = LAYER_MODELS[P][0]
            md5 = LAYER_MODELS[L][0]
            source = f"migrated {today}: {LAYER_MODELS[L][1]}"
            chosen[r[header.index("host")]] = L
        new = list(r)
        for f in fields:
            new[pos[f][0]] = vals[f]
        out_rows.append([new[i] for i in keep] + [prev[f] for f in fields]
                        + [md5, md5_prev, "unknown" if md5 else "", source])

    with open(a.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(out_header)
        w.writerows(out_rows)
    from collections import Counter
    by_model = Counter(LAYER_MODELS[L][0] for L in chosen.values())
    print(f"{a.input}: {len(header)} -> {len(out_header)} columns, {len(body)} rows; "
          f"{len(layer_cols)} layer columns collapsed into {len(fields)} canonical fields")
    print(f"  scored_model_md5: {dict(by_model)}; rows with no probability in any layer: "
          f"{len(body) - len(chosen)}")
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
