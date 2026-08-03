"""exofop_vetting.py -- surface TFOP working-group vetting notes as evidence.

WHAT THIS IS
ExoFOP-TESS publishes, per TOI, a free-text `Comments` field written by the
TFOP follow-up working groups ("TFOP FP; retired as NEB", "depth-aperture
correlation; centroid image towards SW in past QLP sectors; v-shaped
transits"). That is expert human reasoning in expert language. It is EXTERNAL
to this project -- not model output, not generated here -- and is attributed as
such wherever it is shown.

WHY "NOT FOUND" IS THE INTERESTING CASE HERE

`06_download_unknown.build_exclusion_set()` excludes confirmed planets AND
every TOI disposition before candidates are ever selected. Candidates are
therefore TOI-free by construction, and a live check confirmed it: 0 of 296
current candidates match an ExoFOP TOI.

So this panel is not a routinely-populated evidence block -- it is a
CROSS-CHECK, and its usual answer is a useful one: "this star carries no TFOP
record", i.e. nobody has flagged it, which is positive evidence of novelty
rather than a blank space.

The found-branch is a genuine safety net rather than dead code. TOIs are
alerted continuously, so a star surfaced today can be alerted next month, and
the exclusion set is only as fresh as the last pipeline run. If that ever
happens the user needs to know immediately, before spending telescope time.

SUBTYPE BADGES ARE DERIVED, AND LABELLED AS SUCH
The category (eclipsing_binary, nearby_eb_blend, ...) comes from keyword
matching over free text, using the conservative vocabulary from the multi-task
feasibility work. It is a reading aid, never authoritative -- the raw comment
is always displayed alongside as the source of truth.
"""
import os
import re
import csv
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_CSV = os.path.join(SCRIPT_DIR, "..", "code", "experiments",
                          "exofop_toi_export.csv")

# Same vocabulary as code/experiments/fp_subtype_feasibility.py, most-specific
# first. Kept in sync deliberately: this is a display aid derived from the same
# analysis, and divergence between the two would be a silent inconsistency.
SUBTYPE_PATTERNS = [
    ("nearby EB / blend", r"\bNEB\b|nearby eclipsing binary|blended? (?:with )?(?:nearby )?EB|contaminat"),
    ("eclipsing binary", r"\bEB\b|eclipsing binary|\bSB1\b|\bSB2\b|secondary eclipse|v-shaped|grazing"),
    ("stellar variability", r"variab|puls|rotat|spot|flare|RR Lyr|delta scuti"),
    ("systematic / artifact", r"systematic|artifact|artefact|momentum dump|scattered light|instrument|asteroid"),
    ("centroid offset", r"centroid|offset|photocenter|difference image"),
    ("odd/even mismatch", r"odd.even|odd/even"),
    ("stellar companion", r"companion|binary star|speckle|adaptive optics"),
]

_INDEX = None
_AS_OF = None


def _load():
    """Parse the cached export once, into {tic_id: record}.

    Uses csv rather than pandas: this is imported by the Flask process on every
    worker start, and the module should not drag pandas in just to read one
    file. Rows with a comment win over rows without when a TIC has several TOIs,
    so the most informative record is the one surfaced.
    """
    global _INDEX, _AS_OF
    if _INDEX is not None:
        return _INDEX
    idx = {}
    if not os.path.exists(EXPORT_CSV):
        _INDEX = idx
        return idx
    _AS_OF = datetime.datetime.fromtimestamp(
        os.path.getmtime(EXPORT_CSV)).strftime("%Y-%m-%d")
    with open(EXPORT_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            raw = (row.get("TIC ID") or "").strip()
            m = re.search(r"(\d+)", raw)
            if not m:
                continue
            tic = int(m.group(1))
            comment = (row.get("Comments") or "").strip()
            prev = idx.get(tic)
            if prev is not None and prev.get("comment") and not comment:
                continue      # keep the more informative record
            idx[tic] = {
                "tic_id": tic,
                "toi": (row.get("TOI") or "").strip(),
                "tfopwg_disposition": (row.get("TFOPWG Disposition") or "").strip(),
                "tess_disposition": (row.get("TESS Disposition") or "").strip(),
                "comment": comment,
                "sg": {k: (row.get(k) or "").strip()
                       for k in ("SG1A", "SG1B", "SG2", "SG3", "SG4", "SG5")},
            }
    _INDEX = idx
    return idx


DISPOSITION_LABELS = {
    "FP": "False positive",
    "FA": "False alarm",
    "KP": "Known planet",
    "CP": "Confirmed planet",
    "PC": "Planet candidate",
    "APC": "Ambiguous planet candidate",
}


def classify_comment(text):
    """Best-effort category for a free-text comment. Returns None when nothing
    matches -- deliberately, rather than forcing an 'other' bucket that would
    imply more confidence than keyword matching earns."""
    if not text:
        return None
    for name, pat in SUBTYPE_PATTERNS:
        if re.search(pat, text, flags=re.I):
            return name
    return None


def as_of_date():
    _load()
    return _AS_OF


def lookup(tic_id):
    """Vetting record for one star, shaped for direct template use.

    Always returns a dict with `found`; never raises and never returns None, so
    the template has no special-casing to do and a missing export file degrades
    to 'no record' rather than a 500.
    """
    idx = _load()
    try:
        tic = int(tic_id)
    except (TypeError, ValueError):
        tic = None
    rec = idx.get(tic) if tic is not None else None

    base = {
        "found": False,
        "as_of": _AS_OF,
        "catalog_size": len(idx),
        "exofop_url": (f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic}"
                       if tic is not None else None),
    }
    if rec is None:
        return base

    disp = rec["tfopwg_disposition"]
    # SG columns are a status code, not a subtype -- 1,146 of 1,152 negatives
    # read '5'. Only surface one when it deviates from that default, so the
    # panel shows signal rather than six identical fives.
    sg_notable = {k: v for k, v in rec["sg"].items() if v and v != "5"}

    return {
        **base,
        "found": True,
        "toi": rec["toi"],
        "disposition_code": disp,
        "disposition_label": DISPOSITION_LABELS.get(disp, disp or "unspecified"),
        "is_negative_disposition": disp in ("FP", "FA"),
        "comment": rec["comment"],
        "subtype": classify_comment(rec["comment"]),
        "sg_notable": sg_notable,
        "tess_disposition": rec["tess_disposition"],
    }
