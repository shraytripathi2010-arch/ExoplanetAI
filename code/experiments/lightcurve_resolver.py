"""lightcurve_resolver.py -- host -> processed light curve, scoped by population.

Twelve scripts used to carry their own lookup, and every variant was wrong in
some way:

  label -> folder (05c, 05d, compute_new_features)
      label 1 -> data/processed, label 0 -> data/processed_negative. Misses the
      label-0 stars the retrain pipeline wrote into data/processed before
      2026-09-27, and every curve in data/retrain_pipeline/processed/.
  first match over a fixed folder list (the other nine)
      Training-only scripts searched processed + processed_negative and missed
      data/retrain_pipeline/processed/. trapezoid_shape, oddeven_timing and
      timeseries_stats searched both pools too, so a widesector host that also
      has a single-sector curve resolved to processed_unknown/ first -- 13 wrong
      trapezoid rows (RESULTS_SUMMARY.md register section F).

Scopes -- a resolver searches its own scope and nothing else:

  training    data/processed, data/processed_negative,
              data/retrain_pipeline/processed
  pool        data/processed_unknown
  widesector  data/processed_unknown_widesector

Never searched: data/processed_8sector (multi-sector stitch),
code/kepler_pilot/processed and code/k2_pilot/processed (Kepler/K2, not
TESS), and every raw folder (not preprocessed). Each is a different product,
and substituting it for a processed curve is silently wrong.

Rules:
  * Within a scope a host matches at most one file. Two is a
    DuplicateCurveError: nothing here chooses between them.
  * Across scopes the same host is expected. processed_unknown and
    processed_unknown_widesector hold genuinely different curves of the same
    star; a widesector resolver never sees the single-sector one.
  * The folder says nothing about the label. data/processed holds label-0
    stars written by the retrain pipeline, and retrain_pipeline/processed
    holds both labels. Labels come from training.csv, never from here.
  * Every caller reports how many hosts it resolved and how many it could not
    (provenance() / summary()) -- the same rule as candidate_pool.py.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

SCOPES = {
    "training": ("data/processed", "data/processed_negative",
                 "data/retrain_pipeline/processed"),
    "pool": ("data/processed_unknown",),
    "widesector": ("data/processed_unknown_widesector",),
}

NEVER_SEARCHED = ("data/processed_8sector", "code/kepler_pilot/processed",
                  "code/k2_pilot/processed")

for _folders in SCOPES.values():
    for _f in _folders:
        assert _f not in NEVER_SEARCHED and "raw" not in _f.split("/"), _f


class DuplicateCurveError(RuntimeError):
    pass


class CurveResolver:
    """Resolves hosts within one scope and keeps count of what it found.

    ROOT is read when the resolver is built, so a check can point it at
    another checkout's data/ without touching the scripts that use it.
    """

    def __init__(self, scope):
        if scope not in SCOPES:
            raise KeyError(f"unknown scope {scope!r}; expected one of {sorted(SCOPES)}")
        self.scope = scope
        self.folders = SCOPES[scope]
        self._dirs = [os.path.join(ROOT, f) for f in self.folders]
        self._hits = {}

    def find(self, host):
        """The one processed curve for `host` in this scope, or None."""
        host = str(host)
        if host in self._hits:
            return self._hits[host]
        hits = [p for p in (os.path.join(d, host + ".csv") for d in self._dirs)
                if os.path.isfile(p)]
        if len(hits) > 1:
            raise DuplicateCurveError(
                f"{host} has {len(hits)} curves in scope {self.scope!r}: {hits}")
        self._hits[host] = hits[0] if hits else None
        return self._hits[host]

    def resolve(self, hosts):
        """find() for each host, in order."""
        return [self.find(h) for h in hosts]

    def provenance(self):
        """Counts for the caller to store in its own output."""
        found = {h: p for h, p in self._hits.items() if p is not None}
        missed = sorted(h for h, p in self._hits.items() if p is None)
        by_folder = {f: 0 for f in self.folders}
        for p in found.values():
            by_folder[os.path.relpath(os.path.dirname(p), ROOT)] += 1
        return {"scope": self.scope, "folders": list(self.folders),
                "n_hosts": len(self._hits), "n_resolved": len(found),
                "n_unresolved": len(missed), "resolved_by_folder": by_folder,
                "unresolved_examples": missed[:10]}

    def summary(self):
        p = self.provenance()
        where = ", ".join(f"{f.removeprefix('data/')} {n}"
                          for f, n in p["resolved_by_folder"].items())
        return (f"light curves [{p['scope']}]: {p['n_resolved']}/{p['n_hosts']} "
                f"resolved ({where}); {p['n_unresolved']} not found")
