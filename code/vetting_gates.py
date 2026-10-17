"""
vetting_gates.py -- physics checks run on every candidate before it is shown as scored.

The checks are the vetting's own (audit/vetting/report.md, register section M), unchanged in
definition. dip, odd_even, phase_scan and geometry are verbatim copies of
audit/vetting/part_b_lib.py, part_b.py and part_b2.py; `raw_checks` is the Phase 1 gate run
(2026-10-17) minus the two checks deliberately not implemented. The binning (bins of
duration/5, baseline >= 1 duration away, +/-3 durations) is web/folded_plot.zoom_stats, shared
with the page's folded figure by design: the checks measure what the plot shows.

Gates FLAG; they never delete a candidate. Roles, from the Phase 1 results (register M):

  BLOCK  geometry   orbit inside the star even at 5 Msun, or a transit longer than 2x the
                    longest possible even at the most favorable plausible mass. 0 of 829 real
                    planets flagged. Failing only at the stored/default mass is a WARN.
  WARN   secondary  dip at phase 0.5 >= 5 sigma (3-5 sigma is reported as weaker info).
         odd/even   odd vs even transit depths differ >= 5.24 sigma.
         structure  a larger dip or bump elsewhere in the orbit than the transit itself.
         visibility binned depth < 3x the bin-to-bin scatter.
  INFO   alias      folding at P/2, P/3, 2P or 3P gives a dip > 2x stronger than at P.

NOT IMPLEMENTED, on purpose:
  - strict "stored P must be the best of the alias set": flagged 462 of 829 real planets
    (55.7%) and 45.9% of injected transits; the alias estimator is too noisy (median winning
    margin 1.29x). Only the > 2x margin is kept, as INFO.
  - off-phase dip scan as its own flag: flagged 52.2% of real planets against 39.7% of false
    positives, i.e. no separation. The scan is still run, because the structure check
    compares the transit against its strongest dip and strongest bump.
"""
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np

_WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
if _WEB not in sys.path:
    sys.path.insert(0, _WEB)
import folded_plot as fp  # noqa: E402

GATE_CODE_VERSION = "2"   # 2: geometry blocks only at any plausible mass (2026-10-17)

# ---- calibration constants, with their source ------------------------------------------------
SIG = 3.0                          # base significance used inside the verbatim checks (part_b.py)
OE_SIG = 5.238778034856423         # odd/even WARN: 99th percentile of the transit-free null
                                   # (audit/vetting/null_calibration.json, oe_null_p99, 2026-10-12)
SECONDARY_WARN_SIG = 5.0           # phase-0.5 WARN: masked-injection false-alarm rate 2.4% (2026-10-17)
SECONDARY_INFO_SIG = 3.0           # phase-0.5 info: false-alarm rate ~10% (9.8%) on the same test
VISIBILITY_MIN = 3.0               # binned depth / bin scatter; the vetting's visible-transit bar
GEOMETRY_MAX_DURATION_RATIO = 2.0  # BLOCK above this; at 1.5 two real planets would be flagged
# The geometry BLOCK must hold at every plausible stellar mass, not just the stored one, because
# a/R* scales as M^(1/3) and the stored mass is often a solar default. The real-planet validation
# (829 planets) had 1 defaulted mass and hosts <= 3.5 Rsun, so for giants the block rests on the
# physics alone (register M, 2026-10-17). Blocks that hold only at the stored/default mass WARN.
GEOMETRY_MASS_CEILING_MSUN = 5.0   # "inside the star" blocks only if inside even at 5 Msun: giants span
                                   # ~1-3 Msun and bright giants reach 3-5; TIC gives no mass for them
GEOMETRY_MASS_FLOOR_MSUN = 0.08    # hydrogen-burning limit: the most favorable mass when none was measured
GEOMETRY_MEASURED_MASS_LOW = 0.5   # the most favorable plausible mass for a catalogue mass is half of it
                                   # (a lower mass makes the longest possible transit longer)
ALIAS_INFO_MARGIN = 2.0            # alias winner / stored-P ratio; real planets 7.5%, FPs 7.6%
GIANT_RADIUS_RSUN = 2.0            # above this the block message asks for the stellar parameters
                                   # to be checked: a misclassified giant is the gate's failure mode
AU_RSUN = 215.032

LIGHTCURVE_DIRS = ("processed_unknown", "processed_unknown_widesector")


# ---- verbatim from audit/vetting/part_b_lib.py ------------------------------------------------
def dip(phase, flux, lay, centre=0.0):
    """Binned dip at `centre` (phase units): depth, error, significance, visibility ratio."""
    ph = np.mod(phase - centre + 0.5, 1.0) - 0.5
    st = fp.zoom_stats(ph, flux, lay)
    if st["ratio"] is None or not st["scatter"]:
        return dict(depth=None, err=None, sig=None, ratio=None)
    err = st["scatter"] / np.sqrt(st["n_in"])
    return dict(depth=st["depth"], err=err, sig=st["depth"] / err, ratio=st["ratio"])


# ---- verbatim from audit/vetting/part_b.py ----------------------------------------------------
def odd_even(t, f, P, T0, dh):
    """Fold at 2P: transits with even epoch number land at phase 0, odd at 0.5."""
    lay2 = fp.layout(2 * P, dh); ph2 = fp.fold(t, T0, 2 * P)
    ev, od = dip(ph2, f, lay2, 0.0), dip(ph2, f, lay2, 0.5)
    if ev["depth"] is None or od["depth"] is None:
        return dict(oe_even_ppm=None if ev["depth"] is None else ev["depth"] * 1e6,
                    oe_odd_ppm=None if od["depth"] is None else od["depth"] * 1e6, oe_sig=None, oe_testable=False)
    s = abs(ev["depth"] - od["depth"]) / np.hypot(ev["err"], od["err"])
    return dict(oe_even_ppm=ev["depth"] * 1e6, oe_odd_ppm=od["depth"] * 1e6, oe_sig=s, oe_testable=True)


def phase_scan(phase, flux, lay):
    """Deepest dip away from the primary (and the same scan on -flux, a noise-only null)."""
    d = lay["dur_phase"]; step = max(d / 2, 1.0 / 400)
    grid = np.arange(2 * d, 1 - 2 * d + 1e-12, step)
    best, best_phi, null = -np.inf, None, -np.inf
    for phi in grid:
        a = dip(phase, flux, lay, phi); b = dip(phase, -flux + 2 * np.median(flux), lay, phi)
        if a["sig"] is not None and a["sig"] > best: best, best_phi = a["sig"], phi
        if b["sig"] is not None and b["sig"] > null: null = b["sig"]
    return (None if best_phi is None else float(best)), (None if best_phi is None else float(best_phi)), (None if null == -np.inf else float(null))


# ---- verbatim from audit/vetting/part_b2.py ---------------------------------------------------
def geometry(ch):
    """Longest possible (circular, b = 0) transit duration on this star, a/R*, and the impact
    parameter the observed duration would need. Mass defaults to 1 Msun where the pipeline
    defaulted it (flagged)."""
    P, Rs = ch.get("period"), ch.get("st_rad"); Ms = ch.get("st_mass") or 1.0
    if not (P and Rs): return (None, None, None)
    a = (Ms * (P / 365.25) ** 2) ** (1 / 3) * AU_RSUN            # R_sun
    k = np.sqrt(max((ch.get("transit_depth_ppm") or 0) * 1e-6, 0))
    if a <= Rs: return (None, a / Rs, None)                       # the orbit would lie inside the star
    tmax = 24 * P / np.pi * np.arcsin(min(1.0, Rs * (1 + k) / a))
    r = ch["transit_duration_hours"] / tmax
    b = float(np.sqrt(max(0.0, 1 - r ** 2)) * (1 + k)) if r <= 1 else None
    return (tmax, a / Rs, b)


# ---- the Phase 1 gate computation ----------------------------------------------------------------
def _num(v):
    """None for missing/NaN/non-positive, else float -- the Phase 1 inputs arrived as JSON
    nulls, so a CSV NaN must become None before geometry() sees it."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def raw_checks(t, f, P, T0, dh, depth_frac, st_rad, st_mass, mass_defaulted=None):
    """Measured values for every check (no roles applied)."""
    lay = fp.layout(P, dh); ph = fp.fold(t, T0, P); prim = dip(ph, f, lay)
    al = {}
    for name, k in (("P", 1.0), ("P/2", 0.5), ("P/3", 1 / 3), ("2P", 2.0), ("3P", 3.0)):
        al[name] = dip(fp.fold(t, T0, P * k), f, fp.layout(P * k, dh))["ratio"]
    fin = {k: v for k, v in al.items() if v is not None}
    best = max(fin, key=fin.get) if fin else None
    oe = odd_even(t, f, P, T0, dh)
    sec = dip(ph, f, lay, 0.5)
    bsig, bphi, null = phase_scan(ph, f, lay)
    geo = geometry_values(P, dh, depth_frac, st_rad, st_mass, mass_defaulted)
    return dict(visibility_ratio=prim["ratio"], primary_sig=prim["sig"], alias_best=best,
                alias_margin=(None if best in (None, "P") or not al.get("P") else fin[best] / al["P"]),
                oddeven_sig=oe["oe_sig"], secondary_sig=sec["sig"], structure_best_dip_sig=bsig,
                structure_best_dip_phase=bphi, structure_best_bump_sig=null, **geo)


def geometry_values(P, dh, depth_frac, st_rad, st_mass, mass_defaulted=None):
    """geometry() at the stored (or 1 Msun default) mass, at the mass ceiling, and at the most
    favorable plausible mass for the duration test."""
    base = dict(period=P, st_rad=st_rad, transit_depth_ppm=(depth_frac or 0) * 1e6, transit_duration_hours=dh)
    tmax, a_rs, b = geometry(dict(base, st_mass=st_mass))
    if mass_defaulted is None:
        mass_defaulted = st_mass is None
    m_used = st_mass if st_mass else 1.0
    m_fav = GEOMETRY_MASS_FLOOR_MSUN if mass_defaulted else max(GEOMETRY_MASS_FLOOR_MSUN, GEOMETRY_MEASURED_MASS_LOW * m_used)
    _, a_c, b_c = geometry(dict(base, st_mass=GEOMETRY_MASS_CEILING_MSUN))
    t_f, _, _ = geometry(dict(base, st_mass=m_fav))
    return dict(a_over_rstar=a_rs, duration_max_h=tmax, duration_over_max=(dh / tmax if tmax else None),
                a_over_rstar_at_ceiling=a_c, impact_needed_at_ceiling=b_c, mass_favorable=m_fav,
                duration_over_max_at_favorable=(dh / t_f if t_f else None))


def apply_roles(v, ran):
    """Flags from measured values. `ran` = the light-curve checks were computed; when they were,
    the Phase 1 rules apply exactly (an unmeasurable dip counts as not visible / not dominant).
    Returns the flag fields plus block_reason / warnings / info."""
    inside = v.get("a_over_rstar") is not None and v["a_over_rstar"] <= 1
    long_ = v.get("duration_over_max") is not None and v["duration_over_max"] > GEOMETRY_MAX_DURATION_RATIO
    inside_c = v.get("a_over_rstar_at_ceiling") is not None and v["a_over_rstar_at_ceiling"] <= 1
    long_f = (v.get("duration_over_max_at_favorable") is not None
              and v["duration_over_max_at_favorable"] > GEOMETRY_MAX_DURATION_RATIO)
    blocked = inside_c or long_f
    flags = dict(
        geometry_flag=bool(blocked), geometry_warn=bool((inside or long_) and not blocked),
        orbit_inside_star=bool(inside), orbit_inside_at_ceiling=bool(inside_c),
        secondary_flag=bool(v.get("secondary_sig") is not None and v["secondary_sig"] >= SECONDARY_WARN_SIG),
        secondary_info=bool(v.get("secondary_sig") is not None and SECONDARY_INFO_SIG <= v["secondary_sig"] < SECONDARY_WARN_SIG),
        oddeven_flag=bool(v.get("oddeven_sig") is not None and v["oddeven_sig"] >= OE_SIG),
        structure_flag=bool(ran and not (
            v.get("primary_sig") is not None and v.get("structure_best_dip_sig") is not None
            and v.get("structure_best_bump_sig") is not None
            and v["primary_sig"] > max(v["structure_best_dip_sig"], v["structure_best_bump_sig"]))),
        visibility_flag=bool(ran and not (v.get("visibility_ratio") is not None and v["visibility_ratio"] >= VISIBILITY_MIN)),
        alias_info=bool(v.get("alias_margin") is not None and v["alias_margin"] > ALIAS_INFO_MARGIN),
    )
    block = []
    if inside_c: block.append(f"orbit inside the star at any mass up to {GEOMETRY_MASS_CEILING_MSUN:g} Msun")
    elif long_f: block.append("transit longer than the star allows at any plausible mass")
    warn = [n for n, k in (("geometry fails only at the assumed stellar mass", "geometry_warn"),
                           ("secondary eclipse at phase 0.5", "secondary_flag"), ("odd/even depth difference", "oddeven_flag"),
                           ("larger structure elsewhere in the orbit", "structure_flag"), ("transit not clearly visible", "visibility_flag")) if flags[k]]
    info = [n for n, k in (("weak dip at phase 0.5", "secondary_info"), ("stronger dip at an alias period", "alias_info")) if flags[k]]
    return dict(flags, block_reason="; ".join(block), warnings="; ".join(warn), info="; ".join(info))


def lightcurve_path(host, project_root):
    for d in LIGHTCURVE_DIRS:
        p = os.path.join(project_root, "data", d, f"{host}.csv")
        if os.path.exists(p):
            return p, f"data/{d}/{host}.csv"
    return None, None


def gate_fields(row, project_root, computed_at=None):
    """All vet_* fields for one candidate (a characterization row or dict). Never raises: a
    light curve that cannot be read leaves the light-curve checks empty and says why, and the
    geometry gate (which needs no light curve) is still evaluated."""
    host = row.get("host")
    P, T0, dh = _num(row.get("period")), row.get("T0"), _num(row.get("transit_duration_hours"))
    depth_ppm = _num(row.get("transit_depth_ppm"))
    depth_frac = depth_ppm * 1e-6 if depth_ppm else 0.0
    st_rad, st_mass = _num(row.get("st_rad")), _num(row.get("st_mass"))
    mass_defaulted = st_mass is None or bool(row.get("stellar_mass_was_defaulted_to_solar") in (True, "True", 1))
    out = dict(vet_code_version=GATE_CODE_VERSION,
               vet_computed_at=computed_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
               vet_geom_st_rad=st_rad, vet_geom_st_mass_used=(st_mass if st_mass else 1.0),
               vet_geom_mass_defaulted=mass_defaulted,
               vet_geom_mass_ceiling=GEOMETRY_MASS_CEILING_MSUN,
               vet_geom_giant=bool(st_rad is not None and st_rad > GIANT_RADIUS_RSUN))
    values = dict(visibility_ratio=None, primary_sig=None, alias_best=None, alias_margin=None, oddeven_sig=None,
                  secondary_sig=None, structure_best_dip_sig=None, structure_best_dip_phase=None,
                  structure_best_bump_sig=None, a_over_rstar=None, duration_max_h=None, duration_over_max=None,
                  a_over_rstar_at_ceiling=None, impact_needed_at_ceiling=None, mass_favorable=None,
                  duration_over_max_at_favorable=None)
    if P and dh:
        values.update(geometry_values(P, dh, depth_frac, st_rad, st_mass, mass_defaulted))
    path, rel = lightcurve_path(host, project_root) if host else (None, None)
    status = "ok"
    if not (P and dh) or T0 is None or (isinstance(T0, float) and T0 != T0):
        status = "no stored period/T0/duration: light-curve checks not run"
    elif path is None:
        status = "processed light curve not found: light-curve checks not run"
    else:
        try:
            t, f = fp.load_curve(path)
            values = raw_checks(t, f, P, float(T0), dh, depth_frac, st_rad, st_mass, mass_defaulted)
            out["vet_lc_source"], out["vet_lc_points"] = rel, int(len(t))
        except Exception as e:
            status = f"light-curve checks failed: {type(e).__name__}: {str(e)[:100]}"
    roles = apply_roles(values, ran=(status == "ok"))
    out.update({f"vet_{k}": v for k, v in values.items()})
    out.update({f"vet_{k}": v for k, v in roles.items()})
    out["vet_status"] = status
    out.setdefault("vet_lc_source", rel)
    return out


def add_gate_fields(full, project_root, computed_at=None):
    """Adds/overwrites the vet_* columns on a characterization DataFrame, every row."""
    import pandas as pd
    stamp = computed_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rows = [gate_fields(r.to_dict(), project_root, stamp) for _, r in full.iterrows()]
    vet = pd.DataFrame(rows, index=full.index)
    full = full.drop(columns=[c for c in full.columns if c.startswith("vet_")])
    return pd.concat([full, vet], axis=1)


def geometry_blocked(full):
    """Boolean Series for the combined filter. A characterization written before the gates
    existed has no vet_geometry_flag column, and is treated as not blocked."""
    import pandas as pd
    if "vet_geometry_flag" not in full.columns:
        return pd.Series(False, index=full.index)
    return full["vet_geometry_flag"].map(lambda v: v in (True, "True", 1, 1.0)).astype(bool)
