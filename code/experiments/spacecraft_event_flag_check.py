"""Is AttitudeTweak (bit 1) a genuinely distinct, usable spacecraft-event class,
or does it face the same stripping problem as Desat (bit 32)?

DRNs say bit 1 was unused early and appears after ~Sector 27. Bit 1 IS in
lightkurve's DEFAULT_BITMASK (17087), so like bit 32 it is stripped at download.
The momentum-dump task showed the schedule can still be reconstructed from
quality_bitmask=0 reference downloads -- so the fair question is whether
AttitudeTweak events EXIST and are frequent enough to carry information."""
import warnings, numpy as np; warnings.filterwarnings("ignore")
import lightkurve as lk

BITS = {1:"AttitudeTweak", 2:"SafeMode", 4:"CoarsePoint", 8:"EarthPoint",
        16:"Argabrightening", 32:"Desat(momentum dump)", 128:"ManualExclude"}
targets = ["TIC 261136679","TIC 38846515","TIC 100100827","TIC 260647166"]
seen = {b:0 for b in BITS}; ev = {b:0 for b in BITS}
nsec = 0
for t in targets:
    try:
        sr = lk.search_lightcurve(t, mission="TESS", author="SPOC", cadence=120)
    except Exception:
        continue
    if not len(sr): continue
    # spread across the mission, including post-S27 where bit 1 is documented
    for i in list(range(0, len(sr), max(1, len(sr)//6)))[:6]:
        try:
            lc = sr[i].download(quality_bitmask=0)
        except Exception:
            continue
        q = np.asarray(lc.quality); tm = np.asarray(lc.time.value)
        sec = lc.meta.get("SECTOR")
        nsec += 1
        line = []
        for b,name in BITS.items():
            m = (q & b).astype(bool)
            n = int(m.sum()); seen[b]+=n
            if n:
                d = np.sort(tm[m & np.isfinite(tm)])
                grp = 1 + int((np.diff(d) > 0.05).sum()) if len(d)>1 else 1
                ev[b]+=grp
                line.append(f"{name}={n}c/{grp}ev")
        print(f"  s{sec:<4} n={len(q):<7} " + ("  ".join(line) if line else "(no flagged cadences)"), flush=True)
print(f"\n=== {nsec} sector-downloads at quality_bitmask=0 ===")
print(f"{'bit':>5} {'flag':<24}{'cadences':>10}{'events':>9}")
for b,name in BITS.items():
    print(f"{b:>5} {name:<24}{seen[b]:>10,}{ev[b]:>9}")
