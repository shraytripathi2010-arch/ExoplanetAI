"""
apply_vetting_gates.py -- compute the physics checks OFFLINE for the existing characterized
candidates and rebuild the two lists, without a full 08 run (no network, no rescoring).

  python3 apply_vetting_gates.py                 # dry run: report only, write nothing
  python3 apply_vetting_gates.py --apply         # write the three results files

Reads   results/unknown_candidates/characterized_candidates.csv and the processed light curves
        (data/processed_unknown*/), read-only.
Writes  (--apply only) characterized_candidates.csv with the vet_* columns added, and
        final_best_candidates.csv / needs_manual_blend_review.csv rebuilt by 08's own
        build_candidate_lists(), which now includes the geometry block.
Never removes a candidate from characterized_candidates.csv.

Guard: before touching anything, the two lists are rebuilt from the current file WITHOUT the
gates and compared with the lists on disk. If they differ, this script is not reproducing
08's output and it stops.
"""
import argparse
import importlib
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vetting_gates  # noqa: E402

m08 = importlib.import_module("08_characterize_candidates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=m08.RESULTS_FOLDER)
    ap.add_argument("--project-root", default=m08.PROJECT_ROOT, help="where data/processed_unknown*/ live")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    char_p = os.path.join(a.results_dir, "characterized_candidates.csv")
    best_p = os.path.join(a.results_dir, "final_best_candidates.csv")
    rev_p = os.path.join(a.results_dir, "needs_manual_blend_review.csv")
    for p in (char_p, best_p, rev_p):
        m08.refuse_dup_columns(p)
    full = pd.read_csv(char_p)
    best0, rev0 = pd.read_csv(best_p), pd.read_csv(rev_p)

    # guard: 08's list logic, without gates, must reproduce the lists on disk
    plain = full.drop(columns=[c for c in full.columns if c.startswith("vet_")])
    b, r, *_ = m08.build_candidate_lists(plain)
    same = set(b.host) == set(best0.host) and set(r.host) == set(rev0.host)
    print(f"guard: rebuilt without gates -> best {len(b)} (on disk {len(best0)}), review {len(r)} (on disk {len(rev0)}); "
          f"same membership: {same}")
    if not same:
        sys.exit("STOP: the rebuilt lists differ from the files on disk; not applying.")

    gated = vetting_gates.add_gate_fields(full, a.project_root)
    best, rev, strict, lenient, review = m08.build_candidate_lists(gated)
    g = gated.set_index("host")
    print(f"\ngates for {len(gated)} candidates: status {g.vet_status.value_counts().to_dict()}")
    for k in ("geometry_flag", "orbit_inside_star", "secondary_flag", "secondary_info", "oddeven_flag",
              "structure_flag", "visibility_flag", "alias_info"):
        print(f"  vet_{k:<20} {int(g['vet_' + k].astype(bool).sum())}")
    print(f"\nfinal_best_candidates: {len(best0)} -> {len(best)}; needs_manual_blend_review: {len(rev0)} -> {len(rev)}")
    for name, before, after in (("final_best", best0, best), ("needs_review", rev0, rev)):
        left = sorted(set(before.host) - set(after.host)); joined = sorted(set(after.host) - set(before.host))
        print(f"  {name}: {len(left)} leave, {len(joined)} join")
        for h in left:
            print(f"    - {h}: {g.loc[h, 'vet_block_reason']} (a/R* {g.loc[h, 'vet_a_over_rstar']:.2f}, "
                  f"duration/max {g.loc[h, 'vet_duration_over_max'] if pd.notna(g.loc[h, 'vet_duration_over_max']) else float('nan'):.2f}, "
                  f"R* {g.loc[h, 'vet_geom_st_rad']}, mass defaulted {g.loc[h, 'vet_geom_mass_defaulted']})")
    if not a.apply:
        print("\ndry run: nothing written (pass --apply to write the three files)")
        return
    gated.to_csv(char_p, index=False)
    best.to_csv(best_p, index=False)
    rev.to_csv(rev_p, index=False)
    print(f"\nwritten: {char_p}, {best_p}, {rev_p}")


if __name__ == "__main__":
    main()
