"""centroid_variant_sensitivity.py -- why the DIRECT photocenter shift is the
weaker form of a test this project already closed.

Run this before re-proposing any "center-of-light shift in-transit vs
out-of-transit" feature. It is the evidence behind the PROPOSED AND REJECTED
entry in RESULTS_SUMMARY.md.

THE TWO VARIANTS

  difference-image (BUILT, and closed as inert at 77.6% coverage):
      centroid( median_OOT_image - median_IT_image ), compared to the target's
      catalog position via the TPF WCS. Isolates only the flux that CHANGED,
      so the centroid sits on whatever star actually dimmed.

  direct photocenter shift (the recurring proposal):
      centroid(IT image) - centroid(OOT image). Measures how far the total
      light of the whole stamp moved.

WHY THEY ARE NOT INTERCHANGEABLE

The direct shift is diluted by every photon that did NOT change. For a transit
of fractional aperture depth d and a true source offset r, the whole-stamp
photocenter moves by roughly d*r, while the difference image places the
centroid at r regardless of d. Real transits here are ~0.01-1% deep, so the
direct statistic is ~100-10,000x smaller than the quantity the difference
image measures directly -- and it shrinks exactly as the signal gets harder.

This simulation makes that concrete on an 11x11 TESS-like stamp with the
transit deliberately placed on a CONTAMINANT 2 px from the target: the blend
scenario both variants exist to catch.
"""
import numpy as np
from scipy.ndimage import center_of_mass

NY = NX = 11
TARGET_RC = (5.0, 5.0)
CONTAM_RC = (5.0, 7.0)          # 2.0 px away
TARGET_AMP, CONTAM_AMP = 1000.0, 300.0
PSF_SIGMA = 1.0
DIPS = (0.50, 0.20, 0.05, 0.01)  # fraction of the CONTAMINANT's flux removed


def gaussian_source(ny, nx, r0, c0, amp, sigma=PSF_SIGMA):
    y, x = np.mgrid[0:ny, 0:nx]
    return amp * np.exp(-((y - r0) ** 2 + (x - c0) ** 2) / (2 * sigma ** 2))


def main():
    true_offset = float(np.hypot(CONTAM_RC[0] - TARGET_RC[0],
                                 CONTAM_RC[1] - TARGET_RC[1]))
    target = gaussian_source(NY, NX, *TARGET_RC, TARGET_AMP)
    contam = gaussian_source(NY, NX, *CONTAM_RC, CONTAM_AMP)

    print("=" * 78)
    print("CENTROID VARIANT SENSITIVITY -- transit placed on a contaminant "
          f"{true_offset:.3f} px away")
    print("=" * 78)
    print(f"  {'aperture depth':>16}{'DIRECT shift':>16}{'DIFF-image offset':>20}"
          f"{'direct/true':>14}")
    rows = []
    for dip in DIPS:
        oot = target + contam
        itr = target + contam * (1.0 - dip)
        ap_depth = 1.0 - itr.sum() / oot.sum()

        c_oot = np.array(center_of_mass(oot))
        c_itr = np.array(center_of_mass(itr))
        direct = float(np.hypot(*(c_itr - c_oot)))

        diff = np.clip(oot - itr, 0, None)
        c_diff = np.array(center_of_mass(diff))
        diff_off = float(np.hypot(*(c_diff - np.array(TARGET_RC))))

        print(f"  {ap_depth*100:>15.2f}%{direct:>16.5f}{diff_off:>20.5f}"
              f"{direct/true_offset:>14.4f}")
        rows.append((ap_depth, direct, diff_off))

    print()
    print(f"  true contaminant offset: {true_offset:.3f} px")
    print("  The difference image recovers it at EVERY depth. The direct")
    print("  photocenter shift scales with depth and collapses toward zero,")
    print("  reaching ~0.004 px at the 0.2%-deep end -- far below the noise")
    print("  floor of a real per-cadence TESS centroid measurement.")
    print()
    print("  CONCLUSION: the direct variant is not a different test, it is the")
    print("  same test with the signal divided by the transit depth.")
    return rows


if __name__ == "__main__":
    main()
