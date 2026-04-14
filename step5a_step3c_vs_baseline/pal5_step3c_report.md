# Pal 5 step 3c: Bonaca-style comparison report

This report compares the current Pal 5 strict-sample morphology baseline to the
headline reference values described in Bonaca+2020 Figure 3 / Section 3.

## Inputs

- Baseline run: **step5a empirical bg + MAP**
- eta_mode: `offstream_empirical_fixedbg`
- sampler: `map`
- Strict sample size: 444232
- Alternate run: **step3b control + MAP**
- Alternate eta_mode: `control`
- Alternate sampler: `map`

## Bonaca reference values used for quick comparison

- Magnitude range: 20 < g0 < 23
- Total stream stars excluding progenitor: 3000 ± 100
- Stream extent: trailing ≈ 16 deg, leading ≈ 8 deg
- Trailing / leading within 5 deg: ≈ 1.5
- Near cluster width: ≈ 0.15 deg
- Leading fan width: ≈ 0.40 deg at phi1 ≈ 7 deg

## Derived metrics

| metric | baseline | alternate | Bonaca ref |
|---|---:|---:|---:|
| successful bins | 41 | 39 | 41 bins modeled |
| successful bins excluding cluster | 39 | 37 | -- |
| trailing extent [deg] | 20.00 | 19.25 | 16.0 |
| leading extent [deg] | 10.00 | 9.25 | 8.0 |
| integrated stars, |phi1| < 8 | 4839 | 2872 | 3000 |
| trailing / leading, |phi1| < 8 | 1.18 | 1.68 | ~1.0 |
| trailing / leading, |phi1| < 5 | 1.14 | 1.75 | 1.50 |
| near-cluster width [deg] | 0.110 | 0.118 | 0.150 |
| leading max width in [5, 8] [deg] | 0.311 @ 7.00 | 0.287 @ 7.00 | 0.400 @ 7.0 |
| trailing max width in [-15, -5] [deg] | 0.358 @ -12.50 | 0.554 @ -10.25 | trailing should stay relatively thin |

## Notes

- The integrated star counts above use the **linear density profile times the phi1 bin spacing**.
  This is the closest like-for-like quantity to Bonaca's integrated stream-star count.
- Cluster bins are excluded from the like-for-like totals and from the arm polynomial fits.
- Width is usually the least stable profile; use the MAP run as the baseline morphology
  and use the emcee run primarily as a posterior sanity check.

## Wide / suspicious baseline bins

| phi1_center | sigma_map [deg] | track_resid_map [deg] |
|---:|---:|---:|
| -20.00 | 0.463 | -0.087 |
| -19.25 | 0.475 | -0.273 |
| -18.50 | 0.355 | 0.055 |
| -17.75 | 0.433 | -0.008 |
| -12.50 | 0.358 | 0.219 |
