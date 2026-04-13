# Pal 5 step 3c: Bonaca-style comparison report

This report compares the current Pal 5 strict-sample morphology baseline to the
headline reference values described in Bonaca+2020 Figure 3 / Section 3.

## Inputs

- Baseline run: **refined DM + control + MAP**
- eta_mode: `control`
- sampler: `map`
- Strict sample size: 460278
- Alternate run: **refined DM + control + emcee**
- Alternate eta_mode: `control`
- Alternate sampler: `emcee`

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
| successful bins | 40 | 38 | 41 bins modeled |
| successful bins excluding cluster | 38 | 36 | -- |
| trailing extent [deg] | 20.00 | 20.00 | 16.0 |
| leading extent [deg] | 10.00 | 10.00 | 8.0 |
| integrated stars, |phi1| < 8 | 2912 | 2821 | 3000 |
| trailing / leading, |phi1| < 8 | 1.63 | 1.78 | ~1.0 |
| trailing / leading, |phi1| < 5 | 1.84 | 1.75 | 1.50 |
| near-cluster width [deg] | 0.123 | 0.123 | 0.150 |
| leading max width in [5, 8] [deg] | 0.293 @ 7.00 | 0.346 @ 7.00 | 0.400 @ 7.0 |
| trailing max width in [-15, -5] [deg] | 0.486 @ -10.25 | 0.556 @ -10.25 | trailing should stay relatively thin |

## Notes

- The integrated star counts above use the **linear density profile times the phi1 bin spacing**.
  This is the closest like-for-like quantity to Bonaca's integrated stream-star count.
- Cluster bins are excluded from the like-for-like totals and from the arm polynomial fits.
- Width is usually the least stable profile; use the MAP run as the baseline morphology
  and use the emcee run primarily as a posterior sanity check.

## Wide / suspicious baseline bins

| phi1_center | sigma_map [deg] | track_resid_map [deg] |
|---:|---:|---:|
| -13.25 | 0.418 | 0.053 |
| -10.25 | 0.486 | -0.069 |
