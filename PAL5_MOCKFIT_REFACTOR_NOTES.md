# Pal 5 Mock-Track Fit Refactor

This repository version adopts the refactor described in
`/Users/island/Desktop/pal5_mockfit_notes.md`, with two project-specific
adjustments:

1. the script is now a runnable CLI instead of a notebook-style `example_main()`
2. it accepts the existing `step3b` profile table directly through column aliases

## Main changes relative to the old notebook

- Observed track extraction is removed from the mock-fit stage.
- The likelihood is track-to-track, not particle-to-curve.
- `dm_dt` is not part of the default free parameter set.
- The Pal 5 frame uses `gala.coordinates.Pal5PriceWhelan18`.
- `MockStreamGenerator` is driven through a `Hamiltonian` in the intended gala style.

## Accepted observed-track inputs

The script accepts either:

- a dedicated track table with `phi1`, `phi2`, `phi2_err`
- the existing step3b profile table with:
  - `phi1_center`
  - `mu`
  - `mu_err`
  - optional `sigma`, `sigma_err`

If a `success` column exists, failed rows are dropped by default.

## Current recommended first pass

Use the current photometric mainline track:

- `/Users/island/Desktop/Pal5/mainline_step4c_rerun_20260417/step4c_step3b_outputs_control/pal5_step3b_profiles.csv`

Run the fitter from the repository root and write outputs into the runtime tree.

## Intended products

- `observed_track_used.fits`
- `mcmc_samples.csv`
- `mcmc_summary.csv`
- `best_fit_model_track.fits`
- `best_fit_mock_stream_particles.fits`
- `best_fit_params.csv`
