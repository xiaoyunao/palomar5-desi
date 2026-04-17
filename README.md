# Palomar 5 Stream Workspace

This repository is the code, backup, and git-tracked workspace for rebuilding the
Palomar 5 stream detection and morphology pipeline.

The active runtime/data directory is:

- `/Users/island/Desktop/Pal5`

Use this repository for:

- code changes
- documentation and project memory
- git commits and pushes

Use `/Users/island/Desktop/Pal5` for:

- large runtime products
- local FITS outputs
- diagnostic figures from actual runs

## Current phase

Phase 0 preprocessing baseline:

- build a clean stellar catalog
- apply dereddening
- transform into the Pal 5 stream frame
- keep a single `g0 < 25` sample
- generate QC plots and cutflow reports

## Main script

- `pal5_preprocess_step1.py`
- `pal5_step2_member_selection.py`
- `pal5_step3b_selection_aware_1d_model.py`
- `pal5_step4c_rrlprior_dm_selection.py`
- `pal5_mock_track_fit_refactor.py`

## Current working baseline

The current photometric working baseline is:

- `/Users/island/Desktop/Pal5/mainline_step4c_rerun_20260417/step4c_outputs/pal5_step4c_rrlprior_members.fits`
- `/Users/island/Desktop/Pal5/mainline_step4c_rerun_20260417/step4c_step3b_outputs_control/pal5_step3b_profiles.csv`

For morphology and Bonaca-style comparisons, the practical track table already
exists in the step3b profile output:

- `phi1_center`
- `mu`
- `mu_err`
- optional `sigma`, `sigma_err`

`pal5_mock_track_fit_refactor.py` can read that table directly; it does not need
the older notebook-style observed-track re-extraction from a filtered catalog.

## Mock-Track Fit

The refactored mock-stream fitter expects a precomputed observed track and fits
the *model centroid track* against it.

Recommended first-pass run from this repository:

```bash
'/Users/island/opt/anaconda3/envs/astro/bin/python' pal5_mock_track_fit_refactor.py \
  --track /Users/island/Desktop/Pal5/mainline_step4c_rerun_20260417/step4c_step3b_outputs_control/pal5_step3b_profiles.csv \
  --outdir /Users/island/Desktop/Pal5/mockfit_mainline_step4c_trackonly \
  --nwalkers 48 \
  --burnin 200 \
  --steps 600
```

If you later want to add the width term, append `--use-width-term`.

The script auto-detects either:

- a clean track table with `phi1`, `phi2`, `phi2_err`
- or the existing step3b profile table with `phi1_center`, `mu`, `mu_err`

## Expected outputs from a real run

- `pal5_preprocessed_glt25.fits`
- `pal5_preprocessed_summary.json`
- `diagnostics_pal5_preproc/`
- `reports_preproc/preprocess_cutflow.txt`
- `tmp_pal5_preproc/`

## Validation

For code changes, start with:

1. `python -m py_compile pal5_preprocess_step1.py`
2. `python -m py_compile pal5_step3b_selection_aware_1d_model.py`
3. `python -m py_compile pal5_mock_track_fit_refactor.py`

For a real preprocessing run, execute in the runtime directory or with explicit
output paths after confirming the input FITS path is available.
