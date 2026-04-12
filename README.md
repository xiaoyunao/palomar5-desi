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

- `pal5_preprocess_glt25.py`

## Expected outputs from a real run

- `pal5_preprocessed_glt25.fits`
- `pal5_preprocessed_summary.json`
- `diagnostics_pal5_preproc/`
- `reports_preproc/preprocess_cutflow.txt`
- `tmp_pal5_preproc/`

## Validation

For code changes, start with:

1. `python -m py_compile pal5_preprocess_glt25.py`

For a real preprocessing run, execute in the runtime directory or with explicit
output paths after confirming the input FITS path is available.
