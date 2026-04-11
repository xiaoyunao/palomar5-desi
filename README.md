# palomar5-desi

Rebuild of the Palomar 5 DESI stream-detection workflow, starting from the raw DESI DR10 catalog and theoretical isochrone inputs, then producing photometric and spatial membership probabilities for the real stream.

## Scope

- Pre-select Pal 5 field sources from the raw DESI catalog
- Apply extinction and residual photometric corrections
- Build CMD and spatial membership models
- Produce final probability-enriched catalogs and stream-detection maps
- Prepare downstream inputs for track extraction and dynamical modeling

## Working layout

- `scripts/`: executable Python pipeline steps
- `README.md`: stable project overview
- `WORKLOG.md`: reverse-chronological session log
- `PLAN.md`: current objective and execution plan

## Data locations

Primary remote workspace:

- NERSC scratch: `/pscratch/sd/y/yunao`
- Existing reference directories:
  - `/pscratch/sd/y/yunao/palomar5`
  - `/pscratch/sd/y/yunao/Palomar_5_new`

Expected main raw input:

- `/pscratch/sd/y/yunao/palomar5/desi-dr10-palomar5-cat.fits`

## Execution model

- Local machine: code editing, validation, git history
- NERSC login node: light inspection, small tests, job submission
- NERSC compute allocation: large FITS chunk processing and full probability runs

## Immediate objective

Replace the notebook-based real-stream detection workflow with a script-based, reproducible pipeline that is robust to non-uniform background depth around Pal 5.
