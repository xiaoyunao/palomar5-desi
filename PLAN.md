# PLAN

## Current Objective

Rebuild the Palomar 5 real-stream detection workflow from raw DESI data and theoretical isochrone inputs, using the newer `Palomar_5_new` approach as the baseline and improving robustness against non-uniform background and depth.

## Milestones

1. Convert the notebook workflow into reproducible Python scripts.
2. Preserve the validated pieces of the current pipeline:
   - chunked pre-selection from the raw DESI catalog
   - extinction and residual correction
   - isochrone-based photometric likelihood
3. Improve the stream-search stage:
   - handle spatially varying background
   - reduce bias from deeper imaging around the stream
   - emit interpretable diagnostics and maps
4. Produce final probability-enriched catalog products for downstream track fitting.
5. Prepare compute-node execution commands for the heavy NERSC run.

## Outstanding Issues

- Current notebooks are not fully reproducible or parameterized.
- The background CMD model in `3_cmd_prob.ipynb` uses a fixed off-stream strip and may be biased by depth variation.
- The current spatial prior hard-codes a rough stream track and can leak assumptions into the detection stage.
- Need a clean split between login-node-safe smoke tests and full production runs.

## Validation Criteria

- Each script passes Python syntax validation locally.
- The pipeline can run on a small sample without errors.
- Output columns and intermediate files are consistent with the intended downstream use.
- Heavy NERSC execution is packaged into explicit compute-node commands or job scripts.

## Next Recommended Steps

1. Initialize git and create the initial project commit.
2. Review literature on non-dynamical stream detection under non-uniform backgrounds.
3. Implement the script-based pre-selection, isochrone, and probability stages.
4. Validate on a small chunk locally or via a login-node-safe remote smoke test.
5. Prepare and launch the full NERSC production run from a compute allocation.
