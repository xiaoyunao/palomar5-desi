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
- The original background CMD model in `3_cmd_prob.ipynb` uses a fixed off-stream strip and may be biased by depth variation.
- The original spatial prior hard-codes a rough stream track and can leak assumptions into the detection stage.
- Need a clean split between login-node-safe smoke tests and full production runs.
- Need to inspect whether the current `P_MEM` thresholding is still too conservative on the full catalog.

## Validation Criteria

- Each script passes Python syntax validation locally.
- The pipeline can run on a small sample without errors.
- Output columns and intermediate files are consistent with the intended downstream use.
- Heavy NERSC execution is packaged into explicit compute-node commands or job scripts.

## Next Recommended Steps

1. Launch the full NERSC compute-node run from raw DESI input through `final_glt24_membership.fits`.
2. Inspect the full-catalog `P_ISO` and `P_MEM` distributions and the extracted stream track.
3. If needed, tune:
   - `logk_cmd`
   - `logk_mem`
   - sideband widths
   - `phi1`/depth background binning
4. Decide whether to reintroduce a Gaussian-plus-background track fit after the full run.
5. Once the front end is stable, connect its outputs to the downstream simulation and dynamical fitting stage.
