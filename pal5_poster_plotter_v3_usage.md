# `pal5_poster_plotter_v3.py` usage

This version now matches the current poster choices that were iterated in the Pal 5 runtime directory.

## Current figure definitions

### Figure 1

- Uses the step4c distance-modulus spline table plus the step4c combined anchor table.
- Also reads the step2 summary JSON to recover the two piecewise DM levels.
- Draws:
  - `piecewise trailing DM`
  - `piecewise leading DM`
  - combined `DM(phi1)`
  - MSTO anchors
  - RRL stream priors
  - RRL cluster anchor
- Uses the current poster y-range:
  - `16.0` to `17.5`

### Figure 2

- Uses the step4c refined member catalog as the observed background.
- Uses the step4c-step3b profile table as the observed track table.
- Draws the current poster version:
  - log density background
  - explicit legend entry for `observed stream background`
  - trailing and leading quadratics with different colors
  - trailing and leading fit bins in matching colors
  - cluster bins in orange
- Uses the current hand-tuned endpoint tweaks before fitting:
  - leftmost fit point: `+1.2` in `phi2`
  - second-rightmost fit point: `-0.1`
  - rightmost fit point: `-0.2`

### Figure 3

- Uses the mock-stream particle table only, not the best-fit mock-track overlay.
- Rotates the full mock particle cloud by `2 deg` counterclockwise about `(0, 0)` before windowing.
- Draws:
  - explicit legend entry for `mock stream particles`
  - observed trailing and leading quadratics
  - observed trailing and leading fit bins
  - observed cluster bins
- Uses the same endpoint-corrected observed track table as Figure 2.

### Figure 4

- Keeps the standardized `q`-mass comparison panel.
- Expects a literature table that already uses the common `M(<20 kpc)` definition.

## Required inputs

- `--members`
  Step4c refined members, e.g. `step4c_outputs/pal5_step4c_rrlprior_members.fits`
- `--obs-track`
  Step4c-step3b profiles table, e.g. `step4c_step3b_outputs_control/pal5_step3b_profiles.csv`
- `--mock-stream`
  Mock-stream particle table with RV column
- `--dm-table`
  Step4c DM track CSV, e.g. `step4c_outputs/pal5_step4c_dm_track.csv`
- `--fig1-anchor-table`
  Step4c combined anchors CSV, e.g. `step4c_outputs/pal5_step4c_combined_anchors.csv`
- `--fig1-step2-summary`
  Step2 summary JSON, e.g. `step2_outputs/pal5_step2_summary.json`
- `--literature-csv`
  Standardized `q`-mass comparison table

## Example

```bash
python pal5_poster_plotter_v3.py \
  --members /Users/island/Desktop/Pal5/step4c_outputs/pal5_step4c_rrlprior_members.fits \
  --obs-track /Users/island/Desktop/Pal5/step4c_step3b_outputs_control/pal5_step3b_profiles.csv \
  --mock-stream /Users/island/Desktop/Pal5/mockfit_simulate_params_bestfit10000/best_fit_mock_stream_particles_with_rv.csv \
  --dm-table /Users/island/Desktop/Pal5/step4c_outputs/pal5_step4c_dm_track.csv \
  --fig1-anchor-table /Users/island/Desktop/Pal5/step4c_outputs/pal5_step4c_combined_anchors.csv \
  --fig1-step2-summary /Users/island/Desktop/Pal5/step2_outputs/pal5_step2_summary.json \
  --literature-csv /Users/island/Desktop/Pal5/poster_figs_v3_actual/pal5_q_mass_m20kpc_filled.csv \
  --fig4-m-col m20_nfw_1e11
```

## Output files

The script writes four figure pairs into `--outdir`:

- `01_dm_track_only`
- `02_obs_density_track_local`
- `03_bestfit_mock_vs_obs_local`
- `04_q_mass_only`

## Notes

- This script now reflects the current poster-specific styling, not a neutral generic plotting preset.
- If the endpoint tweaks or color conventions change again, update this script rather than only editing the runtime PNGs.
