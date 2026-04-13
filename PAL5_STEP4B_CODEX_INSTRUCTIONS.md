# Pal 5 step 4b — MSTO-weighted refined DM(phi1) selection

This step is a **targeted refinement** of step 4, not a redesign of the pipeline.
It keeps the same overall Bonaca-style idea:

1. rebuild the strict `z`-locus parent sample from `final_g25_preproc.fits`
2. estimate coarse distance anchors in 2-degree steps along `phi1`
3. interpolate a smooth `DM(phi1)` track
4. rerun the strict isochrone selection with that varying distance
5. then rerun the step3b / step3c morphology code on the refined members

## Why step 4b exists

Step 4 showed that coarse refined `DM(phi1)` can help, but some anchors were clearly being pulled by a **blue residual contaminant sequence** around roughly `(g-r)_0 ~ 0.2, g_0 ~ 22–23`.

Step 4b fixes that by changing **only the anchor-scoring stage**:

- anchor score uses the **MSTO / upper-MS** region only
- a blue residual sequence is explicitly vetoed/downweighted
- raw 2-degree anchors are smoothed more robustly before interpolation
- **QC CMD panels still show the full `16 < g < 24` range** for visual inspection

## Files expected in the project root

- `final_g25_preproc.fits`
- `pal5.dat`
- `step2_outputs/pal5_step2_summary.json`
- `step2_outputs/pal5_step2_strict_members.fits`
- `pal5_step3b_mu_prior_control.txt`
- `pal5_step4_refined_dm_selection.py`  
  (required because step4b imports helper utilities from it)
- `pal5_step4b_msto_dm_selection.py`

## Main command

```bash
python pal5_step4b_msto_dm_selection.py \
  --preproc final_g25_preproc.fits \
  --iso pal5.dat \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --strict-fits step2_outputs/pal5_step2_strict_members.fits \
  --mu-prior pal5_step3b_mu_prior_control.txt \
  --outdir step4b_outputs
```

## Main outputs

Inside `step4b_outputs/`:

- `pal5_step4b_refined_members.fits`
- `pal5_step4b_dm_anchors.csv`
- `pal5_step4b_dm_track.csv`
- `pal5_step4b_summary.json`
- `pal5_step4b_cutflow.txt`
- `pal5_step4b_report.md`
- `plots_step4b/qc_step4b_dm_track.png`
- `plots_step4b/qc_step4b_segment_cmds.png`
- `plots_step4b/qc_step4b_local_compare.png`
- `plots_step4b/qc_step4b_selected_density_phi12.png`
- `plots_step4b/qc_step4b_selected_density_radec.png`

## Important interpretation notes

### 1. The segment CMD plot is intentionally broader than the score window

The figure `qc_step4b_segment_cmds.png` is drawn over **`16 < g0 < 24`** on purpose.
This is only for visual inspection.

The **actual anchor score** is narrower:

- magnitude emphasis: roughly `19.8 < g0 < 21.7`
- model-color gate: only the MSTO / upper-MS part of the isochrone
- blue residual veto: approximately `(g-r)_0 < 0.15` and `g0 > 21.5`

Do **not** interpret the full plotted range as the part actually used for fitting the anchor.

### 2. Success is not enough; sanity-check the anchor shape

Even if all anchors succeed numerically, inspect:

- whether trailing is still systematically farther than leading
- whether there are obviously nonphysical local jumps in raw anchors
- whether the interpolated `DM(phi1)` is smoother and more Bonaca-like than step 4

### 3. Keep the old step 4 products for comparison

Do not overwrite or delete `step4_outputs/`.
The point of step 4b is to compare:

- old coarse-anchor refined DM
- MSTO-weighted refined DM
- original step2 two-arm baseline

## After step 4b finishes: rerun morphology

### step 3b on refined members — MAP baseline

```bash
python pal5_step3b_selection_aware_1d_model.py \
  --signal step4b_outputs/pal5_step4b_refined_members.fits \
  --preproc final_g25_preproc.fits \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --iso pal5.dat \
  --mu-prior pal5_step3b_mu_prior_control.txt \
  --eta-mode control \
  --sampler map \
  --outdir step4b_step3b_outputs_control
```

### step 3b on refined members — emcee posterior check

```bash
python pal5_step3b_selection_aware_1d_model.py \
  --signal step4b_outputs/pal5_step4b_refined_members.fits \
  --preproc final_g25_preproc.fits \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --iso pal5.dat \
  --mu-prior pal5_step3b_mu_prior_control.txt \
  --eta-mode control \
  --sampler emcee \
  --outdir step4b_step3b_outputs_control_emcee
```

### step 3c comparison on refined members

```bash
python pal5_step3c_bonaca_comparison.py \
  --profiles-map step4b_step3b_outputs_control/pal5_step3b_profiles.csv \
  --summary-map step4b_step3b_outputs_control/pal5_step3b_summary.json \
  --label-map "step4b MSTO DM + control + MAP" \
  --profiles-alt step4b_step3b_outputs_control_emcee/pal5_step3b_profiles.csv \
  --summary-alt step4b_step3b_outputs_control_emcee/pal5_step3b_summary.json \
  --label-alt "step4b MSTO DM + control + emcee" \
  --strict-fits step4b_outputs/pal5_step4b_refined_members.fits \
  --outdir step4b_step3c_outputs
```

## What to report back

Bring back these files / figures:

- `step4b_outputs/pal5_step4b_summary.json`
- `step4b_outputs/pal5_step4b_report.md`
- `step4b_outputs/plots_step4b/qc_step4b_dm_track.png`
- `step4b_outputs/plots_step4b/qc_step4b_segment_cmds.png`
- `step4b_step3b_outputs_control/pal5_step3b_summary.json`
- `step4b_step3c_outputs/pal5_step3c_summary.json`
- the updated Bonaca-style profile figure from the refined-member rerun

## What would count as a real improvement

Relative to the current formal baseline, step 4b is promising if it does at least one of these **without breaking the rest**:

- makes the refined `DM(phi1)` trend smoother and more trailing-far / leading-near
- increases `|phi1| < 8` integrated stars toward Bonaca's `~3000 ± 100`
- reduces the suspicious outer trailing width inflation
- increases the leading `[5,8]` fan width toward Bonaca's `~0.4 deg`
- lowers the trailing/leading asymmetry within `|phi1| < 5`

If step 4b only changes integrated counts slightly but still leaves the leading fan nearly unchanged, that is still useful information: it means the next bottleneck is probably not the coarse distance track.
