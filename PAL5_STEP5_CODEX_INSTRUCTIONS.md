# Pal 5 step 5: empirical background + effective area upgrade

This step keeps the **formal signal sample** fixed to the step-2 strict members and upgrades the **1D morphology model** in a different direction from step 4 / 4b.

## Scientific goal

The current picture after step 4 / 4b is:

- the step-3c `control + MAP` run remains the formal baseline
- refined DM helps somewhat, but it does **not** simultaneously fix
  - leading fan width
  - trailing / leading asymmetry
  - counts
- this points to **background / selection response / effective area** as the next likely bottleneck

So step 5 should test the following model change:

- keep the stream as a **single Gaussian in phi2**
- replace the old background treatment with a more explicit split into
  - an **empirical background shape** from the control sidebands
  - an **effective-area / coverage template** from the full preprocessed parent catalog

This is meant to answer:

1. Does leading-fan width increase once the background shape is fixed more empirically?
2. Does outer trailing width shrink further?
3. Does the trailing / leading ratio move toward Bonaca?
4. Do the local maps become visually cleaner in **log density**?

---

## Script

Use:

- `pal5_step5_empirical_bg_area_model.py`

This script depends on:

- `pal5_step3b_selection_aware_1d_model.py`

because it reuses the step-2 selection / isochrone helper logic.

---

## What step 5 actually fits

Within each overlapping `phi1` window, the observed `phi2` density is modeled as:

- stream term: `A(phi2) * Gaussian(mu, sigma)`
- background term: `A(phi2) * B_emp(phi2) * L(phi2)`

where:

- `A(phi2)` = effective-area / coverage template from the full preprocessed parent catalog
- `B_emp(phi2)` = empirical background shape from the control sidebands
- `L(phi2)` = weak linear residual freedom around the empirical background

Compared to step 3b:

- the control sample is no longer only an `eta` multiplier
- the background shape is now allowed to be **empirical and fixed by the control sample**
- the coverage / footprint response is handled separately

---

## Output conventions

Main outputs in the chosen outdir:

- `pal5_step5_profiles.fits`
- `pal5_step5_profiles.csv`
- `pal5_step5_summary.json`
- `pal5_step5_control_sidebands.fits`
- `pal5_step5_mu_prior.txt`

QC figures:

- `qc_step5_density_phi12_log.png`
- `qc_step5_density_phi12_local_log.png`
- `qc_step5_control_density_phi12_log.png`
- `qc_step5_coverage_phi12_log.png`
- `qc_step5_density_radec_log.png`
- `qc_step5_track.png`
- `qc_step5_track_resid.png`
- `qc_step5_width.png`
- `qc_step5_linear_density.png`
- `qc_step5_stream_fraction.png`
- `qc_step5_example_local_fits.png`
- `qc_step5_template_examples.png`

The density maps in this step are intentionally plotted in **log scale**.

---

## Run order

### 1. Syntax check

```bash
python -m py_compile pal5_step5_empirical_bg_area_model.py
```

### 2. Main MAP run

```bash
python pal5_step5_empirical_bg_area_model.py \
  --signal step2_outputs/pal5_step2_strict_members.fits \
  --preproc final_g25_preproc.fits \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --iso pal5.dat \
  --mu-prior-file step3b_outputs_control/pal5_step3b_mu_prior.txt \
  --support-script pal5_step3b_selection_aware_1d_model.py \
  --outdir step5_outputs_control_map \
  --sampler map
```

### 3. Optional emcee sanity-check run

Only do this if `emcee` is installed and the MAP run completed cleanly.

```bash
python pal5_step5_empirical_bg_area_model.py \
  --signal step2_outputs/pal5_step2_strict_members.fits \
  --preproc final_g25_preproc.fits \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --iso pal5.dat \
  --mu-prior-file step3b_outputs_control/pal5_step3b_mu_prior.txt \
  --support-script pal5_step3b_selection_aware_1d_model.py \
  --outdir step5_outputs_control_emcee \
  --sampler emcee
```

---

## Compare against Bonaca-style reference numbers

### A. Step-5 MAP versus Step-5 emcee

```bash
python pal5_step3c_bonaca_comparison.py \
  --profiles-map step5_outputs_control_map/pal5_step5_profiles.csv \
  --summary-map  step5_outputs_control_map/pal5_step5_summary.json \
  --label-map "step5 empirical bg + MAP" \
  --profiles-alt step5_outputs_control_emcee/pal5_step5_profiles.csv \
  --summary-alt  step5_outputs_control_emcee/pal5_step5_summary.json \
  --label-alt "step5 empirical bg + emcee" \
  --strict-fits step2_outputs/pal5_step2_strict_members.fits \
  --outdir step5_step3c_selfcheck
```

### B. Step-5 MAP versus the current formal baseline (`step3b control + MAP`)

```bash
python pal5_step3c_bonaca_comparison.py \
  --profiles-map step5_outputs_control_map/pal5_step5_profiles.csv \
  --summary-map  step5_outputs_control_map/pal5_step5_summary.json \
  --label-map "step5 empirical bg + MAP" \
  --profiles-alt step3b_outputs_control/pal5_step3b_profiles.csv \
  --summary-alt  step3b_outputs_control/pal5_step3b_summary.json \
  --label-alt "step3b control + MAP" \
  --strict-fits step2_outputs/pal5_step2_strict_members.fits \
  --outdir step5_step3c_vs_baseline
```

---

## What to inspect first

After the MAP run, inspect these in order:

1. `step5_outputs_control_map/pal5_step5_summary.json`
2. `step5_outputs_control_map/qc_step5_density_phi12_local_log.png`
3. `step5_outputs_control_map/qc_step5_template_examples.png`
4. `step5_outputs_control_map/qc_step5_example_local_fits.png`
5. `step5_step3c_vs_baseline/pal5_step3c_report.md`

---

## The most important quantitative checks

Please extract / report these numbers and compare them to the current baseline:

- successful bins
- successful bins excluding cluster
- `|phi1| < 8` integrated stars
- trailing / leading within `|phi1| < 5`
- near-cluster width
- leading max width in `[5, 8]`
- trailing max width in `[-15, -5]`

The decision rule is:

### step 5 is promising if

- leading width increases
- outer trailing width decreases or at least stops being pathological
- counts remain near the current Bonaca-like range
- the asymmetry does not get worse
- local log-density maps look cleaner

### step 5 is not worth adopting as baseline if

- leading width hardly changes
- trailing / leading gets even larger
- counts drift farther from the Bonaca-like comparison
- the empirical-background model becomes unstable in outer bins

---

## Important scope guard

Do **not** modify step 2, step 4, or step 4b while doing this.

For this round:

- keep the **signal sample** fixed to `step2_outputs/pal5_step2_strict_members.fits`
- keep the **mu prior** from the existing control baseline
- do not introduce a new distance model here
- do not add spur / detached components
- do not change the stream from single-Gaussian

This step is specifically about testing whether **empirical background + effective area** helps more than additional DM tweaking.
