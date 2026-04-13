# Pal 5 step 3b: selection-aware Bonaca-style 1D density modeling

## Goal

Run a **stable, selection-aware version of the Bonaca+2020 Section 3 morphology fit**.

The intrinsic model is unchanged:

- stream in each overlapping `phi1` window = **single Gaussian in `phi2`**
- background in each overlapping `phi1` window = **linearly varying function of `phi2`**

The upgrade in step 3b is at the **observation-model level**:

- build a matched **control sample** from the parent preprocessed catalog using
  the same strict magnitude and z-locus cuts as step 2,
- use **isochrone sidebands** instead of the signal ridge,
- turn this into a local multiplicative selection-response template `eta(phi2)`,
- optionally multiply by a **depth-based template** derived from `PSFDEPTH_G`
  and `PSFDEPTH_Z`.

This is motivated by the fact that the targeted deeper stripe is partially aligned
with the Pal 5 stream, so the previous step-3 fit could absorb survey structure
into the fitted stream width and density.

---

## Files expected in the project root

These should already exist:

- `final_g25_preproc.fits`
- `pal5.dat`
- `step2_outputs/pal5_step2_strict_members.fits`
- `step2_outputs/pal5_step2_summary.json`
- `pal5_step3b_selection_aware_1d_model.py`

Optional but recommended:

- `step3_outputs_hw15/pal5_step3_pass1_prior_track.txt`

If the `step3_outputs_hw15/` file does not exist, the script will fall back to
constructing a prior ridge from the strict-member density map itself.

---

## Recommended run order

### Run A: stable baseline (control template only)

This is the primary step-3b run.

```bash
cd ~/Desktop/Pal5
python pal5_step3b_selection_aware_1d_model.py \
  --signal step2_outputs/pal5_step2_strict_members.fits \
  --preproc final_g25_preproc.fits \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --iso pal5.dat \
  --mu-prior-file step3_outputs_hw15/pal5_step3_pass1_prior_track.txt \
  --eta-mode control \
  --sampler map \
  --outdir step3b_outputs_control
```

### Run B: compare against explicit depth modulation

Only after Run A finishes, run this comparison version:

```bash
cd ~/Desktop/Pal5
python pal5_step3b_selection_aware_1d_model.py \
  --signal step2_outputs/pal5_step2_strict_members.fits \
  --preproc final_g25_preproc.fits \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --iso pal5.dat \
  --mu-prior-file step3_outputs_hw15/pal5_step3_pass1_prior_track.txt \
  --eta-mode control_times_depth \
  --sampler map \
  --outdir step3b_outputs_control_depth
```

### Run C: posterior sampling refinement (only if `emcee` is available)

After deciding whether `control` or `control_times_depth` behaves better,
rerun the preferred mode with posterior sampling:

```bash
cd ~/Desktop/Pal5
python pal5_step3b_selection_aware_1d_model.py \
  --signal step2_outputs/pal5_step2_strict_members.fits \
  --preproc final_g25_preproc.fits \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --iso pal5.dat \
  --mu-prior-file step3_outputs_hw15/pal5_step3_pass1_prior_track.txt \
  --eta-mode control \
  --sampler emcee \
  --nwalkers 48 \
  --burn 256 \
  --steps 512 \
  --outdir step3b_outputs_control_emcee
```

If `emcee` is not installed, the script automatically falls back to a
MAP + Laplace approximation when `--sampler auto` is used. For reproducibility,
prefer `--sampler map` unless `emcee` is definitely available.

---

## What the script does

1. Reads the strict member catalog from step 2.
2. Reads `step2_summary.json` to recover the step-2 nuisance alignment and strict
   selection-box configuration.
3. Reconstructs the aligned leading/trailing isochrone ridges from `pal5.dat`.
4. Scans the parent preprocessed catalog and builds:
   - a **control sideband sample**:
     `strict_mag + z_locus + isochrone sidebands`
   - a **z-locus parent sample**:
     `strict_mag + z_locus`
5. Uses the control sidebands to build `eta_control(phi2)` in each overlapping
   `phi1` window.
6. Optionally uses `PSFDEPTH_G / PSFDEPTH_Z` to build `eta_depth(phi2)`.
7. Fits the observed strict-member `phi2` distribution in each overlapping
   `phi1` window with

   `eta(phi2) × [ f * Gaussian(mu, sigma) + (1-f) * linear_background ]`

   using transformed parameters:
   - `u_f = logit(f)`
   - `log_sigma = ln(sigma)`
   - bounded `mu`
   - bounded background tilt
8. Uses the external prior ridge from the exploratory step-3 run if available.
9. Flags cluster bins near `phi1 = 0` and excludes them from arm-polynomial fits
   and the main morphology summaries.
10. Saves profile tables and QC plots.

---

## Output files to inspect

For each output directory (for example `step3b_outputs_control/`), inspect:

Main tables:

- `pal5_step3b_profiles.fits`
- `pal5_step3b_profiles.csv`
- `pal5_step3b_summary.json`
- `pal5_step3b_mu_prior.txt`
- `pal5_step3b_control_sidebands.fits`

Main QC plots:

- `qc_step3b_density_phi12.png`
- `qc_step3b_density_phi12_local.png`
- `qc_step3b_control_density_phi12.png`
- `qc_step3b_track.png`
- `qc_step3b_track_resid.png`
- `qc_step3b_width.png`
- `qc_step3b_linear_density.png`
- `qc_step3b_stream_fraction.png`
- `qc_step3b_example_local_fits.png`
- `qc_step3b_eta_examples.png`

---

## What to check first

### 1. `qc_step3b_control_density_phi12.png`
The control sample should clearly show the same survey / depth imprint that was
polluting step 3.

### 2. `qc_step3b_eta_examples.png`
Check whether `eta_total(phi2)` tracks the local survey structure sensibly.
It should be smooth and strictly positive, not wildly oscillatory.

### 3. `qc_step3b_density_phi12_local.png`
Check whether the fitted ridge now follows the visual stream without obviously
locking onto the deeper stripe.

### 4. `qc_step3b_width.png`
This is the most important science sanity check.
A better run should move toward the Bonaca-like behavior:

- cluster vicinity: both arms relatively narrow,
- trailing arm: mostly thin,
- leading arm: broadening toward positive `phi1`.

### 5. `qc_step3b_linear_density.png`
This should stop being dominated by obvious fit failures or by the cluster spike.
Remember that cluster bins are flagged separately.

### 6. `qc_step3b_example_local_fits.png`
The fit should resemble the local histograms in the windows that visually show
stream signal, and `eta` should account for background modulation rather than
forcing the Gaussian to absorb it.

---

## What counts as a successful step-3b run

A run is acceptable if most of the following are true:

- the number of successful bins is **clearly higher** than the exploratory step-3 run,
- the local fitted ridge follows the visible stream,
- the width profile no longer prefers artificially broad trailing bins,
- the cluster bins are not used to drive the arm polynomial fits,
- the linear-density profile away from the progenitor is interpretable,
- the difference between `control` and `control_times_depth` is physically sensible.

The previous exploratory run had only about 15 successful bins out of 41 and
showed unphysical trailing-width behavior. Step 3b is meant to improve that.

---

## What Codex is allowed to tweak if Run A is still unstable

Only tweak the following, in this order.

### A. fitting half-width around the prior ridge
Default:

- `--fit-halfwidth 1.5`

If needed, try:

- `--fit-halfwidth 1.35`
- `--fit-halfwidth 1.25`

Do not widen it before trying narrower values.

### B. minimum signal stars per bin
Default:

- `--min-signal 60`

If too many bins are skipped purely for lack of stars, try:

- `--min-signal 45`

### C. eta mode
Compare only:

- `--eta-mode control`
- `--eta-mode control_times_depth`

Do **not** switch to `depth`-only unless both of the above look clearly worse.

### D. posterior sampling
Only after the MAP run looks physically sensible should Codex turn on:

- `--sampler emcee`

---

## What Codex must NOT change yet

Do **not** do any of the following in step 3b:

- do not change the strict member-selection definition from step 2,
- do not switch to per-star CMD probabilities,
- do not add a second Gaussian stream component,
- do not add fan- or spur-specific morphology terms,
- do not replace the linear intrinsic background with a spline or high-order polynomial,
- do not change the step-2 z-locus tolerance or isochrone-box widths,
- do not start the deep-sample analysis yet.

The point of step 3b is to keep the **Bonaca intrinsic morphology model** while
fixing the **selection / survey-imprint problem**.

---

## What to report back after running

Please report back with:

1. `pal5_step3b_summary.json` from the preferred run,
2. a short comparison between:
   - `control`
   - `control_times_depth`
3. the following plots from the preferred run:
   - `qc_step3b_control_density_phi12.png`
   - `qc_step3b_eta_examples.png`
   - `qc_step3b_density_phi12_local.png`
   - `qc_step3b_track.png`
   - `qc_step3b_width.png`
   - `qc_step3b_linear_density.png`
   - `qc_step3b_example_local_fits.png`

Also summarize:

- number of successful bins,
- whether the leading arm broadens more than the trailing arm,
- whether the cluster-centered spike is now isolated rather than biasing the full profile,
- whether the background stripe is still being absorbed into the stream model.

---

## Interpretation reminder

This step still belongs to the **strict baseline** branch.

It is intended to answer:

- can we make the Bonaca-style 1D morphology fit numerically stable?
- can we stop the targeted deeper stripe from masquerading as stream density / width?
- do we recover a physically sensible track-width-density baseline before moving on?

It is **not yet** the final completeness-corrected deep-sample analysis for the
southern leading tail extension.
