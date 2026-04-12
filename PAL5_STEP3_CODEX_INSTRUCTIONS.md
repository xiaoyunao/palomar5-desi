# Pal 5 step 3: Bonaca-style 1D stream density modeling

## Goal

Run the **strict baseline spatial modeling** on the step-2 strict member sample.

This step is intended to reproduce the core measurement in Bonaca+2020:
- track `mu(phi1)`
- width `sigma(phi1)`
- linear density along the stream

using a **single Gaussian stream component + local linear background** in overlapping `phi1` bins.

This is **not yet** the final deep-sample or completeness-aware analysis. It is the baseline, Bonaca-style spatial model on the strict selected sample.

---

## Inputs expected in the project root

The following files should already exist:

- `final_g25_preproc.fits`
- `pal5.dat`
- `step2_outputs/pal5_step2_strict_members.fits`
- `pal5_step3_bonaca_1d_model.py`

The script actually uses only:
- `step2_outputs/pal5_step2_strict_members.fits`

---

## Run command

```bash
cd ~/Desktop/Pal5
python pal5_step3_bonaca_1d_model.py \
  --input step2_outputs/pal5_step2_strict_members.fits \
  --outdir step3_outputs
```

---

## What the script does

1. Reads the strict member sample from step 2.
2. Uses `phi1` bin centers from `-20` to `+10` deg with spacing `0.75` deg.
3. Uses overlapping `phi1` windows of width `1.5 * 0.75 = 1.125` deg.
4. In each bin, fits the local `phi2` distribution with:
   - one Gaussian stream component
   - one normalized linear background component
5. Runs a two-pass procedure:
   - **pass 1**: independent local fits using a histogram-mode initialization
   - **pass 2**: refit using a smoothed pass-1 ridge as the `mu` prior center
6. Saves the fitted profiles and a set of QC plots.

---

## Output files to inspect

Main outputs:

- `step3_outputs/pal5_step3_profiles.fits`
- `step3_outputs/pal5_step3_profiles.csv`
- `step3_outputs/pal5_step3_summary.json`

QC plots:

- `step3_outputs/qc_step3_density_phi12.png`
- `step3_outputs/qc_step3_density_phi12_local.png`
- `step3_outputs/qc_step3_density_radec.png`
- `step3_outputs/qc_step3_track.png`
- `step3_outputs/qc_step3_track_resid.png`
- `step3_outputs/qc_step3_width.png`
- `step3_outputs/qc_step3_linear_density.png`
- `step3_outputs/qc_step3_stream_fraction.png`
- `step3_outputs/qc_step3_example_local_fits.png`

Intermediate diagnostic:

- `step3_outputs/pal5_step3_pass1_prior_track.txt`

---

## What to check first

### 1. `qc_step3_density_phi12.png`
Check whether the fitted track overlays the visible stream.

### 2. `qc_step3_track.png`
Check whether the inferred track is continuous over the detected stream.

### 3. `qc_step3_width.png`
Check whether:
- the trailing arm stays relatively thin
- the leading arm broadens toward positive `phi1`

### 4. `qc_step3_linear_density.png`
Check whether obvious density dips appear near the expected locations.

### 5. `qc_step3_example_local_fits.png`
Check whether the local `phi2` histograms are fit sensibly by the Gaussian+linear mixture.

---

## What counts as a successful run

A run is acceptable if:

- most bins in the detected stream region have successful fits
- the fitted track follows the visible ridge in `phi1-phi2`
- the width is not wildly discontinuous bin-to-bin
- the leading arm shows broader inferred width than the trailing arm
- the density profile is not dominated by obvious fit failures

---

## What Codex is allowed to tweak if the fit looks bad

Only tweak these items, in this order:

### A. local `phi2` fitting window
Default:
- pass 1 global range: `[-2.5, +2.5]`
- pass 2 local half-width: `1.75 deg`

If the fit is too background-dominated or obviously misses the stream, Codex may try:
- `--pass2-phi2-halfwidth 1.5`
- `--pass2-phi2-halfwidth 2.0`

### B. minimum stars per bin
Default:
- `--min-stars 60`

If too many bins fail, Codex may try:
- `--min-stars 40`

### C. bin spacing
Default:
- `--phi1-step 0.75`

If the profile is too noisy, Codex may try:
- `--phi1-step 1.0`

but only after trying A and B.

---

## What Codex must NOT change yet

Do **not** do any of the following in this step:

- do not switch to a CMD probability model
- do not change the strict member sample definition
- do not add completeness weighting yet
- do not replace the linear background with the template background yet
- do not add spur/fan-specific extra components yet
- do not change the step-2 selection box or z-locus parameters

This step is meant to stay a **strict Bonaca-style baseline**.

---

## What to report back after running

Please report back with:

1. `step3_outputs/pal5_step3_summary.json`
2. a short text summary of:
   - number of bins fit successfully
   - approximate detected stream extent in `phi1`
   - whether the leading arm broadens relative to the trailing arm
   - whether obvious density dips appear
3. the following plots:
   - `qc_step3_density_phi12.png`
   - `qc_step3_track.png`
   - `qc_step3_width.png`
   - `qc_step3_linear_density.png`
   - `qc_step3_example_local_fits.png`

---

## Interpretation reminder

This step gives the **baseline spatial morphology** on the strict selected sample.

It is appropriate for:
- checking whether the Bonaca-style pipeline reproduces a sensible Pal 5 track/width/density measurement
- establishing a baseline before upgrading the background/completeness treatment

It is **not yet** the final result for the deeper southern leading tail analysis.
