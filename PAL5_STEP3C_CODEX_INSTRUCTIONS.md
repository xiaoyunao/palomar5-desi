# Pal 5 step 3c: Bonaca-style comparison / figure-making

## Goal

This step does **not** re-fit the stream. It takes the already generated step 3b outputs and turns them into:

1. a clean Bonaca-style profile figure,
2. a local `phi1-phi2` density map with the fitted track overplotted,
3. a baseline-vs-alternate comparison figure,
4. a trailing-vs-leading asymmetry figure,
5. machine-readable and human-readable comparison summaries.

The intended interpretation is:

- **control + MAP** = current formal morphology baseline,
- **control + emcee** = posterior / uncertainty sanity check,
- **control_times_depth** = diagnostic only, **not** the preferred baseline.

---

## Files expected in the project root

At minimum:

- `pal5_step3c_bonaca_comparison.py`
- `pal5_step3b_profiles_control.csv`
- `pal5_step3b_summary_control.json`

Recommended:

- `pal5_step3b_profiles_emcee.csv`
- `pal5_step3b_summary_emcee.json`
- `step2_outputs/pal5_step2_strict_members.fits`

---

## Standard run

Run this first:

```bash
python pal5_step3c_bonaca_comparison.py \
  --profiles-map pal5_step3b_profiles_control.csv \
  --summary-map pal5_step3b_summary_control.json \
  --profiles-alt pal5_step3b_profiles_emcee.csv \
  --summary-alt pal5_step3b_summary_emcee.json \
  --strict-fits step2_outputs/pal5_step2_strict_members.fits \
  --outdir step3c_outputs
```

If the strict-member FITS file is not available locally yet, skip the local map remake:

```bash
python pal5_step3c_bonaca_comparison.py \
  --profiles-map pal5_step3b_profiles_control.csv \
  --summary-map pal5_step3b_summary_control.json \
  --profiles-alt pal5_step3b_profiles_emcee.csv \
  --summary-alt pal5_step3b_summary_emcee.json \
  --no-map \
  --outdir step3c_outputs
```

If you want a baseline-only run:

```bash
python pal5_step3c_bonaca_comparison.py \
  --profiles-map pal5_step3b_profiles_control.csv \
  --summary-map pal5_step3b_summary_control.json \
  --no-alt \
  --strict-fits step2_outputs/pal5_step2_strict_members.fits \
  --outdir step3c_outputs_baseline_only
```

---

## What this script computes

### 1. Like-for-like integrated star counts

For Bonaca-style comparison, the relevant quantity is:

- `linear_density * phi1_step`, summed over bins,
- excluding cluster bins,
- with comparison windows such as `|phi1| < 8 deg` and `|phi1| < 5 deg`.

Do **not** use the raw `n_stream` window counts as the main Bonaca-like total, because the bins overlap.

### 2. Current geometry metrics

The script computes:

- trailing extent,
- leading extent,
- trailing/leading asymmetry within `|phi1| < 5` and `|phi1| < 8`,
- near-cluster width,
- leading-arm maximum width in `[5, 8] deg`,
- trailing-arm maximum width in `[-15, -5] deg`.

### 3. Figure products

Expected outputs in `step3c_outputs/`:

- `fig_step3c_bonaca_profiles.png`
- `fig_step3c_local_map.png` (if FITS file supplied)
- `fig_step3c_baseline_vs_alternate.png` (if alternate run supplied)
- `fig_step3c_asymmetry.png`
- `pal5_step3c_profile_table.csv`
- `pal5_step3c_metrics.csv`
- `pal5_step3c_summary.json`
- `pal5_step3c_report.md`

---

## What to inspect after the run

### First priority

Open:

- `fig_step3c_bonaca_profiles.png`
- `pal5_step3c_report.md`

Check whether the following are true:

1. the track is continuous and visually sensible,
2. the baseline total within `|phi1| < 8` is close to Bonaca's `~3000 ± 100`,
3. the trailing/leading asymmetry within `5 deg` is of order `~1.5`,
4. the near-cluster width is of order `~0.15 deg`,
5. the leading width increases toward `phi1 ~ 7 deg`,
6. the outer trailing width is still suspiciously too large.

### Second priority

If the alternate run is present, compare:

- `fig_step3c_baseline_vs_alternate.png`

Interpretation rule:

- if the emcee run gives similar track and density but much wider outer-arm widths,
  then use **MAP as the formal baseline** and use emcee only as the uncertainty / posterior diagnostic.

---

## Current project decision logic

At this stage, the expected decision is:

- keep **control + MAP** as the main baseline,
- keep **control + emcee** as a posterior sanity-check product,
- do **not** adopt `control_times_depth` as the main result.

Only change this if the step 3c report demonstrates otherwise.

---

## What to send back for the next discussion

Bring back these files:

- `step3c_outputs/pal5_step3c_report.md`
- `step3c_outputs/pal5_step3c_summary.json`
- `step3c_outputs/pal5_step3c_metrics.csv`
- `step3c_outputs/fig_step3c_bonaca_profiles.png`
- `step3c_outputs/fig_step3c_asymmetry.png`
- `step3c_outputs/fig_step3c_baseline_vs_alternate.png` (if produced)
- `step3c_outputs/fig_step3c_local_map.png` (if produced)

---

## Important constraints

- Do **not** re-fit step 3b inside this step.
- Do **not** switch back to the depth-multiplied template as the default baseline.
- Do **not** interpret individual density dips as final physical gaps yet.
- Do **not** treat raw `n_stream` window counts as the main Bonaca-like integrated star count.

This step is purely for **comparison, bookkeeping, and figure standardization**.
