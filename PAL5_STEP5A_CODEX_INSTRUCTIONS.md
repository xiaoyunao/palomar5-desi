# PAL 5 step 5a: off-stream anchored empirical-background model

## Goal

Run a **safer** replacement for the original step 5 idea.

This version is designed to answer a very specific modeling question:

> If we keep the stream as a single Gaussian in `phi2`, and we learn the background only from **off-stream control-sideband stars**, can we improve the leading/trailing morphology without letting the stream/background decomposition blow up?

The previous step 5 failed because the model mixed:

1. a parent-count map treated like a multiplicative coverage amplitude,
2. an empirical background template that was too similar to the stream,
3. too much freedom for `sigma` in low-S/N outer bins.

## Important conceptual point

The file previously called `qc_step5_coverage_phi12_log.png` should **not** be interpreted as a pure coverage map.
It is built from counts in the parent catalog, so it mixes:

- the targeted deeper stripe near the stream,
- real sky-density gradients,
- selection effects.

So for step 5a:

- keep that map only as a **diagnostic parent-count proxy**,
- do **not** multiply it directly into the likelihood as an amplitude term,
- let the background be learned from the **control sidebands** and normalized in the **off-stream** region.

## Main modeling changes relative to step 5

### Keep

- `step2_outputs/pal5_step2_strict_members.fits` as the signal sample.
- `step3b_outputs_control/pal5_step3b_mu_prior.txt` as the ridge prior.
- a single-Gaussian stream model in each overlapping `phi1` bin.
- log-scale QC maps.

### Change

- Build the control sample from:
  - strict mag (`20 < g0 < 23`),
  - z-locus,
  - isochrone **sidebands**.
- In each `phi1` window, build the empirical background template using **only off-stream bins** in `phi2`.
- Fix the background normalization using the **signal off-stream counts** and the **control off-stream counts**.
- Fit only the stream term on top of that fixed background.
- Keep `sigma_max = 0.7 deg`.
- MAP only. Do **not** run emcee yet.

## Files to use

- `step2_outputs/pal5_step2_strict_members.fits`
- `final_g25_preproc.fits`
- `step2_outputs/pal5_step2_summary.json`
- `pal5.dat`
- `step3b_outputs_control/pal5_step3b_mu_prior.txt`
- `pal5_step3b_selection_aware_1d_model.py`
- `pal5_step5a_empirical_bg_offstream_model.py`

## Command to run

```bash
python pal5_step5a_empirical_bg_offstream_model.py \
  --signal step2_outputs/pal5_step2_strict_members.fits \
  --preproc final_g25_preproc.fits \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --iso pal5.dat \
  --mu-prior-file step3b_outputs_control/pal5_step3b_mu_prior.txt \
  --support-script pal5_step3b_selection_aware_1d_model.py \
  --outdir step5a_outputs_control_map
```

## After step 5a finishes

Run the Bonaca-style comparison against the current formal baseline (`step3b control + MAP`):

```bash
python pal5_step3c_bonaca_comparison.py \
  --profiles-map step5a_outputs_control_map/pal5_step5a_profiles.csv \
  --summary-map step5a_outputs_control_map/pal5_step5a_summary.json \
  --label-map "step5a empirical bg + MAP" \
  --profiles-alt step3b_outputs_control/pal5_step3b_profiles.csv \
  --summary-alt step3b_outputs_control/pal5_step3b_summary.json \
  --label-alt "step3b control + MAP" \
  --strict-fits step2_outputs/pal5_step2_strict_members.fits \
  --outdir step5a_step3c_vs_baseline
```

## What to look for

### Good signs

- `n_success_excluding_cluster` stays high (`>= 35`).
- `track_poly_trailing` and `track_poly_leading` are not null.
- `sigma` no longer piles up at the upper limit.
- `|phi1| < 8` integrated stars remain in the rough Bonaca-like range (`~2500–3200`).
- leading `[5,8]` width increases relative to the current baseline **without** trailing outer width exploding.

### Bad signs

- many bins hit `sigma_max = 0.7`.
- integrated stars jump to values much larger than `~3000`.
- track polynomials become null again.
- the background template still looks strongly peaked at `mu_prior` in many example bins.

## Deliver back

Bring back these files first:

- `step5a_outputs_control_map/pal5_step5a_summary.json`
- `step5a_outputs_control_map/qc_step5a_example_local_fits.png`
- `step5a_outputs_control_map/qc_step5a_template_examples.png`
- `step5a_outputs_control_map/qc_step5a_track.png`
- `step5a_outputs_control_map/qc_step5a_width.png`
- `step5a_step3c_vs_baseline/pal5_step3c_summary.json`
- `step5a_step3c_vs_baseline/pal5_step3c_report.md`

## If step 5a still fails

Do **not** invent a new model.
Instead report which of these happened:

- `sigma` still hits the cap,
- integrated counts still blow up,
- track becomes unstable,
- leading fan stays too narrow,
- trailing outer bins remain too wide.

That information is enough to decide the next change.
