# PAL5 step4c plotting patch v3

This patch only fixes the **diagnostic plot style** for `qc_step4c_segment_cmds`.
It does **not** change any science result, member selection, DM(track), or morphology model.

## What this patch does

It re-makes the segment CMD panel in the **original step4c excess-Hess style**:

- on-stream minus scaled off-stream Hess
- same orange isochrone overlay
- no contours
- no alternate full-CMD plotting style

The only change is that the Hess is now computed from the **full z-locus parent display sample**, not from the `20 < g0 < 23` strict-mag-limited sample.
So stars in the rest of the displayed CMD range (default `16 < g0 < 24`) now truly appear in the figure.

## Inputs expected in the project root

- `final_g25_preproc.fits`
- `step2_outputs/pal5_step2_summary.json`
- `pal5.dat`
- `step4c_outputs/pal5_step4c_dm_track.csv`
- `step3b_outputs_control/pal5_step3b_mu_prior.txt`

## Run

```bash
python pal5_step4c_plotting_patch_v3.py \
  --preproc final_g25_preproc.fits \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --iso pal5.dat \
  --dm-track-csv step4c_outputs/pal5_step4c_dm_track.csv \
  --mu-prior-file step3b_outputs_control/pal5_step3b_mu_prior.txt \
  --output step4c_outputs/qc_step4c_segment_cmds_stylefixed_fullhess.png
```

## Expected output

- `step4c_outputs/qc_step4c_segment_cmds_stylefixed_fullhess.png`

## Important

This patch is plotting-only.
Keep the current baseline layering unchanged:

- frozen formal baseline v1 = `step3b control + MAP`
- working baseline v2 = `step4c + step3b(control+MAP)`
