# Step 4c plotting patch + frozen baseline convention

This patch does **not** change the science model.

It only does two things:

1. Regenerates the step4c diagnostic plots so that the segment CMD panels show the **full CMD display** from the preprocessed parent catalog, rather than only the score-window-limited stars.
2. Freezes the current **step4c selection-upgraded baseline** as the default input selection for any later background-model experiments.

---

## What is frozen after this patch

Keep both of these baselines:

### Frozen formal baseline v1
Do **not** overwrite or delete:

- `step3b_outputs_control/`
- `step3c_outputs/` or the existing step3c comparison products built from step3b control + MAP

This remains the stable reference baseline.

### Working baseline v2
Use the following as the default **selection-upgraded** input for new background work:

- selected members: `step4c_outputs/pal5_step4c_rrlprior_members.fits`
- DM track: `step4c_outputs/pal5_step4c_dm_track.csv`
- step4c selection report: `step4c_outputs/pal5_step4c_report.md`
- step4c-vs-v1 comparison: `step4c_step3c_vs_step3b_baseline/`

When you run any future background-model experiment, the signal sample should come from the step4c-selected members above unless explicitly testing something else.

---

## What the plotting patch changes

The script:

- `pal5_step4c_plotting_patch.py`

will regenerate these plots:

- `step4c_outputs/qc_step4c_segment_cmds_fullcmd.png`
- `step4c_outputs/qc_step4c_selected_density_phi12_log.png`
- `step4c_outputs/qc_step4c_selected_density_radec_log.png`
- `step4c_outputs/qc_step4c_selected_density_phi12_local_log.png`
- `step4c_outputs/pal5_step4c_plotpatch_summary.json`

### CMD behavior after the patch

The segment CMD panels now:

- use the **preprocessed parent catalog** for the visual background,
- show the full `16 < g0 < 24` range,
- do **not** mask away stars outside `20 < g0 < 23`,
- overlay the strict-zparent excess as contours,
- overlay the step4c DM-track isochrone,
- highlight the actual MSTO score band as an orange band,
- mark the downweighted blue residual region.

This is intentionally a **display** change only. The step4c selection itself is untouched.

---

## Run order

From the runtime directory, for example:

```bash
cd ~/Desktop/Pal5
python -m py_compile pal5_step4c_plotting_patch.py
```

Then run:

```bash
python pal5_step4c_plotting_patch.py \
  --preproc final_g25_preproc.fits \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --iso pal5.dat \
  --dm-track-csv step4c_outputs/pal5_step4c_dm_track.csv \
  --selected-members step4c_outputs/pal5_step4c_rrlprior_members.fits \
  --mu-prior-file step3b_outputs_control/pal5_step3b_mu_prior.txt \
  --output-dir step4c_outputs
```

---

## What to check after running

### 1. Full CMD panel
Open:

- `step4c_outputs/qc_step4c_segment_cmds_fullcmd.png`

and confirm:

- stars outside `20 < g0 < 23` are really visible,
- the full `16 < g0 < 24` CMD is populated,
- the orange score band is only a highlighted region, not a display mask,
- the blue residual sequence is visible where expected.

### 2. Log density maps
Open:

- `step4c_outputs/qc_step4c_selected_density_phi12_log.png`
- `step4c_outputs/qc_step4c_selected_density_phi12_local_log.png`
- `step4c_outputs/qc_step4c_selected_density_radec_log.png`

These are only for visualization/QC. They should not change any scientific result.

### 3. Summary file
Open:

- `step4c_outputs/pal5_step4c_plotpatch_summary.json`

Make sure the file paths and counts look sensible.

---

## Important constraint for later work

Do **not** rewrite the step4c science selection as part of this patch.

The goal here is:

- keep `step4c + step3b(control+MAP)` as the current working baseline v2,
- improve the diagnostic plots,
- then use **that same selection** when testing new background models.

The next background-model code should therefore consume:

- `step4c_outputs/pal5_step4c_rrlprior_members.fits`

as the default signal sample.
