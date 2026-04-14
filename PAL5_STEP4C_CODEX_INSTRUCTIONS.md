# Pal 5 step 4c: RR Lyrae weak-prior DM(track) refinement

## Goal

Build a new **working baseline** on top of the current `step4b + step3b(control+MAP)` direction by adding a sparse RR Lyrae distance prior from **Price-Whelan+2019**.

The logic is:

1. keep the `step4b` idea that the photometric DM anchors should be **MSTO-weighted**;
2. use the small RRL sample only as a **weak prior on the shape of DM(phi1)**, not as a hard absolute zero-point replacement;
3. re-run the variable-DM strict selection; and then
4. re-run `step3b control + MAP` on the updated members.

Do **not** touch the background model in this step.

---

## Files

New files added by GPT:

- `pal5_rrl_price_whelan_2019_subset.csv`
- `pal5_step4c_rrlprior_dm_selection.py`
- `PAL5_STEP4C_CODEX_INSTRUCTIONS.md`

Existing inputs expected in the repo root:

- `final_g25_preproc.fits`
- `pal5.dat`
- `step2_outputs/pal5_step2_summary.json`
- `step3b_outputs_control/pal5_step3b_mu_prior.txt`
- `pal5_step3b_selection_aware_1d_model.py`

---

## External prior used here

The curated CSV is copied from the abbreviated member table in **Price-Whelan et al. 2019**:

- 27 RRLs consistent with Pal 5 overall
- 10 in the cluster, 17 in the tails
- the table gives Gaia DR2 source IDs, periods, amplitudes, distances, membership probabilities, and separations from the cluster center

The script then queries **Gaia DR2 by source_id** (only 27 rows) to recover RA/Dec and transform them into Pal 5 coordinates.

Important: the script locks the **global zero-point** of the RRL distances to the current step2 cluster DM, and only uses the RRLs to improve the **relative shape** of `DM(phi1)`.

---

## One-time dependency

If `astroquery` is not available in your environment, install it first:

```bash
pip install astroquery
```

After the first successful run, the Gaia-enriched RRL file is cached as:

- `step4c_outputs/pal5_step4c_rrl_enriched.csv`

Later re-runs do not need internet as long as that cache exists.

---

## Step 1: run the RR Lyrae-prior DM refinement

From the repo root:

```bash
python pal5_step4c_rrlprior_dm_selection.py \
  --preproc final_g25_preproc.fits \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --iso pal5.dat \
  --mu-prior-file step3b_outputs_control/pal5_step3b_mu_prior.txt \
  --rrl-anchor-csv pal5_rrl_price_whelan_2019_subset.csv \
  --rrl-cache-csv step4c_outputs/pal5_step4c_rrl_enriched.csv \
  --output-dir step4c_outputs \
  --output-members step4c_outputs/pal5_step4c_rrlprior_members.fits \
  --allow-gaia-query
```

### Expected outputs

- `step4c_outputs/pal5_step4c_summary.json`
- `step4c_outputs/pal5_step4c_report.md`
- `step4c_outputs/pal5_step4c_dm_track.csv`
- `step4c_outputs/pal5_step4c_combined_anchors.csv`
- `step4c_outputs/qc_step4c_dm_track.png`
- `step4c_outputs/qc_step4c_rrl_phi12.png`
- `step4c_outputs/qc_step4c_segment_cmds.png`
- `step4c_outputs/qc_step4c_selected_density_phi12.png`
- `step4c_outputs/qc_step4c_selected_density_radec.png`
- `step4c_outputs/pal5_step4c_rrlprior_members.fits`

### Quick QC targets

After the run, inspect:

- whether `DM(phi1=-15)` is larger than `DM(phi1=+8)`
- whether the final `DM(phi1)` is smoother than the raw photometric anchors
- whether the local CMD panels still visibly align with the Pal 5 MSTO in both arms
- whether the selected density maps still show the leading fan candidate and the trailing tail continuously

Do **not** decide on adoption yet.

---

## Step 2: re-run step3b on the new members (control + MAP)

Use the **same command pattern you already used successfully** for `step4b` / `step3b`, but swap the signal file to:

- `step4c_outputs/pal5_step4c_rrlprior_members.fits`

and write to a new output directory such as:

- `step4c_step3b_outputs_control/`

Keep:

- `eta_mode = control`
- `sampler = map`

This rerun is the candidate **new working baseline**.

---

## Step 3: Bonaca-style comparison against the frozen formal baseline

Run the existing step3c comparison again, comparing:

- **baseline / new run**: `step4c_step3b_outputs_control`
- **alternate / frozen formal baseline**: `step3b_outputs_control`

Write the comparison to something like:

- `step4c_step3c_vs_step3b_baseline/`

The key numbers to compare are:

- `|phi1| < 8` integrated stars
- trailing / leading within `|phi1| < 5`
- near-cluster width
- leading max width in `[5, 8]`
- trailing max width in `[-15, -5]`

---

## Adoption rule for this step

Adopt the new `step4c + step3b(control+MAP)` run as the **working baseline v2** only if the following move in the right direction simultaneously relative to the frozen `step3b control + MAP` baseline:

1. `leading [5,8] max width` increases or at least does not degrade;
2. `trailing [-15,-5] max width` decreases or at least does not degrade;
3. `|phi1|<8` integrated stars stays in a Bonaca-like range (roughly around 3000, not wildly above 4000);
4. trailing/leading asymmetry does not blow up.

If the morphology improves but the counts become clearly unreasonable, keep this as a **diagnostic selection run**, not the adopted baseline.

---

## Notes for interpretation

- The RRL prior is intentionally weak. It is there to stabilize the **shape** of the distance-gradient fit.
- The script keeps the **step2 cluster zero-point** and uses the RRLs only to improve the along-stream relative distance trend.
- This step is meant to finish the **selection** side as far as practical before revisiting more ambitious background modeling.
