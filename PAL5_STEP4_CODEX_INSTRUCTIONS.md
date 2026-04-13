# Pal 5 Step 4 — refined Bonaca-style distance-gradient selection

## Goal

Use the current **control + MAP** morphology baseline as the geometric backbone, then do the next Bonaca-like refinement step:

1. rebuild the **strict z-locus parent sample** from `final_g25_preproc.fits`,
2. estimate **coarse distance-modulus anchors** along the stream in 2-degree steps,
3. interpolate a smooth `DM(phi1)` track,
4. re-run the strict isochrone selection with this varying distance track,
5. then re-run step 3b / step 3c on the refined member catalog.

This is the closest automated analogue of Bonaca+2020 Section 2:
- fixed strict magnitude range,
- z-locus star/galaxy cleaning,
- coarse along-stream distance estimates,
- interpolation in `phi1` before final selection. fileciteturn0file14

## Files expected in the working directory

- `final_g25_preproc.fits`
- `pal5.dat`
- `step2_outputs/pal5_step2_summary.json`
- `step2_outputs/pal5_step2_strict_members.fits`
- `pal5_step3b_mu_prior_control.txt`
- `pal5_step3b_selection_aware_1d_model.py`
- `pal5_step3c_bonaca_comparison.py`
- `pal5_step4_refined_dm_selection.py`

## 1) Run step 4

```bash
python pal5_step4_refined_dm_selection.py \
  --preproc final_g25_preproc.fits \
  --iso pal5.dat \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --mu-prior pal5_step3b_mu_prior_control.txt \
  --strict-fits step2_outputs/pal5_step2_strict_members.fits \
  --outdir step4_outputs
```

## 2) Inspect the key QC products

Main files:

- `step4_outputs/pal5_step4_summary.json`
- `step4_outputs/pal5_step4_dm_anchors.csv`
- `step4_outputs/pal5_step4_dm_track.csv`
- `step4_outputs/plots_step4/qc_step4_dm_track.png`
- `step4_outputs/plots_step4/qc_step4_segment_cmds.png`
- `step4_outputs/plots_step4/qc_step4_local_compare.png`
- `step4_outputs/plots_step4/qc_step4_selected_density_phi12.png`

### What “good” looks like

The following should hold qualitatively:

- `DM(phi1)` is **monotonic-ish**: trailing arm more distant than the cluster, leading arm closer than the cluster.
- The anchor curve should not zig-zag violently from one 2-degree bin to the next.
- The refined local density map should still show the same Pal 5 ridge, but the **leading arm should not get weaker** than in step 2.
- The refined member count should stay in the same broad ballpark as step 2, not collapse by a huge factor.

### If the QC is poor

Only then try **one** of the following, not several at once:

1. widen the on-stream stripe slightly:

```bash
--on-halfwidth 0.45
```

2. move the off-stream stripes a bit closer:

```bash
--off-inner 0.70 --off-outer 1.40
```

3. widen the DM scan only if the peak is clearly hitting the grid edge:

```bash
--dm-scan-half 0.50
```

Do **not** change the strict magnitude range, z-locus tolerance, or CMD width function here.

## 3) Re-run step 3b on the refined members

### MAP baseline

```bash
python pal5_step3b_selection_aware_1d_model.py \
  --signal step4_outputs/pal5_step4_refined_members.fits \
  --preproc final_g25_preproc.fits \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --iso pal5.dat \
  --mu-prior-file pal5_step3b_mu_prior_control.txt \
  --eta-mode control \
  --sampler map \
  --outdir step4_step3b_control_map
```

### emcee posterior check

Only run this if the MAP run looks healthy.

```bash
python pal5_step3b_selection_aware_1d_model.py \
  --signal step4_outputs/pal5_step4_refined_members.fits \
  --preproc final_g25_preproc.fits \
  --step2-summary step2_outputs/pal5_step2_summary.json \
  --iso pal5.dat \
  --mu-prior-file pal5_step3b_mu_prior_control.txt \
  --eta-mode control \
  --sampler emcee \
  --outdir step4_step3b_control_emcee
```

## 4) Re-run step 3c comparison on the refined sample

```bash
python pal5_step3c_bonaca_comparison.py \
  --profiles-map step4_step3b_control_map/pal5_step3b_profiles.csv \
  --summary-map step4_step3b_control_map/pal5_step3b_summary.json \
  --label-map "refined DM + control + MAP" \
  --profiles-alt step4_step3b_control_emcee/pal5_step3b_profiles.csv \
  --summary-alt step4_step3b_control_emcee/pal5_step3b_summary.json \
  --label-alt "refined DM + control + emcee" \
  --strict-fits step4_outputs/pal5_step4_refined_members.fits \
  --outdir step4_step3c_outputs
```

## 5) What to compare against the current baseline

The current accepted baseline is:

- `control + MAP`
- `|phi1| < 8` integrated stars ≈ **2872**
- near-cluster width ≈ **0.118 deg**
- leading max width in `[5, 8]` ≈ **0.287 deg @ phi1=7**
- trailing / leading within `|phi1| < 5` ≈ **1.75** fileciteturn0file0turn0file1

### The refined-distance run is an improvement if

- the leading/trailing asymmetry moves **toward** Bonaca’s reference values,
- the leading width in `[5, 8]` moves **upward** toward ~0.4 deg,
- the suspicious outer-trailing broad bin around `phi1 ~ -10` becomes less extreme,
- and the overall integrated stars within `|phi1| < 8` stays near the Bonaca-like value (~3000). fileciteturn0file14

## 6) What not to change at this stage

- Do not change the strict sample philosophy.
- Do not add probabilistic CMD membership yet.
- Do not introduce a 2D detached/fan component yet.
- Do not switch away from the `control` eta template yet.

The point of step 4 is only:

> **same strict Bonaca-style morphology pipeline, but with a better along-stream distance track feeding the CMD selection**.
