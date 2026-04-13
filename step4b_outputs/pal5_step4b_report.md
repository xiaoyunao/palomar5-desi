# Pal 5 step 4b MSTO-weighted refined distance-gradient selection

This run refines the step-4 coarse DM(phi1) idea by scoring anchors only in the distance-sensitive MSTO / upper-MS region, downweighting the blue residual sequence, and robustly smoothing the raw 2-degree anchors before interpolation.

## Summary

- z-parent sample after strict mag + z-locus: **2,525,069**
- successful DM anchors: **12 / 15**
- refined selected members: **451,842**
- DM(phi1=-15 deg): **16.825**
- DM(phi1=0 deg): **16.690**
- DM(phi1=+8 deg): **16.570**

## Anchor-scoring region used for DM fitting

- score window in magnitude: **19.8 < g0 < 21.7**
- model-color gate: **0.12 < (g-r)_iso < 0.58**
- blue-residual veto: **(g-r)_0 < 0.15** and **g0 > 21.5**

## QC files

- `plots_step4b/qc_step4b_dm_track.png`
- `plots_step4b/qc_step4b_segment_cmds.png`
- `plots_step4b/qc_step4b_local_compare.png`
- `plots_step4b/qc_step4b_selected_density_phi12.png`
- `plots_step4b/qc_step4b_selected_density_radec.png`

The segment CMD figure intentionally shows the **full 16 < g0 < 24** range, even though the anchor score itself only uses the narrower MSTO-focused region.
