# Pal 5 step 4c: RR Lyrae weak-prior distance-gradient refinement

This run combines two sources of information for the variable-DM member selection:

1. step4b-style MSTO-weighted photometric anchors; and
2. a small Price-Whelan+2019 RR Lyrae subset used as a weak prior on the *shape* of DM(phi1).

## Summary

- z-parent sample after strict mag + z-locus: **2,723,450**
- photometric anchors used: **15**
- RRL stars in curated subset: **27**
- stream RRL weak priors used: **16**
- RRL cluster DM (weighted median): **16.580 mag**
- zero-point shift applied to RRL DMs: **+0.145 mag**
- DM(phi1=-15 deg): **16.803**
- DM(phi1=0 deg): **16.718**
- DM(phi1=+8 deg): **16.540**
- refined selected members: **456,496**

## Files

- `pal5_step4c_summary.json`
- `pal5_step4c_dm_track.csv`
- `pal5_step4c_photometric_anchors.csv`
- `pal5_step4c_combined_anchors.csv`
- `qc_step4c_dm_track.png`
- `qc_step4c_rrl_phi12.png`
- `qc_step4c_segment_cmds.png`
- `qc_step4c_selected_density_phi12.png`
- `qc_step4c_selected_density_radec.png`
