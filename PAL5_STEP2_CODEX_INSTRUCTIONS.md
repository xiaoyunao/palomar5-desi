# Pal 5 step 2: Codex run instructions

## Goal of this step

This step is **not** the final science selection. It is the **Bonaca-style baseline** member-selection stage.

The goal is to produce a clean, conservative catalog that can be used for:

1. checking that the Pal 5 stream is recovered in a way comparable to Bonaca+2020,
2. validating the preprocessing output,
3. setting up the later 1D morphology stage.

This step should stay close to Bonaca+2020:

- start from the clean stellar catalog,
- apply the **z-locus** cut,
- apply a **hard isochrone CMD cut**,
- produce a **strict member catalog**.

Do **not** turn this into the deep-sample / full matched-filter stage yet.

---

## Files expected in the project root

- `final_g25_preproc.fits`
- `pal5.dat`
- `pal5_step2_member_selection.py`

---

## How to run

From the project root:

```bash
python pal5_step2_member_selection.py \
  --input final_g25_preproc.fits \
  --iso pal5.dat \
  --outdir step2_outputs
```

If memory is tight, reduce chunk size, for example:

```bash
python pal5_step2_member_selection.py \
  --input final_g25_preproc.fits \
  --iso pal5.dat \
  --outdir step2_outputs \
  --chunk 1000000
```

---

## What the script does

### 1. Cluster-center nuisance alignment

It first takes stars within a small radius around the cluster center in the Pal 5 frame and fits a small nuisance warp to the isochrone in `(g-r)_0` vs `g_0`:

- `dmu`: global magnitude / distance-modulus shift,
- `dc0`: global color offset,
- `dc1`: linear color tilt with magnitude.

These are **not** physical stream-distance measurements.
They only absorb small mismatches between the isochrone file and the observed photometric system.

### 2. z-locus cut

It applies the Bonaca-style stellar locus cut:

- use `(g-z)_0 = 1.7 (g-r)_0 - 0.17`,
- keep objects within `±0.1 mag`,
- keep the strict sample at `20 < g_0 < 23`.

### 3. Hard CMD cut

It then applies a **hard** isochrone selection in `(g-r)_0` vs `g_0`.

At this stage the script uses a **two-arm distance model**:

- `phi1 <= 0` → trailing-arm distance,
- `phi1 > 0` → leading-arm distance.

This is the correct conservative baseline before building the smoother `DM(phi1)` model.

---

## Important scientific rules

### A. Do not use the old probability notebook as the main baseline here

The old notebook `3_cmd_prob.ipynb` is a **probabilistic matched-filter style extension**:

- it builds background CMD PDFs,
- computes `logp_iso`, `logp_bg`, `P_CMD`, `P_MEM`,
- combines CMD and spatial priors.

That is useful **later**, but it is **not** what Bonaca+2020 did for the baseline catalog.

For the baseline replication stage:

- use **hard z-locus + hard CMD cuts**,
- do **not** replace the strict sample with `P_MEM`-weighted selection,
- do **not** merge the notebook logic into this script yet.

### B. Do not apply the ad hoc residual-extinction correction

Do **not** run the earlier `apply_extinction_residual(..., frac=0.14)` step in the baseline pipeline.

If the isochrone does not line up perfectly, absorb that with the nuisance alignment (`dmu`, `dc0`, `dc1`) instead of globally modifying the catalog photometry.

### C. Do not force the `r-z` vs `r` isochrone to drive the selection

At this stage:

- `z` is mainly used for the **stellar locus** cut,
- the strict CMD selection should be based on `(g-r)_0` vs `g_0`.

The `r-z` CMD can be kept as a diagnostic plot, but it should **not** become an extra hard selection criterion in the baseline version.

---

## Why the isochrone might not line up perfectly

If the raw isochrone overlay misses the cluster-center CMD, the likely causes are:

1. small filter-system mismatch,
2. small zeropoint mismatch,
3. tiny residual color term,
4. slightly imperfect distance modulus,
5. the isochrone file not exactly matching the survey photometric system.

This is why the script fits the nuisance terms `dmu`, `dc0`, `dc1`.

Interpret them only as **alignment parameters**, not as astrophysical measurements.

---

## What files to inspect after the run

In `step2_outputs/` inspect these first:

- `pal5_step2_alignment.json`
- `pal5_step2_cutflow.txt`
- `pal5_step2_summary.json`
- `pal5_step2_strict_members.fits`

In `step2_outputs/plots_step2/` inspect these first:

- `qc_cluster_cmd_alignment.png`
- `qc_color_color_zlocus.png`
- `qc_selected_density_phi12.png`
- `qc_selected_density_radec.png`
- `qc_selected_cmd_gr_g.png`

---

## What counts as a successful run

### 1. Cluster CMD alignment plot

`qc_cluster_cmd_alignment.png` should show:

- before alignment: raw isochrone near the cluster main sequence but not necessarily perfect,
- after alignment: the adjusted ridge should trace the cluster MSTO and main sequence reasonably well.

A perfect match is **not** required.
The goal is a stable, sensible selection box.

### 2. z-locus plot

`qc_color_color_zlocus.png` should show:

- a tight stellar locus,
- the selected points concentrated around the Bonaca relation,
- the cut not obviously excluding the stellar ridge.

### 3. Selected density maps

`qc_selected_density_phi12.png` and `qc_selected_density_radec.png` should start to reveal the stream much more clearly than the raw preprocessed catalog.

This is not yet the final optimized map, but it should already be qualitatively stream-like.

### 4. Cutflow

The final count should be much smaller than the parent catalog, but not absurdly tiny.

---

## If the isochrone still looks wrong, allowed Codex iterations

Codex may iterate only in this order.

### First allowed iteration

Widen or shift the nuisance-fit search box:

- expand `FIT_DMU_GRID`,
- expand `FIT_DC0_GRID`,
- expand `FIT_DC1_GRID`.

### Second allowed iteration

If the ridge is still systematically too narrow or too broad, adjust the CMD box width parameters:

- `CMD_W0`
- `CMD_W_SLOPE`
- `CMD_W_MIN`
- `CMD_W_MAX`

### Third allowed iteration

If the faint end bends away from the data in a way that `dc1` cannot absorb, Codex may introduce **one additional weak quadratic color warp term** in the isochrone transform,

```python
c = c_iso + dc0 + dc1*(g-gref) + dc2*(g-gref)**2
```

but only if all three conditions are met:

1. the affine warp (`dmu`, `dc0`, `dc1`) clearly fails,
2. the quadratic term is small,
3. it is documented as a **pure nuisance correction**.

Do **not** add higher-order warps beyond this.

---

## What Codex should NOT do in this step

Codex should **not**:

- replace the strict sample with a probability-only sample,
- add spatial priors to the member selection,
- add a full matched filter,
- add background CMD PDFs,
- fit the stream track yet,
- build the full `DM(phi1)` model yet,
- modify the catalog extinction correction globally,
- mix the deep sample into the strict baseline output.

---

## What to bring back for the next discussion

After the run, bring back:

1. `pal5_step2_cutflow.txt`
2. `pal5_step2_alignment.json`
3. these plots:
   - `qc_cluster_cmd_alignment.png`
   - `qc_color_color_zlocus.png`
   - `qc_selected_density_phi12.png`
   - `qc_selected_cmd_gr_g.png`

Then the next step will be:

- assess whether the strict Bonaca-style sample is good enough,
- decide whether to add the smoother `DM(phi1)` model,
- then move to the 1D morphology stage.
