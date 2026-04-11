#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_FITS="${1:-/pscratch/sd/y/yunao/palomar5/desi-dr10-palomar5-cat.fits}"
ISO_DAT="${2:-/pscratch/sd/y/yunao/Palomar_5_new/pal5.dat}"
WORKDIR="${3:-$PWD}"

mkdir -p "$WORKDIR/outputs" "$WORKDIR/tmp"

python "$SCRIPT_DIR/preselect_catalog.py" \
  --input "$RAW_FITS" \
  --out-g24 "$WORKDIR/outputs/final_glt24.fits" \
  --out-g25 "$WORKDIR/outputs/final_glt25.fits" \
  --temp-dir "$WORKDIR/tmp/preselect_chunks"

python "$SCRIPT_DIR/apply_residual_extinction.py" \
  --input "$WORKDIR/outputs/final_glt24.fits" \
  --output "$WORKDIR/outputs/final_glt24_extcorr.fits" \
  --frac 0.14

python "$SCRIPT_DIR/compute_membership.py" \
  --input "$WORKDIR/outputs/final_glt24_extcorr.fits" \
  --isochrone "$ISO_DAT" \
  --output "$WORKDIR/outputs/final_glt24_membership.fits" \
  --diag-output "$WORKDIR/outputs/final_glt24_membership_diag.npz"

python "$SCRIPT_DIR/extract_stream_track.py" \
  --input "$WORKDIR/outputs/final_glt24_membership.fits" \
  --output "$WORKDIR/outputs/pal5_stream_track.fits"

python "$SCRIPT_DIR/plot_detection_results.py" \
  --catalog "$WORKDIR/outputs/final_glt24_membership.fits" \
  --track "$WORKDIR/outputs/pal5_stream_track.fits" \
  --outdir "$WORKDIR/outputs/plots"
