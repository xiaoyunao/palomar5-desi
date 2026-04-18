#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Pal 5 mainline driver: step2 -> step3b(control+MAP) -> step4c -> step3b ->
# mockfit -> visualization
#
# This script is intentionally opinionated:
#   - starts from step2 (not preprocessing)
#   - uses step4c + step3b(control+MAP) as the photometric mainline
#   - skips old notebook-style track re-extraction
#   - feeds the step3b profile table directly into the refactored mock fitter
#
# Typical usage:
#   chmod +x pal5_mainline_step2_to_mockfit.sh
#   CODE_DIR=/path/to/palomar5-desi \
#   DATA_DIR=/path/to/Pal5 \
#   PYTHON_PIPELINE=/path/to/python \
#   PYTHON_MOCKFIT=/path/to/python \
#   ./pal5_mainline_step2_to_mockfit.sh
###############################################################################

###############################################################################
# User config
###############################################################################

CODE_DIR="${CODE_DIR:-$PWD}"
DATA_DIR="${DATA_DIR:-$PWD}"
PYTHON_PIPELINE="${PYTHON_PIPELINE:-python}"
PYTHON_MOCKFIT="${PYTHON_MOCKFIT:-$PYTHON_PIPELINE}"

# If RESUME=1, stages with their key output already present are skipped.
RESUME="${RESUME:-1}"

# Stage toggles
RUN_STEP2="${RUN_STEP2:-1}"
RUN_STEP3B_BASE="${RUN_STEP3B_BASE:-1}"
RUN_STEP4C="${RUN_STEP4C:-1}"
RUN_STEP3B_4C="${RUN_STEP3B_4C:-1}"
RUN_STEP3C_COMPARE="${RUN_STEP3C_COMPARE:-0}"
RUN_MOCKFIT="${RUN_MOCKFIT:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"

# Core inputs
PREPROC_FILE="${PREPROC_FILE:-$DATA_DIR/final_g25_preproc.fits}"
ISO_FILE="${ISO_FILE:-$DATA_DIR/pal5.dat}"
STEP3_SEED_PRIOR_FILE="${STEP3_SEED_PRIOR_FILE:-$DATA_DIR/step3_outputs_hw15/pal5_step3_pass1_prior_track.txt}"
RRL_ANCHOR_CSV="${RRL_ANCHOR_CSV:-$CODE_DIR/pal5_rrl_price_whelan_2019_subset.csv}"
ALLOW_GAIA_QUERY="${ALLOW_GAIA_QUERY:-1}"

# Output directories / files
STEP2_OUTDIR="${STEP2_OUTDIR:-$DATA_DIR/step2_outputs}"
STEP3B_BASE_OUTDIR="${STEP3B_BASE_OUTDIR:-$DATA_DIR/step3b_outputs_control}"
STEP4C_OUTDIR="${STEP4C_OUTDIR:-$DATA_DIR/step4c_outputs}"
STEP4C_MEMBERS="${STEP4C_MEMBERS:-$STEP4C_OUTDIR/pal5_step4c_rrlprior_members.fits}"
STEP4C_RRL_CACHE="${STEP4C_RRL_CACHE:-$STEP4C_OUTDIR/pal5_step4c_rrl_enriched.csv}"
STEP4C_STEP3B_OUTDIR="${STEP4C_STEP3B_OUTDIR:-$DATA_DIR/step4c_step3b_outputs_control}"
STEP4C_STEP3C_OUTDIR="${STEP4C_STEP3C_OUTDIR:-$DATA_DIR/step4c_step3c_vs_step3b_baseline}"
MOCKFIT_OUTDIR="${MOCKFIT_OUTDIR:-$DATA_DIR/mockfit_mainline_step4c_trackonly}"
PLOTS_OUTDIR="${PLOTS_OUTDIR:-$MOCKFIT_OUTDIR/pal5_plots}"

# Step 2 knobs
STEP2_CHUNK="${STEP2_CHUNK:-}"

# Mockfit knobs
MCMC_NCORES="${MCMC_NCORES:-1}"
MCMC_MP_START_METHOD="${MCMC_MP_START_METHOD:-spawn}"
MCMC_NWALKERS="${MCMC_NWALKERS:-16}"
MCMC_BURNIN="${MCMC_BURNIN:-5}"
MCMC_STEPS="${MCMC_STEPS:-5}"
USE_WIDTH_TERM="${USE_WIDTH_TERM:-0}"
MOCKFIT_DT_MYR="${MOCKFIT_DT_MYR:-0.5}"
MOCKFIT_N_STREAM_STEPS="${MOCKFIT_N_STREAM_STEPS:-3000}"
MOCKFIT_RELEASE_EVERY="${MOCKFIT_RELEASE_EVERY:-1}"
MOCKFIT_N_PARTICLES="${MOCKFIT_N_PARTICLES:-2}"
MOCKFIT_TRACK_HALFWINDOW_DEG="${MOCKFIT_TRACK_HALFWINDOW_DEG:-0.75}"
MOCKFIT_MIN_PARTICLES_PER_NODE="${MOCKFIT_MIN_PARTICLES_PER_NODE:-16}"
MOCKFIT_MIN_VALID_FRACTION="${MOCKFIT_MIN_VALID_FRACTION:-0.30}"
MOCKFIT_TRACK_JITTER_DEG="${MOCKFIT_TRACK_JITTER_DEG:-0.03}"
MOCKFIT_WIDTH_JITTER_DEG="${MOCKFIT_WIDTH_JITTER_DEG:-0.03}"
MOCKFIT_INCLUDE_STATIC_BAR="${MOCKFIT_INCLUDE_STATIC_BAR:-0}"

# Plotting knobs
PLOT_STAR_FILE="${PLOT_STAR_FILE:-}"
PLOT_STAR_RA_COL="${PLOT_STAR_RA_COL:-RA}"
PLOT_STAR_DEC_COL="${PLOT_STAR_DEC_COL:-DEC}"
PLOT_STAR_DISTANCE_COL="${PLOT_STAR_DISTANCE_COL:-}"
PLOT_STAR_MAX_DISTANCE="${PLOT_STAR_MAX_DISTANCE:-}"
PLOT_SKIP_RV="${PLOT_SKIP_RV:-1}"
PLOT_SKIP_RV_GRIDS="${PLOT_SKIP_RV_GRIDS:-1}"
PLOT_SKIP_ORBIT_GRID="${PLOT_SKIP_ORBIT_GRID:-0}"
PLOT_SKIP_LITERATURE="${PLOT_SKIP_LITERATURE:-1}"

###############################################################################
# Helpers
###############################################################################

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required file not found: $f" >&2
    exit 1
  fi
}

require_script() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required script not found: $f" >&2
    exit 1
  fi
}

maybe_skip_stage() {
  local sentinel="$1"
  if [[ "$RESUME" == "1" && -e "$sentinel" ]]; then
    log "Skipping stage because output already exists: $sentinel"
    return 0
  fi
  return 1
}

run_cmd() {
  log "Running: $*"
  "$@"
}

append_if_set() {
  local var_value="$1"
  local flag_name="$2"
  if [[ -n "$var_value" ]]; then
    CMD+=("$flag_name" "$var_value")
  fi
}

###############################################################################
# Resolve shared inputs
###############################################################################

if [[ ! -f "$ISO_FILE" && -f "$CODE_DIR/pal5.dat" ]]; then
  ISO_FILE="$CODE_DIR/pal5.dat"
fi

require_file "$PREPROC_FILE"
require_file "$ISO_FILE"
require_file "$RRL_ANCHOR_CSV"

require_script "$CODE_DIR/pal5_step2_member_selection.py"
require_script "$CODE_DIR/pal5_step3b_selection_aware_1d_model.py"
require_script "$CODE_DIR/pal5_step4c_rrlprior_dm_selection.py"
require_script "$CODE_DIR/pal5_mock_track_fit_refactor.py"
require_script "$CODE_DIR/pal5_visualize_suite.py"

mkdir -p "$DATA_DIR"

###############################################################################
# Stage 1: step2 strict member selection
###############################################################################

if [[ "$RUN_STEP2" == "1" ]]; then
  if ! maybe_skip_stage "$STEP2_OUTDIR/pal5_step2_strict_members.fits"; then
    mkdir -p "$STEP2_OUTDIR"
    CMD=(
      "$PYTHON_PIPELINE" "$CODE_DIR/pal5_step2_member_selection.py"
      --input "$PREPROC_FILE"
      --iso "$ISO_FILE"
      --outdir "$STEP2_OUTDIR"
    )
    append_if_set "$STEP2_CHUNK" --chunk
    run_cmd "${CMD[@]}"
  fi
fi

require_file "$STEP2_OUTDIR/pal5_step2_strict_members.fits"
require_file "$STEP2_OUTDIR/pal5_step2_summary.json"

###############################################################################
# Stage 2: step3b baseline on step2 strict members (control + MAP)
###############################################################################

if [[ "$RUN_STEP3B_BASE" == "1" ]]; then
  if ! maybe_skip_stage "$STEP3B_BASE_OUTDIR/pal5_step3b_profiles.csv"; then
    mkdir -p "$STEP3B_BASE_OUTDIR"
    CMD=(
      "$PYTHON_PIPELINE" "$CODE_DIR/pal5_step3b_selection_aware_1d_model.py"
      --signal "$STEP2_OUTDIR/pal5_step2_strict_members.fits"
      --preproc "$PREPROC_FILE"
      --step2-summary "$STEP2_OUTDIR/pal5_step2_summary.json"
      --iso "$ISO_FILE"
      --eta-mode control
      --sampler map
      --outdir "$STEP3B_BASE_OUTDIR"
    )
    if [[ -f "$STEP3_SEED_PRIOR_FILE" ]]; then
      CMD+=(--mu-prior-file "$STEP3_SEED_PRIOR_FILE")
    else
      log "step3 seed prior file not found; letting step3b fall back to internal ridge construction"
    fi
    run_cmd "${CMD[@]}"
  fi
fi

require_file "$STEP3B_BASE_OUTDIR/pal5_step3b_profiles.csv"
require_file "$STEP3B_BASE_OUTDIR/pal5_step3b_mu_prior.txt"
require_file "$STEP3B_BASE_OUTDIR/pal5_step3b_summary.json"

###############################################################################
# Stage 3: step4c RR-Lyrae weak-prior DM refinement
###############################################################################

if [[ "$RUN_STEP4C" == "1" ]]; then
  if ! maybe_skip_stage "$STEP4C_MEMBERS"; then
    mkdir -p "$STEP4C_OUTDIR"
    CMD=(
      "$PYTHON_PIPELINE" "$CODE_DIR/pal5_step4c_rrlprior_dm_selection.py"
      --preproc "$PREPROC_FILE"
      --step2-summary "$STEP2_OUTDIR/pal5_step2_summary.json"
      --iso "$ISO_FILE"
      --mu-prior-file "$STEP3B_BASE_OUTDIR/pal5_step3b_mu_prior.txt"
      --rrl-anchor-csv "$RRL_ANCHOR_CSV"
      --rrl-cache-csv "$STEP4C_RRL_CACHE"
      --output-dir "$STEP4C_OUTDIR"
      --output-members "$STEP4C_MEMBERS"
    )
    if [[ "$ALLOW_GAIA_QUERY" == "1" ]]; then
      CMD+=(--allow-gaia-query)
    fi
    run_cmd "${CMD[@]}"
  fi
fi

require_file "$STEP4C_MEMBERS"
require_file "$STEP4C_OUTDIR/pal5_step4c_summary.json"
require_file "$STEP4C_OUTDIR/pal5_step4c_dm_track.csv"

###############################################################################
# Stage 4: rerun step3b(control + MAP) on step4c refined members
###############################################################################

if [[ "$RUN_STEP3B_4C" == "1" ]]; then
  if ! maybe_skip_stage "$STEP4C_STEP3B_OUTDIR/pal5_step3b_profiles.csv"; then
    mkdir -p "$STEP4C_STEP3B_OUTDIR"
    CMD=(
      "$PYTHON_PIPELINE" "$CODE_DIR/pal5_step3b_selection_aware_1d_model.py"
      --signal "$STEP4C_MEMBERS"
      --preproc "$PREPROC_FILE"
      --step2-summary "$STEP2_OUTDIR/pal5_step2_summary.json"
      --iso "$ISO_FILE"
      --mu-prior-file "$STEP3B_BASE_OUTDIR/pal5_step3b_mu_prior.txt"
      --eta-mode control
      --sampler map
      --outdir "$STEP4C_STEP3B_OUTDIR"
    )
    run_cmd "${CMD[@]}"
  fi
fi

require_file "$STEP4C_STEP3B_OUTDIR/pal5_step3b_profiles.csv"
require_file "$STEP4C_STEP3B_OUTDIR/pal5_step3b_summary.json"

###############################################################################
# Optional: Bonaca-style comparison between baseline and step4c mainline
###############################################################################

if [[ "$RUN_STEP3C_COMPARE" == "1" ]]; then
  require_script "$CODE_DIR/pal5_step3c_bonaca_comparison.py"
  if ! maybe_skip_stage "$STEP4C_STEP3C_OUTDIR/pal5_step3c_report.md"; then
    mkdir -p "$STEP4C_STEP3C_OUTDIR"
    run_cmd \
      "$PYTHON_PIPELINE" "$CODE_DIR/pal5_step3c_bonaca_comparison.py" \
      --profiles-map "$STEP4C_STEP3B_OUTDIR/pal5_step3b_profiles.csv" \
      --summary-map "$STEP4C_STEP3B_OUTDIR/pal5_step3b_summary.json" \
      --label-map "step4c + step3b control + MAP" \
      --profiles-alt "$STEP3B_BASE_OUTDIR/pal5_step3b_profiles.csv" \
      --summary-alt "$STEP3B_BASE_OUTDIR/pal5_step3b_summary.json" \
      --label-alt "step2 + step3b control + MAP" \
      --strict-fits "$STEP4C_MEMBERS" \
      --outdir "$STEP4C_STEP3C_OUTDIR"
  fi
fi

###############################################################################
# Stage 5: mock-stream MCMC fit on the step4c+step3b track table
###############################################################################

if [[ "$RUN_MOCKFIT" == "1" ]]; then
  if ! maybe_skip_stage "$MOCKFIT_OUTDIR/best_fit_params.csv"; then
    mkdir -p "$MOCKFIT_OUTDIR"
    CMD=(
      "$PYTHON_MOCKFIT" "$CODE_DIR/pal5_mock_track_fit_refactor.py"
      --track "$STEP4C_STEP3B_OUTDIR/pal5_step3b_profiles.csv"
      --outdir "$MOCKFIT_OUTDIR"
      --ncores "$MCMC_NCORES"
      --mp-start-method "$MCMC_MP_START_METHOD"
      --nwalkers "$MCMC_NWALKERS"
      --burnin "$MCMC_BURNIN"
      --steps "$MCMC_STEPS"
      --dt-myr "$MOCKFIT_DT_MYR"
      --n-stream-steps "$MOCKFIT_N_STREAM_STEPS"
      --release-every "$MOCKFIT_RELEASE_EVERY"
      --n-particles "$MOCKFIT_N_PARTICLES"
      --min-particles-per-node "$MOCKFIT_MIN_PARTICLES_PER_NODE"
      --min-valid-fraction "$MOCKFIT_MIN_VALID_FRACTION"
      --track-half-window-deg "$MOCKFIT_TRACK_HALFWINDOW_DEG"
      --track-jitter-deg "$MOCKFIT_TRACK_JITTER_DEG"
      --width-jitter-deg "$MOCKFIT_WIDTH_JITTER_DEG"
    )
    if [[ "$USE_WIDTH_TERM" == "1" ]]; then
      CMD+=(--use-width-term)
    fi
    if [[ "$MOCKFIT_INCLUDE_STATIC_BAR" == "1" ]]; then
      CMD+=(--include-static-bar)
    fi
    if [[ "$RUN_PLOTS" == "1" ]]; then
      CMD+=(--make-plots --visualize-script "$CODE_DIR/pal5_visualize_suite.py" --plots-outdir "$PLOTS_OUTDIR")
      if [[ -n "$PLOT_STAR_FILE" ]]; then
        CMD+=(--plot-star-file "$PLOT_STAR_FILE" --plot-star-ra-col "$PLOT_STAR_RA_COL" --plot-star-dec-col "$PLOT_STAR_DEC_COL")
        if [[ -n "$PLOT_STAR_DISTANCE_COL" ]]; then
          CMD+=(--plot-star-distance-col "$PLOT_STAR_DISTANCE_COL")
        fi
        if [[ -n "$PLOT_STAR_MAX_DISTANCE" ]]; then
          CMD+=(--plot-star-max-distance "$PLOT_STAR_MAX_DISTANCE")
        fi
      fi
      if [[ "$PLOT_SKIP_RV" == "1" ]]; then
        CMD+=(--plot-skip-rv)
      fi
      if [[ "$PLOT_SKIP_RV_GRIDS" == "1" ]]; then
        CMD+=(--plot-skip-rv-grids)
      fi
      if [[ "$PLOT_SKIP_ORBIT_GRID" == "1" ]]; then
        CMD+=(--plot-skip-orbit-grid)
      fi
      if [[ "$PLOT_SKIP_LITERATURE" == "1" ]]; then
        CMD+=(--plot-skip-literature)
      fi
    fi
    run_cmd "${CMD[@]}"
  fi
fi

if [[ "$RUN_MOCKFIT" == "1" ]]; then
  require_file "$MOCKFIT_OUTDIR/best_fit_params.csv"
  require_file "$MOCKFIT_OUTDIR/best_fit_model_track.fits"
  require_file "$MOCKFIT_OUTDIR/best_fit_mock_stream_particles.fits"
fi

###############################################################################
# Final summary
###############################################################################

log "Mainline pipeline finished. Key products:"
echo "  Step2 members:          $STEP2_OUTDIR/pal5_step2_strict_members.fits"
echo "  Baseline step3b track:  $STEP3B_BASE_OUTDIR/pal5_step3b_profiles.csv"
echo "  Step4c members:         $STEP4C_MEMBERS"
echo "  Mainline step3b track:  $STEP4C_STEP3B_OUTDIR/pal5_step3b_profiles.csv"
if [[ "$RUN_MOCKFIT" == "1" ]]; then
  echo "  Mockfit params:         $MOCKFIT_OUTDIR/best_fit_params.csv"
  echo "  Mockfit track:          $MOCKFIT_OUTDIR/best_fit_model_track.fits"
  echo "  Mockfit particles:      $MOCKFIT_OUTDIR/best_fit_mock_stream_particles.fits"
fi
if [[ "$RUN_PLOTS" == "1" ]]; then
  echo "  Plot directory:         $PLOTS_OUTDIR"
fi
