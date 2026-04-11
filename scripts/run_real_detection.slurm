#!/bin/bash
#SBATCH -J pal5-detect
#SBATCH -q interactive
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH -t 02:00:00
#SBATCH -o logs/pal5_detect_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /global/common/software/nersc9/python/miniforge3/etc/profile.d/conda.sh || true

bash scripts/run_real_detection.sh "$@"
