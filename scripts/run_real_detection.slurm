#!/bin/bash
#SBATCH -J pal5-detect
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/pal5_detect_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate desi_env
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

bash ./run_real_detection.sh "$@"
