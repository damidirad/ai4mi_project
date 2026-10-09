#!/bin/bash
#SBATCH --job-name=segthor-screening
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --array=0
#SBATCH --output=results/slurm/%x-%A_%a.out
#SBATCH --error=results/slurm/%x-%A_%a.err

# Run one configured experiment per array task; index 0 is M1.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
EXPERIMENT_PYTHON="${EXPERIMENT_PYTHON:-$PWD/ai4mi/bin/python}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
srun "$EXPERIMENT_PYTHON" run_experiments.py \
    --config experiments/screening.json \
    --run-index "${SLURM_ARRAY_TASK_ID:-0}" --workers 4 "$@"
