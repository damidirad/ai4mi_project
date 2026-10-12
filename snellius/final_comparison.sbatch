#!/bin/bash
#SBATCH --job-name=segthor-final_comparison
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --array=0-26%4
#SBATCH --output=results/slurm/%x-%A_%a.out
#SBATCH --error=results/slurm/%x-%A_%a.err

# Run one model/fold/seed per task: BASELINE, TOP1 and TOP2 over 3 folds x 3 seeds.
set -euo pipefail
cd /home/scur0147/ai4mi_project
EXPERIMENT_PYTHON="/home/scur0147/miniconda3/envs/ai4mi/bin/python"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
srun "$EXPERIMENT_PYTHON" run_experiments.py \
    --config experiments/final_comparison.json \
    --run-index "${SLURM_ARRAY_TASK_ID:-0}" --workers 4 "$@"
