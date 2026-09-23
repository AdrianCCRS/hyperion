#!/bin/bash
# Congela el candidato CPU actual sin Optuna y sin --time.
#SBATCH --job-name=hyp_cpu_export_selective
#SBATCH --partition=normal
#SBATCH --nodelist=pacca03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/cpu_export_selective_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/cpu_export_selective_%j.err
set -euo pipefail
export PYTHONPATH="$HOME/hyperion:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
source "$HOME/hyperion-venv/bin/activate"
python3 "$HOME/hyperion/fase2_clasificador/analysis/export_cpu_selective_candidate.py" \
  --dataset "$HOME/hyperion-results/final/training/cpu_feature_audit_20260918/feature_contract_source.csv" \
  --output-dir "$HOME/hyperion-results/final/models/cpu_selective_candidate_20260918" \
  --max-per-family-class 1000 --n-jobs 96 --min-coverage 0.60
