#!/bin/bash
# Selección CPU anidada y abstención; no usa Optuna ni --time.
# La ejecución es independiente de la campaña activa en pacca05.
#SBATCH --job-name=hyp_cpu_selective_nested
#SBATCH --partition=normal
#SBATCH --nodelist=pacca01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/cpu_selective_nested_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/cpu_selective_nested_%j.err

set -euo pipefail
export PYTHONPATH="$HOME/hyperion:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
source "$HOME/hyperion-venv/bin/activate"

OUT="$HOME/hyperion-results/final/training/cpu_selective_nested_20260918_balancefix"
mkdir -p "$OUT" "$HOME/hyperion-results/final/logs"
python3 "$HOME/hyperion/fase2_clasificador/analysis/evaluate_cpu_selective_nested.py" \
  --dataset "$HOME/hyperion-results/final/training/cpu_feature_audit_20260918/feature_contract_source.csv" \
  --output "$OUT/cpu_selective_nested_lofo.json" \
  --max-per-family-class 1000 \
  --n-jobs 96 \
  --min-coverage 0.60 \
  --variants baseline_pmu,pmu_interactions
