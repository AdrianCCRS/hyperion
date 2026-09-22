#!/bin/bash
#SBATCH --job-name=hyp_gpu_export
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/hyp_gpu_export_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/hyp_gpu_export_%j.err

set -euo pipefail

ROOT=/home/latorresn/hyperion
RESULTS=/home/latorresn/hyperion-results/final/analysis/gpu_historical_nested_optuna_20260922
MODELS=/home/latorresn/hyperion-results/final/models/gpu_historical_20260922
PYTHON=/home/latorresn/hyperion-venv/bin/python

mkdir -p "$MODELS"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

"$PYTHON" fase2_clasificador/analysis/export_gpu_historical_candidate.py \
  --source "$RESULTS/historical_gpu_relaxed020.csv" \
  --out "$MODELS" \
  --seed 2000 \
  --split-rajaperf-cuda \
  --extra-features gpu_mem_util_pct_std
