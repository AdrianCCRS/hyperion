#!/bin/bash
#SBATCH --job-name=hyp_gpu_optuna
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/hyp_gpu_optuna_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/hyp_gpu_optuna_%j.err

set -euo pipefail

ROOT=/home/latorresn/hyperion
RESULTS=/home/latorresn/hyperion-results/final/analysis/gpu_historical_nested_optuna_20260922
PYTHON=/home/latorresn/hyperion-venv/bin/python

mkdir -p "$RESULTS"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# A1: reconstruye el dataset historico admitiendo corridas que solo fallan
# por usable_sample_fraction<0.5 (ver rebuild_historical_gpu_relaxed_fraction.py).
"$PYTHON" fase2_clasificador/analysis/rebuild_historical_gpu_relaxed_fraction.py \
  --raw-root /home/latorresn/hyperion-results/final/campaigns \
  --min-usable-sample-fraction 0.20 \
  --output "$RESULTS/historical_gpu_relaxed020.csv"

# A2+A6 (best config de la ronda 2026-09-22) + busqueda anidada de
# hiperparametros para los 5 modelos ajustables, 15 pruebas, semilla unica.
"$PYTHON" fase2_clasificador/analysis/gpu_historical_nested_optuna.py \
  --source "$RESULTS/historical_gpu_relaxed020.csv" \
  --out "$RESULTS" \
  --n-trials 15 \
  --seed 2000 \
  --split-rajaperf-cuda \
  --extra-features gpu_mem_util_pct_std
