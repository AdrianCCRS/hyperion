#!/bin/bash
#SBATCH --job-name=hyp_gpu_rigor
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/hyp_gpu_rigor_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/hyp_gpu_rigor_%j.err

set -euo pipefail

ROOT=/home/latorresn/hyperion
RESULTS=/home/latorresn/hyperion-results/final/analysis/gpu_historical_nested_optuna_20260922
FINAL=/home/latorresn/hyperion-results/final/analysis/gpu_final_rigor_20260922
MODELS=/home/latorresn/hyperion-results/final/models/gpu_historical_20260922
PYTHON=/home/latorresn/hyperion-venv/bin/python

mkdir -p "$FINAL"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# Reemplaza el candidato de threshold/cells (estaba fijo en xgboost) por el
# ganador real (regresion_log).
"$PYTHON" fase2_clasificador/analysis/gpu_quality_report.py \
  --source "$RESULTS/historical_gpu_relaxed020.csv" \
  --out "$FINAL" \
  --seeds 5 --boot 2000 \
  --split-rajaperf-cuda \
  --extra-features gpu_mem_util_pct_std \
  --candidate regresion_log \
  --stages matrix,threshold,cells

# Correlacion/VIF, matriz de confusion + ECE, y validacion externa sellada
# contra Backprop/DWT2D/LavaMD/Myocyte (campana de agosto, nunca tocada hoy).
"$PYTHON" fase2_clasificador/analysis/gpu_final_rigor_audit.py \
  --source "$RESULTS/historical_gpu_relaxed020.csv" \
  --external-campaign-dir /home/latorresn/hyperion-results/campaigns/pacca_gpu_dvfs_20260820 \
  --model "$MODELS/gpu_regresion_log_historical_20260922.joblib" \
  --out "$FINAL" \
  --seeds 5 \
  --candidate regresion_log
