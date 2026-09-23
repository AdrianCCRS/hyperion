#!/bin/bash
# Umbral de abstencion del random forest de GPU sin reloj ni potencia (LOFO sobre corridas ya medidas). Solo CPU. NUNCA local. Sin --time.
#SBATCH --job-name=hyp_gpu_threshold
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/gpu_threshold_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/gpu_threshold_%j.err
set -euo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
source "$HOME/hyperion-venv/bin/activate"
hostname; git -C "$ROOT" log --oneline -1
python3 -m fase2_clasificador.analysis.gpu_threshold_sin_reloj \
  --source "$ROOT/docs/libro/datos/gpu_calidad_20260922/historical_gpu_relaxed020_20260922.csv" \
  --out "$HOME/hyperion-results/final/analysis/gpu_threshold_sin_reloj_20260923"
