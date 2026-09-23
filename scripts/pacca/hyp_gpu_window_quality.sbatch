#!/bin/bash
# Reentrenamiento por ventana del clasificador de GPU con el dataset historico SIN CUPTI: construye ventanas de
# 120 ms desde los samples.csv de las corridas historicas y evalua LOFO por familia (por ventana y por votacion,
# frente al modelo actual por corrida). Solo CPU. NUNCA local. Sin --time.
#SBATCH --job-name=hyp_gpu_window_quality
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/gpu_window_quality_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/gpu_window_quality_%j.err
set -euo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
source "$HOME/hyperion-venv/bin/activate"
hostname; git -C "$ROOT" log --oneline -1
python3 -m fase2_clasificador.analysis.gpu_window_quality \
  --source "$ROOT/docs/libro/datos/gpu_calidad_20260922/historical_gpu_relaxed020_20260922.csv" \
  --out "$HOME/hyperion-results/final/analysis/gpu_window_quality_20260923"
