#!/bin/bash
#SBATCH --job-name=hyp_gpu_temporal
#SBATCH --partition=normal
#SBATCH --nodelist=pacca01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/hyp_gpu_temporal_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/hyp_gpu_temporal_%j.err

set -euo pipefail

ROOT=/home/latorresn/hyperion
RESULTS=/home/latorresn/hyperion-results/final
PYTHON=/home/latorresn/hyperion-venv/bin/python

mkdir -p "$RESULTS/analysis"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

"$PYTHON" -m fase2_clasificador.analysis.extract_historical_gpu_temporal \
  --raw-root "$RESULTS/campaigns" \
  --metadata "$RESULTS/analysis/historical_gpu_merged_20260917.csv" \
  --exclude-family minibude_cuda_bm1 \
  --output "$RESULTS/analysis/historical_gpu_temporal_20260917.csv"
