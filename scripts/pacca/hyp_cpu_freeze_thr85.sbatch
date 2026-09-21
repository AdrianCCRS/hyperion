#!/bin/bash
# Congela el umbral operativo CPU en 0.85 (decision declarada; la regla por exactitud por celda daba 0.93)
# sobre los resultados de hyp_cpu_quality_base6: recalcula final_model y reexporta el modelo. Sin --time.
#SBATCH --job-name=hyp_cpu_freeze_thr85
#SBATCH --partition=normal
#SBATCH --nodelist=pacca03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/cpu_freeze_thr85_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/cpu_freeze_thr85_%j.err

set -euo pipefail
export PYTHONPATH="$HOME/hyperion:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "$HOME/hyperion-venv/bin/activate"

SRC="$HOME/hyperion-results/final/training/cpu_quality_30fam_20260920/source30.csv"
OUT="$HOME/hyperion-results/final/training/cpu_quality_30fam_base6_20260921"
MODEL_DIR="$HOME/hyperion-results/final/models/cpu_selective_candidate_base6_20260921"

python3 -m fase2_clasificador.analysis.cpu_quality_report \
  --source "$SRC" --out "$OUT" --stages final_model --seeds 5 --cap 1000 --n-jobs 96 --threshold 0.85
python3 -m fase2_clasificador.analysis.export_cpu_selective_candidate \
  --dataset "$SRC" --output-dir "$MODEL_DIR" \
  --max-per-family-class 1000 --n-jobs 96 --min-coverage 0.60 --threshold 0.85
