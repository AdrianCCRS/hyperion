#!/bin/bash
# Comparacion completa y pareada de modelos: 5 modelos x 2 representaciones (6 y 12 variables),
# cada uno fijo y con busqueda TPE anidada (15 trials, mismo criterio interno). Sin --time.
#SBATCH --job-name=hyp_cpu_model_grid
#SBATCH --partition=normal
#SBATCH --nodelist=pacca03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/cpu_model_grid_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/cpu_model_grid_%j.err

set -euo pipefail
export PYTHONPATH="$HOME/hyperion:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "$HOME/hyperion-venv/bin/activate"

SRC="$HOME/hyperion-results/final/training/cpu_quality_30fam_20260920/source30.csv"
OUT="$HOME/hyperion-results/final/training/cpu_quality_30fam_base6_20260921"

python3 -m fase2_clasificador.analysis.cpu_quality_report \
  --source "$SRC" --out "$OUT" --stages model_grid \
  --seeds 5 --cap 1000 --optuna-trials 15 --n-jobs 96
