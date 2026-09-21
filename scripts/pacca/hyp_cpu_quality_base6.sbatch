#!/bin/bash
# Recalcula la bateria de calidad CPU con el vector final de 6 tasas base (30 familias) sobre TODOS los
# intervalos elegibles (no se descartan los de cero fallos de cache). El umbral operativo se congela en 0.85:
# threshold_final registra ademas lo que elegiria la regla por exactitud balanceada por celda (0.93).
# Sin --time (cuenta compartida): se corta a mano con scancel.
#SBATCH --job-name=hyp_cpu_quality_base6
#SBATCH --partition=normal
#SBATCH --nodelist=pacca03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/cpu_quality_base6_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/cpu_quality_base6_%j.err

set -euo pipefail
export PYTHONPATH="$HOME/hyperion:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
source "$HOME/hyperion-venv/bin/activate"

SRC="$HOME/hyperion-results/final/training/cpu_quality_30fam_20260920/source30.csv"
OUT="$HOME/hyperion-results/final/training/cpu_quality_30fam_base6_20260921"
MODEL_DIR="$HOME/hyperion-results/final/models/cpu_selective_candidate_base6_20260921"
mkdir -p "$OUT" "$MODEL_DIR" "$HOME/hyperion-results/final/logs"

# threshold_final documenta la rejilla y la eleccion de la regla; final_model usa el umbral congelado.
# Se omiten external, retrain_ext (comparaban 24 vs 30 familias) y power_w (la fuente no trae power_w).
STAGES="inventory,threshold_final,final_model,metrics_suite,matrix,diagnostics,protocols,learning_curve,smoothing,nested_selective,regularization,importance,knn_ceiling,twins,nested_optuna,adaptation,adaptation_phases,temporal,oi_proxy,selection,oi_feature,cap_sensitivity,threshold_nested,latency"

python3 -m fase2_clasificador.analysis.cpu_quality_report \
  --source "$SRC" --out "$OUT" --stages "$STAGES" \
  --seeds 5 --boot 2000 --cap 1000 --n-jobs 96 --threshold 0.85

python3 -m fase2_clasificador.analysis.cpu_book_data --source "$SRC" --out "$OUT/book" --cap 1000

echo "regla por exactitud por celda eligio: $(python3 -c "import json; print(json.load(open('$OUT/threshold_final.json'))['chosen'])"); umbral congelado: 0.85"

python3 -m fase2_clasificador.analysis.export_cpu_selective_candidate \
  --dataset "$SRC" --output-dir "$MODEL_DIR" \
  --max-per-family-class 1000 --n-jobs 96 --min-coverage 0.60 --threshold 0.85
