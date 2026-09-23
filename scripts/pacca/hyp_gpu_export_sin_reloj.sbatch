#!/bin/bash
# Exporta el candidato del clasificador GPU SIN reloj ni potencia (random_forest, 3 variables), elegido tras la
# auditoria gpu_clock_feature_audit (job 7601): sin el reloj la regresion logistica cae a 0.65 de exactitud por
# celda, mientras random_forest no pierde (0.812 sin reloj, 0.819 sin reloj ni potencia). Mide latencia en el
# nodo de destino. Solo CPU. NUNCA local. Sin --time.
#SBATCH --job-name=hyp_gpu_export_sin_reloj
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/gpu_export_sin_reloj_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/gpu_export_sin_reloj_%j.err
set -euo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
source "$HOME/hyperion-venv/bin/activate"
OUT="$HOME/hyperion-results/final/models/gpu_sin_reloj_20260923"
hostname; git -C "$ROOT" log --oneline -1
python3 -m fase2_clasificador.analysis.export_gpu_historical_candidate \
  --source "$ROOT/docs/libro/datos/gpu_calidad_20260922/historical_gpu_relaxed020_20260922.csv" \
  --out "$OUT" --split-rajaperf-cuda --extra-features gpu_mem_util_pct_std \
  --drop-features gpu_sm_clock_mhz_median gpu_power_mw_median \
  --cell-balanced-acc random_forest=0.8191 \
  --tag 20260923_sin_reloj \
  --rationale "Auditoria job 7601: la regresion logistica depende del reloj (0.79 -> 0.65 sin el); random_forest no (0.807 -> 0.812 sin reloj, 0.819 sin reloj ni potencia). El daemon cambia el reloj y la potencia depende de el, asi que se excluyen ambos. Un unico candidato (random_forest) por decision explicita, no por desempate de latencia; la clasificacion ocurre una vez por fase, la latencia de ms es aceptable."
sha256sum "$OUT"/*
ls -la "$OUT"
