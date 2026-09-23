#!/bin/bash
# Contrafactual de freq_khz_observed en el clasificador de CPU vigente. Solo CPU, no toca frecuencias. NUNCA local. Sin --time.
#SBATCH --job-name=hyp_cpu_freq_cf
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/cpu_freq_cf_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/cpu_freq_cf_%j.err
set -euo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
source "$HOME/hyperion-venv/bin/activate"
hostname; git -C "$ROOT" log --oneline -1
python3 -m fase2_clasificador.analysis.cpu_freq_feature_counterfactual \
  --model "$ROOT/fase2_clasificador/models/xgboost_cpu.joblib" \
  --metadata "$ROOT/fase2_clasificador/models/xgboost_cpu.metadata.json" \
  --source "$HOME/hyperion-results/final/training/cpu_quality_30fam_20260920/source30.csv" \
  --out "$HOME/hyperion-results/final/analysis/cpu_freq_counterfactual_20260923/cpu_freq_counterfactual.json"
