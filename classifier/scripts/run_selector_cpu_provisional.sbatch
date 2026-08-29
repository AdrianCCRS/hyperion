#!/bin/bash
#SBATCH --job-name=hyp_selector_cpu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/selector_cpu_provisional_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/selector_cpu_provisional_%j.err
#
# Fase 2 es analisis offline sobre CSV existentes. No reserva paccaA100.
# Optuna usa validacion anidada leave-one-operation-out y estudios SQLite
# reanudables; relanzar este mismo job completa trials faltantes, no duplica
# los ya terminados.
set -e -o pipefail

source /home/latorresn/hyperion-venv/bin/activate
cd /home/latorresn/hyperion

python -m classifier.selector all \
  --cpu-campaign /home/latorresn/hyperion-results/campaigns/pacca_dual_cpu_full_20260828 \
  --gpu-campaign /home/latorresn/hyperion-results/campaigns/pacca_dual_gpu_full_20260828 \
  --catalog /home/latorresn/hyperion/orchestrator/schemas/kernels/catalog.yaml \
  --cpu-manifest /home/latorresn/hyperion/orchestrator/schemas/campaigns/campaign_pacca_dual_cpu_full.yaml \
  --gpu-manifest /home/latorresn/hyperion/orchestrator/schemas/campaigns/campaign_pacca_dual_gpu_full.yaml \
  --output-dir /home/latorresn/hyperion-results/analysis/selector_cpu_provisional_20260828 \
  --mode cpu-provisional \
  --trials 100 \
  --seed 20260828 \
  --latency-warmups 50 \
  --latency-repeats 200

echo SELECTOR_CPU_PROVISIONAL_DONE
