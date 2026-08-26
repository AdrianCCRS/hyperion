#!/bin/bash
#SBATCH --job-name=hyp_gpu_final_v2
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=11:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/gpu_final_dataset_v2_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/gpu_final_dataset_v2_%j.err
#
# CORRIDA FINAL GPU v2 -- dataset consolidado con catalogo ampliado. Ver
# orchestrator/schemas/campaigns/campaign_pacca_gpu_final_dataset_v2.yaml
#
# 850 corridas (17 puntos de carga x 10 niveles GPU x 5 rep), CPU=REF.
# Reemplaza al job 6477 (cancelado 2026-08-25).
set -o pipefail
# ARC-153: sin "-u" a propósito.

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_gpu_final_dataset_v2.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref rodinia_lavamd \
  --campaign-timeout-seconds 39000
'
echo CAMPAIGN_SCRIPT_DONE
