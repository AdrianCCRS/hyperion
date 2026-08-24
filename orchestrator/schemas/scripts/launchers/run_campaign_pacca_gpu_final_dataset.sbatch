#!/bin/bash
#SBATCH --job-name=hyp_gpu_final
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=06:30:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/gpu_final_dataset_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/gpu_final_dataset_%j.err
#
# CORRIDA FINAL GPU -- dataset consolidado. Ver
# orchestrator/schemas/campaigns/campaign_pacca_gpu_final_dataset.yaml
#
# 550 corridas (11 puntos de carga x 10 niveles GPU x 5 rep), CPU=REF.
#
# SE SOMETE CON DEPENDENCIA (afterok de las campañas de prueba). Ojo:
# afterok solo mira el codigo de salida -- NO detecta margenes
# interpolados mal puestos (rechazo I10 masivo, que da exit 0 igual).
# Revisar la prueba V1 sobre el job 6471 antes de dejarla correr.
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
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_gpu_final_dataset.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref rodinia_lavamd \
  --campaign-timeout-seconds 23400
'
echo CAMPAIGN_SCRIPT_DONE
