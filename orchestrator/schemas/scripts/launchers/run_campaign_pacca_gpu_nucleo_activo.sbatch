#!/bin/bash
#SBATCH --job-name=hyp_gpu_nucleo
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/gpu_nucleo_activo_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/gpu_nucleo_activo_%j.err
#
# Núcleo activo de la matriz GPU, 6 niveles completos (ARC-194/Anexo I).
# Ver orchestrator/schemas/campaigns/campaign_pacca_gpu_nucleo_activo.yaml.
#
# Extiende el pre-vuelo de 3 niveles (job 6457, ya validado de punta a
# punta) a los 6 niveles GPU completos, con márgenes de potencia finales
# (no provisionales). 144 corridas, ~1-1.5 h estimadas por comparación con
# las 72 corridas de 6457 (33:38 min).
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
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_gpu_nucleo_activo.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref rodinia_gaussian \
  --campaign-timeout-seconds 5400
'
echo CAMPAIGN_SCRIPT_DONE
