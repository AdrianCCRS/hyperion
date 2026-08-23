#!/bin/bash
#SBATCH --job-name=hyp_gpu_dwt2dsz
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/gpu_dwt2d_sweep_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/gpu_dwt2d_sweep_%j.err
#
# Barrido de tamaño de dwt2d (Anexo N). Ver
# orchestrator/schemas/campaigns/campaign_pacca_gpu_dwt2d_size_sweep.yaml
#
# 90 corridas (5 tamaños x 6 niveles GPU x 3 rep), CPU=REF unicamente.
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
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_gpu_dwt2d_size_sweep.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref rodinia_dwt2d \
  --campaign-timeout-seconds 5400
'
echo CAMPAIGN_SCRIPT_DONE
