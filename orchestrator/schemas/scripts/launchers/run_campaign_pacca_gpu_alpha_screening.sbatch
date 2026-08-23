#!/bin/bash
#SBATCH --job-name=hyp_gpu_alpha
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/gpu_alpha_screening_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/gpu_alpha_screening_%j.err
#
# Tamizaje de alpha en GPU sobre candidatos limitados por memoria.
# Ver orchestrator/schemas/campaigns/campaign_pacca_gpu_alpha_screening.yaml
# y el Anexo K del plan maestro.
#
# 54 corridas (3 kernels x 6 niveles GPU x 3 rep), CPU=REF únicamente.
# Presupuesto de tiempo holgado (2 h) porque en F4 el reloj va recortado
# 6.7x y la duración de myocyte a ese nivel es desconocida.
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
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_gpu_alpha_screening.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref rodinia_dwt2d \
  --campaign-timeout-seconds 7200
'
echo CAMPAIGN_SCRIPT_DONE
