#!/bin/bash
#SBATCH --job-name=hyp_cpu_v2_survivors
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/cpu_v2_survivors_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/cpu_v2_survivors_%j.err
#
# Campaña completa (etiqueta de verdad real, uncore por ventana) sobre los
# 9 sobrevivientes del tamizaje CPU v2 (job 6575). Ver
# orchestrator/schemas/campaigns/campaign_pacca_cpu_screen_v2_survivors.yaml
#
# 9 kernels x 5 niveles x 6 reps = 270 corridas de dataset.
set -e -o pipefail
# ARC-126: sin "-u" a propósito.

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_cpu_screen_v2_survivors.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref cpu_rajaperf_stream_mul \
  --campaign-timeout-seconds 9000
'
echo CAMPAIGN_SCRIPT_DONE
