#!/bin/bash
#SBATCH --job-name=hyp_cpu_cholmod_campaign
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/cpu_cholmod_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/cpu_cholmod_%j.err
#
# Campaña completa CHOLMOD -- ver
# orchestrator/schemas/campaigns/campaign_pacca_cpu_cholmod.yaml
#
# 1 kernel x 6 niveles x 6 reps = 36 corridas (~6.5s a F0, hasta ~26s
# esperados en F4). --mem=8G: peak medido 2042 MB por corrida.
set -e -o pipefail
# ARC-126: sin "-u" a propósito.

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_cpu_cholmod.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref cpu_cholmod \
  --campaign-timeout-seconds 10200
'
echo CAMPAIGN_SCRIPT_DONE
