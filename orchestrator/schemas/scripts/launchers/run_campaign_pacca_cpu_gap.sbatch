#!/bin/bash
#SBATCH --job-name=hyp_cpu_gap_campaign
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/cpu_gap_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/cpu_gap_%j.err
#
# Campaña completa GAP (bfs, pr) -- ver
# orchestrator/schemas/campaigns/campaign_pacca_cpu_gap.yaml
#
# 2 kernels x 6 niveles x 6 reps = 72 corridas, cortas (~7-30s c/u según
# nivel) -- --time=02:00:00 da margen amplio.
set -e -o pipefail
# ARC-126: sin "-u" a propósito.

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_cpu_gap.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref cpu_gap_bfs \
  --campaign-timeout-seconds 6300
'
echo CAMPAIGN_SCRIPT_DONE
