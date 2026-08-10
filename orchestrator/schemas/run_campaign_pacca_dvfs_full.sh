#!/bin/bash
# Campaña DVFS completa contra paccaA100 (campaign_pacca_dvfs.yaml): 7
# kernels x 6 niveles (REF+F0-F4) x 3 repeticiones = 126 corridas, tras
# validar el mecanismo con campaign_pacca_dvfs_smoke.yaml (18/18
# aceptadas, frequency_restored_verified=true, ARC-108).
set -uo pipefail

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive --time=03:00:00 bash -c '
module load openblas/0.3.21 2>/dev/null
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaign_pacca_dvfs.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref npb_mg \
  --campaign-timeout-seconds 9000
'
echo CAMPAIGN_SCRIPT_DONE
