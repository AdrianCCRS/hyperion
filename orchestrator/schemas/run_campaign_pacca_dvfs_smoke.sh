#!/bin/bash
# Smoke test de escritura real de frecuencia (P1, ARC-104) contra
# paccaA100. 1 kernel (npb_mg) x 6 niveles (REF+F0-F4) x 3 reps = 18
# corridas cortas (~4s cada una) -- valida que apply_frequency() realmente
# pinea el hardware antes de comprometer horas de cola en la campaña
# completa (campaign_pacca_dvfs.yaml).
set -uo pipefail

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive --time=00:20:00 bash -c '
module load openblas/0.3.21 2>/dev/null
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaign_pacca_dvfs_smoke.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref npb_mg \
  --campaign-timeout-seconds 900
'
echo CAMPAIGN_SCRIPT_DONE
