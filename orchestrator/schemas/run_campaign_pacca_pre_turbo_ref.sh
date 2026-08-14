#!/bin/bash
# Integración REF del pipeline CPU vigente, ejecutable antes de disponer del
# permiso para desactivar Turbo. No produce ni reemplaza el dataset final.
set -e -o pipefail

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive \
  --time=00:30:00 bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaign_pacca_pre_turbo_ref.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref npb_mg \
  --campaign-timeout-seconds 1500
'
echo PRE_TURBO_REF_DONE
