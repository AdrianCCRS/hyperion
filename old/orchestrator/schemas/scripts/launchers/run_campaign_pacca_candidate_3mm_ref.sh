#!/bin/bash
# Caracterización REF de 3MM: 1 kernel x 1 nivel x 10 repeticiones.
set -e -o pipefail

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive --time=00:40:00 bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/compiler/gcc/12.4.0/lib64:/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_candidate_3mm_ref.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref npb_mg \
  --campaign-timeout-seconds 2100
'
echo CANDIDATE_3MM_SCRIPT_DONE
