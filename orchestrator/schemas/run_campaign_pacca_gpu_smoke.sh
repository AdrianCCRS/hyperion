#!/bin/bash
# Smoke test de la campaña GPU DVFS (ARC-110/111): 1 kernel
# (rodinia_gaussian) x 3 niveles (REF, F0, F4) x 3 repeticiones = 9
# corridas -- valida apply_gpu_frequency() real end-to-end y el kernel
# nuevo, antes de comprometer horas de cola en la campaña completa
# (campaign_pacca_gpu_ref.yaml, 144 combinaciones).
set -uo pipefail

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive --time=00:20:00 bash -c '
module load openblas/0.3.21 2>/dev/null
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaign_pacca_gpu_smoke.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref rodinia_gaussian \
  --campaign-timeout-seconds 900
'
echo GPU_SMOKE_SCRIPT_DONE
