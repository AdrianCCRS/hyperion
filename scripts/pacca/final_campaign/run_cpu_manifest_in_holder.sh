#!/bin/bash
# Corre un manifiesto CPU con el mismo entorno que hyp_final_cpu.sbatch, como
# paso interno del job contenedor: srun --jobid=<id> -N1 -n1 bash este_script <manifiesto>
set -eo pipefail
MANIFEST="${1:?uso: $0 <manifiesto.yaml>}"
KERNEL_ROOT="$HOME/hyperion-kernels"
mkdir -p "$HOME/hyperion-results/final/logs"
module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HOME/hyperion:${PYTHONPATH:-}"
source "$HOME/hyperion-venv/bin/activate"
restore_turbo() { sudo /usr/local/bin/set_turbo_state 0 || true; }
trap restore_turbo EXIT
sudo /usr/local/bin/set_turbo_state 1
cd "$KERNEL_ROOT"
python3 "$HOME/hyperion/fase1_telemetria/run_campaign.py" run-campaign \
  --manifest "$MANIFEST" --node-id pacca-a100 --reference-kernel-ref npb_mg
echo MANIFEST_DONE
