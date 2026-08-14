#!/bin/bash
# ARC-133: prueba de humo de los 7 kernels reales a REF (sin barrear
# frecuencia -- ya probado con npb_mg en ARC-131/132), antes de la campaña
# completa. Objetivo: los 5 binarios NPB nunca ejecutados con telemetria
# en esta sesion, y el fix de OpenBLAS de dgemm_n2048 nunca probado.
set -o pipefail
# ARC-130/133: sin "-u" a proposito -- ver run_campaign_pacca_dvfs_full.sh
# (ARC-126/127) para el detalle, mismo bug de Lmod en este cluster.

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive --time=00:20:00 bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaign_pacca_dvfs_smoke_allkernels.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref npb_mg \
  --campaign-timeout-seconds 900
'
echo CAMPAIGN_SCRIPT_DONE
