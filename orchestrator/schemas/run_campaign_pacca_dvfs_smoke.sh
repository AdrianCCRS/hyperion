#!/bin/bash
# Smoke test de escritura real de frecuencia (P1, ARC-104) contra
# paccaA100. 3 kernels (npb_mg, LavaMD y 3MM) x 10 niveles (REF + nueve
# puntos fijos cada 12,5 %) x 3 reps = 90 corridas. Valida Turbo,
# frecuencia por ventana, suficiencia de F0--F4 y distribución de los
# candidatos antes de comprometer la campaña final.
set -e -o pipefail
# ARC-130: sin "-u" a proposito -- ver run_campaign_pacca_dvfs_full.sh
# (ARC-126/127) para el detalle, mismo bug de Lmod en este cluster.

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

# La primera ejecución con Turbo realmente desactivado no debe quedar
# pegada al límite estimado con relojes nativos. Igual que en la campaña
# completa, el timeout interno deja 15 min para restaurar y persistir.
srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive --time=03:00:00 \
  ~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaign_pacca_dvfs_smoke.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref npb_mg \
  --campaign-timeout-seconds 9900
'
echo CAMPAIGN_SCRIPT_DONE
