#!/bin/bash
#SBATCH --job-name=hyp_fgrid_pre
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/fine_grid_preflight_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/fine_grid_preflight_%j.err
#
# T1.4 del plan maestro: pre-vuelo de rejilla fina 2800-3200 MHz (ARC-181).
# Ver orchestrator/schemas/campaigns/campaign_pacca_fine_grid_preflight.yaml.
#
# Reemplaza a la campaña de rejilla fina 6391, cancelada por un razonamiento
# equivocado: se aplicó al objetivo de energía-con-holgura una conclusión
# que solo vale para el EDP sin restricciones. Bajo holgura `s`, el óptimo
# cae en f* = f_ref/(1+s/alpha), que para los nueve kernels del dataset está
# en 2831-3051 MHz -- una banda sin una sola medición.
#
# --exclusive obligatorio por RAPL de paquete y E11 (uncore).
set -e -o pipefail
# ARC-126: sin "-u" a propósito -- rompe "module load" en este clúster.

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_fine_grid_preflight.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref npb_cg \
  --campaign-timeout-seconds 4500
'
echo CAMPAIGN_SCRIPT_DONE
