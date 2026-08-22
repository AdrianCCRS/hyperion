#!/bin/bash
#SBATCH --job-name=hyp_fine_grid
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/fine_grid_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/fine_grid_%j.err
#
# Campaña suplementaria de rejilla fina de CPU (ver
# orchestrator/schemas/campaigns/campaign_pacca_cpu_fine_grid.yaml para el
# motivo y el diseño). Se encola con sbatch -- NO con srun -- para que
# Slurm la programe cuando paccaA100 quede libre, sin bloquear una sesión
# interactiva ni competir con los trabajos ya en cola de la cuenta
# compartida.
#
# --exclusive es obligatorio, no opcional: la medición es de energía RAPL a
# nivel de paquete, así que cualquier otro inquilino en el nodo se
# contabilizaría como consumo nuestro. También lo exige E11 para uncore.
#
# --gres=gpu:1 se mantiene igual que en run_campaign_pacca_dvfs_full.sh
# aunque la campaña sea de CPU (gpu.enabled=false): paccaA100 está en la
# partición GPU y la reserva del recurso mantiene la asignación coherente
# con la campaña original, cuyo dataset queremos comparar.
set -e -o pipefail
# ARC-126: sin "-u" a propósito -- rompe "module load" en este cluster
# (Lmod referencia LD_PRELOAD sin definir).

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

# El timeout interno (9 h) vence 1 h antes que el límite de Slurm (10 h)
# para que el finally de restauración de frecuencia y Turbo alcance a
# correr y campaign_metadata.json quede consistente. La campaña original
# (540 corridas, incluyendo niveles de 800 MHz) tardó ~8.5 h; esta tiene
# 630 corridas pero todas entre 2200 y 3200 MHz, o sea más cortas.
~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
# ARC-127: gnu12 debe cargarse junto con openblas (dependencia jerárquica);
# dgemm_n2048 lo necesita en tiempo de ejecución.
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_cpu_fine_grid.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref npb_mg \
  --campaign-timeout-seconds 32400
'
echo CAMPAIGN_SCRIPT_DONE
