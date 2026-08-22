#!/bin/bash
#SBATCH --job-name=hyp_phase
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/phase_probe_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/phase_probe_%j.err
#
# Campaña de fases y frontera de alpha (ARC-176). Ver
# orchestrator/schemas/campaigns/campaign_pacca_phase_probe.yaml para el
# motivo y el diseño de la rejilla.
#
# Reemplaza a la campaña de rejilla fina (job 6391, cancelada): aquella
# añadía resolución entre 3200 y 2600 MHz, y la tabla de alpha mostró que
# para cargas de alpha bajo la frecuencia óptima cae en el FONDO del rango.
#
# --exclusive es obligatorio: la medición es de energía RAPL a nivel de
# paquete, así que otro inquilino en el nodo se contabilizaría como consumo
# nuestro. También lo exige E11 para uncore.
#
# Se encola con sbatch para que Slurm la programe cuando paccaA100 quede
# libre, sin bloquear una sesión interactiva ni competir con los trabajos ya
# en cola de la cuenta compartida.
set -e -o pipefail
# ARC-126: sin "-u" a propósito -- rompe "module load" en este cluster.

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

# El timeout interno (5 h) vence 1 h antes que el límite de Slurm para que
# el finally de restauración de frecuencia y Turbo alcance a correr y
# campaign_metadata.json quede consistente.
~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_phase_probe.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref ptrchase \
  --campaign-timeout-seconds 18000
'
echo CAMPAIGN_SCRIPT_DONE
