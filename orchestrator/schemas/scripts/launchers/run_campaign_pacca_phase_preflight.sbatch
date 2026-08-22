#!/bin/bash
#SBATCH --job-name=hyp_phase_pre
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/phase_preflight_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/phase_preflight_%j.err
#
# Pre-vuelo de la campaña de fases (ARC-177). Ver
# orchestrator/schemas/campaigns/campaign_pacca_phase_preflight.yaml para los
# cuatro supuestos que falsa y por qué la rejilla es solo F0/F4.
#
# Va ANTES de campaign_pacca_phase_probe.yaml (~4 h) y del bloque de rejilla
# (~9 h): 27 corridas por ~20 min para no descubrir en la hora 4 que los
# kernels nuevos se rechazan en el fondo del rango o que la etiqueta de
# verdad no se cruza con las ventanas.
#
# --time=01:00:00 en vez de 06:00:00 no es solo cosmético: un límite corto
# lo hace elegible para backfill de Slurm, así que puede colarse en huecos
# donde la campaña larga no cabe.
#
# --exclusive es obligatorio igual que en la campaña grande: la medición es
# de energía RAPL a nivel de paquete y E11 lo exige para uncore. Un
# pre-vuelo con otro inquilino en el nodo mediría otra cosa y no validaría
# nada de lo que va a correr después.
set -e -o pipefail
# ARC-126: sin "-u" a propósito -- rompe "module load" en este cluster.

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

# Timeout interno holgado frente a los ~20 min estimados, pero por debajo
# del límite de Slurm para que el finally de restauración de frecuencia y
# Turbo alcance a correr.
~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_phase_preflight.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref ptrchase \
  --campaign-timeout-seconds 2700
'
echo CAMPAIGN_SCRIPT_DONE
