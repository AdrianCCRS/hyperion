#!/bin/bash
#SBATCH --job-name=hyp_size_pre
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/size_preflight_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/size_preflight_%j.err
#
# T1.2 del plan maestro: pre-vuelo del eje de tamaño (ARC-181). Ver
# orchestrator/schemas/campaigns/campaign_pacca_size_preflight.yaml.
#
# Compara npb_cg y npb_mg en clase B contra clase C, en F0 y F4. Decide si
# subir el tamaño del problema mueve alpha por debajo del umbral 0.226 --la
# palanca más barata del plan, porque los binarios de clase C ya existen y
# no hay que escribir código.
#
# --exclusive es obligatorio: se mide energía RAPL de paquete y E11 lo
# exige para uncore. Sin él, otro inquilino se contabiliza como consumo
# nuestro y la comparación B contra C queda contaminada.
set -e -o pipefail
# ARC-126: sin "-u" a propósito -- rompe "module load" en este clúster.

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

# El timeout interno vence bastante antes que el límite de Slurm para que
# el finally de restauración de frecuencia y Turbo alcance a correr.
~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_size_preflight.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref npb_cg \
  --campaign-timeout-seconds 9000
'
echo CAMPAIGN_SCRIPT_DONE
