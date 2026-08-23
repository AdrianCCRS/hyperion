#!/bin/bash
#SBATCH --job-name=hyp_gpu_bypass
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/gpu_uncore_disabled_preflight_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/gpu_uncore_disabled_preflight_%j.err
#
# Pre-vuelo del bypass ARC-191 (E12 solo mira kernels de dataset, no de
# calibración). Ver
# orchestrator/schemas/campaigns/campaign_pacca_gpu_uncore_disabled_preflight.yaml.
#
# Primera campaña real con uncore.enabled=false. Antes de comprometer la
# matriz GPU completa del Anexo F (~250-320 corridas), confirma con datos
# reales que el bypass y el criterio de potencia por nivel (ARC-189)
# funcionan de punta a punta, no solo en los tests unitarios.
#
# --exclusive sigue siendo obligatorio aunque uncore esté deshabilitado:
# rapl.enabled=true mide energía de paquete, y otro inquilino en el nodo
# contaminaría esa medición igual que contaminaría uncore.
#
# module nvhpc + HYPERION_GPU_FREQ_WRITE_CAPABLE=1: mismo requisito que
# campaign_pacca_gpu_smoke.yaml -- sin nvhpc, gpu_shim no encuentra nvcc y
# el shim de blocking-sync nunca se compila (ARC-70/153/154): cudaDeviceSynchronize()
# cae al spin por defecto de CUDA en vez de dormir.
set -o pipefail
# ARC-153: sin "-u" a propósito -- Lmod referencia $LD_PRELOAD sin definir
# dentro de "module load", "unbound variable" bajo set -u rompe la carga a
# medio camino sin abortar el resto del script.

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_gpu_uncore_disabled_preflight.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref rodinia_gaussian \
  --campaign-timeout-seconds 2700
'
echo CAMPAIGN_SCRIPT_DONE
