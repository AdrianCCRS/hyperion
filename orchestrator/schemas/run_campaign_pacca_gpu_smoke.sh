#!/bin/bash
# Smoke test de la campaña GPU DVFS (ARC-110/111): 1 kernel
# (rodinia_gaussian) x 3 niveles (REF, F0, F4) x 3 repeticiones = 9
# corridas -- valida apply_gpu_frequency() real end-to-end y el kernel
# nuevo, antes de comprometer horas de cola en la campaña completa
# (campaign_pacca_gpu_ref.yaml, 144 combinaciones).
# ARC-153: sin "-u" a proposito -- Lmod referencia $LD_PRELOAD sin definir
# dentro de "module load", "unbound variable" bajo set -u rompe la carga del
# modulo nvhpc a medio camino (nvcc nunca queda en PATH) sin abortar el
# resto del script -- mismo bug ya documentado en run_campaign_pacca_dvfs_full.sh
# (ARC-126) y run_campaign_pacca_dvfs_smoke.sh (ARC-130), nunca aplicado aqui
# porque este script no necesito nvhpc/module load hasta el fix del shim.
set -o pipefail

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive --time=00:20:00 bash -c '
# ARC-153: sin el modulo nvhpc, gpu_shim.compiled_blocking_sync_shim()
# nunca encuentra nvcc (_find_cuda_root() devuelve None) -- confirmado que
# esto hizo que ninguna corrida GPU compilara jamas el shim de blocking
# sync desde ARC-70, sin importar que orchestrator/native/blocking_sync_shim.cpp
# estuviera presente: cudaDeviceSynchronize() siempre cayo al comportamiento
# de spin por defecto de CUDA. gnu12 es prerequisito jerarquico de ambos
# modulos (mismo patron que ARC-127 ya encontro para openblas).
module load gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
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
