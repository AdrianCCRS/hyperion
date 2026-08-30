#!/bin/bash
#SBATCH --job-name=hyp_dual_gpu_big
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/dual_gpu_big_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/dual_gpu_big_%j.err
#
# ############################################################
# NO LANZAR TODAVIA (decision 2026-08-30).
# Este es el barrido COMPLETO de frecuencias. Se decidio medir
# primero solo REF (54 corridas totales entre ambos ejes) con
# run_campaign_pacca_dual_{cpu,gpu}_big_ref.sbatch: la etiqueta
# de dispositivo con solo REF coincide 136/136 con la del
# barrido completo en el catalogo actual, y la decision de
# dispositivo vale ordenes de magnitud mas que la de frecuencia.
# La ampliacion de frecuencias se disenara DESPUES de analizar
# REF, y sera selectiva (solo configuraciones cerca de un cruce
# o con headroom demostrado), no este producto cartesiano.
# ############################################################
#
# Campana suplementaria del selector CPU/GPU -- eje GPU, solo los 9
# config_id "big". Ver
# orchestrator/schemas/campaigns/campaign_pacca_dual_gpu_big.yaml. Manifiesto
# SEPARADO de campaign_pacca_dual_gpu_full.yaml por el mismo motivo que el
# eje CPU (CAM-09, ver run_campaign_pacca_dual_cpu_big.sbatch).
#
# PRESUPUESTO DE TIEMPO (calculado, no inventado):
#   9 kernels x 4 niveles de CPU x 8 niveles de GPU x 3 rep = 864 corridas.
#   El timeout/duracion esperada por corrida escala por el mismo
#   _frequency_slowdown_ratio() del eje CPU (orchestrator/runner.py) --
#   solo 4 niveles de CPU aqui (REF/F0/F3/F6), ratios 1.0/1.0/1.6/4.0
#   (suma=7.6), cada uno repetido para los 8 niveles de GPU.
#   suma(expected_runtime_seconds) de los 9 kernels = 1366 s (catalog.yaml).
#   Tiempo real esperado (sin SAFETY_MARGIN, que es solo el techo de kill):
#     1366 s x 3 rep x 8 niveles_gpu x 7.6 (suma ratios cpu) ~= 249161 s
#     ~= 69.2 h
#   Mas el overhead de asentamiento CPU+GPU por corrida (~30-36 s/corrida,
#   medido real en el pre-vuelo GPU job 6681) x 864 corridas ~= 7.2-8.6 h
#   adicionales. Total estimado ~76-78 h. Esto es sustancialmente mas que
#   la campana GPU full completa (68 config_id, ~48-55h) a pesar de tener
#   solo 9 config_id -- el costo aqui esta dominado por el tiempo de
#   computo real de GEMM/Cholesky/FFT N grandes, no por el overhead de
#   asentamiento como en la campana full (kernels pequenos).
#
# DECISION: multi-sesion obligatoria, NO cabe en una ni en pocas sesiones
# de un dia. Mismo patron que run_campaign_pacca_dual_gpu_full.sbatch
# (CAM-11 reanuda solo lo pendiente). --time=08:00:00 por sesion (limite
# practico para no bloquear el nodo compartido mas de una jornada;
# `sinfo -p GPU` reporta TIMELIMIT=infinite, asi que el limite real es de
# cortesia, no tecnico). --campaign-timeout-seconds deja ~1.5h (5400 s) de
# margen sobre el --time de Slurm -- la corrida individual mas cara en el
# peor caso (GEMM N=16384, ratio CPU F6=4.0, SAFETY_MARGIN 3.0:
# 412 s x 4.0 x 3.0 ~= 4944 s ~= 82 min) cabe sobrada en ese margen para
# que el `finally:` de run_campaign restaure la frecuencia antes del kill
# duro de Slurm.
# Con ~76-78h de trabajo real repartido en sesiones de ~6.5h de campana
# (28800 s - 5400 s de margen = 23400 s), hacen falta entre 11 y 13
# relanzamientos de este mismo script. Es una campana larga -- verificar
# con el usuario si vale la pena antes de encolarla completa, o si conviene
# reducir repeticiones/niveles para esta ronda especifica.
#
# COMO RELANZAR: sbatch este mismo script tantas veces como haga falta.
# Verificar con `squeue`/`sacct` + el conteo de corridas aceptadas en
# campaign_metadata.json (dentro de output_dir) antes de cada relanzamiento.
#
# RECOMENDACION (no automatizada en este sbatch): correr primero un smoke
# REF-only equivalente al de CPU (ver campaign_pacca_dual_cpu_big_smoke.yaml
# como plantilla; para GPU bastaria con frequency_levels: [REF] y
# gpu_frequency_levels: [REF] en un manifiesto derivado) antes de comprometer
# la primera sesion de 8h -- confirma tiempos reales de GEMM/FFT/Cholesky
# N=16384 en GPU y memoria pico antes de encolar el resto.
set -e -o pipefail

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:/opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
bash ~/hyperion/scripts/pacca/reset_cpu_freq_range.sh
sudo -n nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" -rgc
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_dual_gpu_big.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref dual_gemm_gpu_N512 \
  --campaign-timeout-seconds 23400
'
echo CAMPAIGN_SCRIPT_DONE
