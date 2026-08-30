#!/bin/bash
#SBATCH --job-name=hyp_dual_cpu_big
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/dual_cpu_big_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/dual_cpu_big_%j.err
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
# Campana suplementaria del selector CPU/GPU -- eje CPU, solo los 9
# config_id "big" (GEMM/Cholesky/FFT N=8192/12288/16384). Ver
# orchestrator/schemas/campaigns/campaign_pacca_dual_cpu_big.yaml. Manifiesto
# SEPARADO de campaign_pacca_dual_cpu_full.yaml a proposito (bug CAM-09:
# agregar estos kernel_ref al manifiesto de 68 config_id ya aceptados
# cambia compute_protocol_fingerprint() y dispara
# CampaignProtocolMismatchError -- ver commit que revierte esa mezcla).
#
# PRESUPUESTO DE TIEMPO (calculado, no inventado):
#   suma(expected_runtime_seconds) de los 9 kernels = 1366 s (una iteracion
#   cada uno, catalog.yaml).
#   9 kernels x 8 niveles de frecuencia x 3 rep = 216 corridas.
#   El propio orquestador (_frequency_slowdown_ratio en orchestrator/runner.py)
#   escala el timeout de CADA corrida por max_khz/target_khz segun el nivel
#   fijo -- con el rango real del nodo (~3.2GHz -> 0.8GHz una vez Turbo
#   deshabilitado, ARC-136/140) los ratios por nivel son:
#     REF=1.0 F0=1.0 F1=1.143 F2=1.333 F3=1.6 F4=2.0 F5=2.667 F6=4.0
#   (suma=14.743). Tiempo real esperado (sin el SAFETY_MARGIN=3.0 que solo
#   es el techo de kill por corrida, no el tiempo esperado):
#     1366 s x 3 rep x 14.743 ~= 60417 s ~= 16.8 h
#   Overhead de arranque/asentamiento por corrida (~5.5 s/corrida medido en
#   el pre-vuelo del job 6680) anade ~20 min, despreciable frente a lo
#   anterior. Una revision externa, sumando margen adicional de
#   variabilidad termica/DVFS no capturado por la formula anterior, estimo
#   ~22 h. Ambos numeros exceden por mucho las 4h que tenia el sbatch
#   original de la campana "big" (bug real, corregido aqui) y tambien
#   exceden una sola sesion razonable en un nodo compartido.
#
# DECISION: multi-sesion, mismo patron ya usado y documentado en
# run_campaign_pacca_dual_gpu_full.sbatch (CAM-11 reanuda solo lo
# pendiente). --time=08:00:00 por sesion (la particion GPU no tiene limite
# duro -- `sinfo -p GPU` reporta TIMELIMIT=infinite en pacca -- pero 8h es
# el techo razonable para no bloquear el nodo compartido mas de una jornada
# por sesion). --campaign-timeout-seconds deja margen de ~1.5h (5400 s)
# sobre el --time de Slurm: la corrida individual mas cara en el peor caso
# (GEMM N=16384 en F6: 412 s x 4.0 x SAFETY_MARGIN 3.0 ~= 4944 s ~= 82 min)
# debe caber completa dentro de ese margen para que el `finally:` de
# run_campaign restaure la frecuencia ANTES del kill duro de Slurm.
# Con ~16.8-22h de trabajo real repartido en sesiones de ~6.5h de campana
# (28800 s de --time - 5400 s de margen = 23400 s), hacen falta entre 3 y 4
# relanzamientos de este mismo script para completar la campana.
#
# COMO RELANZAR: sbatch este mismo script tantas veces como haga falta.
# Verificar con `squeue`/`sacct` + el conteo de corridas aceptadas en
# campaign_metadata.json (dentro de output_dir) antes de cada relanzamiento
# para saber cuantas sesiones faltan.
#
# RECOMENDACION (no automatizada en este sbatch): antes de la primera
# sesion larga, correr a mano el smoke REF-only
# campaign_pacca_dual_cpu_big_smoke.yaml (9 corridas, ~1366 s + overhead,
# cabe en minutos) para confirmar tiempos reales y memoria pico antes de
# comprometer una sesion de 8h. Ver ese manifiesto para el comando exacto.
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
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_dual_cpu_big.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref dual_gemm_cpu_N512 \
  --campaign-timeout-seconds 23400
'
echo CAMPAIGN_SCRIPT_DONE
