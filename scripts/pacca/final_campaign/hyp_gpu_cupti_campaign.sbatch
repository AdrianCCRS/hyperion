#!/bin/bash
#SBATCH --job-name=hyp_gpu_cupti_campaign
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/gpu_cupti_campaign_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/gpu_cupti_campaign_%j.err
#
# Campana GPU con CUPTI Activity (gpu_cupti_20260923.yaml): 19 kernels x
# 10 niveles GPU (REF+F0-F8) x 3 repeticiones = 570 corridas -- NO el
# mismo conjunto de 19 de gpu_final.yaml (ncu offline): rodinia_lavamd/
# rodinia_dwt2d nunca tuvieron modelo analitico CUPTI, se cambiaron por
# minibude_cuda_bm1 (ver cabecera de gpu_cupti_20260923.yaml para el
# detalle completo y la procedencia de cada parametro analitico).
# Smoke real verificado de punta a punta antes de someter esto (job 7586,
# paccaA100): accepted=3/3, rejected=0, matrix_incomplete=false,
# frequency_restored_verified=true. Mismo patron probado que
# hyp_external_cpu.sbatch.
#
# HALLAZGO SIN RESOLVER, NO BLOQUEANTE (job 7586): el shim de
# blocking-sync (common/hpc/gpu_shim.py, ARC-70) no compilo en este nodo
# ("no se encontro un CUDA toolkit real") pese a que cuda_runtime.h SI
# existe ahi (confirmado por separado via srun) -- causa raiz no
# encontrada, posible diferencia sutil de entorno entre un sbatch batch y
# una sesion interactiva. Efecto: cudaDeviceSynchronize() hace spin en
# vez de bloquear, lo que puede sesgar la clasificacion CPU-side por IPC
# (ver memoria blocking-sync-shim-never-synced) -- NO afecta la medicion
# de potencia/reloj/OI de GPU va NVML/CUPTI, que es el eje central de
# esta campana. Revisar antes de usar los datos CPU-side (rapl/perf) de
# esta campana para nada relacionado con IPC o clasificacion CPU.
#
# SIN --time A PROPOSITO: particion GPU MaxTime=UNLIMITED. Un limite de
# tiempo ya costo el nodo una vez (job 7046). Detener siempre a mano con
# scancel.
#
# DISENO A PRUEBA DE FALLAS: este wrapper NO genera el manifiesto, solo lo
# lee en tiempo de ejecucion -- editable hasta que el job arranque.
# run_campaign.py es resumible (CAM-03/CAM-11): si esto muere a medias,
# volver a someter el mismo manifiesto retoma donde quedo, sin remedir lo
# ya aceptado.

set -eo pipefail

MANIFEST="$HOME/hyperion/scripts/pacca/final_campaign/gpu_cupti_20260923.yaml"
NODE_ID="pacca-a100"
REFERENCE_KERNEL="npb_mg"
KERNEL_ROOT="$HOME/hyperion-kernels"

mkdir -p "$HOME/hyperion-results/final/logs"

fail() { printf '\n[gpu-cupti][FATAL] %s\n' "$*" >&2; exit 2; }

[[ -f "$MANIFEST" ]]    || fail "No existe el manifiesto: $MANIFEST"
[[ -d "$KERNEL_ROOT" ]] || fail "No existe kernel-root: $KERNEL_ROOT"
[[ -f "$HOME/hyperion/common/telemetry/build-cupti/libhyperion_cupti_activity.so" ]] \
  || fail "Falta libhyperion_cupti_activity.so"

module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HOME/hyperion:${PYTHONPATH:-}"
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
source "$HOME/hyperion-venv/bin/activate" || fail "No se pudo activar hyperion-venv"

python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" "$MANIFEST" \
  || fail "El manifiesto no es YAML valido: $MANIFEST"

echo "=== Campana GPU CUPTI ==="
echo "Manifiesto : $MANIFEST"
echo "Kernels    : $(grep -c '^- kernel_ref:' "$MANIFEST")"
echo "Nodo       : $(hostname)"
echo "=========================="

align_smt_siblings() {
  local base sib d s pmax pmin
  for base in 0 1 2 3 4 5; do
    d=/sys/devices/system/cpu/cpu$base
    for sib in $(tr ',-' '  ' < $d/topology/thread_siblings_list); do
      [[ "$sib" == "$base" ]] && continue
      s=/sys/devices/system/cpu/cpu$sib/cpufreq
      pmax=$(cat $d/cpufreq/scaling_max_freq); pmin=$(cat $d/cpufreq/scaling_min_freq)
      echo "[smt] cpu$base min=$pmin max=$pmax | hermano cpu$sib antes: min=$(cat $s/scaling_min_freq) max=$(cat $s/scaling_max_freq)"
      if [[ "$(cat $s/scaling_max_freq)" != "$pmax" || "$(cat $s/scaling_min_freq)" != "$pmin" ]]; then
        echo "$pmin" > $s/scaling_min_freq || fail "no se pudo escribir scaling_min_freq de cpu$sib"
        echo "$pmax" > $s/scaling_max_freq || fail "no se pudo escribir scaling_max_freq de cpu$sib"
        [[ "$(cat $s/scaling_max_freq)" == "$pmax" && "$(cat $s/scaling_min_freq)" == "$pmin" ]] || fail "hermano cpu$sib no quedo alineado"
        echo "[smt] cpu$sib alineado: min=$(cat $s/scaling_min_freq) max=$(cat $s/scaling_max_freq)"
      fi
    done
  done
}
align_smt_siblings
restore_turbo() { sudo /usr/local/bin/set_turbo_state 0 || true; }
trap restore_turbo EXIT
sudo /usr/local/bin/set_turbo_state 1

cd "$KERNEL_ROOT"
python3 "$HOME/hyperion/fase1_telemetria/run_campaign.py" run-campaign \
  --manifest "$MANIFEST" \
  --node-id "$NODE_ID" \
  --reference-kernel-ref "$REFERENCE_KERNEL"

echo GPU_CUPTI_CAMPAIGN_DONE
