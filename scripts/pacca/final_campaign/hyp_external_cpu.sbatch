#!/bin/bash
#SBATCH --job-name=hyp_external_cpu
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/external_cpu_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/external_cpu_%j.err
#
# Prueba externa sellada del clasificador CPU (XSBench, RSBench y otras cargas nuevas), misma
# maquinaria que la campana final (hyp_final_cpu.sbatch). MANIFEST se pasa por entorno:
#   sbatch --export=ALL,MANIFEST=<ruta> hyp_external_cpu.sbatch
#
# SIN --time A PROPOSITO: la particion GPU es MaxTime=UNLIMITED y
# DefaultTime=NONE, asi que omitir la directiva da tiempo ilimitado. Un
# limite de tiempo ya nos costo el nodo una vez (job 7046, 2026-09-12): al
# matarlo Slurm por TIME LIMIT, otro usuario tomo el nodo en el acto. Si hay
# que detener esto, se detiene A MANO con scancel.
#
# DISENO A PRUEBA DE FALLAS: este wrapper NO genera el manifiesto, solo lo
# lee. Slurm copia este script al someterlo, pero el YAML se lee en tiempo
# de ejecucion -> el contenido de la campana sigue siendo editable hasta el
# segundo en que el job arranque, sin perder el puesto en la cola.
# Para sacar un kernel: borrar su linea en el YAML. Nada mas.

set -eo pipefail

MANIFEST="${MANIFEST:-$HOME/hyperion/scripts/pacca/final_campaign/cpu_external_screen_20260921.yaml}"
NODE_ID="pacca-a100"
REFERENCE_KERNEL="npb_mg"
KERNEL_ROOT="$HOME/hyperion-kernels"
LAUNCHER="$HOME/hyperion/common/telemetry/build/telemetry_kernel_launcher"

mkdir -p "$HOME/hyperion-results/final/logs"

# --- Guardas explicitas: fallar ruidosamente y ANTES de tocar el hardware ---
fail() { printf '\n[final-cpu][FATAL] %s\n' "$*" >&2; exit 2; }

[[ -f "$MANIFEST" ]]    || fail "No existe el manifiesto: $MANIFEST"
[[ -d "$KERNEL_ROOT" ]] || fail "No existe kernel-root: $KERNEL_ROOT"
[[ -x "$LAUNCHER" ]]    || fail "Falta el binario de telemetria: $LAUNCHER (correr la etapa validate del cribado)"

module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HOME/hyperion:${PYTHONPATH:-}"
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
source "$HOME/hyperion-venv/bin/activate" || fail "No se pudo activar hyperion-venv"

python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" "$MANIFEST" \
  || fail "El manifiesto no es YAML valido: $MANIFEST"

echo "=== Campana final CPU ==="
echo "Manifiesto : $MANIFEST"
echo "Kernels    : $(grep -c '^- kernel_ref:' "$MANIFEST")"
echo "Niveles    : $(grep -c '^- id: ' "$MANIFEST")"
echo "Nodo       : $(hostname)"
echo "========================="

# --- Turbo: 1 DESACTIVA (escribe no_turbo=1), 0 reactiva. Verificado en
# hardware el 2026-09-12; la convencion inversa estaba mal en un script
# previo y el preflight la rechazaba. Restauracion siempre via trap. ---

# SMT (ARC-162/163): el reloj fisico de un nucleo lo resuelven sus dos hilos logicos, asi que el hermano
# de cada CPU delegado debe tener el mismo rango cpufreq que su compañero. El arnes snapshotea y
# restaura al estado inicial, por lo que un residuo (p. ej. min=max=800 MHz) se perpetua si no se corrige
# ANTES del snapshot. Solo se tocan los hermanos de los CPUs delegados y solo si difieren; cualquier
# fallo de escritura aborta.
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

echo EXTERNAL_CPU_DONE
