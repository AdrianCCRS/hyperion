#!/bin/bash
#SBATCH --job-name=hyp_final_cpu
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/final_cpu_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/final_cpu_%j.err
#
# Campana final de barrido de frecuencia -- CPU (Plan_Detallado §2.4).
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

MANIFEST="$HOME/hyperion/scripts/pacca/final_campaign/cpu_final.yaml"
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
restore_turbo() { sudo /usr/local/bin/set_turbo_state 0 || true; }
trap restore_turbo EXIT
sudo /usr/local/bin/set_turbo_state 1

cd "$KERNEL_ROOT"
python3 "$HOME/hyperion/fase1_telemetria/run_campaign.py" run-campaign \
  --manifest "$MANIFEST" \
  --node-id "$NODE_ID" \
  --reference-kernel-ref "$REFERENCE_KERNEL"

echo FINAL_CPU_DONE
