#!/bin/bash
#SBATCH --job-name=hyp_final_gpu
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/final_gpu_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/final_gpu_%j.err
#
# Campana final de barrido de frecuencia -- GPU (Plan_Detallado §2.4).
#
# PRERREQUISITO DURO (Plan §2.4.1): el barrido de GPU no es valido sin
# T_transicion_gpu medido antes. Lo produce la etapa `transition` del
# cribado; por eso este job se somete con --dependency=afterok sobre la
# cadena de cribado y no debe soltarse a mano sin ese dato.
#
# SIN --time A PROPOSITO (ver nota en hyp_final_cpu.sbatch). Detener a mano
# con scancel, nunca por reloj.
#
# DISENO A PRUEBA DE FALLAS: este wrapper NO genera el manifiesto, solo lo
# lee en tiempo de ejecucion -> editable hasta que el job arranque, sin
# perder el puesto en la cola. Para sacar un kernel: borrar su linea.

set -eo pipefail

MANIFEST="$HOME/hyperion/scripts/pacca/final_campaign/gpu_final.yaml"
NODE_ID="pacca-a100"
REFERENCE_KERNEL="npb_mg"
KERNEL_ROOT="$HOME/hyperion-kernels"
LAUNCHER="$HOME/hyperion/common/telemetry/build/telemetry_kernel_launcher"
SCREEN_RUN="$HOME/hyperion-results/screening/pacca_screen_20260909"

mkdir -p "$HOME/hyperion-results/final/logs"

fail() { printf '\n[final-gpu][FATAL] %s\n' "$*" >&2; exit 2; }

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

# --- Aviso (no bloqueante) de elegibilidad ncu ---
# El plan exige que el barrido de GPU use kernels con
# roofline_label_eligible=true. No se filtra automaticamente a proposito: el
# control es manual y el manifiesto manda. Esto solo imprime la diferencia
# para que quede en el log y se pueda auditar despues.
ELIGIBLE="$SCREEN_RUN/manifests/gpu_eligible.yaml"
if [[ -f "$ELIGIBLE" ]]; then
  echo "--- Elegibilidad ncu (informativo, no filtra) ---"
  python3 - "$MANIFEST" "$ELIGIBLE" <<'PY' || true
import sys, yaml
man = yaml.safe_load(open(sys.argv[1])) or {}
elig = yaml.safe_load(open(sys.argv[2])) or {}
def refs(doc):
    out = []
    for item in doc.get("kernels", []) or []:
        out.append(item["kernel_ref"] if isinstance(item, dict) else item)
    return set(out)
m, e = refs(man), refs(elig)
extra = sorted(m - e)
print(f"  en el manifiesto : {len(m)}")
print(f"  elegibles (ncu)  : {len(e)}")
if extra:
    print(f"  ⚠ EN EL MANIFIESTO PERO NO ELEGIBLES ({len(extra)}): {', '.join(extra)}")
    print("    -> revisar si deben sacarse a mano del YAML antes de confiar en el dato.")
else:
    print("  ✓ todos los kernels del manifiesto estan marcados elegibles")
PY
else
  echo "--- Aviso: no existe $ELIGIBLE (el cribado aun no genero la lista elegible) ---"
fi

echo "=== Campana final GPU ==="
echo "Manifiesto : $MANIFEST"
echo "Kernels    : $(grep -c '^- kernel_ref:' "$MANIFEST")"
echo "Nodo       : $(hostname)"
echo "========================="

# --- Turbo: 1 DESACTIVA, 0 reactiva (verificado en hardware 2026-09-12) ---
restore_turbo() { sudo /usr/local/bin/set_turbo_state 0 || true; }
trap restore_turbo EXIT
sudo /usr/local/bin/set_turbo_state 1

cd "$KERNEL_ROOT"
python3 "$HOME/hyperion/fase1_telemetria/run_campaign.py" run-campaign \
  --manifest "$MANIFEST" \
  --node-id "$NODE_ID" \
  --reference-kernel-ref "$REFERENCE_KERNEL"

echo FINAL_GPU_DONE
