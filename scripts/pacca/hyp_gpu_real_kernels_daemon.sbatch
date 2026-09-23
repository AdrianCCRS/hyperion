#!/bin/bash
# Validacion del daemon de GPU con kernels REALES del catalogo (no los sinteticos del preflight): una aplicacion
# compuesta que encadena un kernel compute_bound y uno memory_bound, con fronteras de fase registradas por el
# driver (composite_known.py). Etapas:
#   1. Sondeo de duracion: cada kernel candidato solo, para saber si dura lo suficiente para la ventana sostenida.
#   2. Ejecuciones de la compuesta con el daemon: brazo sombra (clasifica sin escribir: mide la clasificacion sin
#      retroalimentacion) y brazo activo (escribe el reloj). Se puntua cada decision contra la fase real.
# Toca el reloj de la GPU en el brazo activo: solo con el nodo libre. Sin --time. No intervenir a mano.
#SBATCH --job-name=hyp_gpu_real_kernels
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/gpu_real_kernels_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/gpu_real_kernels_%j.err

set -uo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
KERNEL_ROOT="$HOME/hyperion-kernels"
WORK="$ROOT/fase3_daemon/gpu_loop/real_kernels_work"
SEQ="${SEQ:-gpu_dgemm_n4096 gpu_rajaperf_stream_triad}"
CYCLES="${CYCLES:-3}"
MIN_ACTIVE_S="${MIN_ACTIVE_S:-3}"
mkdir -p "$WORK"; rm -f "$WORK"/*.jsonl "$WORK"/*.out 2>/dev/null

module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
source "$HOME/hyperion-venv/bin/activate"
GPU=0
hostname; git -C "$ROOT" log --oneline -1
cleanup() { sudo -n nvidia-smi -i $GPU -rgc >/dev/null 2>&1 || true; pkill -P $$ -f run_daemon.py 2>/dev/null || true; }
trap cleanup EXIT
sudo -n nvidia-smi -i $GPU -rgc || { echo "sudo -n nvidia-smi -rgc NO permitido" >&2; exit 1; }

echo "=== 0. pytest (poller con ventana sostenida, clasificador, daemon) ==="
(cd "$ROOT" && python3 -m pytest -q fase3_daemon/tests/test_activity_poller.py fase3_daemon/tests/test_gpu_classifier.py \
   fase3_daemon/tests/test_run_daemon.py fase3_daemon/tests/test_gpu_loop.py 2>&1 | tail -6) || { echo "pytest fallo" >&2; exit 1; }

echo; echo "=== 1. duracion de cada kernel candidato (solo, 1 ciclo) ==="
for K in gpu_dgemm_n4096 gpu_cutlass_simt_dgemm_n4096 gpu_rajaperf_stream_triad gpu_rajaperf_stream_copy rodinia_backprop rodinia_dwt2d_s4096; do
  ( cd "$KERNEL_ROOT" && python3 "$ROOT/fase3_daemon/composite_apps/composite_known.py" --node-id pacca-a100 \
      --kernels-root "$KERNEL_ROOT" --sequence $K --cycles 1 --boundaries-out "$WORK/survey_$K.jsonl" 2>&1 | grep -E "^\[ciclo|FALL|Error|no encontrados" | head -2 ) || echo "  $K: no se pudo correr"
done

DWELL_NS=3700000000
run_arm() {  # $1 = sombra|activo
  local arm=$1
  echo; echo "=== 2. compuesta [$SEQ] x $CYCLES ciclos con el daemon, brazo $arm (decision tras ${MIN_ACTIVE_S}s de actividad sostenida) ==="
  ( cd "$KERNEL_ROOT" && python3 "$ROOT/fase3_daemon/composite_apps/composite_known.py" --node-id pacca-a100 \
      --kernels-root "$KERNEL_ROOT" --sequence $SEQ --cycles $CYCLES --boundaries-out "$WORK/phases_$arm.jsonl" \
      > "$WORK/composite_$arm.out" 2>&1 ) & local cp=$!
  python3 "$ROOT/fase3_daemon/run_daemon.py" --policy-table "$ROOT/fase3_daemon/policy_table.yaml" \
    --arm $arm --mode pid --pid $cp --min-dwell-ns $DWELL_NS --gpu-settle-s 0.3 --gpu-index $GPU \
    --delegated-cpus 0 --min-active-s $MIN_ACTIVE_S --log-path "$WORK/daemon_$arm.jsonl" -v > "$WORK/daemon_$arm.out" 2>&1 &
  local dp=$!
  wait $cp; sleep 2; kill $dp 2>/dev/null; wait $dp 2>/dev/null
  cat "$WORK/composite_$arm.out" | cut -c1-160
  python3 - "$WORK/phases_$arm.jsonl" "$WORK/daemon_$arm.jsonl" "$arm" <<'PY'
import json, sys
phases = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
try:
    dec = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
except FileNotFoundError:
    dec = []
ok = n = 0
for r in dec:
    if r.get("label") is None: continue
    ph = next((p for p in phases if p["begin_ns"] <= r["ts_ns"] <= p["end_ns"]), None)
    hint = ph["phase_label_hint"] if ph else None
    kid = ph["kernel_id"] if ph else "(entre fases)"
    good = hint in ("compute_bound", "memory_bound") and hint == r["label"]
    if hint in ("compute_bound", "memory_bound"): n += 1; ok += good
    print(f"  {kid:30s} verdad={hint} clase={r['label']} {'OK' if good else 'ERROR'} objetivo_khz={r['target_freq_khz']} escrito={r['written']} "
          f"actuacion_ms={(r.get('actuation_time_ns') or 0)/1e6:.0f} t_desde_inicio_fase_s={(r['ts_ns']-ph['begin_ns'])/1e9 if ph else float('nan'):.1f}")
print(f"brazo {sys.argv[3]}: clasificacion {ok}/{n} decisiones coinciden con la fase real ({len(phases)} fases, {len(dec)} decisiones)")
PY
  sudo -n nvidia-smi -i $GPU -rgc >/dev/null 2>&1
  echo "reloj tras el brazo: $(nvidia-smi -i $GPU --query-gpu=clocks.sm --format=csv,noheader,nounits) MHz"
}
run_arm sombra
run_arm activo
echo; echo "=== FIN ==="
