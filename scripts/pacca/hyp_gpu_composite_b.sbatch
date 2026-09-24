#!/bin/bash
# Compuesta B de GPU (familias inéditas Rodinia repetidas: dwt2d memory x10 + lavamd compute x4, 2 ciclos) con el daemon de
# GPU en C++ en brazo SOMBRA (decide, nunca escribe). Comprueba que el rastreador mantiene la fase sostenida entre repeticiones
# y puntua la clasificacion contra las fronteras reales. Solo con el nodo libre. Sin --time.
#SBATCH --job-name=hyp_gpu_composite_b
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/gpu_composite_b_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/gpu_composite_b_%j.err

set -uo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
KERNEL_ROOT="$HOME/hyperion-kernels"
CPP="$ROOT/fase3_daemon/gpu_loop_cpp"
WORK="$CPP/real_work_b"
GPU=0
mkdir -p "$WORK"; rm -f "$WORK"/*.jsonl "$WORK"/*.out 2>/dev/null
module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
source "$HOME/hyperion-venv/bin/activate"
BIN="$CPP/build/gpu_loop_main"
hostname; git -C "$ROOT" log --oneline -1
[ -x "$BIN" ] || { echo "falta $BIN" >&2; exit 1; }
sudo -n nvidia-smi -i $GPU -rgc >/dev/null 2>&1 || true

( cd "$KERNEL_ROOT" && python3 "$ROOT/fase3_daemon/composite_apps/composite_gpu_unseen.py" --node-id pacca-a100 \
    --kernels-root "$KERNEL_ROOT" --boundaries-out "$WORK/phases.jsonl" > "$WORK/composite.out" 2>&1 ) & CP=$!
( exec python3 "$CPP/launch_gpu_daemon.py" --policy-table "$ROOT/fase3_daemon/policy_table.yaml" --binary "$BIN" --arm sombra -- \
    --model "$CPP/gpu_rf_sin_reloj.onnx" --features "$CPP/gpu_rf_sin_reloj.features.txt" --gpu-index $GPU \
    --min-dwell-ns 3700000000 --settle-s 0.3 --poll-interval-ms 50 --target-pid $CP --log-path "$WORK/real.jsonl" ) > "$WORK/daemon.out" 2>&1 & DP=$!
wait $CP; sleep 2; kill -TERM $DP 2>/dev/null; wait $DP 2>/dev/null
cut -c1-120 "$WORK/composite.out" | tail -30
python3 - "$WORK/phases.jsonl" "$WORK/real.jsonl" <<'PY'
import json, sys
phases = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
try: dec = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
except FileNotFoundError: dec = []
ok = wrong = abst = 0
for r in dec:
    t_start = r["ts_ns"] - int(r["features"].get("phase_active_for_s", 0) * 1e9)
    ph = next((p for p in phases if p["begin_ns"] - 1_000_000_000 <= t_start <= p["end_ns"]), None)
    hint = ph["phase_label_hint"] if ph else None
    if r["label"] == "revisar": v = "ABSTUVO"; abst += 1
    elif hint in ("compute_bound", "memory_bound") and hint == r["label"]: v = "OK"; ok += 1
    else: v = "ERROR"; wrong += 1
    print(f"  {(ph['kernel_id'] if ph else '(entre fases)'):16s} verdad={hint} clase={r['label']} conf={r['confidence']:.2f} {v} "
          f"activo_s={r['features'].get('phase_active_for_s', 0):.1f}")
print(f"sombra: {ok} correctas, {wrong} erroneas, {abst} abstenciones ({len(phases)} lanzamientos, {len(dec)} decisiones)")
PY
echo "reloj final: $(nvidia-smi -i $GPU --query-gpu=clocks.sm --format=csv,noheader,nounits) MHz"
