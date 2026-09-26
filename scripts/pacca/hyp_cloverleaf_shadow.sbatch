#!/bin/bash
# REF + tres pilotos sombra CloverLeaf. Sin brazo activo; sin --time.
#SBATCH --job-name=hyp_cloverleaf_shadow
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/%x_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/%x_%j.err
set -euo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"; OUT="${OUT:-$HOME/hyperion-results/final/cloverleaf_shadow_${SLURM_JOB_ID}}"
SRC="$HOME/hyperion-kernels/external_src/CloverLeaf_CUDAFortran"; INPUT="$ROOT/scripts/pacca/cloverleaf_screen_6000.in"
ORT="$HOME/.conda/envs/hyperion-cpu-onnx"; CMAKE="$ORT/bin/cmake"; CUDA=/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0
set +u; module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 2>&1 || true; set -u
export LD_LIBRARY_PATH="$CUDA/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"; export PYTHONPATH="$ROOT:${PYTHONPATH:-}"; source "$HOME/hyperion-venv/bin/activate"
mkdir -p "$OUT/build"; sha256sum "$INPUT" > "$OUT/input.sha256"; git -C "$SRC" rev-parse HEAD > "$OUT/source_commit.txt"
test -x "$SRC/clover_leaf" || { echo "falta binario validado de CloverLeaf" >&2; exit 2; }
if [ ! -x "$OUT/build/gpu_loop_main" ]; then "$CMAKE" -S "$ROOT/fase3_daemon/gpu_loop_cpp" -B "$OUT/build/gpu" -DONNXRUNTIME_ROOT="$ORT" -DNVML_INCLUDE_DIR="$CUDA/include" -DNVML_LIB="$CUDA/targets/x86_64-linux/lib/stubs/libnvidia-ml.so" && "$CMAKE" --build "$OUT/build/gpu" -j8 --target gpu_loop_main; cp "$OUT/build/gpu/gpu_loop_main" "$OUT/build/gpu_loop_main"; fi
GER="$OUT/build/gpu_energy_read"; g++ -O2 -o "$GER" "$ROOT/fase3_daemon/composite_apps/gpu_energy_read.cpp" -I"$CUDA/include" "$CUDA/targets/x86_64-linux/lib/stubs/libnvidia-ml.so"
R0=/sys/class/powercap/intel-rapl:0; R1=/sys/class/powercap/intel-rapl:1; RMAX=$(cat "$R0/max_energy_range_uj")
echo 'arm,wall_s,e_cpu_j,e_gpu_j,kinetic,daemon_rc' > "$OUT/results.csv"
run() { local arm=$1 d="$OUT/$1"; mkdir -p "$d"; cp "$INPUT" "$d/clover.in"; local dp='' rc=NA; (cd "$d" && exec "$SRC/clover_leaf") >"$d/app.out" 2>&1 & local app=$!; kill -STOP "$app"
  if [[ "$arm" == sombra* ]]; then (exec python3 "$ROOT/fase3_daemon/gpu_loop_cpp/launch_gpu_daemon.py" --policy-table "$ROOT/fase3_daemon/policy_table.yaml" --binary "$OUT/build/gpu_loop_main" --arm sombra -- --model "$ROOT/fase3_daemon/gpu_loop_cpp/gpu_rf_sin_reloj.onnx" --features "$ROOT/fase3_daemon/gpu_loop_cpp/gpu_rf_sin_reloj.features.txt" --gpu-index 0 --min-dwell-ns 3700000000 --settle-s .3 --target-pid "$app" --threshold .90 --log-path "$d/gpu_decisions.jsonl") >"$d/daemon.out" 2>&1 & dp=$!; fi
  sleep 3; e0=$(cat "$R0/energy_uj"); f0=$(cat "$R1/energy_uj"); g0=$($GER 0); t0=$(date +%s.%N); nvidia-smi --query-gpu=utilization.gpu,utilization.memory,clocks.sm,power.draw --format=csv,noheader,nounits -lms 250 >"$d/nvml.csv" 2>&1 & local sp=$!; kill -CONT "$app"; wait "$app"; ar=$?; t1=$(date +%s.%N); e1=$(cat "$R0/energy_uj"); f1=$(cat "$R1/energy_uj"); g1=$($GER 0); kill "$sp" 2>/dev/null || true; wait "$sp" 2>/dev/null || true; [ -n "$dp" ] && { wait "$dp"; rc=$?; }
  wall=$(python3 -c "print($t1-$t0)"); cpu=$(python3 -c "print(((($e1-$e0)%$RMAX)+(($f1-$f0)%$RMAX))/1e6)"); gpu=$(python3 -c "print(($g1-$g0)/1e3)"); kin=$(awk '/^ step:/{v=$8} END{print v}' "$d/clover.out"); echo "$arm,$wall,$cpu,$gpu,$kin,$rc" >> "$OUT/results.csv"; [ "$ar" = 0 ] || return "$ar"; }
run ref; run sombra1; run sombra2; run sombra3
python3 - "$OUT" <<'PY'
import csv,json,pathlib,sys
o=pathlib.Path(sys.argv[1]); rows=list(csv.DictReader(open(o/'results.csv'))); ref=float(rows[0]['kinetic'])
for r in rows: r['kinetic_rel_diff']=abs(float(r['kinetic'])-ref)/abs(ref)
json.dump({'rows':rows,'gate_pre_edp':{'min_phase_s':60,'memory_confidence':.90,'needed_shadow_pilots':2,'active_not_run':True}},open(o/'screening_summary.json','w'),indent=2)
PY
