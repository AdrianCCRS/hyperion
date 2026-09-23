#!/bin/bash
# Compila y prueba el nucleo de inferencia del loop de CPU (Bloque B,
# Plan_Fase3_Daemon.md): cpu_phase_controller, cpu_feature_builder,
# onnx_cpu_classifier, cpu_loop_tick -- y mide la latencia de inferencia
# ONNX en paccaA100, el nodo de destino real del daemon (mismo criterio
# que hyp_cpu_latency_paccaA100.sbatch, que mide el lado Python).
# Todo el computo (compilar, enlazar, correr) va aqui -- NUNCA local, ver
# memoria feedback-never-run-compute-locally. Sin --time.
#SBATCH --job-name=hyp_cpu_loop_cpp
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/cpu_loop_cpp_build_test_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/cpu_loop_cpp_build_test_%j.err

set -euo pipefail
ROOT="$HOME/hyperion"
ONNXRUNTIME_ROOT="$HOME/.conda/envs/hyperion-cpu-onnx"
BUILD_DIR="$ROOT/fase3_daemon/cpu_loop/build"

hostname
echo "ONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT"
find "$ONNXRUNTIME_ROOT" -iname "onnxruntime_cxx_api.h" -o -iname "libonnxruntime.so" 2>/dev/null

rm -rf "$BUILD_DIR"
cmake -S "$ROOT/fase3_daemon/cpu_loop" -B "$BUILD_DIR" \
  -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
cmake --build "$BUILD_DIR" -j "${SLURM_CPUS_PER_TASK:-4}"

echo "=== ctest ==="
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo "=== latency_bench (fila a fila, presupuesto ~1000us) ==="
freq_state() {
  echo "[$1] no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo n/a)"
  d=/sys/devices/system/cpu/cpu0/cpufreq
  echo "[$1] cpu0 min=$(cat $d/scaling_min_freq 2>/dev/null) max=$(cat $d/scaling_max_freq 2>/dev/null) cur=$(cat $d/scaling_cur_freq 2>/dev/null) gov=$(cat $d/scaling_governor 2>/dev/null)"
}
freq_state antes
# cwd = fase3_daemon/cpu_loop/: OnnxCpuClassifier carga "xgboost_cpu.onnx"
# con ruta relativa (mismo criterio que WORKING_DIRECTORY del ctest de arriba).
(cd "$ROOT/fase3_daemon/cpu_loop" && \
  LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:${LD_LIBRARY_PATH:-}" \
  taskset -c 0 "$BUILD_DIR/cpu_loop_latency_bench")
freq_state despues
