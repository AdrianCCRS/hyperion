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
# cmake no esta en el PATH del nodo de computo por defecto (module
# cmake/3.24.2 ni siquiera existe) -- se instalo cmake en el mismo entorno
# conda que trae onnxruntime-cpp, se invoca por ruta absoluta.
CMAKE="$ONNXRUNTIME_ROOT/bin/cmake"
CTEST="$ONNXRUNTIME_ROOT/bin/ctest"

hostname
echo "ONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT"
find "$ONNXRUNTIME_ROOT" -iname "onnxruntime_cxx_api.h" -o -iname "libonnxruntime.so" 2>/dev/null
"$CMAKE" --version | head -1

rm -rf "$BUILD_DIR"
"$CMAKE" -S "$ROOT/fase3_daemon/cpu_loop" -B "$BUILD_DIR" \
  -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
"$CMAKE" --build "$BUILD_DIR" -j "${SLURM_CPUS_PER_TASK:-4}"

echo "=== ctest: tests propios de fase3_daemon/cpu_loop (puerta dura) ==="
"$CTEST" --test-dir "$BUILD_DIR" --output-on-failure -R \
  '^(cpu_phase_controller_test|cpu_feature_builder_test|cpu_loop_tick_test|cpu_loop_consumer_test|onnx_cpu_classifier_test)$'

echo "=== ctest: suite completa (incluye common/telemetry, arrastrada por add_subdirectory) ==="
# Informativo, NO bloquea el job: common/telemetry ya tenia su propia suite
# antes de este trabajo, y verificarla/depurarla no es parte de Bloque B.
# Si algo falla aqui, se reporta pero no aborta -- lo real de Bloque B ya
# quedo verificado en la puerta dura de arriba.
telemetry_status=0
"$CTEST" --test-dir "$BUILD_DIR" --output-on-failure || telemetry_status=$?
echo "ctest completo, codigo: $telemetry_status (informativo, no bloquea)"

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

echo "=== smoke test de cpu_loop_main (produccion real, PMU real) ==="
# Corrida corta y acotada (10s, timeout la corta con SIGTERM -- cpu_loop_main
# ya maneja SIGTERM limpio): monitorea el propio shell de este script como
# target_pid, solo para confirmar que collector.start() abre PMU real sin
# lanzar excepcion, que el consumidor drena muestras reales, y que
# collector.stop() cierra limpio -- no es un experimento de Fase 4 (eso
# necesita un target real y las aplicaciones compuestas del Bloque C).
smoke_status=0
(cd "$ROOT/fase3_daemon/cpu_loop" && \
  LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:${LD_LIBRARY_PATH:-}" \
  timeout --signal=TERM 10s "$BUILD_DIR/cpu_loop_main" \
    --perf-cpus 0,1,2,3 --collector-cpu 4 --consumer-cpu 5 \
    --target-pid $$ --verbose) || smoke_status=$?
# 0 = cpu_loop_main atendio SIGTERM y salio limpio por su cuenta; 124/143 =
# timeout tuvo que insistir (todavia aceptable, no es una excepcion real);
# cualquier otro codigo (p.ej. una excepcion sin capturar de OnnxCpuClassifier
# o de collector.start() por falta de permisos PMU) SI debe fallar el job.
echo "cpu_loop_main codigo de salida: $smoke_status"
case "$smoke_status" in
  0|124|143) echo "smoke test de cpu_loop_main: OK" ;;
  *) echo "smoke test de cpu_loop_main: codigo inesperado" >&2; exit 1 ;;
esac
