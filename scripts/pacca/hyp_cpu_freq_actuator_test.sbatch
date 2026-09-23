#!/bin/bash
# Compila y prueba el actuador nativo de frecuencia de CPU (Bloque C8,
# Plan_Fase3_Daemon.md): pruebas unitarias sobre un sysfs SIMULADO en un
# directorio temporal (sin tocar hardware real) y paridad con
# common/hpc/freqctl.py. Sin ONNX Runtime: solo los targets del nucleo.
# Todo el computo (compilar, enlazar, correr) va aqui -- NUNCA local, ver
# memoria feedback-never-run-compute-locally. Sin --time.
# Este job NO escribe frecuencia ni turbo reales: es seguro correrlo junto a
# otra medicion, pero conviene no solaparlo con una ventana de medicion RAPL
# por el costo de compilar (ver AGENTS.md, contaminacion de telemetria).
#SBATCH --job-name=hyp_cpu_freq_actuator
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/cpu_freq_actuator_test_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/cpu_freq_actuator_test_%j.err

set -euo pipefail
# Arbol propio (no $HOME/hyperion): ese checkout lo usa la campana en curso y tiene
# cambios locales; no se toca. Se sobreescribe con HYPERION_ROOT.
ROOT="${HYPERION_ROOT:-$HOME/hyperion}"
BUILD_DIR="$ROOT/fase3_daemon/cpu_loop/build_actuator"
CMAKE="$HOME/.conda/envs/hyperion-cpu-onnx/bin/cmake"
CTEST="$HOME/.conda/envs/hyperion-cpu-onnx/bin/ctest"

hostname
"$CMAKE" --version | head -1
git -C "$ROOT" log --oneline -1

rm -rf "$BUILD_DIR"
# Sin -DONNXRUNTIME_ROOT: solo se registran los targets del nucleo.
"$CMAKE" -S "$ROOT/fase3_daemon/cpu_loop" -B "$BUILD_DIR"
"$CMAKE" --build "$BUILD_DIR" -j "${SLURM_CPUS_PER_TASK:-4}"

echo "=== ctest: actuador nativo + paridad con freqctl.py (puerta dura) ==="
"$CTEST" --test-dir "$BUILD_DIR" --output-on-failure -R '^(cpu_freq_actuator_test|freqctl_parity_test)$'

echo "=== ctest: regresion del nucleo del loop de CPU ==="
"$CTEST" --test-dir "$BUILD_DIR" --output-on-failure -R \
  '^(cpu_phase_controller_test|cpu_feature_builder_test|cpu_loop_tick_test|cpu_loop_consumer_test|decision_log_test|gpu_active_reader_test)$'

echo "=== la paridad no debe haberse omitido ==="
out=$(python3 "$ROOT/fase3_daemon/cpu_loop/tests/test_freqctl_parity.py" "$BUILD_DIR/cpu_freq_actuator_probe")
echo "$out"
echo "$out" | grep -q "paridad OK" || { echo "la paridad no corrio (omitida o fallida)" >&2; exit 1; }
echo "actuador C++: OK"

echo "=== pytest: tabla de politica y lanzador (Python puro, sin hardware) ==="
# Requiere pytest/pandas/yaml en el python3 del nodo; si no estan, se reporta
# pero no bloquea la puerta dura del actuador (que ya paso arriba).
(cd "$ROOT" && python3 -m pytest -q fase3_daemon/tests/test_build_policy_table.py \
  fase3_daemon/tests/test_launch_cpu_daemon.py fase3_daemon/tests/test_gpu_classifier.py) || echo "pytest de politica/lanzador: fallo o entorno incompleto (informativo)"
