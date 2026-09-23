#!/bin/bash
# Bloque C4 (Plan_Fase3_Daemon.md SS0.1): compila fase3_daemon/cpu_loop/
# (incluido gpu_active_signal_probe, sin PMU/ONNX) y verifica de punta a
# punta la senal de coordinacion CPU-GPU: un proceso C++ real leyendo lo
# que un proceso Python real escribe, atomicamente, en un archivo
# compartido. NUNCA local, ver memoria feedback-never-run-compute-locally.
#SBATCH --job-name=hyp_verify_gpu_coordination_e2e
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/verify_gpu_coordination_e2e_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/verify_gpu_coordination_e2e_%j.err

set -euo pipefail
ROOT="$HOME/hyperion"
ONNXRUNTIME_ROOT="$HOME/.conda/envs/hyperion-cpu-onnx"
BUILD_DIR="$ROOT/fase3_daemon/cpu_loop/build"
CMAKE="$ONNXRUNTIME_ROOT/bin/cmake"
CTEST="$ONNXRUNTIME_ROOT/bin/ctest"

# Entorno real de campanas (para el lado Python -- ver
# hyp_composite_known_smoke.sbatch para el porque).
module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
source "$HOME/hyperion-venv/bin/activate"

hostname
python3 --version

echo "=== build de fase3_daemon/cpu_loop (incluye gpu_active_signal_probe) ==="
"$CMAKE" -S "$ROOT/fase3_daemon/cpu_loop" -B "$BUILD_DIR" -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
"$CMAKE" --build "$BUILD_DIR" -j "${SLURM_CPUS_PER_TASK:-4}" --target gpu_active_reader_test gpu_active_signal_probe

echo "=== ctest: gpu_active_reader_test (puerta dura de este item) ==="
"$CTEST" --test-dir "$BUILD_DIR" --output-on-failure -R '^gpu_active_reader_test$'

echo "=== tests unitarios Python (parseo, sin subproceso real) ==="
python3 -m pytest "$ROOT/fase3_daemon/gpu_loop/tests/test_verify_gpu_coordination_e2e.py" \
  "$ROOT/fase3_daemon/tests/test_coordination.py" -q

echo "=== verificacion C4: proceso C++ real leyendo lo que Python real escribe ==="
SIGNAL_PATH="$BUILD_DIR/gpu_active_signal_e2e_${SLURM_JOB_ID}.txt"
rm -f "$SIGNAL_PATH"
python3 "$ROOT/fase3_daemon/gpu_loop/verify_gpu_coordination_e2e.py" \
  --probe-path "$BUILD_DIR/gpu_active_signal_probe" \
  --signal-path "$SIGNAL_PATH" \
  --probe-duration-s 3.0
