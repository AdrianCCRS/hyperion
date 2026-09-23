#!/bin/bash
# Exporta el random forest de GPU a ONNX (verificacion fila a fila), compila el daemon de GPU en C++
# (fase3_daemon/gpu_loop_cpp) y corre sus pruebas. No toca GPU ni frecuencias. NUNCA local. Sin --time.
#SBATCH --job-name=hyp_gpu_loop_cpp
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/gpu_loop_cpp_build_test_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/gpu_loop_cpp_build_test_%j.err
set -euo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
ORT="$HOME/.conda/envs/hyperion-cpu-onnx"
CMAKE="$ORT/bin/cmake"; CTEST="$ORT/bin/ctest"
CUDA=/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0
BUILD="$ROOT/fase3_daemon/gpu_loop_cpp/build"
hostname; git -C "$ROOT" log --oneline -1

echo "=== 1. exportar el random forest a ONNX (venv con skl2onnx, sklearn 1.9.0) ==="
"$HOME/hyperion-onnx-venv/bin/python3" "$ROOT/fase3_daemon/gpu_loop_cpp/export_onnx.py" \
  --verify-source "$ROOT/docs/libro/datos/gpu_calidad_20260922/historical_gpu_relaxed020_20260922.csv"
ls -la "$ROOT/fase3_daemon/gpu_loop_cpp/"gpu_rf_sin_reloj.*
sha256sum "$ROOT/fase3_daemon/gpu_loop_cpp/"gpu_rf_sin_reloj.*

echo "=== 2. compilar ==="
rm -rf "$BUILD"
"$CMAKE" -S "$ROOT/fase3_daemon/gpu_loop_cpp" -B "$BUILD" -DONNXRUNTIME_ROOT="$ORT" \
  -DNVML_INCLUDE_DIR="$CUDA/include" -DNVML_LIB="$CUDA/targets/x86_64-linux/lib/stubs/libnvidia-ml.so"
"$CMAKE" --build "$BUILD" -j 8 2>&1 | grep -E "warning|error|Error|Built target|Linking" | head -40
test -x "$BUILD/gpu_loop_main" || { echo "gpu_loop_main NO se construyo" >&2; exit 1; }

echo "=== 3. ctest del nucleo (puerta dura) ==="
"$CTEST" --test-dir "$BUILD" --output-on-failure

echo "=== 4. pytest de fase3_daemon/tests (tras retirar el daemon en Python) ==="
source "$HOME/hyperion-venv/bin/activate"
(cd "$ROOT" && python3 -m pytest -q fase3_daemon/tests 2>&1 | tail -8)
echo "=== FIN ==="
