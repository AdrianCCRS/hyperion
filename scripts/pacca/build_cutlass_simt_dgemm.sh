#!/bin/bash
# Compila cutlass_simt_dgemm_bench (DGEMM via CUTLASS con OperatorClass=Simt
# explicito, sin Tensor Core) -- candidato de la revision externa tras
# confirmar que CUBLAS_PEDANTIC_MATH no desactiva Tensor Cores en este
# entorno (F1-GPU-012/013).
set -e -o pipefail

REPO="${REPO:-$HOME/hyperion}"
DEST="${DEST:-$HOME/hyperion-kernels/libexec/gpu}"
CUTLASS_DIR="${CUTLASS_DIR:-$HOME/cutlass}"
NVCC=/usr/local/cuda/bin/nvcc
GPU_ARCH=sm_80

if [[ ! -d "$CUTLASS_DIR/include" ]]; then
  echo "ERROR: no se encontro $CUTLASS_DIR/include -- CUTLASS no esta clonado ahi" >&2
  exit 1
fi

mkdir -p "$DEST"
cd "$REPO"

echo "== cutlass_simt_dgemm_bench (CUTLASS OpClassSimt, sin Tensor Core) =="
"$NVCC" -O3 -arch=$GPU_ARCH -std=c++17 \
    -I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" \
    kernels/gpu/cutlass_simt_dgemm_bench.cu \
    -o "$DEST/cutlass_simt_dgemm_bench"

echo
echo "== binario en $DEST =="
ls -la "$DEST/cutlass_simt_dgemm_bench"
echo
echo "== checksum del binario real =="
sha256sum "$DEST/cutlass_simt_dgemm_bench"
echo
echo BUILD_CUTLASS_SIMT_DGEMM_DONE
