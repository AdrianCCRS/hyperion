#!/bin/bash
# Compila cublas_dgemm_pedantic_bench (DGEMM cuBLAS con CUBLAS_PEDANTIC_MATH
# forzado, para desactivar el enrutamiento automatico a Tensor Core que usa
# gpu_dgemm_n4096/gpu_dgemm_calibration por defecto -- ver F1-GPU-012/013
# en Seguimiento_Cambios_Plan_Director.md y la nota externa
# Nota_Candidatos_GPU_Compute_Bound_20260914.md.
set -e -o pipefail

REPO="${REPO:-$HOME/hyperion}"
DEST="${DEST:-$HOME/hyperion-kernels/libexec/gpu}"
NVCC=/usr/local/cuda/bin/nvcc
GPU_ARCH=sm_80

mkdir -p "$DEST"
cd "$REPO"

echo "== cublas_dgemm_pedantic_bench (cuBLAS + CUBLAS_PEDANTIC_MATH) =="
"$NVCC" -O3 -arch=$GPU_ARCH \
    kernels/gpu/cublas_dgemm_pedantic_bench.cu \
    -o "$DEST/cublas_dgemm_pedantic_bench" \
    -lcublas

echo
echo "== binario en $DEST =="
ls -la "$DEST/cublas_dgemm_pedantic_bench"
echo
echo "== checksum del binario real =="
sha256sum "$DEST/cublas_dgemm_pedantic_bench"
echo
echo BUILD_CUBLAS_DGEMM_PEDANTIC_DONE
