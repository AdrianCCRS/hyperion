#!/bin/bash
# Compila gemm_native_gpu (GEMM denso via kernel CUDA propio, tiled,
# SIN cuBLAS): reemplazo del GEMM GPU del catalogo. Ver
# Seguimiento_Cambios_Plan_Director.md -- dual_gemm_gpu_N2048 (F1-GPU-007)
# y gpu_dgemm_n4096 miden OI implausible via ncu (DFMA=0, aunque el
# resultado numerico es correcto): el kernel propietario que cuBLAS
# despacha para DGEMM no emite las instrucciones SASS que ncu reconoce
# para contar FLOPs. Este kernel usa CUDA escrito a mano (mismo patron
# que dual_stencil_gpu/dual_cholesky_gpu, que SI miden bien) para evitar
# el problema de raiz.
set -e -o pipefail

REPO="${REPO:-$HOME/hyperion}"
DEST="${DEST:-$HOME/hyperion-kernels/libexec/dual}"
NVCC=/usr/local/cuda/bin/nvcc
GPU_ARCH=sm_80

mkdir -p "$DEST"
cd "$REPO"

echo "== gemm_native_gpu (CUDA propio, tiled, sin cuBLAS) =="
"$NVCC" -O3 -arch=$GPU_ARCH \
    kernels/gemm_native/gemm_native_gpu_bench.cu \
    -o "$DEST/gemm_native_gpu"

echo
echo "== binario en $DEST =="
ls -la "$DEST/gemm_native_gpu"
echo
echo "== checksum del binario real (NO es el que va en el catalogo -- ese es el del wrapper bin/gemm_native_gpu) =="
sha256sum "$DEST/gemm_native_gpu"
echo
echo BUILD_GEMM_NATIVE_DONE
