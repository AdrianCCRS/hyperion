#!/bin/bash
# Compila los OCHO binarios del selector CPU/GPU: cuatro operaciones (GEMM,
# FFT, AXPY, Stencil Jacobi 2D) con una implementacion de CPU y una de GPU
# cada una, mismos parametros de CLI (--size/--iterations) y mismo formato de
# salida, para que las campañas de ambos lados sean comparables fila a fila.
#
#   gemm_cpu     <- kernels/dgemm/dgemm_bench.c           (OpenBLAS)
#   gemm_gpu     <- kernels/dual/gemm_gpu_dispatch.cu     (cuBLAS + transferencias)
#   fft_cpu      <- kernels/dual/fft_cpu_bench.c          (FFTW)
#   fft_gpu      <- kernels/dual/fft_gpu_dispatch.cu      (cuFFT + transferencias)
#   axpy_cpu     <- kernels/dual/axpy_cpu_bench.c         (OpenBLAS BLAS-1)
#   axpy_gpu     <- kernels/dual/axpy_gpu_dispatch.cu     (cuBLAS + transferencias)
#   stencil_cpu  <- kernels/dual/stencil_cpu_bench.c      (OpenMP, kernel propio)
#   stencil_gpu  <- kernels/dual/stencil_gpu_dispatch.cu  (CUDA, kernel propio)
#
# Los binarios *_gpu miden H2D+computo+D2H DENTRO de la ventana a proposito
# (ver el comentario de cabecera de gemm_gpu_dispatch.cu): sin el costo de
# transferencia la frontera CPU/GPU que aprenderia el modelo no seria real.
#
# ARC-126: `set -e -o pipefail` SIN `-u` -- Lmod referencia variables no
# definidas (LD_PRELOAD) y `-u` aborta el script al cargar cualquier modulo.
set -e -o pipefail

REPO="${REPO:-$HOME/hyperion}"
DEST="${DEST:-$HOME/hyperion-kernels/libexec/dual}"

OPENBLAS_ROOT=/opt/ohpc/pub/libs/gnu12/openblas/0.3.21
FFTW_ROOT=/opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10
NVCC=/usr/local/cuda/bin/nvcc
# A100-PCIE-40GB = compute capability 8.0.
GPU_ARCH=sm_80

module load gnu12/12.4.0

mkdir -p "$DEST"
cd "$REPO"

echo "== gemm_cpu (OpenBLAS) =="
gcc -O3 -march=native -fopenmp \
    -I"$OPENBLAS_ROOT/include" \
    kernels/dgemm/dgemm_bench.c \
    -o "$DEST/gemm_cpu" \
    -L"$OPENBLAS_ROOT/lib" -lopenblas -lm

echo "== fft_cpu (FFTW) =="
gcc -O3 -march=native -fopenmp \
    -I"$FFTW_ROOT/include" \
    kernels/dual/fft_cpu_bench.c \
    -o "$DEST/fft_cpu" \
    -L"$FFTW_ROOT/lib" -lfftw3_omp -lfftw3 -lm

echo "== gemm_gpu (cuBLAS) =="
"$NVCC" -O3 -arch=$GPU_ARCH \
    kernels/dual/gemm_gpu_dispatch.cu \
    -o "$DEST/gemm_gpu" \
    -lcublas

echo "== fft_gpu (cuFFT) =="
"$NVCC" -O3 -arch=$GPU_ARCH \
    kernels/dual/fft_gpu_dispatch.cu \
    -o "$DEST/fft_gpu" \
    -lcufft

echo "== axpy_cpu (OpenBLAS BLAS-1) =="
gcc -O3 -march=native \
    -I"$OPENBLAS_ROOT/include" \
    kernels/dual/axpy_cpu_bench.c \
    -o "$DEST/axpy_cpu" \
    -L"$OPENBLAS_ROOT/lib" -lopenblas -lm

echo "== axpy_gpu (cuBLAS) =="
"$NVCC" -O3 -arch=$GPU_ARCH \
    kernels/dual/axpy_gpu_dispatch.cu \
    -o "$DEST/axpy_gpu" \
    -lcublas

echo "== stencil_cpu (OpenMP) =="
gcc -O3 -march=native -fopenmp \
    kernels/dual/stencil_cpu_bench.c \
    -o "$DEST/stencil_cpu" \
    -lm

echo "== stencil_gpu (CUDA) =="
"$NVCC" -O3 -arch=$GPU_ARCH \
    kernels/dual/stencil_gpu_dispatch.cu \
    -o "$DEST/stencil_gpu"

echo
echo "== binarios en $DEST =="
ls -l "$DEST"
echo
echo "== checksums (para binary_checksum de los wrappers) =="
sha256sum "$DEST"/gemm_cpu "$DEST"/gemm_gpu "$DEST"/fft_cpu "$DEST"/fft_gpu \
          "$DEST"/axpy_cpu "$DEST"/axpy_gpu "$DEST"/stencil_cpu "$DEST"/stencil_gpu
echo
echo BUILD_DUAL_DONE
