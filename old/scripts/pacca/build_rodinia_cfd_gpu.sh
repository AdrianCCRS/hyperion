#!/bin/bash
# Reproduce CFD Solver (euler3d_double, FP64) CUDA (Rodinia), sin alterar el algoritmo.
# Fuentes verificadas contra Rodinia @ 9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4 (rodinia-src). Sin parches al algoritmo: helper_cuda.h/helper_timer.h son shims propios (no forman parte del toolkit CUDA desde hace anios; euler3d_double.cu solo usa checkCudaErrors() y StopWatchInterface/sdk*Timer*, reimplementados minimalmente, versionados en patches/rodinia/).
# Nunca muta el checkout vendored: copia/compila desde un directorio aislado
# (mktemp -d), igual que build_rodinia_lavamd_omp.sh. Solo compila (nvcc
# funciona sin GPU en el nodo); ejecutar/perfilar el binario resultante
# requiere paccaA100.
set -euo pipefail

source_dir="${1:-/home/latorresn/rodinia-src}"
output_dir="${2:-/home/latorresn/hyperion-kernels/bin}"
patches_dir="${3:-/home/latorresn/hyperion/old/scripts/pacca/patches/rodinia}"
nvcc="${4:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
gcc="${5:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/gcc}"
cuda_lib="${6:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/lib64}"
arch="sm_80"
expected_commit="9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4"

actual_commit="$(git -C "$source_dir" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "Rodinia HEAD inesperado: $actual_commit (esperado $expected_commit)" >&2
  exit 1
fi

check_hash() {
  local rel="$1" expected="$2" base="$3"
  local actual
  actual="$(sha256sum "$base/$rel" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "Hash inesperado para $rel: $actual (esperado $expected)" >&2
    exit 1
  fi
}
check_hash "cuda/cfd/euler3d_double.cu" "bb45196fc6afab0ed0dde281413433de59a0be719d5d94c6bffdd122cdfc6626" "$source_dir"
check_hash "helper_cuda.h" "9afa9c7809293d4d16b5d544492d4e22c0fde5dc2fedae59a67d2f0e77c85573" "$patches_dir"
check_hash "helper_timer.h" "98c4d01d9f9891c8ef212c7cdc223e33a4304348de1f5ce181f4c9c26ebeaf4e" "$patches_dir"

build_dir="$(mktemp -d -t hyperion_cfd_gpu_build_XXXXXX)"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT

"$nvcc" -O3 -arch=$arch \
  -I "$patches_dir" \
  "$source_dir/cuda/cfd/euler3d_double.cu" \
  -o "$build_dir/rodinia_cfd"

# ARC-186: nvcc no produce binarios reproducibles byte a byte (metadatos de
# depuracion con rutas temporales aleatorias), aunque el codigo ejecutable
# es identico -- confirmado empiricamente aqui (dos builds desde la misma
# fuente/flags dieron sha256 distintos). Se fija hash de FUENTE (arriba),
# no de binario; el sha256 del binario se imprime solo como referencia.
actual_binary_sha256="$(sha256sum "$build_dir/rodinia_cfd" | awk '{print $1}')"
echo "Binario compilado (nvcc, no reproducible byte a byte): $actual_binary_sha256"

mkdir -p "$output_dir"
install -m 0755 "$build_dir/rodinia_cfd" "$output_dir/rodinia_cfd"
sha256sum "$output_dir/rodinia_cfd"
