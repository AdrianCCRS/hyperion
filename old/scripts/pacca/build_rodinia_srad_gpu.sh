#!/bin/bash
# Reproduce SRAD v2 CUDA (Rodinia) sin alterar el algoritmo.
# Fuentes verificadas contra Rodinia @ 9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4 (rodinia-src). Sin parches, fuente original.
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
check_hash "cuda/srad/srad_v2/srad.cu" "7ee19dc1d5d942cee9c18b998b5b2efe5bf85f9fb699fb5e60f4e87604a360d6" "$source_dir"
check_hash "cuda/srad/srad_v2/srad.h" "809653a83d45dbf0f297d8ea288e0446828fb337d36d361edd935838a7995ac9" "$source_dir"
check_hash "cuda/srad/srad_v2/srad_kernel.cu" "4f1551adc54d4df957d7f61668d357d1a28988fadd31b613ff5ca7535268baa0" "$source_dir"

build_dir="$(mktemp -d -t hyperion_srad_gpu_build_XXXXXX)"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT

"$nvcc" -O3 -arch=$arch \
  "$source_dir/cuda/srad/srad_v2/srad.cu" \
  -o "$build_dir/rodinia_srad"

actual_binary_sha256="$(sha256sum "$build_dir/rodinia_srad" | awk '{print $1}')"
expected_binary_sha256="193d8847fd74f536d394ace90602d5705b3932c8f822667bebb6a65dc4ffcf8c"
if [[ "$actual_binary_sha256" != "$expected_binary_sha256" ]]; then
  echo "Binario no reproducible: $actual_binary_sha256 (esperado $expected_binary_sha256)" >&2
  exit 1
fi

mkdir -p "$output_dir"
install -m 0755 "$build_dir/rodinia_srad" "$output_dir/rodinia_srad"
sha256sum "$output_dir/rodinia_srad"
