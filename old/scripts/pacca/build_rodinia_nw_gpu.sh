#!/bin/bash
# Reproduce Needleman-Wunsch CUDA (Rodinia), PARCHEADO.
# Fuentes verificadas contra Rodinia @ 9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4 (rodinia-src). PARCHE F1-XDEV: mismo overflow de enteros de 32 bits que la version CPU (needle_gpu_patched.cu, versionado en patches/rodinia/), needle_kernel.cu y needle.h se usan sin modificar desde el checkout vendored (-I).
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
check_hash "cuda/nw/needle_kernel.cu" "5d1e9448b2ce90b7f0acbbde1289a95108e3480ea6772fba768a985a01914f0c" "$source_dir"
check_hash "cuda/nw/needle.h" "38ffaf31b59432e204dff8b22e10ada572c18eae27abfd0a82ea3f08b178b180" "$source_dir"
check_hash "needle_gpu_patched.cu" "a795f4473098065103a6f7ae854916e6f34fd8036fb11552798294715138e5fe" "$patches_dir"

build_dir="$(mktemp -d -t hyperion_nw_gpu_build_XXXXXX)"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT

"$nvcc" -O3 -arch=$arch \
  -I "$source_dir/cuda/nw" \
  "$patches_dir/needle_gpu_patched.cu" \
  -o "$build_dir/rodinia_nw"

# ARC-186: nvcc no produce binarios reproducibles byte a byte (metadatos de
# depuracion con rutas temporales aleatorias), aunque el codigo ejecutable
# es identico -- confirmado empiricamente aqui (dos builds desde la misma
# fuente/flags dieron sha256 distintos). Se fija hash de FUENTE (arriba),
# no de binario; el sha256 del binario se imprime solo como referencia.
actual_binary_sha256="$(sha256sum "$build_dir/rodinia_nw" | awk '{print $1}')"
echo "Binario compilado (nvcc, no reproducible byte a byte): $actual_binary_sha256"

mkdir -p "$output_dir"
install -m 0755 "$build_dir/rodinia_nw" "$output_dir/rodinia_nw"
sha256sum "$output_dir/rodinia_nw"
