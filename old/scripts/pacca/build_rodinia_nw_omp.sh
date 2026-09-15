#!/bin/bash
# Reproduce Needleman-Wunsch OpenMP (Rodinia), PARCHEADO.
# Fuentes verificadas contra Rodinia @ 9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4 (rodinia-src). PARCHE F1-XDEV: overflow de enteros de 32 bits en malloc(max_rows*max_cols*sizeof(int)) para N>~46000 (max_rows/max_cols retipados a long, mallocs con cast (size_t)). Fuente original verificada primero; el binario se compila desde needle_cpu_patched.cpp (versionado en patches/rodinia/), no desde el checkout vendored.
# Nunca muta el checkout vendored: copia a un directorio aislado (mktemp -d),
# igual que build_rodinia_lavamd_omp.sh.
set -euo pipefail

source_dir="${1:-/home/latorresn/rodinia-src}"
output_dir="${2:-/home/latorresn/hyperion-kernels/bin}"
patches_dir="${3:-/home/latorresn/hyperion/old/scripts/pacca/patches/rodinia}"
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

cc="${4:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/gcc}"
check_hash "openmp/nw/needle.cpp" "9e7237dd24a8390a6829fe57cb6f8ab260cd9f6dbc1fecf5a22520f93728323d" "$source_dir"
check_hash "needle_cpu_patched.cpp" "ca8a53ac91bdeeac591d38a9f6bf289d4422343235c7f5b949360cb947dc8366" "$patches_dir"

build_dir="$(mktemp -d -t hyperion_nw_omp_build_XXXXXX)"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT

"$cc" -O3 -fopenmp -march=native \
  "$patches_dir/needle_cpu_patched.cpp" \
  -lm -o "$build_dir/rodinia_nw_omp"

actual_binary_sha256="$(sha256sum "$build_dir/rodinia_nw_omp" | awk '{print $1}')"
expected_binary_sha256="c43c8c2dd45319d8fde3603cd4bbebad2d4eb9e3569118869c4605916411ea9c"
if [[ "$actual_binary_sha256" != "$expected_binary_sha256" ]]; then
  echo "Binario no reproducible: $actual_binary_sha256 (esperado $expected_binary_sha256)" >&2
  exit 1
fi

mkdir -p "$output_dir"
install -m 0755 "$build_dir/rodinia_nw_omp" "$output_dir/rodinia_nw_omp"
sha256sum "$output_dir/rodinia_nw_omp"
