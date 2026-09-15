#!/bin/bash
# Reproduce SRAD v2 OpenMP (Rodinia) sin alterar el algoritmo.
# Fuentes verificadas contra Rodinia @ 9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4 (rodinia-src). Sin parches, fuente original.
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
check_hash "openmp/srad/srad_v2/srad.cpp" "bd88fbb6ce9d37349cf89abc1dd2cb55412d1464ff34b1a2269fbe6a15831c7e" "$source_dir"

build_dir="$(mktemp -d -t hyperion_srad_omp_build_XXXXXX)"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT

"$cc" -O3 -fopenmp -march=native \
  "$source_dir/openmp/srad/srad_v2/srad.cpp" \
  -lm -o "$build_dir/rodinia_srad_omp"

actual_binary_sha256="$(sha256sum "$build_dir/rodinia_srad_omp" | awk '{print $1}')"
expected_binary_sha256="1d4bc31497befa18859336e64d00b7ee6f26727a78ef3b265c5aa84711aee0b3"
if [[ "$actual_binary_sha256" != "$expected_binary_sha256" ]]; then
  echo "Binario no reproducible: $actual_binary_sha256 (esperado $expected_binary_sha256)" >&2
  exit 1
fi

mkdir -p "$output_dir"
install -m 0755 "$build_dir/rodinia_srad_omp" "$output_dir/rodinia_srad_omp"
sha256sum "$output_dir/rodinia_srad_omp"
