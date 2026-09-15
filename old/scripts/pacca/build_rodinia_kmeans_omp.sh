#!/bin/bash
# Reproduce kmeans OpenMP (Rodinia) sin alterar el algoritmo.
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

check_hash "openmp/kmeans/kmeans_openmp/kmeans_clustering.c" "e2975b0575a33fb23d48602070e31152e0851e5fad341f358aea26f6ee1894a4" "$source_dir"
check_hash "openmp/kmeans/kmeans_openmp/kmeans.c" "86f4d0904446d0cf6a7d81a9386e975c79db408aaa9fc0368d4390e231f1b891" "$source_dir"
check_hash "openmp/kmeans/kmeans_openmp/cluster.c" "c2527677e1e499cacf13ae8f13ee68b15cb0af152cf07483b4ce05cc9ff7a7fa" "$source_dir"
check_hash "openmp/kmeans/kmeans_openmp/getopt.c" "eb2ca4eec4887bff022fec2722fa51a48e38138ec0219cd02c8badcf5198d5f0" "$source_dir"

build_dir="$(mktemp -d -t hyperion_kmeans_omp_build_XXXXXX)"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT

cc="${4:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/gcc}"
"$cc" -O3 -fopenmp -march=native \
  "$source_dir/openmp/kmeans/kmeans_openmp/kmeans_clustering.c" \
  "$source_dir/openmp/kmeans/kmeans_openmp/kmeans.c" \
  "$source_dir/openmp/kmeans/kmeans_openmp/cluster.c" \
  "$source_dir/openmp/kmeans/kmeans_openmp/getopt.c" \
  -lm -o "$build_dir/rodinia_kmeans_omp"

actual_binary_sha256="$(sha256sum "$build_dir/rodinia_kmeans_omp" | awk '{print $1}')"
expected_binary_sha256="PENDIENTE_VERIFICAR_EN_PACCA"
if [[ "$actual_binary_sha256" != "$expected_binary_sha256" ]]; then
  echo "Binario no reproducible: $actual_binary_sha256 (esperado $expected_binary_sha256)" >&2
  exit 1
fi

mkdir -p "$output_dir"
install -m 0755 "$build_dir/rodinia_kmeans_omp" "$output_dir/rodinia_kmeans_omp"
sha256sum "$output_dir/rodinia_kmeans_omp"
