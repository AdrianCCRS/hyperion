#!/bin/bash
# Reproduce el binario OpenMP/FP64 de LavaMD usado en paccaA100 sin
# modificar el algoritmo de Rodinia. La revisión y los hashes de fuente se
# fijan para que un cambio aguas arriba no altere silenciosamente el dataset.
set -euo pipefail

source_dir="${1:-/home/latorresn/rodinia-src}"
output_dir="${2:-/home/latorresn/hyperion-kernels/bin}"
cc="${3:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/gcc}"
expected_commit="9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4"
expected_main_h="14539f64b9c86793d72260eecd5b77312f755ae8925a2a15dc6dd3815ea23a1a"
expected_kernel_c="ff1a1e81b26fadc3566d16e6e73a8d2241066aecd2f001a638bc02379c2be881"
expected_binary_sha256="92cb259bd2ca1d0a67a4d52490b76cdb2eb844568c7e73864023c52d7abb2648"

actual_commit="$(git -C "$source_dir" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "Rodinia HEAD inesperado: $actual_commit (esperado $expected_commit)" >&2
  exit 1
fi

lava_dir="$source_dir/openmp/lavaMD"
actual_main_h="$(sha256sum "$lava_dir/main.h" | awk '{print $1}')"
actual_kernel_c="$(sha256sum "$lava_dir/kernel/kernel_cpu.c" | awk '{print $1}')"
if [[ "$actual_main_h" != "$expected_main_h" || "$actual_kernel_c" != "$expected_kernel_c" ]]; then
  echo "Los fuentes FP64/LavaMD no coinciden con los hashes auditados" >&2
  exit 1
fi

build_dir="$(mktemp -d -p /home/latorresn/yacacerest hyperion_lavamd_build_XXXXXX)"
cleanup() {
  rm -rf -- "$build_dir"
}
trap cleanup EXIT

cp -a "$lava_dir/." "$build_dir/"
mkdir -p "$output_dir"

"$cc" -O3 -march=native -fopenmp \
  "$build_dir/main.c" \
  "$build_dir/kernel/kernel_cpu.c" \
  "$build_dir/util/num/num.c" \
  "$build_dir/util/timer/timer.c" \
  -lm -o "$build_dir/rodinia_lavamd_omp"

actual_binary_sha256="$(sha256sum "$build_dir/rodinia_lavamd_omp" | awk '{print $1}')"
if [[ "$actual_binary_sha256" != "$expected_binary_sha256" ]]; then
  echo "Binario LavaMD no reproducible: $actual_binary_sha256" >&2
  exit 1
fi

install -m 0755 "$build_dir/rodinia_lavamd_omp" "$output_dir/rodinia_lavamd_omp"
sha256sum "$output_dir/rodinia_lavamd_omp"
