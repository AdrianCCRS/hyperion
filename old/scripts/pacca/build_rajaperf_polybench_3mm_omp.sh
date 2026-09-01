#!/bin/bash
# Reproduce RAJAPerf v2025.12.1 con OpenMP/FP64 para paccaA100. No altera
# el kernel: selecciona Base_OpenMP al ejecutar mediante el adaptador vecino.
set -euo pipefail

source_dir="${1:-/home/latorresn/hyperion-kernels/src/RAJAPerf-v2025.12.1}"
cmake_bin="${2:-cmake}"
output_root="${3:-/home/latorresn/hyperion-kernels}"
expected_commit="e3c6197dfa8f1c9ac61635c26775c333411bdcd5"
expected_raja_commit="eca7c5015a5cf8bf7cc8ad1829fd36d3276ab274"
expected_binary_sha256="7f5251ac4c8f4bfd854441b7873f120080affec7bc77abfce1cc0fb9ec165ebb"

if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "RAJAPerf no está en v2025.12.1/$expected_commit" >&2
  exit 1
fi
if [[ "$(git -C "$source_dir/tpl/RAJA" rev-parse HEAD)" != "$expected_raja_commit" ]]; then
  echo "El submódulo RAJA no coincide con la revisión fijada" >&2
  exit 1
fi

build_dir="$(mktemp -d -p /home/latorresn/yacacerest hyperion_rajaperf_build_XXXXXX)"
cleanup() {
  rm -rf -- "$build_dir"
}
trap cleanup EXIT

"$cmake_bin" -S "$source_dir" -B "$build_dir" \
  -DCMAKE_CXX_COMPILER=/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=Off \
  -DENABLE_HIP=Off \
  -DBLT_CXX_STD=c++17
"$cmake_bin" --build "$build_dir" --target raja-perf.exe -j 8

actual_binary_sha256="$(sha256sum "$build_dir/bin/raja-perf.exe" | awk '{print $1}')"
if [[ "$actual_binary_sha256" != "$expected_binary_sha256" ]]; then
  echo "Binario RAJAPerf no reproducible: $actual_binary_sha256" >&2
  exit 1
fi

mkdir -p "$output_root/libexec" "$output_root/bin"
install -m 0755 "$build_dir/bin/raja-perf.exe" "$output_root/libexec/raja-perf-v2025.12.1"
install -m 0755 "$(dirname "$0")/rajaperf_polybench_3mm_omp.sh" \
  "$output_root/bin/rajaperf_polybench_3mm_omp"

sha256sum "$output_root/libexec/raja-perf-v2025.12.1" \
  "$output_root/bin/rajaperf_polybench_3mm_omp"
