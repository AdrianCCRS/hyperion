#!/bin/bash
# Adaptador reproducible para ejecutar únicamente Polybench_3MM/Base_OpenMP
# y convertir el checksum nativo de RAJAPerf en el success_check del catálogo.
set -euo pipefail

binary="/home/latorresn/hyperion-kernels/libexec/raja-perf-v2025.12.1"
expected_binary_sha256="7f5251ac4c8f4bfd854441b7873f120080affec7bc77abfce1cc0fb9ec165ebb"

actual_binary_sha256="$(sha256sum "$binary" | awk '{print $1}')"
if [[ "$actual_binary_sha256" != "$expected_binary_sha256" ]]; then
  echo "RAJAPerf binary checksum mismatch" >&2
  exit 1
fi

run_dir="$(mktemp -d -p /home/latorresn/yacacerest hyperion_rajaperf_3mm_run_XXXXXX)"
cleanup() {
  rm -rf -- "$run_dir"
}
trap cleanup EXIT

export OMP_NUM_THREADS=6
export OMP_PROC_BIND=true
export OMP_PLACES=cores

cd "$run_dir"
"$binary" --warmup-disable -k Polybench_3MM -v Base_OpenMP "$@"

if ! grep -Eq '^Base_OpenMP-default[[:space:]]+PASSED' RAJAPerf-checksum.txt; then
  echo "RAJAPerf Polybench_3MM checksum failed" >&2
  exit 1
fi

grep -E '^Base_OpenMP-default[[:space:]]+PASSED' RAJAPerf-checksum.txt
echo "Verification = SUCCESSFUL"
