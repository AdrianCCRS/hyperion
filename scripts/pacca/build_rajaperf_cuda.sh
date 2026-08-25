#!/bin/bash
# Construye RAJAPerf v2025.12.1 con la variante CUDA (Base_CUDA), la misma
# fuente ya fijada por commit que la variante OpenMP
# (build_rajaperf_polybench_3mm_omp.sh) -- ahora con ENABLE_CUDA=On en vez
# de Off. Es el "impulso" de kernels de GPU que quedó pendiente en
# Estrategia_GPU_Fase2.md (riesgo 6): sin esto, el catálogo GPU sigue
# limitado a las 7 familias de siempre, sin acceso a los otros ~78 kernels
# de RAJAPerf.
#
# nvcc no está en módulo ni en PATH por defecto (mismo hallazgo que
# build_gpu_phase_kernels.sh, ARC-186): toolkit real en
# /home/latorresn/latorresn/cuda-12.3, referenciado por ruta absoluta.
# -arch=sm_80 fija la capacidad real de la A100 (nvidia-smi
# --query-gpu=compute_cap -> 8.0).
#
# REPRODUCIBILIDAD, lección de ARC-186/193 aplicada aquí: nvcc NO produce
# binarios idénticos byte a byte entre builds (metadatos de depuración con
# nombres de archivo temporal aleatorios) -- el ejecutable en sí sí es
# igual tras strip. A diferencia de build_rajaperf_polybench_3mm_omp.sh
# (compilador g++ puro, reproducible sin strip), este script SIEMPRE
# despoja el binario antes de calcular su checksum, o cualquier
# verificación C02 futura fallaría en falso pese a un build legítimo.
set -euo pipefail

source_dir="${1:-/home/latorresn/hyperion-kernels/src/RAJAPerf-v2025.12.1}"
cmake_bin="${2:-cmake}"
output_root="${3:-/home/latorresn/hyperion-kernels}"
cuda_root="${4:-/home/latorresn/latorresn/cuda-12.3}"
expected_commit="e3c6197dfa8f1c9ac61635c26775c333411bdcd5"
expected_raja_commit="eca7c5015a5cf8bf7cc8ad1829fd36d3276ab274"

nvcc="$cuda_root/bin/nvcc"
if [[ ! -x "$nvcc" ]]; then
  echo "build_rajaperf_cuda: no se encontro nvcc en $nvcc" >&2
  exit 1
fi

if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "RAJAPerf no está en v2025.12.1/$expected_commit" >&2
  exit 1
fi
if [[ "$(git -C "$source_dir/tpl/RAJA" rev-parse HEAD)" != "$expected_raja_commit" ]]; then
  echo "El submódulo RAJA no coincide con la revisión fijada" >&2
  exit 1
fi

build_dir="$(mktemp -d -p /home/latorresn/yacacerest hyperion_rajaperf_cuda_build_XXXXXX)"
cleanup() {
  rm -rf -- "$build_dir"
}
trap cleanup EXIT

"$cmake_bin" -S "$source_dir" -B "$build_dir" \
  -DCMAKE_CXX_COMPILER=/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++ \
  -DCMAKE_CUDA_COMPILER="$nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCUDA_TOOLKIT_ROOT_DIR="$cuda_root" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENMP=Off \
  -DENABLE_CUDA=On \
  -DENABLE_HIP=Off \
  -DBLT_CXX_STD=c++17
"$cmake_bin" --build "$build_dir" --target raja-perf.exe -j 8

strip --strip-all "$build_dir/bin/raja-perf.exe"
actual_binary_sha256="$(sha256sum "$build_dir/bin/raja-perf.exe" | awk '{print $1}')"

mkdir -p "$output_root/libexec"
install -m 0755 "$build_dir/bin/raja-perf.exe" "$output_root/libexec/raja-perf-cuda-v2025.12.1"

echo "binario instalado en: $output_root/libexec/raja-perf-cuda-v2025.12.1"
echo "sha256 (tras strip): $actual_binary_sha256"
sha256sum "$output_root/libexec/raja-perf-cuda-v2025.12.1"
