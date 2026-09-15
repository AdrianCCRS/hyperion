#!/bin/bash
# Reproduce B+Tree CUDA (Rodinia) sin alterar el algoritmo.
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
check_hash "cuda/b+tree/main.c" "f461ed1696a757de44b4d3453b5699e5cb8cc0b71ff06be8a650ce44f26ef918" "$source_dir"
check_hash "cuda/b+tree/kernel/kernel_gpu_cuda_wrapper.cu" "2f0715c4bf8f2bfad9ba143e0fa550b4c5d0b7fa7256a88bf665806bd453193b" "$source_dir"
check_hash "cuda/b+tree/kernel/kernel_gpu_cuda_wrapper_2.cu" "4b6706e7071a5ce5faaf9f4657e72cdbfe015093ac78574d5b787daaacc4e572" "$source_dir"
check_hash "cuda/b+tree/util/cuda/cuda.cu" "b07ee56dcf78ba69f874db067c0911f8ddf99437d33196295d43e4b85a452ea5" "$source_dir"
check_hash "cuda/b+tree/util/timer/timer.c" "e437daf303ba449ff7f097ef4dbfb1bdb2248e4347baecfc7e063f017a252620" "$source_dir"
check_hash "cuda/b+tree/util/num/num.c" "5cabf5d35263b1d059d5b586bb995c02327e24d8f9ab4ae9eb45151badaf5c45" "$source_dir"

build_dir="$(mktemp -d -t hyperion_btree_gpu_build_XXXXXX)"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT

mkdir -p "$build_dir/kernel" "$build_dir/util/cuda" "$build_dir/util/timer" "$build_dir/util/num"
cp -a "$source_dir/cuda/b+tree/." "$build_dir/"

cd "$build_dir"
"$nvcc" -O3 -arch=$arch -c kernel/kernel_gpu_cuda_wrapper.cu -o kernel/kernel_gpu_cuda_wrapper.o
"$nvcc" -O3 -arch=$arch -c kernel/kernel_gpu_cuda_wrapper_2.cu -o kernel/kernel_gpu_cuda_wrapper_2.o
"$nvcc" -O3 -arch=$arch -c util/cuda/cuda.cu -o util/cuda/cuda.o
"$gcc" -O3 -c main.c -o main.o
"$gcc" -O3 -c util/timer/timer.c -o util/timer/timer.o
"$gcc" -O3 -c util/num/num.c -o util/num/num.o
"$gcc" main.o kernel/kernel_gpu_cuda_wrapper.o kernel/kernel_gpu_cuda_wrapper_2.o \
  util/timer/timer.o util/num/num.o util/cuda/cuda.o \
  -o rodinia_btree \
  -L"$cuda_lib" -lcudart -lm
cd - > /dev/null

actual_binary_sha256="$(sha256sum "$build_dir/rodinia_btree" | awk '{print $1}')"
expected_binary_sha256="PENDIENTE_VERIFICAR_EN_PACCA"
if [[ "$actual_binary_sha256" != "$expected_binary_sha256" ]]; then
  echo "Binario no reproducible: $actual_binary_sha256 (esperado $expected_binary_sha256)" >&2
  exit 1
fi

mkdir -p "$output_dir"
install -m 0755 "$build_dir/rodinia_btree" "$output_dir/rodinia_btree"
sha256sum "$output_dir/rodinia_btree"
