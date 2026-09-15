#!/bin/bash
# Reproduce kmeans CUDA (Rodinia), PARCHEADO.
# Fuentes verificadas contra Rodinia @ 9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4 (rodinia-src). PARCHE F1-XDEV: API de textura (texture<>/cudaBindTexture/tex1Dfetch), cudaMemcpyToSymbol por string y cudaThreadSynchronize, todas retiradas en CUDA 12 -- reemplazadas por lectura directa de arreglo, simbolo directo y cudaDeviceSynchronize (kmeans_cuda_patched.cu/kmeans_cuda_kernel_patched.cu, versionados en patches/rodinia/). El resto de fuentes (C puro) sin cambios.
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
check_hash "cuda/kmeans/cluster.c" "28f01b1e4b0656ffa163b513889a29526047444595b415e4a31b889ccf032d10" "$source_dir"
check_hash "cuda/kmeans/getopt.c" "eb2ca4eec4887bff022fec2722fa51a48e38138ec0219cd02c8badcf5198d5f0" "$source_dir"
check_hash "cuda/kmeans/kmeans.c" "3544ffbe528072518729e41b7241ceb4a699919a8bbeda7ae5f0bcc176e45646" "$source_dir"
check_hash "cuda/kmeans/kmeans_clustering.c" "4ebdf3a00bc41ebfc340113da816e3aa74dd85a56145df8b4826c0d796660381" "$source_dir"
check_hash "cuda/kmeans/rmse.c" "339d808f51d66e83d54a30a1a51c38b99698c5cfe348010a08a9e88ada759995" "$source_dir"
check_hash "kmeans_cuda_patched.cu" "14a2d6da61983bb80010df6e50f1387480746a1565794ad5da6a968113d202aa" "$patches_dir"
check_hash "kmeans_cuda_kernel_patched.cu" "b6e376d8ec124c2b26614dd96e2b632015061d64af97b8b8fb93830b08983733" "$patches_dir"

build_dir="$(mktemp -d -t hyperion_kmeans_gpu_build_XXXXXX)"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT

cp "$source_dir/cuda/kmeans/cluster.c" "$build_dir/"
cp "$source_dir/cuda/kmeans/getopt.c" "$build_dir/"
cp "$source_dir/cuda/kmeans/kmeans.c" "$build_dir/"
cp "$source_dir/cuda/kmeans/kmeans_clustering.c" "$build_dir/"
cp "$source_dir/cuda/kmeans/rmse.c" "$build_dir/"
cp "$source_dir/cuda/kmeans/kmeans.h" "$build_dir/"
cp "$patches_dir/kmeans_cuda_patched.cu" "$build_dir/kmeans_cuda.cu"
cp "$patches_dir/kmeans_cuda_kernel_patched.cu" "$build_dir/kmeans_cuda_kernel.cu"
cd "$build_dir"

"$gcc" -O2 -c cluster.c getopt.c kmeans.c kmeans_clustering.c rmse.c
"$nvcc" -O2 -arch=$arch -c kmeans_cuda.cu -o kmeans_cuda.o
"$gcc" cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o \
  -o rodinia_kmeans \
  -L"$cuda_lib" -lcudart -lm
cd - > /dev/null

actual_binary_sha256="$(sha256sum "$build_dir/rodinia_kmeans" | awk '{print $1}')"
expected_binary_sha256="d1e877643e640e72a727139adee1cb8f1febe18a6a4d6a7defc87f31c04a9383"
if [[ "$actual_binary_sha256" != "$expected_binary_sha256" ]]; then
  echo "Binario no reproducible: $actual_binary_sha256 (esperado $expected_binary_sha256)" >&2
  exit 1
fi

mkdir -p "$output_dir"
install -m 0755 "$build_dir/rodinia_kmeans" "$output_dir/rodinia_kmeans"
sha256sum "$output_dir/rodinia_kmeans"
