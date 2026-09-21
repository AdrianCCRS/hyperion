#!/usr/bin/env bash
set -euo pipefail

# El módulo cmake/4.3.4 de paccaA100 carga un binario que no encuentra
# libjsoncpp.so.19. Este build directo mantiene el tracer reproducible sin
# depender de ese módulo; el launcher usa su build CMake ya configurado.
repo_root="${HYPERION_ROOT:-$HOME/hyperion}"
cuda_root="${HYPERION_CUDA_ROOT:-/home/latorresn/latorresn/cuda-12.3}"
telemetry_root="$repo_root/common/telemetry"
output_dir="$telemetry_root/build-cupti"

test -f "$cuda_root/include/cuda.h"
test -f "$cuda_root/extras/CUPTI/include/cupti.h"
test -e "$cuda_root/extras/CUPTI/lib64/libcupti.so"
mkdir -p "$output_dir"

/usr/bin/g++ \
  -std=c++17 -O2 -Wall -Wextra -fPIC -shared \
  "$telemetry_root/src/cupti_activity_preload.cpp" \
  -I"$cuda_root/include" \
  -I"$cuda_root/extras/CUPTI/include" \
  -L"$cuda_root/extras/CUPTI/lib64" \
  -Wl,-rpath,"$cuda_root/extras/CUPTI/lib64" \
  -lcupti -pthread \
  -o "$output_dir/libhyperion_cupti_activity.so"

make -C "$telemetry_root/build" -j2 telemetry_kernel_launcher

echo "$output_dir/libhyperion_cupti_activity.so"
