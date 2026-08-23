#!/bin/bash
# Construye gpu_phasic (kernels/phase/gpu_phasic.cu) en paccaA100. ARC-186.
#
# nvcc no está en ningún module ni en el PATH por defecto de este nodo
# (confirmado 2026-08-22: "module avail cuda" no encuentra nada, "which
# nvcc" tampoco). El toolkit SÍ está instalado, en una ruta de usuario:
# /home/latorresn/latorresn/cuda-12.3. Se referencia por ruta absoluta en
# vez de depender de un module que no existe.
#
# -arch=sm_80 fija la capacidad de cómputo real de la A100 (confirmado con
# `nvidia-smi --query-gpu=compute_cap` -> 8.0), en vez de dejar que nvcc
# eligiera un valor por defecto que podría no coincidir con el hardware.
set -euo pipefail

repo_dir="${1:-/home/latorresn/hyperion}"
output_dir="${2:-/home/latorresn/hyperion-kernels/bin}"
cuda_root="${3:-/home/latorresn/latorresn/cuda-12.3}"

nvcc="$cuda_root/bin/nvcc"
if [[ ! -x "$nvcc" ]]; then
  echo "build_gpu_phase_kernels: no se encontró nvcc en $nvcc" >&2
  exit 1
fi

mkdir -p "$output_dir"

"$nvcc" -O3 -arch=sm_80 -o "$output_dir/gpu_phasic" "$repo_dir/kernels/phase/gpu_phasic.cu"

echo "OK  gpu_phasic construido"
sha256sum "$output_dir/gpu_phasic"
