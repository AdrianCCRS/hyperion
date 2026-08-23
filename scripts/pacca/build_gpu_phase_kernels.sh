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

# ARC-186: nvcc NO produce binarios reproducibles byte a byte -- comprobado
# en vivo, dos builds consecutivos de la MISMA fuente con los MISMOS flags
# en el MISMO nodo dan sha256 distintos (diferían desde el byte 897688).
# Es su pipeline de compilacion en varias etapas (cudafe/ptxas) el que
# incrusta nombres de archivo temporal aleatorios en metadatos de depuracion
# -- el codigo ejecutable en si es identico: tras "strip --strip-all" las
# dos build dieron el MISMO sha256. binary_checksum en el catalogo (C02)
# compara el archivo que de verdad se ejecuta, asi que hay que despojar el
# binario ANTES de calcular el checksum, no solo antes de publicarlo --
# de lo contrario el checksum del catalogo nunca coincidiria con una
# reconstruccion legitima futura.
strip --strip-all "$output_dir/gpu_phasic"

echo "OK  gpu_phasic construido y despojado (checksum reproducible)"
sha256sum "$output_dir/gpu_phasic"
