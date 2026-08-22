#!/bin/bash
# Construye los kernels de fase de HYPERION-PHASE (ptrchase, phasic) en
# paccaA100. ARC-177.
#
# POR QUÉ EXISTE ESTE SCRIPT. Estos dos eran los únicos kernels del catálogo
# sin script de build, y se compilaron a mano con "-O2 -fopenmp". Eso dejó la
# fase de cómputo de phasic en mulsd/addsd ESCALAR sobre xmm -- ~1/16 del
# pico de la máquina. El pico de 480 GFLOP/s que fija el ridge del Roofline
# lo mide ert_probe en AVX-512, así que una fase "de cómputo" escalar no es
# comparable con las cargas reales del catálogo, ni en FLOPs alcanzados ni
# --lo que más importa para este trabajo-- en POTENCIA, que es de donde sale
# el umbral de viabilidad alpha <= 0.226.
#
# FLAGS. Los mismos que ert_probe (scripts/felix/build_stream_ert.sh):
# -mprefer-vector-width=512 NO es redundante con -march=native. ARC-125 lo
# encontró con Intel Advisor: -march=native por sí solo dejaba a GCC
# eligiendo AVX2 (ancho 4) aunque el hardware soporta AVX-512.
#
# El script verifica el ancho de vector REALMENTE emitido en vez de confiar
# en que el flag surtió efecto -- que es justo el error que lo originó.
set -euo pipefail

repo_dir="${1:-/home/latorresn/hyperion}"
output_dir="${2:-/home/latorresn/hyperion-kernels/bin}"
cc="${3:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/gcc}"

src_dir="$repo_dir/kernels/phase"
mkdir -p "$output_dir"

flags=(-O3 -march=native -mprefer-vector-width=512 -fopenmp)

# ptrchase no tiene aritmética de punto flotante: es una cadena de cargas
# dependientes. Se compila con los mismos flags por consistencia, no porque
# la vectorización pueda cambiar algo -- una dependencia serial no se
# vectoriza, y ese es exactamente el punto del kernel.
"$cc" "${flags[@]}" -o "$output_dir/ptrchase" "$src_dir/ptrchase.c" -lm
"$cc" "${flags[@]}" -o "$output_dir/phasic"   "$src_dir/phasic.c"   -lm

# VERIFICACIÓN DE ANCHO. Sin esto el script repetiría el fallo original en
# silencio: compilar "con los flags correctos" y producir código escalar.
fma_widths="$(objdump -d "$output_dir/phasic" --no-show-raw-insn \
  | grep -oE 'vfmadd[0-9]*pd[^,]*,%(zmm|ymm|xmm)' \
  | grep -oE 'zmm|ymm|xmm' | sort -u | tr '\n' ' ')"
if [[ "$fma_widths" != *zmm* ]]; then
  echo "phasic: la fase de computo NO emitio FMA empacado de 512 bits" >&2
  echo "  anchos encontrados: '${fma_widths:-ninguno}'" >&2
  echo "  revisar soporte de AVX-512 del nodo y los flags de arriba" >&2
  exit 1
fi

echo "OK  anchos de FMA empacado en phasic: $fma_widths"
sha256sum "$output_dir/ptrchase" "$output_dir/phasic"
