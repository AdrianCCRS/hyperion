#!/bin/bash
# Pre-vuelo del selector CPU/GPU: mide tiempo por (operacion, device, tamaño)
# para (a) calibrar cuantas iteraciones necesita cada tamaño en la campaña y
# (b) COMPROBAR QUE LA FRONTERA CPU/GPU EXISTE.
#
# (b) es lo que decide si el proyecto tiene algo que aprender. Si la GPU gana
# en todos los tamaños, el "selector" degenera en una constante y no hay
# frontera que modelar. Barremos hasta N=64 justamente porque la frontera de
# GEMM, al ser O(N^3) de computo contra O(N^2) de transferencia, deberia caer
# en tamaños pequeños.
#
# Sin telemetria ni DVFS a proposito: aqui solo interesan tiempos relativos.
# ARC-126: sin `-u`.
set -e -o pipefail

DUAL="${DUAL:-$HOME/hyperion-kernels/libexec/dual}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:/opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10/lib:${LD_LIBRARY_PATH:-}"

# N:iteraciones -- iteraciones elegidas para que el lado CPU dure ~1-10 s.
GEMM_GRID="64:20000 128:10000 256:3000 512:500 1024:100 2048:20 4096:5"
FFT_GRID="64:20000 128:10000 256:5000 512:1000 1024:300 2048:60 4096:15"

run_one() {
    local bin="$1" size="$2" iters="$3"
    local out
    out=$("$bin" --size "$size" --iterations "$iters" 2>&1) || { echo "FAIL"; return; }
    if ! grep -q "Verification    =               SUCCESSFUL" <<<"$out"; then
        echo "BADVERIFY"; return
    fi
    grep "Time in seconds" <<<"$out" | awk '{print $NF}'
}

printf "%-6s %-8s %-8s %12s %12s %10s\n" "op" "N" "iters" "t_cpu(s)" "t_gpu(s)" "gana"
for spec in $GEMM_GRID; do
    n="${spec%%:*}"; it="${spec##*:}"
    tc=$(run_one "$DUAL/gemm_cpu" "$n" "$it")
    tg=$(run_one "$DUAL/gemm_gpu" "$n" "$it")
    win=$(awk -v a="$tc" -v b="$tg" 'BEGIN{ if(a+0==0||b+0==0){print "?"} else print (a<b)?"CPU":"GPU" }')
    printf "%-6s %-8s %-8s %12s %12s %10s\n" gemm "$n" "$it" "$tc" "$tg" "$win"
done
for spec in $FFT_GRID; do
    n="${spec%%:*}"; it="${spec##*:}"
    tc=$(run_one "$DUAL/fft_cpu" "$n" "$it")
    tg=$(run_one "$DUAL/fft_gpu" "$n" "$it")
    win=$(awk -v a="$tc" -v b="$tg" 'BEGIN{ if(a+0==0||b+0==0){print "?"} else print (a<b)?"CPU":"GPU" }')
    printf "%-6s %-8s %-8s %12s %12s %10s\n" fft "$n" "$it" "$tc" "$tg" "$win"
done
echo SCREEN_DUAL_DONE
