#!/usr/bin/env bash
# Valida PLAN.md Fase 1.2 / 4.3: los kernels ancla corren y reportan sus propios
# numeros por software (sin depender de contadores de VTune / uncore).
set -uo pipefail
ANCHOR_DIR="${1:-./anchor_bin}"

STREAM_BIN="$ANCHOR_DIR/stream_omp"
DGEMM_BIN="$ANCHOR_DIR/dgemm_bench"

[ -x "$STREAM_BIN" ] || { echo "FALLO: $STREAM_BIN no encontrado/ejecutable" >&2; exit 1; }
[ -x "$DGEMM_BIN" ]  || { echo "FALLO: $DGEMM_BIN no encontrado/ejecutable, o usar bt.C.x/sp.C.x como sustituto y ajustar este test" >&2; exit 1; }

STREAM_OUT=$("$STREAM_BIN")
echo "$STREAM_OUT" | grep -qE "Triad:\s*[0-9]" \
  || { echo "FALLO: STREAM no imprimio una linea Triad con numero reconocible" >&2; exit 1; }

DGEMM_OUT=$("$DGEMM_BIN" 1024 3)
echo "$DGEMM_OUT" | grep -qE "GFLOP/s=[0-9]" \
  || { echo "FALLO: DGEMM no imprimio GFLOP/s reconocible" >&2; exit 1; }

echo "PASO: ambos anclas corren y reportan sus propios numeros por software."
echo "STREAM: $(echo "$STREAM_OUT" | grep Triad)"
echo "DGEMM:  $(echo "$DGEMM_OUT")"
exit 0
