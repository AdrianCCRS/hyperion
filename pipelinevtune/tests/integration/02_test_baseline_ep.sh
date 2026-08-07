#!/usr/bin/env bash
# Valida PLAN.md Fase 3.1 sobre un solo kernel corto. Sin VTune todavia,
# solo confirma que el binario en si mismo es una base valida para medir.
set -uo pipefail
BIN="${1:-./bin/ep.C.x}"

[ -x "$BIN" ] || { echo "FALLO: $BIN no existe o no es ejecutable" >&2; exit 1; }

OUT=$(mktemp)
ERR=$(mktemp)
timeout 300 "$BIN" > "$OUT" 2> "$ERR"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  echo "FALLO: $BIN termino con codigo $STATUS" >&2
  cat "$ERR" >&2
  exit 1
fi

grep -q "VERIFICATION SUCCESSFUL" "$OUT" \
  || { echo "FALLO: no se encontro VERIFICATION SUCCESSFUL en la salida" >&2; exit 1; }

echo "PASO: baseline de $BIN valido (exit 0, VERIFICATION SUCCESSFUL)."
exit 0
