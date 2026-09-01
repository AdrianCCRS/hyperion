#!/usr/bin/env bash
# Valida PLAN.md Fase 3 a 6 de punta a punta sobre un solo kernel corto.
# Requiere run_vtune_pipeline.py, vtune_parser.py y classifier.py ya construidos.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${1:-$SCRIPT_DIR/../..}"
OUT_DIR=$(mktemp -d)

python3 "$PIPELINE_DIR/run_vtune_pipeline.py" \
    --bin-dir ./bin \
    --anchor-dir ./anchor_bin \
    --output-dir "$OUT_DIR" \
    --kernels ep.C.x \
    --threads 8 \
    --repetitions 1

STATUS=$?
if [ "$STATUS" -ne 0 ]; then
  echo "FALLO: run_vtune_pipeline.py termino con codigo $STATUS" >&2
  exit 1
fi

CSV="$OUT_DIR/consolidated_results.csv"
[ -f "$CSV" ] || { echo "FALLO: no se genero $CSV" >&2; exit 1; }

HEADER=$(head -1 "$CSV")
echo "$HEADER" | grep -q "classification_vtune_native" \
  || { echo "FALLO: falta la columna classification_vtune_native en el CSV" >&2; exit 1; }

ROW=$(sed -n '2p' "$CSV")
[ -n "$ROW" ] || { echo "FALLO: el CSV no tiene ninguna fila de datos" >&2; exit 1; }

echo "PASO: pipeline extremo a extremo genero una fila valida para ep.C.x."
echo "Resultados en: $OUT_DIR"
exit 0
