#!/usr/bin/env bash
# Valida PLAN.md Fase 2. Requiere check_vtune.py ya construido.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${1:-$SCRIPT_DIR/../..}"

python3 "$PIPELINE_DIR/check_vtune.py"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  echo "FALLO: check_vtune.py termino con codigo $STATUS (esperado 0)" >&2
  exit 1
fi

echo "PASO: preflight en verde."
exit 0
