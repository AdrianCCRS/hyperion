#!/usr/bin/env python3
"""Punto de entrada del análisis de correlación/VIF y contrato de features
(F1-XDEV-004). No entrena; solo analiza el CSV intermedio final del
dispositivo y propone (opcionalmente congela) el conjunto de features."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fase2_clasificador.analysis.feature_contract import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
