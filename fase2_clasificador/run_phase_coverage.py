#!/usr/bin/env python3
"""Punto de entrada del diagnóstico de cobertura F2-XDEV-001."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fase2_clasificador.analysis.phase_coverage import main  # noqa: E402


if __name__ == "__main__":
    main()
