#!/usr/bin/env python3
"""Punto de entrada de Fase 2 (Objetivo 2): entrenamiento y validación del
clasificador de fase compute_bound/memory_bound.

Envoltura ejecutable de `fase2_clasificador/training/train_phase.py`.
No duplica el parseo de argumentos -- ver `train_phase.main()` para el
`argparse` completo (`--campaign-dir`, `--campaign-id`, `--kernels`,
`--levels`, `--seed`, `--latency-weight`, `--output-dir`, etc.). Este
script solo resuelve `sys.path` para que `fase2_clasificador` y `common`
sean importables sin importar desde qué directorio se invoque.

A diferencia de `fase1_telemetria/run_campaign.py`, este script SÍ puede
invocarse desde cualquier directorio: `--campaign-dir` recibe una ruta
(absoluta o relativa al cwd) al directorio de campaña de Fase 1 que
contiene los `training_cpu_intervals.csv` de origen -- no hay ninguna convención de cwd
implícita como la de `~/hyperion-kernels` en Fase 1.

    python3 fase2_clasificador/run_training.py \
        --campaign-dir ~/hyperion-results/campaigns/mi_campana_20260901 \
        --campaign-id mi_campana_20260901 \
        --output-dir fase2_clasificador/models/

Sin `--output-dir`, corre en modo exploración: compara los modelos e
imprime las tablas, pero no serializa nada. Ver `fase2_clasificador/README.md`
para el significado de cada columna impresa y el formato del modelo/
metadata serializados.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fase2_clasificador.training.train_phase import main  # noqa: E402


if __name__ == "__main__":
    main()
