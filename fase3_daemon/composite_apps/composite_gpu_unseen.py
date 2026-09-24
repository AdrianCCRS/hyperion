#!/usr/bin/env python3
"""Aplicación compuesta B de GPU -- familias INÉDITAS (Fase 3, ítem C2b, contraparte de GPU de `composite_unseen.py`).

Familias Rodinia que NO están entre las 16 del entrenamiento del clasificador de GPU y sí declaran `phase_label_hint`:
  - rodinia_dwt2d    memory_bound,  ~0.8 s por lanzamiento
  - rodinia_lavamd   compute_bound, ~3.0 s por lanzamiento
(medidas en paccaA100, job 7627). Cada lanzamiento dura menos que la ventana de decisión del daemon (3 s de actividad
sostenida), así que cada fase se arma REPITIENDO el mismo kernel varias veces seguidas: la actividad de GPU es continua
salvo el hueco de arranque de cada proceso. Cada lanzamiento queda registrado con su frontera real (reloj monotónico);
para puntuar, las repeticiones consecutivas con el mismo hint forman una sola fase.

Uso (en pacca, con el daemon de GPU aparte apuntando a este proceso con --target-pid):
  python3 composite_gpu_unseen.py --node-id pacca-a100 --kernels-root ~/hyperion-kernels --boundaries-out fronteras.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fase3_daemon.composite_apps.composite_known import main_with_defaults  # noqa: E402

MEMORY_KERNEL, MEMORY_REPEATS = "rodinia_dwt2d", 10      # ~8 s de fase memory
COMPUTE_KERNEL, COMPUTE_REPEATS = "rodinia_lavamd", 4    # ~12 s de fase compute
DEFAULT_SEQUENCE = (MEMORY_KERNEL,) * MEMORY_REPEATS + (COMPUTE_KERNEL,) * COMPUTE_REPEATS


def main() -> int:
    return main_with_defaults(
        default_sequence=DEFAULT_SEQUENCE, description=__doc__, default_cycles=2, label="aplicación compuesta B de GPU",
    )


if __name__ == "__main__":
    raise SystemExit(main())
