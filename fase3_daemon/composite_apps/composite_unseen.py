#!/usr/bin/env python3
"""Aplicación compuesta B -- familias INÉDITAS (Bloque C, ítem C2b,
Plan_Fase3_Daemon.md §0.1, Eje 1/requisito 3).

Contraparte deliberada de la Aplicación A (`composite_known.py`): kernels
que NUNCA entraron a ninguna decisión de entrenamiento/ajuste del
clasificador de Fase 2. Es el equivalente, a nivel de aplicación, del
protocolo *leave-one-familia-out* con el que se validó el modelo -- si A
gana y B no, el resultado es memorización, no generalización. También es
el instrumento con el que se cierran las dos capas de generalización sin
validar documentadas en Bloque A (§0.1, "Límite conocido de A4"): la
exactitud del clasificador con el buffer móvil real (no la agregación
offline de entrenamiento), y si la ganancia de F1 en `memory_bound`
sobrevive fuera del catálogo de política.

Secuencia por defecto -- `cpu_xsbench_omp`/`cpu_rsbench_omp`, el mismo par
que ya usa la prueba externa sellada del clasificador
(`scripts/pacca/final_campaign/cpu_external_screen_20260921.yaml`, cuyo
propio comentario dice "Estos kernels nunca entran a entrenamiento"; ya
corrieron con éxito en paccaA100, job 7530, 18/18 aceptadas):
  - cpu_xsbench_omp (memory_bound, ~9s)
  - cpu_rsbench_omp (compute_bound, ~44s)

Reutiliza `run_composite`/`resolve_entries`/`PhaseRecord` de
`composite_known.py` sin cambios -- la orquestación (verificación de
checksum antes de cada corrida, registro de fronteras reales con reloj
monotónico, criterio de éxito) no depende de si las familias son
conocidas o inéditas, solo cambia CUÁLES kernels se encadenan. Duplicar
esa lógica aquí arriesgaría que un cambio futuro (p.ej. al criterio de
éxito) se aplique a una aplicación y no a la otra sin que nadie lo note.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fase3_daemon.composite_apps.composite_known import main_with_defaults  # noqa: E402

DEFAULT_SEQUENCE = ("cpu_xsbench_omp", "cpu_rsbench_omp")


def main() -> int:
    return main_with_defaults(
        default_sequence=DEFAULT_SEQUENCE,
        description=__doc__,
        default_cycles=2,
        label="aplicación compuesta B",
    )


if __name__ == "__main__":
    raise SystemExit(main())
