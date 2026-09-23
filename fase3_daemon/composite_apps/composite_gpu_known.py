#!/usr/bin/env python3
"""Aplicación compuesta A de GPU -- familias CONOCIDAS (Fase 3, ítem C2, contraparte de GPU de `composite_known.py`).

Encadena dos kernels del catálogo cuyas familias SÍ están entre las 16 del entrenamiento del clasificador de GPU y que
duran bastante más que la ventana de decisión del daemon (3 s de actividad sostenida), con `phase_label_hint` como verdad
de fase (fronteras registradas con reloj monotónico, igual que la app A de CPU):
  - gpu_cutlass_simt_dgemm_n4096  compute_bound, ~31 s  (medido en paccaA100, job 7626)
  - gpu_rajaperf_stream_triad     memory_bound,  ~18 s  (idem; RAJAPerf tarda ~5 s en inicializar en CPU con la GPU ociosa,
                                                        y produce actividad extra hacia el final del proceso)
Se elige el par por decisión del usuario (2026-09-23). Los kernels cortos (gpu_dgemm_n4096, 3.4 s) no sirven: la decisión
llega a los 3 s y la fase termina antes.

Uso (en pacca, con el daemon de GPU aparte apuntando a este proceso con --target-pid):
  python3 composite_gpu_known.py --node-id pacca-a100 --kernels-root ~/hyperion-kernels --boundaries-out fronteras.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fase3_daemon.composite_apps.composite_known import main_with_defaults  # noqa: E402

DEFAULT_SEQUENCE = ("gpu_cutlass_simt_dgemm_n4096", "gpu_rajaperf_stream_triad")


def main() -> int:
    return main_with_defaults(
        default_sequence=DEFAULT_SEQUENCE, description=__doc__, default_cycles=3, label="aplicación compuesta A de GPU",
    )


if __name__ == "__main__":
    raise SystemExit(main())
