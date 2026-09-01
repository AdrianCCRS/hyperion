#!/usr/bin/env python3
"""Punto de entrada de Fase 1 (Objetivo 1): recolección de telemetría.

Envoltura ejecutable de `fase1_telemetria/cli.py`. No duplica el
parseo de argumentos: `cli.py` ya expone un `argparse` completo con
subcomandos (`diagnose`, `calibrate`, `run-campaign`, `postprocess`,
`report`); este script solo resuelve el `sys.path` para que
`fase1_telemetria` y `common` sean importables sin importar desde qué
directorio se invoque, y delega en `cli.main()`.

⚠️ El directorio de trabajo (cwd) desde el que se invoca este script SÍ
importa, y no es la raíz del repositorio: los binarios de terceros del
catálogo (`exec_path` en `catalog.yaml`, p. ej. `bin/stream_c`) se resuelven
relativos al cwd del proceso, no a la ubicación del catálogo ni del repo
-- por convención viven en `~/hyperion-kernels/bin` (un directorio EXTERNO
al repositorio, ver `fase1_telemetria/README.md`). El binario propio del
harness (`telemetry_kernel_launcher`) sí se resuelve aparte, de forma
independiente del cwd, vía `common/hpc_config.toml`.

    cd ~/hyperion-kernels
    python3 /ruta/al/repo/fase1_telemetria/run_campaign.py diagnose \
        --manifest /ruta/al/repo/fase1_telemetria/catalog/campaigns/campaign_example.yaml \
        --output-dir /tmp/diagnose_out

    python3 /ruta/al/repo/fase1_telemetria/run_campaign.py run-campaign \
        --manifest /ruta/al/repo/fase1_telemetria/catalog/campaigns/campaign_example.yaml \
        --node-id nodo01 --reference-kernel-ref stream_official

Ver `fase1_telemetria/README.md` para el significado de cada subcomando,
el formato de `windows.csv` que produce, y los prerrequisitos de
plataforma (§2.0 de `Plan_Detallado_Realineacion_Hyperion.md`) que hay
que verificar antes de correr una campaña real.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Inserta la raíz del repositorio en sys.path para que `import common.hpc`
# y `import fase1_telemetria` funcionen sin importar el cwd del proceso
# que invoca este script (mismo patrón que usan los tests del proyecto).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fase1_telemetria.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
