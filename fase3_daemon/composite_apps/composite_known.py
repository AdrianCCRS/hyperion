#!/usr/bin/env python3
"""Aplicación compuesta A -- familias CONOCIDAS (Bloque C, ítem C2,
Plan_Fase3_Daemon.md §0.1, Eje 1/requisito 3).

Encadena kernels del catálogo de Fase 1 que SÍ estuvieron en el
entrenamiento del clasificador de Fase 2, para darle al daemon fases
reales que clasificar y conmutar. Es la pregunta central de la tesis: un
agente que conmuta por fase puede ganarle a cualquier nivel fijo, aunque
ningún nivel fijo le gane a REF sobre un solo kernel.

Fronteras de fase REGISTRADAS, no inferidas: cada kernel del catálogo ya
trae `phase_label_hint` (ground truth derivado en Fase 1 -- ver
`fase1_telemetria/catalog/catalog.yaml`, nunca inventado aquí), y este
driver marca con reloj monotónico el instante exacto en que cada uno
empieza y termina. Eso permite puntuar la CLASIFICACIÓN del daemon contra
una verdad conocida, no solo comparar EDP agregado.

Secuencia por defecto -- dos familias puras del inventario de 30 familias
(`tmp/cpu_quality_20260918/full/inventory_by_family.csv`), ambas con
`binary_checksum` verificado en paccaA100:
  - dgemm_n2048 (compute_bound puro: memory_share=0.0)
  - npb_cg      (memory_bound puro: memory_share=1.0)

Mismo rigor que Fase 1: nunca se corre un binario sin verificar su
checksum contra lo declarado en el catálogo (`common/hpc/catalog.py::verify_binary`,
C02/CAT-07). El criterio de éxito por kernel reproduce
`fase1_telemetria/runner.py::_check_success` (RUN-05) -- mismos dos tipos
(`exit_code`/`stdout_regex`), sin reimplementar una lógica distinta.

NO usa `telemetry_kernel_launcher` (el harness de medición PMU de Fase 1):
la telemetría de esta aplicación la produce el DAEMON que la observa
(`run_daemon.py`/`cpu_loop_main`, corriendo aparte, apuntado a este mismo
cpuset o a este PID vía `--mode pid`), no este driver -- mezclar ambos
mediría el harness, no la aplicación real.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.hpc.catalog import KernelEntry, load_catalog, verify_binary  # noqa: E402

DEFAULT_CATALOG_PATH = _REPO_ROOT / "fase1_telemetria" / "catalog" / "catalog.yaml"
DEFAULT_SEQUENCE = ("dgemm_n2048", "npb_cg")


@dataclass
class PhaseRecord:
    """Una fase real, con su frontera exacta -- el insumo directo para
    puntuar clasificación contra verdad conocida (§0.1 requisito 3)."""
    cycle: int
    kernel_id: str
    phase_label_hint: str
    begin_ns: int
    end_ns: int
    success: bool
    exit_code: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def resolve_entries(catalog: dict[str, KernelEntry], sequence: tuple[str, ...]) -> list[KernelEntry]:
    """Resuelve `sequence` contra `catalog`, fallando cerrado si algo falta
    -- nunca corre con un subconjunto silencioso de la secuencia pedida."""
    missing = [k for k in sequence if k not in catalog]
    if missing:
        raise ValueError(f"kernel(s) no encontrados en el catálogo: {missing}")
    entries = [catalog[k] for k in sequence]
    without_hint = [e.id for e in entries if not e.phase_label_hint]
    if without_hint:
        raise ValueError(
            f"kernel(s) sin phase_label_hint -- no sirven como frontera de fase conocida "
            f"(Aplicación B, familias inéditas, es la contraparte deliberada de esto): {without_hint}"
        )
    return entries


def _check_success(entry: KernelEntry, exit_code: int, stdout: str) -> bool:
    """Mismo criterio que `fase1_telemetria/runner.py::_check_success`
    (RUN-05) -- reimplementado aquí en memoria (esa función lee de un
    archivo en disco, este driver ya tiene el stdout capturado) en vez de
    forzar una dependencia de I/O compartida entre ambos módulos."""
    check = entry.success_check
    check_type = check.get("type")
    if check_type == "exit_code":
        return exit_code == check.get("expected", 0)
    if check_type == "stdout_regex":
        return re.search(check["pattern"], stdout) is not None
    return False


def run_composite(
    entries: list[KernelEntry],
    *,
    cycles: int,
    node_id: str,
    kernels_root: Path,
    on_phase: Callable[[PhaseRecord], None] | None = None,
    run_fn: Callable[..., Any] = subprocess.run,
    now_fn: Callable[[], int] = time.monotonic_ns,
) -> list[PhaseRecord]:
    """Corre `entries` en orden, `cycles` veces, y devuelve un `PhaseRecord`
    por ejecución con las fronteras REALES (reloj monotónico) de cada fase.

    `run_fn`/`now_fn` son inyectables a propósito -- para poder probar la
    lógica de secuencia/verificación/registro sin lanzar binarios reales
    (ver `tests/test_composite_known.py`, corrible en cualquier máquina).
    En producción, `run_fn` es `subprocess.run` sobre el binario real,
    verificado contra su checksum ANTES de cada ejecución (nunca cacheado
    entre ciclos -- mismo criterio CAT-07 de Fase 1: el checksum se
    revisa antes de cada corrida individual, no solo al arrancar).
    """
    records: list[PhaseRecord] = []
    for cycle in range(cycles):
        for entry in entries:
            if not verify_binary(entry, node_id=node_id):
                raise RuntimeError(
                    f"C02: {entry.id!r} no pasó la verificación de checksum en node_id={node_id!r} -- "
                    "nunca se corre un binario sin verificar (mismo rigor que Fase 1, CAT-07)"
                )
            argv = [str(kernels_root / entry.exec_path), *shlex.split(entry.exec_args)]
            begin_ns = now_fn()
            proc = run_fn(argv, cwd=str(kernels_root), capture_output=True, text=True)
            end_ns = now_fn()
            record = PhaseRecord(
                cycle=cycle, kernel_id=entry.id, phase_label_hint=entry.phase_label_hint,
                begin_ns=begin_ns, end_ns=end_ns,
                success=_check_success(entry, proc.returncode, proc.stdout),
                exit_code=proc.returncode,
            )
            records.append(record)
            if on_phase is not None:
                on_phase(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--sequence", nargs="+", default=list(DEFAULT_SEQUENCE),
                         help="IDs del catálogo a encadenar, en orden (default: %(default)s).")
    parser.add_argument("--cycles", type=int, default=3,
                         help="Cuántas veces repetir la secuencia completa (default: 3).")
    parser.add_argument("--node-id", required=True,
                         help="Nodo para resolver binary_checksum (p.ej. 'pacca-a100'), mismo id que usa "
                              "fase1_telemetria/campaign.py -- sin default: elegirlo a propósito.")
    parser.add_argument("--kernels-root", type=Path, default=Path.cwd(),
                         help="Raíz de binarios compilados (default: cwd -- misma convención que "
                              "fase1_telemetria, ver el encabezado de catalog.yaml).")
    parser.add_argument("--boundaries-out", type=Path, required=True,
                         help="Ruta del registro JSONL de fronteras de fase reales (requisito 3, §0.1).")
    args = parser.parse_args()

    catalog = load_catalog(str(args.catalog))
    entries = resolve_entries(catalog, tuple(args.sequence))

    args.boundaries_out.parent.mkdir(parents=True, exist_ok=True)
    with args.boundaries_out.open("w", encoding="utf-8") as out:
        def on_phase(record: PhaseRecord) -> None:
            out.write(record.to_json())
            out.write("\n")
            out.flush()
            status = "OK" if record.success else "FALLÓ"
            duration_s = (record.end_ns - record.begin_ns) / 1e9
            print(f"[ciclo {record.cycle}] {record.kernel_id} ({record.phase_label_hint}) "
                  f"{status} en {duration_s:.2f}s", flush=True)

        records = run_composite(
            entries, cycles=args.cycles, node_id=args.node_id,
            kernels_root=args.kernels_root, on_phase=on_phase,
        )

    n_failed = sum(1 for r in records if not r.success)
    print(f"aplicación compuesta A: {len(records)} fases, {n_failed} fallidas, "
          f"registro en {args.boundaries_out}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
