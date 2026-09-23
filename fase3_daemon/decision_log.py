"""Registro estructurado de decisiones del daemon (Plan_Fase3_Daemon.md
§0.1, requisito 2): un renglón JSONL por decisión (una por fase de GPU, una
por tick de CPU), con el MISMO esquema en los tres brazos y en ambos
dispositivos. `fase3_daemon/cpu_loop/include/decision_log.hpp` es el
espejo en C++ de este mismo esquema -- cualquier campo que se agregue aquí
debe agregarse allá también, o el análisis de Fase 4 no puede unir ambas
fuentes con un solo parser.

Sin este registro, el brazo *sombra* no mide nada comparable contra
*activo* (requisito 1: sombra hace el mismo trabajo que activo salvo la
escritura final) -- es el insumo directo de Fase 4 (objetivo 4) y de los
criterios de cierre de Bloque C (puntuar clasificación contra las
fronteras de fase conocidas, no solo comparar EDP agregado).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

ARMS = ("base", "sombra", "activo")
DEVICES = ("cpu", "gpu")


@dataclass
class DecisionRecord:
    ts_ns: int
    arm: str
    device: str  # "cpu" | "gpu"
    label: str | None  # None si no se llegó a clasificar (p.ej. features inválidas)
    confidence: float | None
    features: dict[str, float]
    policy_action: str  # "actuar" | "no_actuar" | "n/a" (no se llegó a consultar la tabla)
    target_freq_khz: int
    applied_freq_khz: int
    written: bool
    write_failed: bool
    inference_time_ns: int | None = None
    actuation_time_ns: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm inválido: {self.arm!r} (debe ser uno de {ARMS})")
        if self.device not in DEVICES:
            raise ValueError(f"device inválido: {self.device!r} (debe ser uno de {DEVICES})")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class DecisionLogWriter:
    """Append-only, una línea JSON por decisión, con flush inmediato --
    una decisión de fase GPU o un tick de CPU cuestan órdenes de magnitud
    más que escribir unas pocas decenas de bytes, así que no vale la pena
    bufferizar y arriesgar perder las últimas decisiones si el proceso
    muere sin pasar por el manejo de señales (§4.2/§4.3 punto 8)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = path.open("a", encoding="utf-8")

    def write(self, record: DecisionRecord) -> None:
        self._fh.write(record.to_json())
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "DecisionLogWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def now_ns() -> int:
    """Mismo reloj que `fase3_daemon/gpu_loop/loop.py::now_ns()` y que
    `CpuPhaseController` en C++ (monotónico) -- para que `ts_ns` de ambos
    dispositivos sea comparable en el mismo eje temporal cuando corren en
    el mismo nodo."""
    return time.monotonic_ns()
