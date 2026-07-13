from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeProfile:
    """Perfil mínimo consumido por preflight antes de iniciar una campaña."""

    pmc_count: int
    perf_events_supported: tuple[str, ...] = ()
