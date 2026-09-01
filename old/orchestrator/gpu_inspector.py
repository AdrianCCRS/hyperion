from __future__ import annotations

import subprocess

from . import gpu_freqctl as gpu_freqctl_module


class NvidiaSmiGpuInspector:
    """ARC-171: implementación real de `preflight.GpuInspector` -- nunca
    existió antes de esta sesión. `preflight.check_gpu()`/
    `check_gpu_foreign_activity()` (G01-G03) siempre bloqueaban con "se
    requiere un inspector NVML" en cualquier campaña con `gpu.enabled=true`
    porque `cli.py` nunca construía ni pasaba un `gpu_inspector` real -- los
    smokes GPU previos (ARC-153/154) nunca lo expusieron porque corrían con
    `gpu.enabled=false` (G01-G03 se saltan por completo cuando eso es falso,
    `preflight.py::run_campaign_preflight`). Confirmado el hueco al intentar
    la primera campaña real con `gpu.enabled=true` (job 6342, ARC-170).

    Mismo patrón ya establecido en `gpu_freqctl.py`: subprocesos de
    `nvidia-smi`, sin bindings NVML directos (`pynvml`) -- consistente con
    cómo el resto del proyecto ya lee/escribe el reloj de GPU.
    """

    def __init__(self, gpu_index: int | str | None = None):
        self._gpu_index = gpu_freqctl_module._resolve_gpu_index(gpu_index)

    def _query_gpu_field(self, field: str) -> str | None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(self._gpu_index), f"--query-gpu={field}", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().splitlines()
        return lines[0].strip() if lines else None

    def active_processes(self) -> list[int]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(self._gpu_index), "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if result.returncode != 0:
            return []
        pids: list[int] = []
        for line in result.stdout.strip().splitlines():
            stripped = line.strip()
            if stripped.isdigit():
                pids.append(int(stripped))
        return pids

    def persistence_mode(self) -> bool | None:
        value = self._query_gpu_field("persistence_mode")
        if value is None:
            return None
        return value.lower() == "enabled"

    def mig_configuration(self) -> str | None:
        return self._query_gpu_field("mig.mode.current")
