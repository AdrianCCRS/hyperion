from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_SHIM_SOURCE = Path(__file__).resolve().parent / "native" / "blocking_sync_shim.cpp"
_shim_binary_cache: Path | None = None


def compiled_blocking_sync_shim() -> Path | None:
    """ARC-70: compila (una sola vez, cacheado en disco) el shim LD_PRELOAD
    que fuerza cudaDeviceScheduleBlockingSync en un binario CUDA de terceros
    (Rodinia u otro kernel GPU del catálogo), sin tocar su fuente. Mismo
    patrón que environment._compiled_native_probe(): source en
    orchestrator/native/, compilado on-demand, nunca falla con excepción --
    None significa "este nodo no puede compilarlo ahora" (sin nvcc en PATH,
    o sin headers CUDA encontrados), y el llamador debe tratarlo igual que
    "corre sin el shim", nunca como una falla dura, porque no todo nodo
    tiene GPU.

    Solo necesita g++ (el shim es host-only, sin código de dispositivo,
    confirmado en paccaA100 -- ver docs/retoma/pacca/Diseno_Politica_DVFS_CPU_GPU.md
    sección 8), la ruta de CUDA se infiere de dónde está `nvcc` en PATH, no
    de una variable de entorno adivinada.
    """
    global _shim_binary_cache
    if _shim_binary_cache is not None and _shim_binary_cache.exists():
        return _shim_binary_cache
    if not _SHIM_SOURCE.exists():
        return None

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        logger.warning("ARC-70: nvcc no está en PATH, no se puede localizar CUDA para el shim de blocking sync")
        return None
    cuda_root = Path(nvcc).resolve().parent.parent
    include_dir = cuda_root / "include"
    lib_dir = cuda_root / "lib64"
    if not (include_dir / "cuda_runtime.h").exists():
        logger.warning("ARC-70: cuda_runtime.h no encontrado en %s", include_dir)
        return None

    binary_path = Path(tempfile.gettempdir()) / "hyperion_blocking_sync_shim.so"
    if not binary_path.exists():
        try:
            result = subprocess.run(
                [
                    "g++", "-O2", "-fPIC", "-shared", str(_SHIM_SOURCE), "-o", str(binary_path),
                    f"-I{include_dir}", f"-L{lib_dir}", "-lcudart",
                ],
                capture_output=True, text=True, timeout=60, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0 or not binary_path.exists():
            logger.warning("ARC-70: compilación del shim de blocking sync falló: %s", result.stderr)
            return None

    _shim_binary_cache = binary_path
    return binary_path


def cuda_lib_dir() -> Path | None:
    """Directorio de libcudart.so real, para agregar a LD_LIBRARY_PATH del
    proceso medido -- necesario porque ni el binario GPU ni el shim
    necesariamente lo traen resuelto vía rpath (confirmado empíricamente en
    paccaA100 con binarios compilados con nvcc por defecto)."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None
    return Path(nvcc).resolve().parent.parent / "lib64"
