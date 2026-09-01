from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_SHIM_SOURCE = Path(__file__).resolve().parent / "native" / "blocking_sync_shim.cpp"
_shim_binary_cache: Path | None = None


def _find_cuda_root() -> Path | None:
    """Localiza el árbol del CUDA toolkit real (el que trae cuda_runtime.h
    y libcudart), no necesariamente donde apunta `which nvcc`.

    ARC-74: en paccaA100, `nvcc` en PATH resuelve al binario empaquetado
    dentro del NVIDIA HPC SDK bajo `compilers/bin/` -- el entorno
    "AdaptiveCpp" que carga el nodo lo expone primero (mismo problema que
    ARC-53 ya encontró para el compilador C++ de CMake). Ese nvcc en
    particular NO trae `include/cuda_runtime.h` ni `lib64/libcudart` --
    el CUDA toolkit real vive en un directorio HERMANO dentro del mismo
    HPC SDK: `<raíz_hpc_sdk>/cuda/<versión>/`. Se prueba primero la ruta
    obvia (junto al nvcc encontrado) y, si falta cuda_runtime.h, se busca
    ese hermano antes de rendirse -- nunca se asume una ruta fija ni una
    versión concreta.
    """
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None
    nvcc_dir = Path(nvcc).resolve().parent.parent  # <algo>/bin/nvcc -> <algo>
    if (nvcc_dir / "include" / "cuda_runtime.h").exists():
        return nvcc_dir

    # Patrón NVIDIA HPC SDK: .../Linux_x86_64/<ver>/compilers/bin/nvcc ->
    # el toolkit real vive en .../Linux_x86_64/<ver>/cuda/<cuda_ver>/.
    hpc_sdk_root = nvcc_dir.parent
    cuda_siblings_dir = hpc_sdk_root / "cuda"
    if not cuda_siblings_dir.is_dir():
        return None
    for candidate in sorted(cuda_siblings_dir.glob("*"), reverse=True):  # versión más reciente primero
        if (candidate / "include" / "cuda_runtime.h").exists():
            return candidate
    return None


def _find_cublas_lib_dir(cuda_root: Path) -> Path | None:
    """libcublas vive en `math_libs/`, un árbol hermano de `cuda/<ver>/` en
    el mismo HPC SDK -- no en `cuda/<ver>/lib64` (confirmado empíricamente
    en paccaA100: `cublas_dgemm_bench` fallaba con
    `libcublas.so.12: cannot open shared object file` usando solo
    `cuda_lib_dir()`). Mismo patrón de búsqueda que `_find_cuda_root()`,
    nunca una ruta fija."""
    lib64 = cuda_root / "lib64"
    if any(lib64.glob("libcublas.so*")):
        return lib64
    hpc_sdk_root = cuda_root.parent.parent  # cuda/<ver> -> cuda -> raíz
    math_libs_root = hpc_sdk_root / "math_libs"
    if not math_libs_root.is_dir():
        return None
    for candidate in sorted(math_libs_root.glob("*/targets/*/lib"), reverse=True):
        if any(candidate.glob("libcublas.so*")):
            return candidate
    return None


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
    sección 8), la ruta de CUDA se infiere con `_find_cuda_root()`.
    """
    global _shim_binary_cache
    if _shim_binary_cache is not None and _shim_binary_cache.exists():
        return _shim_binary_cache
    if not _SHIM_SOURCE.exists():
        return None

    cuda_root = _find_cuda_root()
    if cuda_root is None:
        logger.warning("ARC-70/74: no se encontró un CUDA toolkit real (con cuda_runtime.h) en este nodo")
        return None
    include_dir = cuda_root / "include"
    lib_dir = cuda_root / "lib64"

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


def cuda_lib_dirs() -> list[Path]:
    """Directorios reales de libcudart/libcublas, para agregar a
    LD_LIBRARY_PATH del proceso medido -- necesario porque ni el binario GPU
    ni el shim necesariamente los traen resueltos vía rpath (confirmado
    empíricamente en paccaA100 con binarios compilados con nvcc por
    defecto). ARC-74: son dos árboles distintos del mismo HPC SDK
    (cuda/<ver>/lib64 para cudart, math_libs/<ver>/.../lib para cublas), no
    uno solo -- se devuelven ambos cuando existen, nunca se asume que un
    kernel GPU necesita solo uno de los dos."""
    cuda_root = _find_cuda_root()
    if cuda_root is None:
        return []
    dirs = [cuda_root / "lib64"]
    cublas_dir = _find_cublas_lib_dir(cuda_root)
    if cublas_dir is not None and cublas_dir not in dirs:
        dirs.append(cublas_dir)
    return dirs
