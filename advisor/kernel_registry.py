"""Descubrimiento y verificacion de kernels para la campana de Advisor.

No compila nada -- verifica que lo YA compilado sea apropiado para Advisor
(optimizacion + simbolos) y registra evidencia reproducible (flags reales,
checksum). La precision (FP32/FP64) que aqui se registra es un HINT de
codigo fuente, no la fuente de verdad -- la fuente de verdad es la columna
real 'Data Types' que Advisor reporta por loop (ver classify_roofline.py),
medida, no leida del codigo. Se documenta explicitamente cual es cual, tal
como exige docs/advisor/pipeline_advisor_diseno_y_arquitectura.md.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

KERNEL_CLASS_RE = re.compile(r"^([a-z]+)\.([A-Za-z])\.x$")

# Anclas STREAM/DGEMM/ERT -- mismos hints exactos que ya uso raperezp/run_validation.py
# (la campana de VTune) para no inventar un segundo criterio de deteccion.
# No tienen clase A/B/C real (no son parte de NPB); se registran con
# klass="anchor" en el consolidado, mismo convenio que ya uso VTune.
# "ert" -> kernels/ert/ert_probe.c de este mismo repo (ver ARC-31/CAL-03):
# microbenchmark de FLOPs pico tipo ERT Kernels/kernel1.c, working set
# 1KiB-4MiB (cabe en L1/L2 a proposito), standalone, sin argumentos.
ANCHOR_NAME_HINTS = {
    "stream": ("stream_omp", "stream_c", "stream"),
    "dgemm": ("dgemm_bench", "dgemm"),
    "ert": ("ert_probe", "ert"),
}
# DGEMM necesita tamano de problema + repeticiones como argumentos
# posicionales (mismo patron que ../pipelinevtune/run_vtune_pipeline.py y
# la campana de VTune de este proyecto: dgemm_bench 4096 5). STREAM y ERT no
# necesitan argumentos.
ANCHOR_EXTRA_ARGS: dict[str, list[str]] = {
    "dgemm": ["4096", "5"],
}

# NPB3.4-OMP/config/make.def: buscamos estas dos senales, nunca asumimos
# que estan presentes sin leerlas.
_OPT_FLAG_RE = re.compile(r"-O[23]\b")
_DEBUG_FLAG_RE = re.compile(r"-g\b")

_PRECISION_SOURCE_HINTS = {
    "double_precision_declared": re.compile(r"\bDOUBLE\s+PRECISION\b|\bREAL\*8\b", re.IGNORECASE),
    "single_precision_declared": re.compile(r"\bREAL\*4\b", re.IGNORECASE),
}


@dataclass
class KernelBuildInfo:
    kernel: str
    klass: str
    binary_path: Path
    binary_checksum: str | None
    fflags: str | None
    cflags: str | None
    has_optimization: bool | None
    has_debug_symbols: bool | None
    precision_source_hint: str | None  # "double_precision_declared" | "single_precision_declared" | "mixed" | None
    warnings: list[str] = field(default_factory=list)


def discover_kernels(bin_dir: Path, kernels: tuple[str, ...] | None, classes: tuple[str, ...] | None
                      ) -> list[tuple[str, str, Path]]:
    """Mismo mecanismo de descubrimiento que raperezp/run_validation.py --
    <kernel>.<clase>.x, nunca un nombre inventado. kernels/classes None =
    sin filtro, se toma todo lo encontrado."""
    found = []
    for f in sorted(bin_dir.glob("*.x")):
        if not os.access(f, os.X_OK):
            continue
        m = KERNEL_CLASS_RE.match(f.name)
        if not m:
            continue
        kernel, klass = m.group(1), m.group(2)
        if kernels and kernel.lower() not in kernels:
            continue
        if classes and klass.upper() not in classes:
            continue
        found.append((kernel, klass, f))
    return found


def _is_elf(f: Path) -> bool:
    try:
        with f.open("rb") as fh:
            return fh.read(4) == b"\x7fELF"
    except OSError:
        return False


def discover_anchors(anchor_dir: Path) -> dict[str, Path | None]:
    """Mismo criterio exacto que raperezp/run_validation.py::discover_anchors
    -- coincidencia por nombre de archivo (nunca por carpeta contenedora,
    para no confundir p.ej. OpenBLAS/c_check con el binario real
    dgemm_bench). Devuelve {"stream": Path|None, "dgemm": Path|None}."""
    found: dict[str, Path | None] = {name: None for name in ANCHOR_NAME_HINTS}
    if not anchor_dir.is_dir():
        return found
    elf_files = [f for f in anchor_dir.rglob("*") if f.is_file() and os.access(f, os.X_OK) and _is_elf(f)]
    for anchor, hints in ANCHOR_NAME_HINTS.items():
        candidates = [f for f in elf_files
                      if f.stem.lower() in hints or any(f.stem.lower().startswith(h + "_") for h in hints)]
        found[anchor] = sorted(candidates, key=str)[0] if candidates else None
    return found


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _extract_flags(make_def_text: str, var_name: str) -> str | None:
    m = re.search(rf"^{var_name}\s*=\s*(.+)$", make_def_text, re.MULTILINE)
    return m.group(1).strip() if m else None


def read_build_flags(config_dir: Path | None) -> tuple[str | None, str | None, list[str]]:
    """Lee NPB3.4-OMP/config/make.def REAL -- nunca asume flags. Devuelve
    (fflags, cflags, warnings). config_dir=None es el caso esperado para
    anclas STREAM/DGEMM (no son parte del arbol NPB, no tienen make.def) --
    no es un error, no genera warning, simplemente no hay flags que
    verificar por esta via para ellas."""
    if config_dir is None:
        return None, None, []
    warnings: list[str] = []
    make_def = config_dir / "make.def"
    if not make_def.is_file():
        warnings.append(f"No se encontro {make_def} -- no se pueden verificar flags de compilacion reales.")
        return None, None, warnings
    text = make_def.read_text(errors="replace")
    fflags = _extract_flags(text, "FFLAGS")
    cflags = _extract_flags(text, "CFLAGS")
    if fflags is None:
        warnings.append(f"{make_def}: no se encontro FFLAGS.")
    if cflags is None:
        warnings.append(f"{make_def}: no se encontro CFLAGS (puede ser normal si el kernel es solo Fortran).")
    return fflags, cflags, warnings


def _check_flag_quality(flags: str | None, warnings: list[str], label: str) -> tuple[bool | None, bool | None]:
    if flags is None:
        return None, None
    has_opt = bool(_OPT_FLAG_RE.search(flags))
    has_debug = bool(_DEBUG_FLAG_RE.search(flags))
    if not has_opt:
        warnings.append(
            f"{label}: no se encontro -O2/-O3 en los flags reales ({flags!r}) -- Advisor "
            "va a perfilar codigo no representativo del rendimiento real, no solo un problema "
            "de simbolos."
        )
    if not has_debug:
        warnings.append(
            f"{label}: no se encontro -g en los flags reales ({flags!r}) -- la atribucion a "
            "linea de codigo fuente en Advisor (y la inspeccion posterior en la GUI) puede "
            "degradarse a nivel de funcion/direccion, sin romper la coleccion en si."
        )
    return has_opt, has_debug


def _precision_source_hint(source_path: Path | None) -> str | None:
    """HINT de codigo fuente -- nunca autoritativo, ver docstring del modulo."""
    if source_path is None or not source_path.is_file():
        return None
    text = source_path.read_text(errors="replace")
    dp = bool(_PRECISION_SOURCE_HINTS["double_precision_declared"].search(text))
    sp = bool(_PRECISION_SOURCE_HINTS["single_precision_declared"].search(text))
    if dp and sp:
        return "mixed"
    if dp:
        return "double_precision_declared"
    if sp:
        return "single_precision_declared"
    return None


def build_kernel_info(kernel: str, klass: str, binary: Path, config_dir: Path | None,
                       source_hint_path: Path | None = None) -> KernelBuildInfo:
    warnings: list[str] = []
    fflags, cflags, flag_warnings = read_build_flags(config_dir)
    warnings.extend(flag_warnings)
    has_opt, has_debug = _check_flag_quality(fflags or cflags, warnings, f"{kernel}.{klass}")
    checksum = _sha256(binary)
    if checksum is None:
        warnings.append(f"No se pudo calcular sha256 de {binary} -- ¿existe y es legible?")
    precision_hint = _precision_source_hint(source_hint_path)
    return KernelBuildInfo(
        kernel=kernel, klass=klass, binary_path=binary, binary_checksum=checksum,
        fflags=fflags, cflags=cflags, has_optimization=has_opt, has_debug_symbols=has_debug,
        precision_source_hint=precision_hint, warnings=warnings,
    )
