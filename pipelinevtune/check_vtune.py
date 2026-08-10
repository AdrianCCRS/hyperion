#!/usr/bin/env python3
"""Preflight check para el pipeline VTune de Cartagena.

Ver PLAN.md Fase 2. Corre dentro de una reserva Slurm activa en paccaA100, con
los modulos de VTune ya cargados (module load devtools/intel/oneapi/2023 &&
module load vtune/2023.0.0 -- ver context/04_vtune_selfchecker_resultados.md
para por que el modulo vtune es jerarquico y no aparece sin el modulo padre).

Codigo de salida: 0 si no hay errores bloqueantes, 1 si los hay.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# 2026-08-07: alineado con el dominio del orquestador principal en este
# mismo nodo (campaign_pacca_ref.yaml: delegated_cpus=0-5), no con "todo el
# socket" -- ver context/02_decisiones.md D6 (actualizada).
D6_OMP_NUM_THREADS = "6"
D6_OMP_PLACES = "cores"
D6_OMP_PROC_BIND = "close"

REQUIRED_ANALYSES = ("hotspots", "hpc-performance")
KERNEL_CLASS_RE = re.compile(r"^([a-z]+)\.([A-Za-z])\.x$")
MIN_FREE_DISK_GB = 1.0

ANCHOR_NAME_HINTS = {
    "stream": ("stream_omp", "stream_c", "stream"),
    "dgemm": ("dgemm_bench", "dgemm"),
}


@dataclass
class PreflightResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def check_vtune_in_path(result: PreflightResult) -> str | None:
    vtune_path = shutil.which("vtune")
    if vtune_path is None:
        result.error(
            "vtune no esta en PATH. Secuencia confirmada en Fase 0 para este "
            "nodo: 'module load devtools/intel/oneapi/2023' primero (el modulo "
            "vtune es jerarquico y no aparece en 'module avail vtune' sin su "
            "padre), luego 'module load vtune/2023.0.0'."
        )
        result.summary["VTune disponible"] = "no"
        return None
    result.summary["VTune disponible"] = f"si ({vtune_path})"
    return vtune_path


def check_vtune_version(result: PreflightResult, vtune_path: str | None) -> None:
    if vtune_path is None:
        result.summary["Versión"] = "NA"
        return
    try:
        proc = subprocess.run(
            [vtune_path, "--version"], capture_output=True, text=True, timeout=30
        )
    except Exception as exc:  # noqa: BLE001 - se reporta como error bloqueante
        result.error(f"'vtune --version' fallo al ejecutarse: {exc}")
        result.summary["Versión"] = "error"
        return
    if proc.returncode != 0:
        result.error(
            f"'vtune --version' termino con codigo {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
        result.summary["Versión"] = "error"
        return
    text = (proc.stdout or proc.stderr).strip()
    version_line = text.splitlines()[0] if text else "(vacio)"
    result.summary["Versión"] = version_line
    if "2023" not in version_line:
        result.warn(
            f"La version reportada no contiene '2023' ({version_line}). El "
            "plan y las capturas reales de Fase 0 se validaron contra VTune "
            "2023.0.0 -- confirma que sigue siendo la misma version antes de "
            "confiar en los nombres de campo ya documentados."
        )


def check_collect_list(result: PreflightResult, vtune_path: str | None) -> None:
    if vtune_path is None:
        result.summary["Hotspots HW disponible"] = "NA"
        result.summary["HPC Performance disponible"] = "NA"
        return
    try:
        proc = subprocess.run(
            [vtune_path, "-collect-list"], capture_output=True, text=True, timeout=30
        )
    except Exception as exc:  # noqa: BLE001
        result.error(f"'vtune -collect-list' fallo al ejecutarse: {exc}")
        result.summary["Hotspots HW disponible"] = "error"
        result.summary["HPC Performance disponible"] = "error"
        return
    text = proc.stdout + proc.stderr
    for analysis, label in (
        ("hotspots", "Hotspots HW disponible"),
        ("hpc-performance", "HPC Performance disponible"),
    ):
        # Los nombres de analisis aparecen como token propio en su linea,
        # ej. "   hotspots\n      Identify...". Basta con un match de palabra
        # completa en cualquier parte del listado.
        found = re.search(rf"^\s*{re.escape(analysis)}\s*$", text, re.MULTILINE)
        result.summary[label] = "si" if found else "no"
        if not found:
            result.error(
                f"El analisis '{analysis}' no aparece listado por "
                "'vtune -collect-list'. Ver PLAN.md Fase 0/2."
            )


def check_ebs_smoke(
    result: PreflightResult, vtune_path: str | None, work_dir: Path
) -> None:
    """Corre una coleccion real y corta para confirmar que EBS funciona de
    verdad (no solo que la opcion exista). No valida contenido de metricas
    (eso ya se hizo una vez en Fase 0 contra un kernel NPB real, ver
    context/04) -- aqui solo se valida el mecanismo: collect, finalize,
    report sin error fatal, sobre un target trivial y rapido."""
    if vtune_path is None:
        result.summary["EBS funcional"] = "NA"
        return

    hs_dir = work_dir / "preflight_hotspots"
    target = ["sleep", "1"]

    try:
        collect = subprocess.run(
            [
                vtune_path,
                "-collect",
                "hotspots",
                "-knob",
                "sampling-mode=hw",
                "-r",
                str(hs_dir),
                "--",
                *target,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        result.error(f"Smoke test de EBS (collect) fallo al ejecutarse: {exc}")
        result.summary["EBS funcional"] = "no"
        return

    if collect.returncode != 0 or re.search(
        r"^\s*vtune:\s*Error", collect.stdout + collect.stderr, re.MULTILINE
    ):
        result.error(
            "Smoke test de EBS (hotspots -knob sampling-mode=hw) termino con "
            f"error. Codigo={collect.returncode}. stderr:\n"
            f"{collect.stderr.strip()[-800:]}"
        )
        result.summary["EBS funcional"] = "no"
        return

    try:
        report = subprocess.run(
            [vtune_path, "-report", "summary", "-r", str(hs_dir)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        result.error(f"Smoke test de EBS (report) fallo al ejecutarse: {exc}")
        result.summary["EBS funcional"] = "no"
        return

    if report.returncode != 0:
        result.error(
            "VTune no pudo generar el reporte del resultado de prueba "
            f"(codigo {report.returncode})."
        )
        result.summary["EBS funcional"] = "no"
        return

    result.summary["EBS funcional"] = "si"


def check_bin_dir(result: PreflightResult, bin_dir: Path) -> list[Path]:
    if not bin_dir.is_dir():
        result.error(f"El directorio de binarios no existe: {bin_dir}")
        result.summary["Binarios encontrados"] = "0"
        return []

    candidates = sorted(bin_dir.glob("*.x"))
    valid_kernels = []
    for f in candidates:
        if not os.access(f, os.X_OK):
            result.warn(f"Sin permiso de ejecucion, se ignora: {f}")
            continue
        m = KERNEL_CLASS_RE.match(f.name)
        if not m:
            result.warn(
                f"Nombre de archivo no encaja con <kernel>.<clase>.x, se "
                f"ignora: {f.name}"
            )
            continue
        valid_kernels.append(f)

    result.summary["Binarios encontrados"] = str(len(valid_kernels))
    if not valid_kernels:
        result.error(
            f"No se encontro ningun binario valido (<kernel>.<clase>.x) en "
            f"{bin_dir}."
        )
    return valid_kernels


def _is_elf_binary(f: Path) -> bool:
    try:
        with f.open("rb") as fh:
            return fh.read(4) == b"\x7fELF"
    except OSError:
        return False


def _best_anchor_match(candidates: list[Path], hints: tuple[str, ...]) -> Path | None:
    """Elige el candidato mas especifico: el que matchea el hint mas
    temprano (mas especifico) en la lista de preferencia, y a igualdad,
    el de ruta mas corta/alfabeticamente primero para ser determinista."""
    best: tuple[int, Path] | None = None
    for f in candidates:
        stem = f.stem.lower()
        for idx, hint in enumerate(hints):
            if stem == hint or stem.startswith(hint + "_") or stem == hint.rstrip("_"):
                if best is None or idx < best[0] or (idx == best[0] and str(f) < str(best[1])):
                    best = (idx, f)
                break
    return best[1] if best else None


def check_anchor_binaries(result: PreflightResult, anchor_dir: Path) -> dict[str, Path | None]:
    found: dict[str, Path | None] = {name: None for name in ANCHOR_NAME_HINTS}
    if not anchor_dir.is_dir():
        result.warn(
            f"El directorio de anclas no existe: {anchor_dir}. La Fase 4 "
            "(eje de techos STREAM/DGEMM) no tendra con que trabajar hasta "
            "que existan."
        )
    else:
        all_files = [
            f for f in anchor_dir.rglob("*") if f.is_file() and os.access(f, os.X_OK)
        ]
        elf_files = [f for f in all_files if _is_elf_binary(f)]
        for anchor, hints in ANCHOR_NAME_HINTS.items():
            # El nombre del archivo debe matchear el hint -- nunca el
            # directorio contenedor (ej. "OpenBLAS/c_check" es un script de
            # build, no el binario dgemm_bench, aunque viva bajo esa carpeta).
            candidates = [f for f in elf_files if any(h in f.stem.lower() for h in hints)]
            found[anchor] = _best_anchor_match(candidates, hints)

    disponibles = [name for name, path in found.items() if path is not None]
    faltantes = [name for name, path in found.items() if path is None]
    detalle = ", ".join(f"{name}={found[name]}" for name in disponibles) or "ninguno"
    estado = "si" if not faltantes else ("parcial" if disponibles else "no")
    result.summary["Kernels ancla disponibles"] = f"{estado} ({detalle})"
    if faltantes:
        result.warn(
            f"Ancla(s) no encontradas en {anchor_dir}: {', '.join(faltantes)}. "
            "No bloquea el preflight, pero Fase 4.3 no podra calcular "
            "roofline_vs_ceilings sin ellas."
        )
    return found


def check_omp_domain(result: PreflightResult) -> None:
    threads = os.environ.get("OMP_NUM_THREADS")
    places = os.environ.get("OMP_PLACES")
    bind = os.environ.get("OMP_PROC_BIND")

    result.summary["Dominio OMP"] = (
        f"OMP_NUM_THREADS={threads or '(sin definir)'} "
        f"OMP_PLACES={places or '(sin definir)'} "
        f"OMP_PROC_BIND={bind or '(sin definir)'}"
    )

    if not (threads and places and bind):
        result.warn(
            "OMP_NUM_THREADS/OMP_PLACES/OMP_PROC_BIND no estan todos "
            "definidos en el entorno. El pipeline debe fijarlos "
            "explicitamente antes de correr binarios (ver D6 en "
            "context/02_decisiones.md)."
        )
        return

    if (threads, places, bind) != (D6_OMP_NUM_THREADS, D6_OMP_PLACES, D6_OMP_PROC_BIND):
        result.warn(
            f"El dominio OMP activo (OMP_NUM_THREADS={threads} "
            f"OMP_PLACES={places} OMP_PROC_BIND={bind}) difiere del default "
            f"D6 (6/cores/close, cores 0-5, alineado con el orquestador). Si es intencional, "
            "documentalo explicitamente en los metadatos de la campaña -- "
            "no debe mezclarse silenciosamente con corridas de dominio "
            "estandar."
        )


def check_slurm_context(result: PreflightResult) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    node = os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST")
    cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE")
    ntasks = os.environ.get("SLURM_NTASKS")
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")

    if job_id is None:
        result.warn(
            "No se detecto una reserva Slurm activa (SLURM_JOB_ID vacio). "
            "El pipeline real y sus smoke tests de VTune deben correr dentro "
            "de 'srun --exclusive -w paccaA100 ...', no en el nodo de login."
        )
        result.summary["Slurm detectado"] = "no"
        return

    result.summary["Slurm detectado"] = (
        f"si (job={job_id} nodo={node or '?'} "
        f"cpus_nodo={cpus_on_node or '?'} ntasks={ntasks or '?'} "
        f"cpus_por_tarea={cpus_per_task or '?'})"
    )


def check_disk_space(result: PreflightResult, path: Path) -> None:
    path = path if path.exists() else path.parent
    try:
        usage = shutil.disk_usage(path)
    except Exception as exc:  # noqa: BLE001
        result.warn(f"No se pudo determinar el espacio en disco en {path}: {exc}")
        return
    free_gb = usage.free / (1024**3)
    result.summary["Espacio libre"] = f"{free_gb:.1f} GB en {path}"
    if free_gb < MIN_FREE_DISK_GB:
        result.error(
            f"Espacio en disco insuficiente en {path}: {free_gb:.2f} GB libres "
            f"(minimo {MIN_FREE_DISK_GB} GB). Los resultados de VTune pueden "
            "pesar decenas de MB por corrida."
        )


def print_report(result: PreflightResult) -> None:
    print("=== Preflight VTune — Cartagena (paccaA100) ===")
    for key, value in result.summary.items():
        print(f"{key}: {value}")

    print(f"\nErrores bloqueantes: {len(result.errors)}")
    for e in result.errors:
        print(f"  - {e}")

    print(f"\nAdvertencias: {len(result.warnings)}")
    for w in result.warnings:
        print(f"  - {w}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=Path("bin"))
    parser.add_argument("--anchor-dir", type=Path, default=Path("anchor_bin"))
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("./.preflight_tmp"),
        help="Directorio para el resultado de prueba del smoke test de EBS.",
    )
    args = parser.parse_args(argv)

    result = PreflightResult()

    vtune_path = check_vtune_in_path(result)
    check_vtune_version(result, vtune_path)
    check_collect_list(result, vtune_path)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        check_ebs_smoke(result, vtune_path, args.work_dir)
    finally:
        shutil.rmtree(args.work_dir, ignore_errors=True)

    check_bin_dir(result, args.bin_dir)
    check_anchor_binaries(result, args.anchor_dir)
    check_omp_domain(result)
    check_slurm_context(result)
    check_disk_space(result, args.bin_dir)

    print_report(result)
    return 1 if result.errors else 0


if __name__ == "__main__":
    sys.exit(main())
