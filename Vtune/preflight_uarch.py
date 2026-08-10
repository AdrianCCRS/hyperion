#!/usr/bin/env python3
"""Preflight especifico de Microarchitecture Exploration para la campana de
validacion cruzada VTune (paccaA100 / Cartagena).

Distinto del preflight de pipelinevtune/check_vtune.py en un punto central:
ese preflight verificaba Hotspots HW + HPC Performance Characterization,
confirmados disponibles en la Fase 0 de ese proyecto pero con
Microarchitecture Exploration confirmado NO disponible por falta de acceso a
PMU/uncore (ver pipelinevtune/context/04_vtune_selfchecker_resultados.md).
Este preflight parte de la premisa de que ese permiso cambio y verifica el
CONTENIDO real de uarch-exploration, no solo si '-collect-list' lo nombra --
la leccion ya aprendida en este proyecto es que listar un analisis no prueba
que produzca metricas pobladas (ver pipelinevtune/PLAN.md Fase 0).

Regla dura, sin excepcion: este script NUNCA mata, señaliza ni modifica
ningun proceso de otro usuario. Los checks de procesos ajenos son
DIAGNOSTICO puro (listar PID/comm/cpu), igual que E06 en
orchestrator/preflight.py del proyecto principal -- se lee /proc, nunca se
escribe ni se envia una señal.

Codigo de salida: 0 si no hay errores bloqueantes, 1 si los hay. Un error
bloqueante aqui debe leerse como "falta un permiso o config concreta",
nunca como "bug generico" -- el mensaje de cada check dice exactamente que
falta, para poder pedirlo a administracion si aplica.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Mismo dominio D6 que pipelinevtune (context/02_decisiones.md D6): 6 cores
# fisicos, 0-5, sin SMT, alineado con delegated_cpus del orquestador
# principal en este nodo (orchestrator/schemas/campaign_pacca_ref.yaml).
D6_CORE_RANGE = "0-5"
D6_CORES = list(range(0, 6))

# Confirmado en pipelinevtune/context/01_nodo_cartagena.md (lscpu real).
EXPECTED_CPU_MODEL_SUBSTR = "Gold 5315Y"
EXPECTED_ARCH = "x86_64"

ANALYSIS_NAME = "uarch-exploration"
ANALYSIS_DISPLAY_NAME_HINTS = ("uarch-exploration", "microarchitecture exploration")


@dataclass
class PreflightResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "errors": self.errors,
            "warnings": self.warnings,
            "ok": not self.errors,
        }


# --------------------------------------------------------------------------
# vtune en si: PATH, version, listado de analisis
# --------------------------------------------------------------------------


def check_vtune_in_path(result: PreflightResult) -> str | None:
    vtune_path = shutil.which("vtune")
    if vtune_path is None:
        result.error(
            "vtune no esta en PATH. Secuencia confirmada para este nodo: "
            "'module load devtools/intel/oneapi/2023' (modulo padre, "
            "obligatorio primero) seguido de 'module load vtune/2023.0.0' "
            "(el modulo vtune es jerarquico y no aparece en 'module avail "
            "vtune' sin el padre cargado -- ver "
            "pipelinevtune/context/04_vtune_selfchecker_resultados.md)."
        )
        result.summary["vtune_disponible"] = "no"
        return None
    result.summary["vtune_disponible"] = f"si ({vtune_path})"
    return vtune_path


def check_vtune_version(result: PreflightResult, vtune_path: str | None) -> str | None:
    if vtune_path is None:
        result.summary["version"] = "NA"
        return None
    try:
        proc = subprocess.run([vtune_path, "--version"], capture_output=True, text=True, timeout=30)
    except Exception as exc:  # noqa: BLE001
        result.error(f"'vtune --version' fallo al ejecutarse: {exc}")
        result.summary["version"] = "error"
        return None
    text = (proc.stdout or proc.stderr).strip()
    version_line = text.splitlines()[0] if text else "(vacio)"
    result.summary["version"] = version_line
    return version_line


def check_uarch_listed(result: PreflightResult, vtune_path: str | None) -> bool:
    """Confirma que 'uarch-exploration' aparece en -collect-list. Necesario
    pero NO suficiente -- ver check_uarch_smoke() para la prueba real."""
    if vtune_path is None:
        result.summary["uarch_exploration_listado"] = "NA"
        return False
    try:
        proc = subprocess.run([vtune_path, "-collect-list"], capture_output=True, text=True, timeout=30)
    except Exception as exc:  # noqa: BLE001
        result.error(f"'vtune -collect-list' fallo al ejecutarse: {exc}")
        result.summary["uarch_exploration_listado"] = "error"
        return False
    text = proc.stdout + proc.stderr
    found = re.search(rf"^\s*{re.escape(ANALYSIS_NAME)}\s*$", text, re.MULTILINE) is not None
    result.summary["uarch_exploration_listado"] = "si" if found else "no"
    if not found:
        result.error(
            f"'{ANALYSIS_NAME}' no aparece en 'vtune -collect-list' con esta "
            "instalacion/version de VTune. Esto es distinto de un problema "
            "de permisos -- si el analisis ni siquiera esta empaquetado en "
            "este VTune, ningun permiso adicional lo va a destapar. "
            "Confirmar version de VTune y componentes instalados con "
            "administracion del cluster antes de seguir."
        )
    return found


# --------------------------------------------------------------------------
# PMU / kernel: perf_event_paranoid, kptr_restrict, capacidades
# --------------------------------------------------------------------------


def _read_sysctl(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def check_perf_event_paranoid(result: PreflightResult) -> None:
    value = _read_sysctl("/proc/sys/kernel/perf_event_paranoid")
    if value is None:
        result.error(
            "No se pudo leer /proc/sys/kernel/perf_event_paranoid. Sin este "
            "valor no se puede razonar sobre que eventos de PMU son "
            "accesibles sin privilegios -- puede indicar un kernel sin "
            "soporte perf_event, poco probable en este nodo pero bloqueante "
            "si ocurre."
        )
        result.summary["perf_event_paranoid"] = "error"
        return
    result.summary["perf_event_paranoid"] = value
    try:
        level = int(value)
    except ValueError:
        result.warn(f"perf_event_paranoid tiene un valor no numerico inesperado: {value!r}")
        return
    # Semantica del kernel Linux (ver man perf_event_open(2)):
    #   2  -- solo eventos de usuario, sin CPU-wide, sin acceso crudo
    #   1  -- + eventos CPU-wide para el propio usuario
    #   0  -- + acceso a la mayoria de eventos raw/PMU incluyendo algunos uncore
    #  -1  -- sin restriccion (requiere admin para bajarlo a esto)
    # uarch-exploration necesita bastantes eventos programables simultaneos
    # (TMAM completo); documentar el nivel real en vez de asumir cual hace
    # falta -- eso lo confirma el smoke test empirico (check_uarch_smoke),
    # este check solo dtjena constancia del valor para el reporte.
    if level >= 2:
        result.warn(
            f"perf_event_paranoid={level}: nivel restrictivo (solo eventos "
            "propios de usuario). Si el smoke test de uarch-exploration "
            "falla mas abajo, este es el primer sospechoso a revisar con "
            "administracion -- no es un error todavia, es contexto."
        )


def check_kptr_restrict(result: PreflightResult) -> None:
    value = _read_sysctl("/proc/sys/kernel/kptr_restrict")
    result.summary["kptr_restrict"] = value if value is not None else "no_legible"
    # No bloqueante: afecta resolucion de simbolos de kernel, no el
    # userspace de los binarios NPB/STREAM/DGEMM que este pipeline mide
    # (mismo hallazgo que pipelinevtune/context/04, seccion de restricciones).


def check_cap_perfmon(result: PreflightResult) -> None:
    """CAP_PERFMON (Linux 5.8+) es la capability moderna que reemplaza a
    CAP_SYS_ADMIN para abrir eventos de PMU sin ser root. Se verifica de
    forma indirecta via /proc/self/status (campo CapEff) -- no hay una API
    de shell estandar mas directa sin dependencias adicionales."""
    try:
        status = Path("/proc/self/status").read_text()
    except OSError:
        result.warn("No se pudo leer /proc/self/status para inspeccionar capabilities efectivas.")
        return
    m = re.search(r"^CapEff:\s*([0-9a-fA-F]+)", status, re.MULTILINE)
    if not m:
        result.warn("No se encontro el campo CapEff en /proc/self/status.")
        return
    cap_eff = int(m.group(1), 16)
    # CAP_PERFMON = bit 38 (ver capability.h). Informativo: la ausencia no es
    # bloqueante por si sola, porque perf_event_paranoid bajo + driverless
    # perf_event_open puede bastar sin esta capability explicita (asi
    # funcionaba Hotspots HW/HPC Performance en este mismo nodo antes de
    # este permiso nuevo -- ver docs/vtune/Informe_VTune_Profiler.md §3.2).
    has_perfmon = bool(cap_eff & (1 << 38))
    result.summary["cap_perfmon_efectiva"] = "si" if has_perfmon else "no"


# --------------------------------------------------------------------------
# Arquitectura de CPU
# --------------------------------------------------------------------------


def check_cpu_architecture(result: PreflightResult) -> None:
    try:
        proc = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=15)
        text = proc.stdout
    except Exception as exc:  # noqa: BLE001
        result.error(f"'lscpu' fallo al ejecutarse: {exc}")
        return

    arch_m = re.search(r"^Architecture:\s*(\S+)", text, re.MULTILINE)
    model_m = re.search(r"^Model name:\s*(.+)$", text, re.MULTILINE)
    arch = arch_m.group(1) if arch_m else None
    model = model_m.group(1).strip() if model_m else None
    result.summary["arquitectura"] = arch or "desconocida"
    result.summary["modelo_cpu"] = model or "desconocido"

    if arch != EXPECTED_ARCH:
        result.error(f"Arquitectura inesperada: {arch!r} (se esperaba {EXPECTED_ARCH!r}).")
    if model is None or EXPECTED_CPU_MODEL_SUBSTR not in model:
        result.warn(
            f"El modelo de CPU reportado ({model!r}) no contiene "
            f"{EXPECTED_CPU_MODEL_SUBSTR!r} -- si este nodo cambio de "
            "hardware, los nombres de metrica/microarquitectura de Ice "
            "Lake-SP que asume este pipeline (ver docs/vtune/"
            "vtune_cross_validation.md) pueden no aplicar tal cual."
        )


# --------------------------------------------------------------------------
# Afinidad y contexto Slurm
# --------------------------------------------------------------------------


def check_slurm_context(result: PreflightResult) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    node = os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST")
    cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE")
    if job_id is None:
        result.error(
            "SLURM_JOB_ID vacio -- este pipeline debe correr dentro de un "
            "job de Slurm (sbatch_vtune_validation.sh), nunca directo en el "
            "nodo de login ni fuera de una reserva."
        )
        result.summary["slurm_detectado"] = "no"
        return
    result.summary["slurm_detectado"] = f"si (job={job_id} nodo={node or '?'} cpus={cpus_on_node or '?'})"


def check_affinity(result: PreflightResult, expected_cores: list[int]) -> None:
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        result.warn(f"No se pudo leer sched_getaffinity: {exc}")
        return
    result.summary["cpus_afinidad_real"] = ",".join(str(c) for c in allowed)
    missing = [c for c in expected_cores if c not in allowed]
    extra = [c for c in allowed if c not in expected_cores]
    if missing:
        result.error(
            f"La afinidad real del proceso ({allowed}) no cubre el dominio "
            f"D6 esperado ({expected_cores}). Faltan: {missing}. Slurm no "
            "asigno los cores esperados, o taskset aun no se aplico -- "
            "revisar antes de correr VTune, no forzar taskset sobre un "
            "cpuset que no los contiene."
        )
    if extra:
        result.warn(
            f"La afinidad real incluye cores fuera del dominio D6: {extra}. "
            "No bloqueante si --exclusive dio el nodo completo, pero "
            "confirmar que taskset -c 0-5 se aplique de todas formas para "
            "mantener comparabilidad con el resto del proyecto."
        )


def check_foreign_processes(result: PreflightResult, cores: list[int]) -> None:
    """Diagnostico puro: lista procesos de OTROS usuarios corriendo AHORA
    MISMO (state='R') en los cores del dominio D6. Nunca los toca. Mismo
    criterio que E06/PRE-E06 en orchestrator/preflight.py del proyecto
    principal (campo 'processor' de /proc/<pid>/stat + state=='R'), no por
    membresia de cgroup -- ver ARC-44 en la Guia Maestra Fase 1 DVFS para
    por que ese es el criterio correcto y el que se abandono (Cpus_allowed
    marcaba como 'ajeno' a casi cualquier daemon del sistema en reposo)."""
    own_uid = os.getuid()
    foreign: list[str] = []
    for pid_dir in Path("/proc").glob("[0-9]*"):
        pid = pid_dir.name
        try:
            stat_text = (pid_dir / "stat").read_text()
            uid = pid_dir.stat().st_uid
        except (OSError, ValueError):
            continue
        if uid == own_uid:
            continue
        # /proc/<pid>/stat: campo 3 = state, campo 39 = processor (0-indexed
        # tras el nombre entre parentesis, que puede contener espacios).
        m = re.match(r"^\d+\s+\(.*\)\s+(\S)\s+(?:-?\d+\s+){35}(\d+)", stat_text)
        if not m:
            continue
        state, processor = m.group(1), int(m.group(2))
        if state == "R" and processor in cores:
            comm = pid_dir.name
            try:
                comm = (pid_dir / "comm").read_text().strip()
            except OSError:
                pass
            foreign.append(f"pid={pid} comm={comm} uid={uid} core={processor}")

    result.summary["procesos_ajenos_activos_en_dominio"] = str(len(foreign))
    if foreign:
        result.warn(
            "Procesos de otros usuarios corriendo activamente en el dominio "
            f"D6 ({cores}) en este instante: {'; '.join(foreign)}. Esto es "
            "SOLO diagnostico -- no se mata ni se modifica nada. Si "
            "--exclusive realmente reservo el nodo, este listado deberia "
            "salir vacio; si no, confirmar la reserva antes de confiar en "
            "Memory Bound/DRAM Bound como medidas limpias del kernel propio."
        )


# --------------------------------------------------------------------------
# La prueba real: un uarch-exploration corto sobre un binario trivial
# --------------------------------------------------------------------------

UARCH_TOP_LEVEL_LABELS = ("Retiring", "Front-End Bound", "Bad Speculation", "Back-End Bound")


def check_uarch_smoke(result: PreflightResult, vtune_path: str | None, work_dir: Path,
                       pin_prefix: list[str]) -> None:
    """La prueba que realmente importa: availability en -collect-list no
    prueba que el analisis produzca metricas pobladas (leccion ya aprendida
    en pipelinevtune/PLAN.md Fase 0 con hpc-performance). Corre
    uarch-exploration de verdad sobre un target trivial y corto, y confirma
    que las 4 categorias de Nivel 1 del Top-Down salen con numeros reales,
    no 'NA' ni ausentes."""
    if vtune_path is None:
        result.summary["uarch_exploration_funcional"] = "NA"
        return

    smoke_dir = work_dir / "preflight_uarch_smoke"
    shutil.rmtree(smoke_dir, ignore_errors=True)
    # 'sleep' no genera trabajo de CPU real -- se usa un bucle corto en shell
    # con aritmetica para que haya algo de actividad de pipeline que medir,
    # analogo a por que check_ebs_smoke() de pipelinevtune usa un target
    # trivial pero no completamente inerte para el smoke de mecanismo (no de
    # contenido cientifico, eso lo hace la campana real sobre los kernels).
    target = ["bash", "-c", "x=0; for i in $(seq 1 20000000); do x=$((x+i)); done"]

    try:
        collect = subprocess.run(
            [vtune_path, "-collect", ANALYSIS_NAME, "-r", str(smoke_dir), "--", *pin_prefix, *target],
            capture_output=True, text=True, timeout=180,
        )
    except Exception as exc:  # noqa: BLE001
        result.error(f"Smoke test de uarch-exploration (collect) fallo al ejecutarse: {exc}")
        result.summary["uarch_exploration_funcional"] = "no"
        return

    if collect.returncode != 0 or re.search(r"^\s*vtune:\s*Error", collect.stdout + collect.stderr, re.MULTILINE):
        result.error(
            "Smoke test de uarch-exploration termino con error -- esto es "
            "el sintoma directo de un permiso o config faltante (PMU no "
            f"accesible, perf_event_paranoid muy restrictivo, etc). Codigo="
            f"{collect.returncode}. stderr:\n{collect.stderr.strip()[-1000:]}"
        )
        result.summary["uarch_exploration_funcional"] = "no"
        return

    try:
        report = subprocess.run(
            [vtune_path, "-report", "summary", "-r", str(smoke_dir)],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        result.error(f"Smoke test de uarch-exploration (report) fallo al ejecutarse: {exc}")
        result.summary["uarch_exploration_funcional"] = "no"
        return

    text = report.stdout
    populated = []
    missing = []
    for label in UARCH_TOP_LEVEL_LABELS:
        m = re.search(rf"^[ \t]*{re.escape(label)}:\s*(N/A|[\d.]+)", text, re.MULTILINE)
        if m and m.group(1) != "N/A":
            populated.append(label)
        else:
            missing.append(label)

    collector_m = re.search(r"^Collector Type:\s*(.+)$", text, re.MULTILINE)
    result.summary["collector_type"] = collector_m.group(1).strip() if collector_m else "no_reportado"

    if len(populated) == len(UARCH_TOP_LEVEL_LABELS):
        result.summary["uarch_exploration_funcional"] = "si"
        result.summary["categorias_top_level_pobladas"] = ", ".join(populated)
    else:
        result.summary["uarch_exploration_funcional"] = "parcial" if populated else "no"
        result.error(
            "El smoke test corrio sin error, pero el reporte no trae las 4 "
            "categorias de Nivel 1 pobladas con numeros reales. Pobladas: "
            f"{populated or 'ninguna'}. Faltantes/NA: {missing}. Esto es "
            "exactamente el patron esperado si el permiso nuevo cubre "
            "ejecucion del analisis pero no acceso completo a los eventos "
            "de PMU que TMAM necesita -- no tratar esto como bug del "
            "parser, es la señal real de que falta algo. Guardando "
            f"'{smoke_dir}/summary_raw.txt' para inspeccion manual."
        )
        (smoke_dir / "summary_raw.txt").write_text(text)
        return

    shutil.rmtree(smoke_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("./.preflight_uarch_tmp"))
    parser.add_argument("--core-range", type=str, default=D6_CORE_RANGE)
    parser.add_argument("--json-out", type=Path, default=None,
                         help="Si se pasa, ademas del texto imprime/guarda el resultado en JSON.")
    parser.add_argument("--skip-smoke", action="store_true",
                         help="Salta el smoke test real (mas rapido, pero no prueba contenido -- "
                              "solo usar para depuracion rapida del resto de checks).")
    args = parser.parse_args(argv)

    lo, hi = (int(x) for x in args.core_range.split("-"))
    cores = list(range(lo, hi + 1))

    result = PreflightResult()

    vtune_path = check_vtune_in_path(result)
    check_vtune_version(result, vtune_path)
    check_uarch_listed(result, vtune_path)
    check_perf_event_paranoid(result)
    check_kptr_restrict(result)
    check_cap_perfmon(result)
    check_cpu_architecture(result)
    check_slurm_context(result)
    check_affinity(result, cores)
    check_foreign_processes(result, cores)

    if not args.skip_smoke:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        taskset_path = shutil.which("taskset")
        pin_prefix = [taskset_path, "-c", args.core_range] if taskset_path else []
        if not taskset_path:
            result.warn("'taskset' no esta en PATH -- el smoke test corre sin pin explicito de cores.")
        try:
            check_uarch_smoke(result, vtune_path, args.work_dir, pin_prefix)
        finally:
            shutil.rmtree(args.work_dir, ignore_errors=True)
    else:
        result.summary["uarch_exploration_funcional"] = "no_verificado (--skip-smoke)"

    print("=== Preflight Microarchitecture Exploration -- paccaA100 ===")
    for key, value in result.summary.items():
        print(f"{key}: {value}")
    print(f"\nErrores bloqueantes: {len(result.errors)}")
    for e in result.errors:
        print(f"  - {e}")
    print(f"\nAdvertencias: {len(result.warnings)}")
    for w in result.warnings:
        print(f"  - {w}")

    if args.json_out:
        args.json_out.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    return 1 if result.errors else 0


if __name__ == "__main__":
    sys.exit(main())
