#!/usr/bin/env python3
"""Preflight de Intel Advisor para la campana de caracterizacion Roofline.

Ver docs/advisor/pipeline_advisor_diseno_y_arquitectura.md seccion 0 para
un hallazgo importante: el modelo de CPU real de paccaA100 NO coincide con
lo que documenta pipelinevtune/context/01_nodo_cartagena.md (Xeon Gold 5317
medido, 5315Y documentado). Por eso este preflight nunca asume topologia,
tamano de cache ni modelo de CPU -- los lee en vivo cada vez y los deja en
la metadata para que cualquier corrida sea interpretable sin adivinar.

Regla dura, igual que en raperezp/: nunca modifica configuracion global del
nodo, nunca mata ni interfiere con procesos de otros usuarios. Los checks
de PMU/capacidades son solo diagnostico.
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

REQUIRED_MODULE_PARENT = "devtools/intel/oneapi/2023"
REQUIRED_MODULE = "advisor/2023.0.0"


@dataclass
class PreflightResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {"summary": self.summary, "errors": self.errors, "warnings": self.warnings, "ok": not self.errors}


# --------------------------------------------------------------------------
# advisor en si
# --------------------------------------------------------------------------


def check_advisor_in_path(result: PreflightResult) -> str | None:
    path = shutil.which("advisor")
    if path is None:
        result.error(
            f"'advisor' no esta en PATH. Secuencia confirmada para este nodo: "
            f"'module load {REQUIRED_MODULE_PARENT}' (padre, obligatorio primero -- "
            f"modulo jerarquico, mismo patron que vtune/hotspots) seguido de "
            f"'module load {REQUIRED_MODULE}'."
        )
        result.summary["advisor_disponible"] = False
        return None
    result.summary["advisor_disponible"] = True
    result.summary["advisor_path"] = path
    return path


def check_advisor_version(result: PreflightResult, advisor_path: str | None) -> None:
    if advisor_path is None:
        result.summary["version"] = None
        return
    try:
        proc = subprocess.run([advisor_path, "--version"], capture_output=True, text=True, timeout=30)
    except Exception as exc:  # noqa: BLE001
        result.error(f"'advisor --version' fallo: {exc}")
        result.summary["version"] = None
        return
    text = (proc.stdout or proc.stderr).strip()
    version_line = text.splitlines()[0] if text else "(vacio)"
    result.summary["version"] = version_line
    if "2023" not in version_line:
        result.warn(
            f"La version reportada no contiene '2023' ({version_line!r}) -- "
            "los nombres de columna del CSV documentados en "
            "docs/advisor/estudio_intel_advisor_roofline.md se confirmaron contra "
            "2023.0.0 (build 613505), pueden diferir en otra version."
        )


def check_roofline_smoke(result: PreflightResult, advisor_path: str | None, work_dir: Path,
                          pin_prefix: list[str]) -> None:
    """Prueba real, no solo 'esta instalado': corre survey sobre un target
    trivial y confirma que produce un resultado abrible. No mide contenido
    cientifico (eso ya lo hace run_roofline_unit_test.sh sobre un kernel
    real) -- solo confirma que el mecanismo (instrumentacion + benchmarks
    de roofs) funciona en este nodo sin errores de permisos."""
    if advisor_path is None:
        result.summary["advisor_funcional"] = None
        return
    smoke_dir = work_dir / "preflight_advisor_smoke"
    shutil.rmtree(smoke_dir, ignore_errors=True)
    target = ["bash", "-c", "x=0; for i in $(seq 1 2000000); do x=$((x+i)); done"]
    try:
        collect = subprocess.run(
            [advisor_path, "--collect=survey", "--project-dir", str(smoke_dir), "--", *pin_prefix, *target],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        result.error(f"Smoke test de Advisor (collect=survey) fallo al ejecutarse: {exc}")
        result.summary["advisor_funcional"] = False
        return
    if collect.returncode != 0:
        result.error(
            "Smoke test de 'advisor --collect=survey' termino con error -- primer sospechoso: "
            f"permisos de perf_event/PMU. codigo={collect.returncode} stderr:\n{collect.stderr[-800:]}"
        )
        result.summary["advisor_funcional"] = False
        return
    roofs_out = smoke_dir / "roofs_smoke.csv"
    report = subprocess.run(
        [advisor_path, "--report=roofs", "--project-dir", str(smoke_dir), "--format=csv",
         "--report-output", str(roofs_out)],
        capture_output=True, text=True, timeout=60,
    )
    ok = report.returncode == 0 and roofs_out.exists() and roofs_out.stat().st_size > 0
    result.summary["advisor_funcional"] = ok
    if not ok:
        result.error(
            f"Smoke test: collect=survey funciono pero --report=roofs no produjo salida valida. "
            f"stderr:\n{report.stderr[-800:]}"
        )
    else:
        shutil.rmtree(smoke_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Host / CPU / NUMA / cache -- SIEMPRE leido en vivo, nunca asumido
# (ver seccion 0 del documento de diseno: el modelo de CPU documentado del
# nodo esta desactualizado/incorrecto, confirmado empiricamente)
# --------------------------------------------------------------------------


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def collect_host_info(result: PreflightResult) -> None:
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=15).stdout
    except Exception as exc:  # noqa: BLE001
        result.error(f"'lscpu' fallo: {exc}")
        return

    def _grab(label: str) -> str | None:
        m = re.search(rf"^{re.escape(label)}:\s*(.+)$", lscpu, re.MULTILINE)
        return m.group(1).strip() if m else None

    result.summary["hostname"] = _run_hostname()
    result.summary["cpu_model"] = _grab("Model name")
    result.summary["architecture"] = _grab("Architecture")
    result.summary["sockets"] = _grab("Socket(s)")
    result.summary["cores_per_socket"] = _grab("Core(s) per socket")
    result.summary["threads_per_core"] = _grab("Thread(s) per core")
    result.summary["cpus_total"] = _grab("CPU(s)")
    result.summary["numa_nodes"] = _grab("NUMA node(s)")
    for line in lscpu.splitlines():
        if line.strip().startswith("NUMA node") and "CPU(s)" in line:
            key, _, value = line.partition(":")
            result.summary[key.strip().lower().replace(" ", "_").replace("(s)", "")] = value.strip()


def _run_hostname() -> str | None:
    try:
        return subprocess.run(["hostname"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def read_llc_line_size_bytes(cpus: list[int]) -> int | None:
    """Mismo mecanismo exacto que orchestrator/node_profile.py::_cache_data():
    coherency_line_size del indice de cache mas alto (LLC) observado, nunca
    64 asumido. Implementacion independiente (advisor/ no importa
    orchestrator/, mismo desacoplamiento que Vtune/raperezp) pero misma
    logica: nivel mas alto gana, tamano mayor desempata."""
    llc_level = -1
    llc_kb = 0
    llc_line = None
    for cpu in cpus:
        cache_root = Path(f"/sys/devices/system/cpu/cpu{cpu}/cache")
        if not cache_root.exists():
            continue
        for index_dir in sorted(cache_root.glob("index*")):
            level_text = _read_text(index_dir / "level")
            type_text = _read_text(index_dir / "type") or ""
            if level_text is None or type_text == "Instruction":
                continue
            try:
                level = int(level_text)
            except ValueError:
                continue
            size_text = _read_text(index_dir / "size") or ""
            try:
                size_kb = int(size_text[:-1]) if size_text.endswith("K") else int(size_text) // 1024
            except ValueError:
                continue
            if level > llc_level or (level == llc_level and size_kb > llc_kb):
                llc_level = level
                llc_kb = size_kb
                line_text = _read_text(index_dir / "coherency_line_size")
                llc_line = int(line_text) if line_text and line_text.isdigit() else None
    return llc_line


def read_cpu_frequency_snapshot(cpus: list[int]) -> dict:
    """Lectura de SOLO LECTURA de scaling_cur_freq/governor/driver -- mismo
    archivo exacto que orchestrator/freqctl.py::read_observed_frequency_khz()
    ya usa para el orquestador principal (FRQ-10), aqui reimplementado sin
    importar orchestrator/ (mismo desacoplamiento que Vtune/raperezp).

    Este pipeline NUNCA escribe frecuencia -- no hay permiso de escritura
    confirmado para esta campana (P1 de
    docs/retoma/pacca/Solicitud_Permisos_Pacca_Unicartagena.md sigue sin
    resolverse al momento de escribir esto) y, aunque lo hubiera, fijar
    frecuencia no es responsabilidad de una campana de VALIDACION -- es
    responsabilidad de la campana de entrenamiento (`campaign_pacca_ref.yaml`,
    freqctl.py). Lo que si es responsabilidad de este pipeline es registrar
    honestamente a que frecuencia corrio cada medicion, para que un ridge
    point o un GFLOPS que varia entre corridas no se malinterprete como
    ruido de Advisor cuando en realidad es turbo/HWP moviendose."""
    per_cpu: dict[int, dict[str, object]] = {}
    for cpu in cpus:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
        cur_freq_khz = _read_text(base / "scaling_cur_freq")
        per_cpu[cpu] = {
            "scaling_cur_freq_khz": int(cur_freq_khz) if cur_freq_khz and cur_freq_khz.isdigit() else None,
            "scaling_governor": _read_text(base / "scaling_governor"),
            "scaling_driver": _read_text(base / "scaling_driver"),
        }
    freqs_khz = [v["scaling_cur_freq_khz"] for v in per_cpu.values() if v["scaling_cur_freq_khz"] is not None]
    governors = {v["scaling_governor"] for v in per_cpu.values() if v["scaling_governor"] is not None}
    drivers = {v["scaling_driver"] for v in per_cpu.values() if v["scaling_driver"] is not None}
    return {
        "per_cpu": per_cpu,
        "freq_mhz_min": (min(freqs_khz) / 1000.0) if freqs_khz else None,
        "freq_mhz_max": (max(freqs_khz) / 1000.0) if freqs_khz else None,
        "freq_mhz_mean": (sum(freqs_khz) / len(freqs_khz) / 1000.0) if freqs_khz else None,
        "governor": governors.pop() if len(governors) == 1 else (
            "mixed:" + ",".join(sorted(governors)) if governors else None),
        "scaling_driver": drivers.pop() if len(drivers) == 1 else (
            "mixed:" + ",".join(sorted(drivers)) if drivers else None),
    }


def collect_frequency_info(result: PreflightResult, cpus: list[int]) -> None:
    snap = read_cpu_frequency_snapshot(cpus)
    result.summary["governor"] = snap["governor"]
    result.summary["scaling_driver"] = snap["scaling_driver"]
    result.summary["freq_mhz_min"] = snap["freq_mhz_min"]
    result.summary["freq_mhz_max"] = snap["freq_mhz_max"]
    result.summary["freq_mhz_mean"] = snap["freq_mhz_mean"]
    if snap["freq_mhz_min"] is None:
        result.warn(
            "No se pudo leer scaling_cur_freq de ningun core del rango -- las corridas van a "
            "quedar sin frecuencia registrada en la metadata, no se puede diagnosticar despues "
            "si un resultado atipico se debio a turbo/HWP moviendose."
        )
    elif snap["freq_mhz_max"] - snap["freq_mhz_min"] > 200:
        result.warn(
            f"scaling_cur_freq varia {snap['freq_mhz_min']:.0f}-{snap['freq_mhz_max']:.0f} MHz "
            f"entre los cores del rango en este instante -- el nodo no esta en un estado de "
            f"frecuencia estable ahora mismo (turbo/HWP activo). No es bloqueante (no se pidio "
            f"fijar frecuencia para esta campana), pero puede hacer que dos repeticiones del "
            f"mismo kernel den GFLOPS/techos distintos por razones ajenas a Advisor."
        )
    if snap["governor"] and snap["governor"] != "performance":
        result.warn(
            f"Governor activo: {snap['governor']!r}, no 'performance' -- la frecuencia puede "
            "escalar hacia abajo bajo carga variable, afectando la comparabilidad entre "
            "kernels/repeticiones. Registrado en la metadata de cada corrida; no se fuerza a "
            "cambiar porque este pipeline no tiene (ni pidio) permiso de escritura de cpufreq."
        )


def collect_cache_info(result: PreflightResult, cpus: list[int]) -> None:
    line_size = read_llc_line_size_bytes(cpus)
    result.summary["llc_line_size_bytes"] = line_size
    if line_size is None:
        result.warn(
            "No se pudo leer coherency_line_size real de sysfs para ningun nivel de cache -- "
            "el pipeline NO debe caer de vuelta a asumir 64 bytes; si esto ocurre, "
            "classify_roofline.py debe fallar explicito en vez de asumir."
        )
    getconf_line = None
    try:
        getconf_line = subprocess.run(["getconf", "LEVEL1_DCACHE_LINESIZE"], capture_output=True,
                                       text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    result.summary["l1_line_size_bytes_getconf"] = getconf_line


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def check_affinity(result: PreflightResult, expected_cores: list[int]) -> None:
    """Confirma que la afinidad REAL del proceso (heredada de Slurm/srun,
    antes de que taskset entre a fijarla explicitamente por comando) ya
    cubre el dominio D6 pedido -- mismo check que preflight_uarch.py de
    raperezp/ (VTune) ya hacia y que este preflight no tenia. Sin esto,
    "taskset -c 0-5" podria estar fijando una afinidad sobre un dominio que
    Slurm nunca delego, sin que nada lo avisara."""
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        result.warn(f"No se pudo leer sched_getaffinity: {exc}")
        return
    result.summary["cpus_afinidad_real"] = ",".join(str(c) for c in allowed)
    missing = [c for c in expected_cores if c not in allowed]
    if missing:
        result.error(
            f"La afinidad real del proceso ({allowed}) no cubre el dominio D6 pedido "
            f"({expected_cores}). Faltan: {missing}. Slurm no asigno los cores esperados "
            "-- revisar la reserva (--exclusive/--cpus-per-task) antes de correr Advisor, "
            "taskset no puede fijar afinidad sobre cores que el cgroup del job no delego."
        )


def run_preflight(core_range: str, work_dir: Path, skip_smoke: bool) -> PreflightResult:
    result = PreflightResult()
    advisor_path = check_advisor_in_path(result)
    check_advisor_version(result, advisor_path)
    collect_host_info(result)

    lo, hi = (int(x) for x in core_range.split("-"))
    cpus = list(range(lo, hi + 1))
    collect_cache_info(result, cpus)
    collect_frequency_info(result, cpus)
    check_affinity(result, cpus)

    taskset_path = shutil.which("taskset")
    pin_prefix = [taskset_path, "-c", core_range] if taskset_path else []
    if not taskset_path:
        result.warn("'taskset' no esta en PATH -- no se puede fijar afinidad explicita para el smoke test.")

    if not skip_smoke:
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            check_roofline_smoke(result, advisor_path, work_dir, pin_prefix)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
    else:
        result.summary["advisor_funcional"] = "no_verificado (--skip-smoke)"

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-range", type=str, default="0-5")
    parser.add_argument("--work-dir", type=Path, default=Path("./.preflight_advisor_tmp"))
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args(argv)

    result = run_preflight(args.core_range, args.work_dir, args.skip_smoke)

    print("=== Preflight Intel Advisor -- host medido en vivo, ver seccion 0 del diseno ===")
    for key, value in result.summary.items():
        print(f"{key}: {value}")
    print(f"\nErrores bloqueantes: {len(result.errors)}")
    for e in result.errors:
        print(f"  - {e}")
    print(f"\nAdvertencias: {len(result.warnings)}")
    for w in result.warnings:
        print(f"  - {w}")

    if args.json_out:
        args.json_out.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str))

    return 1 if result.errors else 0


if __name__ == "__main__":
    sys.exit(main())
