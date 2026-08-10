#!/usr/bin/env python3
"""Campana de validacion cruzada VTune (Microarchitecture Exploration) para
Hyperion -- paccaA100 / Cartagena.

Que es y que NO es (ver docs/vtune/vtune_cross_validation.md seccion A/B):
esta campana NO produce el dataset de entrenamiento ni reemplaza el
etiquetado principal de Hyperion (PID+inherit sobre perf, ver
docs/retoma/Guia_Maestra_Fase1_DVFS.md). Es una segunda fuente de
observacion, independiente en pipeline de procesamiento y en modelo de
decision, usada para confirmar que los kernels elegidos para ese dataset
se comportan de verdad como compute_bound/memory_bound antes de confiar en
ellos.

Regla dura, sin excepcion: este script NUNCA mata, señaliza ni modifica
procesos de otros usuarios, y nunca solicita liberar recursos del nodo. Si
el nodo esta ocupado, el job de Slurm que lo envuelve simplemente espera en
cola (ver sbatch_vtune_validation.sh) -- este script asume que ya esta dentro
de una reserva propia cuando corre.

Debe correr dentro de sbatch_vtune_validation.sh (ver ese archivo), con los
modulos de VTune ya cargados por el script de shell que lo invoca -- este
script no hace 'module load' por si mismo (esa es responsabilidad del
entorno shell que lo lanza, igual que en pipelinevtune/run_vtune_pipeline.py).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import preflight_uarch
import uarch_parser
import validation_classifier

log = logging.getLogger("run_validation")

KERNEL_CLASS_RE = re.compile(r"^([a-z]+)\.([A-Za-z])\.x$")
VERIFICATION_RE = re.compile(r"verification.*successful", re.IGNORECASE)

# Mismo dominio D6 que pipelinevtune (context/02_decisiones.md D6): 6 cores
# fisicos, 0-5, sin SMT, alineado con delegated_cpus del orquestador
# principal en este nodo (orchestrator/schemas/campaign_pacca_ref.yaml) --
# se conserva a proposito para que esta validacion mida el mismo dominio
# que el resto del proyecto en paccaA100.
D6_CORE_RANGE = "0-5"
D6_THREADS = 6
D6_OMP_PLACES = "cores"
D6_OMP_PROC_BIND = "close"

# Kernels NPB minimos del proyecto (pipelinevtune/context/00_overview_hyperion.md).
# Solo se usan los que realmente existan como binario en --bin-dir -- nunca
# se inventa un nombre que no aparezca ahi.
DEFAULT_KERNELS = ("ep", "cg", "mg", "ft", "lu", "bt")
DEFAULT_CLASSES = ("C",)
DEFAULT_REPETITIONS = 2  # ver README.md: campana de validacion, no de dataset -- razon explicita ahi

ANCHOR_NAME_HINTS = {
    "stream": ("stream_omp", "stream_c", "stream"),
    "dgemm": ("dgemm_bench", "dgemm"),
}
STREAM_TRIAD_RE = re.compile(r"^Triad:\s+([\d.]+)", re.MULTILINE)

CONSOLIDATED_COLUMNS = [
    "campaign_id", "timestamp", "slurm_job_id", "hostname",
    "kernel", "class", "repetition", "is_anchor",
    "expected_behavior", "vtune_validation_class", "agrees_with_hint",
    "validation_confidence", "validation_reason",
    "retiring_pct", "frontend_bound_pct", "bad_speculation_pct", "backend_bound_pct",
    "memory_bound_pct", "core_bound_pct", "dram_bound_pct",
    "ipc", "cpi_rate", "elapsed_time_s", "average_cpu_frequency_ghz",
    "topdown_sum_pct", "quality_status",
    "baseline_valid", "binary_path", "binary_checksum", "result_dir",
]


# --------------------------------------------------------------------------
# Ejecucion de procesos (mismo patron que pipelinevtune/run_vtune_pipeline.py:
# process group propio para poder matar SOLO el arbol que este script creo
# si se pasa del timeout -- nunca toca nada que no haya lanzado el mismo)
# --------------------------------------------------------------------------


@dataclass
class RunOutcome:
    ok: bool
    returncode: int | None
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False
    error: str | None = None


def _run(cmd: list[str], cwd: Path | None, env: dict, timeout: int) -> RunOutcome:
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001
        return RunOutcome(False, None, "", "", time.time() - t0, error=str(exc))
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return RunOutcome(proc.returncode == 0, proc.returncode, stdout, stderr, time.time() - t0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        return RunOutcome(False, None, stdout, stderr, time.time() - t0, timed_out=True,
                           error=f"timeout tras {timeout}s")


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# --------------------------------------------------------------------------
# Descubrimiento de binarios (kernels NPB + anclas STREAM/DGEMM)
# --------------------------------------------------------------------------


def discover_kernels(bin_dir: Path, kernels: tuple[str, ...], classes: tuple[str, ...]) -> list[tuple[str, str, Path]]:
    found = []
    for f in sorted(bin_dir.glob("*.x")):
        if not os.access(f, os.X_OK):
            continue
        m = KERNEL_CLASS_RE.match(f.name)
        if not m:
            continue
        kernel, klass = m.group(1), m.group(2)
        if kernel.lower() in kernels and klass.upper() in classes:
            found.append((kernel, klass, f))
    return found


def _is_elf(f: Path) -> bool:
    try:
        with f.open("rb") as fh:
            return fh.read(4) == b"\x7fELF"
    except OSError:
        return False


def discover_anchors(anchor_dir: Path) -> dict[str, Path | None]:
    found: dict[str, Path | None] = {name: None for name in ANCHOR_NAME_HINTS}
    if not anchor_dir.is_dir():
        return found
    elf_files = [f for f in anchor_dir.rglob("*") if f.is_file() and os.access(f, os.X_OK) and _is_elf(f)]
    for anchor, hints in ANCHOR_NAME_HINTS.items():
        candidates = [f for f in elf_files if any(f.stem.lower() == h or f.stem.lower().startswith(h + "_") for h in hints)]
        found[anchor] = sorted(candidates, key=str)[0] if candidates else None
    return found


# --------------------------------------------------------------------------
# Una corrida: baseline + uarch-exploration + reportes + metadata
# --------------------------------------------------------------------------


def build_env(threads: int) -> dict:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PLACES"] = D6_OMP_PLACES
    env["OMP_PROC_BIND"] = D6_OMP_PROC_BIND
    return env


def process_workload(name: str, klass: str, binary: Path, rep: int, is_anchor: bool,
                      extra_args: list[str], out_dir: Path, vtune: str, env: dict,
                      timeout: int, pin_prefix: list[str], campaign_id: str, slurm_job_id: str,
                      hostname: str) -> dict:
    label = f"{name}.{klass}" if not is_anchor else name
    rep_dir = out_dir / f"{label}" / f"rep_{rep:02d}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    result_dir = rep_dir / "result"

    row: dict = {
        "campaign_id": campaign_id, "timestamp": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": slurm_job_id, "hostname": hostname,
        "kernel": name, "class": klass, "repetition": rep, "is_anchor": is_anchor,
        "expected_behavior": validation_classifier.expected_behavior_for(name),
        "binary_path": str(binary), "binary_checksum": _sha256(binary),
        "result_dir": str(result_dir),
    }

    log.info("[%s rep %02d] baseline (sin VTune)", label, rep)
    baseline_cmd = [*pin_prefix, str(binary), *extra_args]
    baseline = _run(baseline_cmd, binary.parent, env, timeout)
    baseline_valid = baseline.ok
    if baseline_valid and not is_anchor and not VERIFICATION_RE.search(baseline.stdout):
        baseline_valid = False
        baseline.error = "no se encontro 'verification...successful' en stdout"
    row["baseline_valid"] = baseline_valid
    if not baseline_valid:
        log.warning("[%s rep %02d] baseline invalido: %s -- se continua igual con VTune para no perder "
                    "la corrida, pero quality_status lo refleja", label, rep, baseline.error)

    log.info("[%s rep %02d] vtune -collect uarch-exploration", label, rep)
    shutil.rmtree(result_dir, ignore_errors=True)
    vtune_cmd = [vtune, "-collect", "uarch-exploration", "-r", str(result_dir), "--",
                 *pin_prefix, str(binary), *extra_args]
    collect = _run(vtune_cmd, binary.parent, env, timeout)
    row["exact_command"] = " ".join(vtune_cmd)

    if not collect.ok or re.search(r"^\s*vtune:\s*Error", collect.stdout + collect.stderr, re.MULTILINE):
        row.update({
            "quality_status": "vtune_collect_failed",
            "vtune_validation_class": "invalid",
            "validation_confidence": "NA",
            "validation_reason": f"vtune -collect uarch-exploration fallo: {collect.error or collect.stderr[-500:]}",
        })
        _write_metadata(rep_dir, row, baseline, collect, None, None, pin_prefix, env)
        return row

    report_txt_outcome = _run([vtune, "-report", "summary", "-r", str(result_dir)], None, env, timeout)
    (rep_dir / "summary.txt").write_text(report_txt_outcome.stdout)

    report_csv_outcome = _run([vtune, "-report", "summary", "-r", str(result_dir), "-format=csv"], None, env, timeout)
    (rep_dir / "report.csv").write_text(report_csv_outcome.stdout)

    hw_events_outcome = _run([vtune, "-report", "hw-events", "-r", str(result_dir), "-format=csv"], None, env, timeout)
    if hw_events_outcome.ok:
        (rep_dir / "raw_hw_events.csv").write_text(hw_events_outcome.stdout)
    else:
        log.warning("[%s rep %02d] 'vtune -report hw-events' fallo (no bloqueante, se omite raw_hw_events.csv): %s",
                    label, rep, hw_events_outcome.error)

    parsed = uarch_parser.parse_uarch_summary_text(report_txt_outcome.stdout)
    verdict = validation_classifier.classify(parsed)
    total = uarch_parser.top_level_sum(parsed)

    row.update({
        "vtune_validation_class": verdict.vtune_validation_class,
        "agrees_with_hint": validation_classifier.agrees_with_hint(
            verdict.vtune_validation_class, row["expected_behavior"]),
        "validation_confidence": verdict.validation_confidence,
        "validation_reason": verdict.validation_reason,
        "retiring_pct": parsed.get("retiring_pct"),
        "frontend_bound_pct": parsed.get("frontend_bound_pct"),
        "bad_speculation_pct": parsed.get("bad_speculation_pct"),
        "backend_bound_pct": parsed.get("backend_bound_pct"),
        "memory_bound_pct": parsed.get("memory_bound_pct"),
        "core_bound_pct": parsed.get("core_bound_pct"),
        "dram_bound_pct": parsed.get("dram_bound_pct"),
        "ipc": parsed.get("ipc"),
        "cpi_rate": parsed.get("cpi_rate"),
        "elapsed_time_s": parsed.get("elapsed_time_s"),
        "average_cpu_frequency_ghz": parsed.get("average_cpu_frequency_ghz"),
        "topdown_sum_pct": total,
        "quality_status": verdict.quality_status if baseline_valid else "baseline_invalid",
    })

    _write_metadata(rep_dir, row, baseline, collect, report_txt_outcome, hw_events_outcome, pin_prefix, env)
    return row


def _write_metadata(rep_dir: Path, row: dict, baseline: RunOutcome, collect: RunOutcome,
                     report_txt: RunOutcome | None, hw_events: RunOutcome | None,
                     pin_prefix: list[str], env: dict) -> None:
    """Punto 3/8F: comando exacto, parametros, afinidad, hilos y metadata de
    ejecucion suficiente para reproducir la corrida sin adivinar nada."""
    meta = {
        "kernel": row["kernel"], "class": row["class"], "repetition": row["repetition"],
        "is_anchor": row["is_anchor"], "binary_path": row["binary_path"],
        "binary_checksum": row["binary_checksum"],
        "exact_vtune_command": row.get("exact_command"),
        "pin_prefix": pin_prefix,
        "omp_num_threads": env.get("OMP_NUM_THREADS"),
        "omp_places": env.get("OMP_PLACES"),
        "omp_proc_bind": env.get("OMP_PROC_BIND"),
        "affinity_at_run_time": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "ok": baseline.ok, "returncode": baseline.returncode,
            "elapsed_s": baseline.elapsed_s, "timed_out": baseline.timed_out, "error": baseline.error,
        },
        "vtune_collect": {
            "ok": collect.ok, "returncode": collect.returncode,
            "elapsed_s": collect.elapsed_s, "timed_out": collect.timed_out, "error": collect.error,
        },
        "result_dir": row["result_dir"],
    }
    (rep_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (rep_dir / "baseline_stdout.txt").write_text(baseline.stdout)
    (rep_dir / "baseline_stderr.txt").write_text(baseline.stderr)


# --------------------------------------------------------------------------
# Consolidacion
# --------------------------------------------------------------------------


def write_consolidated_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CONSOLIDATED_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin-dir", type=Path, required=True,
                   help="Directorio remoto con los binarios NPB <kernel>.<clase>.x ya compilados "
                        "(mismo convenio que pipelinevtune, NO se compila nada aqui).")
    p.add_argument("--anchor-dir", type=Path, required=True,
                   help="Directorio con los binarios ancla STREAM/DGEMM ya compilados.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--kernels", type=str, default=",".join(DEFAULT_KERNELS))
    p.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASSES))
    p.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    p.add_argument("--threads", type=int, default=D6_THREADS)
    p.add_argument("--core-range", type=str, default=D6_CORE_RANGE)
    p.add_argument("--timeout", type=int, default=600, help="Segundos por corrida individual (baseline o vtune).")
    p.add_argument("--skip-anchors", action="store_true")
    p.add_argument("--skip-preflight", action="store_true",
                   help="NO recomendado -- salta el preflight de PMU/permisos. Solo para depuracion "
                        "del propio pipeline en una maquina sin VTune.")
    args = p.parse_args(argv)

    kernels = tuple(k.strip().lower() for k in args.kernels.split(",") if k.strip())
    classes = tuple(c.strip().upper() for c in args.classes.split(",") if c.strip())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    taskset_path = shutil.which("taskset")
    pin_prefix = [taskset_path, "-c", args.core_range] if taskset_path else []
    if not taskset_path:
        log.warning("'taskset' no esta en PATH -- OMP_PLACES=cores restringe a cores fisicos pero "
                    "no fija cuales; ver D6 en pipelinevtune/context/02_decisiones.md.")

    if not args.skip_preflight:
        log.info("=== Preflight Microarchitecture Exploration ===")
        pf_result = preflight_uarch.PreflightResult()
        vtune_path = preflight_uarch.check_vtune_in_path(pf_result)
        preflight_uarch.check_vtune_version(pf_result, vtune_path)
        preflight_uarch.check_uarch_listed(pf_result, vtune_path)
        preflight_uarch.check_perf_event_paranoid(pf_result)
        preflight_uarch.check_kptr_restrict(pf_result)
        preflight_uarch.check_cap_perfmon(pf_result)
        preflight_uarch.check_cpu_architecture(pf_result)
        preflight_uarch.check_slurm_context(pf_result)
        lo, hi = (int(x) for x in args.core_range.split("-"))
        cores = list(range(lo, hi + 1))
        preflight_uarch.check_affinity(pf_result, cores)
        preflight_uarch.check_foreign_processes(pf_result, cores)
        work_dir = args.output_dir / ".preflight_tmp"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            preflight_uarch.check_uarch_smoke(pf_result, vtune_path, work_dir, pin_prefix)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        (args.output_dir / "preflight_result.json").write_text(
            json.dumps(pf_result.to_dict(), indent=2, ensure_ascii=False))

        for key, value in pf_result.summary.items():
            log.info("preflight: %s = %s", key, value)
        for w in pf_result.warnings:
            log.warning("preflight: %s", w)
        if pf_result.errors:
            for e in pf_result.errors:
                log.error("preflight BLOQUEANTE: %s", e)
            log.error("Preflight fallo -- abortando la campana ANTES de tocar ningun kernel. "
                      "Ver %s para el detalle completo.", args.output_dir / "preflight_result.json")
            return 1
        vtune = vtune_path
    else:
        vtune = shutil.which("vtune")
        if vtune is None:
            log.error("--skip-preflight activo pero 'vtune' no esta en PATH -- no hay forma de continuar.")
            return 1

    bin_dir_kernels = discover_kernels(args.bin_dir, kernels, classes)
    if not bin_dir_kernels:
        log.error("No se encontro ningun binario <kernel>.<clase>.x en %s para kernels=%s clases=%s. "
                  "No se inventan binarios -- confirmar --bin-dir o el conjunto de kernels ya compilados.",
                  args.bin_dir, kernels, classes)
        return 1
    log.info("Kernels descubiertos: %s", [(k, c) for k, c, _ in bin_dir_kernels])

    anchors = {} if args.skip_anchors else discover_anchors(args.anchor_dir)
    for name, path in anchors.items():
        if path is None:
            log.warning("Ancla '%s' no encontrada en %s -- se omite del eje de referencia STREAM/DGEMM.",
                       name, args.anchor_dir)

    env = build_env(args.threads)
    campaign_id = f"vtune_uarch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_slurm")
    hostname = socket.gethostname()

    log.info("=== Campaña %s: %d kernels x %d anclas x %d repeticiones ===",
             campaign_id, len(bin_dir_kernels), sum(1 for v in anchors.values() if v), args.repetitions)

    rows: list[dict] = []

    for kernel, klass, binary in bin_dir_kernels:
        for rep in range(1, args.repetitions + 1):
            row = process_workload(kernel, klass, binary, rep, False, [], args.output_dir,
                                    vtune, env, args.timeout, pin_prefix, campaign_id, slurm_job_id, hostname)
            rows.append(row)
            write_consolidated_csv(rows, args.output_dir / "consolidated_validation.csv")

    if not args.skip_anchors:
        anchor_env = dict(env)
        for anchor_name, binary in anchors.items():
            if binary is None:
                continue
            extra_args = ["4096", "5"] if anchor_name == "dgemm" else []
            if anchor_name == "dgemm":
                anchor_env.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
            for rep in range(1, args.repetitions + 1):
                row = process_workload(anchor_name, "anchor", binary, rep, True, extra_args, args.output_dir,
                                        vtune, anchor_env, args.timeout, pin_prefix, campaign_id, slurm_job_id, hostname)
                rows.append(row)
                write_consolidated_csv(rows, args.output_dir / "consolidated_validation.csv")

    write_consolidated_csv(rows, args.output_dir / "consolidated_validation.csv")
    log.info("Campaña terminada. %d filas en %s", len(rows), args.output_dir / "consolidated_validation.csv")

    failed = [r for r in rows if r.get("vtune_validation_class") == "invalid"]
    if failed:
        log.warning("%d/%d corridas quedaron 'invalid' -- revisar consolidated_validation.csv y "
                    "summary.txt de cada una antes de usar esta campaña como validacion.",
                    len(failed), len(rows))

    return 0


if __name__ == "__main__":
    sys.exit(main())
