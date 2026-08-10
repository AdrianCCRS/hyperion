#!/usr/bin/env python3
"""Orquestador del pipeline VTune para el nodo Cartagena (paccaA100).

Ver PLAN.md Fase 3. Corre baseline + VTune (Hotspots HW + HPC Performance
Characterization) sobre los binarios NPB descubiertos en --bin-dir, mas una
corrida de calibracion opcional sobre los binarios ancla (STREAM/DGEMM) en
--anchor-dir. Produce la estructura de directorios de la Fase 3.2.

Esta fase NO parsea metricas ni clasifica (eso es vtune_parser.py y
classifier.py, Fase 4-6) -- si esos modulos ya existen en el mismo directorio,
se usan al final para escribir consolidated_results.csv; si no, la campaña
de recoleccion igual se completa y se deja una nota clara de que la
consolidacion queda pendiente.

Debe correr dentro de una reserva Slurm exclusiva en paccaA100, con los
modulos de VTune ya cargados. No lanzar la campaña completa (todos los
kernels x todas las clases x varias repeticiones) dentro de una sesion
interactiva de Claude Code -- eso va por sbatch (D7, ver CLAUDE.md).
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
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

KERNEL_CLASS_RE = re.compile(r"^([a-z]+)\.([A-Za-z])\.x$")
VERIFICATION_RE = re.compile(r"verification.*successful", re.IGNORECASE)

D6_OMP_PLACES = "cores"
D6_OMP_PROC_BIND = "close"
# 2026-08-07: alineado con el dominio real que usa el orquestador principal
# de Hyperion en este mismo nodo (campaign_pacca_ref.yaml: delegated_cpus=
# 0-5, collector_cpu=6, consumer_cpu=7) para que las corridas de VTune sean
# comparables kernel-por-kernel con las corridas del orquestador -- antes
# era 8 (todo el socket 0), ver context/02_decisiones.md D6 (actualizada).
D6_DEFAULT_THREADS = 6
D6_CORE_RANGE = "0-5"

ANCHOR_NAME_HINTS = {
    "stream": ("stream_omp", "stream_c", "stream"),
    "dgemm": ("dgemm_bench", "dgemm"),
}

STREAM_TRIAD_RE = re.compile(r"^Triad:\s+([\d.]+)", re.MULTILINE)
DGEMM_GFLOPS_RE = re.compile(r"GFLOP/s=([\d.]+)")

log = logging.getLogger("run_vtune_pipeline")


# --------------------------------------------------------------------------
# Utilidades de binarios (kernels + anclas)
# --------------------------------------------------------------------------


def _is_elf_binary(f: Path) -> bool:
    try:
        with f.open("rb") as fh:
            return fh.read(4) == b"\x7fELF"
    except OSError:
        return False


def discover_kernels(bin_dir: Path, kernel_filter: list[str] | None) -> list[tuple[str, str, Path]]:
    """Devuelve [(kernel, clase, path), ...] descubiertos en bin_dir.

    kernel_filter: lista de tokens tipo "ep" (todas las clases) o "ep.C"
    (solo esa clase). None = sin filtro, se toman todos los descubiertos.
    """
    found = []
    for f in sorted(bin_dir.glob("*.x")):
        if not os.access(f, os.X_OK):
            log.warning("Sin permiso de ejecucion, se ignora: %s", f)
            continue
        m = KERNEL_CLASS_RE.match(f.name)
        if not m:
            log.warning("Nombre de archivo no encaja con <kernel>.<clase>.x, se ignora: %s", f.name)
            continue
        kernel, klass = m.group(1), m.group(2)
        found.append((kernel, klass, f))

    if kernel_filter:
        wanted_kernels = set()
        wanted_pairs = set()
        for tok in kernel_filter:
            tok = tok.strip()
            if "." in tok:
                k, c = tok.split(".", 1)
                wanted_pairs.add((k.lower(), c.upper()))
            else:
                wanted_kernels.add(tok.lower())
        filtered = []
        for kernel, klass, f in found:
            if kernel in wanted_kernels or (kernel, klass) in wanted_pairs:
                filtered.append((kernel, klass, f))
        return filtered
    return found


def _best_anchor_match(candidates: list[Path], hints: tuple[str, ...]) -> Path | None:
    best: tuple[int, Path] | None = None
    for f in candidates:
        stem = f.stem.lower()
        for idx, hint in enumerate(hints):
            if stem == hint or stem.startswith(hint + "_"):
                if best is None or idx < best[0] or (idx == best[0] and str(f) < str(best[1])):
                    best = (idx, f)
                break
    return best[1] if best else None


def find_anchor_binaries(anchor_dir: Path) -> dict[str, Path | None]:
    found: dict[str, Path | None] = {name: None for name in ANCHOR_NAME_HINTS}
    if not anchor_dir.is_dir():
        return found
    elf_files = [
        f for f in anchor_dir.rglob("*")
        if f.is_file() and os.access(f, os.X_OK) and _is_elf_binary(f)
    ]
    for anchor, hints in ANCHOR_NAME_HINTS.items():
        candidates = [f for f in elf_files if any(h in f.stem.lower() for h in hints)]
        found[anchor] = _best_anchor_match(candidates, hints)
    return found


# --------------------------------------------------------------------------
# Ejecucion de procesos (baseline y VTune)
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
    """Corre un comando en su propio process group para poder matar todo el
    arbol de procesos si se pasa del timeout (evita hijos residuales)."""
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
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


def run_baseline(binary: Path, extra_args: list[str], env: dict, timeout: int,
                  is_npb_kernel: bool, pin_prefix: list[str] | None = None) -> RunOutcome:
    cmd = [*(pin_prefix or []), str(binary), *extra_args]
    outcome = _run(cmd, binary.parent, env, timeout)
    if not outcome.ok:
        return outcome
    if is_npb_kernel and not VERIFICATION_RE.search(outcome.stdout):
        outcome.ok = False
        outcome.error = "no se encontro 'verification...successful' (case-insensitive) en stdout"
    return outcome


def vtune_collect(vtune: str, analysis: str, knobs: list[str], result_dir: Path,
                   binary: Path, extra_args: list[str], env: dict, timeout: int,
                   pin_prefix: list[str] | None = None) -> RunOutcome:
    shutil.rmtree(result_dir, ignore_errors=True)
    cmd = [vtune, "-collect", analysis, *knobs, "-r", str(result_dir), "--",
           *(pin_prefix or []), str(binary), *extra_args]
    outcome = _run(cmd, binary.parent, env, timeout)
    if outcome.ok and re.search(r"^\s*vtune:\s*Error", outcome.stdout + outcome.stderr, re.MULTILINE):
        outcome.ok = False
        outcome.error = "vtune reporto 'Error' durante la coleccion"
    return outcome


def vtune_report(vtune: str, report_name: str, result_dir: Path, env: dict,
                  timeout: int, extra_opts: list[str] | None = None) -> tuple[RunOutcome, str]:
    cmd = [vtune, "-report", report_name, "-r", str(result_dir), *(extra_opts or [])]
    outcome = _run(cmd, None, env, timeout)
    return outcome, outcome.stdout


# --------------------------------------------------------------------------
# Un kernel / una repeticion
# --------------------------------------------------------------------------


def process_repetition(kernel: str, klass: str, binary: Path, rep: int, out_dir: Path,
                        vtune: str, env: dict, timeout: int,
                        skip_hotspots: bool, skip_hpc: bool,
                        pin_prefix: list[str] | None = None) -> dict:
    rep_dir = out_dir / kernel.upper() / f"class_{klass}" / f"rep_{rep:02d}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    row = {"kernel": kernel, "class": klass, "repetition": rep, "binary_path": str(binary)}

    log.info("[%s.%s rep %02d] baseline", kernel, klass, rep)
    baseline = run_baseline(binary, [], env, timeout, is_npb_kernel=True, pin_prefix=pin_prefix)
    (rep_dir / "baseline_stdout.txt").write_text(baseline.stdout)
    (rep_dir / "baseline_stderr.txt").write_text(baseline.stderr)
    (rep_dir / "baseline_meta.json").write_text(json.dumps({
        "kernel": kernel, "class": klass, "repetition": rep,
        "binary_path": str(binary),
        "ok": baseline.ok, "returncode": baseline.returncode,
        "elapsed_s": baseline.elapsed_s, "timed_out": baseline.timed_out,
        "error": baseline.error,
    }, indent=2))
    row["baseline_valid"] = baseline.ok
    row["baseline_elapsed_seconds"] = baseline.elapsed_s
    row["verification_successful"] = baseline.ok

    if not baseline.ok:
        log.warning("[%s.%s rep %02d] baseline invalido (%s) -- se omite VTune para esta repeticion",
                    kernel, klass, rep, baseline.error)
        row["quality_status"] = "invalid_baseline"
        row["hotspots_valid"] = False
        row["hpc_valid"] = False
        return row

    if not skip_hotspots:
        log.info("[%s.%s rep %02d] vtune hotspots (HW EBS)", kernel, klass, rep)
        hs_dir = rep_dir / "hotspots"
        hs = vtune_collect(vtune, "hotspots", ["-knob", "sampling-mode=hw"], hs_dir,
                            binary, [], env, timeout, pin_prefix=pin_prefix)
        row["hotspots_valid"] = hs.ok
        if hs.ok:
            _, txt = vtune_report(vtune, "hotspots", hs_dir, env, timeout)
            (rep_dir / "hotspots_summary.txt").write_text(txt)
            _, csv_txt = vtune_report(vtune, "hotspots", hs_dir, env, timeout, ["-format=csv"])
            (rep_dir / "hotspots_summary.csv").write_text(csv_txt)
        else:
            log.warning("[%s.%s rep %02d] hotspots fallo: %s", kernel, klass, rep, hs.error or hs.stderr[-300:])
    else:
        row["hotspots_valid"] = None

    if not skip_hpc:
        log.info("[%s.%s rep %02d] vtune hpc-performance", kernel, klass, rep)
        hpc_dir = rep_dir / "hpc"
        hpc = vtune_collect(vtune, "hpc-performance", [], hpc_dir, binary, [], env, timeout,
                            pin_prefix=pin_prefix)
        row["hpc_valid"] = hpc.ok
        if hpc.ok:
            _, txt = vtune_report(vtune, "summary", hpc_dir, env, timeout)
            (rep_dir / "hpc_summary.txt").write_text(txt)
            _, csv_txt = vtune_report(vtune, "summary", hpc_dir, env, timeout, ["-format=csv"])
            (rep_dir / "hpc_summary.csv").write_text(csv_txt)
            hw_outcome, hw_csv = vtune_report(vtune, "hw-events", hpc_dir, env, timeout, ["-format=csv"])
            if hw_outcome.ok:
                (rep_dir / "hpc_hw_events.csv").write_text(hw_csv)
            else:
                log.info("[%s.%s rep %02d] hw-events no disponible para este resultado (no bloqueante)",
                          kernel, klass, rep)
        else:
            log.warning("[%s.%s rep %02d] hpc-performance fallo: %s", kernel, klass, rep,
                        hpc.error or hpc.stderr[-300:])
    else:
        row["hpc_valid"] = None

    row["quality_status"] = "valid" if (row.get("hotspots_valid") is not False
                                         and row.get("hpc_valid") is not False) else "partial"
    return row


# --------------------------------------------------------------------------
# Calibracion (STREAM + DGEMM)
# --------------------------------------------------------------------------


def run_calibration(anchor_dir: Path, out_dir: Path, vtune: str, env: dict,
                     timeout: int, node: str, domain_config: str,
                     pin_prefix: list[str] | None = None) -> dict:
    cal_dir = out_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "calibrated": False,
        "stream_bandwidth_mb_s": None,
        "dgemm_gflops": None,
        "node": node,
        "domain": domain_config,
        "source": "self-timed (wall clock), sin dependencia de contadores uncore",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": [],
    }

    anchors = find_anchor_binaries(anchor_dir)
    stream_bin, dgemm_bin = anchors["stream"], anchors["dgemm"]

    if stream_bin is None:
        summary["notes"].append(f"STREAM no encontrado en {anchor_dir}")
    else:
        log.info("Calibracion: STREAM baseline (%s)", stream_bin)
        base = run_baseline(stream_bin, [], env, timeout, is_npb_kernel=False, pin_prefix=pin_prefix)
        if base.ok:
            m = STREAM_TRIAD_RE.search(base.stdout)
            if m:
                summary["stream_bandwidth_mb_s"] = float(m.group(1))
            else:
                summary["notes"].append("STREAM corrio OK pero no se pudo extraer 'Triad:' de su stdout")
            hpc_dir = cal_dir / "stream_hpc"
            hpc = vtune_collect(vtune, "hpc-performance", [], hpc_dir, stream_bin, [], env, timeout,
                                 pin_prefix=pin_prefix)
            if hpc.ok:
                _, txt = vtune_report(vtune, "summary", hpc_dir, env, timeout)
                (cal_dir / "stream_hpc_summary.txt").write_text(txt)
                _, csv_txt = vtune_report(vtune, "summary", hpc_dir, env, timeout, ["-format=csv"])
                (cal_dir / "stream_hpc_summary.csv").write_text(csv_txt)
        else:
            summary["notes"].append(f"STREAM baseline invalido: {base.error}")

    if dgemm_bin is None:
        summary["notes"].append(f"DGEMM no encontrado en {anchor_dir}")
    else:
        log.info("Calibracion: DGEMM baseline (%s)", dgemm_bin)
        dgemm_env = dict(env)
        lib_dir = dgemm_bin.parent.parent / "lib"
        if lib_dir.is_dir():
            dgemm_env["LD_LIBRARY_PATH"] = f"{lib_dir}:{dgemm_env.get('LD_LIBRARY_PATH', '')}"
        dgemm_env.setdefault("OPENBLAS_NUM_THREADS", env.get("OMP_NUM_THREADS", str(D6_DEFAULT_THREADS)))
        base = run_baseline(dgemm_bin, ["4096", "5"], dgemm_env, timeout, is_npb_kernel=False,
                             pin_prefix=pin_prefix)
        if base.ok:
            m = DGEMM_GFLOPS_RE.search(base.stdout)
            if m:
                summary["dgemm_gflops"] = float(m.group(1))
            else:
                summary["notes"].append("DGEMM corrio OK pero no se pudo extraer 'GFLOP/s=' de su stdout")
            hpc_dir = cal_dir / "dgemm_hpc"
            hpc = vtune_collect(vtune, "hpc-performance", [], hpc_dir, dgemm_bin, ["4096", "5"],
                                 dgemm_env, timeout, pin_prefix=pin_prefix)
            if hpc.ok:
                _, txt = vtune_report(vtune, "summary", hpc_dir, env, timeout)
                (cal_dir / "dgemm_hpc_summary.txt").write_text(txt)
                _, csv_txt = vtune_report(vtune, "summary", hpc_dir, env, timeout, ["-format=csv"])
                (cal_dir / "dgemm_hpc_summary.csv").write_text(csv_txt)
        else:
            summary["notes"].append(f"DGEMM baseline invalido: {base.error}")

    summary["calibrated"] = summary["stream_bandwidth_mb_s"] is not None and summary["dgemm_gflops"] is not None
    (cal_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------
# Consolidado (Fase 5) -- consolidated_results.csv, consolidated_by_kernel.csv,
# vectorization_detail.csv, classification_summary.md
# --------------------------------------------------------------------------

CONSOLIDATED_COLUMNS = [
    "campaign_id", "timestamp", "hostname", "slurm_job_id", "kernel", "class", "binary_path",
    "binary_checksum", "repetition", "threads", "domain_config",
    "baseline_valid", "verification_successful", "baseline_elapsed_seconds",
    "hotspots_valid", "hpc_valid",
    "dominant_function", "dominant_function_percentage",
    "cpi", "ipc_estimated", "dp_gflops",
    "memory_bound_pct", "dram_bound_pct_or_na", "cache_bound_pct",
    "average_frequency_ghz", "physical_core_utilization_pct", "numa_remote_access_pct",
    "classification_vtune_native", "classification_confidence", "classification_justification",
    "roofline_vs_ceilings_pct_compute", "ceilings_source",
    "quality_status", "error_message", "orchestrator_label",
]

VECTORIZATION_COLUMNS = [
    "kernel", "class", "repetition",
    "sp_gflops", "vectorization_pct", "packed_128_pct", "packed_256_pct", "packed_512_pct",
    "fp_uops_pct", "non_fp_uops_pct", "fp_arith_mem_read_ratio", "fp_arith_mem_write_ratio",
    "instructions_retired", "dominant_function_cpu_time",
]

BY_KERNEL_METRICS = ("dp_gflops", "memory_bound_pct", "dram_bound_pct_or_na", "cpi")


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _read_anchor(cal_dir: Path, name: str) -> dict:
    from vtune_parser import parse_summary_text
    txt_path = cal_dir / f"{name}_hpc_summary.txt"
    if not txt_path.exists():
        return {}
    return parse_summary_text(txt_path.read_text())


def build_consolidated_rows(output_dir: Path, campaign_metadata: dict, margin: float
                             ) -> tuple[list[dict], list[dict]]:
    """Camina la estructura de Fase 3.2 (<KERNEL>/class_<C>/rep_<NN>/) y
    reconstruye una fila por repeticion, sin depender de estado en memoria
    de la corrida que la genero -- se puede correr como paso separado
    contra cualquier output-dir ya recolectado."""
    from classifier import clasificar_nativo, eje_techos, nota_revision_manual
    from vtune_parser import parse_hotspots_text, parse_summary_text

    cal_dir = output_dir / "calibration"
    ancla_compute = _read_anchor(cal_dir, "dgemm")
    ancla_memoria = _read_anchor(cal_dir, "stream")
    ceilings_source = (
        "self-timed (wall clock), sin dependencia de contadores uncore"
        if ancla_compute and ancla_memoria else "no calibrado (--skip-calibration o ancla invalida)"
    )
    dgemm_gflops_ref = None
    cal_summary = _read_json(cal_dir / "calibration_summary.json") or {}
    dgemm_gflops_ref = cal_summary.get("dgemm_gflops")

    rows: list[dict] = []
    vector_rows: list[dict] = []

    skip_dirs = {"calibration", "logs"}
    for kernel_dir in sorted(p for p in output_dir.iterdir() if p.is_dir() and p.name not in skip_dirs):
        for class_dir in sorted(kernel_dir.glob("class_*")):
            klass = class_dir.name.removeprefix("class_")
            for rep_dir in sorted(class_dir.glob("rep_*")):
                baseline_meta = _read_json(rep_dir / "baseline_meta.json") or {}
                kernel = baseline_meta.get("kernel", kernel_dir.name.lower())
                rep = baseline_meta.get("repetition")
                binary_path = baseline_meta.get("binary_path", "")

                hpc_txt = rep_dir / "hpc_summary.txt"
                hs_txt = rep_dir / "hotspots_summary.txt"
                hpc = parse_summary_text(hpc_txt.read_text()) if hpc_txt.exists() else {}
                hs = parse_hotspots_text(hs_txt.read_text()) if hs_txt.exists() else {
                    "dominant_function": None, "dominant_function_percentage": None,
                    "instructions_retired": None, "dominant_function_cpu_time": None,
                }

                baseline_valid = bool(baseline_meta.get("ok"))
                hotspots_valid = hs_txt.exists()
                hpc_valid = hpc_txt.exists()

                if not baseline_valid:
                    quality_status = "invalid_baseline"
                elif hotspots_valid and hpc_valid:
                    quality_status = "valid"
                else:
                    quality_status = "partial"

                if baseline_valid and hpc:
                    clase, confianza, justificacion = clasificar_nativo(
                        hpc, ancla_compute, ancla_memoria, margen=margin
                    )
                else:
                    clase, confianza, justificacion = "invalid", "NA", "Baseline invalido o hpc-performance no disponible"

                nota = nota_revision_manual(kernel, clase)
                if nota:
                    justificacion = f"{justificacion} | {nota}"

                cpi = hpc.get("cpi")
                row = {
                    "campaign_id": campaign_metadata.get("campaign_id"),
                    "timestamp": campaign_metadata.get("timestamp"),
                    "hostname": campaign_metadata.get("hostname"),
                    "slurm_job_id": campaign_metadata.get("slurm_job_id"),
                    "kernel": kernel,
                    "class": klass,
                    "binary_path": binary_path,
                    "binary_checksum": _sha256(Path(binary_path)) if binary_path else None,
                    "repetition": rep,
                    "threads": campaign_metadata.get("threads"),
                    "domain_config": campaign_metadata.get("domain_config"),
                    "baseline_valid": baseline_valid,
                    "verification_successful": baseline_valid,
                    "baseline_elapsed_seconds": baseline_meta.get("elapsed_s"),
                    "hotspots_valid": hotspots_valid,
                    "hpc_valid": hpc_valid,
                    "dominant_function": hs.get("dominant_function"),
                    "dominant_function_percentage": hs.get("dominant_function_percentage"),
                    "cpi": cpi,
                    "ipc_estimated": (1.0 / cpi) if cpi else None,
                    "dp_gflops": hpc.get("dp_gflops"),
                    "memory_bound_pct": hpc.get("memory_bound_pct"),
                    "dram_bound_pct_or_na": hpc.get("dram_bound_pct_or_na"),
                    "cache_bound_pct": hpc.get("cache_bound_pct"),
                    "average_frequency_ghz": hpc.get("average_frequency_ghz"),
                    "physical_core_utilization_pct": hpc.get("physical_core_utilization_pct"),
                    "numa_remote_access_pct": hpc.get("numa_remote_access_pct"),
                    "classification_vtune_native": clase,
                    "classification_confidence": confianza,
                    "classification_justification": justificacion,
                    "roofline_vs_ceilings_pct_compute": eje_techos(hpc.get("dp_gflops"), dgemm_gflops_ref),
                    "ceilings_source": ceilings_source,
                    "quality_status": quality_status,
                    "error_message": baseline_meta.get("error"),
                    "orchestrator_label": "NA",
                }
                rows.append(row)

                vector_rows.append({
                    "kernel": kernel, "class": klass, "repetition": rep,
                    "sp_gflops": hpc.get("sp_gflops"),
                    "vectorization_pct": hpc.get("vectorization_pct"),
                    "packed_128_pct": hpc.get("packed_128_pct"),
                    "packed_256_pct": hpc.get("packed_256_pct"),
                    "packed_512_pct": hpc.get("packed_512_pct"),
                    "fp_uops_pct": hpc.get("fp_uops_pct"),
                    "non_fp_uops_pct": hpc.get("non_fp_uops_pct"),
                    "fp_arith_mem_read_ratio": hpc.get("fp_arith_mem_read_ratio"),
                    "fp_arith_mem_write_ratio": hpc.get("fp_arith_mem_write_ratio"),
                    "instructions_retired": hs.get("instructions_retired"),
                    "dominant_function_cpu_time": hs.get("dominant_function_cpu_time"),
                })

    return rows, vector_rows


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_by_kernel_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["kernel"], row["class"]), []).append(row)

    by_kernel = []
    for (kernel, klass), grupo in sorted(grouped.items()):
        validas = [r for r in grupo if r["quality_status"] == "valid"]
        entry = {
            "kernel": kernel, "class": klass,
            "n_repeticiones": len(grupo),
            "n_validas": len(validas),
            "classification_vtune_native": (
                validas[0]["classification_vtune_native"] if validas else "invalid"
            ),
        }
        for metrica in BY_KERNEL_METRICS:
            valores = [r[metrica] for r in validas if r.get(metrica) is not None]
            if valores:
                media = statistics.mean(valores)
                entry[f"{metrica}_mean"] = media
                entry[f"{metrica}_median"] = statistics.median(valores)
                entry[f"{metrica}_std"] = statistics.stdev(valores) if len(valores) > 1 else 0.0
                entry[f"{metrica}_min"] = min(valores)
                entry[f"{metrica}_max"] = max(valores)
                entry[f"{metrica}_cv"] = (entry[f"{metrica}_std"] / media) if media else None
            else:
                for suf in ("mean", "median", "std", "min", "max", "cv"):
                    entry[f"{metrica}_{suf}"] = None
        by_kernel.append(entry)
    return by_kernel


def write_classification_summary(path: Path, campaign_metadata: dict, cal_summary: dict,
                                  by_kernel: list[dict], rows: list[dict]) -> None:
    lines = []
    lines.append(f"# Resumen de clasificación — campaña `{campaign_metadata.get('campaign_id')}`")
    lines.append("")
    lines.append(f"- Nodo: `{campaign_metadata.get('hostname')}`")
    lines.append(f"- Timestamp: {campaign_metadata.get('timestamp')}")
    lines.append(f"- Slurm job: {campaign_metadata.get('slurm_job_id')}")
    lines.append(f"- Dominio: `{campaign_metadata.get('domain_config')}` "
                 f"({campaign_metadata.get('threads')} threads)")
    lines.append(f"- Repeticiones por kernel/clase: {campaign_metadata.get('repetitions')}")
    lines.append("")
    lines.append("## Calibración (D3-v3 — el veredicto nativo depende de estas anclas)")
    lines.append("")
    if cal_summary.get("calibrated"):
        lines.append(f"- STREAM (ancla de memoria): {cal_summary.get('stream_bandwidth_mb_s')} MB/s")
        lines.append(f"- DGEMM (ancla de cómputo): {cal_summary.get('dgemm_gflops')} GFLOP/s")
    else:
        lines.append("- **Calibración incompleta o `--skip-calibration` usado.** "
                      "`classification_vtune_native` sale `invalid` para todos los kernels "
                      "en esta campaña (ver D3-v3, `context/02_decisiones.md`).")
    lines.append("")
    lines.append("## Clasificación por kernel/clase")
    lines.append("")
    lines.append("| Kernel | Clase | Clasificación | Reps válidas | Memory Bound % (media) | "
                  "DP GFLOPS (media) |")
    lines.append("|---|---|---|---|---|---|")
    for e in by_kernel:
        lines.append(
            f"| {e['kernel']} | {e['class']} | {e['classification_vtune_native']} | "
            f"{e['n_validas']}/{e['n_repeticiones']} | "
            f"{e.get('memory_bound_pct_mean', 'NA')} | {e.get('dp_gflops_mean', 'NA')} |"
        )
    lines.append("")
    lines.append("## Kernels marcados para revisión manual (D4)")
    lines.append("")
    marcados = [r for r in rows if r["kernel"].lower() in ("ep", "is")
                and r["classification_vtune_native"] in ("memory_bound", "ambiguous")]
    if marcados:
        for r in marcados:
            lines.append(f"- `{r['kernel']}.{r['class']}` rep {r['repetition']}: "
                         f"`{r['classification_vtune_native']}` — {r['classification_justification']}")
    else:
        lines.append("Ninguno en esta campaña.")
    lines.append("")
    lines.append("## Restricciones conocidas de este nodo")
    lines.append("")
    lines.append("- No hay LIKWID ni ERT en este nodo (D1) — la única fuente de validación es VTune.")
    lines.append("- `classification_vtune_native` se calibra con STREAM/DGEMM de esta misma campaña "
                 "(D3-v3) — no es un veredicto \"sin calibración\" en el sentido en que se planteó "
                 "originalmente el proyecto; ver `context/02_decisiones.md` D3-v3 para el porqué.")
    lines.append("- `roofline_vs_ceilings_pct_compute` es una columna informativa aparte, nunca "
                 "fusionada con `classification_vtune_native` (D8).")
    lines.append("- No hay eje de memoria (AI = FLOP/byte) para los kernels NPB en este nodo — sin "
                 "uncore ni LIKWID no se puede medir su ancho de banda real (ver `context/04`).")
    path.write_text("\n".join(lines) + "\n")


def write_consolidated_outputs(output_dir: Path, campaign_metadata: dict, cal_summary: dict,
                                margin: float) -> int:
    """Fase 5. Devuelve el numero de filas con quality_status == 'valid'."""
    rows, vector_rows = build_consolidated_rows(output_dir, campaign_metadata, margin)
    write_csv(output_dir / "consolidated_results.csv", CONSOLIDATED_COLUMNS, rows)
    write_csv(output_dir / "vectorization_detail.csv", VECTORIZATION_COLUMNS, vector_rows)
    by_kernel = build_by_kernel_rows(rows)
    by_kernel_columns = ["kernel", "class", "n_repeticiones", "n_validas", "classification_vtune_native"]
    for metrica in BY_KERNEL_METRICS:
        by_kernel_columns += [f"{metrica}_{suf}" for suf in ("mean", "median", "std", "min", "max", "cv")]
    write_csv(output_dir / "consolidated_by_kernel.csv", by_kernel_columns, by_kernel)
    write_classification_summary(output_dir / "classification_summary.md", campaign_metadata,
                                  cal_summary, by_kernel, rows)
    return sum(1 for r in rows if r["quality_status"] == "valid")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin-dir", type=Path, required=True)
    p.add_argument("--anchor-dir", type=Path, default=None,
                   help="Default: igual a --bin-dir (layout real del nodo, ver Fase 1).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--kernels", type=str, default=None,
                   help="Filtro coma-separado: 'ep,cg' (todas las clases) o 'ep.C' (una clase).")
    p.add_argument("--threads", type=int, default=D6_DEFAULT_THREADS)
    p.add_argument("--core-range", type=str, default=D6_CORE_RANGE,
                   help="Rango para 'taskset -c', mismo dominio que delegated_cpus del "
                        "orquestador (campaign_pacca_ref.yaml) para comparabilidad.")
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--timeout", type=int, default=1800, help="Segundos por corrida individual.")
    p.add_argument("--skip-hotspots", action="store_true")
    p.add_argument("--skip-hpc", action="store_true")
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument("--margin", type=float, default=0.15,
                   help="Ancho de la zona ambigua (escala 0-1) en clasificar_nativo(), D3-v3.")
    return p.parse_args(argv)


def setup_logging(output_dir: Path) -> None:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(logs_dir / "pipeline.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Rutas absolutas desde el arranque: process_repetition/vtune_collect
    # fijan `cwd` al correr binarios, y una ruta relativa mezclada con un
    # cwd distinto termina buscando el binario duplicado bajo su propio cwd.
    args.bin_dir = args.bin_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.anchor_dir is not None:
        args.anchor_dir = args.anchor_dir.resolve()

    vtune = shutil.which("vtune")
    if vtune is None:
        print("ERROR: vtune no esta en PATH. Corre check_vtune.py primero "
              "(y carga los modulos: devtools/intel/oneapi/2023, vtune/2023.0.0).",
              file=sys.stderr)
        return 1

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        print(f"ERROR: {args.output_dir} ya existe y no esta vacio. Usa --overwrite "
              "o un --output-dir nuevo (no se sobrescribe silenciosamente, ver PLAN.md Fase 3.2).",
              file=sys.stderr)
        return 1
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(args.output_dir)

    anchor_dir = args.anchor_dir or args.bin_dir
    kernel_filter = [t.strip() for t in args.kernels.split(",")] if args.kernels else None

    if args.threads == D6_DEFAULT_THREADS and args.core_range == D6_CORE_RANGE:
        domain_config = f"{D6_DEFAULT_THREADS}cores_{D6_CORE_RANGE}_noSMT_aligned_orchestrator"
    else:
        domain_config = (f"custom_{args.threads}threads_cores={args.core_range}_"
                          f"places={D6_OMP_PLACES}_bind={D6_OMP_PROC_BIND}")
        log.warning("Threads/core-range difieren del default D6 (%d hilos, cores %s, alineado con "
                    "campaign_pacca_ref.yaml del orquestador) -- dominio marcado como '%s' en los "
                    "metadatos, no se mezcla silenciosamente con corridas estandar.",
                    D6_DEFAULT_THREADS, D6_CORE_RANGE, domain_config)

    taskset_path = shutil.which("taskset")
    if taskset_path is None:
        log.warning("'taskset' no esta en PATH -- no se puede fijar el dominio de cores a %s. "
                    "OMP_PLACES=cores igual restringe threads a cores fisicos, pero SIN taskset "
                    "el SO puede elegir cualquier subconjunto, no necesariamente %s (rompe la "
                    "comparabilidad con el orquestador).", args.core_range, args.core_range)
        pin_prefix: list[str] = []
    else:
        pin_prefix = [taskset_path, "-c", args.core_range]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["OMP_PLACES"] = D6_OMP_PLACES
    env["OMP_PROC_BIND"] = D6_OMP_PROC_BIND

    kernels = discover_kernels(args.bin_dir, kernel_filter)
    if not kernels:
        print(f"ERROR: no se encontraron binarios <kernel>.<clase>.x en {args.bin_dir} "
              f"(filtro={args.kernels!r}).", file=sys.stderr)
        return 1

    campaign_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    node = socket.gethostname()
    log.info("Campaña %s en %s -- %d binarios, %d repeticiones, dominio=%s",
              campaign_id, node, len(kernels), args.repetitions, domain_config)

    rows: list[dict] = []
    for kernel, klass, binary in kernels:
        for rep in range(1, args.repetitions + 1):
            row = process_repetition(kernel, klass, binary, rep, args.output_dir,
                                       vtune, env, args.timeout,
                                       args.skip_hotspots, args.skip_hpc, pin_prefix=pin_prefix)
            rows.append(row)

    if args.skip_calibration:
        log.info("Calibracion omitida (--skip-calibration). Umbrales/techo quedan sin calibrar.")
        cal_summary = {"calibrated": False, "notes": ["--skip-calibration usado"]}
        cal_dir = args.output_dir / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        (cal_dir / "calibration_summary.json").write_text(json.dumps(cal_summary, indent=2))
    else:
        cal_summary = run_calibration(anchor_dir, args.output_dir, vtune, env, args.timeout,
                                       node, domain_config, pin_prefix=pin_prefix)

    version_proc = subprocess.run([vtune, "--version"], capture_output=True, text=True, timeout=30)
    campaign_metadata = {
        "campaign_id": campaign_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": node,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "bin_dir": str(args.bin_dir),
        "anchor_dir": str(anchor_dir),
        "output_dir": str(args.output_dir),
        "threads": args.threads,
        "domain_config": domain_config,
        "repetitions": args.repetitions,
        "kernels_filter": args.kernels,
        "kernels_discovered": [f"{k}.{c}" for k, c, _ in kernels],
        "skip_hotspots": args.skip_hotspots,
        "skip_hpc": args.skip_hpc,
        "skip_calibration": args.skip_calibration,
        "vtune_version": (version_proc.stdout or version_proc.stderr).strip().splitlines()[:1],
        "calibration_calibrated": cal_summary.get("calibrated", False),
    }
    (args.output_dir / "campaign_metadata.json").write_text(json.dumps(campaign_metadata, indent=2))

    n_collected_valid = sum(1 for r in rows if r.get("quality_status") == "valid")
    log.info("Recoleccion completa: %d/%d repeticiones validas.", n_collected_valid, len(rows))

    try:
        n_consolidated_valid = write_consolidated_outputs(args.output_dir, campaign_metadata,
                                                            cal_summary, args.margin)
    except ImportError as exc:
        log.warning(
            "vtune_parser.py y/o classifier.py no se pudieron importar (%s) -- "
            "consolidated_results.csv, consolidated_by_kernel.csv, vectorization_detail.csv y "
            "classification_summary.md quedan pendientes. La recoleccion cruda ya esta completa "
            "en %s.", exc, args.output_dir,
        )
        return 0 if n_collected_valid > 0 else 1

    log.info("Consolidado escrito: %d filas 'valid' en consolidated_results.csv (Fase 5 completa).",
              n_consolidated_valid)

    return 0 if n_collected_valid > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
