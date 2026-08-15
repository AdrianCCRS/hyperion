#!/usr/bin/env python3
"""Orquestador de la campana de caracterizacion Roofline con Intel Advisor.

Ver docs/advisor/pipeline_advisor_diseno_y_arquitectura.md para el diseno
completo (por que esta estructura de resultados, por que estas decisiones
de clasificacion) antes de tocar este archivo.

Flujo por kernel x repeticion:
  1. project_nosim:   collect=survey + collect=tripcounts -flop (SIN
                       --enable-cache-simulation) -- referencia, nunca se
                       usa para clasificar.
  2. project_cachesim: collect=survey + collect=tripcounts -flop
                       --enable-cache-simulation -- fuente real de la
                       clasificacion (Self DRAM GB solo existe aqui).
  3. Reportes oficiales (--report=survey/roofs --format=csv) de ambos.
  4. classify_roofline sobre los datos de project_cachesim.
  5. Filas nuevas en consolidated_characterization.csv (kernel) y
     consolidated_loops.csv (loop, trazabilidad completa).

Regla dura, igual que en raperezp/: nunca mata ni interfiere con trabajos
de otros usuarios, nunca modifica configuracion global del nodo.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import advisor_report_parser as parser
import classify_roofline as clf
import kernel_registry
import preflight_advisor

log = logging.getLogger("run_characterization")

D_THREADS = 6
D_CORE_RANGE = "0-5"
D_OMP_PLACES = "cores"
D_OMP_PROC_BIND = "close"
DEFAULT_REPETITIONS = 2
# Intervalo de muestreo de frecuencia MIENTRAS corre la pasada que clasifica
# (tripcounts+cache-sim) -- ver _run()/characterize_one() y README.md seccion
# de afinidad/frecuencia para por que se reemplazo el esquema anterior
# (una lectura antes + una despues) por muestreo continuo.
FREQ_SAMPLE_INTERVAL_S = 1.0

KERNEL_CSV_COLUMNS = [
    "campaign_id", "timestamp", "hostname", "cpu_model",
    "kernel", "class", "repetition",
    "advisor_roofline_class", "confidence", "reason",
    "hot_loops_considered", "coverage_fraction_achieved", "hot_loops_total_self_time_s",
    "dominant_loop_name", "dominant_loop_source", "dominant_loop_ai", "dominant_loop_precision",
    "dominant_loop_i_ridge_advisor",
    "survey_elapsed_time_nosim_s", "tripcounts_elapsed_time_nosim_s",
    "survey_elapsed_time_cachesim_s", "tripcounts_elapsed_time_cachesim_s",
    "governor", "scaling_driver", "freq_mhz_during_run",
    "binary_checksum", "fflags", "has_optimization", "has_debug_symbols", "precision_source_hint",
    "project_dir_nosim", "project_dir_cachesim",
]

LOOP_CSV_COLUMNS = [
    "campaign_id", "kernel", "class", "repetition",
    "loop_name", "source_location", "self_time_s", "self_time_pct",
    "precision", "arithmetic_intensity", "self_gflop", "self_dram_gb",
    "verdict_class", "i_ridge_used", "reason", "is_hot_loop",
    "project_dir_cachesim",
]


# --------------------------------------------------------------------------
# Ejecucion (mismo patron que raperezp/run_validation.py: process group
# propio, timeout con kill del arbol completo, nunca de procesos ajenos)
# --------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path | None, env: dict, timeout: int,
         freq_sample_cpus: list[int] | None = None
         ) -> tuple[bool, int | None, str, str, float, list[float]]:
    """Si freq_sample_cpus se da, en vez de bloquear en un solo
    communicate(timeout=...), se sondea scaling_cur_freq real cada
    FREQ_SAMPLE_INTERVAL_S mientras el proceso sigue vivo -- reemplaza el
    esquema anterior (una lectura justo antes + una justo despues del
    proceso), que en corridas cortas (ej. ert, ~20s) daba una diferencia de
    cientos de MHz sin decir nada de a que frecuencia corrio el grueso de
    la ejecucion. Devuelve las muestras crudas (MHz, promedio entre cores
    del rango en cada instante); el promedio final se calcula en el
    llamador, nunca aqui, para que quede trazable cuantas muestras hubo."""
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                 start_new_session=True)
    except Exception as exc:  # noqa: BLE001
        return False, None, "", str(exc), time.time() - t0, []

    if not freq_sample_cpus:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return proc.returncode == 0, proc.returncode, stdout, stderr, time.time() - t0, []
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
            return False, None, stdout, f"timeout tras {timeout}s\n{stderr}", time.time() - t0, []

    freq_samples: list[float] = []
    while True:
        if proc.poll() is not None:
            break
        if time.time() - t0 > timeout:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
            return False, None, stdout, f"timeout tras {timeout}s\n{stderr}", time.time() - t0, freq_samples
        snap = preflight_advisor.read_cpu_frequency_snapshot(freq_sample_cpus)
        if snap.get("freq_mhz_mean") is not None:
            freq_samples.append(snap["freq_mhz_mean"])
        time.sleep(FREQ_SAMPLE_INTERVAL_S)
    stdout, stderr = proc.communicate()
    return proc.returncode == 0, proc.returncode, stdout, stderr, time.time() - t0, freq_samples


def _collect_pass(advisor: str, project_dir: Path, extra_collect_args: list[str], binary: Path,
                   pin_prefix: list[str], env: dict, timeout: int,
                   binary_args: list[str] | None = None,
                   freq_sample_cpus: list[int] | None = None) -> dict:
    cmd = [advisor, *extra_collect_args, "--project-dir", str(project_dir), "--",
           *pin_prefix, str(binary), *(binary_args or [])]
    ok, code, stdout, stderr, elapsed, freq_samples = _run(cmd, binary.parent, env, timeout, freq_sample_cpus)
    return {"ok": ok, "returncode": code, "elapsed_s": elapsed, "command": " ".join(cmd),
            "stderr_tail": stderr[-2000:] if stderr else "", "freq_mhz_samples": freq_samples}


def _report(advisor: str, report_name: str, project_dir: Path, out_path: Path,
            extra_report_args: list[str], env: dict, timeout: int) -> dict:
    cmd = [advisor, f"--report={report_name}", "--project-dir", str(project_dir),
           "--report-output", str(out_path), *extra_report_args]
    ok, code, stdout, stderr, elapsed, _freq_samples = _run(cmd, None, env, timeout)
    return {"ok": ok and out_path.exists(), "returncode": code, "elapsed_s": elapsed,
            "command": " ".join(cmd), "stderr_tail": stderr[-1000:] if stderr else ""}


def build_env(threads: int) -> dict:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PLACES"] = D_OMP_PLACES
    env["OMP_PROC_BIND"] = D_OMP_PROC_BIND
    return env


# --------------------------------------------------------------------------
# Una repeticion completa (nosim + cachesim) de un kernel
# --------------------------------------------------------------------------


def characterize_one(kernel: str, klass: str, binary: Path, rep: int, out_root: Path,
                      advisor: str, env: dict, pin_prefix: list[str], timeout: int,
                      config_dir: Path | None, host_info: dict, campaign_id: str,
                      cpus: list[int], extra_args: list[str] | None = None) -> tuple[dict, list[dict]]:
    extra_args = extra_args or []
    label = f"{kernel}.{klass}"
    kernel_dir = out_root / f"{label}_rep{rep:02d}"
    reports_dir = kernel_dir / "reports"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    build_info = kernel_registry.build_kernel_info(kernel, klass, binary, config_dir)
    for w in build_info.warnings:
        log.warning("[%s rep %02d] %s", label, rep, w)

    kernel_row: dict = {
        "campaign_id": campaign_id, "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": host_info.get("hostname"), "cpu_model": host_info.get("cpu_model"),
        "kernel": kernel, "class": klass, "repetition": rep,
        "binary_checksum": build_info.binary_checksum, "fflags": build_info.fflags,
        "has_optimization": build_info.has_optimization, "has_debug_symbols": build_info.has_debug_symbols,
        "precision_source_hint": build_info.precision_source_hint,
    }
    run_meta: dict = {"kernel": kernel, "class": klass, "repetition": rep,
                       "binary_path": str(binary), "build_info": asdict(build_info),
                       "host_info": host_info, "passes": {}}

    # --- project_nosim: referencia, nunca se usa para clasificar --------
    proj_nosim = kernel_dir / "project_nosim"
    shutil.rmtree(proj_nosim, ignore_errors=True)
    survey_nosim = _collect_pass(advisor, proj_nosim, ["--collect=survey"], binary, pin_prefix, env, timeout,
                                  extra_args)
    trip_nosim = _collect_pass(advisor, proj_nosim, ["--collect=tripcounts", "-flop"], binary,
                                pin_prefix, env, timeout, extra_args)
    run_meta["passes"]["survey_nosim"] = survey_nosim
    run_meta["passes"]["tripcounts_nosim"] = trip_nosim
    kernel_row["survey_elapsed_time_nosim_s"] = round(survey_nosim["elapsed_s"], 3)
    kernel_row["tripcounts_elapsed_time_nosim_s"] = round(trip_nosim["elapsed_s"], 3)

    survey_nosim_csv = reports_dir / "survey_nosim.csv"
    roofs_nosim_csv = reports_dir / "roofs_nosim.csv"
    if trip_nosim["ok"]:
        _report(advisor, "survey", proj_nosim, survey_nosim_csv,
                ["--format=csv", "--show-all-columns", "--mix", "--dynamic"], env, timeout)
        _report(advisor, "roofs", proj_nosim, roofs_nosim_csv, ["--format=csv"], env, timeout)

    # --- project_cachesim: fuente real de la clasificacion --------------
    proj_cachesim = kernel_dir / "project_cachesim"
    shutil.rmtree(proj_cachesim, ignore_errors=True)
    survey_cachesim = _collect_pass(advisor, proj_cachesim, ["--collect=survey"], binary,
                                     pin_prefix, env, timeout, extra_args)

    # governor/scaling_driver: no fluctuan durante la corrida (a diferencia
    # de la frecuencia) -- una lectura alcanza, no hace falta muestrearlos.
    freq_static = preflight_advisor.read_cpu_frequency_snapshot(cpus)

    # Frecuencia MUESTREADA EN VIVO mientras corre la pasada que realmente
    # alimenta la clasificacion (tripcounts+cache-sim, no la de survey) --
    # solo lectura, nunca se fija (ver docstring de
    # read_cpu_frequency_snapshot). Reemplaza el esquema anterior de una
    # lectura antes + una despues: en una corrida corta (ert, job 5114,
    # ~20s) ese esquema daba 1457/849 MHz (diferencia de 608 MHz) sin decir
    # nada de a que frecuencia corrio realmente el grueso de la ejecucion --
    # un solo promedio de muestras tomadas DURANTE la pasada es la cifra que
    # de verdad importa para interpretar GFLOPS/techos, no cuanto vario.
    trip_cachesim = _collect_pass(advisor, proj_cachesim,
                                   ["--collect=tripcounts", "-flop", "--enable-cache-simulation"],
                                   binary, pin_prefix, env, timeout, extra_args,
                                   freq_sample_cpus=cpus)
    freq_samples = trip_cachesim.get("freq_mhz_samples") or []
    freq_mhz_during_run = round(sum(freq_samples) / len(freq_samples), 1) if freq_samples else None

    run_meta["passes"]["survey_cachesim"] = survey_cachesim
    run_meta["passes"]["tripcounts_cachesim"] = trip_cachesim
    run_meta["freq_mhz_samples_during_tripcounts_cachesim"] = freq_samples
    kernel_row["survey_elapsed_time_cachesim_s"] = round(survey_cachesim["elapsed_s"], 3)
    kernel_row["tripcounts_elapsed_time_cachesim_s"] = round(trip_cachesim["elapsed_s"], 3)
    kernel_row["project_dir_nosim"] = str(proj_nosim)
    kernel_row["project_dir_cachesim"] = str(proj_cachesim)
    kernel_row["governor"] = freq_static.get("governor")
    kernel_row["scaling_driver"] = freq_static.get("scaling_driver")
    kernel_row["freq_mhz_during_run"] = freq_mhz_during_run
    if freq_mhz_during_run is None:
        log.warning("[%s.%s rep %02d] no se pudo muestrear frecuencia durante tripcounts+cache-sim "
                    "(0 muestras) -- revisar permisos de lectura de scaling_cur_freq.",
                    kernel, klass, rep)

    loop_rows: list[dict] = []

    if not trip_cachesim["ok"]:
        kernel_row.update({
            "advisor_roofline_class": "invalid", "confidence": "NA",
            "reason": f"La coleccion tripcounts+cache-simulation fallo: {trip_cachesim['stderr_tail'][-300:]}",
            "hot_loops_considered": 0, "coverage_fraction_achieved": None, "hot_loops_total_self_time_s": None,
            "dominant_loop_name": None, "dominant_loop_source": None, "dominant_loop_ai": None,
            "dominant_loop_precision": None, "dominant_loop_i_ridge_advisor": None,
        })
        (kernel_dir / "metadata.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False, default=str))
        return kernel_row, loop_rows

    survey_cachesim_csv = reports_dir / "survey_cachesim.csv"
    roofs_cachesim_csv = reports_dir / "roofs_cachesim.csv"
    roofline_html = reports_dir / "roofline_cachesim.html"
    _report(advisor, "survey", proj_cachesim, survey_cachesim_csv,
            ["--format=csv", "--show-all-columns", "--mix", "--dynamic"], env, timeout)
    _report(advisor, "roofs", proj_cachesim, roofs_cachesim_csv, ["--format=csv"], env, timeout)
    _report(advisor, "roofline", proj_cachesim, roofline_html, [], env, timeout)

    survey_rows = parser.parse_survey_csv(survey_cachesim_csv.read_text(errors="replace")) \
        if survey_cachesim_csv.exists() else []
    roofs = parser.parse_roofs_csv(roofs_cachesim_csv.read_text(errors="replace")) \
        if roofs_cachesim_csv.exists() else {}

    if not survey_rows or not roofs:
        kernel_row.update({
            "advisor_roofline_class": "invalid", "confidence": "NA",
            "reason": f"Reporte de survey ({len(survey_rows)} filas) o roofs ({len(roofs)} techos) vacio.",
            "hot_loops_considered": 0, "coverage_fraction_achieved": None, "hot_loops_total_self_time_s": None,
            "dominant_loop_name": None, "dominant_loop_source": None, "dominant_loop_ai": None,
            "dominant_loop_precision": None, "dominant_loop_i_ridge_advisor": None,
        })
        (kernel_dir / "metadata.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False, default=str))
        return kernel_row, loop_rows

    hot_loops, coverage, total_self_time = clf.select_hot_loops(survey_rows)
    hot_verdicts = [clf.classify_loop(row, roofs) for row in hot_loops]

    for row, verdict in zip(hot_loops, hot_verdicts):
        loop_rows.append({
            "campaign_id": campaign_id, "kernel": kernel, "class": klass, "repetition": rep,
            "loop_name": verdict.loop_name, "source_location": verdict.source_location,
            "self_time_s": verdict.self_time_s, "self_time_pct": verdict.self_time_pct,
            "precision": verdict.precision, "arithmetic_intensity": verdict.arithmetic_intensity,
            "self_gflop": verdict.self_gflop, "self_dram_gb": verdict.self_dram_gb,
            "verdict_class": verdict.verdict_class, "i_ridge_used": verdict.i_ridge_used,
            "reason": verdict.reason, "is_hot_loop": True,
            "project_dir_cachesim": str(proj_cachesim),
        })

    kernel_verdict = clf.aggregate_kernel_verdict(kernel, klass, hot_verdicts, coverage)

    dominant = max(hot_verdicts, key=lambda v: (v.self_time_s or 0.0)) if hot_verdicts else None
    kernel_row.update({
        "advisor_roofline_class": kernel_verdict.advisor_roofline_class,
        "confidence": kernel_verdict.confidence, "reason": kernel_verdict.reason,
        "hot_loops_considered": kernel_verdict.hot_loops_considered,
        "coverage_fraction_achieved": round(coverage, 3),
        "hot_loops_total_self_time_s": round(kernel_verdict.hot_loops_total_self_time_s, 4),
        "dominant_loop_name": dominant.loop_name if dominant else None,
        "dominant_loop_source": dominant.source_location if dominant else None,
        "dominant_loop_ai": dominant.arithmetic_intensity if dominant else None,
        "dominant_loop_precision": dominant.precision if dominant else None,
        "dominant_loop_i_ridge_advisor": dominant.i_ridge_used if dominant else None,
    })

    (kernel_dir / "metadata.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False, default=str))
    return kernel_row, loop_rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def write_csv(rows: list[dict], columns: list[str], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_existing_csv_rows(path: Path) -> list[dict]:
    """Si el CSV consolidado de una campana anterior ya existe en
    --output-dir, se precarga para que esta corrida AGREGUE filas en vez de
    pisarlo -- pensado para poder sumar anclas STREAM/DGEMM (u otro kernel
    suelto despues) al consolidado de una campana ya corrida, sin tener que
    fusionar CSVs a mano. No deduplica por kernel/clase/repeticion -- si se
    vuelve a correr exactamente la misma combinacion, queda una fila
    repetida a proposito (mejor eso que borrar silenciosamente una corrida
    anterior)."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin-dir", type=Path, required=True)
    p.add_argument("--config-dir", type=Path, required=True,
                   help="Directorio NPB3.4-OMP/config con make.def real, para verificar flags.")
    p.add_argument("--anchor-dir", type=Path, default=None,
                   help="Directorio con los binarios STREAM/DGEMM ya compilados (busqueda recursiva por "
                        "nombre real, ver kernel_registry.ANCHOR_NAME_HINTS). Opcional -- si no se pasa, "
                        "no se procesa ninguna ancla.")
    p.add_argument("--skip-anchors", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--kernels", type=str, default="")
    p.add_argument("--classes", type=str, default="")
    p.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    p.add_argument("--threads", type=int, default=D_THREADS)
    p.add_argument("--core-range", type=str, default=D_CORE_RANGE)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args(argv)

    kernels = tuple(k.strip().lower() for k in args.kernels.split(",") if k.strip()) or None
    classes = tuple(c.strip().upper() for c in args.classes.split(",") if c.strip()) or None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_preflight:
        pf = preflight_advisor.run_preflight(args.core_range, args.output_dir / ".preflight_tmp", False)
        (args.output_dir / "preflight_result.json").write_text(
            json.dumps(pf.to_dict(), indent=2, ensure_ascii=False, default=str))
        for k, v in pf.summary.items():
            log.info("preflight: %s = %s", k, v)
        for w in pf.warnings:
            log.warning("preflight: %s", w)
        if pf.errors:
            for e in pf.errors:
                log.error("preflight BLOQUEANTE: %s", e)
            return 1
        advisor = pf.summary["advisor_path"]
        host_info = {k: v for k, v in pf.summary.items()
                     if k in ("hostname", "cpu_model", "architecture", "sockets", "cores_per_socket",
                               "threads_per_core", "cpus_total", "numa_nodes", "llc_line_size_bytes")}
    else:
        advisor = shutil.which("advisor")
        if advisor is None:
            log.error("--skip-preflight activo pero 'advisor' no esta en PATH.")
            return 1
        host_info = {"hostname": socket.gethostname()}

    found = kernel_registry.discover_kernels(args.bin_dir, kernels, classes)
    anchors: dict[str, Path | None] = {}
    if args.anchor_dir and not args.skip_anchors:
        anchors = kernel_registry.discover_anchors(args.anchor_dir)
        for name, path in anchors.items():
            if path is None:
                log.warning("Ancla '%s' no encontrada en %s -- se omite.", name, args.anchor_dir)
    if not found and not any(anchors.values()):
        log.error("No se encontro ningun binario <kernel>.<clase>.x en %s ni ninguna ancla en %s.",
                  args.bin_dir, args.anchor_dir)
        return 1
    log.info("Kernels descubiertos: %s", [(k, c) for k, c, _ in found])
    if anchors:
        log.info("Anclas descubiertas: %s", {k: str(v) for k, v in anchors.items() if v})

    taskset_path = shutil.which("taskset")
    pin_prefix = [taskset_path, "-c", args.core_range] if taskset_path else []
    lo, hi = (int(x) for x in args.core_range.split("-"))
    cpus = list(range(lo, hi + 1))
    env = build_env(args.threads)
    campaign_id = f"advisor_roofline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

    kernel_csv = args.output_dir / "consolidated_characterization.csv"
    loop_csv = args.output_dir / "consolidated_loops.csv"
    kernel_rows: list[dict] = load_existing_csv_rows(kernel_csv)
    loop_rows: list[dict] = load_existing_csv_rows(loop_csv)
    if kernel_rows:
        log.info("%d filas existentes precargadas de %s -- esta corrida AGREGA, no reemplaza.",
                 len(kernel_rows), kernel_csv)

    for kernel, klass, binary in found:
        for rep in range(1, args.repetitions + 1):
            log.info("=== %s.%s rep %02d ===", kernel, klass, rep)
            k_row, l_rows = characterize_one(kernel, klass, binary, rep, args.output_dir, advisor, env,
                                              pin_prefix, args.timeout, args.config_dir, host_info, campaign_id,
                                              cpus)
            kernel_rows.append(k_row)
            loop_rows.extend(l_rows)
            write_csv(kernel_rows, KERNEL_CSV_COLUMNS, kernel_csv)
            write_csv(loop_rows, LOOP_CSV_COLUMNS, loop_csv)

    # Anclas STREAM/DGEMM: klass="anchor" (nunca A/B/C, no son parte de NPB),
    # medidas con el MISMO pipeline completo de Advisor (survey+tripcounts,
    # sin simular y con --enable-cache-simulation) -- nunca se usa el GB/s o
    # GFLOP/s que ellas mismas reportan por su cuenta.
    for name, binary in anchors.items():
        if binary is None:
            continue
        extra_args = kernel_registry.ANCHOR_EXTRA_ARGS.get(name, [])
        anchor_env = dict(env)
        if name == "dgemm":
            # dgemm_bench enlaza OpenBLAS dinamico -- mismo patron ya usado
            # en pipelinevtune/run_vtune_pipeline.py y la campana de VTune:
            # LD_LIBRARY_PATH apuntando a openBLAS-dgme/lib, hermano del
            # binario, y OPENBLAS_NUM_THREADS alineado con el resto (D6, 6).
            lib_dir = binary.parent.parent / "lib"
            anchor_env["LD_LIBRARY_PATH"] = f"{lib_dir}:{anchor_env.get('LD_LIBRARY_PATH', '')}"
            anchor_env["OPENBLAS_NUM_THREADS"] = str(args.threads)
        for rep in range(1, args.repetitions + 1):
            log.info("=== ancla %s (anchor) rep %02d ===", name, rep)
            k_row, l_rows = characterize_one(name, "anchor", binary, rep, args.output_dir, advisor, anchor_env,
                                              pin_prefix, args.timeout, None, host_info, campaign_id,
                                              cpus, extra_args)
            kernel_rows.append(k_row)
            loop_rows.extend(l_rows)
            write_csv(kernel_rows, KERNEL_CSV_COLUMNS, kernel_csv)
            write_csv(loop_rows, LOOP_CSV_COLUMNS, loop_csv)

    log.info("Campaña terminada: %d filas de kernel, %d filas de loop.", len(kernel_rows), len(loop_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
