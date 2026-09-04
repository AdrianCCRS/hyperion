#!/usr/bin/env python3
"""F1-CPU-003 -- diagnóstico de la traducción del evento genérico de caché.

`fase1_telemetria/postprocess.py` deriva `cache_miss_rate` de los eventos
GENÉRICOS `PERF_COUNT_HW_CACHE_MISSES` / `PERF_COUNT_HW_CACHE_REFERENCES`
(ver `common/telemetry/src/perf_reader.cpp`). El kernel traduce cada evento
genérico a un evento del PMU concreto, y esa traducción NO está documentada
como exclusivamente LLC/L3. Este script registra, de solo lectura, a qué
evento del PMU se traduce realmente en el nodo actual, para poder afirmar o
descartar la semántica de "último nivel" sin adivinar.

NO escribe nada en el hardware. NO lanza campañas. Produce un JSON de
evidencia. Pensado para correr en `paccaA100` dentro de la asignación de
Slurm (o en cualquier nodo, con `perf` disponible).

Uso:
    python3 fase1_telemetria/diagnose_cache_event.py --out-dir <dir>
    # opcional: --workload "<cmd corto de CPU>"  para un `perf stat` real
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str], timeout: int = 30) -> dict:
    """Ejecuta un comando de solo lectura y captura todo, sin lanzar."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "cmd": " ".join(cmd),
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
        }
    except FileNotFoundError:
        return {"cmd": " ".join(cmd), "error": "binario no encontrado"}
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd), "error": f"timeout tras {timeout}s"}


def _read_text(path: str) -> dict:
    p = Path(path)
    try:
        return {"path": path, "content": p.read_text().strip()}
    except OSError as exc:
        return {"path": path, "error": str(exc)}


def _cpuinfo_family_model() -> dict:
    info: dict[str, str] = {}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if ":" not in line:
                continue
            k, v = (s.strip() for s in line.split(":", 1))
            if k in {"vendor_id", "cpu family", "model", "model name", "stepping"} and k not in info:
                info[k] = v
            if len(info) >= 5:
                break
    except OSError as exc:
        info["error"] = str(exc)
    # Ice Lake-SP == family 6, model 106 (0x6A). El encoding raw del proyecto
    # (perf_reader.cpp) está gated a exactamente eso.
    info["is_ice_lake_sp"] = (
        info.get("cpu family") == "6" and info.get("model") == "106"
    )
    return info


def _pmu_events_dir_dump() -> dict:
    """Nombres de eventos del PMU core que el kernel expone en sysfs -- la
    fuente más directa de cómo se llama el evento al que se traduce."""
    base = Path("/sys/devices")
    out: dict[str, list[str]] = {}
    if not base.is_dir():
        return {"error": f"{base} no existe"}
    for pmu in sorted(base.glob("cpu*/events")):
        try:
            names = sorted(f.name for f in pmu.iterdir() if f.is_file())
        except OSError as exc:
            names = [f"<error: {exc}>"]
        # Solo lo relacionado con caché / LLC / L2 / L3, para no volcar cientos.
        rel = [n for n in names if re.search(r"cache|llc|l2|l3|mem_load", n, re.I)]
        out[str(pmu)] = rel or names[:0]
    return out


def _analyse_perf_list(perf_list_stdout: str) -> dict:
    """Busca en `perf list` la línea del alias genérico y a qué apunta."""
    result: dict[str, object] = {"cache_misses_line": None, "cache_references_line": None,
                                 "llc_lines": []}
    for line in perf_list_stdout.splitlines():
        low = line.lower()
        if "cache-misses" in low and "[hardware event]" in low:
            result["cache_misses_line"] = line.strip()
        if "cache-references" in low and "[hardware event]" in low:
            result["cache_references_line"] = line.strip()
        if re.search(r"\bllc\b|last.level|longest_lat_cache|l3", low):
            result["llc_lines"].append(line.strip())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workload", default=None,
                        help="Comando CPU corto para un `perf stat` real de "
                             "cache-misses/cache-references + eventos LLC crudos. "
                             "Solo lectura; no toca frecuencia.")
    parser.add_argument("--perf", default="perf")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    perf = shutil.which(args.perf) or args.perf
    report: dict[str, object] = {
        "schema": "f1-cpu-003/cache_event_diagnosis/1",
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "node": platform.node(),
        "uname": platform.uname()._asdict(),
        "cpuinfo": _cpuinfo_family_model(),
        "perf_binary": perf,
        "sysfs": {
            "perf_event_paranoid": _read_text("/proc/sys/kernel/perf_event_paranoid"),
            "nmi_watchdog": _read_text("/proc/sys/kernel/nmi_watchdog"),
        },
        "pmu_cache_event_names": _pmu_events_dir_dump(),
        "commands": {},
    }

    cmds = report["commands"]
    cmds["perf_version"] = _run([perf, "--version"])
    cmds["perf_list_cache"] = _run([perf, "list", "cache"])
    cmds["perf_list_hw"] = _run([perf, "list", "hw"])
    # -v hace que perf imprima la config raw a la que traduce cada alias.
    cmds["perf_stat_dryrun_verbose"] = _run(
        [perf, "stat", "-v", "-e", "cache-misses,cache-references", "true"]
    )

    if args.workload:
        wl = ["sh", "-c", args.workload]
        cmds["perf_stat_generic_vs_raw"] = _run(
            [perf, "stat", "-x", ",",
             "-e", "cache-misses,cache-references,"
                   "LLC-load-misses,LLC-store-misses,LLC-loads,LLC-stores",
             *wl],
            timeout=120,
        )

    pl = cmds["perf_list_cache"].get("stdout", "") + "\n" + cmds["perf_list_hw"].get("stdout", "")
    report["perf_list_analysis"] = _analyse_perf_list(pl)
    verbose = cmds["perf_stat_dryrun_verbose"].get("stderr", "") + \
        cmds["perf_stat_dryrun_verbose"].get("stdout", "")
    # `perf stat -v` imprime algo como: "cache-misses: 0x..., 0x..., 0x..."
    raw_hits = re.findall(r"cache-(?:misses|references):[^\n]*", verbose)
    report["perf_stat_verbose_raw_config"] = raw_hits

    # Veredicto conservador: solo "demostrado LLC" si hay evidencia textual
    # directa de que el alias se traduce a un evento de último nivel.
    evidence = json.dumps(report).lower()
    looks_llc = bool(re.search(r"longest_lat_cache|last.level|llc.*miss.*=.*cache-misses", evidence))
    report["verdict"] = {
        "llc_semantics_demonstrated": looks_llc,
        "recommended_feature_name": "cache_miss_rate",
        "note": (
            "El proyecto ya usa `cache_miss_rate` (F1-CPU-003). Este "
            "diagnóstico NO cambia nada por sí solo: si "
            "`llc_semantics_demonstrated` fuera True en varios nodos con "
            "evidencia textual clara, se podría documentar la equivalencia; "
            "mientras no lo sea, `cache_miss_rate` es el nombre correcto."
        ),
    }

    out = args.out_dir / "cache_event_diagnosis.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"escrito {out}")
    print(f"is_ice_lake_sp={report['cpuinfo'].get('is_ice_lake_sp')}  "
          f"llc_semantics_demonstrated={report['verdict']['llc_semantics_demonstrated']}")
    for h in raw_hits:
        print(f"  perf -v: {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
