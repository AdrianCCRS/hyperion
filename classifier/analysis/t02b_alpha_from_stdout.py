#!/usr/bin/env python3
"""T0.2b: recalcular alpha con la duracion que reporta el propio kernel.

Ver docs/general/PLAN_MAESTRO_FASE2.md, Anexo A.2.

POR QUE. Las duraciones usadas hasta ahora se sumaban sobre ventanas de
`windows.csv`, y eso da dos estimaciones y ninguna correcta:

  - FILTRADAS (solo ventanas que pasan el filtro de calidad de frecuencia):
    sesgadas, porque la retencion depende de la frecuencia. Medido en los
    NUEVE kernels, la retencion en F0 es menor que en F4 -- npb_mg pierde
    el 40 % de sus ventanas en F0. Eso subestima T(F0), infla el cociente
    T(f)/T(F0) y con el infla alpha.
  - CRUDAS (todas las ventanas escritas): incluyen tiempo corrido a una
    frecuencia distinta de la solicitada, que es justo lo que el filtro
    existe para excluir.

La duracion correcta es la que el binario imprime por stdout. NPB emite
"Time in seconds = X" y el catalogo ya declara el patron para extraerla en
`runtime_seconds_stdout_pattern`. Es inmune a cualquier filtrado de
ventanas porque no pasa por el pipeline de telemetria.

QUE DECIDE. El diagnostico entero descansa sobre "el alpha minimo del
dataset es 0.242 contra un umbral de 0.226". Si los alpha estan inflados de
forma sistematica, ese numero --y con el la afirmacion de que ningun kernel
alcanza el regimen viable-- hay que rehacerlo.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.analysis.gates_c1_c2_c3 import discover_runs  # noqa: E402
from classifier.features.align import fit_alpha  # noqa: E402

NOMINAL_MHZ = {"F0": 3200.0, "F1": 2600.0, "F2": 2000.0, "F3": 1400.0, "F4": 800.0}
BREAK_EVEN = 0.226


def load_patterns(catalog_path: Path) -> dict[str, str]:
    doc = yaml.safe_load(catalog_path.read_text())
    out = {}
    for entry in doc.get("kernels", []):
        pattern = entry.get("runtime_seconds_stdout_pattern")
        if pattern:
            out[entry["id"]] = pattern
    return out


# ARC-187: respaldo universal cuando el kernel no declara su propio patrón
# (p.ej. 3mm_omp: RAJAPerf imprime una tabla de metricas sin una etiqueta
# de "tiempo total" limpia). telemetry_kernel_launcher.cpp imprime esta
# linea de resumen SIEMPRE, para cualquier kernel, con el tiempo de pared
# de la corrida instrumentada medido por el propio arnes -- no depende de
# que el binario de terceros imprima nada. Es en nanosegundos, de ahi la
# conversion; NO es la misma magnitud que "windows.csv sumado" (esa ya se
# valido contra el stdout propio del kernel a <0.008 en alpha para los 7
# kernels que si tienen patron), pero sirve como fuente independiente.
_LAUNCHER_SUMMARY_RE = re.compile(r"telemetry_mean_ns=([0-9.]+)")


def runtime_from_stdout(run_dir: Path, pattern: str | None) -> float | None:
    stdout = run_dir / "stdout.txt"
    if not stdout.exists():
        return None
    text = stdout.read_text(errors="replace")
    if pattern:
        match = re.search(pattern, text)
        if match:
            try:
                value = float(match.group(1))
            except (TypeError, ValueError):
                value = None
            if value is not None and value > 0:
                return value
    fallback = _LAUNCHER_SUMMARY_RE.search(text)
    if fallback:
        try:
            ns = float(fallback.group(1))
        except (TypeError, ValueError):
            return None
        return (ns / 1e9) if ns > 0 else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    # El directorio REPROCESADO (..._arc174) solo conserva windows.csv,
    # verdict.json y la procedencia; `stdout.txt` se queda en el crudo. El
    # indice de corridas se construye sobre el reprocesado, que es el que
    # tiene la matriz completa, y el stdout se busca por el mismo nombre de
    # corrida en el crudo.
    parser.add_argument("--stdout-dir", required=True,
                        help="directorio CRUDO de la campaña, donde están los stdout.txt")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    index = discover_runs(Path(args.campaign_dir))
    stdout_root = Path(args.stdout_dir)
    patterns = load_patterns(Path(args.catalog))
    print(f"nodo de análisis: {platform.node()}", flush=True)
    print(f"kernels con patrón de runtime en el catálogo: {sorted(patterns)}\n", flush=True)

    rows = []
    sin_patron = []
    used_fallback = []
    for kernel in sorted(index["kernel_ref"].unique()):
        pattern = patterns.get(kernel)
        if not pattern:
            used_fallback.append(kernel)

        # Duracion por nivel: MEDIA de las repeticiones, no suma, porque el
        # numero de repeticiones aceptadas puede diferir entre niveles y una
        # suma las pesaria distinto.
        by_level: dict[str, list[float]] = {}
        for row in index[index["kernel_ref"] == kernel].itertuples():
            run_name = Path(row.windows_path).parent.name
            seconds = runtime_from_stdout(stdout_root / run_name, pattern)
            if seconds is not None:
                by_level.setdefault(row.freq_level_id, []).append(seconds)

        durations = {
            NOMINAL_MHZ[lvl]: float(np.mean(vals))
            for lvl, vals in by_level.items() if lvl in NOMINAL_MHZ and vals
        }
        if len(durations) < 3 or 3200.0 not in durations:
            sin_patron.append(f"{kernel} (datos insuficientes incluso con respaldo)")
            continue

        alpha, r2 = fit_alpha(durations, 3200.0)
        n_reps = {lvl: len(v) for lvl, v in sorted(by_level.items())}
        rows.append({
            "kernel": kernel,
            "alpha_stdout": round(alpha, 4),
            "r2": round(r2, 4),
            "bajo_umbral": bool(alpha <= BREAK_EVEN),
            "fuente": "respaldo (telemetry_mean_ns)" if kernel in used_fallback else "patrón propio",
            "T_F0_s": round(durations.get(3200.0, float("nan")), 3),
            "T_F4_s": round(durations.get(800.0, float("nan")), 3),
            "ratio_F4_F0": round(durations[800.0] / durations[3200.0], 4) if 800.0 in durations else None,
            "reps_por_nivel": n_reps,
        })

    table = pd.DataFrame(rows).sort_values("alpha_stdout")
    print("== alpha con la duración reportada por el kernel ==", flush=True)
    print(table.drop(columns=["reps_por_nivel"]).to_string(index=False), flush=True)
    if sin_patron:
        print(f"\nsin datos utilizables (ni patrón propio ni respaldo): {sin_patron}", flush=True)

    bajo = table[table["bajo_umbral"]]
    print(f"\numbral de viabilidad: alpha <= {BREAK_EVEN}", flush=True)
    print(f"kernels por debajo: {len(bajo)} de {len(table)}"
          f"{'  -> ' + ', '.join(bajo['kernel']) if len(bajo) else ''}", flush=True)
    print(f"alpha mínimo del dataset: {table['alpha_stdout'].min():.4f}", flush=True)

    Path(args.out).write_text(json.dumps({
        "analysis_node": platform.node(),
        "break_even": BREAK_EVEN,
        "per_kernel": rows,
        "sin_patron": sin_patron,
    }, indent=2))
    print(f"\nreporte -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
