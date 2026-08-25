"""V4 (Estrategia_GPU_Fase2.md §10): cruza offline las marcas de verdad de
`gpu_phasic` (PHASE C/M + T0_MONOTONIC_NS en stdout.txt) contra
`gpu_power_mw`/`gpu_sm_clock_mhz` por ventana en windows.csv, para
responder si existe alternancia de fase EXPLOTABLE en GPU -- el hueco
logico que la Estrategia GPU §3 deja abierto (el optimo de corrida
completa es constante por kernel, pero eso no prueba ausencia de
alternancia intra-corrida, solo que no se puede medir con NVML).

Criterio de lectura, declarado en el manifiesto de la campana ANTES de
ver el dato: si las fases son distinguibles en potencia Y su nivel optimo
difiere, hay alternancia explotable; si no, la granularidad por carga
queda confirmada con evidencia positiva.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

BASE = Path.home() / "hyperion-results/campaigns/pacca_gpu_phase_probe_20260824"
CID = "pacca_gpu_phase_probe_20260824"
KERNELS = ["gpu_phasic_p010", "gpu_phasic_p100", "gpu_phasic_p1000"]
GPU_LEVELS = ["REF", "F0", "F1", "F2", "F3", "F4"]
CPU_LEVEL = "REF"
REPS = [1, 2, 3]

_PHASE_RE = re.compile(r"^PHASE ([\d.]+) ([CM])$")
_T0_RE = re.compile(r"^T0_MONOTONIC_NS (\d+)$")


def parse_ground_truth(stdout_path: Path) -> tuple[int, list[tuple[float, str]]] | None:
    t0 = None
    phases: list[tuple[float, str]] = []
    for line in stdout_path.read_text(errors="replace").splitlines():
        m = _T0_RE.match(line)
        if m:
            t0 = int(m.group(1))
            continue
        m = _PHASE_RE.match(line)
        if m:
            phases.append((float(m.group(1)), m.group(2)))
    if t0 is None or not phases:
        return None
    return t0, phases


def label_at(offset_s: float, phases: list[tuple[float, str]]) -> str | None:
    """Ultima fase cuyo offset es <= offset_s (funcion escalon)."""
    label = None
    for phase_offset, kind in phases:
        if phase_offset <= offset_s:
            label = kind
        else:
            break
    return label


def read_windows(run_dir: Path) -> list[dict]:
    windows_path = run_dir / "windows.csv"
    if not windows_path.exists():
        return []
    with windows_path.open(newline="") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row.get("quality_status") == "gpu_telemetry"
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    for kernel in KERNELS:
        print("=" * 84)
        print(f"{kernel}")
        print("=" * 84)
        for level in GPU_LEVELS:
            power_by_label: dict[str, list[float]] = {"C": [], "M": []}
            clock_by_label: dict[str, list[float]] = {"C": [], "M": []}
            n_runs_used = 0
            for rep in REPS:
                run_dir = BASE / f"{CID}__{kernel}__{CPU_LEVEL}__gpu{level}__rep{rep:02d}"
                stdout_path = run_dir / "stdout.txt"
                if not stdout_path.exists():
                    continue
                parsed = parse_ground_truth(stdout_path)
                if parsed is None:
                    continue
                t0, phases = parsed
                rows = read_windows(run_dir)
                if not rows:
                    continue
                n_runs_used += 1
                for row in rows:
                    try:
                        # Filas GPU son passthrough (ARC-70): t_start_ns
                        # viene vacio, solo t_end_ns esta poblado -- es la
                        # marca real de esta lectura NVML, mismo dominio
                        # CLOCK_MONOTONIC que T0_MONOTONIC_NS (verificado:
                        # ambos ~1.21e15 en la misma corrida).
                        t_end = int(row["t_end_ns"])
                        power_mw = float(row["gpu_power_mw"])
                        clock_mhz = float(row["gpu_sm_clock_mhz"])
                    except (KeyError, ValueError, TypeError):
                        continue
                    offset_s = (t_end - t0) / 1e9
                    if offset_s < 0:
                        continue
                    label = label_at(offset_s, phases)
                    if label is None:
                        continue
                    power_by_label[label].append(power_mw)
                    clock_by_label[label].append(clock_mhz)

            n_c, n_m = len(power_by_label["C"]), len(power_by_label["M"])
            if n_c == 0 or n_m == 0:
                print(f"  {level:4s} corridas_usadas={n_runs_used}  SIN VENTANAS SUFICIENTES (C={n_c}, M={n_m})")
                continue
            mean_c = sum(power_by_label["C"]) / n_c
            mean_m = sum(power_by_label["M"]) / n_m
            clk_c = sum(clock_by_label["C"]) / len(clock_by_label["C"])
            clk_m = sum(clock_by_label["M"]) / len(clock_by_label["M"])
            diff_pct = 100.0 * (mean_c - mean_m) / mean_m if mean_m else float("nan")
            print(f"  {level:4s} corridas_usadas={n_runs_used}  "
                  f"P(C)={mean_c:7.1f}mW (n={n_c:4d})  P(M)={mean_m:7.1f}mW (n={n_m:4d})  "
                  f"diff={diff_pct:+6.2f}%  clk(C)={clk_c:6.1f}MHz  clk(M)={clk_m:6.1f}MHz")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
