"""Hueco del oráculo y alpha por kernel en el eje GPU (job 6462).

Responde la pregunta existencial ANTES de diseñar el modelo de ML: ¿existe
un nivel de frecuencia GPU que le gane a "siempre F0" (máximo reloj)? Si no
existe, ningún clasificador puede capturar un ahorro que no está ahí.

METRICA. El ahorro se mide sobre energía TOTAL (GPU + paquete CPU + DRAM),
no solo GPU. Bajar el reloj de la GPU alarga la corrida, y la CPU delegada
sigue consumiendo durante esa extensión. En estos datos la GPU es solo
~47% de la energía total de una corrida, así que evaluar solo el lado GPU
sobreestimaría el ahorro de forma grosera.

ALPHA. Mismo instrumento de la Fase 1, aplicado al eje GPU:
``T(f)/T(f_ref) = (1-alpha) + alpha*(f_ref/f)``, ajustado sobre los 5
niveles fijos con sus MHz reales. alpha bajo = insensible a la frecuencia
= el DVFS paga. Referencia de lectura ya establecida en CPU: STREAM tiene
alpha=0.154 (buen candidato), rodinia_lavamd_omp alpha=1.029 (pésimo).

El nivel REF queda FUERA del ajuste de alpha (no tiene reloj fijo: bajo
carga hace boost libremente) pero sí entra en la tabla del oráculo, porque
"dejar el gobernador nativo" es una política candidata legítima.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

BASE = Path.home() / "hyperion-results/campaigns/pacca_gpu_nucleo_activo_20260823"
CID = "pacca_gpu_nucleo_activo_20260823"

KERNELS = ["rodinia_gaussian", "gpu_dgemm_n4096", "rodinia_heartwall", "rodinia_lavamd"]
CPU_LEVELS = ["REF", "F4"]
GPU_LEVELS = ["REF", "F0", "F1", "F2", "F3", "F4"]
FIXED_GPU_LEVELS = ["F0", "F1", "F2", "F3", "F4"]  # REF no tiene reloj fijo
REPS = [1, 2, 3]

BASELINE_LEVEL = "F0"  # "siempre performance", el rival a vencer


def read_summary(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        try:
            out[key.strip()] = float(value.strip())
        except ValueError:
            pass
    return out


def read_run(run_dir: Path) -> dict[str, float] | None:
    """Extrae energía y tiempo de una corrida. None si falta algo."""
    summary_path = run_dir / "summary.txt"
    metadata_path = run_dir / "metadata.json"
    windows_path = run_dir / "windows.csv"
    if not (summary_path.exists() and metadata_path.exists() and windows_path.exists()):
        return None

    summary = read_summary(summary_path)
    metadata = json.loads(metadata_path.read_text())

    elapsed_s = summary.get("telemetry_elapsed_ns_mean", 0.0) / 1e9
    if elapsed_s <= 0:
        return None

    # Energía CPU desde RAPL (uJ -> J).
    cpu_j = (
        summary.get("rapl_pkg_total_delta_uj", 0.0)
        + summary.get("rapl_dram_total_delta_uj", 0.0)
    ) / 1e6

    # Energía GPU: suma de deltas por ventana (mJ -> J), solo filas con
    # telemetría GPU válida.
    gpu_mj = 0.0
    n_gpu_rows = 0
    with windows_path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("quality_status") != "gpu_telemetry":
                continue
            if row.get("gpu_energy_valid") != "1":
                continue
            raw = row.get("gpu_energy_delta_mj")
            if raw in (None, ""):
                continue
            gpu_mj += float(raw)
            n_gpu_rows += 1
    if n_gpu_rows == 0:
        return None

    gpu_j = gpu_mj / 1e3
    return {
        "elapsed_s": elapsed_s,
        "gpu_j": gpu_j,
        "cpu_j": cpu_j,
        "total_j": gpu_j + cpu_j,
        "gpu_mhz": float(metadata.get("gpu_freq_mhz_applied") or 0.0),
        "n_gpu_rows": float(n_gpu_rows),
    }


def collect() -> dict[tuple[str, str, str], dict[str, float]]:
    """Promedia las repeticiones de cada (kernel, nivel CPU, nivel GPU)."""
    aggregated: dict[tuple[str, str, str], dict[str, float]] = {}
    for kernel in KERNELS:
        for cpu_level in CPU_LEVELS:
            for gpu_level in GPU_LEVELS:
                runs = []
                for rep in REPS:
                    run_dir = BASE / f"{CID}__{kernel}__{cpu_level}__gpu{gpu_level}__rep{rep:02d}"
                    record = read_run(run_dir)
                    if record is not None:
                        runs.append(record)
                if not runs:
                    continue
                keys = ["elapsed_s", "gpu_j", "cpu_j", "total_j", "gpu_mhz"]
                mean = {k: sum(r[k] for r in runs) / len(runs) for k in keys}
                mean["n_reps"] = float(len(runs))
                # Dispersión relativa del tiempo entre repeticiones: si es
                # alta, el promedio no es de fiar para decidir el óptimo.
                times = [r["elapsed_s"] for r in runs]
                mean_t = mean["elapsed_s"]
                if len(times) > 1 and mean_t > 0:
                    var = sum((t - mean_t) ** 2 for t in times) / (len(times) - 1)
                    mean["time_cv_pct"] = 100.0 * math.sqrt(var) / mean_t
                else:
                    mean["time_cv_pct"] = 0.0
                aggregated[(kernel, cpu_level, gpu_level)] = mean
    return aggregated


def fit_alpha(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Ajusta y = (1-alpha) + alpha*x por mínimos cuadrados.

    ``points`` son pares (x, y) con x = f_ref/f y y = T(f)/T(f_ref).
    Devuelve (alpha, intercepto, r2). El intercepto se reporta sin forzar:
    si el modelo es válido debería dar ~= 1-alpha, y una desviación grande
    es señal de que el modelo simple no aplica a ese kernel.
    """
    n = len(points)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    alpha = sxy / sxx
    intercept = mean_y - alpha * mean_x
    ss_tot = sum((p[1] - mean_y) ** 2 for p in points)
    ss_res = sum((p[1] - (intercept + alpha * p[0])) ** 2 for p in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return alpha, intercept, r2


def main() -> int:
    global BASE, CID, KERNELS, CPU_LEVELS, REPS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--campaign-id", default=CID)
    parser.add_argument("--kernels", nargs="+", default=KERNELS)
    parser.add_argument("--cpu-levels", nargs="+", default=CPU_LEVELS)
    parser.add_argument("--reps", type=int, default=len(REPS))
    args = parser.parse_args()
    BASE = args.base
    CID = args.campaign_id
    KERNELS = list(args.kernels)
    CPU_LEVELS = list(args.cpu_levels)
    REPS = list(range(1, args.reps + 1))

    data = collect()
    if not data:
        print(f"ERROR: no se leyó ninguna corrida bajo {BASE}")
        return 1

    print("=" * 78)
    print("TABLA 1 -- Tiempo y energía por nivel (media de 3 rep)")
    print("=" * 78)
    header = (
        f"{'kernel':<20} {'cpu':<4} {'gpu':<4} {'MHz':>5} {'t(s)':>8} "
        f"{'cv%':>5} {'E_gpu':>8} {'E_cpu':>8} {'E_tot':>8}"
    )
    print(header)
    print("-" * len(header))
    for kernel in KERNELS:
        for cpu_level in CPU_LEVELS:
            for gpu_level in GPU_LEVELS:
                record = data.get((kernel, cpu_level, gpu_level))
                if record is None:
                    continue
                print(
                    f"{kernel:<20} {cpu_level:<4} {gpu_level:<4} "
                    f"{record['gpu_mhz']:>5.0f} {record['elapsed_s']:>8.3f} "
                    f"{record['time_cv_pct']:>5.1f} {record['gpu_j']:>8.1f} "
                    f"{record['cpu_j']:>8.1f} {record['total_j']:>8.1f}"
                )
        print()

    print("=" * 78)
    print("TABLA 2 -- alpha en GPU (ajuste sobre los 5 niveles fijos)")
    print("=" * 78)
    print("alpha bajo  => insensible a frecuencia => el DVFS paga")
    print("referencia CPU ya establecida: STREAM 0.154 (bueno), lavamd_omp 1.029 (malo)")
    print()
    header2 = f"{'kernel':<20} {'cpu':<4} {'alpha':>8} {'intercepto':>11} {'1-alpha':>8} {'r2':>8}"
    print(header2)
    print("-" * len(header2))
    for kernel in KERNELS:
        for cpu_level in CPU_LEVELS:
            reference = data.get((kernel, cpu_level, BASELINE_LEVEL))
            if reference is None or reference["gpu_mhz"] <= 0:
                continue
            points = []
            for gpu_level in FIXED_GPU_LEVELS:
                record = data.get((kernel, cpu_level, gpu_level))
                if record is None or record["gpu_mhz"] <= 0:
                    continue
                points.append(
                    (
                        reference["gpu_mhz"] / record["gpu_mhz"],
                        record["elapsed_s"] / reference["elapsed_s"],
                    )
                )
            alpha, intercept, r2 = fit_alpha(points)
            print(
                f"{kernel:<20} {cpu_level:<4} {alpha:>8.3f} {intercept:>11.3f} "
                f"{1 - alpha:>8.3f} {r2:>8.4f}"
            )
    print()

    print("=" * 78)
    print(f"TABLA 3 -- HUECO DEL ORACULO (mejor nivel vs. siempre {BASELINE_LEVEL})")
    print("=" * 78)
    print("Un ahorro <= 0 significa que 'siempre performance' ya es lo óptimo")
    print("y que NO hay nada que un modelo pueda capturar en ese kernel.")
    print()
    header3 = (
        f"{'kernel':<20} {'cpu':<4} {'mejor':>6} {'ahorro_E%':>10} "
        f"{'costo_t%':>9} {'mejor_EDP':>10} {'EDP_gan%':>9}"
    )
    print(header3)
    print("-" * len(header3))
    for kernel in KERNELS:
        for cpu_level in CPU_LEVELS:
            reference = data.get((kernel, cpu_level, BASELINE_LEVEL))
            if reference is None:
                continue
            candidates = [
                (level, data[(kernel, cpu_level, level)])
                for level in GPU_LEVELS
                if (kernel, cpu_level, level) in data
            ]
            if not candidates:
                continue
            best_level, best = min(candidates, key=lambda item: item[1]["total_j"])
            saving_pct = 100.0 * (reference["total_j"] - best["total_j"]) / reference["total_j"]
            time_cost_pct = (
                100.0 * (best["elapsed_s"] - reference["elapsed_s"]) / reference["elapsed_s"]
            )
            ref_edp = reference["total_j"] * reference["elapsed_s"]
            best_edp_level, best_edp = min(
                candidates, key=lambda item: item[1]["total_j"] * item[1]["elapsed_s"]
            )
            edp_gain_pct = (
                100.0 * (ref_edp - best_edp["total_j"] * best_edp["elapsed_s"]) / ref_edp
            )
            print(
                f"{kernel:<20} {cpu_level:<4} {best_level:>6} {saving_pct:>10.2f} "
                f"{time_cost_pct:>9.2f} {best_edp_level:>10} {edp_gain_pct:>9.2f}"
            )
    print()

    print("=" * 78)
    print("TABLA 4 -- ¿el óptimo depende del kernel? (decide si hace falta modelo)")
    print("=" * 78)
    print("Si TODOS los kernels comparten el mismo nivel óptimo, no hace falta")
    print("un modelo: basta una constante. El modelo solo se justifica si el")
    print("nivel óptimo cambia según el kernel.")
    print()
    for cpu_level in CPU_LEVELS:
        best_by_kernel = {}
        for kernel in KERNELS:
            candidates = [
                (level, data[(kernel, cpu_level, level)])
                for level in GPU_LEVELS
                if (kernel, cpu_level, level) in data
            ]
            if not candidates:
                continue
            best_by_kernel[kernel] = min(candidates, key=lambda item: item[1]["total_j"])[0]
        distinct = sorted(set(best_by_kernel.values()))
        print(f"  CPU={cpu_level}: {best_by_kernel}")
        print(f"    niveles óptimos distintos: {distinct}")
        if len(distinct) <= 1:
            print("    => UNA CONSTANTE BASTA en este subconjunto, no hay nada que aprender")
        else:
            print("    => el óptimo varía por kernel: hay señal que un modelo podría capturar")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
