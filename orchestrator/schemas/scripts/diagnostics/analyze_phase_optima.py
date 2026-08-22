"""¿Varía la frecuencia óptima ENTRE FASES de un mismo kernel?

Es la pregunta que decide el diseño del modelo de Fase 2 (ver
docs/general/opciones_modelo_fase2.md):

  - Si el óptimo NO varía dentro de un kernel, predecir "la frecuencia
    óptima" es predecir una constante por kernel y no hay política dinámica
    que aprender: toda la maquinaria de alineación sobra.
  - Si SÍ varía, hay una decisión real por fase y el doble target tiene
    sentido.

De paso mide, por fase:
  - alpha (fracción de tiempo sensible a la frecuencia) y su ajuste R^2,
    que es lo que la opción C propone predecir.
  - la dispersión de alpha DENTRO de cada kernel, que es la señal que esa
    opción necesita.

Lee del directorio reprocesado ARC-174, que es el único con las 540
corridas completas y con la clasificación por ventana vigente.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.home() / "hyperion"))
from classifier.features import align  # noqa: E402

BASE = Path.home() / "hyperion-results/campaigns/pacca_cpu_final_attempt03_20260820_arc174"
CID = "pacca_cpu_final_attempt03_20260820"
KERNELS = [
    "npb_bt", "npb_mg", "npb_cg", "npb_sp", "npb_ft", "npb_lu",
    "dgemm_n2048", "rodinia_lavamd_omp", "rajaperf_polybench_3mm_omp",
]
# REF se excluye del ajuste de alpha: es el gobernador nativo, no un punto
# de frecuencia fija, así que no pertenece a la curva T(f).
LEVEL_MHZ = {"F0": 3200, "F1": 2600, "F2": 2000, "F3": 1400, "F4": 800}
F_REF = 3200
N_BINS = 20
REPS = range(1, 11)

COLS = [
    "kernel_ref", "freq_level_id", "repetition", "window_index",
    "delta_instructions", "delta_t_ns", "pkg_delta_uj",
    "ipc", "mpki", "llc_miss_rate", "stall_backend_ratio", "running_ratio",
    "operational_intensity_uncore_real", "i_ridge_used",
    "quality_status", "frequency_quality_status",
]
FEATURES = [
    "ipc", "mpki", "llc_miss_rate", "stall_backend_ratio", "running_ratio",
    "operational_intensity_uncore_real", "i_ridge_used",
]


def load_kernel(kernel: str) -> pd.DataFrame:
    """Carga las 50 corridas del kernel (5 niveles x 10 reps).

    OJO con ``repetition``: en windows.csv vale siempre 1, porque cada
    corrida se lanza con repetitions=1 y su índice real vive en el nombre
    del directorio. Agrupar por esa columna fusionaría las 10 repeticiones
    en una sola pseudo-corrida y el progreso acumulado cruzaría de una a
    otra. Por eso el índice de repetición se deriva aquí, del run_id.
    """
    frames = []
    for level in LEVEL_MHZ:
        for rep in REPS:
            path = BASE / f"{CID}__{kernel}__{level}__rep{rep:02d}" / "windows.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
            frame["rep_idx"] = rep
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"sin corridas para {kernel}")
    return pd.concat(frames, ignore_index=True)


def spread(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    mean = sum(values) / n
    sd = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    return mean, sd, max(values) - min(values)


def main() -> None:
    print(f"Alineación por instrucciones, {N_BINS} tramos por corrida, "
          f"niveles {list(LEVEL_MHZ)} (REF excluido del ajuste)\n")

    header = (f"{'kernel':<28}{'óptimo/bin':>26}{'alpha medio':>12}"
              f"{'sd':>7}{'rango':>8}{'R2 medio':>10}")
    print(header)
    print("-" * len(header))

    summary = []
    for kernel in KERNELS:
        df = load_kernel(kernel)
        # run_keys por corrida REAL: kernel+nivel+rep_idx identifica una
        # ejecucion unica. Sin rep_idx, las 10 reps se concatenarian.
        df = align.add_instruction_progress(
            df, run_keys=("kernel_ref", "freq_level_id", "rep_idx"))
        df = align.assign_progress_bins(df, n_bins=N_BINS)
        cells = align.aggregate_cells(
            df, feature_cols=FEATURES,
            cell_keys=("kernel_ref", "rep_idx", "progress_bin", "freq_level_id"))
        del df

        cells["edp"] = cells["energy_uj"] * cells["duration_ns"]

        best_levels: Counter = Counter()
        alphas: list[float] = []
        r2s: list[float] = []

        for (_rep, _bin), group in cells.groupby(["rep_idx", "progress_bin"], observed=True):
            per_level = group.set_index("freq_level_id")
            durations = {
                LEVEL_MHZ[lvl]: per_level.loc[lvl, "duration_ns"]
                for lvl in LEVEL_MHZ if lvl in per_level.index
            }
            if len(durations) < len(LEVEL_MHZ):
                continue  # tramo incompleto: no comparable entre niveles
            best_levels[per_level["edp"].idxmin()] += 1
            try:
                alpha, r2 = align.fit_alpha(durations, f_ref_mhz=F_REF)
            except ValueError:
                continue
            alphas.append(alpha)
            r2s.append(r2)

        if not alphas:
            print(f"{kernel:<28}{'(sin tramos completos)':>26}")
            continue

        total = sum(best_levels.values())
        dist = " ".join(f"{lvl}:{100*c/total:.0f}%" for lvl, c in best_levels.most_common(3))
        a_mean, a_sd, a_range = spread(alphas)
        r2_mean = sum(r2s) / len(r2s)
        print(f"{kernel:<28}{dist:>26}{a_mean:>12.3f}{a_sd:>7.3f}"
              f"{a_range:>8.3f}{r2_mean:>10.4f}")
        summary.append((kernel, best_levels, total, a_mean, a_sd, a_range, r2_mean))

    print("\n\n=== ¿Varía el óptimo dentro de cada kernel? ===")
    print(f"{'kernel':<28}{'tramos':>8}{'niveles distintos':>19}{'% en el dominante':>19}")
    print("-" * 74)
    for kernel, best, total, *_ in summary:
        dominant = best.most_common(1)[0][1] / total
        print(f"{kernel:<28}{total:>8}{len(best):>19}{dominant:>18.1%}")

    varying = [k for k, best, total, *_ in summary if best.most_common(1)[0][1] / total < 0.95]
    print(f"\n  kernels con el óptimo repartido (<95% en un solo nivel): "
          f"{len(varying)}/{len(summary)}")
    if varying:
        print(f"  -> {', '.join(varying)}")

    print("\n\n=== ¿Tiene alpha señal DENTRO de cada kernel? ===")
    print("(si el rango intra-kernel es comparable al inter-kernel, hay algo "
          "que aprender por fase)")
    inter = [a for _, _, _, a, _, _, _ in summary]
    if inter:
        lo, hi = min(inter), max(inter)
        print(f"\n  rango INTER-kernel de alpha medio: {lo:.3f} a {hi:.3f} "
              f"(amplitud {hi - lo:.3f})")
        worst = max(summary, key=lambda s: s[5])
        print(f"  mayor rango INTRA-kernel: {worst[0]} con {worst[5]:.3f}")


if __name__ == "__main__":
    main()
