"""Aplica el criterio C1/C2/C3 de `Estrategia_CPU_Fase2.md` §9 al CSV que
produce `scripts/pacca/screen_rajaperf_cpu_alpha.sh`, y decide qué
candidatos entran a la campaña final de CPU.

C1: alpha < 0.226 (umbral derivado del modelo de potencia real de CPU,
    Anexo A.1) Y r2 > 0.95 (el ajuste de Amdahl debe describir el dato,
    no solo dar un numero).
C2: freq_within_5pct == 'yes' en LOS 5 niveles del kernel -- si el
    candado de frecuencia no se sostuvo bajo carga en algun nivel, alpha
    de ese kernel no es confiable (el bug de ARC-162/hermanos SMT que
    invalido la corrida 6475 original).
C3: output_bytes no debe ser desproporcionado frente al resto -- la
    leccion de rodinia_myocyte (Anexo L.1): un kernel que escribe mucho a
    disco puede tener un alpha bajo por costo fijo de I/O, no por ser
    memory-bound. Umbral: >10x la mediana del grupo se marca para
    revision manual, no se descarta automaticamente.

Uso:
    python3 rajaperf_screening_verdict.py <ruta a cpu_rajaperf_screen_NNNN.out>
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.home() / "hyperion"))
from classifier.features.align import fit_alpha  # noqa: E402

ALPHA_THRESHOLD = 0.226
R2_THRESHOLD = 0.95
F_REF_MHZ = 3200.0


HEADER = ("kernel,level,khz_target,elapsed_s,energy_j,freq_mean_khz,freq_min_khz,"
          "freq_max_khz,freq_within_5pct,n_freq_samples,governor,output_bytes")
_N_FIELDS = HEADER.count(",") + 1


def load_rows(path: Path) -> list[dict]:
    # El .out crudo trae ruido de Lmod (banners de "module load") mezclado
    # antes y despues de la tabla real -- filtrar solo por "," no basta,
    # esos banners tambien tienen comas. Exigir el numero exacto de campos
    # es lo unico que distingue una fila real de ruido con forma parecida.
    with path.open() as handle:
        lines = [
            ln for ln in handle
            if ln.count(",") == _N_FIELDS - 1 and not ln.startswith("kernel,")
        ]
    reader = csv.DictReader([HEADER] + lines)
    return list(reader)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    by_kernel: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_kernel[row["kernel"]].append(row)

    output_bytes_by_kernel = {
        k: statistics.median(float(r["output_bytes"]) for r in rs)
        for k, rs in by_kernel.items()
    }
    global_median_bytes = statistics.median(output_bytes_by_kernel.values()) if output_bytes_by_kernel else 0.0

    survivors = []
    print(f"{'kernel':<28}{'alpha':>8}{'r2':>8}{'C1':>6}{'C2':>6}{'C3':>6}{'output_bytes':>14}")
    for kernel, rs in sorted(by_kernel.items()):
        durations_mhz = {float(r["khz_target"]) / 1000.0: float(r["elapsed_s"]) for r in rs}
        try:
            alpha, r2 = fit_alpha(durations_mhz, F_REF_MHZ)
        except ValueError:
            print(f"{kernel:<28}  sin suficientes niveles validos para ajustar alpha")
            continue

        c1 = alpha < ALPHA_THRESHOLD and r2 > R2_THRESHOLD
        c2 = all(r["freq_within_5pct"].strip().lower() == "yes" for r in rs)
        out_bytes = output_bytes_by_kernel[kernel]
        c3 = out_bytes <= 10 * global_median_bytes if global_median_bytes else True

        print(f"{kernel:<28}{alpha:>8.3f}{r2:>8.3f}"
              f"{'OK' if c1 else 'no':>6}{'OK' if c2 else 'no':>6}{'OK' if c3 else '!!':>6}"
              f"{out_bytes:>14.0f}")

        if c1 and c2 and c3:
            survivors.append(kernel)
        elif c1 and not c2:
            print(f"   -> {kernel}: alpha bajo el umbral PERO C2 (pineo de frecuencia) fallo -- "
                  f"no confiable, no se acepta sin remedir")

    print()
    print(f"SOBREVIVIENTES ({len(survivors)}): {survivors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
