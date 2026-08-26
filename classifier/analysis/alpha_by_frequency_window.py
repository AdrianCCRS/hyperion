"""¿Cuanto de nuestro alpha viene de medir sobre un rango de frecuencia
mas ancho que el del estado del arte?

MOTIVO (2026-08-25, tras revisar los papers). Calore et al. 2017
(Concurrency Computat. Pract. Exper. 29(12):e4143) miden DVFS en un Xeon
E5-2630v3 sobre un rango de ~1.2 a 2.4 GHz -- un factor 2x. Su kernel
`propagate` (paso de streaming de Lattice Boltzmann) sale memory-bound con
T_s practicamente constante, y reportan 9% de ahorro de energia por 3% de
coste en tiempo. Ese 3% sobre un 1.41x de reduccion de reloj implica
alpha ~ 0.073.

Nosotros ajustamos alpha sobre F0-F4, es decir 3200 a 800 MHz: un factor
4x, que baja mucho mas. Y alpha NO es invariante al rango sobre el que se
ajusta: si el subsistema de memoria se degrada a frecuencias muy bajas
(sea por el uncore, por perdida de paralelismo de memoria, o por lo que
sea), ese efecto entra en el ajuste y sube alpha, aunque en la ventana
alta el kernel sea perfectamente insensible al reloj.

Si eso es asi, nuestro alpha=0.154 de STREAM y el alpha~0.073 de
`propagate` NO son comparables, y la conclusion "ningun kernel baja del
umbral" podria ser un artefacto de la ventana de ajuste y no una
propiedad de las cargas.

QUE HACE. Reajusta alpha para cada kernel del dataset de CPU sobre
ventanas de frecuencia crecientes, todas ancladas en F0:
    F0-F1  (3200-2600, 1.23x)  <- ventana estrecha, la "zona util"
    F0-F2  (3200-2000, 1.60x)
    F0-F3  (3200-1400, 2.29x)  <- comparable al rango de Calore
    F0-F4  (3200-800,  4.00x)  <- lo que se venia reportando
y muestra como se mueve. Si alpha cae mucho al estrechar la ventana, la
lectura del resultado negativo cambia.

NO es re-analizar para que salga lo que queremos: la ventana correcta es
la que cubre los niveles que la POLITICA usaria de verdad. Si el optimo
de EDP nunca cae por debajo de F2, ajustar alpha incluyendo F4 mete en el
numero una region que la politica jamas visitaria.

Uso: python3 alpha_by_frequency_window.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.analysis import loko_pilot  # noqa: E402
from classifier.features import pair_dataset  # noqa: E402
from classifier.features.align import fit_alpha  # noqa: E402

# kHz reales de cada nivel en la campaña de CPU ya validada.
LEVEL_KHZ = {
    "F0": 3200000, "F1": 2600000, "F2": 2000000,
    "F3": 1400000, "F4": 800000,
}
F_REF_MHZ = 3200.0

WINDOWS = [
    ("F0-F1", ["F0", "F1"]),
    ("F0-F2", ["F0", "F1", "F2"]),
    ("F0-F3", ["F0", "F1", "F2", "F3"]),
    ("F0-F4", ["F0", "F1", "F2", "F3", "F4"]),
]

CPU_THRESHOLD = 0.226


def main() -> int:
    feature_cols = list(pair_dataset.CPU_FEATURES)
    runs = loko_pilot.collect_runs(loko_pilot.CPU_BASE, "cpu", feature_cols)
    # Aqui SI se conservan los kernels de calibracion: `stream_official` es
    # justamente la referencia de saturacion (100% del ancho de banda) y
    # `ert_probe` el control compute-bound. En el LOKO estorban como
    # pliegues; en este analisis son los dos puntos de anclaje.
    print(f"corridas leidas: {len(runs)}")
    print(f"kernels: {sorted(runs['kernel_ref'].unique())}")
    print()

    # Tiempo medio por (kernel, nivel), sobre todas las repeticiones.
    mean_time = (
        runs.groupby(["kernel_ref", "freq_level_id"], observed=True)["elapsed_s"]
        .mean().unstack()
    )

    header = f"{'kernel':<30}" + "".join(f"{name:>10}" for name, _ in WINDOWS)
    print("=== alpha reajustado por ventana de frecuencia ===")
    print(header)
    print("-" * len(header))

    for kernel in sorted(mean_time.index):
        row = f"{kernel:<30}"
        for _, levels in WINDOWS:
            durations = {}
            for level in levels:
                if level in mean_time.columns:
                    value = mean_time.loc[kernel, level]
                    if value == value and value > 0:  # descarta NaN
                        durations[LEVEL_KHZ[level] / 1000.0] = float(value)
            if len(durations) < 2:
                row += f"{'--':>10}"
                continue
            try:
                alpha, _r2 = fit_alpha(durations, F_REF_MHZ)
                row += f"{alpha:>10.3f}"
            except ValueError:
                row += f"{'--':>10}"
        print(row)

    print()
    print(f"umbral de viabilidad de EDP en CPU: alpha <= {CPU_THRESHOLD}")
    print()
    print("Lectura: si alpha cae al estrechar la ventana, el numero que")
    print("veniamos reportando (F0-F4) incluye la degradacion del subsistema")
    print("de memoria a frecuencias muy bajas, region que la politica no")
    print("visitaria. La comparacion con Calore et al. (rango ~2x) exige")
    print("mirar la columna F0-F3, no la F0-F4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
