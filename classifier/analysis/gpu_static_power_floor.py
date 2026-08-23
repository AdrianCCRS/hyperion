"""Por qué el DVFS de GPU casi no puede pagar en paccaA100: el piso estático.

La energía total de una corrida es aproximadamente

    E(f) = P(f) * T(f)

y bajar el reloj de SM solo paga si la potencia cae MAS RAPIDO de lo que
el tiempo crece. El criterio exacto, sin modelo de por medio, es:

    T(f)/T(F0)  <  P(F0)/P(f)

Este script calcula ambos lados para cada kernel y nivel, y dice quién
gana. No depende del ajuste de Amdahl -- que resultó inválido para los
kernels del tamizaje (r2 ~ 0.53-0.63, intercepto incompatible con 1-alpha)
-- sino solo de energía y tiempo medidos.

La hipótesis que pone a prueba: en este nodo el consumo tiene un piso
estático muy alto (CPU delegada a gobernador `performance` + potencia de
reposo de la GPU) que NO baja con el reloj de SM. Si ese piso domina, la
ventana de ganancia del DVFS se cierra casi por completo, y eso sería una
propiedad de la PLATAFORMA, no del catálogo de kernels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from classifier.analysis.gpu_oracle_headroom import collect
from classifier.analysis import gpu_oracle_headroom as core

LEVELS = ["F0", "F1", "F2", "F3", "F4"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--kernels", nargs="+", required=True)
    parser.add_argument("--cpu-level", default="REF")
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    core.BASE = args.base
    core.CID = args.campaign_id
    core.KERNELS = list(args.kernels)
    core.CPU_LEVELS = [args.cpu_level]
    core.REPS = list(range(1, args.reps + 1))

    data = collect()
    if not data:
        print(f"ERROR: no se leyó ninguna corrida bajo {args.base}")
        return 1

    print("=" * 96)
    print("POTENCIA TOTAL POR NIVEL  (P = E_total / T)  --  incluye CPU delegada + GPU")
    print("=" * 96)
    head = (
        f"{'kernel':<20} {'nivel':>5} {'t(s)':>8} {'P_tot(W)':>9} {'P_cpu(W)':>9} "
        f"{'T/T_F0':>8} {'P_F0/P':>8} {'¿paga?':>8} {'dE%':>8}"
    )
    print(head)
    print("-" * len(head))

    for kernel in core.KERNELS:
        reference = data.get((kernel, args.cpu_level, "F0"))
        if reference is None:
            print(f"{kernel:<20} sin nivel F0, se omite")
            continue
        p_ref = reference["total_j"] / reference["elapsed_s"]
        for level in LEVELS:
            record = data.get((kernel, args.cpu_level, level))
            if record is None:
                continue
            power = record["total_j"] / record["elapsed_s"]
            cpu_power = record["cpu_j"] / record["elapsed_s"]
            time_ratio = record["elapsed_s"] / reference["elapsed_s"]
            power_ratio = p_ref / power
            pays = "SI" if time_ratio < power_ratio else "no"
            delta_e = 100.0 * (record["total_j"] - reference["total_j"]) / reference["total_j"]
            print(
                f"{kernel:<20} {level:>5} {record['elapsed_s']:>8.3f} {power:>9.1f} "
                f"{cpu_power:>9.1f} {time_ratio:>8.3f} {power_ratio:>8.3f} "
                f"{pays:>8} {delta_e:>+8.2f}"
            )
        print()

    print("=" * 96)
    print("RESUMEN -- cuánta potencia compra el recorte completo de reloj (F0 -> F4, 6.7x)")
    print("=" * 96)
    head2 = (
        f"{'kernel':<20} {'P(F0)':>8} {'P(F4)':>8} {'caída P':>9} "
        f"{'T(F4)/T(F0)':>12} {'holgura necesaria':>18}"
    )
    print(head2)
    print("-" * len(head2))
    for kernel in core.KERNELS:
        f0 = data.get((kernel, args.cpu_level, "F0"))
        f4 = data.get((kernel, args.cpu_level, "F4"))
        if f0 is None or f4 is None:
            continue
        p0 = f0["total_j"] / f0["elapsed_s"]
        p4 = f4["total_j"] / f4["elapsed_s"]
        drop = 100.0 * (p0 - p4) / p0
        ratio = f4["elapsed_s"] / f0["elapsed_s"]
        print(
            f"{kernel:<20} {p0:>8.1f} {p4:>8.1f} {drop:>8.1f}% "
            f"{ratio:>12.2f} {'< ' + format(p0 / p4, '.2f'):>18}"
        )
    print()
    print("Lectura: el tiempo puede crecer como mucho el factor de la última")
    print("columna. Si T(F4)/T(F0) lo supera, bajar el reloj CUESTA energía.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
