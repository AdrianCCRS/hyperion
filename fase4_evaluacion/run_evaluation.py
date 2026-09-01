#!/usr/bin/env python3
"""Punto de entrada de Fase 4 (Objetivo 4): reporte de evaluación empírica.

⚠️ **Alcance real de este script, léase antes de usar**: genera el reporte
de comparación (§5.2) a partir de `windows.csv` YA PRODUCIDOS por cada
escenario -- no orquesta automáticamente correr el catálogo completo bajo
los 3 escenarios de una sola invocación. Construir esa orquestación
completa requiere que `fase1_telemetria/campaign.py` acepte un wrapper de
escenario de gobernador alrededor de cada corrida (hoy no tiene ese punto
de extensión) y que el daemon de `fase3_daemon/` esté completo (el loop de
CPU real todavía no lo está, ver `fase3_daemon/README.md`) -- ninguna de
las dos piezas se construyó en esta reconstrucción por las razones
documentadas en sus respectivos README.

Lo que SÍ hace este script hoy, de punta a punta:
1. Para los escenarios de gobernador nativo (`ondemand`/`schedutil`/
   `performance`), produce cada uno corriendo manualmente
   `fase1_telemetria/run_campaign.py` dentro de un bloque
   `fase4_evaluacion.governors.governor_scenario(...)` (ver el README de
   esta fase para el procedimiento paso a paso) -- la pieza de código que
   permite eso (`governors.py`) sí está completa y probada.
2. Para el escenario del agente, requiere `windows.csv` de una corrida real
   con `fase3_daemon/run_daemon.py` activo -- sujeto a las limitaciones de
   esa fase.
3. Este script consume los `windows.csv` resultantes de los escenarios que
   ya se corrieron (los que falten se omiten del reporte, nunca se
   fabrican) y produce la tabla de comparación de `edp_report.py`.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fase4_evaluacion.edp_report import (  # noqa: E402
    compare_scenarios,
    format_report,
    load_scenario_windows,
)
from fase4_evaluacion.governors import SCENARIO_GOVERNORS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", action="append", nargs=2, metavar=("NOMBRE", "GLOB_WINDOWS_CSV"),
        required=True,
        help="Repetible. Un escenario y el patrón glob de sus windows.csv, "
             "p.ej. --scenario performance '~/hyperion-results/campaigns/perf/*/windows.csv' "
             "--scenario agente '~/hyperion-results/campaigns/agente/*/windows.csv'. "
             f"Escenarios de gobernador esperados: {', '.join(SCENARIO_GOVERNORS)}.",
    )
    parser.add_argument("--agent-scenario", required=True,
                         help="Nombre (de los pasados en --scenario) que corresponde al agente propuesto.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--devices", default="cpu,gpu")
    parser.add_argument("--labels", default="compute_bound,memory_bound")
    parser.add_argument("--output", type=Path, default=None,
                         help="Si se da, escribe el reporte de texto aquí además de imprimirlo.")
    args = parser.parse_args()

    windows_csv_by_scenario = {}
    for name, pattern in args.scenario:
        expanded = sorted(Path(p) for p in glob.glob(str(Path(pattern).expanduser())))
        if not expanded:
            print(f"AVISO: ningún windows.csv encontrado para el escenario {name!r} con el patrón {pattern!r} "
                  "-- se omite del reporte.", file=sys.stderr)
            continue
        windows_csv_by_scenario[name] = expanded

    if args.agent_scenario not in windows_csv_by_scenario:
        parser.error(
            f"--agent-scenario {args.agent_scenario!r} no tiene windows.csv disponibles -- "
            "no se puede generar ningún reporte sin datos del agente"
        )

    baseline_names = [name for name in windows_csv_by_scenario if name != args.agent_scenario]
    if not baseline_names:
        parser.error("ningún escenario baseline con datos disponibles -- nada que comparar")

    windows_by_scenario = load_scenario_windows(windows_csv_by_scenario)
    comparisons = compare_scenarios(
        windows_by_scenario,
        agent_scenario=args.agent_scenario,
        baseline_scenarios=baseline_names,
        devices=tuple(args.devices.split(",")),
        labels=tuple(args.labels.split(",")),
        alpha=args.alpha,
    )

    if not comparisons:
        print("Sin comparaciones posibles -- revisa que haya al menos 2 kernels en común "
              "por (dispositivo, clase, escenario).", file=sys.stderr)
        return 1

    report = format_report(comparisons)
    print(report)
    if args.output is not None:
        args.output.write_text(report + "\n")
        print(f"\nReporte escrito en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
