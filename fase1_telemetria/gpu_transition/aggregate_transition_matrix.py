#!/usr/bin/env python3
"""F1-GPU-002: agrega varios `gpu_clock_transition_summary.json` en una matriz de
transiciones y deriva `T_transicion_gpu_ns_conservative`.

Regla de derivación (Seguimiento_Cambios_Plan_Director.md, F1-GPU-002; plan
§2.4.1):

- Solo entran corridas con `result == "stable"` y `transition_metrics.valid`.
- Se agrupa por `(from_clock, to_clock_mhz)` -- la dirección importa.
- Por grupo se toma el **máximo** de `conservative_upper_bound_ns` sobre las
  réplicas (no la media, no un percentil: con 3 réplicas el máximo es la cota
  conservadora honesta).
- `T_transicion_gpu_ns_conservative` = máximo de esos máximos por grupo.
- Si la resolución del reloj / cadencia solo permite una cota (siempre, en la
  práctica), ese número es una **cota superior observable**, válida y segura
  para `min_dwell_ns` y `--t-transicion-gpu-ns`, nunca una latencia física
  exacta.

Este módulo NO decide la política ni toca `derive_policy_table.py`: solo produce
el número que se le pasa a mano por `--t-transicion-gpu-ns`.

Uso:
    python3 -m fase1_telemetria.gpu_transition.aggregate_transition_matrix \\
        RESULTS_DIR_O_JSONS...  --output matriz_transiciones.json

Acepta rutas a archivos `*summary*.json`, o directorios (se recorren buscando
`gpu_clock_transition_summary.json`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

MIN_REPLICATES = 3


def _pair_key(summary: dict[str, Any]) -> str:
    return f"{summary.get('from_clock')}->{summary.get('to_clock_mhz')}"


def load_summaries(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Carga los JSON de resumen. Un directorio se recorre en busca de
    `gpu_clock_transition_summary.json`; un archivo se carga tal cual.
    Cada dict cargado lleva `_source_path` para trazabilidad."""
    out: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw in paths:
        p = Path(raw)
        candidates: list[Path]
        if p.is_dir():
            candidates = sorted(p.rglob("gpu_clock_transition_summary.json"))
        else:
            candidates = [p]
        for c in candidates:
            c = c.resolve()
            if c in seen:
                continue
            seen.add(c)
            with c.open(encoding="utf-8") as fh:
                data = json.load(fh)
            data["_source_path"] = str(c)
            out.append(data)
    return out


def conservative_transition_ns(
    summaries: list[dict[str, Any]], *, required_pairs: set[str] | None = None,
) -> dict[str, Any]:
    """Deriva la cota conservadora a partir de resúmenes ya cargados.

    No lee disco: recibe la lista de dicts (facilita las pruebas). Devuelve un
    reporte con la matriz por par, el valor conservador y las advertencias.
    """
    by_pair: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    n_total = len(summaries)
    n_stable = 0
    n_timeout = 0
    n_other = 0
    provenance: tuple[str | None, str | None, str | None] | None = None

    for s in summaries:
        result = s.get("result")
        metrics = s.get("transition_metrics") or {}
        pair = _pair_key(s)
        gpu = s.get("gpu") or {}
        identity = (
            gpu.get("uuid"), gpu.get("driver_version"),
            s.get("workload_checksum_sha256"),
        )
        if provenance is None:
            provenance = identity
        elif identity != provenance:
            warnings.append(
                f"provenance incompatible en {s.get('_source_path', '<inline>')}: "
                "GPU UUID, driver o checksum de carga no coincide con la primera corrida"
            )
        if s.get("dry_run_actuation") is True:
            warnings.append(
                f"{s.get('_source_path', '<inline>')}: dry_run_actuation=true no es una transición medible"
            )
        restoration = s.get("restoration") or {}
        if restoration.get("confirmed") is not True:
            warnings.append(
                f"{s.get('_source_path', '<inline>')}: restauración no confirmada"
            )
        entry = by_pair.setdefault(
            pair,
            {
                "from_clock": s.get("from_clock"),
                "to_clock_mhz": s.get("to_clock_mhz"),
                "stable_replicates": 0,
                "conservative_upper_bound_ns_max": None,
                "conservative_upper_bound_ns_values": [],
                "command_latency_ns_max": None,
                "non_stable_results": [],
                "sources": [],
            },
        )
        entry["sources"].append(s.get("_source_path", "<inline>"))

        if result == "timeout":
            n_timeout += 1
        elif result != "stable":
            n_other += 1

        if result != "stable" or not metrics.get("valid") or s.get("dry_run_actuation") is True \
                or restoration.get("confirmed") is not True:
            if result != "stable":
                entry["non_stable_results"].append(result)
            elif not metrics.get("valid"):
                entry["non_stable_results"].append("stable_metrics_invalid")
            elif s.get("dry_run_actuation") is True:
                entry["non_stable_results"].append("dry_run_actuation")
            else:
                entry["non_stable_results"].append("restoration_unconfirmed")
            continue

        n_stable += 1
        cub = int(metrics["conservative_upper_bound_ns"])
        entry["stable_replicates"] += 1
        entry["conservative_upper_bound_ns_values"].append(cub)
        entry["conservative_upper_bound_ns_max"] = max(
            cub, entry["conservative_upper_bound_ns_max"] or 0
        )
        cmd_lat = metrics.get("command_latency_ns")
        if cmd_lat is not None:
            entry["command_latency_ns_max"] = max(
                int(cmd_lat), entry["command_latency_ns_max"] or 0
            )

    pair_maxima = [
        e["conservative_upper_bound_ns_max"]
        for e in by_pair.values()
        if e["conservative_upper_bound_ns_max"] is not None
    ]

    for pair, e in sorted(by_pair.items()):
        if e["stable_replicates"] == 0:
            warnings.append(
                f"par {pair!r}: 0 réplicas estables "
                f"(resultados: {sorted(set(e['non_stable_results']))}) -- no aporta cota"
            )
        elif e["stable_replicates"] < MIN_REPLICATES:
            warnings.append(
                f"par {pair!r}: solo {e['stable_replicates']} réplica(s) estable(s), "
                f"se requieren >= {MIN_REPLICATES}"
            )
    for pair in sorted(required_pairs or set()):
        if pair not in by_pair:
            warnings.append(f"par requerido {pair!r}: no se entregó ninguna réplica")
    if n_timeout:
        warnings.append(
            f"{n_timeout} corrida(s) con result='timeout': la actuación GPU puede no "
            "converger dentro de --max-wait-ns; revisar el crudo antes de confiar en la cota"
        )
    if n_other:
        warnings.append(
            f"{n_other} corrida(s) con un resultado distinto de 'stable'/'timeout' "
            "(aborted / workload_inactive / source_not_stable / command_error)"
        )

    conservative = max(pair_maxima) if pair_maxima else None
    worst_pair = None
    if conservative is not None:
        worst_pair = max(
            (e for e in by_pair.values() if e["conservative_upper_bound_ns_max"] is not None),
            key=lambda e: e["conservative_upper_bound_ns_max"],
        )
        worst_pair = _pair_key({"from_clock": worst_pair["from_clock"],
                                "to_clock_mhz": worst_pair["to_clock_mhz"]})

    return {
        "schema": "f1-gpu-002/transition_matrix_aggregate/1",
        "n_summaries": n_total,
        "n_stable": n_stable,
        "n_timeout": n_timeout,
        "n_other_non_stable": n_other,
        "min_replicates_per_pair_required": MIN_REPLICATES,
        "t_transicion_gpu_ns_conservative": conservative,
        "t_transicion_gpu_ns_is_observable_upper_bound": True,
        "worst_pair": worst_pair,
        "per_pair": by_pair,
        "warnings": warnings,
        # Fail closed: an incomplete, dry-run, mixed-provenance, timed-out or
        # unrestored matrix must never silently enable GPU actuation.
        "required_pairs": sorted(required_pairs or set()),
        "usable_for_policy": bool(conservative is not None and not warnings),
    }


def aggregate_summaries(
    paths: Iterable[str | Path], *, required_pairs: set[str] | None = None,
) -> dict[str, Any]:
    """load_summaries + conservative_transition_ns."""
    return conservative_transition_ns(load_summaries(paths), required_pairs=required_pairs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+",
                        help="Archivos gpu_clock_transition_summary.json o directorios a recorrer.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Ruta del JSON agregado. Sin esto, solo imprime a stdout.")
    parser.add_argument("--require-pair", action="append", default=[], metavar="FROM->TO",
                        help="Par dirigido exigido para habilitar política; repetir por cada par.")
    args = parser.parse_args(argv)

    report = aggregate_summaries(args.inputs, required_pairs=set(args.require_pair))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    conservative = report["t_transicion_gpu_ns_conservative"]
    print(f"corridas: {report['n_summaries']}  estables: {report['n_stable']}  "
          f"timeout: {report['n_timeout']}  otras: {report['n_other_non_stable']}")
    for pair, e in sorted(report["per_pair"].items()):
        print(f"  {pair:>16}  estables={e['stable_replicates']}  "
              f"cota_max_ns={e['conservative_upper_bound_ns_max']}")
    for w in report["warnings"]:
        print(f"  AVISO: {w}")

    if conservative is None:
        print("\nT_transicion_gpu_ns_conservative: SIN DATOS ESTABLES")
        print("-> no se puede habilitar la actuación GPU; documentar como bloqueo (F1-GPU-002).")
        return 1

    print(f"\nT_transicion_gpu_ns_conservative = {conservative}  "
          f"(cota superior observable; par peor: {report['worst_pair']})")
    print("Alimentar a la derivación de política con:")
    print(f"  python3 fase3_daemon/policy/derive_policy_table.py <windows.csv...> "
          f"--t-transicion-gpu-ns {conservative} --output policy_table.yaml")
    if not report["usable_for_policy"]:
        print("  (revisar los AVISOS de pares arriba antes de usar este valor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
