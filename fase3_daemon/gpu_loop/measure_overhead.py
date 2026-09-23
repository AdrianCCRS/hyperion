#!/usr/bin/env python3
"""Bloque C6 (Plan_Fase3_Daemon.md §0.1): mide la sobrecarga real del loop
de GPU -- tiempo de inferencia (`classify_fn`) + decisión/actuación
(`controller.on_phase_begin`) por transición de fase, contra el
clasificador REAL (`HistoricalGpuClassifier`, el mismo candidato congelado
de Fase 2) y eventos de fase REALES (el mismo kernel de terceros que C3,
`gpu_dgemm_n4096`, checksum verificado), con la tabla de política REAL
(`fase3_daemon/policy_table.yaml`) -- no una tabla sintética.

Complementa la sobrecarga ya medida del lado CPU (Bloque B,
`cpu_loop_latency_bench`: p50=16.4µs/p99=19.3µs contra un presupuesto de
~1ms por tick). Aquí no hay un presupuesto de tick equivalente -- el loop
de GPU decide una vez por FASE, no cada ~1ms -- así que se reporta la
latencia absoluta y se compara contra la duración típica de una fase real
(segundos): es la comparación que importa para el objetivo 2 (§0.1,
"sombra − base = sobrecarga del agente"), no un budget fijo arbitrario.

Reutiliza `build_daemon_gpu_loop` (brazo *sombra*, `dry_run=True`) tal
cual la usa `run_daemon.py` en producción -- corriendo en un hilo mientras
el kernel real se lanza en el hilo principal, mismo patrón de
coordinación que `verify_phase_detection_e2e.py` (C3)."""
from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.hpc.catalog import load_catalog, verify_binary  # noqa: E402
from fase3_daemon.gpu_loop.classifier import HistoricalGpuClassifier  # noqa: E402
from fase3_daemon.gpu_loop.controller import GpuPhaseDecision  # noqa: E402
from fase3_daemon.run_daemon import build_daemon_gpu_loop  # noqa: E402

_DEFAULT_CATALOG_PATH = _REPO_ROOT / "fase1_telemetria" / "catalog" / "catalog.yaml"
_DEFAULT_POLICY_TABLE_PATH = _REPO_ROOT / "fase3_daemon" / "policy_table.yaml"
_DEFAULT_MODELS_DIR = _REPO_ROOT / "fase2_clasificador" / "models"
_DEFAULT_KERNEL_ID = "gpu_dgemm_n4096"


@dataclass
class OverheadReport:
    n_decisions: int
    inference_ns: list[int]
    actuation_ns: list[int]

    @staticmethod
    def _percentiles(values: list[int]) -> dict[str, float]:
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        ordered = sorted(values)
        quantiles = statistics.quantiles(ordered, n=100, method="inclusive") if len(ordered) > 1 else [ordered[0]] * 99
        return {
            "p50": quantiles[49] if len(ordered) > 1 else ordered[0],
            "p95": quantiles[94] if len(ordered) > 1 else ordered[0],
            "p99": quantiles[98] if len(ordered) > 1 else ordered[0],
            "max": float(max(ordered)),
        }

    def to_summary(self) -> dict:
        return {
            "n_decisions": self.n_decisions,
            "inference_ns": self._percentiles(self.inference_ns),
            "actuation_ns": self._percentiles(self.actuation_ns),
        }

    @classmethod
    def from_decisions(cls, decisions: list[GpuPhaseDecision]) -> "OverheadReport":
        inference = [d.inference_time_ns for d in decisions if d.inference_time_ns is not None]
        actuation = [d.actuation_time_ns for d in decisions if d.actuation_time_ns is not None]
        return cls(n_decisions=len(decisions), inference_ns=inference, actuation_ns=actuation)


def run_overhead_measurement(
    *,
    policy_table_path: Path,
    models_dir: Path,
    model_name: str,
    catalog_path: Path,
    kernel_id: str,
    node_id: str,
    kernels_root: Path,
    cycles: int,
    gpu_index=None,
    poll_interval_s: float = 0.05,
    activity_threshold_pct: float = 5.0,
    end_margin_s: float = 2.0,
    poller_warmup_s: float = 1.0,
    run_fn=subprocess.run,
    sleep_fn=time.sleep,
    monotonic_fn=time.monotonic,
) -> OverheadReport:
    catalog = load_catalog(str(catalog_path))
    if kernel_id not in catalog:
        raise ValueError(f"kernel {kernel_id!r} no está en el catálogo")
    entry = catalog[kernel_id]
    resolved_exec_path = str(kernels_root / entry.exec_path)
    if not verify_binary(replace(entry, exec_path=resolved_exec_path), node_id=node_id):
        raise RuntimeError(
            f"C02: {entry.id!r} no pasó la verificación de checksum en node_id={node_id!r} "
            f"(exec_path resuelto: {resolved_exec_path})"
        )

    classifier = HistoricalGpuClassifier.from_export_dir(models_dir, name=model_name)

    decisions: list[GpuPhaseDecision] = []
    stop_polling_at = {"deadline": None}

    def should_continue() -> bool:
        deadline = stop_polling_at["deadline"]
        return deadline is None or monotonic_fn() < deadline

    def worker() -> None:
        nonlocal decisions
        decisions = build_daemon_gpu_loop(
            policy_table_path, gpu_index=gpu_index, min_dwell_ns=0, dry_run=True,
            classify_fn=classifier.classify, on_sample=classifier.record_sample,
            poll_interval_s=poll_interval_s, activity_threshold_pct=activity_threshold_pct,
            should_continue=should_continue, arm="sombra",
        )

    daemon_thread = threading.Thread(target=worker, daemon=True)
    daemon_thread.start()
    sleep_fn(poller_warmup_s)  # ver verify_phase_detection_e2e.py (C3) para por qué

    argv = [resolved_exec_path, *shlex.split(entry.exec_args)]
    for cycle in range(cycles):
        proc = run_fn(argv, cwd=str(kernels_root), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{kernel_id} (ciclo {cycle}) terminó con exit_code={proc.returncode}")

    stop_polling_at["deadline"] = monotonic_fn() + end_margin_s
    daemon_thread.join(timeout=end_margin_s + poll_interval_s * 4 + 15.0)

    return OverheadReport.from_decisions(decisions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG_PATH)
    parser.add_argument("--policy-table", type=Path, default=_DEFAULT_POLICY_TABLE_PATH)
    parser.add_argument("--models-dir", type=Path, default=_DEFAULT_MODELS_DIR)
    parser.add_argument("--model-name", default="gpu_random_forest_historical_20260923_sin_reloj")
    parser.add_argument("--kernel-id", default=_DEFAULT_KERNEL_ID)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--kernels-root", type=Path, default=Path.cwd())
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--gpu-index", default=None)
    parser.add_argument("--out", type=Path, default=None, help="Ruta JSON del reporte (requisito: 'registrada').")
    args = parser.parse_args()

    print(f"midiendo sobrecarga del loop de GPU: {args.kernel_id} x{args.cycles} ciclos, "
          f"clasificador={args.model_name}, política={args.policy_table}...")
    report = run_overhead_measurement(
        policy_table_path=args.policy_table, models_dir=args.models_dir, model_name=args.model_name,
        catalog_path=args.catalog, kernel_id=args.kernel_id, node_id=args.node_id,
        kernels_root=args.kernels_root, cycles=args.cycles, gpu_index=args.gpu_index,
    )
    summary = report.to_summary()
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**summary, "raw": asdict(report)}, indent=2, sort_keys=True))
        print(f"reporte escrito en {args.out}")

    # NO se exige n_decisions == cycles: el sondeo por umbral (Opcion C,
    # activity_poller.py) fusiona dos lanzamientos consecutivos en una
    # sola fase si el hueco entre ellos (el proceso termina y el siguiente
    # arranca) dura menos que poll_interval_s -- es una limitacion de
    # granularidad ya documentada, no un fallo de esta medicion. Lo unico
    # que debe cumplirse es haber medido AL MENOS una decision real.
    if report.n_decisions < 1:
        print("no se midió ninguna decisión real -- el sondeo nunca detectó actividad de GPU", file=sys.stderr)
        return 1
    if report.n_decisions < args.cycles:
        print(f"nota: se pidieron {args.cycles} ciclos pero se midieron {report.n_decisions} "
              "decisiones -- lanzamientos consecutivos sin hueco idle detectable se fusionan "
              "en una sola fase (granularidad de poll_interval_s, no es un fallo)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
