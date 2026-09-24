#!/usr/bin/env python3
"""Aplicación compuesta B de GPU -- familias INÉDITAS (Fase 3, ítem C2b, contraparte de GPU de `composite_unseen.py`).

Un solo kernel por fase, ambos de suites que NO están entre las 16 familias del entrenamiento del clasificador de GPU. Se
usa el mismo binario del catálogo (checksum verificado) con argumentos escalados para que cada fase dure más que la
ventana de decisión del daemon (3 s de actividad sostenida); la verdad de fase se declara aquí porque el catálogo no la
trae para estos argumentos:
  - rodinia_lavamd  -boxes1d 100                  compute_bound, ~10 s (job 7635; el daemon decide compute, conf 1.00)
  - gpu_stream_bw   --arraysize 100000000 --numtimes 1000   memory_bound, ~8 s (BabelStream, job 7636; decide memory, conf 1.00)
BabelStream es un STREAM (OI ~0.08 flop/byte, memory_bound por construcción); su verdad de fase fue aprobada por el
usuario el 2026-09-23. Con los argumentos del catálogo ninguno sostiene 3 s (lavamd 3.5 s, stream_bw 2.4 s).
Repetir lanzamientos cortos NO sirve: la actividad cae bajo el umbral entre procesos y el daemon nunca decide (job 7634).

Uso (en pacca, con el daemon de GPU aparte apuntando a este proceso con --target-pid):
  python3 composite_gpu_unseen.py --node-id pacca-a100 --kernels-root ~/hyperion-kernels --boundaries-out fronteras.jsonl
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.hpc.catalog import load_catalog  # noqa: E402
from fase3_daemon.composite_apps.composite_known import PhaseRecord, build_arg_parser, run_composite  # noqa: E402

# (id del catálogo, argumentos escalados, verdad de fase)
PHASES = (
    ("gpu_stream_bw", "--arraysize 100000000 --numtimes 1000", "memory_bound"),
    ("rodinia_lavamd", "-boxes1d 100", "compute_bound"),
)
DEFAULT_SEQUENCE = tuple(kid for kid, _, _ in PHASES)


def build_entries(catalog: dict) -> list:
    missing = [kid for kid, _, _ in PHASES if kid not in catalog]
    if missing:
        raise ValueError(f"kernel(s) no encontrados en el catálogo: {missing}")
    return [replace(catalog[kid], exec_args=args, phase_label_hint=hint) for kid, args, hint in PHASES]


def main() -> int:
    args = build_arg_parser(default_sequence=DEFAULT_SEQUENCE, description=__doc__, default_cycles=3).parse_args()
    entries = build_entries(load_catalog(str(args.catalog)))
    args.boundaries_out.parent.mkdir(parents=True, exist_ok=True)
    with args.boundaries_out.open("w", encoding="utf-8") as out:
        def on_phase(r: PhaseRecord) -> None:
            out.write(r.to_json() + "\n")
            out.flush()
            print(f"[ciclo {r.cycle}] {r.kernel_id} ({r.phase_label_hint}) {'OK' if r.success else 'FALLÓ'} "
                  f"en {(r.end_ns - r.begin_ns) / 1e9:.2f}s", flush=True)
        records = run_composite(entries, cycles=args.cycles, node_id=args.node_id,
                                kernels_root=args.kernels_root, on_phase=on_phase, gap_s=args.gap_s)
    n_failed = sum(1 for r in records if not r.success)
    print(f"aplicación compuesta B de GPU: {len(records)} fases, {n_failed} fallidas, registro en {args.boundaries_out}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
