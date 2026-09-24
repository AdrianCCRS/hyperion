#!/usr/bin/env python3
"""Aplicación compuesta CONJUNTA CPU + GPU (Fase 4): alterna fases de CPU y de GPU en un mismo proceso, para correr los dos
daemons a la vez con fronteras de fase propias (RAPL y NVML no se mezclan entre fases).

  --set known   familias vistas:   dgemm_n2048 (CPU), npb_cg (CPU), gpu_cutlass_simt_dgemm_n4096 (GPU), gpu_rajaperf_stream_triad (GPU)
  --set unseen  familias inéditas: cpu_xsbench_omp (CPU), cpu_rsbench_omp (CPU), gpu_stream_bw escalado (GPU), rodinia_lavamd escalado (GPU)
La verdad de fase sale del catálogo; los kernels de GPU inéditos usan los argumentos escalados de `composite_gpu_unseen.PHASES`.
Uso: python3 composite_joint.py --set known --node-id pacca-a100 --kernels-root ~/hyperion-kernels --boundaries-out f.jsonl
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.hpc.catalog import load_catalog  # noqa: E402
from fase3_daemon.composite_apps.composite_gpu_unseen import PHASES as GPU_UNSEEN_PHASES  # noqa: E402
from fase3_daemon.composite_apps.composite_known import PhaseRecord, build_arg_parser, run_composite  # noqa: E402

KNOWN = ("dgemm_n2048", "npb_cg", "gpu_cutlass_simt_dgemm_n4096", "gpu_rajaperf_stream_triad")
UNSEEN_CPU = ("cpu_xsbench_omp", "cpu_rsbench_omp")


def build_entries(catalog: dict, which: str) -> list:
    """Entradas en orden CPU, CPU, GPU, GPU. Falla cerrado si algo falta o no trae verdad de fase."""
    if which == "known":
        ids = KNOWN
        entries = [catalog[k] for k in ids if k in catalog]
    elif which == "unseen":
        ids = UNSEEN_CPU
        entries = [catalog[k] for k in ids if k in catalog]
        entries += [replace(catalog[k], exec_args=a, phase_label_hint=h) for k, a, h in GPU_UNSEEN_PHASES if k in catalog]
        ids = ids + tuple(k for k, _, _ in GPU_UNSEEN_PHASES)
    else:
        raise ValueError(f"--set debe ser known o unseen, no {which!r}")
    if len(entries) != len(ids):
        raise ValueError(f"kernel(s) no encontrados en el catálogo: {[k for k in ids if k not in {e.id for e in entries}]}")
    if any(not e.phase_label_hint for e in entries):
        raise ValueError("kernel(s) sin phase_label_hint: no sirven como frontera de fase conocida")
    return entries


def main() -> int:
    parser = build_arg_parser(default_sequence=(), description=__doc__, default_cycles=2)
    parser.add_argument("--set", dest="which", choices=["known", "unseen"], required=True)
    args = parser.parse_args()
    entries = build_entries(load_catalog(str(args.catalog)), args.which)
    args.boundaries_out.parent.mkdir(parents=True, exist_ok=True)
    with args.boundaries_out.open("w", encoding="utf-8") as out:
        def on_phase(r: PhaseRecord) -> None:
            out.write(r.to_json() + "\n")
            out.flush()
            print(f"[ciclo {r.cycle}] {r.kernel_id} ({r.phase_label_hint}) {'OK' if r.success else 'FALLÓ'} "
                  f"en {(r.end_ns - r.begin_ns) / 1e9:.2f}s", flush=True)
        records = run_composite(entries, cycles=args.cycles, node_id=args.node_id, kernels_root=args.kernels_root,
                                on_phase=on_phase, gap_s=args.gap_s)
    n_failed = sum(1 for r in records if not r.success)
    print(f"aplicación compuesta conjunta ({args.which}): {len(records)} fases, {n_failed} fallidas, registro en {args.boundaries_out}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
