#!/usr/bin/env python3
"""Aplicación compuesta del escenario E de la Fase 4 (Plan_Fase4_Escenario_E.md): GPU dominada por fases memory_bound largas.

Un solo kernel por fase, con fases de memoria de 30 s o más: el agente de GPU decide una vez por fase, con ~8 s desde el
inicio hasta el primer reloj aplicado (medido en la matriz A), de modo que una fase de 18 s solo pasa el 55% del tiempo en
F1 y una de 50 s pasa el 84%.

  --set known   familias vistas en la Fase 2 con ganancia de EDP con F1 >= 10%: dual_stencil, dual_spmv, dual_axpy, más
                una fase compute corta (DGEMM de CUTLASS) para que haya conmutación.
  --set unseen  familias memory_bound del catálogo que NO están entre las 16 del entrenamiento (BabelStream escalado,
                Rodinia myocyte escalado; solo dos porque no hay un tercer kernel inédito de memoria que dure 25 s o más), más una fase
                compute inédita (LavaMD escalado).

La verdad de fase de los dual_* se declara aquí desde su clase de la Fase 2 (`kernel_class.csv`, margen 1.0, no ambigua;
OI de 0.09 a 0.27 frente a un ridge fp64 de 3.36), porque el catálogo los marca `intermedio`. Los de la variante inédita
llevan la clase del catálogo (OI << ridge) o la declarada en `composite_gpu_unseen`. Criterio de selección declarado antes
de correr: "familias memory_bound cuya ganancia de EDP con F1 en la Fase 2 fue de al menos 10%, más una variante con
familias no vistas en el entrenamiento".

Uso: python3 composite_gpu_memdom.py --set known --node-id pacca-a100 --kernels-root ~/hyperion-kernels --boundaries-out f.jsonl
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

# (id del catálogo, argumentos o None para los del catálogo, verdad de fase)
KNOWN = (
    ("dual_stencil_gpu_N36864", None, "memory_bound"),
    ("gpu_cutlass_simt_dgemm_n4096", None, "compute_bound"),
    ("dual_spmv_gpu_N200000000", None, "memory_bound"),
    ("dual_axpy_gpu_N1280000000", None, "memory_bound"),
)
# Duraciones medidas en el sondeo (job 7651): BabelStream numtimes 5000 = 36 s, myocyte 1000000 = 48 s. Descartados: backprop
# (falla con los dos tamanos probados), indexlist_3loop y reduce3_int (5 a 6 s, sin argumentos para alargarlos).
UNSEEN = (
    ("gpu_stream_bw", "--arraysize 100000000 --numtimes 5000", "memory_bound"),
    ("rodinia_lavamd", "-boxes1d 100", "compute_bound"),
    ("rodinia_myocyte", "1000000 1 0", "memory_bound"),
)
PHASES = {"known": KNOWN, "unseen": UNSEEN}


def build_entries(catalog: dict, which: str) -> list:
    """Entradas en el orden de PHASES[which], con argumentos y verdad de fase declarados. Falla cerrado si falta un kernel."""
    if which not in PHASES:
        raise ValueError(f"--set debe ser known o unseen, no {which!r}")
    missing = [kid for kid, _, _ in PHASES[which] if kid not in catalog]
    if missing:
        raise ValueError(f"kernel(s) no encontrados en el catálogo: {missing}")
    return [replace(catalog[kid], exec_args=args if args is not None else catalog[kid].exec_args, phase_label_hint=hint)
            for kid, args, hint in PHASES[which]]


def main() -> int:
    parser = build_arg_parser(default_sequence=(), description=__doc__, default_cycles=1)
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
    print(f"aplicación compuesta de GPU dominada por memoria ({args.which}): {len(records)} fases, {n_failed} fallidas, "
          f"registro en {args.boundaries_out}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
