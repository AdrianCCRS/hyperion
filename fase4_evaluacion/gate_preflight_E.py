#!/usr/bin/env python3
"""Compuerta del pre-vuelo del escenario E (Plan_Fase4_Escenario_E.md, paso 4): decide si las corridas largas pueden arrancar.

Lee la carpeta del pre-vuelo (`results.csv` y `cells/*/{phases,gpu_decisions}.jsonl`) y falla (codigo 1) si:
  1. falta alguna celda esperada, o alguna tiene app_rc, rc de daemon distinto de 0 o state_ok distinto de 1;
  2. alguna fase de memoria dura menos de MIN_MEM_S (el agente decide una vez por fase, ~8 s despues del inicio);
  3. la fraccion del ciclo en fases memory_bound es menor que MIN_MEM_FRAC;
  4. en los brazos activos, menos de 2/3 de las fases de memoria recibieron una decision memory_bound con escritura de reloj
     (mas abstenciones que eso: parar y consultar, no cambiar el umbral).
Uso: python3 gate_preflight_E.py CARPETA_PREVUELO
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

MIN_MEM_S = 25.0
MIN_MEM_FRAC = 0.70
MIN_APPLIED = 2 / 3
EXPECTED_ARMS = ("base", "base_noturbo", "activo_gpu", "activo_gpu_cpuobs")
ACTIVE_ARMS = ("activo_gpu", "activo_gpu_cpuobs")


def _jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in path.open() if l.strip()]
    except FileNotFoundError:
        return []


def check(root: Path) -> list[str]:
    problems: list[str] = []
    rows = list(csv.DictReader((root / "results.csv").open())) if (root / "results.csv").exists() else []
    have = {(r["set"], r["arm"]) for r in rows}
    for st in ("known", "unseen"):
        for arm in EXPECTED_ARMS:
            if (st, arm) not in have:
                problems.append(f"falta la celda {st}/{arm}")
    for r in rows:
        if r["app_rc"] != "0" or r["state_ok"] != "1" or r["rc_cpu_daemon"] not in ("0", "NA") or r["rc_gpu_daemon"] not in ("0", "NA"):
            problems.append(f"{r['cell']}: app_rc={r['app_rc']} state_ok={r['state_ok']} rc_cpu={r['rc_cpu_daemon']} rc_gpu={r['rc_gpu_daemon']}")
    applied = {"known": [0, 0], "unseen": [0, 0]}
    for r in rows:
        ph = _jsonl(root / "cells" / r["cell"] / "phases.jsonl")
        mem = [p for p in ph if p["phase_label_hint"] == "memory_bound"]
        total = sum(p["end_ns"] - p["begin_ns"] for p in ph) or 1
        for p in mem:
            if (p["end_ns"] - p["begin_ns"]) / 1e9 < MIN_MEM_S:
                problems.append(f"{r['cell']}: fase de memoria {p['kernel_id']} dura {(p['end_ns'] - p['begin_ns']) / 1e9:.1f} s (< {MIN_MEM_S:.0f})")
        frac = sum(p["end_ns"] - p["begin_ns"] for p in mem) / total
        if r["arm"] == "base" and frac < MIN_MEM_FRAC:
            problems.append(f"{r['cell']}: fraccion en memoria {frac:.2f} (< {MIN_MEM_FRAC})")
        if r["arm"] in ACTIVE_ARMS:
            dec = _jsonl(root / "cells" / r["cell"] / "gpu_decisions.jsonl")
            for p in mem:
                applied[r["set"]][1] += 1
                if any(p["begin_ns"] <= d["ts_ns"] <= p["end_ns"] and d.get("label") == "memory_bound" and d.get("written") for d in dec):
                    applied[r["set"]][0] += 1
    for st, (ok, n) in applied.items():
        if n == 0 or ok / n < MIN_APPLIED:
            problems.append(f"{st}: F1 aplicado en {ok} de {n} fases de memoria (< {MIN_APPLIED:.2f})")
    return problems


def main() -> int:
    problems = check(Path(sys.argv[1]))
    for p in problems:
        print("GATE FALLA:", p)
    print("GATE OK" if not problems else f"GATE: {len(problems)} problema(s), las corridas largas NO arrancan")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
