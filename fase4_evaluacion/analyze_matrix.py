#!/usr/bin/env python3
"""Análisis de la matriz de la Fase 4 (`scripts/pacca/hyp_fase4_matrix.sbatch`): daemon vs REF.

Lee `results.csv` (una fila por celda: duración, energía de CPU y de GPU POR SEPARADO) y, por celda, las fronteras de fase
(`phases.jsonl`) y las decisiones de cada daemon (`cpu_decisions.jsonl`, `gpu_decisions.jsonl`). Produce:
  1. Por (alcance, kernels, brazo): medianas de T, E_cpu, E_gpu, E_total y EDP = (E_cpu + E_gpu) * T, y razones frente a REF
     (brazo `base`, menor es mejor). Con n >= 5 por brazo añade el p de Wilcoxon (bilateral, muestras no pareadas: Mann-Whitney).
  2. Puntuación de clasificación contra las fronteras reales: correctas, erróneas y abstenciones por (alcance, kernels, dispositivo),
     y cobertura (fases con al menos una decisión). Las decisiones se atribuyen a la fase que contiene su `ts_ns`.
Uso: python3 analyze_matrix.py RESULTADOS_DIR [--csv-out resumen.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(path.open()):
        for k in ("wall_s", "e_cpu_j", "e_gpu_j"):
            r[k] = float(r[k])
        r["e_total_j"] = r["e_cpu_j"] + r["e_gpu_j"]
        r["edp"] = r["e_total_j"] * r["wall_s"]
        rows.append(r)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["scope"], r["set"], r["arm"])].append(r)
    out = []
    for (scope, kset, arm), rs in sorted(groups.items()):
        ref = groups.get((scope, kset, "base"), [])
        med = {m: st.median(x[m] for x in rs) for m in ("wall_s", "e_cpu_j", "e_gpu_j", "e_total_j", "edp")}
        row = {"scope": scope, "set": kset, "arm": arm, "n": len(rs), **{k: round(v, 2) for k, v in med.items()}}
        if ref:
            for m, name in (("wall_s", "T"), ("e_cpu_j", "E_cpu"), ("e_gpu_j", "E_gpu"), ("e_total_j", "E_tot"), ("edp", "EDP")):
                row[f"ratio_{name}"] = round(med[m] / st.median(x[m] for x in ref), 3)
            row["p_edp"] = _p_value([x["edp"] for x in rs], [x["edp"] for x in ref]) if arm != "base" else None
        out.append(row)
    return out


def _p_value(a: list[float], b: list[float]) -> float | None:
    if len(a) < 5 or len(b) < 5:
        return None
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return None
    return round(float(mannwhitneyu(a, b, alternative="two-sided").pvalue), 4)


def _read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in path.open() if l.strip()]
    except FileNotFoundError:
        return []


def score_cell(phases: list[dict], decisions: list[dict]) -> dict:
    ok = wrong = abst = 0
    covered = set()
    for d in decisions:
        label = d.get("label")
        if label is None:
            continue
        ph = next((p for p in phases if p["begin_ns"] <= d["ts_ns"] <= p["end_ns"]), None)
        if ph is None:
            continue
        covered.add(id(ph))
        if label == "revisar" or d.get("policy_action") == "revisar":
            abst += 1
        elif label == ph["phase_label_hint"]:
            ok += 1
        else:
            wrong += 1
    return {"ok": ok, "wrong": wrong, "abst": abst, "phases": len(phases), "covered": len(covered)}


def score_all(root: Path, rows: list[dict]) -> list[dict]:
    agg: dict[tuple, dict] = defaultdict(lambda: {"ok": 0, "wrong": 0, "abst": 0, "phases": 0, "covered": 0})
    for r in rows:
        if r["arm"] == "base":
            continue
        cell = root / "cells" / r["cell"]
        phases = _read_jsonl(cell / "phases.jsonl")
        for device in ("cpu", "gpu"):
            decisions = _read_jsonl(cell / f"{device}_decisions.jsonl")
            if not decisions:
                continue
            # los daemons de CPU miran solo fases de CPU y los de GPU solo de GPU: se puntua contra las fases del dispositivo
            phases_dev = [p for p in phases if p["kernel_id"].startswith(("gpu_", "rodinia_")) == (device == "gpu")]
            s = score_cell(phases_dev, decisions)
            a = agg[(r["scope"], r["set"], device, r["arm"])]
            for k, v in s.items():
                a[k] += v
    return [{"scope": k[0], "set": k[1], "device": k[2], "arm": k[3], **v,
             "acc_decided": round(v["ok"] / (v["ok"] + v["wrong"]), 3) if v["ok"] + v["wrong"] else None}
            for k, v in sorted(agg.items())]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--csv-out", type=Path)
    a = ap.parse_args()
    rows = load_results(a.root / "results.csv")
    summ = summarize(rows)
    print(f"{'alcance':6s} {'kernels':7s} {'brazo':14s} {'n':>2s} {'T(s)':>7s} {'E_cpu(J)':>9s} {'E_gpu(J)':>9s} | {'T':>6s} {'E_cpu':>6s} {'E_gpu':>6s} {'E_tot':>6s} {'EDP':>6s} {'p':>7s}")
    for s in summ:
        print(f"{s['scope']:6s} {s['set']:7s} {s['arm']:14s} {s['n']:2d} {s['wall_s']:7.1f} {s['e_cpu_j']:9.0f} {s['e_gpu_j']:9.0f} | "
              + " ".join(f"{s.get('ratio_' + k, float('nan')):6.3f}" for k in ("T", "E_cpu", "E_gpu", "E_tot", "EDP"))
              + f" {s['p_edp'] if s.get('p_edp') is not None else '-':>7}")
    print("\nClasificación contra las fronteras reales (decisiones atribuidas por ts_ns):")
    for s in score_all(a.root, rows):
        print(f"  {s['scope']:6s} {s['set']:7s} {s['device']} {s['arm']:14s} correctas={s['ok']} erroneas={s['wrong']} abstenciones={s['abst']} "
              f"exactitud_decididas={s['acc_decided']} fases_con_decision={s['covered']}/{s['phases']}")
    if a.csv_out:
        keys = sorted({k for s in summ for k in s})
        with a.csv_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
