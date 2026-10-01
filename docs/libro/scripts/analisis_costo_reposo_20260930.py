"""Descomposición del costo en reposo del agente de CPU (job 7829, pacca01).

Por celda: potencia de CPU (RAPL de ambos paquetes / duración), procesadores ocupados
del nodo (deltas de /proc/stat), residencia en C6 de las CPU 0-7 y 16-23, interrupciones
LOC/CAL y tiempo de CPU de los hilos de cpu_loop_main. Imprime una fila por celda y
la media por brazo.

Datos: datos/fase4_20260930/costo_reposo_pacca01_7829/
"""
import csv
import statistics as st
import sys
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "datos/fase4_20260930" / (sys.argv[1] if len(sys.argv) > 1 else "costo_reposo_pacca01_7829")
CPUS = [*range(8), *range(16, 24)]


def leer(p):
    sec, out = None, {"stat": {}, "irq": {}, "idle": {}, "hilos": []}
    for ln in open(p):
        ln = ln.rstrip("\n")
        if ln.startswith("## "):
            sec = ln.split()[1]
            continue
        f = ln.split()
        if not f:
            continue
        if sec == "stat" and f[0].startswith("cpu") and f[0] != "cpu":
            v = list(map(int, f[1:]))
            out["stat"][int(f[0][3:])] = (sum(v) - v[3] - v[4], sum(v))
        elif sec == "interrupts" and f[0] in ("LOC:", "CAL:"):
            out["irq"][f[0][:3]] = sum(int(x) for x in f[1:] if x.isdigit())
        elif sec == "cpuidle":
            out["idle"][(int(f[0][3:]), f[1])] = int(f[2])
        elif sec == "daemon":
            out["hilos"].append(list(map(int, f[1:3])))
    return out


filas = list(csv.DictReader(open(D / "results.csv")))
por_brazo = {}
print("celda,potencia_w,cpus_ocupadas,c6_frac_0_7_16_23,loc_k_s,cal_k_s,agente_cpu_s")
for r in filas:
    a, b = leer(D / "cells" / r["cell"] / "snap_inicio.txt"), leer(D / "cells" / r["cell"] / "snap_fin.txt")
    w = float(r["wall_s"])
    p = float(r["e_cpu_j"]) / w
    ocup = sum(b["stat"][c][0] - a["stat"][c][0] for c in b["stat"]) / 100 / w  # jiffies a 100 Hz
    c6 = st.mean((b["idle"][(c, "C6")] - a["idle"][(c, "C6")]) / 1e6 / w for c in CPUS)
    loc = (b["irq"]["LOC"] - a["irq"]["LOC"]) / w / 1e3
    cal = (b["irq"]["CAL"] - a["irq"]["CAL"]) / w / 1e3
    ag = sum(sum(x) for x in b["hilos"]) / 100 - sum(sum(x) for x in a["hilos"]) / 100 if a["hilos"] else 0.0
    print(f"{r['cell']},{p:.1f},{ocup:.2f},{c6:.3f},{loc:.1f},{cal:.1f},{ag:.2f}")
    por_brazo.setdefault(r["arm"], []).append((p, ocup, c6, loc, cal, ag))
print("\nbrazo,n,potencia_w,cpus_ocupadas,c6_frac,loc_k_s,cal_k_s,agente_cpu_s")
for arm, v in por_brazo.items():
    m = [st.mean(x[i] for x in v) for i in range(6)]
    print(f"{arm},{len(v)}," + ",".join(f"{x:.2f}" for x in m) + f"  (potencias {[round(x[0],1) for x in v]})")
