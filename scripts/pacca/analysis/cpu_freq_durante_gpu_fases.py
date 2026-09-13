"""Campana del selector (pacca_dual_gpu_full_20260828): separa el efecto de
la frecuencia de CPU sobre la FASE CALIENTE (ejecucion en regimen) vs la
FASE FRIA/SETUP (trabajo del host), usando el contrato cold_warm_v1.

Pregunta: cuando bajamos el reloj de CPU durante una carga GPU, ¿que se
degrada? ¿el setup del host, o tambien la ejecucion en regimen?
"""
import json
import os
from collections import defaultdict
from statistics import mean

ROOT = "/home/latorresn/hyperion-results/campaigns/pacca_dual_gpu_full_20260828"

meta_path = os.path.join(ROOT, "campaign_metadata.json")
accepted = set()
if os.path.exists(meta_path):
    accepted = set(json.load(open(meta_path)).get("accepted_run_ids", []))
print(f"aceptadas segun campaign_metadata: {len(accepted)}")

# (kernel, gpu_lvl, rep) -> cpu_lvl -> metricas
data = defaultdict(dict)
cpu_levels_seen = set()
sin_dt = 0

for name in os.listdir(ROOT):
    path = os.path.join(ROOT, name)
    mfile = os.path.join(path, "metadata.json")
    if not os.path.isdir(path) or not os.path.exists(mfile):
        continue
    if accepted and name not in accepted:
        continue
    try:
        m = json.load(open(mfile))
    except Exception:
        continue
    kref = m.get("kernel_ref", "")
    # solo kernels de GPU: la pregunta es sobre carga GPU
    if "gpu" not in kref.lower():
        continue
    gpu_lvl = m.get("gpu_freq_level_id")
    cpu_lvl = m.get("freq_level_id")
    total = m.get("telemetry_elapsed_ns_mean")
    dt = m.get("dispatch_timing") or {}
    if not gpu_lvl or not cpu_lvl or not total:
        continue
    if not dt:
        sin_dt += 1
        continue
    cpu_levels_seen.add(cpu_lvl)
    rep = m.get("repetition_index")
    data[(kref, gpu_lvl, rep)][cpu_lvl] = {
        "total_s": total / 1e9,
        "warm_s": dt.get("warm_total_seconds"),
        "cold_s": dt.get("cold_total_seconds"),
        "setup_s": dt.get("setup_seconds"),
    }

print(f"niveles de CPU encontrados: {sorted(cpu_levels_seen)}")
print(f"corridas GPU sin dispatch_timing (omitidas): {sin_dt}")

lo, hi = "F6", "F0"  # F0 = reloj maximo, F6 = minimo
if lo not in cpu_levels_seen or hi not in cpu_levels_seen:
    raise SystemExit(f"faltan niveles {hi}/{lo}; hay {sorted(cpu_levels_seen)}")

def pct(a, b):
    return (a - b) / b * 100.0 if b else float("nan")

filas = []
for key, by_cpu in data.items():
    if hi in by_cpu and lo in by_cpu:
        a, b = by_cpu[lo], by_cpu[hi]   # a = CPU al minimo, b = CPU al maximo
        if None in (a["warm_s"], b["warm_s"], a["setup_s"], b["setup_s"]):
            continue
        filas.append((
            key[0],
            pct(a["total_s"], b["total_s"]),
            pct(a["warm_s"], b["warm_s"]),
            pct(a["setup_s"], b["setup_s"]),
        ))

print(f"\npares comparables CPU {hi}(max) vs {lo}(min): {len(filas)}")
if not filas:
    raise SystemExit("sin pares")

print(f"\n=== degradacion al bajar CPU de {hi} a {lo} (positivo = mas lento) ===\n")
por_kernel = defaultdict(list)
for k, t, w, s in filas:
    por_kernel[k].append((t, w, s))

print(f"{'kernel':<30} {'n':>4} {'TOTAL %':>9} {'CALIENTE %':>11} {'SETUP %':>9}")
print("-" * 68)
for k in sorted(por_kernel):
    v = por_kernel[k]
    print(f"{k:<30} {len(v):>4} {mean(x[0] for x in v):>9.1f} "
          f"{mean(x[1] for x in v):>11.1f} {mean(x[2] for x in v):>9.1f}")
print("-" * 68)
print(f"{'GLOBAL':<30} {len(filas):>4} {mean(x[1] for x in filas):>9.1f} "
      f"{mean(x[2] for x in filas):>11.1f} {mean(x[3] for x in filas):>9.1f}")
