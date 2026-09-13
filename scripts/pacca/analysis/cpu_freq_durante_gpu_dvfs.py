"""Compara el tiempo de kernel GPU con CPU en REF vs CPU al minimo (F4),
usando la campana real pacca_gpu_dvfs_20260820 (288 combinaciones).

Pregunta que contesta: ¿fijar la CPU al minimo degrada el tiempo de los
kernels de GPU? Se compara pareado por (kernel, nivel_GPU, repeticion).
"""
import json
import os
from collections import defaultdict
from statistics import mean

ROOT = "/home/latorresn/hyperion-results/campaigns/pacca_gpu_dvfs_20260820"

# accepted_run_ids del metadata de campana, para no mezclar corridas rechazadas
meta = json.load(open(os.path.join(ROOT, "campaign_metadata.json")))
accepted = set(meta.get("accepted_run_ids", []))
print(f"corridas aceptadas en la campana: {len(accepted)}")
print(f"rechazadas: {len(meta.get('rejected_run_ids', []))}")

# (kernel, gpu_level, rep) -> {cpu_level: elapsed_ns}
data = defaultdict(dict)
skipped = 0

for name in sorted(os.listdir(ROOT)):
    path = os.path.join(ROOT, name)
    mfile = os.path.join(path, "metadata.json")
    if not os.path.isdir(path) or not os.path.exists(mfile):
        continue
    if accepted and name not in accepted:
        skipped += 1
        continue
    try:
        m = json.load(open(mfile))
    except Exception:
        continue
    gpu_lvl = m.get("gpu_freq_level_id")
    cpu_lvl = m.get("freq_level_id")
    elapsed = m.get("telemetry_elapsed_ns_mean")
    if not gpu_lvl or not cpu_lvl or not elapsed:
        continue  # kernels de calibracion (sin nivel GPU) quedan fuera
    # nombre del kernel: <campaign>__<kernel>__<cpu>__gpu<gpu>__repNN
    parts = name.split("__")
    kernel = parts[1] if len(parts) > 1 else "?"
    rep = parts[-1]
    data[(kernel, gpu_lvl, rep)][cpu_lvl] = elapsed

print(f"corridas no aceptadas omitidas: {skipped}")

# Comparacion pareada: solo tripletas que tengan AMBOS niveles de CPU
pares = []
for (kernel, gpu_lvl, rep), by_cpu in sorted(data.items()):
    if "REF" in by_cpu and "F4" in by_cpu:
        ref, f4 = by_cpu["REF"], by_cpu["F4"]
        delta_pct = (f4 - ref) / ref * 100.0
        pares.append((kernel, gpu_lvl, rep, ref, f4, delta_pct))

print(f"\npares comparables (mismo kernel/nivel GPU/rep, REF vs F4): {len(pares)}")
if not pares:
    raise SystemExit("No hay pares comparables.")

# Resumen por kernel
print("\n=== delta de tiempo con CPU al minimo (F4) vs CPU nativa (REF) ===")
print("    positivo = F4 es MAS LENTO ; negativo = F4 es MAS RAPIDO\n")
por_kernel = defaultdict(list)
for kernel, gpu_lvl, rep, ref, f4, d in pares:
    por_kernel[kernel].append(d)

print(f"{'kernel':<32} {'n':>4} {'media %':>9} {'min %':>8} {'max %':>8}")
print("-" * 66)
for kernel in sorted(por_kernel):
    ds = por_kernel[kernel]
    print(f"{kernel:<32} {len(ds):>4} {mean(ds):>9.2f} {min(ds):>8.2f} {max(ds):>8.2f}")

todos = [d for _, _, _, _, _, d in pares]
print("-" * 66)
print(f"{'GLOBAL':<32} {len(todos):>4} {mean(todos):>9.2f} {min(todos):>8.2f} {max(todos):>8.2f}")

# Resumen por nivel de GPU (¿el efecto cambia con el reloj de GPU?)
print("\n=== mismo delta, agrupado por nivel de GPU ===")
por_gpu = defaultdict(list)
for kernel, gpu_lvl, rep, ref, f4, d in pares:
    por_gpu[gpu_lvl].append(d)
print(f"{'nivel GPU':<12} {'n':>4} {'media %':>9}")
print("-" * 28)
for lvl in sorted(por_gpu):
    ds = por_gpu[lvl]
    print(f"{lvl:<12} {len(ds):>4} {mean(ds):>9.2f}")

# Magnitud: cuantos pares superan umbrales de interes
for umbral in (1.0, 5.0, 10.0):
    n = sum(1 for d in todos if d > umbral)
    print(f"\npares donde F4 es >{umbral:.0f}% mas lento: {n}/{len(todos)} ({n/len(todos)*100:.1f}%)")
