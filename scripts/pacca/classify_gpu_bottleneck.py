"""Clasifica el CUELLO DE BOTELLA real de cada kernel de RAJAPerf-CUDA,
midiendolo en vez de inferirlo.

POR QUE EXISTE (2026-08-25). Hasta ahora la memory-boundness se deducia de
un SINTOMA: ajustar alpha sobre el tiempo de pared a distintos relojes y
ver si el tiempo escalaba. Eso tiene tres problemas: (a) confunde
"insensible al reloj" con "limitado por memoria" -- un kernel dominado por
overhead de host tambien da alpha bajo, que es exactamente como se colo
`dwt2d` (riesgo 8); (b) el ajuste de Amdahl no describe kernels que
saturan a bajo reloj (riesgo 6, r2=0.53-0.75 en varios candidatos); y (c)
cuesta una corrida por nivel de frecuencia.

`ncu` mide la CAUSA directamente, en una sola corrida y sin tocar el
reloj: `dram__throughput` y `sm__throughput` como porcentaje del pico
sostenido. Un kernel limitado por memoria tiene DRAM% alto y SM% bajo; uno
limitado por computo, al reves. Es la posicion en el Roofline MEDIDA, no
derivada de comparar una intensidad operacional declarada contra un ridge
calibrado.

QUE HACE CON ESO. Ordena los 79 kernels por DRAM% para que el tamizaje de
alpha (mas caro: 4-5 corridas por kernel) se gaste solo en los candidatos
que la medicion directa ya señala, en vez de elegirlos por conocimiento
algoritmico como se hizo con los 6 primeros -- criterio que acerto 4 de 6
pero que no escala a 79.

--sizefact 100: mismo motivo que en el resto del pipeline de GPU (a tamaño
por defecto el binario esta dominado por ~380 ms fijos de arranque de
contexto CUDA). Aqui importa aun mas: ncu reporta metricas POR KERNEL de
CUDA, y con un problema minusculo el kernel puede ni siquiera alcanzar
regimen estacionario.

Uso: python3 classify_gpu_bottleneck.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/latorresn/hyperion")
from orchestrator.gpu_shim import cuda_lib_dirs  # noqa: E402

BINARY = "/home/latorresn/hyperion-kernels/libexec/raja-perf-cuda-v2025.12.1"

# pct_of_peak_sustained_elapsed: fraccion del pico sostenible que el kernel
# realmente alcanzo, promediada sobre el tiempo transcurrido. Es la
# magnitud que hace comparables a dos kernels de tamaños distintos.
METRICS = [
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__time_duration.sum",
]

# Umbral de candidatura, deliberadamente PERMISIVO. No decide nada por si
# mismo: solo ordena a quien se le gasta el tamizaje de alpha, que es el
# que aplica los criterios reales. Un umbral estricto aqui volveria a
# introducir el sesgo de seleccion que este script existe para eliminar.
DRAM_CANDIDATE_PCT = 30.0


def list_kernels() -> list[str]:
    completed = subprocess.run(
        [BINARY, "--print-kernels"], capture_output=True, text=True, timeout=120,
    )
    kernels = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if "_" in line and not line.startswith("-") and " " not in line:
            kernels.append(line)
    return kernels


def parse_ncu(csv_text: str) -> dict[str, float]:
    """Suma `gpu__time_duration` y promedia los throughputs pesando por esa
    duracion. Promediar sin pesar mezclaria un kernel de arranque trivial
    con el kernel real de trabajo y correria el resultado hacia el ruido."""
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if not rows:
        return {}

    by_launch: dict[str, dict[str, float]] = {}
    for row in rows:
        name = row.get("Metric Name", "")
        value = row.get("Metric Value", "")
        key = f"{row.get('ID', '')}:{row.get('Kernel Name', '')}"
        if not name or value in (None, ""):
            continue
        try:
            numeric = float(value.replace(",", ""))
        except ValueError:
            continue
        by_launch.setdefault(key, {})[name] = numeric

    total_time = 0.0
    dram_weighted = sm_weighted = 0.0
    for metrics in by_launch.values():
        duration = metrics.get("gpu__time_duration.sum", 0.0)
        if duration <= 0:
            continue
        total_time += duration
        dram_weighted += metrics.get(METRICS[0], 0.0) * duration
        sm_weighted += metrics.get(METRICS[1], 0.0) * duration

    if total_time <= 0:
        return {}
    return {
        "dram_pct": dram_weighted / total_time,
        "sm_pct": sm_weighted / total_time,
        "total_kernel_time_ns": total_time,
        "n_launches": len(by_launch),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sizefact", type=int, default=100)
    args = parser.parse_args()

    env = dict(os.environ)
    dirs = cuda_lib_dirs()
    if dirs:
        env["LD_LIBRARY_PATH"] = ":".join(str(d) for d in dirs) + ":" + env.get("LD_LIBRARY_PATH", "")

    kernels = list_kernels()
    if args.limit:
        kernels = kernels[: args.limit]
    print(f"kernels a clasificar: {len(kernels)}", flush=True)
    print("kernel,dram_pct,sm_pct,total_kernel_time_ns,n_launches,veredicto", flush=True)

    results = []
    for kernel in kernels:
        cmd = [
            "ncu", "--metrics", ",".join(METRICS), "--launch-count", "10", "--csv",
            BINARY, "-k", kernel, "-v", "Base_CUDA", "--sizefact", str(args.sizefact),
        ]
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, env=env,
            )
        except subprocess.TimeoutExpired:
            print(f"{kernel},,,,,TIMEOUT", flush=True)
            continue
        if completed.returncode != 0:
            print(f"{kernel},,,,,NCU_RC={completed.returncode}", flush=True)
            continue

        parsed = parse_ncu(completed.stdout)
        if not parsed:
            print(f"{kernel},,,,,SIN_METRICAS", flush=True)
            continue

        dram, sm = parsed["dram_pct"], parsed["sm_pct"]
        if dram >= DRAM_CANDIDATE_PCT and dram > sm:
            verdict = "MEMORY_BOUND"
        elif sm > dram:
            verdict = "compute_bound"
        else:
            verdict = "bajo_ambos"  # ni satura memoria ni computo: sospechoso de overhead
        print(f"{kernel},{dram:.2f},{sm:.2f},{parsed['total_kernel_time_ns']:.0f},"
              f"{parsed['n_launches']},{verdict}", flush=True)
        results.append((kernel, dram, sm, verdict))

    print(flush=True)
    print("=== RESUMEN: candidatos ordenados por DRAM% ===", flush=True)
    memory_bound = sorted(
        [r for r in results if r[3] == "MEMORY_BOUND"], key=lambda r: -r[1]
    )
    for kernel, dram, sm, _ in memory_bound:
        print(f"  {kernel:<34} DRAM={dram:6.2f}%  SM={sm:6.2f}%", flush=True)
    print(f"\ncandidatos MEMORY_BOUND: {len(memory_bound)} de {len(results)} medidos",
          flush=True)
    print("BOTTLENECK_CLASSIFICATION_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
