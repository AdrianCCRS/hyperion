"""Mide operational_intensity_flops_per_byte real (ncu) para los 6
candidatos de diversidad GPU de RAJAPerf-CUDA que sobrevivieron el
tamizaje de alpha (screen_rajaperf_gpu_alpha.sh, 2026-08-25, ver
Estrategia_GPU_Fase2.md §8). Necesario antes de darlos de alta en
catalog.yaml: cada kernel real (no de calibracion) del catalogo GPU
declara su OI medida con `ncu`, nunca estimada -- mismo requisito que
`rodinia_lavamd`/`gpu_ert_probe_fp32` etc. (ARC-76 y siguientes).

Mismo patron que probe_gaussian_ncu.py: `ncu --metrics <ALL_METRICS>
--launch-count N --csv`, parseado con ncu_gpu_precision.py (ya validado
contra los kernels Rodinia del catalogo).

--sizefact 100: mismo motivo que screen_rajaperf_gpu_alpha.sh -- al
tamano por defecto el binario esta dominado por arranque de contexto
CUDA (~380ms fijos), no por el kernel; a sizefact=100 el computo real
domina y ncu mide trafico DRAM representativo del regimen real, no del
transitorio de arranque.

--launch-count 20 (mismo valor que probe_gaussian_ncu.py): RAJAPerf
repite cada kernel internamente varias veces por corrida (repfact=1.0
por defecto), ncu multiplexa sobre esos lanzamientos reales, no hace
falta --repfact adicional.

Uso: python3 measure_gpu_rajaperf_oi.py
"""
import sys
import os
import subprocess
import csv as csv_module

sys.path.insert(0, "/home/latorresn/hyperion")
sys.path.insert(0, "/home/latorresn/hyperion/docs/justifications/scripts")
from orchestrator.gpu_shim import cuda_lib_dirs  # noqa: E402
from ncu_gpu_precision import (  # noqa: E402
    ALL_METRICS,
    compute_gpu_precision_result,
    parse_ncu_csv_totals,
)

BINARY = "/home/latorresn/hyperion-kernels/libexec/raja-perf-cuda-v2025.12.1"
KERNELS = [
    "Stream_COPY",
    "Stream_TRIAD",
    "Basic_REDUCE3_INT",
    "Basic_INDEXLIST_3LOOP",
    "Polybench_JACOBI_2D",
    "Polybench_HEAT_3D",
]

env = dict(os.environ)
dirs = cuda_lib_dirs()
if dirs:
    env["LD_LIBRARY_PATH"] = ":".join(str(x) for x in dirs) + ":" + env.get("LD_LIBRARY_PATH", "")

results = {}
for kernel in KERNELS:
    cmd = [
        "ncu", "--metrics", ",".join(ALL_METRICS), "--launch-count", "20", "--csv",
        BINARY, "-k", kernel, "-v", "Base_CUDA", "--sizefact", "100",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if completed.returncode != 0:
        print(f"{kernel}: ncu FALLO rc={completed.returncode}")
        print("STDERR:", completed.stderr[-2000:])
        continue

    csv_path = f"/home/latorresn/yacacerest/ncu_{kernel.lower()}_probe.csv"
    with open(csv_path, "w") as handle:
        handle.write(completed.stdout)

    totals, n_launches = parse_ncu_csv_totals(completed.stdout)
    result = compute_gpu_precision_result(totals, n_launches)
    results[kernel] = result
    print(
        f"{kernel:<28} n_launches={n_launches:<5} "
        f"OI={result.operational_intensity:.4f} FLOP/byte "
        f"dram_bytes={result.dram_bytes:.3e} "
        f"flops_fp32={result.flops_fp32:.3e} flops_fp64={result.flops_fp64:.3e} "
        f"frac_fp32={result.fraction_fp32:.3f} frac_fp64={result.fraction_fp64:.3f}"
    )

print()
print("=== RESUMEN (para catalog.yaml) ===")
for kernel, result in results.items():
    precision = "fp64" if result.fraction_fp64 >= result.fraction_fp32 else "fp32"
    print(f"{kernel}: operational_intensity_flops_per_byte={result.operational_intensity:.4f}  gpu_precision={precision}")
