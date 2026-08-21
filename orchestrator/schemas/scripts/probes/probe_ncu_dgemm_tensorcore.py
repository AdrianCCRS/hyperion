#!/usr/bin/env python3
"""Diagnostico de solo lectura (no toca el catalogo ni el harness): perfila
bin/cublas_dgemm_bench (el binario detras de gpu_dgemm_calibration y
gpu_dgemm_n4096) con ncu, pidiendo SIMULTANEAMENTE los contadores clasicos
de FMA/ADD/MUL FP64 y los contadores de Tensor Core (DMMA), para confirmar
si el 68.0 FLOP/byte declarado en el catalogo para gpu_dgemm_n4096 viene de
una ruta Tensor Core (ARC-75/76 ya lo confirmo para gpu_dgemm_calibration,
mismo binario) y si el metodo de conteo clasico (dfma/dadd/dmul) captura
correctamente ese trabajo o lo subestima.
"""
import os
import sys
import csv
import subprocess

sys.path.insert(0, "/home/latorresn/hyperion")
os.chdir("/home/latorresn/hyperion-kernels")

from orchestrator.gpu_shim import cuda_lib_dirs

METRICS = [
    "dram__bytes.sum",
    "sm__sass_thread_inst_executed_op_dfma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_dadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_dmul_pred_on.sum",
    "sm__inst_executed_pipe_tensor.sum",
    "sm__inst_executed_pipe_tensor_op_dmma.sum",
    "sm__pipe_tensor_op_dmma_cycles_active.sum",
    "sm__pipe_shared_cycles_active.sum",
]

env = dict(os.environ)
lib_dirs = cuda_lib_dirs()
if lib_dirs:
    env["LD_LIBRARY_PATH"] = ":".join(str(d) for d in lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")

cmd = [
    "ncu", "--metrics", ",".join(METRICS), "--launch-count", "5", "--csv",
    "bin/cublas_dgemm_bench", "--size", "4096", "--iterations", "10",
]
print("CMD:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
print("RETURNCODE:", result.returncode)
if result.returncode != 0:
    print("STDERR:", result.stderr[-4000:])
    sys.exit(1)

out_path = "/home/latorresn/yacacerest/ncu_dgemm_n4096_tensorcore_probe.csv"
with open(out_path, "w") as f:
    f.write(result.stdout)
print(f"CSV crudo guardado en {out_path}")

lines = result.stdout.splitlines()
header_idx = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
if header_idx is None:
    print("No se encontro encabezado CSV en la salida de ncu")
    sys.exit(1)
reader = csv.DictReader(lines[header_idx:])
rows = list(reader)

totals = {}
for r in rows:
    name = r["Metric Name"]
    v = r["Metric Value"].replace(",", "").strip()
    val = float(v) if v not in ("", "N/A") else 0.0
    totals[name] = totals.get(name, 0.0) + val

print("\n--- Totales agregados (5 lanzamientos) ---")
for k, v in sorted(totals.items()):
    print(f"{k}: {v}")

dfma = totals.get("sm__sass_thread_inst_executed_op_dfma_pred_on.sum", 0.0)
dadd = totals.get("sm__sass_thread_inst_executed_op_dadd_pred_on.sum", 0.0)
dmul = totals.get("sm__sass_thread_inst_executed_op_dmul_pred_on.sum", 0.0)
dram_bytes = totals.get("dram__bytes.sum", 0.0)
tensor_inst = totals.get("sm__inst_executed_pipe_tensor.sum", 0.0)
dmma_inst = totals.get("sm__inst_executed_pipe_tensor_op_dmma.sum", 0.0)

classic_flops = 2 * dfma + dadd + dmul
print(f"\nFLOPs clasicos (2*dfma+dadd+dmul): {classic_flops}")
print(f"dram_bytes: {dram_bytes}")
if dram_bytes > 0:
    print(f"OI clasica (dfma/dadd/dmul): {classic_flops / dram_bytes} FLOP/byte")
print(f"Instrucciones tensor pipe (total): {tensor_inst}")
print(f"Instrucciones tensor pipe DMMA: {dmma_inst}")
print(f"USA TENSOR CORE: {'SI' if tensor_inst > 0 else 'NO'}")
