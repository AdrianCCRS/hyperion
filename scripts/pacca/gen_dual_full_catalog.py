"""Genera las entradas de catalogo.yaml para la rejilla COMPLETA del
selector CPU/GPU (6 operaciones x rejilla de tamanos x 2 devices) y los dos
manifiestos de campana real (CPU-solo y GPU).

POR QUE DOS CAMPANAS, NO UNA. El orquestador aplica una sola lista
`frequency_levels` (CPU) por manifiesto a TODOS los kernels que contiene
(campaign.py: frequency_levels es del manifest, no por kernel). Queremos
8 niveles finos de CPU para los kernels CPU-solo, pero solo unos pocos
niveles de CPU para los kernels GPU (el smoke de 2026-08-27, job 6668,
midio que el reloj de CPU SI afecta el despacho GPU -- hasta 95% mas lento
en F6 -- pero con una forma casi plana en REF/F0 y creciente hacia F6, asi
que no hace falta la misma resolucion). Una sola lista no puede servir a
ambos, asi que se separan en dos manifiestos, exactamente como ya hace el
proyecto con campaign_pacca_cpu_*.yaml vs campaign_pacca_gpu_*.yaml.

ITERACIONES POR TAMANO. warmup_seconds se queda FIJO (0.05s, corrige el bug
de 2026-08-27 job 6668/6657: excluia TODAS las ventanas de una corrida mas
corta que el warmup declarado) -- lo que varia con el tamano es
`--iterations`, calibrado para que el tiempo total en el nivel de
frecuencia MAS LENTO quede muy por encima del warmup. La formula usa la
complejidad asintotica conocida de cada operacion y un punto de referencia
medido de verdad (calibrate_iterations.sh, 2026-08-27), no un numero
supuesto.
"""
from __future__ import annotations

import math

TARGET_SECONDS = 1.5  # tiempo objetivo en el nivel de frecuencia MAS RAPIDO
MIN_ITERATIONS = 5
MAX_ITERATIONS = 2_000_000

# Rejilla "matriz/malla" (GEMM, FFT, Stencil, Cholesky): O(N^2)-O(N^3),
# razon ~1.5x, 13 tamanos, 64..4096. Densa cerca de 256-512 porque ahi cae
# la frontera CPU/GPU medida en el tamizaje (screen_dual_frontier.sh).
GRID_MATRIX = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]

# Rejilla "vector" (AXPY, SpMV): O(N), razon ~sqrt(10), 8 tamanos, 1e4..3.16e7.
# Techo recortado de 1e8 a 3.16e7 -- a 1e8 la memoria REAL de SpMV (CSR con
# 7 no-ceros/fila: row_ptr+col_idx+values+x+y ~= 104*N bytes) da ~10.4 GB
# solo en host, sin contar el lado GPU (VRAM separada, cabria en los 40 GB
# de la A100, pero el host es el limite real con --mem=8-16G ya usado en
# campañas previas).
GRID_VECTOR = [10_000, 31_623, 100_000, 316_228, 1_000_000, 3_162_278,
               10_000_000, 31_622_777]

# Punto de referencia medido de verdad para CPU (calibrate_iterations.sh,
# 2026-08-27, paccaA100). t_per_it en segundos, en el N de referencia. Un
# solo punto (modelo lineal por el origen) es suficiente para CPU. CPU y GPU
# usan conteos de repeticion distintos: el observable comparable es tiempo y
# energia POR DESPACHO, no la duracion total de bucles artificialmente
# igualados. Asi cada lado conserva ~1.5 s de region warm y suficientes
# ventanas de telemetria, incluso cuando N es pequeno.
REFERENCE = {
    # op: (N_ref, t_per_it_cpu, scaling_fn)
    "gemm": (512, 0.00098501, lambda n: n ** 3),
    "fft": (512, 0.00125038, lambda n: (n * n) * math.log2(n * n)),
    "axpy": (1_000_000, 0.00084114, lambda n: n),
    "stencil": (512, 0.00023750, lambda n: n * n),
    "cholesky": (512, 0.00261578, lambda n: n ** 3),
    "spmv": (1_000_000, 0.00258186, lambda n: n),
}

# Modelo de DOS terminos para GPU: t_por_iteracion(N) = A_fijo + B_variable*fn(N).
# Ajustado 2026-08-28 por minimos cuadrados sobre 486 corridas reales
# aceptadas del pase 1 GPU (job 6689, ver
# docs/general/metodologia_selector_cpu_gpu_20260827.md seccion 6.9), usando
# solo niveles gpuREF/gpuF0 (reloj de GPU rapido, para no mezclar el efecto
# de frecuencia con el de tamano) en los 13/8 tamanos de cada rejilla.
#
# POR QUE HACIA FALTA: el modelo lineal-por-el-origen anterior (un solo
# punto en N=512/1e6, igual que CPU) ignoraba el costo fijo de despacho GPU
# (H2D + lanzamiento de kernel + D2H, INDEPENDIENTE de N). A N chico ese
# termino fijo domina por completo -- el modelo viejo pedia ordenes de
# magnitud mas iteraciones de las que caben en el timeout (ej. gemm N=64:
# 779688 iteraciones pedidas, tiempo real de esa cantidad ~20 min, contra
# un timeout de ~90-132s -- el proceso moria sin imprimir ni su primera
# linea, exactamente el patron de las 26 fallas C03 del pase 1 real,
# concentradas en cholesky_gpu por ser el costo fijo mas caro despues de
# gemm). Con el termino fijo explicito, gemm N=64 pide 975 iteraciones en
# vez de 779688 -- una corrida real de 1.5s, no de 20 minutos.
GPU_TWO_TERM = {
    # op: (A_fijo_segundos_por_iteracion, B_variable_segundos_por_unidad_fn)
    "gemm": (1.5383e-3, 7.280e-13),
    "fft": (4.633e-4, 1.447e-10),
    "axpy": (1.629e-4, 2.506e-9),
    "stencil": (1.343e-4, 1.701e-9),
    "cholesky": (6.238e-4, 1.314e-12),
    "spmv": (1.301e-4, 1.780e-9),
}

# suite / gpu_precision / OI medida ncu (representativa por operacion, no
# por tamano -- ese campo solo alimenta la caracterizacion Roofline
# heredada, no el target del selector, ver discusion 2026-08-27).
OP_META = {
    "gemm": dict(suite="Dual-GEMM", oi=0.062215, grid=GRID_MATRIX,
                 wrap_cpu="gemm_cpu", wrap_gpu="gemm_gpu",
                 sha_cpu="caa45d661a38d665193647ff3a0849607d9996ee83f87c046011853bb8b4abf4",
                 sha_gpu="f9b8aa868984f739eea271d5b792a45aafd413df98669ef4b398956693de840b"),
    "fft": dict(suite="Dual-FFT", oi=2.498981, grid=GRID_MATRIX,
                wrap_cpu="fft_cpu", wrap_gpu="fft_gpu",
                sha_cpu="ff3037765fc54f8797dbe0060c0fc8487cc4eed56bdfaa100a3e76fc45f7fea3",
                sha_gpu="ac72a63f28829a73c5ee72364e1499d178ee7c7c2ef977343cd2a21d1f7677fc"),
    "axpy": dict(suite="Dual-AXPY", oi=0.124930, grid=GRID_VECTOR,
                 wrap_cpu="axpy_cpu", wrap_gpu="axpy_gpu",
                 sha_cpu="46c155f9cb971e88606c0dc932e922095e79eb702032f781ee64c4a9cbce0f64",
                 sha_gpu="1e6bb9e3014748d4221c7b96ca085f29a84bcd80027f655f14d56a03d6b2d906"),
    "stencil": dict(suite="Dual-Stencil", oi=0.495370, grid=GRID_MATRIX,
                    wrap_cpu="stencil_cpu", wrap_gpu="stencil_gpu",
                    sha_cpu="3583e61842a749016ebcf85e9479c2daf8516e7e60e722f6f9ab4d2b85a09a23",
                    sha_gpu="46fd0ca330952a17dd14a659d332da99369207a1bd58f7ea14accb08071369e5"),
    "cholesky": dict(suite="Dual-Cholesky", oi=9.887307, grid=GRID_MATRIX,
                     wrap_cpu="cholesky_cpu", wrap_gpu="cholesky_gpu",
                     sha_cpu="b3fdd715d9cac8206f7092829df82ed4ff4b714ab6f2138b13ad2eb9e7827d06",
                     sha_gpu="fe55859d36a0b1de656e949745365d14968a465c5b06cf0bb9849a41d3c7c480"),
    "spmv": dict(suite="Dual-SpMV", oi=0.274176, grid=GRID_VECTOR,
                 wrap_cpu="spmv_cpu", wrap_gpu="spmv_gpu",
                 sha_cpu="591b732e4ff0ff47fead923dff19651bd0b368ae43fba89b9e30ef9aff95aa37",
                 sha_gpu="ab2659d9a62bf47341262f4d2a1e4de0114026c948d246e337ee0a3f28a9097e"),
}


def iterations_for(op: str, n: int, device: str) -> int:
    n_ref, t_cpu_ref, fn = REFERENCE[op]
    if device == "cpu":
        t_per_it = (t_cpu_ref / fn(n_ref)) * fn(n)
    elif device == "gpu":
        a_gpu, b_gpu = GPU_TWO_TERM[op]
        t_per_it = a_gpu + b_gpu * fn(n)
    else:
        raise ValueError(device)
    raw = TARGET_SECONDS / t_per_it
    it = int(round(raw))
    return max(MIN_ITERATIONS, min(MAX_ITERATIONS, it))


def estimated_memory_bytes(op: str, n: int) -> int:
    # Contabilidad real por operacion (no un factor generico), con ~30% de
    # margen -- un valor inflado sin fundamento (ver el bug real: la
    # primera version de esta formula daba 58.8 GB para spmv en N=1e8,
    # 5.6x mas que el calculo real de abajo).
    margin = 1.3
    if op == "gemm":
        # A, B, C: 3 matrices NxN doubles.
        bytes_real = 3 * n * n * 8
    elif op == "cholesky":
        # original, work: 2 matrices NxN doubles (mas el workspace de
        # cuSOLVER en GPU, pequeño frente a esto).
        bytes_real = 2 * n * n * 8
    elif op == "fft":
        # data, original: 2 buffers NxN COMPLEJOS (16 B/elemento).
        bytes_real = 2 * n * n * 16
    elif op == "stencil":
        # a, b, original: 3 buffers NxN doubles.
        bytes_real = 3 * n * n * 8
    elif op == "axpy":
        # x, y, y_original: 3 vectores N doubles.
        bytes_real = 3 * n * 8
    elif op == "spmv":
        # row_ptr (N+1 int32) + col_idx (7N int32) + values (7N double)
        # + x,y (2N double), NNZ_PER_ROW=7 fijo.
        bytes_real = (n + 1) * 4 + 7 * n * 4 + 7 * n * 8 + 2 * n * 8
    else:
        raise ValueError(op)
    return int(bytes_real * margin)


def gen_entries() -> str:
    lines = []
    for op, meta in OP_META.items():
        for n in meta["grid"]:
            config_id = f"{op}_N{n}"
            mem = estimated_memory_bytes(op, n)
            for device in ("cpu", "gpu"):
                it = iterations_for(op, n, device)
                kid = f"dual_{op}_{device}_N{n}"
                wrap = meta[f"wrap_{device}"]
                sha = meta[f"sha_{device}"]
                lines.append(f"  - id: {kid}")
                lines.append(f"    suite: {meta['suite']}")
                lines.append("    role: dataset")
                if device == "gpu":
                    lines.append("    device: gpu")
                    lines.append("    gpu_precision: fp64")
                lines.append(f"    exec_path: bin/{wrap}")
                lines.append(f"    config_id: {config_id}")
                lines.append("    phase_label_hint: intermedio")
                lines.append(f"    size_variant: N{n}")
                lines.append(f'    exec_args: "--size {n} --iterations {it}"')
                lines.append(f"    expected_runtime_seconds: {max(6, int(TARGET_SECONDS * 6) + 2)}")
                lines.append("    warmup_seconds: 0.05")
                lines.append(f"    estimated_memory_bytes: {mem}")
                lines.append('    success_check: {type: stdout_regex, pattern: "Verification\\\\s*=\\\\s*SUCCESSFUL"}')
                lines.append('    flops_rate_stdout_pattern: "Mop/s total\\\\s*=\\\\s*([0-9.]+)"')
                lines.append('    runtime_seconds_stdout_pattern: "Time in seconds\\\\s*=\\\\s*([0-9.]+)"')
                if device == "gpu":
                    lines.append(f"    operational_intensity_flops_per_byte: {meta['oi']}  # ncu real, representativa de la operacion (no por tamano)")
                lines.append("    binary_checksum:")
                lines.append(f'      pacca-a100: "sha256:{sha}"')
                lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(gen_entries())
    total_configs = sum(len(m["grid"]) for m in OP_META.values())
    print(f"config_id totales: {total_configs}  (catalog entries: {total_configs * 2})", file=sys.stderr)
    for op, meta in OP_META.items():
        for device in ("cpu", "gpu"):
            its = [iterations_for(op, n, device) for n in meta["grid"]]
            print(
                f"{op:<10} {device} N={meta['grid'][0]}..{meta['grid'][-1]}  "
                f"iterations={min(its)}..{max(its)}",
                file=sys.stderr,
            )
