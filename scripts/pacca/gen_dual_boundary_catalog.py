#!/usr/bin/env python3
"""Emite las entradas de catalogo para la campana dirigida a fronteras.

No es una ampliacion arbitraria del conjunto de entrenamiento. Los nueve
tamanos nuevos se eligieron antes de medirlos para cerrar los tres cruces
CPU/GPU observados en la campana exploratoria; FFT se vuelve a medir en sus
cuatro tamanos existentes porque el resultado no monotono es precisamente la
hipotesis que se quiere contrastar.

Las iteraciones CPU se obtienen por interpolacion lineal entre los tiempos
por despacho *medidos* de los dos tamanos vecinos de la rejilla original.
No se extrapola ni se inventa una constante de rendimiento. Antes de lanzar,
el manifiesto y el catalogo siguen pasando el preflight normal; una corrida
que no alcance la duracion/resolucion requerida se conserva y se rechaza por
la validacion, no se corrige retrospectivamente.
"""
from __future__ import annotations

import math

from gen_dual_full_catalog import (  # reutiliza checksums y contratos vigentes
    GPU_TWO_TERM,
    MIN_ITERATIONS,
    MAX_ITERATIONS,
    OP_META,
    SCALING_FN,
    TARGET_SECONDS,
    estimated_memory_bytes,
)


# Cada tripleta encierra un cruce estimado previamente con datos exploratorios:
# AXPY ~=208114; SpMV ~=2081139; stencil ~=3584.
BOUNDARY_SIZES = {
    "axpy": (160_000, 208_000, 250_000),
    "spmv": (1_600_000, 2_080_000, 2_500_000),
    "stencil": (3328, 3584, 3840),
}

# Tiempos CPU REF por iteracion medidos en la rejilla original, usados solo
# como vecinos para interpolar. Se mantienen locales para que el artefacto
# sea auditable aun si cambia el generador grande en el futuro.
CPU_NEIGHBORS = {
    "axpy": ((100_000, 7.03062952132e-05), (316_228, 2.11486618005e-04)),
    "spmv": ((1_000_000, 2.10222289157e-03), (3_162_278, 6.82941032609e-03)),
    "stencil": ((3072, 9.51960795455e-03), (4096, 1.69117100000e-02)),
}


def cpu_iterations(op: str, size: int) -> int:
    (lo_n, lo_t), (hi_n, hi_t) = CPU_NEIGHBORS[op]
    if not lo_n < size < hi_n:
        raise ValueError(f"{op} N={size} no queda entre sus vecinos medidos")
    fraction = (size - lo_n) / (hi_n - lo_n)
    seconds_per_iteration = lo_t + fraction * (hi_t - lo_t)
    return max(MIN_ITERATIONS, min(MAX_ITERATIONS, math.ceil(TARGET_SECONDS / seconds_per_iteration)))


def gpu_iterations(op: str, size: int) -> int:
    fixed, variable = GPU_TWO_TERM[op]
    seconds_per_iteration = fixed + variable * SCALING_FN[op](size)
    return max(MIN_ITERATIONS, min(MAX_ITERATIONS, int(round(TARGET_SECONDS / seconds_per_iteration))))


def emit_entry(op: str, size: int, device: str) -> str:
    meta = OP_META[op]
    iterations = cpu_iterations(op, size) if device == "cpu" else gpu_iterations(op, size)
    checksum = meta[f"sha_{device}"]
    wrapper = meta[f"wrap_{device}"]
    lines = [
        f"  - id: dual_{op}_{device}_N{size}",
        f"    suite: {meta['suite']}",
        "    role: dataset",
    ]
    if device == "gpu":
        lines.extend(("    device: gpu", "    gpu_precision: fp64"))
    lines.extend((
        f"    exec_path: bin/{wrapper}",
        f"    config_id: {op}_N{size}",
        "    phase_label_hint: intermedio",
        f"    size_variant: N{size}",
        f'    exec_args: "--size {size} --iterations {iterations}"',
        "    expected_runtime_seconds: 11",
        "    warmup_seconds: 0.05",
        f"    estimated_memory_bytes: {estimated_memory_bytes(op, size)}",
        '    success_check: {type: stdout_regex, pattern: "Verification\\\\s*=\\\\s*SUCCESSFUL"}',
        '    flops_rate_stdout_pattern: "Mop/s total\\\\s*=\\\\s*([0-9.]+)"',
        '    runtime_seconds_stdout_pattern: "Time in seconds\\\\s*=\\\\s*([0-9.]+)"',
    ))
    if device == "gpu":
        lines.append(
            f"    operational_intensity_flops_per_byte: {meta['oi']}  # heredada de la caracterizacion por operacion; no usar para afirmar Roofline por tamano"
        )
    lines.extend(("    binary_checksum:", f'      pacca-a100: "sha256:{checksum}"', ""))
    return "\n".join(lines)


def generate() -> str:
    return "\n".join(
        emit_entry(op, size, device)
        for op, sizes in BOUNDARY_SIZES.items()
        for size in sizes
        for device in ("cpu", "gpu")
    )


if __name__ == "__main__":
    print(generate())
