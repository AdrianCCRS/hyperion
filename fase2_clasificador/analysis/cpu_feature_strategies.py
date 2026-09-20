"""Features CPU candidatas para ablaciones sin fuga Roofline.

El objetivo CPU se deriva offline de FLOPs, bytes DRAM uncore y el ridge. Este
módulo solo prepara variantes calculables durante una observación normal del
lector PMU: instrucciones, ciclos, caché, stalls, tiempo habilitado y reloj.
No entrena ni decide cuál variante gana; esa decisión requiere validación
externa por familia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


LEGACY_FEATURES = [
    "ipc", "mpki", "cache_miss_rate", "stall_mem_ratio", "ips",
    "running_ratio", "freq_khz_observed",
]
DEPLOYABLE_BASE_FEATURES = [feature for feature in LEGACY_FEATURES if feature != "running_ratio"]

# Insumos de la verdad Roofline, identificadores de workload y cantidades de
# adquisición que podrían actuar como huella de la corrida. Ninguno puede ser
# usado por una variante de producción.
FORBIDDEN_FEATURES = frozenset({
    "kernel_ref", "freq_level_id", "run_id", "repetition", "node_id",
    "phase_label_train", "operational_intensity_uncore_real", "i_ridge_used",
    "flops_measured_interval", "uncore_cas_count_read_interval",
    "uncore_cas_count_write_interval", "bytes_moved_uncore_real",
    "uncore_interval_id", "uncore_t_start_ns", "uncore_t_end_ns",
    "uncore_delta_t_ns", "cpu_window_count",
    # Calidad del contador / multiplexación: se usa exclusivamente para
    # aceptar o rechazar filas, no describe la carga y no puede llegar al
    # modelo.
    "delta_running_ns", "delta_enabled_ns", "running_ratio",
})


def _ratio(numerator: pd.Series, denominator: pd.Series, scale: float = 1.0) -> pd.Series:
    """Cociente finito; el cero no se inventa cuando el denominador es cero."""
    n = pd.to_numeric(numerator, errors="coerce")
    d = pd.to_numeric(denominator, errors="coerce")
    return (scale * n / d.where(d > 0)).replace([np.inf, -np.inf], np.nan)


def add_online_physical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Añade transformaciones PMU que no usan la verdad Roofline.

    Las variables base ya son razones. Las nuevas expresan interacciones que
    una regresión lineal no puede recuperar por sí misma (p. ej. referencias
    de caché por ciclo); los árboles se evalúan igualmente como ablación, no
    bajo el supuesto de que una transformación deba ayudarles.
    """
    required = {
        "delta_instructions", "delta_cycles", "delta_cache_references",
        "delta_cache_misses", "delta_stalled_cycles_mem_any",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"faltan contadores PMU para las variantes CPU: {sorted(missing)}")

    out = frame.copy()
    instructions = out["delta_instructions"]
    cycles = out["delta_cycles"]
    references = out["delta_cache_references"]
    misses = out["delta_cache_misses"]
    stalls = out["delta_stalled_cycles_mem_any"]
    out["cache_references_per_ki"] = _ratio(references, instructions, 1000.0)
    out["cache_references_per_cycle"] = _ratio(references, cycles)
    out["cache_misses_per_cycle"] = _ratio(misses, cycles)
    out["stalls_mem_per_ki"] = _ratio(stalls, instructions, 1000.0)
    out["stalls_mem_per_cache_miss"] = _ratio(stalls, misses)
    # Transformación monotónica, robusta frente a la cola larga de MPKI; no
    # altera ni usa la etiqueta.
    out["log1p_mpki"] = np.log1p(pd.to_numeric(out["mpki"], errors="coerce").clip(lower=0.0))
    return out


def feature_variants() -> dict[str, list[str]]:
    """Variantes candidatas, todas disponibles en producción.

    No se incluye identidad de kernel ni duración/índice de intervalo: podrían
    mejorar un split aleatorio memorizando la adquisición sin ayudar sobre una
    familia nueva.
    """
    counter_interactions = [
        "cache_references_per_ki", "cache_references_per_cycle",
        "cache_misses_per_cycle", "stalls_mem_per_ki",
        "stalls_mem_per_cache_miss", "log1p_mpki",
    ]
    return {
        "baseline_pmu": DEPLOYABLE_BASE_FEATURES,
        "pmu_interactions": DEPLOYABLE_BASE_FEATURES + counter_interactions,
        "without_frequency": [f for f in DEPLOYABLE_BASE_FEATURES if f != "freq_khz_observed"] + counter_interactions,
        "cache_stall_focus": [
            "ipc", "cache_miss_rate", "stall_mem_ratio", "freq_khz_observed",
            "cache_references_per_ki", "cache_references_per_cycle",
            "cache_misses_per_cycle", "stalls_mem_per_ki", "stalls_mem_per_cache_miss",
        ],
    }


def production_variants() -> dict[str, list[str]]:
    """Devuelve variantes válidas y comprueba el guardarraíl de fuga."""
    variants = feature_variants()
    for name, features in variants.items():
        leak = set(features) & FORBIDDEN_FEATURES
        if leak:
            raise AssertionError(f"{name} contiene fuga Roofline/identidad: {sorted(leak)}")
    return variants
