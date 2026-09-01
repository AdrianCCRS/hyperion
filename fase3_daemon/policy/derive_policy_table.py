#!/usr/bin/env python3
"""Derivador de la tabla de política clase->frecuencia (§3.4/§3.5 del plan
de realineación). Script offline: corre UNA VEZ por campaña de barrido
cerrada de Fase 1, nunca en producción -- el daemon de fase3_daemon/ carga
la tabla YAML que este script produce, nunca recalcula EDP en línea.

No reimplementa el cálculo de EDP desde cero: envuelve
fase2_clasificador/eval/protocol.py (edp_loss, trivial_baselines,
honest_constant_baseline ya existían y corrían sobre datos reales antes de
esta reconstrucción) y common/stats.py (prueba de significancia pareada,
nueva en esta reconstrucción -- no existía ningún uso de scipy.stats en
ninguna rama de origen).

Entrada: windows.csv de una campaña de Fase 1 ya cerrada (barrido completo
kernel x nivel_frecuencia x repetición), con las columnas que
fase1_telemetria/postprocess.py garantiza (REQUIRED_OUTPUT_COLUMNS).
Salida: policy_table.yaml con 4 entradas (cpu-compute_bound,
cpu-memory_bound, gpu-compute_bound, gpu-memory_bound).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.edp import compute_window_edp, load_windows, median_observed_frequency  # noqa: E402
from common.stats import paired_significance_test  # noqa: E402

REF_LEVEL = "REF"
DEVICES = ("cpu", "gpu")
LABELS = ("compute_bound", "memory_bound")


def filter_gpu_transition_not_settled(
    df: pd.DataFrame, t_transicion_gpu_ns: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """§3.5 paso 2: para GPU, excluye filas cuya corrida completa (mismo
    run_id) dura menos que t_transicion_gpu_ns -- su reloj puede no haber
    convergido al nivel solicitado, y mezclarlas sin distinción contamina
    el EDP agregado sin que el resultado lo señale.

    Devuelve (df_utilizable, df_excluido). Si t_transicion_gpu_ns es None
    (no medido todavía, ver §2.4.1 -- estado real del proyecto hoy: no
    existe esa medición en ningún lado), NINGUNA fila GPU se excluye por
    este criterio, pero el llamador debe tratar la política GPU resultante
    como no verificada -- ver derive_policy().
    """
    if t_transicion_gpu_ns is None:
        return df, df.iloc[0:0]
    is_gpu = df["device"] == "gpu"
    run_duration_ns = df.groupby("run_id")["delta_t_ns"].transform("sum")
    not_settled = is_gpu & (run_duration_ns < t_transicion_gpu_ns)
    return df[~not_settled].copy(), df[not_settled].copy()


def median_edp_by_kernel(df: pd.DataFrame, device: str, label: str, level: str) -> pd.Series:
    """Mediana de EDP por kernel_ref, para (device, phase_label_train,
    freq_level_id) -- unidad robusta a ruido de una corrida individual
    (§3.5 paso 3), y la unidad correcta para la prueba pareada del paso 5
    (un par por kernel entre REF y el nivel candidato).
    """
    level_col = "gpu_freq_level_id" if device == "gpu" else "freq_level_id"
    subset = df[(df["device"] == device) & (df["phase_label_train"] == label) & (df[level_col] == level)].copy()
    subset["edp"] = compute_window_edp(subset)
    subset = subset.dropna(subset=["edp"])
    return subset.groupby("kernel_ref")["edp"].median()


def derive_policy_for_class(
    df: pd.DataFrame, device: str, label: str, alpha: float,
) -> dict:
    """Deriva la entrada de política para un (device, phase_label_train):
    el nivel que minimiza el EDP agregado mediano, solo si la mejora sobre
    REF es estadísticamente defendible (§3.5 paso 5). Si ningún nivel
    mejora sobre REF, o la mejora no es significativa, la política es
    explícitamente "no actuar" -- un resultado válido, no un fallo.
    """
    level_col = "gpu_freq_level_id" if device == "gpu" else "freq_level_id"
    levels = sorted(
        lvl for lvl in df.loc[df["device"] == device, level_col].dropna().unique()
        if lvl != REF_LEVEL
    )
    ref_by_kernel = median_edp_by_kernel(df, device, label, REF_LEVEL)
    if ref_by_kernel.empty:
        return {
            "action": "no_actuar",
            "reason": "sin_datos_ref",
            "chosen_level": None,
        }

    best_level, best_relative_gain, best_test = None, 0.0, None
    for level in levels:
        candidate_by_kernel = median_edp_by_kernel(df, device, label, level)
        common_kernels = sorted(set(ref_by_kernel.index) & set(candidate_by_kernel.index))
        if len(common_kernels) < 2:
            continue
        ref_values = ref_by_kernel.loc[common_kernels].to_numpy()
        candidate_values = candidate_by_kernel.loc[common_kernels].to_numpy()
        relative_gain = 1.0 - float(candidate_values.sum() / ref_values.sum())
        if relative_gain <= 0:
            continue  # el candidato no mejora el EDP agregado sobre REF
        test = paired_significance_test(ref_values, candidate_values, alpha=alpha)
        if test.significant and relative_gain > best_relative_gain:
            best_level, best_relative_gain, best_test = level, relative_gain, test

    if best_level is None:
        return {
            "action": "no_actuar",
            "reason": "ningun_nivel_mejora_edp_de_forma_significativa",
            "chosen_level": None,
            "n_kernels_ref": int(len(ref_by_kernel)),
        }
    return {
        "action": "actuar",
        "chosen_level": best_level,
        # Frecuencia/reloj REAL observado en las filas de windows.csv de
        # este nivel (mediana, no el valor solicitado) -- así la tabla de
        # política es autocontenida: el daemon no necesita volver a
        # resolver "F0" contra la fracción de un manifiesto de campaña que
        # podría no tener a mano en producción. Ver fase3_daemon/gpu_loop/
        # loop.py::build_controller_from_policy().
        "resolved_freq_khz" if device == "cpu" else "resolved_clock_mhz": (
            median_observed_frequency(df, device, label, best_level)
        ),
        "edp_relative_gain": round(best_relative_gain, 4),
        "significance_test": best_test.test_name,
        "p_value": round(best_test.p_value, 6),
        "n_pairs": best_test.n_pairs,
    }


def derive_policy_table(
    df: pd.DataFrame,
    t_transicion_gpu_ns: float | None,
    alpha: float = 0.05,
    campaign_id: str | None = None,
) -> dict:
    usable, excluded = filter_gpu_transition_not_settled(df, t_transicion_gpu_ns)
    n_excluded_gpu = int((excluded["device"] == "gpu").sum())

    table: dict[str, dict] = {}
    for device in DEVICES:
        for label in LABELS:
            key = f"{device}-{label}"
            entry = derive_policy_for_class(usable, device, label, alpha)
            if device == "gpu" and t_transicion_gpu_ns is None:
                # §2.4.1: sin T_transición_gpu medido, cualquier política GPU
                # que sugiera "actuar" no puede confiarse -- se sobreescribe
                # explícitamente a "no actuar", distinguiendo esta causa
                # (viabilidad temporal, no verificada) de un "no actuar" por
                # rango de potencia angosto, tal como exige el plan.
                entry = {
                    "action": "no_actuar",
                    "reason": "t_transicion_gpu_no_medido",
                    "chosen_level": None,
                }
            elif device == "gpu" and entry.get("action") == "no_actuar" and entry.get("reason") == "ningun_nivel_mejora_edp_de_forma_significativa":
                # Conteo de exclusiones POR CLASE, no el total global de GPU
                # (bug corregido: antes usaba n_excluded_gpu, la cuenta de
                # las dos clases juntas -- podía atribuir "pocas fases
                # sobrevivientes por filtro de transición" a una clase cuyas
                # ventanas el filtro nunca tocó, si la OTRA clase sí tuvo
                # exclusiones).
                n_excluded_this_class = int(
                    ((excluded["device"] == "gpu") & (excluded["phase_label_train"] == label)).sum()
                )
                n_survivors = entry.get("n_kernels_ref", 0)
                if n_excluded_this_class > 0 and n_survivors < 2:
                    entry["reason"] = "pocas_fases_sobrevivientes_tras_filtro_transicion"
            table[key] = entry

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign_id,
        "alpha": alpha,
        "t_transicion_gpu_ns": t_transicion_gpu_ns,
        "n_windows_excluded_gpu_transition_not_settled": n_excluded_gpu,
        "policy": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("windows_csv", nargs="+", type=Path,
                         help="Uno o más windows.csv de una campaña de barrido cerrada (§2.4).")
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--alpha", type=float, default=0.05,
                         help="Nivel de significancia para la prueba pareada (default 0.05).")
    parser.add_argument(
        "--t-transicion-gpu-ns", type=float, default=None,
        help="Latencia de conmutación de reloj de GPU medida (§2.4.1). Sin "
             "medirla, la política GPU siempre queda en 'no actuar' -- el "
             "estado real del proyecto hoy, no un default arbitrario.",
    )
    parser.add_argument("--output", type=Path, required=True,
                         help="Ruta del policy_table.yaml a escribir.")
    args = parser.parse_args()

    df = load_windows(args.windows_csv)
    result = derive_policy_table(
        df, t_transicion_gpu_ns=args.t_transicion_gpu_ns,
        alpha=args.alpha, campaign_id=args.campaign_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    print(f"Tabla de política escrita en {args.output}")
    for key, entry in result["policy"].items():
        print(f"  {key}: {entry['action']}" + (f" -> {entry['chosen_level']}" if entry["chosen_level"] else ""))


if __name__ == "__main__":
    main()
