"""Compuerta de arranque/no-arranque sobre la distribucion de la etiqueta.

Un `is_optimal` casi constante (una accion gana casi siempre) no tiene
estructura que un modelo pueda aprender por encima de la constante misma:
comparar familias sobre esa distribucion mide ruido de medicion, no
capacidad predictiva. Esta compuerta se calcula antes de reportar un
`model_comparison.csv` como comparacion valida.

Umbrales (documentados, no ajustados por dataset):
- TOP1_SHARE_MAX: si una sola accion gana en mas de esta fraccion de los
  grupos de decision, la etiqueta no tiene variedad suficiente.
- MIN_EFFECTIVE_CLASSES: numero minimo de acciones que deben ganar al
  menos una vez con una fraccion no despreciable (>= MIN_CLASS_SHARE).
- MEDIAN_MARGIN_FLOOR_PCT: si el margen mediano entre el mejor y el
  segundo candidato cae por debajo de esto, no se distingue de jitter de
  medicion (ver ARC-172: dispersion tipica de la campana CPU). El valor
  actual (2.0) es el piso original, conservador. Una calibracion de ruido
  posterior (sesion aparte) midio ~3.11% de CV mediano entre repeticiones
  del mismo config/accion -- estrictamente mas alto que 2.0 -- pero se deja
  el piso en 2.0 aqui (en vez de subirlo a 3.11) porque CV entre repeticiones
  no es lo mismo que el margen porcentual entre dos acciones distintas: subir
  el piso al valor de CV asumiria que *toda* la dispersion de repeticion se
  traslada 1:1 al margen entre acciones, lo cual es conservador de mas y no
  esta validado aqui. Queda documentado para que una futura revision lo
  suba con evidencia, no por defecto.

Descomposicion device vs frequency (2026-08-30):
El margen mediano original (`median_margin_edp_pct`, calculado sobre TODAS
las acciones candidatas de un grupo, CPU y GPU mezcladas) confunde dos
decisiones de naturaleza distinta:

1. `device_decision`: CPU vs GPU (mejor accion de cada dispositivo, sin
   importar que frecuencia exacta gane dentro de cada uno). Esta decision
   es la que mas energia ahorra y, con evidencia real, casi nunca es
   ruidosa (margenes de ordenes de magnitud tipicos).
2. `frequency_decision`: dentro del dispositivo que resulto ganador en el
   grupo, la mejor accion contra la segunda mejor DEL MISMO dispositivo.
   Aqui si aparece el margen angosto que cae bajo el piso de ruido.

Los campos legacy de nivel superior (`verdict`, `reasons`, `top1_share`,
`effective_classes`, `median_margin_edp_pct`, `action_share`, `thresholds`)
se preservan exactamente con su calculo original (margen mezclado) para no
romper a `search.py`/`eda.py`, que ya los consumen. `device_decision` y
`frequency_decision` son campos nuevos, aditivos.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

TOP1_SHARE_MAX = 0.90
MIN_EFFECTIVE_CLASSES = 3
MIN_CLASS_SHARE = 0.05
MEDIAN_MARGIN_FLOOR_PCT = 2.0
# Mismo piso para el margen de dispositivo: tambien es un margen de EDP
# entre dos acciones (la mejor de cada device), asi que el mismo ruido de
# medicion aplica. En la practica no deberia ser el limitante (ver docstring).
DEVICE_MARGIN_FLOOR_PCT = 2.0


def _device_of(action_id: str) -> str:
    return str(action_id).split(":", 1)[0]


def _margin_pct(best_edp: float, second_edp: float) -> float | None:
    if best_edp is None or second_edp is None:
        return None
    if best_edp <= 0:
        return None
    return float((second_edp - best_edp) / best_edp * 100.0)


def _assess_device_decision(strategy_frame: pd.DataFrame) -> dict[str, Any]:
    """Margen entre la mejor accion CPU y la mejor accion GPU, por grupo."""
    if "edp_mean" not in strategy_frame or "decision_group_id" not in strategy_frame:
        return {
            "verdict": "pipeline_smoke_only",
            "reasons": ["faltan columnas edp_mean/decision_group_id"],
            "n_groups_evaluated": 0,
            "excluded_groups": 0,
            "median_margin_pct": None,
        }

    frame = strategy_frame.copy()
    if "candidate_device" in frame:
        frame["_device"] = frame["candidate_device"].astype(str)
    else:
        frame["_device"] = frame["action_id"].map(_device_of)

    margins = []
    excluded = 0
    for _, group in frame.groupby("decision_group_id", observed=True):
        devices_present = group["_device"].unique()
        if not any(d == "cpu" for d in devices_present) or not any(d == "gpu" for d in devices_present):
            excluded += 1
            continue
        cpu_best = group.loc[group["_device"] == "cpu", "edp_mean"].min()
        gpu_best = group.loc[group["_device"] == "gpu", "edp_mean"].min()
        if pd.isna(cpu_best) or pd.isna(gpu_best):
            excluded += 1
            continue
        best_edp, second_edp = (cpu_best, gpu_best) if cpu_best <= gpu_best else (gpu_best, cpu_best)
        margin = _margin_pct(best_edp, second_edp)
        if margin is not None:
            margins.append(margin)

    median_margin = float(pd.Series(margins).median()) if margins else None
    reasons = []
    if median_margin is None:
        reasons.append("sin grupos con ambos dispositivos presentes para comparar")
    elif median_margin < DEVICE_MARGIN_FLOOR_PCT:
        reasons.append(f"margen mediano de dispositivo {median_margin:.2f}% < piso de ruido {DEVICE_MARGIN_FLOOR_PCT}%")

    return {
        "verdict": "pipeline_smoke_only" if reasons else "comparison_valid",
        "reasons": reasons,
        "median_margin_pct": median_margin,
        "n_groups_evaluated": len(margins),
        "device_margin_excluded_groups": excluded,
        "floor_pct": DEVICE_MARGIN_FLOOR_PCT,
    }


def _assess_frequency_decision(strategy_frame: pd.DataFrame) -> dict[str, Any]:
    """Margen entre la mejor y segunda mejor accion DENTRO del device ganador."""
    required = {"decision_group_id", "action_id", "is_optimal", "edp_mean"}
    if not required.issubset(strategy_frame.columns):
        return {
            "verdict": "pipeline_smoke_only",
            "reasons": [f"faltan columnas {sorted(required - set(strategy_frame.columns))}"],
            "n_groups_evaluated": 0,
            "excluded_groups": 0,
            "median_margin_pct": None,
        }

    frame = strategy_frame.copy()
    if "candidate_device" in frame:
        frame["_device"] = frame["candidate_device"].astype(str)
    else:
        frame["_device"] = frame["action_id"].map(_device_of)

    margins = []
    excluded = 0
    for _, group in frame.groupby("decision_group_id", observed=True):
        optimal_rows = group[group["is_optimal"] == 1]
        if optimal_rows.empty:
            excluded += 1
            continue
        winner_device = str(optimal_rows.iloc[0]["_device"])
        same_device = group[group["_device"] == winner_device].sort_values("edp_mean")
        if len(same_device) < 2:
            excluded += 1
            continue
        best_edp = float(same_device.iloc[0]["edp_mean"])
        second_edp = float(same_device.iloc[1]["edp_mean"])
        margin = _margin_pct(best_edp, second_edp)
        if margin is not None:
            margins.append(margin)

    median_margin = float(pd.Series(margins).median()) if margins else None
    reasons = []
    if median_margin is None:
        reasons.append("sin grupos con >=2 acciones del device ganador para comparar")
    elif median_margin < MEDIAN_MARGIN_FLOOR_PCT:
        reasons.append(f"margen mediano de frecuencia {median_margin:.2f}% < piso de ruido {MEDIAN_MARGIN_FLOOR_PCT}%")

    return {
        "verdict": "pipeline_smoke_only" if reasons else "comparison_valid",
        "reasons": reasons,
        "median_margin_pct": median_margin,
        "n_groups_evaluated": len(margins),
        "frequency_margin_excluded_groups": excluded,
        "floor_pct": MEDIAN_MARGIN_FLOOR_PCT,
    }


def assess_label_health(strategy_frame: pd.DataFrame) -> dict[str, Any]:
    """Evalua si `is_optimal` en `strategy_frame` tiene estructura aprendible.

    Espera las columnas `decision_group_id`, `action_id`, `is_optimal` y,
    si estan presentes, `margin_edp_pct` (una fila por candidato, con
    exactamente un `is_optimal=1` por `decision_group_id`). Para la
    descomposicion device/frequency tambien usa `edp_mean` y, si esta
    presente, `candidate_device` (si no, deriva el device del prefijo de
    `action_id` antes de `:`).
    """
    optima = strategy_frame[strategy_frame["is_optimal"] == 1]
    n_groups = optima["decision_group_id"].nunique()
    if n_groups == 0:
        return {
            "verdict": "pipeline_smoke_only",
            "reason": "sin grupos de decision",
            "n_groups": 0,
            "device_decision": {"verdict": "pipeline_smoke_only", "reasons": ["sin grupos de decision"], "median_margin_pct": None},
            "frequency_decision": {"verdict": "pipeline_smoke_only", "reasons": ["sin grupos de decision"], "median_margin_pct": None},
        }

    action_share = (
        optima.groupby("action_id", observed=True)["decision_group_id"].nunique() / n_groups
    ).sort_values(ascending=False)
    top1_share = float(action_share.iloc[0])
    effective_classes = int((action_share >= MIN_CLASS_SHARE).sum())

    median_margin = None
    if "margin_edp_pct" in optima:
        margins = pd.to_numeric(optima["margin_edp_pct"], errors="coerce").dropna()
        if len(margins):
            median_margin = float(margins.median())

    reasons = []
    if top1_share > TOP1_SHARE_MAX:
        reasons.append(f"accion dominante {action_share.index[0]} gana {top1_share:.1%} > {TOP1_SHARE_MAX:.0%}")
    if effective_classes < MIN_EFFECTIVE_CLASSES:
        reasons.append(f"solo {effective_classes} acciones con masa >= {MIN_CLASS_SHARE:.0%}, se exigen {MIN_EFFECTIVE_CLASSES}")
    if median_margin is not None and median_margin < MEDIAN_MARGIN_FLOOR_PCT:
        reasons.append(f"margen mediano {median_margin:.2f}% < piso de ruido {MEDIAN_MARGIN_FLOOR_PCT}%")

    verdict = "pipeline_smoke_only" if reasons else "comparison_valid"

    device_decision = _assess_device_decision(strategy_frame)
    frequency_decision = _assess_frequency_decision(strategy_frame)

    return {
        # --- campos legacy, calculo sin cambios (margen mezclado CPU+GPU) ---
        "verdict": verdict,
        "reasons": reasons,
        "n_groups": n_groups,
        "top1_action": str(action_share.index[0]),
        "top1_share": top1_share,
        "effective_classes": effective_classes,
        "median_margin_edp_pct": median_margin,
        "action_share": action_share.to_dict(),
        "thresholds": {
            "top1_share_max": TOP1_SHARE_MAX,
            "min_effective_classes": MIN_EFFECTIVE_CLASSES,
            "min_class_share": MIN_CLASS_SHARE,
            "median_margin_floor_pct": MEDIAN_MARGIN_FLOOR_PCT,
        },
        # --- descomposicion nueva, aditiva ---
        "device_decision": device_decision,
        "frequency_decision": frequency_decision,
    }
