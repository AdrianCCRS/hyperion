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
  medicion (ver ARC-172: dispersion tipica de la campana CPU).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

TOP1_SHARE_MAX = 0.90
MIN_EFFECTIVE_CLASSES = 3
MIN_CLASS_SHARE = 0.05
MEDIAN_MARGIN_FLOOR_PCT = 2.0


def assess_label_health(strategy_frame: pd.DataFrame) -> dict[str, Any]:
    """Evalua si `is_optimal` en `strategy_frame` tiene estructura aprendible.

    Espera las columnas `decision_group_id`, `action_id`, `is_optimal` y,
    si estan presentes, `margin_edp_pct` (una fila por candidato, con
    exactamente un `is_optimal=1` por `decision_group_id`).
    """
    optima = strategy_frame[strategy_frame["is_optimal"] == 1]
    n_groups = optima["decision_group_id"].nunique()
    if n_groups == 0:
        return {"verdict": "pipeline_smoke_only", "reason": "sin grupos de decision", "n_groups": 0}

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
    return {
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
    }
