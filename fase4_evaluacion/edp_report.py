"""Reporte final de EDP por clase y dispositivo (§5.2 del plan de
realineación) -- nunca un único número agregado, exigencia explícita del
plan ("resultado reportado con matices... es más interesante que un
resultado uniformemente positivo sin matices").

Construido sobre common/edp.py (compute_window_edp, ya usado por
fase3_daemon/policy/derive_policy_table.py) y common/stats.py
(paired_significance_test, misma prueba pareada que usa el derivador de
política -- la definición de "mejora estadísticamente defendible" no debe
divergir entre la política que se despliega y la evaluación que la juzga).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common.edp import compute_window_edp, load_windows
from common.stats import SignificanceResult, paired_significance_test


@dataclass(frozen=True)
class ScenarioComparison:
    """Resultado de comparar UN escenario baseline contra el agente, para
    UNA clase (device, phase_label_train)."""
    device: str
    phase_label_train: str
    baseline_scenario: str
    n_kernels_compared: int
    agent_edp_relative_change: float  # negativo = el agente gastó MENOS EDP que el baseline
    significance: SignificanceResult


def median_edp_by_kernel_class(df: pd.DataFrame, device: str, label: str) -> pd.Series:
    """Mediana de EDP por kernel_ref, para un (device, phase_label_train) --
    misma unidad que fase3_daemon/policy/derive_policy_table.py usa para
    comparar niveles de frecuencia, aquí para comparar escenarios completos.
    """
    subset = df[(df["device"] == device) & (df["phase_label_train"] == label)].copy()
    subset["edp"] = compute_window_edp(subset)
    subset = subset.dropna(subset=["edp"])
    return subset.groupby("kernel_ref")["edp"].median()


def compare_scenarios(
    windows_by_scenario: dict[str, pd.DataFrame],
    *,
    agent_scenario: str,
    baseline_scenarios: list[str],
    devices: tuple[str, ...] = ("cpu", "gpu"),
    labels: tuple[str, ...] = ("compute_bound", "memory_bound"),
    alpha: float = 0.05,
) -> list[ScenarioComparison]:
    """Compara el escenario del agente contra cada baseline, por separado
    para cada (device, phase_label_train) -- nunca agregando todas las
    clases en un solo número. Un kernel que no tiene datos en ambos lados
    de una comparación (agente y ese baseline) se excluye de esa
    comparación específica, no de todo el reporte.
    """
    if agent_scenario not in windows_by_scenario:
        raise ValueError(f"agent_scenario {agent_scenario!r} no está en windows_by_scenario")

    results: list[ScenarioComparison] = []
    for baseline in baseline_scenarios:
        if baseline not in windows_by_scenario:
            raise ValueError(f"baseline_scenario {baseline!r} no está en windows_by_scenario")
        for device in devices:
            for label in labels:
                agent_edp = median_edp_by_kernel_class(windows_by_scenario[agent_scenario], device, label)
                baseline_edp = median_edp_by_kernel_class(windows_by_scenario[baseline], device, label)
                common_kernels = sorted(set(agent_edp.index) & set(baseline_edp.index))
                if len(common_kernels) < 2:
                    continue  # sin suficientes pares para una prueba pareada -- se omite, no se fabrica un resultado

                agent_values = agent_edp.loc[common_kernels].to_numpy()
                baseline_values = baseline_edp.loc[common_kernels].to_numpy()
                relative_change = float(agent_values.sum() / baseline_values.sum()) - 1.0
                significance = paired_significance_test(baseline_values, agent_values, alpha=alpha)

                results.append(ScenarioComparison(
                    device=device, phase_label_train=label, baseline_scenario=baseline,
                    n_kernels_compared=len(common_kernels),
                    agent_edp_relative_change=relative_change,
                    significance=significance,
                ))
    return results


def format_report(comparisons: list[ScenarioComparison]) -> str:
    """Tabla de texto plano, una fila por (device, clase, baseline) -- el
    formato exacto de reporte (LaTeX/CSV/etc.) queda al capítulo de
    resultados; esto es la vista de trabajo mientras se corre la
    evaluación real."""
    lines = [
        f"{'device':<6}{'clase':<16}{'baseline':<12}{'n_kernels':>10}"
        f"{'Δ EDP agente':>14}{'prueba':>12}{'p-valor':>10}{'sig.':>6}"
    ]
    lines.append("-" * len(lines[0]))
    for c in sorted(comparisons, key=lambda c: (c.device, c.phase_label_train, c.baseline_scenario)):
        sign = "SI" if c.significance.significant else "no"
        pct = c.agent_edp_relative_change * 100
        lines.append(
            f"{c.device:<6}{c.phase_label_train:<16}{c.baseline_scenario:<12}{c.n_kernels_compared:>10}"
            f"{pct:>13.1f}%{c.significance.test_name:>12}{c.significance.p_value:>10.4f}{sign:>6}"
        )
    return "\n".join(lines)


def load_scenario_windows(windows_csv_by_scenario: dict[str, list[Path]]) -> dict[str, pd.DataFrame]:
    """Envoltura de conveniencia sobre common.edp.load_windows para varios
    escenarios a la vez."""
    return {
        scenario: load_windows(paths)
        for scenario, paths in windows_csv_by_scenario.items()
    }
