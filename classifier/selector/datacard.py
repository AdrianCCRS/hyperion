"""Data card obligatorio del dataset compacto (seccion 10.1 del plan).

Se genera ANTES de entrenar. Su proposito no es documentar sino bloquear: si
la distribucion de la etiqueta ya esta decidida por una constante, o si los
margenes viven por debajo del ruido de medicion, entrenar un modelo mide
jitter y no capacidad predictiva.

Produce dos salidas equivalentes -- `datacard.json` para consumo programatico
y `datacard.md` para lectura -- desde una unica estructura, de modo que no
puedan divergir.
"""
from __future__ import annotations

from typing import Any, Mapping
import json

import numpy as np
import pandas as pd

from .compact import NOISE_FLOOR_PCT, REF_ACTIONS


def _describe(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"n": 0, "median": None, "mean": None, "p95": None, "min": None, "max": None}
    return {
        "n": int(len(values)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p95": float(values.quantile(0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def build_datacard(
    compact_frame: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    amortization: pd.DataFrame | None = None,
    headroom: pd.DataFrame | None = None,
    folds: list[tuple[str, pd.DataFrame, pd.DataFrame]] | None = None,
    run_regions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    card: dict[str, Any] = {"noise_floor_pct": NOISE_FLOOR_PCT}

    card["counts"] = {
        "config_ids": int(compact_frame["config_id"].nunique()),
        "compact_rows": int(len(compact_frame)),
        "operations": int(compact_frame["operation"].nunique()),
        "resource_states": sorted(compact_frame["resource_state"].astype(str).unique()),
        "candidate_rows": int(len(candidates)),
        "candidate_actions": int(candidates["action_id"].nunique()),
        "ref_actions": list(REF_ACTIONS),
    }

    card["sizes_by_operation"] = {
        str(operation): sorted(int(size) for size in group["size"].unique())
        for operation, group in compact_frame.groupby("operation", observed=True)
    }

    card["device_winner"] = {
        "by_resource_state": {
            str(state): group["device_label"].value_counts().to_dict()
            for state, group in compact_frame.groupby("resource_state", observed=True)
        },
        "by_operation_and_state": {
            f"{operation}|{state}": group["device_label"].value_counts().to_dict()
            for (operation, state), group in compact_frame.groupby(
                ["operation", "resource_state"], observed=True)
        },
        "by_size_and_state": {
            f"{row['operation']}_N{row['size']}|{row['resource_state']}": str(row["device_label"])
            for row in compact_frame.to_dict("records")
        },
    }

    # Distribucion de acciones ganadoras sobre el catalogo completo de 40
    # acciones, que sigue siendo la referencia de la capa de frecuencia.
    winning_actions: dict[str, dict[str, int]] = {}
    for region, group in candidates.groupby("region", observed=True):
        finite = group[np.isfinite(pd.to_numeric(group["edp_mean"], errors="coerce"))]
        winners = finite.loc[finite.groupby("config_id", observed=True)["edp_mean"].idxmin()]
        winning_actions[str(region)] = winners["action_id"].value_counts().to_dict()
    card["winning_actions_by_region"] = winning_actions

    separated = compact_frame["device_decision_separated"].astype(bool)
    card["device_margin"] = {
        "summary_pct": _describe(compact_frame["device_margin_pct"]),
        "separated": int(separated.sum()),
        "uncertain": int((~separated).sum()),
        "separated_fraction": float(separated.mean()),
        "by_resource_state_pct": {
            str(state): _describe(group["device_margin_pct"])
            for state, group in compact_frame.groupby("resource_state", observed=True)
        },
        "separation_method_counts": (
            compact_frame["device_decision_uncertainty_method"].value_counts().to_dict()
            if "device_decision_uncertainty_method" in compact_frame else {}
        ),
    }

    if headroom is not None:
        card["frequency_margin"] = {
            "dvfs_headroom_pct_by_state": {
                str(state): _describe(group["dvfs_headroom_pct"])
                for state, group in headroom.groupby("resource_state", observed=True)
            },
            "above_noise_floor_by_state": {
                str(state): {
                    "above": int(group["above_noise_floor"].sum()),
                    "total": int(len(group)),
                }
                for state, group in headroom.groupby("resource_state", observed=True)
            },
        }

    # Coeficiente de variacion entre repeticiones: es el piso de ruido real
    # contra el que se contrastan todos los margenes anteriores.
    cv_records: dict[str, Any] = {}
    for column, label in (("edp_cv_pct", "edp"), ("time_cv_pct", "time"), ("energy_cv_pct", "energy")):
        if column in candidates:
            cv_records[label] = _describe(candidates[column])
    if "edp_cv_pct" in candidates:
        cv_records["edp_by_device"] = {
            str(device): _describe(group["edp_cv_pct"])
            for device, group in candidates.groupby("device", observed=True)
        }
        cv_records["edp_by_region"] = {
            str(region): _describe(group["edp_cv_pct"])
            for region, group in candidates.groupby("region", observed=True)
        }
    card["repetition_cv"] = cv_records

    feature_missing: dict[str, dict[str, Any]] = {}
    for column in compact_frame.columns:
        series = compact_frame[column]
        missing = int(series.isna().sum())
        if missing:
            feature_missing[str(column)] = {
                "missing": missing,
                "fraction": float(missing / len(compact_frame)),
                "cause": (
                    "sondeo inexistente en none_ready (el agente aun no ha ejecutado nada)"
                    if str(column).startswith("probe_")
                    else "no determinada"
                ),
            }
    card["missing_by_feature"] = feature_missing

    if run_regions is not None and not run_regions.empty:
        cold_ref = run_regions[
            (run_regions["region"] == "cold") & (run_regions["action_id"].isin(REF_ACTIONS))
        ]
        ratio = pd.to_numeric(cold_ref.get("region_to_sampling_ratio"), errors="coerce")
        card["telemetry_coverage"] = {
            "cold_ref_runs": int(len(cold_ref)),
            "cold_ref_low_resolution_runs": int((ratio < 1.0).sum()),
            "cold_ref_low_resolution_fraction": float((ratio < 1.0).mean()) if len(ratio) else None,
            "rapl_coverage_fraction": _describe(run_regions.get("rapl_coverage_fraction", pd.Series(dtype=float))),
            "gpu_coverage_fraction": _describe(
                run_regions.loc[run_regions["device"] == "gpu", "gpu_coverage_fraction"]
                if "gpu_coverage_fraction" in run_regions else pd.Series(dtype=float)
            ),
        }

    if amortization is not None:
        finite = np.isfinite(amortization["k_break_even"])
        card["amortization"] = {
            "finite_break_even": int(finite.sum()),
            "infinite_break_even": int((~finite).sum()),
            "by_operation": {
                str(operation): {
                    "finite": int(np.isfinite(group["k_break_even"]).sum()),
                    "total": int(len(group)),
                    "min_finite_k": (
                        float(group.loc[np.isfinite(group["k_break_even"]), "k_break_even"].min())
                        if np.isfinite(group["k_break_even"]).any() else None
                    ),
                }
                for operation, group in amortization.groupby("operation", observed=True)
            },
            "analytic_inconsistencies": list(amortization.attrs.get("analytic_inconsistencies", [])),
        }

    if folds:
        card["fold_balance"] = [
            {
                "fold": name,
                "train_configs": int(train["config_id"].nunique()),
                "test_configs": int(test["config_id"].nunique()),
                "train_device_labels": train["device_label"].value_counts().to_dict(),
                "test_device_labels": test["device_label"].value_counts().to_dict(),
                "test_device_labels_gpu_ready": (
                    test[test["resource_state"] == "gpu_ready"]["device_label"]
                    .value_counts().to_dict()
                ),
                "test_operations": sorted(test["operation"].astype(str).unique()),
            }
            for name, train, test in folds
        ]

    return card


def _format_stats(stats: Mapping[str, Any]) -> str:
    if not stats or stats.get("median") is None:
        return "sin datos"
    return (
        f"n={stats['n']}, mediana={stats['median']:.4g}, media={stats['mean']:.4g}, "
        f"p95={stats['p95']:.4g}, rango=[{stats['min']:.4g}, {stats['max']:.4g}]"
    )


def render_markdown(card: Mapping[str, Any]) -> str:
    lines: list[str] = [
        "# Data card -- dataset compacto del selector CPU/GPU",
        "",
        "Generado por `classifier.selector.datacard` (seccion 10.1 del plan de",
        "reformulacion). Piso de ruido de medicion utilizado como referencia:",
        f"**{card['noise_floor_pct']:.2f} %** (CV mediano entre repeticiones).",
        "",
        "## 1. Conteos",
        "",
    ]
    for key, value in card["counts"].items():
        lines.append(f"- `{key}`: {value}")

    lines += ["", "## 2. Tamanos disponibles por operacion", ""]
    for operation, sizes in card["sizes_by_operation"].items():
        lines.append(f"- **{operation}** ({len(sizes)}): {', '.join(str(s) for s in sizes)}")

    lines += ["", "## 3. Dispositivo ganador", "", "### Por estado de recurso", ""]
    for state, counts in card["device_winner"]["by_resource_state"].items():
        lines.append(f"- `{state}`: {counts}")
    lines += ["", "### Por operacion y estado", ""]
    for key, counts in card["device_winner"]["by_operation_and_state"].items():
        lines.append(f"- `{key}`: {counts}")

    lines += ["", "## 4. Acciones ganadoras (catalogo de 40 acciones)", ""]
    for region, counts in card["winning_actions_by_region"].items():
        ordered = sorted(counts.items(), key=lambda item: -item[1])
        lines.append(f"- **region {region}**: " + ", ".join(f"`{a}`={n}" for a, n in ordered))

    margin = card["device_margin"]
    lines += [
        "", "## 5. Margen de la decision de dispositivo", "",
        f"- resumen (%): {_format_stats(margin['summary_pct'])}",
        f"- separados del ruido: **{margin['separated']}**; inciertos: "
        f"**{margin['uncertain']}** ({margin['separated_fraction']:.1%} separados)",
        "",
    ]
    for state, stats in margin["by_resource_state_pct"].items():
        lines.append(f"- `{state}`: {_format_stats(stats)}")

    if "frequency_margin" in card:
        lines += ["", "## 6. Headroom de frecuencia tras fijar el dispositivo", ""]
        for state, stats in card["frequency_margin"]["dvfs_headroom_pct_by_state"].items():
            above = card["frequency_margin"]["above_noise_floor_by_state"][state]
            lines.append(
                f"- `{state}`: {_format_stats(stats)} -- sobre el piso de ruido: "
                f"{above['above']}/{above['total']}"
            )

    lines += ["", "## 7. Coeficiente de variacion entre repeticiones", ""]
    for key, stats in card["repetition_cv"].items():
        if isinstance(stats, dict) and "median" in stats:
            lines.append(f"- `{key}` (%): {_format_stats(stats)}")
        elif isinstance(stats, dict):
            for sub, sub_stats in stats.items():
                lines.append(f"- `{key}.{sub}` (%): {_format_stats(sub_stats)}")

    lines += ["", "## 8. Faltantes por caracteristica y causa", ""]
    if not card["missing_by_feature"]:
        lines.append("- ninguna columna del dataset compacto tiene valores ausentes")
    for column, info in card["missing_by_feature"].items():
        lines.append(
            f"- `{column}`: {info['missing']} ({info['fraction']:.1%}) -- {info['cause']}"
        )

    if "telemetry_coverage" in card:
        lines += ["", "## 9. Cobertura de telemetria", ""]
        coverage = card["telemetry_coverage"]
        lines.append(f"- corridas frias REF: {coverage['cold_ref_runs']}")
        lines.append(
            f"- de baja resolucion (region mas corta que el intervalo de muestreo): "
            f"{coverage['cold_ref_low_resolution_runs']}"
            + (f" ({coverage['cold_ref_low_resolution_fraction']:.1%})"
               if coverage["cold_ref_low_resolution_fraction"] is not None else "")
        )
        lines.append(f"- cobertura RAPL: {_format_stats(coverage['rapl_coverage_fraction'])}")
        lines.append(f"- cobertura NVML (GPU): {_format_stats(coverage['gpu_coverage_fraction'])}")

    if "amortization" in card:
        amortization = card["amortization"]
        lines += [
            "", "## 10. Amortizacion `K_break_even`", "",
            f"- con cruce finito: **{amortization['finite_break_even']}**; "
            f"sin cruce: **{amortization['infinite_break_even']}**",
            "",
        ]
        for operation, info in amortization["by_operation"].items():
            minimum = info["min_finite_k"]
            lines.append(
                f"- **{operation}**: {info['finite']}/{info['total']} con cruce finito"
                + (f", menor K = {minimum:.0f}" if minimum is not None else "")
            )
        inconsistencies = amortization["analytic_inconsistencies"]
        lines += [
            "",
            "Comprobacion analitica (GPU inicialmente peor: cruce finito solo si "
            "gana en caliente; se admite y marca el caso transitorio K=1): "
            + ("**sin inconsistencias**" if not inconsistencies
               else f"**{len(inconsistencies)} inconsistencias**: {inconsistencies}"),
        ]

    if "fold_balance" in card:
        lines += ["", "## 11. Balance de los pliegues", ""]
        for fold in card["fold_balance"]:
            lines.append(
                f"- `{fold['fold']}`: train={fold['train_configs']} cfg, "
                f"test={fold['test_configs']} cfg; etiquetas de prueba en "
                f"`gpu_ready` = {fold['test_device_labels_gpu_ready']}"
            )

    return "\n".join(lines) + "\n"


def write_datacard(card: Mapping[str, Any], output_dir) -> dict[str, Any]:
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "datacard.json"
    markdown_path = output_dir / "datacard.md"
    json_path.write_text(
        json.dumps(card, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(card), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
