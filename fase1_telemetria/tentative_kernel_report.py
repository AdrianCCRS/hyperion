"""Informe de utilidad de los kernels tentativos después del cribado.

No selecciona silenciosamente el catálogo final. Resume procedencia, gate ncu,
calidad de la telemetría, cobertura Roofline, familia y condición metodológica
(externo, control sintético o dual en cuarentena), y emite una recomendación
revisable por kernel.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fase1_telemetria.gpu_phases import GPU_PHASE_DATASET_FILENAME
from fase1_telemetria.postprocess import TRAINING_CPU_INTERVALS_FILENAME
from fase2_clasificador.eval.protocol import derive_kernel_family

PLAN_CANDIDATES_NOT_YET_EXECUTABLE = {
    "cpu": ["npb_ep", "rodinia_kmeans_omp", "rodinia_srad_omp", "rodinia_nw_omp",
            "rodinia_particlefilter_omp"],
    "gpu": ["rodinia_kmeans", "rodinia_srad", "rodinia_nw", "rodinia_b+tree", "rodinia_cfd"],
}
GPU_PROXY_FEATURES = (
    "gpu_power_mw_median", "gpu_util_pct_median", "gpu_mem_util_pct_median",
    "gpu_sm_clock_mhz_median", "gpu_temperature_c_median",
)


def _load_frames(campaign_dir: Path, campaign_id: str, filename: str) -> pd.DataFrame:
    frames = []
    for run_dir in sorted(campaign_dir.glob(f"{campaign_id}__*__rep*")):
        path = run_dir / filename
        if path.is_file():
            frame = pd.read_csv(path, low_memory=False)
            frame["_source_run_dir"] = run_dir.name
            try:
                verdict = json.loads((run_dir / "verdict.json").read_text())
                frame["_run_verdict_accepted"] = verdict.get("accepted") is True
            except (OSError, json.JSONDecodeError):
                frame["_run_verdict_accepted"] = False
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _category(ref: str, entry: dict[str, Any]) -> str:
    suite = str(entry.get("suite", "")).lower()
    if ref.startswith(("phasic_", "gpu_phasic_")) or "hyperion-phase" in suite or ref == "ptrchase":
        return "diagnostic_control"
    if ref.startswith("dual_"):
        return "dual_quarantine"
    if ref == "gpu_dgemm_n4096" or entry.get("role") == "calibration":
        return "infrastructure_control"
    if "gap" in suite or ref in {"gpu_rajaperf_reduce3_int", "gpu_rajaperf_indexlist_3loop"}:
        return "external_needs_flops_review"
    return "external_candidate"


def _safe_median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else None


def _summarize_rows(ref: str, device: str, frame: pd.DataFrame) -> dict[str, Any]:
    subset = frame[frame.get("kernel_ref", pd.Series(dtype=str)) == ref].copy() if not frame.empty else frame
    if device == "cpu":
        usable = subset.get("training_quality_status", pd.Series(dtype=str)).eq("ok")
        oi_col = "operational_intensity_uncore_real"
        freq_col = "freq_level_id"
        quality_col = "training_quality_reason"
    else:
        usable = subset.get("training_eligible", pd.Series(dtype=str)).astype(str).str.lower().isin(("true", "1"))
        usable &= subset.get("phase_quality_status", pd.Series(dtype=str)).eq("ok")
        oi_col = "operational_intensity"
        freq_col = "gpu_freq_level_id"
        quality_col = "phase_quality_reason"
    if len(usable):
        usable &= subset.get("_run_verdict_accepted", pd.Series(False, index=subset.index)).astype(bool)
    good = subset[usable] if len(usable) else subset.iloc[0:0]
    labels = sorted(set(good.get("phase_label_train", pd.Series(dtype=str)).dropna()) - {""})
    ratios = pd.Series(dtype=float)
    if not good.empty and oi_col in good and "i_ridge_used" in good:
        oi = pd.to_numeric(good[oi_col], errors="coerce")
        ridge = pd.to_numeric(good["i_ridge_used"], errors="coerce")
        valid = (oi > 0) & (ridge > 0)
        ratios = (oi[valid] / ridge[valid]).map(math.log2)
    reasons = subset.loc[~usable, quality_col].dropna().astype(str) if quality_col in subset else pd.Series(dtype=str)
    return {
        "rows_total": int(len(subset)), "rows_usable": int(len(good)),
        "usable_fraction": float(len(good) / len(subset)) if len(subset) else 0.0,
        "repetitions_observed": int(pd.to_numeric(good.get("repetition", pd.Series(dtype=float)), errors="coerce").nunique()),
        "frequency_levels_observed": int(good.get(freq_col, pd.Series(dtype=str)).nunique()),
        "classes_observed": labels,
        "compute_rows": int(good.get("phase_label_train", pd.Series(dtype=str)).eq("compute_bound").sum()),
        "memory_rows": int(good.get("phase_label_train", pd.Series(dtype=str)).eq("memory_bound").sum()),
        "median_log2_oi_over_ridge": float(ratios.median()) if not ratios.empty else None,
        "primary_quality_problem": reasons.value_counts().index[0] if not reasons.empty else "",
    }


def build_report(workflow_path: Path) -> dict[str, Any]:
    wf = json.loads(workflow_path.read_text())
    catalog_doc = yaml.safe_load(Path(wf["catalog"]).read_text())
    entries = {entry["id"]: entry for entry in catalog_doc["kernels"]}
    cpu_manifest = yaml.safe_load(Path(wf["cpu_manifest"]).read_text())
    gpu_manifest = yaml.safe_load(Path(wf["gpu_candidates_manifest"]).read_text())
    cpu_refs = [item["kernel_ref"] for item in cpu_manifest["kernels"]]
    gpu_refs = [item["kernel_ref"] for item in gpu_manifest["kernels"]]
    cpu = _load_frames(Path(wf["cpu_campaign_dir"]), wf["cpu_campaign_id"], TRAINING_CPU_INTERVALS_FILENAME)
    gpu = _load_frames(Path(wf["gpu_campaign_dir"]), wf["gpu_campaign_id"], GPU_PHASE_DATASET_FILENAME)
    ncu_reports = {}
    for path in Path(wf["ncu_reports_dir"]).glob("*.json"):
        if path.name == "ncu_batch_summary.json":
            continue
        try:
            report = json.loads(path.read_text())
            ncu_reports[report.get("kernel_ref", path.stem)] = report
        except (OSError, json.JSONDecodeError):
            pass
    warmup_by_device: dict[str, dict[str, Any]] = {}
    for device in ("cpu", "gpu"):
        artifact = Path(wf[f"warmup_{device}_dir"]) / "warmup_calibration.json"
        try:
            warmup_by_device[device] = json.loads(artifact.read_text()).get("per_kernel", {})
        except (OSError, json.JSONDecodeError):
            warmup_by_device[device] = {}

    rows: list[dict[str, Any]] = []
    for device, refs, frame in (("cpu", cpu_refs, cpu), ("gpu", gpu_refs, gpu)):
        for ref in refs:
            entry = entries[ref]
            measured = _summarize_rows(ref, device, frame)
            category = _category(ref, entry)
            ncu_report = ncu_reports.get(ref, {}) if device == "gpu" else {}
            ncu_ok = ncu_report.get("roofline_label_eligible") is True if device == "gpu" else None
            warmup = warmup_by_device[device].get(ref, {})
            warmup_ok = warmup.get("status") in {"measured", "documented_fallback"}
            enough_matrix = (
                measured["rows_usable"] > 0
                and measured["repetitions_observed"] >= 3
                and measured["frequency_levels_observed"] >= 3
            )
            if device == "gpu" and not ncu_ok:
                action = "exclude_from_labeled_screen"
                reason = f"ncu: {ncu_report.get('status', 'missing_report')}"
            elif category == "diagnostic_control":
                action, reason = "keep_as_diagnostic_control", "sintético propio; no entrena"
            elif category == "infrastructure_control":
                action, reason = "keep_as_infrastructure_control", "ancla/calibración; no candidato automático"
            elif category == "dual_quarantine":
                action, reason = "manual_review_quarantine", "procedencia/optimización dual pendiente"
            elif category == "external_needs_flops_review":
                action, reason = "manual_review_flops_semantics", "FLOPs/byte puede no describir trabajo útil"
            elif not warmup_ok:
                action, reason = "repeat_or_reject_warmup", f"warmup: {warmup.get('status', 'missing')}"
            elif not enough_matrix:
                action, reason = "repeat_or_reject_quality", "faltan 3 repeticiones, 3 niveles o filas usables"
            elif not measured["classes_observed"]:
                action, reason = "reject_without_roofline_class", "sin etiqueta Roofline usable"
            else:
                action, reason = "candidate_for_final_campaign", "externo, medible y con verdad usable"
            rows.append({
                "device": device, "kernel_ref": ref,
                "kernel_family": derive_kernel_family(ref), "suite": entry.get("suite"),
                "category": category, "ncu_status": ncu_report.get("status") if device == "gpu" else "NA",
                "ncu_eligible": ncu_ok, "ncu_measured_oi": ncu_report.get("final_operational_intensity"),
                "ncu_catalog_relative_difference": ncu_report.get("operational_intensity_relative_difference"),
                "warmup_status": warmup.get("status", "missing"),
                "warmup_seconds": warmup.get("warmup_seconds"),
                **measured, "recommended_action": action, "recommendation_reason": reason,
            })

    eligible = [row for row in rows if row["recommended_action"] == "candidate_for_final_campaign"]
    coverage: dict[str, Any] = {}
    for device in ("cpu", "gpu"):
        dev = [row for row in eligible if row["device"] == device]
        compute = {row["kernel_family"] for row in dev if "compute_bound" in row["classes_observed"]}
        memory = {row["kernel_family"] for row in dev if "memory_bound" in row["classes_observed"]}
        coverage[device] = {
            "candidate_kernels": len(dev), "compute_families": sorted(compute),
            "memory_families": sorted(memory), "compute_family_count": len(compute),
            "memory_family_count": len(memory),
            "meets_minimum_5_per_class": len(compute) >= 5 and len(memory) >= 5,
        }

    proxy_separation = []
    if not gpu.empty:
        accepted = gpu.get("_run_verdict_accepted", pd.Series(False, index=gpu.index)).astype(bool)
        training = gpu.get("training_eligible", pd.Series(False, index=gpu.index)).astype(str).str.lower().isin(("true", "1"))
        usable_gpu = gpu[accepted & training]
        for feature in GPU_PROXY_FEATURES:
            if feature not in usable_gpu:
                continue
            values = pd.to_numeric(usable_gpu[feature], errors="coerce")
            compute_values = values[usable_gpu["phase_label_train"] == "compute_bound"].dropna()
            memory_values = values[usable_gpu["phase_label_train"] == "memory_bound"].dropna()
            overall = values.dropna()
            q1, q3 = (overall.quantile(0.25), overall.quantile(0.75)) if not overall.empty else (math.nan, math.nan)
            iqr = q3 - q1
            cmed = float(compute_values.median()) if not compute_values.empty else None
            mmed = float(memory_values.median()) if not memory_values.empty else None
            effect = abs(cmed - mmed) / iqr if cmed is not None and mmed is not None and iqr > 0 else None
            proxy_separation.append({
                "feature": feature, "compute_median": cmed, "memory_median": mmed,
                "absolute_median_difference_over_global_iqr": effect,
                "compute_rows": len(compute_values), "memory_rows": len(memory_values),
            })
    return {
        "schema": "f1/tentative_kernel_utility/1", "workflow": wf,
        "coverage": coverage, "kernels": rows,
        "gpu_proxy_separation_descriptive": proxy_separation,
        "plan_candidates_not_yet_executable": PLAN_CANDIDATES_NOT_YET_EXECUTABLE,
    }


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tentative_kernel_utility.json"
    csv_path = output_dir / "tentative_kernel_utility.csv"
    md_path = output_dir / "tentative_kernel_utility.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    csv_rows = []
    for row in report["kernels"]:
        flat = dict(row)
        flat["classes_observed"] = ";".join(row["classes_observed"])
        csv_rows.append(flat)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    lines = ["# Informe de utilidad de kernels tentativos", ""]
    for device in ("cpu", "gpu"):
        cov = report["coverage"][device]
        lines += [
            f"## {device.upper()}", "",
            f"Candidatos recomendados: {cov['candidate_kernels']}. Familias compute: "
            f"{cov['compute_family_count']}; familias memory: {cov['memory_family_count']}. "
            f"Mínimo 5/clase: {'PASS' if cov['meets_minimum_5_per_class'] else 'FAIL'}.", "",
            "| Kernel | Familia | Clases | Filas usables | Mediana log2(OI/ridge) | Acción | Motivo |",
            "|---|---|---|---:|---:|---|---|",
        ]
        for row in (item for item in report["kernels"] if item["device"] == device):
            ratio = row["median_log2_oi_over_ridge"]
            ratio_text = "NA" if ratio is None else f"{ratio:.3f}"
            lines.append(
                f"| {row['kernel_ref']} | {row['kernel_family']} | "
                f"{', '.join(row['classes_observed']) or 'ninguna'} | {row['rows_usable']} | "
                f"{ratio_text} | {row['recommended_action']} | {row['recommendation_reason']} |"
            )
        lines.append("")
    lines += [
        "## Separación descriptiva de proxies NVML", "",
        "Esto no entrena ni valida un modelo. Compara medianas de las filas GPU elegibles; "
        "el efecto es |mediana compute - mediana memory| dividido por el IQR global.", "",
        "| Feature | Mediana compute | Mediana memory | Diferencia/IQR |",
        "|---|---:|---:|---:|",
    ]
    for item in report["gpu_proxy_separation_descriptive"]:
        def fmt(value):
            return "NA" if value is None else f"{value:.4g}"
        lines.append(
            f"| {item['feature']} | {fmt(item['compute_median'])} | "
            f"{fmt(item['memory_median'])} | "
            f"{fmt(item['absolute_median_difference_over_global_iqr'])} |"
        )
    lines += [
        "",
        "## Candidatos del plan todavía no ejecutables", "",
        "Estos kernels no se califican como útiles o inútiles: aún carecen de binario, checksum y "
        "caracterización reproducible para este flujo.", "",
        f"- CPU: {', '.join(report['plan_candidates_not_yet_executable']['cpu'])}",
        f"- GPU: {', '.join(report['plan_candidates_not_yet_executable']['gpu'])}", "",
        "## Interpretación", "",
        "`candidate_for_final_campaign` es una recomendación reproducible, no una selección automática. "
        "La lista definitiva debe congelarse después de revisar cobertura, cercanía al ridge, calidad, "
        "procedencia y redundancia por familia. Los controles y kernels en cuarentena no cuentan para "
        "el mínimo de familias por clase.", "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.workflow)
    wf = report["workflow"]
    paths = write_report(report, args.output_dir or Path(wf["utility_dir"]))
    for device, coverage in report["coverage"].items():
        print(f"{device}: compute={coverage['compute_family_count']} familias, "
              f"memory={coverage['memory_family_count']}, "
              f"mínimo={'PASS' if coverage['meets_minimum_5_per_class'] else 'FAIL'}")
    print("artefactos:", *(str(path) for path in paths), sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
