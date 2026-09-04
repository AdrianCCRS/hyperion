"""Diagnóstico de cobertura Roofline por familia, sin entrenar ni reescribir datos.

F2-XDEV-001 separa esta etapa de la adquisición: Fase 1 conserva las filas
físicas/auditables y este módulo solo informa qué familias, clases y niveles
están disponibles para decidir el catálogo final y el balance posterior.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fase1_telemetria.postprocess import TRAINING_CPU_INTERVALS_FILENAME  # noqa: E402
from fase2_clasificador.eval.protocol import derive_kernel_family  # noqa: E402

CPU_REQUIRED = {
    "kernel_ref", "freq_level_id", "phase_label_train", "training_quality_status",
    "training_quality_reason", "operational_intensity_uncore_real", "i_ridge_used",
}
GPU_REQUIRED = {
    "kernel_ref", "freq_level_id", "gpu_freq_level_id", "phase_label_train",
    "quality_status", "operational_intensity", "i_ridge_used",
}


def _run_directories(campaign_dir: Path, campaign_id: str) -> list[Path]:
    return sorted(path for path in campaign_dir.glob(f"{campaign_id}__*__rep*") if path.is_dir())


def _read_campaign(
    campaign_dir: Path, campaign_id: str, device: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lee todas las corridas existentes, preservando los rechazos para QA."""
    filename = TRAINING_CPU_INTERVALS_FILENAME if device == "cpu" else "windows.csv"
    required = CPU_REQUIRED if device == "cpu" else GPU_REQUIRED
    frames: list[pd.DataFrame] = []
    missing_files: list[str] = []
    for run_dir in _run_directories(campaign_dir, campaign_id):
        path = run_dir / filename
        if not path.exists():
            missing_files.append(str(path))
            continue
        frame = pd.read_csv(path, low_memory=False)
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{path} no tiene las columnas necesarias para diagnóstico: {sorted(missing_columns)}"
            )
        frame["_source_run_dir"] = run_dir.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"ningún {filename} legible bajo {campaign_dir} para campaign_id={campaign_id!r}"
        )
    metadata = {
        "input_filename": filename,
        "runs_discovered": len(_run_directories(campaign_dir, campaign_id)),
        "runs_loaded": len(frames),
        "missing_input_files": missing_files,
    }
    return pd.concat(frames, ignore_index=True), metadata


def _valid_rows(frame: pd.DataFrame, device: str) -> tuple[pd.Series, pd.Series]:
    label_present = frame["phase_label_train"].notna() & (frame["phase_label_train"] != "")
    if device == "cpu":
        usable = (frame["training_quality_status"] == "ok") & label_present
        reason = frame["training_quality_reason"].fillna("")
    else:
        # Esto describe cobertura de las muestras NVML, no fases independientes.
        usable = (frame["quality_status"] == "gpu_telemetry") & label_present
        reason = frame["quality_status"].fillna("")
    return usable, reason


def _primary_reason(values: pd.Series) -> str:
    usable = values[values.notna() & (values != "")]
    if usable.empty:
        return ""
    return str(usable.value_counts().index[0])


def _coverage_tables(
    frame: pd.DataFrame, *, device: str, near_ridge_log2: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = frame.copy()
    frame["family"] = frame["kernel_ref"].map(derive_kernel_family)
    # En GPU la frecuencia que cambia el ridge es gpu_freq_level_id; la CPU
    # anfitriona suele permanecer en REF y no debe agrupar falsamente todos
    # los niveles GPU bajo el mismo valor.
    if device == "gpu":
        frame["analysis_freq_level_id"] = frame["gpu_freq_level_id"].fillna(frame["freq_level_id"])
    else:
        frame["analysis_freq_level_id"] = frame["freq_level_id"]
    usable, rejection_reason = _valid_rows(frame, device)
    frame["_usable"] = usable
    frame["_rejection_reason"] = rejection_reason

    oi_column = "operational_intensity_uncore_real" if device == "cpu" else "operational_intensity"
    frame["_oi"] = pd.to_numeric(frame[oi_column], errors="coerce")
    frame["_ridge"] = pd.to_numeric(frame["i_ridge_used"], errors="coerce")
    valid_oi = (frame["_oi"] > 0) & (frame["_ridge"] > 0)
    frame["log2_oi_over_ridge"] = np.where(
        valid_oi, np.log2(frame["_oi"] / frame["_ridge"]), np.nan
    )
    frame["near_ridge"] = frame["log2_oi_over_ridge"].abs() <= near_ridge_log2

    group_columns = ["family", "kernel_ref", "analysis_freq_level_id"]
    records: list[dict[str, Any]] = []
    quality_records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        family, kernel_ref, frequency = keys
        good = group[group["_usable"]]
        n_total = len(group)
        n_usable = len(good)
        n_compute = int((good["phase_label_train"] == "compute_bound").sum())
        n_memory = int((good["phase_label_train"] == "memory_bound").sum())
        records.append({
            "device": device,
            "family": family,
            "kernel_ref": kernel_ref,
            "freq_level_id": frequency,
            "n_rows_usable": n_usable,
            "n_compute_bound": n_compute,
            "n_memory_bound": n_memory,
            "memory_fraction": n_memory / n_usable if n_usable else math.nan,
            "n_near_ridge": int(good["near_ridge"].sum()),
            "near_ridge_fraction": float(good["near_ridge"].mean()) if n_usable else math.nan,
            "median_log2_oi_over_ridge": float(good["log2_oi_over_ridge"].median()) if n_usable else math.nan,
        })
        rejected = group[~group["_usable"]]
        quality_records.append({
            "device": device,
            "family": family,
            "kernel_ref": kernel_ref,
            "freq_level_id": frequency,
            "n_rows_total": n_total,
            "n_rows_usable": n_usable,
            "n_rows_rejected": len(rejected),
            "usable_fraction": n_usable / n_total if n_total else math.nan,
            "primary_rejection_reason": _primary_reason(rejected["_rejection_reason"]),
        })

    coverage = pd.DataFrame(records)
    quality = pd.DataFrame(quality_records)
    family_rows = coverage.groupby("family", sort=True)[["n_compute_bound", "n_memory_bound"]].sum()
    family_summary: dict[str, str] = {}
    for family, counts in family_rows.iterrows():
        if counts["n_compute_bound"] and counts["n_memory_bound"]:
            family_summary[str(family)] = "mixed"
        elif counts["n_compute_bound"]:
            family_summary[str(family)] = "compute_only"
        elif counts["n_memory_bound"]:
            family_summary[str(family)] = "memory_only"
        else:
            family_summary[str(family)] = "no_usable_label"
    diagnostics = {
        "rows_total": int(len(frame)),
        "rows_usable": int(usable.sum()),
        "rows_rejected": int((~usable).sum()),
        "families_observed": int(frame["family"].nunique()),
        "families_by_coverage": family_summary,
        "families_with_compute": sum(value in {"compute_only", "mixed"} for value in family_summary.values()),
        "families_with_memory": sum(value in {"memory_only", "mixed"} for value in family_summary.values()),
        "near_ridge_log2_threshold": near_ridge_log2,
    }
    return coverage, quality, diagnostics


def analyze_campaign(
    campaign_dir: str | Path,
    campaign_id: str,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    near_ridge_log2: float = 1.0,
) -> dict[str, Any]:
    """Genera artefactos de diagnóstico; no modifica la campaña fuente."""
    if device not in {"cpu", "gpu"}:
        raise ValueError("device debe ser 'cpu' o 'gpu'")
    if near_ridge_log2 < 0:
        raise ValueError("near_ridge_log2 debe ser no negativo")
    source = Path(campaign_dir)
    destination = Path(output_dir)
    frame, input_metadata = _read_campaign(source, campaign_id, device)
    coverage, quality, diagnostics = _coverage_tables(
        frame, device=device, near_ridge_log2=near_ridge_log2
    )
    destination.mkdir(parents=True, exist_ok=True)
    coverage_path = destination / "family_class_frequency_summary.csv"
    quality_path = destination / "kernel_quality_summary.csv"
    report_path = destination / "phase_coverage_report.json"
    coverage.to_csv(coverage_path, index=False)
    quality.to_csv(quality_path, index=False)

    semantics = (
        "one_uncore_interval_per_row"
        if device == "cpu"
        else "raw_nvml_samples_only_not_independent_gpu_phases"
    )
    report = {
        "schema_version": 1,
        "campaign_dir": str(source),
        "campaign_id": campaign_id,
        "device": device,
        "row_semantics": semantics,
        "eligible_for_training_without_additional_aggregation": device == "cpu",
        "input": input_metadata,
        "diagnostics": diagnostics,
        "artifacts": {
            "family_class_frequency_summary_csv": str(coverage_path),
            "kernel_quality_summary_csv": str(quality_path),
        },
        "pending": (
            [] if device == "cpu" else [
                "Agregar muestras NVML por corrida o fase antes de entrenar GPU.",
                "No interpretar las filas NVML periódicas como ejemplos independientes.",
            ]
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostica cobertura Roofline por familia sin entrenar ni alterar los CSV fuente."
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--near-ridge-log2", type=float, default=1.0,
        help="Considera cerca del ridge si abs(log2(OI/I_ridge)) <= este valor (default: 1).",
    )
    args = parser.parse_args()
    report = analyze_campaign(
        args.campaign_dir, args.campaign_id, args.output_dir,
        device=args.device, near_ridge_log2=args.near_ridge_log2,
    )
    diagnostics = report["diagnostics"]
    print(
        f"{args.device}: {diagnostics['rows_usable']:,}/{diagnostics['rows_total']:,} filas usables | "
        f"{diagnostics['families_observed']} familias | "
        f"compute={diagnostics['families_with_compute']} memory={diagnostics['families_with_memory']}"
    )


if __name__ == "__main__":
    main()
