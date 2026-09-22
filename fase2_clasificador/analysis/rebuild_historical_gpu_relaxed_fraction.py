"""A1 del plan de mejora GPU (2026-09-22): reconstruye el dataset histórico
fusionado admitiendo corridas que ``merge_historical_gpu_phase_runs.py``
descarta solo por ``usable_sample_fraction < 0.5``
(``fase1_telemetria/gpu_phases.py::build_gpu_phase_rows``), para las 7
familias que quedaron sin cobertura en REF/F0/F1/F2 del dataset original.

Por qué es defendible relajar el umbral aquí y no en general: el conteo
ABSOLUTO de muestras NVML de las corridas rechazadas es alto (341 a 6176,
ver auditoría de esta sesión) -- el 0.5 exige que la MITAD de las muestras
colectadas sean posteriores al warmup/transición, no que haya pocas muestras
en total. Bajar el umbral a 0.20 conserva corridas con de sobra estadística,
excluyendo solo las que de verdad tienen poquísima señal útil.

No toca `merge_historical_gpu_phase_runs.py` (el contrato oficial sigue
exigiendo phase_quality_status=="ok" sin relajar) -- esto es un paso
adicional, explícito y trazable (columna `quality_gate_relaxed`), no un
cambio silencioso del gate de producción.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

FEATURES = ("gpu_util_pct_median", "gpu_mem_util_pct_median", "gpu_power_mw_median", "gpu_sm_clock_mhz_median")
DEFAULT_IN_SCOPE_FAMILIES = (
    "dual_cholesky", "dual_fft", "gpu_cutlass_simt_dgemm_n4096", "rajaperf_stream",
    "rodinia_gaussian", "rodinia_heartwall", "rodinia_lud",
)


def load_raw(root: Path) -> pd.DataFrame:
    paths = sorted(Path(p) for p in glob.glob(str(root / "**" / "training_gpu_phases.csv"), recursive=True))
    if not paths:
        raise ValueError(f"ningún training_gpu_phases.csv bajo {root}")
    frames = []
    for p in paths:
        frame = pd.read_csv(p, low_memory=False)
        frame["source_file"] = str(p)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def apply_relaxed_gate(raw: pd.DataFrame, families: tuple[str, ...], min_fraction: float) -> pd.DataFrame:
    raw = raw[raw["granularity"] == "run"].copy()
    raw["quality_gate_relaxed"] = False
    already_ok = raw["phase_quality_status"].eq("ok")

    is_target = (
        raw["kernel_family"].isin(families)
        & raw["phase_quality_status"].eq("insufficient_samples")
        & raw["phase_quality_reason"].astype(str).str.contains("usable_sample_fraction", na=False)
        & (pd.to_numeric(raw["usable_sample_fraction"], errors="coerce") >= min_fraction)
    )
    relaxed = raw.loc[is_target].copy()
    relaxed["phase_quality_status"] = "ok"
    relaxed["phase_quality_reason"] = ""
    relaxed["training_eligible"] = True
    relaxed["quality_gate_relaxed"] = True

    keep = raw.loc[already_ok].copy()
    keep["training_eligible"] = True
    combined = pd.concat([keep, relaxed], ignore_index=True)

    eligible = combined["training_eligible"].astype(bool)
    quality_freq_ok = combined["gpu_frequency_quality_status"].isin({"valid", "not_applicable_native"})
    numeric = combined.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    labels = combined["phase_label_train"].isin({"compute_bound", "memory_bound"})
    combined = combined[eligible & quality_freq_ok & numeric & labels].copy()

    duplicated = combined["run_id"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(f"run_id duplicado tras relajar: {sorted(combined.loc[duplicated, 'run_id'].unique())[:5]}")
    return combined.sort_values("run_id").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", required=True, type=Path, help="directorio con training_gpu_phases.csv crudos (aceptados y rechazados)")
    ap.add_argument("--min-usable-sample-fraction", type=float, default=0.20)
    ap.add_argument("--families", nargs="*", default=list(DEFAULT_IN_SCOPE_FAMILIES))
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    raw = load_raw(args.raw_root)
    merged = apply_relaxed_gate(raw, tuple(args.families), args.min_usable_sample_fraction)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    report = {
        "schema": "historical_gpu_relaxed_fraction/1",
        "min_usable_sample_fraction": args.min_usable_sample_fraction,
        "families_targeted": args.families,
        "accepted_rows": len(merged),
        "relaxed_rows": int(merged["quality_gate_relaxed"].sum()),
        "families_in_output": int(merged["kernel_family"].nunique()),
        "class_balance": merged["phase_label_train"].value_counts().to_dict(),
        "relaxed_by_family": merged.loc[merged["quality_gate_relaxed"], "kernel_family"].value_counts().to_dict(),
        "freq_levels_added_by_family": {
            fam: sorted(g.loc[g["quality_gate_relaxed"], "gpu_freq_level_id"].unique().tolist())
            for fam, g in merged.groupby("kernel_family") if g["quality_gate_relaxed"].any()
        },
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(merged)} corridas ({report['relaxed_rows']} admitidas por el gate relajado) -> {args.output}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
