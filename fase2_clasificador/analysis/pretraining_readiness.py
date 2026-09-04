"""Auditoría única de readiness pre-entrenamiento (brecha H del encargo F1).

Determina si un dataset (CPU y/o GPU) está listo para pasar a entrenamiento.
NO entrena. NO fabrica evidencia: un gate que necesita hardware o una campaña
real que todavía no existe se reporta ``BLOCKED``, nunca ``PASS`` con un
fixture.

Cada gate devuelve uno de:

- ``PASS``     -- se verificó y está bien;
- ``FAIL``     -- se verificó y está mal (bloquea entrenamiento);
- ``BLOCKED``  -- no se puede verificar sin hardware / permisos / campaña real;
- ``NA``       -- no aplica a este dispositivo.

El dataset se considera *listo para entrenamiento* solo si NINGÚN gate está en
``FAIL`` ni en ``BLOCKED``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fase2_clasificador.analysis.feature_contract import ROOFLINE_TRUTH_COLUMNS  # noqa: E402

PASS, FAIL, BLOCKED, NA = "PASS", "FAIL", "BLOCKED", "NA"

_CPU_GRANULARITY = "uncore_interval"
_GPU_GRANULARITY = "run"


@dataclass
class GateResult:
    gate: str
    cpu: str = NA
    gpu: str = NA
    detail: str = ""

    def worst(self) -> str:
        order = {PASS: 0, NA: 0, BLOCKED: 1, FAIL: 2}
        return self.cpu if order[self.cpu] >= order[self.gpu] else self.gpu


# ----------------------------------------------------------------- inputs

@dataclass
class ReadinessInputs:
    cpu_dataset: Path | None = None          # training_cpu_intervals.csv (o dir con ellos)
    gpu_dataset: Path | None = None          # training_gpu_phases.csv
    gpu_contract_file: Path | None = None    # training_gpu_phases_contract.json
    feature_contract_cpu: Path | None = None # frozen_feature_contract_cpu.json
    feature_contract_gpu: Path | None = None
    feature_report_cpu: Path | None = None   # feature_contract_cpu.json (Pearson/Spearman/VIF)
    feature_report_gpu: Path | None = None
    ncu_reports_dir: Path | None = None      # F1-GPU-004: dir con convergencia por kernel
    warmup_artifact: Path | None = None      # F1-XDEV-002: warmup_calibration.json
    coverage_report: Path | None = None      # F2-XDEV-001: phase_coverage_report.json
    transition_aggregate: Path | None = None # F1-GPU-002: transition_matrix_aggregate.json


def _load_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if path.is_dir():
        files = sorted(path.rglob("*.csv"))
        if not files:
            return None
        return pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    return None


def _load_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ----------------------------------------------------------------- gates

def _g_provenance(cpu: pd.DataFrame | None, gpu: pd.DataFrame | None) -> GateResult:
    r = GateResult("checksums_y_procedencia")
    for name, df, setter in (("cpu", cpu, "cpu"), ("gpu", gpu, "gpu")):
        if df is None:
            continue
        needed = {"binary_checksum", "kernel_ref"}
        cal_col = "roofline_calibration_ref" if name == "gpu" else None
        missing_cols = needed - set(df.columns)
        empties = [c for c in needed & set(df.columns) if df[c].isna().any() or (df[c] == "").any()]
        st = PASS
        if missing_cols or empties:
            st = FAIL
        setattr(r, setter, st)
        if st == FAIL:
            r.detail += f"[{name}] faltan/vacías {sorted(missing_cols | set(empties))}; "
    return r


def _g_warmup(inp: ReadinessInputs) -> GateResult:
    r = GateResult("warmup_calibrado_y_documentado")
    art = _load_json(inp.warmup_artifact)
    if art is None:
        r.cpu = r.gpu = FAIL
        r.detail = "sin warmup_calibration.json (F1-XDEV-002 no ejecutado sobre campañas reales)"
        return r
    # el artefacto debe declarar estado por kernel: medido / fallback / no apto
    per_kernel = art.get("per_kernel") or art.get("kernels")
    if not per_kernel:
        r.cpu = r.gpu = FAIL
        r.detail = "warmup artifact sin sección por kernel"
        return r
    bad = [k for k, v in per_kernel.items()
           if (v.get("status") if isinstance(v, dict) else None) not in
           ("measured", "documented_fallback")]
    r.cpu = r.gpu = (PASS if not bad else FAIL)
    if bad:
        r.detail = f"kernels sin warmup medido/documentado: {bad[:8]}"
    return r


def _g_roofline_calibration(cpu: pd.DataFrame | None, gpu: pd.DataFrame | None) -> GateResult:
    r = GateResult("calibracion_roofline_presente")
    if cpu is not None:
        ok = "i_ridge_used" in cpu.columns and cpu["i_ridge_used"].notna().any() and \
             cpu.get("training_quality_status", pd.Series(dtype=str)).eq("ok").any()
        # una fila 'ok' con i_ridge_used nulo == calibración ausente para ese nivel
        bad = ("i_ridge_used" not in cpu.columns) or \
              cpu.loc[cpu.get("training_quality_status", "") == "ok", "i_ridge_used"].isna().any() \
              if "training_quality_status" in cpu.columns else ("i_ridge_used" not in cpu.columns)
        r.cpu = PASS if (ok and not bad) else FAIL
    if gpu is not None:
        # F1-GPU-001: el ridge de GPU es por precisión y por frecuencia
        has_ref = "roofline_calibration_ref" in gpu.columns and gpu["roofline_calibration_ref"].notna().any()
        has_ridge = "i_ridge_used" in gpu.columns and gpu["i_ridge_used"].notna().any()
        r.gpu = PASS if (has_ref and has_ridge) else FAIL
        if not (has_ref and has_ridge):
            r.detail += "[gpu] falta ridge/precisión/frecuencia; "
    return r


def _g_label_source(cpu: pd.DataFrame | None, gpu: pd.DataFrame | None) -> GateResult:
    r = GateResult("etiqueta_no_de_hint_ni_proxy")
    for name, df, setter in (("cpu", cpu, "cpu"), ("gpu", gpu, "gpu")):
        if df is None:
            continue
        st = PASS
        # phase_label_hint nunca debe ser la etiqueta de entrenamiento
        if "phase_label_train" not in df.columns:
            st = FAIL
        elif "phase_label_hint" in df.columns and "phase_label_train" in df.columns:
            same = (df["phase_label_train"].astype(str) == df["phase_label_hint"].astype(str))
            # coincidencia perfecta 1:1 es sospechosa solo si hint no es nulo
            if df["phase_label_hint"].notna().all() and same.all() and len(df) > 5:
                st = FAIL
                r.detail += f"[{name}] phase_label_train == phase_label_hint en todas las filas; "
        # CPU: la etiqueta debe venir de uncore real, no del proxy
        if name == "cpu" and "phase_label_uncore_real" in df.columns and "phase_label_train" in df.columns:
            mism = (df["phase_label_train"].astype(str) != df["phase_label_uncore_real"].astype(str))
            elig = df.get("training_quality_status", "") == "ok"
            if (mism & elig).any():
                st = FAIL
                r.detail += "[cpu] phase_label_train difiere de phase_label_uncore_real en filas ok; "
        setattr(r, setter, st)
    return r


def _g_coverage(inp: ReadinessInputs, cpu: pd.DataFrame | None, gpu: pd.DataFrame | None) -> GateResult:
    r = GateResult("cobertura_suficiente_por_clase_y_familia")
    cov = _load_json(inp.coverage_report)
    MIN_FAMILIES = 5
    def check(df, dev):
        if df is None:
            return NA
        elig_col = "training_quality_status" if dev == "cpu" else "training_eligible"
        if elig_col not in df.columns:
            return BLOCKED
        elig = df[df[elig_col].astype(str).isin(["ok", "True", "true"])]
        if elig.empty or "phase_label_train" not in elig.columns:
            return FAIL
        fam_col = "kernel_family" if "kernel_family" in elig.columns else "kernel_ref"
        by_class = elig.groupby("phase_label_train")[fam_col].nunique()
        if set(by_class.index) < {"compute_bound", "memory_bound"}:
            return FAIL
        return PASS if by_class.min() >= MIN_FAMILIES else FAIL
    r.cpu = check(cpu, "cpu")
    r.gpu = check(gpu, "gpu")
    if cov is None and (r.cpu == BLOCKED or r.gpu == BLOCKED):
        r.detail = "sin phase_coverage_report.json y sin columnas de elegibilidad"
    return r


def _g_gpu_rows_not_independent(gpu: pd.DataFrame | None, contract: dict | None) -> GateResult:
    r = GateResult("filas_gpu_no_son_muestras_independientes")
    if gpu is None:
        return r
    # F1-GPU-003: el dataset GPU debe ser por corrida/fase, declarado en el contrato
    if contract is None:
        r.gpu = FAIL
        r.detail = "sin training_gpu_phases_contract.json"
        return r
    if contract.get("nvml_sample_is_independent_example") is True or contract.get("row_unit") not in ("run", "phase"):
        r.gpu = FAIL
        r.detail = "el contrato no declara granularidad por corrida/fase"
        return r
    # además: una fila por run_id (no varias)
    if "run_id" in gpu.columns and gpu["run_id"].duplicated().any():
        r.gpu = FAIL
        r.detail = "run_id duplicado en training_gpu_phases.csv"
        return r
    r.gpu = PASS
    return r


def _g_ncu_convergence(inp: ReadinessInputs, gpu: pd.DataFrame | None) -> GateResult:
    r = GateResult("candidatos_gpu_con_ncu_convergente")
    if gpu is None:
        return r
    if inp.ncu_reports_dir is None or not Path(inp.ncu_reports_dir).is_dir():
        r.gpu = BLOCKED
        r.detail = "F1-GPU-004: sin dir de reportes ncu (requiere ncu en paccaA100)"
        return r
    reports = {p.stem for p in Path(inp.ncu_reports_dir).glob("*.json")}
    kernels = set(gpu.get("kernel_ref", pd.Series(dtype=str)).dropna().unique())
    missing = sorted(kernels - reports)
    if missing:
        r.gpu = BLOCKED
        r.detail = f"sin reporte ncu convergente para: {missing[:8]}"
        return r
    non_conv = []
    for k in kernels:
        rep = _load_json(Path(inp.ncu_reports_dir) / f"{k}.json") or {}
        if not rep.get("converged"):
            non_conv.append(k)
    r.gpu = PASS if not non_conv else FAIL
    if non_conv:
        r.detail = f"ncu no convergente: {non_conv[:8]}"
    return r


def _g_freq_verified_under_load(cpu: pd.DataFrame | None, gpu: pd.DataFrame | None) -> GateResult:
    r = GateResult("frecuencia_verificada_bajo_carga")
    if cpu is not None:
        col = "frequency_quality_status"
        if col not in cpu.columns:
            r.cpu = FAIL
        else:
            elig = cpu.get("training_quality_status", "") == "ok"
            good = cpu.loc[elig, col].astype(str).isin(["valid", "not_applicable_native"]).all() \
                if elig.any() else False
            r.cpu = PASS if good else FAIL
    if gpu is not None:
        # No existe verificación por-ventana del reloj GPU bajo carga todavía
        # (F1-GPU-002 mide transición, no una traza de verificación por corrida).
        r.gpu = BLOCKED
        r.detail = "[gpu] sin traza de verificación de reloj GPU bajo carga por corrida"
    return r


def _g_quality_reported(inp: ReadinessInputs, cpu: pd.DataFrame | None, gpu: pd.DataFrame | None) -> GateResult:
    r = GateResult("calidad_y_rechazos_reportados")
    for name, df, setter, col in (
        ("cpu", cpu, "cpu", "training_quality_status"),
        ("gpu", gpu, "gpu", "phase_quality_status"),
    ):
        if df is None:
            continue
        reason_col = "training_quality_reason" if name == "cpu" else "phase_quality_reason"
        st = PASS if (col in df.columns and reason_col in df.columns) else FAIL
        setattr(r, setter, st)
        if st == FAIL:
            r.detail += f"[{name}] sin columnas de calidad/motivo; "
    return r


def _g_feature_contract_present(inp: ReadinessInputs) -> GateResult:
    r = GateResult("contrato_final_de_features_presente")
    for path, setter in ((inp.feature_contract_cpu, "cpu"), (inp.feature_contract_gpu, "gpu")):
        c = _load_json(path)
        if path is None:
            continue
        if c and c.get("schema", "").startswith("f1-xdev-004/frozen_feature_contract") and c.get("features"):
            setattr(r, setter, PASS)
        else:
            setattr(r, setter, FAIL)
    if inp.feature_contract_cpu is None:
        r.cpu = FAIL
        r.detail += "[cpu] contrato ausente; "
    if inp.feature_contract_gpu is None and inp.gpu_dataset is not None:
        r.gpu = FAIL
        r.detail += "[gpu] contrato ausente; "
    return r


def _g_no_leakage(inp: ReadinessInputs) -> GateResult:
    r = GateResult("sin_columnas_de_fuga_en_el_contrato")
    for path, setter, dev in ((inp.feature_contract_cpu, "cpu", "cpu"),
                              (inp.feature_contract_gpu, "gpu", "gpu")):
        c = _load_json(path)
        if not c:
            continue
        leak = set(c.get("features", [])) & ROOFLINE_TRUTH_COLUMNS
        setattr(r, setter, FAIL if leak else PASS)
        if leak:
            r.detail += f"[{dev}] fuga: {sorted(leak)}; "
    return r


def _g_corr_vif_present(inp: ReadinessInputs) -> GateResult:
    r = GateResult("analisis_pearson_spearman_vif_presente")
    for path, setter, needed in ((inp.feature_report_cpu, "cpu", inp.cpu_dataset),
                                 (inp.feature_report_gpu, "gpu", inp.gpu_dataset)):
        if needed is None:
            continue
        rep = _load_json(path)
        ok = bool(rep and "high_corr_pairs" in rep and "vif" in rep and rep.get("candidate_columns"))
        setattr(r, setter, PASS if ok else FAIL)
        if not ok:
            r.detail += f"[{setter}] falta feature_contract_<dev>.json (Pearson/Spearman/VIF); "
    return r


def _g_granularity_declared(inp: ReadinessInputs, cpu: pd.DataFrame | None, gpu: pd.DataFrame | None) -> GateResult:
    r = GateResult("granularidad_del_dataset_declarada")
    if cpu is not None:
        # F1-CPU-002: una fila == un intervalo uncore. Se declara por columnas.
        r.cpu = PASS if {"uncore_interval_id", "uncore_delta_t_ns"} <= set(cpu.columns) else FAIL
    if gpu is not None:
        c = _load_json(inp.gpu_contract_file)
        r.gpu = PASS if (c and c.get("row_unit") in ("run", "phase") and "granularity" in gpu.columns) else FAIL
    return r


ALL_GATES = [
    "checksums_y_procedencia",
    "warmup_calibrado_y_documentado",
    "calibracion_roofline_presente",
    "etiqueta_no_de_hint_ni_proxy",
    "cobertura_suficiente_por_clase_y_familia",
    "filas_gpu_no_son_muestras_independientes",
    "candidatos_gpu_con_ncu_convergente",
    "frecuencia_verificada_bajo_carga",
    "calidad_y_rechazos_reportados",
    "contrato_final_de_features_presente",
    "sin_columnas_de_fuga_en_el_contrato",
    "analisis_pearson_spearman_vif_presente",
    "granularidad_del_dataset_declarada",
]


def audit(inp: ReadinessInputs) -> dict:
    cpu = _load_csv(inp.cpu_dataset)
    gpu = _load_csv(inp.gpu_dataset)
    gpu_contract = _load_json(inp.gpu_contract_file)

    gates = [
        _g_provenance(cpu, gpu),
        _g_warmup(inp),
        _g_roofline_calibration(cpu, gpu),
        _g_label_source(cpu, gpu),
        _g_coverage(inp, cpu, gpu),
        _g_gpu_rows_not_independent(gpu, gpu_contract),
        _g_ncu_convergence(inp, gpu),
        _g_freq_verified_under_load(cpu, gpu),
        _g_quality_reported(inp, cpu, gpu),
        _g_feature_contract_present(inp),
        _g_no_leakage(inp),
        _g_corr_vif_present(inp),
        _g_granularity_declared(inp, cpu, gpu),
    ]
    worst = {g.gate: g.worst() for g in gates}
    cpu_ready = cpu is not None and all(
        g.cpu in (PASS, NA) for g in gates
    )
    gpu_ready = gpu is not None and all(
        g.gpu in (PASS, NA) for g in gates
    )
    return {
        "schema": "f1/pretraining_readiness/1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "has_cpu_dataset": cpu is not None,
        "has_gpu_dataset": gpu is not None,
        "gates": [asdict(g) for g in gates],
        "summary": {
            "n_fail": sum(1 for v in worst.values() if v == FAIL),
            "n_blocked": sum(1 for v in worst.values() if v == BLOCKED),
            "n_pass": sum(1 for v in worst.values() if v == PASS),
        },
        "cpu_ready_for_training": cpu_ready,
        "gpu_ready_for_training": gpu_ready,
    }


def _print_human(report: dict) -> None:
    print(f"{'GATE':<48}{'CPU':>9}{'GPU':>9}")
    print("-" * 66)
    for g in report["gates"]:
        print(f"{g['gate']:<48}{g['cpu']:>9}{g['gpu']:>9}"
              + (f"\n    {g['detail']}" if g["detail"] else ""))
    s = report["summary"]
    print("-" * 66)
    print(f"PASS={s['n_pass']}  FAIL={s['n_fail']}  BLOCKED={s['n_blocked']}")
    print(f"CPU listo para entrenamiento: {report['cpu_ready_for_training']}")
    print(f"GPU listo para entrenamiento: {report['gpu_ready_for_training']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cpu-dataset", type=Path)
    p.add_argument("--gpu-dataset", type=Path)
    p.add_argument("--gpu-contract-file", type=Path)
    p.add_argument("--feature-contract-cpu", type=Path)
    p.add_argument("--feature-contract-gpu", type=Path)
    p.add_argument("--feature-report-cpu", type=Path)
    p.add_argument("--feature-report-gpu", type=Path)
    p.add_argument("--ncu-reports-dir", type=Path)
    p.add_argument("--warmup-artifact", type=Path)
    p.add_argument("--coverage-report", type=Path)
    p.add_argument("--transition-aggregate", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    inp = ReadinessInputs(
        cpu_dataset=a.cpu_dataset, gpu_dataset=a.gpu_dataset,
        gpu_contract_file=a.gpu_contract_file,
        feature_contract_cpu=a.feature_contract_cpu, feature_contract_gpu=a.feature_contract_gpu,
        feature_report_cpu=a.feature_report_cpu, feature_report_gpu=a.feature_report_gpu,
        ncu_reports_dir=a.ncu_reports_dir, warmup_artifact=a.warmup_artifact,
        coverage_report=a.coverage_report, transition_aggregate=a.transition_aggregate,
    )
    report = audit(inp)
    _print_human(report)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nJSON: {a.out}")
    # rc != 0 si algo no está listo (útil para CI / gate de campaña)
    return 0 if (report["cpu_ready_for_training"] or report["gpu_ready_for_training"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
