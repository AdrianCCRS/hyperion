"""Brecha H: pruebas de la auditoría de readiness pre-entrenamiento.

Verifican que: un dataset incompleto NO pasa; que los gates que necesitan
hardware quedan BLOCKED (no PASS con fixture); y que un bundle completo y
coherente pasa.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase2_clasificador.analysis import pretraining_readiness as pr


# ---------------------------------------------------------------- fixtures

def _cpu_dataset(path: Path, *, families=("gemm", "fft", "cg", "lu", "sp", "mg", "bt")):
    rows = []
    for fam in families:
        for cls in ("compute_bound", "memory_bound"):
            for i in range(4):
                rows.append({
                    "run_id": f"{fam}_{cls}_{i}", "kernel_ref": fam, "kernel_family": fam,
                    "binary_checksum": "sha256:x",
                    "uncore_interval_id": i, "uncore_delta_t_ns": 10_000_000,
                    "training_quality_status": "ok", "training_quality_reason": "",
                    "frequency_quality_status": "valid",
                    "phase_label_train": cls, "phase_label_uncore_real": cls,
                    "i_ridge_used": 3.3,
                    "ipc": 1.0, "mpki": 5.0, "cache_miss_rate": 0.1,
                })
    pd.DataFrame(rows).to_csv(path, index=False)


def _warmup_artifact(path: Path, kernels):
    path.write_text(json.dumps({
        "schema": "f1-xdev-002/warmup_calibration/1",
        "per_kernel": {k: {"status": "measured", "warmup_seconds": 2.0} for k in kernels},
    }))


def _feature_contract(path: Path, device, features):
    path.write_text(json.dumps({
        "schema": "f1-xdev-004/frozen_feature_contract/1",
        "device": device, "features": features,
    }))


def _feature_report(path: Path):
    path.write_text(json.dumps({
        "schema": "f1-xdev-004/feature_contract/1",
        "candidate_columns": ["ipc", "mpki", "cache_miss_rate"],
        "high_corr_pairs": [], "vif": {"ipc": 1.1},
    }))


# ---------------------------------------------------------------- tests

def test_sin_artefactos_nada_esta_listo(tmp_path):
    cpu = tmp_path / "training_cpu_intervals.csv"
    _cpu_dataset(cpu)
    report = pr.audit(pr.ReadinessInputs(cpu_dataset=cpu))
    assert report["cpu_ready_for_training"] is False
    gates = {g["gate"]: g for g in report["gates"]}
    # contrato ausente -> FAIL
    assert gates["contrato_final_de_features_presente"]["cpu"] == pr.FAIL
    # warmup ausente -> FAIL
    assert gates["warmup_calibrado_y_documentado"]["cpu"] == pr.FAIL
    # Pearson/VIF ausente -> FAIL
    assert gates["analisis_pearson_spearman_vif_presente"]["cpu"] == pr.FAIL


def test_gpu_sin_ncu_queda_blocked_no_pass(tmp_path):
    gpu = tmp_path / "training_gpu_phases.csv"
    pd.DataFrame([{
        "run_id": "r1", "kernel_ref": "rodinia_lud", "kernel_family": "lud",
        "binary_checksum": "sha256:x", "granularity": "run",
        "roofline_calibration_ref": "/cal/gpu_fp64.json", "i_ridge_used": 3.3,
        "phase_label_train": "compute_bound",
        "phase_quality_status": "ok", "phase_quality_reason": "", "training_eligible": True,
    }]).to_csv(gpu, index=False)
    contract = tmp_path / "c.json"
    contract.write_text(json.dumps({
        "schema": "f1-gpu-003/gpu_phase_granularity_contract/1",
        "row_unit": "run", "nvml_sample_is_independent_example": False,
    }))
    report = pr.audit(pr.ReadinessInputs(gpu_dataset=gpu, gpu_contract_file=contract))
    gates = {g["gate"]: g for g in report["gates"]}
    assert gates["candidatos_gpu_con_ncu_convergente"]["gpu"] == pr.BLOCKED
    assert gates["frecuencia_verificada_bajo_carga"]["gpu"] == pr.BLOCKED
    assert report["gpu_ready_for_training"] is False


def test_deteccion_de_fuga_en_el_contrato(tmp_path):
    cpu = tmp_path / "training_cpu_intervals.csv"
    _cpu_dataset(cpu)
    fc = tmp_path / "frozen_cpu.json"
    _feature_contract(fc, "cpu", ["ipc", "i_ridge_used"])  # <- fuga
    report = pr.audit(pr.ReadinessInputs(cpu_dataset=cpu, feature_contract_cpu=fc))
    gates = {g["gate"]: g for g in report["gates"]}
    assert gates["sin_columnas_de_fuga_en_el_contrato"]["cpu"] == pr.FAIL


def test_etiqueta_igual_al_hint_falla(tmp_path):
    cpu = tmp_path / "training_cpu_intervals.csv"
    _cpu_dataset(cpu)
    df = pd.read_csv(cpu)
    df["phase_label_hint"] = df["phase_label_train"]  # hint == train en todas
    df.to_csv(cpu, index=False)
    report = pr.audit(pr.ReadinessInputs(cpu_dataset=cpu))
    gates = {g["gate"]: g for g in report["gates"]}
    assert gates["etiqueta_no_de_hint_ni_proxy"]["cpu"] == pr.FAIL


def test_cobertura_insuficiente_por_familia_falla(tmp_path):
    cpu = tmp_path / "training_cpu_intervals.csv"
    _cpu_dataset(cpu, families=("gemm", "fft"))  # solo 2 familias por clase < 5
    report = pr.audit(pr.ReadinessInputs(cpu_dataset=cpu))
    gates = {g["gate"]: g for g in report["gates"]}
    assert gates["cobertura_suficiente_por_clase_y_familia"]["cpu"] == pr.FAIL


def test_bundle_cpu_completo_y_coherente_pasa(tmp_path):
    cpu = tmp_path / "training_cpu_intervals.csv"
    _cpu_dataset(cpu)
    kernels = list(pd.read_csv(cpu)["kernel_ref"].unique())
    warm = tmp_path / "warmup_calibration.json"
    _warmup_artifact(warm, kernels)
    fc = tmp_path / "frozen_cpu.json"
    _feature_contract(fc, "cpu", ["ipc", "mpki", "cache_miss_rate", "stall_mem_ratio"])
    frep = tmp_path / "feature_contract_cpu.json"
    _feature_report(frep)

    report = pr.audit(pr.ReadinessInputs(
        cpu_dataset=cpu, warmup_artifact=warm,
        feature_contract_cpu=fc, feature_report_cpu=frep,
    ))
    pr._print_human(report)  # smoke de la salida humana
    failing = [g for g in report["gates"] if g["cpu"] in (pr.FAIL, pr.BLOCKED)]
    assert not failing, failing
    assert report["cpu_ready_for_training"] is True


def test_main_cli_rc_y_json(tmp_path):
    cpu = tmp_path / "training_cpu_intervals.csv"
    _cpu_dataset(cpu, families=("gemm", "fft"))
    out = tmp_path / "readiness.json"
    rc = pr.main(["--cpu-dataset", str(cpu), "--out", str(out)])
    assert rc == 1  # no está listo
    assert out.exists()
    j = json.loads(out.read_text())
    assert j["schema"] == "f1/pretraining_readiness/1"
    assert len(j["gates"]) == len(pr.ALL_GATES)
