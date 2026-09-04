"""F1-XDEV-004: pruebas del análisis de correlación/VIF y contrato de features.

Fixtures sintéticos -- la selección definitiva se hace sobre el dataset real.
Verifican: filas elegibles, detección de pares muy correlados, VIF, exclusión
dura de columnas de verdad Roofline, manejo de NaN/constantes/inf, y el
congelado del contrato.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase2_clasificador.analysis import feature_contract as fc


def _cpu_frame(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ipc = rng.uniform(0.2, 3.0, n)
    cache_miss_rate = rng.uniform(0.0, 1.0, n)
    frame = pd.DataFrame({
        "ipc": ipc,
        "cache_miss_rate": cache_miss_rate,
        # mpki fuertemente colineal con cache_miss_rate por construcción
        "mpki": cache_miss_rate * 1000.0 + rng.normal(0, 1.0, n),
        "stall_mem_ratio": rng.uniform(0, 1, n),
        "ips": rng.uniform(1e8, 1e10, n),
        "running_ratio": rng.uniform(0.8, 1.0, n),
        "freq_khz_observed": rng.choice([800000, 2000000, 3600000], n),
        # columnas de verdad Roofline: NUNCA deben salir como feature
        "operational_intensity_uncore_real": ipc * 2.0 + rng.normal(0, 0.1, n),
        "i_ridge_used": np.full(n, 3.3),
        "phase_label_train": np.where(ipc < 1.2, "memory_bound", "compute_bound"),
        "training_quality_status": "ok",
    })
    return frame


def test_pares_muy_correlados_detectados_y_se_prefiere_la_medicion_directa():
    df = _cpu_frame()
    rep = fc.analyse(df, "cpu", corr_threshold=0.85)
    pares = {(p["a"], p["b"]) for p in rep.high_corr_pairs}
    assert ("cache_miss_rate", "mpki") in pares or ("mpki", "cache_miss_rate") in pares
    par = next(p for p in rep.high_corr_pairs if set((p["a"], p["b"])) == {"mpki", "cache_miss_rate"})
    # cache_miss_rate es menos indirecto que mpki -> se conserva
    assert par["keep"] == "cache_miss_rate"
    assert par["drop"] == "mpki"


def test_columnas_de_verdad_roofline_nunca_entran_como_feature():
    df = _cpu_frame()
    rep = fc.analyse(df, "cpu")
    assert "operational_intensity_uncore_real" not in rep.recommended_feature_set
    assert "i_ridge_used" not in rep.recommended_feature_set
    assert "phase_label_train" not in rep.recommended_feature_set
    assert set(rep.roofline_truth_columns_seen) >= {
        "operational_intensity_uncore_real", "i_ridge_used", "phase_label_train"
    }
    # tampoco se listan como candidatas numéricas
    assert "operational_intensity_uncore_real" not in rep.candidate_columns


def test_columna_constante_se_marca_no_elegible_y_vif_no_explota():
    df = _cpu_frame()
    df["const_col"] = 7.0
    rep = fc.analyse(df, "cpu")
    diag = {c.name: c for c in rep.column_diagnosis}
    assert diag["const_col"].is_constant
    assert not diag["const_col"].eligible_as_feature
    assert diag["const_col"].reason_excluded == "constant"
    assert "const_col" not in rep.recommended_feature_set


def test_columna_mayormente_ausente_se_excluye():
    df = _cpu_frame()
    col = np.full(len(df), np.nan)
    col[:10] = 1.0
    df["mostly_missing"] = col
    rep = fc.analyse(df, "cpu")
    diag = {c.name: c for c in rep.column_diagnosis}
    assert diag["mostly_missing"].reason_excluded == "over_half_missing"


def test_infinitos_se_cuentan_y_no_rompen_correlacion():
    df = _cpu_frame()
    df.loc[0, "ips"] = np.inf
    rep = fc.analyse(df, "cpu")
    diag = {c.name: c for c in rep.column_diagnosis}
    assert diag["ips"].n_inf == 1


def test_vif_alto_recomienda_descartar():
    df = _cpu_frame()
    # feature redundante casi perfecta -> VIF enorme
    df["ipc_copia"] = df["ipc"] * 2.0 + 1e-9
    rep = fc.analyse(df, "cpu", corr_threshold=0.999, vif_threshold=10.0)
    drops = {d["column"]: d for d in rep.recommended_drops}
    assert any(d["reason"] in ("vif_over_threshold", "high_corr")
               for d in rep.recommended_drops if d["column"] in ("ipc_copia", "ipc"))


def test_freeze_rechaza_fuga_y_columnas_no_elegibles(tmp_path):
    df = _cpu_frame()
    df["const_col"] = 1.0
    rep = fc.analyse(df, "cpu")
    with pytest.raises(ValueError, match="fuga"):
        fc.freeze_contract(rep, ["ipc", "i_ridge_used"], tmp_path / "c.json")
    with pytest.raises(ValueError, match="no elegibles"):
        fc.freeze_contract(rep, ["ipc", "const_col"], tmp_path / "c.json")
    ok = fc.freeze_contract(rep, ["ipc", "stall_mem_ratio", "freq_khz_observed"],
                            tmp_path / "c.json")
    assert ok["features"] == ["ipc", "stall_mem_ratio", "freq_khz_observed"]
    assert (tmp_path / "c.json").exists()


def test_device_gpu_usa_su_propia_columna_de_calidad():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "gpu_util_pct_median": rng.uniform(50, 100, n),
        "gpu_power_mw_median": rng.uniform(1e5, 3e5, n),
        "gpu_sm_clock_mhz_median": rng.choice([1200, 1410], n),
        "phase_label_train": rng.choice(["memory_bound", "compute_bound"], n),
        "phase_quality_status": ["ok"] * (n - 20) + ["rejected"] * 20,
        "gpu_operational_intensity": rng.uniform(1, 50, n),  # verdad -> prohibido
    })
    rep = fc.analyse(df, "gpu")
    assert rep.n_rows_eligible == n - 20
    assert "gpu_operational_intensity" not in rep.recommended_feature_set
    assert "gpu_operational_intensity" not in rep.candidate_columns


def test_device_invalido_falla():
    with pytest.raises(ValueError, match="cpu.*gpu|gpu.*cpu"):
        fc.analyse(_cpu_frame(), "tpu")


def test_cero_filas_elegibles_no_rompe():
    df = _cpu_frame()
    df["training_quality_status"] = "rejected"
    rep = fc.analyse(df, "cpu")
    assert rep.n_rows_eligible == 0
    assert rep.recommended_feature_set == []
    assert any("cero filas" in n for n in rep.notes)


def test_main_cli_escribe_artefactos(tmp_path):
    df = _cpu_frame()
    csv = tmp_path / "training_cpu_intervals.csv"
    df.to_csv(csv, index=False)
    rc = fc.main([str(csv), "--device", "cpu", "--out-dir", str(tmp_path / "out")])
    assert rc == 0
    assert (tmp_path / "out" / "feature_contract_cpu.json").exists()
    assert (tmp_path / "out" / "feature_contract_cpu_pairs.csv").exists()
