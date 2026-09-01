"""Test de integración de run_evaluation.py contra windows.csv reales en
disco (no mockeados) -- verifica el glob de --scenario, la detección de
escenarios sin datos, y que el reporte final se escribe a disco."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


def _write_scenario(base_dir: Path, scenario: str, pkg_uj_base: int) -> None:
    rng = np.random.default_rng(hash(scenario) % (2**32))
    for kernel in ["k1", "k2", "k3", "k4", "k5", "k6"]:
        run_dir = base_dir / scenario / kernel
        run_dir.mkdir(parents=True)
        rows = [{
            "kernel_ref": kernel, "phase_label_train": "compute_bound", "quality_status": "ok",
            "frequency_quality_status": "valid",
            "delta_t_ns": 1_000_000_000, "pkg_delta_uj": int(pkg_uj_base * (1 + rng.normal(0, 0.03))),
            "dram_delta_uj": 0, "energy_valid": True, "gpu_energy_delta_mj": None,
        } for _ in range(3)]
        pd.DataFrame(rows).to_csv(run_dir / "windows.csv", index=False)


def test_run_evaluation_end_to_end_con_archivos_reales(tmp_path):
    _write_scenario(tmp_path, "agente", pkg_uj_base=500_000)
    _write_scenario(tmp_path, "performance", pkg_uj_base=2_000_000)

    output = tmp_path / "reporte.txt"
    result = subprocess.run(
        [
            sys.executable, str(_REPO_ROOT / "fase4_evaluacion" / "run_evaluation.py"),
            "--scenario", "agente", str(tmp_path / "agente" / "*" / "windows.csv"),
            "--scenario", "performance", str(tmp_path / "performance" / "*" / "windows.csv"),
            "--agent-scenario", "agente",
            "--devices", "cpu", "--labels", "compute_bound",
            "--output", str(output),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "performance" in result.stdout
    assert output.exists()
    assert "compute_bound" in output.read_text()


def test_run_evaluation_escenario_sin_datos_se_omite_con_aviso(tmp_path):
    _write_scenario(tmp_path, "agente", pkg_uj_base=500_000)
    _write_scenario(tmp_path, "performance", pkg_uj_base=2_000_000)

    result = subprocess.run(
        [
            sys.executable, str(_REPO_ROOT / "fase4_evaluacion" / "run_evaluation.py"),
            "--scenario", "agente", str(tmp_path / "agente" / "*" / "windows.csv"),
            "--scenario", "performance", str(tmp_path / "performance" / "*" / "windows.csv"),
            "--scenario", "ondemand", str(tmp_path / "no_existe" / "*" / "windows.csv"),
            "--agent-scenario", "agente",
            "--devices", "cpu", "--labels", "compute_bound",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "ondemand" in result.stderr  # aviso de que se omitió
    assert "ondemand" not in result.stdout  # pero no aparece en la tabla real


def test_run_evaluation_sin_datos_del_agente_falla_con_mensaje_claro(tmp_path):
    _write_scenario(tmp_path, "performance", pkg_uj_base=2_000_000)

    result = subprocess.run(
        [
            sys.executable, str(_REPO_ROOT / "fase4_evaluacion" / "run_evaluation.py"),
            "--scenario", "performance", str(tmp_path / "performance" / "*" / "windows.csv"),
            "--agent-scenario", "agente",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "agente" in result.stderr
