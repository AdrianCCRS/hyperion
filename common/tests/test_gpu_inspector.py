from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.hpc import gpu_inspector


def _fake_run(stdout: str, *, returncode: int = 0):
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")
    return run


def test_arc171_active_processes_parsea_pids(monkeypatch):
    inspector = gpu_inspector.NvidiaSmiGpuInspector(gpu_index=0)
    monkeypatch.setattr(subprocess, "run", _fake_run("12345\n67890\n"))
    assert inspector.active_processes() == [12345, 67890]


def test_arc171_active_processes_vacio_sin_procesos(monkeypatch):
    inspector = gpu_inspector.NvidiaSmiGpuInspector(gpu_index=0)
    monkeypatch.setattr(subprocess, "run", _fake_run(""))
    assert inspector.active_processes() == []


def test_arc171_active_processes_vacio_si_nvidia_smi_falla(monkeypatch):
    inspector = gpu_inspector.NvidiaSmiGpuInspector(gpu_index=0)
    monkeypatch.setattr(subprocess, "run", _fake_run("no debería usarse", returncode=1))
    assert inspector.active_processes() == []


def test_arc171_active_processes_vacio_si_nvidia_smi_no_existe(monkeypatch):
    inspector = gpu_inspector.NvidiaSmiGpuInspector(gpu_index=0)

    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", raise_missing)
    assert inspector.active_processes() == []


@pytest.mark.parametrize("raw,expected", [("Enabled\n", True), ("Disabled\n", False)])
def test_arc171_persistence_mode_interpreta_enabled_disabled(monkeypatch, raw, expected):
    inspector = gpu_inspector.NvidiaSmiGpuInspector(gpu_index=0)
    monkeypatch.setattr(subprocess, "run", _fake_run(raw))
    assert inspector.persistence_mode() is expected


def test_arc171_persistence_mode_none_si_consulta_falla(monkeypatch):
    inspector = gpu_inspector.NvidiaSmiGpuInspector(gpu_index=0)
    monkeypatch.setattr(subprocess, "run", _fake_run("", returncode=1))
    assert inspector.persistence_mode() is None


def test_arc171_mig_configuration_devuelve_el_valor_crudo(monkeypatch):
    inspector = gpu_inspector.NvidiaSmiGpuInspector(gpu_index=0)
    monkeypatch.setattr(subprocess, "run", _fake_run("Disabled\n"))
    assert inspector.mig_configuration() == "Disabled"


def test_arc171_mig_configuration_none_si_consulta_falla(monkeypatch):
    inspector = gpu_inspector.NvidiaSmiGpuInspector(gpu_index=0)
    monkeypatch.setattr(subprocess, "run", _fake_run("", returncode=1))
    assert inspector.mig_configuration() is None


def test_arc171_gpu_index_se_resuelve_via_gpu_freqctl(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    inspector = gpu_inspector.NvidiaSmiGpuInspector()
    captured = {}

    def run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    inspector.active_processes()
    assert captured["args"][2] == "3"
