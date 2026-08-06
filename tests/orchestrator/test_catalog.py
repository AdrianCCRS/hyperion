from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import catalog


def _dataset_kwargs(**overrides):
    base = dict(
        id="npb_ep", suite="NPB", role="dataset", exec_path="bin/ep.B.x",
        binary_checksum="sha256:abc", phase_label_hint="compute_bound", size_variant="B",
        expected_runtime_seconds=6, warmup_seconds=1.0, success_check={"type": "exit_code"},
        estimated_memory_bytes=1,
    )
    base.update(overrides)
    return base


def test_cat05_flops_rate_stdout_pattern_valido():
    entry = catalog.KernelEntry(
        **_dataset_kwargs(
            flops_rate_stdout_pattern=r"Mop/s total\s*=\s*([0-9.]+)",
            runtime_seconds_stdout_pattern=r"Time in seconds\s*=\s*([0-9.]+)",
        )
    )
    assert entry.flops_rate_stdout_pattern is not None
    assert entry.runtime_seconds_stdout_pattern is not None


def test_cat05_flops_rate_stdout_pattern_sin_grupo_de_captura_falla():
    with pytest.raises(ValueError, match="CAT-05"):
        catalog.KernelEntry(**_dataset_kwargs(flops_rate_stdout_pattern=r"Mop/s total sin grupo"))


def test_cat05_runtime_seconds_stdout_pattern_regex_invalido_falla():
    with pytest.raises(ValueError, match="CAT-05"):
        catalog.KernelEntry(**_dataset_kwargs(runtime_seconds_stdout_pattern="["))


def test_arc70_device_default_es_cpu():
    entry = catalog.KernelEntry(**_dataset_kwargs())
    assert entry.device == "cpu"


def test_arc70_device_gpu_valido():
    entry = catalog.KernelEntry(**_dataset_kwargs(device="gpu"))
    assert entry.device == "gpu"


def test_arc70_device_invalido_falla():
    with pytest.raises(ValueError, match="CAT-09"):
        catalog.KernelEntry(**_dataset_kwargs(device="tpu"))


def test_load_catalog_lee_flops_rate_y_runtime_seconds(tmp_path):
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "kernels": [
                    {
                        **_dataset_kwargs(),
                        "flops_rate_stdout_pattern": r"Mop/s total\s*=\s*([0-9.]+)",
                        "runtime_seconds_stdout_pattern": r"Time in seconds\s*=\s*([0-9.]+)",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    entries = catalog.load_catalog(str(catalog_path))
    assert entries["npb_ep"].flops_rate_stdout_pattern == r"Mop/s total\s*=\s*([0-9.]+)"
    assert entries["npb_ep"].runtime_seconds_stdout_pattern == r"Time in seconds\s*=\s*([0-9.]+)"


def test_arc70_load_catalog_lee_device_gpu(tmp_path):
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        yaml.safe_dump({"kernels": [{**_dataset_kwargs(), "device": "gpu"}]}),
        encoding="utf-8",
    )
    entries = catalog.load_catalog(str(catalog_path))
    assert entries["npb_ep"].device == "gpu"


def test_arc42_multiplicadores_de_unidad_default_a_uno():
    entry = catalog.KernelEntry(**_dataset_kwargs())
    assert entry.bandwidth_stdout_unit_multiplier == 1.0
    assert entry.flops_stdout_unit_multiplier == 1.0


def test_arc42_load_catalog_lee_multiplicadores_de_unidad(tmp_path):
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        yaml.safe_dump({
            "kernels": [{
                **_dataset_kwargs(id="stream_official", role="calibration", phase_label_hint=None,
                                   size_variant=None, expected_runtime_seconds=None, warmup_seconds=None,
                                   estimated_memory_bytes=None, reports_bandwidth_stdout=True,
                                   bandwidth_stdout_pattern=r"Triad:\s+([0-9.]+)"),
                "bandwidth_stdout_unit_multiplier": 1_000_000,
            }]
        }),
        encoding="utf-8",
    )
    entries = catalog.load_catalog(str(catalog_path))
    assert entries["stream_official"].bandwidth_stdout_unit_multiplier == 1_000_000
