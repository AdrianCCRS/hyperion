"""Verifica que el candidato CPU exportado a ONNX se comporte igual que el
.joblib original -- no repite la verificación completa de 1.17M filas
(vive en tmp/, sin versionar; ver el comando real en el docstring de
export_onnx.py), pero cubre el mismo camino de código con filas
sintéticas dentro de rangos plausibles de las 6 variables."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("skl2onnx")
pytest.importorskip("onnxmltools")

from fase3_daemon.cpu_loop.export_onnx import convert_xgb_classifier_to_onnx, verify_row_by_row

_MODEL_PATH = Path(__file__).resolve().parents[2] / "fase2_clasificador" / "models" / "xgboost_cpu.joblib"

pytestmark = pytest.mark.skipif(not _MODEL_PATH.exists(), reason="xgboost_cpu.joblib no exportado localmente")


@pytest.fixture(scope="module")
def cpu_model():
    import joblib
    return joblib.load(_MODEL_PATH)


def _synthetic_rows(n=500, seed=20260918) -> np.ndarray:
    # rangos plausibles de ipc, mpki, cache_miss_rate, stall_mem_ratio, ips, freq_khz_observed
    rng = np.random.default_rng(seed)
    ipc = rng.uniform(0.1, 4.0, n)
    mpki = rng.uniform(0.0, 100.0, n)
    cache_miss_rate = rng.uniform(0.0, 1.0, n)
    stall_mem_ratio = rng.uniform(0.0, 1.0, n)
    ips = rng.uniform(1e8, 1e10, n)
    freq_khz = rng.uniform(800_000, 3_600_000, n)
    return np.column_stack([ipc, mpki, cache_miss_rate, stall_mem_ratio, ips, freq_khz]).astype(np.float32)


def test_onnx_matches_sklearn_row_by_row(cpu_model, tmp_path):
    onnx_model = convert_xgb_classifier_to_onnx(cpu_model, feature_count=6)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())

    X = _synthetic_rows()
    summary = verify_row_by_row(cpu_model, onnx_path, X, tol=1e-4)
    assert summary["n_rows"] == 500
    assert summary["label_mismatches"] == 0
    assert summary["max_diff_proba"] < 1e-4


def test_verify_row_by_row_raises_on_tight_tolerance_mismatch(cpu_model, tmp_path):
    onnx_model = convert_xgb_classifier_to_onnx(cpu_model, feature_count=6)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())

    X = _synthetic_rows(n=50)
    # tolerancia absurdamente estricta: cualquier ruido de punto flotante entre
    # el backend float64 de sklearn y el float32 de ONNX Runtime debe hacerla fallar,
    # confirmando que la funcion SI detecta diferencias y no siempre pasa.
    with pytest.raises(AssertionError, match="verificación ONNX falló"):
        verify_row_by_row(cpu_model, onnx_path, X, tol=0.0)
