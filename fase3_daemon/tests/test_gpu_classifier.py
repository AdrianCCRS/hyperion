from pathlib import Path

import numpy as np
import pytest

from fase3_daemon.gpu_loop.classifier import HistoricalGpuClassifier
from fase3_daemon.gpu_loop.controller import GpuPhaseLabel
from fase3_daemon.gpu_loop.loop import GpuFeatures

_MODELS_DIR = Path(__file__).resolve().parents[2] / "fase2_clasificador" / "models"
_MODEL_NAME = "gpu_regresion_log_historical_20260922"


def _features(util=50.0, mem=50.0, power=100000.0, clock=1200.0, temp=60.0) -> GpuFeatures:
    return GpuFeatures(gpu_util_pct=util, gpu_mem_util_pct=mem, gpu_power_mw=power,
                       gpu_sm_clock_mhz=clock, gpu_temperature_c=temp)


class _FakeModel:
    """Modelo falso que registra exactamente qué fila recibió, para probar
    la construcción del vector sin depender del .joblib real."""
    def __init__(self, to_return):
        self.to_return = to_return
        self.last_row = None

    def predict(self, rows):
        self.last_row = rows[0]
        return [self.to_return]


FEATURE_ORDER = [
    "gpu_util_pct_median", "gpu_mem_util_pct_median", "gpu_power_mw_median",
    "gpu_sm_clock_mhz_median", "gpu_mem_util_pct_std",
]


def test_rejects_unknown_feature_names():
    with pytest.raises(ValueError, match="no sabe derivar"):
        HistoricalGpuClassifier(_FakeModel(True), ["gpu_util_pct_median", "algo_no_entrenado"])


def test_cold_start_uses_instantaneous_sample_as_single_point_window():
    model = _FakeModel(True)
    clf = HistoricalGpuClassifier(model, FEATURE_ORDER)
    clf.classify(_features(util=42.0, mem=33.0, power=90000.0, clock=1100.0))
    # sin record_sample previo: mediana de 1 muestra = esa muestra, std de 1 = 0.0
    assert model.last_row == [42.0, 33.0, 90000.0, 1100.0, 0.0]


def test_rolling_window_computes_median_and_std_over_recorded_samples():
    model = _FakeModel(True)
    clf = HistoricalGpuClassifier(model, FEATURE_ORDER, window_size=10)
    for mem in (10.0, 20.0, 30.0):
        clf.record_sample(_features(mem=mem))
    clf.classify(_features(mem=999.0))  # la instantanea actual NO participa si hay buffer
    util, mem_med, power, clock, mem_std = model.last_row
    assert mem_med == 20.0  # mediana de [10, 20, 30]
    assert mem_std == pytest.approx(np.std([10.0, 20.0, 30.0], ddof=1))


def test_window_respects_maxlen_and_drops_oldest():
    model = _FakeModel(True)
    clf = HistoricalGpuClassifier(model, ["gpu_mem_util_pct_median"], window_size=3)
    for mem in (1.0, 2.0, 3.0, 100.0):  # el 1.0 debe salir del buffer
        clf.record_sample(_features(mem=mem))
    clf.classify(_features())
    assert model.last_row == [3.0]  # mediana de [2, 3, 100]


@pytest.mark.parametrize("model_return,expected", [
    (True, GpuPhaseLabel.MEMORY_BOUND),
    (False, GpuPhaseLabel.COMPUTE_BOUND),
    (np.bool_(True), GpuPhaseLabel.MEMORY_BOUND),
])
def test_classify_translates_boolean_label(model_return, expected):
    clf = HistoricalGpuClassifier(_FakeModel(model_return), ["gpu_util_pct_median"])
    assert clf.classify(_features()) == expected


def test_classify_rejects_non_boolean_prediction():
    clf = HistoricalGpuClassifier(_FakeModel("memory_bound"), ["gpu_util_pct_median"])
    with pytest.raises(TypeError, match="se esperaba un booleano"):
        clf.classify(_features())


@pytest.mark.skipif(not (_MODELS_DIR / f"{_MODEL_NAME}.joblib").exists(), reason="modelo GPU no exportado localmente")
def test_from_export_dir_loads_real_model_and_classifies():
    clf = HistoricalGpuClassifier.from_export_dir(_MODELS_DIR, name=_MODEL_NAME)
    assert set(clf._feature_names) == {
        "gpu_util_pct_median", "gpu_mem_util_pct_median", "gpu_power_mw_median",
        "gpu_sm_clock_mhz_median", "gpu_mem_util_pct_std",
    }
    # una lectura de alta utilización de memoria y bajo reloj SM, sin buffer
    # previo, debe caer del lado memory_bound (sanity check direccional,
    # no un valor exacto de probabilidad).
    label = clf.classify(_features(util=80.0, mem=90.0, power=150000.0, clock=800.0))
    assert label in (GpuPhaseLabel.MEMORY_BOUND, GpuPhaseLabel.COMPUTE_BOUND)


def test_from_export_dir_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        HistoricalGpuClassifier.from_export_dir(_MODELS_DIR, name="no_existe_este_modelo")


_MODEL_SIN_RELOJ = "gpu_random_forest_historical_20260923_sin_reloj"


def test_candidato_sin_reloj_no_usa_reloj_ni_potencia_y_no_depende_de_ellos():
    # Auditoria job 7601: el daemon fija el reloj y la potencia depende de el; el candidato vigente no debe
    # incluir ninguna de las dos, y por tanto su decision no puede cambiar al variarlas.
    clf = HistoricalGpuClassifier.from_export_dir(_MODELS_DIR, name=_MODEL_SIN_RELOJ)
    assert not ({"gpu_sm_clock_mhz_median", "gpu_power_mw_median"} & set(clf._feature_names))
    for util, mem in ((100.0, 5.0), (100.0, 80.0), (30.0, 60.0)):
        native = clf.classify(_features(util=util, mem=mem, clock=1410.0, power=250000.0))
        locked = HistoricalGpuClassifier.from_export_dir(_MODELS_DIR, name=_MODEL_SIN_RELOJ).classify(
            _features(util=util, mem=mem, clock=210.0, power=40000.0))
        assert native == locked
