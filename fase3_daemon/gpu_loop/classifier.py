"""Clasificador de fase de GPU real (§4.3 punto 6 del plan de
realineación), primera implementación que deja de ser un placeholder.

## El desajuste que este módulo resuelve, y cómo

El candidato congelado en Fase 2 (`fase2_clasificador/models/
gpu_regresion_log_historical_20260922.joblib`) se entrenó sobre CINCO
variables agregadas por CORRIDA COMPLETA: la mediana de cuatro señales
NVML y la desviación estándar intracorrida de la utilización de memoria
(`gpu_mem_util_pct_std`) -- esta última se incorporó explícitamente porque
distingue kernels de lanzamientos cortos y repetidos (p.ej. `kmeans`) cuyas
medianas por sí solas se confunden con una carga poco ocupada (ver
`docs/libro/secciones/02_metodologia.tex`, §metodologia-modelo-gpu).

`gpu_loop/loop.py::query_gpu_features()` da, en cambio, UNA instantánea
NVML en el momento en que `activity_poller` detecta el cruce de actividad.
Alimentar esa instantánea directamente como si fuera la mediana de la
corrida es una aproximación razonable (la mediana de una muestra es esa
muestra), pero `gpu_mem_util_pct_std` de una sola muestra es siempre CERO
-- exactamente la variable que se añadió porque el resto no alcanza, y
justo la que quedaría sistemáticamente mal.

Este módulo mantiene, en cambio, un BUFFER MÓVIL de las últimas
`window_size` muestras que `activity_poller.poll_phase_events()` ya sondea
de todas formas (vía `on_sample`, sección nueva de ese módulo), y calcula
mediana/desviación estándar sobre ese buffer en el instante de clasificar.

**Esto NO reproduce exactamente la definición de entrenamiento.** La
mediana/std de entrenamiento resume una corrida ya terminada (todas sus
muestras); la de este módulo resume una ventana causal de las últimas
`window_size` muestras ANTERIORES al instante de clasificar, que en el
peor caso (arranque en frío, fase muy corta) puede tener muy pocos
elementos. Es la misma clase de aproximación causal que ya hace CPU con
`f_obs` en Fase 2 (la mediana de las lecturas de frecuencia cubiertas por
un intervalo, no de la corrida completa) -- una ventana finita en vez de
la agregación retrospectiva completa. Se documenta aquí porque, a
diferencia de CPU, no hay todavía ninguna medición que diga cuánto cuesta
en exactitud: es una limitación conocida y explícita, no una que deba
descubrirse leyendo el código.

`window_size` por defecto (20 muestras a 50~ms de sondeo = 1~s de
historia) es un valor de arranque razonable, no calibrado -- calibrarlo
contra corridas reales queda para cuando exista una campaña de validación
del propio daemon (Fase 4).
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fase3_daemon.gpu_loop.controller import GpuPhaseLabel  # noqa: E402
from fase3_daemon.gpu_loop.loop import GpuFeatures  # noqa: E402

DEFAULT_WINDOW_SIZE = 200  # ~10 s a DEFAULT_POLL_INTERVAL_S (50 ms); tras reset_window() solo hay muestras de la fase

# Nombre de columna de entrenamiento -> cómo derivarla del buffer de GpuFeatures.
# Debe cubrir exactamente metadata["features"] del modelo cargado, ni más ni menos
# -- ver _build_feature_vector(), que revienta si algo no coincide en vez de
# rellenar en silencio.
_MEDIAN_FIELDS = {
    "gpu_util_pct_median": "gpu_util_pct",
    "gpu_mem_util_pct_median": "gpu_mem_util_pct",
    "gpu_power_mw_median": "gpu_power_mw",
    "gpu_sm_clock_mhz_median": "gpu_sm_clock_mhz",
}
_STD_FIELDS = {
    "gpu_mem_util_pct_std": "gpu_mem_util_pct",
}


class HistoricalGpuClassifier:
    """Envuelve el modelo scikit-learn exportado + su metadata, y mantiene
    el buffer móvil de muestras NVML descrito en el docstring del módulo.

    Uso con `activity_poller.poll_phase_events()`:
        clf = HistoricalGpuClassifier.from_export_dir(Path("fase2_clasificador/models"))
        events = poll_phase_events(query_fn, on_sample=clf.record_sample, ...)
        decisions = gpu_loop.run(events, controller, classify_fn=clf.classify)
    """

    def __init__(self, model: Any, feature_names: list[str], window_size: int = DEFAULT_WINDOW_SIZE):
        unknown = set(feature_names) - set(_MEDIAN_FIELDS) - set(_STD_FIELDS)
        if unknown:
            raise ValueError(
                f"HistoricalGpuClassifier no sabe derivar estas columnas del modelo: {sorted(unknown)} -- "
                "actualizar _MEDIAN_FIELDS/_STD_FIELDS si el candidato cambió de variables."
            )
        self._model = model
        self._feature_names = feature_names
        self._window: deque[GpuFeatures] = deque(maxlen=window_size)

    @classmethod
    def from_export_dir(
        cls, models_dir: Path, name: str = "gpu_random_forest_historical_20260923_sin_reloj",
        window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> "HistoricalGpuClassifier":
        """Carga `{name}.joblib` + `{name}.metadata.json` de
        `fase2_clasificador/models/` -- nunca hardcodea la lista de
        variables ni el nombre del archivo del modelo en el daemon, las
        lee de la metadata exportada por Fase 2 (mismo principio que
        `build_controller_from_policy` con la tabla de política)."""
        import joblib  # import perezoso: no todo consumidor de este módulo necesita joblib instalado

        model_path = models_dir / f"{name}.joblib"
        metadata_path = models_dir / f"{name}.metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"falta {model_path} o {metadata_path} -- exportar primero con "
                "fase2_clasificador/analysis/export_gpu_historical_candidate.py "
                "(ver Plan_Fase3_Daemon.md, Bloque A, A3)"
            )
        metadata = json.loads(metadata_path.read_text())
        model = joblib.load(model_path)
        return cls(model, list(metadata["features"]), window_size=window_size)

    def reset_window(self, *_ignored) -> None:
        """Vacia el buffer. Se engancha a `poll_phase_events(on_active_start=...)`
        para que la mediana/std se calculen SOLO sobre las muestras de la fase
        actual y no arrastren el hueco ocioso anterior (util ~0), que sesga
        hacia abajo la mediana y hacia arriba la desviacion estandar."""
        self._window.clear()

    def record_sample(self, features: GpuFeatures) -> None:
        """Callback para `activity_poller.poll_phase_events(on_sample=...)`
        -- se llama en CADA muestra sondeada, no solo en inicios de fase."""
        self._window.append(features)

    def _build_feature_vector(self, features_now: GpuFeatures) -> list[float]:
        """Construye el vector en el MISMO orden que `self._feature_names`
        (el orden con el que se entrenó el modelo -- nunca se reordena por
        conveniencia). Si el buffer está vacío (arranque en frío: todavía
        no llegó ninguna muestra de `record_sample`), usa la instantánea
        actual como única observación -- mediana de 1 = esa muestra, std
        de 1 = 0.0, el caso límite honesto, no una excepción especial."""
        window = list(self._window) or [features_now]
        row = []
        for name in self._feature_names:
            if name in _MEDIAN_FIELDS:
                attr = _MEDIAN_FIELDS[name]
                row.append(statistics.median(getattr(f, attr) for f in window))
            else:
                attr = _STD_FIELDS[name]
                values = [getattr(f, attr) for f in window]
                row.append(statistics.stdev(values) if len(values) > 1 else 0.0)
        return row

    def classify(self, features_now: GpuFeatures) -> GpuPhaseLabel:
        """`classify_fn` inyectable de `gpu_loop.run()`.

        El modelo NO predice la string `phase_label_train` -- entrena
        sobre `y = frame[LABEL].eq("memory_bound")`
        (`fase2_clasificador/analysis/gpu_quality_report.py::load()`), así
        que `model.predict()` devuelve un booleano: `True` = memory_bound,
        `False` = compute_bound. Traducirlo con cualquier otra suposición
        (p.ej. que devuelve la string, o que `model.classes_[1]` es
        siempre la clase positiva sin verificarlo) falla en silencio o con
        un error que no dice por qué -- de ahí la conversión explícita y
        el `isinstance` estricto, no una comparación laxa tipo `== 1`.
        """
        row = self._build_feature_vector(features_now)
        predicted = self._model.predict([row])[0]
        if not isinstance(predicted, (bool, np.bool_)):
            raise TypeError(
                f"el modelo devolvió {predicted!r} ({type(predicted).__name__}), se esperaba un booleano "
                "(y = phase_label_train.eq('memory_bound') en el entrenamiento) -- "
                "¿cambió el contrato de etiqueta de fase2_clasificador/analysis/gpu_quality_report.py?"
            )
        return GpuPhaseLabel.MEMORY_BOUND if bool(predicted) else GpuPhaseLabel.COMPUTE_BOUND
