"""Pipeline del selector CPU/GPU de Fase 2.

Este paquete es deliberadamente independiente del clasificador historico de
fases por ventana. Su unidad experimental es una operacion completa y su
unidad de decision es un conjunto de configuraciones candidatas.
"""

from .dataset import BuildConfig, build_selector_datasets
from .r1 import run_r1_analysis
from .r2 import run_r2_analysis

__all__ = ["BuildConfig", "build_selector_datasets", "run_r1_analysis", "run_r2_analysis"]
