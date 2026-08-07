# TESTS.md — Estrategia de pruebas del pipeline VTune para Cartagena

Dos suites con propósitos distintos. No mezclarlas: la unitaria valida lógica de
código sin tocar el nodo; la de integración valida que el nodo real se comporta
como el plan asume.

## 1. Tests unitarios (`tests/unit/`)

**Corren en cualquier entorno con Python 3, incluida la sesión de desarrollo antes
de tener acceso al nodo.** Usan `pytest`. No requieren `vtune`, no requieren Slurm.

### Qué cubren

- `test_parser.py` — que `vtune_parser.py` extraiga correctamente los campos de un
  reporte de VTune, tolere `NA` sin excepción, y falle de forma visible (no
  silenciosa) si el formato cambia de manera irreconocible.
- `test_classifier_native.py` — la lógica de `clasificar_nativo()` (Fase 4.2 del
  plan): los tres desenlaces (`memory_bound`, `compute_bound`, `ambiguous`) y el
  caso `invalid` cuando falta `Memory Bound`.
- `test_classifier_ceilings.py` — la lógica de `eje_techos()` (Fase 4.3): cálculo
  del porcentaje respecto al techo de DGEMM, y el caso `NA` cuando falta
  `dp_gflops`.

### Fixtures — estado actual (post Fase 0/4)

`tests/unit/fixtures/real_*` son capturas reales de `vtune -report summary` /
`-report hotspots` en `paccaA100` (texto), tomadas en la Fase 0 y usadas
directamente por `test_parser.py` y `test_classifier_native.py` contra
`vtune_parser.py`/`classifier.py` definitivos — ya no hay una capa de
"implementación de referencia" intermedia (`_reference_impl.py` se eliminó,
cumplió su propósito de scaffolding antes de la Fase 0/4).

Las plantillas ilustrativas que asumían un desglose Top-Down de 4 categorías
(`Core Bound`, `Front-End Bound`, `Bad Speculation`, `Retiring`) se eliminaron
porque la Fase 0 confirmó que el reporte real de `hpc-performance` en este nodo
**no trae esas categorías** — ver `context/04_vtune_selfchecker_resultados.md`
(addendum) y `PLAN.md` §4.2. Solo queda `summary_missing_metrics_template.txt`
(caso sintético de reporte casi vacío, sigue siendo válido porque no depende de
ningún nombre de campo específico, solo prueba tolerancia a `N/A`).

Si se agregan más kernels/clases como fixtures reales en el futuro, seguir el
mismo patrón: `real_summary_<kernel>_<clase>.txt` / `.csv`, con un comentario de
cabecera indicando nodo, versión de VTune y fecha de captura.

### Cómo correr

```bash
cd tests/unit
pip install pytest --break-system-packages   # o el equivalente segun el entorno
pytest -v
```

## 2. Tests de integración (`tests/integration/`)

**Requieren el nodo real, una reserva Slurm activa, y el módulo `vtune/2023`
cargado.** Formalizan los criterios de "paso"/"no paso" que el plan ya describía en
prosa para cada fase, como scripts con código de salida.

### Orden de ejecución y qué valida cada uno

| Script | Fase del plan que valida | Criterio de paso |
|---|---|---|
| `00_test_module_and_smoke.sh` | Fase 0 | Módulo carga, `vtune --version` responde, Hotspots HW y HPC Performance corren sobre `ep.C.x` sin error de reconocimiento de procesador, y al menos `Memory Bound`/equivalente y `DP GFLOPS` salen poblados |
| `01_test_preflight.sh` | Fase 2 | `check_vtune.py` termina con código 0 sobre el estado real del nodo |
| `02_test_baseline_ep.sh` | Fase 3.1 | El binario `ep.C.x` corre fuera de VTune, termina con código 0, y su salida contiene `VERIFICATION SUCCESSFUL` |
| `03_test_anchors.sh` | Fase 1.2 / 4.3 | STREAM y DGEMM (o BT/SP) compilan, corren, y sus binarios imprimen un número de ancho de banda / GFLOP/s reconocible por el parser correspondiente |
| `04_test_single_kernel_full.sh` | Fase 3 a 6, extremo a extremo | Sobre un solo kernel (`ep.C.x`): baseline válido, Hotspots HW y HPC Performance recolectados, fila generada en `consolidated_results.csv` con `classification_vtune_native` no vacío y `quality_status` en estado esperado |

`run_all.sh` corre los cinco en orden y se detiene en el primer fallo, imprimiendo
cuál fase quedó bloqueada.

### Cómo correr

```bash
srun -w paccaA100 --exclusive --cpus-per-task=8 --time=00:30:00 --pty bash -i
module load vtune/2023
cd tests/integration
bash run_all.sh
```

**No correr esta suite fuera de una reserva activa** — varios scripts invocan
`vtune -collect`, que necesita el nodo real con el módulo cargado.

## 3. Qué NO están diseñadas para hacer estas pruebas

- No validan que la clasificación sea "correcta" en el sentido de coincidir con
  Hyperion o con un ground truth externo — eso es una comparación posterior, fuera
  de este pipeline (ver `context/02_decisiones.md` D2).
- No sustituyen la revisión manual que pide D4 para EP cuando sale
  `memory_bound`/`ambiguous` — un test verde en EP no significa que el resultado
  no necesite esa revisión, solo que el pipeline no se rompió al procesarlo.
