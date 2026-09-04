# Fase 2 — Entrenamiento y validación del clasificador de fase

Cumple el **Objetivo 2**: entrenar y validar modelos ligeros de Machine
Learning (árbol de decisión, Random Forest, regresión logística, XGBoost)
capaces de clasificar, en tiempo de ejecución y con baja latencia de
inferencia, la fase de una aplicación (`compute_bound`/`memory_bound`) a
partir de la telemetría barata que produce `fase1_telemetria/`.

No corre nada en producción: entrena, compara y serializa el modelo que
`fase3_daemon/` carga y consulta en el loop de CPU. Ver
`Plan_Detallado_Realineacion_Hyperion.md` §3 para el diseño completo.

## Entrada: `training_cpu_intervals.csv` de Fase 1

`fase1_telemetria/postprocess.py` produce este archivo junto a la traza
auditable `windows.csv`: un directorio de campaña con subdirectorios
`<campaign_id>__<kernel_ref>__<freq_level_id>__rep<NN>/training_cpu_intervals.csv`.
Cada fila representa un intervalo real de `uncore_imc` (~10 ms o más), no un
tick CPU de ~1 ms. Sus tasas se recalculan desde las sumas de deltas del
intervalo; `freq_khz_observed` es la mediana de las ventanas CPU cubiertas.
Columnas que usa: las 7 `FEATURES` (`ipc`, `mpki`, `cache_miss_rate`,
`stall_mem_ratio`, `ips`, `running_ratio`, `freq_khz_observed`), la
etiqueta `phase_label_train`, `kernel_ref`, `training_quality_status` (filtra
`== "ok"`) y `frequency_quality_status` (filtra `valid`/`not_applicable_native`).
Los intervalos rechazados se conservan en ese CSV con `training_quality_reason`
para auditoría y nunca entran al modelo. El entrenador no tiene fallback a
`windows.csv`: campañas históricas deben reprocesarse para evitar duplicar una
misma observación de bytes IMC en varias filas de entrenamiento.

## Uso

```bash
python3 fase2_clasificador/run_training.py \
    --campaign-dir ~/hyperion-results/campaigns/mi_campana \
    --campaign-id mi_campana \
    --output-dir fase2_clasificador/models/
```

Sin `--output-dir`: modo exploración, imprime las tablas comparativas sin
guardar nada. Ver `--help` para el resto de flags (`--kernels`, `--levels`,
`--seed`, `--per-run-sample`, `--latency-weight`).

## Diagnóstico previo de cobertura (`F2-XDEV-001`)

Antes de entrenar o decidir cuotas de balance, ejecutar el diagnóstico sobre
la mini campaña. No altera los CSV de Fase 1 ni entrena un modelo: genera
`family_class_frequency_summary.csv`, `kernel_quality_summary.csv` y
`phase_coverage_report.json` para seleccionar familias y verificar cobertura
por clase.

```bash
python3 fase2_clasificador/run_phase_coverage.py \
  --campaign-dir ~/hyperion-results/campaigns/pacca_phase_coverage_cpu_screen_20260903 \
  --campaign-id pacca_phase_coverage_cpu_screen_20260903 \
  --device cpu \
  --output-dir fase2_clasificador/reports/phase_coverage_cpu/
```

Para GPU, `--device gpu` describe las muestras NVML y sus etiquetas actuales,
pero el JSON marca explícitamente que no son fases independientes. No se debe
usar ese resultado para entrenar hasta implementar la agregación por corrida o
fase estable.

## ⚠️ Fuga de información (data leakage) — la regla más importante de este módulo

La etiqueta `phase_label_train` se calcula en Fase 1 como
`memory_bound if operational_intensity < i_ridge_used else compute_bound`.
Por eso `operational_intensity*`, `i_ridge_used`, `flops_measured_window`,
`bytes_moved_*`, `uncore_cas_count_*` y `phase_label_hint` están en el
conjunto `FORBIDDEN` de `train_phase.py` — un modelo que los reciba no
aprende nada, solo vuelve a aplicar el umbral que ya generó la etiqueta.
`main()` aborta con `SystemExit` si alguna aparece en `FEATURES`. Las 7
features permitidas son deliberadamente baratas: las mismas que estarán
disponibles para el daemon de Fase 3 en producción, sin necesitar `uncore`
ni medir FLOPs en línea.

## Validación: leave-one-familia-out, no leave-one-kernel-out

El catálogo fusionado de Fase 1 tiene familias como `dual_gemm_*` con el
mismo algoritmo barrido sobre ~16 tamaños de problema. Dejar un solo tamaño
fuera de entrenamiento no prueba generalización a un algoritmo nuevo, solo
a un tamaño ya visto. `fase2_clasificador/eval/protocol.py::derive_kernel_family()`
deriva la familia algorítmica de cada `kernel_ref` (por patrón, no por
tabla fija — ver el módulo para las reglas exactas), y
`leave_one_familia_out()` valida agrupando por esa familia. Sobre el
dataset histórico de 9 kernels (ninguno comparte familia con otro) el
resultado es idéntico a `leave_one_kernel_out` — no es un protocolo
distinto, es el mismo protocolo con la unidad de agrupación correcta.

## Modelos comparados y criterio de selección

`build_models()`: línea base mayoritaria (obligatoria — si nada le gana, el
hallazgo es que las features no bastan), árbol de profundidad 1, regresión
logística, árbol de profundidad 6, Random Forest, Extra Trees, XGBoost
(techo de capacidad, comparación no elección por defecto).

El modelo que se serializa **no es el de mayor F1 macro sin más**:
`select_best_model()` combina `(1 - F1_macro_medio)` con la latencia p99
normalizada, ponderada por `--latency-weight` (default `0.2`). Con
`latency-weight=0`, equivale a elegir por F1 solo. Es un punto de partida
documentado, no una fórmula validada empíricamente — si se usa un valor
distinto del default en los resultados finales, repórtese la sensibilidad
a ese parámetro en el capítulo correspondiente.

## Salida

Con `--output-dir`: `<modelo>.joblib` (cargable con `joblib.load`, expone
`.predict(X)` estándar de scikit-learn/XGBoost) y `<modelo>.metadata.json`
(features usadas, seed, tamaño del dataset, número de familias, métricas
de validación cruzada, latencia p50/p95/p99, y la comparación completa
contra el resto de modelos — para que el capítulo de resultados no necesite
re-derivar nada). El modelo se reentrena sobre **todos** los datos
disponibles antes de serializar — los pliegues de leave-one-familia-out son
solo para estimar generalización, no el modelo final.

## Tests

```bash
python3 -m pytest fase2_clasificador/tests/ -q
```

58 tests: 17 del protocolo original (`test_protocol.py`), 33 de
`derive_kernel_family`/`leave_one_familia_out` (verificados contra IDs
reales del catálogo fusionado) y 8 de `train_phase.py` (incluye un
end-to-end real de `main()` con campaña sintética que verifica que el
`.joblib`/`.metadata.json` resultantes son cargables y correctos — no solo
que el script no lance una excepción).

## Limitaciones conocidas

- `select_best_model()` no se ha validado todavía contra resultados reales
  de una campaña completa — el peso de latencia por defecto (`0.2`) es un
  punto de partida razonable, no una elección respaldada por datos.
- No se ha entrenado todavía sobre el catálogo ampliado de 232 kernels — el
  dataset por defecto sigue siendo la campaña original de 9 kernels
  (`DEFAULT_CAMPAIGN_DIR`/`DEFAULT_KERNELS`), hasta que exista una campaña
  real corrida sobre `fase1_telemetria/catalog/catalog.yaml` fusionado.
- El clasificador de GPU (features de NVML: `gpu_util_pct`, `gpu_mem_util_pct`,
  `gpu_power_mw`, `gpu_sm_clock_mhz`, `gpu_temperature_c`, §2.5 del plan) no
  está implementado todavía en este módulo — `train_phase.py` cubre
  únicamente el clasificador de CPU.
