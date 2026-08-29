# classifier/

Fase 2 de Hyperion. El objetivo vigente es puntuar las configuraciones
candidatas de una operación y seleccionar el dispositivo y las frecuencias
que minimizan EDP. El clasificador histórico de ventanas
`compute_bound`/`memory_bound` se conserva para reproducibilidad, pero no
construye el dataset del selector cold/warm.

## Pipeline vigente

`python -m classifier.selector` expone cinco comandos:

- `build`: integra las regiones absolutas `cold` y `warm`, aplica la regla
  energética simétrica y construye los datasets de estrategias A y C.
- `eda`: produce correlaciones Pearson/Spearman, faltantes, distribuciones de
  óptimos, curvas EDP y diagnóstico de resolución temporal.
- `tune`: compara regresión logística, árbol, Random Forest y XGBoost mediante
  Optuna multiobjetivo dentro de una validación anidada
  leave-one-operation-out.
- `evaluate`: resume una búsqueda ya terminada.
- `all`: ejecuta la cadena completa.

Ejemplo de construcción provisional CPU:

```bash
python -m classifier.selector build \
  --cpu-campaign ~/hyperion-results/campaigns/pacca_dual_cpu_full_20260828 \
  --gpu-campaign ~/hyperion-results/campaigns/pacca_dual_gpu_full_20260828 \
  --catalog orchestrator/schemas/kernels/catalog.yaml \
  --cpu-manifest orchestrator/schemas/campaigns/campaign_pacca_dual_cpu_full.yaml \
  --gpu-manifest orchestrator/schemas/campaigns/campaign_pacca_dual_gpu_full.yaml \
  --output-dir ~/hyperion-results/analysis/selector_cpu_provisional_20260828 \
  --mode cpu-provisional
```

El modo provisional exige 68 configuraciones CPU, 8 niveles y 3
repeticiones. Puede leer GPU parcial para auditar el integrador, pero esas
corridas nunca generan labels. `--mode final` exige las 40 acciones
CPU/GPU completas por `config_id` y falla cerrado si falta una.

## Contratos principales

- `cold` contiene un despacho e incluye inicialización de bibliotecas,
  contexto CUDA, recursos, transferencias y resultado.
- `warm` se divide por `iterations`: CPU y GPU usan números distintos de
  despachos para lograr una duración medible y sus totales no son comparables
  directamente.
- El candidato CPU usa RAPL package+DRAM más la línea base GPU nativa de
  34.8379 W (job 6714). El candidato GPU usa RAPL+NVML integrado.
- A solo usa descriptores estáticos. C solo añade telemetría `cold` de una
  primera repetición REF y cambia entre costos cold/warm según qué dispositivo
  ya esté inicializado. Si esa región es menor que el intervalo de muestreo,
  su telemetría queda ausente con indicadores; no se rellena con cero.
- El split es por operación completa. Nunca se dividen filas candidatas ni
  tamaños de una operación entre entrenamiento y prueba.
- La métrica primaria es EDP loss; la exactitud del argmin es secundaria.
- **`EDP_dispatch` es una cota superior, no una ganancia neta**: no incluye el
  costo de actuar la frecuencia entre decisiones sucesivas, que se evalúa por
  separado con el daemon (`energy_rule.edp_interpretation` en
  `provenance.json`).

## Compuerta de salud de la etiqueta (`label_health.py`)

Antes de leer `model_comparison.csv` como una comparación real de familias,
`eda`/`tune` calculan si `is_optimal` tiene variedad aprendible: cuota de la
acción dominante, número de acciones con masa ≥5 % y margen EDP mediano entre
el mejor y el segundo candidato. Si la acción dominante gana >90 %, hay menos
de 3 acciones con masa apreciable, o el margen mediano cae bajo el piso de
ruido de medición (2 %), el veredicto es `pipeline_smoke_only`: la tabla sirve
para confirmar que `build → eda → tune → evaluate` corre de punta a punta,
**no** para decidir qué familia es mejor. El veredicto queda escrito en
`eda/label_health.json` y en `model_contract.json` (`result_status`).

Contra el dataset real (`selector_cpu_provisional_20260828`), A y C no dan
el mismo veredicto: **A es `comparison_valid`** (REF gana 36.8 %, 5 de 6
niveles con masa apreciable, margen mediano 11.3 %) — el paso 6 del plan es
informativo para A-CPU, no solo humo. **C es `pipeline_smoke_only`** (F0
gana 54.4 %, solo 2 acciones con masa, margen mediano 0.73 %): sus
resultados deben leerse como validación de tubería, no como comparación.
La estimación previa de "eje CPU degenerado" venía de una aproximación con
energía de proceso completo, no de la integración real por despacho —
normalizar por despacho separa el costo fijo de arranque del efecto de la
frecuencia sobre el kernel puro y revela más variación de la que esa
aproximación mostraba.

### Sensibilidad de A a los `cold` de baja resolución

A construye su etiqueta y sus candidatos exclusivamente sobre la región
`cold` (`build_strategy_a`, sin telemetría `warm` de respaldo). `eda`
calcula ahora `strategy_a_cold_sensitivity.json`: de las 544 acciones cold
de A, 32 (5.9 %) están marcadas de baja resolución; el ganador actual de un
grupo depende de una de ellas en 5/68 grupos; 3/68 grupos no tienen ninguna
acción de resolución nominal con la que contrastar; y excluir las acciones
de baja resolución cambia el ganador en 2/68 grupos (`axpy_N100000`
F0→F1, `axpy_N316228` REF→F2 — concentrado en `axpy`, como predecía el
diagnóstico de resolución temporal). No se excluyen automáticamente: la
auditoría queda escrita para que se lea junto con `model_comparison.csv`,
no para alterar la etiqueta en silencio.

La selección de familia final tampoco usa solo el 1 % relativo: una familia
es elegible si su EDP loss cae dentro de esa banda **o** dentro del error
estándar combinado entre los folds externos (`edp_loss_std / sqrt(n_folds)`
de ambas familias) — evita que 6 números con dispersión decidan la familia
"ganadora" cuando la diferencia real es ruido entre pliegues.

## Datos y entorno

Los crudos permanecen en `~/hyperion-results/` y no se versionan. Los
artefactos derivados (`run_regions.csv`, `candidate_summary.csv`, datasets
A/C, EDA, estudios Optuna y modelos) se escriben también bajo
`hyperion-results/analysis`.

El venv `~/hyperion-venv` contiene pandas, NumPy, scikit-learn, Optuna,
XGBoost, matplotlib, seaborn y Jupyter. Las búsquedas se ejecutan en un nodo
CPU de la partición `normal`; no necesitan ni deben reservar `paccaA100`.

## Pruebas

```bash
python -m pytest -q tests/classifier
```

La suite cubre integración temporal, fuentes de energía, normalización por
despacho, acciones, estrategias A/C, fugas y splits por operación. Las suites
`tests/classifier` y `tests/orchestrator` se ejecutan por separado por la
colisión histórica de descubrimiento documentada en el repositorio.
