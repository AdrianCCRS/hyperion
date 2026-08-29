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
