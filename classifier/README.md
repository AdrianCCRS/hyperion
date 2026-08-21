# classifier/

Fase 2 del proyecto: entrenamiento del clasificador ligero que infiere la fase
de ejecución (`compute_bound` / `memory_bound`) a partir de telemetría de
hardware. Paquete hermano de `orchestrator/` (Fase 1, el instrumento de
medición) — no depende de él salvo para leer los artefactos que ya produjo
(`windows.csv`, `frequency_quality_summary.json`, etc.), nunca para volver a
correr una campaña.

## Estructura

- `features/` — preprocesamiento y selección de características a partir de
  `windows.csv` (limpieza, normalización, filtrado por las anotaciones de
  calidad de la Fase 1: `quality_status`, `frequency_quality_status`).
- `training/` — definición y entrenamiento de modelos candidatos, búsqueda de
  hiperparámetros (Optuna).
- `eval/` — métricas de clasificación y de latencia de inferencia, comparación
  entre modelos candidatos.
- `notebooks/` — exploración interactiva (matrices de correlación,
  distribución de etiquetas, etc.). Pensados para correr contra un kernel
  Jupyter remoto en `paccaA100` vía VS Code Remote-SSH, no localmente contra
  una copia del dataset.

## Datos

El dataset (`windows.csv` de ambas campañas, CPU y GPU) vive en
`~/hyperion-results/` en `pacca`, no en este repositorio — mismo criterio que
`orchestrator/` ya aplica con las suites de benchmarks de terceros: el
repositorio versiona el código, no los datos ni sus copias. El código de este
paquete recibe la ruta al dataset como parámetro/config, nunca la asume fija
ni la vendoriza.

## Entorno remoto (VS Code + paccaA100/pacca)

El venv `~/hyperion-venv` en `pacca` ya trae `pandas`, `numpy`,
`scikit-learn`, `matplotlib`, `seaborn`, `jupyter`/`ipykernel` y `optuna`
instalados, con un kernel de Jupyter registrado como `hyperion-classifier`
("Hyperion (classifier venv)").

Para conectar VS Code directamente (extensión **Remote-SSH**): el host
`pacca` ya está configurado en `~/.ssh/config` con `ProxyJump
hpc-unicartagena`, así que `ssh pacca` (o "Connect to Host" → `pacca` en VS
Code) conecta en un solo paso sin anidar sesiones a mano. Al abrir un
notebook remoto, seleccionar el kernel `Hyperion (classifier venv)`.

Trabajo puramente exploratorio (EDA, matrices de correlación, distribución de
etiquetas) no necesita GPU ni una asignación exclusiva de `paccaA100` — corre
perfectamente en el nodo de login `pacca` o en cualquier nodo de cómputo CPU
libre (`pacca01`, `pacca03`, ...) vía `srun`. Reservar `paccaA100` solo
cuando el trabajo específicamente necesite la GPU.

## Pruebas

`tests/classifier/`, mismo patrón que `tests/orchestrator/` (sin
`__init__.py`, cada archivo de test resuelve el import con
`sys.path.insert`).
