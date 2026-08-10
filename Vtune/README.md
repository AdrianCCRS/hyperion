# Vtune — campaña de validación cruzada (Microarchitecture Exploration)

Campaña independiente de validación cruzada para Hyperion, usando Intel VTune
Profiler 2023 con el análisis **Microarchitecture Exploration** sobre
`paccaA100` (Cartagena). Documento metodológico completo:
[`docs/vtune/vtune_cross_validation.md`](../docs/vtune/vtune_cross_validation.md).

**Esto NO es el pipeline de entrenamiento de Hyperion ni lo reemplaza.** Es
una segunda fuente de observación, independiente en pipeline de
procesamiento y en modelo de clasificación, para confirmar que los kernels
elegidos para el dataset de entrenamiento (PID+inherit sobre `perf`, ver
`docs/retoma/Guia_Maestra_Fase1_DVFS.md`) se comportan de verdad como
`compute_bound`/`memory_bound`.

**No toca `pipelinevtune/`.** Esa carpeta queda congelada como referencia
histórica de cuando Microarchitecture Exploration no estaba disponible en
este nodo (ver `pipelinevtune/context/04_vtune_selfchecker_resultados.md`) —
esta carpeta parte de cero ahora que sí lo está.

## Requisito antes de enviar el job

Slurm no crea el directorio de logs por ti — `--output`/`--error` del sbatch
fallan si no existe:

```bash
mkdir -p ~/vtune_validation/logs
```

## Cómo lanzar la campaña

```bash
cd Vtune/
sbatch sbatch_vtune_validation.sh
```

Variables de entorno opcionales para override (todas tienen default sensato,
ver comentarios en `sbatch_vtune_validation.sh`):

```bash
VTUNE_VALIDATION_BIN_DIR=$HOME/vtune_selfcheck/bin \
VTUNE_VALIDATION_ANCHOR_DIR=$HOME/vtune_selfcheck/anchor_bin \
VTUNE_VALIDATION_REPETITIONS=2 \
VTUNE_VALIDATION_KERNELS=ep,cg,mg,ft,lu,bt \
VTUNE_VALIDATION_CLASSES=C \
sbatch sbatch_vtune_validation.sh
```

Si el nodo está ocupado, el job queda en cola — nada en este pipeline intenta
liberar recursos, cancelar trabajos ajenos, ni forzar entrada.

## Cómo revisar progreso y resultados

```bash
squeue -u $USER                                  # estado del job
tail -f ~/vtune_validation/logs/slurm_<JOBID>.out # progreso en vivo (logging de run_validation.py)

# Cuando termine:
column -s, -t < ~/vtune_validation/results_vtune/job_<JOBID>/consolidated_validation.csv | less -S
cat ~/vtune_validation/results_vtune/job_<JOBID>/preflight_result.json
```

Si el preflight falla (permiso/PMU faltante), el job termina rápido con
código de salida 1 y `preflight_result.json` documenta exactamente qué
condición bloqueó — no hay una corrida parcial de kernels sobre una base
rota.

## Estructura de resultados

```
results_vtune/job_<JOBID>/
├── preflight_result.json
├── consolidated_validation.csv        <- una fila por kernel/clase/repetición
├── ep.C/
│   ├── rep_01/
│   │   ├── result/                    <- result-dir completo de VTune, para GUI
│   │   ├── summary.txt                <- vtune -report summary (texto)
│   │   ├── report.csv                 <- vtune -report summary -format=csv
│   │   ├── raw_hw_events.csv          <- vtune -report hw-events -format=csv
│   │   ├── metadata.json              <- comando exacto, afinidad, checksum, etc.
│   │   ├── baseline_stdout.txt
│   │   └── baseline_stderr.txt
│   └── rep_02/...
├── cg.C/...  mg.C/...  ft.C/...  lu.C/...  bt.C/...
├── stream/rep_01/...  rep_02/...
└── dgemm/rep_01/...  rep_02/...
```

Adaptación deliberada respecto a la propuesta inicial `results_vtune/<kernel>/`:
se agrega el nivel `rep_NN/` porque la campaña corre repeticiones (mínimo 2,
ver `run_validation.py` `DEFAULT_REPETITIONS` y su justificación) — sin ese
nivel, la segunda repetición sobrescribiría a la primera.

## Archivos

| Archivo | Propósito | Cómo se usa |
|---|---|---|
| `sbatch_vtune_validation.sh` | Job de Slurm: carga módulos, resuelve rutas, invoca el pipeline. | `sbatch sbatch_vtune_validation.sh` |
| `run_validation.py` | Orquestador: preflight → descubre kernels/anclas → corre baseline+VTune por repetición → reportes → CSV consolidado. | Invocado por el sbatch; también corre standalone para depurar un subconjunto (`--kernels ep --repetitions 1`). |
| `preflight_uarch.py` | Preflight específico de Microarchitecture Exploration: PMU, `perf_event_paranoid`, `kptr_restrict`, `CAP_PERFMON`, arquitectura, afinidad Slurm, procesos ajenos (solo diagnóstico). | Se importa desde `run_validation.py`; también corre standalone (`python3 preflight_uarch.py`) para un chequeo rápido sin lanzar la campaña completa. |
| `uarch_parser.py` | Parsea `vtune -report summary`/`hw-events` de un resultado de `uarch-exploration` a un dict plano. | Importado por `run_validation.py`. |
| `validation_classifier.py` | Metodología de clasificación de validación (compute/memory/ambiguous) a partir del Top-Down completo de una sola corrida, sin calibración externa. | Importado por `run_validation.py`. |
| `README.md` | Este archivo. | — |
| `../docs/vtune/vtune_cross_validation.md` | Documento metodológico completo (propósito, independencia metodológica, funcionamiento de VTune, reproducibilidad, flujo manual, GUI). | Referencia para el trabajo de grado. |

## Flujo completo

```
sbatch sbatch_vtune_validation.sh
  ↓
module load devtools/intel/oneapi/2023 && module load vtune/2023.0.0
  ↓
run_validation.py: preflight_uarch (PMU/permisos/afinidad/smoke real de uarch-exploration)
  ↓ (aborta aquí, explícito, si algo falta)
descubrir kernels (--bin-dir) y anclas STREAM/DGEMM (--anchor-dir) -- nunca inventados
  ↓
por cada kernel/ancla × repetición:
  baseline (sin VTune) → vtune -collect uarch-exploration → guardar result-dir
  ↓
  vtune -report summary (texto + csv) + vtune -report hw-events (csv)
  ↓
  uarch_parser: texto → métricas Top-Down (dict)
  ↓
  validation_classifier: métricas → compute_bound | memory_bound | ambiguous | invalid
  ↓
  metadata.json (comando exacto, afinidad, checksum, hilos, timestamps)
  ↓
acumular fila → consolidated_validation.csv (se reescribe tras cada repetición, nunca se pierde progreso)
  ↓
inspección manual (CSV) / apertura en VTune GUI (result/) / docs/vtune/vtune_cross_validation.md
```

## Flujo manual (un solo kernel, sin el pipeline)

Ver `docs/vtune/vtune_cross_validation.md` sección 9 para el detalle completo
y la explicación de cada flag. Resumen:

```bash
module load devtools/intel/oneapi/2023
module load vtune/2023.0.0

export OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close

taskset -c 0-5 vtune -collect uarch-exploration \
  -r ~/vtune_validation/manual/ep_C \
  -- $HOME/vtune_selfcheck/bin/ep.C.x

vtune -report summary -r ~/vtune_validation/manual/ep_C
vtune -report summary -r ~/vtune_validation/manual/ep_C -format=csv > ep_C_summary.csv
vtune -report hw-events -r ~/vtune_validation/manual/ep_C -format=csv > ep_C_hw_events.csv
```
