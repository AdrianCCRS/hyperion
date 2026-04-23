# Guía Total De Uso Del Subsistema Modular De Telemetría

## Propósito

Este documento describe cómo compilar, probar y usar el subsistema modular de
telemetría ubicado en `Code/telemetry`. El objetivo práctico de este subsistema
es capturar métricas de ejecución de kernels HPC representativos para construir
un dataset que permita estudiar y clasificar fases de cómputo, por ejemplo
`memory_bound`, `compute_bound` y `cache_sensitive`.

El flujo actual está centrado en CPU multihilo. GPU/NVML existe como backend
opcional de la librería, pero el runner experimental de kernels todavía no usa
GPU como fuente principal de datos porque la atribución precisa de kernels CUDA
requiere una fase posterior con sincronización explícita y/o herramientas como
CUDA events, NVTX o CUPTI.

El archivo monolítico `telemetry_collection.cpp` se considera legado/MVP y no
forma parte de esta guía.

## Estado Del Sistema

El subsistema tiene dos capas principales:

- Librería `telemetry`: contiene los lectores de métricas, el collector, la ring
  SPSC y utilidades experimentales.
- Ejecutables experimentales: contienen benchmarks manuales, el workload de
  kernels CPU y el launcher que orquesta mediciones reales.

Targets principales:

| Target | Tipo | Uso |
| --- | --- | --- |
| `telemetry` | Librería estática | API base del subsistema. |
| `telemetry_kernel_launcher` | Ejecutable manual | Orquesta baseline, telemetría, cgroup, collector, consumidor y exportación. |
| `telemetry_kernel_workload` | Ejecutable hijo | Ejecuta kernels CPU multihilo controlados por el launcher. |
| `telemetry_overhead_bench` | Benchmark manual | Mide overhead sintético de la librería. No es test unitario. |
| `telemetry_jitter_bench` | Benchmark manual | Mide jitter sintético del productor. No es test unitario. |
| Tests CTest | Ejecutables de prueba | Validan ring, readers, collector y utilidades sin depender de nodo real. |

## Compilación

Compilación CPU-only recomendada para desarrollo local:

```bash
cmake -S telemetry -B /tmp/tg-telemetry-build
cmake --build /tmp/tg-telemetry-build
ctest --test-dir /tmp/tg-telemetry-build --output-on-failure
```

El `CMakeLists.txt` usa C++17, `RelWithDebInfo` por defecto y compila con
`-Wall -Wextra -O2 -march=native`.

Compilación con soporte NVML:

```bash
cmake -S telemetry -B /tmp/tg-telemetry-gpu-build -DWITH_GPU=ON
cmake --build /tmp/tg-telemetry-gpu-build
```

Requisitos para `WITH_GPU=ON`:

- `nvml.h` disponible.
- `libnvidia-ml` disponible.
- Nodo con driver NVIDIA funcional.

Aunque el backend NVML puede compilarse, el runner `telemetry_kernel_launcher`
se mantiene como ruta CPU en esta fase.

## Pruebas Locales

Los tests unitarios están registrados en CTest:

```bash
ctest --test-dir /tmp/tg-telemetry-build --output-on-failure
```

Estos tests deben poder ejecutarse sin permisos especiales de PMU, sin RAPL real
y sin GPU. Los tests que podrían requerir hardware real usan stubs, rutas falsas
o validaciones de fallo temprano.

El criterio mínimo antes de correr en nodo real es:

- `100% tests passed`.
- Build limpio.
- Sin cambios no deseados en archivos legacy.

## Arquitectura De Ejecución Real

La ruta de medición real usa el modelo:

```text
telemetry_kernel_launcher
  ├── baseline child: telemetry_kernel_workload sin collector activo
  └── telemetry child: telemetry_kernel_workload con collector + consumer activo
```

Para cada repetición, el launcher ejecuta dos procesos hijo con los mismos
parámetros:

- `baseline`: workload sin collector activo.
- `telemetry`: workload con producer, consumer y exportación de muestras.

El proceso hijo prepara memoria, crea el pool de hilos y ejecuta warmup antes de
avisar al launcher que está listo. El tiempo medido empieza después de esa fase.
Esto evita mezclar asignación de memoria, creación de hilos y setup con la región
de cómputo del kernel.

## Requisitos Para Nodo Real

Antes de correr una toma de datos real, validar:

| Requisito | Por qué importa |
| --- | --- |
| Cgroup delegado/precreado | El launcher mueve el PID hijo a `cgroup.procs`, pero no crea ni administra la jerarquía global. |
| Permiso de escritura en `cgroup.procs` | Sin esto el workload no entra al cgroup y perf cgroup no mide el proceso esperado. |
| Permiso para `perf_event_open` | Necesario para contadores CPU con perf. Depende de la política del kernel y del entorno. |
| Lista correcta de CPUs en `--perf-cpus` | Perf cgroup abre eventos por CPU. Si el workload corre fuera de esa lista, habrá submedición. |
| RAPL legible si se usa energía | `energy_uj` y preferiblemente `max_energy_range_uj` deben ser legibles. |
| Cgroup limpio | Durante la ventana medida debe contener solo el workload que se desea medir. |
| CPUs libres para collector/consumer si se pinnean | Para aislar telemetría, no deben compartir CPU con el workload. |

## Uso Rápido Sin Perf

Smoke local sin permisos de cgroup/perf:

```bash
/tmp/tg-telemetry-build/telemetry_kernel_launcher \
  --kernel stream_triad \
  --size 10000 \
  --iterations 2 \
  --warmup 1 \
  --threads 2 \
  --repetitions 2 \
  --perf-cpus 0,1 \
  --collector-cpu -1 \
  --consumer-cpu -1 \
  --no-perf \
  --output-dir /tmp/tg-telemetry-smoke \
  --run-id smoke_no_perf
```

Este modo permite comprobar el flujo de launcher, workload, consumer y archivos
de salida sin PMU. Si no se pasa RAPL, el CSV puede no tener filas de métricas de
hardware, pero el flujo de control debe funcionar.

## Primera Toma Real Recomendada

Ejemplo para nodo con cgroup delegado, perf cgroup y RAPL package:

```bash
/tmp/tg-telemetry-build/telemetry_kernel_launcher \
  --kernel stream_triad \
  --size 1000000 \
  --iterations 5 \
  --warmup 1 \
  --threads 4 \
  --repetitions 2 \
  --perf-cpus 2,3,4,5 \
  --collector-cpu -1 \
  --consumer-cpu -1 \
  --cgroup-path /sys/fs/cgroup/<delegated-cgroup> \
  --interval-ns 1000000 \
  --rapl-pkg /sys/class/powercap/intel-rapl/intel-rapl:0 \
  --output-dir runs \
  --run-id smoke_real_stream
```

Después inspeccionar:

```bash
head runs/smoke_real_stream/samples.csv
cat runs/smoke_real_stream/summary.txt
cat runs/smoke_real_stream/metadata.json
```

## Parámetros Del Launcher

Ejecutable:

```bash
telemetry_kernel_launcher [opciones]
```

| Parámetro | Valor | Default | Obligatorio | Descripción |
| --- | --- | --- | --- | --- |
| `--kernel` | `stream_triad`, `reduction`, `stencil_2d`, `gemm_naive` | `stream_triad` | No | Kernel CPU a ejecutar. Define también el label experimental. |
| `--size` | Entero positivo | `1000000` | No | Tamaño lógico del problema. En `stencil_2d` y `gemm_naive` representa dimensión `N` de matrices `N x N`. |
| `--iterations` | Entero positivo | `10` | No | Iteraciones medidas del kernel después del warmup. |
| `--warmup` | Entero no negativo | `1` | No | Iteraciones previas a la medición. Sirven para estabilizar cachés, páginas y estado inicial. |
| `--threads` | Entero positivo | `1` | No | Número de workers del workload. |
| `--repetitions` | Entero positivo | `1` | No | Número de pares baseline/telemetry. |
| `--perf-cpus` | Lista CPU, por ejemplo `2,3,4-7` | Vacío | Sí si perf está activo | CPUs donde `PerfCgroupReader` abre eventos. No pinnea el workload. |
| `--workload-cpus` | Lista CPU | Vacío | Alias | Alias histórico de `--perf-cpus`. Debe interpretarse como alcance de medición perf, no como pinning. |
| `--pin-workload-cpus` | Lista CPU | Vacío | No | Aplica afinidad al proceso hijo del workload. Si no se usa, el scheduler puede migrarlo. |
| `--pin-workers` | Flag | `false` | No | Pinnea un worker por CPU de `--pin-workload-cpus`. Requiere que `threads <= cantidad de CPUs`. |
| `--collector-cpu` | CPU o `-1` | `-1` | No | CPU del productor de telemetría. `-1` significa sin pinning. |
| `--consumer-cpu` | CPU o `-1` | `-1` | No | CPU del consumidor que drena la ring. `-1` significa sin pinning. |
| `--cgroup-path` | Ruta | Vacío | Sí si perf está activo | Cgroup delegado donde se moverá el PID hijo. |
| `--interval-ns` | Nanosegundos positivos | `1000000` | No | Periodo de muestreo del collector. `1000000` equivale a 1 ms. |
| `--no-perf` | Flag | `false` | No | Desactiva lectura de contadores CPU perf. Útil para smoke local o pruebas RAPL-only. |
| `--rapl-pkg` | Ruta a dominio RAPL | Vacío | No | Dominio package con `energy_uj` y opcionalmente `max_energy_range_uj`. |
| `--rapl-dram` | Ruta a dominio RAPL DRAM | Vacío | No | Dominio DRAM opcional. |
| `--output-dir` | Directorio | `runs` | No | Directorio base de resultados. |
| `--run-id` | String | `run_<timestamp>` | No | Nombre de la corrida. El resultado queda en `output-dir/run-id`. |
| `--workload-bin` | Ruta | Junto al launcher | No | Permite usar otro binario compatible con la interfaz del workload. |
| `--help` | Flag | - | No | Muestra uso básico. |

Validaciones actuales:

- `--kernel` debe estar soportado.
- `--size > 0`.
- `--iterations > 0`.
- `--warmup >= 0`.
- `--threads > 0`.
- `--repetitions > 0`.
- `--interval-ns > 0`.
- Si perf está activo, `--cgroup-path` y `--perf-cpus` son obligatorios.
- `--pin-workers` requiere `--pin-workload-cpus`.
- Con `--pin-workers`, `threads` no puede exceder el número de CPUs listadas.

## Parámetros Del Workload

El workload normalmente se invoca desde el launcher. Para depuración puede
ejecutarse directamente:

```bash
telemetry_kernel_workload \
  --kernel stream_triad \
  --size 1000000 \
  --iterations 10 \
  --warmup 1 \
  --threads 4
```

Parámetros internos:

| Parámetro | Descripción |
| --- | --- |
| `--kernel` | Kernel a ejecutar. |
| `--size` | Tamaño lógico del problema. |
| `--iterations` | Iteraciones medidas. |
| `--warmup` | Iteraciones previas no incluidas en `elapsed_ns`. |
| `--threads` | Número de workers. |
| `--worker-cpus` | Lista de CPUs para pinnear workers. La usa el launcher cuando se pasa `--pin-workers`. |
| `--ready-fd` | Descriptor usado por el launcher para saber que el hijo terminó setup y warmup. |
| `--go-fd` | Descriptor usado por el launcher para iniciar la ventana medida. |

Salida esperada del workload:

```text
elapsed_ns=<nanosegundos>
sink=<valor>
```

El launcher parsea `elapsed_ns`.

## Kernels Soportados

| Kernel | Label | Interpretación de `--size` | Patrón esperado |
| --- | --- | --- | --- |
| `stream_triad` | `memory_bound` | Número de elementos `N` | Recorrido lineal de arreglos `a`, `b`, `c`; alta presión de memoria. |
| `reduction` | `memory_bound` | Número de elementos `N` | Suma paralela con parciales por worker; dominado por lectura secuencial. |
| `stencil_2d` | `cache_sensitive` | Dimensión `N` de grilla `N x N` | Acceso local 2D con vecinos; sensible a caché y tamaño. Requiere `N >= 3`. |
| `gemm_naive` | `compute_bound` | Dimensión `N` de matrices `N x N` | Multiplicación de matrices ingenua; alto cómputo y tamaño de memoria `O(N^2)`. |

Precaución de memoria:

- `stream_triad`: usa tres vectores de `double`, alrededor de `24 * N` bytes.
- `reduction`: usa un vector principal y parciales, alrededor de `8 * N` bytes más overhead menor.
- `stencil_2d`: usa dos grillas `N x N`, alrededor de `16 * N^2` bytes.
- `gemm_naive`: usa tres matrices `N x N`, alrededor de `24 * N^2` bytes.

## Pinning Y Alcance De Medición

Hay tres conceptos distintos:

| Concepto | Parámetro | Qué hace |
| --- | --- | --- |
| Alcance perf por CPU | `--perf-cpus` | Abre eventos perf cgroup en esas CPUs. No cambia scheduling. |
| Pinning del workload | `--pin-workload-cpus` | Restringe afinidad del proceso hijo. |
| Pinning de workers | `--pin-workers` | Fija un worker por CPU listada. |
| Pinning del productor | `--collector-cpu` | Fija el hilo productor del collector. |
| Pinning del consumidor | `--consumer-cpu` | Fija el hilo consumidor que drena la ring. |

Para dejar al workload migrar libremente, no usar `--pin-workload-cpus` ni
`--pin-workers`. En ese caso, `--perf-cpus` debe cubrir todas las CPUs donde el
workload puede ejecutarse realmente.

Para estudiar el impacto del pinning de telemetría:

```bash
# Variante A: productor y consumidor libres
--collector-cpu -1 --consumer-cpu -1

# Variante B: productor y consumidor aislados
--collector-cpu 6 --consumer-cpu 7
```

Idealmente, las CPUs `6` y `7` no deben pertenecer al conjunto donde corre el
workload.

## Formato De `samples.csv`

Cada fila tiene una forma unificada CPU/RAPL/GPU. Los campos no aplicables se
dejan vacíos.

| Columna | Aplica a | Descripción |
| --- | --- | --- |
| `run_id` | Todas | Identificador de corrida. |
| `repetition` | Todas | Repetición del par baseline/telemetry. Las muestras solo existen en la ejecución telemetry. |
| `kernel` | Todas | Kernel medido. |
| `label` | Todas | `memory_bound`, `compute_bound`, `cache_sensitive` o `unknown`. |
| `timestamp_ns` | Todas | Timestamp `CLOCK_MONOTONIC` en nanosegundos. |
| `tag` | Todas | `CPU`, `ENERGY` o `GPU`. |
| `instructions` | CPU | Instrucciones escaladas por perf. |
| `cycles` | CPU | Ciclos escalados por perf. |
| `cache_references` | CPU | Referencias de caché escaladas por perf. |
| `cache_misses` | CPU | Fallos de caché escalados por perf. |
| `time_enabled_ns` | CPU | Tiempo habilitado por perf. |
| `time_running_ns` | CPU | Tiempo corriendo por perf. |
| `pkg_uj` | ENERGY | Lectura cruda RAPL package en microjoules. |
| `dram_uj` | ENERGY | Lectura cruda RAPL DRAM en microjoules. Cero si no se configuró DRAM. |
| `pkg_delta_uj` | ENERGY | Delta wrap-aware contra la muestra ENERGY previa de la misma repetición. |
| `dram_delta_uj` | ENERGY | Delta wrap-aware DRAM. |
| `energy_delta_valid` | ENERGY | `1` si el delta es válido. `0` para primera muestra o wrap sin rango máximo legible. |
| `gpu_power_mw` | GPU | Potencia NVML en mW. Actualmente no usado por el launcher CPU. |
| `gpu_util_pct` | GPU | Utilización GPU NVML. Actualmente no usado por el launcher CPU. |

Notas:

- Las filas CPU se usan para jitter y ratio de multiplexación.
- Las filas ENERGY contienen snapshots crudos y deltas derivados.
- El productor no calcula deltas RAPL; los deltas se calculan en exportación.

## Formato De `metadata.json`

Campos principales:

| Campo | Significado |
| --- | --- |
| `run_id`, `kernel`, `label` | Identidad de la corrida. |
| `size`, `iterations`, `warmup`, `threads`, `repetitions` | Parámetros del workload. |
| `interval_ns` | Periodo nominal de muestreo. |
| `enable_perf` | Si perf estuvo activo. |
| `perf_cpus` | CPUs usadas para abrir eventos perf cgroup. |
| `pin_workload_cpus`, `pin_workers` | Pinning del workload y workers. |
| `collector_cpu`, `consumer_cpu` | Pinning de productor y consumidor. |
| `cgroup_path` | Cgroup usado para medición perf. |
| `baseline_elapsed_ns_mean`, `telemetry_elapsed_ns_mean` | Promedios de tiempo por modo. |
| `baseline_elapsed_ns_sd`, `telemetry_elapsed_ns_sd` | Desviación estándar por modo. |
| `overhead_pct_mean`, `overhead_pct_sd` | Overhead medio y dispersión. |
| `baseline_elapsed_ns_values`, `telemetry_elapsed_ns_values` | Valores por repetición. |
| `overhead_pct_values` | Overhead por repetición. |
| `sampling_interval_mean_ns`, `sampling_interval_sd_ns`, `sampling_interval_cv_pct` | Métricas de jitter del productor calculadas con filas CPU. |
| `push_retries` | Veces que el productor no pudo empujar a la ring inmediatamente. |
| `push_retries_by_repetition` | Reintentos por repetición. |
| `perf_running_ratio_min` | Mínimo `time_running / time_enabled` observado. Cerca de `1.0` indica baja multiplexación. |
| `rapl_pkg_max_range_uj`, `rapl_dram_max_range_uj` | Rango máximo RAPL leído para manejo de overflow. |
| `rapl_pkg_total_delta_uj`, `rapl_dram_total_delta_uj` | Energía acumulada por deltas válidos. |
| `rapl_energy_delta_count` | Número de deltas ENERGY válidos acumulados. |
| `samples_collected` | Total de filas/muestras exportadas. |

## Criterios De Calidad Para Una Corrida

Para aceptar una corrida como candidata a dataset:

| Métrica | Criterio inicial |
| --- | --- |
| Tests locales | `100% tests passed`. |
| `push_retries` | Idealmente `0`. Si crece, la ring o el consumidor no sostienen la tasa. |
| `sampling_interval_cv_pct` | Idealmente `< 5%` para muestreo estable. |
| `perf_running_ratio_min` | Cerca de `1.0`. Valores bajos indican multiplexación o presión de eventos. |
| `energy_delta_valid` | Debe ser `1` salvo primeras muestras o casos justificados. |
| Overhead medio | Meta experimental inicial: `< 2%` en workload objetivo. |
| Cgroup | Debe contener solo el workload durante la ventana medida. |

## Receta De Matriz Inicial

Para una primera toma pequeña:

1. Correr `stream_triad` con productor/consumidor sin pinning.
2. Correr `stream_triad` con productor/consumidor pinneados fuera del set del workload.
3. Repetir lo mismo con `gemm_naive`.
4. Usar 3 a 5 repeticiones por configuración.
5. Comparar `overhead_pct_mean`, `sampling_interval_cv_pct`, `push_retries` y `perf_running_ratio_min`.

Ejemplo de variante pinneada:

```bash
/tmp/tg-telemetry-build/telemetry_kernel_launcher \
  --kernel stream_triad \
  --size 100000000 \
  --iterations 20 \
  --warmup 2 \
  --threads 4 \
  --repetitions 5 \
  --perf-cpus 2,3,4,5 \
  --collector-cpu 6 \
  --consumer-cpu 7 \
  --cgroup-path /sys/fs/cgroup/<delegated-cgroup> \
  --interval-ns 1000000 \
  --rapl-pkg /sys/class/powercap/intel-rapl/intel-rapl:0 \
  --output-dir runs \
  --run-id stream_pinned_001
```

## Benchmarks Manuales

Estos targets se compilan pero no se registran como pruebas automáticas de
aceptación científica.

Overhead sintético:

```bash
/tmp/tg-telemetry-build/telemetry_overhead_bench --no-perf
```

Jitter sintético:

```bash
/tmp/tg-telemetry-build/telemetry_jitter_bench --samples 10000 --interval-ns 1000000
```

Ambos devuelven `0` aunque la medición sea mala; son herramientas exploratorias,
no tests unitarios.

## Errores Comunes

| Síntoma | Causa probable | Acción |
| --- | --- | --- |
| `--cgroup-path is required when perf is enabled` | Perf está activo y no se pasó cgroup. | Pasar `--cgroup-path` o usar `--no-perf`. |
| `--perf-cpus is required when perf is enabled` | Perf cgroup necesita CPUs explícitas. | Pasar lista real de CPUs. |
| `failed to open cgroup.procs` | Sin permisos o ruta incorrecta. | Usar cgroup delegado/escribible. |
| `perf_event_open ... failed` | Restricción de kernel/permisos o evento no disponible. | Revisar política PMU del nodo o probar `--no-perf`. |
| CSV sin filas CPU | Perf no abrió, no produjo muestras o se usó `--no-perf`. | Revisar `summary.txt`, permisos y `--perf-cpus`. |
| `perf_running_ratio_min` bajo | Multiplexación o demasiados eventos/CPUs. | Reducir eventos/alcance o revisar disponibilidad PMU. |
| `push_retries > 0` | Consumidor no drena lo suficientemente rápido o ring se llena. | Pinnear consumidor, bajar frecuencia o ampliar diseño. |
| `energy_delta_valid=0` repetido | Primera muestra, wrap sin rango o path RAPL incompleto. | Verificar `max_energy_range_uj`. |
| Workload falla por memoria | `--size` demasiado grande, especialmente `gemm_naive` o `stencil_2d`. | Reducir `--size`. |

## Limitaciones Actuales

- El runner multihilo v1 mide CPU; GPU queda pospuesto.
- `--perf-cpus` no pinnea el workload.
- `--workload-cpus` existe como alias de `--perf-cpus`, pero el nombre puede ser
  confuso.
- El launcher no crea cgroups ni modifica jerarquías globales.
- No hay todavía preflight exhaustivo del nodo.
- No hay todavía export por fases internas del kernel; cada ejecución es una
  ventana completa.
- No hay sweep automático de variantes de pinning.
- Los contadores perf son un conjunto fijo: instrucciones, ciclos, referencias
  de caché y misses de caché.

## Checklist Para Toma Real

Antes:

- Compilar en el nodo.
- Ejecutar CTest.
- Identificar cgroup delegado.
- Identificar CPUs permitidas del job.
- Identificar paths RAPL.
- Decidir si el workload migra libremente o se pinnea.
- Confirmar que collector/consumer no se ubican sobre CPUs del workload cuando
  se busca aislar overhead.

Durante:

- Guardar comando exacto.
- Guardar `samples.csv`, `metadata.json`, `summary.txt`.
- Registrar carga del nodo y política de scheduling/cpuset externa si aplica.

Después:

- Revisar `push_retries`.
- Revisar `sampling_interval_cv_pct`.
- Revisar `perf_running_ratio_min`.
- Revisar conteo de muestras.
- Revisar deltas RAPL.
- Comparar baseline vs telemetry por repetición.

