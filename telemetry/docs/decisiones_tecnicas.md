# Decisiones Técnicas Del Subsistema Modular De Telemetría

## Objetivo De Diseño

El subsistema se diseñó para recolectar métricas de kernels HPC representativos
con el menor overhead posible y con suficiente control experimental para crear
un dataset útil para clasificación de fases de cómputo. La meta no es solo medir
tiempo total, sino obtener series temporales de contadores CPU, energía y, en
fases futuras, métricas GPU, preservando trazabilidad entre parámetros,
condiciones de ejecución y salida exportada.

Las decisiones técnicas priorizan:

- bajo overhead en la ruta de muestreo;
- separación entre workload y telemetría;
- soporte para CPU multihilo;
- datos exportables y reproducibles;
- pruebas unitarias locales sin depender del nodo final;
- extensibilidad para GPU en una fase posterior.

## Resumen De Decisiones

| Decisión | Justificación breve |
| --- | --- |
| Arquitectura `launcher + proceso hijo` | Evita mezclar el workload HPC con la lógica de orquestación, consumo y exportación. |
| Perf por cgroup y CPU explícita | Permite medir procesos multihilo e hijos sin depender de `inherit=1`. |
| `--perf-cpus` separado de pinning | Diferencia alcance de medición de política de scheduling. |
| Baseline y telemetry como ejecuciones separadas | Permite estimar overhead comparando ejecuciones equivalentes. |
| Productor y consumidor separados | La ruta caliente solo captura y empuja muestras; exportación queda fuera. |
| Ring SPSC lock-free | Evita mutexes entre productor y consumidor. |
| Backends configurables | Permite tests sin PMU, sin RAPL o sin GPU. |
| RAPL como snapshots crudos | El productor no calcula deltas ni hace trabajo derivado. |
| Delta RAPL en exportación | Permite manejar overflow sin cargar la ruta caliente. |
| `CLOCK_MONOTONIC` y sleep absoluto | Reduce discontinuidades y deriva en timestamps. |
| GPU pospuesto | NVML no atribuye por sí solo kernels CUDA asíncronos. |

## Arquitectura `Launcher + Proceso Hijo`

La primera idea de ejecutar kernels, consumidor y launcher dentro del mismo
proceso no escala bien para kernels HPC multihilo. Un workload real puede crear
varios hilos, usar librerías externas y eventualmente lanzar kernels GPU
asíncronos. Si la telemetría vive dentro del mismo flujo de ejecución, es más
difícil separar:

- tiempo de setup;
- tiempo medido del kernel;
- overhead del collector;
- lógica de consumo;
- persistencia de resultados.

Por eso se adoptó un modelo con:

- `telemetry_kernel_launcher`: proceso padre que orquesta.
- `telemetry_kernel_workload`: proceso hijo que ejecuta kernels.

El launcher ejecuta dos hijos por repetición:

- baseline sin collector activo;
- telemetry con collector y consumidor activos.

Esta separación permite que el workload se comporte como una carga HPC
representativa mientras el launcher conserva control sobre cgroup, afinidad,
inicio de ventana medida y exportación.

## Handshake De Inicio Medido

El workload crea su thread pool, reserva memoria y ejecuta warmup antes de
enviar la señal de listo al launcher. Luego espera una señal `go`. El launcher
activa el collector antes de enviar `go` en la ejecución telemetry.

Esta decisión evita medir:

- asignaciones de memoria;
- creación de hilos;
- inicialización del kernel;
- warmup.

La ventana medida se aproxima mejor al comportamiento estable del kernel.

## Baseline Y Telemetry Separados

El overhead se calcula como:

```text
overhead_pct = 100 * (telemetry_elapsed_ns - baseline_elapsed_ns) / baseline_elapsed_ns
```

Usar ejecuciones separadas permite comparar el mismo workload con y sin
telemetría. Esta estrategia no elimina toda variabilidad del sistema, pero es
simple, auditable y adecuada para comenzar. Por eso se usan repeticiones y se
exportan tanto valores individuales como medias y desviaciones.

Limitación: baseline y telemetry no ocurren exactamente al mismo tiempo, por lo
que ruido del nodo, frecuencia CPU, interferencia externa o NUMA pueden afectar
la comparación. Por eso el análisis debe usar varias repeticiones y condiciones
controladas.

## Uso De Cgroups Para Perf En CPU Multihilo

La medición CPU multihilo se basa en `perf_event_open` con
`PERF_FLAG_PID_CGROUP`. En este modo, el descriptor perf se asocia a un cgroup y
a una CPU. El contador mide tareas del cgroup cuando se ejecutan en esa CPU.

Razones para usar cgroups:

- un kernel multihilo puede crear varios threads;
- el proceso hijo puede tener dinámica interna;
- medir solo un PID con un `PerfReader` simple no representa bien el conjunto;
- `inherit=1` no es una solución robusta para todas las formas de hilos/hijos;
- cgroup permite atribuir trabajo a un contenedor lógico controlado por el
  launcher.

El lector `PerfCgroupReader` abre un grupo de eventos por CPU listada en
`--perf-cpus`. Luego suma las lecturas por CPU para producir una muestra CPU
agregada.

## Por Qué No Usar `inherit=1` Como Estrategia Principal

`inherit=1` puede parecer una solución simple para contar hilos hijos, pero se
descartó como base principal porque:

- puede no cubrir tareas ya existentes;
- complica la semántica con grupos de eventos;
- tiene restricciones con ciertos formatos de lectura;
- puede submedir workloads con hilos dinámicos;
- no resuelve el problema de atribución GPU;
- obliga a razonar desde el árbol de procesos, no desde el conjunto experimental
  que se desea medir.

El cgroup expresa mejor la unidad experimental: "todo lo que corre dentro de
este grupo durante esta ventana".

## `--perf-cpus` No Es Pinning

Se separaron deliberadamente dos conceptos:

- alcance de medición;
- política de scheduling.

`--perf-cpus` define dónde se abren eventos perf. No fuerza al workload a correr
en esas CPUs. Esto permite estudiar escenarios donde el kernel migra libremente,
siempre que `--perf-cpus` cubra el conjunto real de CPUs donde puede ejecutarse.

El pinning del workload se controla con `--pin-workload-cpus` y `--pin-workers`.
Esto permite comparar:

- workload libre;
- workload con afinidad de proceso;
- workload con workers fijados;
- productor/consumidor libres;
- productor/consumidor aislados.

## Core Pinning Del Productor Y Consumidor

El productor se pinnea con `CollectorConfig::producer_cpu`, configurado por
`--collector-cpu`. La afinidad se aplica en los atributos de `pthread_create`.

El consumidor se pinnea con `--consumer-cpu`, mediante
`pthread_setaffinity_np()` dentro del thread consumidor.

El objetivo del pinning no es forzar siempre la medición, sino permitir estudios
A/B:

- sin pinning: el scheduler decide dónde ubicar telemetría;
- con pinning: productor y consumidor se aíslan de las CPUs del workload.

Esto es importante para medir si la propia telemetría compite por caché, ciclos
o ancho de banda con el kernel.

## Ruta Caliente Del Productor

La ruta caliente vive en `Collector::run()`. La regla de diseño es mantenerla lo
más pequeña posible. Durante el loop de muestreo se permite:

- `clock_gettime`;
- lecturas de perf/RAPL/NVML ya abiertas;
- `lseek` y `read` para RAPL;
- `ring.try_push`;
- `ring.flush_producer`;
- `clock_nanosleep` con timer absoluto.

Se evita en el loop:

- logging;
- construcción de strings;
- `std::vector` growth;
- asignaciones dinámicas;
- escritura CSV/JSON;
- locks;
- fstreams;
- excepciones;
- cálculos derivados como IPC, miss ratio o deltas energéticos.

Las operaciones costosas se ubican en `open()`, configuración, consumidor o
exportación.

## Ring SPSC Lock-Free

El productor y el consumidor se conectan con una ring SPSC
`SPSCRing<Sample, RING_CAPACITY>`.

Razones:

- solo hay un productor y un consumidor;
- no se necesitan mutexes;
- se reducen invalidaciones de caché;
- los índices se separan por líneas de caché para mitigar false sharing;
- la capacidad es potencia de dos para usar máscara bitwise en el wrap.

La ring usa actualizaciones batched de índices compartidos. Esto reduce tráfico
de coherencia porque no se publica cada cambio de índice inmediatamente.

El contador `push_retries` se incrementa cuando `try_push()` falla porque la ring
está llena. En una corrida limpia, se espera `push_retries == 0`.

## Formato Unificado De Muestra

Las muestras se representan con:

- `SampleTag::CPU`;
- `SampleTag::ENERGY`;
- `SampleTag::GPU`.

El struct `Sample` usa una unión etiquetada para evitar asignaciones dinámicas y
mantener tamaño fijo. Esto simplifica el paso por la ring y evita objetos con
gestión compleja de memoria en la ruta caliente.

La exportación a CSV usa una forma rectangular: una misma cabecera con campos
vacíos para columnas no aplicables. Esta decisión facilita ingesta posterior en
Python, pandas, R o pipelines de ML.

## Decisión Sobre Perf Events

Los contadores CPU actuales son:

- `PERF_COUNT_HW_INSTRUCTIONS`;
- `PERF_COUNT_HW_CPU_CYCLES`;
- `PERF_COUNT_HW_CACHE_REFERENCES`;
- `PERF_COUNT_HW_CACHE_MISSES`.

Se usan grupos de eventos para leer una muestra coherente. El líder es
`instructions`. El formato incluye:

- `PERF_FORMAT_GROUP`;
- `PERF_FORMAT_TOTAL_TIME_ENABLED`;
- `PERF_FORMAT_TOTAL_TIME_RUNNING`.

`time_enabled` y `time_running` se usan para diagnosticar multiplexación. Cuando
`time_running` difiere de `time_enabled`, los conteos se escalan con:

```text
scaled = raw * time_enabled / time_running
```

Además, `perf_running_ratio_min = min(time_running / time_enabled)` se exporta
como indicador de confianza. Si se aleja de `1.0`, puede haber multiplexación o
presión sobre PMU. El `time_enabled`/`time_running` que cada muestra conserva
para este diagnóstico corresponde al evento con la razón más baja entre los
diez contadores abiertos en esa lectura, no siempre al mismo contador --
antes de esta corrección solo se conservaba el de `instructions`, así que un
sub-evento de `FP_ARITH_INST_RETIRED` podía multiplexarse sin que el
indicador lo reflejara mientras `instructions` se mantuviera en razón 1.0.

Los eventos excluyen kernel e hipervisor:

- `exclude_kernel = 1`;
- `exclude_hv = 1`.

Esto enfoca la medición en espacio de usuario y reduce ruido, aunque implica que
trabajo del kernel asociado al proceso no se contabiliza.

## PerfReader Simple Vs PerfCgroupReader

`PerfReader` mide un PID/CPU simple y es útil para tests, smoke local o procesos
single-thread.

`PerfCgroupReader` es la ruta recomendada para el launcher multihilo. No
reemplaza al lector simple porque ambos cumplen roles distintos:

- `PerfReader`: integración sencilla y pruebas básicas.
- `PerfCgroupReader`: medición experimental multihilo por cgroup y CPUs.

Esta separación evita sobrecargar un lector simple con semántica que no le
corresponde.

## Decisión Sobre RAPL

RAPL expone contadores de energía acumulada en `energy_uj`. Estos contadores
pueden envolver cuando alcanzan `max_energy_range_uj`.

La decisión fue guardar snapshots crudos en el productor:

- `pkg_uj`;
- `dram_uj`;
- timestamp.

El productor no calcula energía consumida porque hacerlo agregaría lógica
derivada al loop de muestreo. El cálculo de delta se realiza durante la
exportación:

- si `current >= previous`, delta normal;
- si `current < previous` y existe `max_energy_range_uj`, delta wrap-aware;
- si hay wrap sin rango máximo legible, el delta se marca inválido.

El CSV exporta:

- lectura cruda;
- delta;
- `energy_delta_valid`.

Esto preserva información original y evita inventar energía falsa cuando falta
el rango para resolver overflow.

## `energy_delta_valid`

Este campo existe porque no todos los pares de muestras producen un delta
confiable.

Casos inválidos:

- primera muestra ENERGY de cada repetición;
- wrap detectado sin `max_energy_range_uj` legible;
- configuración incompleta de dominio energético.

Esta decisión es importante para dataset ML: es mejor marcar un dato como no
válido que entrenar modelos con energía derivada incorrecta.

## `CLOCK_MONOTONIC` Y Sleep Absoluto

Los timestamps usan `CLOCK_MONOTONIC` para evitar saltos por ajustes de reloj de
sistema. El productor duerme con `clock_nanosleep(..., TIMER_ABSTIME, ...)`.

Motivo:

- un sleep relativo acumula deriva;
- un sleep absoluto intenta mantener una cadencia nominal;
- la variación restante se puede cuantificar con `sampling_interval_cv_pct`.

Esto no garantiza tiempo real estricto, pero proporciona una base mejor para
medir jitter.

## Preasignación En Workload Y Launcher

El workload reserva memoria y crea workers antes de la ventana medida. Esto
reduce contaminación de la medición por:

- page faults iniciales;
- asignaciones dinámicas;
- creación de hilos;
- inicialización de estructuras.

El launcher también reserva espacio para muestras estimando la duración de
baseline y el intervalo de muestreo. Esto evita crecimiento repetido del vector
de muestras durante el drenado.

## Decisión Sobre Exportación Posterior

La escritura de CSV, JSON y summary ocurre después de terminar la ventana medida.
La razón es directa: la persistencia a disco es lenta, variable y puede introducir
ruido fuerte.

Por eso:

- productor captura;
- consumidor drena a memoria;
- exportación escribe archivos al final.

Este patrón mantiene el disco fuera del camino crítico.

## GPU Pospuesto

NVML mide estado del dispositivo, no una región exacta de un kernel CUDA. Además,
los kernels CUDA son asíncronos: lanzar un kernel no significa que terminó ni que
la ventana CPU coincida con la ejecución GPU.

Por eso GPU se pospuso. Una fase GPU seria debe incluir:

- sincronización explícita;
- delimitación de fases;
- CUDA events;
- NVTX o CUPTI si se requiere atribución más fina;
- política clara para solapamiento CPU/GPU.

Mantener GPU fuera de la v1 evita mezclar dos problemas grandes antes de cerrar
CPU multihilo con datos confiables.

## Tests Unitarios Y Benchmarks Manuales

Los tests CTest validan comportamiento determinista y local:

- ring SPSC;
- ciclo de vida del collector;
- collector sin perf;
- lectores básicos;
- parsing y utilidades;
- delta RAPL con overflow simulado;
- stub CPU-only de NVML.

Los benchmarks de overhead y jitter no se registran como tests automáticos
porque dependen de hardware, permisos, carga del sistema y política del nodo.
Devuelven `0` para no confundirse con pruebas unitarias.

Esta separación evita que una máquina local sin PMU o sin RAPL bloquee el
desarrollo.

## Criterios Técnicos De Aceptación

Para considerar una medición como confiable:

- CTest local pasa al 100%.
- `push_retries == 0`.
- `sampling_interval_cv_pct < 5%` como criterio inicial.
- `perf_running_ratio_min` cerca de `1.0`.
- `energy_delta_valid` mayoritariamente válido si RAPL está activo.
- cgroup contiene solo el workload durante la ventana medida.
- productor y consumidor no compiten con workload cuando se evalúa modo
  pinneado.
- overhead medio cercano al objetivo experimental, inicialmente `< 2%` para
  workloads representativos.

## Riesgos Técnicos Restantes

| Riesgo | Impacto | Mitigación |
| --- | --- | --- |
| Cgroup con procesos ajenos | Contadores contaminados. | Preflight y cgroup dedicado. |
| Workload migra fuera de `--perf-cpus` | Submedición de contadores. | Alinear cpuset real y lista perf. |
| Multiplexación PMU | Conteos escalados menos confiables. | Revisar `perf_running_ratio_min`. |
| `push_retries > 0` | Pérdida o bloqueo del productor. | Pinning, bajar frecuencia o redimensionar. |
| RAPL sin rango máximo | Deltas inválidos ante overflow. | Usar dominios con `max_energy_range_uj`. |
| Variabilidad del nodo | Overhead ruidoso. | Repeticiones, aislamiento y registro ambiental. |
| GPU no atribuida | Datos GPU ambiguos. | Posponer hasta tener delimitación CUDA. |

## Decisiones A Justificar En El Trabajo De Grado

Para el texto del libro, las ideas centrales son:

1. La unidad experimental no es un hilo, sino un proceso de workload con posible
   paralelismo interno; por eso se usa launcher más proceso hijo.
2. La medición CPU multihilo requiere una frontera de atribución robusta; por eso
   se usa cgroup con eventos por CPU.
3. La telemetría debe perturbar lo menos posible; por eso el productor evita
   locks, asignaciones, logging y persistencia.
4. El consumidor y la exportación existen para sacar trabajo de la ruta caliente.
5. Los datos derivados se calculan después para preservar muestras crudas y
   permitir auditoría.
6. El overflow de RAPL se trata explícitamente para evitar datos energéticos
   falsos.
7. El pinning es una variable experimental, no una suposición fija.
8. GPU se excluye de la v1 para no producir atribución errónea con NVML.

Estas decisiones sostienen la validez del dataset: cada muestra debe ser
interpretable, cada corrida debe ser reproducible y cada fuente de overhead debe
estar identificada.

