# Informe de arquitectura, estado y visión de la telemetría C++

> Documento de traspaso para otro modelo o desarrollador. Describe el estado observado en el workspace el **14 de julio de 2026**, en la rama `hpc-startup-diagnostic`, con `HEAD dc1047f`. El modo `--exec` del launcher y su documentación tienen cambios locales sin commit; por tanto, este informe describe el árbol de trabajo actual y no solamente `HEAD`.

## 1. Resumen ejecutivo

El subsistema C++17 de Hyperion es un **harness de bajo nivel para una corrida individual**. Su responsabilidad es lanzar un workload, muestrear contadores de CPU y energía con bajo overhead, transportar las muestras fuera de la ruta caliente y escribir tres artefactos: `samples.csv`, `metadata.json` y `summary.txt`.

Está compuesto por:

- una biblioteca estática `telemetry` con readers de `perf_event`, RAPL y NVML, un collector dedicado y una ring SPSC;
- `telemetry_kernel_launcher`, que controla procesos, cgroup, afinidad, ciclo de vida de productor/consumidor y exportación;
- `telemetry_kernel_workload`, que contiene cuatro kernels sintéticos solo para desarrollo y medición del propio harness;
- dos benchmarks manuales de overhead/jitter y nueve pruebas CTest.

El C++ **no es el orquestador de campaña**. No controla frecuencias, no conoce el manifest ni el catálogo formal, no calibra Roofline, no valida una campaña y no produce `windows.csv`. Esas responsabilidades corresponden a Python.

La ruta sintética está implementada y tiene una delimitación razonable de la ventana medida mediante un protocolo `ready/go`. El modo nuevo `--exec` permite invocar binarios externos, compila y ejecuta, pero todavía debe considerarse **provisional**: no tiene una barrera previa al `exec`, de modo que el hijo puede comenzar antes de que el padre termine de moverlo al cgroup e iniciar la telemetría.

La columna `label` que exporta el launcher es histórica/descriptiva. **Nunca debe convertirse en `phase_label_train`**. La etiqueta de entrenamiento se deriva posteriormente, por ventana, comparando intensidad operacional contra el `i_ridge` de la calibración Roofline de la sesión.

## 2. Frontera de responsabilidades

```text
orquestador Python (campaña, todavía incompleto)
  manifest + environment + preflight + freqctl + catálogo
  calibración Roofline + node profile
  runner de una combinación
        |
        | invoca CLI
        v
telemetry_kernel_launcher (C++)
  proceso/cgroup/afinidad + collector + consumidor + exportación
        |
        +-- modo sintético --> telemetry_kernel_workload
        |
        +-- modo externo ---> binario real (NPB/STREAM/ERT) con --exec
        |
        v
  samples.csv + metadata.json + summary.txt
        |
        v
postprocess Python
  deltas + ventanas + calidad + Roofline
        |
        v
  windows.csv con phase_label_train
```

El launcher debe permanecer agnóstico a la campaña. El orquestador es quien debe:

- resolver y verificar el binario desde `kernels/catalog.yaml`;
- realizar preflight y aplicar/restaurar frecuencia solo sobre CPUs delegadas;
- fusionar metadata del launcher con identidad de nodo, checksum, calibración y frecuencia;
- aceptar o rechazar la corrida sin borrar su evidencia;
- transformar muestras acumuladas en ventanas y calcular la etiqueta empírica.

En el workspace actual, `orchestrator/runner.py`, `orchestrator/metadata_schema.py` y `orchestrator/validation.py` existen pero están vacíos. El catálogo contiene rutas y checksums de ejemplo. Por ello, el pipeline completo C++ → campaña Python **todavía no está operativo de extremo a extremo**, aunque hay otros módulos Python parcialmente implementados.

## 3. Mapa de componentes

```text
telemetry/
  CMakeLists.txt
  include/telemetry/
    metrics.hpp              estructuras fijas CPU/ENERGY/GPU
    spsc_ring.hpp            cola lock-free single producer/single consumer
    perf_reader.hpp          perf simple por PID/CPU
    perf_cgroup_reader.hpp   perf de cgroup, abierto por cada CPU
    rapl_reader.hpp          snapshots de energía desde sysfs
    nvml_reader.hpp          potencia/utilización GPU opcional
    collector.hpp            configuración y ciclo de vida del productor
    experiment_utils.hpp     CLI, estadísticas, JSON y deltas RAPL
  src/
    collector.cpp
    perf_reader.cpp
    perf_cgroup_reader.cpp
    rapl_reader.cpp
    nvml_reader.cpp
    experiment_utils.cpp
    spsc_ring.cpp             vacío; SPSCRing es template en el header
  experiments/
    telemetry_kernel_launcher.cpp
    telemetry_kernel_workload.cpp
  benchmarks/
    telemetry_overhead_bench.cpp
    telemetry_jitter_bench.cpp
  tests/
    test_*.cpp
```

Targets CMake:

| Target | Papel |
|---|---|
| `telemetry` | Biblioteca estática con readers, collector y utilidades. |
| `telemetry_kernel_launcher` | Harness principal de una corrida. |
| `telemetry_kernel_workload` | Workload sintético controlable mediante `ready/go`. |
| `telemetry_overhead_bench` | Benchmark exploratorio; no impone umbral de aprobación. |
| `telemetry_jitter_bench` | Benchmark exploratorio de jitter de muestras CPU. |
| nueve targets `*_test` | Pruebas unitarias/smoke portables. |

La construcción requiere Linux, CMake ≥ 3.18, C++17 y `pthread`. Usa `-Wall -Wextra -O2 -march=native`; el build por defecto es `RelWithDebInfo`. NVML se habilita expresamente con `-DWITH_GPU=ON`.

## 4. Arquitectura de la biblioteca `telemetry`

### 4.1 Modelo de muestra

`telemetry::Sample` es una unión etiquetada, de tamaño fijo y sin ownership dinámico:

```text
Sample
  tag = CPU    -> CpuSample
  tag = ENERGY -> EnergySnapshot
  tag = GPU    -> GpuSample
```

Todos los timestamps usan `CLOCK_MONOTONIC` en nanosegundos.

`CpuSample` contiene contadores acumulados de instrucciones, ciclos, referencias y fallos de caché, además de `time_enabled_ns` y `time_running_ns`. Estos últimos permiten evaluar multiplexación. Los conteos se escalan con `time_enabled / time_running` cuando perf no tuvo el evento programado durante todo el intervalo.

`EnergySnapshot` contiene lecturas acumuladas RAPL en microjoules para package y, opcionalmente, DRAM. El collector no calcula potencia ni deltas.

`GpuSample` contiene potencia en mW y utilización global del dispositivo. No atribuye actividad a un kernel o proceso CUDA.

### 4.2 Readers

#### `PerfReader`

Abre un grupo perf simple para un PID/CPU. Mide solo espacio de usuario (`exclude_kernel=1`, `exclude_hv=1`) y no hereda hijos. Es apropiado para smoke o procesos sencillos, pero no es la ruta principal del launcher multihilo.

#### `PerfCgroupReader`

Es el backend CPU principal. Abre un grupo con `PERF_FLAG_PID_CGROUP` en cada CPU de `perf_cpus` y suma los resultados de todos los grupos en una sola `CpuSample`.

Eventos actuales, en orden contractual:

1. `PERF_COUNT_HW_INSTRUCTIONS`;
2. `PERF_COUNT_HW_CPU_CYCLES`;
3. `PERF_COUNT_HW_CACHE_REFERENCES`;
4. `PERF_COUNT_HW_CACHE_MISSES`.

El cgroup decide **qué tareas** se cuentan; `perf_cpus` decide **en qué CPUs** se observan. `perf_cpus` no fija afinidad. Si el workload puede migrar a una CPU que no está en la lista, parte de su actividad quedará fuera.

Los eventos se habilitan al abrir el reader y los valores de cada muestra son acumulados desde ese momento. Los deltas pertenecen al postprocesamiento.

#### `RaplReader`

Abre `energy_uj` una sola vez y en cada lectura hace `lseek` + `read`. Package es obligatorio si se configura RAPL; DRAM es opcional. También intenta leer `max_energy_range_uj`.

Los deltas se calculan fuera del productor mediante `next_rapl_delta()`. El estado se reinicia al cambiar de repetición y maneja wrap solo si conoce un rango máximo válido. La primera muestra de cada repetición se marca como delta inválido.

RAPL mide un dominio físico global —normalmente el package—, no el consumo exclusivo del proceso. La limpieza del nodo y el aislamiento experimental son necesarios para interpretar la energía.

#### `NvmlReader`

En builds CPU-only conserva la API, pero `open()` falla explícitamente. Con NVML inicializa el dispositivo indicado y lee potencia/utilización. Actualmente el launcher no ofrece un flag que ponga `CollectorConfig.enable_gpu=true`; por tanto, la capacidad existe en la biblioteca pero **no está conectada a la ruta experimental principal**.

### 4.3 Collector y ruta caliente

`Collector` recibe una `CollectorConfig` y una ring propiedad del llamador. En `start()` abre solo los backends habilitados y crea un `pthread` productor, opcionalmente fijado a `producer_cpu`.

En cada tick, el productor:

1. lee perf si está abierto y empuja una muestra CPU;
2. lee RAPL y empuja una muestra ENERGY;
3. lee NVML si fue compilado y habilitado;
4. publica el batch con `flush_producer()`;
5. espera el siguiente instante absoluto con `clock_nanosleep(..., TIMER_ABSTIME, ...)`.

No escribe archivos ni calcula IPC, MPKI, potencia o intensidad operacional. Esas operaciones se mantienen fuera de la ruta caliente.

Si la ring está llena, `try_push()` se reintenta en un bucle y se incrementa `push_retries`. No se descarta voluntariamente la muestra, pero el busy-wait altera el timing; metodológicamente una corrida con `push_retries > 0` debe rechazarse (`I04`).

`stop()` es idempotente en uso normal: marca stop, hace `join` y cierra todos los readers. El destructor lo invoca de nuevo de forma segura.

### 4.4 Ring SPSC

`SPSCRing<T,N>` es un template lock-free para exactamente un productor y un consumidor. Mantiene índices producer/consumer en líneas de caché separadas, usa capacidad potencia de dos y publica índices por lotes para reducir tráfico de coherencia.

La ring del collector se instancia con `RING_CAPACITY = 16384`; al reservar una posición para distinguir lleno/vacío, su capacidad utilizable efectiva es 16383 elementos. El productor y consumidor deben llamar sus respectivos `flush_*`.

### 4.5 Utilidades experimentales

`experiment_utils` implementa:

- parsing/formato de listas como `2,4-6`, rechazando duplicados;
- media, desviación estándar poblacional y coeficiente de variación;
- overhead respecto a baseline;
- escape básico para JSON;
- mapeo histórico de kernels sintéticos a labels;
- deltas RAPL válidos por repetición y con wrap.

## 5. Arquitectura del launcher

### 5.1 Recursos que posee

Por ejecución hija, el launcher posee:

- el proceso hijo y sus pipes;
- la escritura del PID en un cgroup ya creado/delegado;
- la afinidad opcional del proceso;
- una ring SPSC local;
- un `Collector` productor;
- un `std::thread` consumidor;
- un `vector<RecordedSample>` en memoria;
- la exportación posterior a disco.

El consumidor drena por batches, duerme 100 µs entre intentos y acumula muestras en memoria. La reserva del vector se hace antes de empezar a medir para reducir reallocations, pero en corridas largas puede ser insuficiente y el vector sí podría crecer durante la medición.

### 5.2 Modo sintético: flujo vigente

Por repetición se ejecutan secuencialmente dos hijos:

```text
baseline (sin collector) -> telemetry (con collector)
```

El protocolo de sincronización es:

```text
launcher                         telemetry_kernel_workload
   | fork/exec                              |
   | hijo aplica afinidad antes de exec     |
   | mueve PID al cgroup                    |
   |                                        | reserva memoria
   |                                        | crea pool de threads
   |                                        | ejecuta warmup
   |<--------------- 'R' ready -------------|
   | inicia consumer + collector             | bloqueado
   |---------------- 'G' go ---------------->|
   |                                        | mide kernel->run(iterations)
   |<--------- stdout elapsed_ns=... --------|
   | espera hijo, detiene y drena            |
```

La afinidad del proceso se aplica en el hijo antes de `execv` y el padre mueve su PID al cgroup antes de esperar `ready`. El hijo no entra en la ventana medida hasta recibir `go`. Setup, asignaciones, creación del pool y warmup quedan fuera del `elapsed_ns` reportado por el workload. Baseline y telemetry usan el mismo mecanismo; solo telemetry inicia collector y consumidor.

El overhead por repetición es `(telemetry - baseline) / baseline * 100`. El orden siempre es baseline primero y telemetry después; no está aleatorizado.

### 5.3 Workload sintético

`telemetry_kernel_workload` crea un pool fijo antes de medir y divide rangos contiguos de forma estática:

| Kernel | Label histórico | Patrón |
|---|---|---|
| `stream_triad` | `memory_bound` | triad lineal sobre tres vectores |
| `reduction` | `memory_bound` | reducción paralela de lectura secuencial |
| `stencil_2d` | `cache_sensitive` | stencil de cuatro vecinos sobre dos grillas |
| `gemm_naive` | `compute_bound` | multiplicación de matrices ingenua |

Estos kernels son herramientas de desarrollo y pruebas del harness. **No son la fuente del dataset final** y sus labels no son verdad de entrenamiento.

### 5.4 Modo externo `--exec`: implementación actual

El árbol local agrega:

```bash
telemetry_kernel_launcher \
  --exec /ruta/al/binario \
  --exec-args "argumentos separados por espacios" \
  ...
```

En este modo:

- `execv()` reemplaza al hijo por el binario externo;
- no se lanza `telemetry_kernel_workload`;
- no existe baseline;
- `size`, `iterations`, `warmup` y `threads` se exportan como cero;
- `pin_workers` se ignora;
- si no se da `--kernel`, se usa el basename del ejecutable como identificador;
- los argumentos se separan por whitespace, sin quoting ni escaping de shell;
- la duración se toma con el reloj monotónico del padre;
- stdout se captura; stderr queda heredado.

`execv` no busca en `PATH`: se necesita una ruta ejecutable válida. La metadata aún no conserva `exec_path`, `exec_args`, comando completo ni checksum.

#### Riesgo crítico de sincronización

El hijo externo no espera una señal del padre antes de `execv()`. Padre e hijo corren en paralelo después de `fork()`, así que el binario puede empezar —o terminar si es muy corto— antes de:

- entrar al cgroup;
- abrir los eventos perf;
- iniciar el collector;
- establecer el inicio de la ventana de pared.

Además, `wall_start_ns` se toma antes de `collector.start()`, por lo que la duración incluye el costo de apertura de backends/creación del productor, pero puede omitir el fragmento inicial que el hijo ejecutó durante la carrera. El comentario actual que afirma que fork/exec/cgroup quedan fuera de la medición no está garantizado por el código.

La solución prevista debe ser una **barrera pre-exec controlada por el launcher**: el hijo se bloquea en un pipe inmediatamente después del fork; el padre aplica cgroup/afinidad e inicia consumer/collector; luego registra el inicio, libera al hijo y este hace `execv`. No requiere que el binario externo conozca el protocolo.

### 5.5 Afinidad y cgroup

Hay cuatro knobs distintos:

| Knob | Efecto |
|---|---|
| `--perf-cpus` | CPUs donde se abren eventos perf de cgroup. |
| `--pin-workload-cpus` | Máscara de afinidad del proceso hijo. |
| `--collector-cpu` | CPU del productor. `-1` deja libre al scheduler. |
| `--consumer-cpu` | CPU del consumidor. `-1` deja libre al scheduler. |

`--pin-workers` solo aplica al pool sintético y fija un worker por CPU de `pin-workload-cpus`.

El launcher no crea el cgroup; solo escribe el PID en `<cgroup>/cgroup.procs`. Necesita un cgroup previamente delegado. Si perf está habilitado, `--cgroup-path` y `--perf-cpus` son obligatorios.

### 5.6 Persistencia

Los artefactos se escriben al terminar correctamente **todas** las repeticiones:

```text
<output-dir>/<run-id>/
  samples.csv
  metadata.json
  summary.txt
```

`samples.csv` es rectangular. Cada fila es CPU, ENERGY o GPU y los campos no aplicables quedan vacíos. Los contadores CPU y RAPL crudos son acumulados; ENERGY agrega deltas derivados y `energy_delta_valid`.

Campos relevantes de `metadata.json`:

- identidad y parámetros sintéticos;
- periodo, CPUs, afinidad y cgroup;
- tiempos baseline/telemetry y overhead;
- jitter calculado solo con timestamps CPU;
- mínimo `time_running / time_enabled`;
- `push_retries` total y por repetición;
- rangos y energía RAPL total válida;
- `samples_collected`.

En modo externo los arrays de baseline/overhead quedan vacíos, pero sus agregados aparecen como `0`; deben interpretarse como **no aplicables**, no como mediciones reales.

El launcher no escribe `windows.csv`, no decide `accepted`, no emite `factor_id` y no conoce `phase_label_train`.

## 6. Estado verificado el 14 de julio de 2026

Se ejecutó desde el workspace actual:

```bash
cmake -S telemetry -B /tmp/hyperion-telemetry-report-build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build /tmp/hyperion-telemetry-report-build -j2
ctest --test-dir /tmp/hyperion-telemetry-report-build --output-on-failure
```

Resultado: compilación correcta y **9/9 pruebas CTest aprobadas**.

Cobertura efectiva de estas pruebas:

- enlace básico y estados iniciales;
- orden/integridad concurrente de la ring SPSC;
- ciclo de vida e idempotencia básica del collector;
- collector sin perf y detección de backpressure con RAPL simulado;
- escala aritmética de perf y fallos de configuración de cgroup;
- parsing y lectura RAPL con sysfs temporal;
- parsing de CPU lists, estadísticas y deltas RAPL por repetición;
- stub NVML de un build CPU-only.

También se realizaron dos smokes con `--no-perf` y sin RAPL:

- sintético: 2 repeticiones de `stream_triad`, completó y generó artefactos;
- externo: 2 repeticiones de `/bin/sleep 0.05`, completó y generó artefactos.

Ambos reportaron `samples_collected=0`, como corresponde al no haber ningún backend activo. Son pruebas de mecánica, **no corridas válidas de dataset**.

No se verificó en esta sesión:

- perf real con cgroup delegado;
- RAPL físico;
- NVML/GPU;
- afinidad real en un layout experimental;
- overhead/jitter bajo carga controlada;
- ejecución NPB/STREAM/ERT real;
- prueba de integración C++/orquestador;
- interrupciones/señales a mitad de corrida.

## 7. Limitaciones y riesgos abiertos

### Prioridad alta

1. **Carrera en `--exec`.** El hijo externo puede correr antes del setup del padre. Debe corregirse antes de usar binarios reales para dataset.
2. **Semántica de `label`.** Para sintéticos contiene un prior hardcodeado; para externos cae al nombre del binario. No es `phase_label_hint` formal ni `phase_label_train`. Conviene renombrarlo o marcarlo explícitamente como legado.
3. **Metadata insuficiente para reproducibilidad externa.** Faltan modo, `exec_path`, argv estructurado, comando, PID/exit/signal, checksum, timestamps de ventana y estado explícito de cada backend.
4. **Fallas sin artefactos.** Si una repetición falla, `main()` retorna antes de crear el directorio y exportar evidencia parcial. Esto no satisface por sí solo el principio de conservar corridas rechazadas.
5. **Sin manejo explícito de señales del launcher.** Una interrupción puede dejar al hijo vivo o artefactos ausentes; esto debe probarse y endurecerse.
6. **Orquestación aún desconectada.** El runner/metadata/validation Python están vacíos y el catálogo aún usa placeholders.

### Prioridad media

1. Un error transitorio de lectura RAPL se transforma en valor `0`, pero `read()` devuelve éxito. Con rango máximo conocido podría parecer un wrap válido y contaminar deltas.
2. `PERF_COUNT_HW_CACHE_MISSES` es un evento genérico. Antes de usarlo como aproximación de bytes movidos hay que validar en cada microarquitectura qué nivel representa y documentar el tamaño de línea usado.
3. RAPL y NVML son globales al dominio/dispositivo; no atribuyen energía exclusivamente al workload.
4. CPU, ENERGY y GPU se leen secuencialmente, con timestamps distintos dentro del mismo tick. No existe un `sample_group_id` común.
5. El collector ignora el retorno de `clock_nanosleep`; no registra deadlines perdidos ni saltos de tick.
6. Si la reserva de `samples` queda corta, el consumidor puede realocar memoria durante la ventana.
7. Los benchmarks manuales retornan código cero incluso si el collector no abre o el overhead es alto; son diagnósticos, no gates automáticos.
8. Jitter se calcula únicamente a partir de filas CPU. Una corrida RAPL-only obtiene jitter cero, que significa “no medido”, no “sin jitter”.
9. El writer CSV no escapa comas/comillas en `run_id` o `kernel`; el writer JSON es manual y solo escapa un subconjunto básico.
10. El orden baseline → telemetry es fijo y puede introducir sesgo térmico/temporal en la estimación de overhead.

### Fuera de alcance inmediato

- atribución GPU por kernel (requeriría CUDA/NVTX/CUPTI o un diseño equivalente);
- control DVFS desde C++;
- clasificación ML en línea;
- campañas multinodo. La capa de `node_id`/perfil/calibración debe prepararse en Python, pero no se debe lanzar un segundo nodo sin decisión explícita.

## 8. Visión objetivo

La visión de Fase 1 es que el C++ sea un **sensor/harness pequeño, determinista y auditable**, mientras Python sea el **control plane experimental**.

### Etapa A: cerrar el contrato del launcher

- agregar barrera pre-exec y tests específicos de orden de eventos;
- representar modo sintético/externo explícitamente en metadata;
- exportar argv como array JSON, checksum recibido o identidad verificable del binario, códigos de salida y timestamps;
- producir metadata parcial aun en fallo o señal;
- distinguir `null`/no aplicable de cero en métricas externas;
- eliminar cualquier ambigüedad entre `label` legado, `phase_label_hint` y `phase_label_train`;
- verificar errores de apertura/escritura de los tres artefactos.

### Etapa B: validar adquisición real

- perf cgroup real sobre CPUs delegadas y un workload multihilo;
- RAPL package/DRAM con wrap y contaminación controlada;
- medición de `push_retries`, jitter y multiplexación con criterios de rechazo;
- prueba de interrupción del launcher y limpieza del hijo;
- campañas piloto locales con sintéticos y luego NPB clase S.

### Etapa C: conectar el orquestador

- implementar `runner.py` para resolver `KernelEntry`, invocar `--exec` y fusionar metadata;
- verificar checksum antes de ejecutar;
- añadir preflight reducido por corrida y frecuencia aplicada/observada;
- conservar aceptadas y rechazadas con `factor_id`;
- postprocesar `samples.csv` a `windows.csv` usando deltas y banderas de calidad;
- calcular `phase_label_train` exclusivamente con intensidad operacional versus `i_ridge` de la sesión.

### Etapa D: endurecimiento científico

- validar que los eventos perf elegidos soportan la estimación de bytes movidos;
- cuantificar overhead y perturbación de productor/consumidor por configuración de CPU;
- versionar esquema de artefactos;
- registrar versiones de kernel, compilador, launcher y hardware;
- establecer gates automáticos y una prueba piloto bare-metal antes de SC3.

## 9. Invariantes que el siguiente modelo debe preservar

1. No escribir a disco, loggear, bloquear con mutex ni calcular features en `Collector::run()`.
2. Mantener `Sample` fijo y sin ownership dinámico en la ring.
3. Respetar exactamente un productor y un consumidor por `SPSCRing`.
4. No confundir alcance perf (`perf_cpus`) con afinidad del workload.
5. No asumir que RAPL/NVML pertenecen exclusivamente al proceso.
6. No usar kernels sintéticos para el dataset final.
7. No hardcodear binarios externos en C++; deben venir del catálogo y verificarse antes.
8. No copiar `label` ni `phase_label_hint` a `phase_label_train`.
9. No ocultar backpressure: `push_retries > 0` es evidencia de una corrida inválida.
10. No declarar listo `--exec` hasta eliminar y probar la carrera pre-exec.
11. No mover control de frecuencia al launcher; `freqctl.py` debe limitarse a CPUs delegadas y restaurar siempre.
12. No considerar una prueba mock como sustituto de las pruebas bare-metal de perf, RAPL, DVFS e interrupción.

## 10. Guía rápida para continuar

Archivos que se deben leer primero:

1. `AGENTS.md`;
2. `telemetry/include/telemetry/collector.hpp` y `telemetry/src/collector.cpp`;
3. `telemetry/experiments/telemetry_kernel_launcher.cpp`;
4. `telemetry/experiments/telemetry_kernel_workload.cpp`;
5. `telemetry/docs/decisiones_tecnicas.md` y `telemetry/docs/guia_total_uso.md`;
6. documentos del orquestador en `docs/orchestator/agents/`.

Comandos mínimos después de modificar C++:

```bash
cmake -S telemetry -B /tmp/hyperion-telemetry-build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build /tmp/hyperion-telemetry-build -j2
ctest --test-dir /tmp/hyperion-telemetry-build --output-on-failure
```

Tests que faltan específicamente para el launcher:

- parseo y exclusiones mutuas de CLI;
- `--exec-args` y paths inválidos;
- barrera pre-exec/cgroup/collector mediante un hijo instrumentado;
- salida externa abundante, señal y código de error;
- artefactos parciales y JSON válido en fallo;
- separación correcta entre repeticiones para jitter y RAPL;
- modo externo sin baseline representado como no aplicable;
- smoke con cgroup/perf real en hardware autorizado.

## 11. Conclusión

La biblioteca C++ tiene una separación de responsabilidades sólida: readers especializados, muestra POD, productor dedicado, ring SPSC y exportación fuera de la ruta caliente. La ruta sintética es útil para desarrollar y medir el propio harness, y la base de perf cgroup/RAPL es adecuada para una plataforma CPU multihilo de un nodo.

El siguiente hito no es añadir más kernels internos ni empezar el clasificador. Es **cerrar y validar el contrato de ejecución de binarios externos**, completar su trazabilidad y conectarlo al orquestador. Solo después de corregir la sincronización de `--exec`, validar adquisición real y consolidar los artefactos se debe considerar que el subsistema está listo para generar dataset científico.
