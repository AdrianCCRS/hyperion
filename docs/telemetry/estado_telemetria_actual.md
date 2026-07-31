# Estado Actual Del Subsistema De Telemetria

Este documento resume el estado funcional actual del subsistema de telemetria de
Hyperion en la rama `fase-1/plataforma-experimental-simplificada`. Su objetivo
es servir como contexto de traspaso para disenar una plataforma experimental mas
sencilla encima de lo que ya existe.

## Alcance Actual

El repositorio contiene una implementacion C++17 enfocada en toma de datos CPU
multihilo en un unico nodo. La ruta vigente no es una plataforma completa de
campanas experimentales, sino un harness manual compuesto por:

- una libreria estatica `telemetry`;
- un launcher de medicion `telemetry_kernel_launcher`;
- un workload hijo `telemetry_kernel_workload`;
- benchmarks manuales de overhead/jitter;
- tests CTest locales sin depender de hardware real.

No existen actualmente, en esta rama, manifests formales, schemas JSON,
`campaign_type`, preflight Python, pipeline de dataset ni validador automatico
de calidad. La ruta formal de trabajo es ejecutar directamente el launcher.

El archivo raiz `telemetry_collection.cpp` debe tratarse como legado/MVP y no
como la ruta vigente de desarrollo.

## Mapa De Archivos

```text
telemetry/
  CMakeLists.txt

  include/telemetry/
    metrics.hpp              tipos fijos de muestra: CPU, ENERGY, GPU
    spsc_ring.hpp            ring SPSC lock-free
    collector.hpp            configuracion/API del productor de muestras
    perf_reader.hpp          lector perf simple por PID/CPU
    perf_cgroup_reader.hpp   lector perf por cgroup y CPU explicita
    rapl_reader.hpp          lector sysfs RAPL
    nvml_reader.hpp          lector NVML opcional
    experiment_utils.hpp     utilidades de CLI, stats y exportacion RAPL

  src/
    collector.cpp
    perf_reader.cpp
    perf_cgroup_reader.cpp
    rapl_reader.cpp
    nvml_reader.cpp
    experiment_utils.cpp
    spsc_ring.cpp

  experiments/
    telemetry_kernel_launcher.cpp
    telemetry_kernel_workload.cpp

  benchmarks/
    telemetry_overhead_bench.cpp
    telemetry_jitter_bench.cpp

  tests/
    test_*.cpp

hardware/
  check_node_readiness.py    diagnostico read-only del nodo
  capabilities.py
  probes.py
  schema.json
```

## Herramientas Y Dependencias

La construccion usa:

- CMake >= 3.18;
- C++17;
- `pthread`;
- Linux `perf_event_open`;
- sysfs RAPL, cuando se mide energia;
- NVML opcional con `-DWITH_GPU=ON`;
- CTest para pruebas locales.

Opciones de compilacion actuales:

```text
-Wall -Wextra -O2 -march=native
```

Si no se define `CMAKE_BUILD_TYPE`, CMake usa `RelWithDebInfo`.

## Targets De Build

| Target | Tipo | Uso |
| --- | --- | --- |
| `telemetry` | libreria estatica | Readers, collector, ring y utilidades. |
| `telemetry_kernel_launcher` | ejecutable | Orquesta baseline/telemetry, cgroup, collector, consumidor y exportacion. |
| `telemetry_kernel_workload` | ejecutable | Ejecuta kernels CPU multihilo bajo control del launcher. |
| `telemetry_overhead_bench` | benchmark manual | Explora overhead sintetico. No es criterio cientifico automatico. |
| `telemetry_jitter_bench` | benchmark manual | Explora jitter del productor. No es CTest obligatorio. |
| `*_test` | tests CTest | Validan piezas locales sin PMU/RAPL/GPU reales. |

## Construccion Y Pruebas

Flujo CPU-only recomendado:

```bash
cmake -S telemetry -B /tmp/tg-telemetry-build
cmake --build /tmp/tg-telemetry-build
ctest --test-dir /tmp/tg-telemetry-build --output-on-failure
```

Compilacion con NVML:

```bash
cmake -S telemetry -B /tmp/tg-telemetry-gpu-build -DWITH_GPU=ON
cmake --build /tmp/tg-telemetry-gpu-build
```

`WITH_GPU=ON` requiere `nvml.h`, `libnvidia-ml` y un driver NVIDIA funcional.
Aunque NVML compile, la ruta experimental principal sigue siendo CPU.

Verificacion realizada durante este traspaso:

```text
cmake -S telemetry -B /tmp/tg-telemetry-audit-build: OK
cmake --build /tmp/tg-telemetry-audit-build: OK
ctest --test-dir /tmp/tg-telemetry-audit-build --output-on-failure: 9/9 OK
```

Tambien se ejecuto un smoke sin perf:

```bash
/tmp/tg-telemetry-audit-build/telemetry_kernel_launcher \
  --kernel stream_triad \
  --size 10000 \
  --iterations 2 \
  --warmup 1 \
  --threads 2 \
  --repetitions 2 \
  --collector-cpu -1 \
  --consumer-cpu -1 \
  --no-perf \
  --output-dir /tmp/tg-telemetry-audit-smoke \
  --run-id smoke_no_perf
```

Resultado observado: el flujo termino correctamente y genero archivos, pero con
`samples=0`, porque no habia ningun backend activo. Esto sirve como smoke de
control, no como corrida de dataset.

## Arquitectura De Ejecucion

La ejecucion real se organiza asi:

```text
telemetry_kernel_launcher
  |
  | por repeticion
  |
  +-- baseline:
  |     telemetry_kernel_workload
  |     collector desactivado
  |
  +-- telemetry:
        telemetry_kernel_workload
        collector productor
        ring SPSC
        consumidor en memoria
        exportacion posterior
```

El launcher ejecuta dos procesos hijo por repeticion:

1. `baseline`: workload sin telemetria.
2. `telemetry`: workload con collector y consumidor activos.

El workload reserva memoria, crea su pool de hilos y ejecuta warmup antes de
avisar `ready`. El launcher envia `go` cuando ya aplico cgroup/afinidad y, en
la corrida `telemetry`, despues de iniciar el consumidor y el collector.

La ventana medida en el workload solo cubre:

```text
kernel->run(iterations)
```

No incluye asignacion de memoria, creacion de hilos, warmup ni sincronizacion
inicial con el launcher.

## Arquitectura Del Collector

El collector es un productor dedicado:

```text
PerfReader / PerfCgroupReader
RaplReader
NvmlReader opcional
        |
        v
Collector::run()
        |
        v
SPSCRing<Sample, 16384>
        |
        v
consumer thread del launcher
        |
        v
vector<RecordedSample>
        |
        v
samples.csv, metadata.json, summary.txt
```

La ruta caliente esta en `Collector::run()`. Actualmente realiza:

- `clock_gettime(CLOCK_MONOTONIC)`;
- lecturas de backends ya abiertos;
- `ring.try_push`;
- `ring.flush_producer`;
- `clock_nanosleep` con `TIMER_ABSTIME`.

No debe introducirse en esta ruta:

- logging;
- escritura a disco;
- crecimiento de contenedores;
- locks;
- construccion de strings;
- excepciones como flujo normal;
- calculo de IPC, ratios, potencia o deltas energeticos.

## Modelo De Muestra

Las muestras viajan como `telemetry::Sample`, una union etiquetada de tamano
fijo:

```text
SampleTag::CPU    -> CpuSample
SampleTag::ENERGY -> EnergySnapshot
SampleTag::GPU    -> GpuSample
```

CPU incluye:

- `timestamp_ns`;
- `instructions`;
- `cycles`;
- `cache_references`;
- `cache_misses`;
- `time_enabled_ns`;
- `time_running_ns`.

ENERGY incluye snapshots acumulados:

- `timestamp_ns`;
- `pkg_uj`;
- `dram_uj`.

GPU/NVML incluye datos globales del dispositivo:

- `timestamp_ns`;
- `power_mw`;
- `util_pct`.

NVML no atribuye por si solo consumo o utilizacion a un kernel CUDA especifico.
Por tanto, GPU debe mantenerse fuera de la ruta experimental principal hasta
disenar sincronizacion/atribucion explicita.

## Perf

Hay dos readers:

- `PerfReader`: lector simple por PID/CPU. Es util para pruebas o casos simples.
- `PerfCgroupReader`: lector principal para workloads CPU multihilo.

La ruta multihilo usa `perf_event_open` con `PERF_FLAG_PID_CGROUP`. El cgroup
define el conjunto de tareas medido y `--perf-cpus` define en que CPUs se abren
eventos. `--perf-cpus` no pinnea el workload.

Eventos actuales:

- `PERF_COUNT_HW_INSTRUCTIONS`;
- `PERF_COUNT_HW_CPU_CYCLES`;
- `PERF_COUNT_HW_CACHE_REFERENCES`;
- `PERF_COUNT_HW_CACHE_MISSES`.

Los eventos se leen como grupo y conservan `time_enabled/time_running` para
diagnosticar multiplexacion. En `PerfCgroupReader`, los conteos se agregan entre
CPUs.

## RAPL

`RaplReader` abre `energy_uj` una vez y reusa el descriptor con `lseek/read`.
El productor guarda snapshots acumulados crudos. Los deltas se calculan al
exportar mediante `next_rapl_delta()`, separando repeticiones para no mezclar
ventanas independientes.

Si existe `max_energy_range_uj`, se usa para manejar wrap-around. Si no existe,
un wrap no puede considerarse confiable.

## Workload CPU

`telemetry_kernel_workload` implementa un pool fijo de threads creado antes de
la medicion. Los kernels actuales son:

| Kernel | Label exportado | Interpretacion de `--size` |
| --- | --- | --- |
| `stream_triad` | `memory_bound` | longitud lineal de vectores |
| `reduction` | `memory_bound` | longitud lineal de vector |
| `stencil_2d` | `cache_sensitive` | dimension `N` de matriz `N x N` |
| `gemm_naive` | `compute_bound` | dimension `N` de matrices `N x N` |

Estos labels son etiquetas experimentales controladas para microbenchmarks. No
son predicciones de un modelo ni verdad universal para cualquier tamano,
compilador o plataforma.

## Uso Del Launcher

Smoke local sin perf:

```bash
/tmp/tg-telemetry-build/telemetry_kernel_launcher \
  --kernel stream_triad \
  --size 10000 \
  --iterations 2 \
  --warmup 1 \
  --threads 2 \
  --repetitions 2 \
  --collector-cpu -1 \
  --consumer-cpu -1 \
  --no-perf \
  --output-dir /tmp/tg-telemetry-smoke \
  --run-id smoke_no_perf
```

Primera toma real CPU con perf cgroup y RAPL:

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

Toma de un binario externo de dataset (sin protocolo ready/go, sin corrida baseline, elapsed por wall-clock del padre):

```bash
telemetry_kernel_launcher \
  --exec /opt/nas/bin/mg.S.x \
  --exec-args "" \
  --repetitions 3 \
  --perf-cpus 2-5 \
  --collector-cpu 0 \
  --consumer-cpu 1 \
  --cgroup-path /sys/fs/cgroup/<delegated-cgroup> \
  --interval-ns 1000000 \
  --rapl-pkg /sys/class/powercap/intel-rapl/intel-rapl:0 \
  --output-dir runs \
  --run-id nas_mg_S
```

Opciones principales:

| Opcion | Descripcion |
| --- | --- |
| `--kernel` | Kernel CPU a ejecutar. |
| `--size` | Tamano logico del problema. |
| `--iterations` | Iteraciones medidas despues del warmup. |
| `--warmup` | Iteraciones previas no medidas. |
| `--threads` | Workers del workload. |
| `--repetitions` | Pares baseline/telemetry. |
| `--perf-cpus` | CPUs donde perf abre eventos. No pinnea workload. |
| `--workload-cpus` | Alias historico de `--perf-cpus`. |
| `--pin-workload-cpus` | Afinidad del proceso workload. |
| `--pin-workers` | Pinning worker-a-CPU usando `--pin-workload-cpus`. |
| `--collector-cpu` | CPU del productor; `-1` sin pinning. |
| `--consumer-cpu` | CPU del consumidor; `-1` sin pinning. |
| `--cgroup-path` | Cgroup delegado/precreado para perf cgroup. |
| `--interval-ns` | Periodo de muestreo. Default: 1 ms. |
| `--no-perf` | Desactiva contadores CPU. |
| `--rapl-pkg` | Dominio RAPL package. |
| `--rapl-dram` | Dominio RAPL DRAM opcional. |
| `--output-dir` | Directorio base de resultados. |
| `--run-id` | Nombre de corrida. |
| `--workload-bin` | Workload alternativo compatible. |
| `--exec` | Modo binario externo: ruta al ejecutable de dataset a medir. Hace que el fork+exec lance ese binario en vez de `telemetry_kernel_workload`. Implica omitir el protocolo ready/go (el binario real no lo conoce) y medir elapsed con wall-clock del padre; no se ejecuta corrida baseline. |
| `--exec-args` | Argumentos whitespace-separados para el binario `--exec` (sin quoting). |

## Salidas

Cada corrida escribe:

```text
<output-dir>/<run-id>/
  samples.csv
  metadata.json
  summary.txt
```

`samples.csv` es rectangular y mezcla filas CPU/ENERGY/GPU. Columnas actuales:

```text
run_id,repetition,kernel,label,timestamp_ns,tag,
instructions,cycles,cache_references,cache_misses,
time_enabled_ns,time_running_ns,
pkg_uj,dram_uj,pkg_delta_uj,dram_delta_uj,energy_delta_valid,
gpu_power_mw,gpu_util_pct
```

`metadata.json` contiene parametros de ejecucion, tiempos baseline/telemetry,
overhead estimado, jitter de muestreo, `push_retries`, ratio perf minimo,
resumen RAPL y numero de muestras.

`summary.txt` contiene un resumen plano de las metricas principales.

## Observaciones Para Corregir En La Siguiente Iteracion

Estas observaciones no son funcionalidades nuevas deseables, sino puntos que se
deben resolver para que la toma de datos sea logicamente confiable antes de
construir encima una plataforma experimental.

### Muestras Instantaneas Vs Deltas

Varias columnas actuales no representan "valor de la fase" por si solas, sino
lecturas acumuladas en un instante. Para construir datos correctos se deben
calcular deltas entre muestras consecutivas de la misma repeticion y, segun el
caso, derivar tasas o agregados por ventana.

Casos principales:

- `instructions`, `cycles`, `cache_references`, `cache_misses`: perf entrega
  contadores acumulados desde que se habilito el grupo. Para features por
  ventana se debe usar `delta_actual = contador_t - contador_t-1`, no el valor
  absoluto de la fila.
- `time_enabled_ns` y `time_running_ns`: tambien deben diferenciarse por
  ventana para diagnosticar multiplexacion del intervalo. El ratio util es
  normalmente `delta_running / delta_enabled`.
- `pkg_uj` y `dram_uj`: RAPL entrega energia acumulada. Ya existe calculo de
  `pkg_delta_uj` y `dram_delta_uj` en exportacion, pero debe tratarse como la
  energia util de la ventana, no como energia instantanea.
- potencia: no debe inferirse desde una sola muestra RAPL. Si se requiere, debe
  calcularse como `delta_energy_j / delta_time_s`.
- IPC, miss ratio y otras features: deben calcularse con deltas del intervalo,
  por ejemplo `ipc = delta_instructions / delta_cycles`, no usando acumulados
  desde el inicio de la corrida.

La primera muestra de cada repeticion no tiene delta valido porque no existe
muestra previa dentro de esa misma ventana. No se debe rellenar artificialmente.

Esto implica que `samples.csv` debe entenderse como salida cruda/intermedia. La
plataforma experimental sencilla deberia producir o validar una segunda vista de
datos por intervalo, por ejemplo `windows.csv` o `features.csv`, donde cada fila
represente un intervalo `[t-1, t]` con deltas, tasas y banderas de validez.

### Observaciones De Robustez Ya Identificadas

- Si la ring SPSC se llena, el collector actualmente reintenta hasta insertar.
  Esto preserva la muestra, pero puede romper la cadencia. Deberia contar la
  perdida/presion y continuar, rechazando la corrida si ocurre.
- RAPL puede devolver `0` ante error de lectura y esa lectura podria propagarse
  como si fuera valida. Hace falta validez explicita por lectura o rechazo de
  deltas sospechosos.
- El launcher necesita timeouts para `ready`, ejecucion del hijo y lectura de
  stdout. Una corrida experimental no debe quedarse colgada indefinidamente.
- Una corrida con backends desactivados puede terminar con `samples_collected=0`.
  Eso sirve para smoke de control, pero no debe aceptarse como toma de datos.
- Si perf esta activo, debe verificarse que el cgroup este limpio antes de medir.
- El directorio de salida deberia fallar si ya existe, salvo opcion explicita de
  overwrite.
- Los labels por kernel son etiquetas experimentales controladas; no son
  predicciones de un clasificador.

## Requisitos Para Nodo Real

Antes de usar esta ruta para datos:

- compilar en el nodo objetivo;
- pasar CTest local;
- tener un cgroup delegado, escribible y limpio;
- verificar permisos de `perf_event_open`;
- asegurar que `--perf-cpus` cubre las CPUs donde corre el workload;
- si se usa energia, verificar lectura de `energy_uj`;
- registrar governor, driver, afinidad, frecuencia y carga externa;
- evitar mezclar procesos ajenos en el cgroup medido.

El launcher solo escribe el PID hijo en `cgroup.procs`. No crea ni administra la
jerarquia global de cgroups.

## Limitaciones Y Riesgos Conocidos

Estos puntos son importantes para el plan de implementacion de la nueva
plataforma sencilla:

1. `Collector::run()` reintenta `try_push()` hasta insertar si la ring esta
   llena. Esto conserva muestras, pero puede romper la cadencia de muestreo. La
   plataforma deberia preferir descartar/contar muestras perdidas y rechazar la
   corrida si hubo presion.

2. `RaplReader` actualmente retorna `0` ante errores de lectura o parseo y aun
   asi puede reportar exito. La plataforma deberia introducir validez explicita
   por muestra o rechazar deltas energeticos sospechosos.

3. El launcher usa lecturas bloqueantes para `ready` y stdout del hijo. Si el
   workload se cuelga, la corrida puede quedar bloqueada. Hacen falta timeouts y
   cleanup robusto.

4. Una corrida puede terminar con exit code 0 y `samples_collected == 0` si no
   hay backend activo. Eso es aceptable como smoke de control, pero no como
   corrida de dataset.

5. El launcher no verifica actualmente que el cgroup este limpio antes de medir.
   Esto puede contaminar perf cgroup.

6. `run_id` existente no se rechaza explicitamente; `create_directories()` no
   impide reusar un directorio. La plataforma deberia fallar por defecto si la
   corrida ya existe, salvo `--overwrite` deliberado.

7. `perf_running_ratio_min` se calcula sobre muestras agregadas. Sirve como
   indicador inicial, pero no reemplaza diagnostico por CPU/intervalo si se
   necesita mayor rigor.

8. No hay test de integracion del launcher en esta rama. Solo hay tests unitarios
   de piezas internas. Para la plataforma sencilla conviene agregar tests con un
   workload falso o una corrida smoke controlada.

## Criterio Propuesto Para La Siguiente Plataforma

La siguiente capa deberia mantenerse pequena y enfocada en reproducibilidad de
corridas, no en una plataforma de usuario final.

Minimo razonable:

- una configuracion simple por archivo o CLI;
- ejecucion repetida de matrices de parametros;
- directorio de resultados por corrida;
- rechazo si no hay muestras esperadas;
- rechazo si `push_retries > 0`;
- timeouts por fase;
- validacion de cgroup limpio cuando perf esta activo;
- metadata suficiente para reproducir: comando, host, fecha, CPUs, cgroup,
  governor/frecuencia si se registra externamente;
- sin ML, sin DVFS online y sin GPU atribuida todavia.

La base C++ actual es util para construir encima, pero antes de tomar dataset
formal debe endurecerse la semantica de "corrida valida".
