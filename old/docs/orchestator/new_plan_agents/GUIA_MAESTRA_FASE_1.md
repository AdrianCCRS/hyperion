# Guía Maestra de Desarrollo — Fase 1: Plataforma de Recolección de Telemetría

**Proyecto:** Agente en Espacio de Usuario para Gestión Dinámica de Frecuencia (DVFS) en Sistemas Heterogéneos mediante Modelos Ligeros de Machine Learning

**Alcance de este documento:** es el documento único y autoritativo para la Fase 1. Unifica la metodología experimental, la especificación técnica del orquestador Python, los cambios necesarios en el subsistema C++ de telemetría, el checklist de validaciones, el plan de tests y la guía de desarrollo asistido por IA. No existe ningún otro documento con prioridad sobre este salvo el Registro ARC de adaptaciones de plataforma (cuando describe hechos observados en hardware real).

**Jerarquía de resolución de conflictos:** (a) el cpuset efectivo del job (`os.sched_getaffinity(0)`), (b) el `EnvironmentProfile` que reporta `environment.py` en tiempo real, (c) el resultado del preflight, (d) el Registro ARC, (e) este documento. No inferir permisos ni capacidades a partir del modelo de CPU o del nombre del nodo — siempre verificar contra la fuente real.

---

## Parte I — Contexto y fundamentos

### 1. Propósito

Este documento guía la construcción de la plataforma de recolección de telemetría (Fase 1 del trabajo de grado). El objetivo es generar, de forma reproducible y auditable, un dataset con el que se entrenará un clasificador ligero de ML que infiera si una aplicación está en régimen `compute_bound` o `memory_bound`, para que un agente DVFS actúe en consecuencia y optimice el Producto Energía-Retardo (EDP).

Lo que se construye aquí NO es el clasificador ni el agente de control todavía. Es la infraestructura de recolección: un orquestador Python que envuelve a un harness C++17 existente, lo conecta a kernels de benchmarking reales, calibra el nodo, ejecuta campañas controladas y produce un dataset etiquetado por el modelo Roofline.

### 2. Plataforma real confirmada: felix como único nodo operativo

Tras los diagnósticos de solo lectura contra hardware real del clúster SC3:

| Campo | Valor confirmado en felix.sc3.uis.edu.co |
|---|---|
| Topología | 4 nodos NUMA. Asignación de prueba CPUs 0-3 en NUMA 0; siblings SMT 32-35 fuera de la asignación. |
| cpufreq | `acpi-cpufreq`, 10 niveles discretos entre 1064 y 2261 MHz. El usuario NO tiene permiso de escritura todavía. |
| RAPL | **No disponible** (`rapl_capable=false`). Ningún dominio de energía expuesto. Sin fuente de energía alternativa confirmada. |
| perf | Eventos genéricos de CPU más mem-loads. Suficiente para telemetría, afinidad, NUMA y etiquetado Roofline en modo nativo. |
| GPU | Pendiente de confirmar y solicitar GPU NVIDIA asignada por Slurm. |
| Slurm | 24.11.5, cgroup v2, `proctrack/cgroup`. |

**`smexa.sc3.uis.edu.co` y `exadell.sc3.uis.edu.co` quedan descartados** — exponen GPU AMD (MI210), el proyecto requiere NVIDIA.

**Consecuencias directas para la Fase 1:**
- Cualquier campaña en felix es válida para telemetría de CPU, calibración Roofline y etiquetado `compute_bound`/`memory_bound`, pero **no puede producir features de energía ni EDP** — no inventar, imputar ni aproximar ese dato.
- Mientras no se resuelva la delegación de escritura de cpufreq, las campañas en felix solo pueden correr en modo de referencia nativa (`REF`), marcadas como `not_eligible_for_training_dataset: true` para DVFS. Esto es válido como validación del pipeline completo y para telemetría no-DVFS.
- El bloqueo administrativo de un cgroup delegado **ya no es necesario para tener datos de telemetría válidos** (ver sección 3).

### 3. Estrategia de medición: PID + inherit, sin dependencia de cgroup

Esta sección documenta el mecanismo real de medición y es el cambio más importante respecto al diseño original del subsistema C++ de telemetría.

#### 3.1 El mecanismo correcto

El harness C++ (`telemetry_kernel_launcher`) NO necesita conocer ni resolver la ruta del cgroup de Slurm para medir correctamente. Usa `perf_event_open` apuntando al **PID real del proceso hijo** (workload), con `pe.inherit = 1`, abierto **después del fork pero antes de que el hijo haga exec()**. El hijo se detiene hasta que los eventos estén armados:

```c
// Secuencia correcta en el launcher (simplificada):

pid_t child = fork();
if (child == 0) {
    // --- En el hijo ---
    raise(SIGSTOP);        // o esperar una barrera (pipe/eventfd)
    execvp(workload, args);
}

// --- En el padre ---
waitpid(child, &status, WUNTRACED);  // esperar a que el hijo se detenga

// Abrir perf sobre el PID REAL del hijo, no sobre sí mismo
struct perf_event_attr pe;
memset(&pe, 0, sizeof(pe));
pe.size    = sizeof(pe);
pe.type    = PERF_TYPE_HARDWARE;
pe.config  = PERF_COUNT_HW_INSTRUCTIONS;
pe.inherit = 1;          // propaga a descendientes del hijo
pe.disabled = 1;
pe.exclude_kernel = 1;
pe.exclude_hv = 1;

int fd = perf_event_open(&pe, child, -1, -1, 0);
// ... abrir los demás eventos (cycles, cache_refs, cache_misses)

ioctl(fd, PERF_EVENT_IOC_RESET, 0);
ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);

kill(child, SIGCONT);    // dejar que el hijo haga exec() del workload

// Ahora el padre puede leer fd periódicamente (cada --interval-ns)
// y obtener valores EN VIVO del progreso del hijo -- NO planos.
```

#### 3.2 Por qué no funciona el patrón "abrir sobre sí mismo y confiar en la herencia"

Un patrón frecuentemente citado pero **incorrecto para muestreo periódico** es `perf_event_open(pid=0, ...)` con `inherit=1` desde el proceso padre, antes del fork:

- Los contadores del hijo heredado son copias independientes cuyo conteo **solo se pliega (se suma) de vuelta al fd del padre cuando el hijo termina** (`perf_event_exit_task` en el kernel), no en cada `read()` mientras el hijo corre.
- Si el padre lee el fd periódicamente mientras el hijo sigue vivo (que es exactamente lo que necesita `samples.csv`: un muestreo cada `--interval-ns`), el valor se queda plano, sin reflejar el progreso real del hijo, hasta que el hijo termina y aparece de golpe todo el trabajo acumulado.
- El mecanismo de ventanas (`windows.csv`, `operational_intensity`, `phase_label_train` por ventana) requiere progreso incremental real en cada lectura. Con el patrón incorrecto, eso no ocurre.

Con el patrón correcto (3.1), el evento apunta al PID real del hijo, y las lecturas del padre son en vivo, correctas, en tiempo real — porque el contador de PMU está multiplexado directamente a esa tarea específica en cada cambio de contexto.

#### 3.3 `inherit=1`: para qué sirve y para qué no

- **Para qué sirve:** si el propio workload genera descendientes adicionales (poco probable en NPB/STREAM/ERT, que son de un solo proceso con hilos OpenMP), `inherit` asegura que esos nietos también queden cubiertos.
- **Para qué NO sirve en este contexto:** para "llegar" al workload desde el orquestador confiando en la herencia padre→hijo, porque el plegado no es en vivo (ver 3.2).
- **Hilos OpenMP:** un evento apuntado a un PID cubre automáticamente todos los hilos de ese grupo de tareas (`tgid`). `inherit` no es necesario para hilos, solo para procesos descendientes.

#### 3.4 Caveat documentado: desfase entre contadores no agrupados

Con `inherit=1` activo, el kernel restringe la lectura agrupada (`PERF_FORMAT_GROUP`). Los eventos (instrucciones, ciclos, cache_refs, cache_misses) deben abrirse como descriptores separados y leerse en secuencia, introduciendo un desfase de microsegundos entre ellos. Este desfase es despreciable frente a la resolución de `--interval-ns` (milisegundos), pero es la única fuente real de "no exactamente simultáneo" que queda — y no tiene nada que ver con cgroups.

#### 3.5 E06: la defensa real contra contaminación — por afinidad, no por cgroup

`perf` midiendo por PID+inherit resuelve la **atribución correcta** (¿el contador mide el proceso correcto?). Pero la **contención de hardware compartido** (otro proceso en un core hermano compitiendo por caché L3/ancho de banda de memoria, distorsionando los *valores* de IPC/miss-rate que sí se atribuyen correctamente) es un efecto físico real independiente de la atribución. Se detecta verificando que no haya procesos ajenos con afinidad a `delegated_cpus` (inspeccionar `Cpus_allowed` de los procesos vivos), no por membresía de cgroup.

---

### 4. Cambios necesarios en el subsistema C++ de telemetría

Esta sección especifica exactamente qué archivos del harness C++ existente (rama `fase-1/plataforma-experimental-simplificada`) deben cambiar y cómo, para implementar la estrategia de medición de la sección 3.

#### 4.1 Tabla de cambios por archivo

| Archivo | Tipo de cambio | Detalle |
|---|---|---|
| `include/telemetry/perf_reader.hpp` / `src/perf_reader.cpp` | **Modificar** | Agregar `pe.inherit = 1` a la configuración de `perf_event_attr`. Cambiar la firma para aceptar un PID externo (el del hijo ya forkeado) en vez de asumir `pid=0` o un PID propio. Abrir cada evento como descriptor separado (no agrupado con `PERF_FORMAT_GROUP` cuando `inherit=1`). |
| `include/telemetry/perf_cgroup_reader.hpp` / `src/perf_cgroup_reader.cpp` | **Deprecar** | Marcar como legacy/deprecated. No eliminar todavía (para no romper tests existentes), pero dejar de usarla como ruta principal del launcher. La ruta principal pasa a ser `PerfReader` con PID del hijo + inherit. |
| `experiments/telemetry_kernel_launcher.cpp` | **Modificar (el cambio más grande)** | Implementar la secuencia stop→open→resume de la sección 3.1. Antes del fork actual, el launcher preparaba los readers y abría perf por cgroup; ahora debe: (1) hacer fork, (2) esperar a que el hijo se detenga (SIGSTOP o barrera), (3) abrir `PerfReader` con el PID real del hijo, (4) activar los eventos, (5) dejar que el hijo continúe hacia exec(). Agregar el modo `--exec <path> --exec-args <args>` para binarios externos (NPB/STREAM/ERT), además del modo `--kernel` existente. |
| `experiments/telemetry_kernel_workload.cpp` | **Modificar (menor)** | Si el workload actualmente emite `ready` por stdout y espera `go` del padre, esa sincronización debe ocurrir DESPUÉS de que el padre haya abierto perf sobre su PID. Puede necesitar una reestructuración del orden de handshake: (1) hijo nace y se detiene (SIGSTOP), (2) padre abre perf, (3) padre envía SIGCONT, (4) hijo hace setup (memoria, hilos, warmup), (5) hijo emite `ready`, (6) padre envía `go`. |
| `include/telemetry/collector.hpp` / `src/collector.cpp` | **Modificar (menor)** | El collector actualmente recibe un reader ya abierto. Eso no cambia — solo cambia *quién* lo abre (el launcher, después del fork, sobre el PID del hijo). Verificar que el collector pueda recibir un `PerfReader` con PID externo en vez de un `PerfCgroupReader`. |
| CLI del launcher (opciones) | **Agregar/Deprecar** | `--cgroup-path` pasa a ser opcional/deprecated (no bloqueante si falta). `--exec <path>` y `--exec-args <args>` se agregan como modo alternativo a `--kernel`. `--perf-cpus` se mantiene para afinidad del workload, pero ya no condiciona dónde se abren eventos cgroup (porque ya no hay cgroup). |
| `include/telemetry/metrics.hpp` | **Sin cambios** | `Sample`, `CpuSample`, `EnergySnapshot`, `GpuSample` no cambian. La estructura de datos es independiente de cómo se abren los eventos. |
| `include/telemetry/spsc_ring.hpp` / `src/spsc_ring.cpp` | **Sin cambios** | El ring buffer es independiente del mecanismo de medición. |
| `include/telemetry/rapl_reader.hpp` / `src/rapl_reader.cpp` | **Sin cambios** | RAPL se lee vía sysfs, independiente de perf/cgroup/PID. |
| `include/telemetry/nvml_reader.hpp` / `src/nvml_reader.cpp` | **Sin cambios** | NVML es independiente de todo lo demás y sigue fuera de la ruta principal. |
| Tests CTest existentes | **Revisar** | Los tests que usen `PerfCgroupReader` directamente necesitan un equivalente con `PerfReader` + PID externo. Los 9 tests actuales que pasan sin hardware real probablemente no se vean afectados (verificar). |

#### 4.2 Secuencia detallada del nuevo flujo del launcher

```text
telemetry_kernel_launcher (proceso padre)
  │
  │ Parse CLI: --kernel o --exec, --perf-cpus, --interval-ns, etc.
  │
  │ POR CADA REPETICIÓN:
  │
  ├── BASELINE (sin telemetría):
  │     fork() → hijo ejecuta workload (sin perf, sin collector)
  │     padre espera a que termine, registra wall-time
  │
  ├── TELEMETRY:
  │     │
  │     │ 1. fork()
  │     │    └── HIJO: raise(SIGSTOP) inmediatamente
  │     │
  │     │ 2. PADRE: waitpid(child, WUNTRACED)
  │     │    └── confirmar que el hijo está detenido
  │     │
  │     │ 3. PADRE: abrir PerfReader(child_pid, inherit=1, eventos)
  │     │    └── un fd por evento (instructions, cycles, cache_refs, cache_misses)
  │     │    └── ioctl(PERF_EVENT_IOC_RESET + ENABLE) en cada fd
  │     │
  │     │ 4. PADRE: iniciar hilo collector (lee los fd periódicamente → ring SPSC)
  │     │    PADRE: iniciar hilo consumer (drena ring → vector<RecordedSample>)
  │     │
  │     │ 5. PADRE: kill(child, SIGCONT)
  │     │    └── HIJO: despierta, hace exec() del workload (si --exec)
  │     │         o continúa con setup interno (si --kernel)
  │     │
  │     │ 6. HIJO ejecuta: setup → warmup → ready → [espera go] → kernel→run(iterations)
  │     │    PADRE: lee "ready", envía "go"
  │     │    Collector muestrea cada --interval-ns: read(fd) → ring
  │     │
  │     │ 7. HIJO termina → PADRE: detener collector, join consumer
  │     │    Exportar samples.csv, metadata.json, summary.txt
  │     │    Cerrar los fd de perf
  │
  │ FIN REPETICIÓN
```

#### 4.3 El modo --exec para binarios externos

Cuando el launcher recibe `--exec /ruta/bin/mg.S.x --exec-args ""`:

- El hijo, después de recibir SIGCONT, hace `execvp(exec_path, exec_args)` en vez de iniciar `telemetry_kernel_workload` internamente.
- El protocolo de handshake (`ready`/`go`) **no aplica** para binarios externos, que no saben cooperar con el launcher. En este caso:
  - El launcher NO espera un `ready` del hijo.
  - El launcher inicia el collector inmediatamente después de enviar SIGCONT.
  - El warmup se resuelve en el post-procesamiento por tiempo de pared (campo `warmup_seconds` del catálogo del orquestador), no dentro del binario.
  - El `success_check` del binario (exit code o patrón en stdout) lo aplica el orquestador Python, no el launcher C++.
- El modo `--kernel` existente sigue funcionando exactamente igual para los kernels sintéticos de desarrollo (smoke tests, pruebas del propio orquestador).

#### 4.4 Qué NO cambia en el subsistema C++

- **El modelo de muestra** (`Sample`, `CpuSample`, `EnergySnapshot`, `GpuSample`): la estructura de datos es independiente de cómo se abren los eventos.
- **El ring SPSC**: es independiente del mecanismo de medición.
- **RAPL**: se lee vía sysfs, independiente de perf/cgroup/PID.
- **La ruta caliente de `Collector::run()`**: sigue haciendo `clock_gettime` → `read(fd)` → `try_push` → `flush_producer` → `clock_nanosleep`. Solo cambia que los `fd` ahora apuntan a un PID externo con inherit, no a un cgroup.
- **Las salidas** (`samples.csv`, `metadata.json`, `summary.txt`): mismo formato, mismas columnas. El orquestador Python se encarga de producir `windows.csv` a partir de `samples.csv`.

---

## Parte II — Diseño experimental

### 5. Principios rectores

Cinco principios gobiernan cada decisión de este plan:

1. **Seguridad del nodo compartido.** Cualquier acción que module frecuencia o afinidad debe estar acotada exactamente a los recursos delegados y debe ser reversible. El aislamiento se garantiza por afinidad de CPU (cpuset de Slurm), no por cgroup. Un cgroup delegado es aislamiento adicional, opcional, útil para límites duros de memoria — nunca un requisito para datos válidos.

2. **Reproducibilidad.** Cada corrida debe poder repetirse exactamente a partir de su metadata: comando ejecutado, commit del harness y del catálogo, hash del binario, host, CPUs (cpuset efectivo, no declarativo), governor/frecuencia vigente, fecha y condiciones ambientales.

3. **Validez estadística del dato, no solo validez técnica.** Que una corrida termine sin error no implica que sea útil para entrenamiento. El plan distingue explícitamente entre "la corrida no falló" y "la corrida es apta para el dataset".

4. **Separación estricta entre datos crudos y datos de entrenamiento.** `samples.csv` es crudo. Nunca se entrena directamente sobre esa vista. Toda campaña produce `windows.csv` con deltas, tasas, intensidad operacional y banderas de validez.

5. **La etiqueta de entrenamiento se mide, no se asume.** `phase_label_train` se calcula comparando `operational_intensity` contra `I_ridge` del Roofline calibrado, por ventana. Cualquier expectativa de la literatura sobre un kernel se conserva como `phase_label_hint`, solo para auditoría.

### 6. Kernels de carga de trabajo: suites externas, no programados por el proyecto

Los kernels NO se programan dentro del proyecto. Se usan binarios pre-compilados de suites de benchmarking reconocidas, conectados al launcher mediante un catálogo declarativo (`kernels/catalog.yaml`) y el modo `--exec`.

**Catálogo de tres capas:**

| Capa | Suite | Rol |
|---|---|---|
| Calibración de ancho de banda | STREAM (McCalpin, binario oficial) | `calibration` — no entra al dataset |
| Calibración de cómputo pico | ERT (Empirical Roofline Toolkit) | `calibration` — no entra al dataset |
| Kernels de dataset | NAS Parallel Benchmarks (clases SER/OMP) | `dataset` — genera las corridas de entrenamiento |

**Kernels NPB propuestos:**

| kernel_ref | Kernel NPB | phase_label_hint | Observación |
|---|---|---|---|
| npb_ep | EP (Embarrassingly Parallel) | compute_bound | Generación de números aleatorios; casi sin tráfico de memoria. |
| npb_mg | MG (Multigrid) | memory_bound | Acceso estructurado intensivo a memoria en malla 3D. |
| npb_cg | CG (Conjugate Gradient) | memory_bound | Acceso disperso a memoria (matriz rala). |
| npb_is | IS (Integer Sort) | memory_bound | Ordenamiento con acceso irregular a memoria. |
| npb_ft | FT (Fast Fourier Transform) | intermedio (mixto) | Balance no trivial entre cómputo y memoria. |
| npb_lu/sp/bt | Solvers dispersos/estructurados | intermedio a compute_bound | Mayor intensidad aritmética que MG/CG/IS. |

`phase_label_hint` es exclusivamente informativo: sirve para detectar si la etiqueta empírica derivada por Roofline contradice fuertemente lo esperado.

**Los kernels internos** (`stream_triad`, `gemm_naive`, `stencil_2d` de `telemetry_kernel_workload`) no desaparecen del repositorio: quedan como kernels sintéticos de desarrollo, usados exclusivamente para pruebas unitarias, la prueba de caos de `freqctl.py`, y validación del pipeline en local/cloud sin depender de tener las suites reales compiladas. Nunca entran al dataset de entrenamiento.

### 7. Calibración Roofline y etiquetado

Antes de ejecutar la matriz de kernels de dataset, cada campaña ejecuta una fase de calibración obligatoria (una sola vez por nodo/sesión):

1. Ejecutar STREAM sobre los cores delegados → `BW_pico` (ancho de banda sostenido).
2. Ejecutar ERT → `P_pico` (rendimiento de cómputo pico).
3. `I_ridge = P_pico / BW_pico` (ridge point, en FLOPs/byte).
4. Para cada ventana de cada corrida de dataset: `I = FLOPs_del_binario / bytes_movidos_por_perf`.
5. `phase_label_train = "memory_bound"` si `I < I_ridge`, `"compute_bound"` si `I ≥ I_ridge`.

**Nota de portabilidad sobre FLOPs:** no se usa el evento de PMU `FP_ARITH_INST_RETIRED` ni equivalentes (no portables entre Intel/AMD ni entre generaciones). Se usa el conteo de FLOPs que el propio binario de la suite reporta por stdout al finalizar (estándar en NPB/STREAM/ERT), combinado con bytes movidos medidos por perf (LLC misses × tamaño de línea de caché).

La calibración corre exclusivamente a F0 (frecuencia máxima o nativa, según los permisos disponibles), porque `P_pico` y `BW_pico` son los límites superiores del hardware.

### 8. Matriz experimental

**Estados de frecuencia:**

| Nivel | Configuración | Propósito |
|---|---|---|
| F0 | Governor userspace, f = f_max (si hay permiso de escritura) | Límite superior |
| F1 | f ≈ 75% del rango [f_min, f_max] | Intermedio alto |
| F2 | f ≈ 50% del rango | Intermedio central |
| F3 | f ≈ 25% del rango | Intermedio bajo |
| F4 | f = f_min | Límite inferior |
| REF | Gobernador dinámico nativo, sin control manual | Referencia; no se usa para entrenar DVFS |

Los niveles F0–F4 solo son ejecutables cuando `frequency_write_capable = True`. Mientras felix no tenga escritura delegada, la campaña corre exclusivamente en REF.

**Parámetros de repetición:**

| Parámetro | Valor propuesto | Justificación |
|---|---|---|
| Repeticiones por combinación | 10 | Relanzamientos independientes del binario. |
| Warmup | Por tiempo de pared (warmup_seconds del catálogo), excluido en post-procesamiento | Sin cooperación interna del binario. |
| Ventanas mínimas por repetición | ≥ 50 tras excluir warmup | Suficientes para deltas. |
| Intervalo de muestreo | 1 ms (validar overhead real) | Balance resolución/sobrecarga. |
| Orden de ejecución | Aleatorizado por combinación, nunca en bloques | Evita confundir deriva temporal. |

**Tamaño total:** 6 kernels × 6 niveles × 10 repeticiones = 360 corridas + 360 baseline + calibración.

---

## Parte III — Módulos del orquestador Python

### 9. Mapa de módulos

| # | Módulo | Responsabilidad | Depende de |
|---|---|---|---|
| 1 | `config.py` | Configuración externa (`orchestrator.toml`): rutas sysfs, flags del harness, detección de tier | — |
| 2 | `manifest.py` | Parsea y valida `campaign.yaml` | config |
| 3 | `environment.py` | Detecta, de solo lectura, qué puede controlarse realmente en este nodo — **única autoridad** | config |
| 4 | `diagnostics.py` | Diagnóstico de arranque de solo lectura antes de cualquier campaña | environment, catalog |
| 5 | `preflight.py` | Verificaciones bloqueantes o de advertencia antes de campaña y por corrida | manifest, environment |
| 6 | `freqctl.py` | Control y **restauración garantizada** de frecuencia — el más sensible | environment |
| 7 | `catalog.py` | Valida binarios externos desde `kernels/catalog.yaml` | manifest |
| 8 | `calibration.py` | Calibración Roofline: P_pico, BW_pico, I_ridge | catalog, runner, freqctl |
| 9 | `node_profile.py` | Perfil de hardware + referencias P95 (multinodo) | environment, runner |
| 10 | `runner.py` | Ejecuta una corrida individual del launcher (modo sintético y `--exec`) | manifest, preflight, catalog |
| 11 | `postprocess.py` | `samples.csv` → `windows.csv`: deltas, intensidad operacional, `phase_label_train`, features relativas | calibration, node_profile |
| 12 | `validation.py` | Acepta/rechaza cada corrida con un `factor_id` explícito | runner, catalog |
| 13 | `campaign.py` | El integrador: genera la matriz, aleatoriza, secuencia, reanuda | todos los anteriores |
| 14 | `metadata_schema.py` / `report.py` | Esquema de trazabilidad y reporte consolidado | validation, campaign |

**Orden de construcción real:**

```text
config.py, manifest.py y environment.py en paralelo
  → diagnostics.py → preflight.py → runner.py (modo sintético)
  → freqctl.py → catalog.py
  → calibration.py y node_profile.py en paralelo
  → runner.py (extensión --exec) → postprocess.py
  → validation.py → campaign.py → metadata_schema.py/report.py
```

### 10. Especificación de cada módulo

#### 10.1 config.py — Configuración externa

```python
# orchestrator.toml — separado del código
[paths]
sysfs_cpu = "/sys/devices/system/cpu"
sysfs_powercap = "/sys/class/powercap"
harness_bin = "build/telemetry_kernel_launcher"

[defaults]
interval_ns = 1000000
safety_margin = 3.0
running_ratio_min = 0.90
calibration_cv_threshold = 5.0
```

Responsabilidad: separar rutas sysfs, flags del harness y detección de tier del código fuente. Permite mockear rutas en tests sin editar lógica de negocio.

#### 10.2 manifest.py — Parsing y validación del manifest

**Dataclasses:** `Manifest`, `Combination`, `FrequencyLevel`, `Cores`, `Timeouts`.

**Reglas de validación:**

| ID | Regla |
|---|---|
| MAN-01 | `cgroup_path` es OPCIONAL en todos los tiers — no es requisito para que perf mida correctamente. |
| MAN-02 | Rechazar si `repetitions_per_combination < 3`. |
| MAN-03 | Calcular y loguear el tamaño total de la matriz antes de continuar. |
| MAN-04 | Rechazar si `output_dir` ya existe y `overwrite: false` (I07). |
| MAN-05 | Rechazar si `seed` está ausente o no es entero — nunca generar semilla aleatoria (rompe reproducibilidad). |
| MAN-06 | Rechazar si `delegated_cpus`, `collector_cpu` y `consumer_cpu` se solapan. |
| MAN-07 | `calibration` debe tener ≥1 kernel `reports_bandwidth_stdout` y ≥1 `reports_flops_stdout`. |
| MAN-08 | Ningún kernel de `calibration` puede aparecer en `kernels` (dataset), ni viceversa. |
| MAN-09 | Todo `kernel_ref` debe existir en `catalog_path`. |
| MAN-10 | `frequency_levels` con exactamente un nivel `native_governor` (REF) y los demás `fixed` con `fraction ∈ [0.0, 1.0]`. |
| MAN-11 | `running_ratio_min ∈ (0.0, 1.0]` y `interval_ns > 0`. |

#### 10.3 environment.py — Detección de capacidades (solo lectura)

```python
@dataclass
class EnvironmentProfile:
    tier: str                          # local | cloud_own | hpc_sc3
    frequency_levels_supported: bool   # el driver/hardware expone niveles controlables
    frequency_write_capable: bool      # el USUARIO tiene permiso real de escritura
    frequency_control_strategy: str    # "discrete_bounds" | "bounded_range" | "unavailable"
    frequency_control_paths: dict      # rutas reales por CPU/policy
    scaling_driver: str
    available_frequencies_khz: list[int]
    rapl_capable: bool
    rapl_domains_available: list[str]  # alias únicos: "package-0", "core-package-0"...
    rapl_domain_paths: dict            # alias → ruta sysfs real
    numa_nodes: int
    smt_siblings: dict[int, list[int]]
    gpu_present: bool
    gpu_vendor: str                    # "nvidia" | "amd" | "none"
    gpu_exclusive_hint: bool
```

**Reglas:**

| ID | Regla |
|---|---|
| ENV-01 | `detect_environment()` es de SOLO LECTURA. Ningún otro módulo repite la detección. |
| ENV-02 | `frequency_levels_supported = False` si el driver no es real o solo hay 1 frecuencia. |
| ENV-03 | RAPL recursivo: si no existe ningún dominio o `energy_uj` no cambia, `rapl_capable = False`. |
| ENV-04 | El manifest no puede forzar `rapl.enabled: true` si `rapl_capable: false`. |
| ENV-05 | `frequency_write_capable` se determina con `os.access(path, os.W_OK)`, INDEPENDIENTE de `frequency_levels_supported`. |
| ENV-06–09 | Topología NUMA, SMT, eventos de perf soportados, `environment_report.json`. |
| ENV-10 | `frequency_control_strategy` se determina por qué atributos son escribibles. |
| ENV-11 | Cada dominio RAPL recibe un alias único (`package-0`, etc.), nunca genéricos. |
| ENV-12 | `gpu_vendor` por detección real de dispositivo, nunca por nombre del nodo. |

#### 10.4 preflight.py — Verificaciones de solo lectura

**CheckResult:** `factor_id`, `name`, `passed`, `blocking`, `observed`, `message`.

**Preflight de campaña (una vez):**

| factor_id | Check | Bloqueante |
|---|---|---|
| E01 | Turbo/HWP (Intel) o CPB/CPPC (AMD): leer y fijar para toda la campaña. | Sí |
| E03 | Si hay cgroup delegado: cgroup HIJO de workload vacío (nunca el cgroup del orquestador). | No (opcional) |
| E04 | NUMA: `delegated_cpus` en un único nodo NUMA. | Sí |
| E05 | SMT: política declarada explícitamente en el manifest. | Sí |
| E09 | Si hay niveles `fixed`: `frequency_write_capable` debe ser True. | Sí si hay fixed |
| I05 | RAPL: si `rapl_domains_available` vacío, forzar `rapl.enabled` a False sin bloquear. | No |
| I07 | `output_dir` no existe (o `overwrite: true`). | Sí |
| I08 | `manifest.rapl.domains ⊆ rapl_domains_available` (alias reales). | Sí si rapl.enabled |
| I09 | Espacio libre en disco ≥ tamaño proyectado. | Sí |
| D05 | Eventos de perf solicitados ≤ PMCs disponibles. | Sí |
| OPS-01 | Presupuesto de hora-núcleo ≥ proyección. | Sí |
| G01/G02/G03 | GPU NVIDIA confirmada, sin procesos CUDA activos, persistence mode, MIG. | Sí si gpu.enabled |
| C01/C02/C03 | Binario existe, checksum, success_check. | Sí |
| D01–D04 | Toolchain, calibración ejecutada/parseable/plausible, CV%. | Según el caso |

**Preflight reducido (por corrida):**

| factor_id | Check | Bloqueante |
|---|---|---|
| E02 | Temperatura de paquete dentro de rango. | Sí si hay sensor |
| E06 | Sin procesos ajenos por afinidad a `delegated_cpus` (Cpus_allowed, no cgroup). | Sí |
| E07 | Atributo real de gobierno coincide con el esperado. | Sí si hay fixed |
| E08 | Carga externa bajo umbral. | Sí |
| I07 | `run_id` no existe. | Sí |
| C01/C02/C03 | Binario/checksum/success_check de esta combinación. | Sí |

#### 10.5 freqctl.py — Control de frecuencia (el módulo más sensible)

**Lectura y discretización:** consume `frequency_control_strategy` de `EnvironmentProfile`:
- `discrete_bounds`: lista de frecuencias seleccionables, discretiza a la más cercana.
- `bounded_range`: rango continuo [min, max], sin discretizar.
- `unavailable`: no toca nada.

**Aplicación y verificación:** `apply_frequency()` verifica `frequency_write_capable` primero. Escribe en las rutas reales de `frequency_control_paths` (nunca asume `scaling_setspeed`). Relee y compara.

**Restauración de emergencia:**
- `snapshot_original_state()`: UNA SOLA VEZ al inicio de campaña.
- `restore_original_state()`: idempotente, verifica por lectura.
- `install_emergency_handlers()`: `atexit`, `SIGINT`, `SIGTERM`.
- **Prueba de caos OBLIGATORIA** antes de usar en hardware real.

Si `frequency_write_capable == False`: no escribe nada, registra `frequency_control: "unavailable"`, marca `not_eligible_for_training_dataset: true`.

#### 10.6 catalog.py — Integridad de binarios externos

**KernelEntry:** `id`, `suite`, `role`, `exec_path`, `binary_checksum`, `phase_label_hint`, `size_variant`, `expected_runtime_seconds`, `warmup_seconds`, `success_check`, `reports_bandwidth_stdout`, `reports_flops_stdout`.

**Reglas:** C01 (existencia + ejecutable), C02 (sha256 real, no tamaño), C03 (success_check compila), roles separados (dataset/calibration), IDs únicos, `resolve_exec_command()` sin inventar argumentos.

#### 10.7 calibration.py — Calibración Roofline

Ejecuta STREAM + ERT a F0 (o nativa). Parsea stdout (BW_pico, P_pico). Calcula `I_ridge`. Verifica plausibilidad (D03). Serializa `roofline_calibration.json`. `load_calibration()` rechaza si `plausibility_check_passed == False`.

#### 10.8 node_profile.py — Perfil de hardware y referencias P95

`build_node_profile()`: solo lectura de `/proc/cpuinfo`, `/sys/.../cache/`, `/sys/.../node/`. `build_calibration_references()`: ≥5 repeticiones del kernel de referencia, P95 de IPC/IPS/MPKI/MissRate, CV%. Si `cv_pct > umbral`, `accepted = False` (advertencia, no bloqueante).

Artefactos: `node_profile.json`, `calibration_references.json`, `roofline_calibration.json`.

#### 10.9 runner.py — Ejecución de corrida individual

`run_single()`: preflight reducido → `freqctl.apply_frequency()` → construir comando del launcher → `subprocess.run(timeout)` → aplicar `success_check` → verificar no procesos hijos vivos → fusionar metadata.

**Modo sintético:** `--kernel <name> --size <n>`.
**Modo --exec:** `--exec <path> --exec-args <args>`.

`run_id` determinista: `f"{campaign_id}__{kernel_ref}__{freq_level.id}__rep{n:02d}"`.

#### 10.10 campaign.py — El integrador

`build_matrix()`: producto cartesiano kernels(dataset) × frequency_levels × repetitions.
`randomize(matrix, seed)`: `random.Random(seed).shuffle`, nunca global.
`run_campaign()`: preflight → calibración → matriz aleatorizada → por combinación (preflight reducido, freqctl, runner, validation) → restaurar frecuencia → postprocess → reporte.

Reanudación: si `metadata.json` existe con `accepted=True`, saltar. Si `accepted=False`, reintentar.

#### 10.11 postprocess.py — De samples.csv a windows.csv

Columnas de salida (REQUIRED_OUTPUT_COLUMNS): `run_id`, `repetition`, `kernel_ref`, `node_id`, `phase_label_hint`, `phase_label_train`, `freq_level_id`, `freq_khz_requested/applied/observed`, `window_index`, `t_start_ns/t_end_ns/delta_t_ns`, deltas de instrucciones/ciclos/cache, `ipc`, `llc_miss_rate`, `mpki`, `ips`, `ipc_relative/mpki_relative/miss_rate_relative`, `running_ratio`, energía, `operational_intensity`, `i_ridge_used`, referencias de calibración, `binary_checksum`, `quality_status`.

**quality_status:** `"ok"`, `"first_sample_no_delta"`, `"warmup_excluded"`, `"pmu_degraded"`, `"energy_invalid"`, `"no_freq_reading"`, `"intensity_undefined"`.

**Regla de oro:** `phase_label_train` SIEMPRE es `operational_intensity` vs `i_ridge`. Nunca se copia de `phase_label_hint`. Las features relativas se calculan SIEMPRE, sin recortarlas a [0,1].

#### 10.12 validation.py — Criterios de rechazo

Orden determinista: I04 (vacío) → C02/C03 (binario) → E06-E08 (contención) → el resto. Corridas rechazadas NUNCA se borran. Rechazo a nivel de ventana no invalida la corrida completa.

#### 10.13 metadata_schema.py / report.py

Metadata por corrida: fusión launcher + orquestador (merge con detección de colisiones). Reporte: tabla aceptadas/rechazadas por factor_id (suma 100%), I_ridge, % ventanas intensity_undefined, CV% de calibración.

---

## Parte IV — Estrategia multinodo

### 11. Tres alternativas, una decisión pendiente

| Alternativa | Idea central | Recomendación |
|---|---|---|
| A. Hardware explícito | Modelo global con descriptores del hardware como variables de entrada. | Reservar para trabajo futuro. |
| B. Features relativas | Modelo global con métricas normalizadas contra referencias de calibración del nodo. | Experimento secundario. |
| C. Modelo por nodo | Pipeline reproducible; el modelo es local al nodo. | **Adoptar como arquitectura oficial.** |

**Estrategia "sin arrepentimiento":** construir ya `node_id`, `node_profile.json`, `calibration_references.json` y features relativas en `windows.csv` — sirve a las tres alternativas sin comprometerse con ninguna.

**NO comprometer tiempo de campaña en un segundo nodo** sin la decisión formal del director.

---

## Parte V — Checklist de validaciones técnicas

### 12. Reglas por módulo

*(Cada regla tiene un ID único MÓDULO-NN y una casilla ☐)*

#### 12.1 Manifest (MAN-01 a MAN-11) — 11 reglas

- ☐ **MAN-01** `cgroup_path` es OPCIONAL en todos los tiers.
- ☐ **MAN-02** Rechazar si `repetitions_per_combination < 3`.
- ☐ **MAN-03** Calcular y loguear tamaño de la matriz antes de continuar.
- ☐ **MAN-04** Rechazar si `output_dir` existe y `overwrite: false` (I07).
- ☐ **MAN-05** Rechazar si `seed` ausente — nunca generar aleatoria.
- ☐ **MAN-06** Rechazar si cores se solapan.
- ☐ **MAN-07** Calibración debe tener ≥1 BW y ≥1 FLOPs.
- ☐ **MAN-08** Roles `calibration`/`dataset` sin solape.
- ☐ **MAN-09** Todo `kernel_ref` debe existir en el catálogo.
- ☐ **MAN-10** `frequency_levels` con un REF y los demás fixed válidos.
- ☐ **MAN-11** `running_ratio_min ∈ (0,1]` y `interval_ns > 0`.

#### 12.2 Catálogo (CAT-01 a CAT-08) — 8 reglas

- ☐ **CAT-01** C01: `exec_path` existe y es ejecutable.
- ☐ **CAT-02** C02: sha256 coincide con `binary_checksum`.
- ☐ **CAT-03** C03: `success_check` tipo reconocido y regex compila.
- ☐ **CAT-04** Kernel dataset tiene todos los campos obligatorios.
- ☐ **CAT-05** Kernel calibración: exactamente uno de BW/FLOPs en true.
- ☐ **CAT-06** `resolve_exec_command()` no inventa argumentos.
- ☐ **CAT-07** C01/C02 reducido antes de cada corrida.
- ☐ **CAT-08** IDs únicos en el catálogo.

#### 12.3 Entorno (ENV-01 a ENV-12) — 12 reglas

- ☐ **ENV-01** SOLO LECTURA. Única autoridad de detección.
- ☐ **ENV-02** `frequency_levels_supported` por driver/hardware.
- ☐ **ENV-03** RAPL recursivo; ausencia es hecho del nodo, no error.
- ☐ **ENV-04** Manifest no fuerza `rapl.enabled` si nodo no lo tiene.
- ☐ **ENV-05** `frequency_write_capable` con `os.access W_OK`, independiente de soporte.
- ☐ **ENV-06** Topología NUMA completa.
- ☐ **ENV-07** Siblings SMT y política en metadata.
- ☐ **ENV-08** Eventos de perf soportados.
- ☐ **ENV-09** `environment_report.json` generado.
- ☐ **ENV-10** `frequency_control_strategy` por atributos escribibles.
- ☐ **ENV-11** Alias únicos por dominio RAPL.
- ☐ **ENV-12** `gpu_vendor` por detección real.

#### 12.4 Preflight (PRE-E01 a PRE-OPS01) — 25 reglas

- ☐ **PRE-E01** Turbo/HWP o CPB/CPPC: fijo para toda la campaña.
- ☐ **PRE-E02** Temperatura de paquete.
- ☐ **PRE-E03** Cgroup hijo vacío (SOLO si declarado, NO bloqueante).
- ☐ **PRE-E04** NUMA: un solo nodo.
- ☐ **PRE-E05** SMT: política declarada.
- ☐ **PRE-E06** Procesos ajenos por afinidad (`Cpus_allowed`), NO por cgroup.
- ☐ **PRE-E07** Governor drift verificado por atributo real.
- ☐ **PRE-E08** Carga externa bajo umbral.
- ☐ **PRE-E09** `frequency_write_capable` si hay niveles fixed.
- ☐ **PRE-I05** `max_energy_range_uj` por dominio.
- ☐ **PRE-I07** `output_dir`/`run_id` no existe.
- ☐ **PRE-I08** `rapl.domains ⊆ rapl_domains_available`.
- ☐ **PRE-I09** Espacio en disco suficiente.
- ☐ **PRE-C01** Binario existe.
- ☐ **PRE-C02** Checksum coincide.
- ☐ **PRE-C03** `success_check` bien configurado.
- ☐ **PRE-D01** Toolchain disponible.
- ☐ **PRE-D02** Calibración ejecutada y parseable.
- ☐ **PRE-D03** BW_pico/P_pico plausibles.
- ☐ **PRE-D04** CV% de referencias P95 bajo umbral (advertencia).
- ☐ **PRE-D05** Eventos perf ≤ PMCs disponibles.
- ☐ **PRE-OPS01** Presupuesto de hora-núcleo suficiente.
- ☐ **PRE-G01** GPU NVIDIA confirmada, sin procesos CUDA ajenos.
- ☐ **PRE-G02** Persistence mode leído.
- ☐ **PRE-G03** MIG leído.

#### 12.5 Control de frecuencia (FRQ-01 a FRQ-10) — 10 reglas

- ☐ **FRQ-01** `snapshot_original_state()` UNA SOLA VEZ.
- ☐ **FRQ-02** Verificar por relectura del atributo real (no asumir `scaling_setspeed`).
- ☐ **FRQ-03** Guardar valor solicitado Y aplicado en metadata.
- ☐ **FRQ-04** `restore_original_state()` idempotente, verifica por lectura.
- ☐ **FRQ-05** Manejadores de emergencia en `atexit`, `SIGINT`, `SIGTERM`.
- ☐ **FRQ-06** Si `frequency_write_capable == False`: no escribir nada.
- ☐ **FRQ-07** Calibración a F0/nativa.
- ☐ **FRQ-08** Prueba de caos OBLIGATORIA.
- ☐ **FRQ-09** Solo sobre `delegated_cpus`, nunca global.
- ☐ **FRQ-10** Registrar frecuencia observada por ventana.

#### 12.6 Calibración (CAL-01 a CAL-11) — 11 reglas

- ☐ **CAL-01** Ejecutar a frecuencia máxima/nativa.
- ☐ **CAL-02** BW del stdout de STREAM, nunca de PMU.
- ☐ **CAL-03** P del stdout de ERT, nunca de `FP_ARITH_INST_RETIRED`.
- ☐ **CAL-04** I_ridge con verificación D03 en la misma función.
- ☐ **CAL-05** `roofline_calibration.json` con todos los campos.
- ☐ **CAL-06** `load_calibration()` rechaza si plausibility=False.
- ☐ **CAL-07** `build_node_profile()` SOLO LECTURA.
- ☐ **CAL-08** `node_profile.json` con todos los campos.
- ☐ **CAL-09** ≥5 repeticiones para P95.
- ☐ **CAL-10** CV% > umbral → `accepted=False`.
- ☐ **CAL-11** Los tres artefactos en la misma fase.

#### 12.7 Runner (RUN-01 a RUN-08) — 8 reglas

- ☐ **RUN-01** Comando desde `KernelEntry`, nunca hardcodeado.
- ☐ **RUN-02** `run_id` determinista.
- ☐ **RUN-03** Timeout = `expected_runtime_seconds × SAFETY_MARGIN`.
- ☐ **RUN-04** Verificar sin procesos hijos vivos al terminar.
- ☐ **RUN-05** `success_check` aplicado contra resultado real.
- ☐ **RUN-06** Metadata fusionada (launcher + orquestador).
- ☐ **RUN-07** stdout/stderr guardados en disco.
- ☐ **RUN-08** Si `frequency_write_capable=False`: no invocar `freqctl.apply_frequency()`.

#### 12.8 Campaign (CAM-01 a CAM-07) — 7 reglas

- ☐ **CAM-01** Aleatorizar siempre con semilla, nunca en bloques.
- ☐ **CAM-02** Semilla y orden en metadata de campaña.
- ☐ **CAM-03** Reanudación: accepted=True → saltar, accepted=False → reintentar.
- ☐ **CAM-04** Baseline/telemetry como par atómico.
- ☐ **CAM-05** Contabilizar hora-núcleo antes de escalar.
- ☐ **CAM-06** Timeouts por fase, nunca colgarse.
- ☐ **CAM-07** Restaurar frecuencia al cierre (normal o interrupción).

#### 12.9 Post-procesamiento (POST-01 a POST-16) — 16 reglas

- ☐ **POST-01** Primera muestra: `first_sample_no_delta`, nunca imputar.
- ☐ **POST-02** Delta negativo sin wrap → invalidar.
- ☐ **POST-03** I02: `running_ratio` bajo umbral → `pmu_degraded`.
- ☐ **POST-04** Usar intervalo real medido, no nominal.
- ☐ **POST-05** Corrección de wrap RAPL, o `energy_valid=false`.
- ☐ **POST-06** RAPL 0 ante error → bandera de invalidez.
- ☐ **POST-07** Warmup por tiempo de pared → `warmup_excluded`.
- ☐ **POST-08** `bytes_moved_window == 0` → `NaN`, nunca dividir por cero.
- ☐ **POST-09** FLOPs del stdout, nunca de PMU.
- ☐ **POST-10** `LLC_LINE_SIZE_BYTES` del node_profile, no hardcodeado.
- ☐ **POST-11** `phase_label_train` solo por Roofline, nunca copiado.
- ☐ **POST-12** Features relativas siempre, sin excepción.
- ☐ **POST-13** Ratios no recortados a [0,1].
- ☐ **POST-14** `node_id`, refs en cada fila.
- ☐ **POST-15** `load_calibration()` rechaza plausibility=False.
- ☐ **POST-16** `windows.csv` con absolutas Y relativas siempre.

#### 12.10 Validación (VAL-01 a VAL-08) — 8 reglas

- ☐ **VAL-01** I04: `samples_collected==0` o `push_retries>0` → rechazo inmediato.
- ☐ **VAL-02** I07: run_id duplicado, incluso si no se detectó en preflight.
- ☐ **VAL-03** C02: checksum discrepante → rechazo aunque la corrida terminara.
- ☐ **VAL-04** C03: success_check no cumplido → rechazo.
- ☐ **VAL-05** D03: calibración no plausible → rechazo de campaña completa.
- ☐ **VAL-06** Corridas rechazadas NUNCA se borran.
- ☐ **VAL-07** Orden determinista de evaluación.
- ☐ **VAL-08** Rechazo por ventana no invalida corrida completa.

#### 12.11 Metadata y reporte (MET-01 a MET-07) — 7 reglas

- ☐ **MET-01** `merge()` detecta colisiones de clave con valores distintos.
- ☐ **MET-02** `governor_restored_verified` por lectura, no por éxito de escritura.
- ☐ **MET-03** `node_id` estable entre campañas del mismo nodo.
- ☐ **MET-04** Reporte: tabla por factor_id que suma 100%.
- ☐ **MET-05** CV% > umbral → advertencia visible.
- ☐ **MET-06** Semilla y orden completo en metadata.
- ☐ **MET-07** Trazabilidad completa corrida → fila de `windows.csv`.

#### 12.12 Estrategia multinodo (MLT-01 a MLT-08) — 8 reglas

- ☐ **MLT-01** `node_id` en cada corrida y cada fila.
- ☐ **MLT-02** `node_profile.json` antes de la matriz.
- ☐ **MLT-03** ≥5 repeticiones para P95.
- ☐ **MLT-04** Features relativas siempre.
- ☐ **MLT-05** Manifests parametrizables cambiando solo tier/cores.
- ☐ **MLT-06** Protocolo versionado (commit hash) en metadata.
- ☐ **MLT-07** `-march=native` aceptable si modelo es por nodo.
- ☐ **MLT-08** No comprometer campaña en segundo nodo sin decisión del director.

#### 12.13 Subsistema C++ (CPP-01 a CPP-08) — 8 reglas (NUEVAS)

- ☐ **CPP-01** `PerfReader` debe aceptar PID externo con `inherit=1`, abriendo cada evento como fd separado (no agrupado con `PERF_FORMAT_GROUP`).
- ☐ **CPP-02** El launcher implementa la secuencia stop→open→resume: fork, hijo se detiene, padre abre perf sobre PID del hijo, padre inicia collector/consumer, padre envía SIGCONT.
- ☐ **CPP-03** Modo `--exec <path> --exec-args <args>` funcional: el hijo, tras SIGCONT, hace `execvp` del binario externo. Sin handshake `ready`/`go` (el orquestador no lo espera).
- ☐ **CPP-04** Modo `--kernel` existente sigue funcionando sin regresión.
- ☐ **CPP-05** `--cgroup-path` es opcional/deprecated. Si no se pasa, no se intenta ninguna operación de cgroup. Si se pasa, se usa para aislamiento adicional pero NO para abrir perf.
- ☐ **CPP-06** `PerfCgroupReader` marcado como deprecated. No se elimina, pero deja de ser la ruta principal del launcher.
- ☐ **CPP-07** El collector recibe un `PerfReader` con PID externo sin cambios en su ruta caliente (`read(fd)` → `try_push` → `flush_producer` → `clock_nanosleep`).
- ☐ **CPP-08** Tests CTest existentes (9/9) siguen pasando. Agregar al menos un test nuevo que valide la apertura de perf sobre un PID externo (puede ser un proceso trivial de sleep).

**Total: 131 reglas + 8 reglas C++ = 139 reglas.**

---

## Parte VI — Plan de tests

*(Los tests completos están en el Plan de Tests del Orquestador — 135 tests unitarios/integración. Aquí se resumen los tests de integración más críticos.)*

### 13. Tests de integración obligatorios

| ID | Precondición | Qué se verifica | Resultado esperado |
|---|---|---|---|
| INT-T01 | PC local, kernels sintéticos en modo --exec | Pipeline completo sin excepción | Directorio de campaña con samples.csv, metadata.json, windows.csv |
| INT-T02 | 1 kernel NPB real (npb_ep clase S) | windows.csv con operational_intensity > 0 | ≥50 ventanas con quality_status='ok' |
| INT-T03 | Campaña con freqctl activo | **PRUEBA DE CAOS: SIGINT a mitad de corrida** | Governor/frecuencia de CADA core = estado previo a la campaña |
| INT-T04 | Interrumpir campaña (corrida 4 de 12) | Relanzar el mismo comando | Corridas 1–3 se saltan; la interrumpida se reintenta |
| INT-T05 | cloud_own con RAPL/cpufreq no disponibles | Campaña con frequency_control='unavailable' | not_eligible_for_training_dataset=True; pipeline funciona |
| INT-T06 | Campaña piloto local | Inspeccionar reporte | ≥90% corridas aceptadas |
| INT-T07 | Campaña local | Overhead baseline-vs-telemetry | Estable entre repeticiones (CV <10%) |
| INT-T08 | 2 kernels NPB (EP + MG) | phase_label_train en windows.csv | EP mayormente compute_bound, MG mayormente memory_bound |
| INT-T09 | Campaña piloto | node_profile.json y calibration_references.json | Todos los campos presentes, cv_pct calculado |
| INT-T10 | windows.csv | Features relativas | Presentes y numéricas para filas válidas |
| INT-T11 | Launcher SIN cgroup, kernel con subprocesos | Comparar conteo launcher vs `perf stat` externo | Conteos coinciden (<5%) — confirma medición por PID+inherit |

---

## Parte VII — Guía de desarrollo asistido por IA

### 14. Principio rector: un módulo, una sesión, una verificación

Cada módulo se desarrolla en su propia sesión con la IA, se verifica contra el checklist (Parte V) y el plan de tests correspondiente, y solo entonces se avanza al siguiente. No pedir a la IA que genere varios módulos a la vez.

### 15. Ciclo de trabajo por módulo

1. Reunir el contexto: copiar en la conversación la sección específica de este documento para ese módulo.
2. Usar como prompt de arranque la descripción del módulo (sección 10.N) + las reglas del checklist (sección 12.N) + los tests aplicables.
3. Pedir módulo + tests juntos — un módulo sin tests no está listo.
4. Correr los tests. Si fallan, iterar mostrando el error concreto.
5. Marcar en el checklist las reglas satisfechas, releyendo el código.
6. Solo entonces avanzar al siguiente módulo.

### 16. Riesgos típicos de la IA por módulo

| Módulo | Riesgo | Cómo verificar |
|---|---|---|
| manifest.py | La IA pone defaults silenciosos (seed=0, overwrite=False) en vez de fallar. | Correr MAN-T04 y MAN-T05 primero. |
| environment.py | Hardcodea rutas `/sys/...` sin capa inyectable para mocks. | Confirmar que tests NO usan rutas reales. |
| preflight.py | Cortocircuito (return al primer check que falla) ocultando los demás. | Provocar 2+ fallas simultáneas. |
| freqctl.py | Código que "se ve bien" pero nunca se probó contra sysfs real. | Prueba de caos INT-T03 obligatoria. |
| catalog.py | Checksum por tamaño de archivo, no por hash. | Modificar 1 byte sin cambiar tamaño; confirmar detección. |
| calibration.py | Agregar método alternativo de FLOPs vía PMU "por si acaso". | Grep de FP_ARITH en el código — no debe haber. |
| runner.py | No matar el proceso explícitamente al expirar timeout. | Verificar con psutil que el PID no existe tras timeout. |
| postprocess.py | Dividir sin verificar denominador; silenciar NaN con 0. | Grep de cada `/` y confirmar verificación del denominador. |
| validation.py | Checks en orden no determinista. | VAL-T07: 2 factores fallando a la vez, verificar mismo factor_id siempre. |
| campaign.py | Reimplementar lógica que ya vive en otro módulo. | Buscar duplicación de checksum, frecuencia, etc. |
| metadata_schema.py | `{**dict1, **dict2}` en vez de merge con detección de colisiones. | MET-T01: clave con valores distintos en ambos diccionarios. |

### 17. No delegables a la IA

- Prueba de caos de freqctl.py (INT-T03): requiere hardware real + presencia humana.
- Campaña piloto de integración (INT-T01 a INT-T11): requiere hardware bare-metal.
- Compilación y verificación de NPB/STREAM/ERT en el nodo real.
- Solicitudes administrativas a SC3 (permisos de cpufreq, GPU, reservas Slurm).

---

## Parte VIII — Preguntas abiertas

### 18. Pendientes de resolución

| Categoría | Pregunta | Bloqueante para |
|---|---|---|
| DVFS | ¿Cuándo se solicita a SC3 delegación de escritura cpufreq en felix? | Dataset DVFS de entrenamiento (F0–F4) |
| Energía | ¿Existe fuente de energía alternativa a RAPL en felix (PDU/rack)? | Features de energía/EDP |
| GPU | ¿Cuándo se solicita GPU NVIDIA por Slurm en felix? | Ruta GPU del proyecto |
| Director | ¿Contribución principal: modelo transferible o pipeline reproducible con modelos locales? | Alcance formal del trabajo de grado |
| Director | ¿Cuántos nodos y qué tan diversos para la validación? | Diseño del experimento cross-node |
| Compilación | ¿Toolchain Fortran disponible en felix para NPB? | Compilación de kernels de dataset |
| Compilación | ¿-march=native por nodo, o binario portable? | Comparabilidad entre nodos (Propuesta B) |
| Nomenclatura | ¿F0 = frecuencia máxima fija, o F0 = nativo? | Consistencia manifest/metadata/tests |

---

*Este documento es la fuente de verdad para la Fase 1. Si una instrucción de una sesión de trabajo contradice este documento, el documento tiene prioridad — señalar la contradicción en vez de resolverla por cuenta propia. El Registro ARC prevalece sobre este documento únicamente en hechos de plataforma observados en hardware real.*
