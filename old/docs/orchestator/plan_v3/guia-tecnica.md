**GUÍA TÉCNICA DE IMPLEMENTACIÓN**

**Orquestador de Campañas de Telemetría**

_Agente en Espacio de Usuario para la Gestión Dinámica de Frecuencia (DVFS)_

_Fase 1 - Recolección de telemetría, features y reproducibilidad_

Universidad Industrial de Santander

Escuela de Ingeniería de Sistemas e Informática

Documento técnico interno - para programar el orquestador

_Versión 3.0 - extiende la calibración para soportar la estrategia multinodo (perfil de hardware + referencias P95)_

# 0\. Punto de partida

Esta guía cubre la implementación del orquestador que envuelve al harness C++17 ya existente (telemetry_kernel_launcher, telemetry_kernel_workload, lectores de perf/RAPL/NVML). El harness resuelve la mecánica de bajo nivel de una corrida individual; el orquestador añade la capa de proceso: manifest declarativo, catálogo de kernels, calibración, control de frecuencia, secuenciación de campaña, validación y post-procesamiento.

Dos decisiones de diseño atraviesan toda la guía: los kernels de carga de trabajo no se programan dentro del proyecto (se usan binarios pre-compilados de suites externas), y la etiqueta de fase de entrenamiento no se asume por diseño del kernel, sino que se deriva empíricamente por nodo mediante el modelo Roofline.

**Atención:** Los kernels que antes vivían dentro de telemetry_kernel_workload (stream_triad, gemm_naive, stencil_2d) no desaparecen del repositorio: quedan como kernels sintéticos de desarrollo, usados exclusivamente para pruebas unitarias, la prueba de caos de freqctl.py, y validación del pipeline en local/cloud sin depender de tener las suites reales compiladas (ver sección 10). Nunca entran al dataset de entrenamiento.

## 0.1 Catálogo inicial de kernels

Se adopta un catálogo de tres capas: STREAM (binario oficial de McCalpin, calibración de ancho de banda), ERT - Empirical Roofline Toolkit (calibración de cómputo pico) y NAS Parallel Benchmarks, clases SER/OMP (kernels de dataset: EP, MG, CG, IS, FT y un solver del grupo LU/SP/BT). STREAM y ERT tienen rol calibration y nunca entran al dataset de entrenamiento; los kernels NPB tienen rol dataset.

## 0.2 Entornos de prueba disponibles

Mientras se gestionan los permisos del clúster (cgroup delegado, escritura en cpufreq, RAPL), se trabaja sobre tres niveles de entorno, y el orquestador debe saber en cuál está corriendo (campo environment_tier del manifest):

| **Tier**  | **Entorno**                                  | **Qué se puede validar**                                                                                                 | **Qué NO se puede validar todavía**                                                       |
| --------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| local     | PC de un investigador (bare-metal, con root) | Mecánica completa: perf, RAPL si el hardware lo expone, cpufreq real, restauración de frecuencia, overhead del collector | Contención multiusuario real, NUMA multi-socket si el PC no lo tiene, GPU compartida real |
| cloud_own | Servidor cloud propio, control total         | Igual que local si y solo si la instancia es bare-metal/dedicada                                                         | RAPL y cpufreq reales casi nunca disponibles en una VM estándar virtualizada (ver 0.3)    |
| hpc_sc3   | Nodo real del clúster SC3                    | Todo, incluyendo aislamiento por cgroup delegado y contención multiusuario                                               | -                                                                                         |

**_Nota:_** _En cualquiera de los tres entornos se necesita, además, un toolchain capaz de compilar NPB (gfortran + make) y STREAM/ERT (gcc). Verificar su disponibilidad es parte del preflight (sección 4)._

## 0.3 Advertencia técnica: RAPL y cpufreq en una VM cloud estándar

Esto puede condicionar una decisión de diseño, no es solo una nota al pie:

- energy_uj (RAPL) se lee vía /sys/class/powercap/intel-rapl/..., que depende de acceso a MSRs físicos. La mayoría de los hipervisores (KVM, Xen, VMware) no exponen esos MSRs a la VM invitada, salvo passthrough explícito. En una VM estándar, ese path de sysfs frecuentemente no existe o devuelve valores fijos/sin sentido.
- scaling_governor/scaling_setspeed de cpufreq dependen de que el kernel invitado tenga control real de P-states del hardware. En una VM estándar, el hipervisor gobierna la frecuencia física; el guest puede ver un cpufreq "acpi-cpufreq" ficticio que no cambia nada real, o directamente no exponer cpufreq.

**Atención:** Si el servidor cloud propio es una VM virtualizada normal (no bare-metal/dedicada), la campaña en ese tier debe correr en modo sin RAPL y sin control de frecuencia, sirviendo únicamente para validar la mecánica de perf, el pipeline de collector/consumer, overhead del propio harness y la lógica del orquestador - no para generar datos de entrenamiento reales de DVFS.

Antes de programar contra este tier, correr:

ls /sys/class/powercap/intel-rapl/ 2>&1

cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>&1

cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies 2>&1

Si el primer comando no lista nada o el segundo devuelve algo distinto de un driver real de hardware (p. ej. acpi-cpufreq con un único valor disponible), el tier cloud_own se marca como rapl_capable: false / freq_control_capable: false en su perfil de entorno, y el orquestador debe respetar esa bandera sin excepción.

**_Nota:_** _Esto no invalida usar el servidor cloud: sigue siendo perfecto para probar el 90% del código (manifest, preflight de SO, ejecución, timeouts, restauración de estado, post-procesamiento, rechazo de corridas) sin arriesgar nada en el SC3. Solo hay que tener claro que el dato de esa campaña no sirve para entrenar el modelo de DVFS, sirve para probar el prototipo._

## 0.4 Las tres alternativas para cuando existan varios nodos, y por qué esta guía las anticipa

Cuando el proyecto llegue a tener más de un nodo disponible, existen tres formas de organizar la relación entre telemetría, modelo y nodo. La decisión final depende del director, que está fuera de la oficina; esta guía adopta una estrategia "sin arrepentimiento": construir ya la capa común que sirve a las tres, sin comprometerse de forma irreversible con ninguna.

| **Alternativa**                                       | **Idea central**                                                                                                                                                                            |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A. Modelo global con hardware explícito               | Un único modelo entrenado con telemetría + descriptores explícitos del hardware de cada nodo (topología, caché, ancho de banda) como variables de entrada.                                  |
| B. Modelo global con features relativas               | Un único modelo entrenado sobre métricas normalizadas contra referencias de calibración propias de cada nodo (p. ej. IPC_relativo = IPC_ventana / IPC_referencia_nodo).                     |
| C. Modelo específico por nodo + pipeline reproducible | El modelo se trata como parte de la configuración de ese nodo. Lo que se reutiliza entre nodos es el pipeline completo (código, protocolo, esquema de datos), no los parámetros del modelo. |

La arquitectura recomendada es C (modelo por nodo, pipeline reproducible), con B como experimento secundario de transferencia y A reservado para trabajo futuro con más nodos diversos. El cuadro comparativo completo, con pros, contras y la recomendación detallada por criterio, está en la sección 13. Las tres alternativas comparten la misma base de trabajo - caracterización del nodo, calibración, campaña, features por ventana, criterios de calidad - y divergen únicamente en qué se hace con los datos después de recolectarlos. Por eso es posible seguir construyendo el orquestador y ejecutando campañas ya, sin esperar la decisión del director, siempre que la capa común capture todo lo que las tres alternativas necesitarían: identificador de nodo, perfil de hardware, referencias de calibración estables y features tanto absolutas como relativas. Esa capa común es exactamente lo que las secciones 5 (calibración) y 7 (post-procesamiento) de esta guía implementan.

# 1\. Arquitectura del orquestador

## 1.1 Qué cambia en el harness C++

El harness gana un modo de ejecución genérico, adicional al que ya existía (que se conserva para los kernels sintéticos de desarrollo):

Modo existente, ahora "modo dev/sintético" (se conserva tal cual):

telemetry_kernel_launcher --kernel gemm_naive --size 512 ...

Modo nuevo, para kernels reales de dataset:

telemetry_kernel_launcher --exec /ruta/bin/mg.S.x --exec-args "" \\

\--perf-cpus 2-5 --collector-cpu 0 --consumer-cpu 1 \\

\--output-dir runs/... --run-id ...

Internamente, --exec cambia el fork+exec de "lanzar telemetry_kernel_workload" a "lanzar el binario indicado". El resto de la mecánica (cgroup, afinidad, perf por PID/cgroup, RAPL, escritura de samples.csv/metadata.json) no cambia, porque ya es agnóstica de qué proceso se está midiendo.

## 1.2 Módulos nuevos u modificados respecto a la versión 1

| **Módulo**                                                                            | **Estado**           | **Responsabilidad**                                                                                            |
| ------------------------------------------------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------- |
| catalog.py                                                                            | Nuevo                | Parsear kernels/catalog.yaml, validar existencia y checksum de binarios, resolver kernel_ref a comando --exec. |
| calibration.py                                                                        | Nuevo                | Ejecutar STREAM y ERT, calcular P_pico, BW_pico, I_ridge, escribir roofline_calibration.json.                  |
| runner.py                                                                             | Modificado           | Construye el comando --exec en vez de --kernel/--size cuando el kernel_ref proviene del catálogo.              |
| postprocess.py                                                                        | Modificado           | Calcula intensidad operacional I por ventana y deriva phase_label_train comparando contra I_ridge.             |
| preflight.py                                                                          | Modificado           | Agrega checks C01-C04 (sección 4).                                                                             |
| environment.py, freqctl.py, campaign.py, validation.py, metadata_schema.py, report.py | Sin cambios de fondo | Se mantienen como en la versión 1.                                                                             |

## 1.3 Flujo de alto nivel

orchestrator run campaign.yaml

|

v

\[manifest.py\] parsear y validar manifest

|

v

\[catalog.py\] resolver kernel_ref -> binario, validar checksum (C01, C02)

|

v

\[environment.py\] detectar tier y capacidades reales del entorno

|

v

\[preflight.py\] preflight de campaña (incluye C01-C04) --> abortar si falla algo bloqueante

|

v

\[calibration.py\] ejecutar STREAM + ERT --> P_pico, BW_pico, I_ridge --> roofline_calibration.json

|

v

\[campaign.py\] generar matriz de kernels de dataset + aleatorizar con semilla

|

v

+---------------------------- por cada combinacion -----------------------------+

| \[preflight.py\] preflight reducido de corrida |

| \[freqctl.py\] fijar frecuencia y verificar aplicada |

| \[runner.py\] invocar telemetry_kernel_launcher --exec &lt;binario del catalogo&gt; |

| \[validation.py\] aplicar criterios de aceptacion/rechazo (seccion 8 del plan) |

+---------------------------------------------------------------------------------+

|

v

\[freqctl.py\] restaurar governor/frecuencia original

|

v

\[postprocess.py\] samples.csv -> windows.csv, con I y phase_label_train por ventana

|

v

\[report.py\] reporte consolidado de campaña

# 2\. Catálogo de kernels y manifest de campaña

## 2.1 Catálogo declarativo (kernels/catalog.yaml)

Vive separado del manifest de campaña para poder cambiar de suite sin tocar el resto del sistema.

\# kernels/catalog.yaml

kernels:

\- id: stream_official

suite: STREAM

role: calibration # nunca entra a la matriz de dataset

exec_path: bin/stream_c.exe

reports_bandwidth_stdout: true

binary_checksum: "sha256:...(se completa al compilar en el nodo objetivo)"

\- id: ert_probe

suite: ERT

role: calibration

exec_path: bin/ert_probe.x

reports_flops_stdout: true

binary_checksum: "sha256:..."

\- id: npb_ep

suite: NPB-OMP

role: dataset

exec_path: bin/ep.S.x

phase_label_hint: compute_bound # prior de literatura, NO la etiqueta de entrenamiento

size_variant: S

expected_runtime_seconds: 4

warmup_seconds: 1.0

success_check: {type: exit_code}

binary_checksum: "sha256:..."

\- id: npb_mg

suite: NPB-OMP

role: dataset

exec_path: bin/mg.S.x

phase_label_hint: memory_bound

size_variant: S

expected_runtime_seconds: 6

warmup_seconds: 1.0

success_check: {type: stdout_regex, pattern: "VERIFICATION SUCCESSFUL"}

binary_checksum: "sha256:..."

\# ... npb_cg, npb_is, npb_ft, npb_lu (mismo esquema)

## 2.2 Manifest de campaña

\# campaign.yaml

campaign_id: pilot_2026_08_local

environment_tier: local

seed: 20260803

output_dir: runs/pilot_2026_08_local

overwrite: false

catalog_path: kernels/catalog.yaml

calibration:

\- kernel_ref: stream_official

\- kernel_ref: ert_probe

kernels: # kernels de dataset, referenciados por id del catalogo

\- kernel_ref: npb_ep

\- kernel_ref: npb_mg

frequency_levels:

\- {id: F0, mode: fixed, fraction: 1.00}

\- {id: F1, mode: fixed, fraction: 0.75}

\- {id: F2, mode: fixed, fraction: 0.50}

\- {id: F3, mode: fixed, fraction: 0.25}

\- {id: F4, mode: fixed, fraction: 0.00}

\- {id: REF, mode: native_governor}

repetitions_per_combination: 10

target_windows_per_repetition: 50

interval_ns: 1000000

running_ratio_min: 0.90

cores:

delegated_cpus: "2-5"

collector_cpu: 0

consumer_cpu: 1

numa_node_pin: 0

cgroup_path: null

perf_enabled: true

rapl:

enabled: true

domains: \[package\]

gpu:

enabled: false

timeouts_seconds:

ready: 15

run: 300

shutdown: 10

## 2.3 Reglas de validación nuevas en manifest.py

- Todo kernel_ref referenciado, tanto en calibration como en kernels, debe existir en catalog_path; si no, rechazar el manifest antes de tocar el nodo.
- La sección calibration debe contener al menos un kernel role=calibration con reports_bandwidth_stdout y al menos uno con reports_flops_stdout; si falta alguno, no se puede calcular I_ridge.
- Ningún kernel con role=calibration puede aparecer en la sección kernels (dataset), ni viceversa; es un error de manifest, no una advertencia.

# 3\. Módulo de catálogo (catalog.py)

@dataclass

class KernelEntry:

id: str

suite: str

role: str # "dataset" | "calibration"

exec_path: str

binary_checksum: str

phase_label_hint: str | None # solo si role == "dataset"

size_variant: str | None

expected_runtime_seconds: int | None

warmup_seconds: float | None

success_check: dict | None

reports_bandwidth_stdout: bool = False

reports_flops_stdout: bool = False

def load_catalog(catalog_path: str) -> dict\[str, KernelEntry\]:

"""Parsea kernels/catalog.yaml a un diccionario id -> KernelEntry."""

def verify_binary(entry: KernelEntry) -> CheckResult:

"""

C01: os.path.isfile(entry.exec_path) y os.access(entry.exec_path, os.X_OK)

C02: sha256(entry.exec_path) == entry.binary_checksum

Retorna CheckResult con factor_id "C01" o "C02" segun cual falle.

"""

def resolve_exec_command(entry: KernelEntry) -> list\[str\]:

"""Traduce un KernelEntry a la lista de argumentos --exec/--exec-args del launcher."""

**_Nota:_** _verify_binary() se ejecuta tanto en el preflight de campaña (todos los kernels referenciados) como, de forma reducida, inmediatamente antes de cada corrida individual (solo el kernel de esa combinación), para detectar si algo cambió el binario a mitad de campaña._

# 4\. Preflight (preflight.py)

Se mantienen todos los checks de la versión 1 (E01-E08, I05-I07, G01-G03). Esta versión agrega la categoría C (catálogo y binarios) y D (calibración Roofline).

| **factor_id** | **Check**                                                                                                                       | **Bloqueante**               | **Momento**                            |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | -------------------------------------- |
| C01           | Binario del kernel existe y es ejecutable.                                                                                      | Sí                           | Campaña + por corrida                  |
| C02           | Checksum del binario coincide con el catálogo.                                                                                  | Sí                           | Campaña + por corrida                  |
| C03           | success_check del kernel está correctamente configurado (regex compila, tipo válido).                                           | Sí                           | Campaña                                |
| D01           | Toolchain necesario disponible (gfortran para NPB, gcc para STREAM/ERT), si se va a recompilar en este preflight.               | Solo si se recompila         | Campaña                                |
| D02           | Calibración STREAM/ERT ejecutada sin error y con salida parseable (BW/FLOPs reportados).                                        | Sí                           | Campaña, antes de generar la matriz    |
| D03           | BW_pico y P_pico dentro de un rango plausible frente a la ficha técnica declarada del hardware.                                 | Sí                           | Campaña, inmediatamente después de D02 |
| D04           | node_profile.json y calibration_references.json generados; cv_pct de las referencias P95 dentro del umbral (≤5%, configurable). | No (solo advertencia; ver 9) | Campaña, junto con D02/D03             |

**_Nota:_** _D03 no pretende ser una validación exacta: basta un rango amplio (por ejemplo, ±40% de lo que indica la ficha técnica del fabricante) para atrapar errores groseros de calibración, como correr STREAM con un solo hilo cuando se delegaron 4 cores, o medir con el governor equivocado._

# 5\. Módulo de calibración Roofline (calibration.py)

Se ejecuta una sola vez por campaña, después del preflight y antes de generar la matriz de kernels de dataset.

@dataclass

class RooflineCalibration:

campaign_id: str

timestamp: str

delegated_cpus: str

bw_pico_bytes_per_s: float

p_pico_flops_per_s: float

i_ridge_flops_per_byte: float

stream_raw_output: str

ert_raw_output: str

plausibility_check_passed: bool

def run_calibration(manifest: Manifest, catalog: dict\[str, KernelEntry\]) -> RooflineCalibration:

"""

1\. Localizar en manifest.calibration los kernel_ref con reports_bandwidth_stdout=True

(STREAM) y reports_flops_stdout=True (ERT).

2\. Ejecutar cada uno con runner.run_single(), igual que una corrida de dataset,

pero con role="calibration": no pasa por freqctl (corre a la frecuencia F0,

el limite superior, para medir el pico real) y no entra a windows.csv.

3\. Parsear la salida estandar de cada binario (regex especifico por suite,

declarado en catalog.py) para extraer BW_pico y P_pico.

4\. Calcular i_ridge_flops_per_byte = p_pico_flops_per_s / bw_pico_bytes_per_s.

5\. Ejecutar preflight D03 contra la ficha tecnica declarada del hardware

(manifest o environment.py); marcar plausibility_check_passed.

6\. Serializar a roofline_calibration.json dentro de output_dir de la campaña.

"""

def load_calibration(output_dir: str) -> RooflineCalibration:

"""Usado por postprocess.py para leer i_ridge_flops_per_byte al clasificar ventanas."""

**Atención:** La calibración corre exclusivamente a F0 (frecuencia máxima), porque P_pico y BW_pico son, por definición, los límites superiores del hardware. Ejecutarla a una frecuencia reducida subestimaría ambos picos y distorsionaría I_ridge para toda la campaña.

## 5.1 Parseo de FLOPs y bytes por ventana en postprocess.py

El detalle de cómo se prorratea el FLOP count reportado por un kernel de dataset a nivel de ventana, y cómo se combina con los bytes movidos medidos por perf, se documenta junto con el resto del post-procesamiento en la sección 7.

## 5.2 node_profile.py: perfil de hardware y referencias de calibración para la estrategia multinodo

Se agrega un módulo hermano de calibration.py que corre en el mismo momento de la campaña (después del preflight, junto con la calibración Roofline) y produce dos artefactos adicionales, necesarios para las Propuestas A y B descritas en la sección 0.4, sin los cuales esas dos alternativas quedarían cerradas para siempre sobre los datos ya recolectados.

@dataclass

class NodeProfile:

node_id: str

hostname: str

cpu_model: str

sockets: int

cores_total: int

threads_per_core: int

numa_nodes: int

cache_l1_kb: int

cache_l2_kb: int

cache_llc_kb: int

cache_llc_shared: bool

freq_min_khz: int

freq_max_khz: int

scaling_driver: str

perf_events_supported: list\[str\] # subconjunto realmente disponible en esta PMU

rapl_domains_available: list\[str\]

def build_node_profile(env: EnvironmentProfile, delegated_cpus: str) -> NodeProfile:

"""

Lee /proc/cpuinfo, /sys/devices/system/cpu/\*/cache/index\*/,

/sys/devices/system/node/, y el resultado de environment.detect_environment()

ya calculado. No ejecuta nada nuevo sobre el hardware: reorganiza informacion

de solo lectura ya disponible en un artefacto versionado y con un node_id estable.

"""

@dataclass

class CalibrationReferences:

node_id: str

ipc_p95: float

ips_p95: float

mpki_p95: float

miss_rate_p95: float

repetitions: int

cv_pct: float # coeficiente de variacion entre repeticiones

accepted: bool # True si cv_pct <= 5.0 (umbral configurable)

def build_calibration_references(calibration_runs: list\[RunResult\]) -> CalibrationReferences:

"""

Corre un microbenchmark de referencia (puede ser el mismo npb_ep para computo

y stream_official para memoria) repetido >= 5 veces, calcula P95 de IPC/IPS/MPKI/

MissRate entre esas repeticiones y su coeficiente de variacion.

Si cv_pct > 5.0, accepted=False y el preflight D03-equivalente para esta pieza

(ver tabla de abajo) bloquea la campaña hasta investigar la inestabilidad.

"""

| **Artefacto**                                                 | **Sirve a**                                                                       | **Se guarda en**                                                               |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| node_profile.json                                             | Propuesta A (variables de hardware explícitas)                                    | output_dir de la campaña, referenciado desde cada corrida vía node_profile_ref |
| calibration_references.json (IPC/IPS/MPKI/MissRate P95 + CV%) | Propuesta B (features relativas/normalizadas)                                     | output_dir de la campaña, referenciado desde cada corrida vía calibration_ref  |
| roofline_calibration.json (P_pico, BW_pico, I_ridge)          | Etiquetado de entrenamiento (sección 7.3), usado en las tres propuestas por igual | Ya definido en la sección 5 de esta guía (sin cambios)                         |

**_Nota:_** _Las tres calibraciones (Roofline, node_profile, calibration_references) se ejecutan en la misma fase de campaña, sobre las mismas corridas de STREAM/ERT ya definidas en 5.0/2.2, más algunas repeticiones adicionales del kernel de referencia para poder calcular el CV%. No se necesita una fase separada ni tiempo de nodo adicional significativo._

# 6\. Control de frecuencia (freqctl.py)

Este módulo se mantiene igual que en la versión 1 de esta guía: lectura y discretización de frecuencias disponibles, aplicación y verificación por lectura, y una rutina de restauración de emergencia registrada con atexit/SIGINT/SIGTERM que debe probarse con una prueba de caos antes de usarse contra hardware real. La única adición es que, durante la fase de calibración (sección 5), freqctl.py se invoca para fijar F0 antes de correr STREAM/ERT y se restaura junto con el resto de cores delegados al cierre de la campaña.

- apply_frequency(cpus, level, available_khz) → aplica y verifica por relectura de scaling_cur_freq.
- snapshot_original_state(cpus) / restore_original_state() → snapshot único al inicio de campaña, restauración idempotente.
- install_emergency_handlers() → atexit, SIGINT, SIGTERM.
- Si environment.freq_control_capable == False, se omite la escritura y las corridas quedan marcadas not_eligible_for_training_dataset: true, igual que en la versión 1.

# 7\. Ejecución de una corrida y post-procesamiento con clasificación Roofline

## 7.1 runner.py - construcción del comando desde el catálogo

def run_single(combination: Combination, env: EnvironmentProfile, manifest: Manifest,

catalog: dict\[str, KernelEntry\]) -> RunResult:

"""

1\. preflight reducido, incluyendo C01/C02 del kernel de esta combinacion.

2\. si env.freq_control_capable: freqctl.apply_frequency(...)

3\. entry = catalog\[combination.kernel_ref\]

cmd = \[

"telemetry_kernel_launcher",

"--exec", entry.exec_path,

"--exec-args", "",

"--perf-cpus", manifest.cores.delegated_cpus,

"--collector-cpu", str(manifest.cores.collector_cpu),

"--consumer-cpu", str(manifest.cores.consumer_cpu),

"--interval-ns", str(manifest.interval_ns),

"--output-dir", manifest.output_dir,

"--run-id", combination.run_id,

\]

4\. timeout = entry.expected_runtime_seconds \* SAFETY_MARGIN (p. ej. 3x)

5\. ejecutar con subprocess.run(cmd, timeout=timeout)

6\. aplicar entry.success_check contra exit_code/stdout (C03)

7\. verificar sin procesos hijos vivos; fusionar metadata del orquestador

(checksum del binario, kernel_ref, roofline_calibration_ref) con

la que produce el launcher.

"""

Combination.run_id se construye igual que en la versión 1, ahora a partir de kernel_ref en vez del nombre de kernel sintético:

run_id = f"{campaign_id}\_\_{kernel_ref}\_\_{freq_level.id}\_\_rep{repetition_index:02d}"

## 7.2 Warmup por tiempo de pared (en vez de iteraciones internas)

Como el binario externo no coopera con una señal de ready, el warmup se resuelve enteramente en el post-procesamiento, usando warmup_seconds del catálogo:

def mark_warmup(windows_df, entry: KernelEntry) -> pd.DataFrame:

"""

Para cada ventana con t_start_ns - t_start_ns_de_la_repeticion < entry.warmup_seconds \* 1e9:

quality_status = "warmup_excluded"

El resto continua el pipeline normal de calculo de deltas.

"""

## 7.3 postprocess.py - cálculo de intensidad operacional y etiqueta de entrenamiento

REQUIRED_OUTPUT_COLUMNS = \[

"run_id", "repetition", "kernel_ref", "node_id", "phase_label_hint", "phase_label_train",

"freq_level_id", "freq_khz_requested", "freq_khz_applied", "freq_khz_observed",

"window_index", "t_start_ns", "t_end_ns", "delta_t_ns",

"delta_instructions", "delta_cycles", "delta_cache_references", "delta_cache_misses",

"ipc", "llc_miss_rate", "mpki", "ips",

"ipc_relative", "mpki_relative", "miss_rate_relative", # ver 5.2

"delta_running_ns", "delta_enabled_ns", "running_ratio",

"pkg_delta_uj", "dram_delta_uj", "power_w", "energy_valid",

"flops_window_estimate", "bytes_moved_window", "operational_intensity",

"i_ridge_used", "roofline_calibration_ref", "node_profile_ref", "calibration_ref",

"binary_checksum",

"quality_status",

\]

def compute_relative_features(window_row, refs: CalibrationReferences) -> dict:

"""

Calcula, ademas de las features absolutas ya definidas:

ipc_relative = window_row.ipc / refs.ipc_p95

mpki_relative = window_row.mpki / refs.mpki_p95

miss_rate_relative = window_row.llc_miss_rate / refs.miss_rate_p95

No se recorta el ratio a \[0,1\]: puede superar 1 legitimamente si la ventana

supera la referencia P95 de calibracion, y ese exceso es informacion valida,

no un error a corregir.

Estas columnas quedan pobladas siempre, se use o no la Propuesta B mas adelante.

"""

def compute_operational_intensity(window_row, run_flops_total, run_duration_ns) -> float:

"""

flops_window_estimate = run_flops_total \* (window_row.delta_t_ns / run_duration_ns)

\-- prorrateo simple por duracion; asume FLOPs distribuidos uniformemente en el tiempo

\-- dentro de una misma repeticion (valido para NPB/STREAM/ERT, que no tienen fases

\-- internas conocidas de intensidad variable).

bytes_moved_window = window_row.delta_cache_misses \* LLC_LINE_SIZE_BYTES

return flops_window_estimate / bytes_moved_window if bytes_moved_window > 0 else float("nan")

def compute_windows(samples_df, run_metadata, calibration: RooflineCalibration,

node_refs: CalibrationReferences) -> pd.DataFrame:

"""

Igual que en la version 2 para deltas/IPC/energia/intensidad operacional, y ademas:

\- llama a compute_relative_features(window_row, node_refs) y agrega esas columnas

\- agrega node_id y las referencias (node_profile_ref, calibration_ref) a cada fila

"""

**Atención:** phase_label_train nunca se calcula por inferencia estadística ni se copia de phase_label_hint. Siempre es el resultado de comparar operational_intensity contra i_ridge_used de la calibración de esa sesión. Esto es lo que reemplaza por completo el mecanismo de phase_label_design/observed de la versión 1.

# 8\. Validación y criterios de rechazo (validation.py)

Se conservan todos los criterios I01-I07, E06-E08 de la versión 1. Se agregan:

| **factor_id** | **Condición de rechazo**                                                   | **Nivel**                                        |
| ------------- | -------------------------------------------------------------------------- | ------------------------------------------------ |
| C02           | Checksum del binario ejecutado no coincide con el catálogo.                | corrida completa                                 |
| C03           | success_check no se cumple (exit code o patrón de verificación).           | corrida completa                                 |
| D03           | Calibración de la sesión marcada no plausible.                             | campaña completa (bloquea toda la matriz)        |
| -             | operational_intensity indefinida (bytes_moved_window == 0) en una ventana. | ventana (quality_status = "intensity_undefined") |

# 9\. Metadata y reporte de campaña

{

"run*id": "pilot_2026_08_local*\_npb*mg*\_F2\_\_rep03",

"campaign_id": "pilot_2026_08_local",

"kernel_ref": "npb_mg",

"node_id": "uis-sc3-node07",

"node_profile_ref": "pilot_2026_08_local/node_profile.json",

"calibration_ref": "pilot_2026_08_local/calibration_references.json",

"binary_checksum": "sha256:...",

"roofline_calibration_ref": "pilot_2026_08_local/roofline_calibration.json",

"phase_label_hint": "memory_bound",

"environment_tier": "local",

"frequency_requested_khz": 2400000,

"frequency_applied_khz": 2400000,

"governor_restored_verified": true,

"accepted": true,

"rejection_factor_id": null

}

El reporte consolidado de campaña (report.py) agrega una fila resumen con i_ridge_flops_per_byte de la sesión, el porcentaje de ventanas con quality_status = "intensity_undefined", y ahora también el cv_pct de las referencias de calibración - si ese valor supera el umbral aceptado, el reporte lo marca como advertencia aunque no bloquee la campaña de por sí, ya que solo afecta a las Propuestas A/B, no a la C.

# 10\. Estrategia de pruebas en el entorno actual

Se mantiene el orden de la versión 1 (unitarias → local bare-metal → cloud propio), sin tocar el SC3. Se agregan pasos específicos del catálogo y la calibración.

## 10.1 Pruebas unitarias adicionales

- test_catalog.py: verificar que verify_binary() detecta correctamente un binario ausente (C01) y un checksum alterado (C02), usando binarios de prueba sintéticos, no NPB real.
- test_calibration.py: alimentar calibration.py con una salida de stdout sintética de STREAM/ERT (fixture) y verificar que I_ridge se calcula correctamente y que D03 rechaza una calibración con valores absurdos (por ejemplo, BW_pico de 1 byte/s).
- test_postprocess.py: agregar un caso con bytes_moved_window == 0 y confirmar que cae en quality_status = "intensity_undefined", no en una división por cero silenciosa.

## 10.2 Pruebas locales - pasos adicionales antes de la campaña piloto

- Compilar NPB (clases SER/OMP), STREAM y ERT en el PC local; registrar los checksums resultantes en kernels/catalog.yaml.
- Ejecutar calibration.py de forma aislada (sin matriz de dataset) y verificar que BW_pico/P_pico son razonables frente a la ficha técnica del PC (por ejemplo, contra el datasheet del fabricante del CPU).
- Ejecutar un kernel NPB de clase S de punta a punta (npb_ep, el más simple) y confirmar que aparece correctamente en windows.csv con operational_intensity y phase_label_train poblados.
- Prueba de caos (igual que en la versión 1): interrumpir con SIGINT a mitad de una corrida NPB real, no solo con el kernel sintético, y confirmar restauración de frecuencia.

## 10.3 Pruebas en el servidor cloud propio

- Repetir el diagnóstico de RAPL/cpufreq de la versión 1 antes de intentar nada.
- Verificar que el toolchain (gfortran, gcc, make) está disponible o se puede instalar sin privilegios especiales adicionales a los ya otorgados; si no, este entorno queda limitado a los kernels sintéticos de desarrollo hasta resolverlo.

# 11\. Roadmap de implementación incremental

| **Etapa** | **Entregable**                                                                                            | **Entorno donde se valida**                                           |
| --------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| A         | manifest.py, environment.py, preflight.py básico, runner.py en modo --kernel sintético (sin catálogo aún) | local + cloud_own                                                     |
| B         | freqctl.py completo con prueba de caos                                                                    | local (bare-metal)                                                    |
| C         | catalog.py + compilación y verificación de NPB/STREAM/ERT en al menos un entorno                          | local                                                                 |
| D         | calibration.py + roofline_calibration.json + preflight D01-D03                                            | local                                                                 |
| E         | runner.py en modo --exec real, postprocess.py con operational_intensity y phase_label_train               | local, con 1-2 kernels NPB                                            |
| F         | campaign.py con matriz completa de kernels de dataset, aleatorización y reporte consolidado               | local + cloud_own, campaña piloto de punta a punta                    |
| G         | Extensión a cgroup_path real y verificación de delegación (hpc_sc3)                                       | únicamente cuando estén los permisos del SC3 confirmados              |
| H         | Módulo GPU                                                                                                | se pospone explícitamente                                             |
| I         | node_profile.py + referencias de calibración P95 (capa "sin arrepentimiento" para A/B, sección 0.4)       | local, en paralelo con las etapas C-F; no bloquea nada de lo anterior |

# 12\. Preguntas que quedan abiertas para esta guía en particular

- ¿El PC local usado para la prueba de caos y la calibración es el mismo en el que después se ejecutará la campaña piloto completa? Recalibrar en cada máquina distinta es obligatorio si cambian.
- ¿El servidor cloud propio es bare-metal/dedicado, y tiene toolchain Fortran disponible para compilar NPB?
- ¿Se fija un único BW_pico/P_pico por campaña, o se recalibra si la campaña se extiende por varias sesiones en días distintos (posible deriva térmica ambiental de largo plazo)? Se recomienda recalibrar por sesión, no reutilizar una calibración antigua entre sesiones separadas por más de un día.

# 13\. Estrategia multinodo: comparación completa y qué módulo sirve a cada alternativa

Esta sección retoma en detalle las tres alternativas presentadas en 0.4, con el cuadro comparativo completo (pros, contras y recomendación por criterio), y traduce esa comparación a decisiones de implementación: qué módulo de esta guía sirve a cada alternativa.

## 13.1 Cuadro comparativo: pros, contras y recomendación

| **Criterio**                                                         | **A. Hardware explícito**                                                                                                              | **B. Valores relativos**                                                                                     | **C. Modelo por nodo**                                                                                          |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| Nodos requeridos para ser defendible                                 | Varios y realmente diversos                                                                                                            | Al menos 2, idealmente 3+                                                                                    | Uno por estudio; repetible sin límite                                                                           |
| Alineación con el alcance intra-nodo ya aprobado en el plan de grado | Baja, exige ampliar el alcance formal                                                                                                  | Media, se puede presentar como extensión                                                                     | Alta, es la continuación natural de lo ya aprobado                                                              |
| Riesgo de domain shift / sobreajuste a los pocos nodos disponibles   | Medio-alto: con 2-3 nodos el modelo puede memorizar diferencias entre nodos en vez de aprender una relación general                    | Medio: la normalización reduce diferencias de escala pero no garantiza equivalencia semántica de eventos PMU | Bajo dentro del nodo: cada modelo solo necesita generalizar a nuevas corridas de su propio dominio              |
| Trabajo experimental adicional requerido                             | Alto: perfiles de hardware, calibración de kernels por nodo, dataset conjunto con splits leave-one-node-out                            | Medio-alto: calibración estable por nodo (CV ≤ 5%) más el mismo entrenamiento conjunto                       | Medio: caracterización y campaña por nodo, pero reutilizando el mismo protocolo versionado                      |
| Qué queda como conclusión defendible con la evidencia esperada       | "Transfiere si se demuestra" - afirmación fuerte, difícil de sostener con pocos nodos                                                  | "Transferencia normalizada si se demuestra" - afirmación moderada, evaluable por ablación                    | "El pipeline es reproducible y cada modelo generaliza dentro de su nodo" - afirmación conservadora y alcanzable |
| Riesgo de sobreprometer en la sustentación                           | Alto                                                                                                                                   | Medio                                                                                                        | Bajo                                                                                                            |
| Recomendación                                                        | Reservar para trabajo futuro, o para una ampliación de alcance explícitamente acordada con el director si aparecen más nodos diversos. | Evaluar como experimento secundario de transferencia (no como requisito de éxito del proyecto).              | Adoptar como arquitectura oficial del proyecto.                                                                 |

**_Nota:_** _Esta recomendación es consistente con la restricción ya existente en el plan de grado aprobado, que limita formalmente la validación a un nodo y aclara explícitamente que no se busca un modelo universal de optimización energética._

## 13.2 Qué se pospone hasta la decisión del director

- Comprometer un número y una diversidad concreta de nodos para la campaña (necesario para A y B, no para C).
- Diseñar los splits leave-one-node-out y el experimento cross-node formal.
- Ampliar el alcance formal del trabajo de grado hacia "modelo transferible" en vez de "pipeline reproducible con modelos locales".
- Decidir si la compilación de los binarios de la suite (NPB/STREAM/ERT) debe ser -march=native por nodo (aceptable si el modelo es local, problemático si se busca comparar nodos directamente).

## 13.3 Qué módulo sirve a cada alternativa

Tabla de referencia rápida para no perder de vista, durante la implementación, por qué se está construyendo cada pieza.

| **Módulo**                                        | **A (hardware explícito)**                       | **B (features relativas)**                                | **C (modelo por nodo)**                                                       |
| ------------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------------------- |
| node_profile.py                                   | Imprescindible: es el insumo directo del modelo. | No se usa directamente.                                   | Útil como metadata de auditoría, no imprescindible.                           |
| calibration.py (referencias P95 + CV%)            | No se usa directamente.                          | Imprescindible: es el insumo directo de la normalización. | Útil, pero no imprescindible para entrenar un modelo local.                   |
| calibration.py (Roofline: P_pico/BW_pico/I_ridge) | Se reutiliza para etiquetar, igual que en C.     | Se reutiliza para etiquetar, igual que en C.              | Imprescindible: es el mecanismo de etiquetado de entrenamiento (sección 7.3). |
| catalog.py + adaptador de kernels externos        | Compartido sin cambios entre las tres.           | Compartido sin cambios entre las tres.                    | Compartido sin cambios entre las tres.                                        |
| postprocess.py (features absolutas + relativas)   | Usa solo las absolutas + node_profile.           | Usa las relativas ya calculadas.                          | Usa solo las absolutas.                                                       |
| campaign.py (matriz mono-nodo, repetible)         | Se repetiría idéntica en cada nodo del estudio.  | Se repetiría idéntica en cada nodo del estudio.           | Es exactamente el modo de uso previsto: una campaña por nodo.                 |

**_Nota:_** _Ningún módulo de esta guía es exclusivo de una sola alternativa salvo el propio entrenamiento del modelo (fuera del alcance de esta guía, que cubre orquestación y telemetría, no la Fase 2 de aprendizaje automático). Esa es precisamente la propiedad que hace posible seguir desarrollando sin esperar la decisión del director._