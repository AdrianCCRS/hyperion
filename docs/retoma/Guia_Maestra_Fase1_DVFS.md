

**GUÍA MAESTRA DE DESARROLLO**

**Fase 1: Plataforma de Recolección de Telemetría**

*Agente DVFS en Espacio de Usuario · Sistemas Heterogéneos · Modelos Ligeros de ML*

Universidad Industrial de Santander

Escuela de Ingeniería de Sistemas e Informática

Semillero de Investigación · Clúster SC3

*Documento único y autoritativo para la Fase 1*

*Unifica: metodología experimental · subsistema C++ · orquestador Python · checklist · tests · guía IA*

**Parte I — Contexto y Fundamentos**

# **1\. Propósito**

Este documento guía la construcción de la plataforma de recolección de telemetría (Fase 1 del trabajo de grado). El objetivo es generar, de forma reproducible y auditable, un dataset con el que se entrenará un clasificador ligero de ML que infiera si una aplicación está en régimen compute\_bound o memory\_bound, para que un agente DVFS actúe en consecuencia y optimice el Producto Energía-Retardo (EDP).

Lo que se construye aquí no es el clasificador ni el agente de control todavía. Es la infraestructura de recolección: un orquestador Python que envuelve al harness C++17 existente, lo conecta a kernels de benchmarking reales (NPB/STREAM/ERT), calibra el nodo, ejecuta campañas controladas y produce un dataset etiquetado por el modelo Roofline.

***Atención:** Si una instrucción de una sesión de trabajo contradice este documento, el documento tiene prioridad — señalar la contradicción en vez de resolverla por cuenta propia.*

# **2\. Plataforma real confirmada: felix como único nodo operativo**

Tras los diagnósticos de solo lectura contra hardware real del clúster SC3:

| Campo | Valor confirmado en felix.sc3.uis.edu.co |
| :---- | :---- |
| Topología | 4 nodos NUMA. CPUs 0-3 en NUMA 0; siblings SMT 32-35 fuera de la asignación de prueba. |
| cpufreq | acpi-cpufreq, 10 niveles discretos entre 1064 y 2261 MHz. El usuario NO tiene permiso de escritura todavía. |
| RAPL | No disponible (rapl\_capable=false). Ningún dominio de energía expuesto. |
| perf | Eventos genéricos de CPU más mem-loads. Suficiente para telemetría y etiquetado Roofline en modo nativo. |
| GPU | Pendiente de confirmar GPU NVIDIA asignada por Slurm. |
| Slurm | 24.11.5, cgroup v2, proctrack/cgroup. |

* **smexa y exadell quedan descartados:** exponen GPU AMD (MI210); el proyecto requiere NVIDIA.

* **Sin RAPL:** cualquier campaña en felix es válida para telemetría y etiquetado, pero no puede producir features de energía ni EDP. No inventar, imputar ni aproximar ese dato.

* **Sin escritura cpufreq:** las campañas corren en modo referencia nativa (REF), marcadas not\_eligible\_for\_training\_dataset para DVFS, hasta que administración SC3 delegue la escritura.

# **3\. Estrategia de medición: PID \+ inherit, sin dependencia de cgroup**

Esta sección documenta el mecanismo real de medición y es el cambio más importante respecto al diseño original del subsistema C++. El harness NO necesita conocer ni resolver la ruta del cgroup de Slurm para medir correctamente.

## **3.1 El mecanismo correcto**

Se usa perf\_event\_open apuntando al PID real del proceso hijo (workload), con pe.inherit \= 1, abierto después del fork pero antes de que el hijo haga exec(). El hijo se detiene con SIGSTOP hasta que los eventos estén armados:

// Secuencia correcta en telemetry\_kernel\_launcher (simplificada):  
   
pid\_t child \= fork();  
if (child \== 0\) {  
    // \--- En el hijo \---  
    raise(SIGSTOP);            // detenerse hasta que padre arme perf  
    execvp(workload, args);    // continuará cuando padre envíe SIGCONT  
}  
   
// \--- En el padre \---  
waitpid(child, \&status, WUNTRACED);  // esperar a que el hijo se detenga  
   
// Abrir perf sobre el PID REAL del hijo, no sobre sí mismo (pid=0)  
struct perf\_event\_attr pe;  
memset(\&pe, 0, sizeof(pe));  
pe.size         \= sizeof(pe);  
pe.type         \= PERF\_TYPE\_HARDWARE;  
pe.config       \= PERF\_COUNT\_HW\_INSTRUCTIONS;  
pe.inherit      \= 1;           // propaga a descendientes si los hay  
pe.disabled     \= 1;  
pe.exclude\_kernel \= 1;  
pe.exclude\_hv  \= 1;  
   
// Un fd por evento (inherit=1 impide PERF\_FORMAT\_GROUP)  
int fd\_instr  \= perf\_event\_open(\&pe, child, \-1, \-1, 0);  
pe.config \= PERF\_COUNT\_HW\_CPU\_CYCLES;  
int fd\_cycles \= perf\_event\_open(\&pe, child, \-1, \-1, 0);  
pe.config \= PERF\_COUNT\_HW\_CACHE\_REFERENCES;  
int fd\_cref   \= perf\_event\_open(\&pe, child, \-1, \-1, 0);  
pe.config \= PERF\_COUNT\_HW\_CACHE\_MISSES;  
int fd\_cmiss  \= perf\_event\_open(\&pe, child, \-1, \-1, 0);  
   
ioctl(fd\_instr,  PERF\_EVENT\_IOC\_RESET,  0);  
ioctl(fd\_instr,  PERF\_EVENT\_IOC\_ENABLE, 0);  
// ... repetir para los demás fd  
   
// Iniciar collector y consumer  
// collector lee fd\_instr, fd\_cycles, etc. periódicamente → ring SPSC  
   
kill(child, SIGCONT);   // el hijo hace exec() del workload  
                         // collector muestrea cada \--interval-ns: EN VIVO

## **3.2 Por qué NO funciona abrir sobre pid=0 y confiar en la herencia**

El patrón perf\_event\_open(pid=0, inherit=1) desde el proceso padre ANTES del fork es incorrecto para muestreo periódico. El kernel solo 'pliega' (suma) los contadores del hijo heredado de vuelta al fd del padre cuando el hijo termina, no en cada read() mientras el hijo sigue vivo.

Consecuencia: si el padre lee el fd periódicamente mientras el hijo corre, el valor se queda plano, sin reflejar el progreso real del hijo, hasta que el hijo termina y entonces aparece de golpe todo el trabajo acumulado. Esto invalida por completo el mecanismo de ventanas temporales (windows.csv), que requiere progreso incremental real en cada lectura.

***Atención:** Este es el error silencioso más difícil de detectar: la corrida termina, samples.csv tiene filas, pero los deltas entre muestras son cero o casi cero durante toda la ejecución, con un salto enorme en la última muestra.*

## **3.3 inherit=1: para qué sirve y para qué no**

* **Para qué sirve:** si el workload genera descendientes adicionales (poco probable en NPB/STREAM/ERT de un solo proceso con hilos OpenMP), inherit asegura que esos nietos también queden cubiertos.

* **Para qué NO sirve:** para 'llegar' al workload desde el orquestador confiando en la herencia padre→hijo, porque el plegado no es en vivo.

* **Hilos OpenMP:** un evento apuntado a un PID cubre automáticamente todos los hilos de ese grupo de tareas (tgid). inherit no es necesario para hilos, solo para procesos descendientes.

## **3.4 Caveat documentado: desfase entre contadores no agrupados**

Con inherit=1 activo, el kernel restringe la lectura agrupada (PERF\_FORMAT\_GROUP). Los eventos deben abrirse como descriptores separados y leerse en secuencia, introduciendo un desfase de microsegundos entre ellos. Este desfase es despreciable frente a la resolución de \--interval-ns (milisegundos). Es la única fuente real de imprecisión temporal que queda, y no tiene nada que ver con cgroups.

## **3.5 E06: la defensa real contra contaminación de datos**

El mecanismo PID+inherit resuelve la atribución correcta. Pero la contención de hardware compartido (otro proceso compitiendo por caché L3/ancho de banda de memoria en los mismos cores) es un efecto físico real independiente de la atribución. Se detecta verificando que no haya procesos ajenos con afinidad a delegated\_cpus, inspeccionando Cpus\_allowed de los procesos vivos — no por membresía de cgroup.

***Clave:** La medición correcta y el aislamiento del dato son dos problemas distintos. PID+inherit resuelve el primero. La verificación de afinidad (E06) resuelve el segundo. Ambas cosas sin cgroup.*

# **4\. Cambios en el subsistema C++ de telemetría**

Esta sección especifica exactamente qué archivos del harness C++ existente (rama fase-1/plataforma-experimental-simplificada) deben cambiar y cómo.

## **4.1 Tabla de cambios por archivo**

| Archivo | Tipo | Detalle |
| :---- | :---- | :---- |
| perf\_reader.hpp / perf\_reader.cpp | MODIFICAR | Agregar pe.inherit \= 1\. Cambiar firma para aceptar PID externo (el del hijo ya forkeado). Abrir cada evento como fd separado (no PERF\_FORMAT\_GROUP con inherit=1 activo). |
| perf\_cgroup\_reader.hpp / perf\_cgroup\_reader.cpp | DEPRECAR | Marcar como legacy/deprecated. No eliminar todavía (para no romper los 9 tests CTest actuales), pero dejar de ser la ruta principal del launcher. |
| telemetry\_kernel\_launcher.cpp | MODIFICAR (el más grande) | Implementar la secuencia stop→open→resume de la sección 3.1. Agregar modo \--exec \<path\> \--exec-args \<args\> para binarios externos. Mantener modo \--kernel sin regresión. |
| telemetry\_kernel\_workload.cpp | MODIFICAR (menor) | Reestructurar el handshake: (1) hijo nace y se detiene (SIGSTOP), (2) padre abre perf, (3) padre envía SIGCONT, (4) hijo hace setup/warmup, (5) hijo emite 'ready', (6) padre envía 'go'. |
| collector.hpp / collector.cpp | MODIFICAR (menor) | Verificar que el collector puede recibir un PerfReader con PID externo en vez de un PerfCgroupReader. La ruta caliente no cambia. |
| CLI del launcher | AGREGAR / DEPRECAR | \--exec \<path\> y \--exec-args \<args\> se agregan. \--cgroup-path pasa a ser opcional y no bloqueante si falta. |
| metrics.hpp, spsc\_ring.\*, rapl\_reader.\*, nvml\_reader.\* | SIN CAMBIOS | La estructura de datos, el ring, RAPL y NVML son independientes del mecanismo de apertura de perf. |
| Tests CTest existentes | REVISAR | Los 9 tests actuales no deben regresar. Agregar al menos un test nuevo: apertura de perf sobre un proceso hijo trivial (sleep) con PID externo. |

## **4.2 Nuevo flujo del launcher: secuencia completa**

telemetry\_kernel\_launcher (proceso padre)  
│  
│ Parse CLI: \--kernel o \--exec, \--perf-cpus, \--interval-ns, etc.  
│ POR CADA REPETICIÓN:  
│  
├── BASELINE (sin telemetría):  
│     fork() → hijo ejecuta workload (sin perf, sin collector)  
│     padre espera a que termine, registra wall-time  
│  
└── TELEMETRY:  
      1\. fork()  
         └── HIJO: raise(SIGSTOP) inmediatamente  
      2\. PADRE: waitpid(child, WUNTRACED) \-- confirmar que el hijo se detuvo  
      3\. PADRE: abrir PerfReader(child\_pid, inherit=1)  
               \-- un fd por evento (instructions, cycles, cache\_refs, cache\_misses)  
               \-- ioctl(RESET \+ ENABLE) en cada fd  
      4\. PADRE: iniciar hilo collector → ring SPSC  
         PADRE: iniciar hilo consumer → vector\<RecordedSample\>  
      5\. PADRE: kill(child, SIGCONT)  
         └── HIJO: despierta, hace exec() (modo \--exec) o setup interno (modo \--kernel)  
      6\. Si modo \--kernel: HIJO emite "ready", PADRE envía "go"  
         Si modo \--exec:   no hay handshake; collector empieza inmediatamente  
         Collector muestrea cada \--interval-ns: read(fd) → ring (EN VIVO)  
      7\. HIJO termina → PADRE: detener collector, join consumer  
         Exportar samples.csv, metadata.json, summary.txt  
         Cerrar los fd de perf

## **4.3 Modo \--exec para binarios externos (NPB, STREAM, ERT)**

* **El hijo, tras SIGCONT, hace execvp(exec\_path, exec\_args).** No hay handshake ready/go — el binario externo no coopera con el launcher.

* El collector empieza a muestrear inmediatamente tras SIGCONT.

* El warmup se resuelve en el post-procesamiento por tiempo de pared (warmup\_seconds del catálogo del orquestador), no dentro del binario.

* El success\_check del binario lo aplica el orquestador Python, no el launcher C++.

* El modo \--kernel existente sigue funcionando exactamente igual para kernels sintéticos de desarrollo.

## **4.4 Qué NO cambia en el subsistema C++**

El modelo de muestra (Sample, CpuSample, EnergySnapshot, GpuSample), el ring SPSC, RAPL (sysfs), la ruta caliente de Collector::run() (clock\_gettime → read(fd) → try\_push → flush\_producer → clock\_nanosleep), y las salidas (samples.csv, metadata.json, summary.txt). Solo cambia que los fd ahora apuntan a un PID externo con inherit, no a un cgroup.

## **4.5 Reglas de calidad del subsistema C++ (CPP-01 a CPP-08)**

### 

| ID | ✓ | Regla de validación / invariante técnica |
| :---- | :---: | :---- |
| CPP-01 | ☑ | PerfReader acepta PID externo con inherit=1; un fd por evento separado (no PERF\_FORMAT\_GROUP con inherit activo). |
| CPP-02 | ☑ | Launcher implementa stop→open→resume: fork, hijo se detiene, padre abre perf sobre PID del hijo, padre inicia collector/consumer, padre envía SIGCONT. |
| CPP-03 | ☑ | Modo \--exec funcional: el hijo, tras SIGCONT, hace execvp del binario externo. Sin handshake ready/go. |
| CPP-04 | ☑ | Modo \--kernel existente sin regresión. |
| CPP-05 | ☑ | \--cgroup-path es opcional/deprecated. Si no se pasa, no se intenta ninguna operación de cgroup. Si se pasa, se usa para aislamiento adicional pero NO para abrir perf. |
| CPP-06 | ☑ | PerfCgroupReader marcado como deprecated; los 9 tests CTest actuales siguen pasando. |
| CPP-07 | ☑ | El collector recibe un PerfReader con PID externo sin cambios en su ruta caliente. |
| CPP-08 | ☑ | Test nuevo: apertura de perf sobre proceso hijo trivial con PID externo; confirmar que las lecturas son en vivo y no planas (`perf_reader_pid_live_test`). |

**Parte II — Diseño Experimental**

# **5\. Principios rectores**

* **1\. Seguridad del nodo compartido.** Cualquier acción que module frecuencia o afinidad debe estar acotada exactamente a los recursos delegados y debe ser reversible. El aislamiento se garantiza por afinidad de CPU (cpuset de Slurm), no por cgroup.

* **2\. Reproducibilidad.** Cada corrida debe poder repetirse exactamente a partir de su metadata: comando, commit del harness y catálogo, hash del binario, host, CPUs (cpuset efectivo, no declarativo), governor/frecuencia, fecha y condiciones ambientales.

* **3\. Validez estadística del dato.** Que una corrida termine sin error no implica que sea útil para entrenamiento. El plan distingue entre 'la corrida no falló' y 'la corrida es apta para el dataset'.

* **4\. Separación estricta crudos/entrenamiento.** samples.csv es crudo. Nunca se entrena directamente sobre esa vista. Toda campaña produce windows.csv con deltas, tasas, intensidad operacional y banderas de validez.

* **5\. La etiqueta de entrenamiento se mide, no se asume.** phase\_label\_train se calcula comparando operational\_intensity contra I\_ridge del Roofline calibrado, por ventana. La expectativa de la literatura (phase\_label\_hint) es solo referencia de auditoría.

# **6\. Kernels: suites externas, no programados por el proyecto**

Los kernels NO se programan en este proyecto. Se usan binarios pre-compilados de suites reconocidas, conectados al launcher mediante un catálogo declarativo (kernels/catalog.yaml) y el modo \--exec.

| Capa | Suite | Rol | Propósito |
| :---- | :---- | :---- | :---- |
| Calibración de BW | STREAM (McCalpin, binario oficial) | calibration | Determina BW\_pico. No entra al dataset. |
| Calibración de FLOPs | ERT (Empirical Roofline Toolkit) | calibration | Determina P\_pico. No entra al dataset. |
| Dataset de entrenamiento | NAS Parallel Benchmarks (clases SER/OMP) | dataset | Genera las corridas de entrenamiento. |

Kernels NPB propuestos:

| kernel\_ref | Kernel NPB | phase\_label\_hint | Observación |
| :---- | :---- | :---- | :---- |
| npb\_ep | EP (Embarrassingly Parallel) | compute\_bound | Generación de números aleatorios; casi sin tráfico de memoria. |
| npb\_mg | MG (Multigrid) | memory\_bound | Acceso estructurado intensivo a memoria en malla 3D. |
| npb\_cg | CG (Conjugate Gradient) | memory\_bound | Acceso disperso a memoria (matriz rala). |
| npb\_is | IS (Integer Sort) | memory\_bound | Ordenamiento con acceso irregular; bajo IPC esperado. |
| npb\_ft | FT (Fast Fourier Transform) | intermedio | Balance no trivial entre cómputo y memoria. |
| npb\_lu / npb\_sp / npb\_bt | Solvers dispersos/estructurados | intermedio → compute\_bound | Mayor intensidad aritmética que MG/CG/IS. |

***Nota:** phase\_label\_hint es exclusivamente informativo. La etiqueta de entrenamiento real (phase\_label\_train) siempre se deriva por Roofline. Los kernels sintéticos stream\_triad, gemm\_naive y stencil\_2d permanecen en el repositorio exclusivamente para smoke tests y desarrollo del orquestador.*

# **7\. Calibración Roofline y etiquetado**

Antes de ejecutar la matriz de dataset, cada campaña ejecuta una fase de calibración obligatoria (una sola vez por nodo/sesión):

* 1\. Ejecutar STREAM sobre los cores delegados → BW\_pico (ancho de banda sostenido).

* 2\. Ejecutar ERT → P\_pico (rendimiento de cómputo pico).

* 3\. I\_ridge \= P\_pico / BW\_pico (en FLOPs/byte).

* 4\. Por cada ventana de cada corrida de dataset: I \= FLOPs\_del\_binario / bytes\_movidos\_por\_perf.

* 5\. phase\_label\_train \= 'memory\_bound' si I \< I\_ridge, 'compute\_bound' si I ≥ I\_ridge.

***Atención:** Los FLOPs NO se obtienen de un contador de PMU (FP\_ARITH\_INST\_RETIRED ni equivalentes — no son portables entre Intel/AMD ni entre generaciones). Se usan los FLOPs que el propio binario de la suite reporta por stdout, combinados con bytes movidos medidos por perf (LLC misses × tamaño de línea de caché del node\_profile).*

La calibración corre exclusivamente a la frecuencia máxima disponible (F0 si hay permiso de escritura, o frecuencia nativa si no), porque P\_pico y BW\_pico son los límites superiores del hardware.

# **8\. Matriz experimental**

## **8.1 Estados de frecuencia**

| Nivel | Configuración | Propósito |
| :---- | :---- | :---- |
| F0 | Governor userspace, f \= f\_max (si frequency\_write\_capable=True) | Límite superior de rendimiento/consumo. |
| F1 | f ≈ 75% del rango \[f\_min, f\_max\] | Intermedio alto. |
| F2 | f ≈ 50% del rango | Intermedio central. |
| F3 | f ≈ 25% del rango | Intermedio bajo. |
| F4 | f \= f\_min (si frequency\_write\_capable=True) | Límite inferior. |
| REF | Gobernador dinámico nativo, sin control manual | Referencia; no se usa para entrenar DVFS. Único nivel disponible en felix actualmente. |

***Nota:** Mientras felix no tenga frequency\_write\_capable=True, la campaña corre exclusivamente en REF. Esto sigue siendo válido para validar el pipeline y para telemetría no-DVFS.*

## **8.2 Parámetros de repetición**

| Parámetro | Valor propuesto | Justificación |
| :---- | :---- | :---- |
| Repeticiones por combinación | 10 | Relanzamientos independientes del binario (no iteraciones internas). |
| Warmup | Por tiempo de pared (warmup\_seconds del catálogo) | Sin cooperación interna del binario externo. |
| Ventanas mínimas por repetición | ≥ 50 tras excluir warmup | Suficientes para calcular deltas y descartar outliers. |
| Intervalo de muestreo | 1 ms (--interval-ns 1000000\) | Balance entre resolución temporal y sobrecarga del collector. |
| Orden de ejecución | Aleatorizado por combinación (seed fija) | Evita confundir deriva térmica/temporal con efecto del kernel o la frecuencia. |

Tamaño total: **6 kernels × 6 niveles × 10 repeticiones \= 360 corridas de telemetría \+ 360 baseline \+ calibración inicial.**

**Parte III — Módulos del Orquestador Python**

# **9\. Mapa de módulos y orden de construcción**

| \# | Módulo | Responsabilidad | Depende de |
| :---- | :---- | :---- | :---- |
| 1 | config.py | Configuración externa (orchestrator.toml): rutas sysfs, flags del harness | — |
| 2 | manifest.py | Parsea y valida campaign.yaml | config |
| 3 | environment.py | Detecta, de solo lectura, qué puede controlarse — única autoridad | config |
| 4 | diagnostics.py | Diagnóstico de arranque de solo lectura antes de cualquier campaña | environment, catalog |
| 5 | preflight.py | Verificaciones bloqueantes o de advertencia, antes de campaña y por corrida | manifest, environment |
| 6 | freqctl.py | Control y restauración garantizada de frecuencia — el más sensible | environment |
| 7 | catalog.py | Valida binarios externos desde kernels/catalog.yaml | manifest |
| 8 | calibration.py | Calibración Roofline: P\_pico, BW\_pico, I\_ridge | catalog, runner, freqctl |
| 9 | node\_profile.py | Perfil de hardware \+ referencias P95 (capa multinodo) | environment, runner |
| 10 | runner.py | Ejecuta una corrida individual del launcher (modo sintético y \--exec) | manifest, preflight, catalog |
| 11 | postprocess.py | samples.csv → windows.csv: deltas, intensidad operacional, phase\_label\_train, features relativas | calibration, node\_profile |
| 12 | validation.py | Acepta/rechaza cada corrida con un factor\_id explícito | runner, catalog |
| 13 | campaign.py | El integrador: genera la matriz, aleatoriza, secuencia, reanuda | todos los anteriores |
| 14 | metadata\_schema.py / report.py | Esquema de trazabilidad y reporte consolidado de campaña | validation, campaign |

Orden de construcción real:  
config.py, manifest.py y environment.py en paralelo  
  → diagnostics.py  
  → preflight.py  
  → runner.py (modo sintético, sin \--exec todavía)  
  → freqctl.py   ← PRUEBA DE CAOS antes de avanzar  
  → catalog.py  
  → calibration.py y node\_profile.py en paralelo  
  → runner.py (extensión \--exec)  
  → postprocess.py  
  → validation.py  
  → campaign.py  
  → metadata\_schema.py / report.py

# **10\. Especificación de cada módulo**

## **10.1 environment.py — Detección de capacidades del nodo**

El EnvironmentProfile separa en campos independientes lo que el hardware soporta de lo que el usuario puede hacer:

@dataclass  
class EnvironmentProfile:  
    tier: str                           \# local | cloud\_own | hpc\_sc3  
    frequency\_levels\_supported: bool    \# el driver/hardware expone niveles controlables  
    frequency\_write\_capable: bool       \# el USUARIO tiene permiso REAL de escritura  
    frequency\_control\_strategy: str     \# "discrete\_bounds" | "bounded\_range" | "unavailable"  
    frequency\_control\_paths: dict       \# rutas reales por CPU/policy  
    scaling\_driver: str  
    available\_frequencies\_khz: list\[int\]  
    rapl\_capable: bool  
    rapl\_domains\_available: list\[str\]   \# alias únicos: "package-0", "core-package-0"...  
    rapl\_domain\_paths: dict             \# alias → ruta sysfs real  
    numa\_nodes: int  
    smt\_siblings: dict\[int, list\[int\]\]  
    gpu\_present: bool  
    gpu\_vendor: str                     \# "nvidia" | "amd" | "none"  
    gpu\_exclusive\_hint: bool

***Atención:** frequency\_levels\_supported y frequency\_write\_capable son independientes. felix tiene levels\_supported=True y write\_capable=False. Nunca inferir uno a partir del otro.*

## **10.2 freqctl.py — Control de frecuencia (el módulo más sensible)**

Consume frequency\_control\_strategy de EnvironmentProfile. Ramifica su lógica según la estrategia (discrete\_bounds: valor discreto de una lista; bounded\_range: valor continuo en \[min, max\]; unavailable: no escribe nada). Nunca asume scaling\_setspeed como único mecanismo.

Funciones obligatorias:

* snapshot\_original\_state(cpus): UNA SOLA VEZ al inicio de campaña.

* apply\_frequency(cpus, level, env): verifica frequency\_write\_capable primero. Escribe en frequency\_control\_paths. Relee y compara (nunca asume éxito).

* restore\_original\_state(): idempotente. Verifica por lectura que la restauración ocurrió.

* install\_emergency\_handlers(): atexit, SIGINT, SIGTERM. Todos apuntando a restore\_original\_state().

***Atención:** Prueba de caos OBLIGATORIA en hardware bare-metal antes de usar contra el SC3: enviar SIGINT a mitad de una corrida y confirmar por lectura de sysfs que cada core delegado volvió exactamente al estado previo. Sin esta prueba, freqctl.py no está terminado.*

## **10.3 runner.py — Ejecución de corrida individual**

Construye el comando desde KernelEntry del catálogo. run\_id determinista: f'{campaign\_id}\_\_{kernel\_ref}\_\_{freq\_level.id}\_\_rep{n:02d}'. Timeout \= expected\_runtime\_seconds × SAFETY\_MARGIN (≥ 3×). Verifica no procesos hijos vivos. Fusiona metadata del launcher \+ orquestador.

\# Modo \--exec (kernels reales de dataset):  
cmd \= \[  
  "telemetry\_kernel\_launcher",  
  "--exec", entry.exec\_path,  
  "--exec-args", entry.exec\_args or "",  
  "--perf-cpus", manifest.cores.delegated\_cpus,  
  "--collector-cpu", str(manifest.cores.collector\_cpu),  
  "--consumer-cpu",  str(manifest.cores.consumer\_cpu),  
  "--interval-ns",   str(manifest.interval\_ns),  
  "--output-dir",    manifest.output\_dir,  
  "--run-id",        combination.run\_id,  
\]

## **10.4 postprocess.py — De samples.csv a windows.csv**

Columnas de salida obligatorias (REQUIRED\_OUTPUT\_COLUMNS):

run\_id, repetition, kernel\_ref, node\_id, phase\_label\_hint, phase\_label\_train,  
freq\_level\_id, freq\_khz\_requested, freq\_khz\_applied, freq\_khz\_observed,  
window\_index, t\_start\_ns, t\_end\_ns, delta\_t\_ns,  
delta\_instructions, delta\_cycles, delta\_cache\_references, delta\_cache\_misses,  
ipc, llc\_miss\_rate, mpki, ips,  
ipc\_relative, mpki\_relative, miss\_rate\_relative,  
delta\_running\_ns, delta\_enabled\_ns, running\_ratio,  
pkg\_delta\_uj, dram\_delta\_uj, power\_w, energy\_valid,  
flops\_window\_estimate, bytes\_moved\_window, operational\_intensity,  
i\_ridge\_used, roofline\_calibration\_ref, node\_profile\_ref, calibration\_ref,  
binary\_checksum, quality\_status

quality\_status válidos: 'ok', 'first\_sample\_no\_delta', 'warmup\_excluded', 'pmu\_degraded', 'energy\_invalid', 'no\_freq\_reading', 'intensity\_undefined'.

***Clave:** phase\_label\_train siempre es operational\_intensity vs i\_ridge. Nunca se copia de phase\_label\_hint. Las features relativas (ipc/mpki/miss\_rate\_relative) se calculan SIEMPRE en todas las filas válidas, sin recortarlas a \[0, 1\].*

## **10.5 preflight.py — Tabla completa de verificaciones**

Preflight de campaña (una vez, bloqueante o advertencia):

| factor\_id | Check | Bloqueante |
| :---- | :---- | :---- |
| E01 | Turbo/HWP (Intel) o CPB/CPPC (AMD): leer y fijar para toda la campaña. | Sí |
| E03 | Si hay cgroup delegado (opcional): cgroup HIJO de workload vacío. Nunca verificar contra el cgroup de la step del orquestador. | No (solo si existe) |
| E04 | NUMA: delegated\_cpus en un único nodo NUMA. | Sí |
| E05 | SMT: política declarada explícitamente en el manifest. | Sí |
| E09 | Si hay niveles fixed: frequency\_write\_capable=True. | Sí si hay fixed |
| E10 | Si hay niveles fixed: el dominio real de control de frecuencia (freqdomain\_cpus/related\_cpus/affected\_cpus) de cada core delegado debe estar contenido en delegated\_cpus (evita afectar cores de otro job en hardware con control por socket, ej. felix). Sin datos de dominio no bloquea. | Sí si hay fixed |
| I05 | Si rapl\_domains\_available vacío: forzar rapl.enabled a False sin bloquear. | No |
| I07 | output\_dir no existe (o overwrite: true). | Sí |
| I08 | manifest.rapl.domains ⊆ rapl\_domains\_available (alias reales). | Sí si rapl |
| I09 | Espacio libre en disco ≥ tamaño proyectado. | Sí |
| D05 | Eventos de perf solicitados ≤ PMCs disponibles del nodo. | Sí |
| OPS-01 | Presupuesto de hora-núcleo ≥ proyección de la campaña. | Sí |
| G01 | Si gpu.enabled: GPU NVIDIA confirmada, sin procesos CUDA ajenos. GPU AMD → deshabilitada. | Sí si gpu.enabled |
| C01/C02/C03 | Binario existe, checksum coincide, success\_check bien configurado. | Sí |
| D01–D04 | Toolchain, calibración ejecutada/parseable/plausible, CV% de referencias. | Según el caso |

Preflight reducido (por corrida):

| factor\_id | Check | Bloqueante |
| :---- | :---- | :---- |
| E02 | Temperatura de paquete dentro de rango normal (si hay sensor). | Sí si hay sensor |
| E06 | Sin procesos ajenos con afinidad a delegated\_cpus (por Cpus\_allowed, NO por cgroup). | Sí |
| E07 | Atributo real de gobierno coincide con el esperado (solo si hay niveles fixed). | Sí si hay fixed |
| E08 | Carga externa del nodo bajo umbral configurable. | Sí |
| I07 | run\_id de esta corrida no existe ya en output\_dir. | Sí |
| C01/C02/C03 | Binario/checksum/success\_check de esta combinación específica. | Sí |

**Parte IV — Estrategia Multinodo**

# **11\. Tres alternativas, una decisión pendiente**

| Alternativa | Idea central | Recomendación |
| :---- | :---- | :---- |
| A. Hardware explícito | Modelo global con descriptores del hardware de cada nodo como variables de entrada. | Reservar para trabajo futuro con más nodos diversos. |
| B. Features relativas | Modelo global con métricas normalizadas contra referencias de calibración del nodo. | Evaluar como experimento secundario de transferencia. |
| C. Modelo por nodo \+ pipeline reproducible | El modelo es local al nodo. Lo que se reutiliza entre nodos es el pipeline completo. | ADOPTAR como arquitectura oficial del proyecto. |

***Clave:** Las tres alternativas comparten la misma base: calibración, campaña, features por ventana. Divergen solo en qué se hace con los datos después. Por eso es posible seguir construyendo ya, sin esperar la decisión del director, siempre que la capa común capture lo que las tres necesitan.*

## **11.1 Estrategia 'sin arrepentimiento': construir ya, sin comprometerse con ninguna alternativa**

| Elemento a construir ya | Por qué sirve a las tres alternativas |
| :---- | :---- |
| node\_id en cada corrida y cada fila de windows.csv | Sin este campo no se puede hacer group-split por nodo bajo ninguna de las tres alternativas. |
| node\_profile.json (topología, caché, NUMA, frecuencia, eventos de perf, dominios RAPL) | Insumo directo de la Propuesta A y precondición para interpretar comparaciones entre nodos. |
| calibration\_references.json (P95 de IPC/MPKI/MissRate, CV%) | Insumo directo de la Propuesta B. Reutiliza la misma fase de calibración que ya se hace para I\_ridge. |
| Features relativas (ipc\_relative, mpki\_relative, miss\_rate\_relative) en windows.csv | Calcularlas ahora es casi gratuito. Recalcularlas retroactivamente exige tener el node\_profile de cada corrida vieja. |
| Pipeline versionado (commit hash) en la metadata de cada corrida | Dos campañas en dos nodos son comparables SOLO si corrieron exactamente el mismo protocolo versionado. |

***Atención:** NO comprometer tiempo de campaña ni presupuesto de hora-núcleo en ejecutar la matriz completa en un segundo o tercer nodo sin la decisión formal del director. Toda la infraestructura anterior puede construirse en un único nodo.*

**Parte V — Checklist de Validaciones Técnicas**

# **12\. Reglas por módulo — 139 reglas totales**

Cada regla tiene un ID único MÓDULO-NN y una casilla para marcar cuando el módulo la satisface. Un módulo no se declara terminado hasta que cada regla de su sección esté marcada — revisando el código, no solo por si los tests pasan.

### **12.1 Manifest (MAN-01 a MAN-11) — 11 reglas**

| ID | ✓ | Regla de validación / invariante técnica |
| :---- | :---: | :---- |
| MAN-01 | ☑ | cgroup\_path es OPCIONAL en todos los tiers. No es requisito para que perf mida correctamente. Excepción: obligatorio si environment\_tier es hpc\_sc3 (`manifest.py`, test\_man\_t02). |
| MAN-02 | ☑ | Rechazar si repetitions\_per\_combination \< 3\. (test\_man\_t03) |
| MAN-03 | ☑ | Calcular y loguear tamaño de la matriz antes de continuar. (test\_man\_t11\_tamano\_de\_matriz\_y\_log\_baseline) |
| MAN-04 | ☑ | Rechazar si output\_dir existe y overwrite: false (factor I07). (test\_man\_t04) |
| MAN-05 | ☑ | Rechazar si seed ausente — nunca generar semilla aleatoria (rompe reproducibilidad). (test\_man\_t05) |
| MAN-06 | ☑ | Rechazar si delegated\_cpus, collector\_cpu y consumer\_cpu se solapan. (test\_man\_t06) |
| MAN-07 | ☑ | calibration debe tener ≥1 kernel reports\_bandwidth\_stdout y ≥1 reports\_flops\_stdout. (test\_man\_t07) |
| MAN-08 | ☑ | Roles calibration/dataset sin solape entre secciones. (test\_man\_t08) |
| MAN-09 | ☑ | Todo kernel\_ref debe existir en catalog\_path. (test\_man\_t09) |
| MAN-10 | ☑ | frequency\_levels con exactamente un REF y los demás fixed con fraction ∈ \[0.0, 1.0\]. (test\_man\_t10, varios casos) |
| MAN-11 | ☑ | running\_ratio\_min ∈ (0.0, 1.0\] y interval\_ns \> 0\. (test\_man\_t11) |

### **12.2 Catálogo (CAT-01 a CAT-08) — 8 reglas**

| ID | ✓ | Regla de validación / invariante técnica |
| :---- | :---: | :---- |
| CAT-01 | ☑ | C01: exec\_path existe y es ejecutable (os.access X\_OK). Verificado con los 8 binarios reales compilados en felix (F3.1). |
| CAT-02 | ☑ | C02: sha256(exec\_path) coincide con binary\_checksum. Hash real, nunca por tamaño ni fecha. `catalog.yaml` tiene los sha256 reales de felix (F3.2). |
| CAT-03 | ☑ | C03: success\_check es tipo reconocido (exit\_code o stdout\_regex) y el regex compila. Los 6 kernels NPB comparten el patrón `Verification\s*=\s*SUCCESSFUL` confirmado contra stdout real (línea común de `print_results`, distinta del texto sugerido originalmente en el plan). |
| CAT-04 | ☑ | Kernel con role=dataset tiene phase\_label\_hint, size\_variant, expected\_runtime\_seconds y warmup\_seconds. `expected_runtime_seconds`/`warmup_seconds` calculados a partir de tiempos medidos en felix (F3.3), no estimados. |
| CAT-05 | ☑ | Kernel con role=calibration tiene exactamente uno de reports\_bandwidth\_stdout/reports\_flops\_stdout en true. Patrones confirmados contra stdout real de STREAM/ert\_probe. |
| CAT-06 | ☑ | resolve\_exec\_command() no inventa argumentos: exec\_args vacío → pasar cadena vacía, no omitir. Los 8 kernels reales usan `exec_args` vacío (sin argumentos CLI, tamaño fijado en compilación vía CLASS). |
| CAT-07 | ☑ | C01/C02 se repiten antes de cada corrida individual (no solo al inicio de campaña). Cubierto por `run_reduced_preflight()`, sin cambios de F3. |
| CAT-08 | ☑ | IDs únicos en el catálogo. Duplicados → CatalogValidationError. 8 IDs únicos en el catálogo real (2 calibration + 6 dataset). |

### **12.3 Entorno (ENV-01 a ENV-12) — 12 reglas**

| ID | ✓ | Regla de validación / invariante técnica |
| :---- | :---: | :---- |
| ENV-01 | ☑ | detect\_environment() es de SOLO LECTURA. Ningún otro módulo repite la detección. (test\_env\_t09\_deteccion\_no\_escribe\_archivos; confirmado en hardware real, F4.2) |
| ENV-02 | ☑ | frequency\_levels\_supported \= False si el driver no es real o solo hay una frecuencia disponible. (test\_env\_t02/t03) |
| ENV-03 | ☑ | RAPL se descubre recursivamente bajo /sys/class/powercap/. Si no existe ningún dominio o energy\_uj no cambia, rapl\_capable \= False. (test\_env\_t04/t05/t06; confirmado en felix: rapl\_capable=False, hardware anterior a RAPL) |
| ENV-04 | ☑ | El manifest no puede forzar rapl.enabled: true si environment.py determina rapl\_capable: false. (test\_env\_t07\_rapl\_del\_manifest\_se\_anula) |
| ENV-05 | ☑ | frequency\_write\_capable se determina con os.access(path, os.W\_OK), INDEPENDIENTE de frequency\_levels\_supported. (test\_entorno\_separa\_niveles...; confirmado en felix: frequency\_levels\_supported=True, frequency\_write\_capable=False, justo el caso que exige distinguir ambos campos) |
| ENV-06 | ☑ | Topología NUMA completa: nodos, cores por nodo, a qué nodo pertenecen los delegated\_cpus. (test\_env\_t08\_topologia\_numa\_delegada; confirmado en felix) |
| ENV-07 | ☑ | Siblings SMT de cada core delegado y política elegida en metadata. (test\_env\_t07\_politica\_smt\_se\_conserva\_en\_metadata; confirmado en felix) |
| ENV-08 | ☑ | Subconjunto real de eventos de perf soportados por esta PMU. (test\_env\_t08\_eventos\_perf\_disponibles; confirmado en felix) |
| ENV-09 | ☑ | environment\_report.json generado al inicio de campaña con todos los campos. (test\_env\_t10\_reporte\_de\_entorno; artefacto real escrito en felix en F4.2) |
| ENV-10 | ☑ | frequency\_control\_strategy ('discrete\_bounds', 'bounded\_range', 'unavailable') por atributos escribibles. Las tres ramas cubiertas explícitamente (test\_entorno\_separa\_niveles..., test\_env\_t02\_amd\_pstate\_es\_controlable, test\_env\_t03\_driver\_desconocido); confirmado en felix: discrete\_bounds (acpi-cpufreq). |
| ENV-11 | ☑ | Alias únicos por dominio RAPL (package-0, core-package-0, ...) nunca nombres genéricos. (test\_rapl\_descubre\_subdominios\_con\_identificadores\_unicos) |
| ENV-12 | ☐ | gpu\_vendor por detección real del dispositivo, nunca por nombre del nodo o modelo de CPU. **No implementado**: no existe el campo `gpu_vendor` en `EnvironmentProfile`, solo `gpu_present` (booleano). Requiere el inspector NVML real de ARC-22, todavía pendiente. |

### **12.4 Preflight (PRE-E01 a PRE-OPS01) — 26 reglas**

| ID | ✓ | Regla de validación / invariante técnica |
| :---- | :---: | :---- |
| PRE-E01 | ☑ | Turbo/HWP (Intel) o CPB/CPPC (AMD): leer y fijar para toda la campaña. (test\_e01\_snapshot\_y\_deriva\_turbo\_hwp; confirmado en felix, F4.2) |
| PRE-E02 | ☑ | Temperatura de paquete dentro de rango normal (si hay sensor). (test\_e02\_temperatura\_y\_e06\_procesos\_ajenos) |
| PRE-E03 | ☑ | Si hay cgroup delegado: cgroup HIJO de workload vacío. NUNCA contra el cgroup de la step del orquestador. (test\_pre\_t03\_cgroup\_con\_procesos) |
| PRE-E04 | ☑ | NUMA: delegated\_cpus en un único nodo NUMA. Bloqueante. (test\_pre\_t01\_numa\_en\_dos\_nodos; confirmado en felix, F4.2) |
| PRE-E05 | ☑ | SMT: política declarada explícitamente en el manifest. Bloqueante. (test\_pre\_t02\_politica\_smt\_obligatoria; confirmado en felix, F4.2) |
| PRE-E06 | ☑ | Sin procesos ajenos con afinidad a delegated\_cpus (Cpus\_allowed), NO por cgroup. Bloqueante. **Corrección 2026-08-01:** `check_foreign_processes()` en sí ya estaba probado (test\_e02\_temperatura\_y\_e06\_procesos\_ajenos), pero recibía la lista de PIDs ajenos ya armada — no existía ningún código que escaneara `/proc/*/status` de verdad. Además `run_reduced_preflight()` (que la invoca) nunca se llamaba desde `campaign.py`. Ambos huecos cerrados: `preflight.detect_foreign_affinity_pids()` escanea `Cpus_allowed` real; `campaign.py` lo corre antes de cada par baseline+telemetry y rechaza la combinación (factor\_id E06) si hay solape, sin llegar a medir. Ver ARC-40. |
| PRE-E07 | ☑ | Atributo real de gobierno coincide con el esperado (solo si hay niveles fixed). (test\_pre\_t04\_governor\_distinto, test\_caminos\_validos\_e05\_e07\_d02...) |
| PRE-E08 | ☑ | Carga externa del nodo bajo umbral configurado en el manifest. (test\_pre\_t05\_carga\_externa\_supera\_umbral) |
| PRE-E09 | ☑ | Si el manifest solicita algún nivel fixed: frequency\_write\_capable=True. Bloqueante. (test\_e09\_requiere\_permisos\_en\_todos\_los\_cores) |
| PRE-E10 | ☑ | Dominio real de control de frecuencia (freqdomain\_cpus/related\_cpus/affected\_cpus) contenido en delegated\_cpus; sin datos de dominio no bloquea. Bloqueante. Regla nueva, fuera del alcance original — ver ARC-30. **Verificado en hardware real (F4.2, 2026-08-01):** el primer preflight contra felix reveló que `_parse_cpu_list()` no entendía el formato real de `freqdomain_cpus` (lista separada por espacios, no por comas), dejando el check sin datos con qué bloquear pese a que los tests con mocks pasaban. Corregido — ver ARC-36. |
| PRE-I05 | ☑ | Si rapl\_domains\_available vacío, forzar rapl.enabled a False. No bloqueante. (test\_pre\_t07\_rapl\_wrap\_no\_disponible; confirmado en felix) |
| PRE-I07 | ☑ | output\_dir y run\_id no existen ya en disco. Bloqueante. (test\_pre\_t06\_run\_id\_existente; confirmado en felix) |
| PRE-I08 | ☑ | manifest.rapl.domains ⊆ rapl\_domains\_available del nodo (alias reales). Bloqueante si rapl.enabled. (test\_d01\_toolchain\_y\_checks\_de\_recursos) |
| PRE-I09 | ☑ | Espacio libre en disco ≥ tamaño proyectado de la campaña completa. Bloqueante. (test\_d01\_toolchain\_y\_checks\_de\_recursos; confirmado en felix con free\_bytes real) |
| PRE-C01 | ☑ | Binario existe y es ejecutable. Bloqueante. (test\_pre\_t08\_binario\_inexistente; confirmado en felix contra los 8 binarios reales de F3) |
| PRE-C02 | ☑ | Checksum del binario coincide con el catálogo. Bloqueante. (test\_pre\_t09\_checksum\_incorrecto; confirmado en felix con checksums reales) |
| PRE-C03 | ☑ | success\_check bien configurado antes de ejecutar. Bloqueante. (test\_c03\_success\_check\_valido\_e\_invalido; confirmado en felix) |
| PRE-D01 | ☑ | Toolchain disponible si se va a recompilar. Bloqueante solo si se recompila. (test\_d01\_toolchain\_y\_checks\_de\_recursos) |
| PRE-D02 | ☑ | Calibración STREAM/ERT ejecutada y parseable. Bloqueante si el nodo tiene RAPL. (test\_pre\_t10\_calibracion\_no\_parseable, test\_caminos\_validos...) |
| PRE-D03 | ☑ | BW\_pico y P\_pico dentro de ±40% de la ficha técnica declarada. Bloqueante si aplica D02. (test\_pre\_t11\_calibracion\_no\_plausible, test\_d04\_es\_advertencia\_con\_d02\_y\_d03\_validos) |
| PRE-D04 | ☑ | CV% de referencias P95 ≤ umbral. Solo advertencia (no bloqueante). (test\_pre\_t12\_calibracion\_inestable\_es\_advertencia) |
| PRE-D05 | ☑ | Eventos de perf solicitados ≤ PMCs disponibles. Bloqueante. `pmc_count` medido empíricamente con `environment.probe_pmc_count()` (nunca por modelo de CPU) — verificado en felix: 5 contadores simultáneos sin multiplexar. Ver ARC-37. |
| PRE-OPS01 | ☑ | Presupuesto de hora-núcleo ≥ proyección. Bloqueante. (test\_d01\_toolchain\_y\_checks\_de\_recursos; confirmado en felix) |
| PRE-G01 | ☑ | GPU NVIDIA confirmada, sin procesos CUDA ajenos. GPU AMD → deshabilitada. Bloqueante si gpu.enabled. Mecánica implementada y testeada con adaptador mock (test\_checks\_gpu\_mediante\_adaptador\_mock, test\_gpu\_reporta\_actividad\_y\_estado\_indisponible); GPU confirmada real vía `--gres=gpu:1` (H5), pero el adaptador NVML real para producción sigue pendiente — ver ARC-22. |
| PRE-G02 | ☑ | Persistence mode leído. No bloqueante. Misma nota que G01: mecánica lista, adaptador NVML real pendiente (ARC-22). |
| PRE-G03 | ☑ | Configuración MIG leída. No bloqueante. Misma nota que G01: mecánica lista, adaptador NVML real pendiente (ARC-22). |

### **12.5 Control de frecuencia (FRQ-01 a FRQ-10) — 10 reglas**

| ID | ✓ | Regla de validación / invariante técnica |
| :---- | :---: | :---- |
| FRQ-01 | ☑ | snapshot\_original\_state() llamado UNA SOLA VEZ al inicio de campaña (contrato: el caller la invoca una vez; la función es idempotente/de solo lectura por diseño). |
| FRQ-02 | ☑ | apply\_frequency() verifica por relectura del atributo real de frequency\_control\_paths (nunca asume scaling\_setspeed): discrete\_bounds usa governor=userspace+scaling\_setspeed, bounded\_range usa scaling\_min\_freq/scaling\_max\_freq. |
| FRQ-03 | ☑ | Guardar TANTO el valor solicitado COMO el aplicado en metadata (AppliedFrequency.requested\_khz/applied\_khz). Nunca solo uno. Verificado de punta a punta: `runner.run_single()` conserva el `AppliedFrequency` en `RunResult.applied_frequency` y lo funde en la metadata.json de la corrida (antes se descartaba); `campaign.py` lo propaga a `postprocess.run_postprocess()`. |
| FRQ-04 | ☑ | restore\_original\_state() es idempotente y verifica por lectura que la restauración ocurrió. |
| FRQ-05 | ☑ | install\_emergency\_handlers(): atexit, SIGINT y SIGTERM. Los tres registrados. |
| FRQ-06 | ☑ | Si frequency\_write\_capable=False: no escribir NADA en sysfs. Registrar 'unavailable' en metadata (write\_skipped\_reason). |
| FRQ-07 | ☐ | La calibración Roofline corre a frecuencia máxima/nativa. freqctl fija F0 antes y restaura al terminar. **Pendiente: requiere calibration.py (F2.3), aún no implementado.** |
| FRQ-08 | ☐ | PRUEBA DE CAOS en hardware real OBLIGATORIA antes de usar contra el SC3. **Pendiente — Parte H3, acción humana en bare-metal con root; no se ejecuta desde este entorno de desarrollo.** |
| FRQ-09 | ☑ | El control de frecuencia afecta SOLO a delegated\_cpus, nunca global del nodo (cpus siempre viene explícito del caller; probado con un cpu no delegado que queda intacto). |
| FRQ-10 | ☑ | Registrar frecuencia observada (scaling\_cur\_freq) en windows.csv: `campaign.py` la lee una vez con `freqctl.read_observed_frequency_khz()` justo después de la corrida y la pasa a `postprocess.run_postprocess()`. **Aproximación declarada:** es una lectura por corrida, no una muestra de sysfs sincronizada con cada ventana de perf (el harness C++ no captura `scaling_cur_freq` por tick); si la frecuencia cambiara a media corrida esto no lo reflejaría por ventana. |

### **12.6 Calibración (CAL-01 a CAL-11) — 11 reglas**

| ID | ✓ | Regla de validación / invariante técnica |
| :---- | :---: | :---- |
| CAL-01 | ☑ | La calibración corre a frecuencia máxima/nativa. Ejecutarla a frecuencia reducida subestimaría I\_ridge. (En felix REF≈F0 por `governor=performance`; el pineo explícito de F0 vía freqctl para tiers con escritura queda en FRQ-07, pendiente.) |
| CAL-02 | ☑ | BW\_pico del stdout de STREAM (auto-reportado por la suite). Nunca de contadores de PMU. |
| CAL-03 | ☑ | P\_pico del stdout de ERT. Nunca de FP\_ARITH\_INST\_RETIRED ni equivalentes. |
| CAL-04 | ☑ | I\_ridge \= P\_pico / BW\_pico. El check D03 ocurre en la misma función. Si D03 falla, excepción bloqueante. |
| CAL-05 | ☑ | roofline\_calibration.json incluye todos los campos: campaign\_id, timestamp, delegated\_cpus, BW/P/I\_ridge, stdout crudo, plausibility\_check\_passed. |
| CAL-06 | ☑ | load\_calibration() rechaza (excepción) si plausibility\_check\_passed=False. No etiquetar con calibración inválida. |
| CAL-07 | ☑ | build\_node\_profile() es de SOLO LECTURA: /proc/cpuinfo, /sys/devices/system/cpu/\*/cache/index\*/, /sys/.../node/. |
| CAL-08 | ☑ | node\_profile.json incluye todos los campos del dataclass NodeProfile. |
| CAL-09 | ☑ | build\_calibration\_references() corre ≥5 repeticiones del kernel de referencia para calcular P95. |
| CAL-10 | ☑ | Si cv\_pct \> umbral (defecto 5.0%): accepted=False. **Parcial:** el warning en el reporte de campaña depende de report.py (F2.8, aún no construido). |
| CAL-11 | ☑ | Los tres artefactos (roofline\_calibration.json, node\_profile.json, calibration\_references.json) se generan en la misma fase de campaña, antes de la matriz de dataset: `campaign.run_campaign()` (F2.7) las orquesta en ese orden exacto antes de `build_matrix()`. |

### **12.7 Runner (RUN-01 a RUN-08) — 8 reglas**

| ID | ✓ | Regla de validación / invariante técnica |
| :---- | :---: | :---- |
| RUN-01 | ☑ | El comando del launcher se construye SIEMPRE desde KernelEntry del catálogo. Nunca hardcodeado. |
| RUN-02 | ☑ | run\_id determinista: f'{campaign\_id}\_\_{kernel\_ref}\_\_{freq\_level.id}\_\_rep{n:02d}'. |
| RUN-03 | ☑ | Timeout \= entry.expected\_runtime\_seconds × SAFETY\_MARGIN (≥3×). Si expira, matar el proceso. |
| RUN-04 | ☑ | Verificar que no quedan procesos hijos vivos antes de continuar con la siguiente combinación. |
| RUN-05 | ☑ | Aplicar entry.success\_check (exit\_code o stdout\_regex) contra el resultado real. |
| RUN-06 | ☑ | Metadata final \= fusión launcher (samples\_collected, push\_retries) \+ orquestador (node\_id, binary\_checksum, refs de calibración). |
| RUN-07 | ☑ | stdout.txt y stderr.txt completos guardados en output\_dir/\<run\_id\>/. |
| RUN-08 | ☑ | Si frequency\_write\_capable=False: NO invocar freqctl.apply\_frequency() (verificado vía inyección de `apply_frequency` en `run_single`; freqctl.py aún no existe — F2.2). |

### **12.8 Campaign (CAM-01 a CAM-08) — 8 reglas**

| ID | ✓ | Regla de validación / invariante técnica |
| :---- | :---: | :---- |
| CAM-01 | ☑ | Aleatorizar SIEMPRE con random.Random(seed).shuffle(). Nunca en bloques por kernel o frecuencia (`build_matrix`, reproducible por seed). |
| CAM-02 | ☑ | La semilla y el orden completo de run\_ids ejecutados quedan en la metadata de campaña (`campaign_metadata.json`, escrito incrementalmente por si la campaña se interrumpe). |
| CAM-03 | ☑ | Reanudación: accepted=True → saltar (el par completo, baseline incluido); accepted=False → reintentar (un rechazo no es lo mismo que hecho). |
| CAM-04 | ☑ | Baseline y telemetry son par ATÓMICO (`schedule_runs`, run_id + sufijo `__baseline`). No se separan en el orden aleatorizado. |
| CAM-05 | ☑ | Se contabilizan hora-núcleo acumuladas (`CampaignProgress.total_core_hours`) por corrida. **Parcial:** la decisión operativa de "detenerse antes de lanzar la campaña completa" es una política humana (OPS-01), no automatizada aquí. |
| CAM-06 | ☑ | `campaign_timeout_seconds` aborta la matriz completa si se excede, además del timeout por corrida ya garantizado por runner.py (RUN-03). |
| CAM-07 | ☑ | Al cierre (normal o por interrupción, incluida una excepción durante la calibración) SIEMPRE se llama freqctl.restore\_original\_state() desde un `finally`, además de `install_emergency_handlers` para SIGINT/SIGTERM/atexit. |
| CAM-08 | ☑ | Overhead de instrumentación (`(telemetry.elapsed_seconds - baseline.elapsed_seconds) / baseline.elapsed_seconds * 100`) calculado por CADA par baseline+telemetry realmente ejecutado (no en pares reanudados por CAM-03), acumulado en `CampaignProgress.overhead_pct_values` y persistido en `campaign_metadata.json`. `report.py` expone `overhead_pct_mean`/`overhead_pct_cv`/`overhead_pct_samples` y una advertencia no bloqueante si `overhead_pct_cv` supera el 10% (gate de F4.4). Regla nueva, fuera del alcance original — ver ARC-34. |

### **12.9 Post-procesamiento (POST-01 a POST-16) — 16 reglas**

| ID | ✓ | Regla de validación / invariante técnica |
| :---- | :---: | :---- |
| POST-01 | ☑ | Primera muestra de cada repetición → quality\_status='first\_sample\_no\_delta'. Nunca imputar un delta artificial. |
| POST-02 | ☑ | Delta negativo de contador de hardware sin corrección de wrap → invalidar la ventana (mapeado a quality\_status='pmu\_degraded'; no hay un status dedicado en la lista de 7 valores). |
| POST-03 | ☑ | running\_ratio \< running\_ratio\_min → quality\_status='pmu\_degraded'. Ventana no apta para entrenar. |
| POST-04 | ☑ | Calcular tasas con el intervalo REAL medido (delta\_t\_ns real), nunca con el nominal de \--interval-ns. |
| POST-05 | ☑ | El launcher C++ ya aplica la corrección de wrap RAPL con max\_energy\_range\_uj y calcula energy\_delta\_valid por muestra (telemetry_kernel_launcher.cpp); postprocess.py solo propaga esos valores, nunca los recalcula ni los trata como reales cuando energy\_delta\_valid=0. |
| POST-06 | ☑ | RAPL inválido (energy\_delta\_valid=0) → pkg\_delta\_uj/dram\_delta\_uj/power\_w quedan en None, nunca se reporta como consumo real. |
| POST-07 | ☑ | Ventanas dentro de warmup\_seconds del catálogo → quality\_status='warmup\_excluded'. Se conservan en windows.csv. |
| POST-08 | ☑ | bytes\_moved\_window \== 0 → operational\_intensity=NaN y quality\_status='intensity\_undefined'. Nunca dividir por cero. |
| POST-09 | ☑ | FLOPs totales del stdout del binario (regex `flops_total_stdout_pattern` nuevo en catalog.py), prorrateados por ventana proporcionalmente a delta\_instructions (Plan\_Implementacion\_Medicion\_SC3.md F2.5). Método confirmado por el director sobre la alternativa de `docs/orchestator/plan_v3/guia-tecnica.md` (prorrateo por tiempo, descartado): los FLOPs son subconjunto de instrucciones retiradas, el tiempo está confundido por stalls de memoria. Ver ARC-27. F3.2 (2026-07-31): NPB confirmado en felix nunca imprime un total absoluto, solo `Mop/s total` (tasa) y `Time in seconds`; se agregaron `flops_rate_stdout_pattern`/`runtime_seconds_stdout_pattern` a `catalog.py`/`postprocess.extract_run_flops_total()` como fallback (tasa × 1e6 × tiempo) cuando `flops_total_stdout_pattern` está ausente. Ver ARC-32. |
| POST-10 | ☑ | LLC\_LINE\_SIZE\_BYTES del node\_profile real (`NodeProfile.cache_line_size_bytes`, leído de `coherency_line_size` en sysfs). Nunca hardcodeado como 64 bytes sin verificar. |
| POST-11 | ☑ | phase\_label\_train SIEMPRE por Roofline (operational\_intensity vs i\_ridge). Nunca copiado de phase\_label\_hint (probado explícitamente con hint contradictorio). |
| POST-12 | ☑ | Features relativas (ipc\_relative, mpki\_relative, miss\_rate\_relative) calculadas SIEMPRE que la feature absoluta y las referencias de calibración existan. |
| POST-13 | ☑ | Las features relativas NO se recortan a \[0, 1\]. Un ratio \> 1 es información válida (probado con ratio=2.0). |
| POST-14 | ☑ | node\_id, node\_profile\_ref y calibration\_ref en CADA FILA de windows.csv, incluida la primera (first\_sample\_no\_delta). |
| POST-15 | ☑ | load\_calibration() rechaza con excepción si plausibility\_check\_passed=False; run\_postprocess() la invoca antes de generar ninguna fila. |
| POST-16 | ☑ | windows.csv contiene TANTO features absolutas COMO features relativas: write\_windows\_csv() escribe las 39 REQUIRED\_OUTPUT\_COLUMNS en cada fila, siempre. |

### **12.10 Validación (VAL-01 a VAL-08) — 8 reglas**

| ID | ✓ | Regla de validación / invariante técnica |
| :---- | :---: | :---- |
| VAL-01 | ☑ | I04: samples\_collected==0 o push\_retries\>0 → rechazo inmediato. |
| VAL-02 | ☑ | I07: run\_id duplicado → rechazo, aunque no se detectara en el preflight (defensa adicional vía `run_id_seen`). |
| VAL-03 | ☑ | C02: checksum del binario ejecutado discrepante → rechazo, aunque la corrida terminara bien (segunda verificación independiente de la de runner.py/CAT-07). |
| VAL-04 | ☑ | C03: success\_check no cumplido → rechazo. |
| VAL-05 | ☑ | D03: calibración no plausible → rechazo de TODA LA CAMPAÑA, no solo de una corrida (`validate_campaign_calibration`; en la práctica ya bloqueado antes por `calibration.run_calibration`/`load_calibration`). |
| VAL-06 | ☑ | Corridas rechazadas NUNCA se borran. `write_verdict()` solo agrega `verdict.json`; ninguna función del módulo borra archivos. |
| VAL-07 | ☑ | Orden determinista de evaluación: I04 primero, luego C02/C03, luego E06-E08, luego I07. |
| VAL-08 | ☑ | Rechazo a nivel de ventana (I01/I02/I03, warmup, intensity\_undefined) no invalida la corrida completa: `validate_run()` no recibe windows.csv ni quality\_status como argumento, estructuralmente no puede verse afectado por eso. |

### **12.11 Metadata y reporte (MET-01 a MET-07) — 7 reglas**

| ID | ✓ | Regla de validación / invariante técnica |
| :---- | :---: | :---- |
| MET-01 | ☑ | merge(launcher\_meta, orchestrator\_meta) detecta colisiones de clave. Nunca {\*\*dict1, \*\*dict2}: `metadata_schema.merge_metadata()` es la única implementación (runner.py refactorizado para usarla). |
| MET-02 | ☑ | `frequency_restored_verified` en campaign\_metadata.json es el booleano que `freqctl.restore_original_state()` retorna tras releer sysfs, nunca del éxito del comando de escritura. |
| MET-03 | ☑ | node\_id es un identificador estable del nodo entre campañas, nunca el hostname de sesión: es un argumento explícito y obligatorio en `campaign.run_campaign()` y en `cli.py --node-id`; ningún código lo deriva de `socket.gethostname()`. |
| MET-04 | ☑ | El reporte de campaña muestra tabla por factor\_id con conteo y porcentaje que suman exactamente 100% (`report.build_factor_table`, la última fila absorbe el residuo del redondeo). |
| MET-05 | ☑ | Si cv\_pct \> umbral, el reporte lo señala como advertencia visible (`report.calibration_stability_warning`, D04). |
| MET-06 | ☑ | La semilla y el orden completo de run\_ids ejecutados quedan en `campaign_metadata.json`, incluyendo `skipped_run_ids` como categoría separada de `accepted_run_ids`. |
| MET-07 | ☑ | Trazabilidad completa: run\_id, kernel\_ref, node\_id, roofline\_calibration\_ref, node\_profile\_ref, calibration\_ref y binary\_checksum en la metadata.json de cada corrida (via `calibration_refs` pasado desde campaign.py) Y en cada fila de windows.csv. |

### **12.12 Estrategia multinodo (MLT-01 a MLT-08) — 8 reglas**

| ID | ✓ | Regla de validación / invariante técnica |
| :---- | :---: | :---- |
| MLT-01 | ☑ | node\_id en CADA corrida y en CADA FILA de windows.csv. Mismo mecanismo que POST-14/MET-03 (`node_id` es argumento explícito obligatorio, nunca `socket.gethostname()`). |
| MLT-02 | ☑ | node\_profile.json generado ANTES de la matriz de dataset, en la fase de calibración. `campaign.run_campaign()`: `build_node_profile()`/`write_node_profile()` se llaman inmediatamente después de `run_calibration()` y antes de `build_matrix()`. |
| MLT-03 | ☑ | calibration\_references.json con ≥5 repeticiones del kernel de referencia. `calibration.MIN_REFERENCE_REPETITIONS`, `ValueError` (CAL-09) si se pide menos. |
| MLT-04 | ☑ | Features relativas calculadas SIEMPRE, aunque la Propuesta B no se adopte. Mismo mecanismo que POST-12. |
| MLT-05 | ☑ | Los manifests son parametrizables cambiando SOLO environment\_tier y cores. Por diseño: ninguna ruta ni valor de felix está hardcodeado en el código del orquestador (todas las rutas sysfs vienen de `orchestrator.toml`/`SysfsPaths`, el catálogo de kernels es externo). No hay un test dedicado que instancie un segundo perfil sintético de nodo para verificarlo explícitamente — brecha menor de cobertura, no de implementación. |
| MLT-06 | ☐ | Commit hash del protocolo completo (harness, catálogo, orquestador) en la metadata de cada corrida. **No implementado**: no existe ningún campo `commit_hash` en `metadata_schema.py` ni en `runner.py`. Confirmado ausente en esta auditoría (2026-08-01). |
| MLT-07 | ☑ | \-march=native es aceptable si el modelo final es por nodo (Propuesta C). Decisión tomada y seguida en la práctica: el harness de telemetría se compila con `-march=native` (`telemetry/CMakeLists.txt`), mientras que NPB/STREAM/ert\_probe se compilan explícitamente SIN esa flag (ARC-32) porque sí son parte del dataset etiquetado. |
| MLT-08 | ☑ | NO ejecutar la matriz completa en un segundo nodo sin la decisión formal del director. Se cumple por ausencia: no existe ningún camino de código para orquestar ejecución multi-nodo hoy (`campaign.py` opera sobre un único `delegated_cpus`/nodo). Revisar esta regla si se implementa esa capacidad — ver H4 (decisión de alcance pendiente). |

### **12.13 Subsistema C++ (CPP-01 a CPP-08) — 8 reglas**

| ID | ✓ | Regla de validación / invariante técnica |
| :---- | :---: | :---- |
| CPP-01 | ☑ | PerfReader acepta PID externo con inherit=1; un fd por evento separado (no PERF\_FORMAT\_GROUP con inherit activo). `perf_reader.cpp::open()`: cada evento es su propio group leader (`group_fd=-1`), `inherit=1`. |
| CPP-02 | ☑ | Launcher implementa stop→open→resume: fork, hijo se detiene (SIGSTOP), padre abre perf sobre PID del hijo, padre inicia collector/consumer, padre envía SIGCONT. `telemetry_kernel_launcher.cpp::run_child()`. |
| CPP-03 | ☑ | Modo \--exec funcional: el hijo, tras SIGCONT, hace execvp del binario externo. Sin handshake ready/go. `build_workload_args()`: en modo externo los fds ready/go no se usan. Verificado en hardware real corriendo NPB/STREAM/ert\_probe bajo `--exec` (Fase 3, F3.4). |
| CPP-04 | ☑ | Modo \--kernel existente sin regresión. Los 9 (ahora 10, con CPP-08) tests CTest pasan. Confirmado en felix: los 10 CTest pasan en el nodo real (F3.4/F4.1). |
| CPP-05 | ☑ | \--cgroup-path es opcional/deprecated. Si no se pasa, no se intenta ninguna operación de cgroup. Comentario explícito en el código ("--cgroup-path is optional (CPP-05)"). |
| CPP-06 | ☑ | PerfCgroupReader marcado como deprecated; deja de ser la ruta principal del launcher. `@deprecated` en el docstring de `perf_cgroup_reader.hpp`/`.cpp`. |
| CPP-07 | ☑ | El collector recibe PerfReader con PID externo sin cambios en su ruta caliente. `collector.cpp`: `perf_reader_(cfg_.target_pid, -1)`, misma ruta `clock_gettime → read(fd) → try_push → flush_producer → clock_nanosleep`. |
| CPP-08 | ☑ | Test nuevo: apertura de perf sobre proceso hijo trivial (sleep) con PID externo; lecturas en vivo y no planas. `perf_reader_pid_live_test` — confirmado pasando en felix (5.35s, F3.4). |

**Parte VI — Plan de Tests de Integración**

# **13\. Tests de integración obligatorios**

Los tests unitarios por módulo (135 tests definidos en el Plan de Tests del Orquestador) deben existir en tests/orchestrator/ y correr en verde antes de avanzar a los tests de integración. Los tests de integración requieren hardware real.

| ID | Precondición | Qué se verifica | Resultado esperado |
| :---- | :---- | :---- | :---- |
| INT-T01 | PC local, kernels sintéticos en modo \--exec | Pipeline completo sin excepción | Directorio de campaña con samples.csv, metadata.json, windows.csv |
| INT-T02 | 1 kernel NPB real (npb\_ep, clase S) compilado | windows.csv con operational\_intensity \> 0 | ≥50 ventanas con quality\_status='ok' |
| INT-T03 | Campaña con freqctl activo (bare-metal con root) | PRUEBA DE CAOS: SIGINT a mitad de corrida | Governor/frecuencia de CADA core delegado \= estado previo exacto a la campaña |
| INT-T04 | Interrumpir campaña a mitad (corrida 4 de 12\) | Relanzar el mismo comando | Corridas 1-3 (accepted) se saltan; la interrumpida se reintenta; las 5-12 siguen |
| INT-T05 | cloud\_own con RAPL/cpufreq no disponibles (VM) | Campaña con frequency\_control='unavailable' | not\_eligible\_for\_training\_dataset=True en todas las corridas; pipeline funciona |
| INT-T06 | Campaña piloto local completa | Inspeccionar reporte de campaña | ≥90% de corridas aceptadas |
| INT-T07 | Campaña local | Overhead real baseline-vs-telemetry | Estable entre repeticiones de la misma condición (CV \<10%) |
| INT-T08 | 2 kernels NPB: EP \+ MG | phase\_label\_train en windows.csv | EP mayormente compute\_bound; MG mayormente memory\_bound |
| INT-T09 | Campaña piloto local | node\_profile.json y calibration\_references.json | Todos los campos presentes; cv\_pct calculado y con valor explícito |
| INT-T10 | windows.csv de la campaña piloto | Columnas ipc\_relative, mpki\_relative, miss\_rate\_relative | Presentes y numéricas para filas con quality\_status='ok' |
| INT-T11 | Launcher SIN cgroup, kernel NPB real con hilos OpenMP | Comparar conteo launcher vs perf stat externo | Conteos coinciden (\<5%) — confirma medición por PID+inherit sin cgroup |

***Atención:** INT-T03 (prueba de caos de freqctl.py) es la única prueba de toda la fase que requiere presencia humana activa durante la ejecución. No se puede delegar a CI. Nunca dar freqctl.py por terminado sin haberla ejecutado.*

**Parte VII — Guía de Desarrollo Asistido por IA**

# **14\. Principio rector y ciclo de trabajo**

Un módulo por sesión, una sesión por verificación. No pedir a la IA que genere varios módulos a la vez: la superficie de error crece más rápido que la capacidad de revisarla, y los módulos más sensibles (freqctl.py) no admiten ese riesgo.

## **14.1 Ciclo de trabajo por módulo**

* 1\. Copiar en la conversación la sección específica de esta guía para ese módulo (sección 10.N \+ reglas del checklist 12.N).

* 2\. Pedir el módulo junto con sus tests unitarios — un módulo sin tests no está listo para revisión.

* 3\. Correr los tests. Si fallan, iterar mostrando el error concreto.

* 4\. Marcar en el checklist las reglas satisfechas, releyendo el código línea a línea.

* 5\. Solo entonces avanzar al siguiente módulo en el orden de la sección 9\.

## **14.2 Riesgos típicos de la IA por módulo**

| Módulo | Riesgo típico | Cómo verificarlo |
| :---- | :---- | :---- |
| manifest.py | Pone defaults silenciosos (seed=0, overwrite=False) en vez de fallar. | Correr MAN-T04 y MAN-T05 primero. |
| environment.py | Hardcodea rutas /sys/... sin capa inyectable para mocks. | Confirmar que ningún test usa rutas reales del sistema de archivos. |
| preflight.py | Cortocircuito (return al primer check) ocultando los demás. | Provocar 2+ fallas simultáneas; confirmar que reporta ambas. |
| freqctl.py | Código que 'se ve bien' pero nunca probado contra sysfs real. | INT-T03 (prueba de caos) es obligatoria, no opcional. |
| catalog.py | Checksum por tamaño de archivo, no por hash sha256. | Modificar 1 byte del binario sin cambiar su tamaño; confirmar detección. |
| calibration.py | Agregar método alternativo de FLOPs vía PMU 'por si acaso'. | Grep de FP\_ARITH en el código — no debe haber ninguna referencia. |
| runner.py | No matar el proceso hijo explícitamente al expirar el timeout. | Verificar con psutil que el PID no existe tras timeout. |
| postprocess.py | Dividir sin verificar el denominador; silenciar NaN con 0\. | Grep de cada '/' en el código y confirmar verificación del denominador. |
| validation.py | Checks en orden no determinista entre ejecuciones. | VAL-T07: 2 factores fallando a la vez, verificar siempre el mismo factor\_id. |
| campaign.py | Reimplementar lógica que ya vive en otro módulo. | Buscar duplicación de checksum, frecuencia, etc. en el código generado. |
| metadata\_schema.py | {\*\*dict1, \*\*dict2} en vez de merge con detección de colisiones. | MET-T01: clave con valores distintos en ambos diccionarios. |

## **14.3 Lo que NO se puede delegar a la IA**

* **Prueba de caos de freqctl.py (INT-T03):** requiere hardware bare-metal con root y presencia humana durante la ejecución.

* **Campaña piloto de integración (INT-T01 a INT-T11):** requiere hardware real con NPB/STREAM/ERT compilados.

* **Compilación y verificación de NPB/STREAM/ERT en el nodo real:** los binarios deben compilarse en felix (o el PC local) y sus checksums registrarse en el catálogo.

* **Solicitudes administrativas a SC3:** permisos de escritura cpufreq, GPU NVIDIA, reservas Slurm.

**Parte VIII — Preguntas Abiertas**

# **15\. Pendientes de resolución**

| Categoría | Pregunta | Bloquea |
| :---- | :---- | :---- |
| DVFS | ¿Cuándo solicitar a SC3 delegación de escritura cpufreq en felix? | Dataset DVFS de entrenamiento (niveles F0–F4) |
| Energía | ¿Existe fuente de energía alternativa a RAPL en felix (PDU/rack)? | Features de energía/EDP |
| GPU | ¿Cuándo solicitar GPU NVIDIA asignada exclusivamente por Slurm en felix? | Ruta GPU del proyecto |
| Director | ¿Contribución principal: modelo transferible entre nodos, o pipeline reproducible con modelos locales? | Alcance formal del trabajo de grado |
| Director | ¿Cuántos nodos y qué tan diversos para la validación? | Diseño del experimento cross-node (Propuestas A/B) |
| Director | ¿F0 \= frecuencia máxima fija, o F0 \= nativa? ¿La nomenclatura de niveles es consistente? | Consistencia manifest/metadata/tests |
| Compilación | ¿Toolchain Fortran (gfortran) disponible en el entorno Conda de felix? | Compilación de kernels NPB en el SC3 |
| Compilación | ¿-march=native por nodo, o binario portable entre nodos? | Comparabilidad entre nodos si se explora Propuesta B |

