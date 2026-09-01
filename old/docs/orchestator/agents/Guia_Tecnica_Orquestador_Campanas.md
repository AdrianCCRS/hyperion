Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

# **GUÍA TÉCNICA DE IMPLEMENTACIÓN** 

### **Orquestador de Campañas de Telemetría** 

_Agente en Espacio de Usuario para la Gestión Dinámica de Frecuencia (DVFS) Fase 1 — Recolección de telemetría, features y reproducibilidad_ 

Universidad Industrial de Santander Escuela de Ingeniería de Sistemas e Informática Documento técnico interno — para programar el orquestador 

_Versión 3.0 — extiende la calibración para soportar la estrategia multinodo (perfil de hardware + referencias P95)_ 

Página 1 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

## **0. Punto de partida** 

Esta guía cubre la implementación del orquestador que envuelve al harness C++17 ya existente (telemetry_kernel_launcher, telemetry_kernel_workload, lectores de perf/RAPL/NVML). El harness resuelve la mecánica de bajo nivel de una corrida individual; el orquestador añade la capa de proceso: manifest declarativo, catálogo de kernels, calibración, control de frecuencia, secuenciación de campaña, validación y post-procesamiento. 

Dos decisiones de diseño atraviesan toda la guía: los kernels de carga de trabajo no se programan dentro del proyecto (se usan binarios pre-compilados de suites externas), y la etiqueta de fase de entrenamiento no se asume por diseño del kernel, sino que se deriva empíricamente por nodo mediante el modelo Roofline. 

**Atención:** Los kernels que antes vivían dentro de telemetry_kernel_workload (stream_triad, gemm_naive, stencil_2d) no desaparecen del repositorio: quedan como kernels sintéticos de desarrollo, usados exclusivamente para pruebas unitarias, la prueba de caos de freqctl.py, y validación del pipeline en local/cloud sin depender de tener las suites reales compiladas (ver sección 13). Nunca entran al dataset de entrenamiento. 

#### **0.1 Catálogo inicial de kernels** 

Se adopta un catálogo de tres capas: STREAM (binario oficial de McCalpin, calibración de ancho de banda), ERT — Empirical Roofline Toolkit (calibración de cómputo pico) y NAS Parallel Benchmarks, clases SER/OMP (kernels de dataset: EP, MG, CG, IS, FT y un solver del grupo LU/SP/BT). STREAM y ERT tienen rol calibration y nunca entran al dataset de entrenamiento; los kernels NPB tienen rol dataset. 

#### **0.2 Entornos de prueba disponibles** 

Mientras se gestionan los permisos del clúster (cgroup delegado, escritura en cpufreq, RAPL), se trabaja sobre tres niveles de entorno, y el orquestador debe saber en cuál está corriendo (campo environment_tier del manifest): 

|**Tier**|**Entorno**|**Qué se puede validar**|**Qué NO se puede**<br>**validar todavía**|
|---|---|---|---|
|local|PC de un investgador<br>(bare-metal, con root)|Mecánica completa: perf, RAPL<br>si el hardware lo expone,<br>cpufreq real, restauración de<br>frecuencia, overhead del<br>collector|Contención<br>multusuario real,<br>NUMA mult-socket si<br>el PC no lo tene, GPU<br>compartda real|
|cloud_own|Servidor cloud propio,<br>control total|Igual que local si y solo si la<br>instancia es<br>bare-metal/dedicada|RAPL y cpufreq reales<br>casi nunca disponibles<br>en una VM estándar<br>virtualizada (ver 0.3)|
|hpc_sc3|Nodo real del clúster SC3|Todo, incluyendo aislamiento|—|



Página 2 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

|**Tier**|**Entorno**|**Qué se puede validar**|**Qué NO se puede**<br>**validar todavía**|
|---|---|---|---|
|||por cgroup delegado y<br>contención multusuario||



**_Nota:_** _En cualquiera de los tres entornos se necesita, además, un toolchain capaz de compilar NPB (gfortran + make) y STREAM/ERT (gcc). Verificar su disponibilidad es parte del preflight (sección 5)._ 

#### **0.3 Advertencia técnica: RAPL y cpufreq en una VM cloud estándar** 

Esto puede condicionar una decisión de diseño, no es solo una nota al pie: 

- energy_uj (RAPL) se lee vía /sys/class/powercap/intel-rapl/..., que depende de acceso a MSRs físicos. La mayoría de los hipervisores (KVM, Xen, VMware) no exponen esos MSRs a la VM invitada, salvo passthrough explícito. En una VM estándar, ese path de sysfs frecuentemente no existe o devuelve valores fijos/sin sentido. 

- scaling_governor/scaling_setspeed de cpufreq dependen de que el kernel invitado tenga control real de P-states del hardware. En una VM estándar, el hipervisor gobierna la frecuencia física; el guest puede ver un cpufreq "acpi-cpufreq" ficticio que no cambia nada real, o directamente no exponer cpufreq. 

**Atención:** Si el servidor cloud propio es una VM virtualizada normal (no bare-metal/dedicada), la campaña en ese tier debe correr en modo sin RAPL y sin control de frecuencia, sirviendo únicamente para validar la mecánica de perf, el pipeline de collector/consumer, overhead del propio harness y la lógica del orquestador — no para generar datos de entrenamiento reales de DVFS. 

Antes de programar contra este tier, correr: 

ls /sys/class/powercap/intel-rapl/ 2>&1 <mark>cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>&1</mark> cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies 2>&1 

Si el primer comando no lista nada o el segundo devuelve algo distinto de un driver real de hardware (p. ej. acpi-cpufreq con un único valor disponible), el tier cloud_own se marca como rapl_capable: false / freq_control_capable: false en su perfil de entorno, y el orquestador debe respetar esa bandera sin excepción. 

**_Nota:_** _Esto no invalida usar el servidor cloud: sigue siendo perfecto para probar el 90% del código (manifest, preflight de SO, ejecución, timeouts, restauración de estado, post-procesamiento, rechazo de corridas) sin arriesgar nada en el SC3. Solo hay que tener claro que el dato de esa campaña no sirve para entrenar el modelo de DVFS, sirve para probar el prototipo._ 

#### **0.4 Las tres alternativas para cuando existan varios nodos, y por qué esta guía las anticipa** 

Cuando el proyecto llegue a tener más de un nodo disponible, existen tres formas de organizar la relación entre telemetría, modelo y nodo. La decisión final depende del director, que está fuera de la oficina; esta guía 

Página 3 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

adopta una estrategia "sin arrepentimiento": construir ya la capa común que sirve a las tres, sin comprometerse de forma irreversible con ninguna. 

|**Alternatva**|**Idea central**|
|---|---|
|A. Modelo global con hardware<br>explícito|Un único modelo entrenado con telemetría + descriptores<br>explícitos del hardware de cada nodo (topología, caché, ancho de<br>banda) como variables de entrada.|
|B. Modelo global con features<br>relatvas|Un único modelo entrenado sobre métricas normalizadas contra<br>referencias de calibración propias de cada nodo (p. ej.<br>IPC_relatvo = IPC_ventana / IPC_referencia_nodo).|
|C. Modelo específco por nodo +<br>pipeline reproducible|El modelo se trata como parte de la confguración de ese nodo.<br>Lo que se reutliza entre nodos es el pipeline completo (código,<br>protocolo, esquema de datos), no los parámetros del modelo.|



La arquitectura recomendada es C (modelo por nodo, pipeline reproducible), con B como experimento secundario de transferencia y A reservado para trabajo futuro con más nodos diversos. El cuadro comparativo completo, con pros, contras y la recomendación detallada por criterio, está en la sección 16. Las tres alternativas comparten la misma base de trabajo — caracterización del nodo, calibración, campaña, features por ventana, criterios de calidad — y divergen únicamente en qué se hace con los datos después de recolectarlos. Por eso es posible seguir construyendo el orquestador y ejecutando campañas ya, sin esperar la decisión del director, siempre que la capa común capture todo lo que las tres alternativas necesitarían: identificador de nodo, perfil de hardware, referencias de calibración estables y features tanto absolutas como relativas. Esa capa común es exactamente lo que las secciones 6 (calibración) y 10 (post-procesamiento) de esta guía implementan. 

## **1. Arquitectura del orquestador** 

#### **1.1 Qué hace el harness C++ y qué le falta al orquestador** 

El harness (telemetry_kernel_launcher + telemetry_kernel_workload) resuelve, por corrida individual: 

- lanzar baseline y telemetry como procesos hijos separados, 

- abrir perf/RAPL/NVML y producir samples.csv + metadata.json + summary.txt, 

- aplicar cgroup y afinidad si se le pasan por CLI, 

- desde esta guía en adelante, lanzar un binario arbitrario en modo --exec, además del modo --kernel sintético original. 

Lo que el harness no hace, y es exactamente el trabajo del orquestador: 

- generar la matriz completa de combinaciones (kernel × frecuencia × repetición) a partir de un manifest declarativo, 

Página 4 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

- resolver un catálogo de kernels externos (suite, binario, checksum) a un comando ejecutable concreto, 

- fijar y restaurar el estado de frecuencia del sistema antes/después de invocar al launcher, 

- calibrar el nodo (Roofline, perfil de hardware, referencias de estabilidad) antes de correr la matriz de dataset, 

- decidir en qué orden ejecutar las combinaciones (aleatorización con semilla), 

- correr el preflight de entorno antes de cada corrida y antes de la campaña completa, 

- aplicar los criterios de aceptación/rechazo por corrida, 

- invocar el post-procesamiento (samples.csv → windows.csv) y agregar los resultados de campaña, 

- producir el reporte consolidado de campaña con conteo de corridas aceptadas/rechazadas por factor. 

#### **1.2 Modo de ejecución genérico del launcher (--exec)** 

El launcher expone dos modos de invocación: 

Modo "dev/sintético" (para pruebas del propio orquestador, sección 13): <mark>telemetry_kernel_launcher --kernel gemm_naive --size 512 ...</mark> 

<mark>Modo "--exec" (para kernels reales de dataset, resueltos desde el catálogo): telemetry_kernel_launcher --exec /ruta/bin/mg.S.x --exec-args "" \ --perf-cpus 2-5 --collector-cpu 0 --consumer-cpu 1 \</mark> --output-dir runs/... --run-id ... 

Internamente, --exec cambia el fork+exec de "lanzar telemetry_kernel_workload" a "lanzar el binario indicado". El resto de la mecánica (cgroup, afinidad, perf por PID/cgroup, RAPL, escritura de samples.csv/metadata.json) es idéntica en ambos modos, porque ya es agnóstica de qué proceso se está midiendo. 

#### **1.3 Lenguaje y ubicación en el repositorio** 

Se usa Python 3.11+ para el orquestador (subprocess, manejo de señales, parsing de sysfs, generación de manifests), dejando el harness C++ intacto como "motor de corrida individual". Ubicación dentro del repositorio: 

|telemetry/<br>...                          (|harness C++, sin cambios)|
|---|---|
|orchestrator/||
|__init__.py||
|cli.py|punto de entrada: orchestrator run campaign.yaml|
|manifest.py|parsing y validacion del manifest de campaña (seccion 2.3)|
|catalog.py|parsing y validacion del catalogo de kernels externos (seccion 4)|
|environment.py<br>|deteccion de tier y capacidades reales del entorno (seccion 3)<br>|
|prefight.py|checks de solo lectura E/I/C/D/G (seccion 5)|
|freqctl.py<br>|lectura/escritura de cpufreq, discretizacion, restauracion (seccion 7)<br>|
|calibration.py|calibracion Roofine (P_pico/BW_pico/I_ridge) (seccion 6.1)|



Página 5 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

<mark>node_profle.py              perfl de hardware + referencias de calibracion P95 (seccion 6.2) runner.py                    ejecuta una corrida individual invocando el launcher C++ (seccion 8) campaign.py                  genera la matriz, aleatoriza, secuencia, reanuda (seccion 9) postprocess.py               samples.csv -> windows.csv (seccion 10) validation.py                criterios de aceptacion/rechazo por corrida (seccion 11) metadata_schema.py           dataclasses/JSON Schema de metadata por corrida y campaña (seccion 12) report.py                    reporte consolidado de campaña (seccion 12)</mark> 

<mark>tests/orchestrator/ test_manifest.py test_catalog.py              con binarios de prueba sinteticos test_environment.py          con sysfs mockeado test_freqctl.py              con mocks de sysfs, sin tocar hardware real test_prefight.py test_calibration.py          con salidas sinteticas de STREAM/ERT test_validation.py test_postprocess.py fxtures/</mark> fake_samples.csv           csv sintetico para probar postprocess sin hardware 

#### **1.4 Flujo de alto nivel** 

orchestrator run campaign.yaml <mark>| v</mark> 

<mark>[manifest.py] parsear y validar manifest (seccion 2.3)</mark> 

<mark>| v</mark> 

<mark>[catalog.py] resolver kernel_ref -> binario, validar checksum (C01, C02) (seccion 4) | v [environment.py] detectar tier y capacidades reales del entorno (seccion 3) | v</mark> 

<mark>[prefight.py] prefight de campaña (E/I/C/D/G) --> abortar si falla algo bloqueante (seccion 5) |</mark> 

<mark>v</mark> 

<mark>[calibration.py + node_profle.py] STREAM + ERT + kernel de referencia repetido --> P_pico, BW_pico, I_ridge, node_profle.json, calibration_references.json (seccion 6) | v</mark> 

<mark>[campaign.py] generar matriz de kernels de dataset + aleatorizar con semilla (seccion 9) |</mark> 

<mark>v</mark> 

<mark>+---------------------------- por cada combinacion -----------------------------+ |  [prefight.py] prefight reducido de corrida                                 | |  [freqctl.py] fjar frecuencia y verifcar aplicada                           | |  [runner.py] invocar telemetry_kernel_launcher --exec <binario del catalogo>  | |  [validation.py] aplicar criterios de aceptacion/rechazo (seccion 11)         | +---------------------------------------------------------------------------------+</mark> 

Página 6 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

<mark>|</mark> 

<mark>v</mark> 

<mark>[freqctl.py] restaurar governor/frecuencia original</mark> 

<mark>| v</mark> 

<mark>[postprocess.py] samples.csv -> windows.csv, con I, phase_label_train y features relativas por ventana (seccion 10) | v</mark> [report.py] reporte consolidado de campaña (seccion 12) 

## **2. Catálogo de kernels y manifest de campaña** 

#### **2.1 Catálogo declarativo (kernels/catalog.yaml)** 

Vive separado del manifest de campaña para poder cambiar de suite sin tocar el resto del sistema. 

# kernels/catalog.yaml <mark>kernels: - id: stream_ofcial suite: STREAM role: calibration                  # nunca entra a la matriz de dataset exec_path: bin/stream_c.exe reports_bandwidth_stdout: true binary_checksum: "sha256:...(se completa al compilar en el nodo objetivo)"</mark> 

<mark>- id: ert_probe suite: ERT role: calibration exec_path: bin/ert_probe.x reports_fops_stdout: true binary_checksum: "sha256:..."</mark> 

<mark>- id: npb_ep suite: NPB-OMP role: dataset exec_path: bin/ep.S.x phase_label_hint: compute_bound    # prior de literatura, NO la etiqueta de entrenamiento size_variant: S expected_runtime_seconds: 4 warmup_seconds: 1.0 success_check: {type: exit_code} binary_checksum: "sha256:..."</mark> 

<mark>- id: npb_mg suite: NPB-OMP role: dataset exec_path: bin/mg.S.x phase_label_hint: memory_bound</mark> 

Página 7 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

<mark>size_variant: S expected_runtime_seconds: 6 warmup_seconds: 1.0 success_check: {type: stdout_regex, pattern: "VERIFICATION SUCCESSFUL"} binary_checksum: "sha256:..."</mark> 

# ... npb_cg, npb_is, npb_ft, npb_lu (mismo esquema) 

#### **2.2 Manifest de campaña** 

# campaign.yaml <mark>campaign_id: pilot_2026_08_local environment_tier: local seed: 20260803</mark> 

<mark>output_dir: runs/pilot_2026_08_local overwrite: false</mark> 

<mark>catalog_path: kernels/catalog.yaml</mark> 

<mark>calibration: - kernel_ref: stream_ofcial - kernel_ref: ert_probe</mark> 

<mark>kernels:                                # kernels de dataset, referenciados por id del catalogo - kernel_ref: npb_ep - kernel_ref: npb_mg</mark> 

<mark>frequency_levels: - {id: F0, mode: fxed, fraction: 1.00} - {id: F1, mode: fxed, fraction: 0.75} - {id: F2, mode: fxed, fraction: 0.50} - {id: F3, mode: fxed, fraction: 0.25} - {id: F4, mode: fxed, fraction: 0.00} - {id: REF, mode: native_governor}</mark> 

<mark>repetitions_per_combination: 10 target_windows_per_repetition: 50 interval_ns: 1000000 running_ratio_min: 0.90</mark> 

<mark>cores: delegated_cpus: "2-5" collector_cpu: 0 consumer_cpu: 1 numa_node_pin: 0</mark> 

<mark>cgroup_path: null perf_enabled: true rapl:</mark> 

Página 8 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

<mark>enabled: true domains: [package] gpu: enabled: false timeouts_seconds: ready: 15 run: 300</mark> shutdown: 10 

#### **2.3 Reglas de validación de manifest.py** 

- Todo kernel_ref referenciado, tanto en calibration como en kernels, debe existir en catalog_path; si no, rechazar el manifest antes de tocar el nodo. 

- La sección calibration debe contener al menos un kernel role=calibration con reports_bandwidth_stdout y al menos uno con reports_flops_stdout; si falta alguno, no se puede calcular I_ridge. 

- Ningún kernel con role=calibration puede aparecer en la sección kernels (dataset), ni viceversa; es un error de manifest, no una advertencia. 

- Rechazar el manifest si environment_tier: hpc_sc3 y no existe cgroup_path (la delegación de cgroup es obligatoria en ese tier, no en local/cloud_own). 

- Rechazar si repetitions_per_combination < 3 (no tiene sentido estadístico por debajo de eso, ni siquiera para piloto). 

- Calcular y loguear el tamaño total de la matriz (len(kernels) × len(frequency_levels) × repetitions_per_combination, doble por baseline) antes de continuar. 

- Si output_dir ya existe y overwrite: false, abortar inmediatamente (factor_id I07). 

- Validar que seed esté presente y sea un entero; si falta, el orquestador no debe generar uno al azar en tiempo de ejecución, porque eso rompería la reproducibilidad del orden de la campaña. 

- Validar que cores.delegated_cpus, cores.collector_cpu y cores.consumer_cpu no se solapen entre sí. 

## **3. Módulo de entorno y capacidades (environment.py)** 

Responsable de convertir "en qué tier estoy" en "qué puedo realmente controlar". Se ejecuta antes del preflight y su salida condiciona qué checks de preflight son bloqueantes. 

@dataclass <mark>class EnvironmentProfle: tier: str                       # local | cloud_own | hpc_sc3 rapl_capable: bool rapl_domains_available: list[str] freq_control_capable: bool scaling_driver: str available_frequencies_khz: list[int] numa_nodes: int</mark> 

Página 9 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

<mark>smt_siblings: dict[int, list[int]] gpu_present: bool</mark> 

<mark>gpu_exclusive_hint: bool        # heuristica; en local suele ser True, en hpc_sc3 nunca asumir True</mark> 

<mark>def detect_environment(delegated_cpus: str) -> EnvironmentProfle: """</mark> 

<mark>Lee /sys/devices/system/cpu/cpu*/cpufreq/scaling_driver, scaling_available_frequencies, /sys/class/powercap/intel-rapl/, /sys/devices/system/node/ y /sys/devices/system/cpu/cpu*/topology/thread_siblings_list. No escribe nada: es de solo lectura.</mark> """ 

#### **3.1 Reglas duras** 

- Si scaling_driver no corresponde a un driver real de hardware conocido (intel_pstate, acpi-cpufreq, amd-pstate) o scaling_available_frequencies tiene un único valor, freq_control_capable = False. 

- Si /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj no existe o no cambia tras una espera corta de control (leer dos veces con 100 ms de diferencia bajo carga sintética mínima), rapl_capable = False. 

- El manifest no puede forzar rapl.enabled: true si environment.py determina rapl_capable: false; el orquestador debe sobrescribir el flag y dejarlo registrado en la metadata de campaña, no fallar en silencio. 

- environment.py es la única fuente autorizada de la advertencia RAPL/cpufreq en VM descrita en la sección 0.3: cualquier otro módulo que necesite saber si puede tocar frecuencia o leer energía debe preguntarle a environment.py, no repetir la detección por su cuenta. 

## **4. Módulo de catálogo (catalog.py)** 

@dataclass <mark>class KernelEntry: id: str suite: str role: str                     # "dataset" | "calibration" exec_path: str binary_checksum: str phase_label_hint: str | None  # solo si role == "dataset" size_variant: str | None expected_runtime_seconds: int | None warmup_seconds: foat | None success_check: dict | None reports_bandwidth_stdout: bool = False reports_fops_stdout: bool = False</mark> 

<mark>def load_catalog(catalog_path: str) -> dict[str, KernelEntry]: """Parsea kernels/catalog.yaml a un diccionario id -> KernelEntry."""</mark> 

<mark>def verify_binary(entry: KernelEntry) -> CheckResult:</mark> 

Página 10 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

<mark>"""</mark> 

<mark>C01: os.path.isfle(entry.exec_path) y os.access(entry.exec_path, os.X_OK) C02: sha256(entry.exec_path) == entry.binary_checksum Retorna CheckResult con factor_id "C01" o "C02" segun cual falle. """</mark> 

<mark>def resolve_exec_command(entry: KernelEntry) -> list[str]:</mark> """Traduce un KernelEntry a la lista de argumentos --exec/--exec-args del launcher.""" 

**_Nota:_** _verify_binary() se ejecuta tanto en el preflight de campaña (todos los kernels referenciados) como, de forma reducida, inmediatamente antes de cada corrida individual (solo el kernel de esa combinación), para detectar si algo cambió el binario a mitad de campaña._ 

## **5. Preflight (preflight.py)** 

Implementa, con IDs explícitos, las verificaciones de solo lectura. Cada función retorna un objeto uniforme: 

@dataclass <mark>class CheckResult: factor_id: str          # "E01", "I05", "C02", "D03", "G02", etc. name: str passed: bool blocking: bool observed: dict          # valores leidos, para la metadata</mark> message: str 

#### **5.1 Preflight de campaña (una sola vez, antes de generar la matriz)** 

|**factor_id**|**Check**|**Bloqueante**|**Momento**|
|---|---|---|---|
|E01|Turbo/HWP: leer estado y dejarlo fjo y<br>registrado para toda la campaña.|Sí|Campaña|
|E04|NUMA: delegated_cpus pertenece a un<br>único nodo NUMA.|Sí|Campaña|
|E05|SMT: identfcar siblings de delegated_cpus<br>y decidir polítca (un hilo por core fsico vs.<br>todos) según manifest.|Sí|Campaña|
|I05|RAPL: max_energy_range_uj disponible o no<br>(solo si rapl.enabled).|No|Campaña|
|I07|output_dir no existe (o overwrite: true).|Sí|Campaña|
|G01/<br>G02/G03|Si gpu.enabled: sin procesos CUDA actvos,<br>persistence mode leído, MIG leído.|Sí si gpu.enabled|Campaña|
|C01|Binario del kernel existe y es ejecutable en<br>el nodo objetvo.|Sí|Campaña + por<br>corrida|
|C02|Checksum del binario coincide con el|Sí|Campaña + por|



Página 11 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

|**factor_id**|**Check**|**Bloqueante**|**Momento**|
|---|---|---|---|
||registrado en el catálogo.||corrida|
|C03|success_check del kernel está<br>correctamente confgurado (regex compila,<br>tpo válido).|Sí|Campaña|
|D01|Toolchain necesario disponible (gfortran<br>para NPB, gcc para STREAM/ERT), si se va a<br>recompilar en este prefight.|Solo si se<br>recompila|Campaña|
|D02|Calibración STREAM/ERT ejecutada sin error<br>y con salida parseable (BW/FLOPs<br>reportados).|Sí|Campaña, antes de<br>generar la matriz|
|D03|BW_pico y P_pico dentro de un rango<br>plausible frente a la fcha técnica declarada<br>del hardware.|Sí|Campaña,<br>inmediatamente<br>después de D02|
|D04|node_profle.json y<br>calibraton_references.json generados;<br>cv_pct de las referencias P95 dentro del<br>umbral (≤5%, confgurable).|No (solo<br>advertencia)|Campaña, junto con<br>D02/D03|



**_Nota:_** _D03 no pretende ser una validación exacta: basta un rango amplio (por ejemplo, ±40% de lo que indica la ficha técnica del fabricante) para atrapar errores groseros de calibración, como correr STREAM con un solo hilo cuando se delegaron 4 cores, o medir con el governor equivocado._ 

#### **5.2 Preflight reducido (antes de cada corrida individual)** 

|**factor_id**|**Check**|**Bloqueante**|
|---|---|---|
|E02|Temperatura de paquete dentro de rango normal (si hay sensor).|Sí si hay sensor<br>disponible|
|E06|Sin procesos ajenos con afnidad a delegated_cpus.|Sí|
|E07|scaling_governor actual coincide con el esperado antes de fjar el<br>nuevo nivel.|Sí|
|E08|Carga externa del nodo (load average normalizado) bajo un<br>umbral confgurable.|Sí|
|I07|run_id de esta corrida no existe ya en output_dir.|Sí|
|C01/C02|Binario del kernel de esta combinación sigue existendo y con el<br>checksum esperado.|Sí|
|C03|success_check de este kernel específco está correctamente<br>confgurado antes de lanzar.|Sí|



Página 12 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

**_Nota:_** _El preflight reducido corre también aunque environment_tier: local. La disciplina de verificación no depende de si el nodo es compartido; depende de si se quiere confiar en el dato._ 

## **6. Calibración: Roofline y perfil de nodo (calibration.py, node_profile.py)** 

Se ejecuta una sola vez por campaña, después del preflight y antes de generar la matriz de kernels de dataset. Produce tres artefactos: la calibración Roofline usada para etiquetar el entrenamiento, el perfil de hardware del nodo, y las referencias de estabilidad P95 — estas dos últimas pensadas para la estrategia multinodo (sección 0.4 y 16). 

#### **6.1 Calibración Roofline (P_pico, BW_pico, I_ridge)** 

@dataclass <mark>class RoofineCalibration: campaign_id: str timestamp: str delegated_cpus: str bw_pico_bytes_per_s: foat p_pico_fops_per_s: foat i_ridge_fops_per_byte: foat stream_raw_output: str ert_raw_output: str plausibility_check_passed: bool</mark> 

<mark>def run_calibration(manifest: Manifest, catalog: dict[str, KernelEntry]) -> RoofineCalibration: """</mark> 

<mark>1. Localizar en manifest.calibration los kernel_ref con reports_bandwidth_stdout=True (STREAM) y reports_fops_stdout=True (ERT).</mark> 

<mark>2. Ejecutar cada uno con runner.run_single(), igual que una corrida de dataset,</mark> 

<mark>pero con role="calibration": no pasa por freqctl (corre a la frecuencia F0, el limite superior, para medir el pico real) y no entra a windows.csv.</mark> 

<mark>3. Parsear la salida estandar de cada binario (regex especifco por suite, declarado en catalog.py) para extraer BW_pico y P_pico.</mark> 

<mark>4. Calcular i_ridge_fops_per_byte = p_pico_fops_per_s / bw_pico_bytes_per_s.</mark> 

<mark>5. Ejecutar prefight D03 contra la fcha tecnica declarada del hardware (manifest o environment.py); marcar plausibility_check_passed.</mark> 

<mark>6. Serializar a roofine_calibration.json dentro de output_dir de la campaña. """</mark> 

<mark>def load_calibration(output_dir: str) -> RoofineCalibration:</mark> """Usado por postprocess.py para leer i_ridge_flops_per_byte al clasificar ventanas.""" 

**Atención:** La calibración corre exclusivamente a F0 (frecuencia máxima), porque P_pico y BW_pico son, por definición, los límites superiores del hardware. Ejecutarla a una frecuencia reducida subestimaría ambos picos y distorsionaría I_ridge para toda la campaña. 

El detalle de cómo se prorratea el FLOP count reportado por un kernel de dataset a nivel de ventana, y cómo 

se combina con los bytes movidos medidos por perf, se documenta junto con el resto del post-procesamiento en la sección 10. 

Página 13 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

#### **6.2 node_profile.py: perfil de hardware y referencias de calibración estables** 

Módulo hermano de calibration.py que corre en el mismo momento de la campaña y produce dos artefactos adicionales, necesarios para las Propuestas A y B descritas en la sección 0.4, sin los cuales esas dos alternativas quedarían cerradas para siempre sobre los datos ya recolectados. 

@dataclass <mark>class NodeProfle: node_id: str hostname: str cpu_model: str sockets: int cores_total: int threads_per_core: int numa_nodes: int cache_l1_kb: int cache_l2_kb: int cache_llc_kb: int cache_llc_shared: bool freq_min_khz: int freq_max_khz: int scaling_driver: str perf_events_supported: list[str]     # subconjunto realmente disponible en esta PMU rapl_domains_available: list[str]</mark> 

<mark>def build_node_profle(env: EnvironmentProfle, delegated_cpus: str) -> NodeProfle: """</mark> 

<mark>Lee /proc/cpuinfo, /sys/devices/system/cpu/*/cache/index*/, /sys/devices/system/node/, y el resultado de environment.detect_environment() ya calculado. No ejecuta nada nuevo sobre el hardware: reorganiza informacion de solo lectura ya disponible en un artefacto versionado y con un node_id estable. """</mark> 

<mark>@dataclass class CalibrationReferences: node_id: str ipc_p95: foat ips_p95: foat mpki_p95: foat miss_rate_p95: foat repetitions: int cv_pct: foat          # coefciente de variacion entre repeticiones accepted: bool          # True si cv_pct <= 5.0 (umbral confgurable)</mark> 

<mark>def build_calibration_references(calibration_runs: list[RunResult]) -> CalibrationReferences: """</mark> 

<mark>Corre un microbenchmark de referencia (puede ser el mismo npb_ep para computo y stream_ofcial para memoria) repetido >= 5 veces, calcula P95 de IPC/IPS/MPKI/ MissRate entre esas repeticiones y su coefciente de variacion. Si cv_pct > 5.0, accepted=False y el check D04 de prefight (seccion 5.1)</mark> 

Página 14 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

<mark>queda marcado como advertencia hasta investigar la inestabilidad.</mark> """ 

|**Artefacto**|**Sirve a**|**Se guarda en**|
|---|---|---|
|node_profle.json|Propuesta A (variables de hardware<br>explícitas)|output_dir de la campaña,<br>referenciado desde cada corrida<br>vía node_profle_ref|
|calibraton_references.json<br>(IPC/IPS/MPKI/MissRate P95 +<br>CV%)|Propuesta B (features<br>relatvas/normalizadas)|output_dir de la campaña,<br>referenciado desde cada corrida<br>vía calibraton_ref|
|roofine_calibraton.json (P_pico,<br>BW_pico, I_ridge)|Etquetado de entrenamiento<br>(sección 10), usado en las tres<br>propuestas por igual|Defnido en 6.1 de esta sección|



**_Nota:_** _Las tres calibraciones (Roofline, node_profile, calibration_references) se ejecutan en la misma fase de campaña, sobre las mismas corridas de STREAM/ERT, más algunas repeticiones adicionales del kernel de referencia para poder calcular el CV%. No se necesita una fase separada ni tiempo de nodo adicional significativo._ 

## **7. Control de frecuencia (freqctl.py)** 

#### **7.1 Lectura y discretización** 

def read_available_frequencies(cpu: int) -> list[int]: 

<mark>"""Lee scaling_available_frequencies; si no existe, cae a [cpuinfo_min_freq, cpuinfo_max_freq]."""</mark> 

<mark>def resolve_level_to_khz(level: dict, available_khz: list[int]) -> int: """</mark> 

<mark>level = {"mode": "fxed", "fraction": 0.75} Calcula f_min + fraction * (f_max - f_min) y ajusta al valor discreto</mark> 

<mark>mas cercano dentro de available_khz. Devuelve tambien el valor solicitado</mark> 

<mark>y el valor aplicado para que ambos queden en metadata (nunca solo uno).</mark> """ 

#### **7.2 Aplicación y verificación** 

def apply_frequency(cpus: list[int], level: dict, available_khz: list[int]) -> AppliedFrequency: <mark>"""</mark> 

<mark>1. set governor 'userspace' en cada cpu de cpus.</mark> 

<mark>2. escribir scaling_setspeed con el valor discretizado.</mark> 

<mark>3. releer scaling_cur_freq en cada cpu y comparar contra el valor aplicado.</mark> 

<mark>4. si difere mas alla de una tolerancia confgurable -> CheckResult(passed=False, factor_id="E01").</mark> """ 

#### **7.3 Restauración de emergencia — no negociable** 

Página 15 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

Este es el punto más sensible de todo el orquestador: debe restaurar el estado original incluso si el proceso del orquestador muere a mitad de una corrida. 

_original_state: dict[int, tuple[str, int]] = {}  # cpu -> (governor, freq_khz) 

<mark>def snapshot_original_state(cpus: list[int]) -> None:</mark> """Guarda governor y frecuencia actuales antes de tocar nada. Se llama una sola vez, al inicio de campaña.""" 

<mark>def restore_original_state() -> None: """Idempotente: puede llamarse mas de una vez sin error. Debe registrar si la restauracion realmente se verifco por lectura posterior, no solo si el comando de escritura no lanzo excepcion.""" def install_emergency_handlers() -> None: """ Registrar restore_original_state() en: - atexit.register(...) - signal.signal(signal.SIGINT, ...) - signal.signal(signal.SIGTERM, ...) para cubrir Ctrl-C, kill, y salida normal del proceso.</mark> """ 

**Atención:** Prueba obligatoria antes de usar este módulo contra hardware real (ver sección 13): simular una interrupción a mitad de una corrida y confirmar, por lectura, que el governor/frecuencia vuelve exactamente al estado original. 

Durante la fase de calibración (sección 6), freqctl.py se invoca para fijar F0 antes de correr STREAM/ERT y se restaura junto con el resto de cores delegados al cierre de la campaña, usando exactamente esta misma rutina de restauración. 

#### **7.4 Cuándo NO tocar frecuencia** 

Si environment.freq_control_capable == False (por ejemplo, en una VM cloud_own sin passthrough de MSR, ver 0.3), freqctl.py debe: 

- omitir por completo la escritura de scaling_governor/scaling_setspeed, 

- registrar en la metadata de cada corrida frequency_control: "unavailable" en vez de fingir que se aplicó un nivel, 

- seguir permitiendo la ejecución de la corrida (para medir overhead del harness y probar el pipeline), pero el orquestador debe etiquetar esas corridas como not_eligible_for_training_dataset: true. 

## **8. Ejecución de una corrida individual (runner.py)** 

def run_single(combination: Combination, env: EnvironmentProfile, manifest: Manifest, <mark>catalog: dict[str, KernelEntry]) -> RunResult: """</mark> 

> <mark>1. prefight reducido (seccion 5.2), incluyendo C01/C02 del kernel de esta combinacion.</mark> 

Página 16 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

<mark>2. si env.freq_control_capable: freqctl.apply_frequency(...)</mark> 

<mark>3. entry = catalog[combination.kernel_ref] cmd = [</mark> 

<mark>"telemetry_kernel_launcher",</mark> 

<mark>"--exec", entry.exec_path,</mark> 

<mark>"--exec-args", "",</mark> 

<mark>"--perf-cpus", manifest.cores.delegated_cpus,</mark> 

<mark>"--collector-cpu", str(manifest.cores.collector_cpu),</mark> 

<mark>"--consumer-cpu", str(manifest.cores.consumer_cpu),</mark> 

<mark>"--interval-ns", str(manifest.interval_ns),</mark> 

<mark>"--output-dir", manifest.output_dir,</mark> 

<mark>"--run-id", combination.run_id,</mark> 

<mark>]</mark> 

<mark>4. timeout = entry.expected_runtime_seconds * SAFETY_MARGIN (p. ej. 3x)</mark> 

<mark>5. ejecutar con subprocess.run(cmd, timeout=timeout)</mark> 

<mark>6. aplicar entry.success_check contra exit_code/stdout (C03)</mark> 

<mark>7. verifcar sin procesos hijos vivos (psutil o /proc); fusionar metadata del orquestador (checksum del binario, kernel_ref, roofine_calibration_ref, node_profle_ref, calibration_ref, environment_tier) con la que produce el launcher.</mark> 

""" 

Combination.run_id se construye de forma determinista a partir de (kernel_ref, frequency_level, repetition_index): 

run_id = f"{campaign_id}__{kernel_ref}__{freq_level.id}__rep{repetition_index:02d}" 

Esto hace que reanudar una campaña interrumpida sea trivial: si output_dir/<run_id>/metadata.json ya existe y quedó marcado como accepted, se salta esa combinación al reanudar (ver sección 9). 

#### **8.1 Warmup por tiempo de pared (en vez de iteraciones internas)** 

Como el binario externo no coopera con una señal de ready, el warmup se resuelve enteramente en el postprocesamiento, usando warmup_seconds del catálogo: 

def mark_warmup(windows_df, entry: KernelEntry) -> pd.DataFrame: <mark>"""</mark> 

<mark>Para cada ventana con t_start_ns - t_start_ns_de_la_repeticion < entry.warmup_seconds * 1e9: quality_status = "warmup_excluded"</mark> 

<mark>El resto continua el pipeline normal de calculo de deltas.</mark> """ 

## **9. Generación y secuenciación de la campaña (campaign.py)** 

def build_matrix(manifest: Manifest) -> list[Combination]: 

<mark>"""Producto cartesiano kernels(role=dataset) x frequency_levels x range(repetitions_per_combination)."""</mark> 

<mark>def randomize(matrix: list[Combination], seed: int) -> list[Combination]:</mark> 

<mark>"""</mark> 

Página 17 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

<mark>random.Random(seed).shufe(matrix) -- nunca usar random global sin semilla.</mark> 

<mark>La semilla y el orden resultante (lista de run_id en el orden ejecutado)</mark> 

<mark>se guardan en la metadata de campaña, no solo en el log.</mark> 

<mark>"""</mark> 

<mark>def run_campaign(manifest: Manifest, catalog: dict[str, KernelEntry]) -> CampaignReport: """</mark> 

<mark>1. prefight.py: prefight de campaña (seccion 5.1).</mark> 

<mark>2. calibration.py + node_profle.py: fase de calibracion (seccion 6).</mark> 

<mark>3. matrix = randomize(build_matrix(manifest), manifest.seed)</mark> 

<mark>4. para cada combination en matrix:</mark> 

<mark>si output_dir/<run_id>/metadata.json ya existe y accepted=True -> saltar (reanudacion) freqctl.apply_frequency(...)</mark> 

<mark>result = runner.run_single(combination, env, manifest, catalog)</mark> 

<mark>verdict = validation.validate_run(result, manifest)</mark> 

<mark>registrar verdict en el reporte de campaña</mark> 

<mark>5. freqctl.restore_original_state()</mark> 

<mark>6. postprocess.compute_windows(...) sobre todas las corridas aceptadas</mark> 

<mark>7. report.build_campaign_report(...)</mark> 

""" 

#### **9.1 Reglas de secuenciación** 

- Orden aleatorio obligatorio (factor_id M01): nunca ejecutar en bloques por kernel o por frecuencia. Se recorre randomize(matrix, seed) en el orden generado, no agrupado. 

- El baseline (sin telemetría) se genera como una combinación paralela con el mismo run_id más el sufijo __baseline, y se ejecuta inmediatamente antes o después de su contraparte telemetry, tal como ya hace el launcher internamente. El orquestador no necesita aleatorizar baseline y telemetry por separado; los trata como un par atómico. 

- La reanudación se basa exclusivamente en la existencia de metadata.json con accepted=True para ese run_id; una corrida rechazada NO se salta automáticamente al reanudar, para permitir reintentarla. 

- El orquestador debe verificar, al final de cada corrida, que no quedaron procesos hijos vivos antes de iniciar la siguiente combinación. 

## **10. Post-procesamiento (postprocess.py)** 

Entrada: samples.csv de todas las corridas aceptadas de la campaña. Salida: windows.csv con una fila por ventana válida y las banderas de calidad. 

REQUIRED_OUTPUT_COLUMNS = [ 

<mark>"run_id", "repetition", "kernel_ref", "node_id", "phase_label_hint", "phase_label_train",</mark> 

<mark>"freq_level_id", "freq_khz_requested", "freq_khz_applied", "freq_khz_observed",</mark> 

<mark>"window_index", "t_start_ns", "t_end_ns", "delta_t_ns",</mark> 

<mark>"delta_instructions", "delta_cycles", "delta_cache_references", "delta_cache_misses",</mark> 

- <mark>"ipc", "llc_miss_rate", "mpki", "ips",</mark> 

<mark>"ipc_relative", "mpki_relative", "miss_rate_relative",   # ver 6.2</mark> 

- <mark>"delta_running_ns", "delta_enabled_ns", "running_ratio",</mark> 

Página 18 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

<mark>"pkg_delta_uj", "dram_delta_uj", "power_w", "energy_valid",</mark> 

<mark>"fops_window_estimate", "bytes_moved_window", "operational_intensity", "i_ridge_used", "roofine_calibration_ref", "node_profle_ref", "calibration_ref", "binary_checksum",</mark> 

<mark>"quality_status",       # "ok" | "warmup_excluded" | "no_freq_reading" | "pmu_degraded" # | "energy_invalid" | "frst_sample_no_delta" | "intensity_undefned" ]</mark> 

<mark>def compute_relative_features(window_row, refs: CalibrationReferences) -> dict: """</mark> 

<mark>Calcula, ademas de las features absolutas: ipc_relative = window_row.ipc / refs.ipc_p95 mpki_relative = window_row.mpki / refs.mpki_p95 miss_rate_relative = window_row.llc_miss_rate / refs.miss_rate_p95 No se recorta el ratio a [0,1]: puede superar 1 legitimamente si la ventana supera la referencia P95 de calibracion, y ese exceso es informacion valida, no un error a corregir.</mark> 

<mark>Estas columnas quedan pobladas siempre, se use o no la Propuesta B mas adelante. """</mark> 

<mark>def compute_operational_intensity(window_row, run_fops_total, run_duration_ns) -> foat: """</mark> 

<mark>fops_window_estimate = run_fops_total * (window_row.delta_t_ns / run_duration_ns)</mark> 

<mark>-- prorrateo simple por duracion; asume FLOPs distribuidos uniformemente en el tiempo</mark> 

- <mark>-- dentro de una misma repeticion (valido para NPB/STREAM/ERT, que no tienen fases</mark> 

<mark>-- internas conocidas de intensidad variable).</mark> 

<mark>bytes_moved_window = window_row.delta_cache_misses * LLC_LINE_SIZE_BYTES return fops_window_estimate / bytes_moved_window if bytes_moved_window > 0 else foat("nan")</mark> 

<mark>def compute_windows(samples_df, run_metadata, calibration: RoofineCalibration, node_refs: CalibrationReferences) -> pd.DataFrame:</mark> 

<mark>"""</mark> 

- <mark>Por cada run_id + repetition, ordenado por timestamp_ns: - fla 0 -> quality_status = "frst_sample_no_delta", sin deltas</mark> 

- <mark>fla i -> delta_x = x[i] - x[i-1]; si delta_x < 0 sin wrap conocido -> invalidar</mark> 

- <mark>ipc = delta_instructions / delta_cycles (NaN si delta_cycles == 0, nunca dividir por cero silenciosamente)</mark> 

- <mark>running_ratio = delta_running_ns / delta_enabled_ns</mark> 

- <mark>power_w = pkg_delta_uj * 1e-6 / (delta_t_ns * 1e-9)</mark> 

- <mark>aplicar correccion de wrap con max_energy_range_uj si run_metadata lo trae</mark> 

- <mark>marcar ventanas dentro de warmup (mark_warmup, seccion 8.1) como "warmup_excluded"</mark> 

- <mark>operational_intensity = compute_operational_intensity(...); si es NaN, quality_status = "intensity_undefned" y la ventana NO se usa para entrenar</mark> 

- <mark>phase_label_train = "memory_bound" si operational_intensity < calibration.i_ridge_fops_per_byte else "compute_bound"</mark> 

- <mark>phase_label_hint se copia tal cual del catalogo, solo para auditoria</mark> 

- <mark>llama a compute_relative_features(window_row, node_refs) y agrega esas columnas</mark> 

- <mark>agrega node_id y las referencias (node_profle_ref, calibration_ref) a cada fla</mark> 

""" 

Página 19 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

**Atención:** phase_label_train nunca se calcula por inferencia estadística ni se copia de phase_label_hint. Siempre es el resultado de comparar operational_intensity contra i_ridge_used de la calibración de esa sesión. 

## **11. Validación y criterios de rechazo (validation.py)** 

Cada corrida se marca automáticamente como aceptada o rechazada, sin intervención manual, aplicando la siguiente tabla completa de criterios: 

|**factor_id**|**Condición de rechazo**|**Nivel**|**Verifcación**|
|---|---|---|---|
|I01|Frecuencia observada ausente en alguna<br>ventana.|ventana|quality_status =<br>"no_freq_reading"|
|I02|running_rato < running_rato_min.|ventana|delta_running_ns /<br>delta_enabled_ns|
|I03|Jiter de muestreo: intervalo real muy<br>distnto del nominal.|ventana|tmestamp[t] -<br>tmestamp[t-1] vs.<br>interval_ns, con<br>tolerancia|
|I04|push_retries > 0 reportado en<br>metadata.json.|corrida<br>completa|rechazo inmediato,<br>sin reparación<br>posible a nivel de<br>ventana|
|I05|Delta energétco negatvo sin corrección de<br>wrap disponible.|ventana|energy_valid = false<br>si no hay<br>max_energy_range_<br>uj|
|I06|Lectura RAPL con bandera de invalidez<br>propagada desde el reader.|ventana|requiere que<br>RaplReader nunca<br>devuelva 0 silencioso<br>ante error|
|I07|run_id duplicado.|corrida<br>completa|prefight, no<br>postprocess|
|E06|Proceso ajeno detectado en el cgroup/cores<br>delegados durante la corrida.|corrida<br>completa|prefight reducido|
|E07|Governor drif: el governor efectvo no<br>coincide con el esperado.|corrida<br>completa|lectura pre/post<br>corrida|
|E08|Carga externa del nodo por encima del<br>umbral durante la ventana medida.|corrida<br>completa|monitor de carga|
|M02|Ventanas dentro del rango de<br>warmup_seconds del catálogo.|ventana|quality_status =<br>"warmup_excluded",<br>nunca descartadas<br>en silencio|



Página 20 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

|**factor_id**|**Condición de rechazo**|**Nivel**|**Verifcación**|
|---|---|---|---|
|C02|Checksum del binario ejecutado no coincide<br>con el catálogo.|corrida<br>completa|prefight/runner|
|C03|success_check no se cumple (exit code o<br>patrón de verifcación en stdout).|corrida<br>completa|runner|
|D03|Calibración de la sesión marcada no<br>plausible.|campaña<br>completa|bloquea toda la<br>matriz, no solo una<br>corrida|
|—|operatonal_intensity indefnida<br>(bytes_moved_window == 0) en una<br>ventana.|ventana|quality_status =<br>"intensity_undefned<br>"|



def validate_run(run_result: RunResult, manifest: Manifest) -> Verdict: <mark>""" Aplica, en orden, I01-I07, E06-E08, M02, C02, C03, D03, y cuenta samples_collected == 0 como I04-equivalente (bufer/backend caido). Retorna Verdict(accepted=bool, factor_id=str|None, message=str). Nunca borra la corrida rechazada: la deja en output_dir marcada como rejected/<factor_id>, conservando el crudo para auditoria posterior.</mark> """ 

**_Nota:_** _El umbral exacto de tolerancia de frecuencia (E07) y del ratio running/enabled (I02) se fija empíricamente durante la campaña piloto (sección 13.2) y queda documentado como parámetro de configuración, no como número embebido en el código._ 

## **12. Metadata y reporte de campaña (metadata_schema.py, report.py)** 

#### **12.1 Metadata por corrida** 

{ <mark>"run_id": "pilot_2026_08_local__npb_mg__F2__rep03", "campaign_id": "pilot_2026_08_local", "kernel_ref": "npb_mg", "node_id": "uis-sc3-node07", "node_profle_ref": "pilot_2026_08_local/node_profle.json", "calibration_ref": "pilot_2026_08_local/calibration_references.json", "binary_checksum": "sha256:...", "roofine_calibration_ref": "pilot_2026_08_local/roofine_calibration.json", "phase_label_hint": "memory_bound", "environment_tier": "local", "rapl_capable": true, "freq_control_capable": true, "frequency_requested_khz": 2400000, "frequency_applied_khz": 2400000, "frequency_observed_khz": [2398000, 2401000, 2400000],</mark> 

Página 21 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

<mark>"governor_before": "schedutil", "governor_during": "userspace", "governor_restored_verifed": true, "prefight_reduced": [{"factor_id": "E07", "passed": true}], "not_eligible_for_training_dataset": false, "accepted": true, "rejection_factor_id": null</mark> } 

#### **12.2 Reporte consolidado de campaña** 

report.py genera, al final de cada campaña, una tabla resumen a partir de los Verdict acumulados: 

|**factor_id**|**Corridas afectadas**|**% sobre el total**|**Ejemplo de run_id**|
|---|---|---|---|
|I04|3|1.2%|...|
|E08|1|0.4%|...|
|— (aceptadas)|236|98.3%|—|



Además de esta tabla, el reporte agrega una fila resumen con i_ridge_flops_per_byte de la sesión, el porcentaje de ventanas con quality_status = "intensity_undefined", y el cv_pct de las referencias de calibración — si ese valor supera el umbral aceptado, el reporte lo marca como advertencia aunque no bloquee la campaña de por sí, ya que solo afecta a las Propuestas A/B, no a la C (sección 16). Este reporte es el artefacto que decide si la campaña es utilizable o si hay que repetir combinaciones específicas antes de pasar a la Fase 2. 

## **13. Estrategia de pruebas en el entorno actual** 

Dado que ahora mismo solo hay acceso a PCs locales de investigadores y a un servidor cloud propio, la implementación se valida en el siguiente orden, sin tocar el SC3 en ningún punto de esta lista: pruebas unitarias sin hardware real, pruebas en un PC local bare-metal, y pruebas en el servidor cloud propio. 

#### **13.1 Pruebas unitarias (sin hardware real)** 

- test_manifest.py: manifest válido e inválido (repetitions<3, output_dir ya existente, kernel_ref inexistente en el catálogo). 

- test_environment.py: sysfs mockeado — driver real, driver ficticio (una sola frecuencia), RAPL ausente. 

- test_freqctl.py: mockear open()/read()/write() de rutas sysfs; probar discretización de fracciones a valores disponibles, y que restore_original_state() es idempotente. 

- test_catalog.py: verificar que verify_binary() detecta correctamente un binario ausente (C01) y un checksum alterado (C02), usando binarios de prueba sintéticos, no NPB real. 

Página 22 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

- test_calibration.py: alimentar calibration.py con una salida de stdout sintética de STREAM/ERT (fixture) y verificar que I_ridge se calcula correctamente y que D03 rechaza una calibración con valores absurdos (por ejemplo, BW_pico de 1 byte/s). 

- test_validation.py: simular metadata.json con push_retries > 0 y confirmar rechazo con factor_id="I04". 

- test_postprocess.py: usar fixtures/fake_samples.csv sintético con casos a propósito — primera fila sin delta, un delta negativo de energía sin wrap, un running_ratio bajo, un caso con bytes_moved_window == 0 — y confirmar que cada uno cae en el quality_status correcto, nunca en una división por cero silenciosa. 

#### **13.2 Pruebas locales (PC de investigador, bare-metal, con root)** 

- Ejecutar environment.py y confirmar que detecta correctamente freq_control_capable y rapl_capable en ese hardware concreto. 

- Compilar NPB (clases SER/OMP), STREAM y ERT en el PC local; registrar los checksums resultantes en kernels/catalog.yaml. 

- Ejecutar calibration.py de forma aislada (sin matriz de dataset) y verificar que BW_pico/P_pico son razonables frente a la ficha técnica del PC (por ejemplo, contra el datasheet del fabricante del CPU). 

- Ejecutar un kernel NPB de clase S de punta a punta (npb_ep, el más simple) y confirmar que aparece correctamente en windows.csv con operational_intensity y phase_label_train poblados. 

- Campaña piloto mínima: 1–2 kernels × 2 niveles de frecuencia × 3 repeticiones, con environment_tier: local. 

- Prueba de caos obligatoria: lanzar la campaña y enviar SIGINT a mitad de una corrida (sintética primero, luego con un kernel NPB real); verificar por lectura de sysfs que el governor/frecuencia quedó restaurado exactamente al valor previo. 

- Medir el overhead real baseline-vs-telemetry reportado por el launcher y confirmar que es estable entre repeticiones de la misma condición. 

#### **13.3 Pruebas en el servidor cloud propio** 

- Ejecutar primero los tres comandos de diagnóstico de la sección 0.3. Documentar el resultado en un archivo environment_report_cloud_own.json versionado junto al código. 

- Verificar que el toolchain (gfortran, gcc, make) está disponible o se puede instalar sin privilegios especiales adicionales a los ya otorgados; si no, este entorno queda limitado a los kernels sintéticos de desarrollo hasta resolverlo. 

- Si freq_control_capable: false (lo más probable en una VM estándar), correr la misma campaña piloto en modo frequency_control: unavailable, sirviendo exclusivamente para validar el pipeline completo de principio a fin bajo una configuración de cores distinta a la del PC local, medir el overhead del collector en un entorno con características de CPU distintas, y probar la ruta de reanudación de campaña interrumpida (matar el proceso orquestador y volver a lanzarlo, confirmando que no repite run_id ya aceptados). 

Página 23 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

#### **13.4 Checklist antes de pedir acceso formal al SC3** 

|**#**|**Verifcación**|
|---|---|
|1|Suite de tests unitarios en verde.|
|2|Prueba de caos de restauración de frecuencia exitosa en al menos un PC local bare-metal, con<br>un kernel sintétco y con un kernel NPB real.|
|3|Reporte de campaña piloto (sección 12.2) generado correctamente en local y en cloud_own.|
|4|environment_report_cloud_own.json documentado, para saber de antemano qué esperar de<br>RAPL/cpufreq en ese servidor.|
|5|Reanudación de campaña interrumpida probada al menos una vez.|
|6|Código de freqctl.py revisado en la parte de restauración de emergencia: es la única pieza que<br>puede afectar a otros usuarios el día que se ejecute contra el SC3.|
|7|node_profle.json y calibraton_references.json generados y revisados en al menos un nodo<br>local, para confrmar que la capa "sin arrepentmiento" multnodo (sección 0.4) funciona antes<br>de escalar.|



## **14. Roadmap de implementación incremental** 

|**Etapa**|**Entregable**|**Entorno donde se**<br>**valida**|
|---|---|---|
|A|manifest.py, environment.py, prefight.py básico, runner.py en<br>modo --kernel sintétco (sin catálogo aún)|local + cloud_own|
|B|freqctl.py completo con prueba de caos|local (bare-metal)|
|C|catalog.py + compilación y verifcación de NPB/STREAM/ERT en<br>al menos un entorno|local|
|D|calibraton.py + roofine_calibraton.json + prefight D01–D03|local|
|E|runner.py en modo --exec real, postprocess.py con<br>operatonal_intensity y phase_label_train|local, con 1–2 kernels<br>NPB|
|F|campaign.py con matriz completa de kernels de dataset,<br>aleatorización y reporte consolidado|local + cloud_own,<br>campaña piloto de<br>punta a punta|
|G|Extensión a cgroup_path real y verifcación de delegación<br>(hpc_sc3)|únicamente cuando<br>estén los permisos del<br>SC3 confrmados|
|H|Módulo GPU|se pospone<br>explícitamente|
|I|node_profle.py + referencias de calibración P95 (capa "sin<br>arrepentmiento" para A/B, sección 0.4)|local, en paralelo con<br>las etapas C-F; no<br>bloquea nada de lo|



Página 24 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

|**Etapa**|**Entregable**|**Entorno donde se**<br>**valida**|
|---|---|---|
|||anterior|



## **15. Preguntas que quedan abiertas para esta guía en particular** 

- ¿El PC local usado para la prueba de caos y la calibración es el mismo en el que después se ejecutará la campaña piloto completa? Recalibrar en cada máquina distinta es obligatorio si cambian. 

- ¿El servidor cloud propio es bare-metal/dedicado, y tiene toolchain Fortran disponible para compilar NPB? 

- ¿Se fija un único BW_pico/P_pico por campaña, o se recalibra si la campaña se extiende por varias sesiones en días distintos (posible deriva térmica ambiental de largo plazo)? Se recomienda recalibrar por sesión, no reutilizar una calibración antigua entre sesiones separadas por más de un día. 

## **16. Estrategia multinodo: comparación completa y qué módulo sirve a cada alternativa** 

Esta sección retoma en detalle las tres alternativas presentadas en 0.4, con el cuadro comparativo completo (pros, contras y recomendación por criterio), y traduce esa comparación a decisiones de implementación: qué módulo de esta guía sirve a cada alternativa. 

#### **16.1 Cuadro comparativo: pros, contras y recomendación** 

|**Criterio**|**A. Hardware explícito**|**B. Valores relatvos**|**C. Modelo por nodo**|
|---|---|---|---|
|Nodos requeridos para ser<br>defendible|Varios y realmente<br>diversos|Al menos 2,<br>idealmente 3+|Uno por estudio;<br>repetble sin límite|
|Alineación con el alcance<br>intra-nodo ya aprobado en<br>el plan de grado|Baja, exige ampliar el<br>alcance formal|Media, se puede<br>presentar como<br>extensión|Alta, es la<br>contnuación natural<br>de lo ya aprobado|
|Riesgo de domain shif /<br>sobreajuste a los pocos<br>nodos disponibles|Medio-alto: con 2–3<br>nodos el modelo<br>puede memorizar<br>diferencias entre<br>nodos en vez de<br>aprender una relación<br>general|Medio: la<br>normalización reduce<br>diferencias de escala<br>pero no garantza<br>equivalencia<br>semántca de eventos<br>PMU|Bajo dentro del nodo:<br>cada modelo solo<br>necesita generalizar a<br>nuevas corridas de su<br>propio dominio|
|Trabajo experimental<br>adicional requerido|Alto: perfles de<br>hardware, calibración<br>de kernels por nodo,<br>dataset conjunto con<br>splits leave-one-node-<br>out|Medio-alto:<br>calibración estable<br>por nodo (CV ≤ 5%)<br>más el mismo<br>entrenamiento<br>conjunto|Medio:<br>caracterización y<br>campaña por nodo,<br>pero reutlizando el<br>mismo protocolo<br>versionado|



Página 25 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

|**Criterio**|**A. Hardware explícito**|**B. Valores relatvos**|**C. Modelo por nodo**|
|---|---|---|---|
|Qué queda como<br>conclusión defendible con<br>la evidencia esperada|"Transfere si se<br>demuestra" —<br>afrmación fuerte,<br>difcil de sostener con<br>pocos nodos|"Transferencia<br>normalizada si se<br>demuestra" —<br>afrmación moderada,<br>evaluable por<br>ablación|"El pipeline es<br>reproducible y cada<br>modelo generaliza<br>dentro de su nodo" —<br>afrmación<br>conservadora y<br>alcanzable|
|Riesgo de sobreprometer<br>en la sustentación|Alto|Medio|Bajo|
|Recomendación|Reservar para trabajo<br>futuro, o para una<br>ampliación de alcance<br>explícitamente<br>acordada con el<br>director si aparecen<br>más nodos diversos.|Evaluar como<br>experimento<br>secundario de<br>transferencia (no<br>como requisito de<br>éxito del proyecto).|Adoptar como<br>arquitectura ofcial<br>del proyecto.|



**_Nota:_** _Esta recomendación es consistente con la restricción ya existente en el plan de grado aprobado, que limita formalmente la validación a un nodo y aclara explícitamente que no se busca un modelo universal de optimización energética._ 

#### **16.2 Qué se pospone hasta la decisión del director** 

- Comprometer un número y una diversidad concreta de nodos para la campaña (necesario para A y B, no para C). 

- Diseñar los splits leave-one-node-out y el experimento cross-node formal. 

- Ampliar el alcance formal del trabajo de grado hacia "modelo transferible" en vez de "pipeline reproducible con modelos locales". 

- Decidir si la compilación de los binarios de la suite (NPB/STREAM/ERT) debe ser -march=native por nodo (aceptable si el modelo es local, problemático si se busca comparar nodos directamente). 

#### **16.3 Qué módulo sirve a cada alternativa** 

Tabla de referencia rápida para no perder de vista, durante la implementación, por qué se está construyendo cada pieza. 

|**Módulo**<br>node_profle.py|**A (hardware**<br>**explícito)**<br>Imprescindible: es el<br>insumo directo del<br>modelo.|**B (features relatvas)**<br>No se usa<br>directamente.|**C (modelo por nodo)**<br>Útl como metadata<br>de auditoría, no<br>imprescindible.|
|---|---|---|---|
|calibraton.py (referencias|No se usa|Imprescindible: es el|Útl, pero no|
|P95 + CV%)|directamente.|insumo directo de la|imprescindible para|



Página 26 de 27 

Guía Técnica — Orquestador de Campañas de Telemetría (Fase 1) 

|**Módulo**|**A (hardware**<br>**explícito)**|**B (features relatvas)**|**C (modelo por nodo)**|
|---|---|---|---|
|||normalización.|entrenar un modelo<br>local.|
|calibraton.py (Roofine:<br>P_pico/BW_pico/I_ridge)|Se reutliza para<br>etquetar, igual que<br>en C.|Se reutliza para<br>etquetar, igual que<br>en C.|Imprescindible: es el<br>mecanismo de<br>etquetado de<br>entrenamiento<br>(sección 10).|
|catalog.py + adaptador de<br>kernels externos|Compartdo sin<br>cambios entre las<br>tres.|Compartdo sin<br>cambios entre las<br>tres.|Compartdo sin<br>cambios entre las<br>tres.|
|postprocess.py (features<br>absolutas + relatvas)|Usa solo las absolutas<br>+ node_profle.|Usa las relatvas ya<br>calculadas.|Usa solo las<br>absolutas.|
|campaign.py (matriz<br>mono-nodo, repetble)|Se repetría idéntca<br>en cada nodo del<br>estudio.|Se repetría idéntca<br>en cada nodo del<br>estudio.|Es exactamente el<br>modo de uso<br>previsto: una<br>campaña por nodo.|



**_Nota:_** _Ningún módulo de esta guía es exclusivo de una sola alternativa salvo el propio entrenamiento del modelo (fuera del alcance de esta guía, que cubre orquestación y telemetría, no la Fase 2 de aprendizaje automático). Esa es precisamente la propiedad que hace posible seguir desarrollando sin esperar la decisión del director._ 

Página 27 de 27 

