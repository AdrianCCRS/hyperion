# Plan detallado de ejecución — Agente DVFS en espacio de usuario para sistemas heterogéneos CPU–GPU

**Trabajo de grado:** Diseño e Implementación de un Agente en Espacio de Usuario para la Gestión Dinámica de Frecuencia (DVFS) en Sistemas Heterogéneos mediante Modelos Ligeros de Machine Learning
**Autores:** Yeison Adrian Caceres Torres, Ricardo Andres Perez Porras
**Director:** Gilberto Javier Diaz Toro, Ph.D.
**Documento preparado para:** revisión y validación técnica del proyecto (revisor: Luis Alejandro)
**Estado:** guía de ejecución técnica para realinear la implementación con los objetivos aprobados — auditado dos veces contra el repositorio real (2026-08-31): una confrontación general documento-vs-código, y una segunda pasada exclusivamente sobre código/scripts ejecutables (sin apoyarse en `AGENTS.md`, changelog ni ningún otro documento de texto) para verificar que ejecutar este plan efectivamente cierra los 4 objetivos aprobados. Las correcciones de ambas rondas están incorporadas inline, marcadas con ⚠️/✅ y ancladas a archivo:línea real donde aplica.

---

## 0. Cómo leer este documento

Este documento traduce la pregunta de investigación y los cuatro objetivos específicos del `plan_trabajo_grado.md` aprobado en abril de 2026 a un procedimiento de ejecución completo, con cada decisión técnica justificada. No reemplaza el plan aprobado como documento académico — es el nivel de detalle de ingeniería que el plan no especifica, y que debe incorporarse al capítulo de Metodología antes de la sustentación.

**Nota de auditoría:** una primera versión de este documento mezcló, sin darse cuenta, datos de dos ramas distintas del repositorio (`main`, donde vive todo lo que corresponde a los 4 objetivos aprobados, y `origin/fase-02`, donde vive el "selector de dispositivo" que §7.2 declara fuera de alcance). La corrección de esa mezcla, y una verificación adicional de que el código real (no solo la documentación) respalda cada pieza que este documento da por existente, están incorporadas en §2.1.1, §3.2–§3.5, §4.1, §5.1–§5.2 y §6.

Cada sección marcada con 🔶 **Desviación del plan original** documenta explícitamente un punto donde la implementación necesaria se aparta de la letra del plan aprobado, junto con la justificación técnica y la recomendación de cómo formalizarla (enmienda documentada, consulta al director, o ambas). Esto es intencional: el objetivo de este documento es que el proyecto avance de forma técnicamente correcta *sin* que esa corrección quede oculta o sin trazabilidad frente al comité evaluador.

---

## 1. Recordatorio: los cuatro objetivos que este documento debe cumplir

| # | Objetivo aprobado | Fase |
|---|---|---|
| 1 | Caracterizar comportamiento computacional y energético de cargas representativas (CPU y GPU, compute e memory-bound) vía Perf/RAPL/NVML | Fase 1 |
| 2 | Entrenar y validar un clasificador ligero (árbol/bosque) que infiera la fase compute_bound/memory_bound en tiempo real, con baja latencia | Fase 2 |
| 3 | Desarrollar un daemon en espacio de usuario que, según esa clasificación, aplique políticas de DVFS vía interfaces estándar del SO | Fase 3 |
| 4 | Evaluar el impacto empírico vía EDP, determinando si el ahorro compensa el overhead de inferencia, frente a gobernadores nativos de Linux | Fase 4 |

**Restricción de alcance explícita del plan (Sección 6.1–6.2):** sin kernel-space, sin deep learning/RL, sin multinodo, un solo nodo heterogéneo x86+NVIDIA. Este documento respeta esas restricciones en su totalidad.

🔶 **Decisión de diseño explícita de este documento:** la variable de control del sistema es **la frecuencia del dispositivo donde ya se ejecuta cada fase**, no la elección de en qué dispositivo ejecutarla. El "selector de dispositivo CPU/GPU" que aparece en trabajo reciente del repositorio (`classifier/selector/`, `docs/general/metodologia_selector_cpu_gpu_20260827.md`) queda **fuera del alcance de este plan**, porque no está en los objetivos aprobados. Ver §7 para la justificación completa de esta decisión y cómo tratar ese trabajo ya hecho.

---

## 2. Fase 1 — Recolección de telemetría y construcción del dataset

### 2.0 Prerrequisitos de plataforma — verificar antes de ejecutar cualquier campaña

Estos dos puntos no son parte del diseño de este documento — son condiciones del clúster, fuera del control directo del código del proyecto, y **bloquean por completo** las secciones que siguen si no están resueltas. Verificarlas es el primer paso de Fase 1, no una validación posterior.

#### 2.0.1 Permiso de escritura de frecuencia de CPU (Turbo Boost) — cómo verificarlo

1. Dentro de una asignación **exclusiva** de Slurm real (no una sesión de desarrollo compartida), escribir `scaling_max_freq`/`scaling_min_freq` a un nivel bajo (p. ej. el mínimo soportado) sobre los cores delegados.
2. **Releer inmediatamente el archivo sysfs** para confirmar que el kernel registró la escritura — esto confirma permiso de escritura, no actuación física.
3. Lanzar un kernel de CPU real y sostenido (p. ej. `ert_probe` o cualquier microbenchmark de cómputo intensivo del catálogo) sobre esos mismos cores.
4. Mientras corre, **muestrear `scaling_cur_freq` a granularidad fina (cada ~5 ms)**, misma metodología que ya usó el repositorio en `ARC-159` — confirmar que el reloj efectivo converge al valor solicitado y se sostiene ahí, no que toque ese valor una sola vez.
5. Si el reloj efectivo se queda en un valor de Turbo Boost en vez de converger al solicitado: leer el estado de Turbo del nodo (`/sys/devices/system/cpu/intel_pstate/no_turbo` o el equivalente de la plataforma) y confirmar si está en `0` (Turbo activo).
6. Intentar `sudo -n set_turbo_state 1` (el mecanismo que el propio repositorio ya identificó como la vía prevista) dentro de la misma asignación exclusiva. Dos resultados posibles:
   - **Éxito sin pedir contraseña:** el permiso está resuelto — repetir el paso 4 para confirmar que ahora sí converge, y documentar la confirmación con timestamp.
   - **Pide contraseña o falla:** el permiso sigue bloqueado del lado del clúster.
7. **Si queda bloqueado**, reunir la evidencia exacta (comando ejecutado, salida completa, timestamp, `job_id` de Slurm, lectura de `scaling_cur_freq` mostrando la no convergencia) y prepararla para escalar al administrador — no intentar workarounds que oculten el problema (p. ej. promediar sobre muchas repeticiones esperando que el efecto de Turbo se diluya; eso contamina el dato, no lo corrige).
8. Repetir la verificación completa después de cualquier cambio del lado del administrador, antes de dar por resuelto el prerrequisito.

#### 2.0.2 Restricción real del reloj de GPU bajo carga — cómo verificarlo

1. Lanzar un kernel GPU real bajo **carga de cómputo sostenida** — confirmar primero, con `nvidia-smi`/NVML, que la utilización está efectivamente en 100%, no en reposo (la restricción de reloj puede funcionar en reposo y fallar bajo carga real, que es exactamente el caso ya documentado en el repositorio).
2. Aplicar `nvidia-smi -lgc <objetivo>,<objetivo>` hacia un valor distinto del que la GPU tomaría por defecto bajo esa carga.
3. **Releer `nvmlDeviceGetClockInfo` de forma continua durante la ejecución** (no una sola lectura al final) y confirmar que el reloj observado converge al objetivo y se mantiene ahí mientras dura la carga.
4. Repetir con varios valores objetivo distintos a lo largo del rango soportado (no confiar en que un solo punto probado generaliza a todo el rango) y con al menos dos binarios distintos del catálogo, para descartar que el problema sea específico de un kernel.
5. **Si el reloj no converge:** antes de escalar, agotar las verificaciones que ya se pueden hacer sin el administrador, siguiendo el mismo orden que el repositorio ya documentó como efectivo (`ARC-113`):
   - Revisar `nvidia-smi -q -d PERFORMANCE` → `Clocks Event Reasons` **durante la carga activa**, no en reposo (en reposo, "Idle: Active" es trivial y no informa nada).
   - Revisar `nvidia-smi -q -d POWER` para descartar que un límite de potencia esté forzando el reloj hacia abajo.
   - Confirmar `GPU Operation Mode` y estado de `MIG` (`Current`/`Pending`) como configuración limpia, no una causa oculta.
   - Confirmar que no hay otro proceso de gestión/monitoreo de GPU (DCGM u otro) corriendo en paralelo que pueda estar sobrescribiendo el candado.
   - Probar variantes de sintaxis del comando (`-lgc <t>,<t>` vs. `-lgc <min>,<max>` vs. `-lgc <t>` sin coma) por si el driver interpreta alguna de forma distinta a la esperada.
6. **Si tras agotar lo anterior el reloj sigue sin converger bajo carga real**, reunir la evidencia completa (todas las verificaciones del paso 5, con su salida exacta, más los valores objetivo probados y el resultado de cada uno) y escalar al administrador — el propio repositorio ya armó este tipo de reporte una vez para el caso de CPU y puede usarse como plantilla.
7. Repetir la verificación completa después de cualquier cambio de driver o configuración del lado del administrador, antes de confiar en cualquier dato de §2.4 o §2.4.1.

**Criterio de decisión:** si alguno de los dos puntos no se puede confirmar en un plazo razonable, el resultado se documenta explícitamente como bloqueo de infraestructura (con la evidencia técnica reunida, lista para escalar al administrador) y se reporta como tal en el objetivo 4 — nunca se fabrica o asume un dato de frecuencia que no fue verificado bajo carga real.

### 2.1 Selección de cargas de trabajo

El plan exige una muestra intencional que cubra los **cuatro escenarios base**: CPU compute-bound, CPU memory-bound, GPU compute-bound, GPU memory-bound. Reglas para esta selección:

1. **Nunca asumir la etiqueta por el nombre o la reputación del kernel en la literatura.** Un kernel "reputado" como compute-bound (p. ej. `hotspot` de Rodinia) puede medir memory-bound en el hardware real. La etiqueta se deriva empíricamente (§2.3), nunca del catálogo de origen del benchmark.
2. Usar binarios de terceros verificables (NAS Parallel Benchmarks, Rodinia, RAJAPerf, STREAM/ERT para calibración) — nunca inventar ni hardcodear un kernel sintético como parte del dataset de entrenamiento. Los microbenchmarks sintéticos propios sirven solo para desarrollo/pruebas del harness, nunca para poblar el dataset final.
3. Cada binario debe declararse en un catálogo versionado con checksum verificado antes de ejecutarse (evita que una recompilación silenciosa invalide corridas pasadas sin que nadie lo note).
4. Descartar cualquier kernel cuya naturaleza aritmética invalide la intensidad operacional (p. ej. kernels de enteros puros sin ninguna operación de punto flotante) — no aportan una medida válida de FLOPs/byte.

#### 2.1.1 Ampliación del catálogo para viabilidad estadística de Fase 2

⚠️ **Corrección importante frente a una versión anterior de este documento — mezcla de ramas detectada y resuelta.** Una versión anterior de esta sección afirmaba, citando directamente `orchestrator/schemas/kernels/catalog.yaml`, que el catálogo real tenía **119 entradas de dataset en CPU y 107 en GPU** con seis familias `dual_gemm_*`/`dual_fft_*`/`dual_axpy_*`/`dual_stencil_*`/`dual_cholesky_*`/`dual_spmv_*`. Una auditoría directa contra el repositorio (rama `main`, HEAD en el momento de la auditoría: `aa4df75`) mostró que esa afirmación era **incorrecta para `main`**: el catálogo real de `main` tiene **23 kernels en total** (10 no-GPU: STREAM, ERT, NPB bt/mg/cg/sp/ft/lu, DGEMM, RAJAPerf 3MM; 13 GPU: Rodinia gaussian/backprop/lavamd/heartwall/lud/myocyte/dwt2d + microbenchmarks GPU propios), **sin ninguna familia `dual_*`**. El catálogo de 232 entradas (114 `device: gpu`) con las seis familias `dual_*` existe, verificado por número de archivo idéntico, únicamente en la rama remota `origin/fase-02` — la misma rama de `classifier/selector/` que §7.2 de este documento declara fuera de alcance. Es decir: la versión anterior de esta sección diseñó la viabilidad estadística de Fase 1/2 sobre datos que pertenecen exactamente al workstream que el propio documento excluye.

**Decisión tomada para resolver esto (confirmada explícitamente con el autor de este documento):** en vez de descartar el catálogo ampliado, se **fusiona `orchestrator/schemas/kernels/catalog.yaml` de `origin/fase-02` hacia `main`** como primer paso de ejecución (antes de §2.4), precisamente porque ese catálogo es un superset verificado de los 23 `id` de `main` (las 23 entradas comunes tienen el mismo `id`; hay que diferenciar antes de fusionar si algún `exec_path`/`binary_checksum`/`success_check` cambió de forma incompatible con el runner de `main`, campo por campo, no solo por `id`). Esto es una decisión operativa de esta realineación, no algo que ya estuviera resuelto en `main` — hasta que se ejecute, el catálogo real de `main` sigue siendo el de 23 entradas, y cualquier trabajo de Fase 1/2 que se apoye en el catálogo ampliado debe esperar a que la fusión esté hecha y verificada.

**Dato adicional, verificado en código real de entrenamiento (no en el catálogo declarativo):** `classifier/training/train_phase.py` (rama `fase-02`) — el script que ya entrena y valida el clasificador compute/memory-bound sobre una campaña real (`pacca_cpu_final_attempt03_20260820_arc174`) — usa una lista fija de **9 kernels** (`npb_bt`, `npb_mg`, `npb_cg`, `npb_sp`, `npb_ft`, `npb_lu`, `dgemm_n2048`, `rodinia_lavamd_omp`, `rajaperf_polybench_3mm_omp`), todos presentes en el catálogo de 23 de `main`. El propio docstring de `classifier/eval/protocol.py` (el módulo de protocolo de validación que usa ese script) documenta esto como una limitación de diseño explícita: *"El dataset tiene 9.95 M de ventanas entrenables pero solo 9 kernels"*, y por eso exige leave-one-kernel-out en vez de split aleatorio. Esto confirma, de forma independiente al catálogo declarativo, que el entrenamiento real ya ejecutado hasta ahora trabajó con 9 kernels, no con 119 — y que el propio código ya reconoce la necesidad de una validación agrupada, aunque agrupada por kernel individual, no todavía por familia algorítmica (ver corrección al punto 1 más abajo).

Con el catálogo ampliado ya verificado como real (solo que vive en `fase-02`, pendiente de fusión) el análisis original de esta subsección se mantiene, con una corrección:

1. **El conteo bruto de entradas no es la métrica correcta para leave-one-kernel-out.** La gran mayoría de esas entradas son familias `dual_gemm_*`, `dual_fft_*`, `dual_axpy_*`, `dual_stencil_*`, `dual_cholesky_*`, `dual_spmv_*` — el mismo algoritmo barrido sobre docenas de tamaños de problema distintos (`N64` a `N16384` y similares). Dejar fuera un tamaño de GEMM no prueba generalización a un algoritmo nuevo, solo a un tamaño nuevo del mismo algoritmo ya visto. La validación de §3.3 debe agruparse por **familia algorítmica** (`gemm`, `fft`, `axpy`, `stencil`, `cholesky`, `spmv`, `npb_bt`, etc.), no por `kernel_ref` individual — de lo contrario el leave-one-kernel-out sobreestima la generalización real. ⚠️ **Corrección de código, no de catálogo:** `classifier/training/train_phase.py` importa `classifier.eval.protocol.leave_one_kernel_out()`, que agrupa por `kernel_ref` (`protocol.py`, parámetro `kernel_col: str = "kernel_ref"`), no por familia. Una vez fusionado el catálogo ampliado, este script generalizará mal si se lo apunta a las familias `dual_*` sin antes añadirle una función `kernel_ref → familia_algorítmica` y sustituir `leave_one_kernel_out` por un `leave_one_familia_out` equivalente — esa función no existe todavía en ninguna rama, hay que construirla como parte de §3.5/§6.
2. **Contando por familia algorítmica distinta (no por tamaño), el catálogo real tiene aproximadamente 25–28 familias en CPU** (NPB: `bt`/`mg`/`cg`/`sp`/`ft`/`lu`; `dgemm`, `lavamd_omp`, `rajaperf_3mm`, `ptrchase`, `phasic`, RAJAPerf `stream`/`lcals`/`polybench`/`basic`, `lulesh`, `hpcg`, GAP `bfs`/`pr`, `cholmod`, y las 6 familias `dual_*`) — un número razonable para leave-one-familia-out, mejor de lo que se había asumido.
3. **Verificar antes de reutilizar la familia `dual_*`:** ese conjunto se construyó para el trabajo de "selector" de tamaños (§7.2, fuera de alcance de este plan) — hay que confirmar explícitamente que su telemetría pasa por el mismo pipeline de etiquetado de §2.3 (Roofline real vía `uncore_imc`/`ncu`, no una métrica ad hoc del selector) antes de darla por utilizable para los clasificadores de §3. Si no es compatible, se re-etiqueta con el pipeline de este plan antes de usarla, no se descarta solo por su origen.
4. **Con todo lo anterior, el hueco real no es de cantidad sino de diversidad de patrón de acceso a memoria** — varias familias del catálogo actual (NPB, RAJAPerf `stream`/`axpy`) son intensivas en ancho de banda de forma bastante uniforme; el catálogo tiene menos representación de patrones **irregulares** (recorridos de estructuras, acceso disperso) que suelen ser los casos límite más interesantes para un clasificador compute/memory-bound. Candidatos concretos a agregar, con motivo:

| Kernel | Fuente | Dispositivo | Por qué complementa el catálogo actual |
|---|---|---|---|
| `npb_ep` (Embarrassingly Parallel) | NAS Parallel Benchmarks | CPU | Referencia fuerte de compute-bound puro (generación de números aleatorios + funciones trigonométricas, casi sin tráfico de memoria) — el catálogo actual no tiene un ancla explícita de este extremo |
| `rodinia_kmeans` (versión OMP) | Rodinia | CPU | Acceso disperso a memoria por clustering iterativo — patrón irregular no representado hoy |
| `rodinia_srad` (versión OMP) | Rodinia | CPU | Stencil de difusión sobre imagen — memory-bound estructurado pero con dependencias vecinas distintas a `dual_stencil_*` |
| `rodinia_nw` (Needleman-Wunsch, OMP) | Rodinia | CPU | Programación dinámica con acceso a memoria en diagonal — patrón irregular, buen caso límite |
| `rodinia_particlefilter` (OMP) | Rodinia | CPU | Mezcla de fases compute-bound y memory-bound dentro del mismo kernel — útil para probar la granularidad de ventana de clasificación |
| `rodinia_kmeans` | Rodinia | GPU | Mismo patrón disperso que su versión CPU, para tener el par CPU/GPU del mismo algoritmo |
| `rodinia_srad` | Rodinia | GPU | Contraparte GPU del stencil de difusión |
| `rodinia_nw` | Rodinia | GPU | Contraparte GPU de la programación dinámica en diagonal |
| `rodinia_b+tree` | Rodinia | GPU | Recorrido de árbol — acceso a memoria más irregular que cualquier kernel ya presente en el catálogo GPU |
| `rodinia_cfd` | Rodinia | GPU | Solver de volúmenes finitos, candidato fuerte a compute-bound en GPU para balancear la clase frente a los kernels memory-bound de la tabla |

Estos diez cierran específicamente el hueco de patrones irregulares y dan pares CPU/GPU del mismo algoritmo (`kmeans`, `srad`, `nw`), útil para comparar si la clase asignada a un mismo patrón de acceso coincide entre dispositivos. No reemplazan el resto del proceso de esta subsección — cada uno sigue las reglas 1–4 de §2.1 (verificación empírica, checksum, descarte si su naturaleza aritmética invalida la intensidad operacional).

5. **Fijar un objetivo mínimo explícito por clase, no solo por familia:** tras el etiquetado empírico de §2.3, cada clase (`compute_bound`, `memory_bound`) debe quedar representada por un mínimo de 5–6 familias algorítmicas distintas por dispositivo — no basta con que el catálogo sea grande si termina concentrado en una sola clase.
6. **Verificar el balance de clases después del etiquetado empírico, no antes.** Si tras ejecutar §2.3 sobre las familias existentes y las nuevas la distribución sigue desbalanceada, volver a la tabla anterior y buscar deliberadamente más candidatos del patrón que falte — la selección final se valida contra el resultado medido, nunca contra la expectativa inicial.
7. Esta verificación (familias existentes agrupadas correctamente, familia `dual_*` confirmada compatible o re-etiquetada, kernels nuevos incorporados) debe completarse **antes de correr el barrido de frecuencia de §2.4** sobre el catálogo completo, para no repetir campañas de barrido dos veces sobre el mismo hardware.

### 2.2 Medidas exactas a recolectar

#### CPU (vía `perf_event`, ventana de muestreo ≈ 1 ms)

| Contador crudo | Evento perf | Uso |
|---|---|---|
| `instructions` | `PERF_COUNT_HW_INSTRUCTIONS` | base de IPC |
| `cycles` | `PERF_COUNT_HW_CPU_CYCLES` | base de IPC y stall ratio |
| `cache_references` | `PERF_COUNT_HW_CACHE_REFERENCES` | base de miss rate |
| `cache_misses` | `PERF_COUNT_HW_CACHE_MISSES` | base de miss rate/MPKI |
| `stalled_cycles_backend` | `PERF_COUNT_HW_STALLED_CYCLES_BACKEND` | base de stall ratio |
| `fp_scalar_double`, `fp_128b_packed_double`, `fp_256b_packed_double`, `fp_512b_packed_double` | `FP_ARITH_INST_RETIRED` (raw, event=0xC7) | FLOPs reales retirados, ponderados por ancho de vector (1/2/4/8) |
| `uncore_cas_count_read`, `uncore_cas_count_write` | `uncore_imc` CAS_COUNT_READ/WRITE (ámbito socket, requiere `CAP_PERFMON` y nodo con `--exclusive`) | bytes reales movidos hacia/desde DRAM |

Energía: RAPL `pkg_delta_uj` (paquete) y `dram_delta_uj` (memoria), de donde se deriva `power_w`.
Estado de actuación (no es feature de fase, es covariable de control): `freq_khz_observed`, releído del sysfs real, nunca asumido por el valor solicitado.

**Requisito de exclusividad:** dado que `uncore_imc` es de ámbito socket completo, cualquier campaña que lo habilite exige reserva exclusiva del nodo (`--exclusive` en Slurm). Sin esto, la medición de bytes reales queda contaminada por otros procesos del nodo compartido.

#### GPU (vía NVML, muestreo ≈ 100 ms — más lento que CPU por limitación del propio driver)

| Métrica cruda | Fuente NVML | Uso |
|---|---|---|
| `gpu_util_pct` | `nvmlDeviceGetUtilizationRates().gpu` | % de tiempo con actividad en SM |
| `gpu_mem_util_pct` | `nvmlDeviceGetUtilizationRates().memory` | % de tiempo con actividad en el controlador de memoria de GPU |
| `gpu_power_mw` | `nvmlDeviceGetPowerUsage` | potencia instantánea |
| `gpu_energy_mj` / `gpu_energy_delta_mj` | `nvmlDeviceGetTotalEnergyConsumption` | energía acumulada y su delta por ventana |
| `gpu_sm_clock_mhz` | `nvmlDeviceGetClockInfo` | reloj de SM real, confirma que el nivel DVFS solicitado se sostuvo |
| `gpu_temperature_c` | `nvmlDeviceGetTemperature` | descarta contaminación térmica de la medición |

**Asimetría estructural que hay que aceptar de diseño, no intentar disimular:** NVML no mide FLOPs ni bytes movidos. No existe un equivalente de `ipc`/`llc_miss_rate` para GPU en tiempo real. La intensidad operacional de GPU se caracteriza **offline, una sola vez por kernel**, con Nsight Compute (`ncu`), y ese valor (constante por kernel, no depende de la frecuencia) se guarda en el catálogo — nunca se invoca `ncu` en producción, porque su overhead es incompatible con inferencia en línea.

### 2.3 Metodología de etiquetado (`phase_label_train`)

La etiqueta **nunca se asume por el nombre del kernel ni se copia de un `phase_label_hint` de literatura**. Se deriva comparando la intensidad operacional medida contra el punto de ridge de la calibración Roofline del propio dispositivo:

- **CPU:** `operational_intensity = flops_measured_window / bytes_moved_uncore_real`, comparado contra `i_ridge` calibrado con STREAM (ancho de banda pico) y ERT (rendimiento pico), **exclusivamente con bytes reales de `uncore_imc`**. Si no hay cobertura de uncore para esa ventana, la fila queda `quality_status = "intensity_undefined"` — nunca se cae a un proxy sesgado (p. ej. `cache_misses × line_size`) ni se asigna una etiqueta con una fuente de menor calidad sin marcarla como tal.
- **GPU:** intensidad estática por kernel medida con `ncu`, comparada contra el ridge de GPU calibrado **por precisión (fp32/fp64) y por nivel de frecuencia** (no existe un ridge único de GPU, la relación FLOPs/byte alcanzable cambia con el reloj y con la precisión aritmética).
- Cualquier expectativa de la literatura sobre si un kernel "debería" ser compute o memory-bound se guarda aparte como `phase_label_hint`, únicamente para auditoría — nunca alimenta el entrenamiento.

### 2.4 Diseño de la campaña de barrido de frecuencia

Este es el paso que el plan aprobado no especifica con suficiente detalle (ver la discusión de "cómo sé que la frecuencia elegida es óptima" más abajo, §3.4) y que hay que ejecutar explícitamente en Fase 1, no diferirlo a la Fase 3.

⚠️ **Orden obligatorio para GPU: ejecutar primero §2.4.1 (medición de `T_transición_gpu`) antes del paso 1 de esta sección.** El paso 1 para GPU asume ese valor ya medido — correr el barrido de GPU sin haberlo medido antes invalida en silencio cualquier fase cuya duración sea menor a la latencia de conmutación real, sin que nada en la campaña lo señale como tal.

1. Para cada kernel del catálogo, ejecutar un barrido completo sobre **todos los niveles de frecuencia soportados** del dispositivo correspondiente (CPU: `scaling_min_freq`/`scaling_max_freq`, típicamente 5 niveles REF+F0–F4; GPU: `nvidia-smi -lgc`, sobre el rango real de `SUPPORTED_CLOCKS`, nunca asumido de la hoja de datos — **solo después de completar §2.4.1**).
2. Repetir cada combinación (kernel × nivel de frecuencia) un mínimo de 3 veces para tener variabilidad estadística.
3. **Verificar por relectura, nunca por código de retorno**, que el reloj efectivo corresponde al nivel solicitado — el driver puede ignorar la solicitud silenciosamente (Turbo Boost en CPU, throttling térmico en GPU).
4. Registrar, por cada combinación: tiempo total, energía (RAPL/NVML), potencia media, y el EDP resultante.
5. **Nunca borrar una corrida rechazada.** Se conserva en disco con `accepted: false` y su código de rechazo (`factor_id`) — es evidencia de auditoría para la sustentación, no ruido a limpiar.

#### 2.4.1 Medición obligatoria de la latencia de conmutación de reloj de GPU (`T_transición_gpu`) — ejecutar antes del paso 1 anterior para GPU

Este paso es un prerrequisito del barrido de GPU (paso 1 anterior), no una validación posterior — sin él, el EDP medido en el barrido de GPU puede estar contaminado por fases que terminan antes de que el reloj realmente haya convergido al nivel solicitado, y la campaña completa quedaría con datos no confiables sin que nada lo señale.

**Qué se mide:** el tiempo real, bajo carga sostenida, entre emitir `nvidia-smi -lgc <objetivo>` y el momento en que el reloj de GPU efectivo se estabiliza en ese valor — no la primera lectura que lo toca, sino que se mantenga dentro de una banda de tolerancia durante varias lecturas consecutivas, para descartar overshoot o inestabilidad transitoria del propio driver.

**Por qué no se puede asumir un valor de la ficha técnica:** el propio repositorio ya encontró, para CPU, que la latencia de conmutación real bajo carga puede ser mucho mayor que la nominal — `ARC-159` documentó que ni siquiera una pausa de asentamiento de 300 ms fue suficiente para que `scaling_cur_freq` convergiera al valor del candado aplicado, muy por encima de los ~1–10 ms que cita la literatura para P-states con HWP. No hay ninguna razón para asumir que el reloj de GPU se comporte mejor, y no existe hoy ningún valor medido de esto en el repositorio.

**Cómo se mide, paso a paso (misma metodología que `ARC-159`, aplicada a GPU):**

1. Lanzar un kernel GPU real y sostenido, de duración conocida y larga (varios segundos), para tener margen suficiente de observación.
2. Mientras corre, emitir `nvidia-smi -lgc <objetivo>` hacia un nivel distinto del actual.
3. Desde ese instante, muestrear `nvmlDeviceGetClockInfo` a la cadencia más fina que NVML permita — **nunca a 100 ms**, que es demasiado grueso para resolver esta latencia — y registrar el timestamp en que el reloj observado entra y se sostiene dentro de una banda de tolerancia del valor objetivo.
4. Repetir para varios pares (nivel de origen, nivel de destino), sin asumir que la latencia es simétrica ni constante en todo el rango — puede ser distinta subiendo que bajando, o distinta según qué tan lejos esté el salto entre niveles.
5. Repetir cada par un mínimo de 3 veces, con el mismo criterio de variabilidad estadística del paso 2 del barrido general.

**Qué implica el resultado:**

- Si `T_transición_gpu` medido resulta muy por debajo de la duración típica de las fases del catálogo real → la actuación de frecuencia de GPU por fase (§4.1) es viable tal como está diseñada.
- Si resulta comparable o mayor a la duración típica de fase → define un **umbral mínimo de duración de fase** por debajo del cual actuar frecuencia en GPU no solo no ayuda, sino que puede empeorar el EDP frente a no tocar nada (se paga el costo de la transición sin llegar a correr realmente al nivel decidido).

**Para qué se requiere — dónde se usa después de medido:**

1. **Alimenta directamente el filtro del derivador de la tabla de política (§3.5):** las fases del barrido más cortas que `T_transición_gpu` se excluyen (o se marcan aparte, nunca se mezclan sin distinción) del cálculo de EDP agregado por nivel, porque su medición está contaminada por una transición sin converger.
2. **Fija el valor mínimo de `min_dwell_ns` en `gpu_clock_controller.hpp`** — no puede fijarse arbitrariamente ni por defecto del código; debe ser, como mínimo, el `T_transición_gpu` medido, o el daemon podría solicitar un nuevo nivel antes de que el anterior se hubiera asentado.
3. **Es insumo necesario para reportar honestamente en el objetivo 4** si actuar frecuencia en GPU es o no un mecanismo que vale la pena para la distribución real de duraciones de fase del catálogo del proyecto — incluyendo la posibilidad legítima de que la respuesta sea "no, para la mayoría de las fases del catálogo", tal como puede ocurrir en CPU por el hallazgo del rango dinámico de potencia angosto (§3.4).

### 2.5 Construcción del dataset y control de correlación entre features

Antes de fijar el conjunto final de columnas:

1. Calcular la matriz de correlación (Pearson y Spearman) sobre todas las columnas candidatas del `windows.csv` real (no sobre una muestra sintética).
2. Para cada par con `|ρ| > 0.85`, conservar una sola columna, priorizando la medición física más directa sobre el proxy derivado (ejemplo: `uncore_cas_count_*`/`bytes_moved_uncore_real` sobre `delta_cache_misses`, porque el primero mide bytes de DRAM directamente y el segundo es un proxy ya documentado como sesgado por no ver tráfico de prefetch).
3. Calcular VIF (Variance Inflation Factor) sobre el conjunto resultante para detectar colinealidad multivariable que un análisis pareja-a-pareja no captura.
4. Documentar explícitamente, en el capítulo de metodología, qué columnas se descartaron y por qué — esto es evidencia de rigor metodológico frente al comité.

**Conjunto final de features recomendado, por clasificador:**

**Clasificador CPU** (una fila = una ventana de ~1 ms):
- `ipc` (eficiencia de cómputo)
- `llc_miss_rate` **o** `mpki` — una sola de las dos
- `stall_backend_ratio` **o** `ipc` — una sola de las dos (son inversamente redundantes)
- `operational_intensity_uncore_real` (la variable más cercana a la etiqueta física en sí, útil como feature aunque también sea insumo de la etiqueta — su inclusión debe evaluarse con cuidado para no crear fuga de información, ver nota abajo)
- `power_w`
- `freq_khz_observed` (covariable de control)

⚠️ **Nota sobre fuga de información (data leakage):** si `operational_intensity_uncore_real` es tanto el insumo de `phase_label_train` como una feature de entrada, el clasificador puede terminar "memorizando" la regla de etiquetado en vez de aprender un patrón generalizable a partir de contadores más baratos de leer en producción. Evaluar dos variantes del clasificador: una que incluye `operational_intensity` como feature (límite superior de desempeño alcanzable) y otra que solo usa `ipc`/`llc_miss_rate`/`power_w` (el escenario realista de producción, donde calcular la intensidad exacta en línea puede no ser viable). Reportar ambas en el capítulo de resultados.

**Clasificador GPU** (una fila = una fase, no una ventana de tiempo fijo):
- `gpu_util_pct`
- `gpu_mem_util_pct`
- `gpu_power_mw`
- `gpu_sm_clock_mhz` (covariable de control)
- `gpu_temperature_c` (descarta contaminación térmica como confusor)

### 2.6 Partición del dataset

**El split debe ser por operación/kernel completo, nunca por fila.** Dividir ventanas de la misma corrida entre entrenamiento y prueba produce fuga por autocorrelación temporal (ventanas consecutivas de un mismo kernel son casi idénticas) y sobreestima el desempeño real. Además, **agrupar por familia algorítmica, no solo por `kernel_ref`** (§2.1.1/§3.3): varias entradas del catálogo (`dual_gemm_*`, `dual_fft_*`, etc.) son el mismo algoritmo en distintos tamaños — dejarlas repartidas entre entrenamiento y prueba filtra información del mismo patrón computacional y vuelve a sobreestimar la generalización, aunque técnicamente sean `kernel_ref` distintos. Usar validación **leave-one-familia-out** (k-fold agrupado por familia algorítmica), nunca k-fold aleatorio sobre filas ni sobre `kernel_ref` individuales cuando varios comparten familia.

---

## 3. Fase 2 — Entrenamiento y validación del clasificador

### 3.1 Formulación del problema

Dos problemas de clasificación binaria **independientes**, uno por dispositivo:
- `f_cpu(vector_features_cpu) → {compute_bound, memory_bound}`
- `f_gpu(vector_features_gpu) → {compute_bound, memory_bound}`

🔶 **Desviación del plan original — nota de alcance:** el plan (§5.2) describe "un modelo" sin distinguir por dispositivo. Usar dos modelos, uno por dispositivo, es una necesidad técnica (las fuentes de sensores son distintas, §2.2) y no una ampliación del objetivo — la salida sigue siendo binaria compute/memory-bound, tal como pide el objetivo 2, y la frecuencia sigue siendo la única variable de control (objetivo 3). Documentar esto explícitamente en el capítulo de metodología como una precisión de diseño, con una frase corta que la distinga del "selector de dispositivo" (§7), que sí es una ampliación real de alcance.

### 3.2 Modelos candidatos a comparar

Consistente con el marco conceptual del plan (§4.1.8): árbol de decisión, Random Forest, regresión logística como baseline lineal simple, y XGBoost como techo de capacidad (el plan ya lo describe conceptualmente, aunque no aparece en los objetivos específicos como obligatorio — usarlo como comparación, no como elección por defecto, para no introducir un modelo más pesado sin justificar que su ganancia predictiva compensa su mayor costo de inferencia).

✅ **Verificado en código (auditoría exclusiva de código, no de documentación) — esto ya existe y corre, con 3 huecos concretos.** `classifier/training/train_phase.py` (rama `fase-02`) es un script funcional, no un borrador: `build_models()` construye y compara `DummyClassifier` (línea base obligatoria), `DecisionTreeClassifier(max_depth=1)`, `LogisticRegression` (con `StandardScaler`), `DecisionTreeClassifier(max_depth=6)`, `RandomForestClassifier` y `ExtraTreesClassifier`; `measure_latency()` mide p50/p99 de una predicción aislada (no de lote, a propósito — el daemon decide sobre una ventana a la vez). Antes de "reactivar" este módulo tal como dice §6, hay tres huecos verificados por lectura directa del archivo, no asumidos:

1. **No incluye XGBoost.** `build_models()` no lo importa ni lo instancia — hay que añadirlo como comparación, tal como pide este mismo apartado.
2. **No serializa el modelo elegido en ningún punto.** El script entero es de evaluación comparativa (imprime tablas por stdout); no hay una sola llamada a `joblib.dump`/`pickle.dump`/exportación ONNX. Revisado los 40 archivos de `classifier/` en `fase-02`: la única serialización de modelo existente está en `classifier/selector/{r2,search,structured}.py` — el selector fuera de alcance. **Hoy no hay ningún camino de código que produzca un artefacto cargable por el daemon de §4.3** — hay que construirlo, no reactivarlo.
3. **Mide p50/p99, no p95/p99** como pide §3.3 punto 4 de este mismo documento — ajuste menor pero real en `measure_latency()`.

El listado de modelos (`FEATURES`) usado hoy es `ipc, mpki, llc_miss_rate, stall_backend_ratio, ips, running_ratio, freq_khz_observed` — un superconjunto razonable de la propuesta de §2.5, y ya incluye `freq_khz_observed` con la misma justificación de covariable de control que este documento propone. El chequeo de fuga de información de §2.5 también ya está implementado, y de forma más estricta que "evaluar dos variantes": `train_phase.py` define un conjunto `FORBIDDEN` (`operational_intensity*`, `i_ridge_used`, `flops_measured_window`, `bytes_moved_*`, `uncore_cas_count_*`, `phase_label_hint`) y aborta con `SystemExit` si alguna columna prohibida aparece en `FEATURES` — más una prohibición dura que una comparación de variantes. Si se quiere reportar ambas variantes tal como pide §3.3 punto 2, hay que relajar esa comprobación explícitamente para una corrida de auditoría, documentando por qué se hace solo esa vez.

### 3.3 Protocolo de entrenamiento y comparación

⚠️ **Nota sobre potencia estadística — leave-one-familia-out, no leave-one-kernel-out.** El catálogo real es grande en número de entradas (119 CPU / 107 GPU en `catalog.yaml`), pero buena parte son barridos de tamaño del mismo algoritmo (`dual_gemm_*`, `dual_fft_*`, etc.). Validar dejando fuera una sola entrada de esas familias no prueba generalización a un patrón nuevo. La partición externa debe agruparse por **familia algorítmica** (§2.1.1) — con eso, el catálogo tiene ~25–28 familias distintas en CPU, un número razonable para leave-one-familia-out, siempre que la ampliación y el balance por clase de §2.1.1 se completen (incluyendo confirmar que la familia `dual_*` es compatible con el etiquetado de §2.3 antes de contarla). Si por restricciones de tiempo la ampliación o la verificación de `dual_*` no alcanzan a completarse, el número final de familias usado y esta limitación deben quedar reportados explícitamente en el capítulo de resultados, no omitidos.

1. Validación anidada: partición externa leave-one-familia-out para estimar generalización, con búsqueda de hiperparámetros (grid o Bayesiana) en un split interno.
2. **Para el clasificador de CPU, entrenar y reportar las dos variantes de features definidas en §2.5** (con `operational_intensity_uncore_real` incluida vs. solo `ipc`/`llc_miss_rate`/`power_w`) — este paso no es opcional, es el chequeo de fuga de información que evita reportar una exactitud inflada por memorizar la regla de etiquetado en vez de aprender un patrón generalizable a partir de contadores realmente disponibles en producción.
3. Métricas de clasificación: exactitud, F1 por clase (no solo global, porque las clases pueden estar desbalanceadas si el catálogo tiene más kernels de un régimen que del otro), matriz de confusión.
4. **Latencia de inferencia medida en el hardware real de destino**, no estimada — es una restricción dura del objetivo 2 y del alcance del proyecto. Medir percentil 95 y 99, no solo el promedio, porque el daemon necesita un peor caso acotado para decidir la cadencia de muestreo viable.
5. Selección final: el modelo que minimice una función que combine error de clasificación y latencia de inferencia — nunca elegir por exactitud sola si el modelo más preciso introduce una latencia que compromete el objetivo de "no degradar el rendimiento global" (pregunta de investigación explícita del plan). **La variante sin `operational_intensity` (§2.5) es la que se serializa para el daemon real de §4.3**, salvo que se justifique explícitamente lo contrario.
6. Serializar el modelo elegido (formato liviano, p. ej. ONNX o el nativo de scikit-learn/XGBoost) para integrarlo al daemon.

### 3.4 Cómo se deriva la política de frecuencia a partir de la clase (el paso que el plan no detalla)

La clasificación por sí sola no dice qué frecuencia aplicar — solo dice en qué régimen está el sistema. La tabla clase → frecuencia óptima se construye **a partir de los datos del barrido de Fase 1 (§2.4)**, no del clasificador:

1. Para cada clase (`compute_bound`, `memory_bound`) y cada dispositivo, agregar el EDP observado en el barrido, por nivel de frecuencia, sobre todos los kernels que cayeron en esa clase.
2. Elegir, por clase, el nivel de frecuencia que minimiza el EDP agregado.
3. **Aceptar explícitamente el resultado "no cambiar la frecuencia" como una salida válida de este análisis**, si ningún nivel por debajo del nativo mejora el EDP para esa clase en ese hardware — este es exactamente el caso que ya se documentó preliminarmente para la CPU de la plataforma experimental (reducción de reloj 4× baja la potencia solo 28% mientras el tiempo se alarga 2.2–4×). ⚠️ **Ese número viene de una campaña anterior a esta reorganización de Fase 1.** Es una medición de potencia/tiempo puro, no depende de qué features de ML se elijan después, así que es razonable esperar que se sostenga — pero antes de usarlo como base firme de la tabla de política, debe **re-verificarse con la campaña reorganizada de §2.4** (reserva exclusiva de nodo, verificación por relectura tal como quedó especificada aquí), no darse por válido solo por existir en la documentación previa del repositorio. Una política de "no actuar" en CPU sigue siendo un resultado científico legítimo del objetivo 4, no un fracaso del diseño.

✅ **Confirmación independiente, encontrada en código (no en documentación) durante la auditoría de esta sección.** `classifier/eval/protocol.py::trivial_baselines()` ya calcula esto sobre datos reales, y su docstring registra el resultado ya obtenido: *"Con los datos de CPU actuales el óptimo es la frecuencia máxima en 9 de 9 kernels"*. Esta función ya corrió sobre la campaña real (`pacca_cpu_final_attempt03_20260820_arc174`) y encontró que la línea base "siempre a la frecuencia máxima" logra EDP-loss = 1.0 en los 9 kernels disponibles — es decir, en el código que ya existe, DVFS por sí solo no ganó en CPU en ningún caso probado hasta ahora. Esto corrobora, desde una fuente de evidencia distinta e independiente (código que ya ejecutó el cálculo, no un informe narrativo), el hallazgo del rango dinámico de potencia angosto. No sustituye la re-verificación con la campaña reorganizada de §2.4 que pide el párrafo anterior (sigue haciendo falta, sobre todo tras fusionar el catálogo ampliado de §2.1.1), pero sí es una señal fuerte de que "no actuar" en CPU es la salida más probable de este análisis, y de que la tabla de política debe estar preparada para documentar ese resultado con la misma seriedad que un nivel elegido.

4. Esta tabla (2 clases × 2 dispositivos = 4 entradas) es la política estática que consulta el daemon en producción — el daemon nunca recalcula el EDP en línea, solo aplica la tabla ya derivada offline.

### 3.5 Cómo construir el derivador de la tabla de política, paso a paso

Esta pieza no existe todavía **como script independiente**, pero su núcleo de cálculo sí existe y ya corre en producción de análisis: `classifier/eval/protocol.py` (rama `fase-02`) ya implementa `edp_loss()` (razón entre el EDP obtenido y el EDP del oráculo), `trivial_baselines()` (línea base "siempre máxima"/"al azar" vs. oráculo) y `honest_constant_baseline()` (mejor frecuencia constante única, recalculada por pliegue para no hacer trampa con información del kernel de prueba). Esto cubre gran parte de los pasos 3–5 de abajo (agregación de EDP, comparación contra baseline, honestidad de la partición) — falta envolverlo en un script offline dedicado que además: (a) filtre por `T_transición_gpu` (paso 2, no existe hoy en ningún lado), (b) corra la prueba de significancia estadística explícita del paso 5 (ver el hueco de código verificado en §5.2 — hoy no hay ningún uso de `scipy.stats` en el repositorio), y (c) serialice el resultado a la tabla de 4 entradas del paso 6. Es decir: no es "construir desde cero", es **extraer y envolver la lógica de `classifier/eval/protocol.py` en un módulo nuevo**, p. ej. `classifier/policy/derive_policy_table.py`, añadiéndole exactamente esas tres piezas que hoy le faltan.

1. **Cargar el dataset del barrido de §2.4** (`windows.csv` de las campañas CPU y GPU ya cerradas), filtrando únicamente filas con `quality_status="ok"` (CPU) o `"gpu_telemetry"` con etiqueta válida (GPU) — nunca corridas rechazadas.
2. **Para GPU, filtrar además por duración de fase contra `T_transición_gpu` (§2.4.1):** cualquier fase cuya duración sea menor al `T_transición_gpu` medido se excluye del cálculo de EDP por nivel (paso 4), marcada explícitamente como `excluded_transition_not_settled`, nunca mezclada sin distinción con fases donde el reloj sí llegó a converger — de lo contrario el EDP agregado de GPU quedaría contaminado por transiciones incompletas sin que el resultado lo señale.
3. **Agrupar por `(device, phase_label_train, freq_level_id)`** y calcular, por grupo, el EDP agregado usando una estadística robusta (mediana, no media, para no dejar que una corrida con ruido de medición domine la decisión) de energía y tiempo.
4. **Fijar el nivel `REF` (frecuencia nativa) como baseline de cada `(device, phase_label_train)`.** Para cada nivel de frecuencia distinto de REF, calcular la diferencia relativa de EDP contra ese baseline.
5. **Seleccionar el nivel que minimiza el EDP agregado**, pero solo si la mejora es estadísticamente defendible: correr una prueba pareada por kernel (Wilcoxon, o bootstrap sobre las repeticiones) entre el EDP a REF y el EDP al nivel candidato. Si no hay significancia, o si ningún nivel mejora sobre REF, la política resultante para esa clase es explícitamente `"no actuar"` — se documenta igual que una elección de frecuencia, no se omite la entrada. **Para GPU, si tras el filtro del paso 2 una clase queda dominada por fases excluidas (pocas fases sobrevivientes con señal confiable), la política resultante es también `"no actuar"`, documentando que la causa es de viabilidad temporal (duración de fase insuficiente) y no de rango de potencia — son dos motivos distintos para la misma conclusión y deben quedar distinguidos en el reporte.**
6. **Serializar la tabla resultante** (4 entradas: `cpu-compute_bound`, `cpu-memory_bound`, `gpu-compute_bound`, `gpu-memory_bound`) en un archivo de configuración simple y versionado (YAML o JSON), incluyendo por entrada: el nivel elegido, el `campaign_id` de origen, el tamaño de muestra usado, el intervalo de confianza de la mejora de EDP y, para GPU, el motivo de "no actuar" cuando aplique. Este archivo es el que el daemon carga en `§4.3`.
7. **Nunca hardcodear esta tabla en el código del daemon.** Si se repite la campaña de barrido (cambio de hardware, nueva versión del catálogo de kernels), el archivo se regenera corriendo el script de nuevo — el daemon no cambia.

---

## 4. Fase 3 — Daemon de control en espacio de usuario

### 4.1 Arquitectura: dos loops desacoplados, coordinados por una señal compartida

🔶 **Desviación del plan original:** el plan (§5.3) describe un único bucle iterativo. La arquitectura real necesaria son **dos loops dentro del mismo proceso**, porque CPU y GPU tienen sensores, cadencias de muestreo y costos de transición de frecuencia radicalmente distintos. Ver §7 para la justificación completa y cómo formalizar esta desviación frente al plan aprobado.

#### Loop CPU
- Cadencia fija ≈ 1 ms (misma granularidad de la Fase 1).
- En cada ventana: lee `perf`/RAPL, arma el vector de features de CPU, corre `f_cpu`, consulta la tabla de política (§3.4), aplica el nivel de frecuencia correspondiente vía gobernador `userspace` (`scaling_min_freq`/`scaling_max_freq`).

#### Loop GPU
- No corre por tiempo fijo — corre **por fase**, delimitada por los puntos de lanzamiento de kernel.
- ⚠️ **Hueco de diseño resuelto aquí:** `on_phase_begin`/`on_phase_end` de `gpu_clock_controller.hpp` reciben la etiqueta ya decidida y solo administran histéresis/permanencia mínima — **no detectan por sí solos cuándo empieza una fase**. ✅ **Verificado con `grep` sobre todo el repositorio (auditoría exclusiva de código):** la clase `GpuClockController` tiene exactamente dos referencias en todo el árbol de código — su propia definición en `gpu_clock_controller.hpp` y su test unitario `test_gpu_clock_controller.cpp`. **Cero callers de producción.** Tampoco existe, en ningún `.cpp` de `orchestrator/` o `telemetry/` fuera de tests/benchmarks/experiments, ningún `int main()` que combine lectura de `collector.hpp` en vivo + `model.predict()` + `freqctl.apply_frequency()` en el mismo loop (`predict(` no aparece en ningún `.py`/`.cpp` de producción de ninguna de las dos ramas). Esto confirma con evidencia directa de código, no solo por ausencia documental, que el daemon de este apartado no tiene ni un esqueleto parcial en ningún lado — ni siquiera en `fase-02`. El catálogo de GPU usa binarios de terceros (Rodinia), no código propio, y el objetivo 3 exige operar sin modificar el binario, así que no se puede insertar la llamada directamente en el código fuente del kernel. El disparo real se hace **interceptando `cudaLaunchKernel`/`cudaDeviceSynchronize` vía el mismo mecanismo `LD_PRELOAD`** que ya usa `blocking_sync_shim.cpp` (§4.1, Señal de coordinación) — se extiende ese shim para que, además de forzar `cudaDeviceScheduleBlockingSync`, marque el inicio de una fase en cada llamada a `cudaLaunchKernel` y su fin en el `cudaDeviceSynchronize`/`cudaStreamSynchronize` correspondiente, comunicando esos eventos al loop de GPU del daemon (p. ej. vía una cola o socket local liviano, dado que el shim vive inyectado en el proceso del binario de terceros y el daemon corre aparte). Esta decisión debe validarse con una prueba dirigida (lanzar un kernel real, confirmar que el daemon recibe exactamente un evento de inicio y uno de fin por lanzamiento) antes de integrarla al loop completo.

  🔴 **Corrección posterior, con evidencia real (no en el texto original de este párrafo):** esa prueba dirigida se hizo, con CUDA real, y **la intercepción de `cudaLaunchKernel` vía `LD_PRELOAD` no funciona** para la sintaxis estándar `<<<>>>` en ningún modo de enlace de cudart (`nvcc` resuelve esa llamada en tiempo de compilación, nunca a través de la tabla de símbolos dinámicos que `LD_PRELOAD` puede alterar — confirmado con `nm -D` y una build de depuración). El mecanismo elegido en su lugar es sondeo de `gpu_util_pct` desde el propio daemon (`fase3_daemon/gpu_loop/activity_poller.py`), sin instrumentar el binario objetivo — ver `fase3_daemon/README.md` para el hallazgo completo y las dos alternativas (intercepción a nivel de driver CUDA, `CUDA_INJECTION64_PATH`/CUPTI) documentadas como trabajo futuro, no descartadas.
- Al iniciar una fase (evento recibido del shim extendido): lee NVML, arma el vector de features de GPU, corre `f_gpu`, consulta la tabla de política, aplica el reloj de GPU una sola vez para toda la fase vía `nvidia-smi -lgc`.
- Incluye banda de histéresis y `min_dwell_ns` (permanencia mínima antes de permitir un nuevo cambio) para no oscilar en fases muy cortas — este valor se fija con el `T_transición_gpu` medido en §2.4.1, nunca arbitrariamente.

#### Señal de coordinación
- Problema real a resolver primero: `cudaDeviceSynchronize()` hace spin-wait por defecto, lo que hace ver a la CPU como "ocupada"/`compute_bound` aunque solo esté esperando a la GPU sin hacer trabajo útil.
- Solución de raíz: shim `LD_PRELOAD` que fuerza `cudaDeviceScheduleBlockingSync` en los binarios GPU de terceros, sin modificar su código fuente (respeta la restricción de "sin intervenir el binario", igual que ya se aplica a los kernels de CPU de NPB).
- Medida defensiva adicional, de bajo costo: mientras `gpu_util_pct` reporte actividad, forzar el reloj de CPU al mínimo, independientemente de lo que diga `f_cpu` en ese instante — si la CPU está de verdad bloqueada esperando, bajar su reloj casi no afecta el consumo porque no hay conmutación que escale con la frecuencia.

### 4.2 No negociables de implementación (aplican a ambos loops)

1. **Nunca escribir fuera de los CPUs/GPU delegados a la campaña o al daemon** — el nodo puede ser compartido con otros usuarios.
2. **Toda escritura de frecuencia se verifica por relectura**, nunca se asume éxito por el código de retorno de la llamada.
3. **Restauración obligatoria e idempotente** del estado original del hardware, registrada en `atexit`, `SIGINT` y `SIGTERM` — probada explícitamente con una prueba de caos real (interrumpir a mitad de ejecución y confirmar por lectura de sysfs/NVML que todo volvió al estado previo).
4. Medir y registrar el **overhead del propio daemon** (tiempo de inferencia + tiempo de actuación) en cada ciclo — es un insumo directo del objetivo 4, no un detalle de implementación.

### 4.3 Cómo construir el daemon, paso a paso

Esta pieza tampoco existe todavía. A diferencia del derivador de política (§3.5), el daemon corre en producción y su presupuesto de latencia es una restricción dura — especialmente en el loop de CPU, que debe decidir y actuar dentro de la ventana de ~1 ms sin volverse él mismo la fuente de overhead que el objetivo 2 pide evitar.

1. **Definir explícitamente el alcance de monitoreo/actuación del daemon antes de escribir el loop.** El objetivo 3 exige que el daemon opere sobre aplicaciones en ejecución en general, no solo sobre el catálogo de benchmarks de la campaña — el documento no puede dejar esto implícito. Dos mecanismos posibles, consistentes con el no-negociable de §4.2 ("nunca escribir fuera de los CPUs/GPU delegados"): (a) el daemon se lanza acotado a un **cpuset/cgroup delegado** (el mismo mecanismo que ya usa `campaign.py` para las corridas de Fase 1) y monitorea/actúa sobre todo proceso que corra dentro de ese cpuset, sin necesidad de conocer su PID de antemano; o (b) el daemon **se adjunta a un PID objetivo** específico, pasado como argumento al arrancar. Elegir (a) por defecto para el diseño del daemon (más cercano al caso de uso real de un nodo HPC compartido, y no requiere que el usuario conozca el PID de antemano), documentando (b) como modo alternativo para pruebas dirigidas contra un solo binario del catálogo. ⚠️ **Consistencia con el shim extendido de §4.1:** elegir (a) implica que el shim `LD_PRELOAD` (necesario para el disparo de fase en GPU) debe quedar activo automáticamente para todo proceso lanzado dentro de ese cpuset/cgroup — vía la variable `LD_PRELOAD` propagada en el entorno del job de Slurm que delega el cpuset, no como un paso manual por aplicación. Sin esto, el mecanismo de §4.1 solo funcionaría en el modo (b), no en el (a) elegido por defecto — hay que verificarlo explícitamente al probar el daemon, no asumirlo.
2. **Decisión de lenguaje por loop, no uniforme para todo el daemon.** El loop de GPU (cadencia de fases, no de milisegundos) puede implementarse en Python, reutilizando `orchestrator/gpu_freqctl.py` y el patrón de `gpu_clock_controller.hpp` como referencia de diseño. El loop de CPU, en cambio, debe implementarse en C++ dentro del propio harness de `telemetry/` (extendiendo `collector.hpp`), porque el overhead de un intérprete Python en cada ciclo de 1 ms puede por sí solo violar la restricción de "no degradar el rendimiento global" que pide la pregunta de investigación. Documentar esta decisión explícitamente como parte del diseño del daemon, no dejarla implícita.
3. **Exportar los modelos entrenados en §3 a un formato de inferencia liviano invocable desde C++** (por ejemplo, convertir el árbol/Random Forest elegido a código C++ generado automáticamente, o usar ONNX Runtime en su build C++) para el clasificador de CPU. El clasificador de GPU puede quedarse en su formato nativo (scikit-learn/XGBoost) porque corre desde el loop en Python.
4. **Cargar la tabla de política de §3.5** (el archivo YAML/JSON versionado) al arrancar el daemon — nunca recalcularla en línea.
5. **Loop de CPU (C++, dentro de `telemetry/`):** en cada tick de ~1 ms, leer el snapshot ya recolectado por `collector.hpp` (`perf`/`uncore`/RAPL), construir el vector de features de §2.5, correr la inferencia exportada, consultar la tabla de política, y **solo si la clase cambió respecto al tick anterior**, invocar la actuación de frecuencia (reutilizando la lógica de verificación por relectura de `freqctl.py`, pero implementada nativamente para no cruzar la frontera Python↔C++ en el camino caliente).
6. **Loop de GPU (Python):** consume los eventos de inicio/fin de fase que emite la extensión del shim `LD_PRELOAD` descrita en §4.1 (intercepción de `cudaLaunchKernel`/`cudaDeviceSynchronize`) — no invoca `on_phase_begin`/`on_phase_end` directamente desde el código del kernel, porque el binario es de terceros. Al recibir un evento de inicio de fase: lee NVML, construye el vector de features de GPU, corre la inferencia, consulta la tabla de política, aplica `nvidia-smi -lgc` solo si la clase cambió y ya se cumplió el `min_dwell_ns`.
7. **Señal de coordinación:** una variable atómica compartida entre ambos loops (viven en el mismo proceso, así que puede ser un flag simple con las garantías de atomicidad del lenguaje/runtime que se use). El loop de GPU la activa mientras `gpu_util_pct` esté por encima de un umbral de ruido; el loop de CPU la consulta antes de aplicar su propia decisión y, si está activa, fuerza el piso de frecuencia sin importar lo que haya dicho `f_cpu` en ese ciclo.
8. **Manejo de señales y restauración**, reutilizando el patrón ya probado de `campaign.py::install_emergency_handlers()`: un único cierre combinado que restaura CPU y GPU juntos, registrado una sola vez en `atexit`/`SIGINT`/`SIGTERM` (dos registros separados pisarían el manejador del otro).
9. **Modo `dry-run` primero:** el daemon corre completo (clasifica, consulta la tabla, decide) pero solo registra en log qué habría hecho, sin escribir frecuencia real. Validar este modo contra corridas ya grabadas de la campaña de Fase 1 antes de tocar hardware real — permite verificar que las decisiones del daemon coinciden con lo esperado sin arriesgar una corrida en el clúster compartido.
10. **Instrumentar el propio daemon con logging estructurado** (features leídas, clase inferida, frecuencia aplicada, tiempo de inferencia, tiempo de actuación) por cada decisión — este log es el insumo directo de la columna "overhead del agente" que pide la Fase 4.
11. **Solo después de validar en `dry-run`**, habilitar la escritura real de frecuencia y correr contra el catálogo completo — este es el punto en el que el daemon pasa de pieza en construcción a estar listo para la Fase 4.

---

## 5. Fase 4 — Validación experimental

### 5.1 Escenarios de comparación

1. Gobernador nativo de Linux (`ondemand`/`schedutil`, o `powersave` bajo `intel_pstate` según lo que exponga la plataforma real).
2. Frecuencia fija de alto rendimiento (`performance`).
3. El agente propuesto (dos clasificadores + daemon de dos loops).

⚠️ **Hueco de código verificado, no solo un paso pendiente genérico.** `orchestrator/freqctl.py` ya tiene un modo `native_governor` (función `_apply_native_governor()`) — pero eso significa "dejar el CPU con el `scaling_governor` que el nodo ya tenía configurado" (que, por la propia lógica de restauración del código, hoy es `performance`), **no** un mecanismo para conmutar explícitamente hacia `ondemand` o `schedutil` y ejecutar el catálogo bajo cada uno. Se buscaron las cadenas literales `"ondemand"` y `"schedutil"` en todo `orchestrator/` y en los 40 archivos de `classifier/` de `fase-02`: **cero apariciones en cualquiera de las dos ramas.** El escenario 1 de esta lista, tal como está escrito ("gobernador nativo... `ondemand`/`schedutil`"), **no tiene ningún soporte de código hoy** — hace falta añadir, no reutilizar, la capacidad de escribir `scaling_governor` a un valor distinto de `userspace`/`performance` y ejecutar una campaña completa bajo ese gobernador, con la misma disciplina de verificación por relectura que ya usa el resto de `freqctl.py`.

### 5.2 Métricas y protocolo

- Por cada escenario y cada kernel del catálogo: tiempo total de ejecución, energía CPU (RAPL) y GPU (NVML), potencia media, EDP, y overhead del agente (solo aplica al escenario 3).
- Repeticiones suficientes para análisis estadístico (mínimo 3, idealmente 5+ si el presupuesto de cómputo lo permite).
- Prueba de hipótesis apropiada según distribución de los datos (Wilcoxon/Mann-Whitney si no se puede asumir normalidad, t-test pareado si sí) para determinar si la diferencia de EDP entre el agente y cada baseline es estadísticamente defendible, no solo una diferencia numérica.

⚠️ **Hueco de código verificado: este módulo no existe todavía en ninguna forma, en ninguna rama.** Se buscó `scipy.stats`, `wilcoxon`, `mannwhitneyu`, `ttest_rel`, `ttest_ind` en todo `orchestrator/`, `telemetry/` y los 40 archivos de `classifier/` de `fase-02`: **cero resultados.** `classifier/eval/protocol.py` sí calcula razones de EDP (`edp_loss()`, `trivial_baselines()`, `honest_constant_baseline()`, ver §3.4/§3.5) pero ninguna de esas funciones produce un p-valor ni un intervalo de confianza — son comparaciones de magnitud, no pruebas de significancia. Este punto del protocolo necesita un módulo nuevo (p. ej. `classifier/analysis/statistical_tests.py` o una extensión de `protocol.py`), no una adaptación de algo existente.

- Reportar el resultado **incluso si es negativo o mixto** (p. ej. "el agente mejora el EDP en fases memory-bound de GPU pero no en CPU, consistente con el hallazgo de rango dinámico de potencia de §3.4") — es un resultado científicamente válido y defendible del objetivo 4, y de hecho más interesante que un resultado uniformemente positivo sin matices.

---

## 6. Mapeo módulo por módulo: qué conservar, ajustar, completar o aislar

Para no partir de cero: gran parte de la infraestructura de Fase 1 ya construida en `hyperion` es directamente reutilizable bajo este plan, porque la telemetría en sí (perf/RAPL/NVML/uncore) no cambia — lo que cambia es qué se hace con ella después. Esta tabla es el punto de partida operativo para reestructurar el código.

⚠️ **Nota de procedencia, verificada con `git ls-tree`/`git show`, no con documentación:** todo lo que en esta tabla dice `classifier/*` vive **únicamente en la rama remota `origin/fase-02`**, no en `main`. La columna "Acción" asume que la fusión de catálogo de §2.1.1 y el porteo de código de `classifier/` hacia `main` ya se decidieron (confirmado con el autor) — hasta que ese porteo ocurra, ninguna de estas rutas existe en el árbol de trabajo de `main`.

| Módulo del repo | Rama | Estado frente a este documento (verificado por lectura directa de código) | Acción |
|---|---|---|---|
| `telemetry/` (harness C++: `perf_reader`, `uncore_reader`, `nvml_reader`, `rapl_reader`, `collector`) | `main` | Cumple §2.2 casi sin cambios — confirmado: eventos `PERF_COUNT_HW_*`/raw, `energy_uj` de RAPL pkg+dram, llamadas NVML reales (`GetPowerUsage`/`GetUtilizationRates`/`GetClockInfo`/`GetTotalEnergyConsumption`/`GetTemperature`), `uncore_imc` CAS_COUNT. `Collector::run()` ya tiene un loop continuo real (`while(!stop_flag_)` con ring buffer), reutilizable como base del loop CPU del daemon (§4.3) | **Conservar** |
| `orchestrator/calibration.py` | `main` | Cumple §2.3 (Roofline, ridge por dispositivo/precisión) | **Conservar** |
| `orchestrator/postprocess.py` (`phase_label_train`, ventanas) | `main` | Genera `windows.csv` con las columnas de §2.2, pero mezcla columnas redundantes entre sí (§2.5) | **Ajustar**: no quitar columnas crudas (siguen siendo evidencia de auditoría), pero añadir un paso explícito de selección de features hacia el dataset de entrenamiento |
| `orchestrator/freqctl.py` / `gpu_freqctl.py` | `main` | Actuación con verificación por relectura y restauración — cumple §4.2. Tiene un modo `native_governor` (`_apply_native_governor()`), pero eso es "dejar el gobernador que el nodo ya tenía" (`performance`, verificado), no conmutar hacia `ondemand`/`schedutil` — cero apariciones de esas dos cadenas en todo el archivo | **Conservar el mecanismo de pin/relectura; extender** con conmutación explícita de gobernador para §5.1 |
| `telemetry/.../gpu_clock_controller.hpp` | `main` | Base del loop GPU por fase (§4.1) — histéresis/dwell ya implementados. **Verificado con `grep` en todo el repo: cero callers de producción**, las únicas dos referencias a `GpuClockController` en todo el árbol son su propia definición y su test unitario | **Completar**: falta el wiring real hacia un daemon vivo, no existe ni un esqueleto parcial |
| `orchestrator/native/blocking_sync_shim.cpp` | `main` | Shim necesario para §4.1 — existe, fuerza `cudaDeviceScheduleBlockingSync`, sigue vigente sin cambios. 🔴 **La extensión para interceptar `cudaLaunchKernel` SE INTENTÓ y se retiró**: verificado con CUDA real que esa intercepción no se dispara para la sintaxis `<<<>>>` en ningún modo de enlace de cudart — ver `fase3_daemon/README.md` | **Conservar tal cual** (el mecanismo base sigue siendo correcto); la detección de fase de §4.1 se resolvió por sondeo de `gpu_util_pct` (`fase3_daemon/gpu_loop/activity_poller.py`), no extendiendo este archivo |
| `classifier/training/train_phase.py` (205 líneas) | `fase-02` | Clasificador de fase — **ya funcional, no un borrador**: entrena Dummy/árbol prof1/logreg/árbol prof6/RandomForest/ExtraTrees, valida con `leave_one_kernel_out` (agrupado por `kernel_ref`, no por familia todavía), mide latencia p50/p99 por predicción aislada, bloquea por `SystemExit` cualquier feature de `FORBIDDEN` (fuga de etiqueta) de forma más estricta que "evaluar ambas variantes". Usa 9 kernels reales (confirmado en el propio código, §2.1.1). **No incluye XGBoost. No serializa el modelo elegido en ningún punto del archivo** (sin `joblib`/`pickle`/ONNX en todo `classifier/training/`) | **Reactivar como base, no tal cual**: añadir XGBoost, añadir serialización, cambiar `kernel_ref` por familia algorítmica una vez fusionado el catálogo ampliado (§2.1.1), medir p95 además de p99 |
| `classifier/eval/protocol.py` | `fase-02` | **Ya implementa el núcleo de cálculo de §3.4/§3.5**: `edp_loss()`, `trivial_baselines()`, `honest_constant_baseline()`, `leave_one_kernel_out()`, `fold_summary()` — no es "no existe todavía", es lógica real ya usada por `train_phase.py`. No tiene ninguna prueba de significancia estadística (`scipy.stats` no aparece en el archivo ni en ningún otro de `classifier/`) | **Extender**, no construir desde cero: envolver en el script de §3.5, añadir filtro `T_transición_gpu` y pruebas de hipótesis (§5.2) |
| `classifier/selector/` (17 archivos), `docs/general/*selector*` | `fase-02` | El "selector de dispositivo" — fuera de alcance según §7.2 de este documento. Confirmado por `classifier/README.md`: *"El objetivo vigente es puntuar las configuraciones candidatas... y seleccionar el dispositivo y las frecuencias"* — es decir, es el foco activo de esa rama, no un experimento secundario | **Aislar**: no construir más sobre esto, no borrar |
| `classifier/analysis/` (23 scripts) | `fase-02` | Son scripts de EDA **específicos del selector** (roofline policy, calibración alpha de GPU, headroom CPU/GPU) — no utilidades genéricas de correlación/VIF reutilizables tal cual para §2.5 | **No asumir reutilización directa**: revisar cuáles de los 23 aplican al clasificador de fase (no al selector) antes de apoyarse en ellos |
| *(no existe en ninguna rama)* | — | Daemon vivo con dos loops + señal de coordinación (§4.1) — confirmado: cero `int main()` fuera de tests/benchmarks/experiments en todo el repo | **Construir desde cero** |
| *(no existe en ninguna rama)* | — | Conmutación explícita de gobernador nativo (`ondemand`/`schedutil`) para el escenario 1 de §5.1 — confirmado: cero apariciones de esas cadenas en `orchestrator/`+`classifier/` | **Construir desde cero** en `freqctl.py` |
| *(no existe en ninguna rama)* | — | Módulo de pruebas de significancia estadística (Wilcoxon/Mann-Whitney/t-test pareado) para §5.2 — confirmado: cero uso de `scipy.stats` en todo el repositorio | **Construir desde cero** |
| *(no existe en ninguna rama)* | — | Serialización del modelo de fase entrenado (§3.3 punto 6) — confirmado: la única serialización existente en `classifier/` es la del selector, fuera de alcance | **Construir desde cero**, dentro de `classifier/training/` |
| *(no existe en ninguna rama)* | — | Función `kernel_ref → familia_algorítmica` y `leave_one_familia_out()` equivalente a `leave_one_kernel_out()` (§2.1.1/§2.6/§3.3) | **Construir desde cero**, en `classifier/eval/protocol.py` |
| *(no existe todavía)* | — | Paso de análisis de correlación/VIF sobre el dataset antes de fijar columnas (§2.5) | **Construir desde cero**, revisando primero cuáles scripts de `classifier/analysis/` son adaptables |

Las piezas que más bloquean avanzar, en orden de dependencia: (1) la fusión del catálogo (§2.1.1, bloquea todo lo demás), (2) el porteo de `classifier/` desde `fase-02`, (3) el daemon vivo y (4) el derivador de la tabla de política — estas dos últimas ya tienen bastante lógica de cálculo reutilizable (`protocol.py`, `gpu_clock_controller.hpp`), pero cero wiring de producción en ningún lado. El resto del trabajo pendiente es reactivar/ajustar componentes ya maduros, más las cuatro piezas de código nuevo que la auditoría de §5.1/§5.2/§3.2 confirmó que no existen en ninguna rama (conmutación de gobernador, pruebas estadísticas, serialización de modelo, agrupación por familia). Los pasos concretos para construir el daemon y el derivador de política están detallados en §3.5 y §4.3.

---

## 7. Justificación completa de las desviaciones frente al plan aprobado

Esta sección deja trazabilidad explícita de en qué puntos la implementación necesaria se aparta de la letra del plan aprobado, y por qué cada punto sigue sirviendo a los mismos cuatro objetivos.

### 7.1 Desviaciones que este plan SÍ incorpora (necesarias, no cambian el objetivo de fondo)

| Desviación | Objetivo afectado | Justificación técnica | Por qué no cambia el objetivo en el fondo |
|---|---|---|---|
| Dos clasificadores independientes (uno por dispositivo) en vez de "un modelo" | Obj. 2 | CPU y GPU no comparten sensores compatibles (PMU vs. NVML); un modelo único requeriría normalización artificial que fragiliza el sistema sin necesidad | La salida sigue siendo binaria compute/memory-bound por dispositivo, tal como pide el objetivo |
| Daemon de dos loops desacoplados + señal de coordinación, en vez de un bucle único | Obj. 3 | Costos de transición y cadencias de muestreo radicalmente distintos entre CPU (~1 ms) y GPU (fases completas); forzar un solo ritmo desperdicia la ventaja de uno u obliga al otro a decisiones mal informadas | La actuación sigue siendo únicamente sobre frecuencia, vía las mismas interfaces (`cpupower`/`nvidia-smi`) que pide el objetivo |
| Barrido de frecuencia explícito en Fase 1 para derivar la tabla clase→frecuencia óptima | Obj. 1 y 3 | El plan no especifica cómo se deriva "la configuración de frecuencia más adecuada" que el daemon debe aplicar; sin este paso, el objetivo 3 no es ejecutable | Es una precisión metodológica que llena un vacío del plan, no una ampliación de alcance |
| Análisis de correlación/VIF sobre las features del dataset | Obj. 1 y 2 | El plan no exige explícitamente evitar redundancia entre features; sin este control, el modelo es menos interpretable y el capítulo de resultados es más débil frente al comité | Es rigor metodológico adicional, no cambia qué se mide ni qué se predice |

### 7.2 Desviación que este plan explícitamente NO incorpora (recomendación: descartar o separar como trabajo futuro)

**El "selector de dispositivo" (`classifier/selector/`, decidir si una fase corre en CPU o en GPU) no forma parte de este plan.** Razones:

1. **No está en los objetivos aprobados.** El objetivo 3 dice literalmente "aplicando políticas proactivas de DVFS... en función de la fase de ejecución inferida" — la variable de control es la frecuencia, el dispositivo de ejecución es un dato de entrada, no una salida del sistema. El objetivo 2 pide una etiqueta binaria de fase, no una decisión de enrutamiento entre dispositivos.
2. **El alcance del plan (§6.1) fija el proyecto como "gestión intra-nodo de frecuencia"**, no como un planificador de tareas que decide dónde ejecutarlas.
3. **La justificación empírica que motivó el selector es real y valiosa** (en la CPU de la plataforma experimental, DVFS por sí solo casi nunca mejora el EDP porque el rango dinámico de potencia es angosto: -28% de potencia por -75% de reloj, con el tiempo alargándose 2.2–4×) — pero esa evidencia es exactamente el tipo de resultado que corresponde al **objetivo 4** (evaluar empíricamente si el DVFS solo, tal como está definido en el plan, logra o no mejorar el EDP). Es un resultado legítimo y publicable dentro del alcance original, no una razón para ampliar el sistema a decidir dispositivo.
4. **Recomendación concreta:** el trabajo ya invertido en el selector (dataset cold/warm, `metodologia_selector_cpu_gpu_20260827.md`, el código de `classifier/selector/`) no se pierde — se documenta en el capítulo de "Trabajo futuro" de la tesis como una extensión natural motivada por los hallazgos del objetivo 4, explícitamente fuera del alcance evaluado en este trabajo de grado. Esto es honesto académicamente, aprovecha el esfuerzo ya hecho, y evita que el proyecto se evalúe contra un objetivo que nunca fue aprobado por el comité.

### 7.3 Cómo tratar el trabajo ya hecho en el selector de dispositivo

El código de `classifier/selector/` y los documentos `metodologia_selector_cpu_gpu_20260827.md`/`plan_reformulacion_selector_tamanos_20260830.md` no se descartan ni se borran — se aíslan de la ruta principal de desarrollo (por ejemplo, dejándolos donde están pero dejando de construir sobre ellos) y se referencian en el documento final como trabajo futuro motivado por el hallazgo empírico del objetivo 4. Esto evita perder el esfuerzo ya invertido sin dejar que siga absorbiendo tiempo de las Fases 2–4 que sí están dentro del alcance evaluado.

---

## 8. Checklist de cierre por fase (para usar como criterio de aceptación)

- [ ] **Prerrequisitos de plataforma (§2.0):** escritura de frecuencia de CPU verificada bajo carga real dentro de asignación exclusiva (Turbo Boost no anula el candado); restricción de reloj de GPU verificada bajo carga real sostenida (no solo en reposo) — ambos con evidencia registrada, no asumidos.
- [ ] **Fase 1:** catálogo ampliado con las familias algorítmicas de §2.1.1, verificado ≥5–6 familias por clase y por dispositivo, con checksum verificado; campaña CPU y GPU completas con verificación por relectura bajo carga en todos los niveles de frecuencia; `T_transición_gpu` medido bajo carga real por par de niveles (§2.4.1), nunca asumido de la ficha técnica; dataset con etiquetas derivadas exclusivamente de intensidad real (uncore/ncu), nunca de proxy; matriz de correlación y VIF documentados con las columnas finales justificadas.
- [ ] **Fase 2:** catálogo de `fase-02` fusionado y verificado campo a campo contra las 23 entradas de `main` (§2.1.1); `classifier/training/train_phase.py` porteado a `main` con XGBoost añadido, serialización de modelo añadida, `leave_one_kernel_out` sustituido por `leave_one_familia_out` (función nueva, no existe en ninguna rama), y latencia p95 añadida junto a p99; tabla clase→frecuencia óptima derivada del barrido de Fase 1 extendiendo `classifier/eval/protocol.py` (no reescribiéndolo), incluyendo explícitamente el caso "no actuar" donde aplique — con el hallazgo ya calculado en código (`trivial_baselines()`: máxima frecuencia óptima en 9/9 kernels CPU) tratado como señal a confirmar, no como resultado final.
- [ ] **Fase 3:** daemon con los dos loops funcionando de forma independiente y verificada por separado — partiendo de que hoy no existe ni un `int main()` parcial en ninguna rama, ni callers de producción de `GpuClockController`; mecanismo de detección de fase de GPU (shim `LD_PRELOAD` extendido sobre `cudaLaunchKernel`, §4.1) probado de punta a punta contra un kernel real de terceros; señal de coordinación CPU-GPU probada; restauración de estado verificada con prueba de caos real; overhead del daemon medido y registrado.
- [ ] **Fase 4:** conmutación real de gobernador (`ondemand`/`schedutil`) construida en `freqctl.py` — hoy `native_governor` solo deja el gobernador que el nodo ya tenía (`performance`), no hay código que pruebe los otros dos; módulo de pruebas de significancia estadística construido (`scipy.stats`, cero uso hoy en todo el repositorio); los tres escenarios de comparación ejecutados sobre el catálogo completo; resultado reportado con matices (por clase, por dispositivo), no como un único número agregado.
- [ ] **Documentación:** capítulo de Objetivos y Metodología del documento final actualizados para reflejar §2–§5 de este plan; selector de dispositivo documentado como trabajo futuro, no como parte del sistema evaluado; la procedencia del catálogo ampliado (fusionado desde `fase-02`, §2.1.1) documentada explícitamente en metodología, no presentada como si siempre hubiera vivido en `main`; cualquier limitación de tamaño de catálogo o bloqueo de plataforma no resuelto a tiempo, reportada explícitamente.
