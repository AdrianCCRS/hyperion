# Plan de ejecución — Fase 3 (daemon de control DVFS)

Complementa `Plan_Detallado_Realineacion_Hyperion.md` §4. Donde este
documento y aquel discrepen, **manda este**: el plan de realineación se
escribió antes de tener los resultados de Fase 2, y dos de sus supuestos
quedaron refutados por medición (§0.2 y §0.3 de abajo).

Estado de partida (2026-09-22): `fase3_daemon/` tiene 31 pruebas Python y
1 de C++ en verde, con el loop de GPU completo, el actuador, la máquina de
decisión de CPU y `run_daemon.py` operando en `--dry-run`. El plan de
realineación afirma que esta pieza "no tiene ni un esqueleto parcial";
eso ya no es cierto y no debe usarse para dimensionar el trabajo restante.

---

## 0. Decisiones de diseño (Bloque 0)

### 0.1 Qué hace el daemon dado que la política de CPU es "no actuar"

**Contexto.** La tabla de política de CPU resultó *no actuar* en ambas
clases, y la de GPU *actuar en `memory_bound` a F1* (+8.9% de EDP) pero
con el candado de reloj averiado. Un daemon que solo traduzca "clase →
nivel" sería, hoy, un no-op en CPU.

**Decisión.** El daemon no se escribe para ejecutar la tabla de política y
ya; se escribe para **poder ejecutar el experimento de Fase 4**, que es lo
que da valor a la tabla (incluso a una tabla que dice "no actuar"). El
experimento tiene dos ejes.

**Eje 1 — la carga.** La tabla de política se derivó sobre corridas de un
solo kernel, donde ningún nivel *fijo* gana sobre REF. Una aplicación HPC
compuesta, que alterna fases de distinto régimen, es una pregunta
distinta y es justamente la premisa de la tesis: un agente que conmuta por
fase puede ganarle a cualquier nivel fijo aunque ningún nivel fijo le gane
a REF. Se construyen a mano aplicaciones compuestas de varios kernels:

- **A. Compuesta con familias conocidas** — kernels que sí estuvieron en
  el entrenamiento del clasificador.
- **B. Compuesta con familias inéditas** — kernels que nunca entraron en
  ninguna decisión de ajuste. Es el equivalente, a nivel de aplicación,
  del protocolo *leave-one-familia-out* con el que se validó el modelo: si
  A gana y B no, el resultado es memorización, no generalización.

**Eje 2 — el brazo de control.** Tres brazos, no dos, porque "sin actuar"
es ambiguo y esconde dos efectos distintos:

| Brazo | Daemon | Escribe frecuencia | Qué aísla |
|---|---|---|---|
| **base** | apagado | no | Línea base pura: gobernador nativo |
| **sombra** | encendido | no | **Costo** del agente: clasifica y decide en cada ciclo, sin actuar. `sombra − base` = sobrecarga de inferencia (objetivo 2) |
| **activo** | encendido | sí | Efecto total. `activo − sombra` = **beneficio** de la actuación, ya descontada su propia sobrecarga |

Sin el brazo *sombra*, cualquier ganancia de *activo* frente a *base*
mezcla el beneficio de actuar con el costo de inferir, y el objetivo 2
(que la sobrecarga no anule el beneficio) no se puede responder por
separado. Es una corrida más por aplicación, y es barata.

**Expectativa declarada de antemano, para no ajustarla después.** En CPU,
el piso de potencia medido (85 W, 84% del total a 3.2 GHz) acota el
oráculo por kernel a 2.2% de ganancia media. Una aplicación compuesta
puede superar a cualquier nivel fijo, pero no puede escapar de ese piso:
**no se espera una ganancia grande en CPU**, y un resultado plano es un
hallazgo válido, no un fallo del daemon. Lo que sí debe quedar medido es
la sobrecarga (brazo *sombra*) y el comportamiento de conmutación. En GPU
la expectativa es la contraria, porque ahí sí hay margen medido.

**Requisitos que esto impone al daemon** (todos verificables sin cluster):

1. **Brazo como parámetro de primera clase**, no un `--dry-run` de
   conveniencia: en el brazo *sombra* el daemon debe ejecutar exactamente
   el mismo trabajo que en *activo* (leer, construir el vector, inferir,
   consultar la tabla, decidir) y detenerse justo antes de la escritura.
   Si *sombra* saltara la inferencia, no mediría nada útil.
2. **Registro estructurado por decisión, idéntico en los tres brazos**:
   marca de tiempo, variables leídas, clase inferida, confianza, decisión
   de la tabla, si se escribió, tiempo de inferencia y tiempo de
   actuación. Es el insumo directo de la Fase 4 y del objetivo 4.
3. **Fronteras de fase conocidas en las aplicaciones compuestas.** Al
   construirlas nosotros, cada transición de kernel queda registrada con
   su marca de tiempo. Eso permite puntuar la **clasificación** contra una
   verdad conocida, no solo comparar EDP agregado — si el EDP no mejora,
   hay que poder distinguir "clasificó bien pero no había margen" de
   "clasificó mal".
4. **GPU lista aunque no se pueda probar.** Toda la ruta de GPU
   (detección de fase, inferencia, consulta de tabla, decisión,
   histéresis) se implementa y se valida en *sombra*, con la escritura del
   reloj como el único paso detrás de la bandera de brazo. Cuando H1 se
   repare, el brazo *activo* de GPU corre sin cambios de código.
5. **Restauración y caos** cubriendo los tres brazos (en *base* y
   *sombra* no hay nada que restaurar, y eso también debe verificarse:
   que el daemon no dejó rastro).

### 0.2 La medida defensiva de §4.1 queda refutada — CERRADA

§4.1 del plan de realineación prescribe: *"mientras `gpu_util_pct` reporte
actividad, forzar el reloj de CPU al mínimo"*, bajo el supuesto explícito
de que *"si la CPU está de verdad bloqueada esperando, bajar su reloj casi
no afecta el consumo"*.

Ese supuesto está **refutado por medición** (F1-XDEV-006, 2026-09-13):
fijar la CPU al mínimo durante una carga GPU la alarga **63-66%**
(134/134 y 1632/1632 pares más lentos en dos campañas), con la fase en
régimen estacionario casi duplicada (+93.8%) y una degradación que crece
con el tamaño del problema.

**Semántica corregida, ya implementada** en
`fase3_daemon/cpu_loop/include/cpu_phase_controller.hpp`: la señal de
coordinación pasa de ser un *objetivo* (bajar a él) a ser una *barrera*
(nunca pedir por debajo de él). Concretamente:

- Con la GPU activa, una petición de la política por debajo de
  `gpu_active_floor_khz` se **eleva** hasta ese valor.
- Si la política de la clase es "no actuar", no se escribe nada, ni
  siquiera con la GPU activa: la barrera solo puede elevar una petición
  que la política ya decidió hacer, nunca introduce una escritura propia.

El shim de blocking-sync (`common/hpc/native/blocking_sync_shim.cpp`)
sigue siendo la parte válida del mecanismo original: evita que una espera
bloqueante se lea como IPC alto ante el clasificador. Lo que se retira es
bajar el reloj por esa señal.

### 0.3 Un solo derivador de política, y que sea el correcto

Existen dos derivadores y **no coinciden**:

| | `fase3_daemon/policy/derive_policy_table.py` | `fase2_clasificador/analysis/{cpu,gpu}_policy_table.py` |
|---|---|---|
| Unidad de EDP | **ventana** (~1 ms) | **corrida completa** |
| IC95 por bootstrap | no | sí |
| Efecto mínimo relevante | no | sí |
| Des-duplicación por familia | no | sí (complemento por familia) |
| Resuelve la frecuencia física | sí (`resolved_*`) | **no** |
| Produjo los resultados del libro | no | sí |

El EDP por ventana es la unidad que el libro **rechaza explícitamente**:
con ventanas de duración fija, a frecuencia baja una ventana cubre menos
trabajo que a frecuencia alta, así que `E×Δt` a Δt fijo mide potencia, no
energía-retardo del trabajo. El derivador de Fase 3 produciría, en
silencio, una tabla distinta de la que sustenta el libro.

**Decisión.** Se retira `fase3_daemon/policy/derive_policy_table.py` como
derivador. La derivación vive en Fase 2 (como ya dice el libro en
§`sec:metodologia-politica-cpu`) y Fase 3 solo **carga y valida**.

**Dos defectos del artefacto canónico que hay que cerrar antes** (ambos
detectados al revisar el consumo real del daemon):

1. **`chosen_level` de GPU dice `F2`; el libro defiende `F1`.** El JSON
   sale del análisis por kernel, donde F2 aparece como el mejor nivel
   significativo; pero al des-duplicar los tres algoritmos medidos en dos
   tamaños (`dual_axpy`, `dual_spmv`, `dual_stencil`), F2 pierde
   significancia y F1 es el nivel que se sostiene. El artefacto debe
   reflejar la decisión por familia, que es la defendida.
2. **Falta `resolved_clock_mhz`/`resolved_freq_khz`.** Sin la frecuencia
   física resuelta, `build_controller_from_policy()` lanza `ValueError` y
   el daemon no puede actuar. La tabla debe ser autocontenida: el daemon
   no debe resolver un ID de nivel contra un manifiesto de campaña.

---

## 1. Bloques de trabajo

### Bloque A — Sin dependencia de hardware — CERRADO (2026-09-22)

- [x] **A1/A2.** `fase3_daemon/policy/build_policy_table.py` combina
  `policy_cpu.json` (kernel) con `policy_by_family.json` de GPU (familia,
  des-duplicado) y resuelve la frecuencia real (F1 → 1260 MHz). Verificado
  contra `build_controller_from_policy()` real. `derive_policy_table.py`
  retirado. Artefactos GPU comprometidos en
  `docs/libro/datos/gpu_calidad_20260922/politica/`.
- [x] **A3.** Candidato GPU vigente (regresión logística, 483 corridas, 16
  familias, 5 variables) exportado a `fase2_clasificador/models/`. El
  `.joblib` de 15-sep quedó retirado. Latencia: la cifra de `paccaA100` se
  preserva en la metadata con su procedencia documentada; no se remidió
  localmente (número sin sentido fuera del hardware de destino).
- [x] **A4.** `gpu_loop/classifier.py::HistoricalGpuClassifier` cablea el
  modelo real. Encontrado y resuelto en el camino: (a) el modelo se
  entrenó con mediana+std de múltiples muestras NVML por corrida, el loop
  solo daba una instantánea por evento — resuelto con un buffer móvil
  (`activity_poller` gana un `on_sample` aditivo), documentado como
  aproximación causal, no la definición exacta de entrenamiento; (b) el
  modelo predice booleano, no la string de la etiqueta.
- [x] **A5.** `fase3_daemon/cpu_loop/export_onnx.py`: ONNX verificado fila
  a fila contra las 1 169 833 filas reales de `tmp/cpu_quality_20260918/
  source.csv` (no una muestra): `max_diff_proba=2.98e-07`, 0 discrepancias.

49/49 tests de `fase3_daemon` en verde (36 previos del Bloque 0 + 13 nuevos).

#### Límite conocido de A4: dos capas de generalización sin validar

La tabla de política no está indexada por kernel (solo tiene 2 entradas,
`compute_bound`/`memory_bound`); que "funcione" con un kernel nunca visto
depende enteramente de que el **clasificador** generalice, no de que la
tabla lo conozca. Hay evidencia real de eso (LOFO 0.792, validación
externa sellada 0.779 sobre 4 familias nunca tocadas), pero esa evidencia
cubre solo una de dos capas necesarias, y de las otras dos no hay ninguna
medición todavía:

1. **Generalización del clasificador** (¿acierta la clase?) — medida:
   ~0.79-0.85 según cobertura. Evidencia fuerte.
2. **Generalización de la propia tabla** (¿la ganancia de F1 en
   `memory_bound` se sostiene en un kernel `memory_bound` nuevo?) — la
   validación LOFO de la Tabla \ref{tab:gpu-politica-ic} da 6.5% de
   ganancia realizada fuera de muestra con **n=8 familias** y un IC95 que
   casi toca cero ($-4.0$ a $13.1\%$). Dirección correcta, evidencia
   débil por tamaño de muestra pequeño, no por defecto de método.
3. **Aproximación del buffer móvil** (`HistoricalGpuClassifier`, A4b):
   sustituye la mediana/desviación estándar de la corrida completa con la
   que se entrenó el modelo por la de una ventana causal de ~20 muestras
   recientes. **Nunca se comparó contra la agregación real de
   entrenamiento** -- no hay ninguna cifra de cuánto se degrada el 0.792
   de LOFO al usar esta versión online en vez de la offline.
4. **Kernels que cruzan el punto de inflexión Roofline** (`rajaperf_gemm`,
   `rodinia_lud`, sección \ref{sec:resultados-gpu-gemm-evolucion} del
   libro): su clase real cambia con el nivel de frecuencia. Un kernel
   nuevo con ese comportamiento probablemente se clasifica mal, sin forma
   de saberlo de antemano con las cinco variables actuales.

No es un defecto de diseño del daemon: es exactamente la pregunta que la
aplicación compuesta **B (familias inéditas)** de §0.1 está pensada para
responder de forma empírica, en vez de asumir que el 0.792 de LOFO se
traduce directamente en beneficio real. Un resultado donde "el
clasificador acierta pero la ganancia no se sostiene" sería un hallazgo
válido de Fase 4, no un fallo del daemon -- pero hasta que esa medición
exista, el brazo *activo* de GPU sobre un kernel fuera del catálogo de
entrenamiento se apoya en una cadena de tres aproximaciones no verificadas
apiladas sobre una que sí lo está.

### Bloque B — Loop de CPU en C++ — núcleo de inferencia CERRADO (2026-09-22), falta la integración en vivo

El SDK C++ de ONNX Runtime no estaba disponible en ningún entorno local
(solo el binding Python) -- instalado vía un entorno conda-forge dedicado
y liviano (sin CUDA, la inferencia de CPU no la necesita):
`conda create -n hyperion-cpu-onnx -c conda-forge cmake onnxruntime-cpp onnx`.
Construidos y probados (4/4 tests C++ en verde, además de los 3 de
`cpu_phase_controller_test` ya existentes):

- [x] **`cpu_feature_builder.hpp`**: deltas de `telemetry::CpuSample` -> las
  6 variables, reproduce EXACTAMENTE las fórmulas de `postprocess.py`
  (líneas ~700-712). Rechaza explícitamente deltas de tiempo <= 0,
  contadores que retroceden (overflow/reset) y denominadores en cero --
  nunca fabrica un valor.
- [x] **`onnx_cpu_classifier.hpp`**: envoltorio de `Ort::Session`, verificado
  contra tres filas calculadas con el `.joblib` real en Python
  (`model.predict_proba`), no valores inventados. El grafo (`zipmap=False`)
  expone dos salidas (`label` int64, `probabilities` float32 [N,2]) --
  verificado con `session.get_outputs()` antes de asumir el orden en C++,
  y `classes_=[0,1]` del modelo real antes de asumir qué índice es
  `memory_bound`.
- [x] **`cpu_loop_tick.hpp`**: une construcción de variables -> inferencia ->
  la ecuación de decisión selectiva del libro (q=max(p,1-p), umbral 0.85)
  -> `CpuPhaseController`. Decisión de diseño nueva, no heredada de Fase 2:
  un tick "revisar" (baja confianza) **no llama a `on_window()`** -- la
  frecuencia queda en lo último aplicado, coherente con la conclusión del
  libro de que la utilidad del clasificador está en decidir cuándo
  abstenerse. Probado con `predict_proba` inyectado, sin modelo real.
- [x] **Latencia de inferencia medida en `paccaA100` real**
  (`scripts/pacca/hyp_cpu_loop_cpp_build_test.sbatch`, job 7562,
  `latency_bench.cpp`, fila a fila, 2000 repeticiones, `taskset -c 0`,
  gobernador `performance`, turbo activo): **p50=16.4µs, p95=18.4µs,
  p99=19.3µs**, contra un presupuesto de ~1000µs -- **~2% del presupuesto
  en el peor caso**. Cifra autoritativa, no una estimación de orden de
  magnitud: corrida vía `sbatch` en el nodo de destino real, nunca en la
  laptop local (ver [[feedback-never-run-compute-locally]] -- el
  ONNX Runtime C++ SDK y el propio `cmake` se instalaron con
  `conda create -n hyperion-cpu-onnx -c conda-forge cmake onnxruntime-cpp onnx`
  en el nodo de LOGIN de pacca, que no es cómputo; compilar/enlazar/correr
  sí lo es y fue todo dentro del job).

**Cerrado (job 7565, paccaA100):** `cpu_loop_consumer.hpp` drena
`telemetry::Collector::Ring` en vivo (probado también con muestras
sintéticas empujadas al ring con `try_push()`, sin PMU ni modelo real,
`cpu_loop_consumer_test`) y `cpu_loop_main.cpp` es el binario de
producción real, equivalente C++ de `run_daemon.py`. Una prueba de humo
de 10s contra PMU real en `paccaA100` (`--perf-cpus 0,1,2,3`,
auto-monitoreo, sin escritura de frecuencia real) confirma el pipeline
completo de punta a punta: **9886 de 9933 ticks (99.5%) construyeron
features válidas y clasificaron con éxito** (9139 actuando con confianza
≥0.85, 747 abstenidos; solo 47 fallos de construcción de features, 0
reintentos de anillo lleno). La primera versión de este smoke test
monitoreaba el PID equivocado (el shell padre, bloqueado esperando, sin
actividad de CPU propia) y todos los ticks fallaban en silencio
(`features_fallidas=9920/9920`) sin que el código de salida del proceso
lo reflejara -- corregido monitoreando al propio `cpu_loop_main`
(`target_pid=0`) y añadiendo una aserción explícita sobre el resumen, no
solo sobre el código de salida.

Tres decisiones de alcance quedaron documentadas explícitamente en el
docstring de `cpu_loop_main.cpp`, no como huecos escondidos: (1) no
parsea `policy_table.yaml` -- la política de CPU es `no_actuar` en ambas
clases hoy, así que un script wrapper resolvería el YAML y pasaría flags
concretos el día que cambie; (2) no escribe frecuencia real -- no existe
todavía un escritor nativo de `scaling_min/max_freq` con verificación por
relectura, y construirlo sin ninguna política en `actuar` sería código
muerto; (3) `gpu_active` siempre `false` -- no hay mecanismo de
coordinación entre este proceso C++ y `run_daemon.py` (Python, proceso
separado, el plan original asumía un único proceso con ambos loops), y no
hace falta uno mientras la política GPU siga bloqueada por H1.

**Hallazgo colateral, fuera de alcance de Bloque B:** `add_subdirectory`
sobre `common/telemetry` arrastra su propia suite de pruebas (20 tests en
total); 3 de ellas, preexistentes y no escritas en esta ronda
(`collector_no_perf_test`, `rapl_reader_test`, `cpu_freq_reader_test`),
fallan con `std::bad_alloc` en `paccaA100` específicamente, reproducible
en dos corridas independientes (jobs 7563 y 7564). No es una regresión de
Bloque B (los 5 tests propios pasan 5/5) ni bloquea el cierre, pero queda
como hallazgo real sin investigar -- posible causa: estas pruebas no se
habían corrido antes en `paccaA100` (nodo GPU, topología de sysfs
distinta a los nodos donde normalmente se verifica `telemetry`).

### Bloque C — Brazos de experimento e integración

- **C1. Cerrado (job 7566, paccaA100).** Brazo como parámetro de primera
  clase: `run_daemon.py` reemplazó `--dry-run` por `--arm {sombra,activo}`
  (obligatorio, sin default -- `build_daemon_gpu_loop()` rechaza `arm=base`
  y cualquier combinación `arm`/`dry_run` inconsistente). `cpu_loop_main.cpp`
  recibió la misma bandera `--arm`. Registro estructurado idéntico en ambos
  dispositivos: `fase3_daemon/decision_log.py` (GPU) y
  `fase3_daemon/cpu_loop/include/decision_log.hpp` (CPU) escriben el mismo
  esquema JSONL (marca de tiempo, brazo, variables leídas, clase inferida,
  confianza, decisión de la política, si se escribió, tiempo de inferencia
  y de actuación) -- una línea por decisión de GPU, una línea por tick de
  CPU sin importar el desenlace (actuó/abstuvo/features fallidas). Probado
  con `decision_log_test` (C++, 4 casos) y `test_decision_log.py` (Python,
  6 casos), además de 4 casos nuevos en `test_run_daemon.py` que verifican
  el rechazo de `arm=base`/inconsistencias y el esquema completo del
  registro en brazo *sombra*. En el smoke test real de 10s sobre
  `cpu_loop_main` (PMU real, `--arm sombra --log-path ...`), el registro
  quedó con 9944 líneas = `actuo(9873)+abstuvo(59)+features_fallidas(12)`,
  verificado por el propio sbatch -- confirma que el wiring de producción
  escribe una línea por tick, no solo el test unitario. Suite completa de
  `fase3_daemon/tests/` en verde localmente (55 passed, 1 skipped); puerta
  dura de `cpu_loop` 6/6 en verde en paccaA100 (se sumó `decision_log_test`
  a la puerta dura y al smoke test). Los mismos 3 tests preexistentes de
  `common/telemetry` (`std::bad_alloc`, ver cierre de Bloque B) siguen sin
  investigar, sin bloquear.
- **C2a. Aplicación A (familias conocidas) — Cerrado (job 7572, paccaA100).**
  `fase3_daemon/composite_apps/composite_known.py` encadena `dgemm_n2048`
  (compute_bound puro, memory_share=0.0) y `npb_cg` (memory_bound puro,
  memory_share=1.0) -- ambos del inventario de 30 familias
  (`tmp/cpu_quality_20260918/full/inventory_by_family.csv`), ambos ya
  vistos en el entrenamiento del clasificador de Fase 2, ambos con
  `binary_checksum` declarado para `pacca-a100` en el catálogo. Nunca
  corre un binario sin verificar su checksum antes de cada ejecución
  (`common/hpc/catalog.py::verify_binary`, mismo rigor CAT-07 de Fase 1).
  Fronteras de fase = `phase_label_hint` del catálogo (ground truth ya
  derivado en Fase 1, no inventado aquí) + reloj monotónico real de
  inicio/fin de cada kernel, registrado en JSONL. Corrida real en
  paccaA100 (2 ciclos, 4 fases): 4/4 exitosas, alternando compute_bound
  (dgemm, ~1.7s) / memory_bound (npb_cg, ~2.7s), checksum verificado en
  cada ejecución. 9/9 tests unitarios locales
  (`fase3_daemon/composite_apps/tests/test_composite_known.py`, todos con
  `run_fn`/`now_fn` inyectados -- sin binarios reales, corrible en
  cualquier máquina). No usa `telemetry_kernel_launcher`: la telemetría de
  esta aplicación la produce el daemon que la observa
  (`run_daemon.py`/`cpu_loop_main`), corriendo aparte -- este driver solo
  orquesta y registra fronteras, no mide PMU.

  **Hallazgos colaterales durante la verificación (jobs 7567-7572), ambos
  corregidos, ninguno afectó trabajo ya cerrado:**
  - `common/hpc/catalog.py::verify_binary`/`verify_cupti_activity_binary` y
    `fase1_telemetria/campaign.py::_launcher_checksum` usaban
    `hashlib.file_digest` (stdlib 3.11+); reemplazado por lectura en
    bloques de 1 MiB, portable. Verificado que esto NUNCA afectó campañas
    reales: todas activan `~/hyperion-venv` (Python 3.11.13), nunca el
    Python 3.10.20 del sistema -- el fix es una mejora de portabilidad
    real (útil para scripts nuevos que, como el propio
    `hyp_composite_known_smoke.sbatch` en su primer intento, no activen
    ese venv), no una corrección de un bug en producción.
  - `hyp_composite_known_smoke.sbatch` no replicaba el entorno real de
    campaña (`module load ... openblas`, `LD_LIBRARY_PATH`,
    `source ~/hyperion-venv/bin/activate`, ver `hyp_external_cpu.sbatch`)
    -- causó los tres fallos en cadena (pytest ausente, `hashlib` roto,
    `dgemm_bench` sin `libopenblas.so.0`). Corregido replicando ese mismo
    patrón.

- **C2b. Aplicación B (familias inéditas) — Cerrado (job 7573, paccaA100).**
  `fase3_daemon/composite_apps/composite_unseen.py` encadena
  `cpu_xsbench_omp` (memory_bound) y `cpu_rsbench_omp` (compute_bound) --
  el mismo par que `cpu_external_screen_20260921.yaml` declara "nunca
  entran a entrenamiento" y que ya corrió con éxito como prueba externa
  sellada (job 7530). Contraparte deliberada de la Aplicación A (§0.1
  Eje 1): si A gana y B no, el resultado es memorización, no
  generalización -- también el instrumento con el que Fase 4 puede cerrar
  las dos capas de generalización sin validar documentadas en Bloque A.
  Reutiliza `run_composite`/`resolve_entries` de `composite_known.py` sin
  duplicar lógica (`build_arg_parser()`/`main_with_defaults()`
  compartidos, para que un cambio futuro al criterio de éxito o al
  registro se aplique a ambas aplicaciones a la vez). Corrida real en
  paccaA100 (2 ciclos, 4 fases): 4/4 exitosas, alternando memory_bound
  (xsbench, ~9.15s) / compute_bound (rsbench, ~43.6s), checksum
  verificado en cada ejecución. 13/13 tests unitarios locales
  (`fase3_daemon/composite_apps/tests/`, incluye una verificación contra
  el catálogo real -- sin mocks -- de que A y B son conjuntos de familias
  disjuntos).
- **C3. Cerrado (job 7574, paccaA100).** `fase3_daemon/gpu_loop/verify_phase_detection_e2e.py`
  lanza `gpu_dgemm_n4096` (checksum verificado, compute_bound) como
  subproceso real mientras un hilo aparte corre
  `activity_poller.poll_phase_events()` con `query_gpu_features()` real
  (NVML vía `nvidia-smi`, no `GpuFeatures` sintéticas como los tests
  unitarios existentes). Reutiliza `should_continue` (ítem C7) para
  detener el sondeo de forma limpia con un margen tras el fin del kernel,
  en vez de un `for`/`break` sobre el generador infinito. Corrida real:
  kernel exitoso (3.55s), inicio de fase detectado (latencia 687.5ms) y
  fin de fase detectado (latencia 381.6ms) -- ambas dentro del orden de
  magnitud esperado para un sondeo cada 50ms más el tiempo real que tarda
  `gpu_util_pct` en cruzar el umbral, confirmando de punta a punta lo que
  antes solo se sabía por tests unitarios con datos inyectados. 5 tests
  locales
  (`fase3_daemon/gpu_loop/tests/test_verify_phase_detection_e2e.py`) cubren
  solo la lógica determinista (criterio de éxito, propiedades de
  `E2EResult`, rechazo por checksum) -- el camino feliz con hilos+GPU real
  es intrínsecamente de punta a punta y se verifica en el cluster, no con
  relojes falsos (decisión de alcance explícita, evita un test frágil sin
  garantía real adicional).
- **C4. Cerrado (job 7575, paccaA100).** Mecanismo elegido: archivo de un
  byte, reemplazado atómicamente (`os.replace`/POSIX `rename()`) por el
  loop de GPU en cada transición de fase (`fase3_daemon/gpu_loop/coordination.py::GpuActiveSignalWriter`,
  enganchado a `on_decision`/`on_end` de `build_daemon_gpu_loop` vía
  `--gpu-active-signal-path`), leído por el loop de CPU cada tick
  (`fase3_daemon/cpu_loop/include/gpu_active_reader.hpp`, enganchado a
  `cpu_loop_main` vía la misma bandera). Falla cerrado a `false` ante
  cualquier problema de lectura (archivo ausente, vacío, contenido
  irreconocible) -- el mismo default que existía antes de que esta señal
  existiera. Probado en tres capas: 6 tests Python del escritor
  (`fase3_daemon/tests/test_coordination.py`), 6 tests C++ del lector
  (`fase3_daemon/cpu_loop/tests/test_gpu_active_reader.cpp`), y una
  verificación real de punta a punta entre DOS PROCESOS del sistema
  operativo (`fase3_daemon/gpu_loop/verify_gpu_coordination_e2e.py`
  lanza `gpu_active_signal_probe`, un binario C++ real sin PMU/ONNX, como
  subproceso mientras Python escribe una secuencia programada con esperas
  reales) -- confirma que la señal funciona entre procesos reales, no
  solo que cada lado pasa sus propios tests en aislamiento. Corrida real:
  el proceso C++ observó exactamente `[False, True, False, True, False]`,
  coincidencia exacta con la secuencia escrita por Python (incluido el
  estado inicial `False` por archivo ausente antes de la primera
  escritura), `exit_code=0`.
- **C5. Cerrado (local, sin necesidad de pacca -- es software puro de
  señales/subprocesos, no HW real).** Tres pruebas de caos reales
  (`fase3_daemon/tests/test_daemon_restore_chaos.py`), mismo patrón ya
  establecido en `common/tests/test_freqctl.py::test_frq05_sigint_heredada_como_ignorada_restaura_y_termina`
  (proceso real y separado, señal real enviada de verdad), pero aplicado
  al WIRING propio de `run_daemon.py::_install_restore_handlers` -- no
  repite la cobertura ya existente de `gpu_freqctl.restore_gpu_state` en
  sí (`common/tests/test_gpu_freqctl.py`, incluida la garantía de "nunca
  lanza"):
  - Brazo *activo*, SIGINT heredada como `SIG_IGN` (la misma herencia
    adversa que motivó la regresión original de `freqctl`): el proceso
    restaura y muere por la señal.
  - Brazo *activo*, SIGTERM (lo que Slurm envía al cancelar un job, el
    caso real de "nos quitan el nodo antes de tiempo"): restaura y muere.
  - Brazo *sombra*, SIGTERM: el proceso muere sin dejar ningún rastro --
    nunca se registró un manejador de restauración porque nunca se tocó
    hardware real, confirmando §0.1 punto 5 ("en base y sombra no hay
    nada que restaurar, y eso también debe verificarse").
  Complementado con 2 tests locales de wiring (`test_run_daemon.py`):
  `sombra` nunca llama a `detect_environment()`/`install_emergency_handlers`,
  `activo` sí y el manejador registrado sí invoca `restore_gpu_state()`.
  El brazo *base* no requiere prueba: es, literalmente, no correr el
  script, no hay proceso que matar.

  **Fuera de alcance de esta ronda, explícito:** una prueba de caos que
  confirme la restauración del reloj de GPU FÍSICO tras una escritura real
  (no un doble de prueba) depende de que H1 (candado de reloj averiado,
  Bloque D) esté reparado -- hoy escribir un reloj de GPU real ya no es
  fiable (`nvidia-smi -lgc/-rgc` sale con éxito pero el reloj queda
  pegado), así que verificar su restauración con el mismo instrumento roto
  no probaría nada. El wiring de software (qué se llama, cuándo, con qué
  argumentos) es exactamente lo que sí se puede y se debe verificar ahora,
  independiente de H1.
- **C6. Cerrado (jobs 7576/7577, paccaA100).** Lado CPU ya medido en
  Bloque B (`cpu_loop_latency_bench`: p50=16.4µs/p99=19.3µs contra un
  presupuesto de ~1ms por tick, ~1.6-1.9%). Lado GPU:
  `fase3_daemon/gpu_loop/measure_overhead.py` corre `build_daemon_gpu_loop`
  real (brazo *sombra*, clasificador real, `policy_table.yaml` real) en un
  hilo mientras `gpu_dgemm_n4096` (mismo kernel de C3, checksum verificado)
  se lanza 5 veces en el hilo principal -- mismo patrón de coordinación
  que `verify_phase_detection_e2e.py`. Reporta p50/p95/p99/max de
  `inference_time_ns`/`actuation_time_ns` (los mismos campos que ya
  registra `decision_log.py`, ítem C1) y los escribe a JSON (`--out`,
  "registrada"). Sin presupuesto de tick equivalente al de CPU -- el loop
  de GPU decide una vez por FASE, no cada ~1ms -- así que se compara
  contra la duración típica de la fase, no contra un budget fijo. 5 tests
  locales (`fase3_daemon/gpu_loop/tests/test_measure_overhead.py`) cubren
  la agregación de percentiles con datos sintéticos.

  **Resultado real (job 7577, 5/5 decisiones medidas):**
  `actuation_ns` p50=10.6µs/p99=26.9µs (mismo orden de magnitud que el
  lado CPU). `inference_ns` p50≈898µs, con un arranque en frío de 7.76ms
  en la primera llamada (probablemente JIT/caché de numpy del primer
  `model.predict()`) y consistentemente ~850-910µs en las 4 siguientes --
  **~50-60x más lento que la inferencia ONNX C++ del lado CPU (16µs)**,
  esperable: el candidato GPU es sklearn puro en Python, no ONNX, y corrió
  concurrente con un kernel que satura CPU (contención real de
  scheduling). Aun así, insignificante frente a la duración real de una
  fase (~1.7s del kernel): <0.1% de sobrecarga -- no amenaza la
  comparación *sombra − base* que hará Fase 4.

  **Hallazgo colateral, no bloqueante, corregido en el camino:** el primer
  intento (job 7576) exigía `n_decisions == cycles` y falló pese a medir
  3 decisiones reales válidas -- el sondeo por umbral (`activity_poller.py`)
  fusiona lanzamientos consecutivos del kernel en una sola fase si el
  hueco entre ellos dura menos que `poll_interval_s` (50ms), limitación de
  granularidad ya documentada, no un fallo de la medición. Corregido para
  exigir solo `n_decisions >= 1`.

**Bloque C completo: C1, C2a, C2b, C3, C4, C5, C6, C7 todos cerrados.**
- **C7. Cerrado.** Modo `--pid` ahora ata el ciclo de vida del loop de GPU
  al proceso objetivo: `activity_poller.poll_phase_events()` acepta
  `should_continue` (consultado al inicio de cada iteración, antes de
  sondear NVML) y `run_daemon.py::pid_alive()` lo resuelve con
  `os.kill(pid, 0)`. Con `--mode pid`, el script rechaza arrancar si el
  proceso ya no existe, y el loop se detiene solo cuando termina -- sin
  eso, seguía sondeando indefinidamente contra un PID muerto. `--mode
  cpuset` (default) queda sin cambio de comportamiento, documentado
  explícitamente como "corre mientras dure la asignación de Slurm". Límite
  declarado, no oculto: `--pid` decide CUÁNDO parar, no filtra qué
  muestras NVML cuentan -- `query_gpu_features()` sigue siendo una lectura
  de todo el dispositivo (mismo límite estructural que ya tiene el
  clasificador GPU). Probado con 2 casos nuevos en `test_activity_poller.py`
  y 3 en `test_run_daemon.py` (60 passed, 1 skipped, todo local -- Python
  puro con datos sintéticos, sin GPU/PMU real, no requiere pacca).

### Bloque C8 — Brazo activo de CPU (EN CONSTRUCCIÓN)

**Decisión (2026-09-23).** El daemon de CPU deja de ser un no-op. La
comparación de la tesis es contra el gobernador nativo (REF), que **no es
un estado del daemon**: es el brazo *base* (daemon apagado). Con el daemon
activo el reloj siempre está en un nivel fijo, con el turbo apagado:

| Situación | Nivel |
|---|---|
| Daemon activo, estado base | **F0** (3200 MHz, turbo apagado) |
| `memory_bound`, q >= 0.85 sostenido | **F1** |
| `compute_bound` o abstención | F0 (o mantiene el último; un tick "revisar" no llama a `on_window()`) |

- F0 como base está respaldado por los datos de entrenamiento (kernel
  único): +0.56% de EDP en compute_bound (p=0.008) y +0.2% en
  memory_bound (p=0.94, plano; energía y tiempo idénticos a REF). No es un
  costo. Apagar el turbo es parte legítima de la intervención (impide
  llegar a 3600 MHz); por encima de 3200 sin turbo no existe punto fijable.
- F1 es la apuesta: en agregado pierde 5.4% en memory_bound (ahorra 1.1%
  de energía, alarga 2.3% el tiempo; 11/28 kernels mejoran). Solo paga si
  el clasificador acierta la fase y la fase es larga.
- El mejor nivel por kernel (oráculo) NO se usa: se elige con los mismos
  datos con que se mide la ganancia, e implica conocer el kernel.

**Brazos:** *base* (REF, daemon apagado), *sombra* (sin escribir;
`sombra - base` = sobrecarga) y *activo* (`activo - base` = efecto total
contra el gobernador nativo). Variante **activo-F0** (en memory se queda en
F0) para ver si F1 agrega algo sobre solo fijar F0; ambas se reportan, sin
elegir la mejor después de ver los resultados.

**Actuador en C++, portando las reglas de `freqctl.py`.** El mecanismo de
fijar frecuencia ya existe y está probado en `common/hpc/freqctl.py`, pero es
Python y el loop de CPU es C++. Un actuador Python persistente habría añadido
un segundo proceso (intérprete, sondeo de un archivo, latencia decisión ->
reloj) cuya energía cuenta en el RAPL de paquete, o sea, dentro de la
sobrecarga que el brazo *sombra* debe medir. Decisión: el actuador vive en
`cpu_loop_main` y implementa el `FrequencySetter` del controlador en C++,
**portando sin cambiar las reglas** de `freqctl.py`: orden protegido al
escribir min/max, relectura de cada escritura, hermanos SMT, snapshot del
estado original, restauración idempotente que prueba todos los CPU aunque uno
falle, y manejadores de señal que restauran. Prueba de paridad: el mismo
escenario sobre un sysfs simulado, corrido con `freqctl.py` y con el C++,
comparando los archivos resultantes. Turbo: el C++ hace `fork/exec` de
`sudo /usr/local/bin/set_turbo_state` (ruta absoluta, `1` = desactiva) solo
al entrar y salir del daemon, con relectura de `no_turbo`; no se escribe
directo porque el permiso lo da el wrapper. Verificar la ubicación actual del
wrapper antes de construir (en `scripts/pacca/` solo aparece referenciado
desde los sbatch; `with_cpu_turbo_disabled.sh` está en `old/`).

**Estado (2026-09-23).** Escrito, NO verificado (no se compila ni corre en
local, ver feedback-never-run-compute-locally): `cpu_freq_actuator.hpp`
(actuador con turbo, snapshot, restauración idempotente y falla cerrado),
`tests/test_cpu_freq_actuator.cpp` (10 casos sobre sysfs simulado),
`tools/cpu_freq_actuator_probe.cpp` + `tests/test_freqctl_parity.py`
(paridad con `freqctl.py`), integración en `cpu_loop_main.cpp` (solo brazo
`activo`, con `--sysfs-cpu-root`/`--no-manage-turbo` para pruebas) y
`scripts/pacca/hyp_cpu_freq_actuator_test.sbatch` (puerta dura). Pendiente:
correr ese sbatch en pacca; luego la lectura de la tabla de política (hoy los
niveles entran por `--compute-freq-khz`/`--memory-freq-khz`), el caos real y
la medición de latencia de conmutación.

Tabla y lanzador (escritos, sin correr): `build_policy_table.py` acepta
`--cpu-experimental-base F0 --cpu-experimental-memory F1` (F0 en ambos =
variante activo-F0) y emite `action: actuar_experimental` con
`resolved_freq_khz` (rejilla final: F0=3200000, F1=2900000) y
`measured_action: no_actuar`, para que nadie la lea como una política que
ganó; el `policy_table.yaml` versionado NO se regeneró. `cpu_loop/
launch_cpu_daemon.py` traduce la tabla a los flags de `cpu_loop_main` y hace
`exec` (las señales llegan directo al proceso C++ que restaura); exige nivel
base si memory actúa.

**Hallazgos del preflight real (job 7592, 2026-09-23, paccaA100).**
- El actuador C++ escribe y restaura de verdad: la sonda (turbo apagado,
  100 x F0<->F1 con carga en 0-5) terminó con el estado IDÉNTICO al inicial.
  El estado tras la cancelación del job 7590 estaba limpio (0-5 y 16-21 en
  800000/3200000, turbo activo).
- **Latencia de conmutación: set_khz p50 27.7 ms (p95 28.2 ms)** para 12 CPU
  lógicos (6 + hermanos), enter(turbo off) 19 ms, restore(turbo on) 48 ms,
  asentamiento de `scaling_cur_freq` a ±5% p50 37.7 ms (200/200 < 500 ms).
  Es ~28 veces el tick de 1 ms: el controlador actual (sin piso de
  permanencia) NO es viable tal cual, porque una clase que oscila entre ticks
  dispara una escritura de ~28 ms en cada cambio. Hace falta permanencia
  mínima/histéresis (p.ej. N ventanas consecutivas de la misma clase) y/o
  reducir el costo por escritura.
- **Bug encontrado: `run_consumer_loop` no atiende `stop` bajo backlog.**
  `stop` se consulta solo fuera del bucle interno `while (try_pop())`; si el
  consumidor va más lento que el productor (aquí, por las escrituras de
  28 ms), el ring nunca se vacía y SIGTERM/SIGINT no detienen el proceso.
  Consecuencia: la restauración por señal no corre. Corregir consultando
  `stop` dentro del drenaje. La prueba de caos por señal (SIGTERM/SIGINT) NO
  se completó y sigue pendiente.
- Contexto de la prueba: `cpu_loop_main` se monitoreó a sí mismo
  (`--target-pid` 0) y su clase osciló entre memory y compute casi cada tick,
  lo que agravó lo anterior; con un target real habrá que repetirla.
- Incidente: la intervención manual (kill -9 + reescritura de turbo) se solapó
  con la segunda ronda y la invalidó; el job terminó FAILED y el nodo se
  verificó/reparó con `hyp_node_state_check.sbatch` (jobs 7593/7594). La v1 de
  ese script asumió max=3200000 en TODOS los CPU y bajó el max de los no
  delegados (nativo 3600000); la v2 (job 7594) lo devolvió a cpuinfo_max.

**Correcciones y optimización de la conmutación (job 7596, 2026-09-23).**
- `run_consumer_loop` ahora consulta `stop` dentro del drenaje (y el drenaje
  final se acota a 64 muestras): SIGTERM y SIGINT detienen `cpu_loop_main`,
  restauran y el estado queda IDÉNTICO al inicial (caos por señal real, ambos
  casos, con `dd` como target). Prueba de regresión con productor rápido y
  consumidor lento en `test_cpu_loop_consumer.cpp` (no se comprobó que falle
  contra el código anterior; el watchdog aborta si el bucle no vuelve).
- Histéresis: `CpuPhaseControllerConfig::min_consecutive_windows` (default 1
  en el controlador; `cpu_loop_main --min-dwell-windows`, default 50) y
  `mark_applied()` para no reescribir el nivel base al arrancar. Con ella la
  ronda de caos hizo 1 y 7 escrituras reales (antes, una por cambio de clase).
- Costo por cambio F0<->F1 sobre 12 CPU lógicos (200 cambios por variante):

  | Variante | set_khz p50 | asentamiento cur_freq p50 |
  |---|---|---|
  | min+max, secuencial (original) | 27.9 ms | 37.7 ms |
  | min+max, **un hilo por CPU** | **2.85 ms** | 10.4 ms |
  | solo techo, secuencial | 14.1 ms | 24.1 ms |
  | solo techo + hilos | 1.8 ms | 10.6 ms |

  El asentamiento (~10 ms) ya está dominado por el hardware, no por las
  escrituras. Default de `cpu_loop_main`: `--switch-parallel 1` con min+max
  fijos (misma semántica de candado que las campañas medidas); `--switch-pin-min 0`
  queda como opción, con 200/200 asentamientos correctos bajo carga pero sin
  usarse por defecto para no cambiar la condición frente a las mediciones.
- Pendiente: el target `dd` del caos produjo 96% de ventanas con
  `zero_cycles` (5171 de ~5400): no es representativo; el caos y la
  clasificación deben repetirse con un target real de las aplicaciones A/B.

**Qué construir**
1. Actuador C++ (port de `freqctl.py`, con prueba de paridad) + control de
   turbo con relectura.
2. Lectura de la tabla en `cpu_loop_main` (hoy no parsea el YAML) con un
   estado `actuar_experimental` (`base_level: F0`, `memory_level: F1`),
   distinto de una política ganadora.
3. Restauración por caos (C5) ampliada a turbo y rango de frecuencia, incluida
   la muerte del daemon con el nodo en F1 y turbo apagado.
4. Guardas: estado inicial de los núcleos 0-5 (contaminación de cpufreq en
   paccaA100) y turbo siempre desactivado en corridas de frecuencia fija.

**Riesgos y cosas a medir**
- Latencia de F0<->F1: escritura, relectura y asentamiento real del reloj,
  más la llamada `sudo` del turbo (el turbo solo se toca al entrar/salir del
  daemon, no en cada transición F0<->F1, salvo que se decida lo contrario).
  El controlador actual no tiene piso de permanencia mínima; revisar si hace
  falta a la luz de esta medición.
- Fases cortas: las aplicaciones compuestas necesitan fases más largas que el
  costo de conmutar, o el experimento no puede mostrar nada.
- Aplicaciones compuestas solo de CPU: bajar el reloj de CPU alarga 63-66% las
  cargas GPU (F1-XDEV-006).
- Mientras corra una campaña, las shells adjuntas al nodo son solo inspección.

**Orden:** (1) local con sysfs simulado y pruebas de tabla/política;
(2) preflight en pacca (~20 min) de escritura real con relectura y latencia,
revisando antes `docs/general/Estado_Cola_Slurm.md`; (3) caos real;
(4) aplicaciones A y B, tres brazos + variante activo-F0, >= 3 repeticiones,
con registro por decisión para puntuar la clasificación contra las fronteras
de fase conocidas.

**Expectativa declarada de antemano:** por el piso de potencia (85 W, 84% del
total) la ganancia esperada en CPU es pequeña; un resultado plano o negativo
es un hallazgo válido.

### Bloque D — Bloqueado por H1

Brazo *activo* de GPU. No se arranca hasta que el candado de reloj esté
reparado. Todo lo demás corre en *sombra* mientras tanto.

---

## 2. Criterios de cierre

Los del checklist §8 del plan de realineación, más los de §0.1:

- [ ] Ambos loops funcionando y verificados por separado.
- [ ] Detección de fase de GPU probada contra un kernel real de terceros.
- [ ] Señal de coordinación probada, con la semántica corregida de §0.2.
- [ ] Restauración verificada con prueba de caos real.
- [ ] Sobrecarga del daemon medida y registrada.
- [ ] Los tres brazos corren sobre las aplicaciones compuestas A y B, con
      registro por decisión suficiente para puntuar clasificación (no solo
      EDP) contra las fronteras de fase conocidas.
- [ ] Ruta de GPU completa y validada en *sombra*, de modo que el brazo
      *activo* no requiera cambios de código cuando H1 se repare.
- [ ] Aplicación B (familias inéditas) mide, sobre las decisiones
      reales del daemon, las tres capas de generalización sin validar
      documentadas en A4: exactitud del clasificador con el buffer móvil
      real (no la agregación offline de entrenamiento), y si la ganancia
      de F1 en `memory_bound` (n=8, IC95 casi en cero) se sostiene fuera
      del catálogo de política. Resultado negativo en cualquiera de las
      dos es un hallazgo válido, no un bloqueador de cierre.
