# Plan de la Fase 4: escenario E (GPU dominada por memoria) y REF sin turbo

Fecha: 2026-09-24. Complementa `Plan_Fase4_Evaluacion.md` (matriz y escenarios A a D). Este documento está pensado para que
otro agente lo ejecute de punta a punta: contiene el contexto, las decisiones ya tomadas por el usuario, los cambios de
código, las corridas y los criterios de parada. **Leer completo antes de tocar nada**, y leer también `AGENTS.md` (raíz) y
`docs/libro/AGENTS.md`.

---

## 0. Reglas no negociables (resumen; la fuente es AGENTS.md y la memoria del proyecto)

- **Slurm en pacca (cuenta compartida `latorresn`)**: nunca `--time`; todo `--job-name` empieza con `hyp_`; nunca
  `scancel` de jobs ajenos; revisar `squeue -u latorresn` antes de encolar.
- **Nunca medir con una shell adjunta** (`srun --jobid`) mientras corre una celda: RAPL de paquete y la GPU son
  compartidos y la contaminación no se detecta.
- **Sync local/GitHub/pacca solo con git push/pull.** En pacca el código de Fase 3/4 vive en `~/hyperion_c8`
  (`HYPERION_ROOT`); **no hacer pull en `~/hyperion`** (lo usa la campaña CUPTI y tiene cambios locales).
- **Commits**: `git add` con rutas explícitas (nunca `-A`); **sin** trailer `Co-Authored-By` en este repo.
- **Turbo**: el argumento de `set_turbo_state` se escribe literal en `no_turbo`: `1` APAGA el turbo, `0` lo enciende.
  Siempre por ruta absoluta: `sudo -n /usr/local/bin/set_turbo_state 1`.
- **Pre-vuelo antes de una corrida larga**: primero los chequeos gratuitos por lectura de código y datos, luego una corrida
  corta (~20 min) que pueda falsar los supuestos, y solo después la matriz.
- **Consultar decisiones de diseño**: si aparece una decisión no cubierta aquí (qué kernel usar si un candidato falla, qué
  hacer con un resultado inesperado), parar y preguntar al usuario, no decidir en silencio.
- **Campaña CUPTI** (job 7644, `scripts/pacca/final_campaign/hyp_gpu_cupti_campaign.sbatch`, desde `~/hyperion`) quedó
  cancelada a propósito; se reanuda cuando el usuario lo indique, no antes.

---

## 1. Contexto: qué se midió y por qué los resultados fueron malos

Datos en `~/hyperion-results/final/` (pacca) y copia en `docs/libro/datos/fase4_20260924/`. Análisis:
`python3 fase4_evaluacion/analyze_matrix.py <carpeta>` (en pacca, porque las decisiones por celda solo están allá).

| Corrida | Job | Carpeta | Contenido |
|---|---|---|---|
| Matriz A (1 ms) | 7639 | `fase4_matrix_main` | 82 celdas, cpu/gpu/joint x A/B x brazos |
| CPU a 10 ms | 7642 | `fase4_cpu_10ms` | cpu, base/sombra/activo |
| Conjunta a 10 ms | 7643 | `fase4_joint_10ms` | joint, base/sombra/activo |
| Diagnóstico | 7645 | `fase4_cpu_diag` | cpu A, variantes de sombra |
| Escenario C | 7646 | `fase4_C_ort1` | cpu+joint con ORT de 1 hilo (en curso al escribir esto) |

Resultados (EDP del nodo frente a REF): clasificación CPU 94 a 99.9%, GPU 100% en lo decidido con abstenciones; activo CPU
1.7 a 2.0; conjunta 1.3 a 1.6; GPU 1.00.

Causas identificadas, con evidencia:

1. **Costo del agente de CPU (resuelta)**: los hilos por defecto de ONNX Runtime giraban y competían con la app. Con
   `--ort-threads 1` (ahora default, commit 78a2b08) el brazo sombra queda en tiempo 1.00, E_cpu 1.06, EDP 1.05.
2. **La acción de CPU no puede pagar (límite físico)**: activo y activo_f0 dan casi lo mismo, así que la pérdida viene de
   fijar F0 = 3.2 GHz sin turbo frente a REF con turbo (hasta 3.6 GHz) en las fases de cómputo. Unos 85 W de paquete no
   escalan con el reloj. La Fase 2 ya concluyó `no_actuar` para ambas clases de CPU (`fase3_daemon/policy_table.yaml`).
   Activo/activo_f0 en EDP: CPU A 1.06, CPU B 1.005: conmutar a F1 en memoria no ayudó ni frente al nivel fijo.
3. **La aplicación de GPU tenía poco que ganar**:
   - Se usó `gpu_rajaperf_stream_triad`, una de las familias memory_bound más débiles con F1 (5.6% de mejora de EDP en
     Fase 2) frente a 15 a 17% de dual_stencil, dual_spmv, rodinia_lud, dual_axpy y jacobi_2d
     (`docs/libro/datos/gpu_calidad_20260922/politica/family_log_ratios.csv`; columna = log de la razón de EDP frente a
     REF, negativo = mejora; mejora % = (1 - exp(log)) x 100 con signo invertido).
   - Solo 36% (A) y 49% (B) del tiempo en fases de memoria.
   - La energía de CPU es ~44% de la del nodo en el alcance GPU (15 176 J frente a 19 489 J) y bajar el reloj de GPU no la
     reduce: diluye cualquier ahorro de GPU a la mitad en el EDP del nodo.
   - En B, BabelStream (`gpu_stream_bw`, 8 s por fase) recibió una sola decisión por fase y fue abstención.
   - Cuenta de control: 0.36 x 5.6% x 0.56 ~ 1%, coherente con el 0.4% medido.

---

## 2. Decisiones ya tomadas por el usuario (no reabrir)

- **Escenario E**: aplicación dominada por fases memory_bound de GPU, con las familias que más ganan con F1 según la Fase 2.
- **Política de GPU congelada**: F1 = 1260 MHz en memory_bound, nativo en compute_bound (tabla versionada). **No** usar el
  mejor nivel por kernel (F2/F3): sería ajustar al resultado.
- **CPU en E: medir ambas opciones** como brazos distintos:
  - (a) **sin agente de CPU** (la política de CPU es `no_actuar` en ambas clases, no hay nada que hacer);
  - (b) **agente de CPU en observación** (misma política, `--arm sombra`), para medir clasificación de CPU y su costo.
- **REF sin turbo**: se agrega como **línea base secundaria** (brazo `base_noturbo`: sin agente, turbo apagado, gobernador
  nativo). **REF con turbo sigue siendo la referencia principal**; todo resultado se reporta frente a ambas, nunca solo
  frente a la sin turbo. Pregunta que responde: si conmutar por fase le gana a un nivel fijo.
- **Repeticiones**: 3 por celda en todo.
- **Anti-selección**: A y C quedan en el libro tal cual. E se declara antes de correrse (sección 6) como *escenario de
  aplicabilidad*, con el criterio de selección explícito, y se reporta salga como salga.

---

## 3. Paso 1: chequeos gratuitos (sin nodo)

Hacerlos todos antes de escribir código de la corrida. Registrar las respuestas en la sección 8 de este archivo.

1. **Latencia de aplicación de F1**: en `fase4_matrix_main/cells/gpu_known_activo_*/`, cruzar `gpu_decisions.jsonl`
   (`ts_ns`, `written`, clase) con `phases.jsonl` (`begin_ns`): cuántos segundos después del inicio de cada fase de triad
   se escribió F1. Si es una fracción grande de la fase, las fases de E deben ser más largas.
2. **¿Re-decide tras una abstención?**: leer `fase3_daemon/gpu_loop_cpp` (controlador de fase) y confirmar si, después de
   un `revisar`, el agente vuelve a evaluar dentro de la misma fase o pierde la fase entera. Si la pierde, anotarlo como
   limitación; **no** cambiar el umbral (0.90 congelado).
3. **Qué energía mide la ganancia de Fase 2**: en `docs/libro/scripts/generar_figuras_politica_gpu.py` y sus datos, ver si
   `energy_j` es solo GPU (NVML) o GPU+CPU. Recalcular la ganancia esperada de E con el dato correcto.
4. **Kernels candidatos** (sección 4): para cada uno, confirmar en `fase1_telemetria/catalog/catalog.yaml` que existe el
   `id`, su `phase_label_hint`, `binary_checksum` para `pacca-a100`, `expected_runtime_seconds` y `exec_args`.
   - Ojo: `gpu_rajaperf_jacobi_2d` está como `phase_label_hint: intermedio` en el catálogo. Averiguar con qué etiqueta
     entró a la Fase 2 (`docs/libro/datos/gpu_calidad_20260922/politica/kernel_class.csv`). Si no hay una etiqueta
     memory_bound defendible, **no usarlo**.
   - Los tamaños con mejor ganancia en `best_level_by_kernel.csv` (`dual_spmv_gpu_N200000000`,
     `dual_stencil_gpu_N36864`, `dual_axpy_gpu_N1280000000`) **no aparecen con esos ids** en el catálogo de Fase 1 (hay,
     por ejemplo, `dual_stencil_gpu_N4096`, `dual_spmv_gpu_N2500000`). Localizar de dónde salen (campaña histórica) y si
     el binario con esos argumentos está disponible en `~/hyperion-kernels` de pacca. Si no, usar el tamaño de catálogo
     más grande de la misma familia con argumentos escalados (patrón de `composite_gpu_unseen.PHASES`) y declarar la verdad
     de fase igual que ahí.
5. **Familias inéditas de memoria**: listar las 16 familias del entrenamiento de GPU
   (`docs/libro/datos/gpu_calidad_20260922/`) y elegir kernels memory_bound de GPU del catálogo **fuera** de esas 16, con
   OI claramente por debajo del ridge (p. ej. `gpu_stream_bw` con otros argumentos, u otros de RAJAPerf no usados).
   `gpu_rajaperf_stream_copy` es del linaje de stream y **no** cuenta como inédito.

---

## 4. Paso 2: aplicaciones del escenario E

Crear dos compuestas nuevas en `fase3_daemon/composite_apps/`, siguiendo el patrón de `composite_gpu_unseen.py`
(lista `PHASES` de `(id, exec_args, verdad_de_fase)` y `run_composite` de `composite_known.py`):

- `composite_gpu_memdom_known.py` (**E-A, familias vistas**): tres fases memory_bound de familias con mejora de F1 >= 10%
  en Fase 2 (candidatas en orden de preferencia: dual_stencil, dual_spmv, dual_axpy, rodinia_lud, jacobi_2d si pasa el
  chequeo 4) y **una** fase compute_bound corta (`gpu_cutlass_simt_dgemm_n4096` o `rodinia_lavamd -boxes1d 100`) para que
  haya conmutación.
- `composite_gpu_memdom_unseen.py` (**E-B, familias inéditas**): fases memory_bound de familias fuera del entrenamiento
  (chequeo 5) y la misma fase compute corta.

Objetivos de diseño (verificar midiendo, ver paso 4):
- Cada fase de memoria **>= 20 s** (el agente decide una vez por fase, con 3.7 s de permanencia mínima y 0.3 s de
  asentamiento).
- **~80% del tiempo del ciclo en memoria.**
- 2 ciclos por aplicación, 1 s de reposo entre fases (igual que la matriz A; sin el reposo el agente no abre fase nueva,
  job 7637).
- Checksum verificado y criterio de éxito del catálogo, como en las compuestas existentes.
- Tests en `fase3_daemon/composite_apps/tests/` al estilo de los existentes (construcción de entradas, falla cerrada si
  falta un kernel).

Criterio de selección a declarar en el libro, textual: *"familias memory_bound cuya ganancia de EDP con F1 en la Fase 2 fue
de al menos 10%, más una variante con familias no vistas en el entrenamiento"*.

---

## 5. Paso 3: cambios en `scripts/pacca/hyp_fase4_matrix.sbatch`

1. **Alcance nuevo `gpumem`** en `app_cmd` y en el plan de celdas:
   - `gpumem:known` -> `composite_gpu_memdom_known.py`; `gpumem:unseen` -> `composite_gpu_memdom_unseen.py`.
   - Brazos: `base`, `base_noturbo`, `sombra`, `activo_gpu` (opción a), `activo_gpu_cpuobs` (opción b).
2. **Brazo `base_noturbo`** (sin agentes): antes de lanzar la app, `sudo -n /usr/local/bin/set_turbo_state 1` y verificar
   que `no_turbo` quedó en `1` (si no, marcar la celda con error y no medir). Al terminar la celda, el `restore_cpu`
   existente ya devuelve el turbo al estado inicial porque `state()` registra `no_turbo`; **verificarlo** en
   `state_final.txt` de la primera celda del pre-vuelo.
   Agregarlo también a los alcances `cpu` y `joint` (solo si se pide en `ONLY_ARMS`), para la corrida complementaria del
   paso 5.
3. **Brazos de agentes en `gpumem`**:
   - `sombra`: agente de GPU en `--arm sombra`; sin agente de CPU.
   - `activo_gpu`: agente de GPU en `--arm activo` con la política versionada `$ROOT/fase3_daemon/policy_table.yaml`;
     **sin agente de CPU**.
   - `activo_gpu_cpuobs`: igual, más el agente de CPU con `launch_cpu_daemon.py --policy-table
     $ROOT/fase3_daemon/policy_table.yaml --arm sombra` (política versionada, `no_actuar` en ambas clases: no escribe
     nada). Mismos flags de CPU que hoy (`--perf-cpus 0,1,2,3 --collector-cpu 6 --consumer-cpu 7 --interval-ns
     $INTERVAL_NS`, ORT de 1 hilo por default).
   - Para `gpumem` usar la política versionada, **no** `policy_activo.yaml` (esa tiene la CPU experimental).
4. **No cambiar** el comportamiento de los alcances y brazos existentes (la matriz A debe poder reproducirse igual).
5. `bash -n` del script y una corrida con `ONLY_*` mínima en el pre-vuelo.

## 6. Paso 3b: cambios en `fase4_evaluacion/analyze_matrix.py`

- Parámetro `--ref-arm` (por defecto `base`) y, cuando exista `base_noturbo` en los datos, imprimir **las dos** tablas de
  razones: frente a `base` y frente a `base_noturbo`.
- Columna adicional **EDP de GPU sola** = `E_gpu x T` y su razón, además del EDP del nodo (la energía ya se registra por
  separado).
- Puntuación de clasificación: sin cambios (ya separa CPU y GPU por prefijo de kernel; confirmar que los ids nuevos de GPU
  caen del lado GPU: la regla actual es `startswith(("gpu_", "rodinia_"))`; los `dual_*_gpu_*` **no** empiezan así y se
  clasificarían como CPU. **Corregirlo** (p. ej. usar el campo `device` del catálogo o `"_gpu_" in id`) y añadir un test.
- Tests en `fase4_evaluacion/tests/test_analyze_matrix.py`.

Antes de correr, añadir en `Plan_Fase4_Evaluacion.md` (sección de escenarios) la fila del escenario E con el criterio de
selección, los brazos, la ganancia esperada (sección 7) y la frase: *"se reporta salga como salga; A y C siguen siendo la
referencia"*. Commit de eso **antes** de la corrida (queda la marca de tiempo de la declaración).

---

## 7. Paso 4: pre-vuelo (~20 min, un job)

`ONLY_SCOPES=gpumem ONLY_ARMS="base base_noturbo activo_gpu activo_gpu_cpuobs" REPS_GENERAL=1 CYCLES=1`,
`OUT=$HOME/hyperion-results/final/fase4_E_preflight`, `--job-name` con `hyp_` (el sbatch ya lo trae).

Criterios para seguir (todos):
- Todas las celdas con `app_rc=0`, `state_ok=1`, daemons con rc 0.
- En `base_noturbo`, `no_turbo=1` durante la app y restaurado al final.
- En `activo_gpu`, **cada** fase de memoria tiene una decisión `memory_bound` con `written=true` y F1 aplicado; anotar
  cuántas abstenciones hubo. En E-B, si hay abstención en la mayoría de fases de memoria, **parar y consultar**.
- Duración real de cada fase >= 20 s y fracción de memoria ~80% (de `phases.jsonl`).
- Si algo falla: corregir, repetir el pre-vuelo. No lanzar la matriz con un supuesto sin verificar.

Ganancia esperada (a recalcular con el chequeo 3): 0.8 (fracción en memoria) x ~15% (F1 en esas familias) x ~0.56
(dilución por la energía de CPU) ~ **6 a 7% de EDP del nodo** frente a REF, **~12% de EDP de GPU sola**. Frente a
`base_noturbo`, algo más (la CPU sin turbo gasta menos en la línea base, pero también es más lenta: no predecir el signo).

## 8. Paso 5: corridas completas

Una sola cola, en este orden (sin `--dependency`: se someten seguidas y la cola FIFO las ordena; si una falla, **revisar
antes** de que corra la siguiente o cancelarla, es nuestra):

1. **Matriz E**: `ONLY_SCOPES=gpumem ONLY_ARMS="base base_noturbo sombra activo_gpu activo_gpu_cpuobs"`,
   `OUT=$HOME/hyperion-results/final/fase4_E`, 3 reps. 2 aplicaciones x 5 brazos x 3 = 30 celdas.
2. **Complemento REF sin turbo para A/C**: `ONLY_SCOPES="cpu joint" ONLY_ARMS="base base_noturbo activo"`,
   `OUT=$HOME/hyperion-results/final/fase4_C_noturbo`, 3 reps. Incluye `base` y `activo` en la misma corrida para que la
   comparación con `base_noturbo` no dependa de deriva entre días. 2 alcances x 2 apps x 3 brazos x 3 = 36 celdas.

Estimar la duración con los tiempos de fase medidos en el pre-vuelo y avisar al usuario antes de encolar si pasa de ~4 h.

## 9. Paso 6: análisis y libro

- Correr `analyze_matrix.py` (en pacca) sobre `fase4_E`, `fase4_C_noturbo` y `fase4_C_ort1`; copiar los `results.csv` y la
  puntuación a `docs/libro/datos/fase4_20260924/` (mismo mecanismo que ya existe; ver
  `docs/libro/scripts/generar_figuras_fase4_20260924.py`).
- Extender ese script (no crear otro): tabla y figura del escenario E (EDP del nodo y de GPU sola, frente a REF y a REF sin
  turbo, por brazo), tabla de REF sin turbo para A/C, y actualizar las tablas de C.
- Libro (`docs/libro/secciones/03_resultados.tex`, sección `sec:resultados-fase4`; metodología
  `sec:metodologia-fase4-escenarios` en `02_metodologia.tex`; discusión y conclusiones):
  - E como escenario de aplicabilidad, con su criterio de selección textual.
  - REF con turbo como referencia principal siempre; REF sin turbo como control, nunca como titular.
  - Resultados de A, C y E juntos; nada se omite por ser negativo.
  - Estilo: sin guiones largos `---` en la prosa, sin narrativa de bugs del instrumento (solo metodología final y resultado).
  - Compilar en una carpeta aparte (`latexmk -outdir=<scratch>`) porque el editor del usuario compila `main.pdf` en la
    misma carpeta y pisa los auxiliares; con `\input` de tablas, el fragmento debe traer el `tabular` completo
    (un `\input` dentro de `tabular` rompe la alineación).
- **No hacer commit del libro sin preguntar**: el usuario tiene cambios propios sin confirmar en
  `02_metodologia.tex` y `A1_anexo_metodologia_detalle.tex`.

## 10. Criterios de interpretación (fijados antes de ver los datos)

- **Mejora** frente a REF: mediana de EDP del nodo del brazo activo < 1.00 **y** las 3 repeticiones del activo por debajo
  de las 3 de REF (con 3 contra 3 el p exacto mínimo es 0.05; reportar razón, rango y p).
- **Neutral**: razón dentro de ±2% o solapamiento de repeticiones.
- (a) frente a (b): la diferencia entre `activo_gpu` y `activo_gpu_cpuobs` es el costo de observar la CPU; se reporta
  aparte.
- Si E no mejora: se reporta igual y se discute que el techo de la Fase 2 no se traslada a aplicaciones compuestas (con
  la causa medida: latencia, abstención o dilución).

## 11. Registro de respuestas a los chequeos (2026-09-24)

| Chequeo | Respuesta | Fuente |
|---|---|---|
| 1. Latencia de F1 tras inicio de fase | F1 se aplica ~8 s (7.9 a 8.6 s) después del inicio de cada fase de memoria; la decisión de compute llega a los ~3.7 s. Una fase de 18 s pasa ~55% en F1; una de 50 s, ~84%. Por eso las fases de E duran 30 s o más (la compuerta exige >= 25 s) | `cells/gpu_known_activo_{1,2,3}` de `fase4_matrix_main` |
| 2. Re-decisión tras abstención | No: hay una decisión por fase; tras `revisar` toda la fase queda en el reloj nativo (`gpu_known_activo_3`: triad `revisar` a los 8.6 s, sin F1 en esa fase). Limitación asumida; umbral 0.90 sin tocar | mismas celdas y `gpu_loop_main.cpp` |
| 3. Energía de la ganancia de Fase 2 | Solo GPU (`gpu_energy_delta_mj_sum`, NVML): el 8.9% es EDP de GPU sola. La dilución por la energía de CPU aplica al EDP del nodo, como se estimó; `analyze_matrix.py` ahora reporta también EDP de GPU sola (`EDPg`) | `historical_gpu_relaxed020_20260922.csv` |
| 4. Kernels E-A elegidos | dual_stencil_gpu_N36864 (60 s), dual_spmv_gpu_N200000000 (55 s), dual_axpy_gpu_N1280000000 (50 s), gpu_cutlass_simt_dgemm_n4096 (31 s, compute). Los tres dual_* existen en el catálogo con checksum, marcados `intermedio`; su clase memory_bound sale de `kernel_class.csv` (margen 1.0, no ambiguos, OI 0.09 a 0.27 frente a ridge 3.36). `gpu_rajaperf_jacobi_2d` también es memory_bound en `kernel_class.csv`, pero no se usó (no hacía falta) | catálogo y `kernel_class.csv` |
| 5. Kernels E-B elegidos | `gpu_stream_bw --arraysize 100000000 --numtimes 5000` (36 s), `rodinia_myocyte 1000000 1 0` (48 s), ambos memory_bound, y `rodinia_lavamd -boxes1d 100` (compute, ~10 s). Descartados por el sondeo: `rodinia_backprop` (falla con 4194240 y 8388480), `gpu_rajaperf_indexlist_3loop` y `reduce3_int` (5 a 6 s, sin argumentos para alargarlos). E-B queda con 2 fases de memoria, no 3. Ninguno está entre las 16 familias del entrenamiento | sondeo job 7651 |
| Sondeo de duración | job 7651, completado en 5 min, sin errores | `logs/fase4_E_survey_7651.out` |
| Pre-vuelo | job 7652 | |
| Matriz E | job 7653 | |
| Complemento sin turbo | job 7654 | |

## 12. Cambios respecto del diseño original y hallazgos de la implementación

- **1 ciclo por aplicación** (no 2) en el alcance `gpumem`: con fases de 30 a 60 s, 2 ciclos duplicarían la duración de la matriz.
  Cada aplicación tiene 3 fases de memoria y 1 de compute (E-A ~196 s, ~84% en memoria). Variable `CYCLES_GPUMEM`.
- **Compuerta automática** (`fase4_evaluacion/gate_preflight_E.py`, invocada por `scripts/pacca/hyp_fase4_E_run.sbatch`): las
  corridas largas (7653 y 7654) leen el pre-vuelo y terminan sin medir si falla algún criterio (celdas faltantes, rc distinto
  de 0, fase de memoria < 25 s, fracción de memoria < 70%, F1 aplicado en < 2/3 de las fases de memoria).
- **Sondeo previo** (`scripts/pacca/hyp_fase4_E_survey.sbatch`, job 7651): mide con argumentos escalados los candidatos de E-B
  (`survey_kernels.py` acepta ahora `id::argumentos`).
- **Los jobs 7652 a 7654 se sometieron retenidos** (`scontrol hold`) hasta confirmar E-B con el sondeo; se liberan juntos y
  conservan su orden relativo.
- **REF sin turbo probablemente iguale a REF.** En la matriz A, `activo_f0` (3.2 GHz, turbo apagado) tardó lo mismo que
  `sombra` (1.207 frente a 1.205 en CPU A; 1.155 frente a 1.154 en CPU B) y con menos energía de CPU (1.591 frente a 1.656).
  Es decir, con turbo REF no corre más rápido que 3.2 GHz en los núcleos 0 a 5 (su `scaling_max_freq` es 3200000). Esto
  **corrige la causa 2** de la sección 1: perder el turbo no explica el exceso de tiempo de la CPU (era el costo del agente);
  lo que sí se ve es un ahorro de energía de CPU de ~4% con F0, sin costo de tiempo. `base_noturbo` lo confirma o lo refuta.
- **Error latente corregido**: `analyze_matrix.py` trataba como CPU los `dual_*_gpu_*`; ahora `is_gpu_kernel` usa el
  dispositivo del id (con test).

## 13. Resultado del pre-vuelo y desviación decidida por el usuario (2026-09-24)

Pre-vuelo (job 7652, 8 celdas, 20 min, sin errores; `fase4_E_preflight`):
- **E-A**: el agente aplicó F1 en 2 de las 3 fases de memoria (la tercera, dual_axpy, quedó con una decisión memory_bound sin
  escritura, es decir sin cambio de reloj). La primera decisión llegó a los 13 a 16 s del inicio de la fase (no 8 s). Con una
  repetición, la energía de GPU bajó 6.5% (15 293 J frente a 16 363 J de REF) con la misma duración (187.2 s frente a 187.4 s).
  Es un solo dato, sin valor estadístico.
- **E-B**: BabelStream se abstuvo (`revisar`, confianza 0.57 y 0.67); myocyte no generó ninguna decisión en 40 s (el agente no
  vio actividad de GPU suficiente para abrir fase). El agente no actuó en ninguna fase de memoria inédita.
- **REF sin turbo**: en E-A `base_noturbo` fue igual a `base` (186.8 s y 28 574 J de CPU frente a 187.4 s y 28 621 J); en E-B
  93.2 s frente a 88.4 s (una repetición). Turbo restaurado en todas las celdas (`state_ok=1`).

La compuerta bloqueó las corridas largas (7653 y 7654) por E-B, como está diseñada. Decisión del usuario, consultado según
la regla de este plan: **correr E-A y E-B tal cual**, con la compuerta informativa (no bloqueante) para E-B y estricta para
E-A, y reportar E-B como resultado (límite de generalización del clasificador de GPU a memoria inédita, no un fallo de
la corrida). El complemento sin turbo (7654) se rehízo como job 7656 con una compuerta que solo exige `base_noturbo` sano.

## 14. Escenario D: LAMMPS como aplicación real de terceros (2026-09-24)

- **Aplicación**: LAMMPS 2023 con paquete GPU del clúster (`/opt/ohpc/pub/lammps/2023_gpu/bin/lmp`, API CUDA), inputs de benchmark de
  `/usr/local/src/lammps/lammps-23Jun2022/bench` (lj y eam a 256 000 átomos, chain y rhodo a 32 000), un proceso MPI en los núcleos
  0 a 3, `-sf gpu -pk gpu 1`. Pasos fijados por calibración (job 7678, ~60 s por corrida): lj 4400, eam 3200, chain 8000, rhodo 1500.
- **Sondeo (jobs 7668 a 7674)**: con el agente de GPU en sombra, la GPU queda casi en reposo (utilización mediana 6 a 12%, 765 MHz,
  37 a 38 W) y no hay fases a escala de segundos; el agente no abre ninguna fase. Por eso D no puede mostrar mejora de GPU: mide
  validez externa sobre el costo (¿el agente es inocuo en una aplicación real sin fases de memoria de GPU?).
- **Arranque de LAMMPS**: el módulo `lammps/2023_GPU` no carga bajo `set -u`; `lmp` sin lanzador se cuelga en `MPI_Init`. Se usa
  `mpirun -np 1` de nvhpc con `OPAL_PREFIX`, `LD_LIBRARY_PATH` explícito (fftw, mpi de nvhpc) y `OMPI_MCA_*` (ver
  `hyp_fase4_matrix.sbatch`, alcance `lammps`).
- **Corridas**: pre-vuelo job 7679 (1 repetición, base y activo_gpu_cpuobs, 8 celdas) y matriz job 7680 (RUN_KIND=D, 48 celdas: 4
  benchmarks x base, sombra, activo_gpu, activo_gpu_cpuobs x 3 repeticiones), protegida por `gate_preflight_E.py --lammps`.
- **Expectativa declarada antes de medir**: EDP del nodo con el agente de GPU ~1.00 frente a REF; con el agente de CPU, ~+6% de
  energía de CPU (costo fijo medido). No se espera mejora.
