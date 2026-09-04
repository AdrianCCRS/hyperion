# Seguimiento de cambios al plan de realineación

Este documento registra decisiones técnicas tomadas después de la redacción de
`Plan_Detallado_Realineacion_Hyperion.md`. No reemplaza el plan ni implica por
sí solo aprobación académica del director: conserva la motivación, el impacto y
el estado de implementación de cada ajuste para que pueda revisarse y
ratificarse sin perder trazabilidad.

## Convención de identificadores

Se adopta la forma `F<fase>-<alcance>-<secuencia>`:

- `F1-CPU-001`: cambio 001 de Fase 1 que afecta CPU.
- `F1-GPU-001`: cambio 001 de Fase 1 que afecta GPU.
- `F2-XDEV-001`: cambio 001 de Fase 2 que afecta ambos dispositivos.

Los alcances previstos son `CPU`, `GPU`, `XDEV` (ambos dispositivos) y `GEN`
(sin dispositivo específico). La numeración es independiente por combinación
de fase y alcance. Esta convención mantiene el formato propuesto
`F1-CPU-001` y permite ampliarlo sin volver ambiguos los cambios transversales.

## Resumen

| ID | Decisión | Estado |
|---|---|---|
| `F1-CPU-001` | Sustituir el supuesto contador de stalls de backend por `CYCLE_ACTIVITY.STALLS_MEM_ANY` | Implementado y validado |
| `F1-CPU-002` | Alinear el dataset de entrenamiento CPU con los intervalos reales de `uncore_imc` | Implementado (offline) |
| `F1-CPU-003` | Renombrar `llc_miss_rate` → `cache_miss_rate` (evento genérico, no LLC demostrada) | Implementado y validado; validación física del PMU pendiente en paccaA100 (no bloquea) |
| `F1-GPU-001` | Usar Nsight Compute solo para construir la verdad Roofline offline y NVML como proxy ligero online | Parcialmente implementado (captura completa; entrenador GPU pendiente) |
| `F1-GPU-002` | Caracterizar la cadencia efectiva de NVML y medir `T_transición_gpu` bajo carga | Infraestructura implementada (probe C++ + lógica pura + agregador + comparador de cadencia + pruebas); **pendiente de medición real en paccaA100** |
| `F1-GPU-003` | Contrato de granularidad GPU + dataset intermedio por corrida/fase con agregación NVML robusta | Implementado y validado; a la espera de campaña GPU real para poblarlo |
| `F1-GPU-004` | Convergencia y procedencia de la verdad Roofline GPU (`ncu`) por kernel | Parcialmente implementado (parser + lógica de convergencia + runbook, validados); **ejecución de `ncu` bloqueada por hardware** |
| `F1-XDEV-001` | Tratar las 232 entradas como banco de candidatos y seleccionar por cobertura Roofline/familia | En preparación (manifiestos de cribado creados; generador de manifiesto definitivo implementado) |
| `F1-XDEV-002` | Recalibrar y congelar el warmup de cada candidato, plegado dentro de la campaña real (sin mini-campaña aparte) | Implementado (módulo + CLI + `warmup_seconds_override` de manifiesto + `repostprocess_campaign.py` + pruebas); **calibración real bloqueada por campaña en paccaA100** |
| `F1-XDEV-003` | Usar cribado de frecuencia reducido solo para selección y rejilla fina en el dataset definitivo | Implementado (generador de manifiesto por lista congelada + resolución MHz→fracción + gate); manifiestos finales pendientes del resultado del cribado |
| `F1-XDEV-004` | Análisis Pearson/Spearman/VIF y contrato versionado de features antes de entrenar | Implementado y validado con fixtures; **selección definitiva pendiente del dataset real** |
| `F1-XDEV-005` | Orquestar el cribado hasta un informe de utilidad, con `ncu` obligatorio antes de GPU | Implementado; pendiente ejecución real en paccaA100 |
| `F2-XDEV-001` | Diagnosticar cobertura Roofline y calidad antes de seleccionar/balancear entrenamiento | Implementado (sin datos de campaña aún) |
| `H` (gate) | Auditoría única de readiness pre-entrenamiento (PASS/FAIL/BLOCKED por gate) | Implementada y validada; a la espera de un dataset real para dictaminar |

---

## F1-CPU-001 — Sustitución del contador de stalls

**Fecha de registro:** 2026-09-03
**Estado:** implementado en captura, esquema, postproceso, entrenamiento,
pruebas y manual.
**Commit de implementación:** `98370d1`.

### Situación anterior

La señal se exportaba como `stalled_cycles_backend`. En paccaA100, el evento
genérico de Linux no está mapeado para el PMU de Ice Lake-SP y el instrumento
usaba como alternativa el evento crudo `CYCLE_ACTIVITY.STALLS_TOTAL`. Aunque
este evento abre correctamente, cuenta stalls de ejecución totales y no stalls
atribuibles específicamente al backend o a memoria. Su nombre exportado podía
inducir una interpretación física incorrecta en el modelo y en la tesis.

### Decisión

Usar exclusivamente el evento crudo de Ice Lake-SP
`CYCLE_ACTIVITY.STALLS_MEM_ANY` (`event=0xA3`, `umask=0x14`, `cmask=0x14`),
protegido por el gate CPUID familia 6/modelo 106. Renombrar la señal y sus
derivados como:

- `stalled_cycles_mem_any`
- `delta_stalled_cycles_mem_any`
- `stall_mem_ratio`

La señal cuenta ciclos de ejecución bloqueados mientras existe una carga
pendiente en el subsistema de memoria. Es un predictor de presión de memoria;
no es la etiqueta Roofline ni prueba por sí sola que una carga sea
`memory_bound`.

### Consecuencias

- Los datasets anteriores con `stall_backend_ratio` no son compatibles de
  forma automática con el nuevo vector de features.
- Deben repetirse la adquisición, el postproceso y el entrenamiento para
  construir un modelo que use la señal corregida.
- El encoding es específico de la plataforma y no debe abrirse en otra
  microarquitectura sin validación independiente.

---

## F1-GPU-001 — Separación entre verdad offline y proxy NVML online

**Fecha de registro:** 2026-09-03
**Estado:** etiquetado y captura implementados (la corrección de representación
NVML se cerró el 2026-09-03, ver más abajo); entrenador GPU pendiente.

### Restricción observada

En la NVIDIA A100, las llamadas NVML usadas por el proyecto entregan estado
agregado del dispositivo: actividad GPU y de memoria, potencia, reloj SM,
energía acumulada y temperatura. No entregan los contadores
microarquitectónicos por kernel necesarios para medir directamente FLOPs,
bytes, throughput de SM/HBM, ocupación o causas de stall.

Nsight Compute (`ncu`) sí puede obtener FLOPs y bytes por kernel mediante
contadores de hardware, pero es un perfilador: puede requerir control de la
ejecución, serialización y repetición de kernels. Sus resultados no quedan
expuestos posteriormente a NVML y no es adecuado invocarlo en cada decisión
del daemon ligero. Incorporar esas métricas en vivo exigiría diseñar e integrar
un colector CUPTI, medir su intrusión y revisar el alcance del proyecto.

### Decisión

Separar las fuentes y responsabilidades:

1. **Verdad de entrenamiento offline:** perfilar cada kernel/tamaño con `ncu`,
   calcular su intensidad operacional y compararla con el ridge Roofline
   calibrado para su precisión y nivel de frecuencia.
2. **Inferencia online:** utilizar como candidatos las señales NVML de baja
   intrusión `gpu_util_pct`, `gpu_mem_util_pct`, `gpu_power_mw`,
   `gpu_sm_clock_mhz` y `gpu_temperature_c`.
3. **Energía:** conservar `gpu_energy_mj` y `gpu_energy_delta_mj` para EDP y
   derivación de política, no como feature primaria de clasificación. El valor
   acumulado crudo nunca debe entrar al modelo.

### Limitación que debe reportarse

NVML es un proxy, no una medición directa del régimen Roofline. Una carga
`compute_bound` y otra `memory_bound` pueden mantener simultáneamente valores
altos y similares de actividad GPU/memoria. Por tanto, el desempeño del modelo
puede ser bajo incluso con una implementación correcta. El conjunto de cinco
features solo se acepta si, sobre datos reales y con validación
`leave-one-familia-out`, supera claramente la línea base mayoritaria. Un
resultado negativo significa que la observabilidad NVML de la A100 es
insuficiente para generalizar; no debe ocultarse añadiendo `ncu` al vector de
producción.

### Trabajo pendiente

- Agregar las muestras NVML por corrida o fase estable; no tratarlas como
  ejemplos temporalmente independientes.
- Implementar el entrenador GPU separado y su serialización.
- Medir F1 por clase, matriz de confusión, latencia p95/p99 y desempeño por
  familia algorítmica.
- ~~Corregir la representación de métricas NVML opcionales no disponibles.~~
  **Cerrado el 2026-09-03** — ver "Corrección de captura" abajo.

### Corrección de captura — representación de métricas NVML opcionales

**Fecha:** 2026-09-03
**Estado:** cerrado.
**Commit de implementación:** `fix(telemetry): distinguir métrica NVML opcional
no disponible de un 0 real`.

#### Situación anterior

`common/telemetry/src/nvml_reader.cpp` invocaba `nvmlDeviceGetClockInfo`,
`nvmlDeviceGetTotalEnergyConsumption` y `nvmlDeviceGetTemperature` **ignorando
el código de retorno**. Si el driver/GPU no soportaba una de ellas, el valor
quedaba en `0` y el launcher lo escribía en `samples.csv` como `0` (no celda
vacía). Aguas abajo, `fase1_telemetria/postprocess.py` veía
`previous_energy_mj == current_energy_mj == 0` y, desde la segunda ventana GPU,
marcaba `gpu_energy_delta_mj = 0` con **`gpu_energy_valid = True`**: una lectura
energética fabricada que alimentaba el EDP de GPU y la derivación de política.
`gpu_sm_clock_mhz` y `gpu_temperature_c` pasaban como `0` sin ningún bit que los
distinguiera de una medición real.

#### Decisión

Aplicar la misma convención "no medido ≠ 0 real" que ya usan
`stalled_cycles_mem_any` y `UncoreSnapshot::interval_valid`:

1. `GpuSample` (`metrics.hpp`) gana `sm_clock_valid`, `energy_valid` y
   `temperature_valid`.
2. `nvml_reader.cpp` comprueba `NVML_SUCCESS` de cada una de las tres llamadas
   opcionales; si falla, el valor queda en `0` y su bit `*_valid` en `false`.
   Una métrica opcional ausente nunca invalida la lectura de potencia/util.
3. `telemetry_kernel_launcher.cpp` escribe celda **vacía** (no `0`) para la
   métrica cuyo `*_valid` es `false`.
4. `postprocess.py` trata además como ausente cualquier `gpu_energy_mj <= 0`
   (un contador acumulado soportado nunca es 0), y cualquier `gpu_sm_clock_mhz`
   / `gpu_temperature_c` de `0` — red de seguridad para `samples.csv` grabados
   con el contrato viejo. Con esto `gpu_energy_valid` solo es `True` cuando hay
   dos lecturas acumuladas reales consecutivas.

#### Verificación

- Build C++ (`common/telemetry/`) limpio; `ctest` 14/14 (1 skip no relacionado).
- `pytest fase1_telemetria/ fase2_clasificador/ common/` — 582 pasan.
- Test de regresión nuevo
  `test_f1_gpu_001_energia_cero_en_todas_las_filas_no_fabrica_validez`: energía,
  reloj SM y temperatura en `0` en todas las filas ⇒ `gpu_energy_valid` queda
  `False` y las tres columnas quedan `None`; `gpu_power_mw` / `gpu_util_pct`
  reales intactos.

---

## F1-GPU-002 — Cadencia efectiva NVML y `T_transición_gpu` bajo carga

**Fecha de registro:** 2026-09-04
**Estado:** infraestructura de medición **implementada** el 2026-09-04 (ver
"Implementado" más abajo); pendiente la ejecución en paccaA100 con NVML real.
Sigue siendo prerrequisito del barrido GPU y de cualquier actuación GPU, no una
medición que pueda diferirse a Fase 3.

### Hecho establecido y distinción necesaria

`gpu_interval_ns` es la cadencia solicitada al bucle del colector, no la
frecuencia con que NVML actualiza necesariamente una señal. En un barrido real
anterior sobre la A100 de pacca (ocho kernels, 1/5/10/50/100 ms, tres
repeticiones por punto), 100 ms fue frágil para cargas cortas: por ejemplo,
`rodinia_backprop` produjo 9.0 muestras NVML útiles de media a 100 ms, frente
a 150.3 a 5 ms y 80.3 a 10 ms. Por tanto, bajar el sondeo desde 100 ms evitó
perder ubicación temporal y dejar corridas cortas con muy pocas lecturas.

La misma evidencia mostró que las lecturas repetidas no son muestras físicas
independientes. En esa A100, potencia y utilización exhibieron escalones
observados de aproximadamente 105--120 ms: con sondeo a 1 ms se vieron muchos
valores consecutivos idénticos. El conteo de cambios de valor solo es una cota
inferior de actualizaciones físicas (dos actualizaciones pueden devolver el
mismo número); no prueba una tasa universal de refresco NVML y no se traslada
sin medir a `gpu_sm_clock_mhz`.

En consecuencia, ambas afirmaciones son simultáneamente verdaderas:

1. **100 ms es demasiado grueso para algunas corridas y para localizar el
   primer cambio publicado por el sensor.**
2. **5 ms no convierte potencia/utilización en observaciones independientes
   de 5 ms.** Es un sondeo fino que reduce la incertidumbre de cuándo se vio
   un escalón y conserva redundancia deliberada.

La documentación de NVML tampoco autoriza equiparar llamadas con actualizaciones:
para `nvmlDeviceGetUtilizationRates`, NVIDIA declara una ventana interna
dependiente del producto; para `nvmlDeviceGetClockInfo` no publica una cadencia
de refresco general. La evidencia local se conserva en
`old/docs/justifications/report/sections/gpu_interval_ns.tex` y sus CSV.

### Objetivo y unidades de resultado

Medir, bajo carga GPU sostenida, la latencia observable entre solicitar un
cambio de reloj y verificar que el reloj graphics (el dominio de `-lgc`) se
mantiene en el destino; el reloj SM se conserva como señal auxiliar:

\[
T_{actuacion}=t_{estable}-t_{solicitud}.
\]

Se registrará por separado `t_command_return - t_solicitud` para no confundir
el costo de invocar `nvidia-smi` con la respuesta posterior del driver. El
resultado primario no es un único número: es una matriz dirigida
`(reloj_origen, reloj_destino, repetición)`. La dirección importa. Solo después
se deriva `T_transicion_gpu_ns_conservative`, el máximo de las repeticiones y
pares que la política puede solicitar; con tres repeticiones el máximo se
reporta como cota conservadora, no como percentil estadístico estable.

### Etapa A — Caracterizar la observabilidad antes de elegir cadencia

Con el mismo driver, GPU, carga sostenida, aislamiento y colector de la futura
campaña, ejecutar un probe de señal con bordes externos conocidos (actividad
activa/inactiva o marcas monotónicas emitidas por la carga). Probar al menos
5, 10, 50 y 100 ms; 5 ms es el baseline fino y no una conclusión anticipada.
Para cada señal (`gpu_util_pct`, `gpu_mem_util_pct`, `gpu_power_mw`,
`gpu_sm_clock_mhz`, temperatura y energía) preservar crudo y reportar:

- distribución de `delta_timestamp_ns` real entre llamadas, incluido p50/p95;
- cambios consecutivos y duración de escalones como **tasa de refresco
  observada/cota inferior**, no como actualización física demostrada;
- error entre cada borde conocido y la primera observación que lo refleja;
- muestras y escalones observados después de warmup en la corrida más corta;
- overhead del sondeo frente a una corrida sin telemetría.

La cadencia de campaña `q_produccion` será la más gruesa que, frente al baseline
de 5 ms, no pierda bordes observables ni reduzca materialmente los escalones
post-warmup en la carga más corta, mantenga el objetivo de cobertura y tenga
menor o igual perturbación. Si 10 ms satisface esas condiciones, se preferirá
por menor redundancia; si no, se conservarán 5 ms. 100 ms queda descartado si
vuelve a producir cobertura frágil o localización de borde peor que el límite
aceptable medido. La decisión, el driver y los artefactos se versionan; no se
extrapolan a otra GPU ni a otra versión de driver.

### Etapa B — Medición de transición de reloj

Usar una carga CUDA sostenida, con utilización y temperatura verificadas, que
dure varios segundos después del warmup. Para cada par de niveles fijos que
pueda ser solicitado por la política, y también desde `REF` hacia cada nivel
fijo candidato:

1. fijar y confirmar el origen bajo carga;
2. comenzar el sondeo de reloj graphics con la cadencia fina seleccionada para el
   **probe** (inicialmente 5 ms, aun si `q_produccion` resulta 10 ms);
3. marcar `t_solicitud` inmediatamente antes de invocar `nvidia-smi -lgc` y
   `t_command_return` al recibir su resultado;
4. registrar cada lectura de reloj, utilización, potencia, temperatura,
   razones de throttling si están disponibles y timestamp monotónico;
5. declarar estable el destino solo si hay al menos tres lecturas consecutivas
   dentro de una tolerancia documentada respecto al reloj soportado objetivo,
   con GPU activa y sin throttling que invalide la interpretación;
6. repetir cada transición dirigida al menos tres veces y restaurar el estado
   GPU al terminar, incluso ante fallo.

La tolerancia no se toma de la hoja de datos: debe ser menor que la mitad del
salto al reloj soportado vecino y quedar escrita en el reporte. Si la primera
lectura estable aparece al límite de la resolución de `gpu_sm_clock_mhz`, el
resultado se declara una **cota superior observable**, no una latencia física
exacta. Esa cota sigue siendo válida y segura para fijar permanencia mínima.

### Artefactos, gates y consumo posterior

El módulo futuro debe producir por ejecución `gpu_clock_transition_raw.csv`,
un `gpu_clock_transition_summary.json` y un resumen de matriz CSV. Deben incluir
UUID/modelo GPU, driver/CUDA, clocks soportados, comando, carga/checksum,
frecuencias origen/destino, timestamps, cadencia real, criterio de estabilidad,
réplica, fallos y restauración. No se aceptan resultados agregados sin crudo.

- Si `T_transicion_gpu_ns_conservative` es comparable o mayor que la duración
  de las fases/corridas elegibles, GPU queda en `no_actuar`; ese es un resultado
  válido.
- Si es menor, `derive_policy_table.py` recibe ese valor en
  `--t-transicion-gpu-ns`; sus exclusiones deben conservar el motivo y el
  reporte de origen.
- `min_dwell_ns` del daemon se fija como mínimo a esa cota conservadora. Un
  multiplicador adicional solo se permite tras medir sensibilidad/histéresis,
  nunca como constante implícita.
- La caracterización de cadencia no convierte filas NVML en fases ML
  independientes: el futuro entrenador GPU sigue requiriendo agregación por
  corrida o fase estable (`F1-GPU-001`).

### Implementado (2026-09-04)

Probe independiente y documentado. **No** cambia la cadencia de las campañas,
el entrenador GPU ni `derive_policy_table.py`.

- `common/telemetry/include/telemetry/gpu_transition_analysis.hpp` — lógica
  pura, sin NVML/CUDA: `detect_stability` (declara estable solo con N lecturas
  consecutivas dentro de tolerancia, GPU activa y sin throttling invalidante;
  N por defecto 3), `compute_transition_metrics` (`command_latency_ns`,
  `t_actuacion_ns`, `settle_after_command_ns`, `conservative_upper_bound_ns`
  = cota superior observable segura para `min_dwell_ns`, `optimistic_ns` solo
  como contexto), `compute_cadence_stats` (p50/p95/min/max de
  `delta_timestamp_ns` real) y `analyze_signal_steps` (cambios consecutivos y
  duración de escalones **como cota inferior**, con nota explícita de que
  5 ms no implica información independiente cada 5 ms).
- `common/telemetry/experiments/gpu_clock_transition_probe.cpp` — ejecutable.
  Verifica el reloj **graphics** (el dominio que fija `-lgc`) mediante
  `nvmlDeviceGetClockInfo` y exporta el reloj SM como señal auxiliar;
  `nvidia-smi` solo actúa (`-lgc`/`-rgc`) con `sudo -n` y timeout. El timestamp
  de estabilidad se toma después de recibir la lectura graphics, por lo que la
  cota no antecede a su observación. Fija y confirma el reloj origen bajo carga,
  lanza la carga vía `sh -c`, verifica actividad por NVML, solicita el destino
  a mitad de la carga, registra cada lectura (reloj/util/potencia/temp/energía/
  `throttle_reasons`) con timestamp monotónico, y restaura ante `atexit`,
  SIGINT/SIGTERM, fallo de comando o timeout; una segunda señal no omite la
  restauración.
  Compila CPU-only (imprime aviso `-DWITH_GPU` y sale 2). Produce
  `gpu_clock_transition_raw.csv`, `gpu_clock_transition_summary.json` y
  `gpu_clock_transition_matrix.csv`, con crudo incluso al fallar.
- `fase1_telemetria/gpu_transition/aggregate_transition_matrix.py` — junta los
  `summary.json` de varias corridas, agrupa por par dirigido y deriva
  `T_transicion_gpu_ns_conservative` = **máximo** de `conservative_upper_bound_ns`
  sobre pares y réplicas (nunca promedio). Exige declarar todos los pares de
  política y falla cerrado si falta uno, hay `timeout`/dry-run, restauración no
  confirmada o procedencia incompatible; sin datos estables devuelve `None`.
- Pruebas: `common/telemetry/tests/test_gpu_transition_analysis.cpp` (8 grupos:
  convergencia, primer toque + salida de tolerancia, timeout, timestamps
  irregulares, NVML ausente, GPU ociosa, throttling invalidante vs. el bit de
  `-lgc`, escalones/redundancia, cálculo de `T_actuacion`/latencia/cota) y
  `fase1_telemetria/tests/test_aggregate_transition_matrix.py` (9 casos,
  incluidos dry-run/restauración no confirmada y par requerido ausente).
- Documentación y procedimiento: `fase1_telemetria/gpu_transition/README.md`
  (build `-DWITH_GPU=ON`, Etapa A de cadencia, Etapa B de matriz dirigida con
  ≥ 3 réplicas y `REF→fijo` separado, agregación y alimentación de
  `--t-transicion-gpu-ns`).

**Verificado localmente (sin GPU):** build CPU-only del probe; `ctest`
`common/telemetry` 15/15; `pytest fase1_telemetria common` 530.
**Pendiente en paccaA100:** build `-DWITH_GPU=ON` contra el `nvml.h`/driver
del nodo, confirmar `sudo nvidia-smi -lgc/-rgc` sin contraseña bajo Slurm,
elegir carga CUDA sostenida y calibrar tiempos, correr Etapa A + Etapa B,
registrar el número (o el bloqueo) aquí y en `fase3_daemon`.

### Criterio de salida

El barrido GPU completo solo puede comenzar tras guardar una selección
versionada de `q_produccion` y un reporte de transición reproducible, o tras
documentar que la resolución de la A100/driver solo permite una cota tan alta
que la actuación GPU no es viable. En ambos casos se conserva la evidencia y
la política GPU no puede pasar silenciosamente a `actuar`.

---

## F1-CPU-002 — Dataset alineado con intervalos `uncore_imc`

**Fecha de registro:** 2026-09-03
**Estado:** implementado en postproceso y entrenamiento offline; la adaptación
del daemon a esta cadencia queda pendiente.
**Commit de implementación:** `d0c22377c88b9690fcf998c645ddfe01e456f413`.

### Restricción observada

Los contadores de núcleo y `FP_ARITH_INST_RETIRED` se leen aproximadamente
cada 1 ms, mientras que `perf stat -I` entrega los CAS del controlador de
memoria en intervalos prácticos de aproximadamente 10 ms o más. Los bytes de
un intervalo `uncore` no pueden atribuirse físicamente a una sola ventana de
1 ms.

El postproceso actual actúa correctamente para construir la verdad: suma los
FLOPs de todas las ventanas CPU cubiertas por un intervalo `uncore`, calcula
`OI = FLOPs_sumados / ((CAS_read + CAS_write) * 64)` y difunde esa OI y su
etiqueta a las ventanas cubiertas. Sin embargo, el entrenador todavía recibe
esas ventanas de 1 ms como filas separadas y las submuestrea por corrida. Así,
varias filas correlacionadas comparten una única observación física de bytes y
una misma etiqueta de aproximadamente 10 ms.

### Decisión

Conservar `windows.csv` con resolución de núcleo de aproximadamente 1 ms como
artefacto crudo/auditable, pero construir un dataset de entrenamiento CPU
adicional con **una fila por intervalo `uncore`**. La granularidad declarada de
la clasificación Roofline CPU pasa a ser la del intervalo `uncore`, no 1 ms.

Para cada intervalo deben agregarse los deltas crudos y recalcularse las tasas:

- `ipc = sum(delta_instructions) / sum(delta_cycles)`
- `mpki = 1000 * sum(delta_cache_misses) / sum(delta_instructions)`
- `llc_miss_rate = sum(delta_cache_misses) / sum(delta_cache_references)`
- `stall_mem_ratio = sum(delta_stalled_cycles_mem_any) / sum(delta_cycles)`
- `ips = sum(delta_instructions) / duración_real_del_intervalo`
- `running_ratio = sum(delta_running_ns) / sum(delta_enabled_ns)`
- `freq_khz_observed`: mediana o promedio ponderado por duración, dejando el
  criterio fijado en metadata.

No deben promediarse directamente razones como IPC o MPKI, porque ventanas con
distinta cantidad de trabajo quedarían ponderadas de forma incorrecta.

### Implementación

1. Implementado: el postproceso añade un identificador y límites temporales explícitos del
   intervalo, por ejemplo `uncore_interval_id`, `uncore_t_start_ns`,
   `uncore_t_end_ns` y `uncore_delta_t_ns`.
2. Implementado: produce, sin reemplazar el CSV auditable,
   un archivo como `training_cpu_intervals.csv` con una fila por intervalo.
3. Implementado: rechaza intervalos sin CAS válidos o cobertura utilizable,
   que crucen el warmup o que contengan ventanas PMU/frecuencia degradadas. Registra el
   motivo; no imputar ni recurrir a `cache_misses * 64`.
4. Implementado: el cargador de Fase 2 consume el dataset agregado y registra
   `training_granularity = "uncore_interval"` en la metadata del modelo.
5. Se mantiene la validación externa `leave-one-familia-out`; ninguna fila de una
   familia puede aparecer simultáneamente en entrenamiento y prueba.
6. Implementado: pruebas de agregación, intervalos rechazados y recomputación
   de ratios; el conjunto `FORBIDDEN` mantiene la ausencia de fuga de etiqueta.
7. Pendiente: evaluar el daemon CPU a una cadencia coherente con la evidencia disponible,
   inicialmente alrededor de 10 ms. Una inferencia a 1 ms solo puede
   presentarse como predicción con etiqueta débil de 10 ms, no como verdad
   Roofline medida independientemente a 1 ms.

### Criterio de cierre

Estado: **implementado en la canalización offline CPU**. El cambio se considera cerrado cuando cada fila usada por el entrenador
representa exactamente un intervalo `uncore`, las features y la etiqueta se
calculan sobre los mismos límites temporales, la metadata conserva esa
granularidad y las pruebas demuestran que ningún intervalo se multiplica en
varias observaciones de entrenamiento.

---

## F1-XDEV-001 — Selección por cobertura del plano Roofline

**Fecha de registro:** 2026-09-03
**Estado:** manifiestos de cribado creados y validados contra el catálogo; diez
candidatos del plan pendientes de compilación, catálogo, checksum y
caracterización — los cinco Rodinia GPU son bloqueantes del balance por clase,
ver "Observaciones de revisión" al final de esta sección.

### Problema observado

El catálogo fusionado contiene 232 entradas, pero 172 son variantes `dual_*`
de seis operaciones sobre distintos tamaños. El conteo bruto no representa
232 familias estadísticamente independientes. También mezcla calibradores,
microbenchmarks sintéticos y cargas creadas para el selector CPU/GPU con las
cargas externas que pueden alimentar el clasificador de fase.

El cribado histórico por alpha y margen EDP responde si una carga puede
beneficiarse de DVFS, no si aporta una clase, frontera o patrón nuevo al modelo
`compute_bound`/`memory_bound`. En particular, los nueve sobrevivientes CPU de
RAJAPerf aportan margen real en siete casos, pero se concentran en cuatro
familias de acceso regular; los seis sobrevivientes GPU fueron seleccionados
deliberadamente por presión de DRAM y no balancean la clase compute.

### Decisión

Usar las 232 entradas como banco de candidatos, no como dataset final. La
unidad de diversidad y de validación será la familia algorítmica. La selección
definitiva se hará después de medir, para cada familia y dispositivo:

- fracción de intervalos `compute_bound` y `memory_bound`;
- distribución y distancia de `OI/I_ridge`;
- cobertura y causas de rechazo de calidad;
- presencia/duración de transiciones de fase;
- diversidad de las features disponibles en producción;
- patrón de acceso y validez de FLOPs como medida de trabajo.

El objetivo mínimo sigue siendo 5--6 familias por clase y dispositivo. El
dataset debe incluir extremos, observaciones cercanas al ridge y, cuando la
medición lo confirme, cargas con mezcla temporal. El número de variantes de
una familia no le dará mayor peso estadístico. Alpha y EDP se conservarán para
la política DVFS, pero no serán criterios de inclusión del clasificador.

### Composición estimada

| Alcance | Candidatos externos | Controles/cuarentena | Calibración | Total previsto |
|---|---:|---:|---:|---:|
| CPU | 28 | 10 | 2 | 40 |
| GPU | 18 | 10 | 4 | 32 |
| **Total** | **46** | **20** | **6** | **72** |

De los 46 candidatos externos, 36 ya están representados en los manifiestos
actuales (23 CPU y 13 GPU). Faltan los diez propuestos en §2.1 del plan:
`npb_ep`, cuatro Rodinia CPU y cinco Rodinia GPU. No se añadirán al YAML como
referencias ficticias: primero requieren binario reproducible, checksum,
validación numérica y, en GPU, OI medida con `ncu`.

Los controles `ptrchase`, `phasic_*` y `gpu_phasic_*` pueden validar extremos,
resolución y transiciones, pero no entrenarán el modelo final por ser
sintéticos propios. `dual_*` se reduce provisionalmente a un tamaño por cada
una de sus seis familias y dispositivo; queda en cuarentena hasta validar
procedencia, optimización y etiquetado Roofline. `GAP` y los RAJAPerf enteros
deben demostrar que FLOPs/byte describe trabajo útil antes de ser elegibles.

### Campañas de cribado

- CPU: `campaign_pacca_phase_coverage_cpu_screen.yaml`, 33 kernels ejecutables,
  niveles `REF/F2/F4`, tres repeticiones: 297 corridas.
- GPU: `campaign_pacca_phase_coverage_gpu_screen.yaml`, 23 kernels ejecutables,
  CPU en `REF`, GPU en `REF/F3/F6`, tres repeticiones: 207 corridas.
- Ambos manifiestos declaran `warmup_seconds_override: 0.0` (F1-XDEV-002,
  actualización 2026-09-04): esta misma corrida sirve también como fuente de
  calibración de warmup, sin mini-campaña aparte. El dataset de cribado real
  es el que resulta de re-postprocesar con
  `fase1_telemetria/repostprocess_campaign.py` una vez calibrado y aplicado el
  catálogo — no el `windows.csv` crudo con warmup en 0.

La separación por dispositivo impide que una dependencia o permiso de GPU
bloquee CPU y viceversa. `MAN-02` exige tres repeticiones como mínimo; por eso
la campaña ejecutable no usa la repetición única considerada inicialmente.
Las variantes NPB clase C y los cuatro tamaños adicionales de DWT2D quedan para
una segunda prueba de robustez al tamaño, no cuentan como familias nuevas.

### Criterio de salida

Después del cribado se congelará una lista versionada de familias elegibles.
Solo esas familias pasarán al barrido completo. La evaluación seguirá siendo
`leave-one-familia-out`, con balance o pesos por familia para evitar que
RAJAPerf, NPB por clase o cualquier barrido de tamaños domine la matriz.

### Observaciones de revisión (2026-09-03)

Revisión de los dos manifiestos contra el catálogo real: **validan limpio** —
los 56 `kernel_ref` existen, ninguno tiene `device` incoherente, los 23 de GPU
declaran `operational_intensity_flops_per_byte` y `gpu_precision`, y la
aritmética cuadra (33×3×3 = 297; 23×1×3×3 = 207). Quedan tres puntos abiertos
que deben resolverse antes de interpretar los resultados del cribado.

#### 1. Los criterios de selección deben separarse por dispositivo

La lista de la sección "Decisión" solo es medible en su totalidad del lado CPU.
En `fase1_telemetria/postprocess.py` (bloque de filas GPU) la etiqueta sale de
`context.gpu_operational_intensity`, una **constante por kernel** tomada del
catálogo (`ncu` offline): todas las filas de una corrida GPU comparten la misma
etiqueta por construcción. Por tanto, para GPU:

- «fracción de intervalos `compute_bound`/`memory_bound`» es siempre 100/0;
- «presencia y duración de transiciones de fase» es siempre cero;
- «distribución de `OI/I_ridge`» es un solo punto por (kernel, nivel),
  calculable con el catálogo y la calibración **sin ejecutar nada**.

Lo que el cribado GPU sí aporta, y justifica correrlo, son otras dos cosas:
(a) qué familias quedan lo bastante cerca del ridge como para que su
desplazamiento entre `REF`/`F3`/`F6` les cambie la clase, y (b) si los cinco
proxies NVML separan esas clases — la pregunta abierta de `F1-GPU-001`. La
mezcla temporal intra-corrida solo puede medirse en CPU, donde la OI se calcula
por intervalo `uncore` (`F1-CPU-002`). Redactar dos listas de criterios, no una.

#### 2. El objetivo de balance se decide en compute-bound de GPU

El número operativo no es «46 candidatos» sino cuántas familias sobreviven a
las reglas de cuarentena de esta misma sección (sintéticos fuera, `dual_*` en
cuarentena, GAP y RAJAPerf enteros condicionales): del orden de ~16 familias
CPU y ~11 GPU. Contra el objetivo de 5–6 familias **por clase y por
dispositivo**, el punto tenso previsible es **compute-bound en GPU**: los seis
sobrevivientes RAJAPerf-CUDA se eligieron por presión de DRAM y
`gpu_dgemm_n4096` está declarado como control/infraestructura, con lo que
quedarían poco más que `rodinia_lavamd` y `rodinia_myocyte`.

Consecuencia: los cinco Rodinia GPU sin compilar (`rodinia_cfd` en particular,
propuesto en §2.1 precisamente como refuerzo compute) no son un pendiente
cosmético — están en la ruta crítica del criterio de salida. Se registran como
**bloqueantes declarados** del balance por clase en GPU, no como trabajo
opcional.

#### 3. Cadencia NVML del cribado GPU (preexistente, no introducida aquí)

`gpu_interval_ns: 5000000` (5 ms) es la convención ya usada por las campañas
GPU anteriores, pero §2.2 del plan fija la cadencia real de NVML en ~100 ms por
límite del propio driver. Si eso se confirma, ~20 filas consecutivas repiten la
misma lectura interna. No invalida el cribado, pero debe resolverse antes de
tratar esas filas como observaciones independientes — es el mismo punto que
encabeza el "Trabajo pendiente" de `F1-GPU-001` (agregar por corrida o fase
estable, no como ejemplos temporalmente independientes).

---

## F1-XDEV-002 — Calibración trazable de warmup antes de campaña

**Fecha de registro:** 2026-09-04
**Estado:** pendiente. Es un gate de preparación: no lanzar las campañas de
cribado ni el barrido completo con valores de `warmup_seconds` no verificados.

### Problema observado

`warmup_seconds` no ordena al harness ejecutar una corrida de calentamiento
separada. La telemetría se captura desde el inicio; al postprocesar, las
ventanas CPU cuyo inicio y las muestras GPU cuyo `timestamp` ocurren antes de
`primera_muestra_CPU + warmup_seconds` se preservan en `windows.csv` pero se
marcan `warmup_excluded` y no entran al conjunto utilizable. Por ello, un valor
heredado, puesto conservadoramente o no medido puede descartar datos válidos o
dejar dentro un transitorio de arranque.

Algunos valores actuales están respaldados por mediciones históricas; otros
son fallbacks conservadores. Ninguno debe considerarse automáticamente válido
para un candidato nuevo o para una configuración que cambie binario, tamaño,
afinidad, número de hilos, dispositivo, driver, cadencia del colector o
frecuencias de la campaña.

### Decisión y procedimiento obligatorio

Antes de una campaña de datos, ejecutar una mini-campaña de calibración para
cada candidato y dispositivo con `warmup_seconds: 0`, para conservar todo el
transitorio. Mantener el mismo binario/checksum, argumentos, tamaño, hilos,
pinning, nodo, colector y configuración de frecuencia previstos para la
campaña posterior. Recoger como mínimo tres repeticiones y cubrir referencia y
los extremos de frecuencia que se usarán; como el catálogo vigente admite un
único valor por kernel, se adopta el máximo valor robustamente detectado entre
esas condiciones.

El detector histórico `old/scripts/pacca/measure_warmup.py` analiza IPC en CPU y
`gpu_util_pct` en GPU. Busca dos ventanas móviles consecutivas con
`CV <= 5%`; si no las encuentra, usa segmentación por puntos de cambio y toma
el primer segmento que alcanza 80% de la meseta de actividad. El valor
propuesto es el instante detectado con 20% de margen:

\[
    warmup\_seconds = 1.2 \times t_{detectado}.
\]

En GPU, la cadencia efectiva de NVML puede impedir resolver transitorios muy
cortos. Si no hay señal suficiente para una detección fiable, la carga debe
alargarse o declararse no apta para telemetría NVML; no se debe sustituir por
un valor arbitrario.

### Evidencia que se debe congelar

Para cada `kernel_ref`, registrar junto al cambio de catálogo: checksum y
argumentos medidos; dispositivo y niveles de frecuencia; identificadores de
las tres corridas; señal usada; método (`cv_threshold` o `changepoint`);
instante bruto, margen aplicado y valor final; y estado de confianza.
`fallback_conservative` solo es admisible si incluye razón, duración y riesgo
documentados; no equivale a «warmup medido».

### Criterio de salida

Los manifiestos de cribado `F1-XDEV-001` solo pueden lanzarse después de que
todos sus kernels tengan una de estas dos condiciones explícitas: (a) warmup
calibrado y trazable, o (b) exclusión razonada del candidato por falta de
señal/aptitud de medición. Tras congelar esos valores, no se modifican durante
las repeticiones de una misma campaña.

---

## F1-XDEV-003 — Rejilla fina para el dataset definitivo

**Fecha de registro:** 2026-09-04
**Estado:** decisión adoptada; pendiente materializar los manifiestos finales
después de seleccionar el catálogo.

### Problema observado

La rejilla gruesa histórica muestreó `REF` y cinco puntos fijos en CPU
(`F0`--`F4`). El salto superior de 3,2 a 2,6 GHz dejó sin observar precisamente
la región en la que una reducción pequeña de frecuencia puede conservar el
presupuesto de rendimiento y reducir energía. El análisis posterior no puede
concluir que no existe un óptimo interior si el experimento no midió esa zona.
El mismo fenómeno apareció en GPU: el salto de 1410 a 1110 MHz era demasiado
grande para estudiar presupuestos de degradación pequeños.

El repositorio ya conserva dos manifiestos que documentan esa evidencia y las
rejillas suplementarias usadas entonces:
`campaign_pacca_cpu_fine_grid.yaml` y
`campaign_pacca_gpu_fine_grid_dataset.yaml`. Son antecedentes experimentales,
no los manifiestos definitivos del catálogo nuevo.

### Decisión

Separar dos propósitos que no requieren el mismo costo:

1. **Cribado de cobertura:** conservar las rejillas reducidas de
   `F1-XDEV-001` (`REF/F2/F4` en CPU y `REF/F3/F6` en GPU). Su objetivo es
   descartar candidatos, comprobar señal y estimar cobertura Roofline; no
   derivar el óptimo energético ni alimentar por sí solas el dataset final.
2. **Campaña definitiva:** ejecutar los kernels/familias seleccionados sobre
   una rejilla fina que conserve los extremos y aumente la resolución en la
   zona alta, donde la rejilla gruesa ya mostró pérdida de información.

Como punto de partida reproducible, la rejilla CPU unificada debe incluir los
niveles históricos y los suplementos: `REF`, 3200, 3100, 3000, 2900, 2800,
2600, 2400, 2200, 2000, 1400 y 800 MHz. La rejilla GPU de referencia es `REF`,
1410, 1350, 1290, 1230, 1170, 1110, 810, 510 y 210 MHz. Los valores se deben
resolver y verificar contra los relojes realmente soportados por el nodo en la
sesión de campaña; no se aceptan únicamente por aparecer comentados en un YAML.

### Consecuencias de implementación

- Crear manifiestos nuevos después de congelar el catálogo; no ampliar los
  manifiestos de cribado ni reutilizar como finales los históricos de siete o
  nueve kernels.
- Ejecutar calibración Roofline por dispositivo, precisión y nivel de
  frecuencia de la rejilla definitiva.
- Verificar el reloj observado bajo carga en cada nivel y conservar los
  rechazos, sin interpolarlos como mediciones válidas.
- Remedir los márgenes de potencia/actividad GPU de los niveles intermedios que
  en los YAML históricos figuran como interpolados.
- Aplicar antes `F1-XDEV-002` (warmup) y, para GPU, cerrar `F1-GPU-002`
  (cadencia efectiva y `T_transición_gpu`).
- Derivar cualquier tabla de frecuencia óptima o análisis EDP solamente del
  barrido fino; la mini campaña sigue siendo evidencia diagnóstica.

### Criterio de salida

Cada kernel elegido debe tener observaciones aceptadas en todos los niveles
aplicables de la rejilla fina, o una exclusión explícita y documentada. El
reporte final debe demostrar cobertura de la región alta y no presentar la
ausencia de un óptimo interior como resultado si existen huecos de frecuencia
sin medir.

---

## F2-XDEV-001 — Diagnóstico de cobertura y selección reproducible

**Fecha de registro:** 2026-09-03
**Estado:** implementado como diagnóstico offline; pendiente de ejecutar sobre
las campañas de cribado y de fijar la política de balance.

### Decisión

No balancear ni reescribir el dataset físico de Fase 1. `windows.csv` conserva
la traza auditable y `training_cpu_intervals.csv` conserva un intervalo uncore
por fila. La selección de familias y cualquier balance ocurren en Fase 2,
después de medir la cobertura real y dentro del conjunto de entrenamiento de
cada fold; el fold de prueba se deja intacto.

### Implementado

Se añadió `fase2_clasificador/analysis/phase_coverage.py`, accesible mediante
`fase2_clasificador/run_phase_coverage.py`. El comando recibe un directorio e
identificador de campaña y escribe, sin modificar las fuentes:

- `family_class_frequency_summary.csv`: filas utilizables, clase, familia,
  frecuencia, proximidad al ridge y mediana de `log2(OI/I_ridge)`;
- `kernel_quality_summary.csv`: filas totales/utilizables/rechazadas y la causa
  principal de rechazo;
- `phase_coverage_report.json`: metadatos de entrada, familias compute,
  memory o mixtas, cobertura por clase y artefactos generados.

En CPU solo acepta para cobertura las filas
`training_quality_status="ok"` de `training_cpu_intervals.csv`. La versión
inicial de GPU leía `windows.csv` y bloqueaba su uso como filas independientes.
Ese pendiente quedó resuelto por `F1-XDEV-005`: ahora lee
`training_gpu_phases.csv`, exige calidad de fase y `verdict.json` aceptado, y
declara una corrida o fase alineada como unidad de observación.

Se añadieron pruebas herméticas para intervalos CPU válidos/rechazados y para
el guardarraíl GPU. El módulo no cambia `train_phase.py`, no entrena modelos,
no selecciona familias automáticamente y no altera los CSV fuente.

### Pendiente y orden de continuación

1. Ejecutar los manifiestos `F1-XDEV-001` y conservar los reportes emitidos.
2. Revisar la cobertura medida frente al mínimo de 5--6 familias por clase y
   dispositivo; compilar/caracterizar los candidatos bloqueantes si falta una
   clase, especialmente compute-bound GPU.
3. Congelar una lista versionada de familias elegibles para el barrido completo.
4. Implementar en `train_phase.py` un balance configurable por familia/clase,
   aplicado solo al índice de entrenamiento de cada fold LOFO y registrado en
   la metadata del modelo. Las cuotas no se fijan antes de observar la mini
   campaña.
5. ~~Implementar la agregación GPU por corrida o fase estable antes de crear
   el entrenador GPU.~~ Resuelto por `F1-XDEV-005`; sigue pendiente implementar
   el entrenador GPU que consuma esas filas.

---

## F1-CPU-003 — `llc_miss_rate` → `cache_miss_rate` (evento genérico, no LLC demostrada)

**Fecha de registro:** 2026-09-04
**Estado:** implementado y validado localmente; validación física del PMU en
paccaA100 pendiente (no bloquea el entrenamiento).

### Problema

`fase1_telemetria/postprocess.py` calculaba `delta_cache_misses /
delta_cache_references` y exportaba la columna como `llc_miss_rate`. Esos
deltas vienen de los eventos **genéricos** `PERF_COUNT_HW_CACHE_MISSES` /
`PERF_COUNT_HW_CACHE_REFERENCES` (`common/telemetry/src/perf_reader.cpp:189-193`).
El kernel traduce cada evento genérico a un evento del PMU concreto, y esa
traducción **no está documentada como exclusivamente LLC/L3** ni verificada para
el Ice Lake-SP de paccaA100. El nombre `llc_miss_rate` afirmaba una semántica de
último nivel sin evidencia; `L2_LINES_IN_ALL` (otro evento del harness) tampoco
es LLC y no alimenta esta columna. El entrenador CPU además llevaba a la vez
`mpki` y `llc_miss_rate`, dos caminos de la misma señal.

### Decisión

1. Renombrar la feature a `cache_miss_rate` (y `miss_rate_relative` →
   `cache_miss_rate_relative`, misma cantidad subyacente) en esquema,
   postproceso, entrenador, pruebas y documentación. Los dos nombres **no**
   coexisten: nunca se produce el nombre viejo.
2. Compatibilidad de lectura: `train_phase.load()` renombra
   `llc_miss_rate → cache_miss_rate` si un CSV histórico trae el nombre viejo.
3. La referencia de calibración `calibration_references.miss_rate_p95` queda como
   está: ya es genérica ("miss rate"), sin afirmación de LLC.
4. Diagnóstico ejecutable en paccaA100 (`fase1_telemetria/diagnose_cache_event.py`)
   que registra, de solo lectura, a qué evento del PMU se traduce el alias
   genérico (`perf list`, `perf stat -v`, sysfs PMU) y emite un veredicto
   conservador. No cambia nada por sí solo.

### Archivos modificados

- `fase1_telemetria/postprocess.py` (11 ocurrencias + comentario de rationale)
- `fase2_clasificador/training/train_phase.py` (`FEATURES`, `LEGACY_COLUMN_RENAMES`, `load()`)
- `fase2_clasificador/README.md`, `MANUAL_ESTUDIANTES.md`
- `fase1_telemetria/tests/test_postprocess.py`, `fase2_clasificador/tests/test_train_phase.py`
- Nuevos: `fase1_telemetria/diagnose_cache_event.py`,
  `fase1_telemetria/tests/test_diagnose_cache_event.py`

### Contrato de datos

Columna `cache_miss_rate` (y `cache_miss_rate_relative`) en `windows.csv` y
`training_cpu_intervals.csv`. Semántica: fracción de referencias de caché
(evento genérico) que fueron miss. **No** se afirma que sea LLC/L3.

### Pruebas ejecutadas y resultados

- `pytest fase1_telemetria/tests/test_postprocess.py
  fase1_telemetria/tests/test_diagnose_cache_event.py
  fase2_clasificador/tests/test_train_phase.py` → **72 passed**.
- Suite completa `fase1_telemetria fase2_clasificador fase3_daemon common` →
  **684 passed**.
- Test nuevo `test_load_acepta_csv_historico_con_llc_miss_rate` verifica la
  compatibilidad de lectura.

### Evidencia de hardware

Ninguna todavía. `diagnose_cache_event.py` no se ha corrido en paccaA100.

### Limitaciones

- El diagnóstico da un veredicto textual; incluso si mostrara equivalencia con
  un evento de último nivel, `cache_miss_rate` sigue siendo el nombre correcto
  (no se revierte sin evidencia fuerte y multi-nodo).

### Trabajo pendiente

- Correr `diagnose_cache_event.py` en paccaA100 y adjuntar el JSON a este ID.

### Criterio exacto de cierre

Cerrado cuando: (a) ninguna ruta de código produce `llc_miss_rate`; (b) el
entrenador y sus pruebas usan `cache_miss_rate`; (c) el JSON de
`diagnose_cache_event.py` de paccaA100 está adjunto con su veredicto. (a) y (b)
ya cumplen; (c) pendiente.

---

## F1-GPU-003 — Contrato de granularidad GPU y dataset intermedio por fase

**Fecha de registro:** 2026-09-04
**Estado:** implementado y validado localmente; a la espera de una campaña GPU
real para poblarlo.

### Problema

`postprocess.py` producía, para GPU, **una fila por muestra NVML periódica**,
todas con la misma intensidad operacional `ncu` (constante por kernel). Sirve
para clasificar el régimen predominante, pero: (i) una muestra NVML aislada no
es un ejemplo ML independiente — la evidencia de F1-GPU-002 mostró escalones de
~105-120 ms en potencia/utilización; (ii) no hay marcas de fase para kernels de
terceros (la intercepción de `cudaLaunchKernel` no funciona), así que no se
pueden probar transiciones internas.

### Decisión (contrato de granularidad GPU, formal)

- Unidad de fila del dataset de entrenamiento GPU = **una corrida**
  (`run_id` = kernel_ref × nivel_frecuencia_gpu × repetición), o una **fase
  estable** si en el futuro hay marcas de fase alineadas con verdad offline.
  Nunca una muestra NVML periódica.
- Features NVML = agregados robustos sobre las muestras NVML **post-warmup y
  válidas** de la corrida: mediana, media recortada 10%, desviación, IQR,
  min/max, `n_distinct` (frescura / cota inferior de actualizaciones físicas),
  `valid_frac`, duración cubierta, nº de muestras, fracción usable.
- `phase_label_train`, `operational_intensity` (`ncu`) y `i_ridge_used` se
  conservan **solo para trazabilidad/verdad**; el entrenador GPU no puede
  leerlas como features (fuga).
- `gpu_phasic_*` (sintéticos con fases programadas): **no elegible** para
  entrenamiento con la etiqueta constante del catálogo; queda como control
  diagnóstico (`training_eligible = False`, `phase_quality_status =
  phasic_control_needs_marks`) salvo que existan marcas de fase + verdad
  offline alineada.

### Archivos modificados

- Nuevo: `fase1_telemetria/gpu_phases.py` (contrato + builder + writers)
- Nuevo: `fase1_telemetria/tests/test_gpu_phases.py`
- `fase1_telemetria/postprocess.py` (`run_postprocess`: para `device=gpu` escribe
  `training_gpu_phases.csv` + `training_gpu_phases_contract.json`)

### Contrato de datos

`training_gpu_phases.csv` — una fila por `run_id`. Columnas: trazabilidad
(`run_id`, `repetition`, `kernel_ref`, `node_id`, `freq_level_id`,
`gpu_freq_level_id`, `binary_checksum`, `roofline_calibration_ref`,
`operational_intensity`, `i_ridge_used`, `phase_label_train`), `kernel_family`,
`granularity` (`run`), `phase_quality_status` /
`phase_quality_reason` / `training_eligible`, contadores de muestras y cobertura,
`gpu_energy_delta_mj_sum` / `gpu_energy_covered`, y `<señal>_<agg>` para las 5
señales NVML × 8 agregados. Sidecar
`training_gpu_phases_contract.json` con el contrato formal.

### Pruebas ejecutadas y resultados

- `pytest fase1_telemetria/tests/test_gpu_phases.py
  fase1_telemetria/tests/test_postprocess.py` → **69 passed**.
- Test clave `test_muchas_muestras_de_una_corrida_producen_una_sola_fila`:
  30 muestras NVML de una corrida → **1 fila**, no 30.

### Evidencia de hardware

Ninguna: no hay campaña GPU nueva. El builder se probó con `windows.csv`
sintéticos.

### Limitaciones

- Sin marcas de fase, `granularity` es siempre `run`; no se resuelven fases
  intra-corrida (coherente con `[[intra-kernel-phase-hunt-negative]]`).
- Los agregados son robustos pero siguen dependiendo de la cadencia NVML real
  (F1-GPU-002).

### Trabajo pendiente

- Poblar `training_gpu_phases.csv` con una campaña GPU real (tras F1-GPU-002 y
  la selección de catálogo).
- Implementar el entrenador GPU que consuma este CSV (F1-GPU-001).

### Criterio exacto de cierre

Cerrado cuando una campaña GPU real produce `training_gpu_phases.csv` con
`training_eligible=True` en ≥ 5-6 familias por clase y el gate H no reporta
`filas_gpu_no_son_muestras_independientes` en FAIL.

---

## F1-GPU-004 — Convergencia y procedencia de la verdad Roofline GPU (`ncu`)

**Fecha de registro:** 2026-09-04
**Estado:** parcialmente implementado (parser + lógica de convergencia + runbook,
validados); ejecución de `ncu` bloqueada por hardware.

### Problema

Los kernels GPU históricos tuvieron análisis de convergencia (p. ej.
`rodinia_lud`); los candidatos nuevos **no lo heredan**. Aceptar una etiqueta
Roofline para un kernel GPU sin evidencia de que su intensidad operacional
convergió — y sin distinguir FP32/FP64/mezcla/entero — es asignar una etiqueta
sin fundamento.

### Decisión

Herramienta que: perfila un kernel con cantidades crecientes de trabajo;
registra launches solicitados vs. observados; calcula FLOPs (fadd+fmul+2·ffma,
y las dobles) y bytes DRAM coherentes con la precisión; detecta
`fp32`/`fp64`/`mixed`/`integer_no_flops`/`no_flops`; aplica un criterio de
convergencia **declarado antes**: cambio relativo de la OI < 1% entre los dos
puntos con más trabajo, con launches observados ≈ solicitados; conserva salida
cruda de `ncu`, comandos y versiones; **no** permite `roofline_label_eligible`
sin convergencia; marca kernels enteros/sin FLOPs como
`not_suitable_for_roofline_truth`.

### Archivos modificados

- Nuevo: `fase1_telemetria/ncu_convergence.py` (parser CSV de `ncu`,
  `flops_and_precision`, `assess_convergence`, `build_kernel_report`, runner con
  fallback a runbook si no hay `ncu`)
- Nuevo: `fase1_telemetria/tests/test_ncu_convergence.py`

### Contrato de datos

`<kernel_ref>.json` por kernel: `precision`, `points[]`
(`launch_count_requested/observed`, `flops`, `dram_bytes`,
`operational_intensity`), `converged`, `converged_at_launch_count`,
`final_operational_intensity`, `roofline_label_eligible`, `status`
(`converged`/`not_converged`/`not_suitable_for_roofline_truth`),
versiones `ncu`/driver/CUDA, `binary_checksum`, `kernel_args`. Es el archivo
que lee el gate H (`candidatos_gpu_con_ncu_convergente`).

### Pruebas ejecutadas y resultados

- Las pruebas de `test_ncu_convergence.py` cubren el formato largo real y el
  fixture ancho heredado, conteo de launches distintos, precisión
  fp32/fp64/mixta, saturación de una carga, convergencia y exclusiones.

### Evidencia de hardware

No se generó evidencia nueva porque `ncu` no está en el entorno local. La
corrección del parser se contrastó además con el CSV largo histórico conservado
de paccaA100 (`ID`, `Metric Name`, `Metric Value`); falta revalidarlo con la
versión actualmente instalada en el servidor.

### Limitaciones

- Los nombres de métrica de `ncu` cambian entre versiones; el parser mapea por
  subcadena, tolerante, pero debe re-verificarse contra la versión de `ncu` de
  paccaA100.
- `ncu --launch-count` limita cuántos lanzamientos coincidentes se perfilan; no
  cambia el tamaño del problema. Los comandos de catálogo se mantienen fijos.
  Si un kernel no expone suficientes launches, el reporte registra saturación
  del workload o falta de convergencia, sin sustituir argumentos del kernel.

### Trabajo pendiente

- Correr el runbook por candidato GPU en paccaA100 y adjuntar los
  `<kernel_ref>.json`.
- Alimentar `--ncu-reports-dir` del gate H con esos JSON.

### Criterio exacto de cierre

Cerrado cuando todos los kernels GPU del catálogo congelado tienen un
`<kernel_ref>.json` con `converged=True` y `roofline_label_eligible=True`, o
están marcados `not_suitable_for_roofline_truth` y excluidos del dataset GPU.

---

## F1-XDEV-004 — Análisis Pearson/Spearman/VIF y contrato de features

**Fecha de registro:** 2026-09-04
**Estado:** implementado y validado con fixtures; selección definitiva pendiente
del dataset real.

### Problema

El plan (§2.5) exige Pearson, Spearman y VIF sobre las columnas candidatas del
dataset real antes de fijar las features, y documentar los descartes. Hoy el
entrenador CPU lleva a la vez `mpki` y `cache_miss_rate` (misma señal por dos
caminos) y no hay ningún módulo que haga ese análisis.

### Decisión

Módulo de análisis pre-entrenamiento (no entrenador) que: consume el CSV
intermedio final por dispositivo; opera solo sobre filas elegibles; calcula
Pearson y Spearman; reporta pares con `|ρ| > 0.85`; calcula VIF tras el primer
filtrado; trata ausencias/constantes/infinitos/escala explícitamente; recomienda
descartes priorizando la medición física más directa; **nunca** propone una
columna de verdad Roofline como feature; produce CSV+JSON; permite **congelar**
un contrato versionado por dispositivo (`freeze_contract`, que rechaza fuga y
columnas no elegibles). CPU y GPU se analizan por separado (fuente de verdad y
columnas de calidad distintas).

### Archivos modificados

- Nuevo: `fase2_clasificador/analysis/feature_contract.py`
- Nuevo: `fase2_clasificador/run_feature_contract.py`
- Nuevo: `fase2_clasificador/tests/test_feature_contract.py`

### Contrato de datos

`feature_contract_<device>.json` (diagnóstico: candidatas, diagnóstico por
columna, `high_corr_pairs`, `vif`, `recommended_drops`,
`recommended_feature_set`, `roofline_truth_columns_seen`) +
`feature_contract_<device>_pairs.csv`. `frozen_feature_contract_<device>.json`
(contrato revisado a mano: `features[]`, `device`, `frozen_at_utc`).
`ROOFLINE_TRUTH_COLUMNS` es la lista compartida de columnas prohibidas.

### Pruebas ejecutadas y resultados

- `pytest fase2_clasificador/tests/test_feature_contract.py` → **11 passed**.
- Cubren: detección de par muy correlado y preferencia por la medición directa
  (`cache_miss_rate` sobre `mpki`), exclusión dura de columnas Roofline,
  constante/mayormente-ausente/infinito, VIF alto, `freeze` que rechaza fuga y
  no elegibles, dispositivo GPU con su propia columna de calidad, 0 filas
  elegibles.

### Evidencia de hardware

No aplica (análisis sobre CSV).

### Limitaciones

- Los umbrales (`|ρ|>0.85`, VIF>10) son puntos de partida; se ajustan sobre el
  dataset real.
- `recommended_feature_set` es una propuesta; la selección final se fija con
  `freeze_contract` tras revisar el reporte real.

### Trabajo pendiente

- Correr sobre `training_cpu_intervals.csv` y `training_gpu_phases.csv` reales.
- Congelar `frozen_feature_contract_cpu.json` / `_gpu.json`.
- Alinear `train_phase.py::FEATURES` con el contrato congelado (paso manual con
  artefacto real; este módulo no lo toca automáticamente).

### Criterio exacto de cierre

Cerrado cuando existen los dos `frozen_feature_contract_<device>.json` derivados
del dataset real, sin fuga, y el gate H reporta
`analisis_pearson_spearman_vif_presente` y `contrato_final_de_features_presente`
en PASS.

---

## Gate H — Auditoría de readiness pre-entrenamiento

**Fecha de registro:** 2026-09-04
**Estado:** implementada y validada con fixtures; a la espera de un dataset real
para dictaminar. No es una decisión nueva del plan: es el gate que verifica que
las demás (`F1-*`, `F2-XDEV-001`) están cumplidas antes de entrenar.

### Problema

No existía una verificación única y ejecutable de "¿este dataset está listo para
entrenamiento?". Los criterios estaban repartidos entre secciones.

### Decisión

Auditoría con 13 gates, cada uno `PASS` / `FAIL` / `BLOCKED` / `NA` por
dispositivo. Un dataset está *listo para entrenamiento* solo si ningún gate está
en `FAIL` ni `BLOCKED`. `BLOCKED` (no `PASS`) para lo que necesita hardware,
permisos o campaña real. Gates: checksums/procedencia; warmup calibrado y
documentado; calibración Roofline presente (por dispositivo/precisión/frecuencia);
etiqueta no de hint ni proxy; cobertura ≥ 5 familias por clase; filas GPU no
independientes (contrato); candidatos GPU con `ncu` convergente; frecuencia
verificada bajo carga; calidad/rechazos reportados; contrato final de features
presente; sin columnas de fuga; Pearson/Spearman/VIF presente; granularidad
declarada.

### Archivos modificados

- Nuevo: `fase2_clasificador/analysis/pretraining_readiness.py`
- Nuevo: `fase2_clasificador/run_pretraining_readiness.py`
- Nuevo: `fase2_clasificador/tests/test_pretraining_readiness.py`

### Contrato de datos

Entrada: rutas a `training_cpu_intervals.csv` / `training_gpu_phases.csv`, a los
contratos de features y sus reportes, al dir de reportes `ncu`, al artefacto de
warmup, al reporte de cobertura y al agregado de transición. Salida:
`readiness.json` (`schema: f1/pretraining_readiness/1`, `gates[]` con cpu/gpu/
detail, `summary`, `cpu_ready_for_training`, `gpu_ready_for_training`) + tabla
humana. `rc=0` si algún dispositivo está listo, `rc=1` si no.

### Pruebas ejecutadas y resultados

- `pytest fase2_clasificador/tests/test_pretraining_readiness.py` → **7 passed**.
- Cubren: sin artefactos nada está listo; GPU sin `ncu` queda `BLOCKED` (no
  `PASS` con fixture); detección de fuga en el contrato; etiqueta == hint falla;
  cobertura insuficiente por familia falla; bundle CPU completo y coherente pasa;
  CLI `rc` y JSON.

### Evidencia de hardware

No aplica (opera sobre artefactos).

### Limitaciones

- Algunos gates dependen de artefactos que hoy no existen (warmup real, reportes
  `ncu`, contratos congelados) → hoy el gate reportaría `FAIL`/`BLOCKED` en
  varios puntos, que es el resultado correcto: **ningún dataset está listo**.
- El gate `frecuencia_verificada_bajo_carga` para GPU queda `BLOCKED`: no existe
  en datasets históricos. Las campañas nuevas agregan la traza NVML por
  corrida, la comparan contra `gpu_freq_mhz_applied` y exponen
  `gpu_frequency_quality_status`; el gate permanece `BLOCKED` únicamente para
  archivos anteriores sin esas columnas.

### Trabajo pendiente

- Ejecutarlo cuando existan el dataset y los artefactos reales; adjuntar el
  `readiness.json` resultante.

### Criterio exacto de cierre

Cerrado (para un dispositivo) cuando `<device>_ready_for_training` es `True`
sobre artefactos reales.

---

## Actualizaciones a decisiones ya registradas (2026-09-04)

- **F1-XDEV-002** pasa de "pendiente" a **implementado (módulo)**: se añadió
  `fase1_telemetria/warmup_calibration.py` (detección portada y auditada de
  `old/scripts/pacca/measure_warmup.py`: CV de dos ventanas + segmentación por
  puntos de cambio; margen ×1.2; criterio robusto = máximo entre ≥3 corridas;
  estados `measured`/`insufficient_signal`/`documented_fallback`/`not_suitable`;
  artefacto CSV+JSON; propuesta al catálogo sin reemplazo silencioso, con backup
  `.bak` y verificación de checksum) + CLI + `fase1_telemetria/tests/
  test_warmup_calibration.py` (**9 passed**). **Bloqueado**: la calibración real
  necesita correr una campaña real con `warmup_seconds` en 0 en paccaA100. Ver
  el rediseño de este mismo flujo, más abajo.
- **F1-XDEV-001 / F1-XDEV-003** ganan generador de manifiesto definitivo:
  `fase1_telemetria/campaigns/generate_final_manifest.py` (exige la lista
  congelada de kernels; resuelve la rejilla fina MHz → `fraction` contra el
  rango real del nodo; sin datos del nodo marca
  `frequency_grid_status: assumed_range_pending_node_verification` y
  `verify_grid_against_node()` falla) + `test_generate_final_manifest.py`
  (**6 passed**, uno carga el manifiesto generado con el parser real).
- **F1-GPU-002** gana el comparador de cadencia de la Etapa A:
  `fase1_telemetria/gpu_transition/cadence_sweep.py` (agrega los `summary.json`
  del probe a 5/10/50/100 ms y recomienda `q_produccion` = la cadencia más
  gruesa que conserva ≥ 80% de los escalones observados frente a 5 ms) +
  `test_cadence_sweep.py` (**5 passed**). Sigue **pendiente de medición real**.

---

## F1-XDEV-002 (actualización) — Calibración de warmup plegada dentro de la campaña real

**Fecha:** 2026-09-04
**Estado:** implementado y validado localmente; calibración real bloqueada por
campaña. Reemplaza el flujo de "mini-campaña separada" descrito arriba por uno
plegado dentro de la campaña real, sin dejarlo de soportar como alternativa.

### Problema

El flujo original de F1-XDEV-002 pedía una mini-campaña de calibración previa
a cada campaña real, con `warmup_seconds: 0` en el catálogo, replicando
binario/args/tamaño/hilos/pinning/nodo/colector/frecuencias de la campaña
posterior. Auditando el código se confirmó que **`warmup_seconds` solo se lee
en el postproceso** (`postprocess.py:492`, vía `cli.py::cmd_postprocess` y
`campaign.py`); `runner.py` no lo referencia en ningún punto — la recolección
siempre captura la traza completa desde el inicio, sea cual sea el valor
declarado. Por tanto, dos campañas (una de calibración, otra de datos) miden
exactamente lo mismo si comparten manifiesto; la separación era trabajo de
clúster duplicado sin necesidad técnica.

### Decisión

Plegar la calibración dentro de la campaña real, con la MISMA verificación por
análisis sobre las filas ya recolectadas:

1. La campaña real (p. ej. el cribado `F1-XDEV-001`) declara
   `warmup_seconds_override: 0.0` en su manifiesto — nuevo campo opcional de
   `Manifest`, consumido solo por `cli.py::cmd_postprocess` y el postproceso en
   vivo de `campaign.py`. Ausente (el default, y el único valor de todo
   manifiesto anterior) preserva el comportamiento de siempre: usar
   `kernel_entry.warmup_seconds` del catálogo. Con el override, ninguna ventana
   queda `warmup_excluded` al postprocesar esa campaña — se conserva el
   transitorio completo, con `>= 3` repeticiones y cobertura de REF + extremos
   de frecuencia garantizadas por ser la matriz real, no una reserva aparte.
2. `warmup_calibration.py` (sin cambios de lógica) calibra sobre los
   `windows.csv` que esa misma campaña ya produjo.
3. La propuesta se aplica al catálogo real con
   `apply_proposals_to_catalog(..., apply=True)` (ya con backup `.bak` y
   verificación de checksum, sin reemplazo silencioso).
4. `fase1_telemetria/repostprocess_campaign.py` (nuevo) **re-postprocesa la
   misma campaña sin relanzar ningún kernel**: reutiliza `samples.csv`/
   `metadata.json` ya escritos, localiza cada corrida por el `run_id` real
   (`build_matrix()` + `runner.build_run_id()`, nunca una heurística de nombre
   de directorio), y llama a `run_postprocess()` de nuevo con el catálogo YA
   CORREGIDO. Ignora `manifest.warmup_seconds_override` **a propósito**
   (`ignore_manifest_override=True` por defecto) — ese campo es solo para el
   paso 1; el paso 4 debe reflejar siempre el valor calibrado, nunca repetir
   el forzado a 0.
5. El flujo de mini-campaña separada (documentado arriba) sigue siendo válido
   como alternativa — por ejemplo, para un chequeo barato antes de comprometer
   tiempo de clúster a la campaña completa — pero deja de ser el camino
   recomendado.
6. `compute_protocol_fingerprint()` (CAM-09) incluye ahora
   `warmup_seconds_override`: dos manifiestos que solo difirieran en ese campo
   antes compartían huella de protocolo, lo que podía mezclar corridas con
   distinto criterio de exclusión bajo el mismo `run_id` en una reanudación.
7. **Re-validación del veredicto accepted/rejected** (añadido el mismo día,
   tras una revisión posterior). El accept/reject de cada corrida se decide,
   en la campaña en vivo, sobre el `windows.csv` PROVISIONAL (warmup=0, nada
   excluido) — es el único que existe en ese momento, antes de calibrar. Una
   corrida al límite de `target_windows_per_repetition` puede tener MENOS
   ventanas usables una vez excluido el warmup real, y seguiría figurando como
   `accepted` si nadie la reevaluara. `repostprocess_campaign.py`, al
   reprocesar con éxito, ahora también corre `validation.validate_windows()`
   sobre el `windows.csv` ya corregido y sobrescribe `verdict.json`
   (`validation.write_verdict()`) — nunca borra ni mueve la corrida (VAL-06),
   solo dice honestamente si sigue aceptada. Cada resultado trae
   `verdict_accepted`/`verdict_factor_id`/`verdict_message` y
   `verdict_changed` (si difiere del veredicto que ya estaba en disco); el CLI
   imprime cada cambio de veredicto explícitamente y los cuenta en el resumen,
   para que se revisen a mano antes de dar la campaña por cerrada.

### Archivos modificados

- `common/hpc/manifest.py`: campo `Manifest.warmup_seconds_override: float |
  None`, parseado con `_parse_optional_non_negative_number` (reutiliza el
  helper ya existente para `load_threshold`, mismo código de error `MAN-00`).
- `fase1_telemetria/cli.py::cmd_postprocess`, `fase1_telemetria/campaign.py`
  (postproceso en vivo): usan el override cuando está declarado.
- `fase1_telemetria/campaign.py::compute_protocol_fingerprint`: incluye el
  campo nuevo.
- Nuevo: `fase1_telemetria/repostprocess_campaign.py` (+ test).
- `fase1_telemetria/warmup_calibration.py`: docstring reescrito con el flujo
  plegado como recomendado.
- `fase1_telemetria/catalog/campaigns/campaign_pacca_phase_coverage_{cpu,gpu}_
  screen.yaml`: `warmup_seconds_override: 0.0` + comentario del flujo de 4
  pasos. **`catalog_path` no cambia** (sigue `../catalog.yaml`, el catálogo
  real) — no hace falta un catálogo temporal aparte.
- Tests: `common/tests/test_manifest.py` (+2), `fase1_telemetria/tests/
  test_cli.py` (+1 override, +1 assert en el existente), `fase1_telemetria/
  tests/test_campaign.py` (+1 variante de fingerprint), nuevo
  `fase1_telemetria/tests/test_repostprocess_campaign.py` (12 casos, incluida
  la re-validación).

### Contrato de datos

`warmup_seconds_override` (manifiesto, opcional, `float >= 0` o ausente):
fuerza el `warmup_seconds` usado por **todo** kernel de esa campaña al
postprocesar, sin tocar `catalog.yaml`. No afecta la recolección. Nunca debe
quedar declarado en el manifiesto usado para producir el dataset final leído
por Fase 2 — `repostprocess_campaign.py` existe exactamente para volver a
generar ese dataset final ignorándolo. `verdict.json` de cada corrida queda
sobrescrito con el veredicto recalculado sobre el `windows.csv` corregido.

### Pruebas ejecutadas y resultados

- `pytest common/tests/test_manifest.py fase1_telemetria/tests/test_cli.py
  fase1_telemetria/tests/test_campaign.py
  fase1_telemetria/tests/test_repostprocess_campaign.py` → todo verde.
- Suite completa `fase1_telemetria fase2_clasificador fase3_daemon common` →
  **702 passed**.
- Casos clave: el override pisa el catálogo en la recolección
  (`test_postprocess_respeta_warmup_seconds_override_del_manifiesto`);
  `repostprocess_campaign` usa el catálogo dado y NO el override por defecto
  (`test_ignora_warmup_seconds_override_por_defecto`); una corrida sin
  `samples.csv` se reporta `skipped`, nunca se fabrica; un fallo real de una
  corrida se reporta `error` sin detener las demás; el fingerprint cambia si
  cambia el override; una corrida cuyo `windows.csv` corregido cae por debajo
  de `target_windows_per_repetition` pasa de `accepted` a `rejected` y
  `verdict.json` en disco queda actualizado
  (`test_corregir_el_warmup_puede_hacer_que_una_corrida_al_limite_se_rechace`);
  sin `verdict.json` previo, `verdict_changed` es `False` (no se fabrica un
  "cambio" contra la nada); el CLI imprime cada veredicto cambiado.
- Los dos manifiestos de cribado modificados se verificaron cargando de
  verdad con `common.hpc.manifest.load()`: `warmup_seconds_override=0.0`,
  33/23 kernels intactos, `catalog_path` sigue apuntando al catálogo real.

### Evidencia de hardware

Ninguna: el flujo completo (recolección → calibración → aplicación →
re-postproceso) no se ha corrido en paccaA100.

### Limitaciones

- El campo es un interruptor de campaña completa (todo o nada): no permite
  forzar 0 solo para un subconjunto de kernels dentro de la misma campaña. Si
  hiciera falta, habría que filtrar por `--kernel` en un manifiesto aparte.
- `repostprocess_campaign.py` no borra ni archiva el `windows.csv`/
  `training_cpu_intervals.csv` anterior (el del override en 0): los
  sobrescribe. Quien necesite conservar la traza "sin calibrar" para auditoría
  debe copiar el directorio antes de re-postprocesar.
- ~~El accept/reject de cada corrida seguía reflejando el `windows.csv`
  provisional (warmup=0) después de corregir el warmup.~~ **Resuelto el mismo
  día** (punto 7 de la Decisión, arriba): `repostprocess_campaign.py` ahora
  recalcula el veredicto y sobrescribe `verdict.json`. `run_campaign.py`
  sigue sin re-archivar una corrida que pase de aceptada a rechazada tras la
  corrección (VAL-06: nunca se borra); si eso importa para el reporte de
  cobertura, `phase_coverage.py` debe filtrar por `verdict.json` actualizado,
  no asumir que `accepted_run_ids` de `campaign_metadata.json` (que sigue
  reflejando la decisión en vivo) está al día tras un re-postproceso.

### Trabajo pendiente

- Ejecutar el cribado real con `warmup_seconds_override: 0.0` en paccaA100.
- Calibrar, aplicar al catálogo, re-postprocesar con
  `repostprocess_campaign.py`, y adjuntar `warmup_calibration.json` +
  el resumen de `repostprocess_campaign` a este ID.

### Criterio exacto de cierre

Cerrado cuando el cribado real (CPU y GPU) tiene un `warmup_calibration.json`
con `status="measured"` para cada kernel candidato (o `not_suitable`/
`documented_fallback` explícitamente justificado), el catálogo real quedó
actualizado con esos valores, y `repostprocess_campaign.py` regeneró
`windows.csv`/`training_cpu_intervals.csv`/`training_gpu_phases.csv` finales
sobre esa misma campaña sin relanzar ningún kernel.

---

## Notas de literatura externa (no son cambios al plan)

Registro de trabajos publicados que informan decisiones metodológicas ya
tomadas o pendientes. No modifican el plan; sirven como respaldo citable y como
lista de tareas menores para Fase 2 y para el documento final.

### LIT-001 — Littman & Deakin, "Classifying Performance Bounds Using Machine Learning" (póster SC25)

**Revisado:** 2026-09-04. Universidad de Bristol; deriva de una tesis de
pregrado. `doi` del dataset: `10.5281/zenodo.17194638`.

**Qué es:** estudio *preliminar* que hace únicamente el clasificador
compute-bound / bandwidth-bound a nivel de programa completo (1 registro = 1
corrida agregada, sin dimensión temporal ni de fase). No hay DVFS, daemon, EDP
ni actuación. Plataforma declarada explícitamente "arbitraria" (Xeon E5-2680 v4
Broadwell). 8 códigos (SGEMM, DGEMM, miniBUDE compute; STREAM, LBM D2Q9, HPCCG,
3D-Heat, LU-MKL bandwidth), 100 registros c/u, 1 200 filas. Features: GFLOPs,
FLOPc, IPC, %retiring/%frontend/%backend/%bad-speculation (Top-Down), ratios de
vectorización SP y DP, y cache-miss ratio L1/L2/L3. Modelos DT/k-NN/LogReg/RF/
SVM/MLP + 3 baselines (uniforme/proporcional/mayoría). Accuracy 0.83–0.92;
reportan *accuracy perfecta con hiperparámetros por defecto* y un t-SNE con un
cluster nítido por código.

**Coincidencias con Hyperion (refuerzan lo ya decidido):**

- Etiqueta por régimen conocido + confirmación Roofline — igual criterio que
  §2.3 del plan.
- Reconocen escasez de cargas FP-bound como limitación aceptada y
  *"indicative of much current high-performance software"* — encuadre
  reutilizable casi literal para la escasez de compute-bound GPU de
  `F1-XDEV-001`.

**Divergencias (Hyperion es más estricto, no cambiar de rumbo):**

- Su CV es leave-one-out **por registro**: train y test comparten el mismo
  kernel. Con t-SNE mostrando un "performance fingerprint" por código, su
  0.92 mide probablemente reconocimiento de código, no de régimen. Es
  evidencia citable a favor de `leave-one-familia-out` (§2.6 / `F2-XDEV-001`),
  no en contra del tamaño del catálogo.
- Señal de memoria por cache-miss ratio (sin visibilidad de prefetch);
  Hyperion usa bytes DRAM reales de `uncore_imc`. Mantener cache-miss fuera
  también de las features (ya está en `FORBIDDEN`).
- Funden LLC-bandwidth y DRAM-bandwidth en una clase; el `memory_bound` de
  Hyperion es específicamente DRAM.
- Multiplexado por 3 corridas fusionadas; Hyperion abre los ~10 contadores en
  una sola corrida sin multiplexado.

**Tareas menores que aporta (Fase 2, no bloqueantes):**

1. Reportar el trío de baselines explícito (uniforme / proporcional / mayoría)
   como su Tabla 3, además del `DummyClassifier` ya presente en `train_phase.py`.
2. Añadir un paso de visualización t-SNE (o UMAP) a la EDA de Fase 2 como
   diagnóstico de separabilidad trivial / fingerprint por familia; si aparece
   un blob por familia con clases limpias, documentarlo en resultados.
3. Usar el dataset de Zenodo como prueba de humo externa del pipeline de
   entrenamiento, sin correr campañas.

**Dónde cita al trabajo:** ver más abajo el mapeo al `docs/libro/main.tex`.

### LIT-002 — Antici et al., "MCBound" (SC24)

**Revisado:** 2026-09-04. `doi:10.1109/SC41406.2024.00062`. Ya citado en el
libro como `Antici2024` (planteamiento del problema); esta nota amplía su uso.

**Qué es:** primer framework *online* que clasifica *jobs* HPC como memory- o
compute-bound **antes de ejecutarlos**, a partir de metadatos de envío +
histórico de jobs. Fin: guiar scheduling / co-scheduling / asignación de
recursos / selección de frecuencia de nodo. Fugaku (A64FX), 2,2 M de jobs
(dic-2023 a mar-2024), evaluación sobre >700 000 jobs de febrero 2024.

**Etiquetado (verdad de referencia):** Roofline sobre totales de job. Con
`p_j = #flops_j / (duration_j · #nodes_j)` y
`mb_j = #moved_bytes_j / (duration_j · #nodes_j)`, etiqueta = compute-bound si
`op_j = p_j/mb_j > op_r`. Ridge de nodo Fugaku ≈ 3,3 Flops/Byte. `#flops` de
eventos PMU del A64FX (`FP_FIXED_OPS_SPEC` + `FP_SCALE_OPS_SPEC·4`);
`#moved_bytes` de `BUS_READ/WRITE_TOTAL_MEM · 256 B / 12`. **Es la misma lógica
OI-vs-ridge del §2.3 de Hyperion**, solo que sobre el total del job en vez de
por ventana `uncore` — triangulación fuerte del criterio de etiquetado.

**Predictor:** *no usa contadores en inferencia* (predice pre-ejecución). Usa
metadatos de envío (usuario, nombre del job, #cores, frecuencia solicitada)
codificados con SBERT (`all-MiniLM-L6-v2`, 384-dim). Modelos KNN y Random
Forest (scikit-learn). Validación **temporal** (entrena α∈{15,30,45,60} días,
reentrena cada β∈{1,2,5,10}), no agrupada por kernel — su regimen (2,2 M jobs
reales distintos) hace que la fuga por kernel no aplique igual que en Hyperion.

**Resultados:** F1-macro RF = 0,90 (α=15, β=1); KNN = 0,89 (α=30, β=1).
Desbalance de clases **memory:compute ≈ 3,5:1** (1 643 477 vs 477 975) — misma
dirección que la escasez FP-bound de LIT-001 y que la escasez compute-bound GPU
de `F1-XDEV-001`; **tercer testimonio independiente** del mismo sesgo, muy
citable. Overhead: caracterización ~1e-6 s/job, inferencia RF ~2e-6 s/job,
KNN ~2,3e-3 s/job; corre en **máquina desacoplada**, cero overhead en los nodos
de cómputo.

**Estimación de impacto (insumo directo para el encuadre del Objetivo 4):**
con 90% de acierto, selección semi-automática de frecuencia; a escala Fugaku,
mover 750 k jobs memory-bound de *boost* a modo normal ahorraría ~450 MW de
potencia y 14 GJ de energía; 330 k jobs compute-bound en normal que deberían
ir en *boost* cuestan >1700 h de cómputo. Sobre el nodo, **54% de los jobs
memory-bound corren a 2,0 GHz y solo 30% de los compute-bound en boost** — "no
hay correlación observable entre la frecuencia elegida por el usuario y la
posición Roofline": la motivación de automatizar la decisión.

**Contrastes con Hyperion (para la discusión):**

- MCBound actúa a granularidad de **job entero** y con **2 niveles discretos**
  (normal/boost); Hyperion actúa por **fase intra-corrida** y sobre un barrido.
  Granularidad más fina = novedad, pero también hace que el overhead del
  agente sea un problema real (ellos lo resuelven trivialmente con máquina
  aparte; Hyperion corre *en* el nodo → Objetivo 2/4).
- MCBound predice de metadatos, no de telemetría: su 0,90 **no** es evidencia
  de que los contadores clasifiquen bien, sino de que hasta el nombre del job
  correlaciona con la clase — el mismo "performance fingerprint" de LIT-001.
- Ni MCBound ni LIT-001 hacen clasificación de fases intra-job. Coherente con
  `[[intra-kernel-phase-hunt-negative]]`: la "fase" de Hyperion es de hecho
  cercana a whole-run a configuración fija; conviene ser explícito en el libro.

---

## F1-XDEV-005 — Orquestador de cribado hasta informe de utilidad

**Fecha de registro:** 2026-09-04
**Estado:** implementado y validado localmente; ejecución de hardware pendiente
**Commit de implementación:** `cfccb58`

### Problema

El orden previo permitía lanzar el cribado GPU usando la OI histórica del
catálogo antes de demostrar con `ncu` que esa OI, su precisión y su
convergencia eran válidas. Una etiqueta errónea habría producido falsa
cobertura Roofline. Además, los pasos de transición, warmup, agregación GPU y
cobertura existían como comandos separados, sin un gate operacional que
impidiera ejecutarlos fuera de orden.

Durante la implementación se encontró que el parser nuevo de F1-GPU-004 solo
entendía un fixture ancho artificial. El CSV real conservado de paccaA100 usa
el formato largo de Nsight Compute: una fila por métrica y lanzamiento, con
`ID`, `Metric Name` y `Metric Value`. También se estaba sustituyendo el valor
de `--launch-count` dentro de `{N}`/`{launches}` del comando del kernel, aunque
`--launch-count` es un filtro propio de `ncu` y no el tamaño del problema.

### Decisión

Añadir `run_screening_to_report.sh`, separado de `run_all.sh`, con etapas
reanudables `prepare`, `validate`, `screen-cpu`, `transition`, `ncu`,
`screen-gpu`, `warmup` y `report` (`screen` conserva el atajo conjunto). El
flujo termina antes de seleccionar o ejecutar la rejilla fina.

1. CPU conserva su OI medida en vivo por intervalo `uncore_imc`; no depende de
   `ncu` y su cribado puede ejecutarse en una reserva independiente. `all` lo
   serializa para evitar interferencia dentro de un único nodo.
2. GPU ejecuta F1-GPU-004 antes del cribado. Se usa el comando fijo real del
   catálogo y límites `ncu --launch-count` crecientes.
3. El parser acepta el CSV largo real y cuenta IDs de lanzamiento distintos,
   no filas de métricas. Precisión mixta, ausencia de FLOPs y Tensor Core sin
   regla explícita quedan no elegibles.
4. Solo `roofline_label_eligible=true` entra a `gpu_eligible.yaml`; su OI y
   precisión medidas actualizan una copia de trabajo del catálogo, nunca el
   archivo versionado sin revisión.
5. El cribado conserva el transitorio, calibra warmup sobre las mismas
   corridas y después re-postprocesa los crudos.
6. El diagnóstico GPU consume `training_gpu_phases.csv` agregado por corrida y
   filtra por `verdict.json`, en vez de ponderar cada muestra NVML como ejemplo.
7. El informe `tentative_kernel_utility.{csv,json,md}` separa candidatos
   externos, controles, semántica FLOPs dudosa y `dual_*` en cuarentena; el
   mínimo de familias no cuenta controles.
8. El barrido de cadencia reporta todas las señales NVML del probe. La decisión
   usa potencia, utilización GPU/memoria y relojes SM/gráfico; temperatura y
   energía quedan diagnósticas porque son lenta y acumulativa.

### Correcciones relacionadas

- `generate_final_manifest.py` elimina explícitamente
  `warmup_seconds_override` heredado del template: el manifiesto final usa el
  warmup calibrado del catálogo.
- `training_gpu_phases.csv` incorpora requested/applied MHz, fracción de
  muestras NVML dentro de tolerancia y `gpu_frequency_quality_status`. El gate
  H ya puede verificar reloj GPU bajo carga en campañas nuevas.
- La guía ejecutable y el significado de cada etapa quedan en
  `fase1_telemetria/SCREENING_TO_REPORT.md`.

### Criterio de salida

Se cierra en hardware cuando una misma ejecución versionada produce: reporte
de cadencia y matriz de transición; reporte `ncu` terminal para cada candidato
GPU; cribados CPU/GPU re-postprocesados con warmup medido; cobertura por
familia; e informe de utilidad. El informe puede terminar en FAIL de cobertura:
ese es un resultado válido que ordena compilar y caracterizar candidatos
adicionales antes de construir la campaña fina.
