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
| `F1-GPU-005` | Corregir 6 wrappers `gpu_rajaperf_*` que rechazaban corridas correctas por una etiqueta de tuning inexistente | Corregido y validado en pacca; catálogo actualizado |
| `F2-XDEV-001` | Diagnosticar cobertura Roofline y calidad antes de seleccionar/balancear entrenamiento | Implementado (sin datos de campaña aún) |
| `F2-CPU-001` | Evaluar representación PMU y abstención con validación LOFO anidada | Implementado y ejecutado; sus cifras (0.664, 68.9 %) fueron sustituidas por `F2-CPU-002` y `F2-CPU-003` (población y métrica finales) |
| `F1-CPU-004` | `cpu_gap_bfs`/`ptrchase` fuera del dominio del clasificador (latency-bound) | Cerrado (`ptrchase` reaparece solo como sonda de alfa en `F2-CPU-004`, nunca en el dataset del clasificador) |
| `F1-CPU-005` | Barrido de tamaño en NPB (`npb_ft`) | Cerrado, negativo |
| `F1-CPU-006` | `--repfact` resuelve `lcals_tridiag_elim` e `init3` | Cerrado, re-incluidos |
| `F1-CPU-007` | Segundo punto cerca del ridge en `dual_*` | Cerrado: éxito en `stencil`, negativo en `axpy`/`spmv`/`fft` |
| `F1-CPU-008` | `lbm_cpu` (LBM D2Q9) | Cerrado, negativo |
| `F1-CPU-009` | `hpccg_cpu` (CG 3D) | Cerrado, éxito (`N46`, 66.3 % cerca del ridge) |
| `F1-CPU-010` | Segundo intento en `dual_fft` | Cerrado, éxito (`N472`, 97.1 %) |
| `F1-CPU-011` | Revisión del set final de 43 kernels | Cerrado, sin cambios |
| `F2-CPU-002` | Informe de calidad y batería de pruebas del clasificador CPU | Implementado y ejecutado |
| `F2-CPU-003` | Sección de resultados CPU y modelo final (XGBoost, 12 variables, umbral 0.85) | Implementado |
| `F2-CPU-005` | Población final del clasificador CPU: 30 familias (24 más seis compute RAJAPerf); batería completa y libro regenerados | Implementado |
| `F2-CPU-004` | Tabla de política CPU (EDP por corrida), modelo de potencia P(f), candidatas compute nuevas | Tabla y modelo ejecutados (resultado: no actuar); barrido de candidatas en curso |
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

## F1-XDEV-006 — La frecuencia de CPU degrada la ejecución de cargas GPU (corrige a ARC-155/ARC-170)

**Fecha de registro:** 2026-09-13
**Estado:** medido y validado sobre dos campañas independientes ya ejecutadas; sin cómputo nuevo
**Campañas analizadas:** `pacca_gpu_dvfs_20260820` (288 combinaciones) y `pacca_dual_gpu_full_20260828` (8801 corridas)
**Scripts de reproducción:** `scripts/pacca/analysis/cpu_freq_durante_gpu_dvfs.py` y `scripts/pacca/analysis/cpu_freq_durante_gpu_fases.py`

### Problema

El proyecto arrastraba una conclusión registrada en ARC-155 (2026-08-19) y
usada como base de la decisión de diseño de ARC-170 (2026-08-20): *"no hay
overhead de lanzamiento medible sensible a la frecuencia de CPU"*, de donde se
concluyó que fijar la CPU al mínimo durante combinaciones GPU no introducía un
confusor. Esa conclusión es incorrecta, y la decisión que sostuvo (fijar la CPU
al mínimo siempre durante la campaña GPU) quedó apoyada sobre ella.

El error de ARC-155 no fue de ejecución sino de diseño experimental: midió un
solo kernel, `rodinia_gaussian`, elegido explícitamente por ser *"el caso más
sensible del catálogo a overhead de lanzamiento acumulado"* (unos 8190
lanzamientos CUDA por corrida). La premisa de esa elección es falsa: el
mecanismo de degradación no es el conteo de lanzamientos. Medido ahora sobre el
catálogo completo, `rodinia_gaussian` resulta ser el kernel **menos** afectado
de todos (+7,0 %), de modo que ARC-155 generalizó desde la sonda menos
representativa disponible.

### Decisión

Se corrige el registro y se mantiene la decisión ya tomada el 2026-09-13 para
la campaña final de GPU (`scripts/pacca/final_campaign/gpu_final.yaml`): el eje
de CPU se declara **únicamente en `REF`**, sin nivel fijo al mínimo.

La justificación cambia respecto a la que se había anotado. No es solo que la
pregunta ya esté contestada y re-medirla cueste horas de cola: es que fijar la
CPU al mínimo **contamina la medición del eje que sí interesa**. Con la CPU al
mínimo, el tiempo total de una carga GPU crece en promedio un 63-66 %, de modo
que el EDP atribuido a un nivel de reloj de GPU quedaría mezclado con una
penalización de origen distinto.

### Evidencia de hardware

**Verificación previa obligatoria (estrategia de espera del host).** Antes de
interpretar nada se descartó que el efecto viniera del modo de sincronización.
`pacca_gpu_dvfs_20260820` corrió con el shim de blocking-sync **activo**: 0
ocurrencias del aviso "ARC-70" en las 603 corridas con `stderr` (en modo spin
ese aviso aparecía 33 veces por corrida, ver ARC-153/154). El host dormía
bloqueado esperando a la GPU, no hacía *busy-wait*. La degradación no se
explica por la estrategia de espera.

**Medición 1: catálogo real, sin separación de fases.** Sobre
`pacca_gpu_dvfs_20260820`, comparación pareada por (kernel, nivel GPU,
repetición), variando solo el nivel de CPU entre `REF` y `F4` (mínimo), 134
pares comparables de 275 corridas aceptadas:

| Kernel | n | CPU mínima vs CPU nativa |
|---|---:|---:|
| `rodinia_lud` | 12 | +189,8 % |
| `rodinia_lavamd` | 18 | +152,5 % |
| `rodinia_backprop` | 14 | +96,2 % |
| `rodinia_myocyte` | 18 | +68,3 % |
| `rodinia_dwt2d` | 18 | +38,3 % |
| `gpu_dgemm_n4096` | 18 | +15,6 % |
| `rodinia_heartwall` | 18 | +9,1 % |
| `rodinia_gaussian` | 18 | +7,0 % |
| **Global** | **134** | **+66,1 %** |

En 134 de 134 pares (100 %) la CPU al mínimo resultó más lenta. Agrupando por
nivel de GPU, la media va de +57,3 % a +79,9 %, es decir el efecto se sostiene
a lo largo de todo el barrido de reloj de GPU y no es un artefacto de un nivel
particular.

**Medición 2: separación de fases (contrato `cold_warm_v1`).** La campaña del
selector `pacca_dual_gpu_full_20260828` sí instrumenta `dispatch_timing`
(`setup_seconds`, `cold_total_seconds`, `warm_total_seconds`,
`first_dispatch_seconds`), que la campaña de agosto no tenía. Comparación
pareada CPU `F0` (3,2 GHz) contra `F6` (800 MHz), es decir la reducción de
reloj 4x, sobre 1632 pares de kernels GPU:

| Fase | Degradación media |
|---|---:|
| Total del benchmark | **+63,3 %** |
| Caliente (ejecución en régimen) | **+93,8 %** |
| Setup (lado host) | +36,2 % |

La fase caliente se degrada casi el doble que el setup. Esto responde la
pregunta clave: **la penalización no es solo preparación del lado host**, la
ejecución en régimen también se degrada, y más.

**Dependencia del tamaño del problema.** El barrido de tamaños del catálogo
`dual_*` muestra que la degradación de la fase caliente escala con el problema:

| Kernel | Fase caliente |
|---|---:|
| `dual_stencil_gpu_N64` | -6,9 % |
| `dual_spmv_gpu_N10000` | -1,1 % |
| `dual_fft_gpu_N64` | +3,8 % |
| `dual_fft_gpu_N4096` | +169,6 % |
| `dual_stencil_gpu_N4096` | +171,9 % |

En problemas diminutos la frecuencia de CPU es irrelevante (o marginalmente
favorable al reloj bajo); en problemas grandes la fase caliente se duplica o
triplica. El patrón es consistente con un mecanismo de movimiento de datos
host-device dirigido por CPU que escala con el tamaño, y es incompatible con la
hipótesis de overhead fijo de lanzamiento sobre la que se construyó ARC-155.

**Validación cruzada.** Dos campañas independientes, con catálogos distintos
(kernels reales Rodinia/cuBLAS contra sintéticos `dual_*`), fechas distintas y
metodologías distintas (sin y con separación de fases), coinciden en la
magnitud del efecto total: +66,1 % y +63,3 %.

### Consecuencia sobre el plan detallado

El Plan detallado, §4.1, describe como medida defensiva del demonio: *"mientras
`gpu_util_pct` reporte actividad, forzar el reloj de CPU al mínimo,
independientemente de lo que diga `f_cpu` en ese instante; si la CPU está de
verdad bloqueada esperando, bajar su reloj casi no afecta el consumo porque no
hay conmutación que escale con la frecuencia"*.

El supuesto explícito de esa medida ("la CPU está de verdad bloqueada
esperando") queda refutado por los datos: aun con blocking-sync activo, la CPU
realiza trabajo cuya velocidad determina el tiempo de la carga GPU. Aplicada
tal como está redactada, la medida degradaría el tiempo de ejecución entre
+63 % y +66 % en promedio, y hasta +190 % en el peor kernel medido, empeorando
el EDP en vez de mejorarlo.

Esta sección debe reescribirse antes de la Fase 3. La redacción de reemplazo no
se fija aquí porque exige decidir el criterio de sustitución (por ejemplo,
condicionar el piso de frecuencia de CPU al tamaño de problema o a la
utilización de memoria observada), y esa decisión no está soportada todavía por
una medición dirigida.

### Limitaciones

1. La métrica de la medición 1 es `telemetry_elapsed_ns_mean`, tiempo total de
   corrida. Por sí sola no distingue host de dispositivo. Esa separación la
   aporta la medición 2, sobre un catálogo distinto (`dual_*` sintéticos), no
   sobre los kernels reales de la medición 1.
2. La medición 2 usa kernels sintéticos construidos para el estudio de
   selección de dispositivo, cuyo balance de trabajo host/device puede no ser
   representativo del catálogo final. La coincidencia del efecto total con la
   medición 1 (+63 % contra +66 %) mitiga esta objeción pero no la elimina.
3. No se ha medido el consumo energético asociado a esta degradación. La
   conclusión es sobre tiempo; afirmar el signo del efecto sobre EDP exige
   cruzar con la energía registrada en las mismas corridas, lo cual no se hizo
   aquí.
4. El mecanismo propuesto (movimiento de datos host-device) es una hipótesis
   consistente con la dependencia del tamaño, no una causa medida
   directamente. Confirmarla exigiría perfilar transferencias, por ejemplo con
   `ncu` o `nsys`, lo que está fuera del alcance de esta anotación.

### Trabajo pendiente

- Reescribir §4.1 del plan detallado con la medida defensiva corregida.
- Cruzar tiempo con energía sobre las mismas corridas para pronunciarse sobre
  EDP y no solo sobre tiempo (limitación 3).
- Al citar la campaña previa como evidencia del efecto de CPU al mínimo en el
  capítulo de Fase 4, verificar que sea posterior al 2026-08-19: los datos GPU
  anteriores a esa fecha corrieron en modo spin (ARC-153/154) y no son
  comparables.

### Criterio exacto de cierre

Se cierra cuando: (a) §4.1 del plan quede reescrito con el supuesto corregido;
(b) el capítulo de metodología del libro recoja la corrección a ARC-155 con los
números de ambas mediciones; y (c) la campaña final de GPU haya corrido con el
eje de CPU en `REF` únicamente, dejando constancia en su manifiesto de por qué
no incluye un nivel fijo.

## F1-GPU-005 — Los 6 wrappers `gpu_rajaperf_*` rechazaban corridas correctas por un `grep` contra una etiqueta de tuning inexistente

**Fecha de registro:** 2026-09-13
**Estado:** corregido y validado directamente en pacca; catálogo (`fase1_telemetria/catalog/catalog.yaml`) actualizado y sincronizado
**Kernels afectados:** `gpu_rajaperf_stream_copy`, `gpu_rajaperf_stream_triad`, `gpu_rajaperf_reduce3_int`, `gpu_rajaperf_indexlist_3loop`, `gpu_rajaperf_jacobi_2d`, `gpu_rajaperf_heat_3d` (los 6 kernels RAJAPerf-CUDA del catálogo GPU)

### Problema

En una sesión anterior se había detectado, sin diagnosticar, que los 6 kernels
`gpu_rajaperf_*` tienen 0 corridas con `samples.csv` válido en las 198
apariciones registradas en el historial completo del proyecto
(`hyperion-results/campaigns/pacca_*`). La hipótesis abierta era que estos
kernels no ejercitaban la GPU lo suficiente para producir telemetría útil.

Esa hipótesis era incorrecta. La causa real se encontró al intentar correr la
etapa `ncu` del cribado (job 7135, 2026-09-13): los 6 kernels fallaron con
`profiling_error` y el proceso completo terminó con `ExitCode 2:0`. Reproducido
cada uno directamente (sin `ncu` de por medio, con el shim de blocking-sync
inyectado igual que lo haría `runner.py`), los 6 imprimen `RAJAPerf <kernel>
checksum failed` y retornan `exit=0` — el fallo es 100 % reproducible,
independiente de `ncu`, y afecta por igual a kernels triviales
(`Stream_COPY`, una simple copia de arreglo) y complejos, lo cual ya era
indicio de que no era un error numérico real de cada kernel sino algo
estructural compartido por los 6.

Cada `bin/gpu_rajaperf_*` es un script adaptador que ejecuta el binario crudo
de RAJAPerf-CUDA v2025.12.1 y luego valida el resultado grepeando su propio
`RAJAPerf-checksum.txt` en busca de la línea `^Base_CUDA-default[[:space:]]
+PASSED`. Inspeccionando el `RAJAPerf-checksum.txt` real que genera el
binario, la variante siempre se reporta como `PASSED` — el checksum de
RAJAPerf nunca falló — pero bajo la etiqueta `Base_CUDA-block_256` (la mayoría
de kernels) o `Base_CUDA-blkatm_direct_256`/`Base_CUDA-blkatm_occgs_256`
(`Basic_REDUCE3_INT`, que usa dos tunings propios de reducción con átomos).
La cadena literal `Base_CUDA-default` que el wrapper buscaba nunca la genera
RAJAPerf: no es una convención real de nombrado de esta versión de la
suite. El wrapper llevaba desde su creación (2026-08-25, jobs 6517/6528)
rechazando corridas cuyo cómputo siempre fue correcto.

### Decisión

Se corrigen los 6 wrappers en `~/hyperion-kernels/bin/` (no versionados en
este repositorio; conservan copia `.bak_base_cuda_default` de respaldo). Para
los 5 con tuning único se reemplaza el literal por `Base_CUDA-block_256`. Para
`gpu_rajaperf_reduce3_int`, que reporta dos tunings distintos, el patrón se
generaliza a `^Base_CUDA-[A-Za-z0-9_]+[[:space:]]+PASSED` en vez de fijar un
segundo literal, para no repetir el mismo tipo de fragilidad si RAJAPerf
cambia de nuevo el nombre de un tuning.

Se actualiza `binary_checksum.pacca-a100` de los 6 kernels en
`fase1_telemetria/catalog/catalog.yaml` a la suma SHA-256 real de cada wrapper
corregido (CAT-10 valida ese checksum contra el archivo en disco en cada carga
del catálogo; sin esta actualización el catálogo fallaría a cargar, que es el
comportamiento correcto — nunca se ajustó CAT-10 para tolerar la discrepancia).

Se revisó también el wrapper análogo de CPU (`bin/rajaperf_polybench_3mm_omp`,
mismo patrón, etiqueta `Base_OpenMP-default`). En ese caso la etiqueta sí es
la real: OpenMP en esta versión de RAJAPerf solo tiene un tuning por kernel
(no requiere sufijo de tamaño de bloque como CUDA), así que no comparte el
bug. Verificado que corre y pasa tal como está; no se modifica.

### Evidencia de hardware

Reproducción directa en `paccaA100` (job 7144, sesión interactiva vía
`srun --jobid=7144`), con `LD_PRELOAD` del shim de blocking-sync inyectado:

| Kernel | Antes del fix | Después del fix |
|---|---|---|
| `gpu_rajaperf_stream_copy` | `checksum failed`, exit=0 | `Verification = SUCCESSFUL`, exit=0 |
| `gpu_rajaperf_stream_triad` | `checksum failed`, exit=0 | `Verification = SUCCESSFUL`, exit=0 |
| `gpu_rajaperf_reduce3_int` | `checksum failed`, exit=0 | `Verification = SUCCESSFUL`, exit=0 |
| `gpu_rajaperf_indexlist_3loop` | `checksum failed`, exit=0 | `Verification = SUCCESSFUL`, exit=0 |
| `gpu_rajaperf_jacobi_2d` | `checksum failed`, exit=0 | `Verification = SUCCESSFUL`, exit=0 |
| `gpu_rajaperf_heat_3d` | `checksum failed`, exit=0 | `Verification = SUCCESSFUL`, exit=0 |

El catálogo actualizado carga sin excepción (`common.hpc.catalog.load_catalog`)
con los 6 checksums nuevos, confirmando que los binarios en disco coinciden
exactamente con lo declarado.

### Consecuencia sobre el plan detallado

Los 6 `gpu_rajaperf_*` están declarados en `gpu_final.yaml`, la campaña final
de GPU. Sin este fix, la campaña habría gastado horas de nodo ejecutando 6
kernels que nunca habrían producido una corrida `accepted` (el `success_check`
del catálogo, `stdout_regex: "Verification = SUCCESSFUL"`, nunca habría
encontrado esa cadena), sin ninguna señal de alerta hasta revisar los
resultados al final. El hallazgo se originó, además, porque se pausó
deliberadamente el arranque automático de la campaña final para terminar de
diagnosticar por qué había fallado la etapa `ncu` del cribado — si se hubiera
saltado directo a la campaña final sin ese cribado, este bug habría quedado
enmascarado como "estos 6 kernels simplemente no producen datos", reforzando
la hipótesis errónea original.

### Limitaciones

1. No se investigó por qué RAJAPerf-CUDA v2025.12.1 nombra la variante
   `Base_CUDA-block_256` en vez de `Base_CUDA-default` — puede ser una
   convención de esta versión específica, de esta configuración de build, o
   de la GPU/arquitectura de compilación (A100, sm_80). Si el proyecto migra
   a otra versión de RAJAPerf o a otro tipo de GPU, el nombre de tuning podría
   volver a cambiar; el patrón generalizado de `reduce3_int` es más resistente
   a esto que el literal fijo usado en los otros 5.
2. No se corrió aún una corrida completa a través del harness de telemetría
   (`telemetry_kernel_launcher`) con pines de CPU/GPU reales para estos 6
   kernels — la verificación se hizo invocando el wrapper directamente. La
   verificación end-to-end queda pendiente de la próxima ejecución real de
   `screen-gpu` o de la campaña final.

### Trabajo pendiente

- Verificar con una corrida real del harness completo (no solo el wrapper) que
  los 6 kernels producen `samples.csv`/`metadata.json` válidos.
- Reejecutar la etapa `ncu` del cribado (bloqueada anteriormente por este
  mismo bug en 5 de los 6 casos, más el bug de `libnvJitLink.so.12` en
  `dual_cholesky_gpu`/`dual_spmv_gpu`) para completar la verdad Roofline de
  estos kernels, pendiente en `F1-GPU-004`.

### Criterio exacto de cierre

Se cierra cuando una corrida real de la campaña final de GPU produzca
`samples.csv` válido para los 6 `gpu_rajaperf_*`, confirmando que el fix
sobrevive al harness completo y no solo a la invocación directa del wrapper.

## F1-GEN-002 — `no_freq_reading` se calculaba contra el contexto de toda la corrida, no contra la lectura por ventana

**Fecha de registro:** 2026-09-13
**Estado:** corregido y verificado contra datos reales (`fase1_telemetria/postprocess.py`)
**Kernels afectados:** cualquier corrida reprocesada vía `repostprocess_campaign.py` (afectó de forma confirmada a `npb_bt` y `npb_mg`, probablemente a más)

### Problema

`postprocess.py` (línea ~788) calculaba `no_freq_reading = context.freq_khz_observed is None`.
`campaign.py` (ruta en vivo) sí llena `freq_khz_observed` al llamar a
`run_postprocess()`, pero `repostprocess_campaign.py` (re-postproceso de
`samples.csv` ya capturado, sin relanzar el kernel) nunca pasa ese parámetro,
así que queda `None` por defecto — sin importar si la columna real
`scaling_cur_freq_khz` del `samples.csv` sí tenía datos válidos por ventana.
Esto marcaba ventanas como `no_freq_reading` (y por lo tanto las excluía de
`target_windows_per_repetition`) de forma sistemática en todo reprocesamiento,
independientemente de si la frecuencia real se había leído bien.

### Decisión

Usar la lectura POR VENTANA (`row.get("freq_khz_observed")`, ya llenada en la
línea ~546 desde la columna real `scaling_cur_freq_khz` de `samples.csv`, per
ARC-135) en vez de `context.freq_khz_observed`. La fila ya tiene el valor
correcto independientemente de qué le pasen a `run_postprocess()`.

### Evidencia

Verificado en datos reales: las 1876 ventanas de `npb_mg` marcadas
`no_freq_reading` pasaron a `ok` correctamente. Los 9 runs de `npb_mg` y los 9
de `npb_bt` pasaron de `accepted=False` a `accepted=True` tras el fix.

### Criterio exacto de cierre

Cerrado: verificado contra corridas reales, sin regresión detectada.

---

## F1-GEN-003 — Un outlier de duración de 43.86s envenenaba el cálculo del warmup máximo de `cpu_rajaperf_polybench_jacobi_1d`

**Fecha de registro:** 2026-09-13
**Estado:** corregido y verificado contra datos reales (`fase1_telemetria/warmup_calibration.py`)
**Kernels afectados:** `cpu_rajaperf_polybench_jacobi_1d` (confirmado); cualquier kernel con una repetición anómala de duración

### Problema

`calibrate_kernel()` adopta el MÁXIMO del warmup detectado entre repeticiones
("criterio robusto para cubrir el peor caso observado"), sin excluir
outliers. La repetición `F4__rep02` de `cpu_rajaperf_polybench_jacobi_1d`
corrió 43.86s frente a ~6.2s de sus hermanas `F4` (rep01/rep03) — un outlier
~7x. Su punto de estabilización, detectado tarde, propuso
`warmup_seconds ≈ 51.7s` (43.1s × MARGIN 1.2), superando la duración
COMPLETA de cualquier otra corrida del kernel. Resultado: I10 rechazaba las 9
corridas con "0 ventanas ok".

### Decisión

`_reject_span_outliers()`: por cada `freq_level_id`, excluir del pool que
alimenta el máximo cualquier corrida con `total_span_s > 3.0x` la mediana de
duración de su propio nivel (nunca comparar entre niveles distintos, la
duración escala legítimamente con la frecuencia). `n_runs_analyzed` y
`per_run` conservan el conjunto completo para trazabilidad; solo el cómputo
del máximo usa el pool filtrado.

### Evidencia

Verificado en `/tmp/jacobi_calib_test`: nuevo valor 0.1705s calculado a partir
de 8/9 corridas, con nota explícita de exclusión del outlier
(`total_span_s=43.53 > 3x la mediana de su nivel (5.87s)`).

### Criterio exacto de cierre

Cerrado: verificado contra datos reales.

---

## F1-GEN-004 — `parse_ncu_csv` tomaba la primera línea de banner como encabezado real

**Fecha de registro:** 2026-09-13
**Estado:** corregido y verificado contra CSV real capturado en paccaA100 (`fase1_telemetria/ncu_convergence.py`)
**Kernels afectados:** todos los kernels GPU perfilados con `ncu --page raw` (impacto más amplio de esta sesión: pasó de 0/23 a 13/23 kernels con OI convergente)

### Problema

`parse_ncu_csv()` asumía `rows[0]` como encabezado y `rows[1:]` como datos.
Pero la salida cruda de `ncu` (versión 2026.1.1.0, instalada en paccaA100)
siempre antepone líneas de banner (`==PROF== Connected to process...`, el
stdout propio del programa perfilado, `==PROF== Disconnected...`) antes de la
tabla CSV real, y la tabla real tiene una fila de unidades inmediatamente
después del encabezado. Resultado: `metric_names_present` siempre `[]`,
`dram_bytes` siempre `0.0`, sin importar si el CSV real tenía datos válidos.

### Decisión

Buscar la fila cuyo conjunto de celdas normalizadas contenga `{"ID", "Kernel
Name"}` como el encabezado real (en vez de asumir `rows[0]`), y saltar la
fila de unidades que sigue inmediatamente (`rows[header_idx + 2:]`).

### Evidencia

Verificado directamente contra `gpu_dgemm_n4096__lc5.csv` capturado en
paccaA100: antes del fix, `metric_names_present: []`, `dram_bytes: 0.0`;
después, 7 métricas reales presentes, `launches_observed: 5`,
`dram_bytes: 10085416320.0`, `flops: 83886080.0`, `precision: fp64`.

### Criterio exacto de cierre

Cerrado: verificado contra datos reales; efecto confirmado en la corrida
completa de `ncu` sobre los 23 candidatos (13 convergieron tras el fix).

---

## F1-GPU-006 — Tamaños de screening GPU demasiado pequeños: ninguno sostenía uso real de GPU por 30s+

**Fecha de registro:** 2026-09-13
**Estado:** corregido en catálogo (`fase1_telemetria/catalog/catalog.yaml`); pendiente de re-medición con `ncu` para confirmar que la OI no se movió
**Kernels afectados:** `gpu_phasic_p010/p100/p1000`, `gpu_rajaperf_stream_triad`, `gpu_rajaperf_jacobi_2d`, `gpu_rajaperf_heat_3d`, `dual_axpy_gpu_N10000000`, `dual_cholesky_gpu_N2048`, `dual_spmv_gpu_N1000000`

### Problema

El director de tesis advirtió sobre el riesgo de que los kernels GPU no
ejercitaran el dispositivo lo suficiente. Verificado con datos reales de
`samples.csv`: de 12 kernels elegibles medidos, ninguno alcanzó 30s de
duración real (rango 2.3s-11.4s, todos por debajo incluso de su propio
`expected_runtime_seconds` declarado), y 6 de 12 mostraron uso promedio de
GPU por debajo del 25% (`gpu_util_pct`, muestreado por NVML cada ~47ms,
resolución suficiente para descartar que fuera un artefacto de muestreo).

Causa raíz para `gpu_phasic_*`: el harness corre por tiempo de pared fijo
(`--total-seconds`), no por conteo de lanzamientos; el overhead de
instrumentación de `ncu` competía por ese mismo presupuesto de tiempo,
impidiendo alcanzar los 50 lanzamientos solicitados sin importar cuántas
veces se reintentara.

Causa raíz para `dual_axpy_gpu`/`dual_cholesky_gpu`/`dual_spmv_gpu`: el
catálogo tenía 11-16 variantes de tamaño (N) preexistentes por familia, pero
TODAS calibradas al mismo `expected_runtime_seconds` nominal (~11s) —
`--iterations` se reduce proporcionalmente al subir N para mantener ese
mismo objetivo nominal, así que escoger una fila de N mayor no habría
ayudado: el problema no es el tamaño N, es que el número de iteraciones fue
calibrado contra un supuesto de velocidad de GPU equivocado (mucho más lento
que la A100 real).

Causa raíz para `gpu_rajaperf_stream_triad/jacobi_2d/heat_3d`: `--sizefact
100` fijo dentro del wrapper (elegido en su momento solo para escapar el
overhead fijo de arranque de contexto CUDA, ~380ms), sin considerar el
objetivo de sostener carga real.

### Decisión

- `gpu_phasic_*`: `--total-seconds` 20→60.
- RAJAPerf (stream_triad/jacobi_2d/heat_3d): `--sizefact` 100→400 en los
  wrappers (`bin/gpu_rajaperf_*`, no versionados en git, editados
  directamente en pacca con respaldo `.bak_sizefact100_*`); checksum
  recalculado y actualizado en el catálogo.
- `dual_axpy_gpu`/`dual_cholesky_gpu`/`dual_spmv_gpu`: se mantiene el mismo
  N (no se cambia de fila del catálogo) y se sube `--iterations`
  directamente, calculado a partir del tiempo real medido por iteración
  (0.0656s/iter, 0.0363s/iter, 0.01377s/iter respectivamente), apuntando a
  ~30-35s. Al ser el mismo N por iteración, la intensidad operacional por
  iteración no cambia — el catálogo ya documentaba estos valores de OI como
  "representativos de la operación, no del tamaño".

### Limitaciones

- `rodinia_dwt2d` ya usa el dataset más grande disponible en el catálogo
  (16384×16384) y aun así mide 2.3s/7% de uso — no hay margen de tamaño
  disponible sin generar un dataset sintético nuevo. Queda sin resolver.
- `rodinia_lavamd` solo tiene una variante de tamaño (`-boxes1d 70`) en el
  catálogo. Queda sin resolver por ahora.
- Los valores de `--sizefact`/`--iterations` nuevos son estimaciones
  lineales a partir de una sola medición por kernel, no verificadas aún con
  una segunda corrida real.

### Trabajo pendiente

- Re-ejecutar `ncu` sobre los 6 kernels tocados para confirmar que la OI
  medida no cambió de forma significativa (era el riesgo explícito que
  motivó esta entrada) y que la duración real se acerca al objetivo.
- Investigar datasets/tamaños alternativos para `rodinia_dwt2d` y
  `rodinia_lavamd`.

### Criterio exacto de cierre

Se cierra cuando una corrida real de `ncu` + `screen-gpu` confirme, para los
6 kernels tocados, duración real ≥25s y uso de GPU promedio ≥30%, sin que la
etiqueta Roofline (compute/memory-bound) haya cambiado respecto a la medida
a tamaño pequeño.

---

## F1-GPU-007 — `dual_gemm_gpu` mide como memory-bound extremo (OI=0.0153); implausible para GEMM denso, excluido hasta investigar

**Fecha de registro:** 2026-09-13
**Estado:** excluido del manifiesto de candidatos GPU (`campaign_pacca_phase_coverage_gpu_screen.yaml`); causa raíz NO confirmada, solo indicios

### Problema

`dual_gemm_gpu_N2048` midió `OI=0.0153` FLOP/byte vía `ncu` — muy por debajo
del ridge fp64 (3.36), es decir, memory-bound. Esto es físicamente
implausible para una multiplicación de matrices densa (cómputo O(N³) vs
memoria O(N²), debería ser de los kernels MÁS compute-bound del catálogo).

Evidencia: el contador `sm__sass_thread_inst_executed_op_dfma_pred_on.sum`
(instrucciones fused-multiply-add, el núcleo de cualquier GEMM real) da
**0** para `dual_gemm_gpu`. Para comparar, `dual_cholesky_gpu` al mismo N
(2048) muestra ~3.72e9 DFMA reales — confirma que el contador de `ncu`
funciona correctamente cuando el kernel sí hace el cómputo. Los bytes DRAM
medidos (~273MB/iteración) son razonables para el tamaño del problema; los
FLOPs contados (4.2M/iteración) son ~4000x menores que los ~1.72e10
esperados para una GEMM N=2048 completa.

El binario (`libexec/dual/gemm_gpu`, fuente embebida
`kernels/dual/gemm_gpu_dispatch.cu`, no versionada en el repo — solo existe
compilada) llama a `cublasDgemm_v2` real (confirmado por símbolos del
binario) y resuelve sus dependencias correctamente en el entorno de
medición (se descartó biblioteca faltante: con el entorno de la campaña
cargado, `ldd` no muestra "not found"; sí se encontró que resuelve
`libcublas.so.13` desde una instalación de CUDA 13.1 en `/usr/local/cuda`,
ajena al `nvhpc/23.1`/CUDA 12.0 que usa el resto del pipeline — divergencia
de entorno real, pero no se confirmó que sea LA causa). También se descartó
que fuera un artefacto del patrón de 12-13 procesos de vida corta que `ncu
--target-processes all` ve antes de conectarse al proceso real: ese mismo
patrón aparece en `dual_cholesky_gpu`/`dual_axpy_gpu`/`dual_spmv_gpu`/
`dual_stencil_gpu`, que sí miden correctamente.

Hipótesis más probable, sin confirmar: la llamada real a `cublasDgemm_v2`
dentro del binario usa una dimensión K casi degenerada (o un parámetro
alpha/beta incorrecto), moviendo los datos completos pero computando muy
poco — no se pudo verificar sin el código fuente.

### Decisión

Excluir `dual_gemm_gpu_N2048` del manifiesto de candidatos GPU
(`campaign_pacca_phase_coverage_gpu_screen.yaml`) hasta encontrar la causa
raíz real. No usar su etiqueta actual (memory-bound) para entrenar el
clasificador — es casi seguro que está mal.

### Limitaciones

- No hay código fuente disponible en el repositorio ni en pacca para este
  binario — la investigación se hizo por ingeniería inversa (símbolos,
  `ldd`, salida cruda de `ncu`), sin poder confirmar la causa exacta línea
  por línea.
- La instalación paralela de CUDA 13.1 en `/usr/local/cuda` no se investigó
  a fondo — no se sabe si afecta a otros binarios del sistema compartido
  (cuenta `pacca` es compartida con otro grupo de tesis).

### Trabajo pendiente

- Conseguir o reconstruir el código fuente de `gemm_gpu_dispatch.cu` para
  confirmar la causa exacta.
- Alternativa si no se puede diagnosticar: reemplazar `dual_gemm_gpu` con
  otro kernel GEMM ya validado del catálogo (`gpu_dgemm_n4096`/
  `cublas_dgemm_bench` ya miden correctamente) como representante de la
  familia GEMM compute-bound, y descartar `dual_gemm_gpu` definitivamente.

### Criterio exacto de cierre

Se cierra cuando se confirme la causa raíz del conteo de FLOPs anómalo (o se
decida formalmente reemplazar el kernel), y una nueva medición de `ncu`
muestre una OI fisicamente consistente con GEMM denso (muy por encima del
ridge fp64).

---

## F1-CPU-004 — `cpu_gap_bfs`/`ptrchase` excluidos del clasificador compute/memory: fuera de dominio del modelo Roofline clásico (latency-bound, no bandwidth-bound)

**Fecha de registro:** 2026-09-14
**Estado:** cerrado; excluidos del entrenamiento del clasificador. Documentado
como límite de alcance del modelo, no como bug ni como cobertura pendiente.

### Problema

Al correr `fase2_clasificador/run_phase_coverage.py --device cpu` sobre el
cribado CPU (`pacca_screen_20260909`, 498,796 filas usables/516,865, 25
familias), el diagnóstico mostró una dispersión enorme en
`log2(operational_intensity_uncore_real / i_ridge_used)`: de -26.7 hasta
+4.57 entre familias, con la franja cercana al ridge (|log2| <= 1, la región
que define la frontera de decisión del clasificador) sostenida casi en
solitario por `npb_bt` (47.2% de sus filas) y `npb_lu` (47.0%) -- 9 de 25
familias no aportan ni una sola fila compute_bound en ningún nivel de
frecuencia.

Dentro de ese grupo memory_only, `cpu_gap_bfs` (BFS sobre grafo, familia
`gap_bfs`) y `ptrchase` (recorrido de lista enlazada, puntero-persecución)
son un caso aparte: su mediana de `log2(OI/ridge)` es -26.70 y -26.71
respectivamente, es decir su intensidad operacional medida está ~10^8 veces
por debajo del ridge. Ningún barrido de tamaño de problema los va a acercar
-- el cuello de botella de ambos no es ancho de banda de memoria (lo que el
modelo Roofline FLOPs/bytes sí captura), es **latencia de acceso irregular**
(cada acceso depende del resultado del anterior, sin *prefetching* efectivo
posible): un régimen que el Roofline clásico no modela, sin importar cuántos
datos se muevan.

### Decisión

Excluir `cpu_gap_bfs` y `ptrchase` del conjunto de entrenamiento del
clasificador compute/memory-bound. No es un hueco de cobertura a llenar con
más corridas: es un límite de alcance del modelo Roofline FLOPs/bytes, que
por diseño solo separa "limitado por cómputo" vs "limitado por ancho de
banda de memoria" y no tiene una tercera clase para "limitado por latencia
de acceso irregular". La extensión que sí cubre ese régimen es el **Cache-Aware
Roofline Model (CARM, Ilic et al. 2014)**, fuera del alcance de este
proyecto de pregrado (ver [[feedback-undergrad-scope-discipline]] en
memoria de sesión).

Esta decisión queda respaldada por literatura reciente sobre el mismo
problema (clasificación ML compute/memory-bound vía Roofline): el póster
SC25 "*Classifying Performance Bounds Using Machine Learning*" (Littman y
Deakin, University of Bristol)
(https://sc25.supercomputing.org/proceedings/posters/poster_files/post215s2-file3.pdf)
reporta la misma asimetría estructural -- pocos ejemplos *floating-point
bound* en su muestra -- y la explica como un reflejo real de que la mayoría
del software HPC está limitado por *throughput* de datos, no como un defecto
de su conjunto de datos. Nuestro hallazgo (28% compute / 72% memory global,
9/25 familias sin ningún ejemplo compute_bound) es consistente con esa
observación: el desbalance es una propiedad del dominio (HPC real), no
necesariamente un artefacto de la campaña de cribado.

### Limitaciones

- La exclusión es binaria (todo o nada por kernel); no se evaluó si alguna
  variante de tamaño/entrada de `gap_bfs`/`ptrchase` podría, en el extremo,
  acercarse al ridge -- se descartó esa vía por la magnitud de la desviación
  (-26.7 en log2 no es un caso límite, es ~4 órdenes de magnitud más extremo
  que el siguiente kernel más memory-bound del catálogo, `gap_pr` en -10.63).
- El resto de familias memory_only del cribado (`dual_axpy`, `dual_fft`,
  `dual_spmv`, `dual_stencil`, `npb_cg`, `npb_mg`, `gap_pr`) NO se excluyen
  por esta misma entrada -- su desviación del ridge es mucho menor
  (-3.7 a -10.6) y sí son candidatos razonables para un barrido de tamaño de
  problema o de precisión (fp32/fp64), evaluado aparte.

### Trabajo pendiente

- Ninguno para esta entrada específica (excluidos, cerrado).
- Sigue abierto, en otra entrada futura: barrido de tamaño de problema
  dirigido a `dual_stencil`/`dual_spmv`/`dual_fft` y barrido de precisión
  fp32/fp64 sobre el resto del catálogo, para densificar la franja cercana
  al ridge más allá de `npb_bt`/`npb_lu`. Diseño acordado, ejecución
  pendiente de que `paccaA100` se libere del cribado GPU en curso.

### Criterio exacto de cierre

Cerrado en esta misma entrada: decisión tomada y documentada, sin trabajo de
cómputo pendiente. Se reabriría solo si el proyecto decide en el futuro
extender el clasificador con un régimen adicional de latencia (CARM u otro),
fuera del alcance actual.

---

## F1-XDEV-001 (actualización) — Cuarentena `dual_*` verificada e inspeccionada: 5/6 familias habilitadas, fuente fusionado a `main`

**Fecha de registro:** 2026-09-14
**Estado:** cuarentena de §2.1.1 punto 3 resuelta para
`dual_gemm`/`dual_cholesky`/`dual_fft`/`dual_stencil`/`dual_axpy`;
`dual_spmv` habilitado con una reserva documentada. Fuente fusionado a
`main`. Ejecución del barrido de tamaños ampliado sigue pendiente (bloqueada
por el cribado GPU en curso sobre `paccaA100`).

### Contexto

F1-XDEV-001 dejó la familia `dual_*` en cuarentena porque se construyó para
el trabajo de "selector de dispositivo" (`classifier/selector/`, pivote
2026-08-27), explícitamente fuera de alcance según
`Plan_Detallado_Realineacion_Hyperion.md` §7.2. El propio plan (§2.1.1 punto
3) fijó el criterio real para levantar la cuarentena: confirmar que la
telemetría de `dual_*` pasa por el pipeline de etiquetado Roofline real
(`uncore_imc`), no por una métrica ad hoc del selector -- y que su origen no
la descalifica automáticamente.

Esta entrada verifica ese criterio, motivada por el análisis de cobertura
del plano Roofline de CPU (ver `F1-CPU-004`): las familias `dual_stencil`,
`dual_axpy`, `dual_fft`, `dual_spmv` (junto con otras) resultaron
`memory_only` en el cribado, y ya existen en `catalog.yaml` barridos
log-espaciados completos de tamaño para las seis familias `dual_*` (13-16
puntos cada una, `N64` a `N16384`/`N31622777` según la operación) sin usar
en el manifiesto de cribado actual, que solo toma un tamaño por familia.

### Verificación 1 -- pipeline de etiquetado

Confirmado sin ambigüedad: el cribado `pacca_screen_20260909` ya midió
`dual_stencil_cpu_N1024`, `dual_axpy_cpu_N10000000`, etc. con el mismo
`run_campaign.py`/`postprocess.py` basado en `uncore_imc` que el resto del
catálogo -- no se reutilizó ninguna métrica del selector. Los valores de
`operational_intensity_uncore_real`/`i_ridge_used` en
`family_class_frequency_summary.csv` son mediciones reales de esta campaña.
Este punto queda satisfecho para las seis familias.

### Verificación 2 -- procedencia y optimización del código fuente

El fuente (`kernels/dual/*.c`/`*.cu`, generado y compilado según
`scripts/pacca/gen_dual_full_catalog.py`/`build_dual_kernels.sh`) nunca se
había fusionado a `main` -- solo el catálogo (232 entradas, `68a387b`) y los
binarios compilados en pacca. Se inspeccionó línea por línea cada una de las
seis implementaciones CPU (traídas ahora de `origin/fase-02`, HEAD real
`0ba20df`, la versión que coincide con los checksums ya declarados en
`catalog.yaml`):

| Familia | Implementación | Veredicto |
|---|---|---|
| `dual_gemm_cpu` | Reutiliza literalmente `kernels/dgemm/dgemm_bench.c` (mismo fuente que `dgemm_n2048`, ya en el catálogo original de 23 kernels) | Limpio -- no es código nuevo |
| `dual_cholesky_cpu` | `LAPACKE_dpotrf` real vía OpenBLAS, matriz SPD por `A = BᵀB + NI` | Limpio, LAPACK estándar |
| `dual_fft_cpu` | FFTW real (`fftw_plan_dft_2d`, `FFTW_ESTIMATE` para determinismo); FLOPs reportados por fórmula analítica estándar (no afecta la OI real, que sale de contadores `uncore`/`perf`, no del stdout del kernel) | Limpio |
| `dual_stencil_cpu` | Jacobi 2D de 5 puntos, OpenMP, doble buffer, fp64 | Limpio, mismo calibre que un stencil de Rodinia |
| `dual_axpy_cpu` | `cblas_daxpy` real, BLAS-1 puro | Limpio |
| `dual_spmv_cpu` | Matriz "dispersa" con **banda fija sintética** (7 no-ceros/fila, columnas `i±3 mod N`) -- el propio comentario del código lo describe como "misma conectividad que un stencil 1D de 7 puntos, generada sintéticamente" | Usable, pero **no representa acceso irregular real**; es un stencil con una capa de indirección, no un sustituto de un SpMV irregular genuino (ver el hueco de patrón irregular ya identificado en `Plan_Detallado_Realineacion_Hyperion.md` §2.1.1 punto 4, candidatos `rodinia_kmeans`/`nw`/`particlefilter`) |

Ninguna de las seis reutiliza una métrica ad hoc del selector ni toma
atajos de implementación -- todas llaman a rutinas numéricas reales
(OpenBLAS/LAPACK/FFTW) o implementan el algoritmo directamente.

### Decisión

1. **Cuarentena levantada** para `dual_gemm`/`dual_cholesky`/`dual_fft`/
   `dual_stencil`/`dual_axpy`: elegibles para ampliar el manifiesto de
   cribado con más puntos de tamaño (ya compilados, ya con checksum en
   `catalog.yaml`, sin trabajo de compilación nuevo).
2. `dual_spmv` **elegible con reserva documentada**: si se usa, el capítulo
   correspondiente debe aclarar que su "irregularidad" es sintética de banda
   fija, no un patrón de acceso disperso real.
3. **Fuente fusionado a `main`** en este mismo cambio: `kernels/dual/*.c`,
   `kernels/dual/*.cu`, `kernels/dual/dispatch_timing.h`,
   `scripts/pacca/gen_dual_full_catalog.py`,
   `scripts/pacca/build_dual_kernels.sh`,
   `scripts/pacca/screen_dual_frontier.sh` -- cierra el hueco de
   reproducibilidad/procedencia (antes solo existían en `origin/fase-02`,
   una rama fuera de alcance). Efecto colateral útil: el fuente de
   `gemm_gpu_dispatch.cu` que `F1-GPU-007` necesitaba para diagnosticar el
   conteo de FLOPs anómalo (`sm__sass_thread_inst_executed_op_dfma_pred_on.sum=0`)
   ahora también está disponible en `main`.

### Limitaciones

- No se verificó que los binarios YA COMPILADOS en pacca correspondan
  bit-a-bit a este fuente (solo se comparó que los checksums existentes en
  `catalog.yaml` provienen del mismo commit `0ba20df` de `fase-02` del que
  se trajo el fuente) -- si algún binario en `~/hyperion-kernels/bin` en
  pacca quedó de una compilación más vieja, el checksum del catálogo lo
  detectaría en el preflight (C02), no esta entrada.
- No se inspeccionaron los `.cu` de GPU con el mismo detalle línea por línea
  (la cuarentena que motivó esta entrada es específicamente CPU, vía
  `F1-CPU-004`); `F1-GPU-007` es la entrada correcta para retomar el lado
  GPU ahora que su fuente ya está disponible.

### Trabajo pendiente

- Ampliar `cpu_screen.yaml` (o un manifiesto de seguimiento nuevo) con 3-4
  tamaños intermedios adicionales para `dual_stencil`/`dual_spmv`/
  `dual_fft`/`dual_axpy`, dirigido a densificar la franja cercana al ridge
  identificada en `F1-CPU-004`. Bloqueado por `paccaA100` ocupado con el
  cribado GPU.
- Retomar `F1-GPU-007` con el fuente de `gemm_gpu_dispatch.cu` ya
  disponible en `main`.

### Criterio exacto de cierre

Esta actualización se cierra con la fusión del fuente y la tabla de
veredictos por familia. El punto pendiente (ampliar el manifiesto y correr
el barrido) se rastrea aparte, no bloquea cerrar esta entrada.

---

## F1-XDEV-005 (actualización) — `cpu_final.yaml` actualizado antes de su primer intento real: excluidos F1-CPU-004, ampliados `dual_*`

**Fecha de registro:** 2026-09-14
**Estado:** manifiesto actualizado, sin ejecutar todavía. La campaña final
(`job 7144`, `hyp_final_holder`) intentó `final_cpu`/`final_gpu` una vez
(2026-09-13 14:46, ambos fallaron en segundos, antes de medir nada real) y
quedó viva a la espera; desde entonces la asignación se usa para el cribado
GPU en curso (adjuntada manualmente, no un job aparte, para no perder el
puesto en la cola FIFO). Este cambio llega a tiempo: ningún dato real de la
campaña final se descarta ni se invalida.

### Problema

`scripts/pacca/final_campaign/cpu_final.yaml` (comprometido 2026-09-13,
antes de esta sesión) tenía la misma lista de 32 kernels del manifiesto de
cribado -- un solo tamaño por familia `dual_*`, incluía `cpu_gap_bfs`/
`ptrchase` (excluidos en `F1-CPU-004`) y los dos RAJAPerf demasiado cortos
para producir ventanas `ok` (`cpu_rajaperf_lcals_tridiag_elim`/
`basic_init3`). De reintentarse tal cual, la campaña final (960 corridas,
~500 horas-núcleo proyectadas) habría heredado exactamente los mismos
huecos de cobertura diagnosticados en `F1-CPU-004`/`F1-XDEV-001`, gastando
cómputo real en configuraciones ya sabidas inviables.

### Decisión

Editar `cpu_final.yaml` directamente (el propio archivo documenta que es
editable hasta que el job arranque, sin necesidad de re-someter):

1. **Quitar** `cpu_gap_bfs`, `ptrchase` (`F1-CPU-004`, latency-bound, fuera
   de dominio del modelo).
2. **Quitar** `cpu_rajaperf_lcals_tridiag_elim`, `cpu_rajaperf_basic_init3`
   (con su warmup real de 2.3-3.6s dan 0 ventanas `ok` en cualquier nivel de
   frecuencia -- correr 30 repeticiones cada uno en la campaña final de
   nada serviría). El arreglo real (tamaño de problema mayor) sigue
   diferido, no bloquea esta campaña.
3. **Ampliar `dual_*`** con la cuarentena ya levantada (`F1-XDEV-001`
   actualización anterior): +2 tamaños nuevos cada una para
   `dual_fft`/`dual_axpy`/`dual_stencil`/`dual_cholesky` (además del que ya
   tenía el cribado), y `dual_spmv` agregado por primera vez a esta campaña
   (2 tamaños, no tenía ninguno) -- con la reserva ya documentada de que su
   "dispersión" es banda fija sintética. `dual_gemm` se deja con su único
   tamaño (ya es ancla compute-bound confiada, sin urgencia de ampliarlo
   ahora).

Resultado: **38 kernels x 10 niveles x 3 repeticiones = 1140 corridas**
(antes 960). `projected_campaign_bytes`/`projected_core_hours` escalados
proporcionalmente (96 GiB de margen, ~600 horas-núcleo). Verificado que el
YAML resultante parsea y la aritmética cuadra (`38*10*3=1140`).

### Limitaciones

- Los tamaños `dual_*` añadidos se eligieron por criterio de cobertura
  (uno pequeño para cruzar el límite de caché, uno intermedio), no por una
  búsqueda exhaustiva del punto óptimo -- si el barrido real muestra que
  ninguno se acerca más al ridge que el tamaño ya usado en el cribado, hay
  margen en el catálogo (`N64` a `N16384`/`N31622777`) para iterar.
- No se tocó `gpu_final.yaml` en este cambio -- el lado GPU sigue
  pendiente del propio cribado en curso bajo `job 7144`.

### Trabajo pendiente

- Reintentar `final_cpu` bajo `job 7144` (o el que corresponda cuando el
  cribado GPU cierre) con el manifiesto ya actualizado.
- Actualizar `gpu_final.yaml` con los hallazgos equivalentes del lado GPU
  una vez cierre su propio cribado.

### Criterio exacto de cierre

Se cierra cuando la campaña final CPU corra de punta a punta con este
manifiesto y produzca el dataset que alimentará el entrenamiento real del
clasificador con `leave_one_familia_out` (⚠️ corrección 2026-09-14: esta
función YA EXISTE en `fase2_clasificador/eval/protocol.py`, junto con
`derive_kernel_family`/`assert_no_familia_leak`, XGBoost y serialización en
`fase2_clasificador/training/train_phase.py` -- una sesión anterior dijo
por error que faltaba construirla, citando el estado de la rama `fase-02`
en vez de revisar `main`. Lo que sí falta, verificado leyendo el código:
`class_weight` sin balancear en ningún modelo, solo F1 macro reportado
(falta F1 por clase + matriz de confusión), y las "dos variantes" de
features con/sin `operational_intensity` no automatizadas).

---

## F1-CPU-005 — Barrido de tamaño en NPB: probado en `npb_ft`, resultado negativo

**Fecha de registro:** 2026-09-14
**Estado:** cerrado, negativo. No se repite con `cg`/`mg`/`sp`.

### Problema / hipótesis a probar

Tras confirmar que achicar el tamaño de problema en `dual_*` sí acerca el
OI al ridge (actualización de F1-XDEV-001, mismo día), se probó si el
mismo mecanismo aplica a NPB -- dirigido a `npb_ft` por ser el único de
los 6 NPB con evidencia previa (ARC-71, clase B vs C) de que su
clasificación SÍ responde al tamaño.

### Procedimiento

Compilado en `paccaA100` (gfortran 12.4.0, módulo `gnu12` con `source`
explícito del init de Lmod -- sin eso resuelve en silencio al gfortran de
sistema 8.5.0, encontrado y corregido en el camino). Clases probadas en
orden creciente:

- **S** (64×64×64): `Time in seconds = 0.00` -- demasiado corto, mismo
  modo de fallo que los RAJAPerf excluidos en F1-CPU-004 (0 ventanas
  posibles a `target_windows_per_repetition=50`). Descartada sin someter
  campaña.
- **W** (128×128×32): `Time in seconds = 0.01` -- mismo problema.
  Descartada sin someter campaña.
- **A** (256×256×128): `0.32s` con 6 hilos pineados (`taskset`) -- único
  tamaño intermedio disponible bajo la clase B ya catalogada. Sometida a
  smoke test real (REF, 3 repeticiones): **3/3 aceptadas**.

### Resultado

`npb_ft` clase A: `log2(OI/ridge) = -3.65`, 0% compute, 0% cerca del
ridge. **Clase B (ya en catálogo): `log2(OI/ridge) = -1.98`.** Achicar el
tamaño alejó el kernel del ridge en vez de acercarlo -- el resultado
opuesto al de `dual_*`.

### Por qué no funcionó (interpretación, no solo el número)

`dual_*` son microbenchmarks de un solo patrón (stencil/axpy/fft/spmv/
cholesky) donde el tamaño controla directamente cuánto del working set
cabe en caché. `npb_ft` hace FFT 3D con transposiciones y comunicación
multi-fase entre pasadas -- a menor tamaño, el costo fijo de esas fases
pesa proporcionalmente más frente al cómputo real, así que el kernel se
vuelve MÁS memory-bound, no menos. El mecanismo que funcionó en `dual_*`
no es generalizable a solvers multi-fase sin más evidencia.

### Decisión

No se repite el experimento con `npb_cg`/`npb_mg`/`npb_sp` -- eran los
candidatos con MENOS evidencia previa a favor que `npb_ft` (que ya
fracasó), y ARC-71 ya había mostrado que esos tres son los más estables
frente a cambios de tamaño (B vs C, ≤1.1pp de diferencia). Se documenta el
catálogo (`npb_ft_a`) y el binario compilado, pero no se usa en
`cpu_final.yaml` -- el hallazgo es negativo, no hay ganancia que capturar.

### Limitaciones

- Muestra chica (72 filas usables, corrida de 0.32s) -- el resultado es
  consistente en dirección pero no tiene la robustez estadística de una
  campaña completa. No cambia la decisión: incluso con más muestras, la
  dirección (más lejos del ridge, no más cerca) ya contradice la hipótesis
  que se quería confirmar.
- `warmup_seconds` de `npb_ft_a` no se calibró (se dejó el valor heredado
  de clase B, 0.5s) -- no se justifica calibrar un kernel que ya se
  descartó.

### Trabajo pendiente

Ninguno -- cerrado, negativo.

### Criterio exacto de cierre

Cerrado en esta misma entrada.

---

## F1-CPU-006 — `--repfact` resuelve `cpu_rajaperf_lcals_tridiag_elim`/`basic_init3`, re-incluidos en la campaña final

**Fecha de registro:** 2026-09-14
**Estado:** cerrado, positivo. Verificado con smoke test real (REF+F8, 12/12
aceptadas). Re-incluidos en `cpu_final.yaml` (40 kernels, 1200 corridas).

### Problema

F1-CPU-004/actualización de F1-XDEV-005 había dejado estos dos kernels
fuera de la campaña final: con su warmup real calibrado (2.3-3.6s), su
duración base (medida F4≈2.9s/2.2s en la rejilla del cribado) es más corta
que su propio transitorio -- 0 ventanas `ok` garantizado (`I10`) sin
importar la frecuencia. Quedó registrado como "diferido", pendiente de un
tamaño de problema mayor.

### Solución

El wrapper (`bin/cpu_rajaperf_*`, script bash que invoca el binario real de
RAJAPerf con `--memory-touched 125829120` fijo) ya reenviaba argumentos
extra al binario real (`"$@"`) -- no hizo falta recompilar nada. RAJAPerf
expone `--repfact <F>` (multiplicador de repeticiones) como opción
**distinta** de `--size`/`--memory-touched` (tamaño de problema): confirmado
leyendo el CSV de salida de RAJAPerf que `BytesTouched/rep` se mantiene
igual con `--repfact 30` que sin él -- el tamaño/OI que ya pasó el tamizaje
v2 (alpha<0.226) queda intacto, solo se repite más veces el mismo trabajo.

Medido directamente (no estimado): `--repfact 2` en `lcals_tridiag_elim` da
4.16s a velocidad nativa (turbo); `--repfact 3` en `basic_init3` da 3.71s.
A la frecuencia más lenta de la rejilla (~4x más lenta) eso proyecta
~17-18s y ~16s respectivamente -- margen amplio sobre el warmup real.

Aplicado en dos catálogos: el del repo (`fase1_telemetria/catalog/
catalog.yaml`, vía commit) y el desplegado en pacca que usa realmente
`cpu_final.yaml` (`screening/pacca_screen_20260909/config/
catalog.screening.yaml`, editado directo con backup `.bak_f1cpu006` porque
NO está versionado en git y ya traía `warmup_seconds` calibrados de una
sesión anterior que no se querían perder). Solo se tocaron `exec_args` y
`expected_runtime_seconds` en la copia desplegada, nada más.

### Verificado con hardware real

Smoke test (`pacca_cpu_smoke_rajaperf_fix_20260914`, REF + F8, 3 reps c/u):
**12/12 aceptadas, 0 rechazadas, matriz completa.** F8 es la frecuencia más
restrictiva de la rejilla (peor caso), y pasó limpio.

### Decisión

Re-incluidos en `cpu_final.yaml`: **40 kernels × 10 niveles × 3 reps =
1200 corridas** (antes 1140). `projected_core_hours`/`projected_campaign_
bytes` reescalados proporcionalmente.

### Limitaciones

- `warmup_seconds` en ambos catálogos sigue siendo el valor calibrado con
  la corrida CORTA (2.3197/3.5689) -- no se recalibró con la corrida larga
  (`--repfact` aplicado). Es razonable esperar que el warmup real (tiempo
  hasta que el IPC se estabiliza) no dependa de cuántas repeticiones se
  hagan después, solo del arranque -- pero no está confirmado con
  `warmup_calibration.py` sobre la corrida larga, queda como supuesto
  hasta que la campaña final produzca datos para verificarlo.
- El repositorio y la copia desplegada en pacca del catálogo pueden
  divergir con el tiempo si alguna sesión futura edita una sin la otra --
  anotado aquí explícitamente para que quien retome esto lo sepa (la copia
  desplegada acumula calibraciones reales de campaña que el repo no tiene).

### Trabajo pendiente

Ninguno para esta entrada -- cerrado. La recalibración de warmup (si hace
falta) es trabajo de la campaña final en sí, no de esta corrección.

### Criterio exacto de cierre

Cerrado: arreglo verificado con hardware real, re-incluido en el
manifiesto que va a correr.

---

## F1-CPU-007 — Caza dirigida de un segundo punto cerca del ridge en `dual_*`: éxito en `stencil`, negativo en `axpy`/`spmv`/`fft`

**Fecha de registro:** 2026-09-14
**Estado:** cerrado. `dual_stencil_cpu_N512` incluido en `cpu_final.yaml`
(41 kernels, 1230 corridas). No se persigue más tamaño intermedio en
`axpy`/`spmv`/`fft` -- requeriría catálogo nuevo, mismo costo que el
experimento NPB ya descartado.

### Motivación

Tras confirmar que `dual_cholesky_cpu_N1024` cae casi exacto sobre el
ridge (87.6% cerca, actualización de F1-XDEV-001), se probó el mismo
truco con las otras cuatro familias `dual_*` que solo tenían un punto
"chico" (compute, sobrepasado de largo) y uno "grande" (memory): elegir el
punto medio geométrico del tamaño de problema entre ambos y ver si cae
cerca del ridge, igual que pasó con cholesky.

### Procedimiento

Smoke test (REF, 3 repeticiones, 4 kernels nuevos): `dual_stencil_cpu_
N512` (entre N256 y N1024), `dual_axpy_cpu_N1000000` (entre N100000 y
N3162278), `dual_fft_cpu_N384` (entre N128 y N1024), `dual_spmv_cpu_
N316228` (media geométrica exacta de N100000/N1000000, ya existía en
catálogo). **12/12 aceptadas.**

### Resultado

| kernel | log2(OI/ridge) | cerca del ridge |
|---|---|---|
| `dual_stencil_cpu_N512` | **+0.70** | **67.2%** -- éxito |
| `dual_fft_cpu_N384` | +6.18 | 0% -- más adentro del lado compute que N128 (+5.87), no monótono con N |
| `dual_axpy_cpu_N1000000` | -6.87 | 0% -- casi igual de memory que el extremo N3162278 (-7.32) |
| `dual_spmv_cpu_N316228` | -5.50 | 0% -- también cerca del extremo memory pese a ser la media geométrica exacta |

### Interpretación

Solo `stencil` repitió el éxito de `cholesky`. En `axpy`/`spmv`, la media
geométrica del TAMAÑO no corresponde a la media geométrica del OI -- la
transición compute→memory es marcadamente no lineal y, por lo que se ve,
está sesgada mucho más cerca del extremo compute (N100000) de lo que el
punto medio geométrico del tamaño sugiere; haría falta un tamaño bastante
más chico que N316228/N1000000 para acercarse, y esos tamaños finos no
están en el catálogo (mismo problema de "hay que compilar/generar uno
nuevo" que en F1-CPU-005). En `fft`, el resultado ni siquiera fue
monótono con N -- probable efecto de eficiencia de radix de FFTW (N=384
factoriza más favorablemente que valores vecinos), no un problema de
"tamaño más grande/chico" en el sentido simple que aplica a stencil/axpy.

### Decisión

Se incluye `dual_stencil_cpu_N512` en `cpu_final.yaml`. No se persigue
más precisión en `axpy`/`spmv`/`fft` -- el costo (generar tamaños nuevos
vía `scripts/pacca/gen_dual_full_catalog.py`, o para `fft` explorar qué
valores de N factorizan bien) no se justifica frente a lo ya ganado
(`cholesky` + `stencil`, dos anclas nuevas reales cerca del ridge, además
de `npb_bt`/`npb_lu`).

### Limitaciones

- Solo REF, una repetición corta -- mismo nivel de evidencia que los
  demás smoke tests de esta sesión, no la robustez de una campaña
  completa.
- La hipótesis de "media geométrica del tamaño ≈ media geométrica del
  OI" (que sí funcionó para stencil y cholesky) queda refutada como regla
  general -- no generalizar a otras familias sin verificar primero.

### Trabajo pendiente

Ninguno para esta entrada -- cerrado. Si en el futuro se quiere refinar
`axpy`/`spmv`, el próximo paso sería un tamaño entre N100000 y N316228
(más cerca del extremo compute), generado con el script existente.

### Criterio exacto de cierre

Cerrado: decisión tomada (incluir stencil, no perseguir el resto),
documentada con los cuatro resultados reales.

## F1-CPU-008 — Kernel nuevo `lbm_cpu` (LBM D2Q9-BGK): compilado y verificado, pero el barrido de tamaño da resultado negativo (filo de cuchillo, no banda cerca del ridge)

**Fecha de registro:** 2026-09-14
**Estado:** cerrado. Kernel `lbm_cpu` queda en el catálogo (util como
familia algorítmica distinta, documentada), pero NINGUNO de los 13 tamaños
probados entra a `cpu_final.yaml` -- ninguno cae cerca del ridge.

### Motivación

El usuario pidió buscar en internet alternativas de benchmark usadas por
el paper de Littman & Deakin (SC25 poster) para densificar la zona cerca
del ridge, dado que 4 anclas (`npb_bt`, `npb_lu`,
`dual_cholesky_cpu_N1024`, `dual_stencil_cpu_N512`) seguían pareciendo
pocas. El paper cita LBM D2Q9 como ejemplo de benchmark "LLC cache or
main memory bandwidth bound" -- justo el régimen intermedio buscado.

### Procedencia de la fuente

El repo que usa el paper como fuente de ese benchmark
(`UoB-HPC/advanced-hpc-lbm`) es material de entrega de un curso de la
Universidad de Bristol y **no tiene archivo LICENSE** (confirmado 404 en
`/license` de la API de GitHub, 2026-09-14) -- no se vendorizó por falta
de procedencia clara para una tesis. En su lugar se implementó
`kernels/lbm/lbm_d2q9_cpu_bench.c`: método LBM D2Q9-BGK estándar
(Bhatnagar-Gross-Krook, formulación de Kruger et al. 2017) desde cero,
sin copiar ninguna base de código específica. Frontera periódica, datos
sintéticos generados en el binario (mismo criterio que `dual_*`).
Verificación por invariante analítico (masa total conservada bajo
streaming periódico + colisión BGK), no por datos de referencia externos
-- confirmado con error relativo ~1e-15 en pruebas locales y en
paccaA100.

Compilado en paccaA100 vía `srun --jobid=7144` (holder), con
`module load gnu12/12.4.0` explícito (mismo patrón de todo el proyecto).
Checksum del wrapper `bin/lbm_cpu`:
`308fb92642d1a38095d15dbc165987a00438f491bc77540cfee5eb2cbf2ef8f3`.

### Procedimiento

Tres rondas de smoke test (REF, 3 repeticiones), acortando el rango cada
vez: N128-N512 (6 tamaños, 18/18 aceptadas) → N208-N248 (4 tamaños,
12/12 aceptadas) → N212-N220 (3 tamaños, 9/9 aceptadas).

### Resultado

| N | huella total (3 buffers) | log2(OI/ridge) | cerca del ridge |
|---|---|---|---|
| 128 | 3.5 MB | +3.86 | 0% |
| 192 | 8.0 MB | +2.39 | 0.8% |
| 208 | 9.3 MB | +2.67 | 0% |
| 212 | 9.7 MB | +1.83 | 1.8% |
| 216 | 10.1 MB | +2.75 | 0.8% |
| 220 | 10.5 MB | **-3.07** | 0% |
| 224 | 10.8 MB | -3.09 | 0% |
| 240 | 12.4 MB | -3.10 | 0% |
| 256 | 14.2 MB | -3.10 | 0% |
| 320 | 22.1 MB | -3.25 | 0% |
| 384 | 31.9 MB | -3.72 | 0% |
| 512 | 56.6 MB | -4.21 | 0% |

### Interpretación

No hay banda cerca del ridge en ningún punto -- el kernel es
**biestable**: por debajo de un umbral de huella (~10.1-10.5 MB total,
~1.7 MB por hilo delegado, compatible con el tamaño de L2 por núcleo, no
L3) se comporta como fuertemente compute-bound (log2 entre +1.8 y +3.9);
por encima, colapsa de golpe a memory-bound (log2 ≈ -3.1, valor que
**satura** y ya no cambia entre N=220 y N=512, un factor 5x en tamaño).
No existe una pendiente intermedia que cruce log2=0.

La causa física es coherente con el diseño del kernel: cada paso de
colisión+streaming visita cada celda un número FIJO de veces (patrón de
barrido, sin reutilización entre iteraciones más allá de un paso), igual
que AXPY (BLAS-1). A diferencia de un kernel con bloqueo/tiling (GEMM),
donde la fracción de datos que debe re-transferirse desde DRAM crece
suavemente con el tamaño relativo al cache, aquí en cuanto la huella no
cabe en L2 el barrido entero se vuelve un fallo de cache en cada pasada
-- de ahí el salto abrupto en vez de una rampa. Es la misma familia de
problema que ya cerró en negativo `axpy`/`spmv` en F1-CPU-007, con un
mecanismo de fondo distinto (límite de capacidad de cache vs. intensidad
aritmética intrínsecamente baja).

### Decisión

No se agrega ningún tamaño de `lbm_cpu` a `cpu_final.yaml`. El kernel
queda en el catálogo (`fase1_telemetria/catalog/catalog.yaml`, 13
entradas `lbm_cpu_N*`) documentado como intento negativo, por si en el
futuro se reimplementa con bloqueo por bandas (streaming en tiles que
quepan en L2, con más reutilización) -- fuera de alcance de este
trabajo de pregrado (ver disciplina de alcance ya establecida).

### Limitaciones

- Solo REF, una repetición corta por ronda -- mismo nivel de evidencia
  que el resto de los smoke tests de esta sesión.
- No se probó una implementación con bloqueo/tiling explícito, que en
  teoría podría dar una rampa más suave -- eso es una reescritura no
  trivial del kernel, no un simple barrido de tamaño.
- El umbral de cache observado (~10.1-10.5 MB de huella total para 6
  hilos) es una medición indirecta, no una lectura directa de `lscpu`
  para el tamaño de L2/L3 de los Gold 5315Y de paccaA100.

### Trabajo pendiente

Ninguno para esta entrada -- cerrado. Si se retoma en el futuro, el
siguiente paso sería reimplementar `collide_stream` con bloqueo espacial
(procesar la malla en franjas que quepan en L2, con más de un paso por
franja antes de pasar a la siguiente) para intentar una transición
gradual en vez de un salto de cache.

### Criterio exacto de cierre

Cerrado: 13 tamaños probados con evidencia real de campaña (smoke test,
39/39 corridas aceptadas en total), decisión tomada de no incluir
ninguno en la campaña final, documentado con los 12 resultados de OI
medidos y la interpretación física del salto abrupto.

## F1-CPU-009 — Kernel nuevo `hpccg_cpu` (CG sobre Laplaciano 3D): éxito real, `hpccg_cpu_N46` con 66.3% cerca del ridge

**Fecha de registro:** 2026-09-14
**Estado:** cerrado. `hpccg_cpu_N46` se agrega a `cpu_final.yaml` (42
kernels, 1260 corridas). Tercera ancla nueva cerca del ridge de la
sesión, junto a `dual_cholesky_cpu_N1024` (87.6%) y `dual_stencil_cpu_
N512` (67.2%).

### Motivación

Tras el resultado negativo de `lbm_cpu` (F1-CPU-008), el usuario pidió
explícitamente no asumir que el resto de benchmarks del paper Littman/
Deakin fallarían igual -- "experimenta, aseguremos, no supongamos". Se
implementó `kernels/hpccg/hpccg_cg_cpu_bench.c`: CG estándar sobre un
Laplaciano discreto 3D de 7 puntos (malla NxNxN), CSR explícito,
diagonal 6.1 (dominancia diagonal estricta -> SPD garantizado). No
vendorizado de Mantevo/HPCCG -- implementación propia, mismo criterio
que `lbm_cpu`. Verificación por decrecimiento monótono del residuo
(propiedad garantizada de CG sobre SPD), confirmada localmente y en
paccaA100 (residuo final << ||b|| en todos los tamaños probados).

Hipótesis explícita a probar (no asumida): a diferencia de LBM (un paso
por despacho, sin reutilización), CG hace `CG_STEPS=20` pasadas por
despacho sobre la MISMA matriz -- si la matriz cabe en cache, esa
reutilización podría dar una transición más gradual que el salto de
cache abrupto de `lbm_cpu`.

Compilado en paccaA100 vía `srun --jobid=7144`, checksum del wrapper
`bin/hpccg_cpu`: `837339bd72c5477c52bc28e01b28ead9dda06eaa4f879d1c193973be1db2f383`.

### Procedimiento

Tres rondas de smoke test (REF, 3 repeticiones), acortando el rango:
N16-N128 (11 tamaños, 33/33 aceptadas) → N42-N46 (3 tamaños, 9/9
aceptadas, advertencia `CAL-10/D04: cv_pct=15.17%` en la calibración de
referencia, por encima del umbral 5% -- no bloqueante, aceptado igual,
ver Limitaciones).

### Resultado

| N | huella total (aprox.) | log2(OI/ridge) | cerca del ridge |
|---|---|---|---|
| 16 | 0.6 MB | +4.72 | 0.1% |
| 24 | 2.1 MB | +5.67 | 0.2% |
| 32 | 4.9 MB | +5.14 | 0% |
| 40 | 9.6 MB | +5.48 | 0% |
| 42 | 11.1 MB | +3.94 | 0% |
| 44 | 12.8 MB | +2.04 | 0.3% |
| **46** | **14.6 MB** | **-0.27** | **66.3% -- éxito** |
| 48 | 16.6 MB | -2.81 | 0% |
| 56 | 26.4 MB | -5.04 | 0% |
| 64 | 39.4 MB | -5.49 | 0% |
| 80 | 77.0 MB | -5.81 | 0% |
| 96 | 133.1 MB | -6.19 | 0% |
| 112 | 211.3 MB | -6.40 | 0% |
| 128 | 315.5 MB | -6.49 | 0% |

### Interpretación

A diferencia de `lbm_cpu` (plateau plano en el lado memoria, sin
gradiente), aquí el lado memoria SÍ decae de forma gradual y monótona
conforme N crece (-2.81 en N48 hasta -6.49 en N128, asíntota que no
termina de aplanarse ni en N=128) -- consistente con la hipótesis de
reutilización de matriz: cuanto más grande la matriz relativa al
cache, más se acerca el comportamiento a "sin reutilización" (similar
a SpMV puro), y el arrastre es progresivo, no un colapso de golpe.

El cruce compute→memory sigue siendo abrupto en términos de N (de
+2.04 en N44 a -0.27 en N46, salto de solo 2 unidades de malla), pero
a diferencia de `lbm_cpu`, el punto de cruce mismo SÍ cae casi exacto
sobre el ridge (log2=-0.27, no un vacío entre dos valores lejanos) --
la ventana de "cerca del ridge" (|log2|<1, criterio del proyecto) tiene
ancho suficiente en el espacio de N para que un solo tamaño entero
(N=46) capture una fracción alta y estable de corridas.

### Decisión

Se agrega `hpccg_cpu_N46` a `cpu_final.yaml`. Tercera familia
algorítmica nueva de la sesión con ancla real cerca del ridge (CG con
matriz dispersa estructurada, distinta de GEMM/Cholesky denso y de
stencil Jacobi simple). No se agrega ningún otro tamaño de `hpccg_cpu`
-- ya hay una ancla de calidad comparable a `dual_stencil_cpu_N512`.

### Limitaciones

- Solo REF, una repetición corta por ronda -- mismo nivel de evidencia
  que el resto de los smoke tests de la sesión.
- La ronda N42-N46 disparó `CAL-10/D04` (dispersión de la calibración
  de referencia 15.17% > umbral 5%) -- indicio de ruido del sistema
  (nodo compartido) en ese momento puntual. No bloqueó la aceptación
  de las 9 corridas y el resultado (66.3% cerca del ridge, muy por
  encima del ruido esperable) es robusto a esa dispersión, pero
  conviene confirmar en la campaña final con las 3 repeticiones
  completas y los 10 niveles de frecuencia, no solo REF.
- CG_STEPS=20 es fijo (no ajustado a tolerancia de convergencia) --
  correcto para reproducibilidad de FLOPs por despacho, pero significa
  que el punto de cruce (N=46) es específico de ese conteo de pasadas;
  cambiar CG_STEPS movería el punto (más pasadas = más reutilización
  de matriz = el cruce se movería hacia N más grandes).

### Trabajo pendiente

Ninguno para esta entrada -- cerrado. La campaña final confirmará si
el 66.3% se sostiene con calibración estable y los 10 niveles de
frecuencia.

### Criterio exacto de cierre

Cerrado: 14 tamaños probados con evidencia real de campaña (smoke
test, 42/42 corridas aceptadas en total), resultado positivo
identificado (N=46, 66.3% cerca del ridge), decisión tomada de
incluirlo en `cpu_final.yaml`, documentado con los 14 resultados de OI
medidos y la interpretación de por qué CG (con reutilización de
matriz) se comporta distinto a LBM (sin reutilización).

## F1-CPU-010 — Segundo intento en `dual_fft`: éxito, `dual_fft_cpu_N472` con 97.1% cerca del ridge, la mejor ancla nueva de la sesión

**Fecha de registro:** 2026-09-14
**Estado:** cerrado. `dual_fft_cpu_N472` se agrega a `cpu_final.yaml` (43
kernels, 1290 corridas). Cuarta ancla nueva cerca del ridge de la sesión.

### Motivación

Tras el éxito de `hpccg_cpu_N46` (F1-CPU-009), el usuario pidió un
último intento antes de cerrar el hilo: "intentemos ya por ultimo con
esos que recomiendas, y ya si no da, seguimos asi". Se decidió retomar
`dual_fft` (dejada negativa en F1-CPU-007 por un muestreo no monótono:
N=384 daba OI MÁS compute que N=128, atribuido a eficiencia de radix de
FFTW) en vez de escribir un kernel nuevo tipo "hot" (heat diffusion vía
CG, redundante en mecanismo con `hpccg_cpu` ya probado) -- FFT no
necesita compilar nada nuevo (el binario `fft_cpu` ya existe) y tiene
una estructura de reutilización genuinamente distinta (múltiples pasadas
internas O(log N) de FFTW sobre el mismo buffer), ni un solo paso
(`lbm_cpu`) ni pasadas explícitas completas (`hpccg_cpu`).

Esta vez se usaron tamaños "5-smooth" (factorizan en 2/3/5, eficientes
para el planificador de FFTW) en vez de un punto medio geométrico
arbitrario, para eliminar el ruido de radix que arruinó el intento
anterior.

### Procedimiento

Tres rondas de smoke test (REF, 3 repeticiones, turbo desactivado):
N512-N1024 (7 tamaños, reutilizando `dual_fft_cpu_N512`/`N768` ya
existentes en el catálogo -- un intento inicial con un `N512` duplicado
falló por `CAT-08: id de catálogo duplicado`, corregido reutilizando la
entrada existente) → N400-N480 (4 tamaños nuevos, 12/12 aceptadas) →
N456-N472 (3 tamaños nuevos, 9/9 aceptadas).

### Resultado

| N | huella total (2 buffers) | log2(OI/ridge) | cerca del ridge |
|---|---|---|---|
| 384 (F1-CPU-007) | 4.7 MB | +6.18 | 0% |
| 400 | 5.1 MB | +5.76 | 1.5% |
| 432 | 6.0 MB | +4.82 | 1.3% |
| 448 | 6.4 MB | +4.78 | 0% |
| 456 | 6.7 MB | +6.20 | 0% |
| 464 | 6.9 MB | +5.58 | 0% |
| **472** | **7.1 MB** | **-0.37** | **97.1% -- éxito** |
| 480 | 7.4 MB | -2.39 | 0% |
| 512 | 8.4 MB | -2.60 | 0% |
| 576 | 10.6 MB | -2.20 | 0% |
| 640 | 13.1 MB | -2.52 | 0% |
| 720 | 15.8 MB | -2.34 | 0% |
| 768 | 18.9 MB | -2.86 | 0% |
| 800 | 19.5 MB | -2.29 | 0% |
| 896 | 24.5 MB | -2.89 | 0% |
| 1024 | 33.6 MB | -4.23 | 0% |

Verificado en el catálogo DESPLEGADO (el que usa `cpu_final.yaml`) con
REF + F8, turbo desactivado: 6/6 aceptadas, 0 rechazadas. Aviso `CAL-07`
no bloqueante (traza de calibración F8 completa dentro de
grace/tail_grace, ya no bloquea desde ARC-167).

### Interpretación

Igual que `lbm_cpu`, el cruce es abrupto en términos de N (ancho de solo
~24-32 unidades de malla), pero a diferencia de `lbm_cpu` (plateau plano
sin banda cerca del ridge en ningún punto) y de forma aún más marcada
que `hpccg_cpu`, aquí el punto de cruce cae CASI exacto sobre el ridge
(log2=-0.37, 97.1% de las corridas dentro de |log2|<1) -- la mejor
fracción de las cuatro anclas nuevas de la sesión, mejor incluso que
`dual_cholesky_cpu_N1024` (87.6%).

El lado memoria (N≥480) se mantiene relativamente plano alrededor de
log2≈-2.2 a -2.9 (con un descenso adicional en N=1024, -4.23) -- no tan
plano como `lbm_cpu` pero tampoco tan gradual como `hpccg_cpu`; FFT
parece ocupar un punto intermedio entre ambos mecanismos de
reutilización.

Que dos de tres intentos "genuinamente nuevos" de esta sesión
(`hpccg_cpu`, `dual_fft` retomado) hayan tenido éxito, frente a uno
negativo (`lbm_cpu`), confirma que no había manera de predecir el
resultado sin medir -- exactamente el motivo por el que el usuario pidió
no asumir y experimentar en ambos casos.

### Decisión

Se agrega `dual_fft_cpu_N472` a `cpu_final.yaml`. Cuarta ancla nueva de
la sesión (junto a `dual_cholesky_cpu_N1024`, `dual_stencil_cpu_N512`,
`hpccg_cpu_N46`), la de mejor fracción cerca del ridge. Con esto se
cierra el hilo de búsqueda de anclas de esta sesión -- seis anclas reales
en total (`npb_bt`, `npb_lu`, y las cuatro nuevas), acuerdo explícito con
el usuario de parar aquí salvo que él pida continuar.

### Limitaciones

- Solo REF (más la verificación puntual en F8), una repetición corta por
  ronda -- mismo nivel de evidencia que el resto de los smoke tests de
  la sesión; la campaña final confirmará con los 10 niveles completos.
- El ancho del cruce (~24-32 unidades de N) es sensible a la
  granularidad del barrido -- no se puede descartar que exista un N
  intermedio no probado con una fracción aún mejor, pero 97.1% ya está
  cerca del techo práctico (100% no es alcanzable por ruido de sistema
  normal, ver la propia `dual_cholesky_cpu_N1024` en 87.6%).

### Trabajo pendiente

Ninguno para esta entrada -- cerrado. La campaña final confirmará si el
97.1% se sostiene con los 10 niveles de frecuencia completos.

### Criterio exacto de cierre

Cerrado: 16 tamaños de `dual_fft` probados en total entre F1-CPU-007 y
esta entrada (evidencia real de campaña, 48/48 corridas aceptadas en
las tres rondas de esta entrada), resultado positivo identificado
(N=472, 97.1% cerca del ridge), decisión tomada de incluirlo en
`cpu_final.yaml`, verificado en el catálogo desplegado con REF+F8 y
turbo desactivado.

## F1-CPU-011 — Set final de 43 kernels de `cpu_final.yaml`: revisión de redundancia, se deja sin cambios

**Fecha de registro:** 2026-09-14
**Estado:** cerrado. `cpu_final.yaml` se mantiene en 43 kernels / 1290
corridas (sin cambios respecto a F1-CPU-010) -- decisión explícita del
usuario tras revisar candidatos a recorte.

### Motivación

Con las 6 anclas cerca del ridge ya cerradas (F1-CPU-007/009/010), el
usuario preguntó si había redundancia en el set de 43 kernels que
valiera la pena recortar antes de lanzar la campaña final.

### Candidatos identificados

- **`dual_axpy_cpu_N100000/N3162278/N10000000` (3 tamaños).** Evidencia
  directa de F1-CPU-007: el punto "intermedio" (`N1000000`, no incluido
  en la campaña final) dio log2(OI/ridge)=-6.87, casi igual al extremo
  grande `N3162278` (-7.32) -- indicio fuerte de que los tres tamaños
  caen en la misma zona de OI (memory-bound extremo), sin aportar
  diversidad de intensidad operacional entre sí.
- **`phasic_p010/p100/p1000` (3 kernels).** Mismo tamaño (512 MiB),
  varían solo `--phase-seconds` (0.010/0.100/1.000s) -- no son tamaños
  de un algoritmo, sino la misma carga cambiando la velocidad de
  alternancia de fase. Se señaló la duda de si siguen aportando algo al
  clasificador principal o son remanente de la línea de caza de fases
  intra-kernel, cerrada en negativo (ver nota de memoria "NO reabrir").
- Descartados como candidatos: `dual_stencil` (4 tamaños) y
  `dual_cholesky` (3 tamaños) cubren tramos distintos de la curva
  compute→memory, no solo extremos; STREAM `mul/triad/add` son 3 de los
  4 kernels canónicos del benchmark real, con OI distinta entre sí
  aunque todos memory-bound; `dual_spmv` solo tiene 2 tamaños.

### Análisis de costo (por qué no es una decisión de tiempo)

Se calculó el ahorro real de wall-clock/core-horas de recortar los
candidatos, para separar el argumento de "tiempo" del argumento de
"calidad de dataset":

| Recorte | corridas quitadas | wall-clock | core-horas (×32 nucleos reservados) |
|---|---|---|---|
| 1 kernel axpy (11s/corrida) | 30 | ~5.5 min | ~2.9 |
| 3 kernels phasic (24s/corrida c/u) | 90 | ~36 min | ~19 |
| **total** | **120** | **~42 min** | **~22 (~3% de 687 proyectadas)** |

Conclusión: el ahorro de tiempo es insignificante frente al presupuesto
total (~687 core-horas, ~11-17h de wall-clock estimadas) -- cualquier
recorte aquí sería una decisión de **balance/calidad del dataset**
(evitar sobre-representar una zona de OI con puntos casi duplicados en
espacio de features), nunca de velocidad de ejecución.

### Decisión

El usuario decidió NO recortar nada: "dejamos asi". `cpu_final.yaml`
se mantiene en **43 kernels / 1290 corridas** tal como quedó tras
F1-CPU-010. El análisis de redundancia queda documentado por si se
retoma en el futuro (p. ej. si el desbalance de clases resulta ser un
problema real durante el entrenamiento del clasificador, ver
`fase2_clasificador/training/train_phase.py`), pero no bloquea el
lanzamiento de la campaña.

### Set final de 43 kernels (referencia)

**NPB (6):** `npb_bt`, `npb_mg`, `npb_cg`, `npb_sp`, `npb_ft`, `npb_lu`

**RAJAPerf-OpenMP (9):** `cpu_rajaperf_stream_mul`,
`cpu_rajaperf_stream_triad`, `cpu_rajaperf_stream_add`,
`cpu_rajaperf_lcals_first_sum`, `cpu_rajaperf_lcals_tridiag_elim`,
`cpu_rajaperf_polybench_jacobi_1d`, `cpu_rajaperf_polybench_fdtd_2d`,
`cpu_rajaperf_basic_daxpy`, `cpu_rajaperf_basic_init3`

**Standalone reales (10):** `dgemm_n2048`, `rodinia_lavamd_omp`,
`rajaperf_polybench_3mm_omp`, `cpu_lulesh`, `cpu_hpcg`, `cpu_gap_pr`,
`cpu_cholmod`, `phasic_p010`, `phasic_p100`, `phasic_p1000`

**`dual_*` selector CPU/GPU original (16):** `dual_gemm_cpu_N2048`,
`dual_fft_cpu_N128`, `dual_fft_cpu_N1024`, `dual_fft_cpu_N4096`,
`dual_axpy_cpu_N100000`, `dual_axpy_cpu_N3162278`,
`dual_axpy_cpu_N10000000`, `dual_stencil_cpu_N256`,
`dual_stencil_cpu_N512`, `dual_stencil_cpu_N1024`,
`dual_stencil_cpu_N1536`, `dual_cholesky_cpu_N256`,
`dual_cholesky_cpu_N1024`, `dual_cholesky_cpu_N2048`,
`dual_spmv_cpu_N100000`, `dual_spmv_cpu_N1000000`

**Anclas nuevas de esta sesión (2):** `hpccg_cpu_N46`,
`dual_fft_cpu_N472`

**Excluidos del set (documentados, no en la lista):** `cpu_gap_bfs`,
`ptrchase` (F1-CPU-004). **Probados y descartados:** `lbm_cpu`
(F1-CPU-008, 13 tamaños, ninguno entra).

**Anclas cerca del ridge (6 de 43):** `npb_bt`, `npb_lu`,
`dual_cholesky_cpu_N1024` (87.6%), `dual_stencil_cpu_N512` (67.2%),
`hpccg_cpu_N46` (66.3%), `dual_fft_cpu_N472` (97.1%).

### Criterio exacto de cierre

Cerrado: candidatos identificados con evidencia (axpy) o duda explícita
(phasic), costo de recorte calculado y mostrado, decisión tomada por el
usuario de no modificar el manifiesto. `cpu_final.yaml` queda
congelado en 43 kernels / 1290 corridas, listo para lanzar.

## F1-GPU-008 — Cinco kernels de `gpu_final.yaml` sin evidencia de cribado: dos por precisión mixta, tres por FLOPs de punto flotante cero

**Fecha de registro:** 2026-09-14
**Estado:** cerrado. Los 5 se excluyen de `gpu_final.yaml` -- razón
metodológica genuina en todos los casos, no un hueco de cobertura.

### Motivación

Al replicar en GPU el mismo proceso de auditoría de cobertura que se
hizo con CPU esta sesión, se encontró que `rodinia_backprop`,
`rodinia_myocyte`, `gpu_rajaperf_stream_copy`, `gpu_rajaperf_reduce3_int`
y `gpu_rajaperf_indexlist_3loop` -- los 5 listados en `gpu_final.yaml`
-- tienen **cero** carpetas de corrida en el cribado GPU real
(`pacca_screen_20260909_gpu_screen`, 323 corridas). El manifiesto
`gpu_eligible.yaml` (el que realmente se sometió) los excluye desde el
principio; `gpu_candidates.yaml` (la lista completa antes del filtro de
`ncu`) sí los incluye.

### Investigación

El campo `workflow_notes.ncu_gate` de `gpu_eligible.yaml` confirma que
la exclusión viene de la etapa `ncu` (F1-GPU-004), no de un fallo de la
campaña de cribado en sí. Revisando los reportes JSON individuales
(`ncu/<kernel>.json`):

- **`rodinia_backprop`/`rodinia_myocyte`**: `status:
  not_suitable_for_roofline_truth`, `reason: "precisión FP32/FP64
  mixta: no se asigna automáticamente un único ridge Roofline"`.
  Confirmado en los contadores crudos: ambos muestran actividad
  simultánea en `fadd_pred_on`/`fmul_pred_on` (fp32) Y
  `dadd_pred_on`/`dfma_pred_on`/`dmul_pred_on` (fp64) -- no hay un
  ridge único que asignarles bajo el modelo Roofline de una sola
  precisión que usa el proyecto.
- **`gpu_rajaperf_stream_copy`/`reduce3_int`/`indexlist_3loop`**:
  `status: not_suitable_for_roofline_truth`, `reason: "sin FLOPs de
  punto flotante útiles: FLOPs/byte no describe trabajo"`. Son kernels
  puramente enteros/de movimiento de datos (una copia de arreglo, una
  reducción entera, una compactación de índices) -- FLOPs/byte no tiene
  sentido para ellos, sin importar si el wrapper que los ejecuta
  funciona o no.

**Corrección de un supuesto propio equivocado durante la
investigación**: se pensó inicialmente que la causa de los 3
`gpu_rajaperf_*` era exclusivamente el bug de wrapper ya documentado en
F1-GPU-005 (`Base_CUDA-default` vs `Base_CUDA-block_256`), y se
reperfiló con `ncu` tras confirmar que el fix seguía vigente. El
reperfilado de `stream_copy` y `reduce3_int` (exitoso) confirmó el
MISMO resultado ("sin FLOPs") que ya tenían sus reportes previos al
fix -- es decir, el fix de F1-GPU-005 era necesario para los otros 3
`gpu_rajaperf_*` (`stream_triad`/`jacobi_2d`/`heat_3d`, que sí tienen
FLOPs reales) pero nunca iba a cambiar el destino de estos 3
específicos. `indexlist_3loop` no se pudo reperfilar (ver incidente
abajo), pero su reporte previo al fix YA mostraba la misma razón
("sin FLOPs"), consistente con sus dos kernels hermanos.

### Incidente: `ncu` se colgó dos veces reperfilando `gpu_rajaperf_indexlist_3loop`

Dos intentos independientes de reperfilar este kernel quedaron
congelados indefinidamente (0:00 de tiempo de CPU, sin proceso de
cómputo activo en `nvidia-smi`, más de 20 minutos en ambos casos,
superando el timeout interno de 1200s del script -- que no se
propagó correctamente al árbol de procesos que crea
`TreeLauncherSubreaper` de `ncu`). Se terminaron ambos procesos
huérfanos manualmente (con autorización explícita del usuario en cada
caso, dentro del job propio 7144, no de otro usuario de la cuenta
compartida). No se investigó la causa raíz del cuelgue -- podría ser
un problema específico de cómo este binario interactúa con el modo
`--target-processes all` de `ncu`, o un problema de recursos del nodo
compartido en ese momento. Como su reporte previo al fix ya daba la
respuesta correcta (mismo motivo que sus hermanos), no se insistió en
un tercer intento.

### Decisión

Se excluyen los 5 de `gpu_final.yaml`. Documentado como límite de
alcance del modelo (mismo criterio que F1-CPU-004 en CPU), no como
hueco de cobertura a llenar.

### Limitaciones

- No se investigó la causa del cuelgue de `ncu` con
  `gpu_rajaperf_indexlist_3loop` -- si se necesita reperfilar este
  kernel en el futuro por otra razón, puede volver a colgarse.
- La aceptación del resultado de `indexlist_3loop` se apoya en su
  reporte pre-fix (no confirmado con una corrida fresca) más la fuerte
  consistencia con sus dos kernels hermanos, que sí se confirmaron en
  fresco -- evidencia indirecta, no directa.

### Criterio exacto de cierre

Cerrado: 5 kernels investigados con evidencia real (reportes `ncu`
existentes + 2 reperfilados en fresco), razón metodológica confirmada
para cada uno, decisión tomada de excluirlos de `gpu_final.yaml`.

## F1-GPU-009 — Reemplazo del GEMM GPU: ambos representantes existentes medían OI implausible; kernel propio (CUDA, sin cuBLAS) da OI plausible pero cerca del ridge, no compute-bound profundo

**Fecha de registro:** 2026-09-14
**Estado:** cerrado. `gpu_gemm_native_n4096` reemplaza a
`dual_gemm_gpu_N2048` y `gpu_dgemm_n4096` en `gpu_final.yaml`. Séptima
ancla real cerca del ridge de la sesión (sexta si se cuenta solo GPU).

### Motivación

`dual_gemm_gpu_N2048` (F1-GPU-007) ya estaba excluido por OI implausible
(0.0153 FLOP/byte, DFMA=0). Revisando `gpu_dgemm_n4096` con el mismo
criterio se encontró el MISMO síntoma: OI=0.0083 FLOP/byte medido vs.
68.0 declarado en catálogo (desfase ~8175x), DFMA=0 en los tres puntos
de `ncu`. Esto corrige una afirmación de F1-GPU-007, que proponía
`gpu_dgemm_n4096` como reemplazo asumiendo que "ya mide correctamente"
sin haber comparado el OI medido contra el declarado.

### Investigación

Se leyó el código fuente de `dual_gemm_gpu_dispatch.cu`
(`kernels/dual/gemm_gpu_dispatch.cu`, disponible en el repo -- contra
lo que decía F1-GPU-007, que no la encontró en su momento). La llamada
a `cublasDgemm` usa parámetros correctos: M=N=K=n, `alpha=1`,
`beta=0`, matrices A/B con datos aleatorios reales, sin degeneración
de ninguna dimensión. Esto descarta la hipótesis de F1-GPU-007 sobre
una K casi degenerada.

`gpu_dgemm_n4096` (`bin/cublas_dgemm_bench`) enlaza correctamente
contra `libcublas.so.12` (la misma versión que usa el resto del
pipeline, `nvhpc/23.1`/CUDA 12.0) -- descarta también la hipótesis de
divergencia de entorno/versión de CUDA que quedó abierta para
`dual_gemm_gpu`.

Con el código fuente confirmado correcto en ambos casos, la conclusión
es que el kernel interno y propietario que `cuBLAS` despacha para
DGEMM en esta combinación de GPU/driver/cuBLAS no emite las
instrucciones SASS que el método de conteo de FLOPs de `ncu`
reconoce (`sm__sass_thread_inst_executed_op_dfma_pred_on.sum` y
similares) -- es una brecha de instrumentación, no un bug de código.
Los demás kernels `dual_*_gpu` (stencil, cholesky, spmv, axpy, fft) sí
miden OI plausible con `ncu`, y todos son CUDA escrito a mano, ninguno
usa cuBLAS -- patrón consistente con esta hipótesis.

### Solución: `gemm_native_gpu`

Kernel CUDA propio (`kernels/gemm_native/gemm_native_gpu_bench.cu`):
GEMM tiled con memoria compartida (bloque 16x16), sin cuBLAS,
verificación por muestreo contra el cálculo esperado en CPU. Compilado
y verificado en paccaA100 (`Verification SUCCESSFUL`).

Primera medición a N=1024: **OI=105.14 FLOP/byte** (`ncu`, convergió,
cambio relativo 0.0001) -- muy por encima del ridge fp64 (~3.4-3.7),
con DFMA real y masivo (a diferencia de los dos binarios basados en
cuBLAS). Pero el usuario señaló que N=1024 es un tamaño trivial para
la GPU y no sostiene duración real -- recordando F1-GPU-006 (kernels
GPU necesitan ~30-35s de duración real). Medido en paccaA100: N=1024
da 0.0567s/iter (~10s con 176 iter, insuficiente); N=4096 da 0.0975s/iter
(iterations=330 para ~32.2s, dentro del objetivo).

**Sorpresa real al reperfilar a N=4096**: OI cae a **4.05 FLOP/byte**
-- casi exacto sobre el ridge, NO compute-bound profundo como a N=1024.
Explicación física: el kernel usa un tile de memoria compartida chico
(TILE=16) sin bloqueo por registros, con poca reutilización de datos
entre bloques. A N=1024 la huella completa (3 buffers x 1024² x 8 =
25MB) cabe entera en la L2 del A100 (40MB) -- el tráfico real a DRAM es
mínimo y el OI medido queda artificialmente alto (mismo artefacto de
cache-residencia que ya se documentó repetidamente en el lado CPU esta
sesión, F1-CPU-008/009/010). A N=4096 (huella de 402MB, no cabe en L2)
el tráfico real a DRAM revela la intensidad operacional verdadera de
este bloqueo simple, que resulta ser baja.

### Decisión

Se acepta `gpu_gemm_native_n4096` con `phase_label_hint: intermedio`
(no `compute_bound`) como séptima ancla real cerca del ridge de la
sesión, en vez de invertir en bloqueo por registros (cada hilo computa
varios elementos de C, técnica estándar para subir la intensidad
aritmética) para recuperar un GEMM genuinamente compute-bound --
decisión explícita del usuario. Reemplaza a `dual_gemm_gpu_N2048` y
`gpu_dgemm_n4096` en `gpu_final.yaml`.

**Consecuencia que queda documentada**: el catálogo GPU se queda SIN
ningún representante de GEMM denso profundamente compute-bound. No es
un hueco a llenar de inmediato -- fue una decisión consciente de
alcance, no un olvido.

### Limitaciones

- No se investigó si un `TILE` más grande (32/64) o bloqueo por
  registros movería el OI de forma significativa -- se optó por
  aceptar el resultado en vez de iterar sobre el diseño del kernel.
- La causa exacta de por qué `ncu` no captura las instrucciones de
  `cuBLAS` (¿instrucciones tensor-core no marcadas como tal?, ¿un
  opcode SASS distinto en esta versión de driver/cuBLAS?) sigue sin
  confirmarse -- se descartaron las hipótesis de código y de entorno,
  pero no se identificó la causa positiva.
- Solo se verificó con `ncu` (3 puntos de convergencia), no con una
  corrida real de campaña con los 9 niveles de frecuencia de GPU.

### Trabajo pendiente

Ninguno para esta entrada -- cerrado. Si en el futuro se quiere un GEMM
denso compute-bound real para el catálogo GPU, el punto de partida
sería agregar bloqueo por registros a `gemm_native_gpu` (mismo binario,
nuevo `TILE`/blocking) en vez de escribir uno nuevo desde cero.

### Criterio exacto de cierre

Cerrado: causa raíz de ambos GEMM anteriores investigada con evidencia
real (código fuente revisado, versión de biblioteca confirmada),
kernel propio escrito, compilado, verificado numéricamente y medido
con `ncu` en dos tamaños, resultado final (near-ridge, no
compute-bound) aceptado explícitamente por el usuario.

## F1-GPU-010 — `gpu_gemm_native_n4096` (código propio) sustituido por `gpu_rajaperf_gemm` (suite RAJAPerf): mismo problema resuelto sin escribir CUDA a mano

**Fecha de registro:** 2026-09-14
**Estado:** cerrado. `gpu_rajaperf_gemm` reemplaza a `gpu_gemm_native_n4096`
en `gpu_final.yaml`. La campaña final vuelve a estar compuesta
enteramente por kernels de suites de terceros, sin excepciones de
código propio.

### Motivación

El usuario planteó que un kernel GEMM escrito a mano (la solución de
F1-GPU-009) implica un riesgo experimental y de cuestionamiento por
parte de evaluadores externos que un kernel de una suite reconocida no
tiene, y pidió resolver el problema original (GEMM de suite con OI
implausible vía `ncu`) sin código propio salvo que fuera estrictamente
necesario.

### Investigación

El punto de partida de F1-GPU-009 seguía siendo válido: el problema de
`dual_gemm_gpu_N2048`/`gpu_dgemm_n4096` no es un bug de código, sino que
`ncu` no puede contar instrucciones SASS dentro del kernel propietario
que despacha cuBLAS. Cualquier GEMM de suite que sea un envoltorio de
cuBLAS/cuDNN/CUTLASS tendría el mismo problema. La pregunta correcta
era entonces: ¿alguna suite ya integrada en el proyecto implementa GEMM
con su propio código CUDA (no una biblioteca BLAS de proveedor)?

Rodinia no incluye GEMM denso (su análogo más cercano es `rodinia_lud`,
descomposición LU, ya usado). RAJAPerf (`raja-perf-cuda-v2025.12.1`,
ya integrado para `gpu_rajaperf_stream_triad`/`jacobi_2d`/`heat_3d`)
incluye el grupo Polybench, que sí tiene `Polybench_GEMM`: se confirmó
con `--print-kernels` que el binario CUDA ya compilado en pacca lo
tiene disponible, y con `-v Base_CUDA --dryrun` que soporta la variante
CUDA. RAJA genera el kernel CUDA real a partir de su propio código C++
(lambda → kernel CUDA), no llama a cuBLAS -- por diseño, sus
instrucciones SASS sí son visibles para `ncu`.

Se midió tiempo de pared real (sin instrumentar) para elegir un tamaño
que cumpliera el piso de ~30-35s de F1-GPU-006: `--sizefact 200` → 25.2s,
`--sizefact 250` → 34.3s, `--sizefact 400` → 63.6s. Se eligió 250. El
wrapper reproducible `bin/gpu_rajaperf_gemm` (mismo patrón que los demás
`gpu_rajaperf_*`, checksum-verificado) invoca únicamente
`Polybench_GEMM`/`Base_CUDA` con ese tamaño y valida el checksum nativo
de RAJAPerf (`RAJAPerf-checksum.txt`, `Base_CUDA-block_256 PASSED`).

Perfilado con `ncu` (mismo pipeline `run_screening_to_report.sh --stage
ncu --ncu-kernel gpu_rajaperf_gemm`, tres puntos de convergencia
lc5/lc20/lc50): convergió (cambio relativo 0.0005), con instrucciones
DFMA/DMUL reales y distintas de cero (37.9 billones de DFMA sumados),
a diferencia de los dos GEMM cuBLAS. OI final medido: **2.81 FLOP/byte,
fp64**, `roofline_label_eligible: true`. Contra el ridge fp64 (~3.4-3.7),
queda cerca del ridge del lado memory-bound -- simétrico a
`gemm_native_gpu` (OI=4.05, cerca del ridge del lado compute-bound).

Nota operativa: el perfilado con `ncu` de este kernel tardó
considerablemente más que el de kernels previos (~35 min totales,
contra <10 min para kernels ligeros como `rodinia_dwt2d_s2048`) porque
la carga base sin instrumentar ya es ~100x más pesada (34s vs <1s);
no fue un colgamiento (a diferencia de los incidentes de F1-GPU-008),
el proceso mantuvo uso de CPU activo (~98%) durante todo el perfilado.

### Decisión

Se reemplazó `gpu_gemm_native_n4096` por `gpu_rajaperf_gemm` en
`gpu_final.yaml` (decisión del usuario, 2026-09-14, entre mantener solo
el de suite o dejar ambos como anclas simétricas cerca del ridge -- se
optó por la opción de menor riesgo experimental). La campaña queda de
nuevo en 14 kernels/420 corridas (reemplazo 1 a 1, sin cambio de
conteo). `gemm_native_gpu` se deja documentado en el catálogo
(`fase1_telemetria/catalog/catalog.yaml`, entrada `gpu_gemm_native_n4096`)
y su código fuente (`kernels/gemm_native/gemm_native_gpu_bench.cu`) se
mantiene en el repo, pero fuera de la campaña final.

### Limitaciones

- Igual que `gemm_native_gpu`, este resultado es near-ridge, no
  profundamente compute-bound -- no resuelve la brecha de cobertura en
  el extremo compute-bound profundo identificada antes de esta entrada
  (`rodinia_lavamd` sigue siendo la única ancla realmente profunda).
- No se investigó si otras variantes de RAJAPerf (`Polybench_2MM`,
  `Polybench_3MM`) o un `--sizefact` distinto cambiarían el OI medido de
  forma significativa.
- Solo se verificó con `ncu` (tres puntos de convergencia), no con una
  corrida real de campaña con los 9 niveles de frecuencia de GPU.

### Trabajo pendiente

Ninguno para esta entrada -- cerrado.

### Criterio exacto de cierre

Cerrado: se identificó y verificó (vía `--print-kernels`/`--dryrun`/
ejecución real) un kernel GEMM de una suite de terceros ya integrada en
el proyecto que resuelve el mismo problema de instrumentación que
motivó `gemm_native_gpu`, sin escribir código CUDA nuevo; wrapper
reproducible desplegado y checksum-verificado; medido con `ncu` hasta
convergencia; `gpu_final.yaml` y `catalog.yaml` actualizados;
decisión de reemplazo (vs. mantener ambos) tomada explícitamente por
el usuario.

## F1-GPU-011 — Barrido de memoria real en GPU: tamaños nuevos a ~50% de VRAM del A100 para las 5 familias `dual_*`, 3 entran a la campaña final

**Fecha de registro:** 2026-09-14
**Estado:** cerrado. `gpu_final.yaml` pasa de 14 a **17 kernels / 510
corridas**. `dual_axpy_gpu_N1280000000`, `dual_spmv_gpu_N200000000` y
`dual_stencil_gpu_N36864` se agregan JUNTO A (no en reemplazo de) sus
variantes pequeñas ya usadas. `dual_fft_gpu_N25600` y
`dual_cholesky_gpu_N32768` quedan documentados pero fuera de la
campaña.

### Motivación

El director de tesis del usuario pidió explícitamente tener muestras
que ocupen una porción significativa de la memoria de la GPU (A100,
40GB de VRAM) en los casos donde aplique, mismo principio que ya se
usó en CPU para poblar zonas cercanas al ridge con un barrido de
tamaños (F1-CPU-010, `dual_fft_cpu_N472`).

### Investigación

Se encontró que, salvo `gemm_native_gpu`, `rodinia_dwt2d` y
parcialmente `rodinia_lavamd` (ya investigados), **ninguna de las 5
familias `dual_*` de GPU tenía evidencia real de que su OI cambiara
con el tamaño**: el catálogo declara un único OI "representativo" fijo
por familia y lo copia a todas las variantes de N, incluyendo tamaños
grandes nunca perfilados con `ncu` ("heredados"). Todos los kernels
`dual_*` usados en la campaña ocupaban menos del 2% de la VRAM del
A100 antes de esta entrada (excepción: `rodinia_dwt2d`, 30%).

Se derivó la ley de escalamiento real de memoria de cada familia a
partir de los datos ya existentes en catálogo (`estimated_memory_bytes`
vs. N), y se calcularon tamaños candidato apuntando a ~50% de VRAM.
Las estimaciones por fórmula resultaron sistemáticamente por encima de
lo real (ej. FFT: se estimó 21.1GB a N=22528, la medición real con
`nvidia-smi` dio 15.92GB) -- se verificó cada tamaño con ejecución real
(`Verification = SUCCESSFUL`) y memoria medida en vivo (`nvidia-smi
--query-gpu=memory.used` en un loop durante la corrida), no solo por
fórmula, antes de comprometer tiempo de `ncu`:

| Familia | N final | Memoria real | % VRAM | Tiempo real sin instrumentar |
|---|---|---|---|---|
| `dual_fft_gpu` | 25600 | 20.43GB | 49.9% | 28.1s |
| `dual_cholesky_gpu` | 32768 | 8.64GB | 21.1% | 57.1s |
| `dual_axpy_gpu` | 1280000000 | 19.5GB | 48.8% | 48.4s |
| `dual_spmv_gpu` | 200000000 | 19.8GB | 49.5% | 52.3s |
| `dual_stencil_gpu` | 36864 | 20.66GB | 51.7% | 57.8s |

Nota sobre Cholesky: se probó primero N=49152 (18.9GB, 46% VRAM), pero
tardó 3 minutos solo para 3 iteraciones sin instrumentar -- proyectando
el mismo overhead de `ncu` visto en `gpu_rajaperf_gemm` (~60x), el
perfilado habría tomado 3-5+ horas en el nodo compartido. El usuario
eligió explícitamente N=32768 (21% VRAM, perfilado ~1h) sobre N=49152
(46% VRAM, perfilado de horas) para no comprometer el nodo compartido
tanto tiempo.

Se perfilaron los 5 con `ncu` (mismo pipeline, tres puntos de
convergencia lc5/lc20/lc50). Dos intentos fallaron antes del exitoso:
(1) el catálogo requiere un campo `operational_intensity_flops_per_byte`
placeholder incluso para kernels nuevos sin medir aún (validación
CAT-10), se había omitido; (2) el primer parche para agregar ese
placeholder corrompió el YAML por un escape mal anidado en un comando
remoto (indentación rota, `binary_checksum:` perdió su nivel) --
se resolvió restaurando desde backup y reconstruyendo el fragmento
limpio localmente antes de subirlo, en vez de parchear in-place en el
cluster.

**Resultados reales:**

| Kernel | OI real | OI anterior | Estado formal | Interpretación |
|---|---|---|---|---|
| `dual_fft_gpu_N25600` | 1.71 (estable) | 2.499 | Excluido: precisión mixta | Hallazgo nuevo, no visto en N4096/N16384 |
| `dual_cholesky_gpu_N32768` | 0.217 (subiendo aún) | 9.887 (heredado) | No converge (1.55%>1%) | Cache-residencia confirmada, caída ~45x |
| `dual_axpy_gpu_N1280000000` | 0.0858 | 0.125 | Converged, elegible | Memory-bound, esperado |
| `dual_spmv_gpu_N200000000` | 0.266 (estable) | 0.274 | No converge (técnico, launches insuficientes) | Casi idéntico, consistente |
| `dual_stencil_gpu_N36864` | 0.2577 (estable) | 0.495 | No converge (técnico, launches insuficientes) | Cache-residencia confirmada, caída a la mitad |

Ninguno resultó ser una ancla nueva cerca del ridge fp64 (~3.4-3.7
FLOP/byte) -- todos se confirman o se alejan más hacia memory-bound a
esta escala. Este es el resultado real, no un fallo del barrido: a la
escala de memoria que realmente usa una GPU grande, el catálogo es
consistentemente memory-bound, que es el dato correcto para el
clasificador (no todo barrido de tamaño tiene que revelar una ancla
nueva para ser útil).

**Hallazgo colateral -- `dual_fft_gpu_N25600`, precisión mixta:** a
N=16384 esta familia es fp64 puro; a N=25600 `ncu` detecta
instrucciones fp32 Y fp64 mezcladas, algo no visto en ningún tamaño
anterior. No afecta a `dual_fft_gpu_N4096` (ya en la campaña, fp64
puro). Causa raíz no investigada (posible cambio de estrategia interna
del kernel a partir de cierto tamaño, ej. tabla de twiddle factors en
fp32) -- queda como hallazgo documentado, no resuelto.

### Decisión

Se agregan `dual_axpy_gpu_N1280000000`, `dual_spmv_gpu_N200000000` y
`dual_stencil_gpu_N36864` a `gpu_final.yaml` JUNTO A sus variantes
pequeñas (no las reemplazan): aunque ambas escalas caen en la misma
región memory-bound del espacio Roofline, dan puntos reales a
magnitud de memoria realista de GPU, cumpliendo el pedido del
director sin sacrificar la cobertura a escala pequeña ya validada.
`dual_cholesky_gpu_N32768` y `dual_fft_gpu_N25600` quedan documentados
en el catálogo pero fuera de la campaña (no convergen / precisión
mixta sin investigar). La campaña final queda en **17 kernels, 510
corridas** (antes: 14/420).

### Limitaciones

- `dual_cholesky_gpu_N32768` y las 2 entradas "not_converged por
  launches insuficientes" (`dual_spmv_gpu_N200000000`,
  `dual_stencil_gpu_N36864`) no cumplen el criterio formal de
  convergencia del pipeline, aunque su OI es visualmente muy estable
  entre los tres puntos medidos -- se aceptan con esta nota
  metodológica, mismo criterio ya usado en `dual_fft_gpu_N16384`
  (F1-GPU-011 previo).
- No se investigó la causa raíz de la precisión mixta en
  `dual_fft_gpu_N25600`.
- No se investigó si un N intermedio entre 32768 y 49152 permitiría a
  Cholesky converger con un tiempo de perfilado más razonable.
- Solo se verificó con `ncu`, no con una corrida real de campaña con
  los 9 niveles de frecuencia de GPU.

### Trabajo pendiente

Ninguno para esta entrada -- cerrado. Si se quiere investigar la causa
de la precisión mixta en FFT o intentar converger Cholesky a mayor
escala, sería una entrada nueva.

### Criterio exacto de cierre

Cerrado: ley de escalamiento de memoria derivada de datos reales del
catálogo, 5 tamaños candidato verificados con ejecución real y memoria
medida en vivo (no solo estimada), perfilados con `ncu` hasta
convergencia o hasta obtener una medición estable con nota
metodológica, 3 de 5 incorporados a `gpu_final.yaml` con decisión
explícita del usuario, 2 documentados como excluidos con motivo
verificable; `gpu_final.yaml` y `catalog.yaml` actualizados.

## F1-GPU-012 — Corrige F1-GPU-009: el DGEMM cuBLAS usa Tensor Cores (no una "brecha de instrumentación"), y el parser de `ncu_convergence.py` nunca detectaba Tensor Core en corridas reales

**Fecha de registro:** 2026-09-14
**Estado:** cerrado el bug del parser (código corregido, prueba de
regresión agregada). Abierto el candidato de reemplazo GEMM
(`CUBLAS_PEDANTIC_MATH`/CUTLASS SIMT) -- ver Trabajo pendiente.

### Motivación

El usuario adjuntó una revisión externa (`Nota_Candidatos_GPU_Compute_Bound_20260914.md`,
generada por otra sesión de Claude, sin ejecutar nada en pacca) que
cuestionaba la conclusión de F1-GPU-009 ("brecha de instrumentación de
`ncu` con binarios cuBLAS") citando un comentario `ARC-76` ya existente
en `catalog.yaml` (líneas 400-409) que documentaba, ANTES de esta
sesión, que `cublas_dgemm_bench` enruta a un kernel Tensor Core
(`cutlass_80_tensorop_d884gemm`) sin que se le pida, y que el pipeline
excluye Tensor Cores del conteo de FLOPs por contrato explícito
(`fase1_telemetria/ncu_convergence.py::build_kernel_report`, guardrail
en `tensor_instructions_observed > 0`).

### Investigación

Se verificó la afirmación contra el CSV crudo real de `ncu` ya generado
para `gpu_dgemm_n4096` durante F1-GPU-009
(`pacca_screen_20260909/ncu/gpu_dgemm_n4096__lc5.csv`): la columna
"Kernel Name" confirma literalmente `cutlass_80_tensorop_d884gemm_64x64_16x4_nn_align1`
-- el mismo kernel Tensor Core de ARC-76. **La conclusión de F1-GPU-009
está corregida**: no es una brecha de instrumentación de `ncu` con
binarios cuBLAS en general, es que ESTE binario/config específico
(`cublas_dgemm_bench --size 4096`) enruta a Tensor Cores, y el pipeline
correctamente excluye esa ruta del conteo escalar de FLOPs -- por
diseño, no por incapacidad de `ncu`.

Al verificar por qué `tensor_instructions_observed` había salido 0.0 en
el reporte (en vez de disparar el guardrail y marcar el kernel como
`not_suitable_for_roofline_truth: Tensor Core`), se encontró un bug real
en `ncu_convergence.py`: la función `parse_ncu_csv` tiene dos rutas de
parseo, una para el formato "largo" (una fila por métrica, usada en los
fixtures de prueba) que sí detecta columnas `pipe_tensor` correctamente
vía `_metric_bucket`, y una para el formato "ancho" (una fila por
lanzamiento, todas las métricas como columnas) -- que es el formato que
`ncu --page raw` REALMENTE produce en toda corrida real de este
proyecto (confirmado contra el CSV crudo). En la rama de formato ancho,
la lista de columnas a sumar (`fp32_cols`, `fp64_cols`, `int_cols`,
`fma_cols`, `dram_cols`) **nunca incluía `tensor_cols`**, y el bucle de
acumulación no iteraba sobre "tensor" -- el campo `tensor_inst` se
inicializaba en 0.0 y nunca se tocaba, sin importar lo que el CSV
crudo realmente contuviera.

**Impacto:** el guardrail de exclusión de Tensor Core nunca se disparó
para NINGUNA corrida real de `ncu` en todo el proyecto hasta ahora (el
formato ancho es el único que se usa en producción). Cualquier kernel
que secretamente enrutara por Tensor Cores habría sido medido con
FLOPs≈0 (near-zero OI, "converged") en vez de excluido correctamente
como `not_suitable_for_roofline_truth`. Para GEMM esto llevó a la
conclusión equivocada de F1-GPU-009 ("brecha de instrumentación") en
vez de la correcta ("Tensor Core, excluido por diseño") -- pero la
conclusión práctica de F1-GPU-009 y F1-GPU-010 (reemplazar el GEMM
cuBLAS en la campaña, GEMM propio o de RAJAPerf) sigue siendo correcta,
solo el diagnóstico estaba mal.

### Corrección aplicada

`fase1_telemetria/ncu_convergence.py`, rama de formato ancho de
`parse_ncu_csv`: se agregó `tensor_cols = col_idx(lambda h: "pipe_tensor"
in h or "tensor_op_" in h)` y se agregó `("tensor", tensor_cols,
"tensor_inst")` a la tupla que itera el bucle de acumulación --
mismo criterio de detección que ya usaba la rama de formato largo
(`_metric_bucket`), ahora consistente entre ambas rutas. Prueba de
regresión agregada (`test_formato_ancho_real_detecta_tensor_cores`)
con un CSV sintético que imita el formato ancho real (columnas "ID"/
"Kernel Name" + `sm__inst_executed_pipe_tensor.sum`), verifica que
`tensor_inst` se acumula correctamente y que `build_kernel_report`
dispara el guardrail con esa evidencia.

Se descubrió, al correr la suite completa, que 6 pruebas de
`test_ncu_convergence.py` ya fallaban ANTES de este cambio (confirmado
con `git stash`) -- no relacionado con este fix, no se investigó ni se
corrigió (fuera de alcance de esta entrada).

### Sobre el segundo punto de la nota externa (2MM/3MM)

La nota también cuestiona la explicación causal que di para la caída
de OI en `gpu_rajaperf_2mm`/`gpu_rajaperf_3mm` (F1-GPU-011 previo,
"round-trip por DRAM entre etapas") -- señala correctamente que no se
verificó desglosando FLOPs/bytes por nombre de kernel individual dentro
del perfil agregado de 2-3 lanzamientos. Esto queda como una
afirmación no verificada, no retractada ni confirmada -- el resultado
medido (OI 2.81→2.50→1.95) sigue siendo válido, solo la explicación
mecanicista de por qué es especulación sin verificar.

### Reprocesamiento y auditoría (segunda parte, mismo día)

Con el parser corregido, se reprocesaron desde sus CSV ya existentes
(`--from-csv`, sin gastar `ncu` de nuevo) `gpu_dgemm_n4096` y
`dual_gemm_gpu_N2048`: ambos ahora se marcan correctamente
`not_suitable_for_roofline_truth` por Tensor Core. Confirma la
corrección.

Se auditaron los 37 kernels GPU con reporte `ncu` ya generado: **17
mostraron `tensor_instructions_observed > 0`** al reprocesar. Un umbral
binario `> 0` (igual al del guardrail original) habría excluido de
golpe a `rodinia_lud`, `rodinia_gaussian`, `rodinia_dwt2d`,
`gpu_rajaperf_heat_3d`, `dual_stencil_gpu_*` y los tres `gpu_phasic_*`
-- kernels escritos a mano o sintéticos, sin ninguna dependencia
cuBLAS/cuSOLVER de por medio, donde un uso real de Tensor Core es
implausible. Se calculó la fracción `tensor / (tensor + escalar)`
(sumando ambos across los 3 puntos lc5/lc20/lc50, mismo criterio en
numerador y denominador) para los 17: apareció un hueco limpio en la
distribución real entre ~3.3% (máximo del grupo "ruido de medición",
`dual_cholesky_gpu_N2048`) y ~20.8% (mínimo del grupo "Tensor Core
real", `dual_cholesky_gpu_N16384`). Se corrigió el guardrail de
`> 0` absoluto a `tensor_fraction > TENSOR_MIN_FRACTION` (constante
nueva, 0.10, decisión del usuario, cae a mitad del hueco observado).

**Autocorrección importante:** el primer cálculo de esta fracción (antes
de implementar el fix en el pipeline) comparó el tensor sumado en los 3
puntos contra el escalar de un solo punto (denominadores no
comparables), lo que infló `dual_cholesky_gpu_N2048` a un 20.0%
aparente -- con esa cifra equivocada se le pidió al usuario decidir si
sacarlo de `gpu_final.yaml`, y el usuario aceptó sacarlo. Al recalcular
con el mismo criterio de suma en ambos lados (el que efectivamente
quedó implementado en `build_kernel_report`/`main()`), la fracción real
de `dual_cholesky_gpu_N2048` es **3.3%**, bajo el umbral -- NO se
excluye, NO se cambia `gpu_final.yaml`. La decisión del usuario de
sacarlo se revirtió antes de tocar el archivo (el error se detectó
antes de aplicar el cambio, así que no hubo que deshacer nada en
`gpu_final.yaml`). Sí es real que Cholesky se contamina con Tensor
Core a partir de tamaños grandes (`N16384`: 20.8%, `N32768`: 34.2%),
pero esos tamaños ya estaban fuera de la campaña por no converger
(F1-GPU-011) -- la contaminación tensor es una segunda razón
independiente, no cambia esa decisión.

### Corrección aplicada (parte 2 -- umbral de fracción)

`fase1_telemetria/ncu_convergence.py`: nueva constante
`TENSOR_MIN_FRACTION = 0.10`. `build_kernel_report()` gana un parámetro
`total_scalar_instructions`, calcula `tensor_fraction =
tensor_instructions_observed / (tensor_instructions_observed +
total_scalar_instructions)` y solo excluye si supera el umbral (antes:
cualquier valor `> 0`). `main()` calcula `total_scalar_instructions`
sumando las mismas claves `_FP32_KEYS`/`_FP64_KEYS`/`_INT_KEYS` across
todos los puntos, con el mismo criterio de suma que ya usaba para
`tensor_instructions_observed`. Dos pruebas de regresión nuevas:
`test_tensor_bajo_umbral_no_excluye_ruido_de_medicion` (4% no excluye,
caso tipo `rodinia_gaussian`) y `test_tensor_sobre_umbral_excluye_como_cholesky_real`
(20% sí excluye, caso tipo `dual_cholesky_gpu_N16384`).

### Limitaciones

- El umbral 10% es una decisión razonada sobre el hueco observado en
  ESTOS 17 kernels, no una constante física -- si aparecen más
  kernels con fracciones intermedias (entre 4% y 20%) en el futuro,
  habría que revisar si el hueco sigue siendo limpio.
- No se investigó `CUBLAS_PEDANTIC_MATH` ni CUTLASS SIMT (candidatos de
  la nota externa para recuperar un GEMM optimizado sin Tensor Cores).
- No se investigó la causa raíz de la caída de OI en 2MM/3MM por
  kernel individual.
- No se reprocesaron con el fix los 20 kernels restantes (de 37) que
  no mostraron `tensor_instructions_observed > 0` en la auditoría --
  se asume que su resultado no cambia (no tenían tensor que
  reclasificar), pero no se verificó cada uno explícitamente.

### Trabajo pendiente

1. Evaluar `CUBLAS_PEDANTIC_MATH`/CUTLASS SIMT como vía para un GEMM
   optimizado real sin Tensor Cores (prioridad 1 de la nota externa).
2. Si se agregan kernels nuevos al catálogo GPU en el futuro, incluir
   `tensor_instructions_observed`/`tensor_fraction` en la revisión
   estándar, no solo OI y convergencia.

### Criterio exacto de cierre

Cerrado: dos bugs reales del parser identificados y corregidos con
precisión de línea de código (falta de `tensor_cols` en el formato
ancho; umbral binario `>0` sin piso de ruido), ambos confirmados contra
CSV crudo real y contra una auditoría de 37 kernels ya perfilados, no
solo teoría. Pruebas de regresión agregadas y pasando (5 nuevas en
total). Un error propio de cálculo (denominadores no comparables) se
detectó y corrigió ANTES de aplicarlo a `gpu_final.yaml` -- documentado
explícitamente en vez de omitido. El candidato de reemplazo GEMM (parte
2 de la nota externa, `CUBLAS_PEDANTIC_MATH`/CUTLASS) queda abierto, no
cerrado en esta entrada.

## F1-GPU-013 — Segunda ancla compute-bound de suite/proveedor: `CUBLAS_PEDANTIC_MATH` falla, DGEMM via CUTLASS `OpClassSimt` funciona (OI=26.73 FLOP/byte)

**Fecha de registro:** 2026-09-14
**Estado:** cerrado. `gpu_cutlass_simt_dgemm_n4096` se agrega a
`gpu_final.yaml`. Campaña final queda en **18 kernels / 540 corridas**.

### Motivación

Continuación directa de F1-GPU-012: la revisión externa
(`Nota_Candidatos_GPU_Compute_Bound_20260914.md`) proponía, en orden de
prioridad, (1) `CUBLAS_PEDANTIC_MATH` o CUTLASS SIMT para recuperar un
GEMM optimizado por el proveedor sin Tensor Core, y (2) `Apps_LTIMES`
de RAJAPerf como alternativa. El usuario pidió intentarlo pese a que ya
habían fallado tres búsquedas anteriores de un segundo compute-bound
profundo (GEMM propio near-ridge, 2MM/3MM peor que GEMM simple).

### Investigación

**Intento 1, `CUBLAS_PEDANTIC_MATH` -- falló.** Se creó
`kernels/gpu/cublas_dgemm_pedantic_bench.cu`, copia exacta de
`old/kernels/gpu/cublas_dgemm_bench.cu` (el binario detrás de
`gpu_dgemm_n4096`/`gpu_dgemm_calibration`) con
`cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)` agregado antes de
`cublasDgemm`. Compilado, verificado numéricamente (`Verification =
SUCCESSFUL`), pero perfilado con `ncu` (3 lanzamientos, sin ciclo
completo de convergencia -- no hacía falta) mostró el MISMO kernel:
`void Kernel2<cutlass_80_tensorop_d884gemm_64x64_16x4_nn_align1>(Params)`,
con `sm__inst_executed_pipe_tensor.sum` no-cero por lanzamiento. La
documentación de NVIDIA (CUDA 12.0) dice que pedantic desactiva Tensor
Core; en esta combinación específica de driver/cuBLAS/CUDA (CUDA 13.1
detectado en el nodo) NO lo hizo -- resultado real y verificado, no
teórico.

**Intento 2, CUTLASS `OpClassSimt` -- funcionó.** Se encontró CUTLASS
v2.11.0 ya clonado en pacca (`~/cutlass`, sin compilar). En vez de
compilar el `cutlass_profiler` completo (pesado, requiere CMake no
disponible en PATH), se escribió un programa standalone
(`kernels/gpu/cutlass_simt_dgemm_bench.cu`) que instancia
`cutlass::gemm::device::Gemm<double, ..., cutlass::arch::OpClassSimt,
cutlass::arch::Sm50, ...>` con el tile `128x32x8`/`32x16x8` -- copiado
EXACTO de `cutlass/test/unit/gemm/device/simt_dgemm_nn_sm50.cu` (una
configuración ya validada por los tests propios de CUTLASS, no una
elección sin precedente). Compilado directo con `nvcc -I
cutlass/include` (sin CMake), verificado numéricamente.

Confirmación con `ncu`: el nombre del kernel SASS contiene
`MmaSimt` (no `MmaTensorOp`), y una medición rápida dio
`sm__inst_executed_pipe_tensor.sum = 0` exacto junto con
`sm__sass_thread_inst_executed_op_dfma_pred_on.sum = 68,719,476,736`
-- coincide EXACTO con N³ para N=4096 (4096³ = 68,719,476,736), la
cuenta de FLOPs esperada para una GEMM completa. Perfilado con el
ciclo completo de convergencia (lc5/lc20/lc50): **OI = 26.73 FLOP/byte,
fp64, `converged: true` (cambio relativo 0.0001), extremadamente
estable (26.721 → 26.724 → 26.727), `roofline_label_eligible: true`,
`tensor_instructions_observed: 0.0`**.

Contra el ridge fp64 (~3.4-3.7 FLOP/byte), esto da
log2(OI/ridge) ≈ **+2.9** -- comparable a `rodinia_heartwall` (+2.06),
la segunda ancla compute-bound de suite/proveedor real de la sesión
(`rodinia_lavamd`, +7.52, sigue siendo la más profunda).

Tamaño elegido por tiempo de pared real (F1-GPU-006, piso ~30-35s):
N=4096 con `--iterations 1650` da 31.1s medido. La huella (3 matrices
4096²×8B ≈ 402MB) no cabe en L2 (40MB del A100), así que no hay riesgo
de artefacto de cache-residencia como el visto en `gemm_native_gpu`.

### Decisión

Se agrega `gpu_cutlass_simt_dgemm_n4096` a `gpu_final.yaml` **junto a**
`gpu_rajaperf_gemm` (no lo reemplaza): tienen roles distintos --
`gpu_rajaperf_gemm` cubre la zona cerca del ridge, este cubre
compute-bound genuino. Campaña final: **18 kernels / 540 corridas**
(antes: 17/510).

### Limitaciones

- Solo se probó un tile SIMT (`128x32x8`/`32x16x8`, el de la referencia
  de CUTLASS); no se exploró si otro tile daría un OI aún más alto o
  más bajo -- no hacía falta para el objetivo (encontrar UNA ancla
  compute-bound adicional), pero significa que 26.73 no es
  necesariamente el máximo alcanzable con CUTLASS SIMT.
- No se investigó por qué `CUBLAS_PEDANTIC_MATH` no tuvo el efecto
  documentado en este entorno (¿versión de cuBLAS?, ¿comportamiento
  específico de FP64 DGEMM?) -- se aceptó el resultado negativo y se
  pasó al siguiente candidato en vez de depurar la causa.
- CUTLASS v2.11.0 (clonado en pacca, enero 2023) es una versión vieja
  frente al CUDA 13.1 detectado en el nodo -- compiló con warnings de
  deprecación (`long4`/`double4` etc.), no errores; funciona, pero no
  se investigó si una versión más nueva de CUTLASS cambiaría el
  resultado.
- No se probó `Apps_LTIMES` (prioridad 2 de la nota externa) ni
  miniBUDE (prioridad 3) -- innecesario, el candidato de prioridad 1
  ya dio resultado positivo.

### Trabajo pendiente

Ninguno para esta entrada -- cerrado. Si en el futuro se quiere seguir
poblando la zona compute-bound, el punto de partida sería probar otros
tiles SIMT de CUTLASS o `Apps_LTIMES` de RAJAPerf (candidato de
prioridad 2 de la nota externa, aún sin probar).

### Criterio exacto de cierre

Cerrado: dos candidatos investigados en orden de prioridad de la
revisión externa, el primero (`CUBLAS_PEDANTIC_MATH`) descartado con
evidencia real de `ncu` (mismo kernel Tensor Core con o sin pedantic),
el segundo (CUTLASS `OpClassSimt`) confirmado con evidencia real en
tres niveles independientes (nombre de kernel SASS, conteo exacto de
DFMA=N³, `tensor_instructions_observed=0` en las tres mediciones de
convergencia) -- no solo un número de OI aislado. Instanciación de
CUTLASS copiada de un test ya validado por el propio proyecto CUTLASS,
no inventada. `gpu_final.yaml` y `catalog.yaml` actualizados.

## F2-XDEV-002 — Primer clasificador de fase GPU real: `train_phase_gpu.py`, contrato de features con evidencia (temperatura descartada), F1 macro=0.576

**Fecha de registro:** 2026-09-15
**Estado:** cerrado. Primer modelo GPU entrenado y serializado
(`fase2_clasificador/models/arbol_prof1_gpu.joblib`) sobre datos reales
de la campaña final GPU.

### Motivación

Mientras la campaña final CPU corría sin supervisión, se adelantó el
trabajo de Fase 2 que no depende de ella: el entrenador GPU no existía
todavía en `fase2_clasificador/` (solo el de CPU, `train_phase.py`) --
hueco explícito documentado en el propio `README.md` del módulo. La
campaña final GPU (18 kernels, F1-GPU-011/012/013) ya había terminado,
con datos reales disponibles y ociosos.

### Investigación y construcción

**Datos ya listos, sin trabajo adicional de postproceso:** cada
directorio de corrida de la campaña GPU ya tenía `training_gpu_phases.csv`
(F1-GPU-003, ya implementado desde antes) -- una fila por corrida
completa (`kernel_ref x gpu_freq_level_id x repetición`), agregada con
estadísticos robustos (mediana/media recortada/std/IQR/min/max/nº
distintos/fracción válida) sobre las muestras NVML de esa corrida.
Verificado con datos reales (540 archivos, `training_eligible=True` en
525 de ellos): el hueco que el `README.md` señalaba ("las muestras
NVML no son fases independientes") ya estaba resuelto por diseño desde
F1-GPU-003 -- no hacía falta construir agregación nueva.

**`fase2_clasificador/training/train_phase_gpu.py` (nuevo):** espejo de
`train_phase.py` (CPU) adaptado a la granularidad de una fila-por-corrida
(no hace falta submuestrear por corrida como en CPU, cada corrida ya es
exactamente una fila). Carga por `rglob` de todos los
`training_gpu_phases.csv` bajo el directorio de campaña, en vez de
reconstruir rutas combinatoriamente kernel×nivel×repetición (evita el
problema de que algunos directorios tienen sufijo `__baseline` y otros
no). Mismos modelos comparados, mismo guardarrail `FORBIDDEN`, mismo
criterio de selección F1+latencia, misma serialización `joblib`+
`metadata.json` que CPU.

**Contrato de features corregido con evidencia real, no con la
propuesta del plan sin verificar (§2.5 exige exactamente este paso
antes de fijar columnas).** Se corrió `run_feature_contract.py --device
gpu` sobre las 540 filas reales (525 elegibles) combinadas de la
campaña: el plan proponía 5 señales por mediana (`gpu_util_pct`,
`gpu_mem_util_pct`, `gpu_power_mw`, `gpu_sm_clock_mhz`,
`gpu_temperature_c`). El análisis real mostró que **las 8 variantes
agregadas de `gpu_temperature_c` correlacionan `|ρ|>0.85` con
`gpu_power_mw` en TODOS los pares** -- en fases de esta duración, la
temperatura no se desacopla de la potencia instantánea, así que no
aporta información independiente. Contrato congelado con las 4 señales
restantes (`gpu_util_pct_median`, `gpu_mem_util_pct_median`,
`gpu_power_mw_median`, `gpu_sm_clock_mhz_median`), VIF razonable
(2.5-6.5) entre ellas. Reporte completo en
`hyperion-results/final/campaigns/gpu/_feature_contract/` en pacca.

### Resultado del entrenamiento (honesto, no inflado)

525 corridas elegibles, 18 kernels, 13 familias algorítmicas
(`derive_kernel_family` ya cubre casi todos los `kernel_ref` de la
campaña GPU; `gpu_cutlass_simt_dgemm_n4096` y `gpu_gemm_native_n4096`
quedan como familia propia -- ningún patrón de `_FAMILY_PATTERNS` los
agrupa con `dgemm`, no bloqueante pero documentado como mejora futura).
Distribución de clase: 387 memory_bound / 138 compute_bound.

**Modelo elegido: árbol de profundidad 1. F1 macro=0.576 (línea base
mayoritaria: 0.563) -- apenas por encima de no hacer nada.** Varias
familias (`dual_cholesky`, `rodinia_gaussian`, `rodinia_heartwall`,
`rodinia_lavamd`) dan F1=0.0 al dejarlas fuera del entrenamiento: el
modelo no generaliza a ellas con solo 4 señales NVML agregadas. Esto es
consistente con lo que el plan anticipa como resultado científicamente
válido a reportar sin ocultar (§5.2) -- no se intentó maquillar
eligiendo otra métrica o descartando familias difíciles.

**Hallazgo colateral real, no un bug:** Random Forest y Extra Trees
miden p99 de ~26,000-27,000 microsegundos (26-27 **milisegundos**) por
predicción de una sola fila -- comportamiento ya documentado de
scikit-learn (overhead de coordinación de threads con `n_jobs=-1` en
predicción de una sola muestra, no amortizado como en un lote). El
criterio de selección F1+latencia los descarta correctamente pese a
tener F1 competitivo con el árbol simple.

### Limitaciones

- F1 macro=0.576 es un resultado débil -- el clasificador GPU, con las
  features disponibles hoy, generaliza poco mejor que la línea base
  mayoritaria. No se investigó si features adicionales (duración de
  fase, fracción de muestras válidas, agregados de dispersión que
  `feature_contract.py` sí proponía en su lista completa de 16
  columnas) mejorarían el resultado -- se priorizó parsimonia y
  fidelidad al espíritu del plan (features físicas simples,
  disponibles en producción) sobre exprimir F1.
- `select_best_model()` con `latency_weight=0.2` (default, sin ajustar)
  -- misma limitación ya documentada para CPU.
- No se comparó contra una variante con más familias/kernels que las
  18 de esta campaña -- es el primer y único modelo GPU entrenado hasta
  ahora, no hay punto de comparación histórico como en CPU (9 kernels
  de `attempt03`).
- `gpu_cutlass_simt_dgemm_n4096`/`gpu_gemm_native_n4096` no se agrupan
  con la familia `dgemm` en `_FAMILY_PATTERNS` -- quedan como familia
  propia de un solo miembro cada una, técnicamente correcto pero
  subóptimo (podría refinarse el patrón para reconocerlos como
  variantes del mismo algoritmo).

### Trabajo pendiente

1. Correr `run_pretraining_readiness.py --gpu-dataset ...` apuntando a
   la campaña real para confirmar si esto cierra los gates GPU
   pendientes.
2. Evaluar si agregar features de dispersión (std/iqr) o duración de
   fase mejora F1 macro, reportando la comparación explícitamente en
   vez de solo el mejor resultado.
3. Refinar `_FAMILY_PATTERNS` para agrupar `gpu_cutlass_simt_dgemm_*`/
   `gpu_gemm_native_*` con la familia `dgemm`.
4. Repetir este mismo proceso para CPU en cuanto la campaña final CPU
   termine (F1-XDEV-002 warmup + contrato de features + entrenamiento
   sobre el catálogo completo de 43 kernels, no solo los 9 históricos).

### Criterio exacto de cierre

Cerrado: entrenador GPU construido siguiendo exactamente el mismo
patrón validado de CPU, contrato de features fijado con evidencia real
(no con la propuesta sin verificar del plan), modelo entrenado y
serializado sobre datos de una campaña real completa (no un fixture
sintético), resultado reportado con sus limitaciones explícitas
(F1 débil, hallazgo de latencia de sklearn) en vez de presentado como
éxito. Primer artefacto de modelo GPU versionado en el repo.

## F1-XDEV-002 (cierre para GPU) — Calibración real de warmup ejecutada: 46 corridas cambian de veredicto, 2 kernels quedan sin datos de entrenamiento

**Fecha de registro:** 2026-09-15
**Estado:** cerrado para GPU (los 18 kernels de `gpu_final.yaml`
calibrados con evidencia real). CPU queda pendiente para cuando su
campaña final termine.

### Motivación

El usuario preguntó directamente si el warmup se había medido antes de
entrenar el clasificador GPU. Verificación honesta: `gpu_final.yaml`
declara `warmup_seconds_override: 0.0` (línea 86) -- **intencional**,
es el primer paso del procedimiento documentado de F1-XDEV-002 (correr
la campaña real sin excluir el transitorio, para poder medirlo
después), pero los pasos 2-4 (calibrar, aplicar, reprocesar) nunca se
habían ejecutado. Confirmado directamente contra los datos: las 540
filas de `training_gpu_phases.csv` tenían
`n_nvml_samples_warmup_excluded = 0` en el 100% de los casos -- el
modelo entrenado hasta ese momento tenía el transitorio de arranque
mezclado en cada agregado (mediana de potencia/utilización/reloj
calculada incluyendo las primeras muestras post-arranque, no solo el
estado estable).

### Ejecución (en `pacca03`, nodo libre -- sin tocar `paccaA100` mientras
corría la campaña final CPU, pedido explícito del usuario)

1. `warmup_calibration.py --device gpu` sobre los 18 kernels de
   `gpu_final.yaml`, usando `windows.csv` de la campaña real ya
   ejecutada (28-30 corridas analizadas por kernel, método
   `cv_threshold` -- el principal, no un método de respaldo). Los 18
   dieron `status=measured`, ninguno cayó a `insufficient_signal`.
2. **Los valores reales resultaron mayores que los declarados en 17 de
   18 casos** (ej. `dual_stencil_gpu_N1024`: 0.05s declarado → 4.33s
   real; `dual_cholesky_gpu_N2048`: 0.05s → 4.35s; `rodinia_dwt2d`:
   1.0s → 6.41s) -- confirma que el transitorio real es
   sistemáticamente más largo de lo que se había puesto a mano en
   distintos momentos de la sesión.
3. `--apply` sobre el catálogo desplegado en pacca (con backup
   automático `.bak`) y sobre `fase1_telemetria/catalog/catalog.yaml`
   local (edición directa, verificada contra `yaml.safe_load`, 280
   entradas intactas).
4. `repostprocess_campaign.py` sobre la campaña real con el catálogo
   ya corregido (`ignore_manifest_override` por defecto -- ignora el
   `warmup_seconds_override: 0.0` del manifiesto a propósito, usa el
   `warmup_seconds` ya calibrado del catálogo). **Hueco de diseño
   encontrado y sorteado, no corregido de raíz:** el cargador de
   manifiestos (`manifest_module.load`) rechaza con `I07` cualquier
   manifiesto cuyo `output_dir` ya exista y `overwrite: false` --
   correcto para lanzar una campaña nueva, pero bloquea reprocesar una
   ya completada (el caso de uso explícito de este script, que nunca
   relanza kernels). Se sorteó con una copia temporal del manifiesto
   con `overwrite: true` solo para esta invocación
   (`gpu_final_repostprocess_tmp.yaml`, no versionada). Pendiente:
   `repostprocess_campaign.py` debería tener su propio mecanismo para
   esto en vez de depender de una copia manual del manifiesto.

### Resultado -- 46 corridas cambian de veredicto, no se ocultó

Al excluir correctamente el warmup real (más largo), **46 de 540
corridas (8.5%) pasan de `accepted=true` a `accepted=false`**
(`I10: ventanas gpu_telemetry insuficientes, <5 vs.
target_windows_per_repetition=5`) -- todas de `rodinia_lud`,
`rodinia_lavamd` y `rodinia_dwt2d`, los 3 kernels con el warmup
calibrado más largo (6.4-6.8s) relativos a su duración total de
corrida. **Estas corridas estaban siendo aceptadas por error antes de
esta corrección**: el código contaba muestras del transitorio de
arranque como si fueran ventanas de estado estable válidas, porque
nada se estaba excluyendo.

Efecto sobre el dataset de entrenamiento: de 525 a **334 corridas
elegibles**, de 18 a **16 kernels**, de 13 a **11 familias algorítmicas**
-- `rodinia_lavamd` (la ancla compute-bound más profunda del catálogo,
log2(OI/ridge)=+7.52) y `rodinia_dwt2d` quedan **sin ninguna corrida
utilizable** para entrenamiento: ninguna de sus repeticiones/niveles
alcanza las 5 ventanas de estado estable reales una vez excluido el
transitorio correcto.

### Reentrenamiento (F2-XDEV-002, actualización)

| | Sin balanceo, warmup sin corregir | Con balanceo, warmup sin corregir | Con balanceo, warmup corregido |
|---|---|---|---|
| Corridas | 525 | 525 | **334** |
| Familias | 13 | 13 | **11** |
| F1 macro | 0.576 | 0.594 | **0.617** |
| F1 compute_bound | *(no se medía)* | 0.148 | **0.197** |
| F1 memory_bound | *(no se medía)* | 0.579 | 0.583 |

Las tres correcciones (balanceo de clase, F1 por clase/matriz de
confusión, warmup real) se suman en la dirección correcta -- F1 macro
sube de 0.576 a 0.617, F1 compute_bound casi se duplica (0.148→0.197)
-- pero el modelo sigue siendo débil en términos absolutos, y ahora con
menos datos y sin su ancla compute-bound más fuerte.

### Limitaciones

- `rodinia_lavamd`/`rodinia_dwt2d` quedan efectivamente fuera del
  catálogo de entrenamiento GPU tal como está dimensionado hoy -- no
  se investigó si aumentar su duración de corrida (mismo criterio de
  F1-GPU-006 aplicado a otros kernels) recuperaría suficientes
  ventanas de estado estable. Sería el paso natural si se quiere
  recuperar la representación compute-bound profunda.
- `T_transición_gpu` (§2.4.1 del plan, medido en job 7055,
  ~170ms conservador) NO está wireado como filtro explícito en
  `gpu_phases.py` (`excluded_transition_not_settled` no existe como
  estado) -- verificado que no cambia nada en el dataset actual (todas
  las corridas duran 25-34s, muy por encima de 170ms), pero sigue
  siendo una pieza real del plan sin construir.
- No se calibró warmup para CPU todavía -- la campaña final CPU sigue
  corriendo.
- El mecanismo de "copia temporal del manifiesto con overwrite:true"
  para poder reprocesar es un sorteo, no una solución -- documentado
  como pendiente en la sección de ejecución arriba.

### Trabajo pendiente

1. Evaluar si agrandar `rodinia_lavamd`/`rodinia_dwt2d` (más
   iteraciones/tamaño, mismo criterio F1-GPU-006) recuperaría
   ventanas de estado estable suficientes.
2. Implementar el filtro `T_transición_gpu` en `gpu_phases.py` por
   completitud metodológica, aunque hoy no cambie el resultado.
3. Repetir este mismo proceso de calibración de warmup para CPU en
   cuanto su campaña final termine.
4. Darle a `repostprocess_campaign.py` su propio mecanismo para
   reprocesar una campaña existente sin depender de una copia manual
   del manifiesto con `overwrite: true`.

### Criterio exacto de cierre

Cerrado para GPU: calibración ejecutada con evidencia real (no
teórica) sobre los 18 kernels, aplicada al catálogo (local y
desplegado) con backup, campaña completa reprocesada, 46 corridas
reclasificadas correctamente en vez de ocultadas, modelo reentrenado
con los datos limpios y el resultado (incluida la pérdida de
`rodinia_lavamd`/`rodinia_dwt2d`) reportado sin maquillar.

## F2-XDEV-003 — Búsqueda de hiperparámetros (Optuna/TPE) añadida: no existía ninguna, contradiciendo §3.3 punto 1 del plan

**Fecha de registro:** 2026-09-15
**Estado:** cerrado (módulo construido, probado, e integrado en ambos
entrenadores; reentrenamiento real del modelo GPU en curso al momento de
este registro).

### Motivación

El usuario preguntó directamente si el entrenamiento usaba ajuste de
hiperparámetros/Optuna. Verificación honesta contra el código real (no
asumida): `build_models()` en `train_phase.py`/`train_phase_gpu.py` tenía
TODOS los hiperparámetros fijos a mano (`max_depth=12`, `n_estimators=100`,
etc.), sin ningún `GridSearchCV`, `RandomizedSearchCV` ni Optuna en
ninguna parte del repositorio -- confirmado con `grep` y con `pip show
optuna` (no instalado). Esto contradice directamente el plan, §3.3 punto 1:
*"Validación anidada: partición externa leave-one-familia-out para estimar
generalización, con búsqueda de hiperparámetros (grid o Bayesiana) en un
split interno."* El modelo GPU documentado en el capítulo de resultados
(F1 macro=0.617) se había entrenado con la primera configuración razonable,
no con una búsqueda.

### Diseño

Se instaló Optuna (TPE/Bayesiana, sobre grid search puro: los espacios de
`random_forest`/`extra_trees`/`xgboost` combinan enteros y reales en varias
dimensiones, una grilla con la misma resolución sería cara) y se construyó:

1. **`fase2_clasificador/eval/hyperparam_search.py`**: `search_best_params()`
   maximiza el F1 macro medio de un split agrupado INTERNO (mismo criterio
   de agrupación por familia que `protocol.py`, nunca por fila ni por
   `kernel_ref` individual) sobre el `df` que recibe -- que debe ser ya el
   conjunto de entrenamiento de un pliegue EXTERNO, nunca el dataset
   completo, para que la familia de prueba del pliegue externo nunca
   participe en la búsqueda (anidación real, no una búsqueda compartida
   entre comparación y selección). `n_groups()` cuenta cuántos pliegues
   produciría un `fold_fn` dado sobre un `df`, sin fabricar un split si hay
   menos de 2 familias -- lanza `ValueError` explícito en ese caso.
2. **`fase2_clasificador/training/model_specs.py`**: separa los 7 modelos
   del plan (§3.2) en dos grupos -- `build_fixed_models()` (`mayoritaria` y
   `arbol_prof1`, líneas base deliberadamente NO optimizadas: la primera es
   "no hacer nada", la segunda es el punto de comparación interpretable del
   plan) y `tunable_specs()` (regresión logística, árbol de profundidad
   variable, Random Forest, Extra Trees, XGBoost -- cada uno con un
   constructor y un espacio de búsqueda). `build_models()` se conserva
   como configuración fija de respaldo (usada cuando un pliegue no tiene
   familias suficientes para un split interno, o con `--n-trials 0`).
3. **`main()` de ambos entrenadores, reestructurado**: el bucle externo
   ahora itera primero por pliegue (leave-one-familia-out/leave-one-kernel-out
   sobre `kernel_family`), y dentro de cada pliegue busca hiperparámetros
   por separado para cada modelo sintonizable usando SOLO los datos de
   entrenamiento de ese pliegue -- nunca los del pliegue de prueba. El
   modelo final que se serializa reentrena la búsqueda una vez más sobre el
   dataset COMPLETO (ya no queda ninguna familia por proteger) para elegir
   los hiperparámetros de producción. Metadata nueva por modelo:
   `hyperparameter_search.best_params_final_model` y
   `best_params_por_pliegue_externo` (para auditar si los hiperparámetros
   elegidos son estables entre pliegues o oscilan).

### Verificación

90/90 tests pasan (80 previos + 10 nuevos: `test_hyperparam_search.py`,
`test_model_specs.py`), incluidos casos de: `ValueError` explícito con <2
familias (no se fabrica un split), `n_groups()` cuenta familias reales y no
`kernel_ref` únicos, y una búsqueda sobre un problema sintéticamente
separable converge a F1>0.9. Suite completa: ~29s (antes ~15s, el aumento
es el costo real de ejercitar la búsqueda en el test de integración de
`main()`, con `--n-trials 3` para mantenerlo rápido).

### Pendiente al momento de este registro

Reentrenar el modelo GPU real con `--n-trials 30` sobre el dataset
combinado de 522 corridas (`/tmp/combined_training_gpu_phases.csv`) está en
curso; el resultado (F1 macro con hiperparámetros buscados vs. los
0.617 fijos anteriores, y si los hiperparámetros elegidos son estables
entre las 11 familias) se registra en una entrada de seguimiento aparte una
vez complete. No se ha ejecutado ninguna búsqueda para CPU todavía -- la
campaña final CPU sigue sin confirmarse terminada.

## F2-CPU-001 — Selección anidada corta y abstención para el clasificador CPU

**Fecha de registro:** 2026-09-18
**Estado:** ejecutado y documentado. Job `7459` en `pacca01`, 14 min 12 s,
sin ejecutar una búsqueda Optuna adicional.

### Motivación

La campaña CPU final aporta 1\,169\,833 intervalos elegibles, pero solo 24
familias algorítmicas independientes. La evaluación preliminar con XGBoost,
interacciones PMU y ponderación por familia alcanzó F1 macro medio LOFO de
0.653 (cifra descartada el 2026-09-19 por posible doble balanceo; ver `F2-CPU-002`). La diferencia grande frente a un split aleatorio mostró que el límite
principal es la transferencia entre algoritmos, no la cantidad de intervalos.

Se descartó eliminar retrospectivamente la familia `gap_pr`. Su exclusión
elevó la F1 macro de la evaluación preliminar de 0.653 a 0.658, una variación
insuficiente y metodológicamente inválida como solución. Se confirmó además
que `freq_khz_observed` es una covariable de producción válida porque el
techo de cómputo y el ridge dependen de la frecuencia física observada. En
cambio, `delta_enabled_ns`, `delta_running_ns` y `running_ratio` quedaron
reservados para aceptación de calidad de los contadores y se excluyeron del
vector de entrada.

### Diseño

El experimento compara dos candidatos XGBoost con ponderación inversa por
familia y clase. El candidato base usa IPC, MPKI, tasa de fallos de caché,
proporción de ciclos detenidos por memoria, instrucciones por segundo y
frecuencia observada. El segundo añade razones PMU calculables en línea entre
referencias de caché, fallos, instrucciones, ciclos y ciclos detenidos.

Cada familia externa se reserva por completo. Dentro de sus familias de
entrenamiento se ejecuta otra validación LOFO para escoger la representación
y el umbral de confianza. Los umbrales candidatos son 0.50 a 0.90 en pasos
de 0.05 y la selección exige cobertura interna media de al menos 60\%. La
salida operativa es `compute_bound` o `memory_bound` cuando
`max(p, 1-p)` alcanza el umbral y `revisar` en caso contrario. La etiqueta
Roofline y sus insumos, FLOPs, bytes `uncore`, intensidad y ridge, nunca
ingresan al modelo.

El cálculo utiliza un límite reproducible de 1000 intervalos por celda
familia--clase, que produjo 36\,310 filas. Este límite reduce el costo de la
validación anidada y evita que las familias largas dominen la selección. No
convierte las filas de una misma familia en unidades independientes.

### Resultado inicial descartado por doble balance

Los valores de esta subsección corresponden al job `7459` y se conservan
solo como registro histórico. No deben citarse como resultado del modelo,
porque combinaban pesos por familia--clase con `scale_pos_weight`. El resultado
corregido y vigente se encuentra en la subsección de actualización del balance.

| Medida | Resultado |
|---|---:|
| Familias externas | 24 |
| Representación con interacciones elegida | 17 de 24 pliegues |
| Umbral 0.90 elegido | 21 de 24 pliegues |
| Cobertura automática | 63.1\% |
| Salida `revisar` | 36.9\% |
| F1 macro agrupada dentro de la cobertura | 0.871 |
| Exactitud agrupada dentro de la cobertura | 0.874 |
| F1 macro media por familia dentro de la cobertura | 0.664 |
| Desviación estándar entre familias | 0.231 |
| Peor familia, `gap_pr` | 0.308 |

La F1 agrupada de 0.871 no es una F1 de cobertura total ni sustituye la F1
media por familia. Los casos cubiertos están concentrados en familias donde
la confianza es alta, mientras que varias familias difíciles conservan baja
cobertura o baja F1. La abstención mejora la calidad de las decisiones que el
sistema emite, pero no resuelve la falta de identificabilidad universal de la
etiqueta Roofline a partir de las señales PMU disponibles.

### Decisión operativa

Se conserva XGBoost con interacciones PMU y ponderación por familia como
candidato de producción. La abstención se incorpora como una tercera salida
operativa. Una salida `revisar` debe conducir a una microcaracterización
ligera o mantener una frecuencia conservadora, nunca a forzar una clase. El
siguiente aumento de capacidad debe priorizar nuevas familias independientes
con acceso irregular o comportamiento intermedio, y evaluar eventos PMU
adicionales solo después de verificar que caben sin degradar la calidad de los
contadores.

### Artefactos

- `fase2_clasificador/analysis/evaluate_cpu_selective_nested.py`
- `scripts/pacca/hyp_cpu_selective_nested.sbatch`
- `/home/latorresn/hyperion-results/final/training/cpu_selective_nested_20260918/cpu_selective_nested_lofo.json`
- `docs/libro/secciones/03_resultados.tex`
- `docs/libro/secciones/04_discusion.tex`

### Actualización de balance y auditoría de abstención

**Fecha de registro:** 2026-09-18

La primera evaluación selectiva combinaba pesos explícitos por familia--clase
con `scale_pos_weight` de XGBoost. Esa combinación corregía dos veces el
desbalance de clase. Se corrigió fijando `scale_pos_weight=1.0` cuando los
pesos familia--clase están presentes y se repitió la evaluación como job
`7461`.

El resultado corregido cubre 68.9\% de los intervalos y devuelve `revisar`
en 31.1\%. La F1 macro agrupada dentro de la cobertura es 0.860, mientras la
F1 macro media por familia es 0.664, con desviación estándar 0.242. El peor
pliegue externo es `npb_mg`, con F1 0.240. La representación con
interacciones PMU se seleccionó en 15 de las 24 familias externas.

La auditoría LOFO del candidato congelado se ejecutó como job `7463`. Las
familias con menor cobertura fueron `npb_lu` (31.4\%), `npb_bt` (34.5\%) y
`rodinia_lavamd` (35.2\%). Los mayores errores entre decisiones automáticas
ocurrieron en `rodinia_lavamd` (56.3\%), `npb_mg` (55.2\%), `npb_ft`
(51.1\%) y `gap_pr` (49.8\%). Los rangos de frecuencia observada más bajos
concentraron más abstenciones y errores. Esto confirma que una probabilidad
alta no garantiza transferencia a una familia no vista.

La regla operativa provisional es emitir `compute_bound` o `memory_bound`
solo cuando la confianza sea al menos 0.85. En otro caso se emite `revisar`,
se conserva la frecuencia actual o nativa y se esperan dos intervalos
`uncore` válidos consecutivos. Si ambos permiten una caracterización física
consistente se usa esa evidencia; de lo contrario se mantiene la política
conservadora. Esta regla no asigna todavía una frecuencia de menor EDP.

Artefactos adicionales:

- `fase2_clasificador/analysis/audit_cpu_selective_errors.py`
- `/home/latorresn/hyperion-results/final/training/cpu_error_audit_20260918/`
- `/home/latorresn/hyperion-results/final/models/cpu_selective_candidate_20260918/`

### Documentación exhaustiva y trazabilidad numérica

**Fecha de registro:** 2026-09-18

El libro se amplió para separar el procedimiento reproducible, los resultados
observados y su interpretación. La metodología ahora documenta las fórmulas de
las 12 entradas, la auditoría Pearson/Spearman/VIF, el muestreo con semilla
20260918, el peso por celda familia--clase, la validación LOFO anidada y las
fórmulas de cobertura, confianza y F1 macro. Los resultados incorporan tablas
completas de muestreo y auditoría para las 24 familias, matriz de correlación,
curva de selección del umbral, matriz de confusión, desempeño por familia y
resultado por cuartil de frecuencia.

Se dejó explícita la diferencia entre los dos porcentajes de cobertura. La
evaluación anidada corregida del job `7461` cubre 25 020 de 36 310 intervalos,
equivalentes a 68.9066 %, porque selecciona representación y umbral dentro de
cada pliegue externo. La auditoría congelada del job `7463` cubre 25 111,
equivalentes a 69.1573 %, porque aplica interacciones PMU y umbral 0.85 a todos
los pliegues. La diferencia de 91 intervalos no procede de otro dataset.

La documentación también registra los requisitos todavía abiertos del plan.
Falta medir latencia p95/p99 en el nodo objetivo, cerrar la selección conjunta
entre error y latencia, y realizar una prueba externa única con familias nuevas
selladas. La política de frecuencia de menor EDP continúa bloqueada hasta
obtener niveles CPU físicamente distintos bajo carga.

Artefactos incorporados al repositorio:

- `docs/libro/datos/cpu_modelo_20260918/`
- `docs/libro/scripts/generar_figuras_resultados_cpu.py`
- seis figuras de auditoría CPU con fecha `20260918` bajo `docs/libro/figuras/`
- `docs/libro/secciones/02_metodologia.tex`
- `docs/libro/secciones/03_resultados.tex`
- `docs/libro/secciones/04_discusion.tex`
- `docs/libro/secciones/05_conclusiones.tex`

## F2-XDEV-004 — Resultado real de la búsqueda de hiperparámetros GPU: ningún modelo optimizado supera al árbol de profundidad 1

**Fecha de registro:** 2026-09-15
**Estado:** cerrado. Ejecutado en `pacca03` (job `sbatch` 7245, nodo libre,
sin tocar `paccaA100`), sobre las 540 corridas reales de la campaña final
(no una copia local) -- `--n-trials 30`.

### Incidente operativo previo

El primer intento se lanzó con `srun` encadenado dentro de una sesión SSH
anidada desde la máquina local del usuario, en `run_in_background` -- el
usuario detectó correctamente el problema ("manda eso en pacca, por qué en
mi PC") antes de que se materializara, y además señaló que iba a suspender
su equipo. Al matar el proceso local para reubicarlo, el job de `srun` en
pacca murió con él (confirmado por notificación `failed`, exit 144) --
`srun` depende de que el proceso que lo lanzó siga vivo, a diferencia de
`sbatch`. Relanzado correctamente como `sbatch` (script con
`--nodelist=pacca03`, sin `--time`), que sí sobrevive a la desconexión del
cliente -- terminó sin problema, confirmando que el `sbatch` es el
mecanismo correcto para trabajo que debe sobrevivir a que el usuario cierre
su sesión, no solo para campañas largas de Fase 1.

### Resultado

| Modelo | F1 macro (antes, sin búsqueda) | F1 macro (con búsqueda Optuna real) |
|---|---|---|
| `arbol_prof1` (línea base fija, nunca se optimiza) | 0.617 | **0.617 (sin cambio)** |
| `mayoritaria` (línea base fija) | 0.553 | 0.553 (sin cambio) |
| `random_forest` | 0.420 | 0.470 |
| `extra_trees` | 0.403 | 0.463 |
| `arbol_prof6` | 0.444 | 0.404 |
| `xgboost` | 0.377 | 0.392 |
| `regresion_log` | 0.430 | 0.360 |

El modelo elegido para servir **no cambió**: sigue siendo `arbol_prof1`
(F1 macro=0.617), porque es una línea base deliberadamente fija (nunca se
le buscan hiperparámetros, por diseño -- ver `model_specs.py`). De los 5
modelos que sí se optimizaron, el resultado es mixto pero la conclusión es
clara: **ninguno superó al árbol simple**, incluso con una búsqueda real de
30 intentos de Optuna por modelo, anidada correctamente (la familia de
prueba de cada pliegue externo nunca participa en la búsqueda de
hiperparámetros de ese pliegue). Dos de los cinco (`random_forest`,
`extra_trees`) mejoraron con la búsqueda; dos empeoraron
(`arbol_prof6`, `regresion_log`) frente a su configuración fija anterior;
uno mejoró marginalmente (`xgboost`).

### Interpretación

No es un resultado negativo del módulo de búsqueda -- es información real
sobre el dataset: con 11 familias algorítmicas y una búsqueda ANIDADA (el
conjunto de entrenamiento de cada pliegue externo se subdivide otra vez
para la búsqueda interna), cada evaluación de un candidato de
hiperparámetros ve muy pocas familias. Los modelos de alta varianza
(bosques, XGBoost) tienen margen para sobreajustar el ruido de esa
partición pequeña en vez de encontrar una configuración que generalice,
mientras que la simplicidad forzada de un árbol de profundidad 1 actúa
como una regularización más fuerte que cualquier hiperparámetro que la
búsqueda pueda encontrarle a un modelo más flexible. Reportado en el libro
sin maquillar (mismo criterio que el resto de esta reconstrucción):
`docs/libro/secciones/03_resultados.tex`, párrafo nuevo tras la Figura
`fig_gpu_pareto_calidad_latencia_20260915.png` (regenerada con los números
reales), sección §3.9.2.

### Artefactos actualizados

- `fase2_clasificador/models/arbol_prof1_gpu.joblib` y
  `.metadata.json` (local, fetched vía `base64` sobre la cadena SSH --
  `scp` directo no funciona con el `ProxyJump` configurado para `pacca`).
  Metadata ahora incluye `hyperparameter_search` con
  `best_params_final_model`/`best_params_por_pliegue_externo` (vacíos para
  `arbol_prof1`, correcto: es la línea base fija).
- `docs/libro/figuras/fig_gpu_pareto_calidad_latencia_20260915.png`
  regenerada con los 7 valores reales post-búsqueda.
- Libro recompilado limpio: 134 páginas, cero overfull/underfull, cero
  referencias indefinidas.

### Pendiente

- Ejecutar el mismo proceso para CPU en cuanto la campaña final cierre.
- Los 10 kernels de patrón irregular de §2.1.1 y el wireado de
  `T_transición_gpu` siguen pendientes (ver mensaje de seguimiento directo
  al usuario).

## F2-CPU-002 - Informe de calidad del clasificador CPU y batería de pruebas (2026-09-18)

- **Qué se hizo:** batería reproducible `fase2_clasificador/analysis/cpu_quality_report.py` sobre `feature_contract_source.csv` (1 169 833 intervalos, 24 familias): matriz de 14 configuraciones con IC95 por bootstrap de familias y 5 semillas, protocolos de validación, curva de aprendizaje, calibración, abstención, distancia al ridge, techo k-vecinos, perfiles gemelos, sensibilidad al tope de muestreo, variables temporales, regresor de OI, umbral y complejidad anidados, calibración en línea y latencia en pacca01.
- **Resultado:** exactitud balanceada por celda 0.709 (IC95 0.633 a 0.780) para el candidato; todas las variantes razonables empatan (0.68 a 0.71); techo de información (k-vecinos 0.68 a 0.70); brecha aleatorio contra familia 0.894 contra 0.709; +0.034 estimado por duplicar el catálogo; abstención +0.03 con mala calibración (ECE 0.161).
- **Corrección de métrica:** el F1 macro medio por familia (0.664) queda sustituido como cifra de calidad por la exactitud balanceada por celda con IC95 (definición y motivo en el informe).
- **Documento y artefactos:** `docs/general/Informe_Calidad_Modelo_CPU_20260918.md`, datos en `docs/libro/datos/cpu_calidad_20260918/`, figuras `docs/libro/figuras/fig_cpu_calidad_*_20260918.png` (script `docs/libro/scripts/generar_figuras_calidad_cpu.py`).
- **Libro:** el árbol de trabajo de `03_resultados.tex` había eliminado respecto de `HEAD` secciones de instrumento, GPU anterior y catálogo (sección 7 del informe). Restauradas las partes CPU vigentes (DVFS verificado, uncore, aislamiento, techos Roofline con Advisor, kernels irregulares con la campaña de 3 familias nuevas); lo GPU queda pendiente.
- **Calibración en línea causal con cambios de fase:** con 2.6 % de intervalos etiquetados en la primera mitad, exactitud balanceada 0.85 en bloques mixtos (0.62 sin adaptar; repetir la última etiqueta 0.51).
- **Familias nuevas:** verificado nodo paccaA100, turbo desactivado; solo `particlefilter` completa, `kmeans` y `srad` parciales; las tres casi 100 % memory.
- **Decisión (2026-09-19):** la cifra 0.653 del cribado `7455` se descarta y no se reejecuta; la reemplaza la batería con balanceo único. Turbo y nodo de las 3 familias nuevas ya verificados (arriba). **Pendiente:** auditar ruido de etiquetas.

## F2-CPU-003 - Reescritura de la sección de resultados CPU y cierre del modelo final (2026-09-19)

- **Búsqueda de hiperparámetros anidada (Optuna TPE) sobre la representación final:** no mejora. XGBoost 0.682 frente a 0.708 fijo (diferencia pareada -0.025, IC95 -0.072 a +0.015); logística 0.674 frente a 0.688. El criterio interno no discrimina (mejor puntaje interno 0.693 a 0.746 frente a resultado externo 0.343 a 1.000) y las configuraciones elegidas no convergen entre pliegues.
- **Cifras finales unificadas** (una población, una configuración, 5 semillas): exactitud balanceada 0.709, F1 macro agrupada 0.709, exactitud 0.747; con umbral 0.85, cobertura 0.726 y exactitud balanceada 0.723. Sustituyen a las cifras de cobertura 68.9 % / 69.2 % y F1 por familia 0.664.
- **Libro:** subsección del clasificador CPU reescrita completa (seis apartados más síntesis), sin números de job, sin *proxy* de L2 y sin relato de configuraciones descartadas; métrica principal definida en metodología; discusión y conclusiones actualizadas; guion largo eliminado de la prosa de ambas. 144 páginas, compila limpio.
- **Artefactos:** `docs/libro/datos/cpu_calidad_20260918/` (43 archivos), `docs/libro/scripts/generar_figuras_modelo_cpu.py`, informe en `docs/general/Informe_Calidad_Modelo_CPU_20260918.md`.
- **Métricas de evaluación (F2-CPU-003, ampliación):** apartado nuevo con definición de celda, nueve métricas definidas, ejemplo numérico, tabla de valores con desviación entre semillas e IC95, comparación de estabilidad (0.004 frente a 0.019 a 0.047) y declaración del límite de las celdas de soporte mínimo (4 celdas, 44 intervalos, 9.1 % del peso). Cifras de una sola semilla dadas antes quedan sustituidas por las de cinco.

## F2-CPU-004 - Tabla de política CPU, modelo de potencia y familias compute nuevas (2026-09-19)

- **Tabla clase -> frecuencia (plan §3.4/§3.5).** EDP por corrida completa (RAPL paquete + DRAM por tiempo), con trabajo fijo entre niveles; Wilcoxon pareado por kernel contra REF. Resultado: no actuar en compute (16 kernels) ni en memory (27). El EDP por ventana de 1 ms de `derive_policy_table.py` no es comparable entre niveles y no se usó. `phasic` se excluye (duración fija, alfa 0.005). Script `fase2_clasificador/analysis/cpu_policy_table.py`.
- **Modelo de potencia.** P(f) = P0 + c f^k: P0 mediana 85 W, 84 % de la potencia a 3.2 GHz; bajar a 0.8 GHz ahorra 15 %. Alfa crítico mediano 0.14. Solo 5 kernels tipo STREAM (alfa 0.18-0.22) ganan 4-12 %. Script `cpu_power_model.py`.
- **Techo con clasificador perfecto por kernel:** 2.9 % medio de EDP, 12.5 % máximo.
- **Familias compute nuevas.** Nueve adaptadores RAJAPerf; tamizaje (REF/F0/F8): edge3d, fir, mass3dpa, mat_mat_shared, pi_reduce y trap_int salen compute (~100 %); convection3dpa, diffusion3dpa y ltimes salen memory. Barrido completo (10 niveles x 3 rep) de las seis más `ptrchase` en curso (`cpu_compute_sweep_20260919`).
- Detalle en `docs/general/Informe_Calidad_Modelo_CPU_20260918.md` secciones 14 a 16.

## F2-CPU-005 - Población final de 30 familias (2026-09-20)

- **Decisión:** las seis familias compute añadidas (`apps_edge3d`, `apps_fir`, `apps_mass3dpa`, `basic_mat_mat_shared`, `basic_pi_reduce`, `basic_trap_int`) pasan a formar parte del conjunto; cada una es su propia familia (`protocol.derive_kernel_family`). `ptrchase` queda fuera del clasificador.
- **Resultado:** LOFO en 30 pliegues, exactitud balanceada por celda 0.727 (IC95 0.670 a 0.803); con abstención 0.85, 0.750 con cobertura 0.733. Detalle en el informe, sección 18, y datos en `docs/libro/datos/cpu_calidad_30fam/`.
- **Libro:** sección de resultados CPU, metodología, discusión y conclusiones actualizadas a esta población.
- **Pendiente:** reexportar el modelo serializado con las 30 familias (el exportado en el clúster se ajustó con 24).

## F2-CPU-006 - Vector final de 6 variables, población completa y umbral por exactitud por celda (2026-09-21)

- **Decisión:** el vector final CPU pasa de 12 a 6 variables (`ipc`, `mpki`, `cache_miss_rate`, `stall_mem_ratio`, `ips`, `freq_khz_observed`). Las seis extra salían de los mismos contadores, así que no cambia el costo de lectura. Con 30 familias: exactitud por celda 0.728 (6) frente a 0.724 (12), diferencia de -0.004 (IC95 -0.026 a +0.011); F1 macro agrupada 0.760 frente a 0.783 (diferencia -0.023 para las 6, IC95 -0.053 a +0.002). La regresión logística con 12 variables queda 0.067 por debajo de XGBoost (IC95 -0.105 a -0.030); con 6, 0.048 (-0.092 a 0.000).
- **Población evaluada:** los 1 309 755 intervalos elegibles. Los 12 774 intervalos con cero fallos de caché (99.98 % compute) se descartaban solo porque `stalls_mem_per_cache_miss` queda indefinida; el modelo final no la usa. Las referencias de 12 variables imputan la mediana.
- **Resultado final** (job 7507, pacca03, 24 min; 5 semillas): exactitud por celda 0.728, F1 macro agrupada 0.760, exactitud 0.769; sensibilidad compute 0.714 y memory 0.806. Con umbral 0.85: cobertura 0.722, exactitud por celda 0.738, F1 macro 0.842.
- **Umbral:** ahora se elige con exactitud balanceada por celda sobre lo decidido (cobertura media por celda >= 0.60), no con F1 macro por familia. La regla propone 0.90 (0.752, cobertura 0.649); **el autor congela 0.85** (0.749, cobertura 0.714) porque la curva es plana y 0.90 pierde 6.5 puntos de cobertura. Etapa `threshold_final` registra la rejilla.
- **Búsqueda anidada TPE con 6 variables (15 trials por pliegue):** XGBoost +0.019 frente a la configuración fija (IC95 -0.007 a +0.050; mejora en 14 familias, empeora en 8), regresión logística +0.002. No distinguible; se conserva la configuración fija. La búsqueda global de días (jobs 7355 y 7390, `train_phase.py`, representación anterior con `running_ratio`, métrica F1 por familia) no decide nada.
- **Auditoría §2.5 del vector final** (`cpu_feature_audit_final.py`, 1 309 755 intervalos): ningún par supera 0.85 (máximo 0.741, `ipc` con `stall_mem_ratio`, signo negativo); VIF máximo 5.5 con Pearson y 7.3 con rangos, ambos de `ipc`.
- **Tope por nivel de frecuencia:** repartir el tope entre los 10 niveles cambia la exactitud por celda en -0.003 (IC95 -0.026 a +0.017); el sorteo simple desbalancea los niveles 4.1 veces (mediana).
- **Latencia en el nodo de destino** (job 7512, paccaA100 libre, cores 0 a 3): XGBoost 6 variables p50 243, p95 264, p99 404 us; regresión logística 450/467/475; Random Forest 5297/5373/5453; Extra Trees 5323/5406/5477; árbol de profundidad 1 105/108/109. El core 16 de paccaA100 estaba fijado en 800 MHz (min = max); no se modificó.
- **Modelo serializado:** `/home/latorresn/hyperion-results/final/models/cpu_selective_candidate_base6_20260921/cpu_xgboost_baseline_pmu.joblib` (42 458 filas, umbral 0.85). Sustituye al exportado con 12 variables el 2026-09-20.
- **Comparación completa de modelos** (etapa `model_grid`, job 7513): 5 modelos x 2 representaciones, cada uno fijo y con búsqueda anidada, mismo presupuesto y criterio. Resultado pendiente.
- **Cierre del conjunto de kernels (decisión del autor):** no se añaden más kernels de entrenamiento; no habrá prueba externa sellada y se declara como limitación.
- **Libro:** capítulos 2 a 5 actualizados; commits `6168736`, `2a659b5`, `cd7135b`. Incidente: el commit `72547e6` incluyó por error seis archivos sin versionar del autor (se retiraron del índice en `858ebfa`, siguen en el historial).
- **Archivos derivados (2026-09-21):** modelo CPU en el repo, `fase2_clasificador/models/xgboost_cpu.joblib` y `xgboost_cpu.metadata.json` (sha256 `48172683...bc4cb`, 6 variables, umbral 0.85); tabla de política regenerada en `docs/libro/datos/cpu_calidad_30fam/politica/policy_cpu.json` con el formato de plan §3.5 punto 6 (nivel, campañas de origen, kernels, IC95 bootstrap de la ganancia de EDP por nivel, prueba de Wilcoxon). Resultado sin cambios: no actuar en compute (22 kernels) ni en memory (28).
- **Kernels `nw` y `npb_ep`:** no se medirán; el conjunto de 30 familias medido se da por suficiente (decisión del autor). No se reentrena añadiendo kernels; pueden proponerse kernels solo como prueba del modelo congelado.
- **Tabla de política CPU, corrección (2026-09-21):** se excluyen las tres variantes de `phasic` (duración fija de 20.6 s en los diez niveles; su trabajo no es igual entre niveles, así que su EDP no es comparable). Quedan 19 kernels compute y 28 memory. Con ellos fuera, compute mejora 0.56 % en F0 (IC95 0.27 a 1.38 %, p = 0.008); F0 fija 3.2 GHz, el mismo reloj que REF con turbo desactivado, por lo que la diferencia viene de fijar el reloj frente al gobernador. **Decisión del autor:** efecto mínimo relevante de 1 % (declarado después de ver el resultado); por debajo de él la entrada es `no_actuar` con motivo `efecto_bajo_minimo`. Resultado: no actuar en ambas clases. `cpu_policy_table.py` recibe `--min-effect` (0.01) y registra `excluded_kernels`; `policy_cpu.json` con sobre versionado y IC95 por nivel.
- **Prueba externa del modelo congelado:** kernels XSBench y RSBench compilados en paccaA100 (commits fijos, doble precisión en el código) y catalogados; tamizaje en curso (job 7525). Descartados: miniBUDE (`float`, el instrumento cuenta solo FLOPs double), GAP BFS (entero puro, plan §2.1 punto 4) y NPB EP (ya descartado antes). `kmeans` y `particlefilter` no sirven como prueba: solo 4 % y 1.6 % de sus ventanas tienen FLOPs medidos. Alineación de hermanos SMT (cpu16 a cpu21 estaban fijados en 800 MHz) integrada en `hyp_external_cpu.sbatch`.

