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
| `F1-CPU-001` | Sustituir el supuesto contador de stalls de backend por `CYCLE_ACTIVITY.STALLS_MEM_ANY` | Implementado |
| `F1-GPU-001` | Usar Nsight Compute solo para construir la verdad Roofline offline y NVML como proxy ligero online | Parcialmente implementado (captura completa; entrenador GPU pendiente) |
| `F1-CPU-002` | Alinear el dataset de entrenamiento CPU con los intervalos reales de `uncore_imc` | Implementado (offline) |

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
