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
| `F1-XDEV-001` | Tratar las 232 entradas como banco de candidatos y seleccionar por cobertura Roofline/familia | En preparación (manifiestos de cribado creados) |
| `F2-XDEV-001` | Diagnosticar cobertura Roofline y calidad antes de seleccionar/balancear entrenamiento | Implementado (sin datos de campaña aún) |

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
`training_quality_status="ok"` de `training_cpu_intervals.csv`. En GPU lee
`windows.csv` para describir las muestras NVML y usa `gpu_freq_level_id`, pero
marca el reporte como `eligible_for_training_without_additional_aggregation:
false`: una muestra periódica NVML no equivale a una fase independiente.

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
5. Implementar la agregación GPU por corrida o fase estable antes de crear el
   entrenador GPU o interpretar sus muestras NVML como filas ML.
