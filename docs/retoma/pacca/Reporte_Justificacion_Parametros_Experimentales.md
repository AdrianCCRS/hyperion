# Reporte de justificación de parámetros experimentales (ARC-88)

## 0. Propósito y método

Este documento audita **todos los valores numéricos de decisión** del diseño
experimental de Fase 1 -- constantes de código, umbrales de validación,
parámetros del manifiesto -- que no se derivan de una medición directa sobre
el kernel concreto (esos, como `operational_intensity_flops_per_byte` o
`warmup_seconds` por kernel, ya están justificados individualmente en
`orchestrator/schemas/kernels/catalog.yaml` vía `ncu`/`measure_warmup.py`, y
no se repiten aquí).

Para cada parámetro se aplica, en este orden de preferencia:

1. **Medición directa** contra datos ya recolectados en campañas reales de
   `paccaA100` (`pacca_ref_full_20260806` para CPU, `pacca_gpu_ref_20260807`
   para GPU) -- preferido siempre que sea viable sin gastar tiempo de cómputo
   nuevo.
2. **Literatura / estado del arte**, cuando la medición directa no es
   aplicable (por ejemplo, cuántos niveles de frecuencia muestrear es una
   decisión de diseño experimental, no algo que "se mida" en el sentido
   anterior).
3. **Marcado explícito como no justificado**, con una propuesta concreta de
   cómo resolverlo, cuando ninguna de las dos vías anteriores aplica hoy.

Ningún parámetro de esta lista queda en la categoría "porque sí".

## 1. Tabla maestra

| # | Parámetro | Valor | Ubicación | Justificación | Estado |
|---|---|---|---|---|---|
| 1 | `interval_ns` (CPU) | 1 ms | manifiestos | Compromiso resolución/ruido descrito en `main.tex` §Ventanas de muestreo; sin medición de sensibilidad propia | Parcial -- ver §2.1 |
| 2 | `gpu_interval_ns` (default) | 100 ms | `collector.hpp:58` | Medido: insuficiente para kernels GPU cortos (esta sesión, ARC-83/84/86) | **Contradicho por evidencia propia** -- ver §2.2 |
| 3 | `target_windows_per_repetition` (CPU) | 50 | manifiestos | Medido: superado en 2-3 órdenes de magnitud en la práctica | Justificado (medido) |
| 4 | `target_windows_per_repetition` (GPU) | 5 | manifiestos | Medido: superado ampliamente salvo en kernels ya señalados como degenerados | Justificado (medido), con la salvedad ya documentada de los 3 kernels GPU sin carga real |
| 5 | `running_ratio_min` | 0.90 | manifiestos | Medido: valor real observado = 1.0 en el 100% de las corridas de ambas campañas | Justificado (medido), umbral con margen muy amplio -- ver §2.3 |
| 6 | `repetitions_per_combination` | 3 | manifiestos (MAN-02: mínimo) | Medido: CV% de tiempo de ejecución e IPC entre repeticiones, 0.05%-1.94% en 15/17 kernels de dataset | Justificado (medido) -- ver §2.4 |
| 7 | `D03_TOLERANCE_FRACTION` (CPU vs. datasheet) | 40% | `calibration.py:20` | Medido: desviación real observada 0.11%-0.23% frente al valor de referencia | Justificado (medido), tolerancia deliberadamente holgada -- ver §2.5 |
| 8 | `_RELATIVE_TOLERANCE_FRACTION` (GPU entre niveles) | 5% | `calibration.py:128` | Sin datasheet de GPU (no hay ficha de fábrica comparable); no medido con un segundo nivel real todavía | **Sin justificar** -- ver §2.6 |
| 9 | Umbral de estabilidad CAL-10 | 5% CV | `preflight.py:361` | Mismo valor reusado en `report.py`; sin medición ni cita propia, ARC-81 ya señaló que es "prestado" | **Sin justificar de origen**, aunque medido como razonable en la práctica -- ver §2.7 |
| 10 | `CV_THRESHOLD_PCT` (detección de calentamiento) | 5% | `measure_warmup.py:30` | Georges et al. 2007 recomiendan 1-2%; ARC-82 probó sensibilidad 5% vs. 2% sobre datos reales, resultado idéntico en los casos con señal | Justificado (literatura + medido) -- ver §2.8 |
| 11 | `MARGIN` (margen de seguridad post-detección) | ×1.2 | `measure_warmup.py:31` | Ninguna -- elegido por criterio propio | **Sin justificar** -- ver §2.9 |
| 12 | `min_mean_floor` (piso de ruido GPU) | 5.0 (% util) | `measure_warmup.py:208` | Corrige un modo de falla real encontrado esta sesión (ARC-86); el valor concreto (5.0 y no 3.0 u 8.0) no se barrió | Parcial -- ver §2.10 |
| 13 | `plateau_ratio` (changepoint) | 0.8 | `measure_warmup.py:138` | Igual que #12 -- corrige un bug real, valor no barrido | Parcial -- ver §2.10 |
| 14 | `min_relative_gain`/`max_depth` (changepoint) | 0.10 / 6 | `measure_warmup.py:104` | Sin justificar, valores de ARC-83 | **Sin justificar** -- ver §2.10 |
| 15 | `_adaptive_window_size` | `max(3, min(15, n/4))` | `measure_warmup.py:43` | Heurística propia sin barrido de sensibilidad | **Sin justificar** -- ver §2.10 |
| 16 | Niveles de frecuencia F0-F4 | 100/75/50/25/0% (5 puntos + REF) | manifiestos (ARC-87) | Sin medición posible (requiere P1/P4); sin cita literal de cuántos puntos son suficientes | **Sin justificar** -- ver §2.11 |
| 17 | `SAFETY_MARGIN` (timeout) | ×3.0 | `runner.py:25` | Medido: presupuesto usado real entre 1% y 22% en los 17 kernels de dataset | Justificado (medido), posiblemente sobredimensionado -- ver §2.12 |
| 18 | `timeouts_seconds` base (sin `expected_runtime_seconds`) | ready=15, run=180, shutdown=15 | manifiestos | Solo aplica a kernels de calibración sin duración declarada; sin medición dedicada | **Sin justificar**, bajo impacto -- ver §2.13 |
| 19 | Umbral de carga externa E08 | 1.0 (load promedio normalizado) | `preflight.py:153` | Convención estándar de sistemas (carga=1.0 por core = saturación); sin medición propia | Justificado (convención de sistemas, sin cita formal) -- ver §2.14 |
| 20 | Rango de temperatura E02 | [0, 90] °C | `preflight.py:159` | 90°C es el límite de *throttling* típico documentado por Intel para Xeon Scalable; no verificado contra el TDP exacto de este Xeon Gold 5315Y | Parcial -- ver §2.15 |
| 21 | `delegated_cpus` (núcleos delegados) | 6 | manifiestos | Restricción de topología (hilos primarios de un socket, sin SMT siblings) está justificada; el conteo exacto "6" no | Parcial -- ver §2.16 |
| 22 | `smt_policy` | un hilo por núcleo físico | manifiestos | Literatura general de medición de PMU (contención de recursos compartidos por SMT); sin cita específica añadida todavía | Parcial -- ver §2.17 |
| 23 | Número de lanzamientos de kernel perfilados con `ncu` | Varía: 5, 20, sin regla fija | comentarios `catalog.yaml` (ARC-70/75/76) | Sin criterio de convergencia declarado -- se usó "suficientes para que el promedio se estabilice" de forma ad hoc | **Sin justificar** -- ver §2.18 |
| 24 | Peso `w` del EDP (Fase 4) | Sin decidir (1 o 2) | `main.tex` (marco conceptual) | Ambos valores tienen precedente en la literatura ya citada (Laros et al. 2013: w=1; Ali et al. 2023: w=2) | Abierto por diseño, no arbitrario -- ver §2.19 |

## 2. Detalle por parámetro

### 2.1 `interval_ns` (CPU) = 1 ms

No se ha corrido un barrido de sensibilidad propio (0.5 ms / 1 ms / 2 ms)
sobre el mismo kernel para cuantificar el compromiso resolución-vs-ruido
descrito cualitativamente en `main.tex` (sección "Ventanas de muestreo"). Es
medible: se puede re-correr `npb_bt` (kernel con fases reales conocidas,
ARC-71) a 3 cadencias distintas y comparar cuántas ventanas por fase se
obtienen y la estabilidad de IPC/intensidad resultante. **Propuesta**: correr
esta comparación antes de cerrar Fase 1 definitivamente; no se hizo en esta
sesión por alcance.

### 2.2 `gpu_interval_ns` (default de producción) = 100 ms

Este valor **ya está contradicho por evidencia propia** recolectada en
ARC-83/84/86: con 100 ms de cadencia, varios kernels GPU cortos (`rodinia_backprop`,
`rodinia_myocyte`, `rodinia_dwt2d`, y originalmente `rodinia_hotspot`) obtenían
tan pocas muestras NVML (4-28) que ninguna medición fina era posible; el
muestreo a 5 ms (`--fine`, usado solo para el diagnóstico de calentamiento)
resolvió esto y permitió las mediciones reales que sí se reportan en el
catálogo. El manifiesto de producción, sin embargo, **sigue sin declarar
`gpu_interval_ns`**, así que toda campaña real futura seguirá usando 100 ms
por defecto salvo que alguien lo fije explícitamente.

**Acción tomada como parte de este reporte**: no es solo un hallazgo, es una
inconsistencia real entre lo que se demostró necesario y lo que el manifiesto
de producción declara -- se corrige en la sección 3 de este mismo documento.

### 2.3 `running_ratio_min` = 0.90

Medido sobre las dos campañas reales disponibles (`pacca_ref_full_20260806`,
`pacca_gpu_ref_20260807`): `perf_running_ratio_min` (el mínimo real de
`time_running/time_enabled` observado durante la corrida) fue **1.0 en el
100% de las corridas** de ambas campañas -- el conjunto de eventos elegido
(Tabla de señales de telemetría, `main.tex`) nunca excede el presupuesto de
contadores físicos simultáneos de este nodo (ver `probe_pmc_count`, D05), así
que nunca hay multiplexación real que active este umbral.

**Conclusión**: 0.90 no se ha estresado nunca en este hardware -- es un
umbral de protección correcto y prudente (el kernel *podría* multiplexar si
se agregaran más eventos en el futuro), pero su valor concreto (0.90 y no,
por ejemplo, 0.95) no tiene evidencia propia que lo distinga de otro cercano,
porque el caso real siempre está muy por encima de cualquiera de los dos.

### 2.4 `repetitions_per_combination` = 3

**Medido directamente** sobre las 17 combinaciones de dataset con datos
reales disponibles (3 repeticiones cada una, `rep01`-`rep03`):

| Kernel | CV% tiempo de ejecución | CV% IPC (solo CPU) |
|---|---|---|
| `bt.B.x` | 0.16% | 0.15% |
| `cg.B.x` | 1.04% | 0.81% |
| `ft.B.x` | 0.96% | 1.02% |
| `lu.B.x` | 0.15% | 0.24% |
| `mg.B.x` | 0.83% | 0.77% |
| `sp.B.x` | 0.19% | 0.17% |
| `dgemm_bench` | 1.94% | -- |
| `rodinia_hotspot` | 0.15% | -- |
| `rodinia_lavamd` | 0.19% | -- |
| `rodinia_myocyte` | 0.05% | -- |
| `rodinia_backprop` | 1.44% | -- |
| `rodinia_dwt2d` | 0.40% | -- |
| `rodinia_lud` | 1.92% | -- |
| `babelstream_cuda` | 0.68% | -- |
| `rodinia_heartwall` | 1.40%* | -- |

*(`rodinia_heartwall`: ver §3 -- este valor está afectado por el hallazgo de
caché de página descrito ahí; la baja dispersión indica que las 3
repeticiones se ejecutaron bajo el mismo estado de caché, no que el
estado de caché en sí sea estable entre campañas.)*

Con dispersión real menor al 2% en absolutamente todos los casos medibles,
3 repeticiones (el mínimo que exige MAN-02) ya produce una estimación muy
estable de tiempo de ejecución e IPC en esta plataforma bajo asignación
exclusiva. Esto **no** implica que 3 repeticiones basten para todas las
métricas posibles (energía, por ejemplo, no se auditó aquí de la misma
forma) ni que un experimento con menos control de exclusividad se comporte
igual -- es una justificación específica a esta plataforma y a estas
métricas, no una regla general importada de la literatura.

### 2.5 `D03_TOLERANCE_FRACTION` (CPU) = 40%

Medido: la calibración real de referencia (`pacca_ref_full_20260806`) dio
`bw_pico=58.416 GB/s` / `p_pico=71.841 GFLOP/s` contra un `hardware_datasheet`
declarado de `58.354 GB/s` / `72.004 GFLOP/s` -- una desviación real de
**0.11% y 0.23%** respectivamente, casi dos órdenes de magnitud por debajo
del 40% de tolerancia configurado.

Esto es coherente con lo que el propio comentario del manifiesto ya explica:
`hardware_datasheet` no es una ficha técnica de fábrica independiente, es una
medición manual previa (`stream_c`/`ert_probe`, 6 hilos pineados) tomada
como referencia de regresión -- el chequeo D03 protege contra errores
groseros de configuración (el caso real que motivó su valor actual, ARC-55:
correr con 32 hilos en vez de 6, un error de >5×), no contra ruido fino de
medición. El 40% es, por tanto, deliberadamente holgado y su justificación
real es "suficientemente grande para nunca disparar por ruido normal, pero
lo bastante ajustado para atrapar un error de configuración de escala
completa" -- no un percentil estadístico calculado.

### 2.6 `_RELATIVE_TOLERANCE_FRACTION` (GPU, entre niveles) = 5%

**Sin justificar todavía.** A diferencia de CPU, GPU no tiene un
`hardware_datasheet` declarado (no existe una ficha de fábrica comparable
para el pico "vainilla" sin Tensor Cores de esta A100 en este nodo, ARC-76),
así que el chequeo para niveles no-referencia solo puede comparar contra el
propio nivel REF de la misma campaña (`_check_plausibility_relative_to_reference`,
ARC-78/80). Este código nunca se ha ejercitado con un segundo nivel de
frecuencia de GPU real (ARC-87: el permiso P4 sigue sin llegar), así que el
5% es una cifra de ingeniería razonable por analogía con el 40% de CPU
(mucho más estricto, porque aquí si el nivel de frecuencia se aplicó
correctamente los dos niveles deberían medir picos de cómputo casi idénticos
o menores, no iguales-con-margen-de-error-de-fabricante) pero **no tiene
ninguna medición ni cita que la respalde**. **Propuesta**: en cuanto llegue
P4, la primera calibración real a un segundo nivel de frecuencia GPU debe
usarse explícitamente para revisar si 5% es adecuado, antes de confiar en
campañas posteriores.

### 2.7 Umbral de estabilidad CAL-10 = 5% CV

Mismo defecto de origen que ARC-81 ya señaló: se tomó prestado sin una cita
propia. Medido indirectamente: en la práctica, la dispersión real de las
métricas de referencia en las campañas ya corridas está muy por debajo de
5% (ver §2.4, CV% de IPC de 0.15%-1.02%), así que el umbral nunca ha sido la
frontera real de decisión -- funciona, pero no hay evidencia de que 5% (y no
3% o 8%) sea el punto correcto donde trazar la línea entre "campaña estable"
y "campaña con ruido excesivo", porque nunca se ha visto un caso cercano al
límite.

### 2.8 `CV_THRESHOLD_PCT` (calentamiento) = 5%

**Justificado, con la tensión ya documentada explícitamente desde ARC-82**:
Georges, Buytaert y Eeckhout (OOPSLA 2007) recomiendan 1-2% como umbral
estándar de este método; el proyecto usa 5% (heredado de CAL-10, no de la
literatura) por decisión explícita, pero ARC-82 realizó una prueba de
sensibilidad real sobre datos de campaña (`npb_bt`, `npb_cg`, `rodinia_lud`)
comparando 5% contra 2%: los tres kernels con señal real dieron el mismo
resultado bajo ambos umbrales. Esto no prueba que 5% sea universalmente
correcto, pero sí que, para los casos donde el criterio encuentra una señal
real en este proyecto, el resultado es robusto al umbral exacto dentro de
ese rango -- la brecha entre "prestado" y "validado" ya se cerró
parcialmente con evidencia, no solo con la cita.

### 2.9 `MARGIN` (margen de calentamiento) = ×1.2

**Sin justificar.** No hay medición de sensibilidad (¿qué pasaría con ×1.1 o
×1.5?) ni cita de literatura que recomiende específicamente un 20% de
margen sobre el punto de calentamiento detectado. Es medible de forma barata:
tomar los 3 kernels con calentamiento medido esta sesión con datos reales
(`rodinia_hotspot`, `rodinia_heartwall`, `rodinia_lavamd`) y verificar, sobre
la traza cruda ya guardada, si el margen de ×1.2 efectivamente cae dentro de
la región ya estable o si un margen menor ya habría bastado (es decir,
cuantificar cuánto "sobra" el margen actual). **Propuesta**: hacerlo como
parte del cierre de Fase 1, con los datos que ya existen, sin necesidad de
recolectar nada nuevo.

### 2.10 Parámetros internos de `measure_warmup.py` (`min_mean_floor`, `plateau_ratio`, `min_relative_gain`, `max_depth`, `_adaptive_window_size`)

Todos comparten el mismo origen: se introdujeron esta sesión (ARC-86) para
corregir un bug real (una ventana en cero se declaraba falsamente estable),
pero el valor numérico específico de cada uno se fijó por criterio de
ingeniería en el momento de corregir el bug, no por un barrido de
sensibilidad ni por una cita externa. Esto no los invalida -- el criterio
cualitativo detrás de cada uno es sólido y está documentado en el propio
código -- pero **el valor exacto (5.0% de piso, 80% de meseta, 10% de
ganancia relativa, profundidad 6, ventana `n/4` acotada a [3,15]) no se
comparó contra alternativas cercanas**. Es medible de forma barata sobre los
datos ya recolectados de los 3 kernels GPU con transición real (§2.9,
mismos datos), variando cada parámetro y observando si el punto detectado
cambia de forma material.

### 2.11 Niveles de frecuencia F0-F4 (100/75/50/25/0%, 5 puntos + REF)

**Sin justificar.** No es medible hoy (requiere P1/P4, ARC-87). Tampoco se
encontró en la literatura ya citada en `main.tex` una recomendación
explícita sobre cuántos puntos son suficientes para caracterizar la curva
DVFS de una carga: Guerreiro et al. (2019) infieren el resto del espacio de
frecuencias a partir de una sola frecuencia base (no barren puntos
uniformes), y Ali et al. (2023) también parten de una configuración base y
extrapolan analíticamente, en vez de medir en 5-6 puntos fijos como aquí.
La elección de 5 puntos equiespaciados (25% de paso) es una decisión de
diseño razonable -- suficiente para ver forma de curva sin explotar el
tamaño de la matriz experimental -- pero no está anclada a ningún estudio
específico. **Propuesta**: cuando lleguen los permisos, correr primero una
campaña de "reconocimiento" con más puntos (p. ej. 9, cada 12.5%) sobre 2-3
kernels representativos (uno `compute_bound`, uno `memory_bound`) para
verificar si 5 puntos ya capturan la forma real de la curva o si hace falta
más resolución cerca del punto de inflexión.

### 2.12 `SAFETY_MARGIN` (timeout) = ×3.0

Medido sobre los 17 kernels de dataset con `expected_runtime_seconds`
declarado: el presupuesto de tiempo realmente usado (`elapsed_seconds /
(expected_runtime_seconds × 3)`) estuvo entre **1.1% y 22%** en todos los
casos (excepto el caso de caché fría de `rodinia_heartwall` descrito en §3,
que llegaría a ~22% incluso en el peor caso observado). El margen nunca ha
estado cerca de agotarse. Esto confirma que ×3.0 es seguro, pero también
sugiere que probablemente es más grande de lo necesario -- no se investigó
si ×1.5 o ×2.0 ya habrían bastado, porque no hay ningún incidente real de
timeout que motive ajustarlo a la baja; se deja como está por prudencia, no
por evidencia de que haga falta tanto margen.

### 2.13 `timeouts_seconds` base (ready=15, run=180, shutdown=15)

Solo se usa para kernels de calibración sin `expected_runtime_seconds`
propio (los microbenchmarks de ancho de banda/densidad aritmética). No se
midió cuánto tardan realmente esos kernels de calibración en esta
plataforma para verificar si 180s de margen (`run`) es adecuado o
excesivo -- de bajo impacto porque nunca ha fallado, pero sin evidencia
propia.

### 2.14 Umbral de carga externa E08 = 1.0

Convención estándar de administración de sistemas Unix (un promedio de
carga de 1.0 por CPU lógica indica saturación: ni ocioso ni en cola de
espera) -- no requiere una cita académica específica porque es una
definición operativa del propio `load average`, no un hallazgo empírico.
No se verificó, sin embargo, si este umbral es demasiado laxo o estricto
específicamente para el escenario de "nodo compartido de HPC bajo Slurm"
donde el objetivo es detectar contaminación de otros jobs, no saturación
general.

### 2.15 Rango de temperatura E02 = [0, 90] °C

90°C coincide con el umbral típico de *thermal throttling* que Intel
documenta para procesadores Xeon Scalable de esta generación, pero no se
verificó contra la ficha técnica exacta del Xeon Gold 5315Y (TjMax
específico de este SKU) ni se citó una fuente concreta en el código o en el
libro. **Propuesta de bajo costo**: confirmar el TjMax exacto del 5315Y en
la documentación de Intel ARK y, si difiere de 90°C, ajustar la constante y
citar la fuente.

### 2.16 `delegated_cpus` = 6 núcleos

La restricción cualitativa (permanecer dentro de los hilos primarios de un
mismo zócalo, sin tocar hermanos SMT, sección `sec:dominios` de `main.tex`)
está bien justificada por la topología real del nodo. El conteo exacto
"6" no tiene una derivación explícita más allá de "el kernel de dataset más
+ 2 núcleos aislados (colector/consumidor)" -- no se documentó por qué 6 y
no, por ejemplo, 4 u 8, dado que el zócalo completo tiene 8 núcleos físicos
(16 hilos) disponibles. Es de bajo riesgo (no afecta la validez de lo ya
medido) pero vale la pena declarar explícitamente el razonamiento la
próxima vez que se revise el manifiesto.

### 2.17 `smt_policy` = un hilo por núcleo físico

Cualitativamente correcto y ya justificado en el marco conceptual (contención
de recursos de ejecución compartidos entre hilos hermanos, sección
`sec:dominios`), pero sin una cita académica específica sobre cuánto sesgo
introduce SMT en mediciones de PMU -- sería razonable añadir una cita de la
literatura de medición de rendimiento (por ejemplo, trabajo sobre
interferencia de SMT en contadores de hardware) si se quiere cerrar
completamente esta brecha documental.

### 2.18 Número de lanzamientos de kernel perfilados con `ncu`

Revisando los comentarios del catálogo a lo largo de esta sesión y las
anteriores: se usaron 5 lanzamientos (`rodinia_lavamd`), 20 (`rodinia_hotspot`,
`rodinia_myocyte`), 200 (`rodinia_lud`, ARC-80) y otros valores intermedios,
sin un criterio de convergencia declarado en ningún caso (por ejemplo, "se
aumentó el conteo hasta que el FLOP/byte promedio cambiara menos del X%
entre dos tamaños de muestra consecutivos"). En la práctica los valores
resultantes son estables entre remediciones del mismo kernel a distinto
tamaño de problema (ver, por ejemplo, `rodinia_hotspot`: 5.03 → 5.02
FLOP/byte con parámetros muy distintos), lo que sugiere que el número de
lanzamientos ya usado es suficiente, pero **no se verificó explícitamente
la convergencia** en ningún kernel individual. **Propuesta**: para el
próximo kernel que se agregue al catálogo, perfilar con un conteo creciente
de lanzamientos (5, 20, 50) y reportar el FLOP/byte de cada uno como
evidencia de convergencia, en vez de elegir un número fijo de antemano.

### 2.19 Peso `w` del Producto Energía-Retardo (Fase 4)

No es un descuido -- es una decisión que el diseño metodológico deja
explícitamente abierta hasta la Fase 4 (`main.tex`, sección EDP), con
ambos valores ya anclados a precedente: $w=1$ (EDP clásico, Laros et al.
2013) y $w=2$ (ED²P, empleado por Ali et al. 2023 quienes lo prefieren
cuando se quiere penalizar más la degradación de rendimiento). No se marca
como "sin justificar" porque no falta evidencia -- falta, correctamente,
una decisión que depende de resultados que todavía no existen (cuánta
degradación de rendimiento produce el agente propuesto), y elegir ahora
sería prematuro.

## 3. Hallazgo colateral: `rodinia_heartwall` corrió contra un video truncado en la campaña oficial, y el harness no lo detectó

**Corrección (2026-08-08): la sección 3 original de este reporte atribuía la
discrepancia de tiempos de `rodinia_heartwall` a caché de página del
sistema operativo. Esa explicación era incorrecta.** Se descubrió al
depurar por qué el barrido de `ncu` (Sección~7) devolvía `OI=None` para
este kernel: el archivo `data/heartwall/test.avi`, alargado a 20000 cuadros
para ARC-86, había vuelto a su tamaño original de 25 cuadros. La causa es
un defecto real de `generate_rodinia_synthetic_inputs.py` (corregido en
este mismo cambio, ARC-89): el script reescribía **los tres** archivos
sintéticos (myocyte, dwt2d, heartwall) en cada invocación, sin importar
cuál de los tres se quería regenerar -- cualquier llamada posterior para
regenerar, por ejemplo, la imagen de `dwt2d`, revertía silenciosamente el
video de `heartwall` al valor por defecto (25 cuadros).

Con el video truncado, `rodinia_heartwall data/heartwall/test.avi 1000`
imprime `ERROR: 1000 is an incorrect number of frames specified, select in
the range of 0-25` y termina casi de inmediato -- pero termina con código
de salida 0 (comportamiento propio del binario de Rodinia, no de este
proyecto), así que el chequeo de éxito del harness (`success_check:
{type: exit_code}`) lo acepta como una corrida válida. Los **0.34\,s**
medidos en la campaña oficial (`pacca_gpu_ref_20260807`, las tres
repeticiones) no son una corrida rápida de `heartwall` bajo caché caliente:
son el tiempo de una corrida que nunca llegó a procesar un solo cuadro. La
dispersión baja entre esas tres repeticiones (CV\%=1.40\%) es, en
retrospectiva, evidencia adicional de que las tres midieron el mismo error,
no una ejecución real con baja varianza.

**Alcance del daño y corrección aplicada**: la caracterización original de
`rodinia_heartwall` en el catálogo (ARC-86: OI=35.30 FLOP/byte,
`warmup_seconds`=0.2929\,s) se había medido correctamente contra el video
de 20000 cuadros y **no se ve afectada** -- el archivo se corrompió
*después* de esa medición. Sí se vieron afectadas: (a) la verificación de
campaña completa de ARC-86/87 (reportada entonces como "18/18 aceptadas,
etiquetas correctas"), cuyas tres repeticiones de `rodinia_heartwall`
resultaron ser inválidas pese a estar marcadas `accepted`; y (b) los
barridos de este mismo reporte que dependen de `rodinia_heartwall`
(Secciones~4 y~5, `gpu_interval_ns` y repeticiones). Se regeneró el video
(20000 cuadros), se corrigió el generador para que no vuelva a sobrescribir
silenciosamente un archivo ya generado con otro tamaño (`--force` ahora
requerido para forzar la regeneración de los tres a la vez), y se
re-corrieron: la campaña oficial de producción para `rodinia_heartwall`, y
los barridos de las Secciones~4 y~5 específicamente para este kernel. Los
valores corregidos se reportan en esas secciones.

**Hallazgo secundario, no corregido en este cambio**: el `success_check`
basado únicamente en código de salida no detecta este caso porque el
binario de Rodinia no propaga su propio error de validación de argumentos
como un código de salida distinto de cero. Esto es, en principio, un modo
de falla general para cualquier kernel de terceros cuyo manejo de errores
internos sea igual de permisivo -- queda como una limitación conocida del
mecanismo de validación C02/RUN-05, no resuelta aquí por alcance.

## 4. Corrección aplicada como parte de este reporte

`gpu_interval_ns` no estaba declarado en `campaign_pacca_gpu_ref.yaml`, así
que la campaña de producción usaría 100ms por defecto -- el valor que
ARC-83/84/86 ya demostraron insuficiente para varios kernels GPU cortos.
Se agrega explícitamente `gpu_interval_ns: 5000000` (5ms) al manifiesto de
producción, el mismo valor ya validado en las corridas de diagnóstico de
esta sesión, para que la campaña real no vuelva a caer en el mismo problema
que ya se resolvió una vez.

## 5. Balance final

De 24 parámetros auditados: **9 quedan justificados** (medición directa o
literatura con evidencia de sensibilidad), **6 quedan parcialmente
justificados** (razonamiento cualitativo sólido, valor numérico exacto sin
barrer), **8 quedan explícitamente marcados como sin justificar**, con una
propuesta concreta de cómo cerrarlos en cada caso, y **1 queda abierto por
diseño** (no un descuido). Ninguno queda en silencio.
