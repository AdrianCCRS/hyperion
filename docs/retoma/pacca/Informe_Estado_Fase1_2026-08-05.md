# Informe de estado — Fase 1 (Medición DVFS/Roofline), hasta 2026-08-05

**Propósito de este documento:** insumo para redactar el libro/tesis del proyecto (destinado a ser procesado por otra IA como base de escritura). Resume qué se construyó, qué se validó, qué se encontró, y qué sigue abierto en la Fase 1 (instrumentación de medición de energía/rendimiento y clasificación Roofline bajo distintos estados de frecuencia).

**Restricción deliberada de alcance:** este informe describe el *pipeline y la metodología*, no un nodo de cómputo específico. El primer nodo real donde se ejecutó (nombre en clave "felix", clúster SC3) sirvió para primer contacto con hardware real y validación de la arquitectura del pipeline, pero **no es el nodo final del proyecto** y tiene limitaciones de hardware propias (ausencia de RAPL, dominio de frecuencia por socket) que no deben generalizarse como propiedades del proyecto ni de la Fase 1. Cualquier afirmación aquí sobre "funciona en hardware real" se refiere a la validez del pipeline como tal, verificada en al menos un nodo real, no a una propiedad garantizada de un nodo particular.

---

## 1. Objetivo de la Fase 1

Construir un instrumento de medición reproducible que, para un conjunto de kernels de cómputo representativos, capture rendimiento (instrucciones, ciclos, accesos a memoria) y — donde el hardware lo permita — energía, bajo distintos estados de frecuencia de CPU controlados explícitamente, y clasifique cada corrida en el modelo Roofline (`compute_bound` vs `memory_bound`) a partir de la intensidad operacional medida. El resultado es la base empírica sobre la que se apoyará el resto de la tesis (relación DVFS–eficiencia–arquitectura).

---

## 2. Qué se construyó

### 2.1 Subsistema de telemetría (C++)

Un harness (`telemetry_kernel_launcher`) que lanza un kernel objetivo como proceso hijo y adjunta contadores de hardware (`perf_event_open`) sobre su PID con herencia (`inherit=1`), evitando cualquier dependencia de un cgroup dedicado (la migración fuera de un diseño basado en cgroup fue una decisión deliberada de portabilidad — ver sección 3 de `Guia_Maestra_Fase1_DVFS.md` — precisamente para no atarse a la configuración de aislamiento de un clúster específico, algo que resultó acertado al encontrar dos clústeres con jerarquías de cgroup incompatibles entre sí, v1 vs v2). Soporta también un modo `--exec` para envolver binarios de suites externas (NPB, STREAM) sin modificarlos.

### 2.2 Orquestador (Python)

Conjunto de módulos independientes, cada uno con reglas de calidad numeradas y verificadas por tests: manifiesto de campaña, catálogo de kernels, detección de capacidades del entorno, verificaciones de preflight (E01–E10 y otras), control de frecuencia (`freqctl.py`, con dos estrategias — `discrete_bounds` y `bounded_range` — seleccionadas automáticamente según el driver de cpufreq detectado, sin intervención manual), ejecución de corridas individuales, orquestación de campañas completas, post-procesamiento de muestras crudas a ventanas de análisis, validación de calidad de datos, y generación de reportes. El diseño explícitamente anticipa portabilidad entre nodos con hardware distinto (detección de driver de frecuencia, presencia/ausencia de RAPL, número de contadores simultáneos disponibles, etc., todo detectado en tiempo de ejecución, no hardcodeado).

Al día de este informe: **139 reglas de diseño documentadas** en la guía maestra, de las cuales el checklist técnico reporta **119/124 verificadas (96%)** contra la primera ejecución real completa.

### 2.3 Catálogo de kernels

Siete kernels: seis de la suite NAS Parallel Benchmarks (NPB) cubriendo distintos perfiles de intensidad operacional, más un DGEMM (BLAS) como referencia de kernel fuertemente compute-bound. El catálogo es una capa de metadatos sobre binarios de terceros, no código de cómputo propio — decisión deliberada para no introducir sesgos de implementación propios en la clasificación Roofline.

### 2.4 Calibración de referencia

Metodología de calibración (`calibration_references`) que ejecuta un kernel de referencia varias repeticiones y usa el percentil 95 de cuatro métricas (IPC, IPS, MPKI, tasa de fallos de caché) para fijar un umbral de estabilidad (`cv_pct`, coeficiente de variación) como gate de calidad no bloqueante. Este mecanismo fue puesto a prueba y ajustado durante la primera campaña real (ver sección 4).

---

## 3. Validación en hardware real: qué se probó y qué resultó

La primera ejecución completa de campaña real (7 kernels × 3 repeticiones = 21 corridas) se ejecutó de punta a punta sobre hardware físico, con el harness adjuntando contadores reales de CPU. Resultado: **21/21 corridas aceptadas**, con un total de ~305.554 filas de muestras crudas procesadas, y una clasificación Roofline (`phase_label_train`) coherente con la expectativa declarada por catálogo para cada kernel (`phase_label_hint`) en todos los casos evaluados. El coeficiente de variación de la calibración de referencia terminó en 0.88%, muy por debajo del umbral de alerta de 5%.

Esto demuestra que el pipeline completo — desde el lanzamiento del kernel hasta la clasificación Roofline final — es funcionalmente correcto de extremo a extremo bajo condiciones reales de ejecución, no solo en pruebas unitarias sintéticas.

**Importante:** esta campaña se ejecutó con un único estado de frecuencia (la frecuencia de referencia, REF — el estado por defecto del nodo, sin control activo de DVFS). La matriz experimental completa con múltiples estados de frecuencia controlados **no se ha ejecutado en ningún nodo todavía**, porque el permiso de escritura sobre los registros de control de frecuencia de CPU no ha sido otorgado por ninguna administración de clúster contactada hasta la fecha. Este es el riesgo de cronograma más importante de la Fase 1 (ver sección 6).

---

## 4. Bugs reales encontrados y corregidos durante el primer contacto con hardware real

El valor de haber ejecutado contra hardware real (en vez de quedarse en simulación/tests unitarios) quedó demostrado por la cantidad y naturaleza de los problemas que solo aparecieron ahí:

1. **Inestabilidad del kernel de referencia de calibración**: el kernel originalmente elegido como referencia tenía un conteo de fallos de caché casi nulo (cientos a pocos miles de eventos totales), lo que hacía que dos de las cuatro métricas de calibración (MPKI, tasa de fallos) fueran estadísticamente inestables por tener muy pocos eventos de base, aunque las otras dos (IPC, IPS) fueran perfectamente estables. Diagnóstico preciso vía descomposición de CV por métrica; solución operativa (cambio de kernel de referencia a uno con actividad de caché sustancial), sin cambios de código.
2. **Bug crítico de post-procesamiento**: se descubrió que, para toda repetición de campaña distinta de la primera, el archivo de ventanas de análisis salía completamente vacío — afectando el 100% de las repeticiones 2 y 3 de los 7 kernels (14 de 21 corridas) en la primera ejecución real. Causa raíz: el post-procesador filtraba las muestras crudas usando el número de repetición *de campaña* (1, 2, 3…), pero el harness C++ numera sus propias repeticiones internamente siempre desde 1 en cada invocación de proceso — dos esquemas de numeración con el mismo nombre de campo pero significados distintos, que el código conflacionaba silenciosamente. El bug no producía ningún error visible; simplemente descartaba datos sin avisar, y un resumen superficial ("21/21 aceptadas") lo habría dejado pasar. Se corrigió desacoplando ambos esquemas de numeración, se agregó una prueba de regresión específica, y se volvió a ejecutar la campaña completa para confirmar datos completos.
3. Otros ajustes menores de manejo de unidades nativas del stdout de kernels externos, y de fuente correcta de datos de procesador/estado de CPU para uno de los checks de preflight (E06), corregidos durante el mismo ciclo de validación.

El patrón general — bugs silenciosos que una revisión superficial de "todo aceptado" no habría detectado, encontrados solo por inspección directa de los datos crudos — es evidencia de la importancia de la validación empírica sobre hardware real como parte formal del proceso, no un paso opcional posterior al desarrollo.

---

## 5. Estado del checklist técnico

119 de 124 ítems del checklist verificados contra la primera ejecución real completa (96%). Los ítems pendientes están relacionados con la matriz multi-frecuencia (bloqueada por permisos, no por diseño) y con la validación cruzada de un metodo de medición secundario (ver sección 6).

---

## 6. Brechas y riesgos abiertos

1. **Sesgo no corregido en la medición de bytes movidos.** La métrica `bytes_moved_window`, base del cálculo de intensidad operacional y por lo tanto de la clasificación Roofline, se calcula a partir de un contador genérico de fallos de caché multiplicado por el tamaño de línea de caché — un proxy, no una medición directa de tráfico de memoria (que requeriría contadores de uncore/memory controller). Una validación anterior (sobre un kernel de ancho de banda puro, STREAM) mostró que este proxy subestima el tráfico real en aproximadamente 30–34%. **Este sesgo nunca se corrigió ni se validó de forma cruzada con contadores de uncore en ningún nodo**, porque el acceso a esos contadores requiere un permiso que ninguna administración de clúster contactada ha otorgado todavía. Que la clasificación Roofline haya salido coherente con lo esperado en la campaña real no prueba que el sesgo haya desaparecido — kernels cerca del punto de inflexión (ridge point) del modelo Roofline son los más sensibles a este error y podrían clasificarse incorrectamente sin que sea evidente a simple vista.
2. **Matriz DVFS multi-frecuencia sin ejecutar en ningún nodo.** El control de frecuencia está implementado y probado en aislamiento, pero requiere un permiso de escritura sobre los registros de control de frecuencia de CPU que, hasta la fecha, ningún administrador de clúster contactado ha otorgado. Sin esto, no existe todavía ningún dato real de la variable central de la tesis (el efecto de la frecuencia sobre eficiencia/rendimiento) — solo datos de un único estado de frecuencia de referencia. Este es, con diferencia, el mayor riesgo de cronograma de la Fase 1: toda la campaña experimental completa depende de este permiso, independientemente de en qué nodo termine ejecutándose.
3. **Energía real medida solo parcialmente.** El harness soporta una vía de telemetría de energía (`ENERGY`), pero su disponibilidad depende de que el nodo objetivo tenga la tecnología de medición de energía correspondiente (p. ej. RAPL) y de que sea accesible sin privilegios especiales — algo que varía por hardware y no puede asumirse resuelto de forma genérica hasta confirmarse en el nodo final.

---

## 7. Recomendación dado el cronograma (3 semanas para cierre del proyecto)

La prioridad inmediata y crítica es **obtener el permiso de escritura de frecuencia de CPU** en el nodo donde efectivamente se correrá la campaña final — sin esto, la Fase 1 no puede considerarse cerrada porque falta el eje central del experimento (DVFS). Todo lo demás (kernels, orquestador, clasificación Roofline, calibración) ya está validado de extremo a extremo contra hardware real y no requiere más trabajo de ingeniería para arrancar una campaña completa en cuanto el permiso llegue. El sesgo de bytes movidos (punto 1 de la sección 6) es importante pero no bloqueante para producir resultados — debe documentarse explícitamente como limitación conocida en cualquier resultado que se publique mientras no se resuelva.

---

## 8. Qué NO afirma este informe

Para mantener la separación pedida entre metodología y nodo: este informe no afirma en qué nodo específico correrá la campaña final, ni asume que las características de hardware del primer nodo de prueba (ausencia de RAPL, dominio de frecuencia por socket, `perf_event_paranoid=1`, etc.) sean representativas de ningún otro nodo. Los hechos de arquitectura y permisos de cualquier nodo candidato deben documentarse por separado, nodo por nodo, en su propio informe de auditoría (siguiendo el mismo formato usado para el primer nodo de prueba).
