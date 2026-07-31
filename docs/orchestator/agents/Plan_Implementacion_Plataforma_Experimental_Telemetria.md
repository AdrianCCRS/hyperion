Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

# **PLAN DE IMPLEMENTACIÓN** 

## **Plataforma Experimental para la Recolección de Telemetría — Fase 1** 

_Agente en Espacio de Usuario para la Gestión Dinámica de Frecuencia (DVFS) en Sistemas Heterogéneos mediante Modelos Ligeros de Machine Learning_ 

Universidad Industrial de Santander Escuela de Ingeniería de Sistemas e Informática Clúster SC3 — Semillero de investigación 

_Documento de trabajo interno — versión consolidada (incluye la estrategia multinodo: generalización, portabilidad y ruta de decisión con el director)_ 

Página 1 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

## **1. Propósito y punto de partida** 

Este documento define el plan de implementación de la plataforma experimental que se usará en la Fase 1 (Recolección de información y caracterización de cargas) del trabajo de grado. Su objetivo es dejar establecido, antes de tomar cualquier muestra en un nodo real del clúster SC3, qué se va a medir, en qué orden, bajo qué controles de seguridad y con qué criterios de aceptación o rechazo, de modo que el dataset resultante sea utilizable para entrenar el clasificador compute-bound / memory-bound sin introducir sesgos ni artefactos de medición. 

El punto de partida es el harness C++17 ya existente (telemetry_kernel_launcher, 

telemetry_kernel_workload, lectores de perf/RAPL/NVML). Ese harness resuelve la mecánica de bajo nivel de la toma de datos. Este plan define la capa de proceso, disciplina experimental y verificación que debe envolverlo antes de que una corrida pueda llamarse "dato de dataset" y no "smoke de control". 

**Atención:** Los kernels de carga de trabajo no se programan dentro del proyecto. Se usan binarios precompilados de suites de benchmarking reconocidas (NPB, STREAM, ERT), conectados al launcher mediante un catálogo declarativo y un modo de ejecución genérico. Además, la etiqueta de fase 

(compute_bound/memory_bound) no se asume por diseño del kernel: se deriva empíricamente por nodo mediante el modelo Roofline, calibrando el pico de cómputo y el pico de ancho de banda de cada nodo real. Ambos elementos se explican en detalle en la sección 3, y hacen innecesario un kernel "sonda" dedicado únicamente a detectar transiciones de régimen: cualquier kernel se etiqueta con el mismo procedimiento matemático, sin necesitar un caso especial. 

**Atención:** El proyecto tiene tres alternativas metodológicas en discusión para cuando exista más de un nodo disponible — un modelo global con perfil de hardware explícito (A), un modelo global con features normalizadas por calibración local (B), o un modelo específico por nodo con un pipeline reproducible (C). La decisión final depende del director, que está fuera de la oficina, así que este plan adopta una estrategia "sin arrepentimiento": todo lo que se construye (calibración, metadata, esquema de features) queda diseñado para servir a las tres alternativas, sin comprometerse de forma irreversible con ninguna. El desarrollo del catálogo de kernels y del etiquetado por Roofline de la sección 3 incorpora, desde el inicio, los campos que las tres alternativas necesitan. El detalle completo — comparación, recomendación y plan de desarrollo mientras se espera la decisión — está en la sección 13. 

### **1.1 Objetivos de este plan** 

- Definir el catálogo de kernels (suite externa, no programados por el proyecto) y el protocolo de calibración Roofline que sustenta el etiquetado de fase. 

- Definir la matriz experimental (kernels × estados de frecuencia × repeticiones) necesaria para la Fase 1. 

- Establecer el protocolo de seguridad para operar en un nodo HPC compartido sin afectar a otros usuarios ni comprometer la estabilidad del nodo. 

- Enumerar, de forma exhaustiva, los factores de confusión que pueden invalidar una corrida o contaminar el dataset, junto con el control correspondiente — incluyendo los nuevos factores propios de usar binarios externos. 

Página 2 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

- Especificar un protocolo estándar de ejecución de campaña: calibración, preflight, ejecución, validación y registro. 

- Fijar criterios explícitos de aceptación/rechazo por corrida, para que incluir un dato en el dataset no sea una decisión subjetiva. 

- Dejar explícitas las decisiones que dependen de información del clúster que aún no se ha confirmado (sección 12). 

## **2. Principios rectores** 

Cinco principios gobiernan cada decisión de este plan. Cuando exista una disyuntiva entre velocidad de recolección y alguno de estos principios, el plan prioriza el principio. 

### **2.1 Seguridad del nodo compartido** 

El clúster SC3 no es un laboratorio dedicado al proyecto. Cualquier acción que module frecuencia, afinidad o cgroups debe estar acotada exactamente a los recursos delegados a la campaña, y debe ser reversible. 

### **2.2 Reproducibilidad** 

Cada corrida debe poder repetirse exactamente a partir de su metadata: comando ejecutado, commit del harness y del catálogo de kernels, hash del binario ejecutado, host, CPUs, cgroup, governor/frecuencia vigente, fecha y condiciones ambientales relevantes. 

### **2.3 Validez estadística del dato, no solo validez técnica** 

Que una corrida termine sin error no implica que sea útil para entrenamiento. El plan distingue explícitamente entre "la corrida no falló" y "la corrida es apta para el dataset". 

### **2.4 Separación estricta entre datos crudos y datos de entrenamiento** 

samples.csv es una vista cruda con contadores acumulados y snapshots instantáneos. Nunca se entrena directamente sobre esa vista. Toda campaña produce, además, una vista derivada por ventana (windows.csv) con deltas, tasas, la intensidad operacional medida y la etiqueta de fase derivada por Roofline. 

### **2.5 La etiqueta de entrenamiento se mide, no se asume** 

Ningún kernel tiene una etiqueta de fase fija "por ser quien es". La etiqueta que efectivamente se usa para entrenar (phase_label_train) se calcula, por ventana, comparando la intensidad operacional medida contra el ridge point del Roofline calibrado de ese nodo en esa sesión. Cualquier expectativa de la literatura sobre el comportamiento típico de un kernel se conserva únicamente como phase_label_hint, un campo de referencia para auditoría, nunca como la etiqueta que ve el modelo. 

## **3. Diseño de la matriz experimental** 

Página 3 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

La matriz experimental es el conjunto de combinaciones (kernel × estado de frecuencia × repetición) que se ejecutará durante la Fase 1. Los kernels no se diseñan ni se programan dentro del proyecto: se seleccionan de suites de benchmarking ya establecidas y se conectan al launcher mediante un catálogo declarativo. 

**_Nota:_** _Alcance: la matriz que se describe a continuación se ejecuta sobre un solo nodo por sesión de campaña, tal como la Propuesta C recomienda como arquitectura principal (sección 13). Nada de lo definido aquí impide repetir exactamente el mismo protocolo sobre un segundo o tercer nodo el día que el director confirme cuántos nodos habrá disponibles; de hecho, esa repetibilidad exacta es precisamente lo que la sección 13.5 exige que quede garantizado desde ya._ 

### **3.1 Catálogo de kernels: suites externas, no programadas por el proyecto** 

Se adopta un catálogo de tres capas, cada una con un rol distinto en la campaña: 

|**Capa**|**Suite**|**Rol**|**Propósito**|
|---|---|---|---|
|Calibración de ancho<br>de banda|STREAM (McCalpin, binario<br>ofcial)|calibraton|Determinar BW_pico del nodo.<br>No entra al dataset de<br>entrenamiento.|
|Calibración de<br>cómputo pico|ERT (Empirical Roofine Toolkit)|calibraton|Determinar P_pico del nodo. No<br>entra al dataset de<br>entrenamiento.|
|Kernels de dataset|NAS Parallel Benchmarks, clases<br>SER/OMP|dataset|Generar las corridas que sí<br>entran al dataset de<br>entrenamiento, con diversidad<br>de intensidad ya documentada<br>en la literatura.|



Catálogo inicial de kernels de dataset propuesto a partir de NPB (clases discretas S/W/A/B/C, según el tamaño de problema soportado por cada nodo): 

|**kernel_ref**|**Kernel NPB**|**phase_label_hint (prior de**<br>**literatura)**|**Observación**|
|---|---|---|---|
|npb_ep|EP (Embarrassingly<br>Parallel)|compute_bound|Generación de números<br>aleatorios y evaluación de<br>función; casi sin tráfco de<br>memoria relevante.|
|npb_mg|MG (Multgrid)|memory_bound|Acceso estructurado intensivo a<br>memoria en malla 3D; ancho de<br>banda dominante.|
|npb_cg|CG (Conjugate Gradient)|memory_bound|Acceso disperso a memoria<br>(matriz rala); presión sobre la<br>jerarquía de caché.|
|npb_is|IS (Integer Sort)|memory_bound|Ordenamiento con acceso<br>irregular a memoria; bajo IPC|



Página 4 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

|**kernel_ref**|**Kernel NPB**|**phase_label_hint (prior de**<br>**literatura)**|**Observación**<br>esperado.|
|---|---|---|---|
|npb_f|FT (Fast Fourier<br>Transform)|intermedio (mixto)|Uso intensivo de FFT; balance no<br>trivial entre cómputo y memoria,<br>buen caso límite.|
|npb_lu /<br>npb_sp /<br>npb_bt|Solvers de sistemas<br>dispersos/estructurados|intermedio a<br>compute_bound|Mayor intensidad aritmétca que<br>MG/CG/IS, pero con tráfco de<br>memoria no despreciable.|



**_Nota:_** _phase_label_hint es exclusivamente informativo: sirve para detectar, en el reporte de campaña, si la etiqueta empírica derivada por Roofline (sección 3.1.2) contradice fuertemente lo esperado por la literatura, lo cual dispararía una revisión manual del kernel o de la calibración, no un descarte automático._ 

- **Por qué no se necesita un kernel "sonda" dedicado:** un kernel de régimen intermedio, como stencil_2d en un diseño propio, existiría únicamente para detectar el punto de transición entre regímenes, porque sin un mecanismo objetivo de etiquetado no habría otra forma de saber en qué régimen cae. Con el etiquetado por Roofline calibrado (3.1.2), cualquier kernel — incluido uno de comportamiento intermedio como FT — se etiqueta con el mismo procedimiento matemático que los demás, sin necesitar un caso especial. 

#### **3.1.1 Catálogo declarativo (contrato launcher-agnóstico de suite)** 

El catálogo vive en un archivo separado del manifest de campaña, para poder cambiar de suite sin tocar el resto del sistema. A continuación se documenta el contrato conceptual que debe respetar cualquier entrada del catálogo: 

|**Campo**|**Obligatorio**|**Descripción**|
|---|---|---|
|id|Sí|Identfcador único referenciado desde el manifest de<br>campaña.|
|suite|Sí|NPB-OMP, STREAM, ERT, u otra que se incorpore más<br>adelante.|
|role|Sí|dataset o calibraton.|
|exec_path|Sí|Ruta al binario ya compilado en el nodo objetvo.|
|phase_label_hint|Solo si<br>role=dataset|Prior de literatura, nunca la etqueta de entrenamiento.|
|size_variant|Sí|Clase de tamaño NPB (S/W/A/B/C) u otro identfcador<br>discreto de tamaño de la suite correspondiente.|
|expected_runtme_se<br>conds|Sí|Usado para calcular tmeouts razonables por corrida.|
|success_check|Sí|exit_code o stdout_regex, para decidir si la corrida terminó|



Página 5 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

|**Campo**|**Obligatorio**|**Descripción**|
|---|---|---|
|||correctamente.|
|binary_checksum|Sí|Hash del binario compilado, para detectar recompilaciones<br>accidentales entre sesiones/nodos (ver 5.6).|



#### **3.1.2 Protocolo de calibración Roofline por sesión de campaña** 

Antes de ejecutar la matriz de kernels de dataset, cada campaña ejecuta una fase de calibración obligatoria, una sola vez por nodo/sesión: 

- 1. Ejecutar STREAM (binario oficial) sobre los cores delegados y registrar el ancho de banda sostenido medido → BW_pico. 

- 2. Ejecutar ERT (o el micro-benchmark de FLOPs puros equivalente) sobre los mismos cores y registrar el rendimiento de cómputo pico medido → P_pico. 

- 3. Calcular el ridge point del nodo: I_ridge = P_pico / BW_pico (en FLOPs/byte). 

- 4. Para cada ventana de cada corrida de dataset, calcular la intensidad operacional observada: I = FLOPs_reportados_por_el_binario / bytes_movidos_medidos_por_perf (ver nota de portabilidad abajo). 

- 5. Derivar la etiqueta de entrenamiento por ventana: phase_label_train = memory_bound si I < I_ridge, compute_bound si I ≥ I_ridge. 

- 6. Guardar P_pico, BW_pico, I_ridge y los metadatos de la calibración en un artefacto roofline_calibration.json versionado junto con la campaña, referenciado desde la metadata de cada corrida. 

**Atención:** Nota de portabilidad sobre FLOPs: no todos los procesadores exponen de forma confiable, vía perf, un evento de PMU para "operaciones de punto flotante retiradas" (varía entre Intel, AMD y entre generaciones). Por eso el paso 4 no depende de un contador de hardware de FLOPs: usa el conteo de FLOPs que el propio binario de la suite reporta por stdout al finalizar (estándar en NPB/STREAM/ERT), combinado con bytes movidos medidos por perf (LLC misses × tamaño de línea de caché), que sí es un contador universal y portable entre plataformas. 

### **3.2 Estados de frecuencia a explorar** 

Se define la siguiente rejilla, que combina extremos operativos con puntos intermedios, evitando depender de un gobernador dinámico durante la toma de datos de entrenamiento. 

|**Nivel**|**Confguración CPU**|**Propósito**|
|---|---|---|
|F0|governor userspace, f = f_max soportada|Límite superior de<br>rendimiento/consumo.|
|F1|governor userspace, f ≈ 75% del rango<br>[f_min, f_max]|Punto intermedio alto.|
|F2|governor userspace, f ≈ 50% del rango|Punto intermedio central.|



Página 6 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

|**Nivel**|**Confguración CPU**|**Propósito**|
|---|---|---|
|F3|governor userspace, f ≈ 25% del rango|Punto intermedio bajo.|
|F4|governor userspace, f = f_min soportada|Límite inferior de<br>rendimiento/consumo.|
|REF|gobernador dinámico natvo<br>(ondemand/schedutl), sin control manual|Referencia de comparación; no se usa<br>para entrenar.|



Los valores exactos de f_min/f_max dependen de las P-states que exponga el hardware asignado (ver 12.3); el preflight debe leerlas y discretizar F1–F3 al valor real más cercano, dejando registrada la frecuencia efectivamente aplicada. 

### **3.3 Repeticiones y tamaño de muestra** 

Con binarios externos, cada repetición es un relanzamiento completo e independiente del binario (proceso nuevo), no una iteración interna medida dentro de un mismo proceso, porque no se tiene control sobre la cooperación interna del kernel. El warmup, por la misma razón, se define por tiempo de pared (segundos declarados en el catálogo, campo expected warmup) y se excluye en el post-procesamiento, no dentro del propio binario. 

|**Parámetro**|**Valor propuesto inicial**|**Justfcación**|
|---|---|---|
|Repetciones por<br>combinación|10|Relanzamientos independientes del<br>binario; permite detectar y descartar<br>outliers sin perder la condición.|
|Warmup (tempo de pared<br>excluido)|según catálogo por kernel (p.<br>ej. 1–2 s); ventanas marcadas<br>warmup_excluded en post-<br>procesamiento|Evita capturar el transitorio de arranque sin<br>depender de cooperación interna del<br>binario.|
|Duración medida por<br>repetción|la que tome el binario en su<br>clase de tamaño (S/W/A…),<br>verifcando ≥ 50 ventanas<br>útles tras excluir warmup|Sufcientes flas por repetción tras aplicar<br>deltas.|
|Intervalo de muestreo (--<br>interval-ns)|1 ms como punto de partda;<br>validar overhead real|Balance entre resolución temporal y<br>sobrecarga del propio collector.|
|Orden de ejecución<br>dentro de la campaña|aleatorizado por<br>combinación, no bloque por<br>bloque|Evita confundir deriva térmica/temporal<br>con el efecto del kernel o la frecuencia.|



**_Nota:_** _Si la clase de tamaño más pequeña de un kernel NPB (S) no alcanza las ~50 ventanas útiles al muestrear a 1 ms, se sube a la siguiente clase disponible (W) antes que reducir el intervalo de muestreo por debajo de lo validado como seguro para el overhead del collector._ 

Página 7 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

### **3.4 Tamaño total estimado de la campaña** 

Con 6 kernels de dataset (EP, MG, CG, IS, FT y un solver del grupo LU/SP/BT) × 6 configuraciones de frecuencia (5 fijas + referencia dinámica) × 10 repeticiones se obtienen 360 corridas de medición, más 360 corridas baseline. A esto se suma, una única vez por sesión de campaña, la fase de calibración (STREAM + ERT), que no escala con el tamaño de la matriz. Antes de ejecutar la campaña completa en el nodo real, se corre una campaña piloto reducida (2 kernels × 2 configuraciones × 3 repeticiones + calibración completa) para validar el protocolo de punta a punta, incluyendo que I_ridge calculado sea plausible frente a la ficha técnica del hardware. 

## **4. Protocolo de seguridad en el nodo HPC compartido** 

Ningún paso de la campaña debe poder degradar el servicio para otros usuarios del clúster ni dejar el nodo en un estado distinto al que tenía antes de la campaña. 

### **4.1 Delimitación explícita de recursos** 

- Confirmar por escrito, con el administrador del SC3 o el semillero, qué núcleos, qué cgroup delegado y, si aplica, qué GPU están asignados exclusivamente a la campaña durante la ventana de ejecución. 

- Verificar que el cgroup delegado es escribible y que cgroup.procs está vacío antes de iniciar cualquier corrida. 

- No ejecutar workloads de medición fuera del conjunto de CPUs confirmado como propio. 

- Si el clúster usa un gestor de colas (SLURM u otro), solicitar la reserva mediante ese gestor en lugar de asumir exclusividad manual (ver 12.2). 

### **4.2 Control de frecuencia sin comprometer el nodo** 

- El governor userspace, si se usa, debe fijarse únicamente sobre los cores delegados a la campaña. 

- Antes de cada corrida se debe leer y guardar el governor/frecuencia original; al finalizar la corrida, la campaña, o ante fallo/interrupción, restaurar ese estado de forma automática. 

- Debe existir una rutina de restauración de emergencia ejecutable en cualquier momento, incluso si la corrida se interrumpe a mitad de camino. 

- Si no hay permisos para escribir en scaling_governor/scaling_setspeed, formalizar la solicitud con el administrador antes de la campaña (ver 12.3). 

### **4.3 Convivencia con otros procesos y usuarios** 

- Verificar, antes de cada corrida, que no hay procesos ajenos con carga significativa en los cores delegados, usando únicamente herramientas de solo lectura. 

- No usar nice/renice, taskset ni cgroups para desplazar procesos de otros usuarios. 

- Anunciar en el canal del semillero la ventana horaria de la campaña, especialmente si se van a fijar frecuencias bajas. 

Página 8 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

### **4.4 Límites temporales y de recursos** 

- Cada corrida debe tener timeout de fase, calculado a partir de expected_runtime_seconds del catálogo más un margen, para evitar que un binario colgado consuma la ventana reservada del nodo. 

- El orquestador debe verificar, al final de cada corrida, que no quedaron procesos hijos vivos antes de iniciar la siguiente combinación. 

- Llevar contabilidad del consumo real de hora-núcleo de la campaña piloto para proyectar el consumo de la campaña completa. 

## **5. Factores de confusión y amenazas a la calidad del dato** 

Se identifican cinco grupos de factores que pueden hacer que una corrida técnicamente exitosa produzca datos inservibles o engañosos para el clasificador: hardware/firmware, sistema operativo, energía, GPU y factores metodológicos. Se agrega, además, una categoría propia de usar binarios pre-compilados de suites externas en lugar de kernels programados por el proyecto. 

### **5.1 Hardware y firmware** 

- Turbo Boost/HWP: registrar la frecuencia real observada por ventana, no solo la solicitada. 

- Throttling térmico: registrar temperatura del paquete si el sensor está disponible; insertar enfriamiento entre repeticiones si se detecta deriva. 

- C-states: mantener el workload activo durante toda la ventana medida. 

- Topología NUMA: fijar afinidad de memoria al mismo nodo NUMA que los cores delegados. 

- Hyperthreading/SMT: decidir explícitamente la política (un hilo por core físico vs. todos) y mantenerla constante entre condiciones. 

- ASLR y alineación de memoria: no requiere desactivarse; se compensa con el número de repeticiones. 

### **5.2 Sistema operativo** 

- Procesos ajenos dentro del cgroup o del core delegado. 

- Deriva de gobernador (governor drift): releer scaling_governor y scaling_cur_freq antes y después de cada corrida. 

- Multiplexación de contadores de PMU: usar time_enabled_ns/time_running_ns y rechazar ventanas con ratio bajo umbral. 

- Jitter del scheduler y del propio collector: usar el intervalo real medido, no el nominal. 

- Presión sobre la ring buffer SPSC: rechazar corrida si push_retries > 0. 

- Carga externa del nodo fuera del cgroup. 

### **5.3 Energía** 

- Wrap-around de contadores RAPL: corregir con max_energy_range_uj cuando esté disponible. 

- Lecturas RAPL inválidas devueltas como cero: propagar una bandera explícita de validez. 

Página 9 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

- Dominio Package vs. DRAM vs. consumo total del nodo: ser consistente en qué dominios se comparan entre condiciones. 

- Periodicidad de actualización del contador: no muestrear por debajo de la resolución real validada en el piloto. 

### **5.4 GPU (fuera de alcance por ahora, ver 12.4)** 

Se mantiene fuera de la ruta experimental principal hasta contar con exclusividad confirmada del dispositivo. 

### **5.5 Factores metodológicos** 

- Orden de ejecución y deriva temporal: aleatorizar el orden entre combinaciones dentro de la campaña. 

- Fuga entre baseline y telemetry: mantener la separación entre ambos procesos. 

- Warmup insuficiente: validar en el piloto que las ventanas ya excluidas de warmup no muestran tendencia sistemática distinta del resto de la repetición. 

- run_id reutilizado: el orquestador falla por defecto si el directorio de salida ya existe, salvo overwrite explícito. 

- **Calibración implausible:** la etiqueta de entrenamiento (phase_label_train) se deriva empíricamente por Roofline (3.1.2), no se asume por diseño del kernel. Por eso lo que debe vigilarse no es el kernel en sí, sino que la calibración Roofline (P_pico, BW_pico, I_ridge) sea plausible: si I_ridge resulta absurdo frente a la ficha técnica del hardware, todas las etiquetas derivadas de esa sesión quedan en duda. 

### **5.6 Catálogo y binarios externos** 

#### **Reproducibilidad del build entre nodos/sesiones** 

Un binario de NPB/STREAM/ERT compilado en un nodo con un compilador y flags de optimización distintos a otro nodo puede tener un comportamiento de rendimiento distinto, aunque sea "el mismo kernel". Control: registrar el checksum (binary_checksum) del binario efectivamente ejecutado en cada corrida, y rechazar la corrida si el checksum no coincide con el registrado para ese kernel en el catálogo de esa campaña. 

#### **Ausencia de cooperación interna del binario para warmup/iteraciones** 

A diferencia de un kernel propio, un binario externo no emite una señal de "ya terminé el warmup". Control: warmup por tiempo de pared declarado en el catálogo (3.1.1) y validado empíricamente en el piloto, no por conteo de iteraciones internas. 

#### **Verificación de corrección del binario, no solo de su terminación** 

Un binario puede terminar con código de salida 0 y aun así no haber completado su verificación interna de resultados (muchos kernels de NPB imprimen explícitamente si la verificación fue exitosa). Control: success_check en el catálogo debe validar, cuando la suite lo soporte, el mensaje de verificación exitosa vía stdout_regex, no solo el código de salida. 

#### **Portabilidad del conteo de FLOPs** 

Página 10 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

Ver la nota de portabilidad en 3.1.2: no se debe depender de un contador de hardware de FLOPs para calcular la intensidad operacional, porque no es consistente entre plataformas Intel/AMD ni entre generaciones. Control: usar el FLOP count auto-reportado por el binario de la suite. 

#### **Clases de tamaño discretas (NPB CLASS)** 

A diferencia de un kernel propio donde el tamaño se controla de forma continua, NPB expone clases discretas (S, W, A, B, C…). Control: el catálogo declara explícitamente qué clase se usa por kernel y nodo, y el piloto valida que la clase elegida produce suficientes ventanas útiles (3.3) sin exceder el timeout de fase (4.4). 

## **6. Preflight obligatorio previo a cada campaña** 

Antes de iniciar cualquier campaña se ejecutan las siguientes verificaciones de solo lectura, agrupadas por categoría: E (entorno del nodo) y C (catálogo de binarios externos). 

|**Verifcación**|**Fuente**|**Acción si falla**|
|---|---|---|
|Cores delegados coinciden con lo<br>confrmado por el administrador.|cgroup, afnidad|Abortar campaña.|
|cgroup.procs vacío.|sysfs cgroup|Abortar hasta limpiar.|
|Governor actual y frecuencias<br>disponibles por core delegado.|scaling_governor,<br>scaling_available_freque<br>ncies|Registrar; abortar si no<br>coincide con lo esperado.|
|Permisos de escritura sobre<br>scaling_setspeed/scaling_governor.|sysfs cpufreq|Abortar y escalar con<br>administrador.|
|Dominios RAPL disponibles y<br>energy_uj legible.|sysfs powercap|Abortar si energía es<br>objetvo de la corrida y no<br>es legible.|
|C01: binarios del catálogo existen y<br>son ejecutables en el nodo objetvo.|flesystem|Abortar campaña.|
|C02: checksum de cada binario<br>coincide con el registrado en el<br>catálogo.|hash del binario|Abortar; posible<br>recompilación no<br>documentada.|
|C04: calibración STREAM/ERT<br>ejecutada y BW_pico/P_pico dentro<br>de un rango plausible frente a la fcha<br>del hardware.|roofine_calibraton.json|Abortar; revisar<br>calibración antes de<br>contnuar.|
|run_id/directorio de salida no existe<br>previamente.|flesystem|Abortar salvo overwrite<br>explícito.|



**_Nota:_** _El preflight reducido de cada corrida individual incluye cgroup limpio, governor esperado, ausencia de procesos ajenos, y agrega C03 (verificación de que el success_check del kernel específico está correctamente configurado antes de lanzar)._ 

Página 11 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

## **7. Protocolo estándar de ejecución de una campaña** 

### **7.1 Campaña piloto** 

- Ejecutar la fase de calibración completa (STREAM + ERT) y verificar que I_ridge resultante es plausible. 

- Ejecutar al menos 2 kernels de dataset en 2 configuraciones de frecuencia con 3 repeticiones cada una. 

- Confirmar que el preflight (sección 6) detecta problemas simulados a propósito (checksum alterado, binario faltante, cgroup sucio). 

- Medir el tiempo real por corrida, el overhead real del collector y el jitter real de muestreo. 

- Confirmar que la rutina de restauración de frecuencia funciona incluso ante una interrupción forzada. 

- Producir al menos una vista windows.csv de ejemplo con la columna de intensidad operacional y phase_label_train calculadas, para validar el post-procesamiento antes de escalar. 

### **7.2 Secuencia por campaña** 

- 1. Preflight de campaña (sección 6). 

- 2. Fase de calibración: correr STREAM y ERT, calcular I_ridge, escribir roofline_calibration.json. 

- 3. Generar y aleatorizar la matriz de combinaciones de kernels de dataset (kernel × frecuencia × repetición). 

- 4. Por cada combinación: preflight reducido → fijar frecuencia → ejecutar el binario vía el modo genérico del launcher → validar (sección 8) → registrar metadata. 

- 5. Restaurar governor/frecuencia original de todos los cores delegados y confirmarlo por lectura. 

- 6. Post-procesar samples.csv de las corridas aceptadas a windows.csv, aplicando la clasificación Roofline con el I_ridge calculado en el paso 2. 

- 7. Consolidar el reporte de campaña. 

## **8. Criterios de aceptación y rechazo por corrida** 

Una corrida se marca automáticamente como no apta para el dataset, sin intervención manual, si ocurre cualquiera de las siguientes condiciones. El rechazo no borra la corrida: se conserva en crudo, etiquetada como rechazada y con el motivo. 

|**Condición de rechazo**|**Verifcación**|
|---|---|
|samples_collected == 0 para algún backend actvo esperado.|metadata.json|
|push_retries > 0.|metadata.json|
|Governor o frecuencia efectva no coincide con la solicitada, fuera<br>de tolerancia.|lectura pre/post corrida|



Página 12 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

|**Condición de rechazo**|**Verifcación**|
|---|---|
|Rato delta_running/delta_enabled por debajo del umbral defnido<br>en alguna ventana relevante.|perf tme_enabled/tme_running|
|Delta energétco negatvo o fuera de rango plausible sin corrección<br>de wrap disponible.|RAPL|
|Proceso ajeno detectado en el cgroup durante la corrida.|prefight por corrida|
|Timeout de alguna fase.|orquestador|
|Menos del mínimo de ventanas válidas tras aplicar deltas y excluir<br>warmup.|post-procesamiento|
|C02: checksum del binario ejecutado no coincide con el registrado<br>en el catálogo.|prefight/runner|
|C03: success_check del kernel no se cumple (exit code o patrón de<br>verifcación en stdout).|runner|
|I_ridge de la sesión no plausible (calibración fallida o degradada).|roofine_calibraton.json|



## **9. Post-procesamiento, metadata y trazabilidad** 

### **9.1 De samples.csv a windows.csv, con clasificación Roofline** 

Se calculan los siguientes deltas, tasas y campos derivados por ventana, incluyendo el cálculo de la intensidad operacional y la derivación de la etiqueta de entrenamiento: 

- Calcular deltas de instructions, cycles, cache_references, cache_misses entre muestras consecutivas de la misma repetición. 

- Calcular delta_running/delta_enabled por ventana y descartar ventanas por debajo del umbral de PMU. 

- Calcular energía útil por ventana y potencia, corrigiendo wrap-around cuando esté disponible. 

- **Calcular la intensidad operacional observada por ventana:** I = FLOPs_reportados_por_el_binario (prorrateados por ventana según su duración relativa dentro de la repetición) / bytes_movidos_medidos_por_perf en esa ventana. 

- **Derivar phase_label_train:** memory_bound si I < I_ridge, compute_bound si I ≥ I_ridge, usando el I_ridge de roofline_calibration.json correspondiente a esa sesión/nodo. 

- Conservar phase_label_hint (del catálogo) como columna de auditoría, sin usarla nunca como variable de entrenamiento. 

- Marcar como inválida la primera muestra de cada repetición, y como warmup_excluded las ventanas dentro del tiempo de pared declarado en el catálogo. 

- Propagar una bandera de validez agregada por ventana y otra por corrida completa. 

- Adjuntar a cada fila el estado de frecuencia, el kernel_ref, el checksum del binario y el run_id, para trazar cualquier fila hasta su corrida de origen. 

Página 13 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

### **9.2 Metadata mínima por corrida** 

**_Nota:_** _Se incluyen node_id y node_profile_ref porque, sin importar qué alternativa (A/B/C) apruebe el director, cualquier análisis posterior necesita poder agrupar y separar datos por nodo. Añadir estos campos ahora es barato; reconstruirlos retroactivamente sobre un dataset ya recolectado no lo es. Ver sección 13.5._ 

|**Campo**|**Descripción**|
|---|---|
|run_id, repetción, kernel_ref|Identfcación de la corrida.|
|node_id|Identfcador estable del nodo donde se ejecutó la corrida.<br>Obligatorio para poder hacer group-split por nodo bajo<br>cualquiera de las tres alternatvas.|
|node_profle_ref|Referencia al node_profle.json de esa sesión (topología,<br>caché, NUMA, rango de frecuencia, referencias de<br>calibración). Necesario para la Propuesta A y para la Propuesta<br>B.|
|Comando completo ejecutado (modo<br>--exec)|Reproducibilidad exacta.|
|binary_checksum efectvo|Verifcación de reproducibilidad del build.|
|roofine_calibraton_ref (referencia a<br>la sesión de calibración usada)|Traza qué I_ridge se usó para etquetar esta corrida, y sirve<br>como base de las referencias P95 de la Propuesta B (sección<br>13.5).|
|Commit/versión del harness, del<br>catálogo y del script de post-<br>procesamiento|Trazabilidad de código.|
|Hostname y modelo de CPU|Contexto de hardware.|
|CPUs delegadas, cgroup usado,<br>topología NUMA|Contexto de aislamiento.|
|Governor solicitado y efectvo,<br>frecuencia solicitada y efectva<br>(pre/post)|Validación del estado de frecuencia.|
|Timestamp absoluto de inicio y fn|Detección de deriva temporal.|
|Resultado del prefight por corrida|Auditoría de condiciones previas.|
|Motvo de rechazo, si aplica|Auditoría de calidad.|



## **10. Cronograma de implementación dentro de la Fase 1** 

|**Semana**|**Actvidad**|
|---|---|
|1|Confrmar con administrador/semillero los recursos delegados y permisos de<br>frecuencia. Compilar y verifcar (checksum) los binarios de NPB, STREAM y ERT en cada<br>entorno de prueba disponible.|



Página 14 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

|**Semana**|**Actvidad**|
|---|---|
|1–2|Defnir catálogo de kernels (kernels/catalog.yaml), manifest de campaña con<br>referencia a calibración, y script de orquestación con tmeouts, restauración de<br>frecuencia y rechazo de run_id duplicado.|
|2|Ejecutar la fase de calibración Roofine (STREAM + ERT) en el nodo real y validar que<br>I_ridge es plausible; ejecutar campaña piloto (7.1).|
|2–3|Ajustar parámetros de 3.3 con datos empíricos del piloto; implementar el post-<br>procesamiento con clasifcación Roofine (9.1) y criterios de aceptación/rechazo<br>(sección 8).|
|3|Ejecutar campaña completa de CPU (sección 3.4) en bloques compatbles con la<br>ventana de nodo disponible, con orden aleatorizado y monitoreo contnuo.|
|4|Consolidar dataset, reporte de campaña, y validar cobertura de las dos clases de<br>entrenamiento (memory_bound/compute_bound) antes de pasar a la Fase 2.|



## **11. Riesgos y mitigaciones** 

|**Riesgo**|**Impacto**|**Mitgación**|
|---|---|---|
|No se obtenen permisos de<br>escritura sobre cpufreq en el<br>nodo compartdo.|No se puede fjar frecuencia;<br>la matriz de 3.2 no es<br>ejecutable.|Escalar con el administrador antes<br>de la campaña piloto (12.3).|
|El evento de PMU de FLOPs no<br>está disponible o no es<br>comparable entre plataformas.|Cálculo erróneo de<br>intensidad operacional.|Usar FLOPs auto-reportados por el<br>binario de la suite en vez de<br>contador de hardware (3.1.2).|
|Los binarios de<br>NPB/STREAM/ERT se compilan<br>distnto en cada nodo<br>(compilador, fags).|Corridas no comparables<br>entre sesiones/nodos.|Verifcación de checksum<br>obligatoria (C02) antes de aceptar<br>cualquier corrida.|
|La calibración Roofine de una<br>sesión resulta implausible<br>(I_ridge absurdo).|Todas las etquetas de esa<br>sesión quedan en duda.|Prefight C04 bloqueante; comparar<br>contra fcha técnica del hardware<br>antes de aceptar la calibración.|
|El presupuesto de hora-núcleo se<br>agota antes de completar la<br>matriz.|Dataset incompleto o<br>desbalanceado entre<br>condiciones.|Priorizar cobertura de las 2 clases<br>de entrenamiento sobre los 5<br>niveles de frecuencia si hay que<br>recortar.|
|Otro usuario ejecuta carga alta<br>en el nodo durante la ventana<br>reservada.|Contaminación de<br>mediciones.|Monitor de carga externa en<br>prefight y por corrida; posponer si<br>supera umbral.|



## **12. Preguntas abiertas y decisiones pendientes** 

Página 15 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

### **12.1 Acceso y delegación de recursos** 

- ¿Qué nodo(s) concretos del SC3 tiene asignados el semillero para este proyecto, y con qué exclusividad temporal? 

- ¿Existe ya un cgroup delegado y escribible para el usuario del proyecto? 

### **12.2 Gestor de colas / reserva de recursos** 

- ¿El clúster usa SLURM u otro gestor de colas, o el acceso es por sesión interactiva directa sobre un nodo compartido? 

### **12.3 Permisos y capacidades de DVFS** 

- ¿El usuario del proyecto tiene permisos para escribir en scaling_governor/scaling_setspeed en los cores que se le deleguen? 

- ¿Qué modelo de CPU tiene el nodo asignado, y qué rango de P-states soporta realmente? 

- ¿RAPL expone qué dominios en ese hardware, y existe max_energy_range_uj disponible? 

### **12.4 GPU** 

- ¿El nodo tiene GPU asignada con exclusividad, o es un recurso compartido? Esto sigue determinando si la GPU entra a la Fase 1 o permanece fuera. 

### **12.5 Suites y compilación** 

- ¿Quién compila NPB/STREAM/ERT en cada entorno de prueba (PC local, cloud propio, eventualmente SC3), y con qué toolchain/flags, para poder fijar el proceso de verificación de checksum entre sesiones? 

- ¿Se dispone de compilador Fortran (gfortran) en todos los entornos de prueba? NPB clásico lo requiere para varios de sus kernels. 

### **12.6 Ventana de tiempo y coordinación** 

- ¿Cuánto tiempo de ventana exclusiva de nodo se tiene disponible por sesión de campaña? 

- ¿Hay algún canal formal donde se deba anunciar la ventana de la campaña antes de fijar frecuencias no estándar? 

## **13. Estrategia multinodo: generalización, portabilidad y ruta de decisión con el director** 

Cuando el proyecto llegue a tener más de un nodo disponible, existen tres formas distintas de organizar la toma de datos, el entrenamiento y el despliegue del clasificador. Esta sección compara esas tres alternativas y condiciona directamente cómo debe diseñarse la metadata, la calibración y el esquema de features desde ahora, aunque la decisión final todavía no esté tomada. 

### **13.1 Las tres alternativas, en corto** 

Página 16 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

|**Alternatva**|**Idea central**|**Qué se despliega en un nodo**<br>**nuevo**|
|---|---|---|
|A. Modelo global con<br>hardware explícito|Un único modelo entrenado con<br>telemetría + descriptores explícitos<br>del hardware de cada nodo<br>(topología, caché, ancho de banda,<br>etc.) como variables de entrada.|Se construye el perfl de<br>hardware del nodo nuevo y se<br>alimenta al modelo global ya<br>entrenado, sin reentrenar<br>necesariamente.|
|B. Modelo global con<br>features relatvas|Un único modelo entrenado sobre<br>métricas normalizadas contra<br>referencias de calibración propias de<br>cada nodo (p. ej. IPC_relatvo =<br>IPC_ventana / IPC_referencia_nodo).|Se corre una calibración local<br>(no supervisada) para obtener<br>las referencias del nodo, y se<br>usa el modelo global ya<br>entrenado con esas<br>referencias.|
|C. Modelo específco por<br>nodo + pipeline<br>reproducible|El modelo entrenado se trata como<br>parte de la confguración de ese<br>nodo. Lo que se reutliza entre nodos<br>es el pipeline completo (código,<br>protocolo, esquema de datos), no los<br>parámetros del modelo.|Se repite la campaña de<br>caracterización y<br>entrenamiento completa en el<br>nodo nuevo antes de poder<br>predecir en él.|



### **13.2 Cuadro comparativo: pros, contras y recomendación** 

|**Criterio**|**A. Hardware explícito**|**B. Valores relatvos**|**C. Modelo por nodo**|
|---|---|---|---|
|Nodos requeridos para ser<br>defendible|Varios y realmente<br>diversos|Al menos 2,<br>idealmente 3+|Uno por estudio;<br>repetble sin límite|
|Alineación con el alcance<br>intra-nodo ya aprobado en<br>el plan de grado|Baja, exige ampliar el<br>alcance formal|Media, se puede<br>presentar como<br>extensión|Alta, es la<br>contnuación natural<br>de lo ya aprobado|
|Riesgo de domain shif /<br>sobreajuste a los pocos<br>nodos disponibles|Medio-alto: con 2–3<br>nodos el modelo<br>puede memorizar<br>diferencias entre<br>nodos en vez de<br>aprender una relación<br>general|Medio: la<br>normalización reduce<br>diferencias de escala<br>pero no garantza<br>equivalencia<br>semántca de eventos<br>PMU|Bajo dentro del nodo:<br>cada modelo solo<br>necesita generalizar a<br>nuevas corridas de su<br>propio dominio|
|Trabajo experimental<br>adicional requerido|Alto: perfles de<br>hardware, calibración<br>de kernels por nodo,<br>dataset conjunto con<br>splits leave-one-node-<br>out|Medio-alto:<br>calibración estable<br>por nodo (CV ≤ 5%)<br>más el mismo<br>entrenamiento<br>conjunto|Medio:<br>caracterización y<br>campaña por nodo,<br>pero reutlizando el<br>mismo protocolo<br>versionado|
|Qué queda como<br>conclusión defendible con|"Transfere si se<br>demuestra" —|"Transferencia<br>normalizada si se|"El pipeline es<br>reproducible y cada|



Página 17 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

|**Criterio**|**A. Hardware explícito**|**B. Valores relatvos**|**C. Modelo por nodo**|
|---|---|---|---|
|la evidencia esperada|afrmación fuerte,<br>difcil de sostener con<br>pocos nodos|demuestra" —<br>afrmación moderada,<br>evaluable por<br>ablación|modelo generaliza<br>dentro de su nodo" —<br>afrmación<br>conservadora y<br>alcanzable|
|Riesgo de sobreprometer<br>en la sustentación|Alto|Medio|Bajo|
|Recomendación de este<br>documento|Reservar para trabajo<br>futuro, o para una<br>ampliación de alcance<br>explícitamente<br>acordada con el<br>director si aparecen<br>más nodos diversos.|Evaluar como<br>experimento<br>secundario de<br>transferencia (no<br>como requisito de<br>éxito del proyecto).|Adoptar como<br>arquitectura ofcial<br>del proyecto.|



**_Nota:_** _Esta recomendación es consistente con la restricción ya existente en el plan de grado aprobado, que limita formalmente la validación a un nodo y aclara explícitamente que no se busca un modelo universal de optimización energética._ 

### **13.3 Por qué no hay que decidir hoy para poder seguir avanzando** 

Las tres alternativas comparten una base de trabajo idéntica: caracterización del nodo, calibración, campaña de telemetría, construcción de features por ventana, criterios de calidad y versionado. Donde divergen es en qué se hace con esos datos después: entrenar un modelo con perfil de hardware (A), entrenar un modelo con features normalizadas (B), o entrenar un modelo local (C). Esa divergencia ocurre después de recolectar el dataset, no antes. Por eso es posible — y es lo que recomienda esta sección — seguir construyendo el orquestador y ejecutando campañas ya, sin esperar la respuesta del director, siempre que la capa común capture todo lo que las tres alternativas necesitarían. 

### **13.4 Qué se pospone hasta la decisión del director** 

- Comprometer un número y una diversidad concreta de nodos para la campaña (necesario para A y B, no para C). 

- Diseñar los splits leave-one-node-out y el experimento cross-node formal. 

- Redactar el alcance formal del trabajo de grado como "modelo transferible" en vez de "pipeline reproducible con modelos locales". 

- Decidir si la compilación de los binarios de la suite (NPB/STREAM/ERT) debe ser -march=native por nodo (aceptable si el modelo es local, problemático si se busca comparar nodos directamente). 

### **13.5 Estrategia "sin arrepentimiento": qué construir ya, sirva la decisión que sirva** 

Página 18 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

Esta es la capa común obligatoria, independiente de cuál alternativa se termine adoptando. Es una extensión directa de lo que ya estaba planeado en las secciones 3 a 9 de este documento, no un trabajo paralelo distinto. 

|**Elemento a construir ya**|**Por qué sirve a las tres alternatvas**|**Dónde se defne en**<br>**este plan**|
|---|---|---|
|node_id explícito en toda corrida<br>y toda fla de windows.csv|Sin esto no se puede hacer group-split<br>por nodo bajo A, B ni C.|Sección 9.2<br>(metadata)|
|node_profle.json por sesión<br>(topología, caché, NUMA, rango<br>de frecuencia)|Es exactamente el insumo de la<br>Propuesta A y una precondición para<br>interpretar cualquier comparación entre<br>nodos.|Extende la<br>calibración ya<br>defnida en 3.1.2|
|Referencias de calibración<br>estables (P95 de<br>IPC/MPKI/MissRate, con criterio<br>de estabilidad CV ≤ 5% entre<br>repetciones)|Es exactamente el insumo de la<br>Propuesta B, y reutliza el mismo<br>mecanismo de calibración que ya se<br>construye para I_ridge (Roofine, sección<br>3.1.2).|Extende<br>calibraton.py|
|Guardar en windows.csv tanto<br>las features absolutas como sus<br>versiones relatvas ya calculadas<br>(IPC_relatve, MPKI_relatve,<br>MissRate_relatve)|Calcular esto ahora es práctcamente<br>grats; recalcularlo retroactvamente<br>exige volver a tener el node_profle de<br>cada corrida vieja disponible y<br>consistente.|Sección 9.1 (post-<br>procesamiento)|
|sample_status para ventanas de<br>transición/ambiguas, separado<br>de la etqueta binaria|Evita que una decisión tomada hoy sobre<br>cómo tratar ventanas ambiguas<br>contamine cualquiera de los tres análisis<br>posteriores.|Ya cubierto por<br>quality_status en la<br>sección 9.1; se deja<br>explícito que cumple<br>este rol|
|Esquema de datos, catálogo y<br>pipeline versionados<br>(commit/hash de cada<br>componente en la metadata)|Es indispensable para poder decir, más<br>adelante, que dos campañas en dos<br>nodos son comparables porque corrieron<br>exactamente el mismo protocolo.|Ya cubierto en la<br>sección 9.2|
|Campaña de un solo nodo por<br>manifest, pero manifest<br>parametrizado para poder<br>apuntarse a otro nodo sin<br>reescribir nada más que<br>environment_ter/cores|Permite repetr la campaña en un<br>segundo nodo el día que se apruebe, sin<br>rediseñar el orquestador.|Ya cubierto por el<br>diseño del manifest<br>en la sección 3|



**Atención:** Lo único que NO se debe hacer todavía es comprometer recursos (tiempo de campaña, presupuesto de hora-núcleo) en ejecutar la matriz completa sobre un segundo o tercer nodo. Eso sí depende de la decisión del director, porque cambia el alcance formal del trabajo de grado. Todo lo demás en esta tabla es trabajo de infraestructura que hay que construir de todas formas para el nodo único, y que además deja la puerta abierta a A, B o C sin retrabajo. 

Página 19 de 20 

Plan de Implementación — Plataforma Experimental de Telemetría (Fase 1) 

### **13.6 Preguntas para el director (para cuando regrese de vacaciones)** 

- ¿La contribución principal del trabajo de grado será un modelo transferible entre nodos, o un pipeline reproducible que produce modelos locales? 

- ¿El uso de varios nodos es parte del alcance formal aprobado, o una validación complementaria/exploratoria? 

- ¿Cuántos nodos concretos estarán disponibles para el proyecto, y qué tan diferentes son entre sí (misma microarquitectura o distinta)? 

- ¿Se acepta que el éxito principal del trabajo sea intra-nodo (Propuesta C), dejando cross-node como exploración secundaria (Propuesta B)? 

- ¿Se acepta recalibrar y reentrenar cada vez que se despliegue en un nodo nuevo, en vez de perseguir un modelo verdaderamente universal (Propuesta A)? 

- ¿Qué eventos de perf se consideran obligatorios y cuáles opcionales, de cara a comparar nodos con PMUs potencialmente distintas? 

- ¿Se conserva compilación nativa (-march=native) por nodo, o se exige un binario común para poder comparar nodos directamente? 

- ¿Qué criterios cuantitativos definirían una "transferencia exitosa" si se llegara a intentar la Propuesta B? 

Página 20 de 20 

