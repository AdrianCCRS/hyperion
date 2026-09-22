# Plan de reducción del Marco de referencia (Capítulo 1)


## 0. Numeración objetivo

La numeración del libro empieza en 1 para el capítulo. El capítulo 1 es el **Marco de referencia** y contiene dos secciones.

```
1      Marco de referencia
1.1    Marco Conceptual
1.1.1  Potencia CMOS y fundamento de DVFS
1.1.2  Modelo Roofline y regímenes de ejecución
1.1.3  Telemetría microarquitectónica con PMU
1.1.4  Espacios de ejecución user-space y kernel-space
1.1.5  Interfaces de observabilidad
1.1.6  P-states, gobernadores y DVFS
1.1.7  Cargas de calibración y modelos de clasificación ligera
1.1.8  Hiperparámetros, Optuna y validación
1.1.9  Producto Energía–Retardo (EDP)
1.2    Estado del Arte


Reglas de estructura.

- Se usan `\subsection` para 1.1.x y 1.2.x. Se eliminan casi todos los `\subsubsection` actuales (hoy hay unos 20). Cuando un subtema deba conservar una etiqueta, se usa un párrafo corto dentro de la subsección con el `\label` correspondiente (ver sección 5).
- Cada subsección justifica una pieza del proyecto y termina abriendo la dependencia con la siguiente, sin cierre enfático.
- La síntesis final (`sec:marco-sintesis`) deja de ser subsección. Pasa a ser un párrafo de cierre de 1.1.9. La etiqueta no se referencia desde otros archivos.

## 1. Lectura obligatoria antes de escribir

1.  `README.md`, `MANUAL_ESTUDIANTES.md`, `Plan_Detallado_Realineacion_Hyperion.md` y `Seguimiento_Cambios_Plan_Director.md`, para saber qué es primordial en el marco y qué ya se explica en Metodología.

2. main.pdf, para conservar solo citas ya existentes.

## 2. Meta de extensión y criterio de decisión

- El director pide Marco Conceptual con figuras y Estado del Arte en unas 20 páginas como máximo.
- Medida actual en `main.pdf` (el `.toc` está desactualizado y no sirve). El capítulo 1 ocupa las páginas 16 a 50. Marco Conceptual ronda las 31 páginas y Estado del Arte las 3.
- Meta. Marco Conceptual entre 15 y 17 páginas y Estado del Arte entre 2.5 y 3. Es una reducción cercana al 45 % del marco conceptual, superior al 30 %–40 % inicial porque el tope de 20 páginas incluye el Estado del Arte. Si el resultado cabe en 20 páginas con una reducción menor (mínimo 30 %), no se recorta más.
- Un tema permanece si responde a qué es y para qué se usa en este proyecto, sin describir procedimiento (Metodología) ni valorar resultados (Discusión). Los evaluadores dominan IA y arquitectura de computadores, así que no se explican conceptos estándar.
- Se conserva únicamente la teoría que sostiene decisiones del proyecto. Queda prohibida la teoría huérfana, es decir, la que no se usa después.

## 3. Elementos que deben prevalecer

| Figura | Ubicación final |
|---|---|
| Modelo Roofline conceptual (`fig:modelo-roofline-conceptual`) | 1.1.2 |
| Unidad de Monitorización de Rendimiento (PMU) | 1.1.3 |
| Ámbito de atribución de contadores, núcleo y `uncore` | 1.1.3 |
| Espacios `user-space` y `kernel-space` | 1.1.4 |
| P-states y C-states, integrada con la jerarquía de mecanismos DVFS (una sola figura) | 1.1.5|


Se eliminan las figuras de `perf_event`, RAPL, SMT, dominios GPU, Pareto y validación anidada.

Fórmulas que se conservan. Potencia dinámica CMOS, `eq:roofline`, `eq:ridge`, `eq:intensidad` y las dos ecuaciones del EDP. Se eliminan las ecuaciones de F1, precisión, recall y afines (los evaluadores las conocen).

## 4. Contenido por subsección (Marco Conceptual)

En cada subsección se indican las líneas actuales de `01_marco_referencia.tex` que sirven de fuente, lo que debe permanecer, lo que sale y el enlace con la siguiente.

### 1.1 Marco Conceptual, párrafo de apertura (líneas 3 a 33)

Un único párrafo de unas 6 a 8 líneas que enuncie la cadena de suposiciones del trabajo (régimen de ejecución → traza en contadores → clasificador → política de frecuencia juzgada con EDP). Se elimina el segundo párrafo, que anuncia el orden del capítulo, porque es un preámbulo que anticipa lo que sigue.

### 1.1.1 Potencia CMOS y fundamento de DVFS (líneas 35 a 45, etiqueta `sec:marco-cmos`)

- Permanece. La ecuación de potencia dinámica (proporcional a la capacitancia efectiva, al cuadrado del voltaje y a la frecuencia) y la relación entre voltaje y frecuencia. De ella se deriva por qué DVFS puede ahorrar energía y por qué el ahorro depende de que el tiempo de ejecución no crezca en la misma proporción.
- Sale. La discusión termodinámica extensa del silicio y todo lo que no se use después.
- Enlace. La reducción de frecuencia solo compensa si la carga no está limitada por cómputo, lo que exige saber qué régimen limita cada fase.

### 1.1.2 Modelo Roofline y regímenes de ejecución (líneas 46 a 157, etiquetas `sec:roofline`, `sec:marco-ridge`, `sec:intensidad`)

- Permanece. La figura Roofline, `eq:roofline`, `eq:ridge` (punto de inflexión) y `eq:intensidad`, el criterio de clasificación `compute-bound` y `memory-bound`, y la dependencia del punto de inflexión respecto de la frecuencia y la precisión aritmética (Metodología lo referencia).
- Se fusionan "Punto de inflexión y criterio de clasificación" e "Intensidad operacional observada" en un solo bloque, sin subsubsecciones. Se conservan los tres `\label` (`sec:marco-ridge`, `sec:intensidad`, `eq:ridge`).
- Sale. Derivaciones repetidas y explicaciones básicas del modelo.
- Enlace. La etiqueta de régimen requiere medir FLOPs y bytes, lo que solo es posible fuera de línea y con contadores.

### 1.1.3 Telemetría microarquitectónica con PMU (líneas 158 a 458, etiquetas `sec:marco-telemetria`, `sec:multiplexacion`, `sec:eventos-genericos-crudos`, `sec:uncore`, `sec:ventanas`)

- Permanece. Qué es la PMU y para qué se usa aquí (figura PMU). Distinción entre contadores de núcleo y de `uncore` con la figura de ámbito de atribución. Los tres conceptos con etiqueta referenciada desde Metodología se reducen a párrafos de 3 a 5 líneas cada uno (multiplexación y escalado, eventos genéricos y crudos, naturaleza temporal de la ventana de muestreo). Una sola frase que mencione IPC y MPKI como métricas derivadas de los contadores, sin fórmulas ni interpretación.
- Sale. La subsección de métricas microarquitectónicas de eficiencia, salvo la frase de IPC y MPKI, y cualquier detalle operativo de ventanas que ya esté en Metodología 2.2.4.
- Cautela epistémica. Los contadores describen actividad y no intensidad operacional; no se presentan como mediciones directas de la etiqueta.
- Enlace. Leer contadores depende de en qué espacio de ejecución corre el lector.

### 1.1.4 Espacios de ejecución `user-space` y `kernel-space` (líneas 459 a 534, etiqueta `sec:marco-espacios-ejecucion`)

- Permanece. La figura y un párrafo que explique por qué esta frontera determina la ubicación del agente y las restricciones de lectura de contadores y de actuación.
- Sale. Explicación general de llamadas al sistema y de anillos de protección.
- Enlace. Qué interfaces atraviesan esa frontera.

### 1.1.5 Interfaces de observabilidad (líneas 535 a 654, etiquetas `sec:marco-interfaces`, `sec:perfevent`)

- Permanece. Un párrafo o una tabla corta que indique qué expone cada interfaz (`perf_event` para contadores de CPU, RAPL para energía de paquete y DRAM, NVML y Nsight Compute para GPU). Se conserva la etiqueta `sec:perfevent` en el párrafo de `perf_event`. Se mantiene la cautela de que RAPL mide el paquete completo y no los núcleos reservados.
- Sale. Las figuras de `perf_event` y de dominios RAPL, la subsección de niveles de privilegio (`paranoid`, con etiqueta `sec:paranoid`; no se referencia desde otros archivos) y toda comparación sobre cuál interfaz es mejor (eso se hace en Metodología 2.2.3).
- Enlace. Observar es la mitad del lazo; la otra mitad es actuar sobre la frecuencia.

### 1.1.6 P-states, gobernadores y DVFS (líneas 655 a 881, etiquetas `sec:dvfs-cpu`, `sec:dominios`, `sec:dvfs-gpu`)

- Permanece. P-states y C-states, papel de los gobernadores (`performance`, `powersave` y la política del controlador de frecuencia) y la jerarquía entre solicitud del SO, firmware y hardware, todo en una sola figura. El párrafo de dominios de frecuencia y `SMT` (dos hermanos que solicitan frecuencias distintas y el efecto sobre la frecuencia efectiva) se reduce a 3 o 4 líneas y **conserva la etiqueta `sec:dominios`**, referenciada tres veces desde Metodología. Una nota breve sobre las diferencias de DVFS en GPU (relojes de cómputo y de memoria), con la etiqueta `sec:dvfs-gpu`.
- Sale. Las figuras de solicitudes SMT y de dominios GPU, y la tecnología detallada de cada gobernador.
- Enlace. Con qué señal se decide cambiar de estado, lo que requiere un modelo que infiera el régimen.

### 1.1.7 Cargas de calibración y modelos de clasificación ligera (líneas 882 a 1351, etiquetas `sec:marco-cargas`, `sec:marco-ml-ligero`, `sec:marco-familias-modelos`)

- Permanece. El benchmark se une aquí con el modelo. Un párrafo inicial indica que los kernels y microbenchmarks de calibración generan los datos con la etiqueta de régimen (detalle en Metodología 2.3.2 y 2.3.4). Luego se presenta el sentido de "ligero" (inferencia de baja latencia compatible con un bucle de control) y las familias candidatas (regresión logística como línea base, árbol de decisión, Random Forest, Extra-Trees y XGBoost). Para cada una se dice solo lo pertinente y en cuáles se espera mejor comportamiento con pocas características tabulares y fronteras no lineales, con las cautelas correspondientes.
- Sale. La sección independiente "Cargas de trabajo", las definiciones básicas de árbol y de ensamble, la evaluación del clasificador (F1, precisión, recall y sus ecuaciones), las líneas base y el costo de inferencia como subsecciones (líneas 1352 a 1498). De ese material solo queda una frase que indique que el costo de inferencia se compara junto con la calidad predictiva. Las tres figuras de RF, Extra-Trees y XGBoost se reducen a una figura combinada o se eliminan.
- Enlace. Los modelos tienen hiperparámetros que hay que fijar sin contaminar la evaluación.

### 1.1.8 Hiperparámetros, Optuna y validación (líneas 1499 a 1790, etiquetas `sec:marco-hiperparametros`, `sec:marco-validacion-anidada`)

Esta subsección incorpora Optuna siguiendo el flujo (modelos, luego su ajuste, luego su validación) y absorbe dos subsecciones actuales.

- Permanece. Qué es un hiperparámetro (una frase). Por qué se usa una búsqueda bayesiana con TPE (`Bergstra2011`) implementada en Optuna (`Akiba2019`) y no una rejilla, porque los espacios de RF, Extra-Trees y XGBoost mezclan enteros y reales en varias dimensiones. Un párrafo corto con la etiqueta `sec:marco-validacion-anidada` (referenciada desde Metodología línea 567) que explique la separación por familia de kernel y la anidación (la búsqueda opera solo sobre el conjunto de entrenamiento de cada pliegue externo, de modo que la familia de prueba nunca interviene), y la exclusión de la magnitud que origina la etiqueta del vector de entrada. Una frase que indique que la calidad predictiva y el costo de inferencia son criterios en conflicto (sin formalismo de Pareto).
- Sale. La comparación entre rejilla y muestreo aleatorio, la relación de dominancia y el frente de Pareto, la figura de Pareto y la de validación cruzada anidada, y la subsección de comparaciones estadísticas pareadas (líneas 1791 a 1821), que se traslada a Metodología Fase 4 por decisión del autor.
- No se incluyen números de intentos, valores de F1 antes y después ni espacios de búsqueda. Son metodología y resultados (Seguimiento de Cambios, entrada `F2-XDEV-003`, y `03_resultados.tex` línea 328).
- Verificación previa. El estado actual de `02_metodologia.tex` no menciona Optuna. Si la Fase 2 no describe el procedimiento, se deja anotado como pendiente en el informe final y no se duplica aquí.
- Enlace. Con el modelo elegido se necesita el criterio para juzgar la política que dispara.

### 1.1.9 Producto Energía–Retardo (EDP) (líneas 1822 a 1890, etiqueta `sec:marco-edp`)

- Permanece. Las dos ecuaciones y su papel como criterio con el que se juzga la política DVFS (ahorro de energía que no compensa un aumento equivalente de tiempo).
- Cierra con un párrafo corto sobre los límites de lo que el marco permite afirmar (clasificación binaria respecto a un umbral calibrado, observabilidad asimétrica entre CPU y GPU y costo temporal de actuación). Es el contenido útil de la actual subsección de síntesis (líneas 1863 a 1890), reducido a unas 8 líneas.

## 5. Etiquetas y referencias cruzadas que no se pueden romper

Estas etiquetas se referencian desde archivos externos a `01_marco_referencia.tex` y deben seguir existiendo, o cada referencia debe actualizarse.

| Etiqueta | Referenciada desde | Ubicación final |
|---|---|---|
| `eq:ridge` | Metodología | 1.1.2 |
| `sec:marco-ridge` (2 veces) | Metodología 2.3.4 y 2.3.8 | 1.1.2 |
| `sec:intensidad` | Metodología | 1.1.2 |
| `sec:multiplexacion` (2) | Metodología | 1.1.3 |
| `sec:eventos-genericos-crudos` | Metodología | 1.1.3 |
| `sec:uncore` | Metodología | 1.1.3 |
| `sec:ventanas` (3) | Metodología | 1.1.3 |
| `sec:perfevent` | Metodología | 1.1.5 |
| `sec:dominios` (3) | Metodología | 1.1.6 |
| `sec:marco-validacion-anidada` | Metodología línea 567 | 1.1.8 |

Antes de terminar, ejecuta una búsqueda de todos los `\ref`, `\eqref` y `\pageref` en `00_frontmatter.tex`, `02_metodologia.tex`, `03_resultados.tex`, `04_discusion.tex` y `05_conclusiones.tex`. Compila y confirma que no quedan advertencias `undefined reference`. Si un párrafo referenciado desde Metodología ya no explica lo que Metodología dice que explica, se ajusta la frase de Metodología con la mínima edición posible y se anota en el informe final.

Cada `\label` referenciado desde otros archivos va en la primera frase del párrafo o subsección que conserve el concepto. Los `\label` nuevos no se inventan; se reutilizan los existentes.

## 6. Contenido que sale del marco y destino

| Tema retirado | Destino |
|---|---|
| Métricas de evaluación del clasificador (F1, precisión, recall) | Metodología Fase 2 |
| Líneas base y costo de inferencia como dimensión de evaluación | Metodología Fase 2 |
| Comparaciones estadísticas pareadas | Metodología Fase 4 (decisión del autor) |
| Frente de Pareto y relación de dominancia | Metodología Fase 2 |
| Métricas de eficiencia (IPC, MPKI) y detalle de ventanas | Metodología 2.2.4 |
| Interfaces en detalle (RAPL, `perf_event`, `paranoid`, NVML) | Metodología 2.2.3 |
| Catálogo de cargas | Metodología 2.3.2 |

No hace falta trasladar texto literal a Metodología. Antes de mover algo, se verifica que Metodología no lo tenga ya. Solo se agrega lo que falte y sea indispensable, y se informa al autor.

## 7. Estado del Arte (1.2)

Sección corta y con tono analítico, no de resumen de manual. Se mantiene en 2.5 a 3 páginas. **No se añaden ni se cambian citas** (`Calore2017`, `Guerreiro2019`, `Ali2023`, `Antici2024`, `Simsek2024`, `Wang2024`); no se inventan referencias. Se conservan todas las cifras (por ejemplo 29.6 % de ahorro con 5.2 % de pérdida en GA100, F1-macro de al menos 0.89, 7.82 % con 2.95 %, 22 % con menos de 3 %). Se reorganiza el texto actual, hoy sin subsecciones, en cuatro bloques y se elimina lo repetido.

- **1.2.1 Adaptación de la frecuencia según el comportamiento de la carga.** Calore et al. (una misma frecuencia no es igual de conveniente para todas las regiones, CPU Haswell y GPU K80), Guerreiro et al. (clasificación por sensibilidad a la frecuencia de cómputo y memoria a partir de métricas de una configuración inicial) y Ali et al. (selección automatizada de frecuencia de GPU con modelos de potencia y tiempo). Una sola frase de transición que señale que estos enfoques parten de una caracterización previa y suponen una carga estable.
- **1.2.2 Clasificación de cargas `compute-bound` y `memory-bound`.** Antici et al. (MCBound, F1-macro de al menos 0.89 sobre datos de Fugaku). Se indica con precisión que su clasificación caracteriza trabajos y no gobierna directamente estados DVFS.
- **1.2.3 Actuación dinámica de frecuencia durante la ejecución.** Simsek et al. (SPH-EXA, reducción de hasta 7.82 % por GPU con 2.95 % de degradación, dependiente de la instrumentación del código) y Wang et al. (DRLCAP, aprendizaje por refuerzo, 22 % de energía de GPU con menos de 3 % de degradación, orientado solo a GPU).
- **1.2.4 Brecha identificada y posición del trabajo.** Lo que queda abierto: integrar clasificación y control de baja intrusión en CPU y GPU. Se conserva el argumento de asimetría de observabilidad y de mecanismos entre CPU y GPU. Se elimina el párrafo que recapitula uno a uno los seis trabajos ya presentados, o se reduce a 2 líneas.

Se conservan las cautelas actuales (no atribuir a un trabajo lo que no hizo). Sin frases del tipo "diversos estudios señalan".

## 8. Parámetros de redacción (obligatorios en todo el capítulo)

Se examina el texto completo de cada subsección antes de modificar oraciones sueltas. Se conservan siempre los datos duros, las cifras, los nombres propios, las citas y las relaciones de causalidad.

### 8.1 Patrones de redacción artificial que se deben eliminar

1. Falsos contrastes del tipo «no X, sino Y» cuando se niega un planteamiento ficticio solo para resaltar la afirmación siguiente. Se expone la idea principal directamente.
2. Cierres enfáticos y fragmentos dramáticos. Se eliminan oraciones finales que solo reiteran lo dicho o recalcan conclusiones evidentes sin datos nuevos.
3. Falsa profundidad y aforismos («el corazón del problema», «en última instancia»). Se reemplazan por la formulación técnica exacta.
4. Preámbulos innecesarios («conviene comenzar señalando», «a continuación se detalla») cuando la idea puede enunciarse directamente.
5. Objeciones no formuladas. No se descartan alternativas que nadie ha planteado, salvo que formen parte de una justificación metodológica documentada.
6. Enumeraciones forzadas en tríadas que no respondan a una necesidad conceptual. Se conservan los elementos con diferencias reales o se formula una síntesis en prosa.
7. Comienzos repetitivos. Se alterna la estructura oracional para no encadenar frases con el mismo sujeto o pronombre.
8. Calificativos exagerados y vocabulario inflado («crucial», «clave», «robusto», «meticuloso», «panorama», «transformador») salvo que correspondan a definiciones matemáticas o técnicas formales.
9. Atribución vaga de autoridad («diversos estudios señalan», «los expertos coinciden») en ausencia de una cita bibliográfica puntual.
10. Gerundios finales («garantizando», «demostrando», «reflejando») que agreguen juicios no demostrados en la frase principal.
11. Residuos conversacionales o de asistencia (cortesía, saludos, descargos de conocimiento, preguntas al usuario) dentro del cuerpo del texto.

### 8.2 Estilo

- Evitar tono de manual o de clase introductoria. No explicar conceptos estándar más de lo necesario. No acumular definiciones antes de mostrar su utilidad.
- Evitar frases que anuncien lo que se explicará después. Evitar repetir la misma relación con conectores diferentes.
- No usar metáforas para explicar relaciones técnicas cuando una formulación directa basta.
- Preferir verbos concretos como `mide`, `define`, `calcula`, `requiere`, `limita`, `depende`, `permite`, `observa` o `clasifica`.
- Evitar «permite comprender», «resulta fundamental», «es importante señalar», «cabe destacar», «constituye un aspecto clave» y equivalentes que no añadan contenido.
- Mantener las cautelas técnicas cuando una relación no sea directa. No convertir correlaciones, proxies o señales indirectas en mediciones directas. Conservar limitaciones y condiciones de validez aunque el texto se compacte.
- Usar artículos y determinantes cuando ayuden a que la frase suene escrita y no telegráfica (por ejemplo «de la frecuencia y del voltaje» y no «de frecuencia y voltaje» sistemáticamente).
- Favorecer verbos relacionales como `depende de`, `responde a`, `se asocia con` o `queda condicionado por` sobre verbos genéricos como `cambia`, `presenta` o `muestra`.
- Enlazar oraciones con referentes ya introducidos («esta relación», «ese comportamiento», «dicha condición», «a partir de ello») siempre que el antecedente sea inequívoco.
- Alternar oraciones breves con otras de desarrollo medio. Evitar que todos los enunciados tengan la misma longitud y estructura sujeto + verbo + complemento.
- Variar la sintaxis con subordinadas causales, relativas y comparativas cuando la relación conceptual lo justifique.
- Evitar secuencias repetidas de «X permite, Y permite, Z permite». Reescribir parte de ellas con relaciones causales o dependencias explícitas.
- En párrafos introductorios o de hoja de ruta, los conectores de secuencia («primero», «después», «a partir de ahí») se usan de forma puntual, sin enumeración rígida ni «en primer lugar / en segundo lugar / por último» mecánicos.
- Al comparar dos condiciones, preferir una comparación desarrollada si mejora la precisión (por ejemplo «no responde de la misma forma que una carga condicionada por memoria»), evitando oposiciones tan cortas que parezcan titulares.
- No cerrar cada párrafo con una conclusión enfática. Cuando el concepto siguiente dependa del anterior, usar el cierre para abrir esa dependencia.
- No sustituir términos técnicos consolidados por sinónimos (`Roofline`, `DVFS`, `PMU`, `uncore`, `EDP`, `compute-bound`, `memory-bound`). La variación recae en la sintaxis y en los verbos.

### 8.3 Función del marco

- El Estado del Arte y el Marco Teórico son antecedentes analizados críticamente y fundamentación conceptual directa. No funcionan como manual introductorio general. Se conserva exclusivamente la teoría que sostiene decisiones del proyecto. Queda prohibido citar teoría huérfana.
- Cada subsección justifica una pieza del proyecto. `Roofline` explica el criterio físico de clasificación. `PMU`, `uncore`, `RAPL` y `NVML` explican qué puede observarse y con qué limitaciones. `user-space` explica la ubicación y las restricciones del agente. El ML ligero explica por qué la inferencia debe tener baja latencia. La validación explica cómo evitar una estimación optimista del clasificador. El `EDP` explica el criterio con el que se juzga la política DVFS.

### 8.4 Sintaxis y puntuación

- Enlazar ideas mediante subordinación natural, comas o punto y coma, evitando la fragmentación telegráfica y las oraciones desmedidas.
- Queda prohibido el uso de guiones o rayas como recurso de acotación o conector universal (por ejemplo «- ... -» o «--- ... ---»). Los incisos se resuelven con comas, paréntesis o punto y coma. Se exceptúan los rangos numéricos y los guiones de palabras compuestas (por ejemplo «CPU--GPU», «Energía--Retardo» tal como ya se escriben en LaTeX).
- No usar `:` en la prosa académica salvo cuando la oración introduzca directamente una ecuación. Las enumeraciones, aclaraciones y relaciones se reformulan como oraciones completas. Se exceptúan los elementos técnicos cuya sintaxis requiere dos puntos (`\label{sec:uncore}`, `\ref{sec:ventanas}`, rutas, identificadores, código, formatos de herramientas).

## 9. Orden de trabajo

.
2. Recompilar `main.tex` y medir las páginas reales por subsección para confirmar la meta.
3. Leer los archivos de la sección 1 y verificar qué cubre ya Metodología (sección 6).
4. Reescribir el capítulo subsección por subsección en el orden de la sección 4, y después el Estado del Arte (sección 7).
5. Al terminar **cada** subsección, releerla contra la sección 8 (patrones, estilo, función del marco y puntuación) y corregir antes de pasar a la siguiente. Se vuelve a buscar, como mínimo, dos puntos en la prosa, guiones o rayas como inciso, gerundios finales y los términos prohibidos.
6. Ajustar las referencias cruzadas de la sección 5 y compilar sin advertencias de `undefined reference` ni de citas faltantes.
7. Medir la extensión (Marco Conceptual entre 15 y 17 páginas, capítulo 1 en 20 o menos) y comprobar que los subtítulos se leen como una explicación continua.
8. Revisión final del capítulo completo contra la sección 8, buscando redundancias entre subsecciones y verificando que ninguna figura o ecuación conservada quedó huérfana (referenciada, con leyenda y con fuente).
9. Informe final al autor con las páginas antes y después, lo que se movió a Metodología, las referencias de Metodología que se ajustaron y lo que quedó pendiente (por ejemplo, Optuna en la Fase 2).

## 10. Criterios de aceptación

- Estructura exactamente igual a la sección 0, con numeración 1, 1.1, 1.1.1 a 1.1.9 y 1.2, 1.2.1 a 1.2.4.
- Marco Conceptual entre 15 y 17 páginas y capítulo 1 en 20 o menos.
- Se conservan las figuras Roofline, PMU, ámbito de atribución, `user-space`/`kernel-space` y P-states/DVFS, y las fórmulas CMOS, `eq:roofline`, `eq:ridge`, `eq:intensidad` y EDP.
- IPC y MPKI aparecen en una sola frase dentro de 1.1.3.
- Optuna aparece integrado en 1.1.8 con `Akiba2019` y `Bergstra2011`.
- Estadística y evaluación del clasificador quedan fuera del marco, en Metodología.
- Sin referencias rotas, sin citas nuevas, sin duplicación con Metodología y sin cifras de las Fases 2 a 4 en el marco.
- Sin dos puntos, guiones ni rayas de acotación en la prosa y sin los patrones de la sección 8.
