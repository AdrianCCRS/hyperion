# Revisión del libro de tesis (versión final), 2026-09-29

Documento revisado: `docs/libro/main.tex` y `docs/libro/secciones/*.tex` (76 páginas compiladas).
Nivel de exigencia: trabajo de grado de **pregrado** (UIS). No se exige lo que se le exigiría a un artículo de revista; se señala lo que un jurado podría objetar o lo que es inconsistente dentro del propio documento.

**Excluido a propósito:** la ausencia de resultados *powersave* en la matriz inicial, E-A, D y CloverLeaf (job 7789 pendiente). Ese hueco no se reporta. Sí se reporta una contradicción del texto *ya escrito* sobre el experimento E-B con *powersave* (hallazgo M13), porque no depende de los resultados que faltan.

Cómo leer las ubicaciones: `03:446` significa `secciones/03_resultados.tex`, línea 446.

---

## 0. Procedencia de la revisión (léase primero)

- El "panel" de cinco revisores se simuló **en un solo contexto y con un solo modelo** (Claude Opus 5.5). Los cinco puntos de vista son separación de roles, no revisiones independientes: comparten sesgos y pueden coincidir en un error.
- No se ejecutaron los scripts del protocolo del skill (contrato de sprint, procedencia del panel, calibración). La revisión es `NOT_CALIBRATED`.
- Tres afirmaciones se verificaron contra el código antes de reportarlas:
  - **Descartada:** "fracción de ciclos detenidos por memoria" es correcta. `common/telemetry/src/perf_reader.cpp:73-83` usa `CYCLE_ACTIVITY.STALLS_MEM_ANY`.
  - **Confirmada:** el modelo GPU se entrenó con agregados de corridas completas y en el agente se aplica a la ventana de cada episodio (`fase3_daemon/gpu_loop_cpp/include/gpu_window_classifier.hpp:17-20`). Ver M3.
  - **Confirmada en parte:** la energía de CPU de la Fase 4 suma los paquetes RAPL de los dos zócalos (`scripts/pacca/hyp_cloverleaf_confirm.sbatch:24-25,91-99`, `hyp_fase4_F_f1compute.sbatch:22`). No verifiqué el script de la matriz inicial ni los dominios de la Fase 2. Ver M7.

---

## 1. Configuración del panel (Fase 0)

| Asiento | Perspectiva asignada |
|---|---|
| Ajuste al programa | Jurado de pregrado en Ingeniería de Sistemas: cumplimiento de objetivos, estructura UIS, coherencia pregunta, objetivos y conclusiones |
| R1 Metodología | Diseño experimental, validez estadística, reproducibilidad (HPC y ML aplicado) |
| R2 Dominio | Gestión de energía y DVFS en CPU/GPU, modelo Roofline, telemetría PMU/NVML |
| R3 Perspectiva | Uso práctico en un clúster compartido, operación real de un agente, validez externa |
| Abogado del diablo | Ataca el argumento central: ¿demuestra el trabajo lo que dice demostrar? |

---

## 2. Decisión editorial

**Aprobable con correcciones.** Casi todas se resuelven en el texto y **no requieren experimentos nuevos**. Solo dos correcciones piden un análisis adicional sobre datos que ya existen (C1, M6).

El trabajo es sólido para un pregrado: valida el instrumento, usa validación por familia en vez de mezclar intervalos, tiene un brazo sombra, no oculta resultados negativos y no afirma significancia donde no la hay. Los problemas son de tres tipos:

1. **El diseño confirmatorio no podía dar significancia** y el texto no lo dice (C1).
2. **Lo que el agente hace en línea no es exactamente lo que se validó fuera de línea**: detección de fase por inactividad, y entrenamiento con corridas completas frente a inferencia sobre ventanas de 3 s (C2, M3).
3. **Contradicciones internas** entre Metodología y Resultados, o entre párrafos de Resultados (M5, M6, M11, M13, M14 y varios menores).

---

## 3. Hallazgos, ordenados por severidad

Severidad:
- **Crítico:** un jurado atento lo usaría para cuestionar la conclusión principal.
- **Mayor:** error o contradicción que debe corregirse.
- **Menor:** claridad, consistencia o forma.

### Críticos

#### C1. El diseño confirmatorio no podía alcanzar significancia al nivel 0.05
> **Estado (2026-09-29): corregido con el arreglo recomendado.** Se declaró el $p$ mínimo y se añadió la media geométrica por bloque con IC95 (*t* pareada sobre el logaritmo), marcada como análisis posterior a la medición. Script: `docs/libro/scripts/ic_efecto_bloques_20260929.py`; salida: `docs/libro/datos/fase4_20260929/ic_efecto_bloques.csv`.

- **Dónde:** `02:448` (prueba de signos exacta bilateral por bloques); `03:446` y `03:489` ($p=0.0625$ con 5 de 5 bloques favorables); conclusiones (`05`, párrafo 4); resumen y abstract.
- **Problema:** con 5 bloques y prueba de signos bilateral, el $p$ más bajo posible es $2\cdot(1/2)^5 = 0.0625$. Aunque los cinco bloques favorezcan al agente, el experimento **no puede** llegar a 0.05. El texto presenta "no alcanzó significancia" como un resultado, cuando es una propiedad del diseño elegido. El objetivo específico 4 pide "determinar estadísticamente si el ahorro compensa la sobrecarga". Tal como está, un jurado puede decir que el diseño impedía responderlo.
- **Arreglo mínimo (solo texto):** en Metodología, declarar que con 5 bloques el $p$ mínimo es 0.0625 y que la prueba de signos se usa como evidencia de dirección, no de significancia. En Resultados y Conclusiones, decir "5 de 5 bloques favorables, el máximo posible con este diseño".
- **Arreglo recomendado (análisis sobre datos existentes):** reportar el tamaño de efecto con intervalo, por ejemplo la media del log-cociente activo/base por bloque con IC95 por *t* pareada o *bootstrap* de bloques. Eso responde "cuánto" y no solo "hacia dónde".
- **Opción fuerte (experimento):** con 6 bloques el $p$ mínimo baja a 0.031. Solo si hay tiempo de nodo.

#### C2. La "detección de fase" de GPU es detección de episodios de actividad; los dos resultados favorables no aíslan el valor del agente dinámico
- **Dónde:** `02:298`, `02:356` (fases separadas por 1 s de reposo); `03:285` ("el agente decide una sola vez por fase"); `03:487` (CloverLeaf: una sola decisión a los ~3 s y 1260 MHz durante toda la corrida); `03:450` (brazo F1 fijo de E-A inválido). Código: `gpu_activity_tracker.hpp:12-19`.
- **Problema:** el agente GPU no detecta cambios de régimen dentro de una actividad continua. Abre un episodio cuando la utilización de SM cruza 5%, decide una vez a los 3 s y mantiene esa decisión hasta que la utilización cae. En las aplicaciones compuestas, el reposo de 1 s entre fases hace que **cada frontera de fase coincida por construcción con una frontera de episodio**. En CloverLeaf, que es continua, hubo **una sola decisión** y el reloj quedó en 1260 MHz toda la corrida: el agente se comportó como un candado estático en F1. Como el brazo F1 fijo de E-A no fue válido, **ninguno de los dos resultados favorables muestra que la clasificación dinámica aporte algo sobre fijar F1 en toda la aplicación.** El libro reconoce lo del F1 fijo, pero no la equivalencia de CloverLeaf con un candado estático, ni que las fases de las compuestas se separan por inactividad.
- **Arreglo:**
  - Metodología §Fase 3: decir explícitamente que la unidad de decisión de GPU es el episodio de actividad y que las compuestas insertan 1 s de reposo para que cada fase sea un episodio.
  - Resultados CloverLeaf: una frase que diga que, al decidir una sola vez, el efecto medido equivale al de fijar F1 en toda la ejecución.
  - Discusión y Limitaciones: decir que no se evaluó un cambio de régimen dentro de un episodio continuo.
  - Resumen e Introducción: moderar "clasifica la fase de ejecución" para GPU.
  - Unificar "fase" y "episodio" (`03:285` dice "una vez por fase").

#### C3. En CPU, el instrumento ya calcula la etiqueta en línea; falta justificar por qué el agente usa un clasificador de 0.728
> **Estado (2026-09-29): corregido.** Párrafo nuevo en Metodología §Formulación (uncore de ámbito de zócalo, nodo exclusivo, `CAP_PERFMON`; en GPU, Nsight Compute ~60 veces más lento en el GEMM de RAJAPerf, según `Seguimiento_Cambios_Plan_Director.md:3467`). El Marco remite a ese párrafo, Resultados acota "los contadores" a "las seis variables", y Conclusiones añade el aporte de bajo costo del agente de GPU.

- **Dónde:** `01:69` ("sin reconstruir en línea la intensidad operacional"); `02:111` ("$I$ se calcula por intervalo durante la ejecución"); `03:190` (con $I$ como entrada, 0.9989); `03:175` y `03:190` ("el límite proviene de la información de los contadores").
- **Problema:** el instrumento de la Fase 1 mide FLOPs (`FP_ARITH_INST_RETIRED`) y bytes de DRAM (*uncore*) por intervalo de ~10 ms mientras la carga corre. Un jurado preguntará por qué el agente no calcula $I$ y lo compara con el *ridge* (exactitud ≈ 1 por definición) en vez de usar un modelo con 0.728. Puede haber buenas razones (el *uncore* es de todo el zócalo y no se puede atribuir a un proceso en un nodo compartido, requiere privilegios, cuesta más), pero **el libro no las da**. La frase de `01:69` se afirma sin justificar.
- **Arreglo:** 3 a 5 líneas en el Marco (§telemetría) o en Metodología (§Fase 2) que expliquen la restricción operativa que impide usar $I$ en producción. Cambiar también "la información contenida en los contadores" (`03:175`, `03:190`) por "la información de estas seis variables": los contadores de FLOPs sí transportan esa información, pero están excluidos a propósito.

### Mayores

#### M3. El modelo GPU se entrena con corridas completas y se aplica a los primeros 3 s de cada episodio
- **Dónde:** `02:160-162` (unidad GPU = corrida completa, sin calentamiento); `03:233`; `02:298`; código `gpu_window_classifier.hpp:17-20` ("se aproximan con las muestras de la fase actual").
- **Problema:** las medianas del entrenamiento excluyen el transitorio inicial. En el agente, la ventana de 3 s está justo al inicio del episodio, donde ocurre ese transitorio. Es un cambio de distribución entre entrenamiento e inferencia que el libro no declara. Puede explicar las abstenciones con confianza 0.56 a 0.67 en BabelStream (`03:452`). Además, 0.819 y 0.900 son exactitudes fuera de línea, no la del agente en línea.
- **Arreglo:** declararlo en Metodología §Fase 3 y en Limitaciones. En Conclusiones, separar la exactitud fuera de línea (LOFO) de la observada en línea (Fase 4, conteos pequeños).

#### M4. La política GPU se adoptó con la estimación dentro de muestra; la validación fuera de muestra incluye el cero
- **Dónde:** `02:241` (regla: la mejora debe estar "respaldada estadísticamente" y "se valida dejando una familia fuera"); `03:313`, `03:320` (LOFO: 6.5%, IC95 de $-4.0$ a $13.1$); `03:331` (en cómputo se usa el LOFO de $-4.8\%$ para **rechazar**); `03:412` y resumen (se cita 8.9%).
- **Problema:**
  1. Con la regla que declara Metodología, F1 en memoria no pasa la validación por familia. Se adoptó de todos modos, y el texto no dice que se hizo una excepción.
  2. El LOFO se usa en un sentido para cómputo (rechazar) y se ignora para memoria (aceptar).
  3. Hay comparaciones múltiples (unos 9 niveles × 2 clases) sin corrección. $p=0.020$ no sobrevive un Bonferroni por 9 (0.0056).
  4. El resumen da la cifra más favorable (8.9%, dentro de muestra).
- **Arreglo:** poner el 6.5% fuera de muestra al lado del 8.9% en el resumen y en `03:412`. Declarar en Resultados que F1 se adoptó con evidencia fuera de muestra no concluyente, como decisión de diseño, y justificarla (7 de 8 familias, dirección consistente). Una línea sobre la multiplicidad.

#### M5. El umbral de abstención de CPU: Metodología describe un procedimiento que no se usó
- **Dónde:** `02:221` y Figura `fig:met-lofo` (τ se elige con un LOFO interno maximizando EB con cobertura ≥ 0.60); `03:158` ("Es una decisión de diseño posterior al resultado"; 0.90 daba +0.003 con 6.5 puntos menos de cobertura, 65.7%, que cumple ≥ 0.60).
- **Problema:** la regla declarada habría elegido 0.90 o más, no 0.85. Metodología y Resultados se contradicen. Resultados es honesto y muestra que el impacto es mínimo (por pliegue da 0.734), pero Metodología describe algo que no ocurrió.
- **Arreglo:** que Metodología diga que el umbral de CPU se fijó en 0.85 como decisión de diseño que prioriza la cobertura, y que el LOFO interno se usó como verificación. O aplicar la regla y reportar 0.90.

#### M6. La unidad de la política de CPU es el kernel, pero la tabla de amenazas dice "familia"
- **Dónde:** `02:241` (*bootstrap* de kernels); caption de `03:205` ("cómputo: 19 kernels; memoria: 28"); `02:465` (tabla de amenazas: "la familia como unidad de validación **y de comparación de la política**"); `03:219` (la dispersión entre familias domina).
- **Problema:** es una contradicción directa. Además, en CPU las variantes de tamaño de un mismo algoritmo cuentan como réplicas (pseudo-replicación), justo lo que la tabla de amenazas dice controlar. En GPU sí se usó la familia. Es poco probable que cambie la conclusión (los efectos en CPU son negativos), pero el texto es inconsistente.
- **Arreglo:** rehacer el *bootstrap* de CPU por familia (análisis sobre datos existentes) o corregir la tabla de amenazas para que diga que en CPU la unidad fue el kernel y explicar por qué.

#### M7. "EDP del nodo" y la energía de CPU: dominios sin declarar e inconsistentes entre fases
- **Dónde:** `02:438-441`; caption de `03:215` (Fase 2: "potencia de paquete y DRAM"); `03:210` (84%); `03:448` ("la CPU consume el 64%"). Código de la Fase 4: paquetes RAPL de los dos zócalos, sin DRAM.
- **Problema:**
  1. El agente actúa sobre 6 núcleos físicos (12 lógicos) de 16, pero la energía de CPU suma **los dos paquetes completos**. El piso del 84% y el 64% incluyen núcleos no delegados y un segundo zócalo cuyo reloj el agente no toca. Eso infla la fracción "que no depende de la frecuencia" y hace que la conclusión "DVFS de CPU no paga" dependa de esta configuración parcial. El resumen la presenta como general.
  2. Al parecer, la Fase 2 incluye DRAM y la Fase 4 no (verificar).
  3. "EDP del nodo" es CPU+GPU; no incluye placa, ventiladores ni fuente.
- **Arreglo:** en Metodología (Plataforma o Medición), declarar los dominios RAPL de cada fase y qué núcleos se escalan. En el resumen y en Conclusiones, acotar: "con la frecuencia aplicada solo a los núcleos delegados, cerca del 84% de la potencia de paquete se mantiene". Renombrar a "EDP de CPU+GPU" o justificar el nombre en una frase.

#### M8. La parte de CPU del agente se recomienda desactivar, pero el resumen y la introducción la presentan como parte del agente
- **Dónde:** `03:448` ("Conviene, por tanto, un agente de GPU sin agente de CPU"); `03:394-398` (el agente de CPU cuesta de 5 a 9% de energía de CPU y su sombra ya está en 1.10 de EDP); resumen, introducción `00:136` ("ajusta la frecuencia de CPU y GPU").
- **Problema:** el resultado práctico es que la parte de CPU no aporta y cuesta, y la política de CPU es no actuar. La pregunta de investigación incluye "sin que el costo de la inferencia degrade el rendimiento". Para CPU la respuesta es negativa, y en ningún lado se dice así de claro.
- **Arreglo:** en Conclusiones, una frase explícita: el componente de CPU se implementó y se evaluó, su costo de observación supera cualquier ganancia posible con la política derivada, y se recomienda operar solo el de GPU. En el resumen, media frase.

#### M9. Las conclusiones no responden la pregunta de investigación ni cierran los cuatro objetivos
- **Dónde:** `05` completo; pregunta `00:158`; objetivos `00:170-186`.
- **Problema:** las conclusiones resumen hallazgos por dispositivo, pero no dicen "el objetivo 1 se cumplió con…, el 2…", ni responden "¿en qué medida…?". Un jurado de pregrado lo busca primero.
- **Arreglo:** un párrafo breve por objetivo y uno que responda la pregunta, con los matices de C1, C2 y M8. Algunos objetivos se cumplieron con salvedades que conviene declarar: el objetivo 3 dice "políticas proactivas" y el agente es reactivo (decide tras 3 s de actividad); el objetivo 4 dice "determinar estadísticamente" (ver C1).

#### M10. Latencia de decisión sin explicar: 13 a 16 s en E y ~3 s en CloverLeaf
- **Dónde:** `03:426` ("La primera decisión de GPU llegó entre 13 y 16 s después del inicio de cada fase de memoria (la ventana de observación sostenida es de 3 s)"); `03:487` ("tras aproximadamente 3 s").
- **Problema:** el texto no explica de dónde salen los 10 a 13 s adicionales. De esa latencia dependen el "65 a 78% de la fase en F1" y la conclusión de que el agente necesita fases largas. Si la causa es, por ejemplo, que cada fase empieza con preparación en el host antes de ocupar la GPU, eso cambia la interpretación: el agente no sería lento, la GPU todavía estaría ociosa.
- **Arreglo:** una frase con la causa, o declararla sin identificar.

#### M11. Escenario C: dos frases se contradicen
- **Dónde:** `03:398` ("Usar F0 también en *memory* no cambia el resultado de forma apreciable") frente a `03:407` ("bajar a F1 en memoria en CPU con kernels vistos dio 1.154 de EDP frente a 1.085 al mantener F0") y la misma `03:398` ("bajar el reloj en memoria empeora el EDP 5%").
- **Problema:** una diferencia de unos 6 puntos entre F1 y F0 en memoria no es "no apreciable".
- **Arreglo:** reescribir la frase de `03:398` para que diga qué comparación "no cambia" (¿activo-F0 frente a sombra?).

#### M12. Costo del agente de CPU: no queda claro con qué configuración de hilos se midió
- **Dónde:** `03:398` ("El agente de CPU usa por defecto un único hilo de inferencia. Con el conjunto de hilos por defecto de ONNX Runtime, cada hilo espera de forma activa…"); Discusión `04:19` y Conclusiones citan "entre 5 y 9%"; E-A exploratorio `03:448` (3.4%); LAMMPS `03:471` (2.9 a 3.4%).
- **Problema:** el lector no sabe si el 5 a 9% se midió con el conjunto de hilos por defecto, que espera de forma activa (un defecto ya corregido), o con un hilo. Si fue lo primero, la Discusión cita un costo que ya no es el del agente actual.
- **Arreglo:** indicar la configuración de cada cifra. Si la matriz inicial usó el conjunto por defecto, decirlo y citar en la Discusión las cifras medidas después de la corrección.

#### M13. Contradicción sobre el turbo en el E-B con *powersave* ya reportado
*(Es texto existente, no los resultados que faltan.)*
- **Dónde:** `02:7` y Limitaciones `05` ("en ambos casos el turbo de CPU permanece desactivado"); `03:464` ("La REF sin turbo bajo *powersave* queda en 1.005…", lo que implica que la REF de ese experimento tenía turbo activo; en el registro del proyecto, el job 7779 corrió con `no_turbo=0`); `03:462` ("los mismos cinco brazos del escenario E", que no se enumeran en ningún lugar).
- **Arreglo:** declarar el estado del turbo del E-B con *powersave* ya reportado y aclarar que la serie nueva (7789) sí corre con turbo apagado. Enumerar los cinco brazos de E en Metodología (Tabla del mapa de escenarios).

#### M14. La REF de las Fases 1 y 2 no está definida
- **Dónde:** `02:7` define solo la base de la Fase 4; `03:9`, `03:44` ("nivel nativo", "referencia nativa"); `03:200` ("F0 mejora 0.56% frente a la base"); `02:340` (acción de CPU en cómputo: "Fija F0 (3.2 GHz)").
- **Problema:** no se dice qué gobernador ni qué estado de turbo tenía la REF de las campañas de las Fases 1 y 2. Si era *performance* sin turbo, entonces REF ≡ F0 (3.2 GHz), el 0.56% es diferencia entre dos estados idénticos, y la acción "fijar F0" es nula frente a la base de la Fase 4. Si era otra cosa, falta decirlo.
- **Arreglo:** una frase en Metodología §Matriz experimental que defina la REF de las Fases 1 y 2.

### Menores

| # | Dónde | Problema | Arreglo |
|---|---|---|---|
| m1 | `03:54`, `03:80`, `03:452`, `03:462` | Núcleos delegados descritos como "seis", "cuatro", "0 a 5" y "12" en distintos lugares | Definirlos una vez en la Tabla de plataforma (p. ej. 6 físicos + 6 hermanos SMT = 12 lógicos, zócalo X) |
| m2 | `02:177`, caption de `fig:met-lofo` | "La familia retirada no interviene en ninguna decisión… ni en la selección de representación", pero la elección entre 6 y 12 variables (`03:74`), el modelo (`03:80`) y la variante GPU de 3 señales (`03:251`, justo la de mayor EB) se decidieron con el mismo LOFO externo | Matizar: la selección entre pocas variantes usó el LOFO externo; el sesgo es pequeño porque las diferencias lo son |
| m3 | `03:80` | "la regresión logística rinde menos", pero su F1 macro (0.765) es la más alta de la lista | Aclarar que "rinde menos" se refiere a la EB por celda |
| m4 | `03:251`, Conclusiones | "Tres señales invariantes": la prueba solo excluyó reloj y potencia; no se mostró que la utilización y su dispersión no cambien con el reloj fijado | Usar "señales que no dependen directamente de la acción" o mostrar su distribución por nivel |
| m5 | `03:383` | "La exactitud del brazo activo y la del sombra son indistinguibles, como corresponde a un clasificador cuya entrada no depende de la acción": es circular, porque la decisión se toma antes de actuar | Quitar la inferencia o reformular |
| m6 | `03:285`, `03:353`; Trabajo futuro | Nunca se reporta la latencia de inferencia de GPU dentro del agente en C++ (solo en Python, 4.7 ms) | Reportarla si existe o declararla como no medida |
| m7 | `02:316`, `03:353` | "Permanencia mínima 3.7 s" aparece como parámetro, pero no se explica qué hace | Una frase: qué restringe y cuándo aplica |
| m8 | `02:356`, Tabla `tab:agente-compuestas` | "Una tercera aplicación alterna trabajo CPU y GPU", pero la tabla no la describe y se evalúa en el alcance conjunto | Añadir su fila |
| m9 | `03:342` | "Antes de esa verificación" sin antecedente (es el inicio de la sección) | Reescribir la frase inicial |
| m10 | Tabla `tab:fase4-mapa` | Escenarios C, D, E, F sin A ni B (A y B son aplicaciones), lo que se confunde con E-A y E-B | Una nota que explique el esquema de letras |
| m11 | Tabla `tab:fase4-matriz` | "3 (5 en la matriz inicial)", pero la tabla *es* la matriz inicial | Aclarar a qué corresponde cada número |
| m12 | `01:18`, `03:219`, `03:322` | $\alpha$ se usa como factor de actividad CMOS y como pendiente de $\log T$ frente a $\log f$ | Renombrar la pendiente (p. ej. $\beta$) |
| m13 | Discusión `04:11`, Conclusiones | "La diversidad de familias influyó más que cambiar el algoritmo": la propia curva (`03:188`) dice que el diseño no separa el número de familias del tamaño de la muestra, y que la ganancia es pequeña | Añadir esa salvedad en las Conclusiones |
| m14 | Resumen `00:72`, abstract `00:88` | "a partir de telemetría de hardware (contadores PMU y RAPL en CPU…)": RAPL no es entrada del clasificador, solo mide energía | "infiere el régimen con contadores PMU (CPU) y NVML (GPU); la energía se mide con RAPL y NVML" |
| m15 | Marco `01:167-174` | El Marco explica `perf_event_open` y RAPL, pero no NVML ni el bloqueo de reloj (`nvidia-smi -lgc`), que son la fuente y el actuador de todo el lado GPU | 2 o 3 frases sobre NVML en §interfaces |
| m16 | `01:38`, `01:188`, casi todas las figuras | Mezcla de "Nota. Tomado de…", "Fuente: Tomado de…" y "Fuente: elaboración propia." La guía UIS usa "Nota" y no pide referencia en figuras propias | Unificar el formato con el que pida el director |
| m17 | `03:450` | El brazo F1 fijo se descartó por salir de 1230 a 1290 MHz en ~21% de las muestras; no se dice si las fases de memoria del brazo activo pasaron el mismo criterio | Declarar si el activo se verificó igual; si no, es una aceptación asimétrica |
| m18 | `03:383` | "Toda fase recibió al menos una decisión" cuenta las abstenciones como decisiones | "al menos una salida del clasificador (clase o abstención)" |
| m19 | Planteamiento, `00` ("A nivel nacional y regional…") | Afirmación sin cita; la propuesta citaba [1]–[5] solo para el contexto internacional | Citar, suavizar ("no se identificaron…") o quitar (ya señalado antes) |
| m20 | `03:322` frente a `03:260` y `03:320` | *rodinia_gaussian* es "*memory_bound* sin ambigüedad" y a la vez "la familia más cercana al *ridge*" | Explicar que la etiqueta es estable aunque esté cerca del *ridge* |

---

## 4. Lo que cada revisor destacaría

### Ajuste al programa (jurado de pregrado)
- **A favor:** alcance y rigor por encima de lo habitual en un pregrado. Cuatro fases completas con instrumento propio, validación por familia, código y datos públicos (`02:167`), y honestidad con los resultados negativos.
- **En contra:** las Conclusiones no cierran los objetivos ni responden la pregunta (M9). El título y el resumen prometen un agente CPU–GPU, pero el resultado recomienda solo GPU (M8).

### R1 Metodología
- **A favor:** separación correcta de las unidades de generalización (`03:115`: 0.874 mezclando intervalos, 0.733 dejando fuera un kernel y 0.728 dejando fuera una familia); abstención aleatoria como control (`03:158`); comprobación de equivalencia ONNX (`03:342`); validación independiente de los techos con Advisor (`03:54`); verificación de la frecuencia efectiva con el hallazgo de SMT (`03:14`); control de repetibilidad de la base (`03:398`).
- **En contra:** C1, M4, M5, M6, m2 y m17. La más seria es C1, porque afecta al objetivo 4.

### R2 Dominio
- **A favor:** el análisis del piso de potencia y de la pendiente $\alpha$ (`03:219`, `03:335`) explica de forma física por qué la CPU no gana y la GPU sí; el retiro de *hotspot* por precisión mixta (`03:228`) es un buen ejemplo de disciplina sobre la etiqueta.
- **En contra:** C3 (por qué no usar $I$ en línea en CPU), M7 (dominios RAPL y núcleos parciales condicionan la conclusión de CPU), m15 (NVML ausente del Marco).

### R3 Perspectiva (operación real)
- **A favor:** restauración ante terminación anómala verificada (`03:355`); el brazo sombra separa el costo de observar del de actuar, algo que muchos trabajos no hacen.
- **En contra:** en uso real (LAMMPS) el agente no actuó, y en una aplicación continua (CloverLeaf) equivale a fijar F1 (C2). La latencia de decisión de 13 a 16 s (M10) limita qué fases se pueden aprovechar. Para un administrador del clúster, la recomendación práctica, "fijar 1260 MHz cuando la carga es de memoria", no requiere el clasificador si ya se sabe qué aplicación corre. Conviene que la Discusión diga en qué escenario el clasificador sí es necesario: cargas desconocidas o mezcladas.

### Abogado del diablo

**Contraargumento más fuerte.** El trabajo demuestra que fijar el reloj de la A100 en 1260 MHz ahorra energía de GPU en cargas limitadas por memoria, y que en esta CPU, midiendo los dos paquetes completos y escalando solo 6 núcleos, bajar la frecuencia no paga. No demuestra que un agente que *clasifica fases en línea* aporte algo sobre esas dos reglas estáticas:
- El único experimento con fases detectadas separa las fases con reposo artificial (C2).
- El contraste contra F1 fijo quedó inválido.
- En la aplicación continua el agente decidió una vez y se comportó como un candado estático.
- En la aplicación real con fases naturales (LAMMPS) no actuó.
- En CPU, la etiqueta que el clasificador aproxima con 0.728 se puede calcular con el mismo instrumento (C3).
- La política de GPU adoptada no supera la validación fuera de muestra que la propia Metodología exige (M4).
- Ninguna mejora de aplicación podía resultar significativa con el diseño elegido (C1).

Es un buen trabajo de caracterización y de instrumento. Como evidencia de que el *agente* mejora el EDP frente a los gobernadores nativos, la conclusión correcta es "dirección favorable en dos casos seleccionados, sin poder de prueba".

**Cómo responderlo sin experimentos nuevos:** el libro ya tiene casi todo el material. Basta con que el resumen, la Discusión y las Conclusiones:
1. digan explícitamente que el valor demostrado es el de la política (F1 en memoria), no el de la conmutación dinámica;
2. declaren la limitación de la detección por episodios;
3. expliquen la restricción operativa que obliga a no usar $I$ en línea;
4. reporten el efecto con un intervalo en lugar de solo la prueba de signos.

Con eso, el argumento queda proporcionado a la evidencia y el jurado no encuentra la grieta por su cuenta.

**Caminos alternativos que el texto no discute:**
- Una regla estática por aplicación (lista de aplicaciones de memoria → F1) como línea base frente al clasificador.
- Usar la utilización de memoria de NVML con un umbral simple, sin modelo, dado que la importancia por permutación (`03:295`) dice que es la señal dominante.

---

## 5. Hoja de ruta sugerida (orden de trabajo)

1. **Solo texto, alto impacto:** C1 (declarar el $p$ mínimo), C2 (episodios y CloverLeaf ≈ F1 estático), C3 (por qué no usar $I$ en línea), M4 (6.5% fuera de muestra junto al 8.9%), M8 y M9 (Conclusiones por objetivo y respuesta a la pregunta).
2. **Solo texto, contradicciones:** M5, M6 (o rehacer el *bootstrap*), M11, M12, M13, M14, m1, m2, m3.
3. **Declaraciones de medición:** M3, M7, M10, m17.
4. **Análisis opcional sobre datos existentes:** IC del efecto por bloques (C1), *bootstrap* de la política de CPU por familia (M6).
5. **Forma:** los demás menores.

**Impacto en el control del proyecto:** ninguno de estos cambios altera resultados registrados en `Plan_Detallado_Realineacion_Hyperion.md` ni en `Seguimiento_Cambios_Plan_Director.md`. Si se decide rehacer el *bootstrap* de CPU por familia (M6) o calcular IC por bloques (C1), conviene registrarlo como análisis adicional.

**Nota aparte (fuera del libro):** `docs/libro/AGENTS.md` sigue diciendo que el contador es `stalled-cycles-backend` y que la bibliografía es un bloque `thebibliography` manual. Las dos cosas están desactualizadas (el código usa `STALLS_MEM_ANY` y el libro usa BibTeX con `IEEEtran`).
