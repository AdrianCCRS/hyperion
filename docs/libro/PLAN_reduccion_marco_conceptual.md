# Plan de reducción del Marco Conceptual

Archivo objetivo: `docs/libro/secciones/01_marco_referencia.tex` (respaldo sin seguimiento: `01_marco_referenciaV1.tex`).

## 1. Objetivo y meta de extensión

- El director pide que Marco Conceptual (con figuras) y Estado del Arte no pasen de unas 20 páginas.
- Medida actual en `main.pdf`: el capítulo 1 ocupa las páginas 16 a 50 (Marco Conceptual unas 31 páginas, Estado del Arte unas 3). El `.toc` está desactualizado y no sirve como referencia.
- Meta: Marco Conceptual de 15 a 17 páginas. Equivale a una reducción cercana al 45 % o 50 %, superior al 30 %–40 % inicial, porque el tope de 20 incluye el Estado del Arte. Si al recompilar el Estado del Arte resulta más corto, la meta se relaja hacia el 40 %.
- No se añaden ni se eliminan referencias en `main.bib`. Se usan solo las ya verificadas (`Akiba2019` para Optuna y `Bergstra2011` para TPE ya están en el `.bib`).

## 2. Criterio de decisión

Un tema permanece si responde a qué es y para qué se usa en este proyecto. Sale si describe procedimiento (va a Metodología), si valora o compara resultados (va a Discusión) o si es un concepto estándar que los evaluadores conocen (IA, arquitectura de computadores). No se conserva teoría que el proyecto no use.

## 3. Estructura final y presupuesto de páginas

El orden de los subtítulos sigue una cadena en la que cada subsección abre la dependencia de la siguiente.

| # | Subsección final | Contenido que permanece | Contenido que sale o se condensa | Meta |
|---|---|---|---|---|
| 1 | Potencia CMOS y fundamento de DVFS | Ecuación de potencia dinámica y su dependencia de la frecuencia y del voltaje. Origen de la idea de que DVFS puede ahorrar energía. | Discusión termodinámica extensa del silicio. | ~1 pág |
| 2 | Modelo Roofline | Figura Roofline. Ecuaciones del punto de inflexión (`eq:ridge`) y de la intensidad operacional. Criterio `compute-bound` y `memory-bound`. Se fusionan "punto de inflexión" e "intensidad observada" en un solo bloque. Se conservan `sec:marco-ridge` y `eq:ridge`. | Derivaciones repetidas y notación redundante. | ~2.5 pág |
| 3 | Telemetría con PMU | Figura PMU. Figura de ámbito de atribución (núcleo y `uncore`). Definición de PMU y para qué sirve. Una frase que menciona IPC y MPKI como métricas derivadas de los contadores, sin fórmulas ni interpretación. | Multiplexación y eventos genéricos/crudos (2 o 3 frases). Sección de métricas de eficiencia y de ventanas de muestreo (a Metodología). | ~3 pág |
| 4 | Espacios de ejecución `user-space` y `kernel-space` | Figura user/kernel. Un párrafo sobre por qué esta frontera determina qué puede leer y qué puede actuar el agente. | Explicación general de llamadas al sistema. | ~1 pág |
| 5 | Interfaces de medición | Un párrafo o una tabla corta que indique qué expone cada interfaz (`perf_event`, RAPL, NVML, Nsight). Sin comparar cuál es mejor. | Figuras de `perf_event` y RAPL. Detalle de niveles de privilegio (`paranoid`). Todo ello ya está tratado en Metodología 2.2.3. | ~0.75 pág |
| 6 | P-states, C-states y gobernadores | Una sola figura que une P-states/C-states y la jerarquía DVFS. Papel de los gobernadores. Nota breve sobre GPU. | Figuras de dominios SMT y de dominios GPU (su idea pasa a 2 o 3 frases). | ~2.5 pág |
| 7 | Modelos de clasificación ligera | Qué modelos se usan (regresión logística, árbol, Random Forest, Extra-Trees, XGBoost) y en cuáles se espera mejor comportamiento y por qué. Por qué la inferencia debe ser de baja latencia. Cargas y benchmarks entran aquí como una sola frase sobre kernels y microbenchmarks de calibración. | Definiciones básicas de árbol y ensamble. Las tres figuras de RF, ET y XGB se reducen a una figura combinada o a ninguna. Sección "Cargas de trabajo" independiente (detalle a Metodología 2.3.2). | ~2 pág |
| 8 | Búsqueda de hiperparámetros con Optuna | Ver sección 4 de este plan. | Frente de Pareto y relación de dominancia formal (a 1 o 2 frases). Comparación grid contra aleatoria. | ~1 pág |
| 9 | Producto Energía–Retardo (EDP) | Las dos ecuaciones y su papel como criterio de decisión de la política. | Ninguno relevante. | ~0.75 pág |
| 10 | Cierre del marco | Un párrafo breve con los límites de lo que el marco permite afirmar. | Apertura larga sobre la "cadena de dependencias" (se reduce cerca de un 60 %). | ~0.5 pág |

Subtotal aproximado 15 a 16 páginas, con margen de 1 página.

Figuras que se conservan (5 o 6, hoy son unas 14): Roofline, PMU, ámbito de atribución de contadores, user/kernel, P-states/DVFS y, opcionalmente, la de modelos.

## 4. Incorporación de Optuna siguiendo el flujo

Optuna se introduce como cierre natural de la subsección de modelos, porque la búsqueda de hiperparámetros es el paso que sigue a elegir familias candidatas. Eso permite condensar dos temas en uno.

Contenido:

- Qué es un hiperparámetro y por qué se fija antes del ajuste (2 o 3 frases, sin repetir la definición básica).
- Por qué se usa una búsqueda bayesiana con TPE y no una rejilla. El espacio de RF, Extra-Trees y XGBoost mezcla enteros y reales en varias dimensiones, y una rejilla con resolución equivalente sería costosa. Esto respalda la elección con `Bergstra2011` y `Akiba2019`, que ya están en el `.bib`.
- Relación con la validación. La búsqueda ocurre sobre el conjunto de entrenamiento de cada pliegue externo, de modo que la familia de prueba nunca interviene. Se resume en una sola frase que remite a Metodología Fase 2. Esto sustituye a la sección extensa de validación anidada.
- Objetivos en conflicto. Una frase indica que la calidad predictiva y el costo de inferencia deben considerarse juntos, sin desarrollar el formalismo de Pareto.

Datos que no se presentan en el marco: número de intentos por modelo, valores del F1 antes y después y espacios de búsqueda concretos. Son metodología y resultados y están documentados en `F2-XDEV-003` del Seguimiento de Cambios.

Aviso de verificación. En el estado actual, `02_metodologia.tex` no menciona Optuna, y `03_resultados.tex` (línea 328) sí lo hace. Antes de escribir se debe confirmar que Metodología Fase 2 recoge el procedimiento. Si no lo recoge, se anota como pendiente en Metodología y no se duplica en el marco.

## 5. Contenido que sale del marco y a dónde va

| Tema retirado | Destino |
|---|---|
| Métricas de evaluación del clasificador (F1, precisión, recall y sus ecuaciones) | Metodología Fase 2 |
| Líneas base y costo de inferencia como dimensión de evaluación | Metodología Fase 2 |
| Validación cruzada anidada y fuga de información | Metodología Fase 2 (se conserva la referencia `sec:marco-validacion-anidada`, ver sección 6) |
| Comparaciones estadísticas pareadas | Metodología Fase 4 (decisión ya tomada) |
| Frente de Pareto (formalismo) | Metodología Fase 2 |
| Ventanas de muestreo y métricas de eficiencia | Metodología 2.2.4 |
| Interfaces en detalle (RAPL, `perf_event`, paranoid, NVML) | Metodología 2.2.3 |
| Catálogo de cargas | Metodología 2.3.2 |

## 6. Referencias cruzadas que no se pueden romper

- `02_metodologia.tex` referencia `sec:marco-ridge`, `eq:ridge` y `sec:marco-validacion-anidada` (línea 567).
- Si se elimina la subsección de validación anidada, hay dos salidas. Una es mover a Metodología el párrafo que la sustenta y actualizar la referencia. La otra es conservar un párrafo corto dentro de la subsección de modelos con la misma etiqueta. Se elige la segunda si el párrafo cabe en cinco líneas.
- Tras cada cambio se compila y se buscan advertencias de `undefined reference` y de citas faltantes.

## 7. Parámetros de redacción (aplican a todo lo que se reescriba)

Se examina cada subsección completa antes de tocar oraciones sueltas. Se conservan datos, cifras, nombres propios, citas y relaciones de causalidad.

Patrones a eliminar:

1. Falsos contrastes del tipo "no X, sino Y".
2. Cierres enfáticos que repiten lo dicho.
3. Falsa profundidad y aforismos ("el corazón del problema", "en última instancia").
4. Preámbulos y anuncios de lo que se explicará después.
5. Objeciones que nadie ha planteado.
6. Enumeraciones forzadas en tríadas.
7. Comienzos de oración repetitivos.
8. Calificativos inflados ("crucial", "clave", "robusto", "meticuloso", "panorama", "transformador").
9. Atribución vaga ("diversos estudios señalan") sin cita puntual.
10. Gerundios finales que añaden juicios no demostrados.
11. Residuos conversacionales o de asistencia.

Estilo:

- Sin tono de manual ni de clase introductoria. No se explican conceptos estándar más de lo necesario ni se acumulan definiciones antes de mostrar su utilidad.
- Cada subsección justifica una pieza del proyecto (Roofline explica el criterio de clasificación, PMU/`uncore`/RAPL/NVML explican qué puede observarse y con qué límites, `user-space` explica la ubicación y restricciones del agente, ML ligero explica la baja latencia, EDP explica el criterio de juicio de la política).
- Verbos concretos (mide, define, calcula, requiere, limita, depende, permite, observa, clasifica) y verbos relacionales (depende de, responde a, se asocia con, queda condicionado por).
- Evitar "permite comprender", "resulta fundamental", "es importante señalar", "cabe destacar", "constituye un aspecto clave". Evitar secuencias repetidas de "X permite, Y permite, Z permite".
- Alternar oraciones breves con otras de desarrollo medio, y combinar subordinadas causales, relativas y comparativas cuando la relación lo justifique.
- Enlazar con referentes ya introducidos ("esta relación", "ese comportamiento", "dicha condición"), con antecedente inequívoco.
- Usar artículos y determinantes ("de la frecuencia y del voltaje") para evitar un tono telegráfico.
- Los conectores de secuencia ("primero", "después") solo de forma puntual en párrafos introductorios.
- No cerrar cada párrafo con una conclusión enfática. El cierre debe abrir la dependencia con el concepto siguiente.
- No sustituir términos técnicos consolidados por sinónimos (`Roofline`, `DVFS`, `PMU`, `uncore`, `EDP`, `compute-bound`, `memory-bound`).
- Mantener las cautelas técnicas. No convertir correlaciones, proxies o señales indirectas en mediciones directas. Conservar limitaciones y condiciones de validez aunque el texto se compacte.

Puntuación:

- Sin dos puntos en la prosa, salvo cuando la oración introduce directamente una ecuación. Se exceptúan los elementos técnicos (`\label{sec:uncore}`, `\ref{sec:ventanas}`, rutas, identificadores, código).
- Sin guiones ni rayas como acotación o conector. Los incisos se resuelven con comas, paréntesis o punto y coma.
- Las enumeraciones y aclaraciones se reformulan como oraciones completas.

Respeto a las reglas del `AGENTS.md` de `docs/libro/`: no inventar referencias, no mencionar felix/SC3, no tocar el Marco Legal, y ante conflicto con el código, manda el código. Se conserva la disciplina epistémica (medido frente a estimado).

## 8. Orden de trabajo

1. Commit del estado actual (`01_marco_referencia.tex` y el respaldo `V1`) para poder volver atrás.
2. Recompilar `main.tex` y medir las páginas reales por subsección.
3. Verificar en Metodología 2.2.3, 2.2.4, 2.3.2 y en las Fases 2 y 4 qué ya está cubierto, para no duplicar al mover contenido.
4. Reescribir el marco subsección por subsección, en el orden de la sección 3.
5. Al terminar cada subsección, releer contra la sección 7 (patrones, estilo y puntuación) y corregir antes de pasar a la siguiente.
6. Ajustar referencias cruzadas y compilar sin advertencias.
7. Comprobar la extensión (Marco + Estado del Arte ≤ 20 páginas) y que los subtítulos se lean como una explicación continua.
8. Revisión final del capítulo completo contra la sección 7, buscando redundancias entre subsecciones.
9. Opcional. Pasar el capítulo por la skill de editor académico UIS para revisar APA y estilo.

## 9. Criterios de aceptación

- Marco Conceptual entre 15 y 17 páginas y Marco + Estado del Arte no mayor de 20.
- Se conservan las figuras Roofline, PMU, ámbito de atribución y user/kernel, y las fórmulas de CMOS, `eq:ridge`, intensidad operacional y EDP.
- IPC y MPKI aparecen en una sola frase dentro de la subsección de PMU.
- Optuna aparece integrado al final de la subsección de modelos con las citas `Akiba2019` y `Bergstra2011`.
- La estadística y la evaluación del clasificador quedan en Metodología.
- Sin referencias rotas, sin citas nuevas sin verificar, sin duplicación con Metodología.
- Sin dos puntos, guiones ni rayas de acotación en la prosa, y sin los patrones de la sección 7.
