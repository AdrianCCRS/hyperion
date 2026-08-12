# Estudio a fondo: Intel Advisor Roofline (2023.0.0) como validación cruzada para Hyperion

Documento escrito a partir de documentación oficial de Intel (citada en cada
sección — enlaces al final) y verificación empírica propia en `paccaA100`.
Reemplaza el enfoque de validación cruzada basado en VTune (`docs/vtune/`,
`pipelinevtune/`, ambos congelados como referencia histórica) — no porque
VTune esté mal, sino porque, tras probarlo, Advisor encaja mejor con lo que
esta campaña necesita: el propio Advisor ya expresa su resultado en el mismo
lenguaje que usa Hyperion (intensidad aritmética vs. techos), sin que
nosotros tengamos que traducir un Top-Down de microarquitectura a esa
pregunta con un árbol de reglas propio.

Versión confirmada en el nodo: **2023.0.0**, mismo módulo jerárquico que
VTune (`module load devtools/intel/oneapi/2023` → `module load advisor/2023.0.0`).

---

## 1. Qué es Intel Advisor, en una frase

Intel Advisor es una herramienta de diseño y caracterización de rendimiento
de software — no un profiler de propósito general como VTune — centrada en
tres preguntas: ¿mi código vectoriza bien?, ¿mi código está limitado por
cómputo o por memoria (Roofline)?, ¿vale la pena paralelizar/threadear esta
sección? [[Intel Advisor Roofline — Developer Guide]](https://www.intel.com/content/www/us/en/developer/articles/guide/intel-advisor-roofline.html).
Este proyecto usa exclusivamente su análisis **Roofline** (cache-aware),
sección de esta nota.

## 2. Qué calcula y cómo lo calcula — el mecanismo de dos pasadas

A diferencia de VTune (una sola colección de muestreo por hardware), obtener
un Roofline completo en Advisor **requiere dos análisis separados y
secuenciales**, confirmado en la documentación oficial de comandos
[[CPU Roofline — Run from Command Line]](https://www.intel.com/content/www/us/en/docs/advisor/user-guide/2023-0/run-cpu-roofline-perspective-from-command-line.html):

```bash
advisor --collect=survey    --project-dir=./proj -- ./mi_app
advisor --collect=tripcounts -flop --enable-cache-simulation --project-dir=./proj -- ./mi_app
# equivalente abreviado de las dos corridas de arriba:
advisor --collect=roofline --enable-cache-simulation --project-dir=./proj -- ./mi_app
```

1. **`survey`** — pasada de muestreo liviano: identifica los loops/funciones
   más costosos y mide su tiempo de ejecución real. Documentado como "no
   noticeable overhead" — es la pasada que da el eje Y (tiempo → GFLOPS,
   junto con el conteo de FLOPs del segundo paso).
2. **`tripcounts -flop`** — pasada de **instrumentación binaria** (no
   muestreo): cuenta cuántas veces se ejecuta cada loop (*trip counts*) y
   cuántas operaciones de punto flotante ejecuta realmente cada uno. Tiene
   "significant overhead" — por eso Intel obliga a separarla de `survey`,
   para que ese overhead extra no contamine la medición de tiempo del primer
   paso [[Getting Started with Advisor Roofline]](https://www.intel.com/content/www/us/en/developer/articles/guide/intel-advisor-roofline.html).

**Por qué esto importa para nuestra metodología:** el tiempo (para GFLOPS) y
el conteo de FLOPs/bytes (para intensidad aritmética) vienen de **dos
mecanismos de medición distintos, corridos por separado, y luego fusionados**
— no de una sola corrida con un solo mecanismo, como sí es el caso del
Top-Down de VTune (todo sale de una corrida de `uarch-exploration`). Es un
diseño más caro en tiempo de campaña (dos ejecuciones del binario por
kernel), pero elimina la interferencia entre "medir cuánto tarda" y "contar
cuántas operaciones hizo".

## 3. Cómo halla los FLOPs — de la aplicación y del techo (son dos preguntas distintas)

### 3.1 FLOPs de la aplicación (el punto que se grafica)

**Por instrumentación, no por muestreo ni por contador de PMU.** La pasada
`tripcounts -flop` cuenta las operaciones de punto flotante realmente
ejecutadas, instrumentando el binario para interceptar las instrucciones
aritméticas — Intel lo describe como conteo de "actual executed floating
point instructions", con la fracción "por segundo" tomada de la pasada
`survey` por separado (esto se llama *selftime-based FLOPS counting*,
asignado a la función que hizo el trabajo, no al loop que la llamó)
[[Cache-Aware Roofline Model — ERCIM News]](https://ercim-news.ercim.eu/en110/r-i/cache-aware-roofline-model-in-intel-advisor).
El tráfico de memoria (bytes movidos, para el eje X) se extrae de la misma
forma: instrumentando las instrucciones que mueven datos hacia/desde
registros — no leyendo contadores de cache-miss del hardware.

### 3.2 FLOPs/bytes "del techo" (los rooflines)

Estos **no** salen de una ficha técnica ni de un cálculo teórico — Advisor
corre sus **propios micro-benchmarks contra el hardware real** antes/aparte
de medir la aplicación, exactamente el mismo principio que
`calibration.py` de Hyperion (STREAM/ERT propios, no valores de catálogo):
"the lines... are representative of hardware limits on kernel performance
based on benchmarks run by Intel Advisor to establish baselines and
performance limits on the host system"
[[Intel Advisor Roofline — Parallel Universe article]](https://www.intel.com/content/www/us/en/developer/articles/guide/intel-advisor-roofline.html).
Confirmado también en las diapositivas oficiales de entrenamiento de Intel:
*"Roofs are based on benchmarks run before the application"*
[[Intel Advisor Vectorization and Roofline — RWTH Aachen, Intel 2022]](https://blog.rwth-aachen.de/itc-events/files/2022/09/Intel_Advisor_Roofline_RWTH.pdf).

Estos micro-benchmarks miden, en el hardware real donde se corre:
- Ancho de banda pico por nivel de caché: `L1 Bandwidth`, `L2 Bandwidth`,
  `L3 Bandwidth`, `DRAM Bandwidth` (con AVX-512, precisión simple y doble
  por separado).
- Pico de cómputo por tipo de instrucción: `Scalar Add Peak`,
  `Vector Add Peak`, `Vector FMA Peak` (SP y DP).

## 4. ¿Es confiable la medición? — instrumentación exacta, no estadística

Diferencia de fondo frente a VTune, y la respuesta directa a tu pregunta: la
instrumentación **cuenta cada operación real**, no extrapola desde una
muestra estadística. No hay incertidumbre de "¿cuántas muestras hacen falta
para que el número sea estable?" que sí existe en un mecanismo de EBS/PMU —
si el binario ejecutó 4 millones de FMA, Advisor contó 4 millones de FMA,
punto. La contrapartida (ver §6) es el costo: instrumentar cada instrucción
aritmética y cada acceso a memoria es mucho más lento que muestrear
periódicamente, de ahí el "significant overhead" ya citado.

**Corrección respecto a una primera redacción de esta sección (2026-08-10):**
el tráfico de memoria por nivel de caché (L1/L2/L3/DRAM) **no** se mide
contando misses reales de hardware — se **simula**, y el comportamiento sin
`--enable-cache-simulation` no es "medir solo L1 sin simular nada", como se
afirmaba antes aquí — es más drástico: **sin ese flag, Advisor no distingue
niveles de caché en absoluto.** Trata cada byte que toca una instrucción de
carga/almacenamiento como si fuera tráfico de DRAM, y grafica un solo techo
(DRAM Bandwidth) — el modelo "clásico" original de Williams et al. (2009),
sin la extensión cache-aware. Confirmado en un hilo de soporte oficial de
Intel: *"cache-config option is used for cache-simulation to get data
transfers values between different memory levels... [without it, you see]
only the DRAM roofline"*
[[Intel Community — cache-config option in Advisor roofline]](https://community.intel.com:443/t5/Analyzers/cache-config-option-in-the-Advisor-roofline-analysis/td-p/1194687).

Con `--enable-cache-simulation`, Advisor sí reconstruye, a partir de la
misma traza de accesos instrumentada, qué habría pasado en una jerarquía de
caché modelada (simula hits/misses/desalojos en cada nivel) — dando bytes
por nivel (L1/L2/L3/DRAM) y, por tanto, los techos y la intensidad
aritmética específicos de cada uno, no solo un DRAM genérico. **Caveat
honesto encontrado en la documentación:** la simulación de caché de Advisor
no simula la jerarquía completa — corre sobre un subconjunto y luego
**extrapola** el resultado al resto
[[Investigate Memory Usage and Traffic — Advisor docs]](https://intel.com/content/www/us/en/develop/documentation/advisor-user-guide/top/analyze-vectorization-perspective/explore-vectorization-and-code-insights-results/investigate-memory-usage-and-traffic.html).
Es un modelo validado por Intel, no una medición directa — la distinción
importa porque significa que el eje de intensidad aritmética depende de qué
tan bien el simulador de
caché de Advisor refleja el comportamiento real de la microarquitectura, no
solo de la instrumentación en sí.

## 4.1 ¿Conviene usar la simulación de caché, o el modelo "clásico" (sin `--enable-cache-simulation`)?

Pregunta de diseño directa para esta campaña, no solo curiosidad — la
respuesta depende de contra qué se va a comparar, y aquí sí hay una
respuesta correcta: **usar `--enable-cache-simulation`, tal como ya hace
`advisor/run_roofline_unit_test.sh`.** Razón, comparando con lo que
`operational_intensity` de Hyperion mide realmente
(`orchestrator/postprocess.py`, ver conversación anterior):

- `bytes_moved_window` de Hyperion = `delta_cache_misses × llc_line_size_bytes`
  — bytes que **fallaron en LLC**, la aproximación más cercana posible a
  tráfico real de DRAM sin acceso a *uncore* (ver
  `docs/vtune/Informe_VTune_Profiler.md` §9 para la misma limitación del
  lado VTune).
- El modelo **clásico** de Advisor (sin simulación) trata **todo** byte
  tocado por una carga/almacenamiento como si fuera tráfico de DRAM — sin
  distinguir qué se sirvió desde L1/L2/L3 con alto reuso. Esto
  **sobreestima sistemáticamente** el tráfico de memoria real, y por lo
  tanto **subestima la intensidad aritmética** — un kernel con buen reuso de
  caché saldría luciendo más "memory bound" de lo que realmente es.
- El modelo **cache-aware** (con simulación) sí separa el tráfico que
  efectivamente llega a DRAM del que se queda en caché — es la magnitud
  **comparable** con el `bytes_moved_window` de Hyperion (ambos aproximan
  "bytes que de verdad salieron del núcleo hacia memoria"), no la magnitud
  del modelo clásico.

**Conclusión:** usar el modelo clásico compararía la intensidad aritmética
de Hyperion (basada en misses de LLC) contra una intensidad de Advisor
basada en tráfico total sin filtrar — no es una comparación justa, sesgaría
sistemáticamente hacia "Advisor dice más memory-bound que Hyperion" sin que
eso refleje una discrepancia real. La simulación de caché es más cara (el
overhead de 17× que medimos en la prueba unitaria ya la incluye) pero es la
única de las dos opciones que mide la misma magnitud física que compara
Hyperion — la corrección aquí no es una preferencia de exactitud en
abstracto, es un requisito de comparabilidad directa con lo que ya
tenemos.

## 5. ¿Utiliza PMU? ¿Hay multiplexación?

**Respuesta corta: no para lo que grafica el Roofline.** Ni el conteo de
FLOPs ni el de bytes (simulados, §4) dependen de contadores de PMU ni de
`perf_event_open` — son instrumentación + simulación de caché, un mecanismo
completamente distinto al que usa VTune. Esto tiene una consecuencia
práctica enorme para este proyecto: **la multiplexación de eventos de PMU
(la razón por la que `uarch-exploration` de VTune necesitaba tiempo extra y
podía perder precisión, ver `docs/vtune/vtune_cross_validation.md` §E.6)
simplemente no aplica aquí** — no hay grupos de eventos que rotar porque no
hay eventos de PMU en el camino crítico de esta medición.

**Dónde SÍ podría entrar PMU, con matiz:** la pasada `survey` (el tiempo de
ejecución de cada loop) es, en espíritu, del mismo tipo de muestreo liviano
que usa VTune Hotspots — la documentación de Intel no detalla si usa
`perf_event_open` internamente o un temporizador de SO, y no encontramos una
página oficial que lo precise a ese nivel para Advisor específicamente. Lo
dejamos como pregunta abierta, no como afirmación — no vamos a inventar el
mecanismo exacto de `survey` sin una fuente que lo confirme. Lo que sí es
seguro, porque la propia CLI lo distingue con un flag dedicado, es que el
FLOP/byte no pasa por ahí.

## 6. Niveles de exactitud y overhead (accuracy presets)

Advisor expone niveles de exactitud configurables — "Low Accuracy" (Basic
Roofline Chart) hasta niveles más altos con call stacks — donde "the higher
accuracy value you choose, the higher runtime overhead is added"
[[CPU Roofline Accuracy Presets — User Guide 2023-0]](https://www.intel.com/content/www/us/en/docs/advisor/user-guide/2023-0/cpu-roofline-accuracy-presets.html).
Para esta campaña usamos el nivel por defecto (equivalente al comando
abreviado `--collect=roofline` de la sección 2) — suficiente para el
veredicto compute/memory-bound por kernel, sin necesitar la atribución por
call-stack completa.

## 7. Qué técnica de medición usa, en resumen (sí, es el modelo Roofline — con una variante)

Confirmado con la cita fundacional exacta: el modelo Roofline lo propusieron
Williams, Waterman y Patterson (UC Berkeley) en *"Roofline: An Insightful
Visual Performance Model for Multicore Architectures"* (2009). Advisor
implementa la variante **cache-aware**, propuesta por Ilic, Pratas y Sousa
(Universidad de Lisboa) en *"Cache-Aware Roofline Model: Upgrading the Loft"*
(2013) — ambas citadas directamente en el material oficial de entrenamiento
de Intel [[Intel Advisor Vectorization and Roofline — CERN 2022]](https://cdrdv2-public.intel.com/671165/roofline-analysis-with-intel-advisor.pdf).

La diferencia frente al Roofline "clásico" (un solo techo de ancho de banda,
DRAM): el cache-aware model dibuja **un techo diagonal por cada nivel de la
jerarquía de memoria** (L1, L2, L3, DRAM) y **un techo horizontal por cada
tipo de capacidad de cómputo** (escalar, vectorial, vectorial+FMA, en
simple/doble precisión) — un mismo punto (un loop) se compara
simultáneamente contra los cuatro techos de memoria y los tres+ techos de
cómputo, no contra uno solo de cada tipo:

```
Gflop/s = min( Platform_PEAK_del_techo_relevante , Platform_BW_del_nivel_de_cache × AI )
```

fórmula confirmada literal en el material de entrenamiento de Intel
(diapositiva "Drawing the Roofline")
[[Intel Advisor Vectorization and Roofline — RWTH Aachen]](https://blog.rwth-aachen.de/itc-events/files/2022/09/Intel_Advisor_Roofline_RWTH.pdf).

---

## 8. Comparación directa con Hyperion — cómo aporta este enfoque como validación cruzada

### 8.1 Coincidencia metodológica de fondo — más cercana que VTune

Esto es lo más importante de esta sección: **Advisor ya habla el mismo
idioma que Hyperion**, sin que nosotros tengamos que traducir nada. La
etiqueta oficial de Hyperion (`phase_label_train`, ver
`docs/retoma/Guia_Maestra_Fase1_DVFS.md`) sale de comparar
`operational_intensity` (FLOPs del binario ÷ bytes movidos, medidos con
`perf`) contra `i_ridge` (de una calibración Roofline propia con
STREAM/ERT). Advisor calcula exactamente ese mismo par de números —
intensidad aritmética (FLOP/byte) vs. techos de cómputo/memoria medidos con
micro-benchmarks propios — y los pone en el mismo gráfico. Con VTune
tuvimos que construir una capa de reglas propia (`validation_classifier.py`)
para traducir un Top-Down de microarquitectura (fracciones de *pipeline
slots*) a un veredicto compute/memory — con Advisor, el propio *dot* en el
gráfico ya está en el lado izquierdo (memoria) o derecho (cómputo) del
gráfico, sin esa traducción.

### 8.2 Dónde SÍ siguen siendo independientes (no es circular)

Aunque el lenguaje es el mismo, el mecanismo de medición es distinto de
principio a fin — sigue cumpliendo la separación de capas ya documentada en
`docs/vtune/vtune_cross_validation.md` §C:

```
misma pregunta (¿AI vs. techos?)  ≠  mismo mecanismo de medición
```

| | Hyperion (oficial) | Advisor (esta validación) |
|---|---|---|
| FLOPs de la app | Autorreportados por el binario (stdout de NPB/STREAM/ERT) | Instrumentación binaria, conteo exacto de instrucciones FP ejecutadas |
| Bytes movidos | `perf_event_open` — contadores de PMU (`cache-misses` × tamaño de línea) | Simulación de caché sobre la traza instrumentada (`--enable-cache-simulation`) |
| Techos (I_ridge / roofs) | Calibración propia con STREAM/ERT, un solo techo de BW y uno de cómputo | Micro-benchmarks propios de Advisor, un techo por nivel de caché y por tipo de instrucción |
| Mecanismo físico | `perf_event_open`, muestreo de PMU | Instrumentación dinámica + simulación, sin PMU en el camino crítico |

Dos técnicas físicamente distintas (PMU real vs. instrumentación+simulación)
llegando al mismo tipo de veredicto — si coinciden, es una validación más
fuerte que la de VTune, precisamente porque **ni siquiera comparten
mecanismo de origen**, no solo pipeline de procesamiento.

### 8.3 Qué NO resuelve, honestamente

- El `I_ridge` de Hyperion depende de STREAM/ERT corridos por nuestro propio
  harness; el de Advisor depende de sus propios micro-benchmarks — si ambos
  coinciden en el orden de magnitud del ridge point de este hardware, es
  evidencia cruzada real; si difieren mucho, hay que investigar cuál de los
  dos micro-benchmarks es menos representativo, no asumir que Hyperion tiene
  la razón por defecto.
- Advisor mide por **loop/función**, no por ventana temporal de ~1 ms como
  el agente DVFS — es una validación de **la selección de kernels para el
  dataset**, igual que se aclaró para VTune, no un sustituto del
  clasificador en vivo.
- El overhead de instrumentación (§4, §6) es alto — nunca sería viable como
  mecanismo de producción del agente DVFS, ni siquiera se lo planteamos.

---

## Fuentes citadas

- [Intel® Advisor Roofline — Developer Guide](https://www.intel.com/content/www/us/en/developer/articles/guide/intel-advisor-roofline.html)
- [CPU Roofline — Run CPU/Memory Roofline Insights Perspective from Command Line (User Guide 2023-0)](https://www.intel.com/content/www/us/en/docs/advisor/user-guide/2023-0/run-cpu-roofline-perspective-from-command-line.html)
- [CPU Roofline Accuracy Presets (User Guide 2023-0)](https://www.intel.com/content/www/us/en/docs/advisor/user-guide/2023-0/cpu-roofline-accuracy-presets.html)
- [Analyze CPU Roofline (User Guide 2023-0)](https://www.intel.com/content/www/us/en/docs/advisor/user-guide/2023-0/analyze-cpu-roofline.html)
- [Cache-Aware Roofline Model in Intel® Advisor — ERCIM News 110](https://ercim-news.ercim.eu/en110/r-i/cache-aware-roofline-model-in-intel-advisor)
- [Frequently Asked Questions about the Intel® Advisor Roofline Analysis Feature](https://www.intel.com/content/www/us/en/developer/articles/troubleshooting/intel-advisor-roofline-feature-qa.html)
- [Memory-Level Roofline Model with Intel® Advisor](https://www.intel.com/content/www/us/en/developer/articles/technical/memory-level-roofline-model-with-advisor.html)
- [Intel Community — cache-config option in the Advisor roofline analysis](https://community.intel.com:443/t5/Analyzers/cache-config-option-in-the-Advisor-roofline-analysis/td-p/1194687) (confirma: sin `--enable-cache-simulation`, solo aparece el techo de DRAM Bandwidth)
- [Investigate Memory Usage and Traffic — Advisor User Guide](https://intel.com/content/www/us/en/develop/documentation/advisor-user-guide/top/analyze-vectorization-perspective/explore-vectorization-and-code-insights-results/investigate-memory-usage-and-traffic.html) (confirma: la simulación de caché corre sobre un subconjunto y extrapola)
- Intel® Advisor Roofline Analysis — *The Parallel Universe* magazine, issue with O'Leary, Gazizov, Shinsel, Matveev, Petunin (Intel Corporation), PDF: https://cdrdv2-public.intel.com/671165/roofline-analysis-with-intel-advisor.pdf
- Intel® Advisor — Vectorization and Roofline Analysis, Heinrich Bockhorst (Intel), aiXcelerate RWTH Aachen, dic. 2022: https://blog.rwth-aachen.de/itc-events/files/2022/09/Intel_Advisor_Roofline_RWTH.pdf
- Intel® Advisor — Vectorization and Roofline Analysis, Klaus-Dieter Oertel (Intel), CERN Software Tools Training, mar. 2022: https://indico.cern.ch/event/1106734/sessions/431205/attachments/2401314/4106693/Intel_Advisor_VectorizationRoofline_CERN_2022-03-03.pdf
- Williams, S., Waterman, A., Patterson, D. — *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, 2009 (citado en el material de Intel arriba).
- Ilic, A., Pratas, F., Sousa, L. — *Cache-Aware Roofline Model: Upgrading the Loft*, 2013 (citado en el material de Intel arriba).

## 9. Primera corrida real (evidencia empírica, no solo documentación)

Prueba unitaria ejecutada en `paccaA100` el 2026-08-10, `EP.A` (NAS Parallel
Benchmarks), 6 hilos (`0-5`), `advisor/run_roofline_unit_test.sh`:

| Pasada | Tiempo real de `EP.A` | Comparación |
|---|---|---|
| Baseline / `survey` | 1.20 s | Igual al tiempo normal del binario (confirma "no noticeable overhead" citado en §2) |
| `tripcounts -flop --enable-cache-simulation` | 20.39 s | **~17× más lento** que el baseline — confirma empíricamente el "significant overhead" citado en §2, no es solo lo que dice la documentación |

Resultado final del Roofline para `EP.A`:

```
GFLOPS:  3.59
GINTOPS: 2.24
```

(`GINTOPS` — operaciones enteras por segundo — aparece porque EP genera
números aleatorios con un generador congruencial de 64 bits, trabajo entero,
no solo de punto flotante; coherente con lo que ya sabíamos de este kernel,
ver `pipelinevtune/context/03_kernels_notas.md`.)

Artefactos generados y verificados: `roofline_report.html` (656 KB,
autocontenido, se abre en cualquier navegador sin necesitar la GUI de
Advisor) y `ep.A.advixeexpz` (7.6 MB, snapshot empaquetado para la GUI local,
ver `advisor/README.md`).

---

**Nota de honestidad metodológica:** varias páginas del User Guide oficial
(`docs.intel.com/.../advisor/user-guide/2023-0/...`) están citadas pero no
se pudieron volcar completas por bloqueo de acceso automatizado (HTTP 403);
lo citado de ellas viene del resumen de búsqueda que sí pudo indexarlas, no
de una lectura directa página por página de nuestra parte. Las citas
extraídas y verificadas palabra por palabra son las de los PDFs (RWTH
Aachen, CERN, *Parallel Universe*) y las de ERCIM News. Si algo de esta nota
necesita reconciliarse contra el User Guide exacto, es lo primero a revisar.
