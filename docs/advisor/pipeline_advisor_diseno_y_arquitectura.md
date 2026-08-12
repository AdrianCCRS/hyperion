# Pipeline de caracterización Roofline con Intel Advisor — diseño, decisiones y arquitectura

Documento de diseño, escrito **antes** de la implementación de la capa de
clasificación (tal como se pidió), con evidencia empírica real recolectada
en `paccaA100` el 2026-08-10/11. Complementa
`docs/advisor/estudio_intel_advisor_roofline.md` (qué es Advisor, cómo mide)
— este documento es sobre **qué decisiones metodológicas toma el pipeline y
por qué**, más la arquitectura del código.

---

## 0. Hallazgo que hay que reportar antes de cualquier otra cosa

**El modelo de CPU real de `paccaA100` no coincide con lo que documenta
`pipelinevtune/context/01_nodo_cartagena.md`.** Confirmado por dos fuentes
independientes en esta sesión (`lscpu` y la detección de CPU de LIKWID, en
una conversación anterior sobre VTune/LIKWID):

| | Documentado (`pipelinevtune/context/01_nodo_cartagena.md`) | Medido ahora (`lscpu`, 2026-08-11) |
|---|---|---|
| Modelo | Xeon Gold **5315Y** | Xeon Gold **5317** |
| Cores/socket | 8 | 12 |
| CPUs lógicas totales | 32 | 48 |
| NUMA node0 | `0-7,16-23` | `0-11,24-35` |
| L3 cache | 12288K | 18432K |

Ambos son SKUs reales de Ice Lake-SP (misma microarquitectura), así que
`-march=icelake-server` en la compilación de NPB sigue siendo correcto — eso
no está en duda. Lo que sí queda en duda es cualquier cifra o supuesto que
dependa del conteo exacto de cores/socket o del tamaño de L3 documentado
para ese nodo en trabajo previo. **No lo investigo más a fondo aquí** —
excede el alcance de este pipeline — pero lo dejo registrado con evidencia
para que se revise por separado; no lo escondo ni lo "arreglo" en la
documentación vieja sin que el equipo lo decida.

**Consecuencia práctica para este pipeline:** nunca asumir esas cifras.
`preflight_advisor.py` (sección 2) lee arquitectura, sockets, cores y NUMA
en vivo, cada vez, y los registra en la metadata de cada corrida — nunca los
codifica.

---

## 1. Qué ya está confirmado en vivo, y con qué comando exacto

Todo lo que sigue viene de ejecutar los comandos reales en `paccaA100`
(2023.0.0, `advisor --help report` y corridas reales sobre `ep.A.x`), no de
documentación web:

- **Exportación oficial estructurada, sin GUI:**
  `advisor --report=survey --project-dir=<proj> --format=csv --show-all-columns --mix --dynamic --report-output=<out>.csv`
  produce un CSV con **145 columnas reales**, incluyendo (nombres exactos
  del header real): `Self GFLOPS`, `Self AI`, `Self GINTOPS`, `Self INT AI`,
  `Self Overall AI`, `Self L2 GB`/`Total L2 GB`, `Self L3 GB`/`Total L3 GB`,
  `Self DRAM GB`/`Total DRAM GB` (con `Loaded`/`Stored` separados), `Data
  Types`, `Vector ISA`, `Traits`, y decenas de columnas `Dynamic <tipo>`
  (`Dynamic dp_compute`, `Dynamic sp_compute`, `Dynamic int_compute`,
  `Dynamic fma_vector_compute`, etc. — conteo dinámico real, no estático).
- **`advisor --report=roofs --project-dir=<proj> --format=csv`** exporta los
  valores de techo reales medidos en este nodo — confirmado con salida real
  (unidades: bytes/s y FLOP/s, no GB/GFLOP, hay que convertir):

  ```
  DRAM Bandwidth (single node),85832856271,memory,CPU
  DRAM Bandwidth,175273218366,memory,CPU
  L3 Bandwidth,579993235472,memory,CPU
  DP Vector FMA Peak,1349734446677,compute,CPU
  DP Vector Add Peak,674784613608,compute,CPU
  Scalar Add Peak,87528623391,compute,CPU
  SP Vector FMA Peak,2699332504425,compute,CPU
  ... (lista completa real archivada en advisor/reference/roofs_ep.A.x_paccaA100.csv)
  ```

- **Precisión real de EP (no asumida):** la columna `Data Types` del loop
  más caliente de `EP.A` (57.4% del self time,
  `MAIN__._omp_fn.1 at ep.f90:184`) dice literalmente `Float64; UInt64` —
  confirma FP64 + trabajo entero mezclado, coherente con el hallazgo
  anterior de `GINTOPS` no nulo. Otro loop (`vranlc_`) es `Float64` puro.
  Esto es evidencia real por loop, no una suposición de "NPB es DP".
- **`Vector ISA` del loop más caliente: `AVX2`**, no AVX-512 — pese a que el
  nodo sí soporta AVX-512 (`avx512f` en las flags de `lscpu`) y el binario
  se compiló con `-march=icelake-server`. El compilador (`gfortran`) no
  siempre generó AVX-512 para este loop específico — otra confirmación de
  por qué no se puede asumir el ISA usado sin mirar el dato real por loop.
- **Flags de compilación reales de NPB en este nodo**
  (`~/vtune_selfcheck/NPB3.4-OMP/config/make.def`):
  `FFLAGS = -O3 -march=icelake-server -mtune=icelake-server -fopenmp -g` —
  **ya incluye `-g`** y optimización — no hace falta recompilar nada para
  que Advisor tenga símbolos y atribución de línea (ya confirmado
  funcionando: `ep.f90:184`, `randi8.f90:59` aparecen correctos en el CSV).
- **Tamaño de línea de caché real: 64 bytes**, confirmado por dos vías del
  sistema operativo (`getconf LEVEL1_DCACHE_LINESIZE` y
  `/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size`) — mismo
  mecanismo exacto que ya usa `orchestrator/node_profile.py`
  (`_cache_data()`, lee `coherency_line_size` del índice de LLC). El
  pipeline reutiliza ese mismo mecanismo, no vuelve a asumir 64.

---

## 2. Las seis preguntas de diseño, respondidas y justificadas

### 2.1 ¿Qué memory roof se usa?

**DRAM Bandwidth (con `--enable-cache-simulation`), no el modelo clásico
sin simular.** Ya se justificó y se acordó en la conversación anterior
(`docs/advisor/estudio_intel_advisor_roofline.md` §4.1): `bytes_moved_window`
de Hyperion mide bytes que **fallaron en LLC** (lo más cercano a tráfico
real de DRAM que Hyperion puede medir sin *uncore*). Comparar contra el
techo de DRAM simulado de Advisor es la comparación de magnitudes
compatibles; comparar contra el modelo clásico (que cuenta *todo* acceso
como si fuera DRAM) compararía dos cosas distintas y sesgaría el resultado.

**Con un matiz nuevo, ahora que se confirmó el dato real:** el CSV expone
`Self DRAM GB` (bytes que el simulador de caché estima que sí llegaron a
DRAM) — **no** hace falta pedirle a Advisor un "techo de L3" aparte para
replicar el criterio de Hyperion; se usa directamente `Self DRAM GB` del
mismo reporte, dividiendo `Self GFLOP / Self DRAM GB` para obtener la
intensidad aritmética a nivel DRAM por loop — Advisor no expone esa columna
ya calculada (`Self AI` es la intensidad por defecto, que no se puede dar
por sentado a qué nivel corresponde sin confirmarlo — ver limitación en §4),
así que el pipeline la calcula explícitamente, no la asume de una columna
con un nombre parecido.

### 2.2 ¿Qué compute roof se usa?

**`DP Vector FMA Peak` o `SP Vector FMA Peak`, elegido dinámicamente por
loop según la columna real `Data Types`/`Dynamic dp_compute` vs `Dynamic
sp_compute` — nunca fijo de antemano, nunca elegido "porque da el número
más parecido a DGEMM".**

Justificación de por qué es el **FMA** peak y no el Scalar/Vector-Add peak,
aun cuando el loop en cuestión no esté vectorizado con FMA: el propósito de
esta comparación es **clasificar** (¿el techo relevante es de memoria o de
cómputo?), no diagnosticar oportunidades de optimización. Usar el techo
alcanzado realmente por ESE loop (p.ej. `Scalar Add Peak` si el loop no
vectorizó) introduciría un sesgo circular: un loop mal vectorizado
"parecería" compute-bound contra su propio techo bajo, aunque en realidad
tenga margen enorme de mejora — exactamente el tipo de conclusión que
Roofline está diseñado para exponer, no para ocultar. Es el mismo principio
que ya usa Hyperion: `P_pico` de ERT es un solo techo de máquina, no un
techo ajustado por kernel.

**Lo que sí varía por loop, con evidencia real, es la precisión** (DP vs
SP) — eso es una capacidad de hardware genuinamente distinta (dos picos
físicos distintos), no un artefacto de qué tan bien está escrito el código.
Por eso el compute roof varía por precisión (confirmada por `Data Types`),
no por ISA/vectorización lograda.

**Decisión pendiente, dejada explícita, no resuelta aquí:** no se confirmó
si `P_pico` de la calibración ERT de Hyperion corresponde específicamente
al pico FMA o al pico de suma vectorial — necesario solo si en algún
momento se quiere comparar numéricamente `i_ridge_advisor` contra
`i_ridge` de Hyperion (no para que cada uno sea internamente consistente,
eso ya está resuelto). Revisar la configuración real de ERT
(`kernels/ert/`, `orchestrator/calibration.py::_measure_bw_and_flops_peak`)
antes de intentar esa comparación numérica directa.

### 2.3 ¿Qué definición de ridge point se usa?

```
i_ridge_advisor = P_peak_flops_per_s / BW_peak_bytes_per_s
```

leído directamente de `advisor --report=roofs` (sección 1) — **nunca**
mezclado con el `i_ridge` de Hyperion (que sale de STREAM/ERT propios). Son
dos ridge points independientes, calculados con benchmarks independientes,
sobre el mismo hardware — si coinciden en orden de magnitud, es evidencia
cruzada real (ver §6); si no coinciden, hay que investigar cuál benchmark
es menos representativo, no promediarlos ni descartar la discrepancia.

### 2.4 ¿Cómo se tratan kernels con loops en regímenes distintos?

1. Se descartan del análisis los loops con `Self Time Percent` irrelevante
   (ver `HOT_LOOP_COVERAGE_FRACTION` abajo) — no se pondera cada loop por
   igual sin importar cuánto tiempo real ocupa.
2. **Umbral declarado, no mágico escondido:** se ordenan los loops por
   `Self Time` descendente y se toman los mínimos necesarios hasta cubrir
   `HOT_LOOP_COVERAGE_FRACTION = 0.80` (80%) del tiempo propio total del
   kernel — constante nombrada en `classify_roofline.py`, con esta misma
   justificación en el docstring: es el criterio estándar de "hot path"
   usado en profiling (regla 80/20), declarado y configurable, no derivado
   matemáticamente de nada más fundamental — **se marca explícitamente como
   parámetro de diseño, no como verdad matemática**.
3. Cada loop caliente se clasifica individualmente (`compute_bound` /
   `memory_bound` / `ambiguous_loop`, ver 2.6) usando su propia AI vs.
   `i_ridge_advisor` (con el compute roof de SU precisión, per §2.2).
4. La clase del **kernel completo** es la clase que concentra la mayor
   fracción de tiempo propio **entre los loops calientes ya clasificados**
   — si ninguna clase supera a las demás por un margen declarado
   (`KERNEL_DOMINANCE_MARGIN = 0.15`, 15 puntos porcentuales de tiempo
   propio entre la clase líder y la segunda), el kernel se marca
   `ambiguous` a nivel de kernel, incluso si loops individuales sí tenían
   veredictos confiados — un kernel genuinamente mixto (el caso ya
   anticipado para MG/FT/LU en `pipelinevtune/context/03_kernels_notas.md`)
   debe poder salir `ambiguous`, no forzarse a una clase por mayoría simple
   de loops.

### 2.5 ¿Qué es medición y qué es estimación/modelado? (resumen, detalle en la documentación del pipeline, sección 5)

| Cantidad | Medición o estimación | Mecanismo |
|---|---|---|
| FLOPs por loop | **Medición exacta** | Instrumentación binaria (Trip Counts + FLOP), cuenta cada instrucción FP ejecutada |
| Bytes en L1 (tráfico bruto) | **Medición exacta** | Instrumentación — cuenta cada acceso de carga/almacenamiento real |
| Bytes en L2/L3/DRAM | **Estimación por simulación** | Simulador de caché de Advisor sobre la traza instrumentada, extrapolado desde un subconjunto (ver limitación en el estudio) |
| Roofs (techos) | **Medición empírica** | Micro-benchmarks propios de Advisor corridos contra el hardware real de este nodo, con la misma afinidad (`taskset -c 0-5`) que el resto de la campaña |
| Tiempo por loop | **Medición por muestreo** | Pasada `survey`, overhead bajo, mecanismo de muestreo no detallado por Intel a nivel de evento (ver estudio §5) |
| Precisión (DP/SP) declarada por loop | **Medición exacta** | `Data Types`/columnas `Dynamic dp_compute`/`sp_compute`, de la instrumentación |
| `i_ridge_advisor` | **Derivado** | Cociente de dos mediciones empíricas (no una medición en sí) |
| Clasificación final | **Derivado + parámetros declarados** | Reglas de Roofline + umbrales nombrados en `classify_roofline.py` |

### 2.6 Relación con la metodología actual de Hyperion — sin mezclarlas

- **Nunca se sobrescribe ni se combina con `phase_label_train`.** La columna
  de este pipeline se llama `advisor_roofline_class` en el CSV consolidado
  — un nombre deliberadamente distinto, igual que
  `vtune_validation_class` en la campaña de VTune.
- **`i_ridge_advisor` nunca reemplaza a `i_ridge`** de la calibración
  STREAM/ERT de Hyperion — son dos números que pueden reportarse lado a
  lado, nunca promediados.
- Esta es la **tercera** fuente de validación cruzada independiente del
  proyecto (después de VTune Top-Down, congelado sin permiso funcional, y
  Advisor Roofline) — cada una con mecanismo físico y pipeline de
  procesamiento propio, exactamente la misma disciplina de capas ya
  documentada en `docs/vtune/vtune_cross_validation.md` §C.

**Umbral de margen de ambigüedad por loop (`AMBIGUOUS_AI_LOG_MARGIN`):**
un loop se marca `ambiguous_loop` (en vez de forzar compute/memory) si su
AI cae dentro de un factor de 1.25× de `i_ridge_advisor` en cualquier
dirección (`0.8 × i_ridge <= AI <= 1.25 × i_ridge`) — margen multiplicativo,
no de puntos porcentuales, porque AI e `i_ridge` son ambos valores de
intensidad (FLOP/byte) que se comparan en escala logarítmica en cualquier
gráfico Roofline, no fracciones de un mismo total como sí lo eran
`Memory Bound`/`Core Bound` en el Top-Down de VTune. Declarado, configurable,
justificado explícitamente por la advertencia ya documentada de que la
simulación de caché extrapola desde un subconjunto (no es exacta al 100%) —
un margen de tolerancia alrededor del ridge es la forma honesta de no
sobre-afirmar precisión que el propio mecanismo de medición no tiene.

---

## 2.7 Frecuencia de CPU — se registra, no se fija (agregado 2026-08-11)

Cada corrida ahora lee `scaling_cur_freq`/`scaling_governor`/`scaling_driver`
(mismo archivo exacto que `orchestrator/freqctl.py::read_observed_frequency_khz`
ya usa, FRQ-10) **antes y después** de la pasada `tripcounts+cache-sim` — la
que realmente alimenta la clasificación — y lo deja en `metadata.json` y en
`consolidated_characterization.csv` (`governor`, `scaling_driver`,
`freq_mhz_mean_before_cachesim`, `freq_mhz_mean_after_cachesim`,
`freq_drift_mhz`).

**Por qué importa:** tanto los roofs (`P_peak`/`BW_peak`, medidos por los
micro-benchmarks propios de Advisor) como el `GFLOPS`/`AI` de cada loop
dependen de la frecuencia real a la que corrió la CPU en ese instante. Si el
nodo tiene turbo/HWP activo (comportamiento por defecto, gobernado por
`intel_pstate`), dos repeticiones del mismo kernel pueden dar techos o
intensidades distintas por variación de reloj, no por ruido de medición de
Advisor — sin este registro, esa diferencia sería indistinguible de un error
real del pipeline.

**Decisión explícita: este pipeline NO fija la frecuencia.** Dos razones:
(1) no hay permiso de escritura de `cpufreq` confirmado para esta campaña
(P1 de `docs/retoma/pacca/Solicitud_Permisos_Pacca_Unicartagena.md` sigue
sin resolverse al momento de escribir esto); (2) aunque lo hubiera, fijar
frecuencia es responsabilidad de una campaña de *entrenamiento* controlada
(`campaign_pacca_ref.yaml` + `freqctl.py`), no de esta campaña de
*validación* — mezclar ambas responsabilidades introduciría acoplamiento
donde no hace falta. Si el `freq_drift_mhz` de una corrida resulta grande,
el pipeline lo deja como advertencia en el log y en el CSV — no aborta la
corrida ni intenta corregir nada.

---

## 3. Arquitectura del pipeline

```
advisor/
├── preflight_advisor.py         módulo + standalone: entorno, permisos, host/CPU/NUMA/cache
├── kernel_registry.py           descubre kernels, verifica compilación (flags/-g/checksum), NO recompila
├── run_characterization.py      orquestador: por kernel, corre survey+tripcounts con y sin cache-sim
├── advisor_report_parser.py     parsea los CSV oficiales de --report=survey/--report=roofs
├── classify_roofline.py         ridge point, hot loops, clasificación por loop y por kernel
├── run_roofline_unit_test.sh    (ya existía) prueba manual de un solo binario
├── sbatch_advisor_characterization.sh   campaña completa, sbatch (mismo patron que raperezp/)
└── README.md
```

Reutiliza deliberadamente el mismo patrón de módulos ya validado en
`Vtune/`/`raperezp/` (preflight separado, parser separado del clasificador,
orquestador que los conecta) — no por copiar sin pensar, sino porque ya se
discutió y aceptó esa separación de responsabilidades para la campaña de
VTune, y el mismo razonamiento aplica aquí.

### 3.1 Por qué la estructura de resultados NO es la sugerida literalmente

La propuesta original (`survey/`, `tripcounts_flop/`, `roofline/`,
`roofline_cache_sim/` como subcarpetas separadas por tipo de análisis)
**no coincide con cómo Advisor organiza sus propios datos**: `survey` y
`tripcounts` no son análisis independientes con resultados separados — son
dos pasadas de colección que escriben **al mismo `--project-dir`**, y los
reportes (incluido el Roofline) se generan leyendo ambas pasadas juntas de
ese mismo proyecto (confirmado en la prueba real: un solo `--project-dir`
contuvo ambas pasadas y el reporte de summary ya reflejaba las dos). Separar
`survey/` y `tripcounts_flop/` en carpetas distintas sugeriría, de forma
incorrecta, que son independientes entre sí.

Estructura real usada:

```
advisor_results/<kernel>_<clase>/
├── project_nosim/            <- --project-dir de Advisor: survey + tripcounts SIN --enable-cache-simulation
├── project_cachesim/         <- --project-dir de Advisor: survey + tripcounts CON --enable-cache-simulation
├── reports/
│   ├── survey_nosim.csv
│   ├── roofs_nosim.csv
│   ├── survey_cachesim.csv
│   ├── roofs_cachesim.csv
│   └── roofline_cachesim.html
└── metadata.json             <- comando exacto, duracion, exit code, checksum, host/CPU/NUMA
```

`nosim`/`cachesim` sí quedan separados — esa distinción es real y
metodológicamente relevante (dos preguntas distintas, tal como se pidió) —
pero `survey` y `tripcounts` viven juntos dentro de cada uno, reflejando
cómo Advisor realmente los trata.

---

## 4. Limitaciones y fuentes de error conocidas (resumen; detalle completo en `docs/advisor/estudio_intel_advisor_roofline.md`)

- La simulación de caché extrapola desde un subconjunto de accesos, no
  simula la jerarquía completa — fuente de error no cuantificada por Intel
  en la documentación disponible.
- `Self AI` (la columna que Advisor calcula por defecto) no se usa
  directamente porque no se pudo confirmar con certeza a qué nivel de
  memoria corresponde exactamente sin ambigüedad — el pipeline calcula su
  propia AI a nivel DRAM (`Self GFLOP / Self DRAM GB`) para evitar esa
  ambigüedad, al costo de no reutilizar un campo que Advisor ya da listo.
- El overhead de la pasada con instrumentación completa es alto (~17×,
  medido, no estimado) — la campaña completa debe presupuestar tiempo de
  cola/ejecución con ese factor, no con el tiempo baseline del kernel.
- La discrepancia de modelo de CPU (sección 0) es una alerta abierta, no
  resuelta en este documento.
- No se confirmó si el `P_pico` de ERT en Hyperion es comparable
  numéricamente al `DP Vector FMA Peak` de Advisor (sección 2.2) — pendiente
  si se quiere comparar los dos ridge points en valor absoluto.
