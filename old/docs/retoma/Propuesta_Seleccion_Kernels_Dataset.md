# Propuesta de selección de kernels para el dataset DVFS

Este informe responde a la pregunta de cuántos y cuáles kernels usar para
entrenar el modelo, partiendo de la lista de benchmarks candidatos
propuesta (STREAM, NPB-MG/CG/FT/IS/EP/BT/LU, HPCG, DGEMM). Amplía esa
lista con más candidatos, y explica con detalle qué significa "cubrir
área" del espacio Roofline — la pregunta que quedó abierta en la
conversación.

## 1. Qué significa "cubrir área" (el concepto detrás de la recomendación)

El modelo Roofline ubica cada kernel en un plano de dos ejes, ambos en
escala logarítmica:

- **Eje X — Intensidad operacional** (`I`, FLOP/byte): cuántas
  operaciones de punto flotante hace el kernel por cada byte que mueve
  desde/hacia memoria. Kernels que reutilizan mucho un dato en cache
  (multiplicación de matrices densas) tienen `I` alto; kernels que tocan
  cada byte una sola vez y no lo reutilizan (STREAM) tienen `I` muy bajo.
- **Eje Y — Rendimiento** (GFLOP/s).

El **ridge point** (`i_ridge = P_pico / BW_pico`, la incógnita que
preguntaste en el mensaje anterior) es el valor de `I` donde el techo de
cómputo (`P_pico`, una recta horizontal) y el techo de memoria (`BW_pico
× I`, una recta diagonal) se cruzan. A la izquierda del ridge point un
kernel es **memory-bound** (el ancho de banda de memoria es el límite);
a la derecha es **compute-bound** (la capacidad de cómputo es el
límite).

**"Cubrir área" quiere decir dos cosas, no una:**

1. **Cobertura a lo largo del eje X.** El modelo no debe entrenarse solo
   con ejemplos en los dos extremos (muy memory-bound / muy
   compute-bound). Las features que sí podemos medir con PMU (`ipc`,
   `llc_miss_rate`, `mpki`) cambian de forma **continua** a medida que
   `I` se acerca al ridge point, no con un salto brusco. Si el dataset
   solo tiene ejemplos lejos del ridge point (STREAM en un extremo, EP en
   el otro), el modelo nunca ve cómo se ven esas features *cerca* de la
   frontera de decisión — que es justamente la región donde más se
   equivocaría en producción.
2. **Diversidad de forma de acceso a memoria, no solo de valor de `I`.**
   Dos kernels pueden tener un `I` parecido pero un patrón de acceso muy
   distinto — acceso secuencial regular (STREAM, FT) vs. acceso disperso
   indirecto (CG) vs. acceso aleatorio (IS) — y eso produce firmas de
   `mpki`/`llc_miss_rate` diferentes aunque cerca en el eje X. Si el
   dataset solo tiene *una* forma de acceso por zona del eje, el modelo
   aprende a reconocer ese kernel específico, no el fenómeno físico
   (memory-bound vs. compute-bound) que necesita generalizar.

En otras palabras: no se trata de maximizar la cantidad de kernels, sino
de que los puntos que sí se toman formen una nube razonablemente
distribuida a lo largo del eje X y con formas de acceso distintas — eso
es "área" en el sentido de área *cubierta* del plano Roofline, no
cantidad de kernels.

## 2. Dónde cae cada candidato (aproximado, literatura — pendiente de confirmar con calibración real en felix)

Los valores de `I` son órdenes de magnitud conocidos en la literatura de
Roofline para estos benchmarks con tamaños de problema típicos; el valor
exacto en felix depende del tamaño de arreglo/clase elegido y **todavía
no se ha medido** (esa es la calibración real pendiente, F4.2/F4.3).

| Kernel | Zona aproximada | Patrón de acceso | Estado |
|---|---|---|---|
| STREAM | Extremo memory-bound (`I` ≈ 0.08–0.25) | Secuencial, sin reuso | ✅ compilado, calibration (no entra al dataset de entrenamiento) |
| NPB-IS | Memory-bound bajo | Disperso/aleatorio (enteros, buckets) | ✅ compilado, dataset |
| NPB-CG | Memory-bound | Disperso indirecto (matriz rala) | ✅ compilado, dataset |
| NPB-MG | Memory-bound a intermedio | Estructurado por niveles (stencil multigrilla) | ✅ compilado, dataset |
| NPB-FT | Intermedio | Regular pero con mucho tráfico (FFT) | ✅ compilado, dataset |
| NPB-LU | Intermedio | Bloques con reuso parcial | ✅ compilado, dataset |
| NPB-EP | Extremo compute-bound (`I` muy alto, casi sin memoria) | Ninguno relevante (generación de aleatorios) | ✅ compilado, dataset |
| **DGEMM/OpenBLAS** | **Ajustable** (cerca del ridge hasta muy compute-bound, según tamaño de bloque/matriz) | Denso, con reuso alto en cache | ❌ propuesto |
| NPB-BT | Intermedio (banda similar a FT/LU) | Bloques tridiagonales | ❌ propuesto (slide) |
| HPCG | Memory-bound (banda similar a CG, más "moderno") | Disperso indirecto (solver iterativo) | ❌ propuesto (slide) |

## 3. Más candidatos (ampliando la lista original)

Para que la decisión no se quede solo con lo que ya trajiste, esta es una
lista más amplia de benchmarks conocidos que podrían aportar diversidad
de patrón de acceso, organizados por la zona del Roofline que ocuparían:

**Memory-bound, patrones distintos a los que ya tenemos:**
- **RandomAccess/GUPS** (HPC Challenge): acceso puramente aleatorio a un
  arreglo enorme — el extremo más agresivo de "sin localidad" que existe,
  útil para anclar el borde izquierdo del eje X con más densidad.
- **Graph500 / BFS** sobre grafos grandes: acceso por *puntero* (seguir
  aristas), un patrón de "pointer chasing" que ni STREAM ni CG capturan
  — memory-bound pero por latencia, no por ancho de banda, un matiz
  distinto que vale la pena si el modelo necesita generalizar a cargas
  de grafos.
- **SpMV standalone** (multiplicación matriz dispersa × vector, sin todo
  el aparato de un solver iterativo completo como HPCG/CG): más simple
  de compilar que HPCG, con el mismo patrón de acceso disperso.

**Intermedio (banda cercana al ridge point):**
- **Stencil genérico** (ej. Laplaciano 7 puntos en 3D, tipo mini-app):
  patrón estructurado distinto al multigrid de MG, más simple de ajustar
  el tamaño de working set con precisión para acercarse al ridge point.

**Compute-bound, alternativas a EP/DGEMM:**
- **FFTW optimizada** (distinta de NPB-FT, que usa una FFT propia del
  suite, no una librería tuneada): mismo tipo de operación pero con una
  estrategia de reuso de cache muy distinta, cambia dónde cae `I`.
- **N-Body / simulación gravitacional**: compute-bound con `I` ajustable
  según el radio de corte o el algoritmo (todos-contra-todos vs.
  Barnes-Hut), otro punto tunable como DGEMM.

Ninguno de estos está compilado ni evaluado — se listan para que la
decisión de ampliar el dataset más adelante tenga opciones concretas, no
solo "buscar algo más".

## 4. Por qué 7 alcanza para la primera versión del dataset

**Los 7** = los 6 ya compilados y verificados en felix (EP, MG, CG, IS,
FT, LU) **+ DGEMM**.

1. **Los 6 actuales ya dan diversidad real de patrón de acceso**, no solo
   de posición en el eje X: secuencial (FT), disperso indirecto (CG),
   estructurado por niveles (MG), aleatorio de enteros (IS), bloques con
   reuso parcial (LU), y el extremo casi-sin-memoria (EP). Eso ya cubre
   la mayoría de las "formas" de acceso que importan para que el modelo
   no memorice un kernel específico.

2. **El hueco real era un punto compute-bound *ajustable* cerca del
   ridge**, porque EP es un extremo tan pronunciado (casi no toca
   memoria) que no ayuda a poblar la zona *cercana* a la frontera de
   decisión desde el lado derecho. **DGEMM resuelve esto con una sola
   entrada de catálogo**: variando el tamaño de matriz/bloque se pueden
   generar varios `size_variant` que caen en distintos puntos del eje X
   — de "cerca del ridge" a "muy compute-bound" — sin compilar un
   binario nuevo por cada punto. Es, en la práctica, varios kernels de
   cobertura en una sola integración.

3. **El costo marginal de cada kernel nuevo no es solo compilarlo.** Ya
   lo vivimos con NPB/STREAM/ERT: cada binario nuevo implica repetir
   F3.2 (checksums y regex reales contra stdout real) y F3.3 (medir
   tiempo real en felix para elegir la clase/tamaño correcto) contra
   hardware real, no en mocks. HPCG en particular añade una dependencia
   de MPI y un formato de configuración propio — el costo de
   integración es visiblemente mayor que su beneficio marginal, dado que
   CG ya cubre esa misma zona del plano (memory-bound, acceso disperso).

4. **Coherente con cómo está diseñado el resto del plan**: F4.4 pide
   arrancar con un piloto *mínimo* (2 kernels: `npb_ep` + `npb_mg`) antes
   de ir a la matriz completa — la filosofía del proyecto ya es "validar
   con lo mínimo primero, ampliar después con evidencia". Los "7" son
   exactamente ese mismo principio aplicado a la selección de kernels: un
   MVP de cobertura razonable, no un techo permanente. Si al entrenar el
   primer modelo real (después de F4.5, con el gate INT-T08 en verde) se
   ve que el modelo generaliza mal en alguna zona específica del plano
   (por ejemplo, cerca del ridge point, o en accesos tipo grafo), la
   lista de la sección 3 ya deja opciones concretas para una segunda
   ronda dirigida por evidencia real, no por intuición.

## 5. Recomendación concreta

- **Implementado (2026-08-01, ARC-38):** `DGEMM/OpenBLAS` agregado al
  catálogo con checksum real de felix (`dgemm_n2048`, N=2048, ~12.18 s,
  `Verification SUCCESSFUL`). El dataset queda en 7 kernels. Preflight
  completo re-verificado con el catálogo de 7: 46/46 checks en verde.
- **Más adelante, solo si el modelo lo pide:** evaluar `RandomAccess`,
  `SpMV` o `Graph500-BFS` como segunda ronda dirigida por dónde falla el
  modelo entrenado, no antes.
- **HPCG y BT:** no los priorizaría — el costo de integración (MPI para
  HPCG; otro binario NPB completo para BT) no se justifica frente a la
  cobertura que ya dan CG (para HPCG) y FT/LU (para BT).

Esto sigue siendo una recomendación, no una decisión tomada — H4 (alcance
definitivo, decisión del director) sigue pendiente y esta lista es
insumo para esa conversación, no un reemplazo de ella.
