# Estrategia GPU — Fase 2 (documento para el director)

**Propósito.** No es un cambio de objetivos del anteproyecto — es la
corrección de *cómo* los estamos cumpliendo, basada en evidencia empírica
propia contrastada contra la literatura ya citada en el marco teórico.
§9 mapea cada punto a los Objetivos Específicos 1–4; §10 da la lista de
pruebas que permite verificar (o refutar) cada afirmación de este
documento.

Fuente completa: `docs/general/PLAN_MAESTRO_FASE2.md`, Anexos K, L, M, N.
Este documento es la síntesis ejecutiva.

> **Nota de auditoría (2026-08-24).** Este documento fue revisado contra
> los datos crudos y se corrigieron cinco errores de la versión anterior:
> (a) atribución de las 144 corridas del job 6462, (b) el rango de ahorro
> decía 8.7% cuando el mínimo medido propio es 7.7%, (c) el conteo de
> kernels de RAJAPerf estaba inflado al doble por un artefacto de conteo,
> (d) una afirmación causal sobre la grilla presentada como establecida
> cuando aún no hay dato, (e) una afirmación de ausencia de fases
> intra-corrida que es circular (§3). Se documentan aquí para que la
> corrección quede trazable, no se borran.

---

## 1. Qué se hizo (Objetivo Específico 1)

Telemetría de bajo nivel (NVML para GPU; RAPL+Perf para el lado CPU
delegado) sobre **7 kernels GPU**, repartidos en dos campañas:

| job | kernels | niveles GPU | niveles CPU | reps | corridas |
|---|---|---|---|---|---|
| 6462 | 4 (`gaussian`, `dgemm_n4096`, `heartwall`, `lavamd`) | 6 | 2 (REF, F4) | 3 | 144 |
| 6463 | 3 (`myocyte`, `backprop`, `dwt2d`) | 6 | 1 (REF) | 3 | 54 |

Ambas 100% aceptadas (144/144 y 54/54, 0 rechazos). Antes de correrlas se
corrigió un problema de reproducibilidad de `nvcc` no detectado hasta
entonces (ARC-193) y se verificaron los checksums de los binarios.

## 2. El error de métrica que se corrigió, y por qué importa contarlo

La primera medición sumó **GPU + paquete CPU + DRAM** (energía total del
nodo). Con esa métrica el DVFS de GPU parecía no poder pagar nunca: un
recorte de reloj de 6.7× compraba solo 10–41% de caída de potencia
(Anexo L).

**Esa métrica es más estricta que la de la línea de trabajo con la que nos
comparamos.** Los trabajos de predicción de frecuencia óptima en GPU
(`Fan2020`, `Guerreiro2019`) miden **energía de GPU**, que es lo que una
política de GPU realmente controla; el piso de potencia del host es una
propiedad del nodo de medición, no del mecanismo bajo estudio. Con esa
métrica el resultado se revierte (Anexo M, CPU=REF, contra "siempre F0"):

| kernel | mejor nivel | ahorro E_gpu | costo de tiempo | ganancia EDP |
|---|---|---|---|---|
| `rodinia_lavamd` | F1 (1110 MHz) | **25.11%** | +10.02% | **17.60%** |
| `rodinia_heartwall` | F1 | **18.24%** | +33.12% | 1.03% |
| `rodinia_gaussian` | F1 | **15.35%** | +25.62% | 1.69% |
| `rodinia_myocyte` | F1 | **7.66%** | +28.02% | 0.00% |

Rango medido propio: **7.7–25.1%**, comparable a lo publicado (8.7–23.1%
en entrenamiento DNN; 20.2–26.7% con escalado consciente de la aplicación
en V100/A100).

**Matiz honesto que hay que declarar, no esconder:** no existe un estándar
único en el campo sobre qué energía medir. Un survey de técnicas de DVFS
en GPU documenta que distintos trabajos miden distinto, y reporta trabajo
previo enfocado en energía *a nivel de sistema* que concluye que el DVFS
de GPU afecta la energía del sistema **menos** que el DVFS de CPU — es
decir, **el hallazgo del Anexo L es un resultado conocido y publicado, no
un artefacto de nuestro montaje**. Por eso este trabajo **reporta ambas
métricas**: energía de GPU como primaria (comparabilidad con la línea de
predicción de frecuencia), energía total del nodo como limitación
declarada del alcance. Reportar las dos es más fuerte que cualquiera sola.

## 3. Granularidad del modelo: por qué no clasifica fase intra-ejecución

**Lo que se creía necesario:** un clasificador de fase por ventana,
análogo al de CPU.

**El impedimento técnico (CAT-10), y su causa real:** en GPU la intensidad
operacional se declara **estática por kernel** en el catálogo, medida
offline con `ncu`. No es una elección de conveniencia: los contadores de
tráfico DRAM (`dram__bytes.sum`) no están en NVML (API de monitoreo) sino
en la Profiling API de CUPTI, que **requiere replay del kernel** —
relanzarlo para multiplexar contadores. Eso (a) distorsiona el tiempo y la
energía que este trabajo mide, y (b) entrega el dato **por lanzamiento de
kernel**, no por ventana de tiempo: para nuestros kernels reales, que son
uno o pocos lanzamientos largos, ni pagando el costo se obtiene
intensidad dinámica por ventana. Es una asimetría de categoría respecto de
`perf_event_open`+uncore en CPU, no de presupuesto. **No es un permiso
caído**: se verificó que `NVreg_RestrictProfilingToAdminUsers` no está
activo en pacca — `ncu` corre sin privilegios.

**Lo que el dato propio muestra (Anexo K, `gpu_policy_headroom.py`):** el
óptimo **de corrida completa** es constante por kernel y varía **entre**
kernels.

> **Límite lógico de esta afirmación, declarado explícitamente.** De que
> el óptimo de corrida completa sea constante **no se sigue** que no
> exista alternancia de fases dentro de la corrida: es precisamente lo
> que la instrumentación de GPU no puede observar. Afirmar su ausencia
> sería circular. Lo defendible es más débil y suficiente: **con la
> telemetría disponible, no hay evidencia de alternancia explotable, y no
> existe vía de medición que pueda producirla.** El test V4 (§10) es el
> único que puede cerrar este hueco, usando `gpu_phasic` — que alterna
> fases por construcción y emite marcas de verdad (`PHASE`,
> `T0_MONOTONIC_NS`) cruzables offline contra `gpu_power_mw`.
>
> **Resultado de V4 (job 6476, 2026-08-25).** Cruzando offline las marcas
> reales contra `gpu_power_mw` por ventana: a período de fase de 1 s las
> dos fases **sí son distinguibles en potencia**, con una diferencia
> grande y consistente en los cuatro niveles fijos (F1: −23.05%, F2:
> −29.20%, F3: −27.47%, F4: −18.72%; la fase de memoria consume *más*
> potencia que la de cómputo — coherente con que la persecución de
> punteros en SIMT mantiene miles de hilos concurrentes saturando el
> controlador de memoria, mientras que el FMA sin tráfico puede no
> alcanzar la misma ocupación). A 100 ms la diferencia ya es débil
> (0.5–8%, sin dirección consistente) y a 10 ms prácticamente
> desaparece (−0.9% a +7.9%) — exactamente lo que predice el límite de
> muestreo de NVML (~100 ms), confirmando el control negativo por
> diseño.
>
> **La segunda mitad del criterio declarado (¿el nivel óptimo difiere
> entre fases?) no se pudo evaluar con este instrumento, y es una
> limitación de diseño, no un dato faltante**: `gpu_phasic` fija la
> duración de cada fase **por tiempo**, no por trabajo (sección
> `sec:resultados-multifasico` del libro) — precisamente para poder
> comparar la proporción cómputo/memoria entre niveles de frecuencia sin
> que un recuento fijo de iteraciones se alargue de forma desigual. Esa
> misma decisión de diseño hace que la duración de una fase **no
> responda a la frecuencia por construcción**, así que no hay señal de
> tiempo por fase de la que derivar un óptimo por fase, solo potencia.
> **Conclusión:** hay evidencia positiva de alternancia real y detectable
> en potencia (cierra el hueco lógico de arriba), pero no evidencia —ni
> en un sentido ni en el otro— sobre si esa alternancia sería
> aprovechable por una política de frecuencia; medirlo exigiría un
> microbenchmark cuya fase termine por trabajo, no por tiempo, lo que
> reintroduciría el problema de comparabilidad entre niveles que el
> diseño actual evita a propósito.

**La granularidad elegida es la que usa la literatura publicada:**

- `Guerreiro2019` (ya citado): clasifica la **aplicación**; observa
  contadores **en una sola frecuencia de referencia** y predice el resto.
  16% de ahorro promedio, 0.74% de desviación respecto al óptimo real.
- `Calore2017` (ya citado): probó ajuste función-por-función y lo abandonó
  explícitamente (*"clock tuning on a function-by-function basis is not
  convenient"*), pivotando a frecuencia constante por programa.
- `Antici2024` (ya citado, MCBound, producción en Fugaku): clasifica el
  **job completo**, F1-macro ≥ 0.89.

El Objetivo 2 pide un modelo clásico que clasifique "las fases de
ejecución… basándose en la telemetría", **en tiempo de ejecución y con
baja latencia**. Nada en ese texto exige que la fase cambie *dentro* de
una corrida; exige que la inferencia sea en vivo y barata. Eso se
mantiene: lo que cambia es la unidad clasificada (carga/kernel, no ventana
de 1 ms), que es la unidad de los tres sistemas publicados citados.

## 4. Arquitectura propuesta

- **Entrada:** telemetría observada en **un solo nivel de referencia**
  (F0) — `gpu_util_pct`, `gpu_mem_util_pct`, potencia sobre reposo, y el
  cociente `gpu_mem_util_pct / gpu_util_pct` como proxy de
  memory-boundness — más la frecuencia candidata como feature.
- **Salida:** dos regresiones por par (carga, nivel): `E(f)/E(F0)` y
  `T(f)/T(F0)`.
- **Política:** minimizar EDP (fiel al Objetivo 4) y, como variante,
  minimizar energía sujeta a un presupuesto de degradación explícito —
  el presupuesto queda como **parámetro de política, no horneado en el
  entrenamiento** (cambiarlo no exige reentrenar).
- **Anti-fuga:** prohibido todo lo que provenga de la corrida del nivel
  candidato (su tiempo, energía o potencia). Solo features de F0 + la
  frecuencia candidata. Verificable por test (V5, §10).
- **Evaluación:** leave-one-kernel-out contra la **mejor constante única**,
  elegida **solo sobre los folds de entrenamiento** — si se elige mirando
  el kernel de prueba, el baseline hace trampa y el modelo parece mejor de
  lo que es (V6, §10).

**Por qué regresión por par y no clasificación por kernel:** con N cargas
y M niveles hay N×M muestras en vez de N, y el presupuesto queda
parametrizable. Es además la formulación de `Fan2020`/`Guerreiro2019`.

## 5. El margen real sobre la mejor constante, medido

`gpu_policy_headroom.py` compara tres políticas para cuantificar cuánto
puede ganar un modelo por encima de una regla trivial. **Tabla sobre 6
kernels** — excluye `rodinia_backprop`, cuya energía de GPU en F0 es
8.2 J (dos órdenes de magnitud bajo el resto: sus porcentajes son ruido
dividido por casi cero):

| presupuesto de degradación | mejor constante | oráculo | margen del modelo |
|---|---|---|---|
| ≤4% | F0 → 0% | 0.58% | **0.58 pts** |
| ≤15% | F0 → 0% | 4.76% | **4.76 pts** |
| sin límite | F1 → 7.71% | 11.41% | **3.70 pts** |

Lectura honesta: con la grilla de 6 niveles, **una sola constante (F1)
captura el 68% del oráculo**, y bajo un presupuesto estricto de 4% el
margen aprovechable es de apenas 0.58 puntos.

**Hipótesis puesta a prueba: la causa era la resolución de la grilla, no
la física — CONFIRMADA (job 6471, 210/210 aceptadas, V1 y V2, §10).** Con
los 4 escalones nuevos entre F0 y F1 (G1–G4, 60 MHz de paso en vez de
300), sobre los mismos 6 kernels con energía medible:

| presupuesto de degradación | mejor constante | oráculo | margen del modelo |
|---|---|---|---|
| ≤4% | REF → 0.56% | 2.36% | **1.80 pts** (antes 0.58) |
| ≤10% | REF → 0.56% | 9.62% | **9.06 pts** |
| ≤15% | G1 → 4.51% | 10.08% | **5.57 pts** (antes 4.76) |
| sin límite | G3 → 8.00% | 12.33% | **4.33 pts** (antes 3.70) |

El margen no solo mejoró — **el mayor margen de todo el proyecto hasta
hoy aparece en ≤10%** (9.06 puntos), justo donde antes no había ni
siquiera un punto de medición. Ahí los óptimos por kernel divergen de
verdad: `heartwall`/`gaussian` prefieren G2, `lavamd` G4, `myocyte` G2,
mientras que `dgemm_n4096` y `dwt2d` casi no se mueven de REF/F0 — es
exactamente el patrón de "diferentes cargas quieren diferentes niveles"
que justifica entrenar un modelo en vez de fijar una constante. Un
hallazgo colateral no anticipado: en los presupuestos más estrictos
(≤4%, ≤10%) **la mejor constante única es `REF`** (gobernador nativo),
no ningún nivel fijo — el autoboost del driver ya se acerca más al óptimo
bajo esas condiciones que cualquier candado fijo.

## 6. Estado de las corridas (2026-08-25, actualizado tras terminar)

El job ajeno que ocupaba el nodo ~15 h terminó (`CANCELLED`, no por
nosotros) el 2026-08-25; toda la batería encolada corrió y terminó.

- **Job 6471** — grilla fina, 210/210 aceptadas. **V1 pasa limpio**: los
  márgenes de potencia interpolados de G1-G4 no causaron rechazo masivo
  (21/21 en los 10 niveles, incluidos los 4 interpolados). **V2 confirma
  la hipótesis de §5**: la rejilla fina abre margen real, con el hallazgo
  más grande del proyecto hasta hoy en ≤10% (9.06 pts).
- **Job 6472** — barrido de tamaño de `dwt2d`, completado. **Hallazgo sin
  cerrar, no un resultado**: energía y tiempo de GPU en F0 salen
  prácticamente idénticos entre los 5 tamaños (192 a 8192 px, 42× de
  rango; energía 271-281 J, tiempo 5.0-5.5 s los 5). Se descartó un
  archivo de entrada truncado (tamaños de archivo verificados: 110 KB a
  201 MB, proporción correcta) y un argumento mal pasado (el binario
  reporta la dimensión correcta en `stdout.txt`). Queda sin aislar si el
  tiempo medido está dominado por overhead fijo del arnés (inicialización
  CUDA, sincronización del colector) sobre un cómputo DWT que en sí
  podría ser rapidísimo en una A100 — **no se reporta todavía como "la
  energía de dwt2d no depende del tamaño"**, es una pregunta abierta.
- **Job 6476** — cruce de fases `gpu_phasic` (V4), resultado en §3.
- **Job 6474** — reloj de memoria (V8), **NEGATIVO confirmado**: la A100
  expone un único reloj de memoria (1215 MHz). Riesgo 2 cerrado con
  evidencia (ver más abajo).
- **Instrumento de tamizaje ya validado** (Anexo K.4): `nvidia-smi -lgc`
  escala solo el reloj de SM, no el de memoria, así que el margen debería
  vivir en kernels limitados por ancho de banda. Verificado gratis con la
  calibración: `gpu_stream_bw` da α=0.071 frente a α=0.6–0.8 de los
  kernels de cómputo.

## 7. Riesgos abiertos, sin adornar

1. ~~Márgenes interpolados del job 6471~~ **CERRADO (2026-08-25)**: V1
   pasó limpio, 210/210 aceptadas, 21/21 en cada uno de los 10 niveles.
2. ~~El reloj de memoria nunca se probó~~ **CERRADO, NEGATIVO (job 6474,
   V8, 2026-08-25)**: la A100 expone un único reloj de memoria soportado
   (1215 MHz). No hay segundo mando de DVFS en este hardware — la línea
   de trabajo de `Fan2020`/`Guerreiro2019` (escalar núcleo y memoria
   juntos) no es aplicable aquí, se cierra con evidencia y no queda
   pendiente.
3. **Con 7–13 kernels el LOKO entrena sobre 6–12** — es un piloto, no un
   resultado estadísticamente robusto. **Conectado con la limitación de
   granularidad de §3 (CAT-10), no independiente de ella (2026-08-25):**
   el modelo trata cada kernel como una unidad de OI estática (§4), así
   que la variabilidad intra-kernel no puede compensar el N pequeño
   aunque exista físicamente. Y existe: la potencia dentro de una sola
   corrida REF/REF varía con CV=51.6% en `lavamd`, 25.4% en `dgemm_n4096`,
   19.5% en `gaussian`, 11.3% en `heartwall`, 1.4% en `myocyte` (coherente
   con que `myocyte` sea el kernel de menor ahorro, §2) — pero sin marcas
   de verdad como las de `gpu_phasic`, esa variación no se puede atribuir
   a alternancia de régimen explotable vs. transitorios de arranque/cierre
   (mismo problema que `dwt2d`, riesgo 8). No es una arista nueva: es
   exactamente el límite que `main.tex` §discusión-granularidad ya declara
   ("en GPU el argumento es más débil… se ha establecido que no puede
   medirse"), documentado aquí explícitamente unido al riesgo 3 en vez de
   por separado.
4. ~~La arquitectura reformulada de §4 nunca se ha entrenado ni
   evaluado~~ **EJECUTADO (2026-08-25, job 6532,
   `classifier/analysis/loko_pilot.py`) — RESULTADO NEGATIVO, y es el
   hallazgo más importante de la sesión.** El modelo **pierde contra no
   hacer nada** en ambos ejes:

   | política (EDP loss, 1.0 = óptimo) | GPU (6 kernels) | CPU (9 kernels) |
   |---|---:|---:|
   | oráculo (techo) | 1.0000 | 1.0000 |
   | **trivial: siempre a F0** | **1.0507** | **1.0010** |
   | modelo aprendido (LOKO) | 1.0925 | 1.0027 |
   | mejor constante honesta (V6) | 1.0940 | 1.0010 |
   | **margen del modelo vs. trivial** | **−0.0418** | **−0.0017** |

   Dos lecturas, que **no deben mezclarse**:

   - **CPU: el techo mismo es 0.1%.** Un oráculo *perfecto* solo mejora
     0.0010 sobre no hacer nada. No es que el modelo falle: no hay nada
     que ganar con este catálogo. Es un resultado reportable y cierra la
     pregunta del lado CPU. (Los números bajaron de 1.0070 a 1.0010 al
     corregir un bug propio del piloto que metía `stream_official` y
     `ert_probe` —kernels de calibración— como pliegues del LOKO;
     `stream_official`, el único con α bajo el umbral, aportaba casi todo
     el margen aparente.)
   - **GPU: hay 5.07 pts de techo real y el modelo no captura ninguno.**
     El modo de fallo es concreto: elige G2 para `dwt2d` (EDP real
     1.2071, un 20% *peor* que no tocar nada) y F1 para `heartwall`
     (1.0630 cuando G2 daba 0.9361). Con 6 kernels, LOKO entrena sobre 5
     — es el riesgo 3 materializado y ahora cuantificado.

   **Corrección metodológica que este resultado obliga:** V6 comparaba
   contra la mejor constante honesta, pero **esa no es rival suficiente**.
   La constante honesta se elige por pliegue y puede salir PEOR que
   quedarse quieto (en el pliegue de `dwt2d` la media de entrenamiento
   favorece G2, que ese kernel detesta): 1.0940 contra 1.0507 del trivial.
   Un modelo que le gana a ella (+0.0015, ruido) pero pierde contra "no
   hacer nada" (−0.0418) **no justifica existir**, y reportar solo la
   primera comparación habría hecho pasar por logro justo lo contrario.
   El margen contra el trivial es ahora el número titular del piloto.

   **Variante con umbral de acción (job 6533): acota la pérdida a cero,
   no produce ganancia — y su diagnóstico explica todo lo anterior.**
   Regla: desviarse de F0 solo si la mejora predicha supera el RMSE del
   propio modelo, calculado únicamente sobre pliegues de entrenamiento
   (honesto por construcción, no sintonizado contra el test). Resultado:
   el umbral **nunca se dispara** y la política degenera exactamente en el
   trivial — GPU 1.0507, CPU 1.0010, ambos idénticos a no hacer nada.

   La razón está en la propia magnitud del umbral, y es **el número que
   explica el fracaso**:

   | eje | RMSE del modelo | techo disponible | razón |
   |---|---:|---:|---:|
   | GPU | 0.2755 | 0.0507 | error **5.4×** mayor que el premio |
   | CPU | 0.0916 | 0.0010 | error **92×** mayor que el premio |

   El modelo no puede encontrar una ganancia de 5% cuando su propia barra
   de error sobre el EDP es de 27.5%. **No es un problema de
   hiperparámetros ni de elección de regresor: la incertidumbre del modelo
   supera al fenómeno que intenta explotar.**

   **CAUSA RAÍZ, medida (job 6535,
   `classifier/analysis/loko_feature_diagnostic.py`): el N efectivo no es
   el número de filas, es el número de KERNELS.** Las features son
   promedios de la corrida de referencia, así que apenas varían entre
   repeticiones del mismo kernel — su CV intra-kernel es de 0.5–5% frente
   a 30–297% entre kernels (razón 17× a 64×). Las repeticiones son
   réplicas del mismo punto, no muestras nuevas:

   | | GPU | CPU |
   |---|---:|---:|
   | filas del dataset por par | 162 | 450 |
   | **vectores de features distintos** | **18** | **90** |
   | kernels = pliegues LOKO | 6 | 9 |
   | **puntos efectivos de entrenamiento** | **5** | **8** |
   | features del modelo | 2 | 7 (6 útiles) |
   | componentes para el 90% de la señal | 2 de 2 | 5 de 7 |

   En GPU son **2 features generalizando desde 5 puntos a un sexto**; con
   eso, RMSE=0.2755 no es una anomalía a diagnosticar sino lo esperable.
   En CPU el problema es de otra forma: **7 features para 8 puntos**
   (p≈n), sobreajuste garantizado por construcción — y una de ellas,
   `ref_running_ratio`, tiene **varianza cero** (constante intra y entre
   kernels): ocupa un grado de libertad sin aportar nada y debe quitarse.

   Causa secundaria, aún sin aislar: el objetivo abarca de 0.85 a 12.4
   —F3/F4 son órdenes de magnitud peores— así que el error cuadrático se
   concentra en niveles extremos que a la política no le importan,
   mientras la región accionable (EDP≈1) queda mal resuelta.

   Lo único aprovechable hoy: **la regla de umbral es un mecanismo de
   seguridad válido para el daemon del Objetivo 3** — convierte una
   pérdida de 4.18 pts en un empate, es decir, garantiza no empeorar
   respecto del gobernador nativo aunque el modelo se equivoque.

   Pendientes de este hilo, en orden de valor esperado:
   1. **Repetir el piloto sobre el dataset de 17 kernels del job 6529**
      (12 pliegues tras exclusiones → 11 puntos efectivos, más del doble
      que hoy). Ataca la causa raíz. Honestidad sobre su alcance: pasar de
      5 a 11 puntos **ayuda, no transforma** — sigue siendo un piloto, que
      es exactamente lo que el riesgo 3 ya declaraba.
   2. **Quitar `ref_running_ratio`** (varianza cero) y, en CPU, reducir la
      dimensionalidad para que p < n en vez de p≈n.
   3. Predecir en espacio logarítmico o restringir los niveles candidatos
      a la región accionable, que ataca la causa secundaria.
   4. **Enriquecer las features sin cambiar de granularidad**: hoy solo se
      usa la media de la corrida de referencia. Percentiles y dispersión
      de la misma telemetría (p10/p50/p90, CV) añadirían dimensiones sin
      necesitar más kernels ni etiqueta por ventana — atacan el N efectivo
      por el lado de la riqueza del punto, no del número de puntos.
5. **La OI de los 3 tamaños intermedios de `dwt2d` está interpolada**, no
   medida con `ncu`. No se usa como feature del modelo (evita CAT-10 por
   diseño), pero está declarado en el catálogo.
6. **El ajuste de α resultó inválido para los kernels del tamizaje**
   (r²=0.53–0.63, Anexo L.1): el modelo de Amdahl no describe kernels que
   saturan a bajo reloj. α sirve como tamiz cualitativo, **no** como
   número reportable para esos casos. Mismo patrón se repitió en el
   tamizaje RAJAPerf-CUDA (§8): 4/6 candidatos con r²>0.97, pero
   `Basic_REDUCE3_INT`/`Basic_INDEXLIST_3LOOP` con r²=0.75/0.58 — incluidos
   igual en el catálogo final (decisión 2026-08-25), con la misma reserva:
   α cualitativo, no reportable como número preciso para esos dos.
7. ~~La variante CUDA de RAJAPerf no está compilada~~ **CERRADO
   (2026-08-25).** Compilada y verificada en `paccaA100`
   (`build_rajaperf_cuda.sh`), tamizaje corrido (§8).
8. ~~Energía y tiempo de `dwt2d` en F0 no varían con el tamaño~~
   **CAUSA RAÍZ ENCONTRADA (2026-08-25).** Cruzando `gpu_util_pct` por
   ventana contra el reloj de pared: incluso el `rodinia_dwt2d` original
   (16384×16384, el que sí muestra actividad real, hasta 93% de
   utilización) tiene su ráfaga de trabajo GPU genuino en solo **0.17 s,
   al final de una ventana total de 4.49 s** — más de 4.3 s son overhead
   de host (decodificar el bitmap, inicialización de CUDA) antes de que
   el kernel arranque. Para los 4 tamaños del barrido (192–8192 px, la
   mitad o menos del original), esa ráfaga real es tan corta que nunca
   se ve con claridad en el muestreo de NVML (utilización máxima
   6–33%, nunca un plateau limpio). **Consecuencia: las 4 variantes de
   tamaño no aportan la diversidad de carga que se buscaba** — miden,
   en la práctica, el mismo overhead de host disfrazado de 4 kernels
   distintos, el mismo modo de falla ya encontrado y tratado para
   `rodinia_lud` (GPU en reposo, excluido del catálogo). No se
   recomienda contarlas como 4 regímenes distintos en el análisis de
   margen ni en el conteo de diversidad de §8 — mismo tratamiento que
   ya recibe `rodinia_backprop` (excluido de `gpu_policy_headroom.py`
   por energía despreciable): excluir de análisis de margen, no
   necesariamente del dataset físico.
   **Verificado en las 3 repeticiones de cada tamaño, no solo la
   primera** (el patrón es idéntico: `s192` 6% en las tres, `s8192`
   32–33% en las tres, el original 82–100% con la ventana activa
   siempre en el último medio segundo de la corrida) — no es ruido de
   una corrida particular. **Decisión operativa para el pipeline de
   entrenamiento:** excluir estos 4 kernels del LOKO y del cálculo de
   margen (`gpu_policy_headroom.py`/`pair_dataset.py`) en el dataset
   final. (Nota: el job 6477, que originalmente correría estos datos,
   fue cancelado el 2026-08-25 en favor del impulso de RAJAPerf-CUDA de
   §8 — el dataset final que reemplaza a 6477 aún no se ha lanzado.)

## 8. Impulso: el banco de kernels disponible es mayor que el usado

Cuántos kernels usa cada trabajo citado: `Guerreiro2019` 35 (5 suites),
`Calore2017` **2** (una sola app, y aun así resultado real y citable),
`Hebbar2022` 43 (SPEC CPU2017 — licencia paga, no reproducible por
nosotros), `Antici2024` producción a escala Fugaku. El rango es enorme:
**el número no es lo que decide, sino la diversidad de régimen cubierta.**

Contra ese criterio, el catálogo GPU actual está sesgado a
compute/balanced, con 1–2 candidatos memory-bound reales. **RAJAPerf ya
está descargado en pacca** y el catálogo usa **1** de sus kernels. Conteo
verificado (2026-08-24, despojando sufijos de backend correctamente —
el conteo anterior estaba inflado al doble):

| categoría | kernels distintos |
|---|---|
| `apps` | 22 |
| `basic` | 20 |
| `polybench` | 13 |
| `lcals` | 11 |
| `algorithm` | 8 |
| `comm` | 6 |
| `stream` | 5 |
| **total** | **85** |

`raja-perf.exe` corre cualquiera con `-k NOMBRE -v <variante>`: agregar un
kernel es un wrapper, no una compilación.

**Resultado del impulso (2026-08-25).** Variante CUDA compilada y
verificada en `paccaA100` (riesgo 7, cerrado). Tamizaje sobre 6
candidatos elegidos por familia de acceso no cubierta (`Stream_COPY`,
`Stream_TRIAD` — ancho de banda, control positivo; `Basic_REDUCE3_INT`,
`Basic_INDEXLIST_3LOOP` — reducción/compactación; `Polybench_JACOBI_2D`,
`Polybench_HEAT_3D` — stencils, que en CPU fallaron el umbral pero en GPU
no tienen por qué):

| kernel | α | r² | veredicto |
|---|---:|---:|---|
| `Stream_TRIAD` | 0.027 | 0.999 | claro |
| `Stream_COPY` | 0.061 | 0.992 | claro |
| `Polybench_HEAT_3D` | 0.085 | 0.999 | claro |
| `Polybench_JACOBI_2D` | 0.123 | 0.973 | claro |
| `Basic_REDUCE3_INT` | 0.010 | 0.749 | α bajo, ajuste ruidoso |
| `Basic_INDEXLIST_3LOOP` | 0.013 | 0.575 | α bajo, ajuste ruidoso |

Los dos stencils son el hallazgo más interesante: fallan el tamizaje CPU
(§ correspondiente en `Estrategia_CPU_Fase2.md`, α 0.71–0.85) pero pasan
aquí — confirma que el rango dinámico de potencia distinto entre
dispositivos (sección 1145 de `main.tex`) cambia la viabilidad, no solo
la magnitud.

**Decisión de catálogo (2026-08-25): los 6 entran**, incluidos los 2 con
ajuste ruidoso — su α es bajo (0.010–0.013, muy por debajo del umbral
0.639) aunque el ajuste de Amdahl no lo describa con precisión (riesgo 6).
El catálogo GPU pasa de 7 a 13 kernels reales.

## 8.bis Estudio de suites adicionales para GPU (2026-08-26, sin lanzar)

**Motivo.** El catálogo actual (13 confirmados + los 43 candidatos del
tamizaje DRAM%/SM% de §"Impulso") viene solo de RAJAPerf-CUDA y Rodinia.
Vale mirar qué usó la propia literatura con la que este eje se compara,
en vez de improvisar una tercera suite a ciegas.

`Guerreiro2019` — el trabajo con el que más se compara este eje (§4, §8) —
valida su modelo con **35 kernels de Rodinia + Polybench + Parboil + SHOC
+ CUDA SDK**, no solo RAJAPerf. Dos de esas suites no se han tocado aquí:

| suite | qué llena | por qué | licencia / esfuerzo |
|---|---|---|---|
| **Parboil** (Stratton et al., UIUC) | mezcla real compute/memoria por diseño — 11 apps de imagenología, biomolecular, dinámica de fluidos y astronomía (CUTCP, MRI-Gridding, SpMV, histograma), no solo streaming | open source, CUDA+OpenCL, sin licencia paga | por verificar tiempo de compilación en `paccaA100` |
| **SHOC** (Danalis et al., ORNL) | diseñada explícitamente para cubrir el espectro compute↔memoria: micro + apps (FFT, MD, Scan, Sort, **SpMV, Stencil2D**, Triad, S3D) — SpMV/Stencil2D son justo la banda intermedia que RAJAPerf-CUDA cubre poco | open source (BSD), CUDA+OpenCL+MPI | por verificar tiempo de compilación |

Ambas, a diferencia de `Hebbar2022` (SPEC CPU2017, licencia paga), son
libres y reproducibles — mismo criterio de licencia que ya rige todo el
catálogo actual. **No se lanza nada todavía**: es candidatera para cuando
el tamizaje de los 43 candidatos (job 6595) cierre el catálogo actual y
se decida si hace falta diversidad que RAJAPerf-CUDA no tiene, no un
reemplazo de lo que ya está en cola.

## 9. Mapeo a los Objetivos Específicos

| Objetivo | Estado | Evidencia |
|---|---|---|
| 1. Caracterizar comportamiento y consumo bajo distintos estados de frecuencia (NVML) | **Cumplido** (198+210+90+54 corridas, 0 rechazos) | §1, §6; jobs 6462/6463/6471/6472/6476 |
| 2. Clasificador ML en vivo, baja latencia | **Granularidad reinterpretada a carga/kernel**, con precedente publicado triple | §3, §4; Anexo K |
| 3. Daemon de espacio de usuario con política DVFS | Sin cambios de diseño; pendiente de implementar sobre el modelo de §4 | V7 mide su presupuesto de latencia |
| 4. Evaluación por EDP contra gobernador nativo | EDP es métrica primaria (§5). En GPU el "nativo" es el autoboost del driver (`native_governor`/REF), incluido en todas las tablas | §5; `gpu_policy_headroom.py` |

## 10. Lista de pruebas — cómo verificar o refutar esta propuesta

Cada prueba tiene un criterio de paso explícito **y** qué se concluye si
falla. Ninguna afirmación de este documento debería sostenerse si su
prueba correspondiente falla.

| # | Qué verifica | Cómo | Pasa si | Si falla |
|---|---|---|---|---|
| **V1** | Márgenes interpolados de los 4 niveles nuevos (§6, riesgo 1) | Job 6471: contar rechazos I10 por nivel en `campaign_metadata.json` | Rechazos en G1–G4 comparables a los niveles medidos (F0/F1) | Remedir línea de reposo con sonda fina (≥30 s/nivel) y recalcular márgenes antes de usar el dataset |
| **V2** | La causa del margen angosto es la grilla, no la física (§5) | Correr `gpu_policy_headroom.py` sobre el dataset de 6471 con `--max-slowdown-pct 4` | Margen del modelo > 2 pts a ≤4% | La hipótesis se refuta: el presupuesto de 4% es inalcanzable aquí; reportar por EDP y declararlo como límite de plataforma |
| **V3** | Reproducibilidad del ahorro de `lavamd@F1` (§2) | Comparar el 25.11% de 6462 contra el mismo punto en 6471 | Diferencia dentro del CV entre repeticiones (≈1–3%) | El número de 6462 no es reproducible: reauditar antes de citarlo |
| **V4** | **Cierra el hueco lógico de §3**: ¿existe alternancia intra-corrida explotable? | Cruzar offline las marcas `PHASE`/`T0_MONOTONIC_NS` de `gpu_phasic` contra `gpu_power_mw` y `gpu_sm_clock_mhz` por ventana | Si las fases son distinguibles en potencia **y** su nivel óptimo difiere → hay alternancia explotable | Si no son distinguibles o el óptimo no difiere: queda confirmado que la granularidad por carga es la correcta, con evidencia positiva y no por ausencia de medición |
| **V5** | Anti-fuga de etiqueta (§4) | Test unitario: entrenar inyectando a propósito una feature del nivel candidato | El test **falla** ruidosamente (guardarraíl activo) | El guardarraíl no protege: cualquier resultado del modelo queda invalidado hasta arreglarlo |
| **V6** | Baseline honesto en LOKO (§4) | Verificar que la "mejor constante" se elige solo con folds de entrenamiento | La constante elegida puede diferir por fold | El baseline hace trampa y el margen reportado del modelo está inflado |
| **V7** | Latencia de inferencia (Objetivo 2) | Medir p50/p99 de una inferencia sobre el modelo entrenado | p99 « período de decisión del daemon | El modelo no sirve para el Objetivo 3 aunque acierte: elegir uno más liviano |
| **V8** | ¿El reloj de memoria es un mando disponible? (riesgo 2) | `nvidia-smi -i 0 --query-supported-clocks=mem` y probar `-lmc` bajo carga | Hay >1 reloj de memoria y `-lmc` se aplica y se sostiene | El segundo mando no existe en esta A100: cerrar esa línea y declararlo |
| **V9** | Sanidad de la energía de GPU | `gpu_energy_valid == 1` en todas las filas `gpu_telemetry` | 100% válidas (ya verificado en 6462/6463) | La métrica primaria pierde piso: no reportar ahorros hasta resolverlo |
| **V10** | Consistencia documento↔dato | Reejecutar `gpu_policy_headroom.py` y comparar contra las tablas de §2/§5 | Coinciden | Actualizar el documento: los números del papel deben salir siempre del script, nunca copiarse a mano |

**V3 — PASA (2026-08-25).** Recalculado con el mismo método en ambas
corridas (suma de `gpu_energy_delta_mj` con `gpu_energy_valid==1`, 3 reps
c/u, CPU=REF/GPU=F1 vs CPU=REF/GPU=REF): job 6462 → 28.15%, job 6471 →
26.71%. Diferencia **1.44 pts**, dentro del CV declarado (≈1-3%) — el
ahorro de `lavamd@F1` reproduce entre campañas independientes. Nota
aparte: ambos valores difieren del 25.11% citado en §2 por 1.6-3 pts —
el número original de §2 salió de un método de cómputo distinto (no
identificado); no invalida la reproducibilidad entre 6462 y 6471, que es
lo que V3 pregunta, pero §2 debería recalcularse con el mismo método
antes de seguir citándose (dejar como pendiente de V10).

---

## Referencias

Ya en `docs/libro/main.tex`: `\cite{Guerreiro2019}`, `\cite{Calore2017}`,
`\cite{Antici2024}`, `\cite{Williams2009}`.

Pendientes de agregar (ver `docs/libro/referencias_pendientes_dvfs_gpu.md`,
que distingue autores verificados de no verificados): `Fan2020`,
`Mei2016`, y el survey de técnicas de DVFS en GPU citado en §2 — **cuya
autoría debe confirmarse antes de citarlo**, igual que el resto de
entradas marcadas en ese archivo.
