# Evaluación de la propuesta *Phase-Based Table Lookup* y plan consolidado de política DVFS

**Fecha:** 18 de agosto de 2026
**Rama:** `advisorIntel`
**Documento evaluado:** `Hyperion_Propuesta_Phase_Based_Table_Lookup.md` (propuesta externa, 30 secciones)
**Alcance:** dictamen técnico sobre la propuesta + especificación consolidada de la política de control DVFS del daemon + plan por fases con sustentación metodológica.

**Documentos que este texto NO sustituye y que tienen prioridad sobre él:**
`docs/general/plan_trabajo_grado.md` (plan aprobado, verdad no negociable),
`docs/retoma/pacca/Diseno_Politica_DVFS_CPU_GPU.md` (diseño de política v2, ARC-65 a ARC-80),
`docs/retoma/Guia_Maestra_Fase1_DVFS.md`, `AGENTS.md`.
Si algo de aquí contradice a esos documentos, el error es de este documento.

---

## 0. Aclaración de numeración de fases (importante, se presta a confusión)

La consulta original habla de "fase 4 (daemon)". En el **plan de trabajo de grado aprobado** la numeración es otra y hay que respetarla en el libro:

| Fase del plan aprobado | Contenido | Estado real hoy |
|---|---|---|
| Fase 1 (§5.1) | Recolección de telemetría y caracterización de cargas | Avanzada; validada a `REF`. **Falta la matriz multi-frecuencia** |
| Fase 2 (§5.2) | Entrenamiento del clasificador ligero | No iniciada (no hay código de ML en el repo) |
| **Fase 3 (§5.3)** | **Implementación del agente de control DVFS (daemon) ← es aquí donde vive esta propuesta** | No iniciada |
| Fase 4 (§5.4) | Validación experimental, EDP y análisis estadístico | No iniciada |

La propuesta evaluada es, por tanto, **una propuesta de política para la Fase 3**, cuya calibración es un producto de la Fase 1 y cuya evaluación pertenece a la Fase 4. Para no arrastrar la ambigüedad, el plan de la sección 8 de este documento usa una numeración propia e inequívoca (`D0`–`D5`, de *daemon*) y la mapea explícitamente contra las fases del plan aprobado.

---

## 1. Veredicto ejecutivo

**Calificación global: 8 / 10.**

La propuesta es **técnicamente correcta, metodológicamente defendible, de nivel de pregrado y compatible con lo ya construido**. No hay razón para descartarla ni para sustituirla por otro paradigma de control. De hecho, describe con precisión algo que el repositorio **ya implementa parcialmente sin haberle puesto nombre**: `telemetry/include/telemetry/gpu_clock_controller.hpp` ya es, literalmente, una tabla de dos entradas (`compute_bound_clock_mhz`, `memory_bound_clock_mhz`) más una máquina de estados de residencia mínima (`min_dwell_ns`). La propuesta formaliza, justifica bibliográficamente y extiende a CPU un diseño que el proyecto ya había tomado de facto.

Los dos puntos que impiden un 9–10 no son de concepto sino de **omisión**, y ambos son subsanables:

1. **No aborda la bistabilidad introducida por un `i_ridge` dependiente de la frecuencia** — un problema que este proyecto ya descubrió y documentó por su cuenta (ARC-78 en CPU, ARC-79/ARC-80 en GPU) y que la histéresis propuesta **no** resuelve, porque no es ruido: es realimentación estructural.
2. **Su única dependencia dura (la campaña de calibración F0–F4) coincide exactamente con el bloqueador número uno del proyecto** (permisos `P1`/`P4`, sin otorgar hasta la última documentación disponible), y la propuesta no ofrece ninguna vía degradada si ese permiso no llega.

**Decisión recomendada: adoptar la propuesta como base, con cinco correcciones obligatorias**, bajo el nombre **Política de dos estados con banda de indecisión, calibrada por EDP normalizado** (*two-state EDP-calibrated phase policy*). La sección 6 la especifica completa y la sección 8 la convierte en plan ejecutable.

### 1.1 Calificación desagregada

| Criterio | Nota | Sustentación |
|---|---:|---|
| Concordancia con el plan aprobado | **10** | Clasificador binario (§5.2), política **discreta** y **proactiva** (§5.3), EDP como métrica (§5.4), actuación vía `cpupower`/`nvidia-smi` (§5.3). No añade ni una clase ni un actuador fuera del plan. |
| Reutilización de lo ya construido | **10** | No exige rehacer telemetría, catálogo, calibración Roofline, `freqctl.py`, `gpu_freqctl.py` ni el harness C++. |
| Nivel de pregrado | **9** | Sin RL, sin power capping, sin predicción de duración de fase, sin modelado energía/tiempo. Complejidad de implementación: una máquina de estados y un `dict`. |
| Rigor metodológico de la calibración | **8** | Normalización por workload, restricción de *slowdown*, *leave-one-workload-out*: todo correcto. Falla en la estadística concreta (§4.4). |
| Diseño experimental de validación | **9** | Los cinco tratamientos `B0`–`B4` son los correctos; incluir **Best Static** (`B2`) y **Oracle** (`B3`) es lo que separa un trabajo evaluable de una demostración. |
| Tratamiento del riesgo de oscilación | **6** | Cubre el *thrashing* por ruido; **ignora la bistabilidad por ridge dependiente de frecuencia** (§4.1). |
| Viabilidad de ejecución en el cronograma real | **5** | Depende íntegramente de un permiso administrativo pendiente; sin plan B declarado (§4.2). |
| Integración CPU↔GPU | **6** | Trata los dominios como independientes y omite la regla de acoplamiento ya decidida en el diseño v2 §4 (§4.3). |
| Honestidad epistémica | **10** | La §27 ("qué NO se afirma") es ejemplar y debe conservarse literalmente en el libro. |

---

## 2. Contra qué se evaluó (evidencia disponible en el repositorio)

El dictamen no es de opinión. Se contrastó contra el estado real medido:

**CPU (`paccaA100`, Xeon Gold 5317, 6 cores delegados):**
- Campañas clase B y clase C a `REF`: **21/21** y **18/18** corridas aceptadas, **1 107 573 ventanas**, validez energética RAPL ≈ 100 %.
- Estabilidad de la etiqueta entre tamaños de problema: cinco de seis kernels NPB varían ≤ 1,1 puntos porcentuales entre clase B y C.
- Comportamiento por kernel: `npb_mg` 99,9 % `memory_bound`; `npb_bt` 85,6 / 85,4 % `compute_bound`; `npb_lu` 88,4 / 89,0 % `compute_bound`; `npb_cg` 92,7 / 93,5 % `memory_bound`; **`npb_sp` 58,2 / 59,3 % `memory_bound`** (mixto); **`npb_ft` 79,7 % → 66,2 % `compute_bound`** al crecer el tamaño.
- Potencia media a `REF` entre cargas: ≈ 114–141 W.

**GPU (A100-PCIe-40GB):**
- BabelStream Triad 1,399 TB/s; pico FP32 10 178,2 GFLOP/s y FP64 4 698,6 GFLOP/s (microbenchmark propio, sin Tensor Cores, ARC-76); ridge **7,28** FLOP/byte (FP32) y **3,36** FLOP/byte (FP64).
- **ARC-77: la GPU permanece a 765 MHz de 1410 posibles incluso bajo carga real**, con 62 W de 250 W y 26 °C — sin límite térmico ni energético que lo explique.

**Bloqueadores vigentes:** permisos `P1` (escritura `cpufreq`), `P4` (reloj de SM), contadores `uncore`/iMC. Sesgo conocido de `bytes_moved_window`: subestima el tráfico real ≈ 30–34 %.

---

## 3. Lo que la propuesta acierta, y por qué debe conservarse

### 3.1 Separar "reconocer el régimen" de "decidir la acción" es la decisión arquitectónica correcta

El plan aprobado (§5.2) fija que la salida del modelo es una etiqueta binaria de régimen, no una frecuencia. Si el clasificador escogiera frecuencia directamente, habría que reformular el problema como multiclase de 5 salidas, rehacer el etiquetado de todo el dataset y justificar ante el jurado una desviación del plan aprobado. La propuesta evita eso por construcción: el modelo responde *qué régimen*, la tabla responde *qué hacer*, la máquina de estados responde *si conviene actuar ahora*. Tres responsabilidades, tres piezas auditables por separado.

### 3.2 Dos estados efectivos por dominio es la elección correcta, y por una razón que la propuesta no explota del todo

La propuesta argumenta que con cinco acciones surge una pregunta que el clasificador binario no puede responder (`memory_bound → ¿F1, F2, F3 o F4?`). Correcto. Pero hay un argumento más fuerte y propio: **con 6 kernels CPU de dataset, calibrar 5 niveles × 2 clases con `n=3` repeticiones deja como máximo 3 observaciones por celda**. Cualquier selección entre 5 niveles sería estadísticamente indefendible en el libro. Con 2 estados efectivos, la campaña F0–F4 se usa para *caracterizar la curva completa* (evidencia rica para el capítulo de resultados) pero la *decisión* recae sobre una comparación de dos alternativas, que sí se sostiene con el `n` disponible.

### 3.3 `Best Static` (`B2`) como baseline obligatorio

Es el aporte más valioso de la propuesta. La pregunta científica real del trabajo no es "¿mejora el agente frente al governor?" sino **"¿hace falta que sea dinámico?"**. Sin `B2` el trabajo no puede responderla, y un jurado técnico la va a hacer. Debe quedar en el diseño experimental sin negociación.

### 3.4 Anclaje bibliográfico correcto y honesto

Carpentieri et al. (IPDPS 2025) es efectivamente el antecedente directo, y la propuesta es explícita en que Hyperion **no replica** su algoritmo (profiling + DAG) sino que adopta el principio con una detección distinta (PMU/NVML + ML). Esa honestidad es exactamente lo que protege el trabajo de una acusación de sobreventa. Igual de valiosa es la cita de Veličká et al. sobre latencias de conmutación dependientes del par origen/destino, que sustenta por qué la residencia mínima se deriva de una **matriz** `L[Fi][Fj]` y no de un escalar.

### 3.5 La sección "qué NO se afirma" (§27)

Debe copiarse casi literalmente al capítulo de limitaciones del libro. Delimita la hipótesis a algo evaluable con los datos que el proyecto va a tener.

---

## 4. Los cinco defectos, en orden de severidad

### 4.1 [CRÍTICO] Bistabilidad por `i_ridge` dependiente de la frecuencia — la histéresis propuesta no la corrige

**El problema.** En Hyperion la etiqueta no es una propiedad fija de la carga. Es el resultado de una comparación:

```text
phase_label_train(w, f) = compute_bound  si  I_op(w) > i_ridge(f)
                          memory_bound   en caso contrario
```

Y este proyecto ya estableció, por su cuenta y contra su propia suposición inicial, que **`i_ridge` depende de la frecuencia**: `i_ridge = P_pico / BW_pico`, donde `P_pico` escala con el reloj y `BW_pico` casi no lo hace (ARC-78 en CPU, ARC-79/ARC-80 en GPU). Bajar la frecuencia **baja el ridge**, y por tanto empuja cargas de intensidad fija hacia el lado `compute_bound` de la frontera.

**La consecuencia que la propuesta no ve.** Se cierra un lazo de realimentación positiva:

```text
ventana clasificada memory_bound
    -> política aplica LOW
    -> baja la frecuencia
    -> baja i_ridge
    -> la MISMA carga, sin cambiar en nada, cruza a compute_bound
    -> política aplica HIGH
    -> sube la frecuencia
    -> sube i_ridge
    -> vuelve a memory_bound ...
```

Esto **no es *thrashing* por ruido**. Es un ciclo límite determinista. Ni el filtro de `N` predicciones consecutivas ni la residencia mínima lo eliminan: solo alargan su periodo. Con `N=3` y residencia de 50 ms, en lugar de oscilar cada ventana, oscila cada ~150 ms — indefinidamente, y consumiendo una transición cada vez.

Los kernels en riesgo no son hipotéticos y están identificados en los datos del propio repositorio: **`rodinia_lud`** (7,6–7,8 FLOP/byte, a menos del 7 % del ridge FP32 medido de 7,28) y **`npb_sp`** (58–59 % de ventanas `memory_bound`, es decir, una carga que ya reparte sus ventanas a ambos lados de la frontera a frecuencia fija).

**Corrección obligatoria — anclar la etiqueta a un ridge de referencia fijo.** La etiqueta que consume el daemon debe definirse contra `i_ridge(F_ref)`, un valor **constante durante toda la ejecución**, no contra el ridge del estado actual:

```text
etiqueta_de_control(w) = I_op(w) vs i_ridge(F_ref)     <- constante, rompe el lazo
etiqueta_física(w, f)  = I_op(w) vs i_ridge(f)         <- se conserva para el análisis del libro
```

Justificación metodológica, no solo práctica: la etiqueta de control debe ser una propiedad **de la carga**, no del estado del actuador. Si depende del actuador, el sistema clasifica su propia decisión anterior en vez de clasificar la aplicación — que es precisamente lo que el objetivo específico 2 pide evitar. Se conserva la etiqueta física dependiente de frecuencia porque es un hallazgo real de la caracterización (Fase 1) y merece su sección en el capítulo de resultados; simplemente **no es la señal que cierra el lazo de control**.

**Consecuencia operativa sobre el entrenamiento:** el dataset de la Fase 2 debe etiquetarse con `i_ridge(F_ref)` para todas las combinaciones `(kernel, F_n)`, y `F_ref` debe quedar congelado y declarado en `roofline_calibration.json` y en `policy.json`. Esto **no invalida** el trabajo de ARC-78/79/80 — lo usa: sin haber calibrado el ridge por nivel, no habría manera de demostrar que el problema existe ni de justificar el anclaje.

### 4.2 [CRÍTICO] La única dependencia dura de la propuesta es el bloqueador número uno del proyecto, y no hay plan B

La tabla se calibra con una campaña `kernel × F0–F4 × repeticiones` con energía. Esa campaña **no existe en ninguna rama** y está bloqueada por `P1` (CPU) y `P4` (GPU). La propuesta la asume disponible en su §6 y no dice qué hacer si no lo está.

**Lo que sí se puede afirmar, y es una buena noticia:** el bloqueador es de *permiso*, no de *tiempo de máquina*. Estimación de orden de magnitud derivada de los conteos de ventanas ya medidos — la campaña clase C acumuló 891 587 ventanas de 1 ms en 18 corridas, ≈ 50 s de tiempo de ventana medido por corrida:

```text
CPU:  7 kernels × 5 niveles × 3 repeticiones × 2 corridas (baseline + telemetría) = 210 ejecuciones
      + calibración Roofline (STREAM + ERT) repetida por nivel = 5 calibraciones
```

A ≈ 50 s de ventana útil por corrida, más warmup, preflight y sobrecosto de orquestación (factor 2–4×), el orden de magnitud es de **horas de nodo, no de semanas**. *Este número es derivado de conteos de ventanas, no de reloj de pared medido; debe confirmarse con un piloto de un solo nivel antes de comprometer la campaña completa (paso `D1.0` del plan).*

**Corrección obligatoria — declarar tres modos de degradación antes de empezar**, para que ninguna decisión de última hora quede sin sustento (detalle completo en la sección 9):

| Modo | Condición de disparo | Qué se entrega |
|---|---|---|
| **A — Completo** | `P1` y `P4` otorgados | Política CPU + GPU, calibración F0–F4, `B0`–`B4` completos |
| **B — Solo CPU** | `P1` sí, `P4` no | Política CPU real; GPU documentada como bloqueada con la evidencia de ARC-77 |
| **C — Traza** | Ningún permiso | Daemon completo evaluado por **reproducción sobre trazas** (`trace-driven replay`) de las campañas ya recolectadas: se mide overhead real de inferencia, tasa de conmutación, latencia de detección y estabilidad de la máquina de estados; el ahorro de EDP se reporta como **proyección**, nunca como medición |

El modo C **no es un fracaso**: entrega el objetivo específico 3 completo (el daemon existe, corre y se mide su overhead) y deja el objetivo 4 explícitamente como medición pendiente por causa externa documentada. Es infinitamente preferible a llegar a la sustentación sin daemon.

### 4.3 [ALTO] Omite la regla de acoplamiento CPU↔GPU ya decidida por el proyecto

La propuesta modela CPU y GPU como dos dominios independientes que consultan la misma estructura de tabla. Pero `Diseno_Politica_DVFS_CPU_GPU.md` §4 ya decidió, con argumento físico, algo distinto: **existe una señal unidireccional "GPU ocupada" que hace que el loop de CPU fuerce su frecuencia al mínimo sin siquiera consultar al modelo de CPU**.

La razón es concreta y está verificada en hardware: durante `cudaDeviceSynchronize()` con espera por *spin* (comportamiento por defecto de CUDA), Perf observa IPC alto y casi cero *cache misses*, y el clasificador de CPU diría `compute_bound` — **subiendo la frecuencia justo cuando el CPU no hace ningún trabajo útil**. El shim `LD_PRELOAD` de bloqueo real ya está implementado y verificado (ARC-72: 99,8 % → 0,0 % de CPU), pero el shim solo aplica a las cargas del catálogo, no a una aplicación arbitraria que el daemon vaya a controlar en producción.

Una política que ignore esto **empeora activamente el EDP en el caso heterogéneo**, que es el caso que da título al trabajo de grado. La sección 6.4 lo incorpora como regla de precedencia explícita.

### 4.4 [MEDIO] La estadística propuesta no es ejecutable con el `n` real

La propuesta exige `slowdown_p95(f) <= delta`. Con `n = 3` repeticiones por celda (el valor ya justificado experimentalmente en este proyecto, con CV < 2 %), **un percentil 95 no es estimable**: con tres observaciones, el P95 empírico es indistinguible del máximo.

**Corrección:** sustituir por criterios que sí se sostienen con `n=3` y declarar el `n` junto a cada cifra:

```text
slowdown_max_observado(f) <= delta          (criterio primario, conservador y honesto)
IC bootstrap 95% de median(EDP_norm)        (percentil, B = 10 000 remuestreos)
fracción de workloads mejorados >= umbral   (recuento, no estimación de cola)
```

El bootstrap sobre la mediana entre *workloads* (6–7 unidades) es defendible; el P95 sobre 3 repeticiones no lo es. Además, la comparación final entre tratamientos debe usar una prueba no paramétrica pareada por workload (Wilcoxon de rangos con signo) en lugar de asumir normalidad — el plan aprobado (§5.4) pide "técnicas estadísticas apropiadas según la distribución y homogeneidad de los datos", lo que obliga a verificar la distribución antes de elegir, no después.

### 4.5 [BAJO] Dos observaciones menores

**(a) El espacio DVFS de GPU está hoy colapsado a un punto.** ARC-77 midió que la GPU permanece a 765 MHz de 1410 incluso bajo carga real con margen térmico y energético amplio. Sin `P4`, `HIGH_GPU` y `LOW_GPU` son el mismo valor y la política de GPU es formalmente inaplicable. Debe declararse así en el libro, con la evidencia, en lugar de presentarse como una política de GPU que "está lista".

**(b) El nombre.** *"Phase-Based Table Lookup"* es un término acuñado por la propuesta, y ella misma lo reconoce (§3). Presentarlo como si fuera terminología establecida es un riesgo innecesario ante un jurado. Solución: en el libro se describe como *"política de dos estados por régimen, inspirada en el enfoque phase-aware de Carpentieri et al."*, con la aclaración terminológica de la §3 de la propuesta conservada como nota al pie.

---

## 5. ¿Hay un método mejor? Comparación con las alternativas reales

Se evaluaron cinco alternativas frente a la propuesta corregida. Ninguna la supera **dentro de las restricciones de este trabajo**.

| Alternativa | Por qué no reemplaza a la propuesta |
|---|---|
| **Reglas PMU con umbrales** (Hebbar y Milenković) | Más ligera, pero exige umbrales manuales específicos del hardware y **duplica el trabajo del clasificador**, que el plan aprobado ya obliga a construir (objetivo específico 2). Adoptarla haría al modelo de ML redundante — se perdería el objetivo 2 completo. Sí sirve como *baseline adicional* si sobra tiempo. |
| **Clasificador que predice frecuencia directamente** (multiclase de 5 salidas) | Desvía del plan aprobado (§5.2 fija salida binaria), multiplica por 5 el espacio de etiquetas sobre el mismo dataset y exige rehacer el etiquetado. Riesgo alto, beneficio no demostrado. |
| **Modelo predictivo de energía/tiempo por frecuencia** | Es más potente y permitiría decisión continua, pero convierte a Hyperion en un proyecto de *performance/energy modeling*, con su propia validación, su propio error de predicción y su propio capítulo. Fuera del alcance declarado (§6.2 del plan: se excluyen "esquemas complejos de optimización en línea"). |
| **RL / bandit contextual** | Explícitamente excluido por el plan aprobado (§6.2, "no contempla ... Aprendizaje por Refuerzo Profundo ni esquemas complejos de optimización en línea"). Además introduce exploración en línea sobre un clúster **compartido**, lo que choca con los no-negociables de `AGENTS.md`. |
| **DVFS por kernel / online probing** | La propia bibliografía citada lo desaconseja: Carpentieri et al. miden 0,30–0,60 ms por transición en GPU y muestran que conmutar en cada invocación puede anular el beneficio. El probing en línea además contamina la aplicación medida. |

**Conclusión:** la propuesta ocupa el punto correcto del espacio de diseño. Lo que le falta no es un paradigma distinto, sino las cinco correcciones de la sección 4. Por eso este documento **consolida** en lugar de reemplazar.

**Una única mejora de diseño se incorpora más allá de la propuesta** (sección 6.2): sustituir la histéresis de "`N` predicciones idénticas consecutivas" por una **banda de indecisión sobre la confianza del clasificador**, con una tercera acción explícita `NO_CHANGE`. Justificación empírica propia: `npb_sp` reparte sus ventanas 58 % / 42 % entre clases a frecuencia fija. Contra una carga así, una decisión binaria dura conmuta permanentemente por diseño; una banda de indecisión la deja quieta en el estado actual, que es la respuesta correcta para una carga genuinamente mixta. El costo es nulo: los árboles y bosques aleatorios que el plan ya obliga a usar exponen `predict_proba` sin cómputo adicional relevante.

---

## 6. Especificación de la política consolidada v1

### 6.1 Definición de la etiqueta de control

```text
i_ridge_ref(d)  := i_ridge del dominio d calibrado a F_ref, CONGELADO en policy.json
etiqueta(w)     := compute_bound  si  I_op(w) > i_ridge_ref(d)
                   memory_bound   en caso contrario
```

`F_ref` se declara explícitamente (`F0` si `P1` está otorgado; la frecuencia nativa del nodo en caso contrario) y se registra en `roofline_calibration.json`, `policy.json` y en la metadata de cada corrida. **No se recalcula durante la ejecución del daemon.**

### 6.2 Tres acciones, no dos

```text
p := P(compute_bound | features)      del clasificador (predict_proba)
tau := margen de decisión, calibrado (valor inicial 0.15)

p >= 0.5 + tau   ->  candidato HIGH
p <= 0.5 - tau   ->  candidato LOW
en otro caso     ->  NO_CHANGE  (banda de indecisión: la carga es mixta o el modelo no está seguro)
```

Sobre el candidato se aplica además un filtro `N`-de-`M` (inicial `N=3`, `M=4`) para tolerar ventanas atípicas. La banda y el filtro atacan problemas distintos y son complementarios: la banda filtra **incertidumbre del modelo**, el filtro `N`-de-`M` filtra **ruido temporal**.

`tau` se calibra sobre el conjunto de validación de la Fase 2, no se elige a ojo: se barre `tau ∈ {0.05, 0.10, 0.15, 0.20, 0.25}` y se selecciona el valor que minimiza la tasa de conmutación sin degradar el EDP en la reproducción sobre trazas. Ese barrido es un experimento offline puro, no cuesta tiempo de nodo con permiso.

### 6.3 Máquina de estados completa (orden de evaluación no negociable)

```text
1.  ¿telemetría válida? (quality_status ok, sin denominador cero)   -> NO -> NO_CHANGE
2.  ¿el dominio está bajo una regla de precedencia? (sección 6.4)   -> SÍ -> aplicar esa regla, salir
3.  inferencia -> p -> candidato { HIGH | LOW | NO_CHANGE }         -> NO_CHANGE -> salir
4.  ¿candidato estable N-de-M?                                      -> NO -> NO_CHANGE
5.  target := policy[dominio][candidato]
6.  ¿target == estado actual?                                       -> SÍ -> NO_CHANGE
7.  ¿residencia mínima cumplida? (min_residence[dominio])           -> NO -> NO_CHANGE
8.  ¿hardware saludable? (térmica, throttling, dominio delegado)    -> NO -> FAILSAFE
9.  aplicar frecuencia
10. RELECTURA obligatoria del estado aplicado                       -> incompatible -> FAILSAFE
11. arrancar temporizador de residencia; registrar la decisión completa en el log de auditoría
```

Los pasos 9–11 no son opcionales: son los no-negociables de `AGENTS.md` ("toda escritura de frecuencia se verifica por relectura"; "la restauración es obligatoria e idempotente, registrada en `atexit`, `SIGINT` y `SIGTERM`"). `FAILSAFE` significa **restaurar el estado original del dominio y dejar de actuar**, no "reintentar".

### 6.4 Reglas de precedencia (paso 2) — lo que la propuesta omitía

Se evalúan antes de consultar el modelo, en este orden:

| Prioridad | Condición | Acción | Sustentación |
|---|---|---|---|
| P-1 | `hardware_safe == false` | `FAILSAFE` + restaurar | `AGENTS.md`, no-negociable |
| P-2 | `gpu_busy == true` (loop GPU señala trabajo activo) | CPU → piso mínimo, **sin consultar `modelo_cpu`** | `Diseno_Politica_DVFS_CPU_GPU.md` §4.1: corrige el falso `compute_bound` del spin-wait |
| P-3 | `telemetry_valid == false` | `NO_CHANGE` | `AGENTS.md`: denominador cero → `NaN` + `quality_status`, nunca un `0` silencioso |
| P-4 | — | Política normal (6.2, 6.3) | — |

La señal `gpu_busy` es **unidireccional**: el loop de GPU la emite, el loop de CPU la consume. El loop de GPU nunca consulta nada del loop de CPU. Esto preserva la independencia de los dos ciclos de decisión ya establecida en el diseño v2 §4 y evita cualquier negociación entre dominios (que sería otro proyecto).

### 6.5 Esquema de `policy.json`

```json
{
  "schema_version": 2,
  "node_id": "paccaA100",
  "policy_type": "two_state_edp_calibrated_phase_policy",
  "calibrated_at": "YYYY-MM-DD",
  "reference": {
    "f_ref_level_id": "F0",
    "i_ridge_ref_cpu_flops_per_byte": null,
    "i_ridge_ref_gpu_fp32_flops_per_byte": 7.28,
    "i_ridge_ref_gpu_fp64_flops_per_byte": 3.36,
    "roofline_calibration_ref": "roofline_calibration.json#sha256"
  },
  "decision": {
    "tau_margin": 0.15,
    "stability_n_of_m": [3, 4],
    "third_action": "NO_CHANGE"
  },
  "cpu": {
    "compute_bound": { "logical_state": "HIGH", "frequency_level": null },
    "memory_bound":  { "logical_state": "LOW",  "frequency_level": null },
    "min_residence_ms": null,
    "selection_metric": "median_normalized_edp",
    "baseline_level": "F0",
    "slowdown_limit": 0.05,
    "slowdown_criterion": "max_observed",
    "n_repetitions": 3
  },
  "gpu": {
    "status": "blocked_by_P4",
    "evidence": "ARC-77: 765/1410 MHz bajo carga real, 62/250 W, 26 C"
  },
  "provenance": {
    "calibration_workloads": [],
    "evaluation_workloads": [],
    "leave_one_workload_out_report": null
  }
}
```

Los `null` son deliberados: **son los valores que solo la campaña puede llenar**. Un `policy.json` con `null` es un artefacto honesto; uno con números inventados es fraude metodológico. La separación explícita `calibration_workloads` / `evaluation_workloads` hace que la afirmación de generalización sea verificable desde el propio artefacto.

### 6.6 Pseudocódigo de referencia

```python
def control_epoch(domain, features, shared_state):
    # 1. telemetría
    if not telemetry_valid(features):
        return log(domain, NO_CHANGE, reason="telemetry_invalid")

    # 2. precedencias
    if not hardware_safe(domain):
        restore_original_state(domain)
        return log(domain, FAILSAFE, reason="hardware_unsafe")
    if domain == "cpu" and shared_state.gpu_busy:
        return apply_floor(domain, reason="gpu_busy_spin_correction")

    # 3. inferencia con banda de indecisión
    p = classifier[domain].predict_proba(features)
    candidate = ( HIGH       if p >= 0.5 + policy.tau
             else LOW        if p <= 0.5 - policy.tau
             else NO_CHANGE )
    if candidate is NO_CHANGE:
        return log(domain, NO_CHANGE, reason="undecided_band", p=p)

    # 4. estabilidad temporal
    history[domain].push(candidate)
    if not stable_n_of_m(history[domain], policy.n, policy.m):
        return log(domain, NO_CHANGE, reason="unstable")

    # 5-7. tabla, estado actual, residencia
    target = policy.lookup(domain, candidate)
    if target == current_frequency(domain):
        return log(domain, NO_CHANGE, reason="already_there")
    if not residence_satisfied(domain):
        return log(domain, NO_CHANGE, reason="min_residence")

    # 8-11. aplicar y VERIFICAR POR RELECTURA
    apply_frequency(domain, target)
    observed = read_frequency(domain)
    if not compatible(target, observed):
        restore_original_state(domain)
        return log(domain, FAILSAFE, reason="readback_mismatch",
                   target=target, observed=observed)

    start_residence_timer(domain)
    return log(domain, SWITCH_OK, target=target, observed=observed, p=p)
```

Cada retorno pasa por `log()` con su `reason`. Ese registro **es** la evidencia de interpretabilidad que se defiende en la sustentación: toda decisión del agente debe poder reconstruirse a posteriori desde el log, sin ejecutar nada.

---

## 7. Calibración de la tabla: procedimiento estadístico ejecutable

**Métrica.** `EDP = E × T`, normalizada por workload contra el nivel de referencia:

```text
EDP_norm(w, d, f) = EDP(w, d, f) / EDP(w, d, F_ref)
```

La normalización es indispensable: sin ella, un kernel largo domina la estadística por su EDP absoluto, no por su comportamiento.

**Procedimiento (todo offline, sobre datos ya recolectados):**

1. Calcular EDP por repetición, nunca sobre promedios agregados.
2. Normalizar por workload contra `F_ref`.
3. Estimar `median_w(EDP_norm)` por `(dominio, clase, nivel)` con IC 95 % bootstrap (`B = 10 000`) entre workloads.
4. Aplicar la restricción de rendimiento: `slowdown_max_observado(f) <= delta`, con `delta ∈ {3 %, 5 %, 10 %}` estudiado en el piloto y **congelado antes** de la campaña final.
5. Reportar además la fracción de workloads mejorados (recuento sobre 6–7 unidades, no estimación de cola).
6. Seleccionar el estado **conservador**: si dos niveles son estadísticamente indistinguibles (IC solapados), escoger el de menor slowdown.
7. **Congelar** `policy.json` y calcular su checksum.
8. Evaluar sobre los workloads reservados, que no participaron en ningún paso anterior.

**Formulación:**

```text
f*(d, c) = argmin_f  median_w( EDP_norm(w, d, f) )
           sujeto a  slowdown_max(f) <= delta
                     fraction_improved(f) >= threshold
```

**Resultados que deben aceptarse si aparecen** (y que la propuesta no contempla explícitamente):

- **`HIGH == LOW`** para un dominio: los datos dicen que una única frecuencia domina. Es un resultado científico válido, no un fallo de la política. En ese caso `B2` (Best Static) gana y hay que decirlo.
- **`Best Static` supera al Oracle**: indica fases demasiado cortas, costo de conmutación alto o granularidad binaria insuficiente. También es un resultado publicable, y la §24 de la propuesta ya prevé cómo interpretarlo.

**Generalización.** *Leave-one-workload-out*: calibrar sin `npb_cg`, seleccionar `LOW`, evaluar sobre `npb_cg`; repetir para cada workload. Con 6–7 kernels son 6–7 re-análisis del **mismo** conjunto de datos — costo de nodo cero. La afirmación defendible resultante es exactamente la que enuncia la propuesta:

> La tabla fue calibrada sobre un subconjunto de cargas de `paccaA100` y evaluada sobre cargas que no participaron en la selección de sus estados.

Nunca "la política es portable".

---

## 8. Plan por fases

Numeración propia `D0`–`D5`, mapeada contra el plan aprobado. Cada fase declara **entregable**, **criterio de salida verificable** y **dependencia dura**.

### D0 — Congelar el contrato de la política *(sin dependencias externas — se puede empezar hoy)*

| | |
|---|---|
| **Entregable** | `policy.json` con esquema completo y valores `null`; `docs/policy/` con este documento; barrido offline de `tau` sobre el conjunto de validación |
| **Salida** | El esquema valida contra un JSON Schema versionado; `F_ref` declarado; los tres modos de degradación (sección 9) escritos y aprobados por el director |
| **Depende de** | Nada |
| **Fase del plan** | Preparatoria de Fase 3 |
| **Por qué primero** | Fija qué números tiene que producir la campaña. Diseñar la campaña sin saber qué va a consumir sus resultados es la causa habitual de campañas que hay que repetir. |

### D1 — Campaña de calibración F0–F4 *(bloqueada por `P1` / `P4`)*

| | |
|---|---|
| **D1.0** | **Piloto de un nivel, un kernel**: medir reloj de pared real, confirmar la estimación de horas de nodo de §4.2 y verificar la restauración de frecuencia (prueba de caos real, operación humana, **no delegable a un agente de IA**) |
| **D1.1** | Recalibración Roofline por nivel (STREAM + ERT a cada `F_n`) — mecanismo ya implementado (ARC-78/80), solo falta el permiso |
| **D1.2** | Matriz completa `7 kernels × 5 niveles × 3 repeticiones`, aleatorizada, con corridas rechazadas conservadas y su `rejection_factor_id` |
| **D1.3** | Etiquetado dual: `etiqueta_control` contra `i_ridge(F_ref)` y `etiqueta_física` contra `i_ridge(F_n)` |
| **Salida** | Tasa de aceptación ≥ 95 %; validez energética RAPL ≥ 99 %; CV de calibración < 5 % en los cinco niveles; `windows.csv` no vacío en **todas** las repeticiones (regresión conocida del bug de numeración de repeticiones) |
| **Fase del plan** | Cierre de Fase 1 |
| **Riesgo** | **Máximo**. Es la variable experimental central del trabajo y no existe en ninguna rama. |

**Pendientes baratos que deben resolverse antes de D1.2, no después:** recalibrar el `warmup_seconds` de `dgemm_n2048` (una campaña reportó **0 de 773 ventanas** en estado `ok` porque el kernel dura menos que su propio warmup), cerrar el caso de entrada truncada de `rodinia_heartwall`, y decidir el tratamiento de los kernels GPU borderline (`rodinia_lud`).

### D2 — Selección de la tabla *(offline puro, ninguna dependencia de hardware)*

| | |
|---|---|
| **Entregable** | `policy.json` poblado y congelado, con checksum; reporte de calibración con medianas, IC bootstrap, slowdown máximo y fracción mejorada por `(dominio, clase, nivel)`; reporte *leave-one-workload-out* |
| **Salida** | `HIGH` y `LOW` seleccionados con criterio conservador y trazables al reporte; `calibration_workloads` y `evaluation_workloads` disjuntos y declarados en el artefacto |
| **Depende de** | D1 |
| **Fase del plan** | Bisagra Fase 1 → Fase 3 |

### D3 — Daemon *(el objetivo específico 3)*

| | |
|---|---|
| **D3.1** | Máquina de estados de §6.3 con clasificadores intercambiables: `Stub`, `Trace`, `Oracle`, `ML`. **Se construye y se prueba entera antes de que exista el modelo entrenado** |
| **D3.2** | Reglas de precedencia de §6.4, incluida la señal unidireccional `gpu_busy` |
| **D3.3** | Actuación sobre `freqctl.py` / `gpu_freqctl.py` con relectura obligatoria, `FAILSAFE` y restauración idempotente en `atexit` / `SIGINT` / `SIGTERM` |
| **D3.4** | Log de auditoría: una línea por época con `reason`, `p`, `target`, `observed`, `residence_remaining` |
| **D3.5** | Medición de overhead: latencia de inferencia (p50/p95/máx), costo de la época completa, e impacto sobre el runtime frente a una corrida sin daemon |
| **Salida** | Prueba de caos real superada (interrumpir a mitad de corrida y confirmar por lectura de `sysfs` que todo volvió a su estado previo); cero escrituras fuera de `delegated_cpus` verificadas por auditoría del log; el daemon corre completo con `Oracle` sin necesitar el modelo |
| **Depende de** | D0 (contrato). **No depende de D1 ni de D2**: con `Stub`/`Trace`/`Oracle` y una tabla de valores provisionales, el daemon se construye y se valida en paralelo a la espera del permiso |
| **Fase del plan** | **Fase 3 (§5.3)** |

**Esta independencia es la decisión de planificación más importante de todo el documento.** Es lo que impide que el bloqueo de permisos deje al proyecto sin el objetivo específico 3.

### D4 — Validación experimental *(el objetivo específico 4)*

Cinco tratamientos, ejecutados en orden aleatorizado sobre los mismos workloads:

| ID | Tratamiento | Pregunta que responde |
|---|---|---|
| `B0` | Governor nativo | ¿Mejora el agente frente al comportamiento original del nodo? |
| `B1` | `F0` fija / performance | ¿Cuál es la referencia de máximo rendimiento? |
| `B2` | **Best Static** (la mejor frecuencia fija única) | **¿Hace falta que la política sea dinámica?** |
| `B3` | **Oracle** (misma política, etiquetas verdaderas) | ¿Funciona la política si la clasificación fuera perfecta? |
| `B4` | Hyperion ML | ¿Cuánto del beneficio del Oracle sobrevive con el clasificador real? |

**Métricas por corrida:** tiempo, energía CPU (RAPL) y GPU (`nvmlDeviceGetTotalEnergyConsumption`, delta acumulado — **nunca** integrando `power_mw`, que es un *gauge* filtrado y con lag), potencia media, EDP, ED²P, overhead del agente, número de conmutaciones y tiempo de residencia por estado.

**Análisis:** verificar distribución y homogeneidad **antes** de elegir la prueba; Wilcoxon pareado por workload como opción por defecto; reportar tamaño del efecto y `n` junto a cada cifra; **corrección por comparaciones múltiples** (Holm) al comparar `B4` contra los cuatro baselines.

**Salida:** `B4` vs `B0` y `B4` vs `B2` decididos con evidencia estadística explícita — **incluido el caso en que la diferencia no sea significativa**, que es un resultado válido y debe reportarse como tal.

**Fase del plan:** **Fase 4 (§5.4)**.

### D5 — Diagnóstico e interpretación

Reservar tiempo explícito para el análisis de fallos que la propuesta describe en su §24: si `B3` gana y `B4` pierde, el problema es de clasificación (features, latencia de detección); si `B3` pierde, el problema es de tabla, de costo de conmutación o de la premisa binaria; si `B2` gana a `B3`, las fases son demasiado cortas para amortizar la transición en estas cargas. **Cada uno de esos tres desenlaces es un resultado publicable**, y planificarlos de antemano es lo que evita reescribir el capítulo de conclusiones a última hora.

### 8.1 Ruta crítica y paralelismo

```text
D0 ─────────────┬──────────────────────────────────────► D3 (daemon, Oracle/Trace) ──┐
                │                                                                     │
   [permiso P1] └──► D1 (campaña F0-F4) ──► D2 (tabla) ─────────────────────────────► D4 ──► D5
```

D3 nunca espera a D1. Si el permiso llega tarde, lo único que falta al final es sustituir los valores provisionales de `policy.json` por los calibrados y volver a correr `D4` — no reescribir el daemon.

---

## 9. Modos de degradación (declarados de antemano, no improvisados)

| | **Modo A — Completo** | **Modo B — Solo CPU** | **Modo C — Traza** |
|---|---|---|---|
| Disparo | `P1` y `P4` otorgados | `P1` sí, `P4` no | Ningún permiso a la fecha de corte |
| D1 | CPU + GPU F0–F4 | Solo CPU F0–F4 | No se ejecuta |
| D2 | Tabla CPU + GPU | Tabla CPU; GPU `status: blocked_by_P4` | Tabla con valores provisionales, declarados como tales |
| D3 | Completo | Completo (GPU en modo observación) | **Completo** |
| D4 | `B0`–`B4` CPU + GPU | `B0`–`B4` CPU | Reproducción sobre trazas: overhead real, tasa de conmutación, latencia de detección, estabilidad |
| Objetivo 3 | Cumplido | Cumplido | **Cumplido** |
| Objetivo 4 | Cumplido | Cumplido en CPU | **Pendiente por causa externa documentada**; EDP reportado como proyección |

**Fecha de corte recomendada:** fijar con el director una fecha a partir de la cual, si `P1` no ha llegado, se activa el Modo C sin más deliberación. Una decisión tomada con antelación y por escrito es defendible ante el jurado; una tomada bajo presión en la última semana, no.

En cualquier modo, la evidencia de ARC-77 (GPU a 765/1410 MHz bajo carga real, 62/250 W, sin límite térmico ni de potencia) se documenta como hallazgo del trabajo, no como una limitación que se disculpa: es una observación medida sobre un nodo de producción real.

---

## 10. Por qué esto está a nivel de pregrado (sustentación explícita)

Esta sección existe porque la pregunta se hizo de forma directa y merece una respuesta directa, no un "sí" tranquilizador.

**Lo que el trabajo hace, en términos de complejidad:**

- El clasificador es un árbol de decisión o un bosque aleatorio: contenido de un curso estándar de aprendizaje automático supervisado.
- La política es un diccionario de dos entradas y una máquina de estados de once pasos: contenido de un curso de sistemas operativos o de sistemas embebidos.
- La calibración es un diseño factorial `workload × frecuencia × repetición` con normalización, restricción de degradación y validación cruzada por unidad experimental: contenido de un curso de estadística aplicada / diseño de experimentos.
- No hay redes profundas, ni aprendizaje por refuerzo, ni optimización en línea, ni modelado predictivo de energía, ni instrumentación de kernel-space. Todo esto está **excluido explícitamente** por el §6.2 del plan aprobado.

**Lo que hace que el trabajo no sea trivial** — y esto es lo que se defiende en la sustentación:

1. La etiqueta no se asume por el nombre del kernel: se **deriva empíricamente** por ventana comparando intensidad operacional medida contra un ridge point calibrado en el nodo. Esa cadena metodológica (STREAM + ERT → `P_pico`/`BW_pico` → `i_ridge` → etiqueta por ventana) es el núcleo defendible del trabajo y ya está construida y verificada.
2. La tabla no se escribe a mano con la heurística intuitiva `compute = máximo, memory = mínimo`: se **selecciona con evidencia experimental** de EDP bajo restricción de degradación de rendimiento. La contribución no es afirmar que memory debe ir lento — es **medir si eso es cierto en este nodo**, y aceptar el resultado si no lo es.
3. El proyecto ya encontró, por instrumentación propia, dos cosas que la intuición no da: que `i_ridge` depende de la frecuencia (ARC-78/79) y que la GPU del nodo no hace boost bajo carga (ARC-77). Ambos hallazgos son evidencia de que la plataforma de medición funciona y produce conocimiento, no solo números.
4. La política incluye una corrección física real y no obvia (el falso `compute_bound` del spin-wait durante `cudaDeviceSynchronize()`), verificada en hardware con el shim de bloqueo (99,8 % → 0,0 % de CPU).

**Lo que el trabajo NO afirma** (se conserva íntegra la §27 de la propuesta): que `compute` deba ir siempre a `fmax`; que `memory` deba ir siempre a `fmin`; que `HIGH`/`LOW` sean universales; que la clasificación binaria sea óptima para todo workload; que la tabla de `paccaA100` sirva en otro nodo; que la política sea globalmente óptima.

**Hipótesis evaluable, en una frase:**

> En un nodo caracterizado, una política ligera que asocia regímenes compute/memory a estados DVFS calibrados por EDP puede mejorar el compromiso energía-rendimiento frente a políticas nativas o estáticas, siempre que las fases sean suficientemente estables para amortizar el costo de transición.

Es falsable, está acotada al nodo, y las cinco condiciones experimentales `B0`–`B4` están diseñadas para poder refutarla.

---

## 11. Trazabilidad contra los objetivos específicos del plan aprobado

| Objetivo específico | Dónde se cumple | Estado |
|---|---|---|
| 1. Caracterizar cargas bajo distintos estados de frecuencia con Perf/RAPL/NVML | D1 | **Bloqueado por `P1`/`P4`**; instrumentación lista y validada a `REF` |
| 2. Entrenar y validar clasificadores ligeros con baja latencia de inferencia | Fase 2 del plan (previa a D3.5) | No iniciada; sin código de ML en el repositorio |
| 3. Desarrollar el daemon que lee contadores, infiere y aplica DVFS | **D3** | Diseño consolidado en §6; **construible hoy** con `Oracle`/`Trace` |
| 4. Evaluar el impacto mediante EDP frente a governors nativos | **D4** | Diseño experimental completo en §8; ejecución sujeta al modo de degradación vigente |

---

## 12. Resumen de las cinco correcciones obligatorias

| # | Corrección | Severidad | Costo de implementarla |
|---|---|---|---|
| 1 | Anclar la etiqueta de control a `i_ridge(F_ref)` fijo; conservar la etiqueta física dependiente de frecuencia solo para el análisis | Crítica | Una constante congelada en `policy.json`; cero costo de nodo |
| 2 | Declarar los tres modos de degradación y construir D3 en paralelo, no después de D1 | Crítica | Cero: es una decisión de planificación |
| 3 | Incorporar las reglas de precedencia CPU↔GPU (`gpu_busy` → piso de CPU) | Alta | Una comparación antes de la inferencia |
| 4 | Sustituir P95 por máximo observado + IC bootstrap de la mediana; Wilcoxon pareado con corrección de Holm | Media | Re-análisis offline; cero costo de nodo |
| 5 | Añadir la tercera acción `NO_CHANGE` con banda de indecisión sobre `predict_proba`; renombrar la política en el libro | Media | Dos comparaciones y un parámetro `tau` barrido offline |

Ninguna de las cinco requiere tiempo de nodo con permiso. Las cinco pueden aplicarse antes de que llegue `P1`.

---

## 13. Referencias

**Internas (prioridad sobre este documento):**
`docs/general/plan_trabajo_grado.md` · `docs/retoma/pacca/Diseno_Politica_DVFS_CPU_GPU.md` · `docs/retoma/Guia_Maestra_Fase1_DVFS.md` · `docs/retoma/pacca/Informe_Estado_Fase1_2026-08-05.md` · `docs/retoma/pacca/Solicitud_Permisos_Pacca_Unicartagena.md` · `docs/CONTEXTO_Y_ESTADO_RAMA_HPC_STARTUP_DIAGNOSTIC.md` · `AGENTS.md` · `telemetry/include/telemetry/gpu_clock_controller.hpp`

**Externas (heredadas de la propuesta evaluada, verificadas como pertinentes):**

[1] L. Carpentieri, A. De Caro, M. S. Beni, K. Fan y B. Cosenza, "Phase-Based Frequency Scaling for Energy-Efficient Heterogeneous Computing," *IPDPS 2025*, pp. 824–836, doi: `10.1109/IPDPS64566.2025.00078`.

[2] D. Velička, O. Vysocký y L. Říha, "Methodology for GPU Frequency Switching Latency Measurement," *IPDPSW 2025*, doi: `10.1109/IPDPSW66978.2025.00133`.

[3] F. Antici, A. Bartolini, Z. Kiziltan, Ö. Babaoglu y Y. Kodama, "MCBound: An Online Framework to Characterize and Classify Memory/Compute-bound HPC Jobs," *SC24*, doi: `10.1109/SC41406.2024.00062`.

[4] R. Hebbar y A. Milenković, "PMU-Events-Driven DVFS Techniques for Improving Energy Efficiency of Modern Processors," *ACM TOMPECS*, vol. 7, 2022, doi: `10.1145/3538645`.

[5] E. S. A. Lozano y A. Gerstlauer, "Learning-based Phase-aware Multi-core CPU Workload Forecasting," *ACM TODAES*, 2022, doi: `10.1145/3564929`.

[6] K. Criswell y T. Adegbija, "A Survey of Phase Classification Techniques for Characterizing Variable Application Behavior," 2019, arXiv: `1908.02238`.

[7] Linux Kernel Documentation, "CPU Performance Scaling," `https://docs.kernel.org/admin-guide/pm/cpufreq.html`.

[8] NVIDIA Corporation, "NVML API Reference Guide," `https://docs.nvidia.com/deploy/nvml-api/`.

---

## 14. Conclusión

La propuesta *Phase-Based Table Lookup* **es viable, es óptima dentro de las restricciones de este trabajo de grado, y está a nivel de pregrado**. Se recomienda adoptarla — **8/10** — con las cinco correcciones de la sección 12, bajo el nombre *política de dos estados con banda de indecisión, calibrada por EDP normalizado*.

Su mayor virtud es que describe algo que el proyecto ya había decidido de facto sin nombrarlo, y le da anclaje bibliográfico y estructura evaluable. Su mayor debilidad no es de concepto sino de planificación: hace depender todo el entregable de un permiso administrativo que lleva meses sin llegar. La corrección número 2 —construir el daemon en paralelo, con clasificadores `Oracle`/`Trace`, y declarar por escrito los tres modos de degradación— es la que decide si el trabajo llega a sustentación con un agente de control funcionando o con una campaña que nunca pudo correr.

El punto que ninguna revisión superficial habría encontrado, y que justifica este documento, es el primero: en Hyperion la etiqueta depende del ridge, el ridge depende de la frecuencia, y la frecuencia es precisamente lo que la política modifica. Sin anclar la etiqueta de control a un ridge de referencia congelado, la política oscila de forma determinista sobre las cargas más interesantes del catálogo — `npb_sp` y `rodinia_lud`, las dos que están sobre la frontera— y ninguna cantidad de histéresis lo arregla, porque no es ruido: es el lazo cerrándose sobre sí mismo.
