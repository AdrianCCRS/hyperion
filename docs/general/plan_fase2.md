# Plan de Fase 2 — Modelo de doble objetivo

Documento de trabajo. Traduce las ideas discutidas con el director (regresión de
doble target, Optuna para hiperparámetros, minimización de EDP) en un diseño
ejecutable sobre el dataset que **ya existe** de Fase 1.

Estado: propuesta. Fecha: 2026-08-21. Rama: `fase-02`.

---

## 0. Resumen

Antes de diseñar nada se midió, sobre los datos reales, si los objetivos
propuestos tienen solución no trivial. Salieron tres hallazgos, y los tres
condicionan el diseño:

1. **En CPU el EDP es monótono**: siempre gana la frecuencia máxima. Piso de
   potencia estática muy alto (~85 W a 800 MHz contra ~118 W a 3200 MHz):
   bajar la frecuencia 4× reduce la potencia apenas 28 %, así que alargar la
   ejecución siempre cuesta más energía. Régimen de *race-to-idle*.
2. **En GPU sí hay óptimo interior y depende del kernel**: `rodinia_lavamd`
   minimiza EDP en F1 (−20.6 % vs. REF) y `rodinia_lud` en F4 (−30.5 %, con
   solo +10.9 % de tiempo), mientras `gpu_dgemm_n4096` y `rodinia_gaussian`
   minimizan en REF.
3. **El ridge del Roofline se mueve con la frecuencia**: cae de 8.733 a 2.992
   FLOP/byte (×0.34) entre REF y F4, idéntico para los 9 kernels porque es una
   propiedad del nodo. **La etiqueta de fase es por tanto relativa a la
   frecuencia a la que se mide**, no una propiedad intrínseca del código.

**Consecuencia:** el aporte de "predecir la frecuencia óptima" vive en la GPU.
En CPU el modelo sigue siendo válido, pero su resultado honesto bajo EDP puro es
la frecuencia máxima — eso se reporta como hallazgo. Y el hallazgo 3 obliga a
una compuerta de validación previa (§8) que antes no estaba en el plan.

---

## 1. Evidencia: EDP frente a frecuencia

### 1.1 CPU — la rampa no se dobla nunca (10 reps por celda, turbo desactivado)

| kernel | nivel | MHz | T (s) | E (J) | EDP (J·s) |
|---|---|---:|---:|---:|---:|
| npb_cg | REF | 3200 | 7.14 | 846.8 | 6044.3 |
| npb_cg | **F0** | 3200 | 7.14 | 841.7 | **6008.3** |
| npb_cg | F1 | 2600 | 8.13 | 871.0 | 7077.7 |
| npb_cg | F2 | 2000 | 9.98 | 990.3 | 9882.7 |
| npb_cg | F3 | 1400 | 13.66 | 1248.9 | 17062.1 |
| npb_cg | F4 | 800 | 23.71 | 2008.9 | 47625.7 |
| npb_ft | **REF** | 3200 | 4.54 | 541.7 | **2459.0** |
| npb_ft | F4 | 800 | 15.16 | 1265.3 | 19178.1 |
| dgemm_n2048 | **REF** | 3200 | 3.04 | 434.7 | **1320.7** |
| dgemm_n2048 | F4 | 800 | 11.62 | 992.2 | 11531.0 |

La energía —no solo el EDP— crece al bajar la frecuencia. Eso cierra también la
puerta a usar "energía mínima" como objetivo alternativo en CPU.

**Alcance de esta medición:** es por **corrida completa** (frecuencia fija
durante toda la ejecución), no por ventana. Responde "¿qué frecuencia estática
sostenida minimiza EDP?", **no** "¿ayuda cambiar de frecuencia dentro de la
misma corrida?", que es la pregunta real de la tesis. Esa segunda pregunta
requiere la infraestructura de §4 y la compuerta de §8.

### 1.2 GPU — el óptimo se mueve (3 reps, CPU en REF)

| kernel | nivel | SM clk | T (s) | E (J) | EDP | vs REF |
|---|---|---:|---:|---:|---:|---:|
| gpu_dgemm_n4096 | **REF** | 1160 | 3.67 | 732.7 | **2686.0** | — |
| gpu_dgemm_n4096 | F4 | 210 | 16.97 | 1094.0 | 18570.2 | +591 % |
| rodinia_gaussian | **REF** | 1410 | 5.43 | 433.3 | **2354.7** | — |
| rodinia_lavamd | REF | 1410 | 5.74 | 497.4 | 2854.2 | — |
| rodinia_lavamd | **F1** | 1110 | 6.28 | 360.8 | **2266.9** | **−20.6 %** |
| rodinia_lud | REF | 1410 | 9.59 | 591.4 | 5669.0 | — |
| rodinia_lud | **F4** | 210 | 10.64 | 370.4 | **3942.0** | **−30.5 %** |

---

## 2. Evidencia: el ridge se mueve con la frecuencia

El ridge del Roofline es `I_ridge = FLOPS_pico / ancho_de_banda`. Al bajar el
reloj cae el pico de cómputo, pero el ancho de banda de memoria casi no cambia
— así que **el ridge se desplaza a la izquierda**:

| nivel | MHz | `i_ridge_used` (FLOP/byte) |
|---|---:|---:|
| REF | 3200 | 8.733 |
| F0 | 3200 | 8.675 |
| F1 | 2600 | 7.258 |
| F2 | 2000 | 5.690 |
| F3 | 1400 | 4.313 |
| F4 | 800 | 2.992 |

Es idéntico para los 9 kernels dentro de cada nivel: es una calibración del
nodo por nivel de frecuencia, no una propiedad del kernel. La caída (×0.34) es
menor que la de la frecuencia (×0.25), lo que indica que el ancho de banda
efectivo también baja algo a menor reloj.

### 2.1 Consecuencia: 2 de 9 kernels invierten su etiqueta

`OI / I_ridge` por kernel y nivel (>1 ⟹ compute_bound, <1 ⟹ memory_bound):

| kernel | REF | F0 | F1 | F2 | F3 | F4 | veredicto |
|---|---:|---:|---:|---:|---:|---:|---|
| npb_cg | 0.020 | 0.020 | 0.024 | 0.030 | 0.038 | 0.054 | estable (memory) |
| npb_sp | 0.042 | 0.042 | 0.061 | 0.104 | 0.123 | 0.177 | estable (memory) |
| npb_mg | 0.043 | 0.044 | 0.052 | 0.073 | 0.097 | 0.133 | estable (memory) |
| npb_ft | 0.150 | 0.151 | 0.181 | 0.233 | 0.315 | 0.455 | estable (memory) |
| **npb_lu** | 0.352 | 0.353 | 0.518 | 0.682 | **0.964** | **1.376** | **INVIERTE** |
| **npb_bt** | 0.732 | 0.738 | 0.883 | **1.147** | **1.498** | **2.125** | **INVIERTE** |
| dgemm_n2048 | 2.457 | 2.458 | 2.953 | 4.018 | 5.561 | 7.644 | estable (compute) |
| rajaperf_polybench_3mm_omp | 5.866 | 6.008 | 5.554 | 7.854 | 10.007 | 12.091 | estable (compute) |
| rodinia_lavamd_omp | 28.393 | 28.623 | 30.640 | 33.919 | 35.622 | 34.916 | estable (compute) |

Fracción de ventanas `compute_bound` en los dos que invierten:

| kernel | REF | F0 | F1 | F2 | F3 | F4 |
|---|---:|---:|---:|---:|---:|---:|
| npb_bt | 0.6 % | 0.8 % | 19.4 % | 78.5 % | 86.0 % | 89.8 % |
| npb_lu | 0.0 % | 0.0 % | 0.0 % | 8.5 % | 34.4 % | 69.1 % |

**Lectura correcta:** no es un bug del pipeline. Es Roofline funcionando como
debe — el roofline es *por configuración*, y a menor frecuencia más cargas
quedan limitadas por cómputo. Los 7 kernels estables están lejos del ridge en
todo el rango; los 2 que invierten tienen `OI` dentro de la banda que el ridge
barre (2.99–8.73).

Para el agente en línea esto **funciona a favor**: siempre observará telemetría
a la frecuencia actual y calculará el label con el ridge de esa frecuencia, que
es exactamente la relatividad correcta.

### 2.2 Advertencia: la etiqueta no es una predicción de escalado

El label Roofline dice **cuál techo te limita**, no garantiza cómo escalará el
tiempo con la frecuencia. Hay evidencia directa: `npb_bt` está etiquetado
`memory_bound` a REF (99.4 % de sus ventanas) y aun así **se alarga 3.74×** al
pasar a F4 — casi el 4× del cociente de frecuencia, es decir, se comporta como
compute-bound. Una carga puede estar por debajo del ridge sin saturar el ancho
de banda (limitada por latencia, dependencias u ocupación).

Esto es lo que la compuerta 0 de §8 tiene que medir antes de construir nada
encima, porque toca la hipótesis central de la tesis: *que clasificar la fase
sirve para elegir la frecuencia*.

> **Corrección de un análisis previo.** Una sonda intermedia pareció mostrar que
> las ventanas `memory_bound` de `npb_lu` apenas se alargaban (1.14×) frente al
> 3.69× del kernel completo — lo que se leyó como señal a favor de DVFS
> dinámico. **Era un artefacto.** A REF el 100 % de las ventanas de `npb_lu`
> están etiquetadas `memory_bound` y a F4 solo el 30.9 %: el cociente comparaba
> conjuntos distintos de ventanas, no la misma fase estirándose. Medir eso bien
> exige una definición de fase invariante a la frecuencia, que es justamente lo
> que §4 construye.

---

## 3. Los dos objetivos, construidos desde el dataset

### 3.1 Target A — grado de acotamiento continuo `b ∈ [0,1]`

El director pidió "0 = compute, 1 = memory" en continuo. No hay que inventarlo:
Fase 1 ya calcula la etiqueta binaria como un umbral duro sobre el Roofline
(`orchestrator/postprocess.py`):

```
phase_label_train = "memory_bound" if operational_intensity < i_ridge_used
                    else "compute_bound"
```

La magnitud continua que ese umbral discretiza es la **distancia al ridge en
escala logarítmica**:

```
b = σ( −k · log₁₀( OI / I_ridge(f) ) )
```

Nótese `I_ridge(f)`: el ridge del nivel al que se tomó la telemetría (§2).

Propiedades:

- `OI = I_ridge` ⟹ `b = 0.5` exactamente. **Umbralizar en 0.5 reproduce bit a
  bit la etiqueta binaria de Fase 1.** Es una generalización estricta.
- `OI ≫ I_ridge` ⟹ `b → 0` (compute); `OI ≪ I_ridge` ⟹ `b → 1` (memory).
- `k` se calibra desde la dispersión observada de `log₁₀(OI/I_ridge)`.

Ventaja adicional a la luz de §2: los kernels que invierten (`npb_bt`,
`npb_lu`) producen un `b` que **cruza suavemente 0.5** en vez de saltar de
`memory` a `compute` entre dos niveles. El target continuo absorbe la
transición que el binario convierte en discontinuidad.

### 3.2 Target B — no predecir el argmin, predecir la superficie de EDP

La formulación literal **no funciona estadísticamente**. Si el target es "la
frecuencia óptima", su cardinalidad efectiva es el número de kernels: 9 en CPU,
8 en GPU. Un modelo entrenado para eso tiene 9 ejemplos independientes, no 10
millones; cualquier métrica alta sería memorización del kernel.

La formulación correcta —y la que usa la literatura ya citada en §4.2 del
anteproyecto (Guerreiro et al., Ali et al.: *medir en una configuración base y
estimar el resto del espacio DVFS*)— es un **modelo sustituto** del EDP
relativo:

```
ÊDP_rel( x , f_actual → f_destino )  ≈  EDP(f_destino) / EDP(f_actual)
```

donde `x` es la telemetría **observada a `f_actual`**. En ejecución, el agente
evalúa las 5 alternativas y toma el argmin.

- **Usa todo el dataset.** Cada celda con 6 niveles genera 30 pares ordenados.
- **Es adimensional.** Predecir un cociente elimina la escala de cada kernel.
- **Da el costo de equivocarse**, necesario para amortizar la latencia de
  conmutación (Velicka et al., §4.2).
- **Degrada bien.** En CPU predecirá `ÊDP_rel > 1` para todo destino distinto
  del máximo — el hallazgo de §1.1 reproducido por el modelo, no impuesto.

---

## 4. El puente que falta: alinear fases entre frecuencias

El EDP de §1 es **por corrida**, pero el agente decide **por ventana**. Y una
ventana a F0 no es la misma ventana a F2.

Hace falta una **coordenada de progreso invariante a la frecuencia**. En CPU
existe una natural: `delta_instructions` — el número de instrucciones retiradas
por un kernel determinista no depende del reloj.

```
p = instrucciones acumuladas hasta la ventana / instrucciones totales   ∈ [0,1]
```

Con `p` en `B` bins (arrancar con `B = 100`), cada celda `(kernel, rep, bin)`
reúne 6 observaciones del mismo punto lógico del programa. De ahí salen las
features agregadas, `E` y `T` del bin, y el `EDP_rel` de cada par de niveles.

Volumen en CPU: `9 × 10 × 100 = 9 000` celdas → **~270 000 filas** para el
sustituto.

**Dos supuestos que hay que verificar antes de construir encima:**

1. Que el conteo total de instrucciones por (kernel, rep) sea estable entre
   niveles (tolerancia ±2 %). Si no, `p` no es válido y hay que caer a fracción
   de tiempo transcurrido, más débil.
2. **Que la etiqueta de fase NO se asuma constante entre niveles** — §2
   demuestra que no lo es para 2 de 9 kernels. En cada celda hay que llevar el
   label (y el `b`) *de cada nivel por separado*, nunca propagar el de REF.

**En GPU no existe `delta_instructions`** — las filas GPU son *passthrough*
(ARC-70). La coordenada de progreso debe salir de otra señal; punto abierto que
se resuelve junto con el bloqueador de §6.

---

## 5. Protocolo de validación

Con 9.95 M de ventanas entrenables pero **solo 9 kernels**, un split aleatorio
pone ventanas de la misma corrida en train y test. El resultado sería una
exactitud cercana a 1.0 sin ningún valor: el modelo reconoce el kernel, no el
régimen. El único protocolo honesto es **leave-one-kernel-out (LOKO)**: 9
pliegues en CPU, 8 en GPU, reportando media *y dispersión* entre pliegues.

| Objetivo | Métrica primaria | Métrica secundaria |
|---|---|---|
| Target A (`b`) | MAE sobre `b` | F1 al umbralizar en 0.5 (comparable con Fase 1) |
| Target B (`ÊDP_rel`) | **EDP loss**: EDP del modelo ÷ EDP del oráculo | top-1 de acierto del argmin |
| Ambos | latencia de inferencia p50/p99 | tamaño del modelo serializado |

La *EDP loss* es la que importa: un modelo que falla el argmin pero elige una
frecuencia casi tan buena es aceptable; uno que acierta a menudo pero falla
catastróficamente, no.

---

## 6. Bloqueador: la campaña GPU rinde 7 %

CPU: 9 953 976 / 10 251 941 ventanas entrenables (**97.1 %**).
GPU: 160 142 / 2 171 803 (**7.4 %**).

La sonda de §1.2 apunta a la causa: `rodinia_lavamd` y `rodinia_lud` reportan
`gpu_util_pct = 0` en casi todos los niveles **mientras la GPU consume energía y
el tiempo transcurre**. No es física, es muestreo o atribución de NVML. Y
`filter_gpu_trainable()` exige `gpu_util_pct ≥ 5`, así que descarta esas
corridas completas.

Como §1.2 demuestra que **la GPU es donde vive el aporte del Target B**, esto es
el **bloqueador crítico** de Fase 2. Requiere `paccaA100`, hoy ocupado.

Adelantable sin `paccaA100`: todo el pipeline de CPU y el diagnóstico offline
sobre los `windows.csv` de GPU ya recolectados.

---

## 7. Dónde entra Optuna

**7.1 Hiperparámetros con LOKO como objetivo.** El valor que Optuna optimiza
debe ser la métrica de §5 promediada sobre los pliegues LOKO, nunca sobre un
split aleatorio — si no, se optimiza la fuga de información.

**7.2 Búsqueda multiobjetivo (precisión ↔ latencia).** §5.2 del anteproyecto
exige "el mejor compromiso entre desempeño predictivo y costo computacional".
Eso es un problema bi-objetivo, y Optuna lo soporta nativamente (NSGA-II):

```
objetivo_1 = EDP loss media en LOKO        → minimizar
objetivo_2 = latencia de inferencia p99    → minimizar
```

El resultado es un **frente de Pareto**, del cual se elige un punto con criterio
explícito y justificable. Más defendible que "probamos varios y este dio mejor".

**Familias a barrer**, respetando §5.2 del anteproyecto (modelos clásicos
ligeros; se excluyen por diseño deep learning y RL profundo):

- `DecisionTreeRegressor` — línea base interpretable.
- `RandomForestRegressor` / `ExtraTreesRegressor` — multi-salida nativa.
- `HistGradientBoostingRegressor` — un regresor por salida.
- Regresión lineal / ridge — piso de comparación obligatorio; si un modelo
  complejo no le gana en LOKO, el hallazgo es que las features no bastan.

---

## 8. Orden de ejecución

Son compuertas, no tareas paralelas: cada una puede invalidar las siguientes.

0. **Test de validez de la premisa** *(nuevo, tras §2.2)*. Para cada
   `(kernel, bin)`, medir el estiramiento real `T(F4)/T(REF)` y comprobar si el
   `b` calculado a REF lo predice. **Si la fase no predice el escalado, la
   hipótesis central de la tesis necesita reformularse antes de entrenar nada.**
   `npb_bt` ya es un contraejemplo parcial. *No requiere `paccaA100`.*
1. **Verificar la invariancia de instrucciones** (§4). Si falla, cambia el
   diseño de alineación.
2. **`features/targets.py` + `02_targets.ipynb`.** Construir `b` con
   `I_ridge(f)`, confirmar que umbralizar en 0.5 reproduce la etiqueta de Fase 1
   y graficar EDP(f) de los 9 kernels de CPU.
3. **`features/align.py`.** Binning por progreso, con label por nivel (§4).
4. **`eval/protocol.py`.** LOKO y métricas **antes** de entrenar nada.
5. **`training/dataset.py` + línea base.** Árbol y ridge bajo LOKO. Aquí se sabe
   si el problema tiene señal.
6. **`training/search.py`.** Optuna mono-objetivo, luego multiobjetivo.
7. **Desbloquear GPU** (§6) y repetir 2–6 sobre GPU.
8. **Serializar el modelo elegido** para Fase 3.

---

## 9. Riesgos declarados

| Riesgo | Impacto | Mitigación |
|---|---|---|
| EDP monótono en CPU | Target B trivial en CPU | Se reporta como hallazgo; el aporte se sostiene en GPU (§1.2) |
| **La fase no predice el escalado** | **Toca la hipótesis central** | **Compuerta 0**; evidencia parcial en contra ya en `npb_bt` (§2.2) |
| Etiqueta relativa a la frecuencia | Alineación ingenua sería incorrecta | Label por nivel, nunca propagar el de REF (§4) |
| Solo 8–9 kernels | Poca generalización | LOKO obligatorio; cocientes adimensionales; reportar dispersión |
| Campaña GPU al 7 % | Bloquea el aporte principal | Prioridad crítica; diagnóstico offline ya iniciable |
| Instrucciones no invariantes | Rompe la alineación de §4 | Compuerta 1 |
| Sin progreso invariante en GPU | Alineación GPU más débil | Resolver junto con §6 |
| Pseudo-replicación | Métricas infladas | Agregar a celdas antes de entrenar; nunca split aleatorio |
