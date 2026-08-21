# Plan de Fase 2 — Modelo de doble objetivo

Documento de trabajo. Traduce las ideas sueltas discutidas con el director
(regresión de doble target, Optuna para hiperparámetros, minimización de EDP)
en un diseño ejecutable sobre el dataset que **ya existe** de Fase 1.

Estado: propuesta. Fecha: 2026-08-21. Rama: `fase-02`.

---

## 0. Resumen para el director

Antes de diseñar nada se midió, sobre los datos reales, si el objetivo
"frecuencia óptima que minimiza EDP" tiene solución no trivial. El resultado
condiciona todo lo demás:

- **En CPU el EDP es monótono**: siempre gana la frecuencia máxima. No hay
  óptimo interior. La causa es un piso de potencia estática muy alto
  (~85 W a 800 MHz frente a ~118 W a 3200 MHz): bajar la frecuencia 4×
  reduce la potencia apenas 28 %, así que alargar la ejecución siempre cuesta
  más energía. Es el régimen clásico de *race-to-idle*.
- **En GPU sí hay óptimo interior y depende del kernel**: `rodinia_lavamd`
  minimiza EDP en F1 (1110 MHz, −20.6 % vs. REF) y `rodinia_lud` en F4
  (210 MHz, −30.5 % vs. REF, con solo +10.9 % de tiempo), mientras
  `gpu_dgemm_n4096` y `rodinia_gaussian` minimizan en REF.

**Consecuencia:** el aporte de "predecir la frecuencia óptima" vive en la GPU.
En CPU el modelo sigue siendo válido y necesario, pero su resultado honesto es
que la política óptima bajo EDP puro es la frecuencia máxima — eso se reporta
como hallazgo, no se disimula.

---

## 1. La evidencia

### 1.1 CPU — EDP monótono (10 reps por celda, turbo desactivado)

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

La energía —no solo el EDP— crece al bajar la frecuencia. Eso cierra también
la puerta a usar "energía mínima" como objetivo alternativo en CPU.

Sí queda una señal aprovechable: la **penalización relativa** por bajar
frecuencia depende del régimen. De F0 a F4 el tiempo crece 3.32× en `npb_cg`
(memory-bound) pero 3.81× en `dgemm_n2048` (compute-bound). El modelo puede
aprender esa diferencia aunque el argmin no cambie.

### 1.2 GPU — óptimo interior, dependiente del kernel (3 reps, CPU en REF)

| kernel | nivel | SM clk | T (s) | E (J) | EDP | vs REF |
|---|---|---:|---:|---:|---:|---:|
| gpu_dgemm_n4096 | **REF** | 1160 | 3.67 | 732.7 | **2686.0** | — |
| gpu_dgemm_n4096 | F4 | 210 | 16.97 | 1094.0 | 18570.2 | +591 % |
| rodinia_gaussian | **REF** | 1410 | 5.43 | 433.3 | **2354.7** | — |
| rodinia_lavamd | REF | 1410 | 5.74 | 497.4 | 2854.2 | — |
| rodinia_lavamd | **F1** | 1110 | 6.28 | 360.8 | **2266.9** | **−20.6 %** |
| rodinia_lud | REF | 1410 | 9.59 | 591.4 | 5669.0 | — |
| rodinia_lud | **F4** | 210 | 10.64 | 370.4 | **3942.0** | **−30.5 %** |

`rodinia_lud` es el caso didáctico: bajar el reloj de 1410 a 210 MHz cuesta
+10.9 % de tiempo y ahorra 37.4 % de energía. Es un kernel de baja ocupación
donde el reloj de SM casi no limita.

---

## 2. Los dos objetivos, construidos desde el dataset

### 2.1 Target A — grado de acotamiento continuo `b ∈ [0,1]`

El director pidió "0 = compute, 1 = memory" en continuo. No hay que inventarlo:
Fase 1 ya calcula la etiqueta binaria como un **umbral duro sobre el modelo
Roofline** (`orchestrator/postprocess.py`):

```
phase_label_train = "memory_bound" if operational_intensity < i_ridge_used
                    else "compute_bound"
```

La magnitud continua que ese umbral está discretizando es la **distancia al
ridge en escala logarítmica**. Por tanto:

```
b = σ( −k · log₁₀( OI / I_ridge ) )
```

con `σ` la sigmoide y `k > 0` un factor de escala.

Propiedades que lo hacen defendible en el documento:

- `OI = I_ridge` ⟹ `b = 0.5` exactamente. **Umbralizar `b` en 0.5 reproduce
  bit a bit la etiqueta binaria de Fase 1.** El target continuo es una
  generalización estricta del anterior, no un reemplazo arbitrario.
- `OI ≫ I_ridge` ⟹ `b → 0` (compute). `OI ≪ I_ridge` ⟹ `b → 1` (memory).
  Coincide con la convención pedida.
- `k` se calibra desde la dispersión observada de `log₁₀(OI/I_ridge)` (p. ej.
  `k = 1/σ_log`), no se elige a dedo.

Ventaja práctica: las ventanas cerca del ridge —donde la etiqueta binaria es
frágil y hace ruido— quedan cerca de 0.5 y aportan poca fuerza al gradiente,
en vez de contarse como errores duros.

### 2.2 Target B — no predecir el argmin, predecir la superficie de EDP

Esto es lo que más hay que discutir con el director, porque la formulación
literal **no funciona estadísticamente**.

Si el target es "la frecuencia óptima", su cardinalidad efectiva es el número
de kernels: 9 en CPU, 8 en GPU. Un modelo entrenado para predecir eso tiene
9 ejemplos independientes, no 10 millones. Cualquier métrica alta sería
memorización del kernel.

La formulación correcta —y la que usa la literatura ya citada en §4.2 del
anteproyecto (Guerreiro et al., Ali et al.: *medir en una configuración base y
estimar el resto del espacio DVFS*)— es un **modelo sustituto (surrogate)** del
EDP relativo:

```
ÊDP_rel( x , f_actual → f_destino )  ≈  EDP(f_destino) / EDP(f_actual)
```

donde `x` es el vector de telemetría **observado a `f_actual`**. En ejecución,
el agente de Fase 3 evalúa las 5 alternativas y toma el argmin.

Por qué esta forma y no otra:

- **Usa todo el dataset.** Cada celda (kernel, rep, fase) con 6 niveles genera
  30 pares ordenados de entrenamiento en vez de 1 etiqueta.
- **Es adimensional.** Al predecir un cociente y no un EDP absoluto, se elimina
  la escala propia de cada kernel, que es justo lo que impide generalizar con
  solo 8–9 kernels.
- **Da el costo de equivocarse.** El agente no solo sabe a dónde ir, sabe
  cuánto gana; puede exigir un margen mínimo antes de pagar la latencia de
  conmutación (la restricción que señala Velicka et al., §4.2).
- **Degrada bien.** En CPU predecirá `ÊDP_rel > 1` para todo destino distinto
  del máximo — que es exactamente el hallazgo de §1.1, reproducido por el
  modelo en vez de impuesto a mano.

---

## 3. El puente que falta: alinear fases entre frecuencias

Problema: el EDP de §1 es **por corrida**, pero el agente decide **por
ventana**. Y una ventana de la corrida a F0 y una de la corrida a F2 no son la
misma ventana — son ejecuciones distintas del mismo trabajo, estiradas en el
tiempo de forma distinta.

Solución: introducir una **coordenada de progreso invariante a la frecuencia**.

En CPU existe una natural: `delta_instructions`. El número de instrucciones
retiradas por un kernel determinista no depende del reloj. Entonces

```
p = (instrucciones acumuladas hasta la ventana) / (instrucciones totales de la corrida)   ∈ [0,1]
```

identifica el mismo punto lógico del programa en las 6 corridas. Con `p`
discretizado en `B` bins (arrancar con `B = 100`):

- cada celda `(kernel, rep, bin)` tiene **6 observaciones**, una por nivel;
- features = agregados de la ventana en ese bin (medias/percentiles de `ipc`,
  `mpki`, `llc_miss_rate`, `stall_backend_ratio`, `running_ratio`, `OI`);
- `E` y `T` del bin salen de `pkg_delta_uj` y `delta_t_ns` sumados;
- de ahí sale el `EDP_rel` de cada par de niveles.

Volumen resultante en CPU: `9 kernels × 10 reps × 100 bins = 9 000` celdas
→ **~270 000 filas** de entrenamiento para el surrogate, más 9 000 filas para
el target A agregado.

**Compuerta de validación obligatoria** antes de construir nada encima:
verificar que el conteo total de instrucciones por (kernel, rep) es
efectivamente estable entre niveles de frecuencia (tolerancia sugerida: ±2 %).
Si no lo es, `p` no es válido y hay que caer a fracción de tiempo transcurrido,
que es más débil.

**En GPU no existe `delta_instructions`** — las filas GPU son *passthrough*
(ARC-70) y no traen PMU de CPU. Allí la coordenada de progreso debe salir de
fracción de tiempo transcurrido o de una señal GPU equivalente. Es un punto
abierto, y se resuelve junto con el bloqueador de §5.

---

## 4. Protocolo de validación — la trampa que hay que evitar

Con 9.95 M de ventanas entrenables pero **solo 9 kernels**, un split aleatorio
pone ventanas de la misma corrida en train y test. El resultado sería una
exactitud cercana a 1.0 que no significa absolutamente nada: el modelo
reconoce el kernel, no el régimen.

Por tanto, el único protocolo honesto es **leave-one-kernel-out (LOKO)**:

- 9 pliegues en CPU, 8 en GPU; en cada uno, un kernel completo queda fuera.
- Se reporta la media y la dispersión entre pliegues. La dispersión importa
  tanto como la media: si un kernel se desploma, eso es el resultado.
- Ninguna repetición del kernel de test puede aparecer en entrenamiento.

Métricas:

| Objetivo | Métrica primaria | Métrica secundaria |
|---|---|---|
| Target A (`b`) | MAE sobre `b` | exactitud/F1 al umbralizar en 0.5 (comparable con Fase 1) |
| Target B (`ÊDP_rel`) | **EDP loss**: EDP alcanzado siguiendo el modelo ÷ EDP del óptimo oráculo | top-1 de acierto del argmin |
| Ambos | latencia de inferencia (p50/p99, µs) | tamaño del modelo serializado |

La *EDP loss* es la métrica que de verdad importa: un modelo que falla el
argmin pero elige una frecuencia casi tan buena es aceptable; uno que acierta
poco pero cuando falla lo hace catastróficamente, no.

---

## 5. Bloqueador: la campaña GPU solo rinde 7 %

El EDA arrojó: CPU 9 953 976 / 10 251 941 ventanas entrenables (**97.1 %**),
GPU 160 142 / 2 171 803 (**7.4 %**).

La sonda de §1.2 apunta a la causa: `rodinia_lavamd` y `rodinia_lud` reportan
`gpu_util_pct = 0` en casi todos los niveles **mientras la GPU consume energía
y el tiempo transcurre**. No es física, es un problema de muestreo/atribución
de NVML. Y `filter_gpu_trainable()` exige `gpu_util_pct ≥ 5`, así que descarta
esas corridas completas.

Dado que §1.2 demuestra que **la GPU es donde vive el aporte del target B**,
esto deja de ser un pendiente menor: **es el bloqueador crítico de Fase 2**.
Requiere `paccaA100`, hoy ocupado.

Trabajo que sí se puede adelantar sin `paccaA100`: todo el pipeline de CPU
(§2.1, §3, §4) y el análisis offline de los `windows.csv` de GPU ya
recolectados, que basta para diagnosticar el 7 % aunque no para recolectar de
nuevo.

---

## 6. Dónde entra Optuna

Optuna no es "para mejorar el modelo" en abstracto; entra en dos lugares
concretos, y en el segundo aporta algo que el anteproyecto ya exige.

**6.1 Búsqueda de hiperparámetros con LOKO como objetivo.** El valor que
optimiza Optuna debe ser la métrica de §4 promediada sobre los pliegues LOKO,
nunca sobre un split aleatorio — si no, se optimiza la fuga de información.

**6.2 Búsqueda multiobjetivo (precisión ↔ latencia).** §5.2 del anteproyecto
exige explícitamente "el mejor compromiso entre desempeño predictivo y costo
computacional". Eso es literalmente un problema bi-objetivo, y Optuna lo
soporta de forma nativa (`directions=["minimize", "minimize"]`, NSGA-II):

```
objetivo_1 = EDP loss media en LOKO        → minimizar
objetivo_2 = latencia de inferencia p99    → minimizar
```

El resultado no es un modelo sino un **frente de Pareto**, del cual se elige un
punto con criterio explícito y justificable en el documento (p. ej. la latencia
máxima tolerable dada la cadencia de decisión del agente). Eso es mucho más
defendible que "probamos varios y este dio mejor".

**Familias de modelos a barrer**, respetando la restricción de §5.2 del
anteproyecto (modelos clásicos ligeros; se excluyen por diseño deep learning y
RL profundo):

- `DecisionTreeRegressor` — línea base interpretable.
- `RandomForestRegressor` / `ExtraTreesRegressor` — soportan multi-salida
  nativamente (los dos targets a la vez).
- `HistGradientBoostingRegressor` — un regresor por salida.
- Regresión lineal / ridge — piso de comparación obligatorio; si un modelo
  complejo no le gana en LOKO, el hallazgo es que las features no bastan.

---

## 7. Estructura de módulos

Sobre el paquete `classifier/` ya creado:

```
classifier/
  features/
    load.py       ✅ ya existe — carga y filtrado de windows.csv
    targets.py    ← construcción de b (§2.1) y de EDP/EDP_rel (§2.2)
    align.py      ← coordenada de progreso invariante y binning (§3)
  training/
    dataset.py    ← ensambla la matriz final (features, targets, grupo=kernel)
    search.py     ← estudios de Optuna (§6)
    train.py      ← entrenamiento final y serialización para Fase 3
  eval/
    protocol.py   ← splitter LOKO y métricas de §4
    latency.py    ← medición de latencia de inferencia p50/p99
  notebooks/
    01_eda.ipynb  ✅ ya ejecutado contra el dataset real
    02_targets.ipynb   ← inspección visual de b y de las curvas EDP(f)
```

Cada módulo con pruebas en `tests/classifier/`, siguiendo la convención ya
establecida.

---

## 8. Orden de ejecución

El orden importa porque cada paso puede invalidar los siguientes. Son
compuertas, no una lista de tareas paralelas.

1. **Verificar la invariancia de instrucciones** (§3). Si falla, el diseño de
   alineación cambia antes de escribir código encima. *No requiere `paccaA100`.*
2. **`features/targets.py` + `02_targets.ipynb`.** Construir `b`, confirmar que
   umbralizar en 0.5 reproduce la etiqueta de Fase 1, y graficar las curvas
   EDP(f) por kernel para tener §1 completo (9 kernels CPU, no 3).
3. **`features/align.py`.** Binning por progreso; producir la tabla de celdas
   `(kernel, rep, bin)` con sus 6 niveles.
4. **`eval/protocol.py`.** LOKO y métricas **antes** de entrenar nada, para que
   ningún número se produzca fuera del protocolo.
5. **`training/dataset.py` + línea base.** Árbol de decisión y ridge, ya bajo
   LOKO. Aquí se sabe si el problema tiene señal.
6. **`training/search.py`.** Optuna mono-objetivo primero, multiobjetivo
   después.
7. **Desbloquear GPU** (§5) en cuanto `paccaA100` libere, y repetir 2–6 sobre
   GPU, que es donde el target B tiene óptimo interior real.
8. **Serializar el modelo elegido** para Fase 3.

---

## 9. Riesgos declarados

| Riesgo | Impacto | Mitigación |
|---|---|---|
| EDP monótono en CPU | El target B en CPU es trivial | Se reporta como hallazgo; el aporte de B se sostiene en GPU (§1.2) |
| Solo 8–9 kernels | Poca capacidad de generalizar | LOKO obligatorio; predecir cocientes adimensionales; reportar dispersión entre pliegues |
| Campaña GPU al 7 % | Bloquea el aporte principal | Prioridad crítica; diagnóstico offline ya iniciable |
| Instrucciones no invariantes | Rompe la alineación de §3 | Compuerta 1 del orden de ejecución |
| Sin progreso invariante en GPU | Alineación GPU más débil | Resolver junto con §5 |
| Pseudo-replicación (10 M ventanas, 9 kernels) | Métricas infladas | Agregar a celdas antes de entrenar; nunca split aleatorio |
