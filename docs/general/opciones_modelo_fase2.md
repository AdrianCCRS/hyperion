# Opciones de diseño para el modelo de Fase 2

Documento para contrastar. Cuatro formas de plantear el modelo, con lo que
cada una gana y pierde **medido contra los datos que ya tenemos**, no en
abstracto.

Fecha: 2026-08-21. Rama: `fase-02`.

---

## 1. El criterio de decisión

Lo que separa a estas opciones no es "cuál usa mejor machine learning". Es
esto:

> **Con los datos que tenemos, ¿el modelo tiene algo que aprender?**

Porque hoy, con el análisis completo (9/9 kernels de CPU, 8/8 de GPU):

- La **frecuencia óptima en CPU es la máxima en 9 de 9 kernels**. Como
  target, es una constante.
- La **sensibilidad a la frecuencia sí varía mucho**: el estiramiento al
  bajar a 800 MHz va de 2.21× a 4.05×, y el parámetro α que lo resume va
  de 0.385 a 1.010.
- En **GPU sí hay óptimos interiores** (2 de 8 en EDP, 7 de 8 en energía
  pura), pero solo el 7.4 % del dataset es utilizable hasta arreglar el
  problema de NVML.

Un modelo que predice una constante no se puede evaluar: un predictor
trivial que siempre responde "máxima" acierta igual. Un modelo que predice
α tiene rango real que recorrer. **Ese es el criterio.**

## 2. Evidencia disponible

| Hecho medido | Valor |
|---|---|
| Fase predice sensibilidad a frecuencia (compuerta 0) | Pearson **−0.82** (n=9, p≈0.007) |
| Rango de estiramiento T(800)/T(3200) | 2.21× – 4.05× |
| Rango de α (fracción de tiempo sensible a frecuencia) | 0.385 – 1.010 |
| Ajuste de la ley de escalado T(f) = T_mem + T_cpu·(f_ref/f) | R² 0.978 – 0.9998 |
| Óptimo de EDP en CPU fuera de f_max | **0 de 9 kernels** |
| Óptimo de EDP en GPU fuera de f_max | 2 de 8 kernels |
| Mínimo de energía en GPU por debajo de f_max | 7 de 8 kernels |
| Ahorro medio de energía en GPU con ≤10 % slowdown | +7.4 % (máx. +27.5 %) |
| Ventanas entrenables CPU / GPU | 97.1 % / 7.4 % |
| Kernels disponibles CPU / GPU | 9 / 8 |
| Desplazamiento del ridge entre 3200 y 800 MHz | 8.733 → 2.992 FLOP/byte |

---

## 3. Opción A — Doble target directo (propuesta del director)

**Qué hace.** Un regresor multi-salida (RandomForest / XGBoost) que recibe
telemetría de la ventana y devuelve:

1. `b` ∈ [0,1] — 0 compute, 1 memory.
2. `f_opt` — la frecuencia óptima, normalizada.

**Cómo se construye la etiqueta de `f_opt`.** Aquí está todo el asunto. Si
se calcula **por kernel** (el óptimo de la corrida completa), los 10
millones de ventanas de `npb_cg` llevan todas la misma etiqueta, y el
modelo tiene 9 ejemplos independientes, no 10 millones — aprende a
reconocer el kernel. Para que funcione hay que calcularla **por tramo de
progreso**, lo que exige toda la infraestructura de alineación (§6).

**A favor**
- Es lo que el director propuso; una sola inferencia da la decisión.
- Directo e interpretable: el modelo dice la respuesta.

**En contra**
- **Hoy el target es constante en CPU** (f_max en 9/9). No hay nada que
  aprender y la evaluación no distingue el modelo de un predictor trivial.
- El *argmin* es discontinuo: cuando dos frecuencias están casi empatadas,
  ventanas casi idénticas llevan etiquetas distintas y la regresión
  promedia.
- No dice **cuánto** se gana, así que el agente no puede decidir si vale la
  pena pagar la latencia de conmutación.
- Queda atado a un objetivo. Si mañana el criterio pasa de EDP a "energía
  con ≤5 % de degradación", hay que reetiquetar y reentrenar.

---

## 4. Opción B — Sustituto de cocientes de EDP

**Qué hace.** Predice `EDP(f_destino) / EDP(f_actual)` para cada
frecuencia candidata; el agente evalúa las alternativas y toma el argmin.

**A favor**
- Cada par de frecuencias es un ejemplo: multiplica los datos por ~30.
- Adimensional, así que generaliza mejor entre kernels de escalas distintas.
- Da la magnitud de la ganancia.

**En contra**
- Requiere una inferencia por frecuencia candidata (5–11 según la rejilla).
  *Nota: medí mal este punto antes — con un bosque pequeño son decenas de
  microsegundos cada una, así que no es el factor decisivo que dije.*
- Sigue necesitando alineación por tramos.
- Menos interpretable: no hay un "estado" del sistema, solo comparaciones.
- Igual que A, atado al objetivo EDP en la propia etiqueta.

---

## 5. Opción C — Ley de escalado (propuesta nueva)

**Qué hace.** En vez de predecir la decisión, predice la **física** de la
que se deduce la decisión. El modelo devuelve:

1. `b` ∈ [0,1] — el score de fase, igual que en A (sin cambios).
2. `α` ∈ [0,1] — la **fracción del tiempo que es sensible a la
   frecuencia**.

De α sale la curva de tiempo completa, analíticamente:

```
T(f) / T(f_ref) = (1 - α) + α · (f_ref / f)
```

α = 0 significa "el reloj no me afecta" (todo el tiempo es espera a
memoria). α = 1 significa "escalo perfectamente con el reloj". La potencia
se modela aparte, una sola vez por nodo (§5.3). Con T(f) y P(f) se calcula
EDP(f), energía, ED²P o lo que sea, y el argmin sale como **consecuencia**,
no como predicción.

### 5.1 Por qué creo que es mejor

**Tiene señal donde las otras no.** α va de 0.385 a 1.010 en los mismos 9
kernels donde `f_opt` es constante. El modelo tiene un rango real que
aprender aunque la decisión resulte ser siempre la misma, y la evaluación
mide algo.

**Está validada contra los datos.** Ajustando la ley a los 5 niveles
medidos de cada kernel:

| kernel | α | R² | peor error |
|---|---:|---:|---:|
| rajaperf_polybench_3mm_omp | 1.010 | 0.9998 | 1.2 % |
| dgemm_n2048 | 0.932 | 0.9995 | 1.8 % |
| npb_bt | 0.902 | 0.9996 | 1.6 % |
| npb_lu | 0.840 | 0.9997 | 1.3 % |
| npb_ft | 0.771 | 0.9997 | 1.2 % |
| npb_cg | 0.765 | 0.9972 | 4.1 % |
| npb_sp | 0.491 | 0.9942 | 4.1 % |
| npb_mg | 0.385 | 0.9777 | 7.9 % |

Un solo número reconstruye la curva completa con 1–2 % de error en la
mayoría.

**Un modelo sirve para todos los objetivos.** EDP, ED²P, energía mínima,
energía con ≤5 % de degradación: todos salen del mismo α sin reentrenar.
Las opciones A y B hornean el objetivo dentro de la etiqueta.

**Es literalmente lo que hace la literatura que ya citas.** Del anteproyecto
§4.2, sobre Guerreiro et al.: *"inferir, a partir de eventos de hardware
recolectados a una sola frecuencia, cómo cambiarán tiempo de ejecución,
potencia y energía al recorrer el resto del espacio de frecuencias"*. Y
sobre Ali et al.: *"modelado analítico de potencia y tiempo de ejecución, y
selección multiobjetivo mediante EDP y ED2P"*. Ninguno de los dos predice
el argmin: los dos modelan tiempo y potencia y después seleccionan.

**α es interpretable y contrastable.** Existe una señal en la telemetría
que mide algo muy parecido de forma directa: `stall_backend_ratio`, la
fracción de ciclos detenidos esperando al backend. Eso da una línea base
física y honesta contra la cual comparar el modelo aprendido.

### 5.2 En contra

- Hay que modelar también la potencia; sus errores se suman a los de α.
- Asume una forma funcional. Ajusta muy bien a nivel de corrida, pero
  **está por verificar a nivel de fase** — que es donde se va a usar.
- El error se propaga por el paso analítico: un error en α se amplifica al
  calcular el óptimo.
- Sigue necesitando alineación por tramos para etiquetar α por fase.

### 5.3 Sobre el modelo de potencia

De los datos completos, la potencia es mucho más una propiedad del **nodo**
que del kernel: a 3200 MHz va de 107 a 143 W (y el 143 es `dgemm`, saturando
AVX-512), y a 800 MHz de 80 a 89 W en los 9. Así que un modelo de potencia
por nodo con un desplazamiento dependiente de la carga debería bastar, sin
un segundo modelo aprendido pesado.

---

## 6. Opción D — Mínimo fiel al anteproyecto

**Qué hace.** Solo el clasificador de fase. La traducción fase → frecuencia
vive en la Fase 3 como una política interpretable (una tabla de reglas), no
como un modelo aprendido.

**A favor**
- Es **exactamente** lo que el anteproyecto promete. La §5.2 dice, textual:
  *"El problema se formula como una tarea de clasificación supervisada […]
  produce como salida una etiqueta que representa la fase de ejecución"*, y
  que los modelos se comparan *"utilizando métricas de clasificación y
  tiempo de inferencia"*.
- No necesita alineación por tramos ni etiquetas derivadas: se entrena
  directo sobre las 9.95 M ventanas ya etiquetadas y balanceadas (50/50).
- Es el único que **no depende** de que exista un óptimo interior.
- Riesgo mínimo, entregable garantizado.

**En contra**
- Deja fuera la idea del doble target, que es una aportación real del
  director por encima de lo comprometido.
- La política de la Fase 3 queda sin fundamento cuantitativo: hay que
  elegir las frecuencias a mano.

---

## 7. Comparación

| | A · doble target | B · cocientes | C · ley de escalado | D · mínimo |
|---|---|---|---|---|
| ¿Aprende algo si el óptimo es constante? | **No** | No | **Sí** (α varía 0.39–1.01) | Sí (no aplica) |
| Datos de entrenamiento | tramos | tramos × 30 pares | tramos | 9.95 M ventanas directas |
| Inferencias por decisión | 1 | 5–11 | 1 | 1 |
| ¿Sirve para varios objetivos sin reentrenar? | No | No | **Sí** | n/a |
| ¿Da la magnitud de la ganancia? | No | Sí | **Sí** | No |
| Necesita alineación por tramos | Sí | Sí | Sí | **No** |
| Fidelidad al anteproyecto | extiende | extiende | extiende | **exacta** |
| Fidelidad a la literatura citada | media | alta | **alta** | media |
| Interpretabilidad | media | baja | **alta** | alta |
| Validado contra los datos hoy | — | — | **R² ≥ 0.978** | — |
| Riesgo de no entregar | medio | medio | medio | **bajo** |

---

## 8. Recomendación

**D + C, en ese orden, y C solo si la señal aparece.**

1. **Construir D primero.** El clasificador de fase es lo prometido, tiene
   fundamento medido (compuerta 0, r = −0.82), datos abundantes y
   balanceados, y no depende de ningún hallazgo pendiente. Es el entregable
   que garantiza la tesis pase lo que pase con la campaña de rejilla fina y
   con el desbloqueo de la GPU.

2. **Medir si el óptimo varía entre fases dentro de un mismo kernel.** Si
   no varía, A y B se quedan sin objeto y C se queda solo con α — que sigue
   siendo útil, pero es otra historia. Esto se responde con los datos
   actuales, sin `paccaA100`.

3. **Si hay señal, añadir C.** Mantiene la salida `b` que el director pidió
   —el 0 a 1 no cambia—, sustituye `f_opt` por α, y la frecuencia óptima
   sale del cálculo. Es un cambio de *qué se predice*, no de la
   arquitectura: sigue siendo un RandomForest o XGBoost de regresión
   multi-salida, exactamente como se planteó.

**Lo que cambiaría respecto a la propuesta original:** solo la segunda
salida. `b` se queda igual. Y el argumento para el cambio no es teórico —
es que α tiene rango medible (0.385–1.010) donde `f_opt` es una constante,
y que la ley que lo conecta con la decisión ajusta con R² ≥ 0.978 sobre
tus propios datos.

**Lo que hay que discutir con el director:** que su idea del doble target
sigue en pie; lo que cambia es que la segunda salida describe la *física de
la carga* en vez de la *decisión*, y que de esa física la decisión se
deduce. Es más cercano a Guerreiro y Ali —los dos antecedentes más fuertes
del anteproyecto— que la formulación directa.

---

## 9. Lo que falta verificar antes de cerrar la elección

| Pregunta | Bloquea | ¿Necesita `paccaA100`? |
|---|---|---|
| ¿Varía el óptimo entre fases de un mismo kernel? | A, B, C | No |
| ¿Ajusta la ley de escalado **por fase**, no solo por corrida? | C | No |
| ¿Son las instrucciones invariantes a la frecuencia? (compuerta 1) | A, B, C | No |
| ¿Aparecen óptimos interiores con la rejilla fina? | A, B | Sí (job 6391 en cola) |
| ¿Se puede recuperar el 93 % descartado de GPU? | todo el aporte GPU | Sí |
