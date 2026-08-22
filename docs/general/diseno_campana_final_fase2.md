# Diseño de la campaña final — Fase 2

Matriz experimental completa del nuevo intento, con la justificación de cada
nivel de frecuencia y de cada bloque. Sustituye al diseño de la campaña 6391
(cancelada) y extiende al de la 6412 (encolada).

Fecha: 2026-08-22. Rama: `fase-02`. Serie de cambios: ARC-176.

---

## 1. El concepto que sostiene todo el diseño: α

Cuando un programa corre, su tiempo se reparte en dos tipos de trabajo:

1. **Trabajo sensible al reloj** — ejecutar instrucciones, hacer aritmética.
   Si el reloj sube, va más rápido.
2. **Trabajo insensible al reloj** — esperar un dato desde DRAM. Esa espera
   son ~80 ns de viaje físico y no cambia porque el núcleo vaya a 3200 o a
   800 MHz.

**α es la fracción del tiempo que es del primer tipo.** La ley que lo
formaliza tiene la misma estructura que la de Amdahl, cambiando
"paralelizable" por "sensible al reloj":

```
T(f) / T(f_ref)  =  (1 − α)  +  α · (f_ref / f)
```

| α | Significado | Al bajar el reloj ×2 |
|---:|---|---|
| 1.0 | todo el tiempo depende del reloj | el tiempo se duplica |
| 0.5 | mitad y mitad | el tiempo crece +50 % |
| 0.0 | nada depende del reloj | el tiempo no cambia |

Se mide corriendo el mismo kernel a varias frecuencias y ajustando la recta.
Sobre los 9 kernels del dataset ajusta con **R² entre 0.976 y 0.9998**: un
solo número reconstruye la curva de tiempo completa con 1–2 % de error.

### 1.1 Por qué α decide si DVFS sirve

Bajando el reloj de 3200 a 2600 MHz en paccaA100:

- la **potencia** cae a 0.903× (medido: 116.5 W → 105.2 W),
- el **tiempo** crece a (1 + 0.231·α)×,
- energía = potencia × tiempo, EDP = energía × tiempo.

Para que el EDP mejore hace falta `0.903 · (1 + 0.231α)² < 1`, es decir
**α < 0.226**. No es un número arbitrario: es dónde se cruzan una potencia
que solo cae un 10 % y un tiempo que crece proporcional a α.

**Los 9 kernels del dataset tienen α ∈ [0.384, 1.026].** Todos por encima
del umbral. Por eso ninguno se beneficia de DVFS: no falta rejilla ni falta
modelo, faltan cargas por debajo de 0.224.

### 1.2 α no es la etiqueta compute/memory

La etiqueta Roofline (`b`) dice **qué techo te limita en principio**,
comparando intensidad aritmética contra el ridge. α dice **cuánto te afecta
de verdad el reloj**, medido. Correlacionan (−0.82) pero no son lo mismo.

`npb_bt` es el contraejemplo: etiquetado `memory_bound` en REF (99.4 % de
sus ventanas) y sin embargo **α = 0.902**. Está por debajo del ridge pero no
satura el ancho de banda — lo limita la latencia o las dependencias, y eso
sí escala con el reloj.

Para decidir frecuencia, **α es lo que hace falta**; `b` es un estimador de
α que a veces falla. Esa distinción es la que justifica que la segunda
salida del modelo sea α y no la frecuencia óptima directa.

---

## 2. La rejilla de frecuencias

### 2.1 CPU — 10 niveles

Rango real del nodo con Turbo deshabilitado: [800 000, 3 200 000] kHz.
`fraction` se resuelve como `round(f_min + fraction·(f_max − f_min))`
(`freqctl.py:157`).

| id | MHz | `fraction` | Por qué está |
|---|---:|---:|---|
| `REF` | — | `native_governor` | Línea base contra la que el agente compite |
| `F0` | 3200 | 1.0 | Óptimo EDP de α ≥ 0.30 (los 9 reales). Ancla de deriva |
| `S3000` | 3000 | 0.9166666667 | **Presupuesto 5 % de slowdown para α ≥ 0.76** (7 kernels) |
| `S2800` | 2800 | 0.8333333333 | Presupuesto 5 % de α ≈ 0.38–0.49; presupuesto 10 % de α ≥ 0.76 |
| `F1` | 2600 | 0.75 | Óptimo de **energía** de α ≈ 0.30–0.46 |
| `F2` | 2000 | 0.5 | Óptimo EDP de α ≈ 0.10 |
| `F3` | 1400 | 0.25 | Óptimo EDP de α ≈ 0.05 |
| `S1200` | 1200 | 0.1666666667 | Banda α ∈ (0, 0.05), donde el óptimo salta de 800 a 1400 |
| `S1000` | 1000 | 0.0833333333 | Íd. |
| `F4` | 800 | 0.0 | Óptimo de α ≈ 0: −28.4 % de EDP **sin slowdown** |

**Los dos extremos de resolución responden preguntas distintas**, y las dos
están en el anteproyecto:

- **Arriba (3000, 2800):** *energía mínima sujeta a una degradación
  tolerable* — el objetivo general ("sin degradar significativamente el
  rendimiento").
- **Abajo (1200, 1000):** *EDP mínimo sin restricción* — §5.4.

### 2.2 Por qué la banda 2600–3200 es obligatoria

La frecuencia que produce exactamente un 5 % de slowdown es
`f = 3200/(1 + 0.05/α)`. Con el α medido de cada kernel real:

| kernel | α | f al 5 % | f al 10 % |
|---|---:|---:|---:|
| npb_mg | 0.384 | 2831 | 2539 |
| npb_sp | 0.491 | 2904 | 2659 |
| npb_cg | 0.761 | 3003 | 2828 |
| npb_ft | 0.774 | 3006 | 2835 |
| npb_lu | 0.840 | 3020 | 2864 |
| npb_bt | 0.902 | 3032 | 2889 |
| dgemm_n2048 | 0.931 | 3037 | 2899 |
| rajaperf_polybench_3mm_omp | 1.007 | 3048 | 2911 |
| rodinia_lavamd_omp | 1.026 | 3051 | 2916 |

**Los nueve caen en 2831–3051 MHz**, y con presupuesto del 10 % ocho de los
nueve siguen dentro del hueco (2659–2916).

Eso explica con precisión por qué los presupuestos de 5 %, 10 % y 20 %
devolvían **exactamente el mismo resultado** en el análisis: el único nivel
alcanzable dentro de cualquiera de ellos era 3200 MHz. No es que no hubiera
ahorro — es que no había ningún punto medido donde buscarlo.

> **Corrección de un razonamiento previo.** Se había concluido que "la
> resolución hace falta abajo" a partir de la tabla de óptimos, que está
> calculada minimizando EDP. Para EDP puro es cierto. Pero para *energía con
> restricción* el razonamiento se invierte: no se busca el mínimo global sino
> el mejor punto dentro del presupuesto, y ese vive pegado a 3200. Se aplicó
> la conclusión de un objetivo al otro.

### 2.3 GPU — 8 niveles

Se conservan los 6 de la campaña existente (comparabilidad con las 288
corridas ya medidas) y se añaden 2.

| id | SM clock | Nota |
|---|---:|---|
| `REF` | 1410 | Gobernador nativo (baja a 1160 en `gpu_dgemm` por techo de potencia) |
| `F0` | 1410 | Máximo |
| `F1` | 1110 | **Óptimo EDP de `rodinia_lavamd`** (−20.6 %) |
| `F2` | 810 | |
| `S660` | 660 | **Nuevo** — parte el salto 510→810 |
| `F3` | 510 | |
| `S360` | 360 | **Nuevo** — parte el salto 210→510 |
| `F4` | 210 | **Óptimo EDP de `rodinia_lud`** (−30.5 %, con +10.9 % de tiempo) |

Motivo de los dos nuevos: `rodinia_lud` optimiza en **el borde inferior**
de la rejilla. Un óptimo en el borde suele significar que el verdadero está
fuera o muy cerca del límite, así que no sabemos si 210 MHz es realmente el
mejor o solo el más bajo que se probó. Es el único borde de rejilla que
quedó sin acotar en todo el trabajo. El A100 acepta relojes en pasos de
~15 MHz vía `nvidia-smi -lgc`, así que ambos son alcanzables.

---

## 3. La matriz experimental

| Bloque | Qué | Kernels | Niveles | Reps | Corridas | Qué responde |
|---|---|---:|---:|---:|---:|---|
| **A** | Completar rejilla de los reales | 9 | 5 | 10 | **450** | Energía con restricción de slowdown; deja los 540 existentes en la rejilla completa |
| **B** | Microbenchmarks puros (eje α) | 3 | 8 | 10 | **240** | ¿Es alcanzable α ≤ 0.224? Ancla el modelo de α de ~0 a ~1 |
| **C** | Multifásicos CPU | 6 | 8 | 10 | **480** | Detección de fases, experimento dinámico, curva de amortización |
| **D** | GPU multifásico | 3 | 8 | 3 | **72** | Lo mismo en el acelerador (§5.3) |
| | | | | | **1242** | |

De esas, **la campaña 6412 (encolada) ya cubre 320**: `ptrchase` completo en
8 niveles × 10 reps, y `phasic` C↔L en sus 3 periodos. **Quedan 922.**

### 3.1 Bloque A — no se relanza lo que ya sirve

Los 9 kernels reales ya están medidos en REF, F0, F1, F2, F3 y F4. Como
todos tienen α ≥ 0.384, su óptimo de EDP es 3200 MHz pase lo que pase, así
que **repetir esos niveles no aportaría nada**. Solo se corren los que
faltan:

`F0` (ancla de deriva) · `S3000` · `S2800` · `S1200` · `S1000`

El ancla no es opcional: entre campañas hay hasta **1.66 % de deriva de
energía** medida (misma configuración, mismo nodo, job de Slurm distinto),
y los efectos que se buscan son de ~3 %. Sin ancla, un ahorro aparente del
3 % podría ser 1.5 % o 4.5 %.

### 3.2 Bloque B — el eje α

Tres microbenchmarks que barren el rango completo:

| kernel | patrón | α esperado |
|---|---|---:|
| `ptrchase` | persecución de punteros, limitado por **latencia** | ~0 |
| `stream_bench` | acceso secuencial, limitado por **ancho de banda** | intermedio |
| `fma_bench` | FMA encadenadas sobre registros, sin memoria | ~1 |

Sin ellos el modelo de α extrapola fuera del rango donde tiene datos: los 9
reales solo cubren [0.384, 1.026] y la zona interesante está *debajo* de eso.

### 3.3 Bloque C — dos pares de fase, tres periodos

| par | fases | contraste de α | papel |
|---|---|---|---|
| **C↔L** | cómputo ↔ latencia | máximo (~1.0 vs ~0) | caso fácil; ya en 6412 |
| **C↔B** | cómputo ↔ ancho de banda | menor | caso realista y difícil |

Cada par a **3 periodos de fase: 10 ms, 100 ms, 1 s**. Ese barrido es lo que
produce la curva de amortización — a partir de qué frecuencia de cambio el
costo de conmutar se come la ganancia, que es la restricción que la §4.2
señala citando a Velicka et al.

**Detalle de diseño:** la fase se mide por **tiempo**, no por número de
iteraciones. Con iteraciones fijas, bajar la frecuencia alargaría la fase de
cómputo más que la de memoria, cambiando la proporción entre fases entre
niveles y arruinando la comparación.

Ambos kernels imprimen **etiqueta de verdad**: el instante exacto de cada
transición.

### 3.4 Bloque D — GPU

Un kernel CUDA multifásico (alterna FMA y accesos con zancada grande) a los
mismos 3 periodos. La GPU ya tiene óptimos interiores medidos (2/8 en EDP,
7/8 en energía); lo que le falta es variación intra-corrida. Solo 72
corridas.

---

## 4. Protocolo de validación

Con familias sintéticas hay un problema de fuga que hay que resolver
explícitamente: `phasic_p010`, `p100` y `p1000` son **el mismo código** con
distinto parámetro. Hacer leave-one-out entre ellos sería probar contra sí
mismo.

La solución es más limpia y además más fuerte:

| Qué se valida | Cómo | Qué demuestra |
|---|---|---|
| Predicción de α y de régimen | **LOKO entre los 9 kernels reales** | Generaliza a cargas reales no vistas |
| Detección de fases | **Entrenar con los reales → probar con los sintéticos** | Un modelo que solo vio cargas homogéneas detecta transiciones que nunca vio |
| Latencia de detección | **Contra la etiqueta de verdad de `phasic`** | Cuántas ventanas tarda en notar cada transición |
| Política DVFS | Frecuencia fija óptima vs. gobernador vs. **oráculo por fase** vs. modelo | Si conmutar dinámicamente gana algo |

El segundo es el más valioso: es exactamente el escenario de despliegue —
el agente se encuentra cargas que no estaban en su entrenamiento.

---

## 5. Costo y estructura de jobs

| Job | Bloques | Corridas | Estimado | Depende de 6412 |
|---|---|---:|---:|---|
| 6412 (encolado) | parte de B y C | 320 | ~4 h | — |
| Job A | A | 450 | ~3.8 h | **No** |
| Job BCD | resto de B, C, D | 472 | ~5.3 h | **Sí** |

**El job A se puede encolar ya**: completa la rejilla de los kernels reales
y su valor no depende de qué salga en 6412.

**El job BCD espera** porque si `ptrchase` da α > 0.224, el par C↔L pierde
contraste de frecuencia óptima y el bloque C cambia de forma — habría que
reponderar hacia GPU, donde los óptimos interiores ya están medidos.

---

## 6. Decisiones abiertas

1. **¿Rejilla uniforme o irregular entre bloques?** El diseño de arriba usa
   10 niveles en el bloque A y 8 en B/C, porque los α de B y C están en los
   extremos y los puntos de 3000/2800 no serían óptimos nunca. Uniformar a
   10 en todo cuesta **~180 corridas más** y simplifica comparar bloques a
   frecuencia fija.
2. **¿Se aceptan los dos niveles nuevos de GPU (660, 360)?** Cuesta 18
   corridas y cierra el único borde de rejilla sin acotar del trabajo.
3. **¿Se encola ya el job A?** No depende de 6412.
