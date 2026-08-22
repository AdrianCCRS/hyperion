# Resultados de las compuertas de Fase 2

Trabajo del 2026-08-21 al 22. Todo medido sobre la campaña reprocesada
`pacca_cpu_final_attempt03_20260820_arc174` (540 corridas completas) y
`pacca_gpu_dvfs_20260820`.

Resumen: **las dos compuertas que podían invalidar el proyecto pasaron**, y
la pregunta que decidía entre las opciones de diseño quedó respondida — con
una respuesta que descarta dos de las cuatro.

---

## 1. Compuerta 0 — ¿la fase predice el escalado con la frecuencia? PASA

La premisa central de la tesis es que clasificar la fase sirve para decidir
la frecuencia. Eso exige que un kernel memory_bound se alargue *poco* al
bajar el reloj y uno compute_bound *mucho*. Medido:

| kernel | b (0=compute, 1=memory) | T(800 MHz) / T(3200 MHz) |
|---|---:|---:|
| rodinia_lavamd_omp | 0.190 | 4.04× |
| rajaperf_polybench_3mm_omp | 0.317 | 4.05× |
| dgemm_n2048 | 0.404 | 3.82× |
| npb_bt | 0.534 | 3.72× |
| npb_lu | 0.611 | 3.53× |
| npb_ft | 0.695 | 3.34× |
| npb_mg | 0.796 | 2.21× |
| npb_sp | 0.799 | 2.49× |
| npb_cg | 0.845 | 3.32× |

**Pearson(b, estiramiento) = −0.817** (n=9, p≈0.007). La relación existe,
es fuerte y tiene el signo correcto.

**Matiz importante:** el efecto es fuerte en dirección pero **débil en
magnitud**. Si la frecuencia baja 4×, un kernel puramente compute-bound
debería alargarse 4× y uno puramente memory-bound 1× (gratis). El rango
observado es 2.21×–4.05×: ni el más memory-bound se acerca a "bajar el
reloj me sale gratis". Bajar el reloj del núcleo también frena el acceso a
memoria (menos peticiones en vuelo, prefetch más lento).

Eso explica por qué en CPU no aparecen ahorros: la señal está, pero es
pequeña, y por eso hace falta **resolución fina de frecuencia** para
encontrar el punto dulce — que es exactamente la campaña 6391 en cola.

## 2. Compuerta 1 — ¿son las instrucciones invariantes a la frecuencia? PASA

Es lo que permite alinear "el mismo momento del programa" entre corridas a
distinto reloj.

- Peor desviación del conteo total entre niveles: **0.34 %** (`npb_lu`).
- La mayoría por debajo de **0.1 %**.
- Criterio del plan: ±2 %.

`delta_instructions` queda validada como coordenada de progreso para CPU.
En GPU no existe (filas passthrough, ARC-70) y hay que usar fracción de
tiempo, que es más débil.

---

## 3. La pregunta que decidía el diseño: ¿varía el óptimo entre fases?

Alineando por instrucciones, 20 tramos por corrida, 10 repeticiones, 5
niveles de frecuencia fija = **200 celdas comparables por kernel**:

| kernel | distribución del óptimo por tramo | niveles distintos |
|---|---|---:|
| npb_bt | F0: 100 % | 1 |
| npb_sp | F0: 100 % | 1 |
| npb_lu | F0: 100 % | 1 |
| dgemm_n2048 | F0: 100 % | 1 |
| rodinia_lavamd_omp | F0: 100 % | 1 |
| rajaperf_polybench_3mm_omp | F0: 100 % | 1 |
| npb_ft | F0: 99.5 % | 2 |
| npb_cg | F0: 99 % | 2 |
| **npb_mg** | **F0: 58 % · F1: 42 %** | **2** |

**El óptimo NO varía entre fases.** 8 de 9 kernels tienen ≥99 % de sus
tramos prefiriendo el mismo nivel. La única excepción es `npb_mg` — que es
también el único kernel con ahorro energético medible (−3.6 % en F1).

### Consecuencia directa sobre las opciones de diseño

Esto **descarta las opciones A y B** de `opciones_modelo_fase2.md` en CPU:
predecir "la frecuencia óptima por fase" es predecir una constante por
kernel. No hay política dinámica intra-kernel que aprender con la rejilla
actual.

## 4. ¿Y α? Varía entre kernels, no dentro

| medida | valor |
|---|---|
| rango INTER-kernel de α | 0.384 – 1.026 (amplitud **0.642**) |
| desviación típica INTRA-kernel | 0.004 – 0.056 |
| ajuste de la ley de escalado (R² medio) | 0.976 – 0.9998 |

α es esencialmente **una propiedad del workload, no de la fase**. La ley
`T(f)/T(f_ref) = (1−α) + α·(f_ref/f)` ajusta con R² ≥ 0.976 en los 9
kernels, ahora sí con los datos completos.

**Consecuencia:** la opción C sigue en pie y sigue siendo la que tiene
señal —α recorre 0.642 de rango donde el óptimo es constante— pero su
aporte es **caracterizar la carga**, no decidir por fase. Es un cambio de
alcance honesto respecto a como la presenté.

---

## 5. Diagnóstico del 7 % de GPU: no es lo que dije, y puede que no necesite `paccaA100`

Yo había atribuido el problema a `gpu_util_pct = 0`. **Es otra cosa:**

| kernel | ventanas | util = NaN | util = 0 | util ≥ 5 |
|---|---:|---:|---:|---:|
| gpu_dgemm_n4096 | 8 899 | 80.9 % | 3.5 % | 14.5 % |
| rodinia_gaussian | 19 208 | 84.9 % | 1.6 % | 13.3 % |
| rodinia_backprop | 3 942 | 85.3 % | 7.9 % | 0.3 % |
| rodinia_lavamd | 20 140 | 85.5 % | 10.5 % | 3.1 % |
| rodinia_heartwall | 13 591 | 85.0 % | 2.2 % | 12.6 % |
| rodinia_lud | 33 821 | 85.0 % | 14.1 % | 0.3 % |
| rodinia_myocyte | 36 041 | 85.3 % | 9.4 % | 2.2 % |
| rodinia_dwt2d | 19 438 | 85.5 % | 4.4 % | 4.2 % |
| **TOTAL** | **155 080** | **85.0 %** | **8.0 %** | **5.1 %** |

**El 85 % de las ventanas no tiene lectura de GPU en absoluto** (NaN), no
un cero. La causa es una diferencia de cadencia: la ventana mediana dura
**0.26 ms** y NVML está configurado a `gpu_interval_ns = 100 ms`. La GPU
produce ~10 muestras por segundo mientras las ventanas cierran a ~4000 por
segundo, así que el 99 % de las ventanas no puede tener una lectura fresca.

**No es un bug: es un desajuste de diseño.** Y tiene un arreglo que **no
requiere recolectar de nuevo**: analizar la GPU a la granularidad de su
propia telemetría (agregando ventanas a ~100 ms) en vez de a la del
muestreo de CPU. El dataset actual ya contiene esa información.

Queda un segundo problema separado: de las ventanas que **sí** tienen
lectura, aproximadamente la mitad reporta `util < 5`. En `rodinia_lud` los
únicos valores observados son {0, 1, 6} mientras la GPU consume 61 W de
media sobre un reposo de ~35 W — está trabajando. Ahí `gpu_util_pct` no es
un buen indicador de actividad, y **la potencia sobre el reposo lo sería
mejor**. Eso también es analizable offline.

**Revisión de prioridad:** yo había marcado esto como bloqueador crítico
que exigía `paccaA100`. Con este diagnóstico, la mayor parte se puede
resolver con los datos existentes.

---

## 5.bis El resultado más serio: el clasificador NO supera a la línea base trivial

Entrenado sobre 1 011 907 ventanas (submuestreo de 2000 por corrida), 7
features baratas —`ipc`, `mpki`, `llc_miss_rate`, `stall_backend_ratio`,
`ips`, `running_ratio`, `freq_khz_observed`— y validación LOKO estricta:

| modelo | F1 macro | sd | peor pliegue | p50 latencia |
|---|---:|---:|---:|---:|
| regresión logística | 0.393 | 0.105 | 0.150 (npb_ft) | 345 µs |
| árbol prof. 6 | 0.390 | 0.076 | 0.228 (dgemm) | 103 µs |
| árbol prof. 1 | 0.387 | 0.290 | 0.000 (dgemm) | 103 µs |
| **mayoritaria (trivial)** | **0.371** | 0.305 | 0.000 (dgemm) | 10 µs |
| extra trees | 0.368 | 0.195 | 0.017 (dgemm) | 33 553 µs |
| random forest | 0.358 | 0.183 | 0.001 (dgemm) | 33 251 µs |

**Ningún modelo aprendido supera de forma significativa al predictor
trivial**, y los dos ensembles quedan *por debajo*. Un árbol de
profundidad 1 iguala a un random forest de 100 árboles.

### La causa: el dataset no tiene variación de fase intra-kernel

Fracción de ventanas de la **clase minoritaria** dentro de cada kernel:

| kernel | REF | F0 | F1 | F2 | F3 | F4 | global |
|---|---:|---:|---:|---:|---:|---:|---:|
| npb_mg | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0 %** |
| npb_cg | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0 %** |
| dgemm_n2048 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0 %** |
| rodinia_lavamd_omp | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.1 % |
| npb_ft | 0.0 | 0.0 | 0.9 | 0.1 | 0.3 | 1.0 | 0.5 % |
| npb_sp | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.4 | 1.0 % |
| rajaperf_polybench_3mm_omp | 5.7 | 5.6 | 4.8 | 3.7 | 2.5 | 1.7 | 3.2 % |
| npb_bt | 0.6 | 0.8 | 19.4 | 21.5 | 14.0 | 10.2 | 11.8 % |
| npb_lu | 0.0 | 0.0 | 0.0 | 8.5 | 34.4 | 30.9 | 19.0 % |

**Media: 4.0 %.** Cada kernel es prácticamente de una sola clase de
principio a fin. Y donde sí hay mezcla (`npb_bt`, `npb_lu`) aparece **solo
a frecuencias bajas**: es el ridge desplazándose y cruzándolos (§ANEXO
ridge), no fases que se alternen durante la ejecución.

El equilibrio 50/50 que mostró el EDA es **entre kernels**, no dentro de
ellos.

Esto explica el resultado por completo: LOKO deja fuera un kernel que es
100 % de una clase, y el modelo —entrenado con 8 kernels mayoritariamente
de la otra— no tiene nada en qué apoyarse. `dgemm_n2048` (100 % compute)
saca F1 = 0.000 en casi todos los modelos.

### Y sin embargo, el target continuo del director sí gana al binario

Variación de `b` **dentro** de cada corrida (a REF):

| kernel | b medio | sd intra | rango p10–p90 |
|---|---:|---:|---:|
| npb_bt | 0.578 | 0.093 | 0.252 |
| npb_lu | 0.651 | 0.081 | 0.207 |
| rajaperf_polybench_3mm_omp | 0.340 | 0.062 | 0.124 |
| npb_sp | 0.777 | 0.059 | 0.149 |
| npb_ft | 0.721 | 0.052 | 0.133 |
| npb_cg | 0.852 | 0.029 | 0.006 |
| rodinia_lavamd_omp | 0.193 | 0.019 | 0.023 |
| npb_mg | 0.792 | 0.015 | 0.040 |
| dgemm_n2048 | 0.402 | 0.015 | 0.032 |

`npb_sp` tiene la etiqueta binaria constante (100 % memory) pero su `b`
recorre 0.149. **La etiqueta binaria tira esa estructura; el score continuo
la conserva.** Es un argumento medido a favor del 0–1 que propuso el
director, independientemente de lo que se decida sobre el segundo target.

La magnitud, eso sí, es modesta: sd intra media 0.047 contra un rango
inter-kernel de ~0.65 — la variación dentro de una corrida es ~7 % de la
que hay entre kernels.

### Qué implica esto para el anteproyecto

La §5.1 eligió deliberadamente *"benchmarks representativos de cuatro
escenarios base: CPU compute-bound, CPU memory-bound, GPU compute-bound y
GPU memory-bound"* — es decir, ejemplos **puros** de cada régimen. El
dataset cumple ese diseño al pie de la letra.

Pero la §4.2 sitúa el vacío que la tesis ataca en *"la adaptación fina a la
multifasicidad intra-ejecución de aplicaciones HPC generales"*. **Los
benchmarks elegidos no tienen multifasicidad.** Hay una distancia real
entre el vacío declarado y lo que el dataset puede demostrar, y hay que
decidir qué hacer con ella:

1. **Reformular el alcance:** el agente clasifica el *régimen de la carga*
   (no fases intra-ejecución) y fija la frecuencia en consecuencia. Honesto
   y alcanzable, pero más estrecho que el vacío declarado.
2. **Añadir kernels multifásicos:** benchmarks que alternen de verdad entre
   cómputo y memoria durante la ejecución. Requiere campaña nueva, pero es
   lo que cumpliría lo prometido en §4.2.
3. **Las dos**, con lo actual como base y una campaña dirigida encima.

Es una decisión de alcance, no técnica, y por eso queda para discutir.

---

## 6. Código construido

| módulo | qué hace | pruebas |
|---|---|---:|
| `classifier/features/align.py` | progreso invariante a frecuencia, binning, `fit_alpha` | 12 |
| `classifier/features/targets.py` | score continuo `b` con ridge por fila | 12 |
| `classifier/eval/protocol.py` | LOKO, guardarraíl anti-fuga, EDP loss, líneas base | 14 |
| `classifier/features/load.py` | carga y filtrado de `windows.csv` | 6 |

`protocol.py` se escribió **antes** que cualquier entrenamiento, a
propósito, para que ningún número del trabajo nazca fuera del protocolo.

### Un error corregido a mitad de camino

`windows.csv` trae `repetition = 1` **siempre** (cada corrida se lanza con
`repetitions=1` y su índice real vive en el nombre del directorio). La
primera versión del análisis agrupó por esa columna, fusionando las 10
repeticiones en una pseudo-corrida: el progreso acumulado cruzaba de una
repetición a la siguiente, así que los "tramos" no eran posiciones del
programa sino trozos de repeticiones distintas. Los resultados de la §3 y
§4 son de la versión corregida.

Además, tres sondas tempranas (`scaling_law`, `gate1`) leyeron del
directorio crudo `..._20260820` en vez del reprocesado `..._arc174`. El
crudo solo tiene `windows.csv` para parte de las corridas, así que
`rodinia_lavamd_omp` desapareció del ajuste y `npb_cg` quedó con 4 puntos
de 6. Los números de α de este documento son los del directorio correcto.

---

## 7. Estado de las decisiones

| Pregunta | Estado |
|---|---|
| ¿La fase predice el escalado? | **Sí**, r = −0.82 |
| ¿Instrucciones invariantes? | **Sí**, 0.34 % peor caso |
| ¿Varía el óptimo entre fases? | **No** (8/9 al 100 %; solo `npb_mg`) |
| ¿Tiene α señal? | Sí entre kernels (0.642), poca dentro |
| ¿Aparecen óptimos con rejilla fina? | Pendiente — job 6391 en cola |
| ¿Se recupera el dataset de GPU? | Probablemente sí, y **offline** |
| ¿El clasificador supera a la línea base? | **No** bajo LOKO (0.39 vs 0.371) |
| ¿Hay fases intra-kernel que clasificar? | **No** — clase minoritaria 4.0 % |

**Recomendación revisada.** El orden que propuse ayer (D primero, C después)
se sostiene, pero D **no es el entregable seguro que yo creía**: bajo LOKO
el clasificador no supera al predictor trivial, y no por culpa del modelo
sino porque el dataset no contiene el fenómeno que se pretende clasificar.

Lo que queda en pie, medido:

- El score continuo `b` conserva estructura que la etiqueta binaria
  destruye (§5.bis). Ese cambio se justifica solo.
- α caracteriza la carga con R² ≥ 0.976 y recorre 0.642 entre kernels.
  Es predecible y útil, como caracterización de workload.
- Las opciones A y B quedan descartadas en CPU por ausencia de variación
  intra-kernel del óptimo.

Lo que hay que decidir antes de seguir programando es de **alcance**: si el
trabajo clasifica regímenes de carga (lo que el dataset soporta) o fases
intra-ejecución (lo que el §4.2 declara como vacío y exigiría kernels
multifásicos nuevos).
