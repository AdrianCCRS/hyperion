# Estado consolidado y decisiones pendientes — Fase 2

Reúne todo lo medido, todos los problemas abiertos y todas las decisiones
que hay que tomar. Sustituye a la lectura dispersa de
`plan_fase2.md`, `opciones_modelo_fase2.md` y
`resultados_compuertas_fase2.md`, que quedan como detalle de respaldo.

Fecha: 2026-08-22. Rama: `fase-02`.

---

## 1. Dónde estamos en una frase

La Fase 1 entregó un dataset sólido (546/546 corridas aceptadas, 97.1 % de
ventanas de CPU utilizables), y al empezar a diseñar el modelo de Fase 2
aparecieron tres hechos que **no invalidan el trabajo hecho pero sí obligan
a redefinir qué puede demostrar la Fase 2**.

---

## 2. Todo lo medido, en un sitio

### 2.1 Lo que salió bien

| Verificación | Resultado |
|---|---|
| Dataset CPU utilizable | 9 953 976 / 10 251 941 ventanas (**97.1 %**) |
| Distribución global de fase | 50.2 % memory / 49.8 % compute |
| **Compuerta 0** — ¿la fase predice el escalado con la frecuencia? | **PASA**, Pearson −0.82 (n=9, p≈0.007) |
| **Compuerta 1** — ¿instrucciones invariantes a la frecuencia? | **PASA**, 0.34 % peor caso vs. criterio ±2 % |
| Ley de escalado `T(f)/T(ref) = (1−α) + α·(f_ref/f)` | R² **0.976 – 0.9998** en los 9 kernels |
| α (fracción de tiempo sensible al reloj) | rango inter-kernel **0.384 – 1.026** |
| Ahorro de energía en GPU con ≤10 % slowdown | **+7.4 %** medio, hasta **+27.5 %** |
| Ahorro de energía en GPU con ≤20 % slowdown | **+9.5 %** medio, hasta **+37.4 %** |

### 2.2 Lo que salió mal

| Verificación | Resultado |
|---|---|
| Óptimo de EDP en CPU fuera de la frecuencia máxima | **0 de 9 kernels** |
| Ahorro de energía en CPU (cualquier presupuesto) | **+0.7 %** medio; solo `npb_mg` (−3.6 % en F1) |
| ¿Varía el óptimo entre fases de un mismo kernel? | **No** — 8/9 con ≥99 % de tramos en un solo nivel |
| Clase minoritaria dentro de cada kernel | **4.0 %** de media; 4 kernels al **0.0 %** |
| Clasificador de fase bajo LOKO (F1 macro) | **0.393** vs. **0.371** del predictor trivial |
| Random forest bajo LOKO | **0.358** — *peor* que el trivial |
| Ventanas de GPU utilizables | 160 142 / 2 171 803 (**7.4 %**) |

### 2.3 Física del nodo que explica lo anterior

| Medida | Valor |
|---|---|
| Potencia a 3200 MHz | 107 – 143 W |
| Potencia a 800 MHz | 80 – 89 W |
| Reducción de potencia al bajar el reloj 4× | solo **28 %** |
| Estiramiento de tiempo observado (800 vs 3200 MHz) | 2.21× – 4.05× |
| Estiramiento teórico si fuera memory-bound puro | 1.00× |
| Desplazamiento del ridge del Roofline (3200 → 800 MHz) | 8.733 → 2.992 FLOP/byte (×0.34) |
| Cores usados / cores del nodo | 6 / 16 (Xeon Gold 5315Y, 2 sockets) |

El piso de potencia estática es tan alto que alargar la ejecución siempre
cuesta más energía de la que ahorra bajar el reloj. Y ni el kernel más
memory-bound se acerca a "bajar el reloj sale gratis": bajar el reloj del
núcleo también frena el acceso a memoria.

---

## 3. Los problemas, ordenados por cuánto amenazan la tesis

### P1 · El dataset no contiene el fenómeno que la tesis quiere detectar

**Gravedad: alta. Es el problema de fondo.**

Cada benchmark es de un solo régimen de principio a fin (clase minoritaria
4.0 % de media, cuatro kernels al 0.0 %). El equilibrio 50/50 que mostró el
EDA es **entre** kernels, no dentro de ellos. Donde hay mezcla (`npb_bt`
11.8 %, `npb_lu` 19.0 %) aparece solo a frecuencias bajas: es el ridge
desplazándose y cruzándolos, no fases que se alternen.

Consecuencia directa: bajo LOKO el clasificador no supera al predictor
trivial, y no por culpa del modelo. Se le pide distinguir fases que no
existen dentro de las corridas.

No es un defecto de ejecución: la §5.1 del anteproyecto eligió
deliberadamente *"benchmarks representativos de cuatro escenarios base"*,
o sea ejemplos **puros**. El dataset cumple ese diseño al pie de la letra.
El problema es que la §4.2 sitúa el vacío a atacar en *"la adaptación fina
a la multifasicidad intra-ejecución"* — y los benchmarks elegidos no tienen
multifasicidad.

### P2 · En CPU no hay beneficio DVFS medible

**Gravedad: media. Pendiente de resolver.**

0 de 9 kernels tienen óptimo interior; el ahorro medio bajo cualquier
presupuesto es 0.7 %. Pero eso es **en parte artefacto de la rejilla**: el
salto F0→F1 es de 3200 a 2600 MHz (18.75 %), que ya implica 14–23 % de
slowdown, así que los presupuestos de 5 %, 10 % y 20 % devuelven el mismo
resultado — no hay nivel intermedio que probar.

**Job 6391 en cola** cubre ese hueco con 7 niveles nuevos (3100, 3000,
2900, 2800, 2400, 2200 MHz + ancla). Hasta que corra, esto no está cerrado.

### P3 · Solo 9 kernels de CPU y 8 de GPU

**Gravedad: media. Estructural, ya asumida.**

Obliga a LOKO como único protocolo honesto y limita de raíz la capacidad de
generalizar. Con 9 pliegues, un kernel que se desploma mueve la media
entera. No tiene arreglo dentro del alcance del trabajo; se declara.

### P4 · El 93 % de las ventanas de GPU no es utilizable

**Gravedad: media-baja. Arreglable sin recolectar de nuevo.**

El 85 % de las ventanas **no tiene lectura de GPU** (NaN), no un cero: la
ventana dura 0.26 ms y NVML está configurado a 100 ms. Es un desajuste de
cadencia, no un fallo de instrumentación.

Se arregla **offline**, agregando ventanas a la granularidad de la
telemetría de GPU. Yo lo había marcado como bloqueador crítico que exigía
`paccaA100`; ya no lo es.

Queda un segundo asunto: de las ventanas que sí tienen lectura, la mitad
reporta `util < 5`. En `rodinia_lud` los únicos valores son {0, 1, 6}
mientras la GPU consume 61 W sobre un reposo de 35 W — está trabajando.
`gpu_util_pct` no es buen indicador de actividad ahí; la potencia sobre el
reposo lo sería mejor.

### P5 · La etiqueta de fase es relativa a la frecuencia

**Gravedad: baja. Entendido y manejable.**

El ridge cae de 8.733 a 2.992 FLOP/byte entre 3200 y 800 MHz, así que la
misma carga cambia de etiqueta sin cambiar de comportamiento (2 de 9
kernels lo hacen). No es un bug: el Roofline es *por configuración*. Para
el agente en línea juega a favor, porque siempre observa a la frecuencia
actual. Solo obliga a no propagar nunca la etiqueta de un nivel a otro.

---

## 4. Las decisiones

### D1 · Alcance: ¿qué demuestra la Fase 2?

**Esta es la decisión que bloquea a todas las demás.**

| Opción | Qué implica | Costo | Riesgo |
|---|---|---|---|
| **D1-a** Reformular: clasificar el *régimen de la carga*, no fases intra-ejecución | Reescribir la §4.2 para que el vacío declarado coincida con lo demostrable | Bajo — solo documento | El jurado puede leerlo como repliegue |
| **D1-b** Añadir un kernel sintético multifásico | Un benchmark que alterne cómputo y memoria con fases controlables. Demuestra que el agente sigue transiciones reales | Medio — ~100 líneas de C, catálogo, una campaña | Es un caso controlado, no una aplicación real |
| **D1-c** Añadir aplicaciones HPC reales multifásicas | Cumpliría el §4.2 literalmente | Alto — buscar, portar, validar, campaña | Puede no caber en el tiempo |
| **D1-d** Las dos: D1-a como marco honesto + D1-b como demostración | Los benchmarks reales dan la caracterización; el sintético demuestra la capacidad dinámica | Medio | — |

**Recomendación: D1-d.** El resultado se cuenta así: *"en benchmarks HPC
reales los regímenes son homogéneos — eso es un hallazgo, no un fallo — y
sobre una carga multifásica controlada el agente sigue las transiciones de
forma demostrable"*. Es honesto, cabe en el tiempo, y convierte P1 de
problema en resultado.

El kernel sintético es barato de verdad: un bucle de FMA encadenadas
(compute-bound) alternando con un recorrido de memoria de zancada grande
(memory-bound), con la duración de cada fase como parámetro. Encaja con la
"muestra intencional" que la §5.1 ya declara.

### D2 · ¿Qué predice el modelo?

Depende de D1, pero con lo medido ya se puede podar:

| Opción | Estado tras las mediciones |
|---|---|
| **A** · doble target con `f_opt` directo | **Descartada en CPU.** El óptimo no varía entre fases (8/9 al 100 %), así que el target es constante |
| **B** · sustituto de cocientes de EDP | **Descartada por lo mismo**, y además necesita más inferencias |
| **C** · predecir α (ley de escalado) | **Viva.** R² ≥ 0.976, rango 0.642 entre kernels. Pero es caracterización de *carga*, no de fase |
| **D** · solo clasificador de fase | **Viva pero debilitada.** No supera al trivial con el dataset actual (P1) |

**Recomendación: `b` continuo + α, ambos como regresión.** Mantiene la
arquitectura de doble target que propuso el director —RandomForest o
XGBoost multi-salida— y las dos salidas tienen señal medida:

- `b` conserva estructura que la etiqueta binaria destruye (`npb_sp` recorre
  0.149 en `b` con etiqueta constante al 100 % memory).
- α recorre 0.642 entre kernels y reconstruye la curva de tiempo completa.

Lo que cambia respecto a la idea original es solo **qué es la segunda
salida**: la física de la carga (α) en vez de la decisión (`f_opt`). De α
la frecuencia óptima se calcula, para cualquier objetivo, sin reentrenar.

### D3 · ¿Qué hacer si la rejilla fina tampoco muestra nada en CPU?

**Recomendación: reportarlo como caracterización y trasladar la actuación a
GPU.** Con la rejilla fina medida, la afirmación *"en este nodo el piso de
potencia estática hace que DVFS en CPU sea contraproducente"* pasa de
sospecha a resultado, y el papel del agente en CPU se vuelve **no empeorar
las cosas**: una heurística ingenua tipo "memory_bound → bajar frecuencia"
costaría hasta 12× más EDP en `rajaperf_polybench_3mm`. Demostrar que el
modelo aprende a *no* bajar donde una regla simple sí lo haría es un
resultado defendible.

### D4 · ¿Cómo se documenta el seguimiento de ahora en adelante?

Pendiente desde que empezamos la Fase 2 y sin resolver. Los ARC-xxx venían
numerando desviaciones del plan original de Fase 1. Opciones: seguir la
misma serie (ARC-175 en adelante, ya la usé en los commits de anoche),
abrir una serie nueva para Fase 2, o dejar de numerar y usar solo commits.

**Recomendación: seguir la misma serie.** Ya hay continuidad de hecho y
partirla complica rastrear cosas que cruzan fases (el ridge móvil afecta a
las dos).

---

## 5. Qué está en vuelo y qué se puede hacer ya

### En vuelo

| Qué | Estado |
|---|---|
| Job **6391** — campaña de rejilla fina de CPU | En cola (`PD`), 720 corridas, ~7-8 h, arranca solo |

### Desbloqueado, sin depender de nada

1. **Reagregar la GPU a la cadencia de NVML** (P4). Recupera el dataset de
   GPU, que es donde está el único ahorro energético real medido.
2. **Modelo de α** como caracterización de carga (D2), con LOKO.
3. **Regresión sobre `b` continuo** en vez de clasificación binaria — el
   cambio ya está justificado por medición.
4. **Escribir el kernel sintético multifásico** (D1-b), si se aprueba.

### Bloqueado

- Cualquier conclusión sobre DVFS en CPU, hasta que corra 6391.
- Recolección nueva de GPU, hasta que `paccaA100` se libere — pero ya
  sabemos que la mayor parte del problema no la necesita.

---

## 6. Código y estado del repositorio

| Módulo | Qué hace | Pruebas |
|---|---|---:|
| `classifier/features/load.py` | carga y filtrado de `windows.csv` | 6 |
| `classifier/features/align.py` | progreso invariante, binning, `fit_alpha` | 12 |
| `classifier/features/targets.py` | score continuo `b` con ridge por fila | 12 |
| `classifier/eval/protocol.py` | LOKO, guardarraíl anti-fuga, EDP loss, líneas base | 14 |
| `classifier/training/train_phase.py` | entrenamiento y comparación bajo LOKO | — |

Suite completa: **555 pruebas**.

### Errores encontrados y corregidos por el camino

- **`catalog_path` roto en los 20 manifiestos**: la reorganización de
  `schemas/` (`df63dfe`) los movió a `campaigns/` pero la ruta se resuelve
  contra el directorio del manifiesto. Cualquier relanzamiento habría
  fallado al arrancar. Corregido.
- **Tres sondas leían del directorio crudo** en vez del reprocesado
  `_arc174`, que solo tiene `windows.csv` para parte de las corridas. Dejó
  `rodinia_lavamd_omp` fuera del ajuste de α y a `npb_cg` con 4 puntos de 6.
- **`repetition` vale 1 siempre** en `windows.csv`; agrupar por esa columna
  fusionaba las 10 repeticiones en una pseudo-corrida y el progreso
  acumulado cruzaba de una a otra.
- **Afirmación mía demasiado fuerte**: dije "el EDP siempre gana a
  frecuencia máxima en CPU" con 3 de 9 kernels medidos. Con los 9 se
  sostiene, pero la conclusión física era prematura (ver P2).
