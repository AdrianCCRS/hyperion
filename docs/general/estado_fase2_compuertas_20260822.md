# Estado de Fase 2 tras las compuertas C1/C2/C3 y el acoplamiento de uncore

**Fecha:** 2026-08-22
**Jobs:** 6424 (compuertas C1/C2/C3), 6426 (acoplamiento uncore), ambos en
`pacca01`, partición `normal`. Ninguno tocó paccaA100.
**Datos:** `pacca_cpu_final_attempt03_20260820_arc174`, 540 corridas,
9 kernels, 9 953 516 ventanas utilizables.

Este documento reúne lo que se sabe hoy, con qué evidencia, y qué falta
antes de relanzar la campaña final de CPU. Está escrito para decidir, no
para tranquilizar.

---

## 0. Verificación previa: `b` es lo que dice ser

Antes de leer nada más: el acuerdo entre `b > 0.5` y la etiqueta de Fase 1
(`phase_label_train`) es **1.000000** sobre los ~10M de ventanas. El score
continuo es una generalización estricta de la etiqueta binaria, no otra
cosa. `k` calibrado = 1.0958.

Si esto hubiera dado distinto de 1.0, todo lo demás mediría otra cosa.

---

## 1. C1 — ¿`b` varía DENTRO de una ejecución? **Sí, con matices**

Fracción de la varianza total de `b` que es **intra-corrida** (mismo
kernel, mismo nivel de frecuencia, misma repetición): **15.3 %**.

Lo decisivo no es ese número global sino el contraste por kernel entre la
dispersión del score continuo y la de la etiqueta binaria:

| kernel | sd(`b`) intra-corrida | p05–p95 de `b` intra-corrida | clase minoritaria binaria |
|---|---|---|---|
| dgemm_n2048 | 0.019 | 0.053 | **0.0 %** |
| npb_bt | 0.105 | 0.344 | 11.1 % |
| npb_cg | 0.029 | 0.113 | **0.0016 %** |
| npb_ft | 0.057 | 0.179 | 0.36 % |
| npb_lu | 0.098 | 0.265 | 12.3 % |
| npb_mg | 0.019 | 0.059 | **0.0 %** |
| npb_sp | 0.059 | 0.158 | 0.57 % |
| 3mm_omp | 0.061 | — | — |
| lavamd_omp | — | — | — |

**Lectura.** En los cinco kernels donde la etiqueta binaria era plana
(minoritaria ≤ 0.6 %, tres de ellos en 0.0 %), el score continuo **sí**
recorre un rango medible dentro de una misma ejecución: `npb_cg` abarca
0.113 y `npb_sp` 0.158 con etiqueta binaria constante. El target continuo
recupera estructura que el binario destruía. Ése era el argumento de diseño
y queda respaldado por medición.

**Matiz que no se puede omitir.** Un p05–p95 de 0.05–0.11 es modesto sobre
un rango disponible de [0,1]. Y —más importante— **esta prueba no
distingue estructura de fase de ruido de muestreo**. Una fluctuación
ventana a ventana del OI medido produce exactamente la misma varianza
intra-corrida. Para separarlas hace falta la autocorrelación de `b` a lo
largo de la coordenada de avance: una fase real persiste sobre ventanas
consecutivas, el ruido no. **Esa prueba no se ha hecho** (pendiente C1b).

---

## 2. C2 — ¿α varía entre tramos? **Sí. ¿Sirve de algo? No**

α ajustado por celda (kernel × repetición × centil de avance), 1000 celdas
por kernel, alineadas con la coordenada de instrucciones retiradas.
r² mediano entre 0.954 y 0.9997 — los ajustes son excelentes.

| kernel | α medio | sd entre tramos | p05 | p95 | α mínimo | % celdas ≤ 0.226 |
|---|---|---|---|---|---|---|
| dgemm_n2048 | 1.159 | 0.064 | 1.073 | 1.282 | 1.020 | 0.0 % |
| npb_bt | 0.907 | 0.004 | 0.901 | 0.913 | 0.895 | 0.0 % |
| npb_cg | 0.782 | 0.092 | 0.750 | 0.880 | 0.251 | 0.0 % |
| npb_ft | 0.963 | **0.302** | 0.397 | 1.327 | 0.315 | 0.0 % |
| npb_lu | 0.892 | 0.026 | 0.849 | 0.932 | 0.827 | 0.0 % |
| npb_mg | 0.664 | **0.194** | 0.400 | 1.058 | 0.242 | 0.0 % |
| npb_sp | 0.499 | 0.023 | 0.437 | 0.526 | 0.422 | 0.0 % |
| 3mm_omp | 1.056 | 0.029 | 1.002 | 1.097 | 0.965 | 0.0 % |
| lavamd_omp | 1.092 | 0.100 | 1.043 | 1.111 | −0.119 | 0.2 % |

**Lo bueno.** α sí varía entre tramos de una misma ejecución, y en
`npb_ft` (sd 0.302, de 0.40 a 1.33) y `npb_mg` (sd 0.194) la variación es
grande. La segunda salida del modelo tiene algo que predecir.

**Lo malo, y es determinante.** De **9000 celdas, ninguna** baja del umbral
de viabilidad α ≤ 0.226, salvo un 0.2 % en lavamd que es artefacto de
ajuste. El α mínimo real en todo el dataset es **0.242**. La decisión que se
deriva de α es siempre la misma: no bajar la frecuencia. El target varía,
pero la política que produce es constante.

---

## 3. C3 — ¿El regresor le gana al trivial bajo LOKO? **No**

Rasgos: `ipc, mpki, llc_miss_rate, stall_backend_ratio, ips,
running_ratio, freq_khz_observed`. Sin fuga: ninguna variable de la que se
deriva el target entra como entrada. 769 478 filas tras submuestreo por
corrida. RandomForest, semilla fija.

| kernel excluido | MAE modelo | MAE trivial | R² modelo |
|---|---|---|---|
| dgemm_n2048 | 0.412 | 0.222 | −85.1 |
| npb_cg | 0.225 | 0.343 | −38.3 |
| npb_ft | 0.169 | 0.161 | −5.7 |
| npb_lu | 0.158 | 0.105 | −1.6 |
| npb_mg | 0.190 | 0.236 | −29.3 |
| npb_sp | 0.194 | 0.242 | −6.9 |
| 3mm_omp | 0.201 | 0.276 | −11.3 |
| lavamd_omp | 0.319 | 0.433 | −251.3 |

- MAE medio: **0.2256** (modelo) contra **0.2347** (trivial) → mejora del
  3.9 %, del orden del ruido.
- Gana en **5 de 9** pliegues. Pierde en 4, y en `dgemm` pierde por casi
  el doble.
- **R² negativo en los nueve pliegues.** R² < 0 significa que el modelo es
  peor que predecir la media del propio conjunto de prueba.

**Veredicto: C3 falla.** El regresor no generaliza a un kernel no visto.
Es el mismo resultado que ya había dado el clasificador binario (F1 0.393
contra 0.371), ahora reproducido con el target continuo. Pasar de binario a
continuo **no** resolvió el problema de generalización.

---

## 4. El hallazgo que reencuadra el proyecto: el uncore está acoplado

C2 devolvió **α > 1 en cuatro de nueve kernels** con r² de 0.96–0.999. Bajo
la ley `T(f)/T_ref = (1−α) + α·(f_ref/f)`, α es una *fracción del tiempo* y
no puede exceder 1. Un ajuste excelente con α > 1 no es ruido: es el modelo
mal especificado.

La ley supone que la parte insensible al reloj —la espera a memoria— es
constante en frecuencia. Eso solo vale si el uncore (malla, controlador de
memoria, L3) corre a frecuencia independiente.

**Prueba (job 6426).** Ancho de banda alcanzado por nivel, relativo a F0, y
su pendiente contra la frecuencia relativa. Con uncore independiente, una
carga que ya satura memoria mantendría su ancho de banda al bajar el reloj
(pendiente ≈ 0). Con uncore acoplado, lo pierde proporcionalmente
(pendiente ≈ 1).

| kernel | pendiente d(BW_rel)/d(f_rel) |
|---|---|
| npb_sp | **+1.024** |
| dgemm_n2048 | **+1.011** |
| npb_lu | **+1.009** |
| npb_bt | **+0.984** |
| 3mm_omp | **+0.968** |
| npb_ft | **+0.960** |
| npb_cg | **+0.889** |
| npb_mg | +0.681 |
| lavamd_omp | +0.571 |

Siete de nueve kernels en ≈ 1.0. Incluso `npb_cg`, el más memory-bound del
catálogo según el score (b medio 0.839), cae al 32.5 % de ancho de banda al
25 % de frecuencia.

**Por qué importa.** Reencuadra el hallazgo central del proyecto:

- Antes: *"los nueve kernels resultaron sensibles a frecuencia"* — suena a
  mala selección de benchmarks, y la respuesta sería agregar cargas.
- Ahora: *"en esta plataforma el DVFS de núcleo también frena el uncore, de
  modo que la fracción insensible a la frecuencia no llega a existir para
  ninguna carga"* — es una propiedad medida del nodo, y agregar cargas no
  la cambiaría.

La segunda es una tesis defendible. La primera es una debilidad.

**Confusión que hay que resolver.** Pendiente ≈ 1 también es lo que se ve
si el kernel *no* está realmente saturando memoria y simplemente emite
peticiones más despacio porque el núcleo va más lento. Las dos
explicaciones están confundidas en estos datos. Lo que las separa es
`ptrchase` (job 6420): una persecución de punteros pura está limitada por
latencia de memoria por construcción. Si `ptrchase` también sale con
pendiente ≈ 1 y α alto, el acoplamiento queda establecido.

---

## 5. Problema nuevo y BLOQUEANTE: el conteo de bytes de uncore

Los valores absolutos de ancho de banda son físicamente imposibles.

| fuente | BW |
|---|---|
| `stream_official`, calibración F0, vía `bytes_moved_uncore_real` | **997 GB/s** |
| `stream_official`, calibración F1 | 992 GB/s |
| Pico del nodo en el manifiesto (`bw_pico_bytes_per_s`) | **59.5 GB/s** |
| `npb_mg` F0 | 738 GB/s |
| `npb_cg` F0 | 624 GB/s |

STREAM existe precisamente para medir el pico de memoria. Que su propio
tráfico de uncore dé 997 GB/s contra los 59.5 GB/s que el mismo STREAM
reporta por stdout es una inconsistencia de **~17×**.

**Por qué bloquea.** La etiqueta depende de las dos cifras a la vez:

- `I_ridge = P_pico / BW` usa el valor de STREAM (59.5 GB/s).
- `OI = FLOPs / bytes_uncore` por ventana usa el conteo CAS.

Si los bytes están inflados ~17×, el OI está deflactado ~17× y **todas las
ventanas quedan empujadas hacia memory_bound** frente a un ridge calculado
sin esa inflación. Es un sesgo sistemático en la etiqueta misma, no en el
modelo.

**Qué NO invalida.** El análisis de pendientes de §4 es un cociente
(BW relativo a F0), así que un factor de escala constante se cancela. La
conclusión de acoplamiento sobrevive **si** el factor es independiente de
la frecuencia — cosa que hay que verificar, no asumir.

**Explicaciones inocentes posibles**, ninguna del tamaño observado:
contar los dos sockets cuando solo se usa NUMA 0 (×2), tráfico de
write-allocate que STREAM no cuenta como útil (×1.5), unidad de línea de
caché distinta de 64 B. Un ×17 no sale de ahí. Hay que auditar
`uncore_reader` contra STREAM antes de relanzar nada.

---

## 6. Dónde queda el proyecto

Con honestidad, y sin suavizarlo:

**La hipótesis original —que un modelo aprenda a elegir frecuencia y ahorre
energía— tiene evidencia negativa fuerte en CPU sobre este nodo.** No por
falta de datos ni de rejilla: por el rango dinámico de potencia del
procesador y, muy probablemente, por el acoplamiento del uncore.

**Eso no es lo mismo que un proyecto fallido.** El alcance aprobado exige
construir el clasificador y la política, y eso se construye igual. Lo que
cambia es la conclusión que reporta: en esta plataforma la política
correcta es no bajar la frecuencia, y el trabajo **explica por qué con
evidencia medida** en vez de afirmarlo. Un resultado negativo con mecanismo
identificado es más fuerte que un positivo débil.

Lo que sí está en riesgo real es la **validez de la etiqueta** (§5). Ese sí
es un problema que hay que cerrar, porque afecta a todo lo demás.

---

## 7. Qué falta, en orden

### Bloqueante antes de relanzar la campaña final de CPU

1. **Auditar el conteo de bytes de uncore contra STREAM.** Comparar el
   ancho de banda que STREAM reporta por stdout con el derivado de los
   contadores CAS, socket por socket. Sin nodo de medición: los datos ya
   existen. **Nada debe relanzarse hasta cerrarlo.**
2. **Verificar que el factor de inflación es independiente de la
   frecuencia.** Si no lo fuera, la conclusión de §4 se cae.

### Encolado, esperando que se libere paccaA100

3. **Job 6420** — preflight de fases, 27 corridas, ~20 min. Responde:
   ¿es alcanzable α ≤ 0.226 (S1)?, ¿pasan la validación en el fondo del
   rango (S2)?, ¿se resuelve la fase en ventanas de 1 ms (S3)?, ¿cruza la
   etiqueta de verdad con las ventanas (S4)?, ¿es comparable la fase de
   cómputo con las cargas reales (S5)?
4. **Job 6412** — campaña de fases, ~4 h, en `hold`. Su diseño depende de
   lo que diga 6420.

### Sin nodo, pendientes

5. **C1b — autocorrelación de `b`** a lo largo de la coordenada de avance.
   Separa estructura de fase de ruido de muestreo. Decide si el 15.3 % de
   varianza intra-corrida es señal.
6. **Revisar C3 con el target corregido**, si la auditoría de §5 cambia las
   etiquetas.

### Solo en paccaA100, y solo si hay modelo que valga la pena

7. **Latencia de inferencia.** Es una afirmación sobre el hardware de
   despliegue; medirla en `pacca01` sería inválido.

---

## 8. Decisiones que este documento NO toma

- **D2** (arquitectura del modelo de Fase 2) sigue abierta. C1 respalda la
  primera salida (`b` continuo). C2 y C3 debilitan la segunda: α varía pero
  no cruza el umbral, y el regresor no generaliza. No se escribe §Fase 2 del
  libro hasta cerrar §5 y 6420.
- **Agregar cargas reales multifásicas** (HPCG u otra) queda supeditado a
  C1b: si la varianza intra-corrida es ruido, agregar cargas no la arregla;
  si es señal, puede que no haga falta agregar nada.
