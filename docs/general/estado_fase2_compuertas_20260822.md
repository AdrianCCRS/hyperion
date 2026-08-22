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

## 4. Tres retractaciones y el hallazgo que sí sostiene la evidencia

Las conclusiones de la primera versión de este documento sobre uncore y
sobre el conteo de bytes eran **incorrectas**. La auditoría (job 6427) las
desmintió. Se dejan escritas porque el error importa tanto como el
resultado.

### 4.1 RETRACTADO — "los bytes de uncore están inflados ~17×"

Se calculó el ancho de banda como `bytes_moved_uncore_real / delta_t_ns`
fila por fila. Eso está mal. `_apply_uncore_intervals`
(orchestrator/postprocess.py:300-317) difunde los bytes de UN intervalo de
uncore a TODAS las ventanas de CPU que cubre: el intervalo dura ~13 ms
(piso de `perf stat -I`) y la ventana ~1 ms, así que cada fila lleva
escrito el total del intervalo, no su parte. El docstring advierte de esta
clase de error exactamente.

Medido: **13.0 ventanas por intervalo, idéntico en los seis niveles de
frecuencia, p05 = p95 = 13.0, sobre las 540 corridas.** 997 / 13 = 76.7.

A la granularidad correcta:

| fuente | BW |
|---|---|
| STREAM F0, por intervalo | **76.80 GB/s** |
| Pico declarado del nodo | 59.50 GB/s |
| cociente | **1.291** |

1.291 no es un error: es el factor de *write-allocate* / RFO. Un fallo de
escritura lee primero la línea, así que la DRAM ve más tráfico del que
STREAM contabiliza como útil (×1.5 en Copy/Scale, ×1.33 en Triad). Para la
mezcla de STREAM, ~1.3 es el valor de libro.

**El conteo de bytes es correcto y queda validado contra física conocida.
La etiqueta no está sesgada.** No hay nada bloqueante aquí.

### 4.2 RETRACTADO — "el uncore está acoplado al reloj del núcleo"

Se infirió del hecho de que 7 de 9 kernels pierden ancho de banda con
pendiente ≈ 1.0 contra la frecuencia. Se advirtió en su momento que esa
señal está confundida con "el kernel no estaba saturando memoria". **El
control resuelve la confusión en contra de la hipótesis.**

STREAM, que satura memoria por construcción:

| nivel | MHz | BW (GB/s) | BW relativo | f relativo |
|---|---|---|---|---|
| F0 | 3200 | 76.80 | 1.000 | 1.000 |
| F1 | 2600 | 76.68 | 0.998 | 0.813 |
| F2 | 2000 | 75.12 | 0.978 | 0.625 |
| F3 | 1400 | 70.72 | 0.921 | 0.438 |
| F4 | 800 | 60.18 | **0.784** | 0.250 |

**STREAM conserva el 78.4 % de su ancho de banda al 25 % del reloj**
(pendiente ≈ 0.29, no ≈ 1.0). El camino a memoria NO se frena con el DVFS
de núcleo en este nodo.

Entonces la pendiente ≈ 1.0 de los otros siete kernels significa otra cosa:
**no estaban saturando memoria.** Al bajar el reloj simplemente emiten
peticiones más despacio. Es comportamiento limitado por núcleo.

Ancho de banda corregido en F0, contra los 76.80 GB/s de STREAM:

| kernel | BW F0 (GB/s) | % de STREAM | pendiente |
|---|---|---|---|
| npb_mg | 57.18 | 74 % | 0.681 |
| npb_cg | 48.35 | 63 % | 0.889 |
| npb_sp | 43.73 | 57 % | 1.024 |
| npb_ft | 21.76 | 28 % | 0.960 |
| dgemm_n2048 | 21.01 | 27 % | 1.011 |
| npb_lu | 5.84 | 8 % | 1.009 |
| npb_bt | 3.88 | 5 % | 0.984 |

Ninguno satura. La tendencia general es la esperada —más saturado, menor
pendiente— con `npb_sp` como excepción sin explicar. `3mm_omp` y
`lavamd_omp` mueven 0.18 y 0.07 GB/s: su "pendiente de ancho de banda" mide
ruido, no memoria, y por eso es errática.

### 4.3 RETRACTADO — "ninguna carga puede bajar del umbral en este nodo"

Ajuste directo de α sobre las duraciones reales de las corridas de
calibración:

| carga | α | r² | T(F4)/T(F0) |
|---|---|---|---|
| **stream_official** | **0.1538** | 0.982 | 1.484 |
| ert_probe | 0.2277 | 0.980 | 1.716 |

**STREAM está en α = 0.154, por debajo del umbral de viabilidad 0.226.**
ERT está justo encima.

Es la existencia que faltaba: **sí hay cargas por debajo del umbral en este
nodo.** Bajar la frecuencia SÍ mejora el EDP de STREAM.

---

## 5. Diagnóstico corregido

El hallazgo central del proyecto no es una propiedad de la plataforma. Es
una propiedad **del catálogo**:

- El umbral de viabilidad es α ≤ 0.226.
- STREAM, saturando memoria, está en 0.154. **Por debajo.**
- El mínimo de los 9 kernels del dataset es 0.242, sobre 9000 celdas.
- Ninguno de los 9 supera el 74 % del ancho de banda alcanzable, y cinco
  no llegan al 30 %.

Los nueve kernels del dataset **no son suficientemente memory-bound**. El
régimen donde el DVFS paga existe en este nodo, y el catálogo no lo toca.

Eso es exactamente el hueco que `ptrchase` (persecución de punteros,
limitada por latencia) y `phasic` fueron construidos para cubrir, y ahora
hay una razón medida para esperar que funcionen en vez de una corazonada.

---

## 6. Dónde queda el proyecto

Mucho mejor que hace unas horas, y por evidencia, no por optimismo.

**Lo que sigue en pie:**
- C3 falla: el regresor no generaliza a un kernel no visto (R² negativo en
  los nueve pliegues). Ese problema es real y no lo toca nada de lo
  anterior.
- La causa sigue siendo la misma: 9 kernels no son una muestra con la que
  LOKO signifique algo, y todos viven en el mismo régimen.

**Lo que cambió:**
- El objetivo es alcanzable en este hardware. Antes parecía físicamente
  cerrado.
- Se sabe qué falta: cargas en el régimen α < 0.226, que es donde está
  STREAM y donde no hay ni un kernel del dataset.
- La etiqueta está validada contra física conocida (§4.1), no solo contra
  sí misma.

**Lo que sigue abierto y no debe minimizarse:**
- C1b: si el 15.3 % de varianza intra-corrida es señal o ruido.
- Que α > 1 en cuatro kernels sigue sin explicación. Descartado el uncore,
  la causa es otra y no se ha identificado.

---

## 7. Qué hacer, en orden

1. **Job 6420 (preflight, ~20 min, en cola).** Ahora tiene un prior fuerte:
   si STREAM da 0.154, `ptrchase` —limitado por latencia y no por ancho de
   banda— debería quedar por debajo. Es la confirmación directa.
2. **Medir la rejilla 2800–3200 MHz.** Bajo el objetivo de minimizar
   energía con holgura de tiempo `s`, el óptimo cae en
   `f* = f_ref / (1 + s/α)`: con α ≈ 0.9 y s = 5 %, en ~3032 MHz; con
   α = 0.5, en ~2909 MHz. La rejilla actual salta 3200 → 2600 y **no tiene
   una sola medición ahí**. Es la campaña de rejilla fina que se canceló
   (job 6391) y que debe volver.
3. **C1b — autocorrelación de `b`.** Sin nodo.
4. **Explicar α > 1.** Sin nodo, sobre los datos existentes.
5. **Añadir cargas en el régimen de STREAM al catálogo del dataset**, no
   solo como calibración. STREAM ya demuestra que el régimen existe.
