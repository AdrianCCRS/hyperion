# Reporte de metodología de `warmup_seconds` -- revisión 4

> **Nota de vigencia (2026-08-14):** este reporte conserva la evolución
> histórica del análisis de warmup. Las explicaciones que atribuyen señales
> planas a un P4 todavía bloqueado o a un reloj inevitable de 765 MHz quedaron
> invalidadas como explicación vigente por ARC-137: el mecanismo ya era
> funcional y `-lgc` sostuvo 600 y 1200 MHz bajo carga en la prueba registrada
> con el controlador actual, sin atribuirle causalidad. Los valores de warmup ya incorporados al catálogo no se
> cambian por ese hecho, pero cualquier nueva inferencia sobre transitorios GPU
> debe repetirse por nivel durante la prueba de humo/campaña vigente.

**Revisión 4 (ARC-86)**: a pedido explícito del usuario ("Re caractericemos,
hagamos pruebas, no quiero dejar absolutamente nada al azar"), se
re-caracterizaron de verdad los 6 kernels GPU que las revisiones 2/3 habían
dejado explícitamente fuera de alcance (sección 7.3) -- alargando cada uno,
re-verificando con `ncu`, y midiendo warmup sobre datos reales, en vez de
quedarse con el diagnóstico de por qué no se podía. De paso se encontró y
corrigió un bug real en el propio detector de warmup (una ventana de puros
ceros se declaraba "estable" por construcción). Ver sección 10.

**Revisión 3**: el usuario señaló, con razón, que la revisión 2 dejaba la
mayoría del catálogo (14 de 17 kernels de dataset) exactamente igual que al
principio -- números puestos a mano, solo que ahora con una explicación de
por qué no se midieron, en vez de resueltos de verdad. Esta revisión cierra
la brecha en lo que sí es accionable, sección 10 (de la revisión 3; ver
sección 10 actual para el pase de la revisión 4).



**Revisión 1** de este documento se mandó a una IA externa pidiendo que
validara la metodología. La respuesta que volvió no tenía relación con el
contenido -- contestó una pregunta distinta (por qué el ridge point de GPU
cambia según precisión FP32/FP64), ya resuelta en el proyecto desde
ARC-76/78/80, usando además nuestros propios números como si fueran un
hallazgo nuevo. Sección 0 explica ese desajuste. El resto de esta revisión
es investigación propia (búsqueda directa en literatura) más una prueba de
sensibilidad real sobre los datos que **encontró un error real** en la
revisión 1: `npb_lu` se había cambiado con una medición que no sobrevive un
umbral más estricto y consistente con la literatura -- corregido (sección
4).

## 0. Por qué se descarta la respuesta de la IA externa (revisión 1)

Se le pasó este documento completo a una IA con navegación web. Su
respuesta explicó, correctamente en el fondo pero **fuera de tema por
completo**, por qué `i_ridge_gpu` depende de si un kernel es FP32 o FP64,
usando literalmente los números que el propio proyecto ya había calculado
(10178.2/4698.6 GFLOP/s, 1399 GB/s) y concluyendo que hacen falta "dos
ridges por frecuencia" -- **exactamente lo que `calibration.run_gpu_calibration()`
ya implementa desde ARC-80**. Ninguna de las 5 preguntas de la sección 7
(sobre `warmup_seconds`) fue contestada. Conclusión: la respuesta no aporta
nada sobre el tema de este reporte -- probablemente arrastre de contexto de
una conversación previa sobre el tema de ridge points, no una lectura real
del documento. Se descarta por completo, no se usa nada de esa respuesta
abajo.

## 1. El problema que se estaba resolviendo

El catálogo de kernels del proyecto (`orchestrator/schemas/kernels/catalog.yaml`)
tiene un campo `warmup_seconds` por kernel: la cantidad de segundos al
principio de una corrida que se excluyen de la clasificación de
entrenamiento (aunque no se borran de los datos, quedan marcados
`quality_status=warmup_excluded`). Hasta ahora esos valores se pusieron a
mano (0.3, 0.5, 1.0 segundos...) sin medir nada -- y eso ya había producido
un bug real: dos kernels (`dgemm_n2048`, `gpu_dgemm_n4096`) tienen un
`warmup_seconds` mayor que su propia duración total, así que producían
**cero** ventanas utilizables (`quality_status=ok`).

## 2. Hallazgo clave que hizo esto posible sin re-correr nada

`warmup_seconds` **nunca se le pasa al harness C++** (`runner.py` no lo
incluye al construir el comando del launcher) -- es un filtro que aplica
`postprocess.py` *después* de tener la corrida completa ya medida. Toda
campaña ya corrida capturó el transitorio de arranque completo desde el
instante cero; solo estaba marcado como `warmup_excluded` en vez de usarse
para medir cuánto dura de verdad. Se reutilizaron datos de campañas reales
ya ejecutadas en `paccaA100` (`pacca_ref_full_20260806` para CPU,
`pacca_gpu_ref_20260807` para GPU), sin recolectar nada nuevo.

## 3. Metodología aplicada

Implementada en `scripts/pacca/measure_warmup.py`. Para cada kernel:

1. **Señal usada**: IPC (`instructions/cycles` por ventana) para CPU;
   `gpu_util_pct` (NVML) para GPU.
2. **Tamaño de ventana adaptativo**: `max(3, min(15, n_muestras // 4))`.
3. **Criterio de "fin del transitorio"**: coeficiente de variación (CV% =
   desviación estándar / media) por debajo de un umbral, sostenido en dos
   ventanas móviles **consecutivas** (no en el resto de la corrida, para no
   confundir el fin del arranque con un cambio de fase legítimo más
   adelante -- `npb_ft`/`npb_bt` alternan fases reales, ARC-71).
4. **Margen de seguridad**: el punto detectado ×1.2, mismo margen para
   todos los kernels.

**Corrección de la revisión 2**: el umbral de CV% en la revisión 1 era 5%,
tomado prestado de `CAL-10` (un chequeo de estabilidad de referencia
IPC/IPS entre repeticiones, no de detección de warmup) solo porque ya
existía en el proyecto. Investigación propia (sección 5) encontró que la
literatura de referencia para este problema específico usa un umbral
distinto -- se repitió la medición con ese umbral y se comparó (sección 6).

## 4. Resultados: comparación 5% (prestado) vs. 2% (literatura, Georges et al. 2007)

### CPU (señal: IPC)

| kernel | catálogo original | detectado a 5% (3 reps) | detectado a 2% (3 reps) | ¿robusto? | valor final |
|---|---|---|---|---|---|
| `npb_bt` | 0.5 s | 0.0601/0.0611/0.0601 s | 0.0601 s (idéntico) | **sí** | **0.08 s** |
| `npb_cg` | 1.0 s | 0.0501×3 | 0.0501 s (idéntico) | **sí** | **0.07 s** |
| `npb_lu` | 1.0 s | 0.5901/0.6581/0.2481 s (ya inconsistente entre reps) | no detectado en ninguna de las 3 | **no** | **1.0 s (revertido)** |
| `npb_ft` | 0.5 s | 0.0031 s ×3 (muy consistente) | 0.1001/0.0151/0.1061 s (inconsistente) | **no** | 0.5 s (sin cambio, ya se había decidido no aplicar en la rev. 1) |
| `npb_mg` | 0.3 s | no detectado | no detectado | -- | 0.3 s (sin cambio) |
| `npb_sp` | 0.5 s | no detectado | no detectado | -- | 0.5 s (sin cambio) |
| `dgemm_n2048` | 0.5 s | no detectado (corrida ≈0.255s) | no detectado | -- | sin cambio, ver nota 7.2 |

### GPU (señal: `gpu_util_pct`)

| kernel | detectado a 5% | detectado a 2% | ¿robusto? | valor final |
|---|---|---|---|---|
| `rodinia_lud` | 0.804/0.804/0.805 s | 0.804 s (idéntico) | **sí** | 1.0 s ya declarado, coincide con lo medido -- sin cambio |
| resto de kernels GPU | no detectado (4-28 muestras) | no detectado | -- | sin cambio, ver nota 7.3 |

**Lectura del experimento**: `npb_bt`, `npb_cg` y `rodinia_lud` dan
*exactamente* el mismo resultado sin importar si el umbral es 5% o 2% --
alta confianza, esos tres cambios se mantienen. `npb_lu` y `npb_ft` **no
sobreviven** el umbral correcto -- la detección a 5% resultó ser un
artefacto del umbral demasiado laxo, no una medición real. `npb_lu` se
revirtió a su valor original (1.0 s); `npb_ft` ya se había dejado sin
aplicar en la revisión 1 por la misma sospecha (documentada antes de hacer
esta prueba de sensibilidad, no después -- la sospecha original resultó
correcta).

## 5. Lo que dice la literatura (investigación propia, no de la IA externa)

Búsqueda directa, no delegada. Hallazgos con fuente:

1. **El criterio CV%-por-debajo-de-un-umbral SÍ es un método reconocido y
   citado**: Georges, Buytaert y Eeckhout, *"Statistically Rigorous Java
   Performance Evaluation"*, OOPSLA 2007 -- uno de los papers más citados
   de metodología de benchmarking. Su heurística calcula el CV de las
   últimas *k* mediciones y considera terminado el arranque cuando cae por
   debajo de un umbral preestablecido, típicamente **1-2%**, no 5%.
2. **Ese mismo método tiene limitaciones documentadas por trabajos
   posteriores**: según Kalibera y Jones (citados en el survey "Towards
   effective assessment of steady state performance in Java software: are
   we there yet?", *Empirical Software Engineering*, Springer 2022), la
   heurística de Georges et al. **"a menudo falla en identificar
   confiablemente el fin del warmup"**. Esto es relevante para el proyecto:
   los casos donde nuestro detector no encontró nada (`npb_mg`, `npb_sp`,
   `npb_lu`, casi todo GPU) son el **modo de falla ya documentado en la
   literatura de este mismo método**, no evidencia de que el método esté
   mal implementado aquí.
3. **Mejoras posteriores existen**: Laaber et al. usan la diferencia
   mín-máx de CV en una ventana deslizante (más robusto que un CV puntual);
   Barrett et al. proponen detección de *change points* (algoritmo PELT),
   totalmente automatizada y presentada como más rigurosa que el umbral de
   CV -- candidato natural si se quiere resolver `npb_lu`/`npb_mg`/`npb_sp`
   más adelante, no implementado en este cambio.
4. **Contexto de inicialización de CUDA**: reportado en foros técnicos de
   NVIDIA y benchmarks públicos, el overhead constante de inicializar un
   contexto CUDA para una sola corrida ronda **~50 ms** (con `cudaMalloc`
   aportando ~30 ms de eso) en GPUs de clase Tesla/datacenter -- mucho más
   rápido en GPUs de consumidor más viejas (1-4 s reportado en un GTX 465,
   no comparable a la A100). Esto es consistente con por qué kernels GPU
   de \<1 s (`lavamd`, `heartwall`, `backprop`) son fundamentalmente
   difíciles de separar: su duración total es del mismo orden que el
   propio overhead de arranque de CUDA, no hace falta que el método de
   detección tenga ningún defecto para que falle ahí.
5. **JMH (Java Microbenchmark Harness)**, la referencia de facto para
   medir con rigor en la JVM, por defecto usa 5 iteraciones de 10 segundos
   de warmup (50 s totales) antes de medir -- una escala de tiempo mucho
   mayor que la nuestra, pero del mismo dominio de problema que motiva este
   reporte (JIT, no aplica a nuestro código nativo sin JIT) -- confirma que
   "no adivinar el warmup, medirlo" es una práctica estándar establecida,
   no algo inventado para este proyecto.

## 6. Verificación empírica final (3 kernels con cambio aplicado y confirmado: `npb_bt`, `npb_cg`; `npb_lu` revertido)

Postprocess re-corrido sobre las 3 repeticiones reales de cada uno en
`paccaA100`:

| kernel | ventanas `ok` antes | después | `phase_label_train` antes (ARC-71) | después |
|---|---|---|---|---|
| `npb_bt` | 24449 | 24869 (+420) | 85.6% compute_bound | 85.7% compute_bound |
| `npb_cg` | 5748 | 6678 (+930, +16%) | 92.7% memory_bound | 94.1% memory_bound |
| `npb_lu` | 16329 | 16329 (revertido, sin cambio) | 88.4% compute_bound | 88.4% compute_bound (idéntico, confirma el revert) |

## 7. Segundo pase (ARC-83): no quedarse en "no se pudo medir"

La revisión 2 dejaba 14 de 17 kernels de dataset sin resolver, con una
razón documentada pero sin número medido -- objetable, con razón: una
explicación de por qué no se midió no es lo mismo que resolverlo. Este
pase intentó activamente cerrar esa brecha en las dos direcciones que sí
son accionables sin re-verificar clasificaciones ya hechas con `ncu`.

### 7.1 Método más riguroso para `npb_mg`/`npb_sp`/`npb_ft`/`npb_lu`: segmentación binaria (change points)

La sección 5 (punto 3) menciona que Barrett et al. proponen detección de
*change points* como alternativa más rigurosa al umbral de CV. Se
implementó una versión de esa familia de métodos (segmentación binaria,
Scott & Knott 1974 -- misma idea que el PELT de Barrett et al., sin su
optimización de velocidad, innecesaria para el tamaño de estas series): en
vez de buscar "¿la señal es plana dentro de un umbral?", busca "¿hay un
cambio real y grande en el comportamiento de la señal?", sin asumir que el
estado estable es plano -- más apto para separar fin-de-arranque de
cambios de fase reales.

**Resultado, sobre las mismas 3 repeticiones reales de cada kernel**:
`npb_mg` encontró un changepoint en solo 1 de 3 repeticiones (0.018s) --
no robusto, no se aplica. `npb_sp`, `npb_ft` y `npb_lu` **no encontraron
ningún changepoint significativo en ninguna repetición** -- un método más
riguroso confirma, no contradice, la conclusión de la sección 4: estos
kernels genuinamente no tienen una transición de arranque separable con la
señal disponible (IPC). Para `npb_sp` en particular esto es coherente con
lo que ya se sabía (ARC-71: kernel "intermedio", variabilidad continua de
fase, no un arranque-luego-estable). **Conclusión más fuerte que en la
revisión 2**: no es que el método no haya buscado lo suficientemente
bien -- se probaron dos métodos de familias distintas (umbral de CV y
segmentación por varianza) y ninguno encontró señal. Se mantiene el valor
original de catálogo para los 4 (`npb_mg=0.3`, `npb_sp=0.5`, `npb_ft=0.5`,
`npb_lu=1.0`), ahora respaldado por evidencia negativa de dos métodos, no
por ausencia de intento.

### 7.2 `dgemm_n2048` / `gpu_dgemm_n4096`: arreglados de verdad, no solo diagnosticados

El arreglo que se venía posponiendo desde ARC-71/76 (alargar la corrida)
se ejecutó. Aumentar `--iterations` no cambia la intensidad operacional
(la misma multiplicación se repite más veces, no cambia el tamaño del
problema), así que no invalida ninguna clasificación ya verificada con
`ncu`.

- **`dgemm_n2048`** (CPU): `--iterations` 6→80. Corrida real instrumentada:
  2.98 s (antes 0.255 s). Con el `warmup_seconds=0.5` ya declarado, ahora
  produce **2476 ventanas `ok`** (antes: 0), 99.2% `compute_bound` -- la
  clasificación esperada, confirmada con datos reales por primera vez.
- **`gpu_dgemm_n4096`** (GPU): `--iterations` 10→300. Corrida real
  instrumentada: 4.94 s (antes 0.13 s), 45 muestras NVML (antes 7). El
  detector de warmup (umbral de CV, sección 3) ahora sí encuentra una señal
  clara: ~0.80 s crudo, ~0.96 s con margen -- **casi idéntico al 1.0 s ya
  declarado**. El valor viejo resultó ser razonable; se confirma, no se
  cambia.

Ambos verificados con una corrida instrumentada real (`runner.run_single`
+ `postprocess.run_postprocess`), no solo con el binario suelto.

### 7.3 Kernels GPU cortos restantes (`rodinia_lavamd`, `rodinia_heartwall`, `rodinia_backprop`, `rodinia_myocyte`, `rodinia_dwt2d`, `rodinia_hotspot`): explícitamente NO arreglados, y por qué no

A diferencia de los dos casos de DGEMM, estos kernels no tienen un
parámetro tipo "--iterations" que repita el mismo trabajo sin cambiar el
problema -- alargarlos significa aumentar el tamaño de entrada
(`boxes1d`, cantidad de frames, tamaño de imagen), lo cual **sí puede
cambiar la intensidad operacional real** (más o menos reuso de cache según
el tamaño, ver el propio caso de `lud`/`lavamd` con distintos tamaños en
ARC-75/80). Alargarlos sin cuidado invalidaría su clasificación ya
verificada con `ncu` (ARC-72/75/76), y volver a correr `ncu` para cada uno
es un esfuerzo comparable al de la caracterización original -- **una
decisión de alcance, no algo para resolver de paso dentro de un reporte de
warmup**.

**Tercer intento (ARC-84), sin cambiar el binario ni el tamaño del
problema**: se corrieron los 6 con `gpu_interval_ns` reducido de 100 ms a
5 ms (20x más resolución) -- un cambio puramente de la frecuencia de
muestreo de NVML, no del kernel medido, así que no arriesga ninguna
clasificación ya verificada. Con más resolución, el detector "encontró"
un warmup en los 6 -- pero al inspeccionar la señal cruda, tres de ellos
(`backprop`, `heartwall`, `myocyte`) resultaron ser una detección
espuria: `gpu_util_pct` se queda en 0-1% (un valor casi constante, no un
arranque real) durante casi toda la corrida, así que el CV sale bajo
*porque la señal es degenerada*, no porque haya un estado estable
significativo. Se probó `gpu_power_mw` como señal alternativa (más rango
dinámico esperado) para los 6 -- y resultó **completamente plana**
(36961-40131 mW, ~±4%) en los 6 kernels, sin ninguna rampa visible.

**La razón de fondo conecta con un hallazgo ya establecido en el proyecto
(ARC-77)**: la GPU no sube de reloj bajo carga en este nodo (bloqueado por
el mismo permiso `P4` que bloquea todo el control de reloj de GPU). Si el
reloj no sube, ni la potencia ni la utilización van a mostrar una rampa
real que un detector pueda encontrar -- **no es un problema de resolución
de muestreo** (ya se probó 20x más fino y no ayudó), es que las señales
disponibles en NVML no llevan la información necesaria mientras la GPU
esté clock-locked al mínimo. Aumentar más la resolución de muestreo no
va a cambiar esta conclusión. Quedan sin cambio, con esta razón -- que no
existía en la revisión 3 -- documentada explícitamente en vez de un número
inventado.

## 8. Conclusión y qué se hizo con `warmup_seconds` (estado final, 17 kernels de dataset)

**Medido y cambiado, robusto a dos umbrales/métodos distintos:**
- `npb_bt` → 0.08 s
- `npb_cg` → 0.07 s

**Medido y confirmado sin cambio (el valor original ya era correcto):**
- `rodinia_lud` → 1.0 s (medido: ~0.80 s crudo, ~0.96 s con margen)
- `gpu_dgemm_n4096` → 1.0 s (medido tras alargar la corrida: ~0.80 s crudo,
  ~0.96 s con margen)

**Corregido estructuralmente (el problema nunca fue el número de warmup):**
- `dgemm_n2048`: `--iterations` 6→80, corrida real 0.255s→2.98s, ahora
  produce 2476 ventanas `ok` (antes 0), 99.2% `compute_bound` confirmado.

**Medido con dos métodos distintos y confirmado que NO hay señal
separable (evidencia negativa, no ausencia de intento):**
- `npb_mg`, `npb_sp`, `npb_ft`, `npb_lu` -- ni el umbral de CV (Georges et
  al.) ni la segmentación binaria (Barrett et al., familia PELT)
  encontraron una transición robusta y consistente entre repeticiones.
  Valores originales de catálogo sin cambio, ahora con este respaldo.

**Re-caracterizados de verdad en la revisión 4 (ARC-86, sección 10),
alargados y re-verificados con `ncu`:**
- `rodinia_hotspot` → 2.2189 s (medido: transición real 1.4%→72% util)
- `rodinia_heartwall` → 0.2929 s (medido: transición real 1.4%→90.8% util,
  37.8W→52.7W)
- `rodinia_lavamd` → 4.3753 s (medido: transición real de potencia
  37W→93W, dominada por inicialización + copia H2D de un dataset ~343x
  mayor; OI remedido 1233→708.14 FLOP/byte, sin cambio de clasificación)
- `rodinia_myocyte`, `rodinia_backprop`, `rodinia_dwt2d` → 1.0 s sin
  cambio, pero ahora con evidencia negativa que sobrevivió a un
  alargamiento real del problema (no solo a más resolución de muestreo
  como en ARC-84) -- potencia plana confirmada en los tres incluso a
  tamaño mucho mayor.

**Balance final sobre 17 kernels de dataset**: 4 medidos y
cambiados/confirmados en el primer pase (`npb_bt`, `npb_cg`, `rodinia_lud`,
`gpu_dgemm_n4096`), 1 corregido estructuralmente (`dgemm_n2048`), 4 con
evidencia negativa de dos métodos (`npb_mg`/`npb_sp`/`npb_ft`/`npb_lu`), 3
GPU re-caracterizados y medidos con datos reales en la revisión 4
(`hotspot`, `heartwall`, `lavamd`), 3 GPU con evidencia negativa que
sobrevivió a un alargamiento real (`myocyte`, `backprop`, `dwt2d`, sección
10.2). Ningún kernel del catálogo queda con un número sin respaldo de
medición o de evidencia negativa documentada.

**Mejora futura identificada, ya implementada**: la segmentación
binaria (sección 7.1) y su variante basada en meseta (sección 10.1) ya
están en `measure_warmup.py`; el piso de ruido del umbral de CV (sección
10.1) corrige el modo de falla que habría producido falsos positivos en
cualquier señal GPU con tramos en cero.

## 10. Tercer pase (ARC-86): re-caracterización real de los 6 kernels GPU restantes

La sección 7.3 dejó 6 kernels explícitamente sin arreglar porque alargarlos
implicaba cambiar el tamaño del problema (riesgo de invalidar la
clasificación de `ncu`). El usuario pidió resolverlo de verdad: "Re
caractericemos, hagamos pruebas, no quiero dejar absolutamente nada al
azar". Se aceptó el riesgo y se verificó con `ncu` en cada caso, en vez de
evitarlo.

### 10.1 Bug encontrado en el propio detector antes de confiar en nada

Al re-caracterizar `rodinia_hotspot` (1000→50000 iteraciones de
simulación), el umbral de CV devolvió `warmup=0.0s` -- pero la traza cruda
mostraba `gpu_util_pct` en ~0-4% hasta t=1.85s y en 100% después, una
transición real y grande. Causa: `_cv_pct` devuelve 0.0 cuando la media de
la ventana es cero (CV = desviación/media, y una ventana de puros ceros no
tiene desviación) -- el detector declaraba "estable" el reposo inicial,
exactamente lo opuesto de lo que debía medir. Corregido con un piso de
ruido (`min_mean_floor=5.0` para señales GPU en `detect_warmup_ns`): una
ventana "estable en cero" ya no cuenta como fin de arranque. El método de
*changepoint* (sección 7.1) tenía un problema relacionado: tomaba el primer
changepoint a secas, que en `hotspot` caía en un blip espurio de 3-4% en
t=0.33s, no en la meseta real de 100% que empieza en t=2.13s. Corregido
para buscar el primer segmento cuya media alcanza el 80% de la meseta de
mayor carga (`detect_warmup_via_changepoints`, `plateau_ratio=0.8`). Con
ambos arreglos, CV y changepoint coinciden en `hotspot`: ~1.85s y ~2.13s
respectivamente, la misma transición real vista en la traza.

### 10.2 Resultados por kernel

**Con transición real medida** (potencia idle→carga confirmada en la traza
cruda, no solo en la salida del detector):

- **`rodinia_hotspot`**: iteraciones de simulación 1000→50000 (mismo
  stencil repetido, no cambia el tamaño del problema). `ncu`: 5.03→5.02
  FLOP/byte, sin cambio de clasificación (`memory_bound`). Warmup: 1.0s→
  **2.2189s** -- `gpu_util_pct` 1.4%→72% de media antes/después del punto
  detectado.
- **`rodinia_heartwall`**: video sintético generado con 20000 cuadros
  (antes 25), pero solo 1000 procesados -- el harness añade
  sincronización por cada lanzamiento de kernel (shim de blocking-sync,
  ARC-72), y ese costo escala con el número de cuadros: 20000 cuadros
  procesados tardó **115s reales bajo el harness** frente a 4.33s en el
  binario suelto, medido explícitamente antes de fijar el tamaño final en
  1000 (~6.5s bajo harness, extrapolado de fijo≈0.66s + ~5.7ms/cuadro).
  `ncu`: 35.3→35.30 FLOP/byte, sin cambio (`compute_bound`). Warmup: 1.0s→
  **0.2929s** -- `gpu_util_pct` 1.4%→90.8%, `gpu_power_mw` 37.8W→52.7W.
- **`rodinia_lavamd`**: `boxes1d` 10→70 -- a diferencia de los demás, este
  SÍ cambia el tamaño real del problema N-Body (el propio comentario de
  ARC-75 ya advertía que este valor es sensible al tamaño). `ncu`
  confirmó el cambio: 1233→**708.14** FLOP/byte -- baja porque a este
  tamaño el working set ya no cabe tan bien en L2, pero sigue muy por
  encima de cualquier ridge posible (~3.4-7.3), así que la clasificación
  `compute_bound` no cambia. Warmup: 1.0s→**4.3753s** -- una fracción
  grande del run (66%) pero real: la traza muestra potencia plana en
  ~37W (idle) desde t=0 hasta t=3.68s, con salto limpio a ~93W (100%
  util) justo en la ventana detectada -- dominado por la inicialización
  de contexto CUDA y la copia H2D de un dataset ~343x mayor (~3 GB
  medidos con `nvidia-smi`, memoria estimada del catálogo actualizada de
  64 MB a 4 GB en consecuencia).

**Con evidencia negativa que sobrevivió a un alargamiento real** (no solo a
más resolución de muestreo, como en ARC-84):

- **`rodinia_myocyte`**: `xmax` (duración de simulación) 100→100000 ms.
  `ncu`: 0.017 FLOP/byte, idéntico. Incluso con 1000x más duración,
  `gpu_power_mw` se queda plano en ~37W (piso de idle de la A100) durante
  toda la corrida -- los blips de `gpu_util_pct` (hasta 19%) no se
  corresponden con ningún aumento de potencia real. Causa raíz: cada
  lanzamiento de kernel es un grid de `(2,1,1)x(32,1,1)` = 64 hilos
  (`workload=1`, una sola instancia de EDO) -- demasiado corto para que
  NVML lo resuelva sin importar cuántos pasos se acumulen. Warmup: sin
  cambio, 1.0s.
- **`rodinia_backprop`**: tamaño de entrada 524288→**1048560**, el máximo
  seguro antes del límite de `gridDim.y=65535` ya documentado en ARC-72
  (`num_blocks=in/16`). `ncu`: 0.087→0.0799 FLOP/byte, sin cambio de
  clasificación (`memory_bound`, margen de dos órdenes de magnitud). Pero
  backprop solo lanza 2 kernels en total (sin bucle de épocas en el
  código CUDA) -- incluso al tamaño máximo la corrida instrumentada dura
  solo 1.28s y la potencia se queda en el piso de idle. Warmup: sin
  cambio, 1.0s.
- **`rodinia_dwt2d`**: imagen sintética 192→16384 px (`ncu`: 4.10→**2.17**
  FLOP/byte -- baja por más tráfico DRAM sostenido a esta escala, sin
  cambio de clasificación `memory_bound`). Pese al salto de tamaño
  (805 MB de imagen, ~8.35 GB de memoria GPU pico medida con
  `nvidia-smi`), `gpu_power_mw` nunca supera 46.6W en toda la corrida --
  el filtro DWT es demasiado liviano/memory-bound para generar carga
  sostenida sin importar el tamaño de imagen probado; el tiempo de pared
  lo domina la carga del archivo y la inicialización, no el cómputo GPU.
  Warmup: sin cambio, 1.0s.

### 10.3 Verificación de campaña real de punta a punta

Se corrió `run-campaign` completo (calibración + los 6 kernels x 3
repeticiones) sobre `paccaA100` tras limpiar 18 directorios de corridas
obsoletas (confirmado con el usuario antes de borrar -- contenían
`verdict.json` de antes de este cambio, y la lógica de resume de
`campaign.py` (CAM-03) salta cualquier `run_id` con un verdict ya aceptado,
así que sin limpiarlos la campaña habría saltado silenciosamente el 100%
de las combinaciones). Resultado: **18/18 aceptadas, 0 rechazadas**,
`phase_label_train`/`operational_intensity`/`i_ridge_used` idénticos entre
las 3 repeticiones de cada kernel (determinismo confirmado), coincidiendo
con `phase_label_hint` del catálogo en los 6 casos.

## 11. Archivos relevantes

- `scripts/pacca/measure_warmup.py` -- el detector (umbral de CV al 5% con
  piso de ruido para señales GPU, segmentación binaria basada en meseta
  como método de respaldo automático, ARC-83/86).
- `scripts/pacca/generate_rodinia_synthetic_inputs.py` -- genera las
  entradas sintéticas de `myocyte`/`dwt2d`/`heartwall`, parametrizado con
  `--heartwall-frames`/`--dwt2d-size` (ARC-86).
- `orchestrator/schemas/kernels/catalog.yaml` -- `npb_bt`/`npb_cg`
  actualizados (ARC-81), `npb_lu` revertido (ARC-82), `dgemm_n2048`/
  `gpu_dgemm_n4096` alargados y verificados (ARC-83), los 6 kernels GPU
  restantes re-caracterizados (ARC-86).
- `docs/orchestator/agents/Registro_Cambios_Fuera_Plan_Original.md`,
  entradas `ARC-81`, `ARC-82`, `ARC-83`, `ARC-84` y `ARC-86`.
