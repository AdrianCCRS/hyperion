# Estrategia GPU — Fase 2 (documento para el director)

**Propósito.** No es un cambio de objetivos del anteproyecto — es la
corrección de *cómo* los estamos cumpliendo, basada en evidencia empírica
propia contrastada contra la literatura ya citada en el marco teórico.
§9 mapea cada punto a los Objetivos Específicos 1–4; §10 da la lista de
pruebas que permite verificar (o refutar) cada afirmación de este
documento.

Fuente completa: `docs/general/PLAN_MAESTRO_FASE2.md`, Anexos K, L, M, N.
Este documento es la síntesis ejecutiva.

> **Nota de auditoría (2026-08-24).** Este documento fue revisado contra
> los datos crudos y se corrigieron cinco errores de la versión anterior:
> (a) atribución de las 144 corridas del job 6462, (b) el rango de ahorro
> decía 8.7% cuando el mínimo medido propio es 7.7%, (c) el conteo de
> kernels de RAJAPerf estaba inflado al doble por un artefacto de conteo,
> (d) una afirmación causal sobre la grilla presentada como establecida
> cuando aún no hay dato, (e) una afirmación de ausencia de fases
> intra-corrida que es circular (§3). Se documentan aquí para que la
> corrección quede trazable, no se borran.

---

## 1. Qué se hizo (Objetivo Específico 1)

Telemetría de bajo nivel (NVML para GPU; RAPL+Perf para el lado CPU
delegado) sobre **7 kernels GPU**, repartidos en dos campañas:

| job | kernels | niveles GPU | niveles CPU | reps | corridas |
|---|---|---|---|---|---|
| 6462 | 4 (`gaussian`, `dgemm_n4096`, `heartwall`, `lavamd`) | 6 | 2 (REF, F4) | 3 | 144 |
| 6463 | 3 (`myocyte`, `backprop`, `dwt2d`) | 6 | 1 (REF) | 3 | 54 |

Ambas 100% aceptadas (144/144 y 54/54, 0 rechazos). Antes de correrlas se
corrigió un problema de reproducibilidad de `nvcc` no detectado hasta
entonces (ARC-193) y se verificaron los checksums de los binarios.

## 2. El error de métrica que se corrigió, y por qué importa contarlo

La primera medición sumó **GPU + paquete CPU + DRAM** (energía total del
nodo). Con esa métrica el DVFS de GPU parecía no poder pagar nunca: un
recorte de reloj de 6.7× compraba solo 10–41% de caída de potencia
(Anexo L).

**Esa métrica es más estricta que la de la línea de trabajo con la que nos
comparamos.** Los trabajos de predicción de frecuencia óptima en GPU
(`Fan2020`, `Guerreiro2019`) miden **energía de GPU**, que es lo que una
política de GPU realmente controla; el piso de potencia del host es una
propiedad del nodo de medición, no del mecanismo bajo estudio. Con esa
métrica el resultado se revierte (Anexo M, CPU=REF, contra "siempre F0"):

| kernel | mejor nivel | ahorro E_gpu | costo de tiempo | ganancia EDP |
|---|---|---|---|---|
| `rodinia_lavamd` | F1 (1110 MHz) | **25.11%** | +10.02% | **17.60%** |
| `rodinia_heartwall` | F1 | **18.24%** | +33.12% | 1.03% |
| `rodinia_gaussian` | F1 | **15.35%** | +25.62% | 1.69% |
| `rodinia_myocyte` | F1 | **7.66%** | +28.02% | 0.00% |

Rango medido propio: **7.7–25.1%**, comparable a lo publicado (8.7–23.1%
en entrenamiento DNN; 20.2–26.7% con escalado consciente de la aplicación
en V100/A100).

**Matiz honesto que hay que declarar, no esconder:** no existe un estándar
único en el campo sobre qué energía medir. Un survey de técnicas de DVFS
en GPU documenta que distintos trabajos miden distinto, y reporta trabajo
previo enfocado en energía *a nivel de sistema* que concluye que el DVFS
de GPU afecta la energía del sistema **menos** que el DVFS de CPU — es
decir, **el hallazgo del Anexo L es un resultado conocido y publicado, no
un artefacto de nuestro montaje**. Por eso este trabajo **reporta ambas
métricas**: energía de GPU como primaria (comparabilidad con la línea de
predicción de frecuencia), energía total del nodo como limitación
declarada del alcance. Reportar las dos es más fuerte que cualquiera sola.

## 3. Granularidad del modelo: por qué no clasifica fase intra-ejecución

**Lo que se creía necesario:** un clasificador de fase por ventana,
análogo al de CPU.

**El impedimento técnico (CAT-10), y su causa real:** en GPU la intensidad
operacional se declara **estática por kernel** en el catálogo, medida
offline con `ncu`. No es una elección de conveniencia: los contadores de
tráfico DRAM (`dram__bytes.sum`) no están en NVML (API de monitoreo) sino
en la Profiling API de CUPTI, que **requiere replay del kernel** —
relanzarlo para multiplexar contadores. Eso (a) distorsiona el tiempo y la
energía que este trabajo mide, y (b) entrega el dato **por lanzamiento de
kernel**, no por ventana de tiempo: para nuestros kernels reales, que son
uno o pocos lanzamientos largos, ni pagando el costo se obtiene
intensidad dinámica por ventana. Es una asimetría de categoría respecto de
`perf_event_open`+uncore en CPU, no de presupuesto. **No es un permiso
caído**: se verificó que `NVreg_RestrictProfilingToAdminUsers` no está
activo en pacca — `ncu` corre sin privilegios.

**Lo que el dato propio muestra (Anexo K, `gpu_policy_headroom.py`):** el
óptimo **de corrida completa** es constante por kernel y varía **entre**
kernels.

> **Límite lógico de esta afirmación, declarado explícitamente.** De que
> el óptimo de corrida completa sea constante **no se sigue** que no
> exista alternancia de fases dentro de la corrida: es precisamente lo
> que la instrumentación de GPU no puede observar. Afirmar su ausencia
> sería circular. Lo defendible es más débil y suficiente: **con la
> telemetría disponible, no hay evidencia de alternancia explotable, y no
> existe vía de medición que pueda producirla.** El test V4 (§10) es el
> único que puede cerrar este hueco, usando `gpu_phasic` — que alterna
> fases por construcción y emite marcas de verdad (`PHASE`,
> `T0_MONOTONIC_NS`) cruzables offline contra `gpu_power_mw`.

**La granularidad elegida es la que usa la literatura publicada:**

- `Guerreiro2019` (ya citado): clasifica la **aplicación**; observa
  contadores **en una sola frecuencia de referencia** y predice el resto.
  16% de ahorro promedio, 0.74% de desviación respecto al óptimo real.
- `Calore2017` (ya citado): probó ajuste función-por-función y lo abandonó
  explícitamente (*"clock tuning on a function-by-function basis is not
  convenient"*), pivotando a frecuencia constante por programa.
- `Antici2024` (ya citado, MCBound, producción en Fugaku): clasifica el
  **job completo**, F1-macro ≥ 0.89.

El Objetivo 2 pide un modelo clásico que clasifique "las fases de
ejecución… basándose en la telemetría", **en tiempo de ejecución y con
baja latencia**. Nada en ese texto exige que la fase cambie *dentro* de
una corrida; exige que la inferencia sea en vivo y barata. Eso se
mantiene: lo que cambia es la unidad clasificada (carga/kernel, no ventana
de 1 ms), que es la unidad de los tres sistemas publicados citados.

## 4. Arquitectura propuesta

- **Entrada:** telemetría observada en **un solo nivel de referencia**
  (F0) — `gpu_util_pct`, `gpu_mem_util_pct`, potencia sobre reposo, y el
  cociente `gpu_mem_util_pct / gpu_util_pct` como proxy de
  memory-boundness — más la frecuencia candidata como feature.
- **Salida:** dos regresiones por par (carga, nivel): `E(f)/E(F0)` y
  `T(f)/T(F0)`.
- **Política:** minimizar EDP (fiel al Objetivo 4) y, como variante,
  minimizar energía sujeta a un presupuesto de degradación explícito —
  el presupuesto queda como **parámetro de política, no horneado en el
  entrenamiento** (cambiarlo no exige reentrenar).
- **Anti-fuga:** prohibido todo lo que provenga de la corrida del nivel
  candidato (su tiempo, energía o potencia). Solo features de F0 + la
  frecuencia candidata. Verificable por test (V5, §10).
- **Evaluación:** leave-one-kernel-out contra la **mejor constante única**,
  elegida **solo sobre los folds de entrenamiento** — si se elige mirando
  el kernel de prueba, el baseline hace trampa y el modelo parece mejor de
  lo que es (V6, §10).

**Por qué regresión por par y no clasificación por kernel:** con N cargas
y M niveles hay N×M muestras en vez de N, y el presupuesto queda
parametrizable. Es además la formulación de `Fan2020`/`Guerreiro2019`.

## 5. El margen real sobre la mejor constante, medido

`gpu_policy_headroom.py` compara tres políticas para cuantificar cuánto
puede ganar un modelo por encima de una regla trivial. **Tabla sobre 6
kernels** — excluye `rodinia_backprop`, cuya energía de GPU en F0 es
8.2 J (dos órdenes de magnitud bajo el resto: sus porcentajes son ruido
dividido por casi cero):

| presupuesto de degradación | mejor constante | oráculo | margen del modelo |
|---|---|---|---|
| ≤4% | F0 → 0% | 0.58% | **0.58 pts** |
| ≤15% | F0 → 0% | 4.76% | **4.76 pts** |
| sin límite | F1 → 7.71% | 11.41% | **3.70 pts** |

Lectura honesta: con la grilla de 6 niveles, **una sola constante (F1)
captura el 68% del oráculo**, y bajo un presupuesto estricto de 4% el
margen aprovechable es de apenas 0.58 puntos.

**Hipótesis en prueba (no resultado):** que la causa sea la resolución de
la grilla y no la física — el salto F0→F1 es de 300 MHz y cuesta 10–33%
de tiempo, así que bajo presupuesto estricto casi ningún nivel es
elegible. **Esto lo decide el job 6471, todavía sin correr** (V2, §10).
Si la grilla fina no abre margen, la conclusión correcta es que el
presupuesto de 4% es inalcanzable en esta plataforma y que la evaluación
debe reportarse por EDP — donde `lavamd@F1` ya gana 17.6%.

## 6. Encolado (esperando nodo)

Los tres jobs están **PENDING** detrás de un job ajeno que lleva ~15 h
ocupando el nodo. Los manifiestos están validados y encolados; lo que
falta es turno de máquina, no diseño.

- **Job 6471** — grilla fina, 210 corridas: 4 escalones nuevos entre F0
  (1410 MHz) y F1 (1110 MHz). Márgenes de potencia de esos 4 niveles
  **interpolados, no medidos** (V1, §10).
- **Job 6472** — barrido de tamaño de `dwt2d`, 90 corridas: 5 tamaños
  (192–16384 px), mismo binario y checksum, márgenes **medidos**.
- **Instrumento de tamizaje ya validado** (Anexo K.4): `nvidia-smi -lgc`
  escala solo el reloj de SM, no el de memoria, así que el margen debería
  vivir en kernels limitados por ancho de banda. Verificado gratis con la
  calibración: `gpu_stream_bw` da α=0.071 frente a α=0.6–0.8 de los
  kernels de cómputo.

## 7. Riesgos abiertos, sin adornar

1. **Márgenes interpolados del job 6471** — si están mal, el síntoma es
   rechazo masivo I10 (0 ventanas `gpu_telemetry`), que además **borra la
   energía de GPU**, hoy la métrica primaria. Detectable, no silencioso.
2. **El reloj de memoria nunca se probó** como segundo mando. `Fan2020` y
   `Guerreiro2019` lo escalan junto al de núcleo; podría ser el mando
   relevante para `dwt2d`/`stream_bw`, que no respondieron al de SM (V8).
3. **Con 7–11 kernels el LOKO entrena sobre 6–10** — es un piloto, no un
   resultado estadísticamente robusto.
4. **La OI de los 3 tamaños intermedios de `dwt2d` está interpolada**, no
   medida con `ncu`. No se usa como feature del modelo (evita CAT-10 por
   diseño), pero está declarado en el catálogo.
5. **El ajuste de α resultó inválido para los kernels del tamizaje**
   (r²=0.53–0.63, Anexo L.1): el modelo de Amdahl no describe kernels que
   saturan a bajo reloj. α sirve como tamiz cualitativo, **no** como
   número reportable para esos casos.
6. **La variante CUDA de RAJAPerf no está compilada** — el impulso de §8
   requiere un build nuevo antes de rendir en GPU.

## 8. Impulso: el banco de kernels disponible es mayor que el usado

Cuántos kernels usa cada trabajo citado: `Guerreiro2019` 35 (5 suites),
`Calore2017` **2** (una sola app, y aun así resultado real y citable),
`Hebbar2022` 43 (SPEC CPU2017 — licencia paga, no reproducible por
nosotros), `Antici2024` producción a escala Fugaku. El rango es enorme:
**el número no es lo que decide, sino la diversidad de régimen cubierta.**

Contra ese criterio, el catálogo GPU actual está sesgado a
compute/balanced, con 1–2 candidatos memory-bound reales. **RAJAPerf ya
está descargado en pacca** y el catálogo usa **1** de sus kernels. Conteo
verificado (2026-08-24, despojando sufijos de backend correctamente —
el conteo anterior estaba inflado al doble):

| categoría | kernels distintos |
|---|---|
| `apps` | 22 |
| `basic` | 20 |
| `polybench` | 13 |
| `lcals` | 11 |
| `algorithm` | 8 |
| `comm` | 6 |
| `stream` | 5 |
| **total** | **85** |

`raja-perf.exe` corre cualquiera con `-k NOMBRE -v <variante>`: agregar un
kernel es un wrapper, no una compilación. **Pero para GPU falta compilar
la variante CUDA** (hoy solo existe la OpenMP): build nuevo, de bajo
riesgo pero no gratis (mismo procedimiento que `gpu_phasic`: verificar
reproducibilidad, despojar, fijar checksum).

## 9. Mapeo a los Objetivos Específicos

| Objetivo | Estado | Evidencia |
|---|---|---|
| 1. Caracterizar comportamiento y consumo bajo distintos estados de frecuencia (NVML) | **Cumplido** (198 corridas, 0 rechazos), ampliándose | §1; jobs 6462/6463; 6471/6472 encolados |
| 2. Clasificador ML en vivo, baja latencia | **Granularidad reinterpretada a carga/kernel**, con precedente publicado triple | §3, §4; Anexo K |
| 3. Daemon de espacio de usuario con política DVFS | Sin cambios de diseño; pendiente de implementar sobre el modelo de §4 | V7 mide su presupuesto de latencia |
| 4. Evaluación por EDP contra gobernador nativo | EDP es métrica primaria (§5). En GPU el "nativo" es el autoboost del driver (`native_governor`/REF), incluido en todas las tablas | §5; `gpu_policy_headroom.py` |

## 10. Lista de pruebas — cómo verificar o refutar esta propuesta

Cada prueba tiene un criterio de paso explícito **y** qué se concluye si
falla. Ninguna afirmación de este documento debería sostenerse si su
prueba correspondiente falla.

| # | Qué verifica | Cómo | Pasa si | Si falla |
|---|---|---|---|---|
| **V1** | Márgenes interpolados de los 4 niveles nuevos (§6, riesgo 1) | Job 6471: contar rechazos I10 por nivel en `campaign_metadata.json` | Rechazos en G1–G4 comparables a los niveles medidos (F0/F1) | Remedir línea de reposo con sonda fina (≥30 s/nivel) y recalcular márgenes antes de usar el dataset |
| **V2** | La causa del margen angosto es la grilla, no la física (§5) | Correr `gpu_policy_headroom.py` sobre el dataset de 6471 con `--max-slowdown-pct 4` | Margen del modelo > 2 pts a ≤4% | La hipótesis se refuta: el presupuesto de 4% es inalcanzable aquí; reportar por EDP y declararlo como límite de plataforma |
| **V3** | Reproducibilidad del ahorro de `lavamd@F1` (§2) | Comparar el 25.11% de 6462 contra el mismo punto en 6471 | Diferencia dentro del CV entre repeticiones (≈1–3%) | El número de 6462 no es reproducible: reauditar antes de citarlo |
| **V4** | **Cierra el hueco lógico de §3**: ¿existe alternancia intra-corrida explotable? | Cruzar offline las marcas `PHASE`/`T0_MONOTONIC_NS` de `gpu_phasic` contra `gpu_power_mw` y `gpu_sm_clock_mhz` por ventana | Si las fases son distinguibles en potencia **y** su nivel óptimo difiere → hay alternancia explotable | Si no son distinguibles o el óptimo no difiere: queda confirmado que la granularidad por carga es la correcta, con evidencia positiva y no por ausencia de medición |
| **V5** | Anti-fuga de etiqueta (§4) | Test unitario: entrenar inyectando a propósito una feature del nivel candidato | El test **falla** ruidosamente (guardarraíl activo) | El guardarraíl no protege: cualquier resultado del modelo queda invalidado hasta arreglarlo |
| **V6** | Baseline honesto en LOKO (§4) | Verificar que la "mejor constante" se elige solo con folds de entrenamiento | La constante elegida puede diferir por fold | El baseline hace trampa y el margen reportado del modelo está inflado |
| **V7** | Latencia de inferencia (Objetivo 2) | Medir p50/p99 de una inferencia sobre el modelo entrenado | p99 « período de decisión del daemon | El modelo no sirve para el Objetivo 3 aunque acierte: elegir uno más liviano |
| **V8** | ¿El reloj de memoria es un mando disponible? (riesgo 2) | `nvidia-smi -i 0 --query-supported-clocks=mem` y probar `-lmc` bajo carga | Hay >1 reloj de memoria y `-lmc` se aplica y se sostiene | El segundo mando no existe en esta A100: cerrar esa línea y declararlo |
| **V9** | Sanidad de la energía de GPU | `gpu_energy_valid == 1` en todas las filas `gpu_telemetry` | 100% válidas (ya verificado en 6462/6463) | La métrica primaria pierde piso: no reportar ahorros hasta resolverlo |
| **V10** | Consistencia documento↔dato | Reejecutar `gpu_policy_headroom.py` y comparar contra las tablas de §2/§5 | Coinciden | Actualizar el documento: los números del papel deben salir siempre del script, nunca copiarse a mano |

---

## Referencias

Ya en `docs/libro/main.tex`: `\cite{Guerreiro2019}`, `\cite{Calore2017}`,
`\cite{Antici2024}`, `\cite{Williams2009}`.

Pendientes de agregar (ver `docs/libro/referencias_pendientes_dvfs_gpu.md`,
que distingue autores verificados de no verificados): `Fan2020`,
`Mei2016`, y el survey de técnicas de DVFS en GPU citado en §2 — **cuya
autoría debe confirmarse antes de citarlo**, igual que el resto de
entradas marcadas en ese archivo.
