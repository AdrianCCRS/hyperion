# Estado de la cola de Slurm — documento vivo

**Propósito.** Con varios jobs propios encolados, jobs ajenos compartiendo
la cuenta `latorresn`, y decisiones que dependen de resultados que aún no
llegan, es fácil perder el hilo de qué está corriendo, qué espera a qué,
y qué falta ni siquiera encolar. Este documento se actualiza cada vez que
cambia el estado — no es un registro histórico como
`Registro_Cambios_Fuera_Plan_Original.md`, es el estado actual.

**Última actualización: 2026-08-26** (mañana) — 6412 y 6530 terminaron
limpios de madrugada, antes de que arrancara el primer job ajeno (6547,
09:xx). Resultados en §"Jobs terminados" abajo.

---

## Regla de fondo sobre el nodo

Todo el trabajo pesado corre en **paccaA100** (partición `GPU`), no en
`pacca01`: es el único nodo con RAPL/uncore funcionando y donde se hizo
toda la caracterización de CPU y GPU del proyecto — los CPUs de ambas
particiones son distintos (Xeon Gold 5315Y vs 5320), así que nada medido
en pacca01 es comparable. **No lanzar nada en pacca01 sin consultar
primero** (lección del 2026-08-25: se lanzó ahí sin preguntar y fue un
error).

`paccaA100` se pide siempre `--exclusive`: los contadores RAPL/uncore son
de ámbito de socket y cualquier otro proceso contamina la medición. Eso
significa que **solo un job nuestro corre a la vez**, todo lo demás
espera en cola — de ahí que el orden importe tanto.

La cuenta `latorresn` es **compartida** con otro usuario. Sus jobs
(`mixed_precision_stencil`, IDs 6544–6568 a la fecha) no se cancelan ni se
reordenan — eso no es una decisión que nos corresponda tomar. Con
prioridades iguales, Slurm ordena por ID de envío: lo que ya tengamos
encolado con ID menor que el de ellos entra antes; lo que encolemos
después, entra detrás.

---

## Jobs propios, orden real de ejecución (actualizado, noche 2026-08-26)

| orden | job | qué hace | estado |
|---|---|---|---|
| — | **6594** | Campaña completa (etiqueta de verdad real) sobre los 9 sobrevivientes del tamizaje CPU v2 — 324 corridas | ✅ **COMPLETED**, 324/324 aceptadas, 0 rechazos. **7 de 9 kernels con margen real de EDP** (hasta −7.09%) — ver `Estrategia_CPU_Fase2.md` §6.octies |
| **en pausa a propósito** | **6600** (era 6595) | Tamizaje α GPU v2 — 43 candidatos | **CANCELADO (2026-08-26 noche), decisión de alcance, no de nodo.** C8 (§7.bis de `Estrategia_CPU_Fase2.md`) mostró que el clasificador de fase por ventana SÍ funciona donde hay mezcla real (F1=0.538 vs 0.170 trivial) — la vía para mejorarlo más es ampliar el catálogo CPU con kernels que produzcan esa mezcla, no seguir el eje GPU en paralelo. **GPU queda en standby**; se retoma cuando el frente CPU cierre o se agote su margen |
| — | **6601** | Tamizaje α GAP (`bfs`/`pr`) directo en `paccaA100` | ✅ **COMPLETED**, 10/10, frecuencia dentro de 5% en las 10. **Resultado negativo**: `bfs` α=0.738, `pr` α=0.690 — muy por encima del umbral 0.226, no candidato de catálogo por margen de EDP. Ver nota resuelta en `Estrategia_CPU_Fase2.md` §6.sexies |
| corriendo | **6612** | Tamizaje α LULESH+HPCG, directo en `paccaA100` — compilados limpio (commits `3e01c40`/`1146024`), 2do y 3er candidato del pivote de catálogo tras GAP | PENDING/RUNNING, ~45 min de presupuesto |

## Jobs propios, orden real de ejecución (histórico, mañana 2026-08-26)

| orden | job | qué hace | estado a 2026-08-26 (mañana) |
|---|---|---|---|
| — | **6412** | `ptrchase` + `phasic_*` | ✅ **COMPLETED**, 320/320 aceptadas, 0 rechazos |
| — | **6530** | Rejilla fina CPU, 7 niveles nuevos | ✅ **COMPLETED**, 638 aceptadas / 82 saltadas / 0 rechazadas |
| — | **6579** | Pre-vuelo Clase C: `npb_cg`/`npb_mg` B vs C (8× memoria) | ✅ **COMPLETED**, 36/36. α baja en ambos (cg 0.765→0.530, mg 0.409→0.335) pero ninguno cruza 0.226 — "el eje de tamaño funciona" según criterio pre-registrado, sin cruzar el umbral en Clase C |
| — | **6575** | Tamizaje CPU v2: ~79 kernels, `--memory-touched` 10× LLC | ✅ **COMPLETED**. **9 sobrevivientes** (de 0 en v1): `Stream_MUL/TRIAD/ADD`, `Lcals_FIRST_SUM/TRIDIAG_ELIM`, `Polybench_JACOBI_1D/FDTD_2D`, `Basic_DAXPY/INIT3` — ver tabla completa en el reporte |
| — | **6571** | Clasificación cuello de botella GPU (`ncu` DRAM% vs SM%) | ✅ **COMPLETED**, parser corregido. **43 de 75 kernels MEMORY_BOUND** (DRAM%>SM%, DRAM%≥30%) — ver lista completa en el reporte |
| — | **6583** | Triage GAP Benchmark (BFS/PR) en **pacca01** | ❌ **COMPLETED sin señal útil** — bloqueado por falta de permiso de escritura de frecuencia en pacca01 (`Permission denied`, no es un resultado sobre los kernels). Binarios sí quedaron compilados en `~/hyperion-kernels/libexec/gapbs/`. Plan: tamizar `bfs`/`pr` directo en `paccaA100` cuando se libere, sin pasar por pacca01 |
| — | *(jobs ajenos)* | `mixed_precision_stencil` | uno RUNNING (6552), resto PENDING |

**Decisión aprobada 2026-08-26**: esperar a que termine 6571 antes de
decidir el catálogo final de GPU y lanzar la campaña de dataset. Sigue en
pie — nada de esto la cambia todavía.

## Jobs terminados — resultados

**6412 (`ptrchase` + `phasic_*`)**: 320/320 aceptadas. Dos resultados
reales, distintos entre sí:

*`ptrchase` confirma el umbral, con datos de campaña completa (no la
sonda rápida):*

| ventana | α | r² |
|---|---:|---:|
| F0–F4 completo | 0.122 | 0.990 |
| F0–F1 (la que la política usaría) | **0.097** | 1.000 |

Ambos por debajo de 0.226. Confirma lo que la sonda de 6542 ya sugería,
ahora con 10 repeticiones por nivel y ajuste r²≈1 en la ventana estrecha.
**Es el segundo kernel viable del catálogo**, junto a `npb_mg`.

*`phasic_p010/p100/p1000` son un resultado de OTRO tipo — no compiten por
el umbral, lo pulverizan:* EDP/F0 cae a **0.82–0.83** en F4 (17–18% de
ahorro), con α≈0.002–0.003. Pero ojo: **duración casi constante en las
tres variantes de periodo de fase** (20.27→20.42–20.47 s, <1% de
variación entre F0 y F4) — es exactamente el diseño del kernel (fase fija
por *tiempo*, no por trabajo, ver `Estrategia_CPU_Fase2.md`), así que su
α bajo es por construcción, no evidencia de que cargas reales se
comporten así. Útil como referencia/control de que el instrumento
detecta memory-boundness limpio cuando existe, no como candidato de
catálogo.

**6530 (rejilla fina CPU, 3200→2000 MHz)**: 638/720 aceptadas (82
saltadas, 0 rechazadas). **Confirma la hipótesis que motivó la campaña**:
`npb_mg` es el único de los 9 kernels con un óptimo de EDP fuera de F0.

| kernel | mejor nivel | EDP/F0 | lectura |
|---|---|---:|---|
| **`npb_mg`** | **S3000 (3000 MHz)** | **0.9927** | único con mínimo real fuera de F0, −0.73% |
| los otros 8 | F0 (3200 MHz) | 1.0000 (monótono creciente al bajar) | ninguno se beneficia de bajar el reloj |

Margen pequeño (0.73%) pero real y no monótono — no es ruido de medición,
es un mínimo genuino en S3000, con S3100 y S2900 a ambos lados subiendo
de nuevo. Todos los demás kernels (npb_bt, npb_cg, npb_sp, npb_ft, npb_lu,
dgemm_n2048, rodinia_lavamd_omp, rajaperf_polybench_3mm_omp) degradan
monótonamente su EDP al bajar frecuencia — sin excepción, ninguno se
acerca a un óptimo distinto de F0 en esta ventana.

---

## Lo que YA está listo pero NO encolado (esperando decisión, no nodo)

**Campaña final de GPU (17 kernels, v2).** El manifiesto
(`campaign_pacca_gpu_final_dataset_v2.yaml`) y el catálogo (6 kernels
nuevos de RAJAPerf-CUDA con OI real medido por `ncu`, job 6528) ya están
completos y el manifiesto **carga sin errores** — podría lanzarse hoy.
**No se lanza todavía a propósito**: si 6571 encuentra más candidatos
memory-bound entre los 79 kernels CUDA, el dataset nacería incompleto
(mismo error que ya se cometió una vez con el job 6529, cancelado por
esto). Se espera el resultado de 6571 antes de decidir el catálogo
definitivo y encolar.

## Lo que NO existe todavía y depende de resultados pendientes

- **Campaña final de CPU con catálogo ampliado.** No hay manifiesto
  escrito. Depende de qué sobreviva a 6575 (tamizaje v2) y de lo que
  confirme 6530 (rejilla fina) sobre `npb_mg`. No se puede escribir hasta
  tener ambos resultados.
- **Piloto LOKO repetido** sobre los datasets nuevos (CPU y GPU por
  separado, sin mezclar los dos ejes — ver nota de estilo abajo). Es el
  único paso que dice si el catálogo ampliado se traduce en un modelo que
  gane al trivial.

## Mejoras del modelo identificadas, sin hacer, no necesitan nodo

Se pueden hacer en cualquier momento, en paralelo a lo que corre en el
clúster:
- Quitar `ref_running_ratio` del vector de features (varianza cero,
  confirmado en `loko_feature_diagnostic.py`).
- Enriquecer features con percentiles/dispersión de la corrida de
  referencia (p10/p50/p90, CV), no solo la media — ataca el N efectivo
  por el lado de la riqueza del punto.
- Probar predicción en espacio logarítmico o restringir niveles
  candidatos a la región accionable del EDP.

---

## Nota de estilo (recordatorio explícito, pedido 2026-08-25)

**No mezclar el modelo de GPU con el de CPU al hablar de resultados** —
son ejes separados con catálogos, umbrales y estados distintos. Confundir
uno con otro genera confusión real, ya pasó en esta sesión.

---

## Cómo actualizar este documento

Cada vez que se encole, cancele, o termine un job propio, o cambie una
decisión de las de arriba, editar este archivo en el mismo turno — no
esperar a que se acumulen varios cambios. La fuente de verdad del estado
en vivo sigue siendo `squeue`/`sacct` en pacca; este documento es el
resumen legible entre sesiones, no reemplaza consultarlo antes de una
acción importante.
