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

## Jobs propios, orden real de ejecución

| orden | job | qué hace | estado a 2026-08-26 (mañana) |
|---|---|---|---|
| — | **6412** | `ptrchase` + `phasic_*` | ✅ **COMPLETED**, 320/320 aceptadas, 0 rechazos |
| — | **6530** | Rejilla fina CPU, 7 niveles nuevos | ✅ **COMPLETED**, 638 aceptadas / 82 saltadas / 0 rechazadas |
| ahora | **6575** | Tamizaje CPU v2: ~79 kernels, `--memory-touched` 10× LLC | PENDING (Priority), detrás de los 25 ajenos |
| después | **6571** | Clasificación cuello de botella GPU (`ncu` DRAM% vs SM%) | PENDING (Priority), detrás de 6575 |
| — | *(jobs ajenos)* | `mixed_precision_stencil` | uno RUNNING (6547), resto PENDING |

**Decisión aprobada 2026-08-26**: esperar a que termine 6571 antes de
decidir el catálogo final de GPU y lanzar la campaña de dataset. Sigue en
pie — nada de esto la cambia todavía.

## Jobs terminados — resultados

**6412 (`ptrchase` + `phasic_*`)**: 320/320 aceptadas. Resultado numérico
(α de `ptrchase`) pendiente de recalcular sobre esta campaña — la sonda
rápida previa (job 6542) ya había dado α≈0.096–0.144, por confirmar con
estos datos reales.

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
