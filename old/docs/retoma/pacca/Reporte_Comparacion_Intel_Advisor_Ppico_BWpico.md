# Reporte de comparación: Intel Advisor vs. instrumento propio (P_pico / BW_pico, paccaA100)

## 0. Propósito y método

Registrado como **ARC-125/ARC-126** en `docs/orchestator/agents/Registro_Cambios_Fuera_Plan_Original.md`.

Este documento persiste la comparación pedida explícitamente por el usuario
("quiero que me des la comparación completa, quiero poder ver cualquier tipo
de bug que tengamos a la clasificación") entre los picos de referencia que
usa el modelo Roofline del proyecto (`BW_pico`, `P_pico`, calibrados por
`kernels/stream/stream.c` y `kernels/ert/ert_probe.c`) y una medición
independiente de la misma cantidad con **Intel Advisor**, herramienta que
mide FLOPs/bytes por instrumentación binaria directa del *loop* (Pin-based),
sin pasar por `perf_event_open`/PMU/uncore -- es decir, un mecanismo de
medición genuinamente distinto al propio del proyecto, útil como validación
cruzada real.

Advisor se corrió sobre `dgemm_bench` (OpenBLAS, referencia de cómputo) y
`stream_c` (referencia de ancho de banda), con los mismos 6 núcleos que usa
la calibración de producción (`OMP_NUM_THREADS=6`), en `paccaA100`.
Comando: `advisor --collect=roofline` (survey + tripcounts&FLOP), reporte
extraído con `advisor --report=survey --format=csv --show-all-columns
--report-output=<archivo>.csv` (el flag `--show-all-columns` es necesario
para ver `Self GFLOPS`/`Self AI`/`Vector ISA`, ausentes por defecto;
`--report=roofline` siempre emite HTML, no CSV, independiente de
`--format`).

## 1. Tabla comparativa

| Magnitud | Nuestro instrumento (antes) | Intel Advisor | Diferencia | Nuestro instrumento (después de corregir) |
|---|---|---|---|---|
| `BW_pico` (`stream_c`, Triad, 6 núcleos) | 58.4-58.8 GB/s | 64.0-66.8 GB/s | ~12-15% | 58.8-59.5 GB/s (sin cambio significativo) |
| `P_pico` (`ert_probe`, 6 núcleos) | **71.8-71.9 GFLOP/s** | 577.7 GFLOP/s (`dgemm_bench`, AI=1.091 FLOP/byte) | **~8x** | 308.5 GFLOP/s (`-march=native`) → **417.8-555 GFLOP/s** (`-march=native -mprefer-vector-width=512`) |
| `Vector ISA` de `ert_probe` (columna Advisor) | `SSE2`, ancho 2/4 | -- | -- | `AVX2` tras `-march=native` solo; `AVX-512` tras agregar `-mprefer-vector-width=512` |
| `i_ridge` implícito (`P_pico/BW_pico`) | ~1.22 FLOP/byte | ~1.091 FLOP/byte (dgemm_bench, coherente) | -- | ~5.19 FLOP/byte tras la primera corrección; ~7.0-9.3 FLOP/byte tras la segunda (ver §3) |

`BW_pico` valida bien contra Advisor desde el principio (diferencia de
~12-15%, coherente con dos mediciones independientes del mismo fenómeno
físico -- ancho de banda de DRAM). `P_pico` NO validaba: una brecha de ~8x
es demasiado grande para ser sesgo de metodología entre dos herramientas
distintas: es un bug real del instrumento propio.

## 2. Causa raíz

`scripts/felix/build_stream_ert.sh` compilaba `ert_probe` sin ninguna flag
de arquitectura:

```
gcc -O3 -fopenmp -o ert_probe kernels/ert/ert_probe.c -lm
```

Sin `-march=native` (o equivalente), GCC nunca genera instrucciones más
anchas que la base garantizada de x86-64 (SSE2), pese a que Ice Lake-SP
soporta AVX-512 -- confirmado inspeccionando la columna `Vector ISA` del
propio reporte de Advisor sobre el *loop* caliente de `ert_probe`. El
harness de telemetría del proyecto (`telemetry/CMakeLists.txt`) sí usa
`-march=native` desde siempre; el kernel de calibración que lo mide no lo
tenía -- una inconsistencia real entre el instrumento y lo que mide.

`stream_c` (mismo script) no mostraba el mismo sesgo porque STREAM está
limitado por ancho de banda de DRAM, no por ejecución vectorial -- un *loop*
sin AVX-512 puede saturar memoria igual, así que la falta de la flag no lo
afectaba.

## 3. Corrección aplicada, en dos pasos

**Paso 1 (ARC-125):** `-march=native` agregado a ambas compilaciones
(`stream_c` y `ert_probe`). `ert_probe` pasó de 71.8 a 308.5 GFLOP/s (4.3x).
Inspeccionado de nuevo con Advisor: GCC eligió `AVX2` (ancho 4), no
`AVX-512` (ancho 8) -- heurística por defecto de GCC en objetivos de
servidor para evitar el *downclocking* que el ancho 512 puede causar.

**Paso 2 (ARC-126):** agregado `-mprefer-vector-width=512` explícito,
solo a `ert_probe`. Resultado: **417.8-555 GFLOP/s**. Se probó el mismo
flag en `stream_c` y se **descartó**: Triad se volvió más bajo y con mucha
más varianza entre corridas (49.3-69.2 GB/s vs. 58.8-59.5 GB/s estable sin
el flag) -- consistente con *downclocking* de AVX-512 sin ningún beneficio
para un kernel limitado por ancho de banda de DRAM, no por ejecución
vectorial. `stream_c` se dejó solo con `-march=native`.

Mismo diagnóstico aplicado también a NPB por consistencia, a pedido
explícito del usuario: `config/make.def.template` compilaba sin
`-march=native`. Se agregó (vía `sed` en `scripts/felix/build_npb.sh`),
deliberadamente **sin** `-mprefer-vector-width=512` -- NPB mezcla kernels
memory-bound (`mg`/`cg`) y compute-bound (`bt`) en un único `make.def`
compartido, y ya había precedente de regresión con ese flag en un kernel
memory-bound (`stream_c`).

## 4. ¿Esto sesga la clasificación Roofline?

**`BW_pico`: no afectado.** Diferencia de ~12-15% con Advisor consistente
desde antes de cualquier fix, dentro de lo esperable entre dos herramientas
que miden el mismo fenómeno por mecanismos distintos.

**`P_pico`/`i_ridge`: sí, de forma directa y grande.** El `i_ridge`
(`P_pico/BW_pico`) pasó de ~1.22 FLOP/byte (con el bug) a ~5.19 FLOP/byte
(paso 1) a un rango de ~7.0-9.3 FLOP/byte (paso 2, dependiendo del extremo
de GFLOP/s tomado dentro de 417.8-555). Un desplazamiento de esta magnitud
en el punto ridge reclasifica ventanas: cualquier ventana con intensidad
operacional entre ~1.2 y ~9 FLOP/byte, que con el `i_ridge` original
clasificaba `compute_bound`, pasa a `memory_bound` con el `i_ridge`
corregido (la intensidad ya no supera el nuevo ridge).

**`operational_intensity` (eje horizontal) del propio dataset: no
afectado.** Por invarianza de Roofline, la relación FLOPs/bytes de un mismo
algoritmo no depende del ancho vectorial usado para ejecutarlo -- los
contadores de hardware (`FP_ARITH_INST_RETIRED`) miden los FLOPs realmente
ejecutados, cualquiera sea el ancho. La falta de `-march=native` en NPB no
sesgaba la intensidad medida de esos kernels, solo su representatividad de
rendimiento absoluto (por eso se corrigió igual, mismo caso: sin motivo
para clasificar mal, pero corregido por consistencia con el resto del
proyecto).

## 5. Impacto en el dataset ya aceptado

El dataset de calibración/campaña de CPU ya aceptado (126/126, ARC-105 en
adelante) tiene ahora **dos motivos acumulados** para descartarse y
repetirse, ninguno relacionado con el otro:

1. Sin la señal real de `uncore` para `operational_intensity`/
   `phase_label_train` (ARC-119/123) -- antes de esa corrección, el
   proyecto usaba un proxy (`cache_misses × line_size`) que no captura
   *prefetching*.
2. Con un `i_ridge` subestimado ~4-8x por falta de vectorización AVX-512
   en el propio instrumento de calibración de `P_pico` (este documento).

**Pendiente:** recalibrar F0-F4+REF con todos los binarios corregidos
(`uncore` + AVX-512 de `ert_probe` + `-march=native` de NPB) en una sola
campaña de repetición, no en pasos separados -- ya planeado, no iniciado
a la fecha de este documento.

## 6. Checksums y verificación

Checksums de `pacca-a100` para `stream_c`/`ert_probe` actualizados en
`orchestrator/schemas/kernels/catalog.yaml` y
`orchestrator/schemas/kernels/class_c_stress/catalog_class_c.yaml`
(mismos binarios reusados ahí). Build confirmado **reproducible**: una
recompilación posterior con los mismos flags produjo checksums idénticos
byte a byte. 378/378 tests Python en verde tras el cambio (no dependen del
contenido del binario). No requirió tocar `postprocess.py` ni la lógica de
clasificación -- el fix está enteramente en cómo se compilan los binarios
de calibración.
