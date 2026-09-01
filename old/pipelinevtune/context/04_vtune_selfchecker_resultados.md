# 04 — Resultados reales de `vtune-self-checker.sh` en Cartagena

Esta es evidencia empírica registrada del nodo, no una suposición sobre lo que
"debería" funcionar por ser Ice Lake. Todo el resto de la planificación se apoya en
esto, no en la documentación general de VTune.

## Carga del módulo

```bash
module purge
module load vtune/2023
which vtune
vtune --version
```

## Análisis confirmados disponibles

- Performance Snapshot
- Hotspots (User Mode Sampling)
- **Hotspots with Hardware Event-Based Sampling**
- **HPC Performance Characterization**
- Hotspots with HW Event-Based Sampling + Call Stacks
- Threading with HW Event-Based Sampling

Los dos que el pipeline necesita (Hotspots HW y HPC Performance Characterization)
**están confirmados funcionales**. Esto reemplaza a la prueba de humo manual que
`PLAN.md` pedía como Fase 0 — ya está hecha, no hace falta repetirla, pero sí
verificar en la primera corrida real sobre un kernel de NPB que las métricas
salgan pobladas y no todo `NA` (ver Fase 0 actualizada).

## Análisis NO disponibles

- Microarchitecture Exploration
- Memory Access
- GPU Compute/Media (no se necesita para este pipeline, CPU-only)

## Restricciones observadas y su consecuencia técnica

| Restricción | Consecuencia para este pipeline |
|---|---|
| **Eventos Uncore no disponibles** | `DRAM Bound` (nivel fino del Top-Down, atribuido al controlador de memoria) puede salir `NA` o degradado. `Memory Bound` de nivel superior (basado en *stalls* de core, no necesariamente uncore) debería sobrevivir — **verificar empíricamente, no asumir**. Microarchitecture Exploration y Memory Access dependen fuertemente de uncore, coincide que ambos aparecen como no disponibles |
| **`perf_event_paranoid` limitado** | Puede restringir qué eventos de PMU están accesibles incluso dentro de Hotspots HW/HPC Performance; si algún campo específico sale `NA` de forma persistente, esta es la primera causa a revisar antes que un bug del parser |
| **`kptr_restrict` activo (símbolos del kernel restringidos)** | Afecta la resolución de símbolos en código de kernel de Linux, no en el userspace de los binarios NPB — no debería afectar `dominant_function` de los kernels NPB, que corren en espacio de usuario |

## Consecuencia directa para el eje STREAM/DGEMM (ver `02_decisiones.md`, D3 revisada)

Sin uncore, un ancho de banda de memoria calculado a partir de contadores de
hardware de VTune (si es que HPC Performance Characterization expone alguno así)
es sospechoso en este nodo. La solución que evita depender de uncore por completo:

- **STREAM ya reporta su propio ancho de banda por software** (temporizado con
  reloj de pared, tamaño de arreglo conocido) en su salida estándar — no necesita
  ningún contador de hardware. Usar ese número, no uno derivado de VTune.
- **El driver DGEMM ya construido en este proyecto también calcula y reporta su
  propio GFLOP/s por software** (mismo mecanismo: tamaño de problema conocido +
  tiempo medido). Tampoco depende de uncore.
- El `DP GFLOPS` que reporta VTune vía HPC Performance Characterization sí debería
  ser confiable sin uncore (viene de contadores de ejecución de FP en el core, no
  del controlador de memoria) — usar el de VTune para los kernels NPB reales, ya
  que ahí no hay un número autorreportado por el binario.

**Limitación honesta que hay que dejar dicha, no escondida:** para los kernels de
NPB (a diferencia de STREAM/DGEMM, que se miden a sí mismos) no hay forma de
obtener bytes movidos/ancho de banda real sin uncore y sin LIKWID. Esto significa
que el eje "Roofline vs. techos" para los kernels de NPB en este nodo **no puede
graficar su posición horizontal completa (AI = FLOP/byte)** — solo se puede
comparar su `DP GFLOPS` medido contra el techo de cómputo (DGEMM) y dejar que el
veredicto de memoria salga del eje nativo de VTune (Memory Bound del Top-Down), no
de un AI calculado. Ver `PLAN.md` Fase 4 actualizada para cómo se maneja esto sin
inventar un número que VTune no puede dar en este nodo.

## Addendum — Fase 0 real ejecutada en este nodo (2026-08-07)

Corrida real de `hpc-performance` sobre `ep.C.x` (NPB clase C) y sobre
`STREAM/stream_omp`, ambos con `OMP_NUM_THREADS=8 OMP_PLACES=cores
OMP_PROC_BIND=close`, reserva exclusiva `srun -p GPU -w paccaA100 --exclusive`.
Capturas completas en `tests/unit/fixtures/real_summary_ep_C.{txt,csv}` y
`real_summary_stream.{txt,csv}`.

**Corrección a lo anticipado arriba — `DRAM Bound` SÍ funciona:** contra lo que
este documento anticipaba como más probable (`NA` por falta de uncore), `DRAM
Bound` salió poblado con un número real y coherente en ambos casos: `0.0%` en EP
(kernel con working set diminuto, consistente con casi nada de tráfico a DRAM) y
`67.7%` en STREAM (kernel deliberadamente memory-bandwidth-bound). No es un campo
roto ni degradado en este nodo — la limitación de uncore documentada arriba no le
impide a VTune calcular este nivel fino del Top-Down aquí. Queda como hallazgo
empírico verificado, reemplaza la incertidumbre anterior.

**Nombres de campo confirmados literalmente en `vtune -report summary -r <dir>`
(hpc-performance), texto y CSV, VTune 2023.0.0 build 624757:**

```
Elapsed Time
SP GFLOPS / DP GFLOPS / x87 GFLOPS
CPI Rate
Average CPU Frequency
Total Thread Count
Effective Physical Core Utilization (con "(N.NNN out of M)" embebido en el string)
Effective Logical Core Utilization  (idem)
Memory Bound                    -- "% of Pipeline Slots"
    Cache Bound                 -- "% of Clockticks"
    DRAM Bound                  -- "% of Clockticks"
    NUMA: % of Remote Accesses
Vectorization                   -- "% of Packed FP Operations"
    Instruction Mix
        SP FLOPs / DP FLOPs / x87 FLOPs / Non-FP  -- "% of uOps"
            Packed / Scalar
                128-bit / 256-bit / 512-bit
    FP Arith/Mem Rd Instr. Ratio
    FP Arith/Mem Wr Instr. Ratio
```

**Hallazgo que sí cambia algo del plan — no hay una "contraparte de cómputo"
explícita junto a `Memory Bound`.** `PLAN.md` §4.2 (D3-native) asume que
`hpc-performance` imprime el desglose Top-Down completo (Retiring / Front-End
Bound / Bad Speculation / Back-End Bound, con Core Bound dentro de este último)
para comparar contra Memory Bound. En la salida real de `-report summary` **eso
no aparece** — `Memory Bound` sale como métrica aislada de nivel superior, sin un
"Core Bound" o "Compute Bound" hermano impreso en el mismo reporte. El desglose
Top-Down completo de 4 categorías vive normalmente en Microarchitecture
Exploration, que ya sabíamos no disponible en este nodo (ver arriba) — por eso no
aparece aquí tampoco, tiene sentido con la restricción de uncore/PMU ya conocida.

Lo que SÍ existe como alternativa, confirmado con `-report hw-events
-format=csv`: los contadores crudos `TOPDOWN.SLOTS` y
`TOPDOWN.BACKEND_BOUND_SLOTS`, más los eventos de stall
(`CYCLE_ACTIVITY.STALLS_L1D_MISS/L2_MISS/L3_MISS/MEM_ANY/TOTAL`,
`MEM_LOAD_RETIRED.*`, `MEM_LOAD_L3_MISS_RETIRED.LOCAL_DRAM/REMOTE_DRAM`, etc.),
por función. Con eso se podría reconstruir Back-End Bound/Core Bound a mano
siguiendo la metodología TMAM de Intel, pero es trabajo adicional no
contemplado en el plan original — pendiente de decisión con el usuario sobre si
vale la pena para este pipeline o si `classifier.py` usa en cambio el
complemento simple `100 - memory_bound_pct` (dentro de Pipeline Slots) como señal
"no-memoria", documentado honestamente como que incluye Retiring + Bad
Speculation + Front-End Bound + Core Bound juntos, no un Core Bound aislado. Ver
discusión pendiente en la sesión de implementación — no reescribir `PLAN.md` §4.2
hasta confirmar con el usuario.
