# Estrategia CPU — Fase 2 (documento para el director)

**Propósito.** Complementa `Estrategia_GPU_Fase2.md`. El eje CPU tiene un
bloqueo de infraestructura real (CAP_PERFMON) que el de GPU no tiene, así
que este documento separa con cuidado **qué está bloqueado de verdad** de
**qué se puede seguir haciendo hoy**, y por qué ninguna de las dos cosas
rompe el contrato de objetivos. §8 mapea a los Objetivos Específicos; §9
da la lista de pruebas para verificar o refutar cada afirmación.

Fuente: `docs/general/resultados_compuertas_fase2.md` (2026-08-21/22) y
el análisis nuevo de esta sesión sobre esos mismos datos.

> **Nota de auditoría (2026-08-24).** Revisado contra los datos crudos; se
> corrigieron cuatro errores de la versión anterior: (a) `npb_mg` estaba
> listado entre los kernels con alternancia de fase intra-corrida cuando
> su clase minoritaria es **0.0% en los seis niveles**, (b) el conteo de
> kernels de RAJAPerf estaba inflado al doble, (c) se comparaba frecuencia
> *observada* de REF contra frecuencia *aplicada* de F0, (d) el Objetivo 4
> se declaraba satisfecho sin señalar que solo cubre uno de los dos
> gobernadores nativos disponibles. Se agregó además un riesgo nuevo
> (riesgo 6) detectado al auditar el propio script de tamizaje.

---

## 1. El bloqueo real, con precisión

`perf_event_open` sobre contadores de **uncore** (`uncore_imc`, ancho de
banda de memoria) requiere `CAP_PERFMON`, que se rompió en pacca (ARC-184,
regresión de un permiso ya concedido). Eso bloquea **generar dataset nuevo
de CPU con etiqueta de verdad**: `phase_label_train` se deriva de
intensidad operacional dinámica, que sale de uncore.

**Lo que NO está bloqueado, y es la aclaración central:** el modelo, una
vez entrenado, **no necesita uncore para inferir**. Verificado en el
código ya escrito (`classifier/training/train_phase.py`):

```python
FEATURES = [
    "ipc", "mpki", "llc_miss_rate", "stall_backend_ratio",
    "ips", "running_ratio", "freq_khz_observed",
]
```

Ninguna de esas siete columnas requiere uncore. El diseño ya declaraba
`operational_intensity*`, `uncore_cas_count_*` y derivados como
`FORBIDDEN` — **no por el bloqueo, sino porque son la fuente de la
etiqueta y usarlas como feature sería fuga de datos**. El daemon del
Objetivo 3 nunca dependió de CAP_PERFMON. Lo que CAP_PERFMON bloquea es
**entrenar con datos nuevos**, no **desplegar**.

## 2. Lo que ya se hizo hoy sin CAP_PERFMON

Existe una campaña válida corrida **antes** de la regresión:
`pacca_cpu_final_attempt03_20260820_arc174` — 424/540 corridas aceptadas,
9 kernels × 6 niveles × 10 rep, con uncore funcionando y energía RAPL
real. Esta sesión construyó `cpu_policy_headroom.py` y lo corrió sobre
esos datos, sin tocar ningún nodo. Energía RAPL (pkg+dram), contra
"siempre F0", **9 kernels**:

| presupuesto de degradación | mejor constante | oráculo | margen del modelo |
|---|---|---|---|
| ≤4% | F0 → 0% | 0.33% | **0.33 pts** |
| ≤10% | F0 → 0% | 0.33% | **0.33 pts** |
| ≤15% | F0 → 0% | 0.33% | **0.33 pts** |
| sin límite | F0 → 0% | 0.33% | **0.33 pts** |

El margen no cambia con el presupuesto porque **el óptimo casi nunca se
aparta de F0**: 7 de 9 kernels lo tienen en F0. Solo `npb_mg` se separa de
forma significativa (**+2.71% en F1**); `npb_ft` (+0.10%) y
`rajaperf_polybench_3mm_omp` (+0.14%) están dentro del ruido.

## 3. Por qué CPU no responde al mismo remedio que GPU

En GPU el margen escondido vivía en el salto F0→F1 porque **el reloj de
núcleo y el de memoria son dominios independientes**: bajar el núcleo no
frena la memoria. **En CPU no existe ese segundo dominio.** El análisis de
compuerta 0 (`resultados_compuertas_fase2.md`, previo a esta sesión) ya lo
documentó: *"bajar el reloj del núcleo también frena el acceso a memoria
(menos peticiones en vuelo, prefetch más lento)"*. El estiramiento
observado al bajar 4× el reloj fue **2.21×–4.05×**, no 1×–4×: ni el kernel
más memory-bound se acerca a "bajar el reloj sale gratis".

**Esto coincide con la literatura citada, no la contradice.**
`Calore2017` (Haswell CPU) reporta *"los ahorros no son grandes, pero
tampoco despreciables"* — más modesto que GPU, con la misma causa física.
Su regla operativa es la nuestra: ahorro cuando el código es memory-bound,
medido por balance de máquina vs. intensidad operacional (Roofline,
`\cite{Williams2009}`, ya usado en este trabajo).

**Consecuencia de diseño:** insistir en más *resolución* de grilla sobre
los mismos 9 kernels no tiene sustento — a diferencia de GPU, el problema
de CPU no es dónde se muestrea sino **qué se muestrea** (§6).

## 4. Un experimento descartado, y qué se rescató de él

Se evaluó comparar contra el gobernador `powersave` (dinámico bajo HWP,
que la documentación del kernel describe como similar a
`schedutil`/`ondemand`), inspirado en que `Hebbar2022` logra 121–183% de
mejora explotando que el gobernador por defecto detecta mal las cargas
memory-bound.

**No se implementó porque ya se intentó y falló.** ARC-160 (2026-08-19,
`Registro_Cambios_Fuera_Plan_Original.md`): escribir
`scaling_governor=powersave` en los 6 CPUs delegados devuelve
`Permission denied` en los 6 — el permiso P1 cubre únicamente
`scaling_min_freq`/`scaling_max_freq`, nunca el gobernador ni el EPP.

**Lo que sí se rescata:** en esta plataforma el driver es `intel_pstate`
con solo dos gobernadores disponibles (`performance`, `powersave`), y el
nivel `REF` corre bajo el activo, `performance`. Su frecuencia **observada
bajo carga** es indistinguible de la de F0 (**3199.88 vs 3199.95 MHz**,
ambas medias observadas sobre `npb_bt`): el gobernador nativo activo,
bajo carga, se comporta como "siempre máxima frecuencia".

**Alcance exacto para el Objetivo 4, sin sobreventa:** `REF` satisface la
comparación contra **el gobernador nativo activo** sin necesitar ningún
permiso nuevo. **No** cubre `powersave`, que es el otro gobernador nativo
disponible y el que la línea de `Hebbar2022` sugiere que sería el rival
interesante. Cerrar esa mitad exigía una solicitud de permiso sobre
`scaling_governor`/EPP que nunca se había pedido (riesgo 3) — **el
administrador la concedió el 2026-08-25** (`sudo /usr/local/bin/set_cpu_gov
<gobernador> <epp>`, restringido a los cores 0-5). El permiso ya no es el
bloqueador; **queda pendiente de verificación empírica antes de usarse en
una campaña real** (riesgo 3, actualizado), tema abierto hasta que haga
falta para la comparación de la Fase 4 — no bloquea nada de lo que corre
hoy.

## 5. El plan concreto

1. **Tamizar el catálogo por α**, buscando kernels más memory-bound que
   los 9 actuales. **No necesita CAP_PERFMON**: mide solo tiempo de pared
   y energía RAPL con `scaling_min/max_freq` pineados (permiso P1, ya en
   uso). **Encolado, ver §6.**
2. **`npb_mg` es la única señal real del catálogo actual** (+2.71% en F1)
   — el equivalente CPU de `lavamd` en GPU: el candidato con margen
   genuino, a estudiar primero.
3. **Construir y evaluar el modelo con la arquitectura corregida**
   (predicción por carga desde un nivel de referencia — detallada en
   `Estrategia_GPU_Fase2.md` §4, es la misma para ambos ejes) **sobre el
   dataset ya válido**, sin esperar nada. Es el reintento del trabajo que
   falló bajo LOKO (`resultados_compuertas_fase2.md` §5.bis: F1 macro
   0.393 del mejor modelo vs. 0.371 del predictor trivial).
4. **El tamizaje (job 6483) no dejó kernels que ampliar** (§6: 0/7
   sobrevivientes) — este paso queda cerrado sin dataset nuevo que
   construir. Lo que sí sigue pendiente y sin correr es la **campaña de
   rejilla fina** (`campaign_pacca_cpu_fine_grid.yaml`, 7 niveles × 9
   kernels × 10 reps, diseñada para cerrar el hueco 3200-2600 MHz nunca
   muestreado — motivada por `npb_mg`, el único kernel cuyo óptimo de
   energía no cae en frecuencia máxima). Preparada desde 2026-08-21, **sin
   ejecutar todavía**: su preflight de verificación uncore falló dos veces
   (jobs 6431 y 6484, mismo patrón E13 intermitente) y quedó bloqueada
   detrás de eso, no por falta de prioridad.
5. **Cuando CAP_PERFMON se repare**: la etiqueta de verdad completa queda
   disponible para la rejilla fina de arriba. Protocolo
   actualizado (2026-08-25), evidencia-basado y no heredado sin más del
   núcleo: **6 niveles × 6 repeticiones** (no 10 — ver
   `docs/justifications/report/sections/repetitions_edp.tex`, análisis de
   convergencia de CV% de EDP sobre las 54 celdas de `arc174` corregido;
   n=6 conserva la misma cobertura a umbral 2% que n=3 y n=10, 40% menos
   costo). El par baseline/telemetry que mide overhead de instrumentación
   ya no se repite en cada corrida — con 540 pares medidos (media 1.95%,
   estable entre kernels) el número ya está caracterizado; el manifiesto
   nuevo declara `baseline_repetition_indices: [1]` como vigilancia contra
   desvío del instrumento, no medición completa de nuevo.

## 6. Impulso: RAJAPerf ya está en pacca, casi sin usar

Cuántos kernels usa cada trabajo citado: `Guerreiro2019` 35,
`Calore2017` **2** (una sola app, y aun así resultado real y citable),
`Hebbar2022` 43 (SPEC CPU2017 — **licencia paga, no reproducible por
nosotros**), `Antici2024` producción a escala Fugaku. **El número no
decide; decide la diversidad de régimen cubierta.**

El catálogo CPU actual (9 kernels) tiene exactamente **un** candidato
memory-bound con margen real. **RAJAPerf ya está descargado y compilado en
pacca**, y el catálogo usa **1** de sus kernels. Conteo verificado
(2026-08-24; el conteo previo estaba inflado al doble por un artefacto):

| categoría | kernels | | categoría | kernels |
|---|---|---|---|---|
| `apps` | 22 | | `algorithm` | 8 |
| `basic` | 20 | | `comm` | 6 |
| `polybench` | 13 | | `stream` | 5 |
| `lcals` | 11 | | **total** | **85** |

`raja-perf.exe` corre cualquiera con `-k NOMBRE -v Base_OpenMP`: agregar
un kernel es un wrapper, **no una compilación** — mismo binario, checksum
ya verificado.

**Job 6483 — COMPLETADO (2026-08-25), resultado negativo cuantificado.**
Tamiza 7 candidatos de Polybench elegidos por conocimiento algorítmico
clásico (stencils y productos matriz-vector: `JACOBI_1D`, `JACOBI_2D`,
`HEAT_3D`, `FDTD_2D`, `ATAX`, `GESUMMV`, `MVT`). Script
`scripts/pacca/screen_rajaperf_cpu_alpha.sh`: bypasea el orquestador por
diseño (solo tiempo + RAPL, sin `perf`/uncore), así que no espera a
CAP_PERFMON. (El primer intento, job 6475, corrió con un bug de pineo —
ver riesgo 6 abajo, ya cerrado — y se descartó sin usar sus datos.)

**Ningún candidato sobrevive.** α medido entre 0.331 (`FDTD_2D`) y 0.852
(`HEAT_3D`), todos muy por encima del umbral 0.226; ajuste de Amdahl
confiable (r² = 0.975–0.999 en los 7); pineo de frecuencia verificado bajo
carga en 5/7 (C2 falla en `ATAX` y `GESUMMV`, pero ambos ya fallan C1 por
márgen amplio, así que no cambia el veredicto):

| kernel | α | r² | C1 | C2 |
|---|---:|---:|---|---|
| `FDTD_2D` | 0.331 | 0.977 | no | OK |
| `GESUMMV` | 0.534 | 0.987 | no | no |
| `JACOBI_1D` | 0.599 | 0.975 | no | OK |
| `ATAX` | 0.635 | 0.999 | no | no |
| `MVT` | 0.652 | 0.999 | no | OK |
| `JACOBI_2D` | 0.714 | 0.989 | no | OK |
| `HEAT_3D` | 0.852 | 0.999 | no | OK |

> ### ⚠️ NOTA DE INTEGRIDAD (2026-08-26): la sección de abajo tenía un
> ### número mal atribuido — corregido aquí, no reescrito en silencio
>
> La versión anterior de este aviso decía «la L3 de este nodo son 39 MB»
> y concluía que los 32 MB/rep de `JACOBI_1D` cabían en cache. **Ese
> número (39 MB / 39936 KiB) es la L3 de `pacca01` (Xeon Gold 5320),
> leída con un `srun -p normal -w pacca01` — no la de `paccaA100`, que es
> donde corrió de verdad el job 6483.** Error de atribución cruzada entre
> nodos, exactamente el tipo de comparación que el proyecto tiene
> documentado como peligrosa
> (`pacca01-vs-paccaA100-node-divergence.md`) y que aun así se cometió
> aquí.
>
> **La L3 real de `paccaA100` (Xeon Gold 5315Y) es 12 MB**
> (`cache_llc_kb: 12288` en `node_profile.json`, consistente con la
> especificación pública de Intel para ese modelo — 8 núcleos, 12 MB de
> caché). Con ese número, los 7 candidatos **sí exceden la LLC real**, no
> caben:
>
> | kernel | tamaño por defecto | LLC real (12 MB) | veces la LLC |
> |---|---:|---:|---:|
> | `ATAX` | 16.0 MB | 12 MB | 1.3× |
> | `GESUMMV` | 16.0 MB | 12 MB | 1.3× |
> | `MVT` | 16.0 MB | 12 MB | 1.3× |
> | `JACOBI_1D` | 32.0 MB | 12 MB | 2.7× |
> | `JACOBI_2D` | 32.1 MB | 12 MB | 2.7× |
> | `HEAT_3D` | 33.0 MB | 12 MB | 2.7× |
> | `FDTD_2D` | 79.9 MB | 12 MB | 6.7× |
>
> **Lectura honesta, sin sobrecorregir en la otra dirección:** que
> excedan la LLC no prueba que el 0/7 original fuera válido — 1.3-2.7×
> es un exceso moderado, no una holgura clara, y con asociatividad de
> caché y prefetching agresivo (todos son accesos con patrón regular,
> stride constante) es plausible que una fracción sustancial del tráfico
> siga sirviéndose desde L3 en vez de DRAM. El diagnóstico de tamaño
> **no está descartado, pero tampoco está probado con el margen que
> antes se afirmaba** — queda en un terreno intermedio, y es exactamente
> lo que el tamizaje v2 (`--memory-touched` a 10× la LLC real, ~120 MB,
> excedente inequívoco) está diseñado para zanjar sin ambigüedad.
>
> Lo que SÍ sigue en pie sin cambios: el hallazgo de Hebbar sobre
> `649.fotonik3d` (un stencil real haciendo *plateau* en la literatura),
> y que `--memory-touched` (que lee la LLC en vivo del nodo donde
> realmente corre, nunca hardcodeada) es la forma correcta de zanjarlo —
> eso no dependía del número erróneo.
>
> Los números de la tabla de arriba (α por kernel) se conservan como
> registro de lo medido; el veredicto sobre SI ese α refleja al kernel o
> al tamaño queda abierto hasta el resultado del tamizaje v2 (job 6575).

## 6.bis ¿Se frena la memoria al bajar el reloj del núcleo? Medido

Es la premisa sobre la que descansa todo el eje CPU: si una carga está
limitada por memoria, bajar el reloj no debería alargarla. Hasta el
2026-08-25 solo se había observado el **síntoma** (el tiempo), nunca la
**causa**. `stream_official` satura el ancho de banda al 100% y aun así
mide α = 0.154, no 0 — a 800 MHz tarda 46% más que a 3200 MHz *pese a
estar completamente limitado por memoria*. Esa brecha había que
explicarla.

**Job 6542** (`scripts/pacca/measure_cpu_memory_vs_frequency.sh`) mide la
frecuencia real del controlador de memoria por sus propios ticks
(`uncore_imc_0/clockticks` sobre reloj de pared conocido) mientras se
varía la del núcleo. Necesario porque `current_freq_khz` del driver
`intel_uncore_frequency` es ilegible sin root en este nodo — pero ese
mismo sysfs confirma que **el uncore es un dominio de frecuencia
separado** (800–2400 MHz) del de los núcleos (800–3200 MHz).

| nivel de núcleo | reloj IMC (STREAM) | reloj IMC (`ptrchase`) | BW alcanzado (STREAM) |
|---|---:|---:|---:|
| F0 (3200 MHz) | 2.755 GHz | 2.886 GHz | 20.8 GB/s |
| F1 (2600) | 2.722 | 2.886 | 20.4 |
| F2 (2000) | 2.685 | 2.889 | 19.2 |
| F3 (1400) | 2.776 | 2.853 | 18.1 |
| F4 (800) | 2.804 | 2.824 | **14.6** |

**El controlador de memoria NO se frena** — su reloj se mantiene en
2.75–2.89 GHz (variación ~2%) mientras el núcleo cae 4×. La hipótesis de
que el uncore baja con los núcleos queda **descartada con medida**.

**Pero el ancho de banda alcanzado sí cae un 30%.** La memoria va a plena
velocidad y el núcleo lento no consigue alimentarla: emite menos accesos
en vuelo y no sostiene el paralelismo de memoria necesario. Ojo con el
signo: si el límite fuera puramente el núcleo, el BW habría caído 4×
como el reloj; cayó 1.43×. **La memoria sigue siendo el limitante
dominante**; el núcleo solo deja de saturarla del todo en el extremo bajo.

**Consecuencia para la tesis:** la premisa es correcta en lo esencial
(la memoria no se ralentiza) pero incompleta — un núcleo lento no
mantiene saturada a una memoria que sí va rápido. Eso es un límite de la
**carga y su paralelismo de memoria**, no de la plataforma, y por tanto
es atacable eligiendo mejor los kernels.

**Caveat declarado:** el control compute-bound (`ert_probe`) no sirvió en
esta corrida — dura 0.15 s, demasiado corto, y su frecuencia no llegó a
estabilizarse (`freq_within_5pct = NO` en 4 de 5 niveles). Sin un control
negativo limpio el argumento no queda cerrado del todo. Lo que sí
discrimina el instrumento: STREAM mueve 20.8 GB/s y `ptrchase` 0.53 GB/s
—40× de diferencia— exactamente lo esperable entre saturar ancho de banda
y estar limitado por latencia.

## 6.ter α depende de la VENTANA de frecuencia sobre la que se ajusta

Hallazgo del job 6543 (`classifier/analysis/alpha_by_frequency_window.py`),
motivado al revisar la literatura. α **no es invariante** al rango sobre
el que se ajusta: si el subsistema de memoria se degrada en el extremo
bajo, ese efecto entra en el ajuste y lo infla, aunque en la ventana alta
el kernel sea insensible al reloj.

| kernel | F0-F1 (1.23×) | F0-F2 (1.60×) | F0-F3 (2.29×) | F0-F4 (4.00×) |
|---|---:|---:|---:|---:|
| `stream_official` | 0.103 | 0.106 | 0.121 | 0.154 |
| **`npb_mg`** | **0.171** | **0.227** | 0.296 | 0.385 |
| `npb_sp` | 0.376 | 0.402 | 0.434 | 0.491 |
| `npb_cg` | 0.600 | 0.655 | 0.700 | 0.760 |
| `npb_ft` | 0.733 | 0.739 | 0.750 | 0.771 |
| `npb_lu` | 0.776 | 0.810 | 0.820 | 0.840 |
| `npb_bt` | 0.861 | 0.860 | 0.875 | 0.902 |
| `dgemm_n2048` | 0.874 | 0.889 | 0.902 | 0.931 |
| `3mm_omp` | 0.986 | 0.975 | 0.985 | 1.007 |
| `lavamd_omp` | 1.068 | 1.012 | 1.015 | 1.027 |

**`npb_mg` da α = 0.171 en la ventana F0–F1, por debajo del umbral
0.226.** La afirmación «ningún kernel del catálogo baja del umbral» era un
artefacto de ajustar α incluyendo F4 (800 MHz) — una región que la
política nunca visitaría, porque su EDP es pésimo para todos los kernels.
**En la ventana que la política usaría de verdad, `npb_mg` sí es viable**,
y eso justifica de forma independiente la campaña de rejilla fina
3200–2600 (`campaign_pacca_cpu_fine_grid.yaml`).

Segundo kernel viable, de la misma tanda: **`ptrchase`** (job 6542) va de
4.733 s en F0 a 6.771 s en F4, lo que da **α ≈ 0.144 en rango completo y
≈ 0.096 en F0–F1** — bajo el umbral en ambas ventanas. Es el sujeto
latency-bound que faltaba, y ya está compilado y en el catálogo.

**Regla metodológica que se deriva:** la ventana correcta para ajustar α
es la que cubre los niveles que la política usaría. Reportar α sobre
F0–F4 mete en el número una región inalcanzable y sesga la decisión de
catálogo hacia el rechazo.

## 6.quater Qué dice la literatura, y en qué nos diferenciamos

Revisión de los dos trabajos de CPU citados (2026-08-25).

**Hebbar & Milenković 2022** (`\cite{Hebbar2022}`), Core i7-8700K de
escritorio, 0.8–4.3 GHz, SPEC CPU2017. Dos aportes directos:

- Confirman explícitamente el mecanismo que medimos en §6.bis: *«uncore
  frequency scaling (UFS), enabling the processor to control the frequency
  of the uncore components (e.g., last-level caches) **independently** of
  the core frequencies. […] The uncore frequency has a significant impact
  on on-die cache-line transfer speeds as well as **on memory
  bandwidth**»*.
- Encuentran *plateaus* reales: `649.fotonik3d` deja de mejorar por encima
  de **1.7 GHz** y `628.pop2` de 2.7 GHz. Ambos son códigos de stencil /
  diferencias finitas sobre mallas grandes — el tipo de carga que nuestro
  tamizaje descartó por correrla en cache (ver el aviso de §6).
- Mejoras de eficiencia energética de 44–92% frente al gobernador
  `ondemand`, y 121–183% en la clase memory-intensive. **No comparable
  directo con nuestro EDP**: su métrica y su línea base son otras, y el
  procesador es de escritorio.

**Calore et al. 2017** (`\cite{Calore2017}`), Xeon E5-2630v3 de servidor,
rango explorado ~1.2–2.4 GHz. Es el más comparable con nuestro nodo:

| rutina | ahorro de energía (CPU) | coste en tiempo |
|---|---:|---:|
| `propagate` (memory-bound) | **9%** | 3% |
| `collide` (compute-bound) | 4% | 4% |
| código completo | 7% | 8% |

- Usan como diagnóstico `f · T_s` frente a `f`: memory-bound ⟹ `T_s`
  constante ⟹ el producto crece lineal. Con eso identifican `propagate`
  (paso de *streaming* de Lattice Boltzmann) como memory-bound con `T_s`
  prácticamente constante en todo su rango.
- Su 3% de coste sobre 1.41× de reducción implica **α ≈ 0.073**, pero
  medido solo en la ventana alta — lo que refuerza la regla de §6.ter:
  su α y nuestro α medido sobre 4× **no son comparables sin fijar la
  ventana**.
- Reportan que el gobernador `powersave` *«has an adverse effect on both
  TS and ES»* **incluso en el kernel memory-bound**. Bajar demasiado
  también les perjudicaba.
- Su conclusión: *«fair energy savings are possible by tuning the
  processor clock to lower values in all cases in which the code is
  memory-bound»* — ganancias **modestas pero reales**, consistente con
  que nuestro margen sea pequeño y no con que sea inexistente.

**Dónde estamos frente a ellos, sin adornar.** Calore obtuvo 9%/3% en un
Xeon de servidor con un rango de frecuencia **más estrecho** que el
nuestro. Así que el resultado no es inalcanzable por hardware. Lo que sí
es una restricción real y propia de este nodo es el **rango dinámico de
potencia de 1.40×** (116.5 W → 83.4 W), del que sale el umbral 0.226; con
un rango más ancho, kernels que aquí fallan pasarían. Las tres cosas son
distintas y conviene no confundirlas:

1. **La velocidad de la memoria NO nos perjudica** (medido, §6.bis).
2. **El rango de potencia de 1.40× SÍ nos perjudica** (medido, es la
   restricción estructural del eje CPU).
3. **El catálogo está mal elegido** (medido: 7 de 9 kernels con α > 0.6,
   y el intento de ampliarlo se corrió en cache).

## 6.quinquies Confirmado con campaña completa: `npb_mg` y `ptrchase` son viables

Jobs 6412 y 6530 (2026-08-26), campaña completa con 10 repeticiones por
nivel — ya no sondas rápidas. **Dos kernels quedan confirmados por debajo
del umbral 0.226, con datos limpios:**

**`npb_mg` (rejilla fina 3200→2000 MHz, 638/720 corridas aceptadas).**
Su óptimo de EDP cae en **S3000 (3000 MHz)**, EDP/F0 = 0.9927 (−0.73%) —
un mínimo real y no monótono, con S3100 y S2900 subiendo de nuevo a
ambos lados. **Es el único de los 9 kernels del catálogo original con un
óptimo fuera de F0**; los otros 8 (`npb_bt`, `npb_cg`, `npb_sp`,
`npb_ft`, `npb_lu`, `dgemm_n2048`, `rodinia_lavamd_omp`,
`rajaperf_polybench_3mm_omp`) degradan su EDP monótonamente al bajar la
frecuencia, sin excepción, en toda la rejilla nueva.

**`ptrchase` (sonda de fases, 320/320 aceptadas, 10 reps/nivel).**

| ventana | α | r² |
|---|---:|---:|
| F0–F4 completo | 0.122 | 0.990 |
| F0–F1 (la que usaría la política) | **0.097** | 1.000 |

Confirma con datos de campaña completa lo que la sonda rápida (job 6542)
ya sugería. Es el sujeto latency-bound que el catálogo no tenía.

**`phasic_p010/p100/p1000`: resultado de otro tipo, no un tercer
candidato de catálogo.** EDP/F0 cae a 0.82–0.83 en F4 (17–18% de ahorro),
con α≈0.002–0.003 — muy por debajo incluso de `ptrchase`. Pero su
duración es casi constante entre F0 y F4 (<1% de variación, 20.27→20.4 s)
por **diseño**: la fase dura un tiempo fijo, no un trabajo fijo (ver §7).
Su α bajo es una propiedad del microbenchmark, no evidencia de que una
carga real se comporte así — sirve como control de que el instrumento
detecta memory-boundness limpio cuando existe, no como sujeto adicional
del catálogo final.

**Balance: el catálogo pasa de 9 kernels con 0 viables a 11 con 2
viables** (`npb_mg`, `ptrchase`), antes incluso de que termine el
tamizaje v2 sobre los ~79 de RAJAPerf (job 6575, en cola). Confirma
directamente la lectura de §6.quater: el problema era el catálogo, no la
plataforma — con kernels mejor elegidos, sí hay margen.

## 6.sexies GAP Benchmark Suite: el hueco de acceso irregular, y por qué `pacca01` sirve para triage aquí

Ni STREAM/Polybench (ancho de banda, acceso regular) ni `ptrchase`
(latencia pura, sin estructura de algoritmo) cubren acceso **irregular
dependiente del dato** — el patrón de recorrer una lista de adyacencia,
donde el próximo salto de memoria no se conoce hasta resolver el actual.
GAP Benchmark Suite (Beamer et al., arXiv:1508.03619) es la suite
académica estándar para eso: BFS, PageRank, componentes conexas, camino
más corto, betweenness centrality, triángulos. El propio paper documenta
que estos algoritmos sufren latencias de memoria largas en SMP porque la
jerarquía de caché está optimizada para acceso local y contiguo — lo
opuesto de lo que hacen.

**Por qué el triage inicial corre en `pacca01` y no en `paccaA100`,
excepción deliberada a la regla de no comparar los dos nodos.** La razón
por la que `paccaA100` está reservado para números finales sigue
intacta (compilación cruzada, umbral de α específico de su modelo de
potencia — ver la nota de integridad de §6). Pero hay una comparación
que **sí** es válida entre ambos: la **L3 por núcleo es idéntica, 1.50
MB** (12 MB / 8 núcleos en `paccaA100`; 39 MB / 26 núcleos en `pacca01`)
— mismo punto de diseño, misma generación de microarquitectura
(Ice Lake-SP), mismos flags AVX-512. Eso hace que el comportamiento
*cualitativo* de un kernel de un solo hilo contra la caché sea
extrapolable: si un kernel de GAP no muestra NINGUNA sensibilidad al
reloj en `pacca01`, es muy poco probable que la muestre en `paccaA100`.

**Lo que NO es extrapolable, y por qué ningún número de aquí es
citable**: el umbral 0.226 sale del modelo de potencia de `paccaA100`
específicamente; `pacca01` tiene otro (no medido, turbo activo por
convención de ese nodo). Y 26 núcleos/socket contra 8 cambia la
contención real de ancho de banda por núcleo activo. `pacca01` responde
"¿este kernel responde al reloj en absoluto?" (sí/no cualitativo) — no
"¿cruza 0.226?". Cualquier candidato que pase el triage se remide en
`paccaA100` antes de entrar al catálogo o citarse en cualquier
documento.

Scripts: `scripts/pacca/build_gap_benchmark.sh` (clona y compila, sin
checksum pineado a propósito — es triage, no un kernel de catálogo),
`scripts/pacca/screen_gap_alpha_pacca01.sh` (BFS y PageRank primero, los
dos más usados en la literatura de GAP; grafo Kronecker sintético
2²²≈4.2M vértices, sin descargar los 275 GB de grafos reales), lanzados
por `run_gap_triage_pacca01.sbatch`.

> ### ⚠️ El triage en `pacca01` está BLOQUEADO por permisos, no dio señal (2026-08-26)
>
> Job 6583 completó (exit 0) pero **sin ningún dato utilizable**: el
> `.err` está lleno de `Permission denied` al escribir
> `scaling_max_freq`/`scaling_min_freq` en los cores de `pacca01`. Nunca
> se concedió permiso de escritura de frecuencia en ese nodo — el
> permiso `set_cpu_gov` que sí existe es específico de cores de
> `paccaA100`. Consecuencia: el reloj se quedó fijo en ~3.39–3.40 GHz en
> los 5 "niveles" pedidos (F0 pasa por casualidad de estar cerca del
> máximo; F1–F4 marcan `freq_within_5pct=NO`), y `bfs`/`pr` dieron
> tiempo prácticamente plano (4.88 s y 5.70 s respectivamente, sin
> variación real). **Eso NO es evidencia de que GAP sea insensible al
> reloj — es un instrumento que nunca movió la variable independiente**,
> el mismo modo de fallo de ARC-162 (tiempo plano por candado roto, no
> por el kernel).
>
> **No se persigue el permiso de `pacca01`**: repetir el ciclo de
> solicitud al administrador (como costó días con `set_cpu_gov` en
> `paccaA100`) no vale la pena para un nodo que solo iba a servir de
> triage. Lo aprovechable del intento: **los binarios ya están
> compilados** (`~/hyperion-kernels/libexec/gapbs/`: `bfs`, `pr`, `cc`,
> `cc_sv`, `sssp`, `tc`, `bc`, `pr_spmv`; commit real `2972aeb2`). Cuando
> `paccaA100` se libere, `bfs`/`pr` se tamizan ahí directamente —
> saltando el paso de triage por completo, con un número citable desde
> el principio, porque ahí sí hay permiso de escritura de frecuencia
> confirmado funcionando (jobs 6412/6530/6575).
>
> **Sí sirve otra vía a frecuencia nativa, sin el permiso bloqueado**:
> `perf stat` (lectura, permiso distinto al de escritura de frecuencia)
> a un solo reloj, comparando la huella de fallos de caché de `bfs`/`pr`
> contra `stream_official` y `ert_probe` como anclas conocidas.
>
> | kernel | IPC | % fallos de caché | % fallos de LLC |
> |---|---:|---:|---:|
> | `bfs` | 0.89 | 23.4% | 10.4% |
> | `pr` | 0.86 | 11.7% | 3.5% |
> | `stream_official` (α=0.154) | 0.23 | 98.0% | 99.8% |
> | `ert_probe` (compute-bound) | 1.73 | 1.8% | 1.5% |
>
> **Templa la expectativa, no la descarta.** Los fallos de LLC de
> BFS/PageRank quedan mucho más cerca del extremo compute-bound que del
> memory-bound puro — no es el caso claro que STREAM. Explicación
> plausible: el grafo a escala 2²² pesa ~570 MB (muy por encima de los
> 12 MB de L3), pero BFS toca solo el *frontier* activo por nivel y
> PageRank re-toca el mismo vector de rank por iteración — localidad
> temporal real del algoritmo, no un artefacto de tamaño. Un 10% de
> fallos de LLC aún puede dominar el tiempo (cada fallo cuesta ~100×
> más ciclos que un acierto, coherente con el IPC de 0.86-0.89 frente al
> 1.73 de cómputo puro), así que esto NO descarta a GAP — deja el
> candidato en un terreno genuinamente incierto, no en una apuesta
> segura como se esperaba. Vale medirlo en `paccaA100` de todas formas.
>
> **RESUELTO con número real (job 6601, 2026-08-26 noche).** Tamizaje de
> α directo en `paccaA100` (mismo método que `screen_rajaperf_cpu_alpha_v2.sh`,
> escritura de frecuencia P1, 10/10 corridas, frecuencia dentro de 5% del
> objetivo en las 10): **`bfs` α=0.738, r²=1.000; `pr` α=0.690, r²=1.000**
> — muy por encima del umbral 0.226. La incertidumbre de arriba queda
> despejada: GAP se comporta como compute-bound en la métrica agregada de
> corrida completa, coherente con la huella de caché ya medida en
> `pacca01`. **No es candidato de catálogo por margen de EDP.** Sigue sin
> descartarse del todo como fuente de *mezcla de fase intra-corrida*
> (pregunta distinta, motivada por C8, §7.bis) — un α agregado alto no
> excluye que existan ventanas minoritarias memory-bound, como pasa en
> `npb_bt`/`npb_lu` a frecuencias bajas — pero confirmarlo exige una
> campaña completa con uncore, un costo mayor que este tamizaje. Dado el
> resultado negativo de α, no se prioriza: ver §6.septies para el orden
> de los siguientes candidatos (LULESH, HPCG).

## 6.octies Campaña real sobre los 9 sobrevivientes: 7 de 9 tienen margen (job 6594, 2026-08-26)

**324/324 corridas aceptadas, 0 rechazos, matriz completa** (9 kernels ×
6 niveles × 6 reps). Único aviso: CAL-07 no válido en la calibración de
`ert_probe` en los 5 niveles (traza entera dentro de `grace_seconds`) —
no bloquea desde ARC-167, solo advertencia.

**Resultado, óptimo real de EDP (RAPL pkg+dram) por kernel:**

| kernel | mejor nivel | EDP/F0 | ahorro | α de tamizaje |
|---|---|---:|---:|---:|
| `Lcals_FIRST_SUM` | F2 | 0.9513 | **7.09%** | 0.113 |
| `Lcals_TRIDIAG_ELIM` | F1 | 0.9798 | 3.46% | 0.125 |
| `Stream_MUL` | F1 | 0.9548 | 4.92% | 0.078 |
| `Basic_INIT3` | F1 | 0.9698 | 4.19% | 0.178 |
| `Polybench_FDTD_2D` | REF | 0.9647 | 1.62% | 0.175 |
| `Polybench_JACOBI_1D` | REF | 0.9681 | 1.37% | 0.148 |
| `Basic_DAXPY` | REF | 0.9932 | 0.21% | 0.161 |
| `Stream_TRIAD` | F0 | 1.0000 | 0.00% | 0.128 |
| `Stream_ADD` | F0 | 1.0000 | 0.00% | 0.147 |

**Tres lecturas, ninguna trivial:**

1. **7 de 9 kernels salen del catálogo con margen real.** El catálogo
   pasa de "1 kernel con margen" (`npb_mg`, −0.73%) a **8 con margen**
   (`npb_mg` + estos 7), con ahorros hasta 10× mayores (7.09% frente a
   0.73%). Es el mejor resultado del eje CPU hasta hoy.
2. **α de tamizaje NO ordena el ahorro real** — y eso es un hallazgo, no
   un fallo del tamizaje. `Stream_MUL` tiene el α más bajo (0.078) pero
   no el mayor ahorro; `Lcals_FIRST_SUM` (α=0.113, segundo más bajo) da
   el mejor resultado; `Stream_TRIAD`/`Stream_ADD` (α medio, 0.128/0.147)
   dan **cero**; `Basic_DAXPY` (α=0.161, casi el más alto) también da
   casi cero. α mide sensibilidad de *tiempo* al reloj bajo un ajuste de
   Amdahl sobre todo el rango F0–F4; el óptimo de *EDP* es otra cosa —
   el punto donde el ahorro de energía todavía compensa el costo de
   tiempo, que puede caer en F1/F2 aunque α sea mediocre. El tamizaje
   cumplió su función real (separar candidatos de los 70 compute-bound),
   no la de predecir la magnitud del ahorro.
3. **Esto responde, con datos, la duda sobre diversidad interna que
   se planteó antes de correr esta campaña** (¿los 9 sobrevivientes son
   demasiado parecidos entre sí?): en **features de entrada** sí se
   agrupan (mismo tipo de kernel, α en banda estrecha); en **etiqueta de
   salida** (nivel óptimo, magnitud de ahorro) **no** — hay tres
   resultados distintos (F1, F2, REF/F0-sin-margen) y un rango de 0% a
   7.09%. Para el modelo esto es la señal correcta: variación real en el
   objetivo que aprender, no solo en el número de kernels.

**Pendiente:** reentrenar el piloto LOKO sobre el catálogo ampliado
(17 kernels: los 8 originales con campaña válida — `npb_mg` viable,
7 restantes sin margen — más estos 9), con las dos correcciones ya
identificadas (quitar `ref_running_ratio`, sin varianza; reconsiderar
dimensionalidad). Es el primer intento real desde el diagnóstico de causa
raíz de `loko_feature_diagnostic.py`.

## 6.nonies Primer resultado positivo del modelo (2026-08-26): gana al trivial en ambas variantes

Con los dos arreglos ya identificados en el riesgo 4 de
`Estrategia_GPU_Fase2.md` aplicados por primera vez — quitar
`ref_running_ratio` (varianza cero) y combinar las dos campañas CPU
(17 kernels efectivos en vez de 8) — el piloto LOKO (`loko_pilot.py`)
**le gana al trivial por primera vez en todo el proyecto**, en cualquiera
de los dos ejes:

| política | EDP loss | margen sobre trivial |
|---|---:|---:|
| oráculo (techo) | 1.0000 | — |
| **modelo sin umbral** | **1.0045** | **+0.0077 (gana)** |
| **modelo + umbral de acción, región REF–F2** | **1.0072** | **+0.0050 (gana)** |
| mejor constante honesta | 1.0173 | −0.0051 (pierde) |
| trivial (siempre F0) | 1.0122 | — |

**El umbral de seguridad, que antes nunca se disparaba, necesitó un tercer
arreglo.** El RMSE de entrenamiento (0.112) estaba inflado por F3/F4 —
`dgemm`/`npb_bt`/`lavamd`/`3mm_omp` llegan a EDP=8×–12× ahí — y ninguna
ganancia real de unos pocos puntos porcentuales lo superaba nunca, así
que la política con umbral degeneraba exactamente al trivial (margen
0.0000). Se agregó `--action-levels` a `loko_pilot.py`: restringe qué
niveles se ofrecen como opción real (oráculo, constante honesta, cálculo
del propio umbral) a REF–F2 — los regresores siguen entrenando con todo
el rango, pero el umbral ya no se mide contra una región que ninguna
política sensata visitaría. Con eso el RMSE baja a 0.048 y el umbral se
dispara en 4 de 17 kernels (`Basic_INIT3`, `Lcals_FIRST_SUM`,
`Stream_MUL` → F1 correcto; `Stream_ADD` → F1, un error real frente al
óptimo F0 de ese kernel, absorbido por las otras tres ganancias).

**Por qué la variante con umbral es la que importa para el Objetivo 3,
no la variante sin umbral.** Sin umbral el modelo se compromete con su
argmin siempre, incluso cuando su propia incertidumbre es mayor que la
ganancia — el modo de fallo que ya costó 4.18 puntos en GPU (§7, riesgo 4
de `Estrategia_GPU_Fase2.md`, caso `dwt2d`). Con umbral, cuando el modelo
no está seguro simplemente no actúa (cae a F0, empata al trivial para ese
kernel) — es la política segura por diseño que un daemon real puede
desplegar sin arriesgar empeorar respecto del gobernador nativo.

**Alcance honesto, sin sobrevender:** 1.22 puntos de margen disponible es
modesto (el catálogo sigue teniendo 10 de 17 kernels sin ningún margen
real, dominados por NPB/`dgemm`/`lavamd`/`3mm`), y el umbral solo captura
41% de ese margen (0.0050 de 0.0122). Pero es la primera vez que hay algo
que capturar y un modelo que lo hace sin arriesgar una regresión — el
Objetivo 2 tiene, por primera vez, un resultado que reportar en lugar de
solo un diagnóstico de por qué no funcionaba.

## 6.septies Estudio de candidatos futuros para llenar el hueco de diversidad (2026-08-26)

> **Reprioridad (2026-08-26 noche, tras C8 §7.bis).** Esta sección se
> escribió pensando en llenar el hueco de α intermedio (EDP). C8 cambió
> el objetivo principal: ahora se busca **kernels que produzcan mezcla
> real de fase intra-corrida** (el mismo fenómeno de `npb_lu`/`npb_bt`/
> `3mm_omp`), no solo α intermedio — son preguntas relacionadas pero no
> idénticas. Con ese criterio, el orden de prioridad cambia:
>
> 1. **GAP (`bfs`/`pr`)** — ya compilado, cero costo de compilación.
>    Acceso dependiente del dato con *frontier* que crece y encoge por
>    nivel del grafo: candidato natural a fases genuinamente distintas
>    dentro de una sola corrida, no solo régimen constante. Primer paso.
> 2. **LULESH** — física explícita por *timestep*: cálculo de tensiones
>    (compute) alternado con actualización de malla no estructurada
>    (memoria dependiente del dato) **por diseño del algoritmo**, no por
>    artefacto de tamaño. El candidato más directo a mezcla real después
>    de GAP.
> 3. **HPCG** — el ciclo multigrid alterna SpMV, suavizado y
>    restricción/prolongación, cada uno con intensidad distinta:
>    candidato razonable, un escalón por debajo de LULESH porque su
>    alternancia es más regular (mismo ciclo repetido) que multifásica.
> 4. **GUPS** y **PARSEC/`canneal`** — bajan de prioridad para este
>    objetivo específico: GUPS es acceso uniforme sin fases por diseño
>    (buen candidato para α, malo para mezcla), y PARSEC sigue con
>    licencia sin confirmar.

**Motivo original.** Los 9 sobrevivientes del tamizaje v2 (§6) no son tan variables
entre sí como parece a primera vista: todos son bucles de un solo paso
dominados por ancho de banda regular (familia STREAM/LCALS/Polybench de
stencil simple), con α agrupado en 0.078–0.178. El catálogo original tiene
7 kernels con α>0.6 (NPB, `dgemm`, `lavamd`, `3mm`). **Falta la banda
intermedia (α≈0.18–0.6) y patrones de acceso distintos al streaming
regular** — el mismo criterio de §6 ("no decide el número, decide la
diversidad de régimen") aplicado hacia adelante, no solo hacia atrás.

Búsqueda dirigida a esos dos huecos, no a "más kernels" en general:

| suite | qué llena | por qué | licencia / esfuerzo |
|---|---|---|---|
| **GAP Benchmark** (ya en curso, §6.sexies) | irregular dependiente del dato (grafos) | BFS/PageRank ya compilados, bloqueados solo por triage en `pacca01`; pendiente medir directo en `paccaA100` | académica, ya resuelta |
| **GUPS/RandomAccess** (HPC Challenge, `github.com/technion-csl/gups`) | acceso disperso **sin** dependencia secuencial — complementa a `ptrchase`, que es una cadena de punteros (un salto a la vez) | un solo fichero, compilación trivial (`make`), sin dependencias externas | libre, sin registro |
| **HPCG** (`hpcg-benchmark.org`, Dongarra et al.) | banda intermedia real: matriz dispersa CSR con **reutilización genuina** (multigrid, no solo streaming) mezclada con acceso indirecto | estándar HPC, open source, MPI+OpenMP, compilación con `make` estándar | libre |
| **LULESH** (LLNL, DOE proxy app) | malla no estructurada, accesos dependientes del dato, reducciones — un régimen que ningún kernel actual cubre | open source, CMake, versión serial/OpenMP sin exigir MPI | libre |
| **PARSEC** (Bienia et al., Princeton) — en particular `canneal` | el kernel de la literatura descrito como el más irregular/memory-bound de esa suite (punteros dispersos sobre un netlist), ya usado en estudios de DVFS de CPU publicados | requiere descarga/registro desde Princeton — **verificar licencia antes de comprometer tiempo**, a diferencia de las demás | por confirmar |

**Nada de esto se ha tocado en el nodo — es investigación de escritorio.**
Antes de comprometer tiempo de `paccaA100`: verificar que compila ahí (no
en `pacca01`, misma lección de §6/riesgo 1), que el tamaño de trabajo
puede fijarse a un múltiplo inequívoco de la LLC real (12 MB) con el mismo
criterio que corrigió el tamizaje v1→v2, y que no arrastra I/O pesado
(lección `myocyte`, Anexo L.1, ya citada en C3 de §9). El paso siguiente,
si se decide perseguir esto, es el mismo patrón ya validado: tamizaje
barato de α con tiempo/RAPL antes de cualquier campaña completa.

## 7. Reconciliación con la variación intra-kernel (Objetivo 2 literal)

No toda la evidencia apunta a "cero variación intra-kernel". La tabla de
clase minoritaria de `resultados_compuertas_fase2.md` muestra **mezcla
real** en:

| kernel | REF | F0 | F1 | F2 | F3 | F4 | global |
|---|---:|---:|---:|---:|---:|---:|---:|
| `npb_lu` | 0.0 | 0.0 | 0.0 | 8.5 | 34.4 | 30.9 | **19.0%** |
| `npb_bt` | 0.6 | 0.8 | 19.4 | 21.5 | 14.0 | 10.2 | **11.8%** |
| `rajaperf_polybench_3mm_omp` | 5.7 | 5.6 | 4.8 | 3.7 | 2.5 | 1.7 | 3.2% |

Los otros seis están por debajo de 1%, y **`npb_mg`, `npb_cg` y
`dgemm_n2048` son 0.0% en los seis niveles** — una sola clase de principio
a fin.

Dos matices que importan y que la versión anterior de este documento no
distinguía:

- En `npb_bt`/`npb_lu` la mezcla aparece **solo a frecuencias bajas**: es
  el ridge de Roofline desplazándose y cruzando el punto de operación
  (ARC-175), no fases que se alternen por naturaleza del algoritmo.
- **`rajaperf_polybench_3mm_omp` es el único con mezcla a REF/F0**
  (5.7%/5.6%), es decir a frecuencia máxima, donde el ridge no se movió.
  Es el caso más interesante para un clasificador en vivo, y **es
  precisamente de la suite que §6 propone ampliar**.

La arquitectura no abandona el clasificador en vivo: lo reorienta hacia
la granularidad de carga como **mecanismo primario** (con el precedente
triple Guerreiro/Calore/Antici) y conserva la clasificación por ventana
como **mecanismo secundario** para los casos con mezcla real —
`npb_lu`, `npb_bt` y `rajaperf_polybench_3mm_omp`. No es una arquitectura
nueva: es priorizar la que el dato respalda, sin descartar la otra.

## 7.ter LULESH y HPCG: negativo en mezcla de fase, pero HPCG da el mayor margen de EDP del catálogo (job 6616+6617, 2026-08-27)

Campaña completa, 72/72 corridas aceptadas (62 en el primer intento, 0
rechazadas; 10 restantes tras que el timeout INTERNO de la campaña
—4.5h, no el de Slurm— cortara antes de tiempo; reanudación automática
del orquestador, ARC-142, completó el resto sin repetir nada).

**Mezcla de fase intra-corrida: NEGATIVO, con número.** Ambos kernels
salen prácticamente homogéneos en los seis niveles — fracción de ventana
minoritaria entre 0.00% y 0.08%, muy por debajo del 11.8%/19.0% real de
`npb_bt`/`npb_lu` (§7). Ni las fases físicas explícitas de LULESH (por
diseño del algoritmo) ni el ciclo multigrid de HPCG (SpMV/suavizado/
restricción) producen alternancia de régimen aprovechable con estas
features. **Con esto, GAP + LULESH + HPCG — los tres candidatos del
pivote de catálogo motivado por C8 — cierran negativo en el objetivo que
los motivó**: el subconjunto con mezcla real aprovechable sigue siendo
`npb_lu`/`npb_bt`/`rajaperf_polybench_3mm_omp`, tal como ya lo confirmó
C8 (§7.bis). No es un resultado menor: tres intentos deliberados,
dirigidos por criterio algorítmico razonado (frontier de grafo, fases
físicas por timestep, ciclo multigrid), y ninguno reprodujo el fenómeno
— refuerza que la mezcla real de fase es una propiedad específica de
esos tres kernels y del desplazamiento del ridge a bajo reloj (ARC-175),
no algo que cualquier kernel "complejo" produzca por tener múltiples
etapas algorítmicas.

**Margen de EDP: mixto, con el mejor resultado del catálogo hasta hoy.**

| kernel | mejor nivel | EDP/F0 | lectura |
|---|---|---:|---|
| **`cpu_hpcg`** | **F2** | **0.9082 (−9.18%)** | óptimo real no monótono (F1 peor que F0, F2 mucho mejor) — mismo patrón que `npb_mg`, pero el margen más grande confirmado en el catálogo |
| `cpu_lulesh` | REF/F0 | 0.9999–1.0000 | sin margen, degrada monótono como la mayoría del catálogo original |

Confirma otra vez el hallazgo de §6.octies: **el α de tamizaje no predice
la magnitud ni siquiera la existencia del margen real de EDP.** LULESH
tenía el α más alto de los dos candidatos (0.533 vs 0.324 de HPCG) y
terminó sin ningún margen; HPCG, con α más bajo y ajuste más ruidoso
(r²=0.903), dio el mejor resultado del catálogo. El tamizaje separa
candidatos de compute-bound puro, no ordena cuáles tendrán mínimo real.

**Balance del catálogo CPU tras este resultado**: 9 kernels con margen
real de EDP confirmado (`npb_mg`, los 6 de RAJAPerf v2 con margen,
`cpu_hpcg`), de 19 con campaña completa. `ptrchase` sigue aparte como
sujeto latency-bound (α bajo umbral, no tabulado en EDP de la misma
forma). `cpu_lulesh`, `Stream_TRIAD`, `Stream_ADD` y los 7 originales
compute-bound completan la lista sin margen.

## 7.bis El clasificador de ventana SÍ funciona donde hay fase real (C8, 2026-08-26)

Pregunta directa, motivada por el texto literal del Objetivo 2 ("clasificar
(...) las fases de ejecución"): el resultado de §5.bis de
`resultados_compuertas_fase2.md` (F1 macro 0.393 vs. 0.371 trivial, sobre
los 9 kernels completos) mostraba que el clasificador de ventana no le
ganaba al trivial — pero ese número mezcla 6 kernels 100% homogéneos con
los 3 que sí tienen mezcla, y un modelo LOKO no tiene de dónde aprender
fase en un kernel que nunca cambia de clase. La pregunta pendiente era si
el fracaso era de las *features/modelo* o de la *composición del
catálogo de prueba*.

**Se entrenó `train_phase.py` (mismas 7 features, mismo protocolo LOKO,
sin CAP_PERFMON) restringido a los 3 kernels con mezcla real —
`npb_lu` (19.0%), `npb_bt` (11.8%), `rajaperf_polybench_3mm_omp` (3.2%).
360 000 ventanas, sin necesitar ningún kernel nuevo: el dato ya existía
en la campaña `arc174`.**

| modelo | F1 macro | vs. trivial (0.170) |
|---|---:|---:|
| `extra_trees` | **0.538** | **3.2×** |
| `random_forest` | 0.518 | 3.0× |
| regresión logística | 0.490 | 2.9× |
| árbol prof. 6 | 0.427 | 2.5× |
| árbol prof. 1 | 0.321 | 1.9× |
| mayoritaria (trivial) | 0.170 | — |

Por pliegue, `extra_trees` generaliza fuerte a un kernel nunca visto:
0.783 en el pliegue `npb_bt` (entrenado con `npb_lu`+`3mm_omp`), 0.643 en
`npb_lu`. El pliegue más débil es `3mm_omp` (0.187) — es también el que
tiene menos mezcla (3.2%, la clase minoritaria casi no aparece ni para
entrenar ni para evaluar), consistente con el patrón, no una excepción.

**Lectura para el Objetivo 2, sin sobrevender:** el fracaso de §5.bis no
era de las features ni del modelo — era que 6 de 9 kernels del catálogo
son homogéneos de principio a fin, y eso ahoga la señal de los otros 3
en el promedio. Donde el fenómeno que el objetivo pide clasificar existe
de verdad, **el clasificador lo aprende y generaliza a un kernel nuevo**,
con las mismas 7 features baratas y el mismo protocolo LOKO estricto. Eso
es evidencia directa a favor de la lectura literal del Objetivo 2, no
solo de su reinterpretación por carga (§4/§7) — **las dos cosas conviven
en el catálogo actual, sin necesitar `GUPS`/`HPCG`/`LULESH`/`PARSEC`
(§6.septies) para demostrarlo.** Esos candidatos siguen valiendo como
trabajo futuro para ampliar el subconjunto con mezcla real más allá de 3
kernels, pero ya no son necesarios para el veredicto de "¿se puede,
cuando el fenómeno existe?" — esa pregunta ya tiene respuesta.

## 8. Mapeo a los Objetivos Específicos

| Objetivo | Estado | Evidencia |
|---|---|---|
| 1. Caracterizar comportamiento y consumo bajo distintos estados de frecuencia (Perf+RAPL) | **Cumplido** (424/540 válidas) y **ampliándose por tres frentes reabiertos**: rejilla fina 3200–2600 (E13 ya destrabado), tamizaje v2 con conjunto de trabajo 10× la LLC, y `ptrchase`/`phasic_*` en curso | `..._arc174`; §2, §5, §6, §6.bis, §6.ter |
| 2. Clasificador ML en vivo, baja latencia | Features ya libres de uncore; granularidad primaria = carga, secundaria = ventana en los 3 kernels con mezcla real | §1, §7 |
| 3. Daemon de espacio de usuario con política DVFS | **No bloqueado por CAP_PERFMON**: el runtime nunca dependió de uncore, y la actuación (`scaling_min/max_freq`) ya tiene permiso P1 | §1 |
| 4. Evaluación por EDP contra gobernador nativo | **Parcial**: `REF` cubre el gobernador nativo *activo* (`performance`) sin permisos nuevos; permiso para `powersave` concedido (2026-08-25), pendiente de verificación empírica antes de usarse | §4; riesgo 3 |

## 9. Lista de pruebas — cómo verificar o refutar esta propuesta

| # | Qué verifica | Cómo | Pasa si | Si falla |
|---|---|---|---|---|
| **C1** | ¿El tamizaje encuentra margen en CPU? (§5.1, §6) | Job 6483: ajustar α por candidato sobre tiempo medido | ≥1 candidato con α < 0.226 **y** r² > 0.95 | **VEREDICTO RETIRADO (2026-08-25).** El 0/7 de 6483 midió el tamaño del problema, no los kernels: 32 MB/rep contra 39 MB de L3, todo en cache. Rehecho con `--memory-touched` a 10× la LLC y sobre los ~79 kernels (`screen_rajaperf_cpu_alpha_v2.sh`) — ver el aviso de §6 |
| **C1b** | ¿Hay margen en el catálogo ACTUAL, con la ventana correcta? (§6.ter) | Reajustar α por ventana de frecuencia (job 6543, sonda); confirmado con campaña completa 10 reps/nivel (jobs 6412/6530, §6.quinquies) | ≥1 kernel con α < 0.226 en la ventana que la política usaría | **PASA, confirmado con campaña completa (2026-08-26): 2 kernels.** `npb_mg` óptimo real en S3000 (EDP/F0=0.9927, −0.73%); `ptrchase` α=0.097 en F0–F1 (r²=1.000) / 0.122 en rango completo |
| **C1c** | ¿Se frena la memoria al bajar el reloj? (§6.bis) | Medir `uncore_imc_0/clockticks` sobre reloj de pared a cada nivel (job 6542) | El reloj del IMC cae junto con el del núcleo | **NO SE CUMPLE, y es buena noticia:** el IMC se mantiene en 2.75–2.89 GHz mientras el núcleo cae 4×. La memoria no se ralentiza; lo que cae es el BW alcanzado (−30%) porque el núcleo lento no la satura |
| **C2** | **El pineo de frecuencia se sostuvo bajo carga** (riesgo 6) | Muestrear `scaling_cur_freq` *durante* la corrida, no antes | Media observada dentro de 5% del objetivo en los 6 CPUs | **PASÓ en 5/7** tras corregir el bug de hermanos SMT (job 6483); `ATAX`/`GESUMMV` fallan pero ya fallan C1 con margen amplio, no cambia el veredicto |
| **C3** | El tamizaje no está contaminado por I/O (lección `myocyte`, Anexo L.1) | Verificar tamaño de los archivos que RAJAPerf escribe por corrida | Salida despreciable frente al tiempo de cómputo | Un α bajo puede ser costo fijo de I/O, no memory-boundness: descontarlo antes de aceptar el candidato |
| **C4** | Robustez de la conclusión de 0.33 pts (§2) | Reejecutar `cpu_policy_headroom.py` optimizando **EDP** en vez de energía | La conclusión no cambia cualitativamente | Si bajo EDP aparece margen, la métrica —no la física— era la limitante, igual que pasó en GPU (Anexo M) |
| **C5** | `REF` ≡ gobernador nativo activo (§4) | Leer `scaling_governor` en cada corrida, no asumirlo | `performance` en los 6 CPUs delegados | La comparación del Objetivo 4 no es contra lo que se afirma: corregir antes de reportar |
| **C6** | Anti-fuga de etiqueta (§1) | Test unitario: inyectar a propósito una columna `FORBIDDEN` | El test **falla** ruidosamente | El guardarraíl no protege: todo resultado del modelo queda invalidado hasta arreglarlo |
| **C7** | ¿La arquitectura corregida mejora sobre el trivial? (§5.3) | Reentrenar por carga, LOKO, contra el predictor trivial (0.371 F1 macro) y contra la mejor constante elegida **solo en folds de entrenamiento** | Supera al trivial de forma significativa, no por décimas | La granularidad por carga tampoco alcanza en CPU: el aporte del eje CPU es la caracterización (Objetivo 1), y hay que decirlo |
| **C8** | Mecanismo secundario por ventana (§7) | Entrenar solo sobre `npb_lu`/`npb_bt`/`3mm_omp`, que sí tienen mezcla | Mejora sobre el trivial *dentro* de ese subconjunto | **PASA (2026-08-26), y con margen grande.** F1 macro 0.538 (`extra_trees`) frente a 0.170 de la mayoritaria — 3.2× mejor, no una diferencia de décimas. Ver §7.bis |
| **C9** | Latencia de inferencia (Objetivo 2) | p50/p99 de una inferencia (ya medido antes: 103 µs árbol, 33 ms ensembles) | p99 « período de decisión del daemon | Descartar los ensembles y quedarse con árbol/regresión, como ya sugieren los datos previos |
| **C10** | Consistencia documento↔dato | Reejecutar `cpu_policy_headroom.py` y comparar contra §2 | Coinciden | Actualizar: los números deben salir del script, nunca copiarse a mano |

---

## Riesgos abiertos, sin adornar

1. **REABIERTO (2026-08-25), y el número que lo reabrió se corrigió al
   día siguiente (2026-08-26).** Se dio por cerrado con el 0/7 del job
   6483; el veredicto se retiró razonando que los 32 MB/rep de
   `JACOBI_1D` cabían en una L3 de 39 MB. **Ese 39 MB era la L3 de
   `pacca01`, leída por error en vez de la de `paccaA100`** (donde 6483
   corrió de verdad) — la real es 12 MB. Con el número correcto, los 7
   candidatos exceden la LLC entre 1.3× y 6.7×, no caben claramente pero
   tampoco la exceden con holgura (ver aviso corregido de §6). El
   diagnóstico de tamaño sigue siendo la hipótesis más plausible —
   ninguno de los 7 estaba realmente lejos de la LLC — pero ya no está
   *probado* con el margen que se afirmó. Sigue abierto hasta
   `screen_rajaperf_cpu_alpha_v2.sh` (job 6575, `--memory-touched` a
   ~120 MB, 10× excedente inequívoco).
   **Lección propia, dos veces en el mismo hilo:** la primera vez por no
   verificar el tamaño de cache en absoluto; la segunda por verificarlo
   en el nodo equivocado — el mismo tipo de comparación cruzada
   `pacca01`/`paccaA100` que este proyecto ya tiene documentada como
   peligrosa y que se cometió de todos modos. La próxima vez que un
   número de hardware entre en un argumento, verificar `hostname` en el
   mismo comando que lo lee.
2. **REABIERTO, mismo motivo.** Los 7 candidatos se eligieron por
   conocimiento algorítmico y no por OI medida. La v2 elimina ese sesgo
   tamizando los ~79 kernels de la suite en vez de 7 elegidos a mano —
   mismo criterio que ya se aplicó en el eje GPU.
2.bis **NUEVO, y es el hallazgo que reabre el eje CPU:** hay **dos**
   kernels que sí cumplen el umbral en la ventana que la política usaría
   — `npb_mg` (α=0.171 en F0–F1) y `ptrchase` (α=0.096 en F0–F1). El
   catálogo CPU **no** está vacío de sujetos viables; lo estaba el
   análisis, por ajustar α sobre una ventana que incluye niveles
   inalcanzables (§6.ter).
3. **`powersave` — el permiso ya se concedió (2026-08-25), pero no está
   verificado.** El administrador entregó `set_cpu_gov <gobernador> <epp>`,
   restringido a los cores 0-5 y sin parámetro de rango de CPUs. Su propio
   correo argumenta que exclusividad + *pinning* de la carga (ya
   implementados en el runner, `--exclusive` + `sched_setaffinity` vía
   `--pin-workload-cpus`) bastan para neutralizar la interferencia de los
   hermanos SMT (16-21). **Nuestra propia Prueba A de ARC-162 contradice
   esa conclusión para `scaling_min_freq`/`scaling_max_freq`**: con la
   carga ya confinada a 0-5 por `taskset` (la misma receta), el reloj
   físico no escaló (razón F0/F4 = 0.995) hasta que el candado de
   frecuencia se aplicó TAMBIÉN a los hermanos 16-21 (razón = 3.78,
   ARC-163). Si el mismo mecanismo de coordinación de P-state aplica a
   gobernador/EPP, `set_cpu_gov` sobre 0-5 podría no bajar el reloj físico
   compartido — y el script entregado no permite aplicarlo también a
   16-21. **Pendiente una prueba tipo Prueba A/B antes de usarlo en
   cualquier campaña real** (`ert_probe` bajo `powersave power`, medir
   `scaling_cur_freq`/potencia real). Abierto a propósito hasta que haga
   falta para la comparación de la Fase 4 contra ambos gobernadores
   nativos — no bloquea ninguna corrida ni prueba en curso hoy.
4. **CAP_PERFMON sigue sin fecha.** El plan de §5 avanza sin esperarlo,
   pero **la ampliación del dataset con etiqueta de verdad sí depende de
   él**: el bypass ARC-191 exime solo datasets 100% GPU, no kernels
   `device: cpu` reales.
5. **El modelo con la arquitectura corregida todavía no se ha corrido** —
   es el paso siguiente, no un resultado obtenido.
6. **RESUELTO (2026-08-25).** El script de tamizaje original (job 6475)
   solo pineaba los cores delegados (0-5), no sus hermanos SMT (16-21) —
   mismo bug de coordinación de P-state que ARC-162/163: bajo
   `intel_pstate`, el reloj físico compartido no baja hasta que el candado
   se aplica también a los hermanos. Corregido en
   `screen_rajaperf_cpu_alpha.sh` con expansión de topología SMT en vivo
   (`smt_siblings_for()`/`_expand_with_smt_siblings()`, leyendo
   `/sys/.../thread_siblings_list`); relanzado como job 6483, verificado
   con variación de frecuencia real (r² 0.975-0.999 en los 7 candidatos,
   frente al bug original que producía frecuencia plana). Los resultados
   del job 6483 (§6) ya no son provisionales.

---

## Referencias

Ya en `docs/libro/main.tex`: `\cite{Calore2017}`, `\cite{Hebbar2022}`,
`\cite{Guerreiro2019}`, `\cite{Antici2024}`, `\cite{Williams2009}`.
