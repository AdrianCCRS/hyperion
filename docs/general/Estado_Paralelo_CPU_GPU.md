# Estado paralelo CPU / GPU — 2026-08-26

Documento vivo, condensado. Actualizar en el mismo turno en que cambie
algo relevante — no dejar que se desactualice como pasó con
`Estado_Cola_Slurm.md` antes de esta convención.

**No mezclar los dos ejes al leer esto** — son catálogos, umbrales y
modelos independientes.

> **GPU en standby a propósito (2026-08-26 noche).** Job 6600 (tamizaje
> α GPU v2) cancelado sin correr — no por el nodo, por alcance. C8
> (`Estrategia_CPU_Fase2.md` §7.bis) mostró que el clasificador de fase
> por ventana funciona donde hay mezcla real; el camino para mejorarlo
> más es ampliar el catálogo CPU con kernels que produzcan esa mezcla
> (GAP → LULESH → HPCG, §6.septies), no seguir el eje GPU en paralelo con
> el tiempo que queda. Se retoma cuando el frente CPU cierre.

---

## MODELO CPU

**Catálogo**: 9 kernels originales (Clase B) + 9 del tamizaje v2, **17
con campaña completa, 8 con margen real de EDP confirmado**.

| viable (margen real) | mejor nivel | EDP/F0 | fuente |
|---|---|---:|---|
| `Lcals_FIRST_SUM` | F2 | 0.9513 (**−7.09%**) | campaña completa, job 6594, 324/324 |
| `Stream_MUL` | F1 | 0.9548 (−4.92%) | campaña completa, job 6594 |
| `Basic_INIT3` | F1 | 0.9698 (−4.19%) | campaña completa, job 6594 |
| `Lcals_TRIDIAG_ELIM` | F1 | 0.9798 (−3.46%) | campaña completa, job 6594 |
| `Polybench_FDTD_2D` | REF | 0.9647 (−1.62%) | campaña completa, job 6594 |
| `Polybench_JACOBI_1D` | REF | 0.9681 (−1.37%) | campaña completa, job 6594 |
| `npb_mg` | S3000 | 0.9927 (−0.73%) | campaña completa, 638/720 |
| `Basic_DAXPY` | REF | 0.9932 (−0.21%) | campaña completa, job 6594 |
| `ptrchase` | — | α=0.097 (F0–F1, r²=1.000) | campaña completa, 320/320 (sujeto latency-bound, no EDP tabulado igual) |

`Stream_TRIAD`/`Stream_ADD` completaron campaña sin margen (EDP/F0=1.0,
óptimo en F0) — confirmados sin viabilidad, no pendientes.

**Hallazgo importante de esta campaña**: el α del tamizaje **no ordena**
el ahorro real de EDP (`Stream_MUL`, el α más bajo, no es el de mayor
ahorro; `Stream_TRIAD/ADD`, α medio, dan cero). El tamizaje sirvió para
separar candidatos de los 70 compute-bound, no para predecir magnitud —
ver detalle en `Estrategia_CPU_Fase2.md` §6.octies.

Los otros 7 originales + 70 del tamizaje v2 son compute-bound genuinos —
no artefacto de tamaño ya corregido (10× LLC real). Excepción parcial:
`npb_cg`/`npb_mg` en Clase C (8× memoria) bajan de α pero no cruzan el
umbral (0.765→0.530 y 0.409→0.335) — el eje de tamaño funciona en
dirección, no alcanza a esa escala.

**Bloqueado**: GAP Benchmark (`bfs`/`pr`) — binarios listos, sin permiso
de escritura de frecuencia en `pacca01`. Huella de caché a reloj nativo
(`perf`) da resultado incierto. Pendiente de tamizar directo en
`paccaA100`.

**El modelo — PRIMER RESULTADO POSITIVO DEL PROYECTO (2026-08-26).**
Reentrenado sobre 17 kernels (arc174 + job 6594), sin `ref_running_ratio`
(varianza cero) y con el umbral de acción restringido a la región REF–F2
(evita que F3/F4, hasta 12× peor, infle el RMSE del umbral). Resultado:

| política | EDP loss | vs. trivial |
|---|---:|---:|
| modelo sin umbral | 1.0045 | **gana +0.0077** |
| modelo + umbral (región accionable) | 1.0072 | **gana +0.0050** |
| trivial (siempre F0) | 1.0122 | — |

Captura 41-63% de los 1.22 puntos de margen disponible. Alcance honesto:
margen modesto (10 de 17 kernels sin ganancia real), pero por primera vez
hay algo que ganar y un modelo que lo hace sin arriesgar una regresión —
detalle completo en `Estrategia_CPU_Fase2.md` §6.nonies.

---

## MODELO GPU

**Catálogo confirmado**: 7 originales + 6 RAJAPerf-CUDA con OI real medida
por `ncu` (job 6528) = 13, checksum verificado. Manifiesto de 17 kernels
listo, **todavía no lanzado**.

**Clasificación de cuello de botella (job 6571, corregido tras el fallo
silencioso de 6539) completada: 43 de 75 kernels CUDA son MEMORY_BOUND**
(DRAM%>SM%, DRAM%≥30%) — salto grande frente a los 6 tamizados a mano
antes. Los más extremos: `Lcals_TRIDIAG_ELIM` (91.5% DRAM), `Apps_PRESSURE`
(90.3%), `Stream_ADD/TRIAD` (~90%), varios `Lcals_*`/`Algorithm_MEMCPY`
`/MEMSET` (~88%). **Esto es DRAM% vs SM%, no α** — clasifica candidatos,
no los confirma; falta el tamizaje de α con reloj variable sobre estos 43
antes de decidir el catálogo final, paso caro (~43×5 corridas) sin
lanzar todavía.

**El modelo** (piloto LOKO, dataset viejo de 6 kernels): pierde contra no
hacer nada, peor que CPU (EDP loss 1.0925 vs trivial 1.0507, −4.18 pts).
Margen real disponible: 5.07 pts. RMSE 5.4× el margen (menos grave que
CPU). Umbral de acción acota la pérdida a cero, no da ganancia. No se ha
reentrenado — pendiente del catálogo final (tamizaje de los 43 + dataset).

---

## En común

- Mismo diagnóstico raíz en los dos ejes: N efectivo pequeño (kernels, no
  filas) es la causa del fracaso del modelo, no un problema de ajuste.
- **CPU ya se reentrenó (2026-08-26) y gana al trivial** — ver arriba.
  Mejoras aplicadas: quitar `ref_running_ratio`, restringir el umbral de
  acción a la región REF–F2. Pendiente en GPU (no aplicable todavía: no
  tiene `ref_running_ratio` como feature, pero sí podría beneficiarse de
  restringir niveles accionables una vez tenga catálogo final).
- Mejora identificada y sin hacer en ninguno de los dos: enriquecer con
  percentiles/dispersión de la corrida de referencia en vez de solo la
  media (ataca N efectivo por el lado de la riqueza del punto, más
  relevante en GPU con solo 2 features).
- **GPU sigue sin reentrenarse** — espera el catálogo final (job 6600).

## Candidatos de catálogo futuro (estudio 2026-08-26, sin lanzar)

Investigación de escritorio, ningún kernel tocado en el nodo todavía.
Detalle completo en `Estrategia_CPU_Fase2.md` §6.septies y
`Estrategia_GPU_Fase2.md` §8.bis.

**CPU** — llenar la banda de α intermedia (0.18–0.6, vacía hoy) y
patrones de acceso más allá del streaming regular de los 9 nuevos:
GUPS/RandomAccess (irregular puro, complementa a `ptrchase`), HPCG
(dispersa con reutilización real, candidato natural de banda intermedia),
LULESH (malla no estructurada), PARSEC/`canneal` (licencia por
confirmar). GAP Benchmark ya está en curso, ver §"Bloqueado" arriba.

**GPU** — mismas suites que ya usa `Guerreiro2019` (35 kernels, el
trabajo con el que más se compara este eje) y que aquí nunca se tocaron:
Parboil (mezcla compute/memoria por diseño) y SHOC (SpMV/Stencil2D como
banda intermedia). Ambas libres, sin licencia paga.

**Condición antes de comprometer nodo en cualquiera de las dos**: esperar
a que cierren 6594 (CPU) y 6595 (GPU) y ver si el catálogo ampliado
resultante ya alcanza — no partir de "más kernels" sin esa evidencia.

## Hallazgos transversales (aplican a ambos ejes o los conectan)

- **La memoria de CPU no se ralentiza con el reloj de núcleo** (medido,
  `uncore_imc_0/clockticks`): el controlador se mantiene en 2.68–2.89 GHz
  mientras el núcleo cae 4×. Lo que cae es el ancho de banda alcanzado
  (−30%) — el núcleo lento no logra saturar una memoria que sí va rápido.
  Límite de carga/catálogo, no de plataforma.
- **GPU solo tiene un reloj de memoria soportado** (1215 MHz, `nvidia-smi
  --query-supported-clocks=mem`) — no hay segundo mando de actuación en
  este acelerador, a diferencia de CPU donde el `uncore` sí es un dominio
  independiente.
- **Overhead de instrumentación caracterizado en CPU**: 1.95% media sobre
  540 pares, estable entre kernels. No cuantificado por separado en GPU.
- **Tamaño de kernel ≠ memory-bound**: exceder la caché es necesario pero
  no suficiente. La intensidad operacional (reutilización de datos, no
  volumen) es lo que decide — mismo principio que llevó a corregir el
  veredicto del tamizaje v1 de CPU y a diseñar v2 con `--memory-touched`.
