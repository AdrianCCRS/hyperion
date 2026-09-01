# Consolidación de kernels para el dataset de Fase 1 (CPU + GPU)

> **Documento histórico, superado como inventario operativo.** La composición
> vigente se consulta en `orchestrator/schemas/kernels/catalog.yaml`: 9 entradas
> CPU (7 dataset + 2 calibración) y 12 GPU (8 dataset + 4 calibración). La
> auditoría dinámica posterior de ambas precisiones retiró `rodinia_hotspot`,
> incorporó `rodinia_gaussian` y corrigió la precisión declarada de otros
> kernels (ARC-110). Los valores obtenidos a 765 MHz y las afirmaciones de P4
> bloqueado que aparecen abajo describen un diagnóstico histórico no concluyente
> bajo 595.45.04; el mecanismo ya era funcional y ARC-137 verificó `-lgc` a 600
> y 1200 MHz bajo carga, sin atribuirlo al driver. No usar
> este cuerpo para generar una campaña actual.

Este documento junta, en un solo lugar, la lista de kernels con la que se
va a construir el dataset de entrenamiento de Fase 1 (caracterización
Roofline, ver `docs/general/plan_trabajo_grado.md`), combinando lo ya
validado en CPU (`docs/retoma/Propuesta_Seleccion_Kernels_Dataset.md`,
ARC-38/57) con lo recién medido en GPU (ARC-70/72/75/76).

No es una decisión de alcance definitivo (eso sigue siendo H4, del
director) -- es el inventario real, con medición real, de lo que ya está
compilado y verificado en hardware, listo para correr una campaña completa
a la frecuencia REF mientras se espera el permiso de `cpufreq` (P4).

**Revisión 3** de este documento. Historial de correcciones:
- Rev. 2: una revisión técnica encontró que el ridge point de GPU mezclaba
  un techo de Tensor Core con kernels que no los usan, y varios problemas
  de redacción (atribuir a "la literatura Rodinia" clasificaciones que en
  realidad eran nuestra propia hipótesis previa, más un error aritmético
  de conteo).
- Rev. 3 (esta): el usuario señaló, con razón, que el plan de tesis nunca
  mencionó Tensor Cores y que la solución de la Rev. 2 (tres ridge points
  distintos según Tensor Core sí/no) metía más profundidad de la
  necesaria en ese tema. Se reemplazó por algo más simple: en vez de
  derivar el techo de cómputo de `cuBLAS` (que puede elegir Tensor Cores
  por su cuenta, sin pedirlo), se escribió un microbenchmark propio
  (`ert_probe_gpu.cu`, ARC-76) que mide el pico real alcanzable en
  aritmética CUDA corriente -- mismo principio que `ert_probe.c` ya usa en
  CPU. `cuBLAS`/Tensor Cores dejan de ser un tema del proyecto: se
  mencionan aquí solo como la razón por la que se descartó `cuBLAS` como
  fuente del ridge, no como algo que el proyecto use o estudie.

## 0. Cómo se mide el techo de cómputo de la GPU (`P_pico`)

**Qué pasó y por qué se corrigió.** Para calcular el ridge point
(`i_ridge = P_pico / BW_pico`, la frontera entre compute-bound y
memory-bound) hace falta un número de "velocidad máxima de cómputo de la
GPU". La primera medición (ARC-70/72) usó `cublas_dgemm_bench`
(`cublasDgemm`, la función de multiplicación de matrices de NVIDIA) para
eso. Pero `ncu` confirmó que `cuBLAS`, por su cuenta y sin que se le
pidiera, ejecutó esa multiplicación con una ruta de hardware acelerada
(Tensor Cores) que los demás kernels del catálogo (Rodinia, CUDA escrito a
mano) no pueden usar -- comparar contra ese número era comparar peras con
manzanas, el mismo tipo de error de unidades que ya se corrigió en CPU con
`npb_ep`/`npb_is` (ARC-57).

**La solución, deliberadamente simple**: en vez de razonar sobre qué ruta
de hardware usa cada kernel, se dejó de usar `cuBLAS` para el ridge por
completo. Se escribió `kernels/gpu/ert_probe_gpu.cu` (ARC-76) -- un
microbenchmark propio, equivalente en CUDA de `ert_probe.c`: cada hilo
hace un bucle de operaciones multiplicar-sumar encadenadas **enteramente
en un registro** (sin `cuBLAS`, sin ninguna librería de NVIDIA que pueda
elegir una ruta de ejecución por su cuenta). Da el mismo tipo de número
que `ert_probe` ya da en CPU: "esto es lo máximo que este hardware puede
sostener con aritmética corriente", sin ningún atajo de por medio.

**Medido en `paccaA100` real** (`gpu_ert_probe_fp32`/`gpu_ert_probe_fp64`):

| Precisión | GFLOP/s alcanzado (probe propio) | % del pico teórico NVIDIA |
|---|---|---|
| FP64 | 4698.6 GFLOP/s | ≈48% de 9.7 TFLOP/s |
| FP32 | 10178.2 GFLOP/s | ≈52% de 19.5 TFLOP/s |

**Por qué ronda el 50% -- verificado, no asumido (ARC-77).** El ~50% podía
deberse a dos causas muy distintas: (a) un error del propio probe (mal
conteo de FLOPs, poco paralelismo), o (b) un límite real del nodo, ajeno
al código. Se investigaron ambas con medición directa, no se asumió
ninguna:

1. **Conteo de FLOPs verificado con `ncu`**: se perfiló `ert_probe_gpu`
   contando las instrucciones FMA que la GPU ejecutó de verdad
   (`sm__sass_thread_inst_executed_op_dfma/ffma_pred_on.sum`). El conteo
   coincide exacto con lo esperado (`elementos × 8 × repeticiones`), y el
   GFLOP/s recalculado desde esas instrucciones reales coincide con lo que
   reporta el propio programa (diferencia <1%). **No hay bug de conteo**
   -- cada FMA ya se contaba como 2 FLOPs correctamente.
2. **Paralelismo a nivel de instrucción**: se probó una variante con 4
   acumuladores independientes por hilo (rompiendo la cadena de
   dependencia secuencial de la versión original) -- la mejora fue de solo
   4-8%, no explica el ~50% faltante.
3. **Reloj real de la GPU durante la ejecución** (no en reposo):
   `nvidia-smi` mientras el kernel corría mostró **765 MHz de 1410 MHz
   máximos (54.3%)**, con consumo de solo 62 W de 250 W disponibles y 26°C
   -- no hay límite térmico ni de potencia, la GPU simplemente no sube de
   reloj. **54.3% de reloj disponible explica casi exactamente el 48-52%
   de FLOPs alcanzados.**

**Conclusión**: el ~50% no es un defecto del probe, es el mismo límite ya
documentado en ARC-62 -- sin el permiso de administrador (`P4`, bloqueado),
no hay forma de fijar ni pedir el reloj máximo de esta GPU (`nvidia-smi
--applications-clocks-permission` está deprecado en este driver). **Esto
hace que el número medido sea el ridge *correcto* a usar, no uno
inferior a un hipotético "real"**: ningún kernel del catálogo (Rodinia,
cuBLAS, o cualquier otro) tiene tampoco ese permiso, así que todos corren
bajo la misma limitación de reloj. Comparar contra un pico de literatura
medido con reloj desbloqueado compararía a nuestros kernels contra una
velocidad que ninguno puede alcanzar hoy en este nodo -- sería reintroducir
exactamente el mismo error de la sección original (comparar contra un
techo que el resto de kernels no puede tocar), solo que por reloj en vez
de por Tensor Cores.

**Ridge points resultantes** (con el ancho de banda medido de
`gpu_stream_bw`, 1.399 TB/s -- GPU confirmada por SKU real, no asumida:
`nvidia-smi --query-gpu=name,memory.total` da A100-PCIe-40GB, 1555 GB/s
teóricos, así que el 89.9% de `gpu_stream_bw` sí aplica a este nodo):

- **FP64 vainilla: i_ridge ≈ 3.36 FLOP/byte** (4698.6 GFLOP/s ÷ 1399 GB/s)
- **FP32 vainilla: i_ridge ≈ 7.28 FLOP/byte** (10178.2 GFLOP/s ÷ 1399 GB/s)

**Precisión real de cada kernel** (verificada leyendo el `.cu`, no
asumida):

| kernel | precisión | ridge aplicable |
|---|---|---|
| `backprop`, `hotspot`, `lud`, `myocyte`, `dwt2d` (9/7), `heartwall` | `float` (FP32) | 7.28 |
| `lavaMD` | `double` (FP64, `#define fp double`) | 3.36 |
| `cublas_dgemm_bench` | `double`, vía `cuBLAS` (ruta acelerada, no vainilla) | no aplica ninguno de los dos -- ver nota abajo |

### Clasificación final, con el ridge medido

| kernel | precisión | OI medida (`ncu`, `dram__bytes.sum`) | ridge | Clasificación |
|---|---|---|---|---|
| `cublas_dgemm_bench` (N=4096) | FP64 (cuBLAS) | 68.0 FLOP/byte | cualquiera (3.36 o 7.28) | compute_bound -- tan por encima de ambos que no importa cuál se use |
| `lavaMD` | FP64 | 1233 FLOP/byte | 3.36 | compute_bound, margen enorme |
| `heartwall` | FP32 | 35.3 FLOP/byte | 7.28 | compute_bound |
| **`lud`** | FP32 | 7.6-7.8 FLOP/byte | **7.28** | **intermedio/borderline** -- la OI cae a menos de 7% por encima del ridge, un caso genuinamente ambiguo, no claramente de un lado |
| `hotspot` | FP32 | 5.03 FLOP/byte | 7.28 | memory_bound, cerca del ridge |
| `dwt2d` (9/7) | FP32 | 4.10 FLOP/byte | 7.28 | memory_bound |
| `backprop` | FP32 | 0.087 FLOP/byte | 7.28 | memory_bound, margen amplio |
| `myocyte` | FP32 | 0.017 FLOP/byte | 7.28 | memory_bound, margen enorme |

**`lud` es el único caso donde el ridge importa de verdad**: con 7.6-7.8
FLOP/byte contra un ridge de 7.28, la clasificación es **intermedio**, no
un lado claro -- se deja que la ventana lo confirme empíricamente en cada
corrida (mismo criterio que `npb_sp`/`npb_bt` en CPU). Todos los demás
kernels están tan lejos del ridge (por encima o por debajo) que ninguna
imprecisión razonable en su medición les cambiaría la clasificación.

**`cublas_dgemm_bench` sigue en el catálogo como kernel de *dataset*
compute-bound** (68.0 FLOP/byte no deja duda) -- lo que cambió es que ya
no se usa como *fuente del ridge*. Sigue siendo útil como referencia de
"qué tan rápido puede ir cuBLAS en este nodo", simplemente no como vara
para medir a los demás.

**Caveat pendiente sobre `lavaMD` (1233 FLOP/byte, el valor más alto del
lote):** la intensidad se mide como FLOPs sobre bytes de tráfico DRAM
(`dram__bytes.sum`, mismo criterio que todos los demás kernels). El valor
es alto porque la configuración medida (`boxes1d=10`) es pequeña: la
mayoría del working set cabe en L2/registros y casi no genera tráfico DRAM
sostenido más allá de la carga inicial. Si el tamaño de `boxes1d` cambia
en el catálogo final, esta intensidad hay que volver a medirla, no
asumirla constante.

## 1. Kernels de CPU (7, ya en catálogo y verificados en pacca/felix)

La columna de porcentajes **no es "grado de compute-boundedness"** de un
kernel -- es la **fracción de ventanas** de esa corrida que el pipeline
etiquetó `compute_bound` (el resto cae en `memory_bound`/`intensity_undefined`).
Un kernel real produce muchas ventanas a lo largo de su ejecución, cada una
con su propia etiqueta.

| id | suite | hint de catálogo | Fracción de ventanas `compute_bound` (ARC-71, clase B / clase C) | Estado |
|---|---|---|---|---|
| `npb_ep` | NPB-OMP | -- | -- | **Retirado** (ARC-57): el contador `Mop/s total` que reporta no corresponde a FLOPs (es "Random numbers generated"), no es válido como numerador de intensidad operacional -- no es un juicio sobre si EP hace o no aritmética de punto flotante internamente |
| `npb_is` | NPB-OMP | -- | -- | **Retirado** (ARC-57): mismo problema de unidades -- su `Mop/s total` es "keys ranked" (ordenamiento entero), no FLOPs |
| `npb_bt` | NPB-OMP | intermedio | 85.6% / 85.4% | ✅ mayoría de ventanas `compute_bound`, estable entre tamaños |
| `npb_mg` | NPB-OMP | memory_bound | 99.9% / 99.9% | ✅ casi todas las ventanas `memory_bound`, muy estable |
| `npb_cg` | NPB-OMP | memory_bound | 92.7% / 93.5% | ✅ mayoría `memory_bound`, estable |
| `npb_sp` | NPB-OMP | intermedio | 58.2% / 59.3% | ✅ mezcla cercana a 50/50, consistente entre clases -- intermedio real |
| `npb_ft` | NPB-OMP | intermedio | 79.7% / 66.2% | ⚠️ la fracción se desplaza con el tamaño de malla -- coherente con ser FFT 3D (mariposas vs. transposición), no es ruido |
| `npb_lu` | NPB-OMP | intermedio | 88.4% / 89.0% | ✅ mayoría `compute_bound`, estable |
| `dgemm_n2048` | DGEMM-OpenBLAS | **compute_bound** (única etiqueta explícita) | **0/773 ventanas `ok`** | ❌ bug de `warmup_seconds` (ver sección 4) -- cuando sí hay dato la clasificación puntual es correcta, pero casi no genera ventanas usables |

Estas fracciones son observaciones empíricas de este hardware, esta
frecuencia (REF) y este tamaño de problema (clase B/C de NPB) -- no
propiedades universales del kernel.

**El hueco real señalado por el usuario**: no es que falte comportamiento
`compute_bound` en CPU (`npb_bt`/`npb_lu` ya lo dan, con datos reales) --
es que el único kernel con **etiqueta explícita** `compute_bound` en el
catálogo (`dgemm_n2048`) casi no produce ventanas utilizables. Corregirlo
es el punto 2 de la sección "Pendiente" más abajo.

## 2. Kernels de GPU

### 2.1 Calibración (no entran al dataset de entrenamiento)

| id | qué mide | resultado real en `paccaA100` |
|---|---|---|
| `gpu_stream_bw` | ancho de banda pico (BabelStream Triad) | 1.399 TB/s (89.9% del teórico de una A100-PCIe-40GB) |
| `gpu_ert_probe_fp32` | pico de cómputo FP32 vainilla (probe propio, ARC-76) | 10178.2 GFLOP/s -- fuente del ridge FP32 (7.28) |
| `gpu_ert_probe_fp64` | pico de cómputo FP64 vainilla (probe propio, ARC-76) | 4698.6 GFLOP/s -- fuente del ridge FP64 (3.36) |
| `gpu_dgemm_calibration` | referencia informativa de cuBLAS (no fuente del ridge, ver sección 0) | ≈10.4 TFLOP/s |

### 2.2 Dataset (ARC-70/72/75)

| id | dominio Rodinia (si aplica) | nuestra hipótesis previa | OI medida (`ncu`) | Clasificación final | ¿Coincidió la hipótesis? |
|---|---|---|---|---|---|
| `gpu_dgemm_n4096` | -- | Compute | 68.0 FLOP/byte | compute_bound | ✅ sí |
| `rodinia_lavamd` | N-Body / Molecular Dynamics | Compute | 1233 FLOP/byte | compute_bound, margen enorme | ✅ sí |
| `rodinia_heartwall` | Structured Grid / Medical Imaging | Compute | 35.3 FLOP/byte | compute_bound | ✅ sí |
| `rodinia_lud` | Dense Linear Algebra | Compute | 7.6-7.8 FLOP/byte | **intermedio/borderline** | ⚠️ ni compute ni memory claro -- "álgebra lineal densa" no implica compute-bound por sí solo |
| `rodinia_hotspot` | Structured Grid / Physics Simulation | intermedio (hint previo) | 5.03 FLOP/byte | memory_bound | ❌ no -- contradice nuestro hint previo de catálogo, no una clasificación oficial de Rodinia |
| `rodinia_dwt2d` (9/7) | Compresión | Memory | 4.10 FLOP/byte | memory_bound | ✅ sí |
| `rodinia_backprop` | -- | memory_bound | 0.087 FLOP/byte | memory_bound, margen amplio | ✅ sí |
| `rodinia_myocyte` | Structured Grid / Biological Simulation | Compute (hipótesis nuestra, no de Rodinia) | 0.017 FLOP/byte | memory_bound, dos órdenes de magnitud por debajo | ❌ no -- refuta la hipótesis inicial con margen grande |

**Nota importante sobre a qué contradice cada hallazgo**: Rodinia clasifica
sus benchmarks por *dominio de aplicación* (N-Body, Structured Grid, Dense
Linear Algebra...), no por posición en un techo Roofline de un hardware
específico. Que `hotspot`, `lud` o `myocyte` salgan `memory_bound`/
`intermedio` no "contradice a Rodinia" -- contradice **nuestra propia
hipótesis previa** (el `phase_label_hint` que habíamos puesto en el
catálogo, o la expectativa inicial del usuario).

**Nota de procedencia (DWT)**: Rodinia trae dos variantes del kernel bajo
el mismo binario -- la transformada 5/3 (`-5`, aritmética entera pura,
mismo defecto que `npb_is`/`npb_ep`: no hay FLOPs que medir) y la 9/7
(`-9`, de punto flotante real, confirmado `float` en el kernel CUDA). Solo
la 9/7 es válida para `operational_intensity`; el catálogo fija
`exec_args` con `-9` explícito para que nadie la cambie sin saber por qué.

## 3. Propuesta final del dataset

**CPU -- 9 kernels en catálogo, 7 de dataset de entrenamiento:**
- Calibración (no entran al dataset de entrenamiento): `stream_official`, `ert_probe`
- Dataset: `npb_bt`, `npb_mg`, `npb_cg`, `npb_sp`, `npb_ft`, `npb_lu`, `dgemm_n2048`

**GPU -- 12 kernels en catálogo, 8 de dataset de entrenamiento:**
- Calibración (no entran al dataset de entrenamiento): `gpu_stream_bw`, `gpu_ert_probe_fp32`, `gpu_ert_probe_fp64`, `gpu_dgemm_calibration`
- Compute-bound: `gpu_dgemm_n4096`, `rodinia_lavamd`, `rodinia_heartwall`
- Memory-bound: `rodinia_hotspot`, `rodinia_backprop`, `rodinia_dwt2d`, `rodinia_myocyte`
- Intermedio: `rodinia_lud`

**Totales:**
- **Catálogo completo**: 9 CPU + 12 GPU = **21 kernels**
- **Dataset de entrenamiento** (excluye `role: calibration`): 7 CPU + 8 GPU = **15 kernels**
- **Kernels de calibración**: 2 CPU + 4 GPU = **6 kernels**

Con la corrección del ridge, GPU queda con 3 compute-bound
(`gpu_dgemm_n4096`, `rodinia_lavamd`, `rodinia_heartwall`), 4 memory-bound
(`rodinia_hotspot`, `rodinia_backprop`, `rodinia_dwt2d`, `rodinia_myocyte`)
y **1 intermedio** (`rodinia_lud`) -- GPU ya tiene un punto borderline real,
igual que CPU (`npb_sp`).

## 4. Pendiente -- explícitamente NO resuelto en este documento

**El problema de `warmup_seconds` inventado por kernel.** Hoy cada entrada
del catálogo trae un `warmup_seconds` puesto a mano (0.3, 0.5, 1.0...) sin
un criterio común, y ya produjo un bug real: `dgemm_n2048` tiene
`warmup_seconds=0.5` sobre un runtime total de ~0.77s (excluye casi toda
la corrida, 0/773 ventanas `ok`, ARC-71) y `gpu_dgemm_n4096` tiene el mismo
problema en peor proporción (`warmup_seconds=1.0` sobre un runtime medido
de ~0.13s -- el warmup completo es más largo que la corrida entera).

No se resuelve aquí a propósito -- inventar un número nuevo por kernel
("total * 10%", "total - 2s", etc.) sin medir primero **repetiría
exactamente el mismo error** que originó el bug: un valor puesto a ojo, no
derivado de datos reales de arranque/transitorio de cada binario. Antes de
tocar `warmup_seconds` en ningún kernel (los 7 de CPU dataset + los 8 de
GPU dataset) hay que:

1. Medir el transitorio real de arranque de cada binario (tiempo hasta que
   las métricas de PMU/NVML se estabilizan), no asumirlo.
2. Definir un criterio único y documentado (ej. fracción del runtime total
   vs. tiempo absoluto vs. detección del primer estado estable) que
   aplique igual a los 21 kernels del catálogo, no una excepción por caso.
3. Re-verificar cuántas ventanas `ok` produce cada kernel con el criterio
   nuevo antes de darlo por bueno.

Esto queda como una tarea aparte, posterior a este documento.

## 5. Siguientes pasos (una vez confirmada esta propuesta)

1. ~~Agregar los 5 kernels GPU nuevos y `gpu_ert_probe_fp32`/`gpu_ert_probe_fp64`
   a `orchestrator/schemas/kernels/catalog.yaml`~~ -- hecho (ARC-75/76).
2. ~~Copiar los binarios compilados a `~/hyperion-kernels/bin/` en pacca~~ -- hecho.
3. ~~Generador reproducible de datos sintéticos
   (`scripts/pacca/generate_rodinia_synthetic_inputs.py`)~~ -- hecho.
4. Extender `campaign_pacca_gpu_ref.yaml` con `gpu_ert_probe_fp32`/
   `gpu_ert_probe_fp64` y volver a correr la campaña completa.
5. Resolver el punto de `warmup_seconds` (sección 4) como tarea propia,
   con medición real antes de cualquier cambio de valores.
6. Registrar la entrada correspondiente en
   `docs/orchestator/agents/Registro_Cambios_Fuera_Plan_Original.md`.
7. Correr la suite de tests completa y confirmar verde antes de dar por
   cerrada la integración.
