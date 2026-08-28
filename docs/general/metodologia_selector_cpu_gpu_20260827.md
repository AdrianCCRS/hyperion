# Metodología del selector CPU/GPU — Fase 2 (pivote 2026-08-27)

Registra a detalle el pivote metodológico del proyecto, todo lo decidido y
construido desde ese momento: arquitectura objetivo, catálogo dual,
features, configuración de campañas, validaciones, bugs encontrados y
corregidos, y el estado de las campañas de datos. Rama: `fase-02`.

---

## 1. El pivote: qué cambió y por qué

### 1.1 Entendimiento anterior (incorrecto)

Antes de esta fecha, el proyecto se estaba construyendo como un
**clasificador de fases intra-kernel**: para cada kernel HPC, dividir su
ejecución en ventanas temporales de ~1 ms, etiquetar cada ventana como
`memory`/`compute`-bound, y entrenar un clasificador que prediga esa
etiqueta a partir de contadores de hardware — con la idea de que distintas
fases dentro de un mismo kernel se beneficiarían de distinta frecuencia o
distinto dispositivo.

Esa línea de trabajo (LULESH, HPCG, GAP, CHOLMOD, C8, features
temporales) **cerró en negativo**: el óptimo de EDP casi nunca varía entre
fases de un mismo kernel (8/9 kernels con ≥99 % de tramos en un solo
nivel de frecuencia), la clase minoritaria dentro de cada kernel es
demasiado pequeña (4.0 % de media) y el clasificador de fase bajo LOKO
(F1 macro 0.393) no supera al predictor trivial (0.371). Ver
`intra-kernel-phase-hunt-negative` en memoria — **no se reabre esa
línea.**

### 1.2 Entendimiento correcto (a partir de la reunión con el profesor)

El proyecto **no es** un clasificador de fases intra-kernel. Es un
**selector**: un componente que, **antes de despachar cada operación (o
fase) de una aplicación HPC**, decide tres cosas a la vez:

1. **Dispositivo**: CPU o GPU.
2. **Frecuencia**: nivel de reloj a usar en ese dispositivo.
3. **Ejecutable**: la variante compilada/dispatchada correspondiente.

La unidad de decisión no es una ventana de 1 ms dentro de un kernel — es
**una llamada a una operación completa** (una fase de una aplicación HPC
mayor). Cada kernel del catálogo dual representa una de esas fases
posibles, no un objeto a subdividir internamente.

Consecuencias directas de este cambio de objetivo:

- **Un solo modelo unificado**, no un modelo CPU y otro GPU por
  separado. El dispositivo es una *feature* más (o el propio target),
  no un separador de datasets.
- El dataset se construye en **formato largo** (long format): cada fila
  es una configuración candidata `(operación, tamaño, device,
  freq_level)`, no una ventana temporal.
- El **target** es un clasificador de `config_id` (identificador de la
  combinación óptima) obtenido por `argmin EDP` sobre todas las
  combinaciones `{device} × {freq_level}` medidas para esa
  configuración — ver §3.

Ver memoria `pivote-selector-cpu-gpu-20260827` para el registro completo
de esta decisión.

---

## 2. Las seis operaciones del catálogo dual

Se decidió construir **6 operaciones**, cada una compilada e
instrumentada en dos variantes — CPU (OpenBLAS/FFTW/propia) y GPU
(cuBLAS/cuFFT/cuSOLVER/cuSPARSE/propia) —, eligiendo bibliotecas
reales de producción en ambos lados, no implementaciones de juguete:

| Operación | CPU | GPU | Complejidad | Rejilla |
|---|---|---|---|---|
| GEMM | OpenBLAS `dgemm` | cuBLAS `cublasDgemm` (mide H2D+compute+D2H) | O(N³) | matriz |
| FFT | FFTW (`fftw_plan_dft`) | cuFFT | O(N² log N²) | matriz |
| AXPY | OpenBLAS `cblas_daxpy` | cuBLAS `cublasDaxpy` | O(N) | vector |
| Stencil | OpenMP, Jacobi 2D 5 puntos propio | CUDA propio (`jacobi_kernel`) | O(N²) | matriz |
| Cholesky | LAPACKE `dpotrf` | cuSOLVER `cusolverDnDpotrf` | O(N³) | matriz |
| SpMV | CSR propio (`NNZ_PER_ROW=7`, banda) | cuSPARSE genérico (`cusparseSpMV`) | O(N) | vector |

Todas en **doble precisión (DP)**, todas compiladas con AVX habilitado
(OpenBLAS despacha kernels AVX-512 vía SKYLAKEX en GEMM/AXPY, confirmado
con `nm -D`/`strings`; FFTW en este nodo solo llega a AVX2, confirmado
igual — es una limitación real de la instalación, no un bug de medición,
y queda correctamente contabilizada, ver §6.2).

Fuente: `kernels/dual/` (12 archivos fuente), compilados por
`scripts/pacca/build_dual_kernels.sh` (gnu12/12.4.0,
`nvcc -arch=sm_80`). Desplegados en pacca como 12 wrappers genéricos en
`~/hyperion-kernels/bin/` que verifican su propio checksum antes de
reenviar `"$@"` (tamaño/iteraciones vienen del catálogo, no hardcodeados
en el wrapper).

### 2.1 Rejilla de tamaños

Dos familias de rejilla, generadas por `scripts/pacca/gen_dual_full_catalog.py`:

- **Rejilla "matriz"** (GEMM, FFT, Stencil, Cholesky) — O(N²)-O(N³),
  razón ~1.5×, **13 tamaños**: `64, 96, 128, 192, 256, 384, 512, 768,
  1024, 1536, 2048, 3072, 4096`. Densa cerca de 256-512 porque ahí cayó
  la frontera CPU/GPU medida en el tamizaje (`screen_dual_frontier.sh`).
- **Rejilla "vector"** (AXPY, SpMV) — O(N), razón ~√10, **8 tamaños**:
  `10 000, 31 623, 100 000, 316 228, 1 000 000, 3 162 278, 10 000 000,
  31 622 777`. Techo recortado de 1e8 a 3.16e7 tras corregir el cálculo
  de memoria real (ver §6.3): a 1e8 SpMV en host ocuparía ~10.4 GB.

Total: 4 operaciones × 13 tamaños + 2 operaciones × 8 tamaños =
**68 config_id**, cada uno con una entrada CPU y una GPU en el catálogo
→ **136 entradas de catálogo dual** (196 kernels totales en
`catalog.yaml`, incluyendo los kernels heredados de Fase 1 y las
calibraciones).

### 2.2 `config_id`: la llave de unión CPU↔GPU

Campo nuevo en `KernelEntry` (`orchestrator/catalog.py`): un string
compartido (`{operación}_N{tamaño}`, p. ej. `gemm_N512`) que identifica
la **misma configuración lógica** en sus dos variantes de catálogo
(`dual_gemm_cpu_N512`, `dual_gemm_gpu_N512`). No lo consume el
orquestador para el despacho — es exclusivamente la llave que usa el
script de construcción del dataset de nivel 2 para unir las mediciones
CPU y GPU de la misma configuración (§3).

### 2.3 Fórmula de `--iterations`

`warmup_seconds` queda **fijo en 0.05 s** en las 136 entradas — no
depende del tamaño (corrige el bug del smoke, ver §6.1). Lo que sí
depende del tamaño es `--iterations`, calculado por
`iterations_for(op, n)` en `gen_dual_full_catalog.py`:

```
k = t_ref / fn(n_ref)              # tiempo por "unidad de complejidad", medido de verdad
raw = TARGET_SECONDS / (k * fn(n)) # TARGET_SECONDS = 1.5 s
it = clamp(round(raw), MIN_ITERATIONS=5, MAX_ITERATIONS=2_000_000)
```

`t_ref` es el **mayor** entre el tiempo por iteración medido en CPU y en
GPU en el tamaño de referencia (`calibrate_iterations.sh`, medido real
en pacca, no supuesto) — así el mismo `--iterations` garantiza margen de
warmup en el lado más lento de cada config_id, y es comparable entre
ambos devices. `fn` es la función de escalado asintótico de cada
operación (`n**3`, `n*n*log2(n*n)`, `n`, etc.).

**Riesgo conocido, no bloqueante**: esta fórmula asume que el
escalamiento asintótico medido en el tamaño de referencia se mantiene en
los extremos de la rejilla (N=4096, N=3.16e7), donde pueden aparecer
efectos de caché no capturados por el modelo. No se ha validado
empíricamente en los extremos — ver §9.

---

## 3. El dataset de dos niveles

### 3.1 Nivel 1 — mediciones crudas

Una fila por **ventana temporal real** (~1 ms en CPU, ~5 ms en GPU)
dentro de una corrida real. Esto es lo que produce
`orchestrator/postprocess.py` en cada `windows.csv`. El muestreo fino
sigue siendo necesario aunque el modelo del selector nunca vea una
ventana individual — dos razones reales, no arbitrarias:

1. **Overflow del contador RAPL de 32 bits**: solo se puede integrar
   energía correctamente sumando deltas periódicos; una sola lectura al
   final de la corrida perdería los overflows intermedios.
2. **Calidad por ventana**: warmup, `running_ratio`, clasificación de
   frecuencia por ventana (`frequency_quality_status`) — sin esto no hay
   forma de excluir contaminación (ventanas de arranque, con eventos de
   PMU degradados, o mal encajadas con la frecuencia declarada) antes de
   promediar.

### 3.2 Nivel 2 — dataset de entrenamiento del selector

Una fila por `config_id`. Se construye agregando (promediando
repeticiones válidas) el nivel 1 y tomando, para cada `config_id`, el
`argmin EDP` sobre **todas** las combinaciones `{device} × {freq_level}`
medidas para ese `config_id`. El tamaño efectivo de entrenamiento es el
número de `config_id` (**68**), no el número de corridas ni el número de
ventanas — mismo principio de "N efectivo es el kernel, no la fila" ya
verificado en el trabajo previo de Fase 1 (ver
`dvfs-frequency-actuation-readiness` en memoria).

**Aún no construido** — script pendiente (§9).

### 3.3 Features

Categorías de features disponibles para el nivel 2 (agregadas desde las
columnas reales de `windows.csv`, `orchestrator/postprocess.py:20-78`):

**A. Descriptores de configuración (conocidos antes del despacho, sin medir nada):**
- Operación (identificador/one-hot), tamaño `N`, `config_id`.
- `device` (cpu/gpu) — feature cuando se entrena el clasificador con
  ambas opciones presentes como candidatas, o parte del propio target.
- `freq_level_id`, `gpu_freq_level_id`, `freq_khz_requested`.
- Descriptores estáticos de hardware (núcleos delegados, NUMA, etc. —
  fijos para este nodo, `pacca-a100`).

**B. Telemetría agregada por ventana (promediada sobre las ventanas
`quality_status="ok"` de cada corrida, luego sobre repeticiones):**
- Reloj observado: `freq_khz_observed`, `freq_khz_observed_spread`.
- PMU/IPC: `ipc`, `llc_miss_rate`, `mpki`, `ips`,
  `ipc_relative`, `mpki_relative`, `miss_rate_relative`.
- Ocupación del contador: `running_ratio` (gate de calidad, no
  necesariamente feature de entrada).
- Energía: `pkg_delta_uj`, `dram_delta_uj`, `power_w`.
- FLOPs y bytes reales: `flops_measured_window` (contador
  `FP_ARITH_INST_RETIRED`, ponderado por ancho de vector — ver §6.2),
  `bytes_moved_uncore_real` (uncore `CAS_COUNT_READ/WRITE`, no el proxy
  L2), `operational_intensity_uncore_real`.
- GPU (cuando `device=gpu`): `gpu_power_mw`, `gpu_util_pct`,
  `gpu_mem_util_pct`, `gpu_sm_clock_mhz`, `gpu_energy_delta_mj`,
  `gpu_temperature_c`.

**C. Probe / sonda de referencia**: una medición corta a nivel de
referencia (`REF`, gobernador nativo) de la misma configuración, tomada
inmediatamente antes de decidir — pensada como la señal que el selector
real tendría disponible en producción sin pagar el costo completo de
probar todas las combinaciones. Mecanismo de uso aún no implementado en
el runtime (pertenece a la Fase D, ver §10).

**D. Target derivado (no es feature de entrada)**: `config_id_óptimo =
argmin_{device,freq_level} EDP(config_id, device, freq_level)`.

El listado completo de columnas de nivel 1 (72 columnas) está en
`orchestrator/postprocess.py:20-78`; la anterior es la agrupación por
categoría relevante para el nivel 2, no una transcripción literal.

---

## 4. Configuración de las campañas

### 4.1 Por qué dos manifiestos, no uno

`campaign.py` aplica una sola lista `frequency_levels` (CPU) por
manifiesto a **todos** sus kernels — no se puede dar 8 niveles finos de
CPU a los kernels CPU-solo y 4 niveles reducidos a los kernels GPU
dentro del mismo archivo. Se separan en dos, igual que ya hacía el
proyecto con `campaign_pacca_cpu_*.yaml` vs `campaign_pacca_gpu_*.yaml`:

- **`campaign_pacca_dual_cpu_full.yaml`** — 68 config_id × 8 niveles CPU
  finos × 3 rep = **1632 corridas**.
- **`campaign_pacca_dual_gpu_full.yaml`** — 68 config_id × 4 niveles CPU
  reducidos × 8 niveles GPU finos × 3 rep = **6528 corridas**.
- **Total: 8160 corridas.**

Ambos generados por `scripts/pacca/gen_dual_campaign_manifests.py` — no
se editan a mano.

### 4.2 Niveles de frecuencia

| Grid | Niveles | Fracciones |
|---|---|---|
| `CPU_LEVELS_FULL` (campaña CPU-solo) | 8: REF, F0..F6 | 1.0 → 0.0 en 7 pasos |
| `CPU_LEVELS_REDUCED` (CPU durante despacho GPU) | 4: REF, F0, F3, F6 | 1.0, 1.0, 0.5, 0.0 |
| `GPU_LEVELS_FULL` | 8: REF, F0..F6 | 1.0 → 0.0 en 7 pasos |

El grid reducido de 4 niveles para el eje "CPU durante GPU" **no es
arbitrario**: reutiliza el mismo grid que ya validó el smoke (job 6668,
2026-08-27) con evidencia real de que el reloj de CPU sí afecta el
despacho GPU (hasta 95 % más lento en F6), con forma de meseta REF≈F0 y
penalización creciente hacia F6 — 4 puntos alcanzan para capturar esa
forma sin pagar el costo combinatorio de una rejilla 8×8.

### 4.3 Repeticiones y `baseline_repetition_indices`

`repetitions_per_combination: 3` en ambos manifiestos completos.

**Decisión 2026-08-27 (auditoría pre-lanzamiento)**:
`baseline_repetition_indices: [1]` en ambos. Sin esto, `campaign.py`
(CAM-04) empareja una corrida baseline (sin instrumentación) con cada
corrida de telemetría, en **cada** repetición — duplicando el total de
lanzamientos de proceso. El overhead de instrumentación ya está
caracterizado (media 1.95 %, estable por nivel de frecuencia, ver
`Estrategia_CPU_Fase2.md`) sobre 540 pares medidos previamente — no hace
falta re-medirlo en las 8160 combinaciones nuevas. Restringir el
baseline a la repetición 1 de cada combinación (spot-check de deriva
silenciosa) recorta el total de lanzamientos de **16 320 a 10 880
(-33 %)** sin perder cobertura de detección.

### 4.4 Otros parámetros de campaña (comunes a ambos manifiestos completos)

| Parámetro | Valor |
|---|---|
| `target_windows_per_repetition` | 5 |
| `interval_ns` (muestreo CPU) | 1 000 000 (1 ms) |
| `gpu_interval_ns` | 5 000 000 (5 ms) |
| `running_ratio_min` | 0.90 |
| `frequency_validation.tolerance_fraction` | 0.05 |
| `frequency_validation.grace_seconds` | 0.05 |
| `cores.delegated_cpus` | `[0,1,2,3,4,5]` |
| `cores.collector_cpu` / `consumer_cpu` | 6 / 7 |
| `numa_node_pin` | 0 |
| `smt_policy` | `one_thread_per_physical_core` |
| `frequency_settle` | habilitado, timeout 30 s, tolerancia 5 %, poll 0.5 s |
| `turbo.require_disabled` | `true` (todo script/manifiesto de nivel fijo se envuelve en `with_cpu_turbo_disabled.sh`) |
| `temperature` | requiere sensor de paquete, rango 0-90 °C |
| Calibración GPU | `gpu_stream_bw`, `gpu_ert_probe_fp32`, `gpu_ert_probe_fp64` |
| `hardware_datasheet.p_pico_flops_per_s` | 509 083 000 000 (medido real post-reparación, `ert_probe` a 6 hilos) |
| `hardware_datasheet.bw_pico_bytes_per_s` | 59 500 000 000 |

### 4.5 Orquestación del cruce CPU×GPU

`build_matrix()` (`orchestrator/campaign.py:338-373`, ARC-129): cuando
`catalog[kernel_ref].device == "gpu"` y el manifiesto declara
`gpu_frequency_levels`, genera el producto cartesiano completo
`(cpu_level, gpu_level)` para cada kernel GPU — no un solo id compartido
recorrido una vez. Sin cambios de código necesarios para soportar el
catálogo dual: este mecanismo ya existía en el orquestador.

**CAM-01**: el orden de ejecución es un **shuffle plano** sobre
`kernel × freq_level × repetición` — nunca por bloques de kernel o de
frecuencia, precisamente para romper el confound térmico/de deriva por
adyacencia. Consecuencia práctica: casi cada corrida cambia de nivel de
frecuencia respecto a la anterior, así que el overhead de asentamiento
de frecuencia (`frequency_settle`, hasta 30 s) ya está incluido en
cualquier tasa de corridas/segundo medida en las campañas — no es un
costo que se pueda amortizar reordenando.

---

## 5. Validaciones que debe pasar cada corrida/campaña

- **D03** (`calibration.py`): plausibilidad del `P_pico`/`BW_pico`
  observado en el nivel de referencia contra
  `manifest.hardware_datasheet` (±40 %), y de cada nivel de frecuencia
  contra el nivel de referencia (±5 % de margen de ruido). Detectó dos
  bugs reales (§6).
- **CAL-10/D04**: coeficiente de variación de las referencias de
  calibración contra un umbral de 5 % — es una **advertencia**, no un
  corte duro (`calibration.py:790`).
- **CAM-09** (`CampaignProtocolMismatchError`): impide reusar un
  `output_dir` cuyo fingerprint de protocolo no coincide con el
  manifiesto actual — nunca se bypasea, se verifica que el directorio
  viejo no tenga datos reales y se borra.
- Checksum de binario por wrapper antes de cada ejecución (evita correr
  contra un binario obsoleto tras una recompilación).
- Contabilidad de FLOPs correcta en AVX-512 vs AVX2: `postprocess.py`
  abre las 4 sub-eventos de `FP_ARITH_INST_RETIRED` (escalar, 128B,
  256B, 512B) y pondera cada uno por su ancho real
  (`_FP_ARITH_DOUBLES_PER_EVENT`) — el conteo de FLOPs es agnóstico al
  ancho de vector por diseño, no algo añadido en esta fase (infraestructura
  Fase 1, ARC-97/98/99).

---

## 6. Bugs encontrados y corregidos durante la construcción

### 6.1 `warmup_seconds` fijo excluía todas las ventanas del smoke

30/240 corridas del smoke rechazadas (`0 ventanas "ok"`) porque
`warmup_seconds: 0.5` excedía la duración total de la corrida en
configuraciones rápidas de N=512 (~0.42 s totales). Diagnosticado
inspeccionando directamente un `windows.csv` real (421/422 ventanas
`warmup_excluded`). Corregido bajando `warmup_seconds` a 0.05 en las 136
entradas duales y estableciendo el principio general: `--iterations` (no
`warmup_seconds`) es la perilla que depende del tamaño.

### 6.2 Corrupción de frecuencia propagada entre jobs

El job 6651 falló dentro de calibración (D03) y dejó
`scaling_min_freq=scaling_max_freq=800000` fijado permanentemente en
cpu0-5 y sus hermanos SMT 16-21. El bloque `finally:` de
`run_campaign()` (CAM-07) restaura correctamente la frecuencia en
cualquier salida — pero al snapshot capturado al **inicio de ese mismo
run**, que ya venía corrupto de un crash anterior, propagando la
corrupción silenciosamente de job en job. Diagnosticado por una
discrepancia de ~4× en GFLOP/s entre el nivel REF (gobernador nativo,
contaminado) y F0 (máximo fijo, correctamente forzado) dentro de la
misma corrida abortada. Reparado manualmente vía sysfs, verificado con
un `ert_probe` limpio (131.7 → 509.083 GFLOP/s). Fix permanente:
`scripts/pacca/reset_cpu_freq_range.sh`, ejecutado antes de que el
orquestador capture su propio snapshot, integrado en **todos** los
sbatch de campañas duales desde entonces. Se verificó por separado que
esto **no** era un bug de emparejamiento SMT — `freqctl.py` ya expande
correctamente a los hermanos SMT tanto al aplicar como al restaurar.

### 6.3 Fórmula de memoria estimada arbitraria

Una heurística inicial ("3.5× margen, ×3 copias") daba 58.8 GB para
SpMV en N=1e8 — 5.6× más que el cálculo real. Corregida con contabilidad
real por operación (conteo exacto de arreglos × tamaño de elemento ×
margen 1.3×) y se redujo el techo de la rejilla vector de 1e8 a 3.16e7
para mantener el peor caso dentro de presupuestos típicos de
`--mem=8-16G`.

### 6.4 FFT: verificación pasando en silencio sobre datos rotos

`fft_cpu_bench.c` encadenaba FFTs directas in-place sin restaurar el
buffer original, desbordando a NaN tras suficientes iteraciones; las
comparaciones con NaN son siempre falsas, así que `max_rel_error` se
quedaba en 0 y el binario imprimía `SUCCESSFUL` con datos completamente
rotos. Corregido restaurando desde un buffer `original` cada iteración y
cambiando a error absoluto con guardas `isfinite()` explícitas en los
tres binarios afectados (`fft_cpu_bench.c`, `fft_gpu_dispatch.cu`,
`gemm_gpu_dispatch.cu`).

### 6.5 `cholesky_gpu_dispatch.cu`: indexado row-major copiado de LAPACKE

La verificación fallaba con `info_host=0` (la factorización siempre era
correcta) porque el código de verificación leía el factor `L` con
indexado row-major (copiado del lado LAPACKE, que sí lo soporta
explícitamente) cuando cuSOLVER **siempre** escribe column-major.
Corregido el indexado de `h_work[hi*n+k]*h_work[lo*n+k]` a
`h_work[k*n+hi]*h_work[k*n+lo]`.

### 6.6 Checksums de wrappers GPU desactualizados tras recompilación

Una recompilación completa (job 6669) cambió los checksums de 4
binarios GPU ya desplegados (gemm_gpu, fft_gpu, axpy_gpu, stencil_gpu)
pese a código fuente sin cambios (metadata de build no determinista),
rompiendo silenciosamente esos wrappers. Corregido regenerando los 12
wrappers con checksums frescos y actualizando las 4 entradas
desactualizadas del catálogo. Verificado de nuevo el 2026-08-27: los 12
checksums desplegados coinciden con el catálogo.

### 6.7 `hardware_datasheet.p_pico_flops_per_s` copiado sin verificar aplicabilidad

Copiado inicialmente de `campaign_pacca_gpu_dvfs.yaml` sin verificar que
correspondía al mismo número de hilos (6). El primer "fix"
(135 781 000 000) se midió sobre un nodo aún contaminado por el bug de
§6.2, y hubo que corregirlo una segunda vez al valor real
(509 083 000 000) tras reparar el nodo.

### 6.9 ABIERTO — el EDP de GPU mediría inicialización de CUDA, no la operación (2026-08-28)

**No corregido todavía; bloquea el lanzamiento de la campaña GPU, no el
de la campaña CPU.**

`classifier/features/pair_dataset.py:86-125`
(`aggregate_cpu_runs`/`aggregate_gpu_runs`) colapsa cada corrida así:

- `elapsed_s` = `max(t_end_ns) − min(t_start_ns)` sobre **todas** las
  ventanas de la corrida → duración del **proceso completo**.
- `energy_j` = suma de `pkg_delta_uj + dram_delta_uj` sobre las ventanas
  con `energy_valid == 1` → energía del **proceso completo**.
- No filtra por `quality_status`, y `warmup_seconds=0.05` solo recorta
  50 ms del arranque.

En el eje CPU eso es una contaminación menor (0.40 s de 2.01 s, ~20 %,
además poco variable entre niveles: 0.36 s en REF → 0.44 s en F6). En el
eje GPU es **el 85 % de la medición**, y — lo decisivo — **escala con el
reloj de GPU** (~6.5 s en `gpuREF` → ~20.6 s en `gpuF6`). Como el target
del selector es `argmin EDP` sobre `{device}×{freq_level}` (§3.2), la
curva EDP-vs-frecuencia que aprendería en GPU sería principalmente *cómo
escala la inicialización de CUDA con el reloj*, no cómo escala la
operación → **etiquetas incorrectas y sesgadas sistemáticamente contra
la GPU**.

Esto contradice el intent explícito del propio binario
(`kernels/dual/gemm_gpu_dispatch.cu:105-108`: *"La creacion del contexto
CUDA tambien queda fuera -- el runtime del selector lo tendria ya
inicializado al decidir"*). El binario lo excluye correctamente de su
bucle medido; el agregador del dataset lo vuelve a meter.

**Por qué no basta con "es un costo real de irse a GPU"** (objeción
planteada 2026-08-27, correcta en el fondo): en una aplicación HPC
multi-fase el contexto CUDA se crea **una vez por proceso**, no por
operación — la fase 1 que va a GPU lo paga, y las fases 5, 12, 30 lo
encuentran vivo. El costo **por despacho** (H2D + cómputo + D2H) sí se
paga siempre, y ese ya está **dentro** del bucle medido, correctamente.
La medición actual le cobra el init íntegro a cada una de las 2176
corridas GPU, como si cada operación reinicializara CUDA. Si la
aplicación tiene N fases en GPU, a cada una le corresponde ~1/N de ese
costo.

**Fix propuesto** (no implementado): medir las dos cosas por separado en
vez de conflacionarlas — el bucle (costo por despacho, por
`config_id`×nivel) y el init (costo único, una vez por nivel). Es
estrictamente más informativo: permite decidir la política de
amortización analíticamente, que es justo el "umbral de amortización de
la sonda" ya previsto en la Fase D (§10). Implementación: la telemetría
y los kernels usan **ambos** `CLOCK_MONOTONIC` en el mismo proceso
(`telemetry/include/telemetry/metrics.hpp:31`,
`kernels/dual/*.cu:34-38`), así que basta con que los binarios impriman
`t0`/`t1` absolutos (ya los calculan) y filtrar las ventanas a ese
intervalo al construir el dataset de nivel 2.

### 6.8 Doble medición baseline+telemetry no restringida (encontrado 2026-08-27)

Ver §4.3 — no es un bug de corrección de datos, sino un desperdicio real
de presupuesto de cómputo detectado en la auditoría previa al
lanzamiento de la campaña completa, ya corregido.

---

## 7. Tamizaje (smoke) y su rol

El smoke (job 6668, 8 kernels, 4 niveles CPU × 4 niveles GPU, 3 rep) fue
el primero en confirmar con datos reales que el reloj de CPU afecta el
despacho GPU (hasta 95 % más lento en F6, forma de meseta REF≈F0). Ese
resultado es la base empírica de la rejilla reducida de 4 niveles CPU
usada en la campaña GPU completa (§4.2) — nunca se pretendió que
sustituyera a la rejilla fina.

También reveló el overhead real por corrida: 7187 s de pared para 240
corridas planeadas (~30 s/corrida) contra ~1.5 s de cómputo puro medido
en `windows.csv` — hallazgo que motivó directamente el diseño de los
pre-vuelos (§8).

---

## 8. Pre-vuelos (2026-08-27)

Dos campañas reducidas, diseñadas explícitamente para responder dos
preguntas **antes** de comprometer las 8160 corridas de la campaña
completa — no para repetir la cobertura de rejilla que ya dio el smoke:

1. **Overhead real por corrida**, con `reps=10` (más estable) sobre un
   subconjunto representativo (1 tamaño por operación, no 68).
2. **CV% de tiempo de ejecución vs. repeticiones acumuladas**, misma
   metodología que `docs/justifications/report/sections/repetitions.tex`
   (que usó el catálogo viejo, nunca corrió sobre estos 6 kernels
   nuevos) — para decidir el umbral de una política de repeticiones
   adaptativa (base 3, escalar donde el margen argmin-EDP sea estrecho o
   el CV% supere un umbral). **Aún no implementada** (§10).

| Campaña | Kernels | Niveles | Reps | Corridas | Job |
|---|---|---|---|---|---|
| `campaign_pacca_dual_preflight_cpu.yaml` | 6 (uno por operación, CPU) | 4 CPU (REF/F0/F3/F6) | 10 | 240 | 6680 |
| `campaign_pacca_dual_preflight_gpu.yaml` | 6 (uno por operación, GPU) | 3 CPU × 3 GPU (REF/F0/F6, extremos) | 10 | 540 | 6681 |

Config_id representativos usados: `gemm_N512`, `fft_N256`,
`axpy_N1000000`, `stencil_N512`, `cholesky_N512`, `spmv_N1000000` — los
mismos ya medidos en el tamizaje, no arbitrarios.

**Estado final**: job 6680 **COMPLETADO** (240/240 corridas, 22:12 min,
único aviso no bloqueante CAL-10/D04 cv_pct=29.48%). Job 6681
**CANCELADO manualmente a 1h12m** (120/540 corridas, 22% de cobertura)
para liberar el nodo compartido — los datos ya escritos no se perdieron
(`write_campaign_metadata` es incremental, CAM-02; cada corrida escribe
su propio directorio al terminar) y la campaña es reanudable (CAM-11)
si hiciera falta completarla.

### 8.1 Resultados medidos

**Desglose en tres partes** (2026-08-28, corrige una lectura previa que
tomaba el tiempo de *proceso* como si fuera tiempo de kernel). El tiempo
de proceso (`telemetry_elapsed_ns`, lo que mide el orquestador) NO es el
tiempo del bucle medido (lo que reporta el binario en
`Time in seconds`) — entre ambos está la inicialización del binario:

| Eje | Bucle medido (binario) | Init del binario | Proceso | Overhead de orquestación | Total/corrida |
|---|---|---|---|---|---|
| CPU (n=240, completo) | 1.61 s | 0.40 s | 2.01 s | 3.54 s | **5.55 s** |
| GPU (n=120, parcial) | **2.14 s** | **11.79 s** | 13.93 s | 22.54 s | **36.47 s** |

El bucle medido de GPU (2.14 s) está cerca del objetivo de 1.5 s de la
fórmula de `--iterations` (§2.3) — **la calibración de iteraciones es
correcta en ambos ejes**. Lo que dispara el costo de GPU es la
inicialización del contexto CUDA y la carga de kernels de las
bibliotecas (cuBLAS/cuFFT/cuSOLVER/cuSPARSE), que además **escala con el
reloj de GPU**: ~6.5 s en `gpuREF`/`gpuF0` frente a ~20.6 s en `gpuF6`.
Solo **2.14 s de los 36.47 s (5.9 %)** de cada corrida GPU es medición
útil. Ver §6.9 para la consecuencia sobre la validez del dataset.

El overhead **de orquestación** (fuera del proceso) en el eje GPU es
~6.4× el del eje CPU. No es una atribución especulativa — son cuatro
mecanismos reales, verificados en el código, que se **apilan** en (casi)
cada corrida GPU por el shuffle plano de CAM-01 (§4.5), mientras que una
corrida CPU-solo solo paga el primero (el cuarto, la inicialización de
CUDA, cae dentro del proceso y es el "Init del binario" de la tabla de
arriba):

1. **Asentamiento de CPU** (`freqctl.settle_if_configured`,
   `orchestrator/freqctl.py:469-491`): cuando el nivel objetivo no es
   `REF`, se lanza una carga sintética real en los CPUs delegados
   (`taskset -c <cpu> yes`, `_start_warmup_load`) y se sondea
   `scaling_cur_freq` hasta que entra en tolerancia (5 %) o hasta 30 s —
   un CPU inactivo nunca confirma un candado de frecuencia alta sin
   carga real (ARC-164/165). Esto se paga en **ambos** ejes.
2. **Actuación de GPU** (`gpu_freqctl.apply_gpu_frequency`,
   `orchestrator/gpu_freqctl.py:140-247`): cada cambio de nivel ejecuta
   **dos** llamadas `sudo nvidia-smi -i <idx> -rgc` (reset) y luego
   `-lgc <t>,<t>` (lock) — dos subprocesos con `sudo` y round-trip real
   al driver, no una escritura de sysfs en memoria como en CPU.
3. **Sin asentamiento propio de GPU pero con verificación por
   consulta**: a diferencia de CPU, `apply_gpu_frequency` no tiene un
   bucle de sondeo separado — pero cada corrida además consulta
   `--query-gpu=clocks.sm` (10 s de timeout propio) para confirmar el
   reloj aplicado antes de medir, otro subproceso más.
4. **Inicialización de contexto/handles CUDA por proceso**: cada
   binario dual-GPU crea su propio contexto CUDA y sus handles de
   biblioteca (cuBLAS/cuFFT/cuSOLVER/cuSPARSE) desde cero en cada
   lanzamiento — un costo fijo conocido de cientos de ms a algunos
   segundos para kernels cortos, inexistente en el lado CPU.

Con el shuffle plano de CAM-01 aplicándose también al eje GPU (8
niveles GPU × 4 niveles CPU, ninguno de los dos fijo entre corridas
consecutivas), (1) y (2)/(3) se pagan casi en cada corrida — de ahí que
el overhead medido (22.54 s) sea un orden de magnitud mayor que el de
CPU-solo (3.54 s), no solo un múltiplo pequeño.

CV%(n=3) replicando la metodología de `repetitions.tex`:

- **CPU**: media 0.84 %, máximo 4.86 % (FFT, nivel F0). FFT es el único
  kernel que repite el patrón ya visto en `repetitions.tex`
  (dwt2d/lud): en F6 el CV% **sube** de 1.94 % (n=3) a 7.66 % (n=10) —
  no converge, hay dispersión real que 3 repeticiones no capturan. Los
  otros 5 kernels (GEMM, AXPY, Cholesky, Stencil, SpMV) están cómodos
  bajo 1.5 % incluso en n=3.
- **GPU** (parcial, 20/54 combinaciones con ≥3 reps): media 1.27-1.39 %,
  máximo 5.27 % (un solo caso, n=3, posible ruido de muestra chica). Sin
  patrón sistemático detectado todavía, pero la cobertura es baja.

**Conclusión sobre repeticiones**: 3 reps son defendibles para la
mayoría de los kernels nuevos, **excepto FFT**, que necesita más
repeticiones o un chequeo adaptativo — confirma directamente lo que ya
advertía `repetitions.tex` sobre kernels con dispersión oculta en
muestras pequeñas.

### 8.2 Hallazgo que bloqueaba el lanzamiento directo: costo de la campaña GPU completa

Proyectando el overhead medido (con `baseline_repetition_indices: [1]`
ya aplicado, §4.3) sobre el diseño de las campañas completas:

- **CPU full** (1632 corridas): ≈ **1.9–2.5 h** — sin problema, cabe en
  una sola sesión.
- **GPU full** (6528 corridas): ≈ **48–55 h ≈ 2–2.3 días** — viola
  directamente el límite de "no perder más de 1 día de cómputo" y no es
  compatible con compartir el nodo con otros usuarios del cluster.

### 8.3 Decisión (2026-08-27): sesiones, no reducción de alcance

Se evaluaron cuatro opciones de recorte, no excluyentes entre sí:

| Opción | Qué sacrifica | Por qué NO se eligió (sola) |
|---|---|---|
| Reducir niveles GPU (8→5) | Resolución de la curva EDP(freq) en GPU | Es exactamente la rejilla fina que se decidió construir tras el pivote (§1.2) — recortarla por presión de tiempo repite el mismo error que ya se corrigió una vez (ver el reclamo del usuario "Por qué hiciste solo 4 niveles CPU?", que forzó reconstruir con rejilla fina) |
| Reducir niveles CPU-durante-GPU (4→3) | Perder el punto F3 de la meseta ya medida en el smoke | El smoke (6668) ya demostró que la forma REF≈F0→creciente hacia F6 necesita al menos un punto intermedio para no asumir linealidad sin evidencia |
| Reducir la rejilla de tamaños GPU | Cobertura de `config_id`, y con ella comparabilidad directa CPU↔GPU (mismo `config_id` en ambos catálogos, §2.2) | Rompe la unión por `config_id` que sostiene todo el dataset de nivel 2 (§3.2) si CPU y GPU dejan de compartir la misma rejilla |
| **Partir en sesiones** | Nada del diseño — solo el tiempo de pared se reparte en bloques | Es la única opción que no cambia qué se mide, solo cuándo — y el orquestador ya tiene reanudación real (CAM-11), así que no cuesta trabajo de ingeniería nuevo |

**Se eligió partir en sesiones**: la rejilla fina de 8 niveles GPU y 4
niveles CPU-durante-GPU se mantiene completa — es la evidencia que ya
costó construir y validar (smoke 6668) — y en su lugar se lanza
`campaign_pacca_dual_gpu_full.yaml` en bloques de ~5.5 h reanudables, no
en un solo job de 2+ días. Esto resuelve la restricción real del
usuario (nodo compartido, no se puede ocupar por días) sin sacrificar
ninguna decisión metodológica ya tomada con evidencia.

Mecanismo: `orchestrator/schemas/scripts/launchers/run_campaign_pacca_dual_gpu_full.sbatch`,
`--time=05:30:00`, `--campaign-timeout-seconds=19800` (5h30m — deja
margen sobre el límite de Slurm para que el `finally:` de
`run_campaign()` restaure la frecuencia con CAM-06 antes de un kill
forzado, nunca dejar que Slurm mate el proceso). Cada sesión vuelve a
correr `reset_cpu_freq_range.sh` antes del snapshot del orquestador
(mismo motivo que §6.2) y vuelve a medir calibración (minutos, costo
pequeño frente a 5.5 h). Reanudar es tan simple como volver a lanzar el
mismo script — CAM-11 detecta las corridas ya aceptadas en el
`output_dir` y las salta.

`campaign_pacca_dual_cpu_full.yaml` sí cabe en una sola sesión:
`run_campaign_pacca_dual_cpu_full.sbatch`, `--time=04:00:00`,
`--campaign-timeout-seconds=13200`.

---

## 9. Riesgos identificados, no bloqueantes hoy

- La fórmula de `--iterations` (§2.3) nunca se validó empíricamente en
  los extremos de la rejilla (N=4096 matriz, N=3.16e7 vector) — solo en
  los 6 tamaños representativos del pre-vuelo. Recomendado: una corrida
  suelta por operación en su extremo antes del lanzamiento completo.
- El CV% de dispersión de las referencias de calibración se observó en
  29.48 % en un punto del pre-vuelo CPU (CAL-10/D04, advertencia, no
  bloqueo) — vigilar si persiste, podría indicar ruido térmico o de
  contención en el nodo.
- FFT muestra CV% no convergente en F6 (1.94 % en n=3 → 7.66 % en
  n=10, §8.1) — 3 repeticiones probablemente subestiman su dispersión
  real. Candidato directo para la política de repeticiones adaptativa
  (§10), o para fijarle un piso de reps más alto que el resto de
  operaciones mientras esa política no exista.
- La cobertura GPU del pre-vuelo quedó parcial (120/540, 20/54
  combinaciones con ≥3 reps) por la cancelación para liberar el nodo —
  el CV% del eje GPU no está tan verificado como el de CPU.

---

## 10. Pendiente

- ~~Analizar overhead real y CV%/convergencia de repeticiones con los
  datos de los pre-vuelos~~ — hecho, ver §8.
- Diseñar e implementar la política de repeticiones adaptativa (base 3,
  escalar por margen EDP estrecho o CV% alto) — FFT ya es un candidato
  concreto con evidencia real (§8.1, §9).
- Lanzar `campaign_pacca_dual_cpu_full.yaml` (una sesión) y
  `campaign_pacca_dual_gpu_full.yaml` (varias sesiones de ~5.5 h,
  reanudables, ver §8.3) — 8160 corridas, 10 880 lanzamientos de proceso
  tras §4.3.
- Construir el script de dataset de nivel 2 (agrupar por `config_id`,
  `argmin EDP` sobre `{device}×{freq_level}`).
- Entrenar el clasificador de `config_id`, validar con leave-one-operación-out.
- Construir la aplicación sintética multi-fase HPC y el runtime
  (lógica sonda-y-decide) — Objetivo 3 / Fase D, no iniciado.
- Experimento de cuatro barras (todo-CPU, todo-GPU, selector, oráculo) +
  medición del umbral de amortización de la sonda — no iniciado.
- Actualizar `Estado_Paralelo_CPU_GPU.md`, `Estrategia_CPU_Fase2.md` /
  `Estrategia_GPU_Fase2.md` y `Estado_Cola_Slurm.md` con este pivote.
- `main.tex` sigue sin redacción de la Fase 2 / pivote.

---

## 11. Referencias cruzadas

- Memoria: `pivote-selector-cpu-gpu-20260827`,
  `intra-kernel-phase-hunt-negative`,
  `feedback-always-disable-turbo-for-fixed-frequency-runs`,
  `feedback-verify-remote-build-not-just-source-sync`,
  `feedback-preflight-before-long-campaigns`.
- Código: `orchestrator/catalog.py`, `orchestrator/campaign.py`,
  `orchestrator/manifest.py`, `orchestrator/calibration.py`,
  `orchestrator/postprocess.py`, `orchestrator/runner.py`.
- Generadores: `scripts/pacca/gen_dual_full_catalog.py`,
  `scripts/pacca/gen_dual_campaign_manifests.py`,
  `scripts/pacca/build_dual_kernels.sh`,
  `scripts/pacca/reset_cpu_freq_range.sh`.
- Kernels: `kernels/dual/*.c`, `kernels/dual/*.cu`.
- Manifiestos: `orchestrator/schemas/campaigns/campaign_pacca_dual_*.yaml`.
- Launchers de campaña completa:
  `orchestrator/schemas/scripts/launchers/run_campaign_pacca_dual_cpu_full.sbatch`,
  `orchestrator/schemas/scripts/launchers/run_campaign_pacca_dual_gpu_full.sbatch`
  (este último pensado para relanzarse varias veces, ver §8.3).
- Reporte previo sobre repeticiones:
  `docs/justifications/report/sections/repetitions.tex`.
