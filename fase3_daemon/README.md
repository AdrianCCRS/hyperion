# Fase 3 — Daemon de control en espacio de usuario

Cumple el **Objetivo 3**: un servicio que lee contadores de hardware,
ejecuta la inferencia del clasificador de Fase 2, y aplica políticas de
DVFS a través de las interfaces estándar del SO, en función de la fase de
ejecución inferida. Ver `Plan_Detallado_Realineacion_Hyperion.md` §4.

## ⚠️ Estado real de este módulo — léase antes de usar

| Pieza | Estado | Por qué |
|---|---|---|
| `actuation/actuator.py` (`HardwareFrequencyActuator`) | ✅ Portado y probado (5/5 tests) | Ya era código Python autocontenido en `fase-02`, sin acoplamiento al selector como se pensó inicialmente |
| `policy/build_policy_table.py` | ✅ Construido, probado de punta a punta contra `build_controller_from_policy()` real | Combina las tablas ya derivadas en Fase 2 (unidad correcta), retirado `derive_policy_table.py` (EDP por ventana, unidad equivocada — ver §"La tabla de política") |
| `gpu_loop/controller.py` | ✅ Puerto fiel de `gpu_clock_controller.hpp`, verificado con los MISMOS casos que su test C++ (2/2) | Máquina de estados pura, sin NVML/CUDA |
| `gpu_loop/activity_poller.py` (fuente de eventos de fase, Opción C) | ✅ Construido y probado (6/6 tests) | Sondeo de `gpu_util_pct` vía NVML -- ver el hallazgo que motivó esta elección más abajo |
| `gpu_loop/loop.py` (incluye `query_gpu_features`) | ✅ Construido y probado (10/10 tests) | `classify_fn` inyectable -- `run_daemon.py` ya cablea `gpu_loop/classifier.py::HistoricalGpuClassifier` real |
| `gpu_loop/classifier.py` (`HistoricalGpuClassifier`) | ✅ Construido y probado (10/10 tests, incluido contra el `.joblib` real) | Carga el candidato de Fase 2 (`fase2_clasificador/models/gpu_regresion_log_historical_20260922.joblib`) y aproxima con un buffer móvil la mediana/std con que se entrenó -- ver limitaciones abajo |
| `run_daemon.py` | ✅ Construido y probado en `--dry-run` (2/2 tests de integración) | Arranca el loop de GPU completo; el loop de CPU no está integrado |
| `cpu_loop/include/cpu_phase_controller.hpp` | ✅ Compilado y probado con CTest (1/1) | Máquina de decisión pura, sin dependencias de ONNX/collector.hpp |
| `cpu_loop/include/cpu_feature_builder.hpp` | ✅ Compilado y probado (1/1) | Deltas de `CpuSample` -> 6 variables, mismas fórmulas que `postprocess.py` |
| `cpu_loop/include/onnx_cpu_classifier.hpp` + `cpu_loop_tick.hpp` | ✅ Compilados y probados EN paccaA100 vía sbatch (2/2, incluido contra `.joblib` real) | Inferencia ONNX C++ + ecuación de decisión selectiva; latencia real en paccaA100 (job 7562): p50=16.4µs, p99=19.3µs contra presupuesto ~1000µs |
| `common/telemetry` con `-DWITH_GPU=ON` real | ✅ Recompilado y probado contra NVML/GPU reales (13/13 CTest, incluido `collector_gpu_cadence_test`) | Verificado con un entorno conda con CUDA real (`environment-hyperion-verify.yml`) |
| `common/hpc/native/blocking_sync_shim.cpp` (mecanismo ARC-70, sin cambios) | ✅ Compila, enlaza y funciona contra `libcudart` real | Sigue siendo válido para forzar blocking-sync en campañas de Fase 1 -- ver más abajo por qué Fase 3 ya no depende de él |
| Loop de CPU real en vivo (`telemetry::Collector` + anillo SPSC) | ❌ No construido | El núcleo de inferencia (features+ONNX+decisión) ya está listo y probado -- falta el consumidor real sobre `SPSCRing<Sample>::try_pop()`, bloqueado por falta de permisos de PMU en esta máquina (`perf_event_paranoid=2`), requiere `pacca` |

## Historial de diseño: por qué la detección de fase de GPU es por sondeo, no por intercepción

La primera versión de esta reconstrucción intentaba detectar fronteras de
fase interceptando `cudaLaunchKernel`/`cudaDeviceSynchronize` vía un shim
`LD_PRELOAD` (`fase3_daemon/shim/`, **eliminado**). Verificado con CUDA
real (compilando el shim contra CUDA 13.3, `libcudart`, GPU NVIDIA real, y
cargándolo contra un kernel de prueba):

1. `cudaDeviceSynchronize`/`cudaStreamSynchronize` sí se interceptaban
   correctamente — pero solo si el binario objetivo enlazaba cudart de
   forma dinámica (`-cudart shared`; el default moderno de `nvcc` es
   `-cudart static`).
2. `cudaLaunchKernel` **nunca se interceptaba, en ningún modo de enlace**.
   La sintaxis `kernel<<<grid,block>>>(args)` no genera una llamada
   dinámica a `cudaLaunchKernel` — `nvcc` la compila en un stub de
   lanzamiento que resuelve la llamada real en tiempo de compilación/
   enlace, nunca a través de la tabla de símbolos dinámicos que
   `LD_PRELOAD` puede alterar. Confirmado con `nm -D`: el símbolo ni
   siquiera aparece como dependencia dinámica del binario compilado, y con
   una build de depuración que nunca imprime al interceptar un lanzamiento
   real.
3. Consecuencia: cero eventos `BEGIN` llegaban jamás al daemon contra un
   kernel real, aunque el canal de transporte (socket Unix) estuviera
   perfectamente probado de forma aislada.

**Se evaluaron 3 caminos; se eligió el (c) para esta reconstrucción,
documentando (a) y (b) como trabajo futuro, no descartado:**

- **(a) Interceptar a nivel de driver CUDA** (`cuLaunchKernel` de
  `libcuda.so`, no la API de runtime) — **trabajo futuro, no
  implementado**. Probablemente tenga el mismo problema: cudart resuelve
  `cuLaunchKernel` internamente vía `dlsym()` sobre un handle propio
  obtenido con su propio `dlopen("libcuda.so.1")`, no a través de la tabla
  de símbolos global que `LD_PRELOAD` altera — arreglarlo requeriría
  además hookear `dlsym()` mismo, más frágil que lo que ya falló.
- **(b) `CUDA_INJECTION64_PATH` + CUPTI callback API** — **trabajo futuro,
  no implementado**. Es el mecanismo oficial de NVIDIA diseñado
  exactamente para esto (funciona sin importar cómo se compiló el binario
  objetivo, es lo que usan las herramientas de profiling reales de
  NVIDIA). Más robusto que (a) y que la intercepción original, pero
  bastante más pesado: una SDK adicional (CUPTI, viene con el CUDA
  toolkit que el proyecto ya necesita, así que no es una dependencia
  nueva) con su propia curva de aprendizaje (dominios de callback,
  gestión de suscriptores, seguridad entre hilos). Vale la pena
  reconsiderarlo si en algún momento se necesita la frontera de fase
  exacta al instante del lanzamiento, en vez de con la latencia de un
  sondeo.
- **(c) Sondeo de `gpu_util_pct` desde el propio daemon, sin instrumentar
  el binario objetivo — ELEGIDA e implementada.** Ver
  `gpu_loop/activity_poller.py`. No requiere ninguna intercepción CUDA,
  funciona sin importar cómo esté compilado el binario de terceros
  (static/dynamic cudart, versión de CUDA, sintaxis de lanzamiento), y
  reutiliza código ya construido y probado (`query_gpu_features()` vía
  `nvidia-smi`, mismo patrón que `common/hpc/gpu_freqctl.py`). A cambio,
  la frontera de fase no es instantánea: se detecta con la latencia de
  `--poll-interval-s` (default 50 ms), no en el instante exacto del
  primer `cudaLaunchKernel`. Se consideró aceptable porque
  `gpu_clock_controller.hpp` ya está diseñado a granularidad de fase con
  histéresis (`min_dwell_ns`) — no para reaccionar al instante — y esa
  latencia de sondeo es pequeña frente al costo real de una transición de
  reloj de GPU.

## Arquitectura: dos loops + señal de coordinación

### Loop de GPU (`gpu_loop/`) — construido y probado

Corre por fase, no por tiempo fijo. `gpu_loop/activity_poller.py` sondea
`gpu_util_pct` (vía `gpu_loop/loop.py::query_gpu_features()`, `nvidia-smi`)
cada `poll_interval_s` y genera un `PhaseBeginEvent` en cada transición
idle→activo (`gpu_util_pct` cruza `activity_threshold_pct`, el "umbral de
ruido" de §4.1) — la transición activo→idle se reporta vía `on_end`
(logging de duración), sin generar un nuevo evento de inicio. Cada
`PhaseBeginEvent` se entrega a `gpu_loop/loop.py::run()`, que clasifica
(función inyectable — no hay clasificador de GPU real todavía) y decide
vía `gpu_loop/controller.py` (puerto fiel de `gpu_clock_controller.hpp`,
con histéresis/`min_dwell_ns`).

### Loop de CPU (`cpu_loop/`) — solo la máquina de decisión, no el binario completo

`cpu_loop/include/cpu_phase_controller.hpp`: recibe una clase ya inferida
por tick (~1ms) y decide si actuar — **solo si la clase cambió** respecto
al tick anterior (sin `min_dwell_ns`: escribir `scaling_min/max_freq` en
CPU es órdenes de magnitud más barato que bloquear el reloj de GPU). Ya
implementa la señal de coordinación (§4.1): si el loop de GPU reporta
actividad, fuerza un piso de frecuencia sin importar lo que diga el
clasificador ese ciclo.

### Actuación (`actuation/actuator.py`)

`HardwareFrequencyActuator`: aplica una acción `cpu:NIVEL` o
`gpu:HOST:NIVEL`, snapshot/restauración conjunta de CPU+GPU, manejadores
de señal ya instalados. Portado de `fase-02:orchestrator/agent_actuator.py`
— corrección frente a lo que se documentó en el plan de realineación: este
archivo NO estaba acoplado a los tipos `Protocol` del selector
(`classifier.selector.agent`) como se pensó inicialmente; solo importaba
`freqctl`/`gpu_freqctl`, ambos ya en `common/hpc/`. El porte fue un cambio
de una línea de import, no una extracción/desacoplamiento.

### Sobre `common/hpc/native/blocking_sync_shim.cpp` (mecanismo ARC-70)

Este shim (fuerza `cudaDeviceScheduleBlockingSync` para que un
`cudaDeviceSynchronize()` bloqueante no aparezca como IPC alto/compute-bound
ante el clasificador de CPU) **sigue siendo válido y compila/enlaza contra
CUDA real** — no depende de interceptar `cudaLaunchKernel`, solo llama a
`cudaSetDeviceFlags` proactivamente antes de `main()`. Fase 1 lo sigue
usando (vía `common/hpc/gpu_shim.py`) para sus campañas GPU. Fase 3 ya no
tiene su propia copia extendida de este shim (eliminada junto con la
intercepción rota) — si el daemon en algún momento necesita lanzar él
mismo un binario GPU (hoy no lo hace: opera sobre procesos ya en marcha o
dentro de un cpuset delegado), puede reutilizar `common/hpc/gpu_shim.py`
directamente para ese caso, sin necesidad de una copia propia.

## La tabla de política (`policy/build_policy_table.py`)

**La derivación NO vive en Fase 3.** Un primer diseño (`derive_policy_table.py`,
retirado 2026-09-23, ver `Plan_Fase3_Daemon.md` §0.3) calculaba el EDP por
*ventana* (~1ms) agregada desde `windows.csv` — exactamente la unidad que
el libro rechaza explícitamente: a frecuencia baja una ventana de duración
fija cubre menos trabajo que a frecuencia alta, así que `energía_ventana ×
Δt_ventana` mide potencia, no energía-retardo del trabajo. Habría producido,
en silencio, una tabla distinta de la que sustenta el libro.

La derivación real vive en Fase 2 (`fase2_clasificador/analysis/
cpu_policy_table.py` y `gpu_policy_table.py`/`gpu_policy_by_family.py`),
con la corrida completa (CPU) o el kernel/familia (GPU) como unidad, IC95
por bootstrap y validación *leave-one-familia-out* — son los scripts que
realmente produjeron los números y figuras del libro.

`policy/build_policy_table.py` **no deriva nada**: combina las dos tablas
ya comprometidas en el repo (`docs/libro/datos/cpu_calidad_30fam/politica/
policy_cpu.json` y `docs/libro/datos/gpu_calidad_20260922/politica/
policy_by_family.json` — nunca `policy_gpu.json`, que es el análisis por
kernel sin des-duplicar y elige F2 en vez del F1 defendido en el libro) y
resuelve la frecuencia física real de cada entrada `actuar` consultando el
dataset de la campaña. La tabla resultante es **autocontenida**: cada
entrada `actuar` incluye `resolved_freq_khz`/`resolved_clock_mhz` (mediana
del reloj REAL observado, no el solicitado) — el daemon nunca necesita
resolver un ID de nivel contra un manifiesto de campaña.

Hoy: `cpu-compute_bound`/`cpu-memory_bound`/`gpu-compute_bound` en
`no_actuar`; `gpu-memory_bound` en `actuar @ F1 (1260 MHz)` — la única
clase con ganancia medida (+8.9% EDP), bloqueada para el brazo `activo`
hasta que el candado de reloj de GPU se repare (Bloque D).

```bash
python3 fase3_daemon/policy/build_policy_table.py \
    --cpu-policy docs/libro/datos/cpu_calidad_30fam/politica/policy_cpu.json \
    --gpu-policy docs/libro/datos/gpu_calidad_20260922/politica/policy_by_family.json \
    --gpu-dataset tmp/historical_gpu_relaxed020_20260922.csv \
    --out fase3_daemon/policy_table.yaml
```

## Uso de `run_daemon.py`

```bash
python3 fase3_daemon/run_daemon.py \
    --policy-table fase3_daemon/policy_table.yaml \
    --min-dwell-ns 10000000000 \
    --poll-interval-s 0.05 \
    --activity-threshold-pct 5.0 \
    --dry-run
```

`--dry-run` clasifica y decide pero solo registra en log (§4.3 punto 9) —
validar así antes de tocar hardware real. Sin `--dry-run`, escribe reloj
GPU real e instala restauración por señal (`atexit`/`SIGINT`/`SIGTERM`).

## Tests

```bash
python3 -m pytest fase3_daemon/tests/ -q          # 31 tests Python
cmake -S fase3_daemon/cpu_loop -B fase3_daemon/cpu_loop/build && \
  cmake --build fase3_daemon/cpu_loop/build && \
  ctest --test-dir fase3_daemon/cpu_loop/build     # 1 test C++
```

## Limitaciones conocidas (además de la tabla de arriba)

- `HistoricalGpuClassifier` aproxima la mediana/desviación estándar con
  que se entrenó el modelo (agregadas sobre una corrida histórica
  completa) mediante un buffer móvil causal de las últimas `window_size`
  muestras NVML sondeadas (default 20, ~1s a 50ms de sondeo) -- **no es la
  misma definición**, y no hay todavía ninguna medición de cuánto cuesta
  esa diferencia en exactitud. Ver el docstring completo de
  `gpu_loop/classifier.py` para la discusión, y `Plan_Fase3_Daemon.md`
  Bloque C para cuándo se cierra (Fase 4, contra las fronteras de fase
  conocidas de las aplicaciones compuestas construidas a mano).
- La detección de fase por sondeo (Opción C) tiene latencia igual a
  `--poll-interval-s`, no es instantánea — ver "Historial de diseño"
  arriba para el porqué y las dos alternativas (a)/(b) que sí serían
  instantáneas, documentadas como trabajo futuro.
- `run_daemon.py` no implementa todavía el modo `(a)` cpuset/cgroup de
  verdad (delegación real vía Slurm) ni el modo `(b)` `--pid` — ambos son
  flags aceptados pero sin wiring de monitoreo por proceso todavía.
