# Fase 3 — Daemon de control en espacio de usuario

Cumple el **Objetivo 3**: un servicio que lee contadores de hardware,
ejecuta la inferencia del clasificador de Fase 2, y aplica políticas de
DVFS a través de las interfaces estándar del SO, en función de la fase de
ejecución inferida. Ver `Plan_Detallado_Realineacion_Hyperion.md` §4.

## ⚠️ Estado real de este módulo — léase antes de usar

| Pieza | Estado | Por qué |
|---|---|---|
| `actuation/actuator.py` (`HardwareFrequencyActuator`) | ✅ Portado y probado (5/5 tests) | Ya era código Python autocontenido en `fase-02`, sin acoplamiento al selector como se pensó inicialmente |
| `policy/derive_policy_table.py` | ✅ Construido y probado (7/7 tests, datos sintéticos) | Envuelve `fase2_clasificador/eval/protocol.py` + `common/stats.py`, ambos ya probados |
| `gpu_loop/controller.py` | ✅ Puerto fiel de `gpu_clock_controller.hpp`, verificado con los MISMOS casos que su test C++ (2/2) | Máquina de estados pura, sin NVML/CUDA |
| `gpu_loop/activity_poller.py` (fuente de eventos de fase, Opción C) | ✅ Construido y probado (6/6 tests) | Sondeo de `gpu_util_pct` vía NVML -- ver el hallazgo que motivó esta elección más abajo |
| `gpu_loop/loop.py` (incluye `query_gpu_features`) | ✅ Construido y probado (10/10 tests) | El clasificador de GPU real no existe (`classify_fn` inyectable, ver limitaciones) |
| `run_daemon.py` | ✅ Construido y probado en `--dry-run` (2/2 tests de integración) | Arranca el loop de GPU completo; el loop de CPU no está integrado |
| `cpu_loop/include/cpu_phase_controller.hpp` | ✅ Compilado y probado con CTest (1/1) | Máquina de decisión pura, sin dependencias de ONNX/collector.hpp |
| `common/telemetry` con `-DWITH_GPU=ON` real | ✅ Recompilado y probado contra NVML/GPU reales (13/13 CTest, incluido `collector_gpu_cadence_test`) | Verificado con un entorno conda con CUDA real (`environment-hyperion-verify.yml`) |
| `common/hpc/native/blocking_sync_shim.cpp` (mecanismo ARC-70, sin cambios) | ✅ Compila, enlaza y funciona contra `libcudart` real | Sigue siendo válido para forzar blocking-sync en campañas de Fase 1 -- ver más abajo por qué Fase 3 ya no depende de él |
| Loop de CPU real (inferencia ONNX sobre `collector.hpp`) | ❌ No construido | El SDK C++ de ONNX Runtime SÍ está disponible (`onnxruntime-cpp` vía conda-forge, verificado) -- falta el código de integración y un modelo real entrenado |

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

## La tabla de política (`policy/derive_policy_table.py`)

Script offline (§3.5): agrega EDP por `(device, phase_label_train,
freq_level_id)` desde `windows.csv` de una campaña de barrido cerrada,
mediana por kernel, prueba de significancia pareada
(`common/stats.py::paired_significance_test`) antes de elegir un nivel
distinto de REF. La tabla resultante es **autocontenida**: cada entrada
`actuar` incluye `resolved_freq_khz`/`resolved_clock_mhz` (mediana del
reloj REAL observado en la campaña, no el solicitado) — el daemon nunca
necesita volver a resolver un ID de nivel contra un manifiesto de campaña.

Para GPU, sin `--t-transicion-gpu-ns` medido, la política de ambas clases
queda **siempre en `no_actuar`** — no es un default conservador arbitrario,
es honesto sobre que no existe todavía esa medición en el proyecto (§2.4.1).

```bash
python3 fase3_daemon/policy/derive_policy_table.py \
    ~/hyperion-results/campaigns/mi_campana/*/windows.csv \
    --campaign-id mi_campana --output fase3_daemon/policy_table.yaml
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

- No hay clasificador de GPU entrenado (`classify_fn` inyectable en
  `gpu_loop/loop.py`, lanza `NotImplementedError` si se usa el placeholder
  fuera de pruebas).
- La detección de fase por sondeo (Opción C) tiene latencia igual a
  `--poll-interval-s`, no es instantánea — ver "Historial de diseño"
  arriba para el porqué y las dos alternativas (a)/(b) que sí serían
  instantáneas, documentadas como trabajo futuro.
- `run_daemon.py` no implementa todavía el modo `(a)` cpuset/cgroup de
  verdad (delegación real vía Slurm) ni el modo `(b)` `--pid` — ambos son
  flags aceptados pero sin wiring de monitoreo por proceso todavía.
