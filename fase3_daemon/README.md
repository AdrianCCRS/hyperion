# Fase 3 — Daemon de control en espacio de usuario

Cumple el **Objetivo 3**: un servicio que lee contadores de hardware,
ejecuta la inferencia del clasificador de Fase 2, y aplica políticas de
DVFS a través de las interfaces estándar del SO, en función de la fase de
ejecución inferida. Ver `Plan_Detallado_Realineacion_Hyperion.md` §4.

## ⚠️ Estado real de este módulo — léase antes de usar

Este es el módulo más nuevo del proyecto (nada de esto existía en ninguna
rama antes de la reconstrucción) y el único donde parte del código **no se
pudo compilar ni probar** en el entorno donde se hizo esta reconstrucción,
porque faltan herramientas que sí deberían existir en el nodo HPC real:

| Pieza | Estado | Por qué |
|---|---|---|
| `actuation/actuator.py` (`HardwareFrequencyActuator`) | ✅ Portado y probado (5/5 tests) | Ya era código Python autocontenido en `fase-02`, sin acoplamiento al selector como se pensó inicialmente (corrección de la propia auditoría de esta reconstrucción) |
| `policy/derive_policy_table.py` | ✅ Construido y probado (7/7 tests, datos sintéticos) | Envuelve `fase2_clasificador/eval/protocol.py` + `common/stats.py`, ambos ya probados |
| `gpu_loop/controller.py` | ✅ Puerto fiel de `gpu_clock_controller.hpp`, verificado con los MISMOS casos que su test C++ (2/2) | Máquina de estados pura, sin NVML/CUDA |
| `gpu_loop/loop.py` | ✅ Construido y probado (7/7 tests) | El clasificador de GPU real no existe (`classify_fn` inyectable, ver limitaciones) |
| `shim/event_listener.py` (lado Python del canal de eventos) | ✅ Probado de punta a punta con un socket Unix real (6/6 tests) | No depende de CUDA, solo de sockets POSIX estándar |
| `run_daemon.py` | ✅ Construido y probado en `--dry-run` (1/1 test de integración) | Arranca el loop de GPU completo; el loop de CPU no está integrado (ver abajo) |
| `cpu_loop/include/cpu_phase_controller.hpp` | ✅ Compilado y probado con CTest (1/1) | Máquina de decisión pura, sin dependencias de ONNX/collector.hpp |
| `shim/blocking_sync_shim.cpp` (extensión CUDA) | ⚠️ **NO compilado, solo verificado con `-fsyntax-only` contra un stub de `cuda_runtime.h`** | El entorno de esta reconstrucción no tiene el CUDA toolkit (`nvcc`) instalado |
| Loop de CPU real (inferencia ONNX sobre `collector.hpp`) | ❌ No construido | Requiere el SDK C++ de ONNX Runtime (no disponible aquí, solo el paquete Python) y un modelo real ya entrenado (no hay campaña real disponible en este entorno) |

**Antes de confiar en este módulo en una campaña real, en orden:**
1. Compilar `fase3_daemon/shim/blocking_sync_shim.cpp` en un nodo con CUDA toolkit real (p. ej. paccaA100) y confirmar que enlaza contra `libcudart`.
2. Correr la prueba dirigida que pide el plan (§4.1): lanzar un kernel GPU real del catálogo con el shim cargado vía `LD_PRELOAD` y `HYPERION_GPU_PHASE_SOCKET` apuntando a un socket real, confirmar exactamente un evento `BEGIN` y un `END` por periodo de actividad GPU (no por cada `cudaLaunchKernel` individual — ver la nota de granularidad de fase en el propio archivo).
3. Confirmar (mismo criterio que ARC-70 ya aplicó al shim original) que no altera la salida de los kernels del catálogo corridos con y sin el shim.
4. Medir `T_transición_gpu` (§2.4.1 del plan — no existe esa medición en ningún lado del proyecto todavía) antes de fijar `--min-dwell-ns` a un valor real.
5. Construir el loop de CPU real: exportar el modelo de `fase2_clasificador/` a ONNX, instalar el SDK C++ de ONNX Runtime, y conectar `cpu_phase_controller.hpp` con `common/telemetry/include/telemetry/collector.hpp`.

## Arquitectura: dos loops + señal de coordinación

### Loop de GPU (`gpu_loop/` + `shim/`) — construido y probado

Corre por fase, no por tiempo fijo. `shim/blocking_sync_shim.cpp` (cargado
vía `LD_PRELOAD` sobre el binario GPU de terceros, sin tocar su código
fuente) intercepta `cudaLaunchKernel`/`cudaDeviceSynchronize`/
`cudaStreamSynchronize` y emite un evento `BEGIN` al primer lanzamiento
tras una sincronización, y `END` en la sincronización que le sigue — una
fase es un **periodo de actividad GPU**, no una llamada individual (un
kernel real puede hacer miles de lanzamientos entre sincronizaciones, p.
ej. `rodinia_gaussian` con ~8190 por corrida).

`shim/event_listener.py` escucha esos eventos por un socket Unix
(`HYPERION_GPU_PHASE_SOCKET`), consulta NVML en vivo (vía `nvidia-smi`,
mismo patrón que `common/hpc/gpu_freqctl.py`) en cada `BEGIN`, y entrega un
`PhaseBeginEvent` a `gpu_loop/loop.py::run()`, que clasifica (función
inyectable — no hay clasificador de GPU real todavía) y decide vía
`gpu_loop/controller.py` (puerto fiel de `gpu_clock_controller.hpp`, con
histéresis/`min_dwell_ns`).

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
    --gpu-phase-socket /tmp/hyperion_gpu_phase.sock \
    --min-dwell-ns 10000000000 \
    --dry-run
```

`--dry-run` clasifica y decide pero solo registra en log (§4.3 punto 9) —
validar así antes de tocar hardware real. Sin `--dry-run`, escribe reloj
GPU real e instala restauración por señal (`atexit`/`SIGINT`/`SIGTERM`).

## Tests

```bash
python3 -m pytest fase3_daemon/tests/ -q          # 29 tests Python
cmake -S fase3_daemon/cpu_loop -B fase3_daemon/cpu_loop/build && \
  cmake --build fase3_daemon/cpu_loop/build && \
  ctest --test-dir fase3_daemon/cpu_loop/build     # 1 test C++
```

## Limitaciones conocidas (además de la tabla de arriba)

- No hay clasificador de GPU entrenado (`classify_fn` inyectable en
  `gpu_loop/loop.py`, lanza `NotImplementedError` si se usa el placeholder
  fuera de pruebas).
- El shim asume que el catálogo no usa streams CUDA concurrentes propios
  más allá del default (documentado explícitamente en el propio archivo) —
  si algún kernel del catálogo llega a necesitarlos, la detección de fin de
  fase por `cudaStreamSynchronize` debe revisarse.
- `run_daemon.py` no implementa todavía el modo `(a)` cpuset/cgroup de
  verdad (delegación real vía Slurm) ni el modo `(b)` `--pid` — ambos son
  flags aceptados pero sin wiring de monitoreo por proceso todavía.
