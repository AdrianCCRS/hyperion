# Fase 3 — Daemon de control en espacio de usuario

Cumple el **Objetivo 3**: un servicio que lee contadores de hardware,
ejecuta la inferencia del clasificador de Fase 2, y aplica políticas de
DVFS a través de las interfaces estándar del SO, en función de la fase de
ejecución inferida. Ver `Plan_Detallado_Realineacion_Hyperion.md` §4.

## ⚠️⚠️ Estado real de este módulo — léase antes de usar (actualizado tras conseguir CUDA real)

Este es el módulo más nuevo del proyecto (nada de esto existía en ninguna
rama antes de la reconstrucción). Una primera verificación se hizo sin
CUDA toolkit disponible; una segunda pasada, con un entorno conda que sí
tiene `nvcc`/CUDA real (`environment-hyperion-verify.yml` en la raíz),
encontró un **fallo confirmado, no solo una limitación de entorno**, en el
mecanismo de detección de fase de GPU:

| Pieza | Estado | Por qué |
|---|---|---|
| `actuation/actuator.py` (`HardwareFrequencyActuator`) | ✅ Portado y probado (5/5 tests) | Ya era código Python autocontenido en `fase-02`, sin acoplamiento al selector como se pensó inicialmente |
| `policy/derive_policy_table.py` | ✅ Construido y probado (7/7 tests, datos sintéticos) | Envuelve `fase2_clasificador/eval/protocol.py` + `common/stats.py`, ambos ya probados |
| `gpu_loop/controller.py` | ✅ Puerto fiel de `gpu_clock_controller.hpp`, verificado con los MISMOS casos que su test C++ (2/2) | Máquina de estados pura, sin NVML/CUDA |
| `gpu_loop/loop.py` | ✅ Construido y probado (7/7 tests) | El clasificador de GPU real no existe (`classify_fn` inyectable, ver limitaciones) |
| `shim/event_listener.py` (lado Python del canal de eventos) | ✅ Probado de punta a punta con un socket Unix real, incluido contra un cliente que envía los datagramas exactos del protocolo (6/6 tests) | No depende de CUDA, solo de sockets POSIX estándar |
| `run_daemon.py` | ✅ Construido y probado en `--dry-run` (1/1 test de integración) | Arranca el loop de GPU completo; el loop de CPU no está integrado |
| `cpu_loop/include/cpu_phase_controller.hpp` | ✅ Compilado y probado con CTest (1/1) | Máquina de decisión pura, sin dependencias de ONNX/collector.hpp |
| `common/telemetry` con `-DWITH_GPU=ON` real | ✅ **Recompilado y probado contra NVML/GPU reales** (13/13 CTest, incluido `collector_gpu_cadence_test`, antes siempre saltado) | Verificación nueva, no estaba hecha en la primera pasada |
| `shim/blocking_sync_shim.cpp`, forzar blocking-sync (mecanismo ARC-70 original) | ✅ **Compila, enlaza y funciona** contra `libcudart` real | Confirmado: no depende de interceptar `cudaLaunchKernel`, solo llama a `cudaSetDeviceFlags` proactivamente |
| `shim/blocking_sync_shim.cpp`, **detección de fase (extensión de esta reconstrucción)** | ❌ **CONFIRMADO ROTO, no solo sin probar** | Ver el hallazgo completo abajo — la intercepción de `cudaLaunchKernel` nunca se dispara para kernels lanzados con `<<<>>>`, en ningún modo de enlace de cudart |
| Loop de CPU real (inferencia ONNX sobre `collector.hpp`) | ❌ No construido | El SDK C++ de ONNX Runtime SÍ está disponible ahora (`onnxruntime-cpp` vía conda-forge, verificado: headers y `libonnxruntime.so` presentes) — lo que falta es un modelo real entrenado (no hay campaña real disponible) y el propio código de integración, no la herramienta |

### 🔴 Hallazgo crítico: la detección de fase por intercepción de `cudaLaunchKernel` no funciona

Verificado compilando el shim contra CUDA 13.3 real (`nvcc`, `libcudart`,
GPU NVIDIA real) y cargándolo con `LD_PRELOAD` contra un kernel CUDA de
prueba mínimo:

1. `cudaDeviceSynchronize`/`cudaStreamSynchronize` **sí se interceptan
   correctamente** — pero solo si el binario objetivo enlaza cudart de
   forma dinámica (`-cudart shared`; el default moderno de `nvcc` es
   `-cudart static`).
2. `cudaLaunchKernel` **nunca se intercepta, en ningún modo de enlace**.
   La sintaxis `kernel<<<grid,block>>>(args)` no genera una llamada
   dinámica a `cudaLaunchKernel` — `nvcc` la compila en un stub de
   lanzamiento que resuelve la llamada real en tiempo de compilación/
   enlace, nunca a través de la tabla de símbolos dinámicos que
   `LD_PRELOAD` puede alterar. Confirmado con `nm -D`: el símbolo ni
   siquiera aparece como dependencia dinámica del binario compilado.
3. Consecuencia: **cero eventos `BEGIN` llegan jamás al daemon** contra un
   kernel real, aunque el canal de transporte (socket Unix +
   `event_listener.py`) esté perfectamente probado y funcional de forma
   aislada (confirmado enviando datagramas manualmente al mismo socket).

El detalle completo, con las tres vías de arreglo evaluadas (interceptar a
nivel de driver CUDA — probablemente con el mismo problema por cómo cudart
resuelve `cuLaunchKernel` internamente vía `dlsym` sobre un handle propio
—, usar el mecanismo oficial de NVIDIA `CUDA_INJECTION64_PATH`/CUPTI, o
abandonar la intercepción y detectar actividad por sondeo de
`gpu_util_pct` desde el propio daemon), está documentado en el
encabezado de `shim/blocking_sync_shim.cpp`. **Ninguna de las tres está
implementada** — es una decisión de diseño pendiente de discutir, no una
corrección menor.

**Antes de confiar en este módulo en una campaña real, en orden:**
1. Decidir el camino de arreglo para la detección de fase (ver las 3 opciones arriba) — bloquea todo lo demás del loop de GPU en producción real.
2. Confirmar (mismo criterio que ARC-70 ya aplicó al shim original) que el mecanismo elegido no altera la salida de los kernels del catálogo corridos con y sin el shim.
3. Medir `T_transición_gpu` (§2.4.1 del plan — no existe esa medición en ningún lado del proyecto todavía) antes de fijar `--min-dwell-ns` a un valor real.
4. Construir el loop de CPU real: exportar el modelo de `fase2_clasificador/` a ONNX (el SDK C++ ya está disponible, ver la tabla de arriba) y conectar `cpu_phase_controller.hpp` con `common/telemetry/include/telemetry/collector.hpp`.

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
