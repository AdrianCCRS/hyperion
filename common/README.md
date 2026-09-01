# `common/` — librería compartida entre las 4 fases

Este directorio **no es una fase**: es la base de hardware/telemetría que usan
Fase 1 (recolección) y Fase 3 (daemon) por igual, y que Fase 2 y Fase 4
consultan indirectamente (formatos de datos, catálogo). Se separó de las
carpetas de fase para que la lógica de lectura/escritura/verificación de
frecuencia —la parte más sensible y más fácil de romper del proyecto— viva en
un solo lugar. Duplicarla por fase habría significado mantener dos copias de
la misma verificación por relectura y arriesgar que divergieran con el
tiempo.

Nada aquí decide nada por sí solo (ni clasifica fase, ni corre una campaña
completa, ni evalúa EDP) — eso vive en `fase1_telemetria/` a `fase4_evaluacion/`.
`common/` solo ofrece primitivas verificadas: leer contadores, escribir y
verificar frecuencia, y saber qué puede controlarse en el nodo actual.

## Qué vive aquí

### `common/telemetry/` — harness C++17

Instrumento de producción que mide `perf`/RAPL/NVML/`uncore_imc` y aplica
`scaling_min/max_freq` de CPU a granularidad de ventana (~1 ms). Se compila
con CMake como una librería estática (`libtelemetry.a`) más varios binarios:

| Binario | Para qué sirve |
|---|---|
| `telemetry_kernel_launcher` | El que usa `fase1_telemetria/` (vía `--exec`) para lanzar un kernel del catálogo con instrumentación completa y escribir `samples.csv`/`metadata.json`. Es el binario de producción. |
| `telemetry_kernel_workload` | Kernels sintéticos internos — **solo para desarrollo/pruebas del harness, nunca para el dataset real**. |
| `telemetry_overhead_bench` / `telemetry_jitter_bench` | Microbenchmarks para medir el overhead propio del harness — insumo de Fase 4 (overhead del agente). |

Cabeceras relevantes en `include/telemetry/`:

- `perf_reader.hpp`, `rapl_reader.hpp`, `nvml_reader.hpp`, `uncore_reader.hpp`, `cpu_freq_reader.hpp` — lectores individuales por fuente.
- `collector.hpp` — orquesta todos los lectores anteriores en un loop continuo real (`Collector::run()`, con ring buffer sin bloqueo); es la base que `fase3_daemon/cpu_loop/` extiende para el loop de CPU del daemon en vivo.
- `gpu_clock_controller.hpp` — máquina de histéresis/`min_dwell_ns` para decidir cuándo vale la pena pagar el costo de una transición de reloj de GPU. Deliberadamente **no clasifica nada** — solo administra la decisión de actuar o no una vez que alguien más (el modelo de Fase 2, o la campaña de caracterización) ya decidió la etiqueta. `fase3_daemon/gpu_loop/` es quien la invoca de verdad.

**Compilar**: ver el comentario al inicio de `CMakeLists.txt` — build genérico Rocky
Linux/Fedora, con las dos situaciones puntuales (compilador no estándar de
un módulo HPC, o NVML sin symlink `.so`) documentadas ahí mismo con la
solución exacta. Resumen rápido:

```bash
cd common/telemetry
cmake -S . -B build -DWITH_GPU=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

⚠️ **`-march=native` está activo por defecto** (`CMakeLists.txt`): el binario
resultante está optimizado para la CPU exacta de la máquina que lo compiló.
Compilar siempre en el mismo nodo (o el mismo tipo de nodo, dentro de la
misma asignación) donde correrá la campaña real — nunca en un login node
genérico que pueda tener una microarquitectura distinta a los nodos de
cómputo, o el binario puede fallar en tiempo de ejecución o, peor, medir mal
el P_pico de FLOPs sin fallar.

### `common/hpc/` — control de hardware en Python 3.11+

| Módulo | Responsabilidad |
|---|---|
| `config.py` | Carga `common/hpc_config.toml` (rutas de sysfs, config del harness, detección de entorno) — separado del manifest de campaña a propósito. |
| `environment.py` | Detecta, de solo lectura, qué puede controlarse realmente en el nodo actual (permiso de escritura de frecuencia, RAPL, hermanos SMT). |
| `freqctl.py` | Control y **restauración garantizada** de frecuencia de CPU — el módulo más sensible del repo. Verificación siempre por relectura, nunca por código de retorno. Incluye el mecanismo de espera activa de asentamiento (`wait_for_frequency_settled`) y la expansión automática a hermanos SMT. |
| `gpu_freqctl.py` | Equivalente de `freqctl.py` para GPU, vía `nvidia-smi -lgc`/`-rgc`, con la misma disciplina de relectura. |
| `catalog.py` | Valida binarios externos del catálogo (existencia, checksum) antes de ejecutarlos. |
| `manifest.py` | Parsea y valida el YAML de una campaña. |
| `preflight.py` | Verificaciones de solo lectura, bloqueantes o de advertencia, antes de campaña y por corrida. |
| `node_profile.py` | Perfil de hardware + referencias de estabilidad P95. |
| `gpu_shim.py` | Compila y localiza el shim `LD_PRELOAD` de `native/blocking_sync_shim.cpp` para forzar `cudaDeviceScheduleBlockingSync` en binarios GPU de terceros sin tocar su código fuente. |
| `gpu_inspector.py` | Inspección de estado GPU real vía `nvidia-smi` (implementación de producción del protocolo `GpuInspector`). |
| `native/blocking_sync_shim.cpp` | Fuente del shim — solo fuerza blocking-sync. Una extensión de este shim para detectar fronteras de fase (interceptando `cudaLaunchKernel`) se intentó en `fase3_daemon/` y se retiró: confirmado que esa intercepción nunca se dispara para la sintaxis `<<<>>>` de CUDA, en ningún modo de enlace de cudart. `fase3_daemon/gpu_loop/activity_poller.py` usa sondeo NVML en su lugar — ver `fase3_daemon/README.md`. |

`common/hpc_config.toml`: `binary_path` se resuelve relativo a la ubicación
del propio archivo TOML (`common/`), no al directorio de trabajo del proceso
— por eso apunta a `telemetry/build/telemetry_kernel_launcher` sin más
prefijo, y sigue siendo correcto sin importar desde dónde se invoque
`fase1_telemetria/run_campaign.py`.

### `common/readiness/` — chequeo de permisos previo a cualquier fase

Porte de lo que antes era `hardware/` en la raíz: `check_node_readiness.py`
es el script que el README global referencia como primer paso antes de
correr cualquier fase — confirma, de solo lectura, que `perf_event_paranoid`,
RAPL, NVML y demás están en un estado usable en este nodo concreto, sin
escribir nada.

## Cómo importar esto desde una fase

El proyecto no se instala como paquete (`pip install -e .`); en su lugar,
`pyproject.toml` en la raíz declara `pythonpath = ["."]` para `pytest`, y
cada `run_*.py` de cada fase inserta la raíz del repositorio en `sys.path`
al arrancar (mismo patrón que ya usan los tests aquí). Así:

```python
from common.hpc import freqctl, gpu_freqctl, catalog, manifest
```

funciona igual desde `fase1_telemetria/`, `fase3_daemon/`, o los tests de
cualquiera de las dos, sin depender de desde qué directorio se invocó el
intérprete.

## Tests

```bash
python3 -m pytest common/tests/ -q
```

248 tests, todos hermáticos (sin depender de hardware real) salvo los que
explícitamente se saltan (`SKIP_RETURN_CODE`) cuando el entorno no lo
permite (p. ej. sin GPU real). Si alguno de los tests de `gpu_freqctl`
falla con un mensaje sobre "relectura ... supera el techo fijado", **no es
un fallo del código**: significa que el test se ejecutó sin inyectar
`query_sm_clock_mhz`/`query_gpu_utilization_pct` en un host con una GPU NVIDIA
real bajo carga en ese instante — los tests de este repo ya inyectan esos dos
parámetros explícitamente para no depender del estado físico de la GPU del
host que ejecuta la suite (ver `common/tests/test_gpu_freqctl.py`).

El harness C++ tiene su propia suite vía CTest, ver más arriba.
