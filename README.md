# Hyperion — Agente DVFS en espacio de usuario para sistemas heterogéneos CPU–GPU

Trabajo de grado: *Diseño e Implementación de un Agente en Espacio de Usuario
para la Gestión Dinámica de Frecuencia (DVFS) en Sistemas Heterogéneos
mediante Modelos Ligeros de Machine Learning*.

Un agente que observa telemetría de hardware de bajo nivel (Perf/RAPL en
CPU, NVML en GPU), infiere si una aplicación está en régimen
`compute_bound` o `memory_bound` con un clasificador ligero, y ajusta
dinámicamente la frecuencia de CPU/GPU vía las interfaces estándar del
sistema operativo — con el objetivo de reducir el Producto Energía-Retardo
(EDP) sin degradar el rendimiento, frente a los gobernadores nativos de
Linux.

## Los 4 objetivos aprobados y dónde vive cada uno

| # | Objetivo | Fase | README |
|---|---|---|---|
| 1 | Caracterizar comportamiento computacional/energético vía Perf/RAPL/NVML bajo distintos estados de frecuencia | Recolección de telemetría | [`fase1_telemetria/README.md`](fase1_telemetria/README.md) |
| 2 | Entrenar y validar un clasificador ligero (árbol/bosque/XGBoost) compute_bound/memory_bound, baja latencia | Clasificador | [`fase2_clasificador/README.md`](fase2_clasificador/README.md) |
| 3 | Daemon en espacio de usuario que aplica políticas de DVFS según la fase inferida | Daemon de control | [`fase3_daemon/README.md`](fase3_daemon/README.md) |
| 4 | Evaluar el impacto empírico vía EDP frente a gobernadores nativos | Validación experimental | [`fase4_evaluacion/README.md`](fase4_evaluacion/README.md) |

Cada fase es independiente: tiene su propio script de lanzamiento
(`run_*.py`), su propio README (manual de uso completo), y su propia
suite de tests. [`common/README.md`](common/README.md) documenta la
librería compartida (harness de telemetría en C++, control de hardware en
Python) que dos o más fases usan — no está duplicada por fase a propósito,
para que la lógica de escritura/verificación de frecuencia (la parte más
sensible del proyecto) viva en un solo lugar.

`Plan_Detallado_Realineacion_Hyperion.md` es el documento de diseño vigente
que esta reconstrucción sigue — auditado dos veces contra el código real
antes de ejecutarse (ver §0 de ese documento). `old/` conserva el árbol de
código completo previo a esta reconstrucción, intacto, como referencia
histórica — nada de lo que hay ahí es la versión vigente de nada.

## Quickstart

```bash
# 1. Clonar y entrar al repositorio
git clone <url> hyperion && cd hyperion

# 2. Entorno Python único para las 4 fases (>=3.11)
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2-bis. Alternativa con conda: entorno CON TODO lo necesario para una
#        verificación completa (incluye CUDA real -- nvcc/cudart/nvml --
#        y el SDK C++ de ONNX Runtime, no solo las dependencias Python de
#        pyproject.toml). Recomendado si se va a correr ./run_all.sh test
#        con -DWITH_GPU=ON o a tocar fase3_daemon/shim/.
#   conda env create -f environment-hyperion-verify.yml
#   conda activate hyperion-verify

# 3. Chequeo de solo lectura de permisos -- primer paso siempre,
#    antes de tocar cualquier fase
./run_all.sh check-readiness

# 4. Compilar el harness C++ + correr toda la suite de tests
./run_all.sh test

# 5. Cada fase por separado (ver su propio --help para el resto de flags)
./run_all.sh fase1 -- --help
./run_all.sh fase2 -- --help
./run_all.sh fase3 -- --help
./run_all.sh fase4 -- --help
```

`run_all.sh` es una conveniencia sobre los mismos `run_*.py` de cada
fase — nunca la única forma de invocarlas; cada README de fase documenta
también la invocación directa.

## Procedimiento de permisos — de punta a punta

Este es el primer paso real antes de cualquier campaña, no un detalle
opcional. `./run_all.sh check-readiness` (envoltura de
`common/readiness/check_node_readiness.py`) verifica todo lo de abajo
automáticamente, de solo lectura, y explica exactamente qué falta y cómo
arreglarlo si algo no está listo — lo que sigue es el porqué de cada
chequeo, para poder actuar sobre lo que el script reporte.

### 1. Perf (contadores de hardware, CPU)

```bash
# Verificar el nivel actual
cat /proc/sys/kernel/perf_event_paranoid

# Bajarlo (root, o vía sudo) -- 1 o menos permite contadores de hardware
# a procesos no privilegiados; -1 no restringe nada (no recomendado fuera
# de un nodo de cómputo dedicado)
sudo sysctl kernel.perf_event_paranoid=1

# Persistente entre reinicios
echo 'kernel.perf_event_paranoid = 1' | sudo tee /etc/sysctl.d/99-hyperion-perf.conf
```

`uncore_imc` (bytes reales de DRAM, usado para la intensidad operacional
real de §2.3 del plan) es de ámbito de socket completo — requiere además
`CAP_PERFMON` (o root) y, en un clúster compartido, **asignación exclusiva
del nodo** (`--exclusive` en Slurm): sin exclusividad, la medición de bytes
reales queda contaminada por otros procesos del nodo.

### 2. RAPL (energía de CPU)

```bash
# Debe ser legible sin privilegios especiales para el usuario del daemon
cat /sys/class/powercap/intel-rapl:0/energy_uj
```

Si da `Permission denied`: en kernels recientes, el acceso a RAPL vía
`powercap` puede requerir pertenecer a un grupo con permiso de lectura
sobre `/sys/class/powercap/intel-rapl:*/energy_uj`, o una regla `udev`
explícita. No hay una única receta universal — depende de la política del
administrador del clúster; si `check-readiness` reporta RAPL no legible,
es un bloqueo de infraestructura a escalar, no algo que el proyecto pueda
resolver por sí solo (mismo criterio que §2.0 del plan de realineación:
nunca fabricar un dato de energía no verificado).

### 3. Escritura de frecuencia de CPU

```bash
# Debe ser escribible por el usuario que corre las fases 1 y 3
echo 3600000 | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
```

Requiere permiso de escritura sobre `scaling_min_freq`/`scaling_max_freq`
(y, para Fase 4, `scaling_governor`) de cada CPU delegado — típicamente vía
una regla `udev`/`sudoers` que el administrador del clúster configura para
el usuario o grupo del proyecto, nunca escribiendo como root de forma
permanente. `common/hpc/environment.py` detecta, de solo lectura, qué es
escribible en este nodo concreto antes de que cualquier fase intente nada.

### 4. NVML / GPU

```bash
nvidia-smi -L   # confirma que el driver ve la GPU
nvidia-smi -q -d PERFORMANCE   # confirma que el control de reloj está disponible
```

Requiere el driver propietario de NVIDIA instalado (no el driver open-source
`nouveau`). El control de reloj (`nvidia-smi -lgc`) típicamente no requiere
privilegios especiales más allá de los que ya tenga el usuario para
`nvidia-smi`, pero verificarlo con `check-readiness` en el nodo real antes
de asumirlo.

### 5. Delegación de cpuset/cgroup (para el daemon de Fase 3)

El daemon (`fase3_daemon/run_daemon.py`, modo `cpuset` por defecto) opera
sobre un cpuset/cgroup ya delegado — no lo crea ni lo descubre por sí solo.
En un clúster con Slurm, esto lo da la propia asignación del job
(`--cpus-per-task`, `--exclusive` según el caso); en una máquina sin
gestor de colas, se delega manualmente con `cgcreate`/`cset` o equivalente
antes de lanzar el daemon.

## Rocky Linux 9 vs. Fedora — diferencias que importan

| | Rocky Linux 9 | Fedora |
|---|---|---|
| Compilador C++ | `dnf install gcc-c++ cmake make` — antes, habilitar el repo de desarrollo: `sudo dnf config-manager --set-enabled crb` | `dnf install gcc-c++ cmake make` — ya disponible por defecto |
| Versión de gcc/g++ | 11.x (repo base) o más nueva vía `gcc-toolset-N` (`dnf install gcc-toolset-13`, luego `scl enable gcc-toolset-13 bash`) | Suele traer una versión más reciente por defecto (verificar con `g++ --version`) |
| Driver NVIDIA + CUDA toolkit | Vía el repo oficial de NVIDIA para RHEL 9 (`dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo`) | Vía RPM Fusion (`dnf install akmod-nvidia`) o el repo oficial de NVIDIA para Fedora |
| `perf` (herramienta CLI, no solo el subsistema del kernel) | `dnf install perf` | `dnf install perf` |
| Python 3.11+ | Puede requerir `dnf install python3.11` explícito si el default del sistema es más viejo | Suele traer 3.11+ por defecto en versiones recientes |

En ambas distros, `common/telemetry/CMakeLists.txt` documenta en su propio
encabezado los dos casos puntuales que sí requieren declarar flags a mano
(un compilador no estándar cargado por un sistema de módulos HPC en vez
del paquete de la distro, o NVML sin el symlink `.so` sin versión) — leerlo
antes de asumir que hace falta algo más que `cmake -S . -B build
-DWITH_GPU=ON`.

## Estado real del proyecto — qué corre hoy de punta a punta y qué no

Fase 1 y Fase 2 están completas y verificadas de punta a punta
(compiladas/corridas, no solo escritas). Fase 4 genera el reporte de
comparación a partir de datos ya producidos, sin orquestar automáticamente
las corridas.

Fase 3 se verificó dos veces: primero sin CUDA toolkit disponible, después
con un entorno conda completo (`environment-hyperion-verify.yml`) que sí
tiene `nvcc`/CUDA/ONNX Runtime C++ reales. La segunda verificación
encontró un **hallazgo crítico confirmado, no solo una limitación de
entorno**: el mecanismo de detección de fase de GPU (intercepción de
`cudaLaunchKernel` vía `LD_PRELOAD`) **no funciona** contra kernels reales
lanzados con la sintaxis estándar `<<<>>>` — confirmado compilando el shim
contra CUDA real y cargándolo contra un kernel de prueba, con evidencia
exacta (`nm -D`, builds de depuración, un socket Unix real) en el
encabezado de `fase3_daemon/shim/blocking_sync_shim.cpp` y en
`fase3_daemon/README.md`. El mecanismo original de ARC-70 (forzar
blocking-sync) sigue funcionando correctamente — el problema es
específico de la extensión de detección de fase construida en esta
reconstrucción, con tres caminos de arreglo evaluados y ninguno
implementado todavía (decisión de diseño pendiente).

El loop de CPU con inferencia ONNX tampoco se construyó — el SDK C++ de
ONNX Runtime ya está disponible (confirmado en el entorno conda), lo que
falta es el código de integración y un modelo real entrenado (no hay
campaña real recolectada en este entorno). El detalle exacto, módulo por
módulo, está en el README de cada fase — no se oculta ninguna limitación
conocida.
