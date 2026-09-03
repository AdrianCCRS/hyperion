# Manual de ejecución — Hyperion (para estudiantes)

Este manual explica, paso a paso y con el porqué de cada cosa, cómo correr
el proyecto completo y qué falta para terminar el trabajo de grado. No es
un README técnico de referencia (esos ya existen, uno por carpeta) — es la
guía que deberías poder seguir sin haber visto el código antes, entendiendo
qué está pasando en cada paso, no solo copiando comandos.

**Cómo usar este documento**: léelo de punta a punta la primera vez, en
orden — cada sección asume que ya hiciste la anterior. Después, úsalo como
referencia: cada sección tiene su propio comando y su propia forma de
verificar que funcionó. Cuando necesites el detalle técnico completo de
algo (todas las columnas de un archivo, todos los flags de un script), este
manual te dice a qué README de fase ir — no repite esa información.

---

## 0. Qué es este proyecto, en una página

### La idea central

Un procesador (CPU o GPU) puede correr a distintas frecuencias de reloj.
Más frecuencia normalmente significa más rendimiento, pero también más
consumo de energía. El sistema operativo ya intenta ajustar la frecuencia
automáticamente (los "gobernadores" como `ondemand`/`schedutil`), pero lo
hace mirando señales genéricas (qué tan ocupado está el CPU), sin entender
*por qué* está ocupado.

Hay una distinción importante que el sistema operativo no ve: una
aplicación puede estar **limitada por cómputo** (`compute_bound` — el
cuello de botella son las operaciones aritméticas) o **limitada por
memoria** (`memory_bound` — el cuello de botella es esperar datos de la
RAM). Cuando una aplicación está limitada por memoria, subir la frecuencia
del procesador casi no mejora el tiempo de ejecución (los núcleos igual
están esperando datos) pero sí sube el consumo de energía. Cuando está
limitada por cómputo, sí vale la pena la frecuencia alta.

Este proyecto construye un **agente en espacio de usuario** (un programa
normal, no algo dentro del kernel de Linux) que:
1. Observa contadores de hardware de bajo nivel (cuántas instrucciones se
   ejecutaron, cuántos accesos a caché fallaron, cuánta energía se gastó,
   qué tan ocupada está la GPU).
2. Con esos datos, **infiere** con un modelo de Machine Learning ligero
   (un árbol de decisión, no una red neuronal) si la aplicación está en
   régimen `compute_bound` o `memory_bound` en ese instante.
3. Según esa clasificación, **ajusta la frecuencia** de CPU/GPU usando las
   mismas interfaces que ya usa Linux (`scaling_min_freq`/`scaling_max_freq`
   para CPU, `nvidia-smi -lgc` para GPU) — no inventa un mecanismo nuevo de
   control de hardware, solo decide mejor cuándo usarlo.
4. Al final, se **mide si esto realmente ayudó**: se compara el consumo de
   energía × tiempo (el "Producto Energía-Retardo", EDP) del agente contra
   los gobernadores nativos de Linux, con una prueba estadística — no solo
   "se ve mejor", sino "la diferencia es significativa o no".

### Los 4 objetivos del trabajo de grado, y dónde vive cada uno

| # | Objetivo (resumen) | Carpeta | Qué produce |
|---|---|---|---|
| 1 | Caracterizar el comportamiento de cargas reales, midiendo con Perf/RAPL (CPU) y NVML (GPU) bajo distintas frecuencias | `fase1_telemetria/` | Un dataset (`windows.csv`) con miles de filas: una fila = una ventana de tiempo con sus contadores y su etiqueta `compute_bound`/`memory_bound` |
| 2 | Entrenar un clasificador ligero que infiera esa etiqueta a partir de los contadores baratos, rápido | `fase2_clasificador/` | Un modelo entrenado (`.joblib`) + una ficha técnica (`.metadata.json`) con su exactitud y su velocidad |
| 3 | Un daemon que, corriendo en vivo, aplique la clasificación para ajustar la frecuencia real | `fase3_daemon/` | Un proceso que corre en el nodo, decide y actúa — **parcialmente construido, ver §4 más abajo** |
| 4 | Medir si esto de verdad mejora el EDP frente a los gobernadores nativos | `fase4_evaluacion/` | Un reporte comparando escenarios, con significancia estadística |

`common/` no es una fase — es la librería compartida (el instrumento de
medición en C++ y el control de frecuencia en Python) que usan las fases
1 y 3 por igual. No la vas a modificar casi nunca; solo la vas a *usar*
indirectamente al correr las fases.

`old/` es el código tal como estaba antes de esta reorganización en 4
fases — queda ahí como referencia histórica. **Nunca trabajes dentro de
`old/`**, todo lo vigente está en las carpetas de arriba.

`Plan_Detallado_Realineacion_Hyperion.md` es el documento de diseño
completo (mucho más detallado que este manual, con la justificación
técnica de cada decisión). Este manual te dice *cómo ejecutar*; ese
documento te dice *por qué se diseñó así*, si alguna vez lo necesitas para
el capítulo de metodología de la tesis.

---

## 1. Preparar el entorno (una sola vez por máquina)

### 1.1 Clonar el repositorio

```bash
git clone <url-del-repo> hyperion
cd hyperion
```

### 1.2 Instalar el entorno con conda (recomendado)

El proyecto necesita Python 3.11+, un compilador C++17, y — si vas a tocar
la parte de GPU — el CUDA toolkit y el SDK de ONNX Runtime. La forma más
simple de tener todo eso junto es el entorno conda ya preparado:

```bash
conda env create -f environment-hyperion-verify.yml
conda activate hyperion-verify
```

Esto instala: Python 3.11, todas las librerías de análisis de datos
(pandas, numpy, scikit-learn, xgboost, scipy), `cmake`, y — esto es lo que
lo hace distinto de una instalación mínima — `nvcc`/CUDA real y el SDK de
ONNX Runtime en C++, que vas a necesitar para terminar la Fase 3 (§4.4).

**Alternativa sin conda** (si tu máquina ya tiene Python 3.11+ y no vas a
tocar CUDA/ONNX todavía):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

⚠️ Nota importante si usas esta alternativa: el `[dev]` al final **no es
opcional** aunque parezca cosmético — sin él, `pytest` no se instala y el
paso 1.4 de abajo falla con "No module named pytest".

### 1.3 Verificar permisos del nodo — el primer paso técnico real

Antes de tocar cualquier fase, corre esto:

```bash
./run_all.sh check-readiness
```

Esto **no escribe nada** en el sistema — solo lee y te dice qué está listo
y qué no. Va a revisar cosas como:

- **`perf_event_paranoid`**: un valor del kernel que controla si un
  usuario normal puede leer contadores de hardware (instrucciones,
  cache-misses, etc.) sin ser root. Si el chequeo falla, alguien con
  permisos de administrador en el clúster necesita correr:
  ```bash
  sudo sysctl kernel.perf_event_paranoid=1
  ```
- **RAPL** (energía de CPU): debe poder leerse
  `/sys/class/powercap/intel-rapl:0/energy_uj` sin privilegios especiales.
- **Escritura de frecuencia de CPU**: debe poder escribirse
  `scaling_min_freq`/`scaling_max_freq`. Esto casi siempre requiere que el
  administrador del clúster te dé un permiso explícito (una regla `sudo`
  o `udev`) — no es algo que puedas resolver tú solo desde tu cuenta.
- **NVML/GPU**: que `nvidia-smi` funcione y vea la GPU.

Si algo sale como "no disponible", el propio comando te dice qué falta y
cómo pedirlo — no sigas a la Fase 1 hasta que al menos la lectura de
contadores (perf/RAPL) esté en verde. La escritura de frecuencia puede
tardar en llegar (depende del administrador del clúster) — puedes avanzar
con partes de la Fase 1 que no requieren escribir frecuencia mientras
tanto, pero la campaña completa sí la necesita.

### 1.4 Verificar que todo compila y los tests pasan

```bash
./run_all.sh test
```

Esto hace 3 cosas en orden: (1) corre toda la suite de tests de Python de
las 4 fases (deberían ver algo como "623 passed"), (2) compila el
instrumento de telemetría en C++ (`common/telemetry/`) y corre sus tests,
(3) compila y prueba la máquina de decisión del loop de CPU del daemon.
Si algo de esto falla, **no sigas** — algo está mal en el entorno, no en
tu campaña. Los errores de compilación C++ suelen ser de compilador
faltante (`cmake`/`g++`) — revisa la sección de Rocky Linux/Fedora del
`README.md` raíz.

Con esto, tu entorno está listo. Lo que sigue son las 4 fases, en orden —
cada una depende de que la anterior haya producido su resultado.

---

## 2. Fase 1 — Recolección de telemetría (Objetivo 1)

### 2.1 Qué hace, conceptualmente

Fase 1 corre kernels reales (programas de benchmark conocidos — NAS
Parallel Benchmarks, Rodinia, RAJAPerf, y otros) bajo distintos niveles de
frecuencia, y mientras corren, mide con altísima resolución temporal
(ventanas de ~1ms en CPU) contadores de hardware: instrucciones
ejecutadas, ciclos de reloj, fallos de caché, energía consumida (RAPL), y
en GPU: utilización, potencia, energía (NVML).

Con esos contadores, calcula una **intensidad operacional** (FLOPs
ejecutados por byte movido desde memoria) y la compara contra un punto de
referencia calibrado para ese hardware específico (el "punto de ridge" del
modelo Roofline) — si la intensidad medida está por debajo del ridge, la
ventana se etiqueta `memory_bound`; si está por encima, `compute_bound`.
**La etiqueta nunca se asume por el nombre del kernel** — un kernel
"famoso" por ser compute-bound en la literatura puede medir memory-bound
en tu hardware real, y el sistema lo detecta empíricamente.

El resultado es un archivo `windows.csv`: cada fila es una ventana de
tiempo con sus contadores y su etiqueta. Este archivo es el dataset que
entrena el clasificador de Fase 2.

### 2.2 El catálogo de kernels

`fase1_telemetria/catalog/catalog.yaml` lista 232 kernels distintos
(agrupados en familias algorítmicas: multiplicación de matrices, FFT,
stencils, etc., cada uno con varios tamaños de problema) con su ruta de
binario y su checksum SHA-256 — el checksum se verifica antes de cada
corrida, para que una recompilación accidental de un binario no invalide
datos anteriores sin que nadie se dé cuenta.

⚠️ **Los binarios de terceros (NPB, Rodinia, RAJAPerf, etc.) no vienen en
este repositorio** — hay que compilarlos aparte y ponerlos en
`~/hyperion-kernels/bin/` (ver §2.3). El catálogo solo declara *dónde*
deberían estar y *qué checksum* deben tener.

### 2.3 Convención de directorios — importante, rompe cosas si no se sigue

```
~/hyperion-kernels/     # EXTERNO al repositorio, no lo crea git clone
  ├── src/               # código fuente de los benchmarks (descargado aparte)
  ├── bin/                # binarios ya compilados -- lo que catalog.yaml referencia
  └── checksums.sha256
~/hyperion-results/
  └── campaigns/          # aquí caen los resultados de cada campaña que corras
```

El comando de Fase 1 debe invocarse **con `~/hyperion-kernels` como
directorio de trabajo** (no la raíz del repo) — los binarios del catálogo
se buscan relativos a ese directorio, a propósito, para que el mismo
catálogo funcione en cualquier clúster sin editar cada ruta.

### 2.4 Cómo correr una campaña

Una **campaña** es: un catálogo de kernels × una lista de niveles de
frecuencia × un número de repeticiones. Se define en un archivo YAML
("manifiesto"). Hay 53 manifiestos ya escritos en
`fase1_telemetria/catalog/campaigns/` — la mayoría son campañas reales ya
usadas en el desarrollo del proyecto (nombres con `pacca_` en el nombre).
Para tu primera vez, usa uno de los "smoke test" (pequeños, rápidos, para
confirmar que todo funciona antes de comprometer horas de cómputo):
`campaign_pacca_dvfs_smoke.yaml` (CPU) o `campaign_pacca_gpu_smoke.yaml`
(GPU).

```bash
cd ~/hyperion-kernels

# Paso 1: diagnóstico de solo lectura -- confirma que el nodo está listo
# para ESTA campaña específica, sin escribir nada.
python3 /ruta/al/repo/fase1_telemetria/run_campaign.py diagnose \
    --manifest /ruta/al/repo/fase1_telemetria/catalog/campaigns/campaign_pacca_dvfs_smoke.yaml \
    --output-dir /tmp/diagnose_smoke

# Paso 2: la campaña completa (calibración Roofline + la matriz de corridas).
# --node-id es una etiqueta tuya para identificar el nodo (p.ej. "pacca01").
# --reference-kernel-ref es el kernel usado como referencia de estabilidad
# (normalmente uno del catálogo, p.ej. "npb_mg").
python3 /ruta/al/repo/fase1_telemetria/run_campaign.py run-campaign \
    --manifest /ruta/al/repo/fase1_telemetria/catalog/campaigns/campaign_pacca_dvfs_smoke.yaml \
    --node-id pacca01 \
    --reference-kernel-ref npb_mg

# Paso 3: reporte de la campaña -- una tabla resumen de qué se aceptó y
# qué se rechazó, y por qué.
python3 /ruta/al/repo/fase1_telemetria/run_campaign.py report \
    --campaign-dir ~/hyperion-results/campaigns/<campaign_id del manifiesto>
```

El `output_dir`/`campaign_id` exactos están dentro del propio archivo
YAML del manifiesto (ábrelo y mira las primeras líneas) — el `report` del
paso 3 los necesita para saber dónde buscar.

### 2.5 Cómo saber si funcionó, y qué mirar

- El **paso `diagnose`** debe terminar sin errores. Si falla, casi
  siempre es un permiso (ver §1.3) o un binario del catálogo que falta en
  `~/hyperion-kernels/bin/`.
- El **paso `run-campaign`** tarda desde minutos (un smoke test) hasta
  horas (una campaña completa real) — depende de cuántos kernels × niveles
  × repeticiones tenga el manifiesto. Al terminar, revisa
  `~/hyperion-results/campaigns/<campaign_id>/`: debería haber un
  subdirectorio por cada combinación `kernel × nivel × repetición`, cada
  uno con `samples.csv` (datos crudos) y `windows.csv` (procesado).
- El **reporte** (`report`) te da una tabla con cuántas corridas se
  aceptaron y cuántas se rechazaron, con el motivo exacto de cada rechazo
  (`factor_id`) — **nunca se borran las corridas rechazadas**, quedan como
  evidencia. Si ves muchos rechazos, revisa el motivo antes de asumir que
  el dataset está listo para Fase 2.

### 2.6 Cómo se ve `windows.csv` — las columnas que más importan

Cada fila es una ventana de ~1ms de un kernel corriendo a un nivel de
frecuencia. Las columnas más importantes para lo que sigue:

| Columna | Qué es |
|---|---|
| `kernel_ref` | Qué kernel generó esta ventana |
| `freq_level_id` / `gpu_freq_level_id` | El nivel de frecuencia solicitado (`REF`, `F0`...`F4`) |
| `freq_khz_observed` | La frecuencia REAL observada (siempre verificada por relectura, nunca asumida) |
| `phase_label_train` | La etiqueta `compute_bound`/`memory_bound` -- lo que entrena el clasificador |
| `ipc`, `mpki`, `llc_miss_rate`, `stall_mem_ratio` | Contadores baratos -- lo que el clasificador SÍ puede usar en producción |
| `quality_status` | `"ok"` si la ventana es usable; otro valor explica por qué no |
| `pkg_delta_uj`, `dram_delta_uj` | Energía RAPL de esta ventana (microjulios) |

Para el detalle completo de las ~60 columnas, ver
`fase1_telemetria/README.md` y los comentarios de
`fase1_telemetria/postprocess.py`.

---

## 3. Fase 2 — Entrenar el clasificador (Objetivo 2)

### 3.1 Qué hace, conceptualmente

Toma el `windows.csv` de Fase 1 y entrena varios modelos de clasificación
ligeros (árbol de decisión, Random Forest, regresión logística, XGBoost)
para predecir `phase_label_train` a partir de **solo los contadores
baratos** (`ipc`, `mpki`, `llc_miss_rate`, `stall_mem_ratio`, `ips`,
`running_ratio`, `freq_khz_observed`) — nunca de la intensidad operacional
en sí, porque eso sería "hacer trampa": esa es literalmente la fórmula que
generó la etiqueta, así que un modelo que la reciba no aprende nada útil,
solo memoriza el umbral.

Valida con una técnica llamada **leave-one-familia-out**: entrena dejando
fuera TODOS los kernels de una familia algorítmica (p.ej. todos los
tamaños de multiplicación de matrices) y prueba solo con esa familia — así
mide si el modelo generaliza a un algoritmo que nunca vio, no solo a un
tamaño nuevo de uno que ya vio.

Al final, elige el mejor modelo combinando exactitud (F1 macro) y
velocidad de inferencia (porque el daemon de Fase 3 tiene que decidir en
menos de 1ms) y lo guarda a disco.

### 3.2 Cómo correrlo

```bash
python3 /ruta/al/repo/fase2_clasificador/run_training.py \
    --campaign-dir ~/hyperion-results/campaigns/<campaign_id de Fase 1> \
    --campaign-id <campaign_id de Fase 1> \
    --output-dir /ruta/al/repo/fase2_clasificador/models/
```

Este script SÍ puede correrse desde cualquier directorio (a diferencia del
de Fase 1) — no depende de `~/hyperion-kernels`.

Sin `--output-dir`, corre en "modo exploración": imprime las tablas
comparativas de los modelos sin guardar nada — útil para ver los números
antes de comprometerte a un modelo.

### 3.3 Qué produce, y cómo leerlo

En la terminal vas a ver una tabla así (números de ejemplo):

```
modelo            F1 macro      sd    peor            familia peor   p50 us   p95 us   p99 us
-------------------------------------------------------------------------------------------
random_forest        0.94    0.03    0.88          dual_gemm            4.2      8.1     12.3
xgboost               0.93    0.04    0.85            npb_bt            3.8      7.5     11.0
arbol_prof6           0.89    0.05    0.79       rodinia_lavamd          0.5      0.9      1.4
regresion_log         0.81    0.08    0.65            npb_cg            1.1      2.0      3.1
arbol_prof1           0.72    0.09    0.51            npb_sp            0.3      0.5      0.7
mayoritaria           0.50    0.00    0.50                 --           0.1      0.1      0.2
```

- **`F1 macro`**: qué tan bien clasifica (1.0 = perfecto). Compáralo
  siempre contra `mayoritaria` (la línea base "elegir siempre la clase más
  común") — si tu modelo no le gana claramente, las features no están
  aportando información real.
- **`peor` / `familia peor`**: el F1 más bajo entre las familias dejadas
  fuera, y cuál fue. Un modelo con F1 promedio alto pero un "peor" muy
  bajo generaliza mal a ciertos patrones — vale la pena investigar por qué.
- **`p50/p95/p99 us`**: latencia de una predicción individual, en
  microsegundos. El daemon de Fase 3 decide cada ~1ms (1000 us) — un
  modelo con p99 de 12us dentro de eso deja margen de sobra; uno con p99
  cercano a 1000us sería un problema real.

Con `--output-dir`, además de la tabla, quedan dos archivos:

- `<modelo>.joblib`: el modelo entrenado, listo para cargar con
  `joblib.load(...)` y llamar `.predict(X)`.
- `<modelo>.metadata.json`: toda la información de la tabla de arriba,
  más las features usadas, el seed, cuántas ventanas/familias vio, y la
  comparación completa contra el resto de modelos — esto es lo que citas
  en el capítulo de resultados de la tesis, no hace falta recalcular nada.

⚠️ **Limitación conocida**: este script entrena el clasificador de **CPU**
únicamente. El clasificador de GPU (features de NVML: `gpu_util_pct`,
`gpu_mem_util_pct`, `gpu_power_mw`, `gpu_sm_clock_mhz`, `gpu_temperature_c`)
todavía no está implementado — es parte del trabajo pendiente, ver §6.

---

## 4. Fase 3 — El daemon de control (Objetivo 3)

### 4.1 Qué hace, conceptualmente — y qué de esto ya funciona

El daemon tiene dos "loops" (bucles) que corren en paralelo, porque CPU y
GPU necesitan ritmos completamente distintos:

- **Loop de CPU**: debería correr cada ~1ms: lee los contadores en vivo,
  corre la inferencia del modelo de Fase 2, y si la clase cambió respecto
  al tick anterior, ajusta la frecuencia de CPU.
- **Loop de GPU**: no corre por tiempo fijo, corre "por fase" — detecta
  cuándo la GPU pasa de inactiva a activa (sondeando `gpu_util_pct` cada
  ~50ms, ver el porqué de este diseño en `fase3_daemon/README.md`, sección
  "Historial de diseño" — se intentó otra técnica primero y no funcionó),
  clasifica, y ajusta el reloj de GPU con histéresis (no cambia de reloj
  más seguido de lo que vale la pena, dado el costo de cada cambio).

**Hoy, el loop de GPU está completo y probado.** El loop de CPU **solo
tiene su lógica de decisión construida** (cuándo actuar, cuándo no) — le
falta la pieza que conecta esa lógica con una inferencia ONNX real sobre
el modelo de Fase 2 y con el instrumento de telemetría en vivo. Esto es
justamente uno de los puntos de trabajo pendiente (§6).

### 4.2 Antes del daemon: derivar la tabla de política

El daemon nunca decide "a qué frecuencia poner el CPU cuando la clase es
compute_bound" en tiempo real — esa decisión se calcula **una vez, offline**,
a partir del barrido de frecuencias de Fase 1, y se guarda en un archivo
YAML que el daemon simplemente consulta.

```bash
python3 /ruta/al/repo/fase3_daemon/policy/derive_policy_table.py \
    ~/hyperion-results/campaigns/<campaign_id>/*/windows.csv \
    --campaign-id <campaign_id> \
    --output /ruta/al/repo/fase3_daemon/policy_table.yaml
```

**Qué hace internamente**: para cada combinación (dispositivo, clase,
nivel de frecuencia), calcula el EDP mediano observado en el barrido, y
compara cada nivel contra la frecuencia de referencia (`REF`) con una
prueba estadística pareada (Wilcoxon o t-test, elegida automáticamente
según la distribución de los datos). Si un nivel mejora el EDP de forma
estadísticamente defendible, ese es el elegido para esa clase. **Si
ningún nivel mejora de forma significativa, la política para esa clase
queda explícitamente en `"no_actuar"`** — esto no es un error del script,
es un resultado científico legítimo (puede significar que el rango
dinámico de potencia del hardware es demasiado angosto para que valga la
pena bajar la frecuencia en esa clase).

Para GPU específicamente: **sin haber medido `T_transición_gpu` primero
(§6), la política de GPU siempre queda en `"no_actuar"`**, sin importar
qué tan buenos se vean los datos — es una salvaguarda a propósito, no un
bug (ver §6 para por qué esta medición es necesaria antes de confiar en
cualquier resultado GPU).

**Cómo leer `policy_table.yaml`**: es un archivo con 4 entradas
(`cpu-compute_bound`, `cpu-memory_bound`, `gpu-compute_bound`,
`gpu-memory_bound`). Cada una dice `action: actuar` o `action: no_actuar`,
y si es `actuar`, el nivel elegido y la frecuencia real (no solo el ID del
nivel) que se observó en esos datos.

### 4.3 Correr el daemon (modo de prueba, sin tocar hardware)

```bash
python3 /ruta/al/repo/fase3_daemon/run_daemon.py \
    --policy-table /ruta/al/repo/fase3_daemon/policy_table.yaml \
    --min-dwell-ns 10000000000 \
    --dry-run
```

`--dry-run` hace que el daemon clasifique y decida normalmente, pero
**nunca escriba una frecuencia real** — solo registra en el log qué
habría hecho. Siempre valida así antes de correr sin `--dry-run` en
hardware real. `--min-dwell-ns` es el tiempo mínimo (en nanosegundos) que
el reloj de GPU debe quedarse en un valor antes de poder cambiar de
nuevo — hoy no hay un valor medido real para esto (ver §6), así que
cualquier valor que uses es un placeholder, no un número confiable
todavía.

⚠️ Con la limitación de §4.1, correr `run_daemon.py` hoy arranca **solo el
loop de GPU** — vas a ver en el log una advertencia explícita de que el
loop de CPU no está integrado todavía.

---

## 5. Fase 4 — Evaluación experimental (Objetivo 4)

### 5.1 Qué hace, conceptualmente

Compara 4 escenarios: el agente propuesto, y 3 líneas base (gobernador
`ondemand`, gobernador `schedutil`, y frecuencia fija `performance`). Para
cada uno, corre el mismo catálogo de kernels y mide tiempo, energía y EDP.
Después compara el EDP del agente contra cada línea base con la misma
prueba de significancia estadística que usa el derivador de política de
Fase 3 — para que "mejora" signifique lo mismo en los dos lugares.

**El reporte se genera a partir de datos ya producidos** — este script no
lanza las 4 campañas por ti automáticamente todavía (parte del trabajo
pendiente, §6). Cada escenario hay que correrlo como una campaña de Fase 1
normal, cambiando el gobernador antes de cada una.

### 5.2 Cómo conmutar el gobernador para las líneas base

```python
from common.hpc import environment
from fase4_evaluacion.governors import governor_scenario

# "2,3,4,5" -- los mismos CPUs delegados que usaste en el manifiesto de
# Fase 1 (cores.delegated_cpus del YAML de la campaña).
env = environment.detect_environment(delegated_cpus="2,3,4,5")
with governor_scenario(env.delegated_cpus, "ondemand", env):
    # aquí adentro: correr fase1_telemetria/run_campaign.py normalmente,
    # con un output_dir/campaign_id propio para este escenario
    ...
# al salir del "with", el gobernador original queda restaurado
# automáticamente, incluso si algo falló adentro
```

Repite esto para `"schedutil"` y `"performance"`, cada uno con su propio
`campaign_id` en el manifiesto de Fase 1 (para no mezclar los resultados).
El escenario del "agente propuesto" son los datos que produzca el daemon
de Fase 3 corriendo de verdad (una vez esté completo, §6).

### 5.3 Generar el reporte final

```bash
python3 /ruta/al/repo/fase4_evaluacion/run_evaluation.py \
    --scenario agente      ~/hyperion-results/campaigns/agente/*/windows.csv \
    --scenario performance ~/hyperion-results/campaigns/performance/*/windows.csv \
    --scenario ondemand    ~/hyperion-results/campaigns/ondemand/*/windows.csv \
    --scenario schedutil   ~/hyperion-results/campaigns/schedutil/*/windows.csv \
    --agent-scenario agente \
    --output /ruta/al/repo/fase4_evaluacion/reporte_final.txt
```

Si te falta un escenario (p.ej. todavía no corriste `schedutil`), el
script lo omite del reporte con un aviso explícito en vez de fallar — así
puedes generar reportes parciales mientras completas las corridas.

### 5.4 Cómo leer el reporte

```
device  clase             baseline       n_kernels  Δ EDP agente      prueba   p-valor  sig.
------------------------------------------------------------------------------------------------
cpu     compute_bound     performance          9          -3.2%    wilcoxon   0.1200    no
cpu     memory_bound      performance          9         -35.1%    wilcoxon   0.0021    SI
gpu     compute_bound     performance          6          -8.4%      ttest    0.0450    SI
```

- **`Δ EDP agente`**: cambio relativo de EDP del agente contra ese
  baseline. Negativo = el agente gastó MENOS EDP (mejor). Positivo = el
  agente gastó más (peor).
- **`sig.`**: si esa diferencia es estadísticamente defendible (`SI`) o
  no (`no`) al nivel de confianza pedido (`--alpha`, default 0.05). Un
  `Δ EDP` grande con `sig.: no` significa que no hay suficiente evidencia
  todavía — no lo reportes como una mejora confirmada.
- Fíjate que el reporte es **por dispositivo y por clase por separado**,
  nunca un solo número global — es normal (y científicamente más
  interesante) que el agente ayude en unas combinaciones y no en otras.
  Repórtalo así en la tesis, con matices, no como un resultado único.

---

## 6. Hoja de ruta para terminar el trabajo de grado

Esto es lo que falta, en el orden en que tiene sentido abordarlo. Cada
punto tiene su propia sección de "Limitaciones conocidas" en el README de
la fase correspondiente, con más detalle técnico.

### 6.1 Medir `T_transición_gpu` (bloquea la política real de GPU)

**Qué es**: cuánto tarda de verdad el reloj de la GPU en estabilizarse
después de pedir un cambio (`nvidia-smi -lgc`). No es instantáneo, y nadie
lo ha medido todavía en este proyecto — por eso la política de GPU siempre
sale en `"no_actuar"` (§4.2).

**Cómo abordarlo**: lanzar un kernel GPU real y sostenido, pedir un
cambio de reloj a mitad de la corrida, y muestrear
`nvmlDeviceGetClockInfo` a la cadencia más fina posible (nunca a 100ms,
demasiado grueso) hasta que el reloj observado se estabilice dentro de una
banda de tolerancia. Repetir para varios pares de niveles (no asumir que
la latencia es simétrica). El resultado alimenta directamente
`--t-transicion-gpu-ns` en `derive_policy_table.py` y `--min-dwell-ns` en
`run_daemon.py`.

### 6.2 Entrenar el clasificador de GPU

`fase2_clasificador/training/train_phase.py` solo cubre CPU. Para GPU
hace falta un script equivalente que use las 5 features NVML
(`gpu_util_pct`, `gpu_mem_util_pct`, `gpu_power_mw`, `gpu_sm_clock_mhz`,
`gpu_temperature_c`) contra `phase_label_train` de las filas GPU de
`windows.csv` — el dataset ya existe si corriste una campaña GPU en Fase 1,
solo falta el script de entrenamiento (puede reutilizar buena parte de la
lógica de `train_phase.py`: la fuga de información, la serialización, la
selección por latencia).

### 6.3 Construir el loop de CPU real del daemon

`fase3_daemon/cpu_loop/include/cpu_phase_controller.hpp` ya tiene la
lógica de decisión (cuándo actuar) construida y probada. Falta: (a)
exportar el modelo `.joblib` de Fase 2 a formato ONNX (el entorno conda ya
trae `skl2onnx` para esto), (b) escribir el código C++ que cargue ese
modelo con ONNX Runtime (ya disponible en el entorno conda, headers y
librería confirmados) y lo conecte con `common/telemetry/collector.hpp`
(el instrumento que ya lee los contadores en vivo cada ~1ms).

### 6.4 Orquestar automáticamente los 4 escenarios de Fase 4

Hoy `run_evaluation.py` solo genera el reporte a partir de datos ya
producidos (§5.1). Para automatizar esto hace falta que
`fase1_telemetria/campaign.py` acepte un "envoltorio de escenario" (correr
la misma campaña bajo cada gobernador sin repetir la configuración a
mano) — un cambio focalizado, no una reescritura.

### Orden recomendado

1. §6.1 (T_transición_gpu) — es la medición más rápida y desbloquea la
   política de GPU real.
2. §6.2 (clasificador de GPU) — en paralelo con lo anterior, si hay dos
   personas en el equipo.
3. §6.3 (loop de CPU) — la pieza más grande, depende de tener ya un
   modelo de Fase 2 entrenado (CPU, que ya existe; GPU si se hizo §6.2).
4. §6.4 (orquestación de Fase 4) — déjalo para el final, cuando ya haya
   datos reales del agente (§6.3) que comparar.

---

## 7. Problemas comunes

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `diagnose` falla con un binario "no encontrado" | Falta compilar ese kernel en `~/hyperion-kernels/bin/` | Revisar `catalog.yaml` para la procedencia del binario y compilarlo |
| `diagnose` falla con un checksum distinto | El binario se recompiló y cambió de bytes | Si el cambio es intencional, actualizar el checksum en `catalog.yaml`; si no, investigar por qué cambió |
| `run-campaign` rechaza casi todas las corridas | Revisar el `report` -- el `factor_id` de cada rechazo dice la causa exacta | No ignorar los rechazos ni recalcular sin entender la causa |
| `pytest` no se encuentra tras `pip install -e .` | Faltó el extra `[dev]` | `pip install -e ".[dev]"` |
| Falla la compilación C++ (`cmake`) | Falta `g++`/`cmake` del sistema | Ver la sección Rocky Linux/Fedora del `README.md` raíz |
| `check-readiness` marca la escritura de frecuencia como no disponible | Falta el permiso del administrador del clúster | Escalar al administrador -- no hay workaround del lado del usuario, y no se debe fabricar un dato de frecuencia no verificado |
| El daemon (`run_daemon.py`) no hace nada visible | Es normal en `--dry-run` si la GPU está inactiva -- el loop de GPU solo actúa cuando `gpu_util_pct` supera el umbral | Lanzar una carga GPU real en paralelo para ver actividad en el log |

---

## 8. Referencia rápida de comandos

```bash
# Preparación (una vez)
conda env create -f environment-hyperion-verify.yml && conda activate hyperion-verify
./run_all.sh check-readiness
./run_all.sh test

# Fase 1 (desde ~/hyperion-kernels)
python3 <repo>/fase1_telemetria/run_campaign.py diagnose --manifest <M> --output-dir <D>
python3 <repo>/fase1_telemetria/run_campaign.py run-campaign --manifest <M> --node-id <N> --reference-kernel-ref <K>
python3 <repo>/fase1_telemetria/run_campaign.py report --campaign-dir <D>

# Fase 2
python3 <repo>/fase2_clasificador/run_training.py --campaign-dir <D> --campaign-id <ID> --output-dir <repo>/fase2_clasificador/models/

# Fase 3
python3 <repo>/fase3_daemon/policy/derive_policy_table.py <D>/*/windows.csv --campaign-id <ID> --output policy_table.yaml
python3 <repo>/fase3_daemon/run_daemon.py --policy-table policy_table.yaml --min-dwell-ns <N> --dry-run

# Fase 4
python3 <repo>/fase4_evaluacion/run_evaluation.py --scenario <nombre> <glob> [--scenario ... repetible] --agent-scenario <nombre> --output reporte.txt

# Cada script tiene --help con todos los flags
python3 <repo>/faseN_.../run_*.py --help
```

Para el detalle técnico completo de cualquier pieza (todas las columnas de
`windows.csv`, todos los campos de un manifiesto, la arquitectura interna
del daemon, etc.), el README de cada carpeta de fase es la referencia
autoritativa — este manual es la puerta de entrada, no el reemplazo.
